from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from math import gcd
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.fftpack import dct, idct
from scipy.io import wavfile
from scipy.signal import resample_poly
from numpy.linalg import svd


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIO_WATERMARKING_DIR = REPO_ROOT / "third_party" / "audio-watermarking"
if str(AUDIO_WATERMARKING_DIR) not in sys.path:
    sys.path.insert(0, str(AUDIO_WATERMARKING_DIR))


DEFAULT_PAYLOAD_LENGTHS = {
    "norm_space": 512,
    "patchwork": 40,
    "fsvc": 40,
}
PATCHWORK_FS = 3000
PATCHWORK_FE = 7000
PATCHWORK_K1 = 0.195
PATCHWORK_K2 = 0.08
FSVC_GAMMA1 = 0.136
FSVC_GAMMA2 = 0.181
FSVC_ALPHA = 3.0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_payload_bits(path: Path) -> np.ndarray:
    bits = np.load(path)
    flat = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if np.any((flat != 0) & (flat != 1)):
        raise ValueError("payload bits must be binary")
    return flat


def _save_bit_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(values, dtype=np.uint8).reshape(-1))


def _save_float_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(values, dtype=np.float64).reshape(-1))


def _normalize_method_name(method_params: dict[str, Any]) -> str:
    raw_name = str(method_params.get("method") or method_params.get("method_name") or "norm_space")
    normalized = raw_name.lower().replace("-", "_")
    aliases = {
        "normspace": "norm_space",
        "norm_space": "norm_space",
        "saadi": "norm_space",
        "patchwork": "patchwork",
        "patchwork_multilayer": "patchwork",
        "fsvc": "fsvc",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown traditional audio watermark method: {raw_name!r}")
    return aliases[normalized]


def _load_audio(path: Path) -> tuple[int, np.ndarray, dict[str, Any]]:
    try:
        sample_rate, data = wavfile.read(path)
        signal, info = _pcm_to_float(data)
        info.update(
            {
                "source_format": "wav",
                "source_dtype": str(data.dtype),
                "source_channels": 1 if np.asarray(data).ndim == 1 else int(np.asarray(data).shape[1]),
            }
        )
        return int(sample_rate), signal, info
    except ValueError as exc:
        if path.suffix.lower() == ".wav":
            raise
        return _load_audio_with_optional_backend(path, exc)


def _load_audio_with_optional_backend(path: Path, original_error: Exception) -> tuple[int, np.ndarray, dict[str, Any]]:
    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(path))
        array = waveform.detach().cpu().numpy()
        if array.ndim == 2:
            signal = array.mean(axis=0) if array.shape[0] > 1 else array[0]
        else:
            signal = np.asarray(array, dtype=np.float32).reshape(-1)
        return (
            int(sample_rate),
            np.asarray(signal, dtype=np.float32),
            {
                "source_format": path.suffix.lower().lstrip(".") or "audio",
                "source_dtype": str(array.dtype),
                "source_channels": int(array.shape[0]) if array.ndim == 2 else 1,
                "audio_backend": "torchaudio",
            },
        )
    except Exception:
        pass

    try:
        import soundfile as sf

        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        signal = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
        return (
            int(sample_rate),
            np.asarray(signal, dtype=np.float32),
            {
                "source_format": path.suffix.lower().lstrip(".") or "audio",
                "source_dtype": str(data.dtype),
                "source_channels": int(data.shape[1]),
                "audio_backend": "soundfile",
            },
        )
    except Exception:
        pass

    raise ValueError(
        f"Could not read {path}. WAV files are supported by scipy; non-WAV audio requires "
        "torchaudio or soundfile in the worker environment."
    ) from original_error


def _pcm_to_float(data: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    array = np.asarray(data)
    if array.ndim == 2:
        array = array.mean(axis=1)
    info: dict[str, Any] = {}
    if np.issubdtype(array.dtype, np.floating):
        return np.asarray(array, dtype=np.float32), {"encoding": "float32"}
    if array.dtype == np.uint8:
        return ((array.astype(np.float32) - 128.0) / 128.0), {"encoding": "uint8"}
    if array.dtype == np.int16:
        return (array.astype(np.float32) / 32768.0), {"encoding": "int16"}
    if array.dtype == np.int32:
        return (array.astype(np.float32) / 2147483648.0), {"encoding": "int32"}
    raise ValueError(f"Unsupported WAV dtype: {array.dtype}")


def _write_audio(path: Path, sample_rate: int, signal: np.ndarray) -> None:
    output = np.asarray(signal, dtype=np.float32).reshape(-1)
    output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
    encoded = np.round(np.clip(output, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, int(sample_rate), encoded)


def _resample_if_needed(
    signal: np.ndarray,
    sample_rate: int,
    method_params: dict[str, Any],
    audio_info: dict[str, Any],
) -> tuple[int, np.ndarray, dict[str, Any]]:
    if method_params.get("sample_rate") is None:
        return int(sample_rate), np.asarray(signal, dtype=np.float32).reshape(-1), audio_info
    target_sample_rate = int(method_params["sample_rate"])
    if target_sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {target_sample_rate}")
    if int(sample_rate) == target_sample_rate:
        audio_info.setdefault("original_sample_rate", int(sample_rate))
        audio_info["resampled_to_target_rate"] = False
        return target_sample_rate, np.asarray(signal, dtype=np.float32).reshape(-1), audio_info

    divisor = gcd(int(sample_rate), target_sample_rate)
    up = target_sample_rate // divisor
    down = int(sample_rate) // divisor
    resampled = resample_poly(np.asarray(signal, dtype=np.float32).reshape(-1), up, down)
    audio_info = dict(audio_info)
    audio_info["original_sample_rate"] = int(sample_rate)
    audio_info["resampled_to_target_rate"] = True
    audio_info["resample_up"] = int(up)
    audio_info["resample_down"] = int(down)
    return target_sample_rate, np.asarray(resampled, dtype=np.float32), audio_info


def _batch_truncate_seconds(method_params: dict[str, Any]) -> float:
    return float(method_params.get("batch_truncate_seconds", 3.0))


def _truncate_audio_for_batch(
    signal: np.ndarray,
    sample_rate: int,
    method_params: dict[str, Any],
    audio_info: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    seconds = _batch_truncate_seconds(method_params)
    info = dict(audio_info)
    info["batch_truncate_seconds"] = float(seconds)
    if seconds <= 0:
        info["truncated_for_batch"] = False
        return np.asarray(signal, dtype=np.float32).reshape(-1), info
    max_samples = int(round(float(seconds) * int(sample_rate)))
    array = np.asarray(signal, dtype=np.float32).reshape(-1)
    if max_samples <= 0 or array.size <= max_samples:
        info["truncated_for_batch"] = False
        return array, info
    info["truncated_for_batch"] = True
    info["pre_truncate_samples"] = int(array.size)
    info["post_truncate_samples"] = int(max_samples)
    return array[:max_samples], info


def _detect_wav_sample_rate(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        return int(handle.getframerate())


def embed_audio(
    signal: np.ndarray,
    payload_bits: np.ndarray,
    sample_rate: int,
    method_params: dict[str, Any],
) -> np.ndarray:
    method = _normalize_method_name(method_params)
    if method == "norm_space":
        from norm_space_watermarking import norm_space_watermark_embedding

        delta_value = float(method_params.get("delta", 0.03))
        return np.asarray(norm_space_watermark_embedding(signal, payload_bits, delta=delta_value), dtype=np.float32)
    if method == "patchwork":
        from patchwork_multylayer_watermarking import patchwork_multilayer_watermark_embedding

        return np.asarray(
            patchwork_multilayer_watermark_embedding(signal, payload_bits, sr=int(sample_rate)),
            dtype=np.float32,
        )
    if method == "fsvc":
        return np.asarray(
            _fsvc_watermark_embedding(signal, payload_bits, sr=int(sample_rate)),
            dtype=np.float32,
        )
    raise ValueError(f"Unsupported method: {method}")


def decode_audio(
    signal: np.ndarray,
    watermark_length: int,
    sample_rate: int,
    method_params: dict[str, Any],
) -> np.ndarray:
    method = _normalize_method_name(method_params)
    if method == "norm_space":
        from norm_space_watermarking import norm_space_watermark_detection

        delta_value = float(method_params.get("delta", 0.03))
        return np.asarray(
            norm_space_watermark_detection(signal, watermark_length=int(watermark_length), delta=delta_value),
            dtype=np.float32,
        )
    if method == "patchwork":
        return np.asarray(
            _patchwork_multilayer_watermark_detection(
                signal,
                watermark_length=int(watermark_length),
                sr=int(sample_rate),
            ),
            dtype=np.float32,
        )
    if method == "fsvc":
        return np.asarray(
            _fsvc_watermark_detection(signal, watermark_length=int(watermark_length), sr=int(sample_rate)),
            dtype=np.float32,
        )
    raise ValueError(f"Unsupported method: {method}")


def _patchwork_multilayer_watermark_detection(
    watermarked_signal: np.ndarray,
    watermark_length: int = 40,
    sr: int = 16000,
) -> np.ndarray:
    signal = np.asarray(watermarked_signal, dtype=np.float64).reshape(-1)
    length = len(signal)
    si = int(PATCHWORK_FS / (float(sr) / length))
    ei = int(PATCHWORK_FE / (float(sr) / length))
    si = max(0, min(si, length - 1))
    ei = max(si, min(ei, length - 1))

    coefficients = dct(signal, type=2, norm="ortho")
    selected = coefficients[si : (ei + 1)]
    selected_length = len(selected)
    if selected_length < int(watermark_length) * 2:
        raise ValueError(
            f"Patchwork selected frequency band is too short for {watermark_length} bits "
            f"at sample_rate={sr}: {selected_length} coefficients"
        )
    if selected_length % (int(watermark_length) * 2) != 0:
        selected_length -= selected_length % (int(watermark_length) * 2)
        selected = selected[:selected_length]

    paired = np.dstack((selected[: selected_length // 2], selected[: (selected_length // 2 - 1) : -1])).flatten()
    segments = np.array_split(paired, int(watermark_length) * 2)
    watermark_bits = []
    for index in range(0, len(segments), 2):
        m1j = np.mean(np.abs(segments[index]))
        m2j = np.mean(np.abs(segments[index + 1]))
        watermark_bits.append(0 if (m1j - m2j) >= 0 else 1)
    return np.asarray(watermark_bits, dtype=np.float32)


def _fsvc_watermark_embedding(signal: np.ndarray, watermark: np.ndarray, sr: int = 16000) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    bits = np.asarray(watermark, dtype=np.uint8).reshape(-1)
    gamma1 = FSVC_GAMMA1 * float(sr) / 44100.0
    gamma2 = FSVC_GAMMA2 * float(sr) / 44100.0
    frames = np.array_split(signal, len(bits))
    watermarked_frames = []
    for index, frame in enumerate(frames):
        first_half, second_half = np.array_split(frame, 2)
        first_coefficients = dct(first_half, type=2, norm="ortho")
        second_coefficients = dct(second_half, type=2, norm="ortho")
        low = int(gamma1 * len(frame))
        high = int(gamma2 * len(frame) + 1)
        high = max(low + 1, min(high, len(first_coefficients), len(second_coefficients)))
        first_band = np.expand_dims(first_coefficients[low:high], axis=-1)
        second_band = np.expand_dims(second_coefficients[low:high], axis=-1)
        u1, s1, v1 = svd(first_band, full_matrices=False)
        u2, s2, v2 = svd(second_band, full_matrices=False)
        l1, l2 = _modify_fsvc_svd_pair(float(s1[0]), float(s2[0]), int(bits[index]))
        first_coefficients[low:high] = np.squeeze(u1 @ np.diag([l1]) @ v1)
        second_coefficients[low:high] = np.squeeze(u2 @ np.diag([l2]) @ v2)
        watermarked_frames.append(
            np.concatenate(
                [
                    idct(first_coefficients, type=2, norm="ortho"),
                    idct(second_coefficients, type=2, norm="ortho"),
                ]
            )
        )
    return np.concatenate(watermarked_frames)


def _fsvc_watermark_detection(
    watermarked_signal: np.ndarray,
    watermark_length: int = 40,
    sr: int = 16000,
) -> np.ndarray:
    signal = np.asarray(watermarked_signal, dtype=np.float64).reshape(-1)
    gamma1 = FSVC_GAMMA1 * float(sr) / 44100.0
    gamma2 = FSVC_GAMMA2 * float(sr) / 44100.0
    frames = np.array_split(signal, int(watermark_length))
    watermark_bits = []
    for frame in frames:
        first_half, second_half = np.array_split(frame, 2)
        first_coefficients = dct(first_half, type=2, norm="ortho")
        second_coefficients = dct(second_half, type=2, norm="ortho")
        low = int(gamma1 * len(frame))
        high = int(gamma2 * len(frame) + 1)
        high = max(low + 1, min(high, len(first_coefficients), len(second_coefficients)))
        first_band = np.expand_dims(first_coefficients[low:high], axis=-1)
        second_band = np.expand_dims(second_coefficients[low:high], axis=-1)
        _, s1, _ = svd(first_band, full_matrices=False)
        _, s2, _ = svd(second_band, full_matrices=False)
        watermark_bits.append(0 if float(s1[0]) / max(float(s2[0]), 1e-12) < 1.0 else 1)
    return np.asarray(watermark_bits, dtype=np.float32)


def _modify_fsvc_svd_pair(l1: float, l2: float, watermark_bit: int) -> tuple[float, float]:
    if int(watermark_bit) == 0:
        if l1 / max(l2, 1e-12) > 1.0 / (1.0 + FSVC_ALPHA):
            l1 = (l1 + l2 * (1.0 + FSVC_ALPHA)) / (FSVC_ALPHA**2 + 2.0 * FSVC_ALPHA + 2.0)
            l2 = (1.0 + FSVC_ALPHA) * l1
    else:
        if l1 / max(l2, 1e-12) < 1.0 + FSVC_ALPHA:
            l2 = (l2 + l1 * (1.0 + FSVC_ALPHA)) / (FSVC_ALPHA**2 + 2.0 * FSVC_ALPHA + 2.0)
            l1 = (1.0 + FSVC_ALPHA) * l2
    return float(l1), float(l2)


def _bits_and_scores(decoded_values: np.ndarray, payload_length: int) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(decoded_values, dtype=np.float64).reshape(-1)[: int(payload_length)]
    if scores.size < int(payload_length):
        scores = np.pad(scores, (0, int(payload_length) - scores.size), constant_values=0.0)
    bits = (scores >= 0.5).astype(np.uint8)
    return bits, scores


def _metadata_for_embed(
    *,
    input_audio_path: Path,
    output_audio_path: Path,
    payload_bits: np.ndarray,
    sample_rate: int,
    audio_info: dict[str, Any],
    method_params: dict[str, Any],
) -> dict[str, Any]:
    method = _normalize_method_name(method_params)
    metadata = {
        "method": method,
        "input_audio_path": str(input_audio_path),
        "output_audio_path": str(output_audio_path),
        "payload_bits": payload_bits.astype(np.uint8).tolist(),
        "payload_length": int(payload_bits.size),
        "sample_rate": int(sample_rate),
        "default_payload_length": int(DEFAULT_PAYLOAD_LENGTHS[method]),
    }
    metadata.update(audio_info)
    for key in ("delta", "device"):
        if key in method_params:
            metadata[key] = method_params[key]
    return metadata


def handle_embed_request(
    request_path: Path,
    response_path: Path,
    embed_impl: Callable[[np.ndarray, np.ndarray, int, dict[str, Any]], np.ndarray] | None = None,
) -> None:
    request = _read_json(request_path)
    response = _handle_embed_payload(request, embed_impl=embed_impl)
    _write_json(response_path, response)


def handle_decode_request(
    request_path: Path,
    response_path: Path,
    decode_impl: Callable[[np.ndarray, int, int, dict[str, Any]], np.ndarray] | None = None,
) -> None:
    request = _read_json(request_path)
    response = _handle_decode_payload(request, decode_impl=decode_impl)
    _write_json(response_path, response)


def _handle_embed_payload(
    request: dict[str, Any],
    *,
    embed_impl: Callable[[np.ndarray, np.ndarray, int, dict[str, Any]], np.ndarray] | None = None,
) -> dict[str, Any]:
    input_audio_path = Path(request["input_audio_path"])
    payload_bits = _load_payload_bits(Path(request["payload_bits_path"]))
    output_audio_path = Path(request["output_audio_path"])
    metadata_output_path = Path(request["metadata_output_path"])
    method_params = dict(request.get("method_params") or {})
    method = _normalize_method_name(method_params)

    sample_rate, signal, audio_info = _load_audio(input_audio_path)
    sample_rate, signal, audio_info = _resample_if_needed(signal, sample_rate, method_params, audio_info)
    if bool(method_params.get("_batch_mode", False)):
        signal, audio_info = _truncate_audio_for_batch(signal, sample_rate, method_params, audio_info)
    impl = embed_impl or embed_audio
    watermarked_signal = impl(signal, payload_bits, sample_rate, method_params)
    _write_audio(output_audio_path, sample_rate, watermarked_signal)
    metadata = _metadata_for_embed(
        input_audio_path=input_audio_path,
        output_audio_path=output_audio_path,
        payload_bits=payload_bits,
        sample_rate=sample_rate,
        audio_info=audio_info,
        method_params=method_params,
    )
    _write_json(metadata_output_path, metadata)
    return {
        "status": "ok",
        "watermarked_audio_path": str(output_audio_path),
        "metadata_path": str(metadata_output_path),
    }


def _handle_decode_payload(
    request: dict[str, Any],
    *,
    decode_impl: Callable[[np.ndarray, int, int, dict[str, Any]], np.ndarray] | None = None,
) -> dict[str, Any]:
    input_audio_path = Path(request["input_audio_path"])
    metadata_path = Path(request["metadata_path"])
    decoded_bits_output_path = Path(request["decoded_bits_output_path"])
    soft_scores_output_path = Path(request["soft_scores_output_path"])
    metadata = _read_json(metadata_path)
    method_params = {**metadata, **dict(request.get("method_params") or {})}
    method_params["method"] = metadata.get("method", method_params.get("method", "norm_space"))
    payload_length = int(metadata.get("payload_length", DEFAULT_PAYLOAD_LENGTHS[_normalize_method_name(method_params)]))
    sample_rate, signal, audio_info = _load_audio(input_audio_path)
    sample_rate, signal, _ = _resample_if_needed(signal, sample_rate, method_params, audio_info)
    if bool(method_params.get("_batch_mode", False)):
        signal, _ = _truncate_audio_for_batch(signal, sample_rate, method_params, audio_info)
    impl = decode_impl or decode_audio
    decoded_values = impl(signal, payload_length, sample_rate, method_params)
    decoded_bits, soft_scores = _bits_and_scores(decoded_values, payload_length)
    _save_bit_array(decoded_bits_output_path, decoded_bits)
    _save_float_array(soft_scores_output_path, soft_scores)
    return {
        "status": "detected",
        "detect_prob": 1.0,
        "decoded_bits_path": str(decoded_bits_output_path),
        "soft_scores_path": str(soft_scores_output_path),
    }


def handle_embed_batch_request(
    request_path: Path,
    response_path: Path,
    embed_impl: Callable[[np.ndarray, np.ndarray, int, dict[str, Any]], np.ndarray] | None = None,
) -> None:
    request = _read_json(request_path)
    method_params = dict(request.get("method_params") or {})
    results = []
    for raw_job in list(request.get("jobs") or []):
        job = {**dict(raw_job), "method_params": {**method_params, "_batch_mode": True}}
        results.append(_handle_embed_payload(job, embed_impl=embed_impl))
    _write_json(response_path, {"status": "ok", "results": results})


def handle_decode_batch_request(
    request_path: Path,
    response_path: Path,
    decode_impl: Callable[[np.ndarray, int, int, dict[str, Any]], np.ndarray] | None = None,
) -> None:
    request = _read_json(request_path)
    method_params = dict(request.get("method_params") or {})
    results = []
    for raw_job in list(request.get("jobs") or []):
        job = {**dict(raw_job), "method_params": {**method_params, "_batch_mode": True}}
        results.append(_handle_decode_payload(job, decode_impl=decode_impl))
    _write_json(response_path, {"status": "ok", "results": results})


def handle_session_request(request: dict[str, Any], default_method: str | None = None) -> dict[str, Any]:
    operation = str(request["operation"])
    response_path = Path(request["response_path"])
    payload = dict(request)
    method_params = dict(payload.get("method_params") or {})
    if default_method and "method" not in method_params and "method_name" not in method_params:
        method_params["method"] = default_method
    payload["method_params"] = method_params

    if operation == "embed":
        response = _handle_embed_payload(payload)
    elif operation == "decode":
        response = _handle_decode_payload(payload)
    elif operation == "embed_batch":
        request_path = response_path.parent / "embed_batch_request_from_payload.json"
        _write_json(request_path, payload)
        handle_embed_batch_request(request_path, response_path)
        return _read_json(response_path)
    elif operation == "decode_batch":
        request_path = response_path.parent / "decode_batch_request_from_payload.json"
        _write_json(request_path, payload)
        handle_decode_batch_request(request_path, response_path)
        return _read_json(response_path)
    else:
        raise ValueError(f"Unknown session operation: {operation}")

    _write_json(response_path, response)
    return response


def serve_json_requests(default_method: str | None = None, session_file: Path | None = None) -> int:
    if session_file is not None:
        count = int(session_file.read_text(encoding="utf-8") or "0") if session_file.exists() else 0
        session_file.write_text(str(count + 1), encoding="utf-8")
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            response = handle_session_request(json.loads(line), default_method=default_method)
        except Exception as exc:
            response = {"status": "error", "message": str(exc)}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traditional audio watermark file-based worker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("embed", "decode", "embed_batch", "decode_batch"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--request-json", required=True)
        sub.add_argument("--response-json", required=True)
        sub.add_argument("--method", choices=sorted(DEFAULT_PAYLOAD_LENGTHS), default=None)
    serve = subparsers.add_parser("serve-json")
    serve.add_argument("--method", choices=sorted(DEFAULT_PAYLOAD_LENGTHS), default=None)
    serve.add_argument("--session-file", required=False)
    return parser


def _apply_default_method_to_request(path: Path, method: str | None) -> Path:
    if not method:
        return path
    request = _read_json(path)
    method_params = dict(request.get("method_params") or {})
    method_params.setdefault("method", method)
    request["method_params"] = method_params
    patched_path = path.parent / f"{path.stem}_{method}.json"
    _write_json(patched_path, request)
    return patched_path


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if args.command == "serve-json":
        session_file = None if not getattr(args, "session_file", None) else Path(args.session_file)
        return serve_json_requests(default_method=args.method, session_file=session_file)

    request_path = _apply_default_method_to_request(Path(args.request_json), args.method)
    response_path = Path(args.response_json)
    if args.command == "embed":
        handle_embed_request(request_path, response_path)
    elif args.command == "decode":
        handle_decode_request(request_path, response_path)
    elif args.command == "embed_batch":
        handle_embed_batch_request(request_path, response_path)
    else:
        handle_decode_batch_request(request_path, response_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
