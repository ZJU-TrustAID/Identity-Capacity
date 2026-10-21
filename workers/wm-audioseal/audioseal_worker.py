from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIOSEAL_SRC = REPO_ROOT / "third_party" / "audioseal" / "src"
if str(AUDIOSEAL_SRC) not in sys.path:
    sys.path.insert(0, str(AUDIOSEAL_SRC))

DEFAULT_GENERATOR_CHECKPOINT = REPO_ROOT / "resources" / "models" / "audioseal" / "generator_base.pth"
DEFAULT_DETECTOR_CHECKPOINT = REPO_ROOT / "resources" / "models" / "audioseal" / "detector_base.pth"
_MODEL_CACHE: dict[tuple[str, str, str], tuple[Any, Any, Any]] = {}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_payload_bits(path: Path) -> list[int]:
    bits = np.load(path)
    flat = np.asarray(bits, dtype=np.uint8).reshape(-1)
    return [int(x) for x in flat.tolist()]


def _save_bit_array(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(values, dtype=np.uint8))


def _save_float_array(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(values, dtype=np.float64))


def handle_embed_request(
    request_path: Path,
    response_path: Path,
    embed_impl: Callable[[Path, list[int], dict[str, Any], Path], dict[str, Any]] | None = None,
) -> None:
    request = _read_json(request_path)
    input_audio_path = Path(request["input_audio_path"])
    payload_bits_path = Path(request["payload_bits_path"])
    output_audio_path = Path(request["output_audio_path"])
    metadata_output_path = Path(request["metadata_output_path"])
    method_params = dict(request.get("method_params") or {})

    payload_bits = _load_payload_bits(payload_bits_path)
    impl = embed_impl or embed_audio_file
    metadata = impl(
        input_audio_path,
        payload_bits,
        method_params,
        output_audio_path,
    )
    _write_json(metadata_output_path, metadata)
    _write_json(
        response_path,
        {
            "status": "ok",
            "watermarked_audio_path": str(output_audio_path),
            "metadata_path": str(metadata_output_path),
        },
    )


def handle_decode_request(
    request_path: Path,
    response_path: Path,
    decode_impl: Callable[[Path, dict[str, Any], dict[str, Any]], tuple[list[int], list[float], float, str]] | None = None,
) -> None:
    request = _read_json(request_path)
    input_audio_path = Path(request["input_audio_path"])
    metadata_path = Path(request["metadata_path"])
    decoded_bits_output_path = Path(request["decoded_bits_output_path"])
    soft_scores_output_path = Path(request["soft_scores_output_path"])
    method_params = dict(request.get("method_params") or {})

    metadata = _read_json(metadata_path)
    impl = decode_impl or decode_audio_file
    decoded_bits, soft_scores, detect_prob, status = impl(
        input_audio_path,
        metadata,
        method_params,
    )

    _save_bit_array(decoded_bits_output_path, decoded_bits)
    _save_float_array(soft_scores_output_path, soft_scores)
    _write_json(
        response_path,
        {
            "status": status,
            "detect_prob": float(detect_prob),
            "decoded_bits_path": str(decoded_bits_output_path),
            "soft_scores_path": str(soft_scores_output_path),
        },
    )


def _embed_job_response(job: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    metadata_output_path = Path(job["metadata_output_path"])
    _write_json(metadata_output_path, metadata)
    return {
        "status": "ok",
        "watermarked_audio_path": str(job["output_audio_path"]),
        "metadata_path": str(metadata_output_path),
    }


def _decode_job_response(
    job: dict[str, Any],
    decoded_bits: list[int],
    soft_scores: list[float],
    detect_prob: float,
    status: str,
) -> dict[str, Any]:
    decoded_bits_output_path = Path(job["decoded_bits_output_path"])
    soft_scores_output_path = Path(job["soft_scores_output_path"])
    _save_bit_array(decoded_bits_output_path, decoded_bits)
    _save_float_array(soft_scores_output_path, soft_scores)
    return {
        "status": status,
        "detect_prob": float(detect_prob),
        "decoded_bits_path": str(decoded_bits_output_path),
        "soft_scores_path": str(soft_scores_output_path),
    }


def handle_embed_batch_request(
    request_path: Path,
    response_path: Path,
    embed_batch_impl: Callable[[list[dict[str, Any]], dict[str, Any]], list[dict[str, Any]]] | None = None,
) -> None:
    request = _read_json(request_path)
    method_params = dict(request.get("method_params") or {})
    jobs: list[dict[str, Any]] = []
    for raw_job in list(request.get("jobs") or []):
        job = dict(raw_job)
        job["input_audio_path"] = Path(job["input_audio_path"])
        job["payload_bits_path"] = Path(job["payload_bits_path"])
        job["output_audio_path"] = Path(job["output_audio_path"])
        job["metadata_output_path"] = Path(job["metadata_output_path"])
        job["payload_bits"] = _load_payload_bits(job["payload_bits_path"])
        jobs.append(job)

    impl = embed_batch_impl or embed_audio_batch
    metadata_rows = impl(jobs, method_params)
    if len(metadata_rows) != len(jobs):
        raise RuntimeError(f"embed batch returned {len(metadata_rows)} results for {len(jobs)} jobs")
    _write_json(
        response_path,
        {
            "status": "ok",
            "results": [
                _embed_job_response(job, metadata)
                for job, metadata in zip(jobs, metadata_rows, strict=True)
            ],
        },
    )


def handle_decode_batch_request(
    request_path: Path,
    response_path: Path,
    decode_batch_impl: Callable[
        [list[dict[str, Any]], dict[str, Any]], list[tuple[list[int], list[float], float, str]]
    ] | None = None,
) -> None:
    request = _read_json(request_path)
    method_params = dict(request.get("method_params") or {})
    jobs: list[dict[str, Any]] = []
    for raw_job in list(request.get("jobs") or []):
        job = dict(raw_job)
        job["input_audio_path"] = Path(job["input_audio_path"])
        job["metadata_path"] = Path(job["metadata_path"])
        job["decoded_bits_output_path"] = Path(job["decoded_bits_output_path"])
        job["soft_scores_output_path"] = Path(job["soft_scores_output_path"])
        job["metadata"] = _read_json(job["metadata_path"])
        jobs.append(job)

    impl = decode_batch_impl or decode_audio_batch
    decoded_rows = impl(jobs, method_params)
    if len(decoded_rows) != len(jobs):
        raise RuntimeError(f"decode batch returned {len(decoded_rows)} results for {len(jobs)} jobs")
    _write_json(
        response_path,
        {
            "status": "ok",
            "results": [
                _decode_job_response(job, decoded_bits, soft_scores, detect_prob, status)
                for job, (decoded_bits, soft_scores, detect_prob, status) in zip(
                    jobs,
                    decoded_rows,
                    strict=True,
                )
            ],
        },
    )


def _load_audioseal_runtime():
    import torch
    import torchaudio
    from audioseal import AudioSeal

    return torch, torchaudio, AudioSeal


def _load_audio_for_model(
    input_audio_path: Path,
    target_sample_rate: int,
):
    _, torchaudio, _ = _load_audioseal_runtime()
    wav, sample_rate = torchaudio.load(input_audio_path)
    downmixed = False
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
        downmixed = True
    resampled = False
    if sample_rate != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sample_rate, target_sample_rate)
        resampled = True
    return wav, sample_rate, downmixed, resampled


def _batch_truncate_seconds(method_params: dict[str, Any]) -> float:
    return float(method_params.get("batch_truncate_seconds", 3.0))


def _truncate_audio_for_batch(wav: Any, sample_rate: int, method_params: dict[str, Any]) -> tuple[Any, bool, float]:
    seconds = _batch_truncate_seconds(method_params)
    if seconds <= 0:
        return wav, False, seconds
    max_samples = int(round(float(seconds) * int(sample_rate)))
    if max_samples <= 0 or int(wav.shape[-1]) <= max_samples:
        return wav, False, seconds
    return wav[..., :max_samples], True, seconds


def _load_models(device_name: str, method_params: dict[str, Any] | None = None):
    torch, _, AudioSeal = _load_audioseal_runtime()
    params = method_params or {}
    generator_checkpoint = str(params.get("generator_checkpoint", DEFAULT_GENERATOR_CHECKPOINT))
    detector_checkpoint = str(params.get("detector_checkpoint", DEFAULT_DETECTOR_CHECKPOINT))
    cache_key = (str(device_name), generator_checkpoint, detector_checkpoint)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    device = torch.device(device_name)
    if getattr(device, "type", None) == "cuda":
        torch.cuda.set_device(device)
    generator = AudioSeal.load_generator(generator_checkpoint, nbits=16, device=device)
    detector = AudioSeal.load_detector(detector_checkpoint, nbits=16, device=device)
    generator = generator.to(device)
    detector = detector.to(device)
    generator.eval()
    detector.eval()
    loaded = (torch, generator, detector)
    _MODEL_CACHE[cache_key] = loaded
    return loaded


def handle_session_request(request: dict[str, Any]) -> dict[str, Any]:
    operation = str(request["operation"])
    response_path = Path(request["response_path"])
    if operation == "embed":
        handle_embed_request_from_payload(request, response_path)
        return _read_json(response_path)
    if operation == "decode":
        handle_decode_request_from_payload(request, response_path)
        return _read_json(response_path)
    if operation == "embed_batch":
        handle_embed_batch_request_from_payload(request, response_path)
        return _read_json(response_path)
    if operation == "decode_batch":
        handle_decode_batch_request_from_payload(request, response_path)
        return _read_json(response_path)
    raise ValueError(f"Unknown session operation: {operation}")


def handle_embed_request_from_payload(request: dict[str, Any], response_path: Path) -> None:
    input_audio_path = Path(request["input_audio_path"])
    payload_bits_path = Path(request["payload_bits_path"])
    output_audio_path = Path(request["output_audio_path"])
    metadata_output_path = Path(request["metadata_output_path"])
    method_params = dict(request.get("method_params") or {})

    payload_bits = _load_payload_bits(payload_bits_path)
    metadata = embed_audio_file(input_audio_path, payload_bits, method_params, output_audio_path)
    _write_json(metadata_output_path, metadata)
    _write_json(
        response_path,
        {
            "status": "ok",
            "watermarked_audio_path": str(output_audio_path),
            "metadata_path": str(metadata_output_path),
        },
    )


def handle_decode_request_from_payload(request: dict[str, Any], response_path: Path) -> None:
    input_audio_path = Path(request["input_audio_path"])
    metadata_path = Path(request["metadata_path"])
    decoded_bits_output_path = Path(request["decoded_bits_output_path"])
    soft_scores_output_path = Path(request["soft_scores_output_path"])
    method_params = dict(request.get("method_params") or {})

    metadata = _read_json(metadata_path)
    decoded_bits, soft_scores, detect_prob, status = decode_audio_file(input_audio_path, metadata, method_params)
    _save_bit_array(decoded_bits_output_path, decoded_bits)
    _save_float_array(soft_scores_output_path, soft_scores)
    _write_json(
        response_path,
        {
            "status": status,
            "detect_prob": float(detect_prob),
            "decoded_bits_path": str(decoded_bits_output_path),
            "soft_scores_path": str(soft_scores_output_path),
        },
    )


def handle_embed_batch_request_from_payload(request: dict[str, Any], response_path: Path) -> None:
    request_path = response_path.parent / "embed_batch_request_from_payload.json"
    _write_json(request_path, request)
    handle_embed_batch_request(request_path, response_path)


def handle_decode_batch_request_from_payload(request: dict[str, Any], response_path: Path) -> None:
    request_path = response_path.parent / "decode_batch_request_from_payload.json"
    _write_json(request_path, request)
    handle_decode_batch_request(request_path, response_path)


def serve_json_requests(session_file: Path | None = None) -> int:
    if session_file is not None:
        count = int(session_file.read_text(encoding="utf-8") or "0") if session_file.exists() else 0
        session_file.write_text(str(count + 1), encoding="utf-8")
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        response = handle_session_request(request)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    return 0


def embed_audio_file(
    input_audio_path: Path,
    payload_bits: list[int],
    method_params: dict[str, Any],
    output_audio_path: Path,
) -> dict[str, Any]:
    torch, torchaudio, _ = _load_audioseal_runtime()
    device_name = str(method_params.get("device", "cpu"))
    target_sample_rate = int(method_params.get("sample_rate", 16000))
    torch_mod, generator, _ = _load_models(device_name, method_params)

    wav, original_sample_rate, downmixed, resampled = _load_audio_for_model(
        input_audio_path,
        target_sample_rate=target_sample_rate,
    )
    if wav.ndim != 2:
        raise ValueError(f"Expected audio tensor with 2 dims, got shape {tuple(wav.shape)}")
    batch = wav.unsqueeze(0).to(torch_mod.device(device_name))

    expected_nbits = int(getattr(getattr(generator, "msg_processor", None), "nbits", len(payload_bits)))
    if len(payload_bits) != expected_nbits:
        raise ValueError(
            f"AudioSeal generator expects {expected_nbits} bits, got {len(payload_bits)}"
        )

    alpha = float(method_params.get("alpha", 1.0))
    message = torch_mod.tensor(payload_bits, dtype=torch_mod.int32, device=batch.device).unsqueeze(0)
    with torch_mod.inference_mode():
        watermarked = generator(
            batch,
            sample_rate=target_sample_rate,
            message=message,
            alpha=alpha,
        )

    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output_audio_path, watermarked.squeeze(0).cpu(), target_sample_rate)

    return {
        "alpha": alpha,
        "detector_checkpoint": str(method_params.get("detector_checkpoint", DEFAULT_DETECTOR_CHECKPOINT)),
        "device": device_name,
        "downmixed_to_mono": downmixed,
        "input_audio_path": str(input_audio_path),
        "generator_checkpoint": str(method_params.get("generator_checkpoint", DEFAULT_GENERATOR_CHECKPOINT)),
        "model_nbits": expected_nbits,
        "model_sample_rate": target_sample_rate,
        "original_sample_rate": original_sample_rate,
        "output_audio_path": str(output_audio_path),
        "payload_bits": payload_bits,
        "payload_length": len(payload_bits),
        "resampled_to_model_rate": resampled,
        "sample_rate": target_sample_rate,
    }


def _pad_audio_batch(torch_mod, waves: list[Any]) -> tuple[Any, list[int]]:
    lengths = [int(wav.shape[-1]) for wav in waves]
    max_length = max(lengths, default=0)
    padded = []
    for wav in waves:
        if int(wav.shape[-1]) < max_length:
            wav = torch_mod.nn.functional.pad(wav, (0, max_length - int(wav.shape[-1])))
        padded.append(wav)
    return torch_mod.stack(padded, dim=0), lengths


def embed_audio_batch(jobs: list[dict[str, Any]], method_params: dict[str, Any]) -> list[dict[str, Any]]:
    if not jobs:
        return []
    torch, torchaudio, _ = _load_audioseal_runtime()
    device_name = str(method_params.get("device", "cpu"))
    target_sample_rate = int(method_params.get("sample_rate", 16000))
    torch_mod, generator, _ = _load_models(device_name, method_params)

    waves = []
    audio_info = []
    for job in jobs:
        wav, original_sample_rate, downmixed, resampled = _load_audio_for_model(
            Path(job["input_audio_path"]),
            target_sample_rate=target_sample_rate,
        )
        if wav.ndim != 2:
            raise ValueError(f"Expected audio tensor with 2 dims, got shape {tuple(wav.shape)}")
        wav, truncated, truncate_seconds = _truncate_audio_for_batch(wav, target_sample_rate, method_params)
        waves.append(wav)
        audio_info.append((original_sample_rate, downmixed, resampled, truncated, truncate_seconds))

    batch, lengths = _pad_audio_batch(torch_mod, waves)
    batch = batch.to(torch_mod.device(device_name))
    payload_rows = [list(job["payload_bits"]) for job in jobs]
    expected_nbits = int(getattr(getattr(generator, "msg_processor", None), "nbits", len(payload_rows[0])))
    for payload_bits in payload_rows:
        if len(payload_bits) != expected_nbits:
            raise ValueError(f"AudioSeal generator expects {expected_nbits} bits, got {len(payload_bits)}")

    alpha = float(method_params.get("alpha", 1.0))
    message = torch_mod.tensor(payload_rows, dtype=torch_mod.int32, device=batch.device)
    with torch_mod.inference_mode():
        watermarked = generator(
            batch,
            sample_rate=target_sample_rate,
            message=message,
            alpha=alpha,
        )

    outputs = []
    for index, job in enumerate(jobs):
        output_audio_path = Path(job["output_audio_path"])
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(output_audio_path, watermarked[index, :, : lengths[index]].cpu(), target_sample_rate)
        original_sample_rate, downmixed, resampled, truncated, truncate_seconds = audio_info[index]
        outputs.append(
            {
                "alpha": alpha,
                "detector_checkpoint": str(method_params.get("detector_checkpoint", DEFAULT_DETECTOR_CHECKPOINT)),
                "device": device_name,
                "downmixed_to_mono": bool(downmixed),
                "input_audio_path": str(job["input_audio_path"]),
                "generator_checkpoint": str(method_params.get("generator_checkpoint", DEFAULT_GENERATOR_CHECKPOINT)),
                "model_nbits": expected_nbits,
                "model_sample_rate": target_sample_rate,
                "original_sample_rate": int(original_sample_rate),
                "output_audio_path": str(output_audio_path),
                "payload_bits": payload_rows[index],
                "payload_length": len(payload_rows[index]),
                "resampled_to_model_rate": bool(resampled),
                "sample_rate": target_sample_rate,
                "batch_truncate_seconds": float(truncate_seconds),
                "truncated_for_batch": bool(truncated),
            }
        )
    return outputs


def decode_audio_file(
    input_audio_path: Path,
    metadata: dict[str, Any],
    method_params: dict[str, Any],
) -> tuple[list[int], list[float], float, str]:
    torch, torchaudio, _ = _load_audioseal_runtime()
    device_name = str(method_params.get("device", metadata.get("device", "cpu")))
    target_sample_rate = int(method_params.get("sample_rate", metadata.get("sample_rate", 16000)))
    torch_mod, _, detector = _load_models(device_name, method_params)

    wav, _, _, _ = _load_audio_for_model(
        input_audio_path,
        target_sample_rate=target_sample_rate,
    )
    if wav.ndim != 2:
        raise ValueError(f"Expected audio tensor with 2 dims, got shape {tuple(wav.shape)}")
    batch = wav.unsqueeze(0).to(torch_mod.device(device_name))

    payload_length = int(metadata["payload_length"])
    if payload_length > int(getattr(detector, "nbits", payload_length)):
        raise ValueError(
            f"AudioSeal detector exposes {detector.nbits} bits, cannot decode {payload_length}"
        )
    message_threshold = float(method_params.get("message_threshold", 0.5))
    detection_threshold = float(method_params.get("detection_threshold", 0.5))
    with torch_mod.inference_mode():
        detect_prob, decoded_message = detector.detect_watermark(
            batch,
            sample_rate=target_sample_rate,
            message_threshold=message_threshold,
            detection_threshold=detection_threshold,
        )
        _, soft_message = detector(batch, sample_rate=target_sample_rate)

    decoded_bits = [
        int(x)
        for x in decoded_message.squeeze(0).detach().cpu().to(dtype=torch_mod.int32).tolist()[:payload_length]
    ]
    soft_scores = [
        float(x)
        for x in soft_message.squeeze(0).detach().cpu().tolist()[:payload_length]
    ]
    detect_value = float(detect_prob.squeeze(0).detach().cpu().item())
    status = "detected" if detect_value >= detection_threshold else "not_detected"
    return decoded_bits, soft_scores, detect_value, status


def decode_audio_batch(
    jobs: list[dict[str, Any]],
    method_params: dict[str, Any],
) -> list[tuple[list[int], list[float], float, str]]:
    if not jobs:
        return []
    torch, _, _ = _load_audioseal_runtime()
    first_metadata = dict(jobs[0]["metadata"])
    device_name = str(method_params.get("device", first_metadata.get("device", "cpu")))
    target_sample_rate = int(method_params.get("sample_rate", first_metadata.get("sample_rate", 16000)))
    torch_mod, _, detector = _load_models(device_name, method_params)

    waves = []
    payload_lengths = []
    for job in jobs:
        metadata = dict(job["metadata"])
        payload_length = int(metadata["payload_length"])
        if payload_length > int(getattr(detector, "nbits", payload_length)):
            raise ValueError(f"AudioSeal detector exposes {detector.nbits} bits, cannot decode {payload_length}")
        wav, _, _, _ = _load_audio_for_model(
            Path(job["input_audio_path"]),
            target_sample_rate=target_sample_rate,
        )
        if wav.ndim != 2:
            raise ValueError(f"Expected audio tensor with 2 dims, got shape {tuple(wav.shape)}")
        wav, _, _ = _truncate_audio_for_batch(wav, target_sample_rate, method_params)
        waves.append(wav)
        payload_lengths.append(payload_length)

    batch, lengths = _pad_audio_batch(torch_mod, waves)
    batch = batch.to(torch_mod.device(device_name))
    message_threshold = float(method_params.get("message_threshold", 0.5))
    detection_threshold = float(method_params.get("detection_threshold", 0.5))
    with torch_mod.inference_mode():
        if hasattr(detector, "detector"):
            detector_input = batch
            normalizer = getattr(detector, "normalizer", None)
            if normalizer is not None:
                detector_input = normalizer.loudness_normalization(detector_input)
            raw_result = detector.detector(detector_input)
            raw_result[:, :2, :] = torch_mod.softmax(raw_result[:, :2, :], dim=1)
            detection_scores = raw_result[:, 1, :]
            message_logits = raw_result[:, 2:, :]
            soft_message = None
        else:
            detection_result, soft_message = detector(batch, sample_rate=target_sample_rate)
            detection_scores = detection_result[:, 1, :]
            message_logits = None

    outputs = []
    detection_cpu = detection_scores.detach().cpu()
    message_logits_cpu = None if message_logits is None else message_logits.detach().cpu()
    soft_cpu = None if soft_message is None else soft_message.detach().cpu()
    max_audio_length = max(lengths, default=1)
    detection_frame_count = int(detection_cpu.shape[-1])
    for index, payload_length in enumerate(payload_lengths):
        valid_frames = int(round(detection_frame_count * int(lengths[index]) / max_audio_length))
        valid_frames = max(1, min(valid_frames, detection_frame_count))
        detection_values = np.asarray(detection_cpu[index, :valid_frames].tolist(), dtype=np.float32)
        detect_value = float(np.count_nonzero(detection_values > detection_threshold) / detection_values.size)
        if message_logits_cpu is not None:
            logits = np.asarray(message_logits_cpu[index, :, :valid_frames].tolist(), dtype=np.float32)
            soft_values = 1.0 / (1.0 + np.exp(-logits.mean(axis=-1)))
        else:
            soft_values = np.asarray(soft_cpu[index].tolist(), dtype=np.float32)
        soft_scores = [float(x) for x in soft_values[:payload_length]]
        decoded_bits = [int(float(x) > message_threshold) for x in soft_scores]
        status = "detected" if detect_value >= detection_threshold else "not_detected"
        outputs.append((decoded_bits, soft_scores, detect_value, status))
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AudioSeal file-based worker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("embed", "decode", "embed_batch", "decode_batch"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--request-json", required=True)
        sub.add_argument("--response-json", required=True)

    serve_parser = subparsers.add_parser("serve-json")
    serve_parser.add_argument("--session-file", required=False)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    if args.command == "serve-json":
        session_file = None if not getattr(args, "session_file", None) else Path(args.session_file)
        return serve_json_requests(session_file=session_file)

    request_path = Path(args.request_json)
    response_path = Path(args.response_json)

    if args.command == "embed":
        handle_embed_request(request_path, response_path, embed_audio_file)
        return 0
    if args.command == "decode":
        handle_decode_request(request_path, response_path, decode_audio_file)
        return 0
    if args.command == "embed_batch":
        handle_embed_batch_request(request_path, response_path, embed_audio_batch)
        return 0
    if args.command == "decode_batch":
        handle_decode_batch_request(request_path, response_path, decode_audio_batch)
        return 0
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
