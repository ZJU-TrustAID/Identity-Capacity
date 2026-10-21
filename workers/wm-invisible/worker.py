from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
INVISIBLE_WATERMARK_ROOT = REPO_ROOT / "third_party" / "invisible-watermark"
RIVAGAN_MODEL_DIR = INVISIBLE_WATERMARK_ROOT / "imwatermark"

DEFAULT_PAYLOAD_LENGTHS = {
    "dwtDct": 64,
    "rivaGan": 32,
}
_RIVAGAN_SESSIONS: dict[str, Any] = {}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_payload_bits(path: Path) -> np.ndarray:
    bits = np.asarray(np.load(path), dtype=np.uint8).reshape(-1)
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("payload bits must be binary")
    return bits


def _save_bits(path: Path, bits: list[int] | np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(bits, dtype=np.uint8).reshape(-1))


def _save_scores(path: Path, scores: list[float] | np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(scores, dtype=np.float32).reshape(-1))


def _normalize_method(params: dict[str, Any] | None) -> str:
    raw = str((params or {}).get("method") or "dwtDct")
    aliases = {
        "dwtdct": "dwtDct",
        "dwt_dct": "dwtDct",
        "dwt-dct": "dwtDct",
        "maxdct": "dwtDct",
        "rivagan": "rivaGan",
        "riva_gan": "rivaGan",
        "riva-gan": "rivaGan",
    }
    normalized = aliases.get(raw.lower(), raw)
    if normalized not in DEFAULT_PAYLOAD_LENGTHS:
        raise ValueError(f"Unsupported invisible-watermark method: {raw!r}")
    return normalized


def _payload_length_for_method(method: str, params: dict[str, Any] | None = None) -> int:
    if params and params.get("payload_length") is not None:
        return int(params["payload_length"])
    return DEFAULT_PAYLOAD_LENGTHS[method]


def _load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _resize_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return np.asarray(image.resize(size, resampling), dtype=np.uint8)


def _save_rgb(path: str | Path, array: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(target)


def _ensure_large_enough(image: np.ndarray, method: str) -> None:
    height, width = image.shape[:2]
    if height * width < 256 * 256:
        raise ValueError(f"{method} requires image area >= 256*256, got {width}x{height}")


def _preprocess_dwt_dct_image(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if (width, height) == (256, 256):
        return image
    return _resize_rgb(image, (256, 256))


def _rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb, dtype=np.float32)
    matrix = np.array(
        [
            [0.299, 0.587, 0.114],
            [-0.14713, -0.28886, 0.436],
            [0.615, -0.51499, -0.10001],
        ],
        dtype=np.float32,
    )
    yuv = array @ matrix.T
    yuv[:, :, 1:] += 128.0
    return yuv


def _yuv_to_rgb(yuv: np.ndarray) -> np.ndarray:
    array = np.asarray(yuv, dtype=np.float32).copy()
    array[:, :, 1:] -= 128.0
    matrix = np.array(
        [
            [1.0, 0.0, 1.13983],
            [1.0, -0.39465, -0.58060],
            [1.0, 2.03211, 0.0],
        ],
        dtype=np.float32,
    )
    rgb = array @ matrix.T
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def _dct2(block: np.ndarray) -> np.ndarray:
    from scipy.fftpack import dct

    return dct(dct(block.astype(np.float32), axis=0, norm="ortho"), axis=1, norm="ortho")


def _idct2(block: np.ndarray) -> np.ndarray:
    from scipy.fftpack import idct

    return idct(idct(block.astype(np.float32), axis=0, norm="ortho"), axis=1, norm="ortho")


def _dwt2(frame: np.ndarray):
    import pywt

    return pywt.dwt2(frame, "haar")


def _idwt2(coefficients: Any) -> np.ndarray:
    import pywt

    return np.asarray(pywt.idwt2(coefficients, "haar"), dtype=np.float32)


def _parse_scales(params: dict[str, Any] | None, default: list[float] | None = None) -> list[float]:
    default_values = [0.0, 36.0, 36.0] if default is None else default
    raw = (params or {}).get("scales", default_values)
    if isinstance(raw, str):
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        values = [float(value) for value in raw]
    if len(values) != 3:
        raise ValueError(f"scales must contain three channel values, got {values!r}")
    return values


def _embed_dwt_dct(rgb: np.ndarray, payload_bits: np.ndarray, params: dict[str, Any] | None) -> np.ndarray:
    scales = _parse_scales(params)
    block_size = int((params or {}).get("block", 4))
    if block_size <= 1:
        raise ValueError("block must be greater than 1")
    yuv = _rgb_to_yuv(rgb)
    height, width = yuv.shape[:2]
    usable_h = height // 4 * 4
    usable_w = width // 4 * 4
    payload = np.asarray(payload_bits, dtype=np.uint8).reshape(-1)
    for channel in range(2):
        scale = float(scales[channel])
        if scale <= 0:
            continue
        ca1, (h1, v1, d1) = _dwt2(yuv[:usable_h, :usable_w, channel])
        rows, cols = ca1.shape
        cursor = 0
        for row in range(rows // block_size):
            for col in range(cols // block_size):
                block_slice = (
                    slice(row * block_size, row * block_size + block_size),
                    slice(col * block_size, col * block_size + block_size),
                )
                dct_block = _dct2(ca1[block_slice])
                flat = np.abs(dct_block.reshape(-1)[1:])
                pos = int(np.argmax(flat) + 1)
                i, j = divmod(pos, block_size)
                value = float(dct_block[i, j])
                sign = -1.0 if value < 0 else 1.0
                magnitude = abs(value)
                bit = int(payload[cursor % payload.size])
                dct_block[i, j] = sign * ((magnitude // scale + 0.25 + 0.5 * bit) * scale)
                ca1[block_slice] = _idct2(dct_block)
                cursor += 1
        restored = _idwt2((ca1, (h1, v1, d1)))
        yuv[:usable_h, :usable_w, channel] = restored[:usable_h, :usable_w]
    return _yuv_to_rgb(yuv)


def _decode_dwt_dct(rgb: np.ndarray, payload_length: int, params: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    scales = _parse_scales(params)
    block_size = int((params or {}).get("block", 4))
    yuv = _rgb_to_yuv(rgb)
    height, width = yuv.shape[:2]
    usable_h = height // 4 * 4
    usable_w = width // 4 * 4
    scores: list[list[float]] = [[] for _ in range(int(payload_length))]
    for channel in range(2):
        scale = float(scales[channel])
        if scale <= 0:
            continue
        ca1, _ = _dwt2(yuv[:usable_h, :usable_w, channel])
        rows, cols = ca1.shape
        cursor = 0
        for row in range(rows // block_size):
            for col in range(cols // block_size):
                block = ca1[
                    row * block_size : row * block_size + block_size,
                    col * block_size : col * block_size + block_size,
                ]
                dct_block = _dct2(block)
                flat = np.abs(dct_block.reshape(-1)[1:])
                pos = int(np.argmax(flat) + 1)
                i, j = divmod(pos, block_size)
                value = abs(float(dct_block[i, j]))
                scores[cursor % int(payload_length)].append(1.0 if (value % scale) > (0.5 * scale) else 0.0)
                cursor += 1
    soft_scores = np.asarray(
        [float(np.mean(row)) if row else 0.0 for row in scores],
        dtype=np.float32,
    )
    decoded = (soft_scores > 0.5).astype(np.uint8)
    return decoded, soft_scores


def _load_rivagan_sessions() -> dict[str, Any]:
    if _RIVAGAN_SESSIONS:
        return _RIVAGAN_SESSIONS
    try:
        import onnxruntime
    except ImportError as exc:
        raise ImportError(
            "RivaGAN requires onnxruntime in the worker environment. "
            "Use a wm-* environment that has onnxruntime installed."
        ) from exc

    _RIVAGAN_SESSIONS["encoder"] = onnxruntime.InferenceSession(
        str(RIVAGAN_MODEL_DIR / "rivagan_encoder.onnx"),
        providers=["CPUExecutionProvider"],
    )
    _RIVAGAN_SESSIONS["decoder"] = onnxruntime.InferenceSession(
        str(RIVAGAN_MODEL_DIR / "rivagan_decoder.onnx"),
        providers=["CPUExecutionProvider"],
    )
    return _RIVAGAN_SESSIONS


def _rivagan_input(rgb: np.ndarray) -> np.ndarray:
    frame = np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0
    return np.transpose(frame[None, ...], (0, 3, 1, 2))[:, :, None, :, :]


def _embed_rivagan(rgb: np.ndarray, payload_bits: np.ndarray, params: dict[str, Any] | None) -> np.ndarray:
    if payload_bits.size != 32:
        raise ValueError(f"rivaGan requires exactly 32 bits, got {payload_bits.size}")
    sessions = _load_rivagan_sessions()
    outputs = sessions["encoder"].run(
        None,
        {
            "frame": _rivagan_input(rgb),
            "data": np.asarray(payload_bits, dtype=np.float32).reshape(1, -1),
        },
    )
    watermarked = np.clip(outputs[0], -1.0, 1.0)
    frame = (np.transpose(watermarked[0, :, 0, :, :], (1, 2, 0)) + 1.0) * 127.5
    return np.clip(np.round(frame), 0, 255).astype(np.uint8)


def _decode_rivagan(rgb: np.ndarray, payload_length: int, params: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    if int(payload_length) != 32:
        raise ValueError(f"rivaGan requires exactly 32 bits, got {payload_length}")
    threshold = float((params or {}).get("threshold", 0.52))
    sessions = _load_rivagan_sessions()
    outputs = sessions["decoder"].run(None, {"frame": _rivagan_input(rgb)})
    scores = np.asarray(outputs[0][0], dtype=np.float32).reshape(-1)[:32]
    return (scores > threshold).astype(np.uint8), scores


def embed_image(
    input_image_path: str | Path,
    payload_bits: list[int],
    method_params: dict[str, Any] | None,
    output_image_path: str | Path,
) -> dict[str, Any]:
    params = dict(method_params or {})
    method = _normalize_method(params)
    payload_array = np.asarray(payload_bits, dtype=np.uint8).reshape(-1)
    expected_length = _payload_length_for_method(method, params)
    if payload_array.size != expected_length:
        raise ValueError(f"{method} expected {expected_length} bits, received {payload_array.size}")
    if np.any((payload_array != 0) & (payload_array != 1)):
        raise ValueError("payload bits must be binary")

    image = _load_rgb(input_image_path)
    original_width = int(image.shape[1])
    original_height = int(image.shape[0])
    if method == "dwtDct":
        image = _preprocess_dwt_dct_image(image)
        watermarked = _embed_dwt_dct(image, payload_array, params)
    elif method == "rivaGan":
        _ensure_large_enough(image, method)
        watermarked = _embed_rivagan(image, payload_array, params)
    else:  # pragma: no cover - guarded by _normalize_method
        raise ValueError(f"Unsupported method: {method}")
    _save_rgb(output_image_path, watermarked)
    return {
        "status": "ok",
        "method": method,
        "payload_length": int(payload_array.size),
        "input_image_size": [original_width, original_height],
        "preprocessed_image_size": [int(image.shape[1]), int(image.shape[0])],
        "output_image_path": str(output_image_path),
        "scales": _parse_scales(params) if method == "dwtDct" else None,
        "threshold": float(params.get("threshold", 0.52)) if method == "rivaGan" else None,
    }


def decode_image(
    input_image_path: str | Path,
    metadata: dict[str, Any],
    method_params: dict[str, Any] | None,
) -> tuple[list[int], list[float], float, str]:
    params = {**dict(metadata.get("method_params") or {}), **dict(method_params or {})}
    if "method" not in params and metadata.get("method"):
        params["method"] = metadata["method"]
    method = _normalize_method(params)
    payload_length = int(metadata.get("payload_length") or _payload_length_for_method(method, params))
    image = _load_rgb(input_image_path)
    if method == "dwtDct":
        image = _preprocess_dwt_dct_image(image)
        decoded, soft_scores = _decode_dwt_dct(image, payload_length, params)
    elif method == "rivaGan":
        _ensure_large_enough(image, method)
        decoded, soft_scores = _decode_rivagan(image, payload_length, params)
    else:  # pragma: no cover - guarded by _normalize_method
        raise ValueError(f"Unsupported method: {method}")
    confidence = np.maximum(soft_scores, 1.0 - soft_scores).mean() if soft_scores.size else 0.0
    return decoded.astype(np.uint8).tolist(), soft_scores.astype(np.float32).tolist(), float(confidence), "decoded"


def embed_image_batch(jobs: list[dict[str, Any]], method_params: dict[str, Any] | None) -> list[dict[str, Any]]:
    results = []
    for job in jobs:
        results.append(
            embed_image(
                job["input_image_path"],
                list(job["payload_bits"]),
                method_params,
                job["output_image_path"],
            )
        )
    return results


def decode_image_batch(
    jobs: list[dict[str, Any]],
    method_params: dict[str, Any] | None,
) -> list[tuple[list[int], list[float], float, str]]:
    return [
        decode_image(job["input_image_path"], dict(job["metadata"]), method_params)
        for job in jobs
    ]


def _write_embed_result(job: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(metadata)
    metadata["method_params"] = dict(job.get("method_params") or {})
    metadata_path = Path(job["metadata_output_path"])
    save_json(metadata_path, metadata)
    return {
        "status": metadata.get("status", "ok"),
        "watermarked_image_path": str(job["output_image_path"]),
        "metadata_path": str(metadata_path),
    }


def _write_decode_result(
    job: dict[str, Any],
    decoded_bits: list[int],
    soft_scores: list[float],
    detect_prob: float,
    status: str,
) -> dict[str, Any]:
    decoded_bits_path = Path(job["decoded_bits_output_path"])
    soft_scores_path = Path(job["soft_scores_output_path"])
    _save_bits(decoded_bits_path, decoded_bits)
    _save_scores(soft_scores_path, soft_scores)
    return {
        "status": status,
        "detect_prob": float(detect_prob),
        "decoded_bits_path": str(decoded_bits_path),
        "soft_scores_path": str(soft_scores_path),
    }


def handle_embed_request(
    request_path: Path,
    response_path: Path,
    embed_impl: Callable[[str, list[int], dict[str, Any], str], dict[str, Any]] | None = None,
) -> None:
    request = load_json(request_path)
    method_params = dict(request.get("method_params") or {})
    job = {
        "input_image_path": request["input_image_path"],
        "payload_bits_path": request["payload_bits_path"],
        "output_image_path": request["output_image_path"],
        "metadata_output_path": request["metadata_output_path"],
        "method_params": method_params,
    }
    payload_bits = _load_payload_bits(Path(job["payload_bits_path"])).tolist()
    impl = embed_image if embed_impl is None else embed_impl
    metadata = impl(
        job["input_image_path"],
        payload_bits,
        method_params,
        job["output_image_path"],
    )
    response = _write_embed_result(job, metadata)
    save_json(response_path, response)


def handle_decode_request(
    request_path: Path,
    response_path: Path,
    decode_impl: Callable[[str, dict[str, Any], dict[str, Any]], tuple[list[int], list[float], float, str]] | None = None,
) -> None:
    request = load_json(request_path)
    metadata = load_json(Path(request["metadata_path"]))
    method_params = dict(request.get("method_params") or {})
    impl = decode_image if decode_impl is None else decode_impl
    decoded_bits, soft_scores, detect_prob, status = impl(
        request["input_image_path"],
        metadata,
        method_params,
    )
    response = _write_decode_result(
        request,
        decoded_bits,
        soft_scores,
        detect_prob,
        status,
    )
    save_json(response_path, response)


def handle_embed_batch_request(
    request_path: Path,
    response_path: Path,
    embed_batch_impl: Callable[[list[dict[str, Any]], dict[str, Any]], list[dict[str, Any]]] | None = None,
) -> None:
    request = load_json(request_path)
    method_params = dict(request.get("method_params") or {})
    jobs: list[dict[str, Any]] = []
    for raw_job in list(request.get("jobs") or []):
        job = dict(raw_job)
        job["payload_bits"] = _load_payload_bits(Path(job["payload_bits_path"])).tolist()
        job["method_params"] = method_params
        jobs.append(job)
    impl = embed_image_batch if embed_batch_impl is None else embed_batch_impl
    metadata_rows = impl(jobs, method_params)
    if len(metadata_rows) != len(jobs):
        raise RuntimeError(f"embed batch returned {len(metadata_rows)} results for {len(jobs)} jobs")
    response = {
        "status": "ok",
        "results": [_write_embed_result(job, metadata) for job, metadata in zip(jobs, metadata_rows, strict=True)],
    }
    save_json(response_path, response)


def handle_decode_batch_request(
    request_path: Path,
    response_path: Path,
    decode_batch_impl: Callable[[list[dict[str, Any]], dict[str, Any]], list[tuple[list[int], list[float], float, str]]] | None = None,
) -> None:
    request = load_json(request_path)
    method_params = dict(request.get("method_params") or {})
    jobs: list[dict[str, Any]] = []
    for raw_job in list(request.get("jobs") or []):
        job = dict(raw_job)
        job["metadata"] = load_json(Path(job["metadata_path"]))
        jobs.append(job)
    impl = decode_image_batch if decode_batch_impl is None else decode_batch_impl
    decoded_rows = impl(jobs, method_params)
    if len(decoded_rows) != len(jobs):
        raise RuntimeError(f"decode batch returned {len(decoded_rows)} results for {len(jobs)} jobs")
    response = {
        "status": "ok",
        "results": [
            _write_decode_result(job, decoded_bits, soft_scores, detect_prob, status)
            for job, (decoded_bits, soft_scores, detect_prob, status) in zip(jobs, decoded_rows, strict=True)
        ],
    }
    save_json(response_path, response)


def handle_session_request(request: dict[str, Any], *, default_method: str | None = None) -> dict[str, Any]:
    if default_method:
        request = dict(request)
        method_params = dict(request.get("method_params") or {})
        method_params.setdefault("method", default_method)
        request["method_params"] = method_params
    operation = str(request["operation"])
    response_path = Path(request["response_path"])
    request_path = response_path.parent / f"{operation}_request_from_payload.json"
    save_json(request_path, request)
    if operation == "embed":
        handle_embed_request(request_path, response_path)
    elif operation == "decode":
        handle_decode_request(request_path, response_path)
    elif operation == "embed_batch":
        handle_embed_batch_request(request_path, response_path)
    elif operation == "decode_batch":
        handle_decode_batch_request(request_path, response_path)
    else:
        raise ValueError(f"Unknown session operation: {operation}")
    return load_json(response_path)


def serve_json_requests(session_file: Path | None = None, default_method: str | None = None) -> int:
    if session_file is not None:
        count = int(session_file.read_text(encoding="utf-8") or "0") if session_file.exists() else 0
        session_file.parent.mkdir(parents=True, exist_ok=True)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="File-based invisible-watermark worker.")
    parser.add_argument("operation", choices=["embed", "decode", "embed_batch", "decode_batch", "serve-json"])
    parser.add_argument("--request-json", required=False)
    parser.add_argument("--response-json", required=False)
    parser.add_argument("--session-file", required=False)
    parser.add_argument("--method", choices=["dwtDct", "rivaGan"], required=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.operation == "serve-json":
        serve_json_requests(
            session_file=None if not args.session_file else Path(args.session_file),
            default_method=args.method,
        )
        return
    if not args.request_json or not args.response_json:
        raise SystemExit("--request-json and --response-json are required")
    request_path = Path(args.request_json)
    response_path = Path(args.response_json)
    if args.operation == "embed":
        handle_embed_request(request_path, response_path)
    elif args.operation == "decode":
        handle_decode_request(request_path, response_path)
    elif args.operation == "embed_batch":
        handle_embed_batch_request(request_path, response_path)
    else:
        handle_decode_batch_request(request_path, response_path)


if __name__ == "__main__":
    main()
