from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image


DEFAULT_PAYLOAD_LENGTH = 64
DEFAULT_IMAGE_SIZE = 256
DEFAULT_NUM_BLOCKS_PER_SIDE = 32
DEFAULT_SEED = 100
DEFAULT_MARGIN = 0.01


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_plain(payload), indent=2, sort_keys=True), encoding="utf-8")


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


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
    raw = str((params or {}).get("method") or "svd_chang")
    normalized = raw.lower().replace("-", "_")
    aliases = {
        "chang": "svd_chang",
        "chang_svd": "svd_chang",
        "ctl_2005": "svd_chang",
        "svd_chang": "svd_chang",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported traditional SVD image watermark method: {raw!r}")
    return aliases[normalized]


def _payload_length(params: dict[str, Any] | None, payload_size: int | None = None) -> int:
    if payload_size is not None:
        return int(payload_size)
    if params and params.get("payload_length") is not None:
        return int(params["payload_length"])
    return DEFAULT_PAYLOAD_LENGTH


def _image_size(params: dict[str, Any] | None) -> int:
    value = int((params or {}).get("image_size", DEFAULT_IMAGE_SIZE))
    if value <= 0:
        raise ValueError(f"image_size must be positive, got {value}")
    return value


def _num_blocks_per_side(params: dict[str, Any] | None) -> int:
    value = int((params or {}).get("num_blocks_per_side", DEFAULT_NUM_BLOCKS_PER_SIDE))
    if value <= 0:
        raise ValueError(f"num_blocks_per_side must be positive, got {value}")
    return value


def _seed(params: dict[str, Any] | None) -> int:
    return int((params or {}).get("seed", DEFAULT_SEED))


def _margin(params: dict[str, Any] | None) -> float:
    value = float((params or {}).get("t", (params or {}).get("margin", DEFAULT_MARGIN)))
    if value <= 0.0:
        raise ValueError(f"t/margin must be positive, got {value}")
    return value


def _preprocess_image(path: str | Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (int(image_size), int(image_size)):
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        image = image.resize((int(image_size), int(image_size)), resampling)
    return np.asarray(image, dtype=np.float64)


def _save_grayscale_as_rgb(path: str | Path, array: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.round(np.asarray(array, dtype=np.float64)), 0, 255).astype(np.uint8)
    Image.fromarray(clipped, mode="L").convert("RGB").save(target)


def _split_blocks(image: np.ndarray, num_blocks_per_side: int) -> tuple[np.ndarray, int]:
    size = int(image.shape[0])
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError(f"expected square grayscale image, got shape={image.shape}")
    if size % int(num_blocks_per_side) != 0:
        raise ValueError(
            f"image_size {size} must be divisible by num_blocks_per_side {num_blocks_per_side}"
        )
    block_size = size // int(num_blocks_per_side)
    if block_size < 3:
        raise ValueError(
            "Chang SVD requires block_size >= 3 because it compares U[1,0] and U[2,0]"
        )
    blocks = np.zeros((num_blocks_per_side * num_blocks_per_side, block_size, block_size), dtype=np.float64)
    index = 0
    for row in range(num_blocks_per_side):
        for col in range(num_blocks_per_side):
            blocks[index] = image[
                row * block_size : (row + 1) * block_size,
                col * block_size : (col + 1) * block_size,
            ]
            index += 1
    return blocks, block_size


def _merge_blocks(blocks: np.ndarray, image_size: int, num_blocks_per_side: int, block_size: int) -> np.ndarray:
    image = np.zeros((int(image_size), int(image_size)), dtype=np.float64)
    index = 0
    for row in range(int(num_blocks_per_side)):
        for col in range(int(num_blocks_per_side)):
            image[
                row * block_size : (row + 1) * block_size,
                col * block_size : (col + 1) * block_size,
            ] = blocks[index]
            index += 1
    return image


def _d_features(blocks: np.ndarray) -> np.ndarray:
    features = np.zeros(blocks.shape[0], dtype=np.int32)
    for index, block in enumerate(blocks):
        _u, singular_values, _vh = np.linalg.svd(block, full_matrices=False)
        features[index] = int(np.count_nonzero(np.round(singular_values, 10) != 0.0))
    return features


def _select_block_ids(blocks: np.ndarray, payload_length: int, seed: int) -> np.ndarray:
    total_blocks = int(blocks.shape[0])
    if int(payload_length) > total_blocks:
        raise ValueError(
            f"payload_length {payload_length} exceeds candidate blocks {total_blocks}"
        )
    features = _d_features(blocks)
    max_feature = int(np.max(features)) if features.size else 0
    candidates = np.flatnonzero(features == max_feature)
    if int(payload_length) > int(candidates.size):
        raise ValueError(
            f"payload_length {payload_length} exceeds candidate blocks {int(candidates.size)} "
            f"with max D-feature {max_feature}"
        )
    rng = np.random.default_rng(int(seed))
    return rng.choice(candidates, size=int(payload_length), replace=False).astype(np.int32)


def _set_signed_abs(value: float, magnitude: float) -> float:
    sign = -1.0 if float(value) < 0.0 else 1.0
    return sign * max(0.0, float(magnitude))


def _adjust_u_feature(u_matrix: np.ndarray, bit: int, margin: float) -> np.ndarray:
    adjusted = np.asarray(u_matrix, dtype=np.float64).copy()
    current_a = abs(float(adjusted[1, 0]))
    current_b = abs(float(adjusted[2, 0]))
    total = max(current_a + current_b, float(margin) + 1e-12)
    target_delta = float(margin) if int(bit) == 1 else -float(margin)
    target_a = (total + target_delta) / 2.0
    target_b = (total - target_delta) / 2.0
    if target_a < 0.0 or target_b < 0.0:
        lift = abs(min(target_a, target_b))
        target_a += lift
        target_b += lift
    adjusted[1, 0] = _set_signed_abs(adjusted[1, 0], target_a)
    adjusted[2, 0] = _set_signed_abs(adjusted[2, 0], target_b)
    return adjusted


def _embed_chang_svd(
    image: np.ndarray,
    payload_bits: np.ndarray,
    *,
    num_blocks_per_side: int,
    seed: int,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    payload = np.asarray(payload_bits, dtype=np.uint8).reshape(-1)
    blocks, block_size = _split_blocks(image, int(num_blocks_per_side))
    block_ids = _select_block_ids(blocks, int(payload.size), int(seed))
    watermarked_blocks = np.asarray(blocks, dtype=np.float64).copy()
    for payload_index, block_index in enumerate(block_ids):
        u_matrix, singular_values, vh_matrix = np.linalg.svd(
            watermarked_blocks[int(block_index)],
            full_matrices=False,
        )
        adjusted_u = _adjust_u_feature(u_matrix, int(payload[payload_index]), float(margin))
        watermarked_blocks[int(block_index)] = adjusted_u @ np.diag(singular_values) @ vh_matrix
    return (
        _merge_blocks(watermarked_blocks, int(image.shape[0]), int(num_blocks_per_side), int(block_size)),
        block_ids,
        block_size,
    )


def _decode_chang_svd(
    image: np.ndarray,
    block_ids: list[int] | np.ndarray,
    *,
    num_blocks_per_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    blocks, _block_size = _split_blocks(image, int(num_blocks_per_side))
    decoded: list[int] = []
    scores: list[float] = []
    for block_index in np.asarray(block_ids, dtype=np.int32).reshape(-1):
        u_matrix, _singular_values, _vh_matrix = np.linalg.svd(blocks[int(block_index)], full_matrices=False)
        delta = abs(float(u_matrix[1, 0])) - abs(float(u_matrix[2, 0]))
        decoded.append(1 if delta > 0.0 else 0)
        scores.append(1.0 / (1.0 + np.exp(-50.0 * delta)))
    return np.asarray(decoded, dtype=np.uint8), np.asarray(scores, dtype=np.float32)


def embed_image(
    input_image_path: str | Path,
    payload_bits: list[int],
    method_params: dict[str, Any] | None,
    output_image_path: str | Path,
) -> dict[str, Any]:
    params = dict(method_params or {})
    method = _normalize_method(params)
    payload_array = np.asarray(payload_bits, dtype=np.uint8).reshape(-1)
    if np.any((payload_array != 0) & (payload_array != 1)):
        raise ValueError("payload bits must be binary")
    expected_length = _payload_length(params)
    if payload_array.size != expected_length:
        raise ValueError(f"{method} expected {expected_length} bits, received {payload_array.size}")
    image_size = _image_size(params)
    num_blocks = _num_blocks_per_side(params)
    seed = _seed(params)
    margin = _margin(params)
    image = _preprocess_image(input_image_path, image_size)
    watermarked, block_ids, block_size = _embed_chang_svd(
        image,
        payload_array,
        num_blocks_per_side=num_blocks,
        seed=seed,
        margin=margin,
    )
    _save_grayscale_as_rgb(output_image_path, watermarked)
    original = Image.open(input_image_path)
    return {
        "status": "ok",
        "method": method,
        "payload_length": int(payload_array.size),
        "input_image_size": [int(original.size[0]), int(original.size[1])],
        "preprocessed_image_size": [int(image_size), int(image_size)],
        "num_blocks_per_side": int(num_blocks),
        "block_size": int(block_size),
        "seed": int(seed),
        "t": float(margin),
        "block_ids": block_ids.astype(int).tolist(),
        "output_image_path": str(output_image_path),
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
    if method != "svd_chang":
        raise ValueError(f"Unsupported method: {method}")
    image_size = int(metadata.get("preprocessed_image_size", [_image_size(params), _image_size(params)])[0])
    num_blocks = int(metadata.get("num_blocks_per_side", _num_blocks_per_side(params)))
    block_ids = list(metadata.get("block_ids") or [])
    payload_length = int(metadata.get("payload_length") or len(block_ids))
    if len(block_ids) != payload_length:
        raise ValueError(f"metadata block_ids length {len(block_ids)} does not match payload_length {payload_length}")
    image = _preprocess_image(input_image_path, image_size)
    decoded, scores = _decode_chang_svd(image, block_ids, num_blocks_per_side=num_blocks)
    return decoded.astype(np.uint8).tolist(), scores.astype(np.float32).tolist(), 1.0, "decoded"


def embed_image_batch(jobs: list[dict[str, Any]], method_params: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [
        embed_image(
            job["input_image_path"],
            list(job["payload_bits"]),
            method_params,
            job["output_image_path"],
        )
        for job in jobs
    ]


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
    metadata = impl(job["input_image_path"], payload_bits, method_params, job["output_image_path"])
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
    decoded_bits, soft_scores, detect_prob, status = impl(request["input_image_path"], metadata, method_params)
    response = _write_decode_result(request, decoded_bits, soft_scores, detect_prob, status)
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
    parser = argparse.ArgumentParser(description="File-based Chang-Tsai-Lin SVD image watermark worker.")
    parser.add_argument("operation", choices=["embed", "decode", "embed_batch", "decode_batch", "serve-json"])
    parser.add_argument("--request-json", required=False)
    parser.add_argument("--response-json", required=False)
    parser.add_argument("--session-file", required=False)
    parser.add_argument("--method", choices=["svd_chang"], required=False)
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
