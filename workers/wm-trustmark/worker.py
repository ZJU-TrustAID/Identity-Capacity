from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TRUSTMARK_PYTHON_DIR = REPO_ROOT / "third_party" / "trustmark" / "python"
_TRUSTMARK_CACHE: dict[tuple[tuple[str, Any], ...], tuple[Any, dict[str, Any]]] = {}


def _import_trustmark() -> tuple[Any, Any]:
    import sys

    trustmark_dir = str(TRUSTMARK_PYTHON_DIR)
    if trustmark_dir not in sys.path:
        sys.path.insert(0, trustmark_dir)

    from PIL import Image
    from trustmark import TrustMark

    return TrustMark, Image


def bits_to_bitstring(bits: np.ndarray) -> str:
    flat = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if np.any((flat != 0) & (flat != 1)):
        raise ValueError("payload bits must only contain 0 or 1")
    return "".join(str(int(bit)) for bit in flat)


def bitstring_to_bits(bitstring: str) -> np.ndarray:
    if any(ch not in {"0", "1"} for ch in bitstring):
        raise ValueError("bitstring must only contain '0' and '1'")
    return np.fromiter((int(ch) for ch in bitstring), dtype=np.uint8)


def validate_payload_bits(bits: np.ndarray, expected_bits: int) -> np.ndarray:
    flat = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if flat.size != expected_bits:
        raise ValueError(f"TrustMark expected {expected_bits} bits, received {flat.size}")
    if np.any((flat != 0) & (flat != 1)):
        raise ValueError("payload bits must only contain 0 or 1")
    return flat


def resolve_method_params(method_params: dict[str, Any] | None) -> dict[str, Any]:
    params = dict(method_params or {})
    model_type = str(params.get("model_type", "Q")).upper()
    encoding_name = str(params.get("encoding_name", "NONE")).upper()
    mode = str(params.get("mode", "binary")).lower()
    device = str(params.get("device", ""))
    if mode != "binary":
        raise ValueError("wm-trustmark worker only supports MODE='binary' for this smoke flow")
    use_ecc = encoding_name not in {"", "NONE", "RAW"}

    return {
        "device": device,
        "model_type": model_type,
        "encoding_name": "NONE" if not use_ecc else encoding_name,
        "mode": mode,
        "wm_strength": float(params.get("wm_strength", 1.0)),
        "wm_merge": str(params.get("wm_merge", "bilinear")),
        "load_remover": bool(params.get("load_remover", False)),
        "load_bbox_detector": bool(params.get("load_bbox_detector", False)),
        "detectfirst": bool(params.get("detectfirst", False)),
        "rotation": bool(params.get("rotation", False)),
        "use_ecc": use_ecc,
        "schema_capacity": None if use_ecc else 100,
        "native_coded_output": use_ecc,
        "raw_bit_channel": not use_ecc,
    }


def load_request(request_path: Path) -> dict[str, Any]:
    return json.loads(request_path.read_text(encoding="utf-8"))


def save_response(response_path: Path, payload: dict[str, Any]) -> None:
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _metadata_for_embed(
    *,
    params: dict[str, Any],
    input_image_path: Path,
    output_image_path: Path,
    payload_bits_path: Path,
) -> dict[str, Any]:
    return {
        "method_name": "trustmark",
        "mode": params["mode"],
        "schema_type": params["encoding_name"],
        "schema_capacity_bits": params["schema_capacity"],
        "native_coded_output": params["native_coded_output"],
        "raw_bit_channel": params["raw_bit_channel"],
        "model_type": params["model_type"],
        "wm_strength": params["wm_strength"],
        "wm_merge": params["wm_merge"],
        "input_image_path": str(input_image_path),
        "watermarked_image_path": str(output_image_path),
        "payload_bits_path": str(payload_bits_path),
    }


def _bits_for_decode_result(
    *,
    decoded_secret: str,
    wm_present: bool,
    params: dict[str, Any],
) -> tuple[np.ndarray, float, str]:
    if wm_present:
        decoded_bits = bitstring_to_bits(decoded_secret)
        return decoded_bits, 1.0, "ok"
    decoded_bits = np.zeros(params["schema_capacity"], dtype=np.uint8)
    return decoded_bits, 0.0, "watermark_not_detected"


def _image_processing_box(tm: Any, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    scale = tm.concentrate_wm_region
    aspect_ratio = width / height if width > height else height / width
    if aspect_ratio > tm.aspect_ratio_lim:
        square_size = min(width, height)
        scaled_size = int(square_size * scale)
        left = (width - scaled_size) // 2
        top = (height - scaled_size) // 2
        return left, top, left + scaled_size, top + scaled_size
    scaled_w = int(width * scale)
    scaled_h = int(height * scale)
    left = (width - scaled_w) // 2
    top = (height - scaled_h) // 2
    return left, top, left + scaled_w, top + scaled_h


def _secret_tensor_for_bitstrings(tm: Any, bitstrings: list[str], params: dict[str, Any]):
    import torch

    if not params["use_ecc"]:
        secrets = [np.fromiter((int(ch) for ch in bitstring), dtype=np.float32) for bitstring in bitstrings]
        secret_array = np.stack(secrets, axis=0)
    elif params["mode"] == "binary":
        secret_array = tm.ecc.encode_binary(bitstrings)
    else:
        secret_array = tm.ecc.encode_text(bitstrings)
    return torch.from_numpy(np.asarray(secret_array, dtype=np.float32)).to(tm.device)


def _encode_images_batch(
    tm: Any,
    images: list[Any],
    bitstrings: list[str],
    params: dict[str, Any],
) -> list[Any]:
    import torch
    from torchvision import transforms
    from PIL import Image

    if not images:
        return []
    covers = []
    crop_sizes: list[tuple[int, int]] = []
    for image in images:
        cover_image = tm.get_the_image_for_processing(image)
        crop_sizes.append(cover_image.size)
        cover = cover_image.resize((tm.model_resolution_enc, tm.model_resolution_enc), Image.BILINEAR)
        covers.append(transforms.ToTensor()(cover).to(tm.encoder.device) * 2.0 - 1.0)
    cover_batch = torch.stack(covers, dim=0)
    secret_batch = _secret_tensor_for_bitstrings(tm, bitstrings, params).to(tm.encoder.device)

    wm_strength = float(params["wm_strength"])
    if params["model_type"] == "P":
        wm_strength *= 1.25
    watermarked_images = []
    with torch.no_grad():
        stego_batch, _ = tm.encoder(cover_batch, secret_batch)
        residual_batch = stego_batch.clamp(-1, 1) - cover_batch
        residual_batch = residual_batch - residual_batch.mean(dim=(2, 3), keepdim=True)
        for image, residual, crop_size in zip(images, residual_batch, crop_sizes, strict=True):
            crop_w, crop_h = crop_size
            resized_residual = torch.nn.functional.interpolate(
                residual.unsqueeze(0),
                size=(crop_h, crop_w),
                mode=params["wm_merge"],
            )
            residual_np = resized_residual.permute(0, 2, 3, 1).cpu().numpy().astype("f4")[0]
            cover_image = tm.get_the_image_for_processing(image)
            stego_patch = np.clip(residual_np * wm_strength + np.array(cover_image) / 127.5 - 1.0, -1, 1)
            stego_patch = stego_patch * 127.5 + 127.5
            stego = tm.put_the_image_after_processing(stego_patch, np.asarray(image).astype(np.uint8))
            watermarked_images.append(Image.fromarray(stego.astype(np.uint8)))
    return watermarked_images


def _decode_images_batch(tm: Any, images: list[Any], params: dict[str, Any]) -> list[tuple[str, bool, int]]:
    import torch
    from torchvision import transforms
    from PIL import Image

    if not images:
        return []
    if params["detectfirst"] or params["rotation"] or params["use_ecc"]:
        return [
            tm.decode(
                image,
                MODE=params["mode"],
                DETECTFIRST=params["detectfirst"],
                ROTATION=params["rotation"],
            )
            for image in images
        ]

    stegos = []
    for image in images:
        stego_image = tm.get_the_image_for_processing(image)
        stego_image = stego_image.resize((tm.model_resolution_dec, tm.model_resolution_dec), Image.BILINEAR)
        stegos.append(transforms.ToTensor()(stego_image).to(tm.decoder.device) * 2.0 - 1.0)
    stego_batch = torch.stack(stegos, dim=0)
    with torch.no_grad():
        secret_binaryarray = (tm.decoder.decoder(stego_batch) > 0).cpu().numpy()
    return [
        ("".join(str(int(x)) for x in row), True, -1)
        for row in secret_binaryarray
    ]


def _build_trustmark(method_params: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    TrustMark, _ = _import_trustmark()
    params = resolve_method_params(method_params)
    cache_key = tuple(sorted(params.items()))
    if cache_key in _TRUSTMARK_CACHE:
        return _TRUSTMARK_CACHE[cache_key]
    device = params["device"]
    if device.startswith("cuda"):
        import torch

        torch_device = torch.device(device)
        if torch_device.index is not None:
            torch.cuda.set_device(torch_device)
    encoding_type = TrustMark.Encoding.BCH_5
    if params["use_ecc"]:
        encoding_type = getattr(TrustMark.Encoding, params["encoding_name"])
    tm = TrustMark(
        use_ECC=params["use_ecc"],
        verbose=True,
        model_type=params["model_type"],
        encoding_type=encoding_type,
        secret_len=100,
        device=device,
        loadRemover=params["load_remover"],
        loadBBoxDetector=params["load_bbox_detector"],
    )
    params["schema_capacity"] = int(tm.schemaCapacity())
    loaded = (tm, params)
    _TRUSTMARK_CACHE[cache_key] = loaded
    return loaded


def embed(request: dict[str, Any]) -> dict[str, Any]:
    _, Image = _import_trustmark()
    tm, params = _build_trustmark(request.get("method_params"))

    input_image_path = Path(request["input_image_path"])
    output_image_path = Path(request["output_image_path"])
    payload_bits_path = Path(request["payload_bits_path"])
    metadata_path = Path(request["metadata_path"])

    payload_bits = validate_payload_bits(np.load(payload_bits_path), params["schema_capacity"])
    bitstring = bits_to_bitstring(payload_bits)

    cover = Image.open(input_image_path).convert("RGB")
    watermarked = tm.encode(
        cover,
        bitstring,
        MODE=params["mode"],
        WM_STRENGTH=params["wm_strength"],
        WM_MERGE=params["wm_merge"],
    )
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    watermarked.save(output_image_path)

    metadata = _metadata_for_embed(
        params=params,
        input_image_path=input_image_path,
        output_image_path=output_image_path,
        payload_bits_path=payload_bits_path,
    )
    save_response(metadata_path, metadata)

    return {
        "status": "ok",
        "watermarked_image_path": str(output_image_path),
        "metadata_path": str(metadata_path),
        "schema_capacity_bits": params["schema_capacity"],
        "native_coded_output": params["native_coded_output"],
        "raw_bit_channel": params["raw_bit_channel"],
    }


def decode(request: dict[str, Any]) -> dict[str, Any]:
    _, Image = _import_trustmark()
    tm, params = _build_trustmark(request.get("method_params"))

    input_image_path = Path(request["input_image_path"])
    metadata_path = Path(request["metadata_path"])
    output_bits_path = Path(request["output_bits_path"])
    output_scores_path = Path(request["output_scores_path"])

    metadata = load_request(metadata_path)
    stego = Image.open(input_image_path).convert("RGB")
    decoded_secret, wm_present, wm_schema = tm.decode(
        stego,
        MODE=params["mode"],
        DETECTFIRST=params["detectfirst"],
        ROTATION=params["rotation"],
    )

    decoded_bits, detect_prob, status = _bits_for_decode_result(
        decoded_secret=decoded_secret,
        wm_present=wm_present,
        params=params,
    )
    if wm_present:
        if params["use_ecc"]:
            schema_match = wm_schema == getattr(tm.Encoding, metadata["schema_type"])
        else:
            schema_match = metadata["schema_type"] == "NONE" and int(wm_schema) == int(tm.Encoding.Undefined)
    else:
        decoded_bits = np.zeros(params["schema_capacity"], dtype=np.uint8)
        detect_prob = 0.0
        status = "watermark_not_detected"
        schema_match = False

    if decoded_bits.size != params["schema_capacity"]:
        status = "decoded_length_mismatch"
        wm_present = False
        detect_prob = 0.0
        schema_match = False
        decoded_bits = np.zeros(params["schema_capacity"], dtype=np.uint8)

    output_bits_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_bits_path, decoded_bits)
    soft_scores = decoded_bits.astype(np.float32)
    np.save(output_scores_path, soft_scores)

    return {
        "status": status,
        "decoded_bits_path": str(output_bits_path),
        "soft_scores_path": str(output_scores_path),
        "decoded_bitstring": bits_to_bitstring(decoded_bits),
        "detect_prob": detect_prob,
        "watermark_present": bool(wm_present),
        "schema_type": metadata["schema_type"],
        "schema_value": int(wm_schema),
        "schema_match": bool(schema_match),
        "native_coded_output": params["native_coded_output"],
        "raw_bit_channel": params["raw_bit_channel"],
    }


def embed_batch(request: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image

    tm, params = _build_trustmark(request.get("method_params"))
    jobs = list(request.get("jobs") or [])
    images = []
    bitstrings = []
    for job in jobs:
        payload_bits = validate_payload_bits(np.load(Path(job["payload_bits_path"])), params["schema_capacity"])
        images.append(Image.open(job["input_image_path"]).convert("RGB"))
        bitstrings.append(bits_to_bitstring(payload_bits))

    watermarked_images = _encode_images_batch(tm, images, bitstrings, params)
    results = []
    for job, watermarked in zip(jobs, watermarked_images, strict=True):
        input_image_path = Path(job["input_image_path"])
        output_image_path = Path(job["output_image_path"])
        payload_bits_path = Path(job["payload_bits_path"])
        metadata_path = Path(job["metadata_path"])
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        watermarked.save(output_image_path)
        metadata = _metadata_for_embed(
            params=params,
            input_image_path=input_image_path,
            output_image_path=output_image_path,
            payload_bits_path=payload_bits_path,
        )
        save_response(metadata_path, metadata)
        results.append(
            {
                "status": "ok",
                "watermarked_image_path": str(output_image_path),
                "metadata_path": str(metadata_path),
                "schema_capacity_bits": params["schema_capacity"],
                "native_coded_output": params["native_coded_output"],
                "raw_bit_channel": params["raw_bit_channel"],
            }
        )
    return {"status": "ok", "results": results}


def decode_batch(request: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image

    tm, params = _build_trustmark(request.get("method_params"))
    jobs = list(request.get("jobs") or [])
    images = [Image.open(job["input_image_path"]).convert("RGB") for job in jobs]
    decoded_rows = _decode_images_batch(tm, images, params)

    results = []
    for job, (decoded_secret, wm_present, wm_schema) in zip(jobs, decoded_rows, strict=True):
        metadata = load_request(Path(job["metadata_path"]))
        decoded_bits, detect_prob, status = _bits_for_decode_result(
            decoded_secret=decoded_secret,
            wm_present=wm_present,
            params=params,
        )
        if wm_present:
            if params["use_ecc"]:
                schema_match = wm_schema == getattr(tm.Encoding, metadata["schema_type"])
            else:
                schema_match = metadata["schema_type"] == "NONE" and int(wm_schema) == int(tm.Encoding.Undefined)
        else:
            schema_match = False

        if decoded_bits.size != params["schema_capacity"]:
            status = "decoded_length_mismatch"
            wm_present = False
            detect_prob = 0.0
            schema_match = False
            decoded_bits = np.zeros(params["schema_capacity"], dtype=np.uint8)

        output_bits_path = Path(job["output_bits_path"])
        output_scores_path = Path(job["output_scores_path"])
        output_bits_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_bits_path, decoded_bits)
        soft_scores = decoded_bits.astype(np.float32)
        np.save(output_scores_path, soft_scores)
        results.append(
            {
                "status": status,
                "decoded_bits_path": str(output_bits_path),
                "soft_scores_path": str(output_scores_path),
                "decoded_bitstring": bits_to_bitstring(decoded_bits),
                "detect_prob": detect_prob,
                "watermark_present": bool(wm_present),
                "schema_type": metadata["schema_type"],
                "schema_value": int(wm_schema),
                "schema_match": bool(schema_match),
                "native_coded_output": params["native_coded_output"],
                "raw_bit_channel": params["raw_bit_channel"],
            }
        )
    return {"status": "ok", "results": results}


def handle_session_request(request: dict[str, Any]) -> dict[str, Any]:
    operation = str(request["operation"])
    response_path = Path(request["response_path"])
    if operation == "embed":
        response = embed(request)
    elif operation == "decode":
        response = decode(request)
    elif operation == "embed_batch":
        response = embed_batch(request)
    elif operation == "decode_batch":
        response = decode_batch(request)
    else:
        raise ValueError(f"Unknown session operation: {operation}")
    save_response(response_path, response)
    return response


def serve_json_requests(session_file: Path | None = None) -> int:
    if session_file is not None:
        count = int(session_file.read_text(encoding="utf-8") or "0") if session_file.exists() else 0
        session_file.write_text(str(count + 1), encoding="utf-8")
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            response = handle_session_request(json.loads(line))
        except Exception as exc:
            response = {"status": "error", "message": str(exc)}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="File-based TrustMark worker.")
    parser.add_argument("operation", choices=["embed", "decode", "embed_batch", "decode_batch", "serve-json"])
    parser.add_argument("--request", required=False, help="Path to a JSON request file.")
    parser.add_argument("--response", required=False, help="Path to write a JSON response file.")
    parser.add_argument("--session-file", required=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.operation == "serve-json":
        session_file = None if not getattr(args, "session_file", None) else Path(args.session_file)
        serve_json_requests(session_file=session_file)
        return
    if not args.request or not args.response:
        raise SystemExit("--request and --response are required for embed/decode")
    request = load_request(Path(args.request))
    if args.operation == "embed":
        response = embed(request)
    elif args.operation == "decode":
        response = decode(request)
    elif args.operation == "embed_batch":
        response = embed_batch(request)
    else:
        response = decode_batch(request)
    save_response(Path(args.response), response)


if __name__ == "__main__":
    main()
