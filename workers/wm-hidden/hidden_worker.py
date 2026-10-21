from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def handle_embed_request(
    request_path: Path,
    response_path: Path,
    embed_impl: Callable[[str, list[int], dict, str], dict] | None = None,
) -> None:
    if embed_impl is None:
        from hidden_runtime import embed_image as embed_impl

    request = load_json(request_path)
    payload_bits = np.load(request["payload_bits_path"]).astype(np.uint8).tolist()
    metadata = embed_impl(
        request["input_image_path"],
        payload_bits,
        request.get("method_params") or {},
        request["output_image_path"],
    )

    metadata_path = Path(request["metadata_output_path"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    response = {
        "status": metadata.get("status", "ok"),
        "watermarked_image_path": request["output_image_path"],
        "metadata_path": str(metadata_path),
    }
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")


def handle_decode_request(
    request_path: Path,
    response_path: Path,
    decode_impl: Callable[[str, dict, dict], tuple[list[int], list[float], float, str]] | None = None,
) -> None:
    if decode_impl is None:
        from hidden_runtime import decode_image as decode_impl

    request = load_json(request_path)
    metadata = load_json(Path(request["metadata_path"]))
    decoded_bits, soft_scores, detect_prob, status = decode_impl(
        request["input_image_path"],
        metadata,
        request.get("method_params") or {},
    )

    decoded_bits_path = Path(request["decoded_bits_output_path"])
    soft_scores_path = Path(request["soft_scores_output_path"])
    decoded_bits_path.parent.mkdir(parents=True, exist_ok=True)
    soft_scores_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(decoded_bits_path, np.asarray(decoded_bits, dtype=np.uint8))
    np.save(soft_scores_path, np.asarray(soft_scores, dtype=np.float32))

    response = {
        "status": status,
        "detect_prob": detect_prob,
        "decoded_bits_path": str(decoded_bits_path),
        "soft_scores_path": str(soft_scores_path),
    }
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")


def _write_embed_result(job: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    metadata_path = Path(job["metadata_output_path"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
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
    decoded_bits_path.parent.mkdir(parents=True, exist_ok=True)
    soft_scores_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(decoded_bits_path, np.asarray(decoded_bits, dtype=np.uint8))
    np.save(soft_scores_path, np.asarray(soft_scores, dtype=np.float32))
    return {
        "status": status,
        "detect_prob": detect_prob,
        "decoded_bits_path": str(decoded_bits_path),
        "soft_scores_path": str(soft_scores_path),
    }


def handle_embed_batch_request(
    request_path: Path,
    response_path: Path,
    embed_batch_impl: Callable[[list[dict[str, Any]], dict], list[dict]] | None = None,
) -> None:
    if embed_batch_impl is None:
        from hidden_runtime import embed_image_batch as embed_batch_impl

    request = load_json(request_path)
    method_params = request.get("method_params") or {}
    jobs: list[dict[str, Any]] = []
    for raw_job in list(request.get("jobs") or []):
        job = dict(raw_job)
        job["payload_bits"] = np.load(job["payload_bits_path"]).astype(np.uint8).tolist()
        jobs.append(job)

    metadata_rows = embed_batch_impl(jobs, method_params)
    if len(metadata_rows) != len(jobs):
        raise RuntimeError(f"embed batch returned {len(metadata_rows)} results for {len(jobs)} jobs")
    response = {
        "status": "ok",
        "results": [
            _write_embed_result(job, metadata)
            for job, metadata in zip(jobs, metadata_rows, strict=True)
        ],
    }
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")


def handle_decode_batch_request(
    request_path: Path,
    response_path: Path,
    decode_batch_impl: Callable[[list[dict[str, Any]], dict], list[tuple[list[int], list[float], float, str]]] | None = None,
) -> None:
    if decode_batch_impl is None:
        from hidden_runtime import decode_image_batch as decode_batch_impl

    request = load_json(request_path)
    method_params = request.get("method_params") or {}
    jobs: list[dict[str, Any]] = []
    for raw_job in list(request.get("jobs") or []):
        job = dict(raw_job)
        job["metadata"] = load_json(Path(job["metadata_path"]))
        jobs.append(job)

    decoded_rows = decode_batch_impl(jobs, method_params)
    if len(decoded_rows) != len(jobs):
        raise RuntimeError(f"decode batch returned {len(decoded_rows)} results for {len(jobs)} jobs")
    response = {
        "status": "ok",
        "results": [
            _write_decode_result(job, decoded_bits, soft_scores, detect_prob, status)
            for job, (decoded_bits, soft_scores, detect_prob, status) in zip(
                jobs,
                decoded_rows,
                strict=True,
            )
        ],
    }
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")


def handle_session_request(request: dict) -> dict:
    operation = str(request["operation"])
    response_path = Path(request["response_path"])
    if operation == "embed":
        handle_embed_request_from_payload(request, response_path)
        return load_json(response_path)
    if operation == "decode":
        handle_decode_request_from_payload(request, response_path)
        return load_json(response_path)
    if operation == "embed_batch":
        handle_embed_batch_request_from_payload(request, response_path)
        return load_json(response_path)
    if operation == "decode_batch":
        handle_decode_batch_request_from_payload(request, response_path)
        return load_json(response_path)
    raise ValueError(f"Unknown session operation: {operation}")


def handle_embed_request_from_payload(request: dict, response_path: Path) -> None:
    from hidden_runtime import embed_image

    payload_bits = np.load(request["payload_bits_path"]).astype(np.uint8).tolist()
    metadata = embed_image(
        request["input_image_path"],
        payload_bits,
        request.get("method_params") or {},
        request["output_image_path"],
    )

    metadata_path = Path(request["metadata_output_path"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    response = {
        "status": metadata.get("status", "ok"),
        "watermarked_image_path": request["output_image_path"],
        "metadata_path": str(metadata_path),
    }
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")


def handle_decode_request_from_payload(request: dict, response_path: Path) -> None:
    from hidden_runtime import decode_image

    metadata = load_json(Path(request["metadata_path"]))
    decoded_bits, soft_scores, detect_prob, status = decode_image(
        request["input_image_path"],
        metadata,
        request.get("method_params") or {},
    )

    decoded_bits_path = Path(request["decoded_bits_output_path"])
    soft_scores_path = Path(request["soft_scores_output_path"])
    decoded_bits_path.parent.mkdir(parents=True, exist_ok=True)
    soft_scores_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(decoded_bits_path, np.asarray(decoded_bits, dtype=np.uint8))
    np.save(soft_scores_path, np.asarray(soft_scores, dtype=np.float32))

    response = {
        "status": status,
        "detect_prob": detect_prob,
        "decoded_bits_path": str(decoded_bits_path),
        "soft_scores_path": str(soft_scores_path),
    }
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")


def handle_embed_batch_request_from_payload(request: dict, response_path: Path) -> None:
    request_path = response_path.parent / "embed_batch_request_from_payload.json"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    handle_embed_batch_request(request_path, response_path)


def handle_decode_batch_request_from_payload(request: dict, response_path: Path) -> None:
    request_path = response_path.parent / "decode_batch_request_from_payload.json"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    handle_decode_batch_request(request_path, response_path)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="File-based HiDDeN worker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("embed", "decode", "embed_batch", "decode_batch"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--request-json", required=True)
        sub.add_argument("--response-json", required=True)

    serve_parser = subparsers.add_parser("serve-json")
    serve_parser.add_argument("--session-file", required=False)

    args = parser.parse_args()

    if args.command == "serve-json":
        session_file = None if not getattr(args, "session_file", None) else Path(args.session_file)
        serve_json_requests(session_file=session_file)
    elif args.command == "embed":
        handle_embed_request(Path(args.request_json), Path(args.response_json))
    elif args.command == "decode":
        handle_decode_request(Path(args.request_json), Path(args.response_json))
    elif args.command == "embed_batch":
        handle_embed_batch_request(Path(args.request_json), Path(args.response_json))
    else:
        handle_decode_batch_request(Path(args.request_json), Path(args.response_json))


if __name__ == "__main__":
    main()
