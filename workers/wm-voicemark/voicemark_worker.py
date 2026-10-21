from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
VOICEMARK_DIR = REPO_ROOT / "third_party" / "VoiceMark"
if str(VOICEMARK_DIR) not in sys.path:
    sys.path.insert(0, str(VOICEMARK_DIR))

DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "resources" / "models" / "VoiceMark"
DEFAULT_CHECKPOINT_NAME = "voicemark.pth"
_SOLVER_CACHE: dict[tuple[str, str, str, bool], tuple[Any, Path, str]] = {}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_payload_bits(path: Path) -> list[int]:
    bits = np.load(path)
    flat = np.asarray(bits, dtype=np.uint8).reshape(-1)
    return [int(x) for x in flat.tolist()]


def _bits_to_string(bits: list[int]) -> str:
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("payload bits must be 0/1")
    return "".join(str(bit) for bit in bits)


def _save_bit_array(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(values, dtype=np.uint8))


def _save_float_array(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(values, dtype=np.float64))


def _write_embed_result(job: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    metadata_output_path = Path(job["metadata_output_path"])
    _write_json(metadata_output_path, metadata)
    return {
        "status": "ok",
        "watermarked_audio_path": str(job["output_audio_path"]),
        "metadata_path": str(metadata_output_path),
    }


def _write_decode_result(
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


def _load_solver(method_params: dict[str, Any]):
    from infer import WatermarkSolver

    checkpoint_dir = Path(method_params.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR))
    checkpoint_name = str(method_params.get("checkpoint_name", DEFAULT_CHECKPOINT_NAME))
    device_name = str(method_params.get("device", "cpu"))
    strict = bool(method_params.get("strict", True))
    cache_key = (str(checkpoint_dir), checkpoint_name, device_name, strict)
    if cache_key in _SOLVER_CACHE:
        return _SOLVER_CACHE[cache_key]
    solver = WatermarkSolver()
    solver.device = device_name
    solver.model.to(solver.device)
    solver.load_model(str(checkpoint_dir), checkpoint_name, strict=strict)
    solver.model.to(solver.device)
    loaded = (solver, checkpoint_dir, checkpoint_name)
    _SOLVER_CACHE[cache_key] = loaded
    return loaded


def embed_audio_file(
    input_audio_path: Path,
    payload_bits: list[int],
    method_params: dict[str, Any],
    output_audio_path: Path,
) -> dict[str, Any]:
    if len(payload_bits) != 16:
        raise ValueError(f"VoiceMark expects 16 payload bits, got {len(payload_bits)}")
    solver, checkpoint_dir, checkpoint_name = _load_solver(method_params)
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    generated = Path(
        solver.infer_for_ui(
            str(input_audio_path),
            _bits_to_string(payload_bits),
            output_audio_path=str(output_audio_path.parent),
        )
    )
    if generated != output_audio_path:
        generated.replace(output_audio_path)

    return {
        "method": "voicemark",
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_name": checkpoint_name,
        "device": solver.device,
        "input_audio_path": str(input_audio_path),
        "output_audio_path": str(output_audio_path),
        "payload_bits": payload_bits,
        "payload_length": len(payload_bits),
        "sample_rate": solver.sample_rate,
    }


def _load_audio_for_solver(input_audio_path: Path, solver: Any):
    import torchaudio

    waveform, sample_rate = torchaudio.load(input_audio_path)
    if sample_rate != solver.sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=solver.sample_rate,
        )
        waveform = resampler(waveform)
    return waveform[:1], sample_rate


def _batch_truncate_seconds(method_params: dict[str, Any]) -> float:
    return float(method_params.get("batch_truncate_seconds", 3.0))


def _truncate_audio_for_batch(waveform: Any, sample_rate: int, method_params: dict[str, Any]) -> tuple[Any, bool, float]:
    seconds = _batch_truncate_seconds(method_params)
    if seconds <= 0:
        return waveform, False, seconds
    max_samples = int(round(float(seconds) * int(sample_rate)))
    if max_samples <= 0 or int(waveform.shape[-1]) <= max_samples:
        return waveform, False, seconds
    return waveform[..., :max_samples], True, seconds


def _pad_waveforms(torch_mod: Any, waveforms: list[Any]) -> tuple[Any, list[int]]:
    lengths = [int(waveform.shape[-1]) for waveform in waveforms]
    max_length = max(lengths, default=0)
    padded = []
    for waveform in waveforms:
        if int(waveform.shape[-1]) < max_length:
            waveform = torch_mod.nn.functional.pad(waveform, (0, max_length - int(waveform.shape[-1])))
        padded.append(waveform)
    return torch_mod.stack(padded, dim=0), lengths


def embed_audio_batch(jobs: list[dict[str, Any]], method_params: dict[str, Any]) -> list[dict[str, Any]]:
    if not jobs:
        return []
    import torch
    import torchaudio

    solver, checkpoint_dir, checkpoint_name = _load_solver(method_params)
    payload_rows = []
    waveforms = []
    original_sample_rates = []
    for job in jobs:
        payload_bits = list(job["payload_bits"])
        if len(payload_bits) != 16:
            raise ValueError(f"VoiceMark expects 16 payload bits, got {len(payload_bits)}")
        waveform, original_sample_rate = _load_audio_for_solver(Path(job["input_audio_path"]), solver)
        waveform, truncated, truncate_seconds = _truncate_audio_for_batch(
            waveform,
            int(solver.sample_rate),
            method_params,
        )
        payload_rows.append(payload_bits)
        waveforms.append(waveform)
        original_sample_rates.append((int(original_sample_rate), bool(truncated), float(truncate_seconds)))

    batch, lengths = _pad_waveforms(torch, waveforms)
    batch = batch.to(solver.device)
    message = torch.tensor(payload_rows, dtype=torch.long, device=solver.device)
    solver.model.eval()
    with torch.no_grad():
        output = solver.model(batch, message=message)
        watermarked = output["recon_wm"]

    outputs: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        output_audio_path = Path(job["output_audio_path"])
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(
            output_audio_path,
            watermarked[index, :, : lengths[index]].cpu(),
            solver.sample_rate,
        )
        payload_bits = payload_rows[index]
        outputs.append(
            {
                "method": "voicemark",
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint_name": checkpoint_name,
                "device": solver.device,
                "input_audio_path": str(job["input_audio_path"]),
                "output_audio_path": str(output_audio_path),
                "payload_bits": payload_bits,
                "payload_length": len(payload_bits),
                "sample_rate": solver.sample_rate,
                "original_sample_rate": original_sample_rates[index][0],
                "batch_truncate_seconds": original_sample_rates[index][2],
                "truncated_for_batch": original_sample_rates[index][1],
            }
        )
    return outputs


def decode_audio_file(
    input_audio_path: Path,
    metadata: dict[str, Any],
    method_params: dict[str, Any],
) -> tuple[list[int], list[float], float, str]:
    merged_params = {
        "checkpoint_dir": metadata.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR),
        "checkpoint_name": metadata.get("checkpoint_name", DEFAULT_CHECKPOINT_NAME),
        **method_params,
    }
    solver, _, _ = _load_solver(merged_params)
    detect_prob, decoded_id = solver.decode_for_ui(str(input_audio_path))
    payload_length = int(metadata.get("payload_length", 16))
    decoded_bits = [int(ch) for ch in decoded_id[:payload_length]]
    soft_scores = [float(bit) for bit in decoded_bits]
    detection_threshold = float(method_params.get("detection_threshold", 1e-2))
    status = "detected" if float(detect_prob) >= detection_threshold else "not_detected"
    return decoded_bits, soft_scores, float(detect_prob), status


def decode_audio_batch(
    jobs: list[dict[str, Any]],
    method_params: dict[str, Any],
) -> list[tuple[list[int], list[float], float, str]]:
    if not jobs:
        return []
    import torch
    import torch.nn.functional as F

    first_metadata = dict(jobs[0]["metadata"])
    merged_params = {
        "checkpoint_dir": first_metadata.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR),
        "checkpoint_name": first_metadata.get("checkpoint_name", DEFAULT_CHECKPOINT_NAME),
        **method_params,
    }
    solver, _, _ = _load_solver(merged_params)
    waveforms = []
    lengths = []
    payload_lengths = []
    for job in jobs:
        metadata = dict(job["metadata"])
        waveform, _ = _load_audio_for_solver(Path(job["input_audio_path"]), solver)
        waveform, _, _ = _truncate_audio_for_batch(waveform, int(solver.sample_rate), method_params)
        waveforms.append(waveform)
        lengths.append(int(waveform.shape[-1]))
        payload_lengths.append(int(metadata.get("payload_length", 16)))

    batch, padded_lengths = _pad_waveforms(torch, waveforms)
    batch = batch.to(solver.device)
    solver.model.eval()
    with torch.no_grad():
        logits, chunk_logits = solver.model.detect_watermark(batch, return_logits=True)

    detected = torch.sigmoid(logits).detach().cpu()
    chunk_probs = F.softmax(chunk_logits.detach().cpu(), dim=-1)
    chunk_indices = torch.argmax(chunk_probs, dim=-1)
    chunk_confidence = torch.max(chunk_probs, dim=-1).values
    detection_threshold = float(method_params.get("detection_threshold", 1e-2))

    outputs: list[tuple[list[int], list[float], float, str]] = []
    max_audio_length = max(padded_lengths, default=1)
    for index, payload_length in enumerate(payload_lengths):
        valid_steps = int(round(int(detected.shape[-1]) * int(lengths[index]) / max_audio_length))
        valid_steps = max(1, min(valid_steps, int(detected.shape[-1])))
        detect_prob = float(detected[index, :valid_steps].mean().item())
        decoded_bits: list[int] = []
        soft_scores: list[float] = []
        for chunk_index in range(chunk_indices.shape[1]):
            chunk_val = int(chunk_indices[index, chunk_index].item())
            chunk_score = float(chunk_confidence[index, chunk_index].item())
            for bit_index in range(4):
                decoded_bits.append(int((chunk_val >> bit_index) & 1))
                soft_scores.append(chunk_score)
        decoded_bits = decoded_bits[:payload_length]
        soft_scores = soft_scores[:payload_length]
        status = "detected" if detect_prob >= detection_threshold else "not_detected"
        outputs.append((decoded_bits, soft_scores, detect_prob, status))
    return outputs


def handle_embed_request(
    request_path: Path,
    response_path: Path,
    embed_impl: Callable[[Path, list[int], dict[str, Any], Path], dict[str, Any]] | None = None,
) -> None:
    request = _read_json(request_path)
    input_audio_path = Path(request["input_audio_path"])
    payload_bits = _load_payload_bits(Path(request["payload_bits_path"]))
    output_audio_path = Path(request["output_audio_path"])
    metadata_output_path = Path(request["metadata_output_path"])
    method_params = dict(request.get("method_params") or {})
    impl = embed_impl or embed_audio_file

    metadata = impl(input_audio_path, payload_bits, method_params, output_audio_path)
    _write_json(metadata_output_path, metadata)
    _write_json(
        response_path,
        {
            "status": "ok",
            "watermarked_audio_path": str(output_audio_path),
            "metadata_path": str(metadata_output_path),
        },
    )


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
                _write_embed_result(job, metadata)
                for job, metadata in zip(jobs, metadata_rows, strict=True)
            ],
        },
    )


def handle_decode_request(
    request_path: Path,
    response_path: Path,
    decode_impl: Callable[[Path, dict[str, Any], dict[str, Any]], tuple[list[int], list[float], float, str]] | None = None,
) -> None:
    request = _read_json(request_path)
    input_audio_path = Path(request["input_audio_path"])
    metadata = _read_json(Path(request["metadata_path"]))
    decoded_bits_output_path = Path(request["decoded_bits_output_path"])
    soft_scores_output_path = Path(request["soft_scores_output_path"])
    method_params = dict(request.get("method_params") or {})
    impl = decode_impl or decode_audio_file

    decoded_bits, soft_scores, detect_prob, status = impl(input_audio_path, metadata, method_params)
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
                _write_decode_result(job, decoded_bits, soft_scores, detect_prob, status)
                for job, (decoded_bits, soft_scores, detect_prob, status) in zip(
                    jobs,
                    decoded_rows,
                    strict=True,
                )
            ],
        },
    )


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
    payload_bits = _load_payload_bits(Path(request["payload_bits_path"]))
    output_audio_path = Path(request["output_audio_path"])
    metadata_output_path = Path(request["metadata_output_path"])
    method_params = dict(request.get("method_params") or {})

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
    metadata = _read_json(Path(request["metadata_path"]))
    decoded_bits_output_path = Path(request["decoded_bits_output_path"])
    soft_scores_output_path = Path(request["soft_scores_output_path"])
    method_params = dict(request.get("method_params") or {})

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
        try:
            response = handle_session_request(json.loads(line))
        except Exception as exc:
            response = {"status": "error", "message": str(exc)}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VoiceMark file-based worker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("embed", "decode", "embed_batch", "decode_batch"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--request-json", required=True)
        sub.add_argument("--response-json", required=True)
    serve = subparsers.add_parser("serve-json")
    serve.add_argument("--session-file", required=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "serve-json":
        session_file = None if not getattr(args, "session_file", None) else Path(args.session_file)
        return serve_json_requests(session_file=session_file)
    if args.command == "embed":
        handle_embed_request(Path(args.request_json), Path(args.response_json))
    elif args.command == "decode":
        handle_decode_request(Path(args.request_json), Path(args.response_json))
    elif args.command == "embed_batch":
        handle_embed_batch_request(Path(args.request_json), Path(args.response_json))
    else:
        handle_decode_batch_request(Path(args.request_json), Path(args.response_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
