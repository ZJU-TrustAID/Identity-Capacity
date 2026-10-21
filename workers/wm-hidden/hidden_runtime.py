from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
HIDDEN_ROOT = REPO_ROOT / "third_party" / "HiDDeN-Pytorch"

if str(HIDDEN_ROOT) not in sys.path:
    sys.path.insert(0, str(HIDDEN_ROOT))

from model.encoder_decoder import EncoderDecoder  # noqa: E402
from noise_layers.noiser import Noiser  # noqa: E402
from options import HiDDenConfiguration  # noqa: E402


DEFAULT_EXPERIMENT = "no-noise adam-eps-1e-4"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "third_party"
    / "HiDDeN-Pytorch"
    / "experiments"
    / DEFAULT_EXPERIMENT
    / "checkpoints"
    / "no-noise--epoch-200.pyt"
)
DEFAULT_OPTIONS = (
    REPO_ROOT
    / "third_party"
    / "HiDDeN-Pytorch"
    / "experiments"
    / DEFAULT_EXPERIMENT
    / "options-and-config.pickle"
)


@dataclass
class RuntimeBundle:
    config: HiDDenConfiguration
    device: torch.device
    model: EncoderDecoder
    checkpoint_path: Path
    options_path: Path
    checkpoint_mode: str


_RUNTIME_CACHE: dict[tuple[str, str, str], RuntimeBundle] = {}


def load_hidden_options(options_path: Path) -> tuple[Any, list[Any], HiDDenConfiguration]:
    with options_path.open("rb") as handle:
        train_options = pickle.load(handle)
        noise_config = pickle.load(handle)
        hidden_config = pickle.load(handle)
    if not hasattr(hidden_config, "enable_fp16") and not hasattr(hidden_config, "enable_mixed"):
        setattr(hidden_config, "enable_fp16", False)
    return train_options, noise_config, hidden_config


def resolve_paths(method_params: dict[str, Any] | None) -> tuple[Path, Path]:
    params = method_params or {}
    checkpoint_path = Path(params.get("checkpoint_path", DEFAULT_CHECKPOINT))
    options_path = Path(params.get("options_path", DEFAULT_OPTIONS))
    return checkpoint_path, options_path


def load_runtime(method_params: dict[str, Any] | None = None) -> RuntimeBundle:
    checkpoint_path, options_path = resolve_paths(method_params)
    device_name = (method_params or {}).get("device")
    if device_name:
        resolved_device_name = str(device_name)
    else:
        resolved_device_name = "cuda" if torch.cuda.is_available() else "cpu"
    cache_key = (str(checkpoint_path), str(options_path), resolved_device_name)
    if cache_key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[cache_key]

    _, noise_config, hidden_config = load_hidden_options(options_path)

    device = torch.device(resolved_device_name)

    noiser = Noiser(noise_config, device)
    model = EncoderDecoder(hidden_config, noiser).to(device)

    checkpoint_mode = "trained"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["enc-dec-model"])
    else:
        checkpoint_mode = "random_init"

    model.eval()
    runtime = RuntimeBundle(
        config=hidden_config,
        device=device,
        model=model,
        checkpoint_path=checkpoint_path,
        options_path=options_path,
        checkpoint_mode=checkpoint_mode,
    )
    _RUNTIME_CACHE[cache_key] = runtime
    return runtime


def load_image_tensor(image_path: Path, config: HiDDenConfiguration, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    resized = image.resize((config.W, config.H))
    image_np = np.asarray(resized, dtype=np.float32) / 127.5 - 1.0
    image_np = np.transpose(image_np, (2, 0, 1))
    tensor = torch.from_numpy(image_np).unsqueeze(0).to(device)
    return tensor


def save_image_tensor(image_tensor: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = image_tensor.detach().cpu().squeeze(0).clamp(-1.0, 1.0)
    image_np = ((clipped.numpy().transpose(1, 2, 0) + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(image_np, mode="RGB").save(output_path)


def normalize_payload(payload_bits: list[int], message_length: int) -> np.ndarray:
    bits = np.asarray(payload_bits, dtype=np.uint8).reshape(-1)
    if bits.size > message_length:
        raise ValueError(
            f"Payload has {bits.size} bits but checkpoint expects at most {message_length} bits."
        )
    if bits.size < message_length:
        padding = np.zeros(message_length - bits.size, dtype=np.uint8)
        bits = np.concatenate([bits, padding], axis=0)
    return bits


def embed_image(
    input_image_path: str | Path,
    payload_bits: list[int],
    method_params: dict[str, Any] | None,
    output_image_path: str | Path,
) -> dict[str, Any]:
    runtime = load_runtime(method_params)
    payload_array = normalize_payload(payload_bits, runtime.config.message_length)
    payload_tensor = (
        torch.from_numpy(payload_array.astype(np.float32))
        .unsqueeze(0)
        .to(runtime.device)
    )
    image_tensor = load_image_tensor(Path(input_image_path), runtime.config, runtime.device)

    with torch.no_grad():
        encoded_image, _, decoded_logits = runtime.model(image_tensor, payload_tensor)

    save_image_tensor(encoded_image, Path(output_image_path))

    metadata = {
        "payload_length": int(len(payload_bits)),
        "message_length": int(runtime.config.message_length),
        "checkpoint_path": str(runtime.checkpoint_path),
        "options_path": str(runtime.options_path),
        "checkpoint_mode": runtime.checkpoint_mode,
        "input_image_size": [int(runtime.config.W), int(runtime.config.H)],
        "embed_decoded_logits": decoded_logits.detach().cpu().squeeze(0).tolist(),
    }
    return metadata


def embed_image_batch(
    jobs: list[dict[str, Any]],
    method_params: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    runtime = load_runtime(method_params)
    payload_rows = [
        normalize_payload(list(job["payload_bits"]), runtime.config.message_length)
        for job in jobs
    ]
    payload_tensor = torch.from_numpy(np.stack(payload_rows, axis=0).astype(np.float32)).to(runtime.device)
    image_tensor = torch.cat(
        [
            load_image_tensor(Path(job["input_image_path"]), runtime.config, runtime.device)
            for job in jobs
        ],
        dim=0,
    )

    with torch.no_grad():
        encoded_images, _, decoded_logits = runtime.model(image_tensor, payload_tensor)

    outputs = []
    for index, job in enumerate(jobs):
        save_image_tensor(encoded_images[index : index + 1], Path(job["output_image_path"]))
        outputs.append(
            {
                "payload_length": int(len(job["payload_bits"])),
                "message_length": int(runtime.config.message_length),
                "checkpoint_path": str(runtime.checkpoint_path),
                "options_path": str(runtime.options_path),
                "checkpoint_mode": runtime.checkpoint_mode,
                "input_image_size": [int(runtime.config.W), int(runtime.config.H)],
                "embed_decoded_logits": decoded_logits[index].detach().cpu().tolist(),
            }
        )
    return outputs


def decode_image(
    input_image_path: str | Path,
    metadata: dict[str, Any],
    method_params: dict[str, Any] | None,
) -> tuple[list[int], list[float], float, str]:
    merged_params = dict(method_params or {})
    if "checkpoint_path" not in merged_params and metadata.get("checkpoint_path"):
        merged_params["checkpoint_path"] = metadata["checkpoint_path"]
    if "options_path" not in merged_params and metadata.get("options_path"):
        merged_params["options_path"] = metadata["options_path"]

    runtime = load_runtime(merged_params)
    image_tensor = load_image_tensor(Path(input_image_path), runtime.config, runtime.device)

    with torch.no_grad():
        logits = runtime.model.decoder(image_tensor).detach().cpu().squeeze(0)

    payload_length = int(metadata.get("payload_length", runtime.config.message_length))
    decoded_values = logits[:payload_length]
    soft_scores = decoded_values.tolist()
    decoded_bits = decoded_values.round().clip(0, 1).to(torch.uint8).tolist()
    clipped_scores = decoded_values.clip(0, 1)
    confidence = torch.maximum(clipped_scores, 1.0 - clipped_scores).mean()
    mean_abs_logit = float(confidence.item())
    status = "decoded"
    if runtime.checkpoint_mode != "trained":
        status = "architecture_smoke"

    return decoded_bits, soft_scores, mean_abs_logit, status


def decode_image_batch(
    jobs: list[dict[str, Any]],
    method_params: dict[str, Any] | None,
) -> list[tuple[list[int], list[float], float, str]]:
    if not jobs:
        return []
    merged_params = dict(method_params or {})
    first_metadata = dict(jobs[0]["metadata"])
    if "checkpoint_path" not in merged_params and first_metadata.get("checkpoint_path"):
        merged_params["checkpoint_path"] = first_metadata["checkpoint_path"]
    if "options_path" not in merged_params and first_metadata.get("options_path"):
        merged_params["options_path"] = first_metadata["options_path"]

    runtime = load_runtime(merged_params)
    image_tensor = torch.cat(
        [
            load_image_tensor(Path(job["input_image_path"]), runtime.config, runtime.device)
            for job in jobs
        ],
        dim=0,
    )

    with torch.no_grad():
        logits = runtime.model.decoder(image_tensor).detach().cpu()

    outputs = []
    for index, job in enumerate(jobs):
        metadata = dict(job["metadata"])
        payload_length = int(metadata.get("payload_length", runtime.config.message_length))
        decoded_values = logits[index, :payload_length]
        soft_scores = decoded_values.tolist()
        decoded_bits = decoded_values.round().clip(0, 1).to(torch.uint8).tolist()
        clipped_scores = decoded_values.clip(0, 1)
        confidence = torch.maximum(clipped_scores, 1.0 - clipped_scores).mean()
        mean_abs_logit = float(confidence.item())
        status = "decoded"
        if runtime.checkpoint_mode != "trained":
            status = "architecture_smoke"
        outputs.append((decoded_bits, soft_scores, mean_abs_logit, status))
    return outputs


def summarize_result(metadata: dict[str, Any], decoded_bits: list[int], soft_scores: list[float], status: str) -> dict[str, Any]:
    return {
        "payload_length": metadata["payload_length"],
        "checkpoint_mode": metadata["checkpoint_mode"],
        "status": status,
        "decoded_bits": decoded_bits,
        "soft_scores_head": soft_scores[: min(8, len(soft_scores))],
    }


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
