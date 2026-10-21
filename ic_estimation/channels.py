from __future__ import annotations

import shutil
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ChannelConfig:
    name: str = "clean"
    params: dict[str, Any] | None = None
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": dict(self.params or {}), "seed": int(self.seed)}


def _rng(config: ChannelConfig, trial_seed: int) -> np.random.Generator:
    return np.random.default_rng(int(config.seed) + int(trial_seed) * 1000003)


def apply_channel(
    input_path: str | Path,
    output_path: str | Path,
    *,
    modality: str,
    config: ChannelConfig,
    trial_seed: int = 0,
) -> Path:
    source = Path(input_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    name = str(config.name or "clean").lower()
    if name in {"clean", "none", "identity"}:
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        return target
    if name not in {"gaussian", "white_noise"}:
        raise ValueError(f"Unsupported channel: {config.name}")
    normalized_modality = str(modality).lower()
    if normalized_modality == "image":
        _apply_image_gaussian(source, target, config, trial_seed)
    elif normalized_modality == "audio":
        _apply_audio_gaussian(source, target, config, trial_seed)
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    return target


def _apply_image_gaussian(source: Path, target: Path, config: ChannelConfig, trial_seed: int) -> None:
    sigma = float((config.params or {}).get("sigma", 0.01))
    rng = _rng(config, trial_seed)
    image = Image.open(source).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    noisy = np.clip(array + rng.normal(loc=0.0, scale=sigma, size=array.shape), 0.0, 1.0)
    Image.fromarray(np.round(noisy * 255.0).astype(np.uint8), mode="RGB").save(target)


def _read_pcm_frames(path: Path) -> tuple[wave._wave_params, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        params = handle.getparams()
        frames = handle.readframes(params.nframes)
    if params.sampwidth == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif params.sampwidth == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif params.sampwidth == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {params.sampwidth}")
    return params, data


def _write_pcm_frames(path: Path, params: wave._wave_params, data: np.ndarray) -> None:
    clipped = np.clip(np.asarray(data, dtype=np.float32), -1.0, 1.0)
    if params.sampwidth == 1:
        encoded = np.round(clipped * 127.0 + 128.0).astype(np.uint8).tobytes()
    elif params.sampwidth == 2:
        encoded = np.round(clipped * 32767.0).astype("<i2").tobytes()
    elif params.sampwidth == 4:
        encoded = np.round(clipped * 2147483647.0).astype("<i4").tobytes()
    else:
        raise ValueError(f"Unsupported WAV sample width: {params.sampwidth}")
    with wave.open(str(path), "wb") as handle:
        handle.setparams(params)
        handle.writeframes(encoded)


def _apply_audio_gaussian(source: Path, target: Path, config: ChannelConfig, trial_seed: int) -> None:
    params, data = _read_pcm_frames(source)
    cfg = config.params or {}
    rng = _rng(config, trial_seed)
    if "snr_db" in cfg:
        signal_power = float(np.mean(data**2))
        snr = 10.0 ** (float(cfg["snr_db"]) / 10.0)
        sigma = 0.0 if signal_power <= 0.0 else np.sqrt(signal_power / snr)
    else:
        sigma = float(cfg.get("sigma", 0.001))
    noisy = data + rng.normal(loc=0.0, scale=sigma, size=data.shape).astype(np.float32)
    _write_pcm_frames(target, params, noisy)
