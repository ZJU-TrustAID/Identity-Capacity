from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ic_estimation.utils import write_jsonl


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    path: Path
    modality: str
    dataset: str
    split: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


def enumerate_samples(
    dataset_root: str | Path,
    *,
    modality: str,
    dataset_name: str,
    split: str = "",
) -> list[SampleRecord]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {root}")

    normalized_modality = str(modality).lower()
    extensions = IMAGE_EXTENSIONS if normalized_modality == "image" else AUDIO_EXTENSIONS
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions)
    samples: list[SampleRecord] = []
    for path in paths:
        relative = path.relative_to(root)
        sample_id = str(relative.with_suffix("")).replace("/", "::")
        samples.append(
            SampleRecord(
                sample_id=sample_id,
                path=path,
                modality=normalized_modality,
                dataset=str(dataset_name),
                split=str(split),
                metadata={"relative_path": str(relative)},
            )
        )
    return samples


def build_subset_manifest(
    samples: list[SampleRecord],
    *,
    subset_count: int | None,
    subset_seed: int,
) -> list[SampleRecord]:
    ordered = list(samples)
    rng = random.Random(int(subset_seed))
    rng.shuffle(ordered)
    if subset_count is not None:
        ordered = ordered[: int(subset_count)]
    return ordered


def write_manifest_jsonl(path: str | Path, samples: list[SampleRecord]) -> None:
    write_jsonl(path, [sample.to_dict() for sample in samples])


def read_manifest_jsonl(path: str | Path) -> list[SampleRecord]:
    import json

    rows: list[SampleRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows.append(
            SampleRecord(
                sample_id=str(payload["sample_id"]),
                path=Path(payload["path"]),
                modality=str(payload["modality"]),
                dataset=str(payload.get("dataset", "")),
                split=str(payload.get("split", "")),
                metadata=dict(payload.get("metadata") or {}),
            )
        )
    return rows
