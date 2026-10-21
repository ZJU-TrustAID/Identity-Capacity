"""IC Estimation command-line toolkit."""

from ic_estimation.adapters import (
    DecodeResult,
    EmbedResult,
    SubprocessWatermarkAdapter,
    WatermarkAdapter,
    make_builtin_adapters,
)
from ic_estimation.bch import BCHMotherCode, DecodeOutcome, ShortenedBCHCode, primitive_bch_mothers
from ic_estimation.channels import ChannelConfig, apply_channel
from ic_estimation.datasets import SampleRecord, build_subset_manifest, enumerate_samples
from ic_estimation.runner import ExperimentConfig, IdentityCapacityRunner

__all__ = [
    "BCHMotherCode",
    "ChannelConfig",
    "DecodeOutcome",
    "DecodeResult",
    "EmbedResult",
    "ExperimentConfig",
    "IdentityCapacityRunner",
    "SampleRecord",
    "ShortenedBCHCode",
    "SubprocessWatermarkAdapter",
    "WatermarkAdapter",
    "apply_channel",
    "build_subset_manifest",
    "enumerate_samples",
    "make_builtin_adapters",
    "primitive_bch_mothers",
]
