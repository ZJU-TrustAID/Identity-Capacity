from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

from ic_estimation.adapters import make_builtin_adapters
from ic_estimation.channels import ChannelConfig
from ic_estimation.code_families import build_code_candidates, normalize_code_family
from ic_estimation.config import load_config
from ic_estimation.datasets import build_subset_manifest, enumerate_samples, read_manifest_jsonl, write_manifest_jsonl
from ic_estimation.l2 import shortened_candidates
from ic_estimation.runner import ExperimentConfig, IdentityCapacityRunner


def _manifest(args: argparse.Namespace) -> int:
    samples = enumerate_samples(
        args.dataset_root,
        modality=args.modality,
        dataset_name=args.dataset_name,
        split=args.split,
    )
    subset = build_subset_manifest(samples, subset_count=args.subset_count, subset_seed=args.subset_seed)
    write_manifest_jsonl(args.output, subset)
    print(f"Wrote {len(subset)} samples to {args.output}")
    return 0


def _prepare_runtime_env(config: dict) -> None:
    runtime = dict(config.get("runtime") or {})
    env = dict(runtime.get("env") or {})
    for key, value in env.items():
        if value is None or value == "":
            continue
        os.environ[str(key)] = str(value)


def _resolve_config_path(value: str | os.PathLike[str], *, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def _default_config_base_dir(config_path: Path) -> Path:
    if config_path.parent.name == "configs" and (config_path.parent.parent / "ic_estimation").exists():
        return config_path.parent.parent
    return config_path.parent


def _enumerate_dataset_samples(dataset_cfg: dict, adapter, adapter_name: str):
    if "root" not in dataset_cfg:
        raise FileNotFoundError("Manifest does not exist and dataset.root is not configured")
    return enumerate_samples(
        dataset_cfg["root"],
        modality=str(dataset_cfg.get("modality", adapter.modality)),
        dataset_name=str(dataset_cfg.get("name", adapter_name)),
        split=str(dataset_cfg.get("split", "")),
    )


def _load_or_refresh_manifest(manifest_path: Path, dataset_cfg: dict, adapter, adapter_name: str):
    if not manifest_path.exists():
        samples_all = _enumerate_dataset_samples(dataset_cfg, adapter, adapter_name)
        write_manifest_jsonl(manifest_path, samples_all)
        print(f"Wrote {len(samples_all)} samples to {manifest_path}")
        return samples_all

    samples = read_manifest_jsonl(manifest_path)
    subset_count = dataset_cfg.get("subset_count")
    if subset_count is None or "root" not in dataset_cfg or len(samples) >= int(subset_count):
        return samples

    samples_all = _enumerate_dataset_samples(dataset_cfg, adapter, adapter_name)
    if len(samples_all) > len(samples):
        write_manifest_jsonl(manifest_path, samples_all)
        print(
            f"Refreshed manifest {manifest_path}: "
            f"{len(samples)} cached samples -> {len(samples_all)} enumerated samples",
            flush=True,
        )
        return samples_all
    return samples


def _run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path).to_dict()
    _prepare_runtime_env(config)
    config_base_dir = _default_config_base_dir(config_path)
    project_root = _resolve_config_path(config.get("paths", {}).get("project_root", "."), base_dir=config_base_dir)
    repo_root = _resolve_config_path(config.get("repo_root", "."), base_dir=project_root)
    adapters = make_builtin_adapters(repo_root)
    adapter_name = str(config["method_name"]).lower()
    if adapter_name not in adapters:
        raise ValueError(f"Unknown built-in adapter {adapter_name!r}; choices: {sorted(adapters)}")
    adapter = adapters[adapter_name]
    configured_payload_length = int(config.get("payload_length", adapter.payload_length))
    if configured_payload_length != int(adapter.payload_length):
        adapter = dataclasses.replace(adapter, payload_length=configured_payload_length)
        post_init = getattr(adapter, "__post_init__", None)
        if callable(post_init):
            post_init()
    dataset_cfg = dict(config.get("dataset") or {})
    if dataset_cfg.get("root"):
        dataset_cfg["root"] = _resolve_config_path(dataset_cfg["root"], base_dir=project_root)
    manifest_path = Path(config.get("manifest_path") or dataset_cfg.get("manifest_path", ""))
    if not manifest_path:
        raise ValueError("Config must set manifest_path or dataset.manifest_path")
    if not manifest_path.is_absolute():
        manifest_path = _resolve_config_path(manifest_path, base_dir=project_root)
    try:
        samples = _load_or_refresh_manifest(manifest_path, dataset_cfg, adapter, adapter_name)
    except FileNotFoundError as exc:
        if str(exc) == "Manifest does not exist and dataset.root is not configured":
            raise FileNotFoundError(f"{exc}: {manifest_path}") from exc
        raise
    if dataset_cfg.get("subset_count") is not None:
        samples = build_subset_manifest(
            samples,
            subset_count=int(dataset_cfg["subset_count"]),
            subset_seed=int(dataset_cfg.get("subset_seed", config.get("message_seed", 1453))),
        )
    channel_cfg = config.get("channel") or {}
    channel = ChannelConfig(
        name=str(channel_cfg.get("name", "clean")),
        params=dict(channel_cfg.get("params") or {}),
        seed=int(channel_cfg.get("seed", 0)),
    )
    code_family = normalize_code_family(str(config.get("code_family", config.get("code_set", "bch"))))
    min_k = int(config.get("min_k", 1))
    max_k = None if config.get("max_k") is None else int(config["max_k"])
    if code_family == "bch":
        candidates = shortened_candidates(
            payload_length=configured_payload_length,
            min_k=min_k,
            max_k=max_k,
        )
    else:
        candidates = build_code_candidates(
            payload_length=configured_payload_length,
            code_family=code_family,
            min_k=min_k,
            max_k=max_k,
            rm_options=dict(config.get("rm_options") or {}),
        )
    candidate_pool_size = int(config.get("candidate_pool_size", len(candidates)))
    candidates = candidates[:candidate_pool_size]
    candidate_min_k = min((int(candidate.k) for candidate in candidates), default=0)
    candidate_max_k = max((int(candidate.k) for candidate in candidates), default=0)
    search_strategy = str(config.get("search_strategy", "brute_force"))
    evaluation_mode = str(config.get("evaluation_mode", "empirical"))
    evaluation_target = _normalize_evaluation_target(str(config.get("evaluation_target", "L2_T+L3")))
    raw_baseline_trial_multiplier = max(1, int(config.get("raw_baseline_trial_multiplier", 1)))
    workers_cfg = dict(config.get("workers") or {})
    method_params = dict(config.get("method_params") or {})
    if adapter_name == "svd_chang":
        method_params["payload_length"] = configured_payload_length
    if adapter.modality == "audio":
        method_params.setdefault("batch_truncate_seconds", 3.0)
    worker_count = int(workers_cfg.get("count", config.get("worker_count", 8)))
    raw_worker_devices = workers_cfg.get("devices", config.get("worker_devices", []))
    if isinstance(raw_worker_devices, str):
        worker_devices = [item.strip() for item in raw_worker_devices.split(",") if item.strip()]
    else:
        worker_devices = [str(item) for item in (raw_worker_devices or []) if str(item)]
    if not worker_devices and method_params.get("device") is not None:
        worker_devices = [str(method_params["device"])]
    max_candidates = None if config.get("max_candidates") is None else int(config["max_candidates"])
    planned_candidates = len(candidates)
    if max_candidates is not None:
        planned_candidates = min(planned_candidates, max(0, max_candidates))
    candidate_trials = len(samples) * planned_candidates * int(config.get("trials_per_sample", 1))
    raw_baseline_trials = len(samples) * int(config.get("trials_per_sample", 1)) * raw_baseline_trial_multiplier
    total_trials = candidate_trials
    worker_calls = (raw_baseline_trials + candidate_trials) * 2
    if evaluation_mode.strip().lower().replace("-", "_") in {"simulated_from_raw", "fast", "simulate", "simulation", "simulated"}:
        worker_calls = raw_baseline_trials * 2
    print(
        "Experiment plan: "
        f"samples={len(samples)} "
        f"candidates={planned_candidates} "
        f"candidate_pool={len(candidates)} "
        f"search_k=[{candidate_min_k},{candidate_max_k}] "
        f"trials_per_sample={int(config.get('trials_per_sample', 1))} "
        f"raw_baseline_trial_multiplier={raw_baseline_trial_multiplier} "
        f"raw_baseline_trials={raw_baseline_trials} "
        f"candidate_trials={candidate_trials} "
        f"total_trials={total_trials} "
        f"worker_calls={worker_calls} "
        f"trial_batch_size={config.get('trial_batch_size', 'adapter_default')} "
        f"batch_truncate_seconds={method_params.get('batch_truncate_seconds', 'n/a')} "
        f"search_strategy={search_strategy} "
        f"evaluation_mode={evaluation_mode} "
        f"evaluation_target={evaluation_target} "
        f"max_candidates={max_candidates if max_candidates is not None else 'all'} "
        f"candidate_pool_size={candidate_pool_size} "
        f"code_family={code_family} "
        f"workers={worker_count} "
        f"worker_devices={','.join(worker_devices)}",
        flush=True,
    )
    experiment = ExperimentConfig(
        run_name=str(config.get("run_name", f"{adapter_name}-{channel.name}")),
        alpha=float(config.get("alpha", 0.05)),
        message_seed=int(config.get("message_seed", 1453)),
        trials_per_sample=int(config.get("trials_per_sample", 1)),
        channel=channel,
        method_params=method_params,
        message_mode=str(config.get("message_mode", "uid_trial")),
        payload_filler=str(config.get("payload_filler", "zeros")),
        worker_count=worker_count,
        worker_devices=worker_devices,
        persist_trial_artifacts=bool(config.get("persist_trial_artifacts", False)),
        trial_batch_size=None if config.get("trial_batch_size") is None else int(config["trial_batch_size"]),
        search_strategy=search_strategy,
        max_candidates=max_candidates,
        evaluation_mode=evaluation_mode,
        evaluation_target=evaluation_target,
        raw_baseline_trial_multiplier=raw_baseline_trial_multiplier,
        code_family=code_family,
    )
    runner = IdentityCapacityRunner(
        adapter=adapter,
        samples=samples,
        candidates=candidates,
        config=experiment,
        output_dir=_resolve_config_path(config["output_dir"], base_dir=project_root),
    )
    summary = runner.run()
    print(f"Wrote run summary to {_resolve_config_path(config['output_dir'], base_dir=project_root) / 'summary.json'}")
    print(
        f"max_M={summary['max_M']} "
        f"log2_M={summary['log2_M']} "
        f"search_min_k={summary.get('search_min_k', candidate_min_k)} "
        f"search_max_k={summary.get('search_max_k', candidate_max_k)} "
        f"search_strategy={summary.get('search_strategy', search_strategy)} "
        f"evaluation_mode={summary.get('evaluation_mode', evaluation_mode)} "
        f"evaluation_target={summary.get('evaluation_target', evaluation_target)}"
    )
    return 0


def _normalize_evaluation_target(value: str | None) -> str:
    target = str(value or "L2_T+L3").strip().lower()
    target = target.replace("-", "_").replace("+", "_").replace(" ", "_")
    aliases = {
        "l2_t_l3": "l2_t_l3",
        "l2t_l3": "l2_t_l3",
        "l2_m_l3": "l2_m_l3",
        "l2m_l3": "l2_m_l3",
        "l2_t_l2_m": "l2_t_l2_m",
        "l2t_l2m": "l2_t_l2_m",
    }
    if target not in aliases:
        raise ValueError("Unsupported evaluation_target: " + str(value))
    return aliases[target]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IC Estimation command-line toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="Build a deterministic sample manifest")
    manifest.add_argument("--dataset-root", required=True)
    manifest.add_argument("--modality", required=True, choices=["audio", "image"])
    manifest.add_argument("--dataset-name", required=True)
    manifest.add_argument("--split", default="")
    manifest.add_argument("--subset-count", type=int, default=None)
    manifest.add_argument("--subset-seed", type=int, default=1453)
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(func=_manifest)

    run = subparsers.add_parser("run", help="Run an Identity Capacity experiment from JSON config")
    run.add_argument("--config", required=True)
    run.set_defaults(func=_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
