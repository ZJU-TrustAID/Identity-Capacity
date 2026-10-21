from __future__ import annotations

import csv
import time
import tempfile
import shutil
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ic_estimation.adapters import WatermarkAdapter
from ic_estimation.channels import ChannelConfig, apply_channel
from ic_estimation.datasets import SampleRecord, write_manifest_jsonl
from ic_estimation.identity import bits_to_int, identity_bits
from ic_estimation.l2 import (
    binomial_failure_probability,
    select_l2_capacity_from_candidates,
    select_l2_capacity_monte_carlo_from_candidates,
)
from ic_estimation.metrics import bit_diagnostics, compute_candidate_metrics
from ic_estimation.uncoded import select_uncoded_l2_capacity, select_uncoded_l3_capacity
from ic_estimation.utils import write_json, write_jsonl


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    alpha: float
    message_seed: int
    trials_per_sample: int
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    method_params: dict[str, Any] = field(default_factory=dict)
    message_mode: str = "uid_trial"
    payload_filler: str = "zeros"
    worker_count: int = 8
    worker_devices: list[str] = field(default_factory=list)
    persist_trial_artifacts: bool = False
    trial_batch_size: int | None = None
    search_strategy: str = "brute_force"
    max_candidates: int | None = None
    evaluation_mode: str = "empirical"
    evaluation_target: str = "l2_t_l3"
    raw_baseline_trial_multiplier: int = 1
    code_family: str = "bch"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "alpha": float(self.alpha),
            "message_seed": int(self.message_seed),
            "trials_per_sample": int(self.trials_per_sample),
            "channel": self.channel.to_dict(),
            "method_params": dict(self.method_params or {}),
            "message_mode": self.message_mode,
            "payload_filler": self.payload_filler,
            "worker_count": int(self.worker_count),
            "worker_devices": list(self.worker_devices or []),
            "persist_trial_artifacts": bool(self.persist_trial_artifacts),
            "trial_batch_size": None if self.trial_batch_size is None else int(self.trial_batch_size),
            "search_strategy": self.search_strategy,
            "max_candidates": None if self.max_candidates is None else int(self.max_candidates),
            "evaluation_mode": self.evaluation_mode,
            "evaluation_target": self.evaluation_target,
            "raw_baseline_trial_multiplier": int(self.raw_baseline_trial_multiplier),
            "code_family": self.code_family,
        }


@dataclass(frozen=True)
class PreparedTrial:
    row_index: int
    sample: SampleRecord
    trial_index: int
    identity_bits: np.ndarray

    def info_for_k(self, k: int) -> np.ndarray:
        return np.asarray(self.identity_bits[: int(k)], dtype=np.uint8)

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": int(self.row_index),
            "sample_id": self.sample.sample_id,
            "sample_path": str(self.sample.path),
            "trial_index": int(self.trial_index),
            "max_k": int(self.identity_bits.size),
            "identity_bits": self.identity_bits.astype(np.uint8).tolist(),
        }


class IdentityCapacityRunner:
    def __init__(
        self,
        *,
        adapter: WatermarkAdapter,
        adapters: list[WatermarkAdapter] | None = None,
        samples: list[SampleRecord],
        candidates: list[Any],
        config: ExperimentConfig,
        output_dir: str | Path,
    ):
        self.adapter = adapter
        self.adapters = list(adapters) if adapters is not None else []
        self.samples = list(samples)
        self.candidates = list(candidates)
        self.config = config
        self.output_dir = Path(output_dir).expanduser().resolve()

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.output_dir / "config.json", self.config.to_dict())
        write_manifest_jsonl(self.output_dir / "subset_manifest.jsonl", self.samples)
        write_json(
            self.output_dir / "method_params.json",
            {
                "adapter_name": self.adapter.name,
                "modality": self.adapter.modality,
                "payload_length": int(self.adapter.payload_length),
                "native_coded_output": bool(getattr(self.adapter, "native_coded_output", False)),
                "method_params": self.config.method_params,
            },
        )
        write_json(self.output_dir / "channel_params.json", self.config.channel.to_dict())
        write_jsonl(self.output_dir / "code_candidates.jsonl", [candidate.to_dict() for candidate in self.candidates])
        prepared_trials = self._prepare_trials()
        write_jsonl(self.output_dir / "prepared_trials.jsonl", [trial.to_dict() for trial in prepared_trials])

        run_started = time.perf_counter()
        try:
            worker_adapters = self._build_worker_adapters()
            worker_devices = self._worker_devices(len(worker_adapters))
            print(
                f"[parallel] worker_count={len(worker_adapters)} "
                f"worker_devices={','.join(worker_devices)}",
                flush=True,
            )
            evaluation_mode = self._normalized_evaluation_mode()
            evaluation_target = self._normalized_evaluation_target()
            raw_baseline = self._run_raw_baseline(
                prepared_trials,
                adapter=worker_adapters[0],
                worker_index=0,
                worker_device=worker_devices[0],
            )
            search_strategy = self._normalized_search_strategy()
            search_start_k = self._actual_search_start_k(raw_baseline)
            raw_l2_max_k = self._raw_l2_max_k(raw_baseline)
            candidate_budget = self._candidate_budget()
            print(
                f"[search] search_strategy={search_strategy} "
                f"search_start_k={_format_optional_int(search_start_k)} "
                f"raw_L2_max_k={_format_optional_int(raw_l2_max_k)} "
                f"max_candidates={_format_optional_int(candidate_budget)} "
                f"candidate_pool_size={len(self.candidates)} "
                f"evaluation_target={evaluation_target}",
                flush=True,
            )
            if evaluation_mode == "simulated_from_raw":
                print(
                    "[simulation] "
                    f"evaluation_mode={evaluation_mode} "
                    f"raw_error_patterns={int(raw_baseline.get('num_trials', 0))} "
                    f"baseline_multiplier={self._raw_baseline_trial_multiplier()}",
                    flush=True,
                )
            candidate_summaries: list[dict[str, Any]] = []
            if evaluation_target != "l2_t_l2_m":
                if search_strategy == "adaptive":
                    candidate_summaries = self._run_adaptive_candidates(
                        prepared_trials,
                        raw_baseline,
                        adapters=worker_adapters,
                        worker_devices=worker_devices,
                    )
                elif len(worker_adapters) == 1:
                    candidate_plan = self._brute_force_candidate_plan()
                    for idx, candidate in candidate_plan:
                        if self._candidate_budget_reached(candidate_summaries):
                            break
                        candidate_summaries.append(
                            self._run_candidate(
                                idx,
                                candidate,
                                prepared_trials,
                                adapter=worker_adapters[0],
                                worker_index=0,
                                worker_device=worker_devices[0],
                                raw_baseline=raw_baseline,
                            )
                        )
                else:
                    candidate_plan = self._brute_force_candidate_plan()
                    if candidate_budget is not None:
                        candidate_plan = candidate_plan[:candidate_budget]
                    results_by_index: dict[int, dict[str, Any]] = {}
                    with ThreadPoolExecutor(max_workers=len(worker_adapters)) as executor:
                        futures = [
                            executor.submit(
                                self._run_worker_candidate_group,
                                worker_index,
                                worker_adapters[worker_index],
                                worker_devices[worker_index],
                                [
                                    (idx, candidate)
                                    for idx, candidate in candidate_plan
                                    if idx % len(worker_adapters) == worker_index
                                ],
                                prepared_trials,
                                raw_baseline,
                            )
                            for worker_index in range(len(worker_adapters))
                        ]
                        for future in as_completed(futures):
                            for result in future.result():
                                results_by_index[int(result["candidate_index"])] = result
                    candidate_summaries = [
                        results_by_index[idx]
                        for idx, _candidate in candidate_plan
                        if idx in results_by_index
                    ]

            passing = [row for row in candidate_summaries if row["pass_alpha"]]
            passing.sort(key=lambda row: (int(row["k"]), -float(row["identity_failure_rate"])), reverse=True)
            best = passing[0] if passing else None
            l2_t_selection = raw_baseline.get("raw_l2_t_selection") or raw_baseline.get("raw_l2_selection")
            l2_m_selection = raw_baseline.get("raw_l2_monte_carlo_selection")
            l2_only_selection = l2_t_selection if isinstance(l2_t_selection, dict) else None
            total_seconds = time.perf_counter() - run_started
            search_min_k = min((int(row["k"]) for row in candidate_summaries), default=0)
            search_max_k = max((int(row["k"]) for row in candidate_summaries), default=0)
            summary = {
                "run_name": self.config.run_name,
                "adapter": self.adapter.name,
                "channel": self.config.channel.to_dict(),
                "alpha": float(self.config.alpha),
                "num_candidates": len(candidate_summaries),
                "candidate_pool_size": len(self.candidates),
                "max_candidates": candidate_budget,
                "search_min_k": search_min_k,
                "search_max_k": search_max_k,
                "search_strategy": search_strategy,
                "evaluation_mode": evaluation_mode,
                "evaluation_target": evaluation_target,
                "code_family": self.config.code_family,
                "raw_baseline_trial_multiplier": self._raw_baseline_trial_multiplier(),
                "search_start_k": search_start_k,
                "raw_l2_max_k": raw_l2_max_k,
                "raw_baseline": raw_baseline,
                "l2_t_selection": l2_t_selection,
                "l2_m_selection": l2_m_selection,
                "uncoded_l2_selection": raw_baseline.get("uncoded_l2_selection"),
                "uncoded_l3_selection": raw_baseline.get("uncoded_l3_selection"),
                "candidate_results": candidate_summaries,
                "best_candidate": best,
                "max_M": (
                    int(l2_only_selection["identity_count"])
                    if evaluation_target == "l2_t_l2_m" and l2_only_selection is not None
                    else 0 if best is None else int(best["identity_count"])
                ),
                "log2_M": (
                    int(l2_only_selection["k"])
                    if evaluation_target == "l2_t_l2_m" and l2_only_selection is not None
                    else 0 if best is None else int(best["k"])
                ),
                "timing": {
                    "total_seconds": total_seconds,
                    "candidates": [
                        {
                            "candidate_index": int(row["candidate_index"]),
                            "seconds": float(row.get("elapsed_seconds", 0.0)),
                            "worker_index": int(row.get("worker_index", 0)),
                            "worker_device": str(row.get("worker_device", "")),
                        }
                        for row in candidate_summaries
                    ],
                },
                "parallel": {
                    "worker_count": len(worker_adapters),
                    "worker_devices": worker_devices,
                },
            }
            write_json(self.output_dir / "summary.json", summary)
            self._write_report(summary)
            self._write_code_features_wide(summary)
            print(f"[timing] run_total seconds={total_seconds:.3f}", flush=True)
            return summary
        finally:
            for adapter in self._unique_adapters():
                close = getattr(adapter, "close", None)
                if callable(close):
                    close()

    def _unique_adapters(self) -> list[WatermarkAdapter]:
        adapters = self.adapters if self.adapters else [self.adapter]
        unique: list[WatermarkAdapter] = []
        seen: set[int] = set()
        for adapter in adapters:
            marker = id(adapter)
            if marker in seen:
                continue
            seen.add(marker)
            unique.append(adapter)
        return unique

    def _build_worker_adapters(self) -> list[WatermarkAdapter]:
        worker_count = max(1, int(self.config.worker_count or 1))
        if self.adapters:
            return self.adapters[:worker_count]
        if worker_count == 1:
            return [self.adapter]
        adapters = [self.adapter]
        for worker_index in range(1, worker_count):
            adapters.append(self.adapter.clone_for_worker(worker_index))
        self.adapters = adapters
        return adapters

    def _worker_devices(self, worker_count: int) -> list[str]:
        configured = [str(device) for device in (self.config.worker_devices or []) if str(device)]
        if not configured:
            configured = [str(self.config.method_params.get("device", "cpu"))]
        return [configured[index % len(configured)] for index in range(int(worker_count))]

    def _normalized_search_strategy(self) -> str:
        strategy = str(self.config.search_strategy or "brute_force").strip().lower().replace("-", "_")
        if strategy in {"bruteforce", "brute"}:
            strategy = "brute_force"
        if strategy not in {"brute_force", "adaptive"}:
            raise ValueError(f"Unsupported search strategy: {self.config.search_strategy}")
        return strategy

    def _normalized_evaluation_mode(self) -> str:
        mode = str(self.config.evaluation_mode or "empirical").strip().lower().replace("-", "_")
        if mode in {"real", "actual"}:
            mode = "empirical"
        if mode in {"fast", "simulate", "simulation", "simulated"}:
            mode = "simulated_from_raw"
        if mode not in {"empirical", "simulated_from_raw"}:
            raise ValueError(f"Unsupported evaluation mode: {self.config.evaluation_mode}")
        return mode

    def _normalized_evaluation_target(self) -> str:
        target = str(self.config.evaluation_target or "l2_t_l3").strip().lower()
        target = target.replace("-", "_").replace("+", "_").replace(" ", "_")
        aliases = {
            "l2_t_l3": "l2_t_l3",
            "l2t_l3": "l2_t_l3",
            "t_l3": "l2_t_l3",
            "l2_m_l3": "l2_m_l3",
            "l2m_l3": "l2_m_l3",
            "m_l3": "l2_m_l3",
            "mc_l3": "l2_m_l3",
            "monte_carlo_l3": "l2_m_l3",
            "l2_t_l2_m": "l2_t_l2_m",
            "l2t_l2m": "l2_t_l2_m",
            "l2_t_m": "l2_t_l2_m",
            "both_l2": "l2_t_l2_m",
        }
        if target not in aliases:
            raise ValueError(f"Unsupported evaluation target: {self.config.evaluation_target}")
        return aliases[target]

    def _uses_monte_carlo_l2(self) -> bool:
        return self._normalized_evaluation_target() in {"l2_m_l3", "l2_t_l2_m"}

    def _raw_baseline_trial_multiplier(self) -> int:
        return max(1, int(self.config.raw_baseline_trial_multiplier or 1))

    def _brute_force_candidate_plan(self) -> list[tuple[int, Any]]:
        return list(enumerate(self.candidates))

    def _candidate_budget(self) -> int | None:
        if self.config.max_candidates is None:
            return None
        return max(0, int(self.config.max_candidates))

    def _candidate_budget_reached(self, rows: list[dict[str, Any]]) -> bool:
        budget = self._candidate_budget()
        return budget is not None and len(rows) >= budget

    def _search_start_k(self, raw_baseline: dict[str, Any]) -> int | None:
        raw_l2_max_k = self._raw_l2_max_k(raw_baseline)
        if raw_l2_max_k is not None:
            return raw_l2_max_k
        if self.candidates:
            return max(int(candidate.k) for candidate in self.candidates)
        return None

    def _raw_l2_max_k(self, raw_baseline: dict[str, Any]) -> int | None:
        selection = self._active_raw_l2_selection(raw_baseline)
        if selection is None:
            return 0
        if isinstance(selection, dict) and selection.get("k") is not None:
            return int(selection["k"])
        return None

    def _active_raw_l2_selection(self, raw_baseline: dict[str, Any]) -> Any:
        if not isinstance(raw_baseline, dict):
            return None
        if self._normalized_evaluation_target() == "l2_m_l3":
            return raw_baseline.get("raw_l2_monte_carlo_selection")
        return raw_baseline.get("raw_l2_selection")

    def _actual_search_start_k(self, raw_baseline: dict[str, Any]) -> int | None:
        grouped = self._candidates_by_k()
        if not grouped:
            return None
        available_ks = sorted(grouped)
        start = self._search_start_k(raw_baseline)
        if start is None:
            return available_ks[-1]
        if int(start) in grouped:
            return int(start)
        return min(available_ks, key=lambda value: (abs(value - int(start)), -value))

    def _candidates_by_k(self) -> dict[int, list[tuple[int, Any]]]:
        grouped: dict[int, list[tuple[int, Any]]] = {}
        for candidate_index, candidate in enumerate(self.candidates):
            grouped.setdefault(int(candidate.k), []).append((candidate_index, candidate))
        return grouped

    def _run_adaptive_candidates(
        self,
        prepared_trials: list[PreparedTrial],
        raw_baseline: dict[str, Any],
        *,
        adapters: list[WatermarkAdapter],
        worker_devices: list[str],
    ) -> list[dict[str, Any]]:
        grouped = self._candidates_by_k()
        if not grouped:
            return []
        worker_adapters = list(adapters or [self.adapter])
        active_worker_devices = list(worker_devices or self._worker_devices(len(worker_adapters)))
        available_ks = sorted(grouped)
        start_k = self._actual_search_start_k(raw_baseline)
        if start_k is None:
            return []

        results: list[dict[str, Any]] = []
        evaluated_ks: set[int] = set()

        def evaluate_k(k_value: int) -> list[dict[str, Any]]:
            if int(k_value) in evaluated_ks:
                return []
            budget = self._candidate_budget()
            remaining = None if budget is None else budget - len(results)
            if remaining is not None and remaining <= 0:
                return []
            evaluated_ks.add(int(k_value))
            assignments = grouped[int(k_value)]
            if remaining is not None:
                assignments = assignments[:remaining]
            rows_by_index: dict[int, dict[str, Any]] = {}
            if len(worker_adapters) == 1:
                for row in self._run_worker_candidate_group(
                    0,
                    worker_adapters[0],
                    active_worker_devices[0],
                    assignments,
                    prepared_trials,
                    raw_baseline,
                ):
                    rows_by_index[int(row["candidate_index"])] = row
            else:
                with ThreadPoolExecutor(max_workers=len(worker_adapters)) as executor:
                    futures = []
                    for worker_index, worker_adapter in enumerate(worker_adapters):
                        worker_assignments = [
                            assignment
                            for assignment_index, assignment in enumerate(assignments)
                            if assignment_index % len(worker_adapters) == worker_index
                        ]
                        if not worker_assignments:
                            continue
                        futures.append(
                            executor.submit(
                                self._run_worker_candidate_group,
                                worker_index,
                                worker_adapter,
                                active_worker_devices[worker_index],
                                worker_assignments,
                                prepared_trials,
                                raw_baseline,
                            )
                        )
                    for future in as_completed(futures):
                        for row in future.result():
                            rows_by_index[int(row["candidate_index"])] = row
            rows = [
                rows_by_index[int(candidate_index)]
                for candidate_index, _candidate in assignments
                if int(candidate_index) in rows_by_index
            ]
            results.extend(rows)
            return rows

        current_rows = evaluate_k(start_k)
        if any(bool(row.get("pass_alpha")) for row in current_rows):
            higher_ks = [k_value for k_value in available_ks if k_value > start_k]
            for k_value in higher_ks:
                if self._candidate_budget_reached(results):
                    break
                rows = evaluate_k(k_value)
                if not any(bool(row.get("pass_alpha")) for row in rows):
                    break
        else:
            lower_ks = [k_value for k_value in reversed(available_ks) if k_value < start_k]
            for k_value in lower_ks:
                if self._candidate_budget_reached(results):
                    break
                rows = evaluate_k(k_value)
                if any(bool(row.get("pass_alpha")) for row in rows):
                    break

        return sorted(results, key=lambda row: int(row["candidate_index"]))

    def _run_worker_candidate_group(
        self,
        worker_index: int,
        adapter: WatermarkAdapter,
        worker_device: str,
        assignments: list[tuple[int, Any]],
        prepared_trials: list[PreparedTrial],
        raw_baseline: dict[str, Any],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for candidate_index, candidate in assignments:
            results.append(
                self._run_candidate(
                    candidate_index,
                    candidate,
                    prepared_trials,
                    adapter=adapter,
                    worker_index=worker_index,
                    worker_device=worker_device,
                    raw_baseline=raw_baseline,
                )
            )
        return results

    def _empty_payload(self, payload_length: int | None = None, *, prepared: PreparedTrial | None = None) -> np.ndarray:
        length = int(self.adapter.payload_length if payload_length is None else payload_length)
        filler = str(self.config.payload_filler or "zeros").strip().lower()
        if filler == "zeros":
            return np.zeros(length, dtype=np.uint8)
        if filler in {"random", "deterministic_random"}:
            if prepared is None:
                key = f"filler|{self.config.message_seed}|length={length}"
            else:
                key = f"filler|{prepared.sample.sample_id}|trial={int(prepared.trial_index)}"
            return identity_bits(key, length, self.config.message_seed, 0, mode="uid")
        raise ValueError(f"Unsupported payload filler: {self.config.payload_filler}")

    def _prepare_trials(self) -> list[PreparedTrial]:
        max_k = max((int(candidate.k) for candidate in self.candidates), default=0)
        rows: list[PreparedTrial] = []
        row_index = 0
        for sample in self.samples:
            for trial_index in range(int(self.config.trials_per_sample)):
                rows.append(
                    PreparedTrial(
                        row_index=row_index,
                        sample=sample,
                        trial_index=trial_index,
                        identity_bits=identity_bits(
                            sample.sample_id,
                            max_k,
                            self.config.message_seed,
                            trial_index,
                            mode=self.config.message_mode,
                        ),
                    )
                )
                row_index += 1
        return rows

    def _prepare_raw_baseline_trials(self, prepared_trials: list[PreparedTrial]) -> list[PreparedTrial]:
        multiplier = self._raw_baseline_trial_multiplier()
        if multiplier <= 1:
            return prepared_trials
        rows: list[PreparedTrial] = []
        row_index = 0
        for repeat_index in range(multiplier):
            for prepared in prepared_trials:
                trial_index = int(prepared.trial_index) + repeat_index * int(self.config.trials_per_sample)
                rows.append(
                    PreparedTrial(
                        row_index=row_index,
                        sample=prepared.sample,
                        trial_index=trial_index,
                        identity_bits=identity_bits(
                            prepared.sample.sample_id,
                            int(prepared.identity_bits.size),
                            self.config.message_seed,
                            trial_index,
                            mode=self.config.message_mode,
                        ),
                    )
                )
                row_index += 1
        return rows

    def _run_raw_baseline(
        self,
        prepared_trials: list[PreparedTrial],
        *,
        adapter: WatermarkAdapter,
        worker_index: int = 0,
        worker_device: str | None = None,
    ) -> dict[str, Any]:
        active_device = str(worker_device or self.config.method_params.get("device", "cpu"))
        method_params = {**dict(self.config.method_params or {}), "device": active_device}
        started = time.perf_counter()
        baseline_dir = self.output_dir / "raw_baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        base_trial_count = len(prepared_trials)
        raw_prepared_trials = self._prepare_raw_baseline_trials(prepared_trials)

        truth_payload_rows: list[np.ndarray] = []
        decoded_payload_rows: list[np.ndarray] = []
        records: list[dict[str, Any]] = []

        def build_trial_job(prepared: PreparedTrial, trial_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
            row_index = int(prepared.row_index)
            trial_dir = trial_root / f"trial_{row_index:06d}"
            payload = identity_bits(
                prepared.sample.sample_id,
                int(adapter.payload_length),
                self.config.message_seed,
                int(prepared.trial_index),
                mode=self.config.message_mode,
            )
            context = {
                "row_index": row_index,
                "sample": prepared.sample,
                "trial_index": int(prepared.trial_index),
                "trial_dir": trial_dir,
                "payload": np.asarray(payload, dtype=np.uint8),
            }
            job = {"sample_path": prepared.sample.path, "payload_bits": payload, "work_dir": trial_dir / "embed"}
            return context, job

        def run_trial_batch(trial_contexts: list[dict[str, Any]], embed_jobs: list[dict[str, Any]]) -> None:
            embed_results = adapter.embed_batch(embed_jobs, method_params=method_params)
            decode_jobs: list[dict[str, Any]] = []
            for context, embed_result in zip(trial_contexts, embed_results, strict=True):
                row_index = int(context["row_index"])
                trial_dir = Path(context["trial_dir"])
                evaluated_suffix = ".wav" if adapter.modality == "audio" else ".png"
                evaluated_path = trial_dir / f"evaluated{evaluated_suffix}"
                apply_channel(
                    embed_result.watermarked_path,
                    evaluated_path,
                    modality=adapter.modality,
                    config=self.config.channel,
                    trial_seed=row_index,
                )
                (trial_dir / "decode").mkdir(parents=True, exist_ok=True)
                context["embed_result"] = embed_result
                decode_jobs.append(
                    {
                        "sample_path": evaluated_path,
                        "metadata_path": embed_result.metadata_path,
                        "work_dir": trial_dir / "decode",
                    }
                )

            decode_results = adapter.decode_batch(decode_jobs, method_params=method_params)
            for context, decode_result in zip(trial_contexts, decode_results, strict=True):
                row_index = int(context["row_index"])
                sample = context["sample"]
                payload = np.asarray(context["payload"], dtype=np.uint8)
                decoded_payload = np.zeros(int(adapter.payload_length), dtype=np.uint8)
                decoded_payload[
                    : min(decoded_payload.size, decode_result.decoded_bits.size)
                ] = decode_result.decoded_bits[: decoded_payload.size]
                truth_payload_rows.append(payload)
                decoded_payload_rows.append(decoded_payload)
                records.append(
                    {
                        "row_index": row_index,
                        "sample_id": sample.sample_id,
                        "sample_path": str(sample.path),
                        "trial_index": int(context["trial_index"]),
                        "payload_bits": payload.astype(np.uint8).tolist(),
                        "decoded_payload_bits": decoded_payload.astype(np.uint8).tolist(),
                        "worker_status": decode_result.status,
                        "embed_status": context["embed_result"].status,
                        "detect_prob": decode_result.detect_prob,
                        "num_payload_errors": int(np.count_nonzero(payload != decoded_payload)),
                    }
                )

        configured_trial_batch_size = self.config.trial_batch_size
        if configured_trial_batch_size is None:
            configured_trial_batch_size = int(getattr(adapter, "default_trial_batch_size", 1) or 1)
        trial_batch_size = max(1, int(configured_trial_batch_size))

        temp_dirs: list[tempfile.TemporaryDirectory[str]] = []
        trial_contexts: list[dict[str, Any]] = []
        embed_jobs: list[dict[str, Any]] = []
        try:
            for prepared in raw_prepared_trials:
                if self.config.persist_trial_artifacts:
                    root = baseline_dir / "trials"
                else:
                    temp_dir = tempfile.TemporaryDirectory(prefix=f"ic-raw_{int(prepared.row_index):06d}-")
                    temp_dirs.append(temp_dir)
                    root = Path(temp_dir.name)
                context, job = build_trial_job(prepared, root)
                trial_contexts.append(context)
                embed_jobs.append(job)
                if len(embed_jobs) >= trial_batch_size:
                    run_trial_batch(trial_contexts, embed_jobs)
                    trial_contexts = []
                    embed_jobs = []
                    for directory in temp_dirs:
                        directory.cleanup()
                    temp_dirs = []
            if embed_jobs:
                run_trial_batch(trial_contexts, embed_jobs)
        finally:
            for directory in temp_dirs:
                directory.cleanup()

        truth_payload = _stack_or_empty(truth_payload_rows, int(adapter.payload_length))
        decoded_payload = _stack_or_empty(decoded_payload_rows, int(adapter.payload_length))
        np.save(baseline_dir / "truth_payload.npy", truth_payload)
        np.save(baseline_dir / "decoded_payload.npy", decoded_payload)
        write_jsonl(baseline_dir / "trial_records.jsonl", records)

        raw_bit_accuracy = (
            float(np.mean(truth_payload == decoded_payload))
            if truth_payload.size and decoded_payload.size
            else 0.0
        )
        raw_bit_error_rate = 1.0 - raw_bit_accuracy
        raw_l2_selection = select_l2_capacity_from_candidates(
            candidates=self.candidates,
            bit_error_rate=raw_bit_error_rate,
            alpha=self.config.alpha,
        )
        raw_l2_monte_carlo_selection = None
        if self._uses_monte_carlo_l2():
            raw_l2_monte_carlo_selection = select_l2_capacity_monte_carlo_from_candidates(
                candidates=self.candidates,
                bit_error_rate=raw_bit_error_rate,
                alpha=self.config.alpha,
                seed=self.config.message_seed,
                start_k=None if raw_l2_selection is None else int(raw_l2_selection["k"]),
                worker_count=int(self.config.worker_count),
            )
        uncoded_l2_selection = select_uncoded_l2_capacity(
            payload_length=int(adapter.payload_length),
            bit_error_rate=raw_bit_error_rate,
            alpha=self.config.alpha,
        )
        uncoded_l3_selection = select_uncoded_l3_capacity(
            truth_payload,
            decoded_payload,
            alpha=self.config.alpha,
        )
        elapsed_seconds = time.perf_counter() - started
        payload = {
            "num_trials": int(truth_payload.shape[0]),
            "num_base_trials": int(base_trial_count),
            "baseline_trial_multiplier": self._raw_baseline_trial_multiplier(),
            "payload_length": int(adapter.payload_length),
            "raw_bit_accuracy": raw_bit_accuracy,
            "raw_bit_error_rate": raw_bit_error_rate,
            "bit_diagnostics": bit_diagnostics(truth_payload, decoded_payload),
            "raw_l2_selection": raw_l2_selection,
            "raw_l2_t_selection": raw_l2_selection,
            "raw_l2_monte_carlo_selection": raw_l2_monte_carlo_selection,
            "uncoded_l2_selection": None
            if uncoded_l2_selection is None
            else uncoded_l2_selection.to_dict(),
            "uncoded_l3_selection": None
            if uncoded_l3_selection is None
            else uncoded_l3_selection.to_dict(),
            "baseline_dir": str(baseline_dir),
            "elapsed_seconds": elapsed_seconds,
            "worker_index": int(worker_index),
            "worker_device": active_device,
        }
        write_json(baseline_dir / "metrics.json", payload)
        selection = self._active_raw_l2_selection(payload)
        selection = selection if isinstance(selection, dict) else {}
        t_selection = payload["raw_l2_t_selection"] if isinstance(payload["raw_l2_t_selection"], dict) else {}
        m_selection = (
            payload["raw_l2_monte_carlo_selection"]
            if isinstance(payload["raw_l2_monte_carlo_selection"], dict)
            else {}
        )
        selection_k = selection.get("k", 0)
        selection_fail = selection.get("failure_probability", 1.0)
        uncoded_l2 = payload["uncoded_l2_selection"] if isinstance(payload["uncoded_l2_selection"], dict) else {}
        uncoded_l3 = payload["uncoded_l3_selection"] if isinstance(payload["uncoded_l3_selection"], dict) else {}
        print(
            "[raw baseline result]\n"
            f"raw_bit_acc={raw_bit_accuracy:.6f}\n"
            f"raw_L2_max_k={int(selection_k) if selection_k is not None else 0}\n"
            f"raw_L2_P_fail={float(selection_fail):.6f}\n"
            f"raw_L2_T_max_k={int(t_selection.get('k', 0) or 0)}\n"
            f"raw_L2_T_P_fail={float(t_selection.get('failure_probability', 1.0)):.6f}\n"
            f"raw_L2_M_max_k={int(m_selection.get('k', 0) or 0)}\n"
            f"raw_L2_M_P_fail={float(m_selection.get('failure_probability', 1.0)):.6f}\n"
            f"uncoded_L2_max_k={int(uncoded_l2.get('k', 0) or 0)}\n"
            f"uncoded_L2_P_fail={float(uncoded_l2.get('failure_probability', 1.0)):.6f}\n"
            f"uncoded_L3_max_k={int(uncoded_l3.get('k', 0) or 0)}\n"
            f"uncoded_L3_P_fail={float(uncoded_l3.get('failure_rate', 1.0)):.6f}\n"
            f"worker={worker_index} device={active_device}",
            flush=True,
        )
        print(
            f"[timing] raw_baseline seconds={elapsed_seconds:.3f} "
            f"worker={worker_index} device={active_device}",
            flush=True,
        )
        return payload

    def _run_candidate_simulated_from_raw(
        self,
        candidate_index: int,
        candidate: Any,
        prepared_trials: list[PreparedTrial],
        *,
        worker_index: int = 0,
        worker_device: str | None = None,
        raw_baseline: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        active_device = str(worker_device or self.config.method_params.get("device", "cpu"))
        candidate_started = time.perf_counter()
        candidate_dir = self.output_dir / "candidates" / f"candidate_{candidate_index:04d}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        write_json(candidate_dir / "candidate_config.json", candidate.to_dict())
        if max(candidate.active_positions, default=-1) >= int(self.adapter.payload_length):
            raise ValueError(f"candidate active position exceeds adapter payload length {self.adapter.payload_length}")
        if raw_baseline is None:
            raise ValueError("simulated_from_raw candidate evaluation requires raw_baseline")

        baseline_dir = Path(str(raw_baseline.get("baseline_dir", self.output_dir / "raw_baseline")))
        raw_truth = np.load(baseline_dir / "truth_payload.npy").astype(np.uint8)
        raw_decoded = np.load(baseline_dir / "decoded_payload.npy").astype(np.uint8)
        if raw_truth.shape != raw_decoded.shape:
            raise ValueError("raw baseline truth and decoded payload matrices must have the same shape")
        if raw_truth.ndim != 2:
            raise ValueError("raw baseline payload matrices must have shape [num_trials, payload_length]")
        if raw_truth.shape[0] <= 0:
            raise ValueError("raw baseline must contain at least one error pattern")
        raw_error_patterns = np.bitwise_xor(raw_truth, raw_decoded).astype(np.uint8)

        truth_payload_rows: list[np.ndarray] = []
        decoded_payload_rows: list[np.ndarray] = []
        truth_codeword_rows: list[np.ndarray] = []
        decoded_codeword_rows: list[np.ndarray] = []
        truth_info_rows: list[np.ndarray] = []
        decoded_info_rows: list[np.ndarray] = []
        success_rows: list[bool] = []
        reject_rows: list[bool] = []
        misattr_rows: list[bool] = []
        records: list[dict[str, Any]] = []
        active_positions = list(candidate.active_positions)
        rng = np.random.default_rng(
            _stable_seed(
                "simulated_from_raw",
                self.config.message_seed,
                candidate_index,
                candidate.n,
                candidate.k,
                raw_error_patterns.shape[0],
            )
        )
        pattern_indices = rng.integers(0, raw_error_patterns.shape[0], size=len(prepared_trials))

        for prepared, pattern_index in zip(prepared_trials, pattern_indices, strict=True):
            row_index = int(prepared.row_index)
            sample = prepared.sample
            trial_index = int(prepared.trial_index)
            info = prepared.info_for_k(candidate.k)
            codeword = candidate.encode(info)
            payload = self._empty_payload(self.adapter.payload_length, prepared=prepared)
            payload[active_positions] = codeword
            error_pattern = raw_error_patterns[int(pattern_index)]
            decoded_payload = np.bitwise_xor(payload, error_pattern[: payload.size]).astype(np.uint8)
            decoded_codeword = decoded_payload[active_positions]
            outcome = candidate.decode(decoded_codeword)
            if outcome.info_bits is None:
                decoded_info = np.zeros(candidate.k, dtype=np.uint8)
            else:
                decoded_info = outcome.info_bits
            identity_success = bool(outcome.status == "decoded" and np.array_equal(decoded_info, info))
            rejected = bool(outcome.status != "decoded")
            misattributed = bool(outcome.status == "decoded" and not np.array_equal(decoded_info, info))

            truth_payload_rows.append(payload)
            decoded_payload_rows.append(decoded_payload)
            truth_codeword_rows.append(codeword)
            decoded_codeword_rows.append(decoded_codeword)
            truth_info_rows.append(info)
            decoded_info_rows.append(decoded_info)
            success_rows.append(identity_success)
            reject_rows.append(rejected)
            misattr_rows.append(misattributed)
            records.append(
                {
                    "row_index": row_index,
                    "candidate_index": candidate_index,
                    "sample_id": sample.sample_id,
                    "sample_path": str(sample.path),
                    "trial_index": trial_index,
                    "identity_bits": info.astype(np.uint8).tolist(),
                    "identity_id": bits_to_int(info),
                    "decoded_identity_id": None if rejected else bits_to_int(decoded_info),
                    "decode_status": outcome.status,
                    "worker_status": "simulated_from_raw",
                    "embed_status": "simulated_from_raw",
                    "detect_prob": None,
                    "identity_success": identity_success,
                    "rejected": rejected,
                    "misattributed": misattributed,
                    "num_codeword_errors": int(np.count_nonzero(codeword != decoded_codeword)),
                    "num_corrected_errors": int(outcome.num_corrected_errors),
                    "raw_error_pattern_index": int(pattern_index),
                }
            )

        truth_payload = _stack_or_empty(truth_payload_rows, int(self.adapter.payload_length))
        decoded_payload = _stack_or_empty(decoded_payload_rows, int(self.adapter.payload_length))
        truth_codeword = _stack_or_empty(truth_codeword_rows, candidate.n)
        decoded_codeword = _stack_or_empty(decoded_codeword_rows, candidate.n)
        truth_info = _stack_or_empty(truth_info_rows, candidate.k)
        decoded_info = _stack_or_empty(decoded_info_rows, candidate.k)
        success = np.asarray(success_rows, dtype=bool)
        rejected = np.asarray(reject_rows, dtype=bool)
        misattributed = np.asarray(misattr_rows, dtype=bool)

        np.save(candidate_dir / "truth_payload.npy", truth_payload)
        np.save(candidate_dir / "decoded_payload.npy", decoded_payload)
        np.save(candidate_dir / "truth_codeword.npy", truth_codeword)
        np.save(candidate_dir / "decoded_codeword.npy", decoded_codeword)
        np.save(candidate_dir / "truth_info.npy", truth_info)
        np.save(candidate_dir / "decoded_info.npy", decoded_info)
        np.save(candidate_dir / "identity_success.npy", success)
        write_jsonl(candidate_dir / "trial_records.jsonl", records)

        metrics = compute_candidate_metrics(
            truth_codeword=truth_codeword,
            decoded_codeword=decoded_codeword,
            truth_info=truth_info,
            decoded_info=decoded_info,
            identity_success=success,
            rejected=rejected,
            misattributed=misattributed,
            alpha=self.config.alpha,
        )
        code_aware_bit_accuracy = float(metrics.raw_bit_accuracy)
        code_aware_bit_error_rate = 1.0 - code_aware_bit_accuracy
        code_aware_l2_p_fail = binomial_failure_probability(candidate.n, candidate.t, code_aware_bit_error_rate)
        raw_bit_accuracy = float(raw_baseline.get("raw_bit_accuracy", code_aware_bit_accuracy))
        raw_bit_error_rate = 1.0 - raw_bit_accuracy
        raw_l2_p_fail = binomial_failure_probability(candidate.n, candidate.t, raw_bit_error_rate)
        elapsed_seconds = time.perf_counter() - candidate_started
        metrics_payload = metrics.to_dict()
        metrics_payload["codeword_bit_accuracy"] = code_aware_bit_accuracy
        metrics_payload["code_aware_bit_accuracy"] = code_aware_bit_accuracy
        metrics_payload["code_aware_bit_diagnostics"] = dict(metrics_payload["bit_diagnostics"])
        metrics_payload["raw_bit_accuracy"] = raw_bit_accuracy
        payload = {
            **candidate.to_dict(),
            **metrics_payload,
            "candidate_index": candidate_index,
            "candidate_dir": str(candidate_dir),
            "raw_bit_error_rate": raw_bit_error_rate,
            "raw_l2_p_fail": raw_l2_p_fail,
            "code_aware_bit_error_rate": code_aware_bit_error_rate,
            "code_aware_l2_p_fail": code_aware_l2_p_fail,
            "l2_bit_error_rate": code_aware_bit_error_rate,
            "l2_theoretical_p_fail": code_aware_l2_p_fail,
            "elapsed_seconds": elapsed_seconds,
            "worker_index": int(worker_index),
            "worker_device": active_device,
            "simulated_from_raw": True,
            "raw_error_pattern_count": int(raw_error_patterns.shape[0]),
            "raw_baseline_trial_multiplier": self._raw_baseline_trial_multiplier(),
        }
        write_json(candidate_dir / "metrics.json", payload)
        ci95 = metrics.identity_failure_ci95
        print(
            f"[candidate {candidate_index + 1}/{len(self.candidates)} result]\n"
            f"(n,k,t)=({candidate.n},{candidate.k},{candidate.t})\n"
            f"Identity Capacity={candidate.identity_count}\n"
            f"pass_alpha={metrics.pass_alpha}\n"
            f"raw_bit_acc={raw_bit_accuracy:.6f}\n"
            f"code_aware_bit_acc={code_aware_bit_accuracy:.6f}\n"
            f"raw_L2_P_fail={raw_l2_p_fail:.6f}\n"
            f"code_aware_L2_P_fail={code_aware_l2_p_fail:.6f}\n"
            f"L3_P_fail={metrics.identity_failure_rate:.6f}\n"
            f"L3_P_fail_CI95=[{float(ci95['lower']):.6f},{float(ci95['upper']):.6f}]\n"
            f"worker={worker_index} device={active_device}",
            flush=True,
        )
        print(
            f"[timing] candidate {candidate_index + 1}/{len(self.candidates)} "
            f"seconds={elapsed_seconds:.3f} worker={worker_index} device={active_device}",
            flush=True,
        )
        return payload

    def _run_candidate(
        self,
        candidate_index: int,
        candidate: Any,
        prepared_trials: list[PreparedTrial],
        *,
        adapter: WatermarkAdapter | None = None,
        worker_index: int = 0,
        worker_device: str | None = None,
        raw_baseline: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        active_adapter = adapter or self.adapter
        active_device = str(worker_device or self.config.method_params.get("device", "cpu"))
        method_params = {**dict(self.config.method_params or {}), "device": active_device}
        candidate_started = time.perf_counter()
        candidate_dir = self.output_dir / "candidates" / f"candidate_{candidate_index:04d}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        write_json(candidate_dir / "candidate_config.json", candidate.to_dict())

        truth_payload_rows: list[np.ndarray] = []
        decoded_payload_rows: list[np.ndarray] = []
        truth_codeword_rows: list[np.ndarray] = []
        decoded_codeword_rows: list[np.ndarray] = []
        truth_info_rows: list[np.ndarray] = []
        decoded_info_rows: list[np.ndarray] = []
        success_rows: list[bool] = []
        reject_rows: list[bool] = []
        misattr_rows: list[bool] = []
        records: list[dict[str, Any]] = []
        if self._normalized_evaluation_mode() == "simulated_from_raw":
            return self._run_candidate_simulated_from_raw(
                candidate_index,
                candidate,
                prepared_trials,
                worker_index=worker_index,
                worker_device=worker_device,
                raw_baseline=raw_baseline,
            )
        if max(candidate.active_positions, default=-1) >= int(active_adapter.payload_length):
            raise ValueError(
                f"candidate active position exceeds adapter payload length {active_adapter.payload_length}"
            )

        persistent_trial_root = candidate_dir / "trials"
        if persistent_trial_root.exists():
            shutil.rmtree(persistent_trial_root)

        def build_trial_job(prepared: PreparedTrial, trial_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
            row_index = int(prepared.row_index)
            sample = prepared.sample
            trial_index = int(prepared.trial_index)
            trial_dir = trial_root / f"trial_{row_index:06d}"
            info = prepared.info_for_k(candidate.k)
            codeword = candidate.encode(info)
            payload = self._empty_payload(active_adapter.payload_length, prepared=prepared)
            payload[list(candidate.active_positions)] = codeword
            context = {
                "row_index": row_index,
                "sample": sample,
                "trial_index": trial_index,
                "trial_dir": trial_dir,
                "info": info,
                "codeword": codeword,
                "payload": payload,
            }
            job = {"sample_path": sample.path, "payload_bits": payload, "work_dir": trial_dir / "embed"}
            return context, job

        def run_trial_batch(trial_contexts: list[dict[str, Any]], embed_jobs: list[dict[str, Any]]) -> None:
            embed_results = active_adapter.embed_batch(embed_jobs, method_params=method_params)
            decode_jobs: list[dict[str, Any]] = []
            for context, embed_result in zip(trial_contexts, embed_results, strict=True):
                row_index = int(context["row_index"])
                trial_dir = Path(context["trial_dir"])
                evaluated_suffix = ".wav" if active_adapter.modality == "audio" else ".png"
                evaluated_path = trial_dir / f"evaluated{evaluated_suffix}"
                apply_channel(
                    embed_result.watermarked_path,
                    evaluated_path,
                    modality=active_adapter.modality,
                    config=self.config.channel,
                    trial_seed=row_index,
                )
                (trial_dir / "decode").mkdir(parents=True, exist_ok=True)
                context["embed_result"] = embed_result
                decode_jobs.append(
                    {
                        "sample_path": evaluated_path,
                        "metadata_path": embed_result.metadata_path,
                        "work_dir": trial_dir / "decode",
                    }
                )

            decode_results = active_adapter.decode_batch(decode_jobs, method_params=method_params)
            for context, decode_result in zip(trial_contexts, decode_results, strict=True):
                row_index = int(context["row_index"])
                sample = context["sample"]
                trial_index = int(context["trial_index"])
                info = np.asarray(context["info"], dtype=np.uint8)
                codeword = np.asarray(context["codeword"], dtype=np.uint8)
                payload = np.asarray(context["payload"], dtype=np.uint8)
                embed_result = context["embed_result"]

                decoded_payload = np.zeros(int(active_adapter.payload_length), dtype=np.uint8)
                decoded_payload[
                    : min(decoded_payload.size, decode_result.decoded_bits.size)
                ] = decode_result.decoded_bits[: decoded_payload.size]
                decoded_codeword = decoded_payload[list(candidate.active_positions)]
                outcome = candidate.decode(decoded_codeword)
                if outcome.info_bits is None:
                    decoded_info = np.zeros(candidate.k, dtype=np.uint8)
                else:
                    decoded_info = outcome.info_bits
                identity_success = bool(outcome.status == "decoded" and np.array_equal(decoded_info, info))
                rejected = bool(outcome.status != "decoded")
                misattributed = bool(outcome.status == "decoded" and not np.array_equal(decoded_info, info))

                truth_payload_rows.append(payload)
                decoded_payload_rows.append(decoded_payload)
                truth_codeword_rows.append(codeword)
                decoded_codeword_rows.append(decoded_codeword)
                truth_info_rows.append(info)
                decoded_info_rows.append(decoded_info)
                success_rows.append(identity_success)
                reject_rows.append(rejected)
                misattr_rows.append(misattributed)
                records.append(
                    {
                        "row_index": row_index,
                        "candidate_index": candidate_index,
                        "sample_id": sample.sample_id,
                        "sample_path": str(sample.path),
                        "trial_index": trial_index,
                        "identity_bits": info.astype(np.uint8).tolist(),
                        "identity_id": bits_to_int(info),
                        "decoded_identity_id": None if rejected else bits_to_int(decoded_info),
                        "decode_status": outcome.status,
                        "worker_status": decode_result.status,
                        "embed_status": embed_result.status,
                        "detect_prob": decode_result.detect_prob,
                        "identity_success": identity_success,
                        "rejected": rejected,
                        "misattributed": misattributed,
                        "num_codeword_errors": int(np.count_nonzero(codeword != decoded_codeword)),
                        "num_corrected_errors": int(outcome.num_corrected_errors),
                    }
                )

        configured_trial_batch_size = self.config.trial_batch_size
        if configured_trial_batch_size is None:
            configured_trial_batch_size = int(getattr(active_adapter, "default_trial_batch_size", 1) or 1)
        trial_batch_size = max(1, int(configured_trial_batch_size))

        if self.config.persist_trial_artifacts:
            trial_contexts: list[dict[str, Any]] = []
            embed_jobs: list[dict[str, Any]] = []
            for prepared in prepared_trials:
                context, job = build_trial_job(prepared, persistent_trial_root)
                trial_contexts.append(context)
                embed_jobs.append(job)
                if len(embed_jobs) >= trial_batch_size:
                    run_trial_batch(trial_contexts, embed_jobs)
                    trial_contexts = []
                    embed_jobs = []
            if embed_jobs:
                run_trial_batch(trial_contexts, embed_jobs)
        else:
            temp_dirs: list[tempfile.TemporaryDirectory[str]] = []
            trial_contexts = []
            embed_jobs = []
            try:
                for prepared in prepared_trials:
                    temp_dir = tempfile.TemporaryDirectory(prefix=f"ic-trial_{int(prepared.row_index):06d}-")
                    temp_dirs.append(temp_dir)
                    context, job = build_trial_job(prepared, Path(temp_dir.name))
                    trial_contexts.append(context)
                    embed_jobs.append(job)
                    if len(embed_jobs) >= trial_batch_size:
                        run_trial_batch(trial_contexts, embed_jobs)
                        trial_contexts = []
                        embed_jobs = []
                        for directory in temp_dirs:
                            directory.cleanup()
                        temp_dirs = []
                if embed_jobs:
                    run_trial_batch(trial_contexts, embed_jobs)
            finally:
                for directory in temp_dirs:
                    directory.cleanup()

        truth_payload = _stack_or_empty(truth_payload_rows, int(active_adapter.payload_length))
        decoded_payload = _stack_or_empty(decoded_payload_rows, int(active_adapter.payload_length))
        truth_codeword = _stack_or_empty(truth_codeword_rows, candidate.n)
        decoded_codeword = _stack_or_empty(decoded_codeword_rows, candidate.n)
        truth_info = _stack_or_empty(truth_info_rows, candidate.k)
        decoded_info = _stack_or_empty(decoded_info_rows, candidate.k)
        success = np.asarray(success_rows, dtype=bool)
        rejected = np.asarray(reject_rows, dtype=bool)
        misattributed = np.asarray(misattr_rows, dtype=bool)

        np.save(candidate_dir / "truth_payload.npy", truth_payload)
        np.save(candidate_dir / "decoded_payload.npy", decoded_payload)
        np.save(candidate_dir / "truth_codeword.npy", truth_codeword)
        np.save(candidate_dir / "decoded_codeword.npy", decoded_codeword)
        np.save(candidate_dir / "truth_info.npy", truth_info)
        np.save(candidate_dir / "decoded_info.npy", decoded_info)
        np.save(candidate_dir / "identity_success.npy", success)
        write_jsonl(candidate_dir / "trial_records.jsonl", records)

        metrics = compute_candidate_metrics(
            truth_codeword=truth_codeword,
            decoded_codeword=decoded_codeword,
            truth_info=truth_info,
            decoded_info=decoded_info,
            identity_success=success,
            rejected=rejected,
            misattributed=misattributed,
            alpha=self.config.alpha,
        )
        code_aware_bit_accuracy = float(metrics.raw_bit_accuracy)
        code_aware_bit_error_rate = 1.0 - code_aware_bit_accuracy
        code_aware_l2_p_fail = binomial_failure_probability(candidate.n, candidate.t, code_aware_bit_error_rate)
        raw_bit_accuracy = (
            float(raw_baseline.get("raw_bit_accuracy", code_aware_bit_accuracy))
            if raw_baseline is not None
            else code_aware_bit_accuracy
        )
        raw_bit_error_rate = 1.0 - raw_bit_accuracy
        raw_l2_p_fail = binomial_failure_probability(candidate.n, candidate.t, raw_bit_error_rate)
        elapsed_seconds = time.perf_counter() - candidate_started
        metrics_payload = metrics.to_dict()
        metrics_payload["codeword_bit_accuracy"] = code_aware_bit_accuracy
        metrics_payload["code_aware_bit_accuracy"] = code_aware_bit_accuracy
        metrics_payload["code_aware_bit_diagnostics"] = dict(metrics_payload["bit_diagnostics"])
        metrics_payload["raw_bit_accuracy"] = raw_bit_accuracy
        payload = {
            **candidate.to_dict(),
            **metrics_payload,
            "candidate_index": candidate_index,
            "candidate_dir": str(candidate_dir),
            "raw_bit_error_rate": raw_bit_error_rate,
            "raw_l2_p_fail": raw_l2_p_fail,
            "code_aware_bit_error_rate": code_aware_bit_error_rate,
            "code_aware_l2_p_fail": code_aware_l2_p_fail,
            "l2_bit_error_rate": code_aware_bit_error_rate,
            "l2_theoretical_p_fail": code_aware_l2_p_fail,
            "elapsed_seconds": elapsed_seconds,
            "worker_index": int(worker_index),
            "worker_device": active_device,
        }
        write_json(candidate_dir / "metrics.json", payload)
        ci95 = metrics.identity_failure_ci95
        print(
            f"[candidate {candidate_index + 1}/{len(self.candidates)} result]\n"
            f"(n,k,t)=({candidate.n},{candidate.k},{candidate.t})\n"
            f"Identity Capacity={candidate.identity_count}\n"
            f"pass_alpha={metrics.pass_alpha}\n"
            f"raw_bit_acc={raw_bit_accuracy:.6f}\n"
            f"code_aware_bit_acc={code_aware_bit_accuracy:.6f}\n"
            f"raw_L2_P_fail={raw_l2_p_fail:.6f}\n"
            f"code_aware_L2_P_fail={code_aware_l2_p_fail:.6f}\n"
            f"L3_P_fail={metrics.identity_failure_rate:.6f}\n"
            f"L3_P_fail_CI95=[{float(ci95['lower']):.6f},{float(ci95['upper']):.6f}]\n"
            f"worker={worker_index} device={active_device}",
            flush=True,
        )
        print(
            f"[timing] candidate {candidate_index + 1}/{len(self.candidates)} "
            f"seconds={elapsed_seconds:.3f} worker={worker_index} device={active_device}",
            flush=True,
        )
        return payload

    def _write_report(self, summary: dict[str, Any]) -> None:
        best = summary.get("best_candidate")
        raw_baseline = summary.get("raw_baseline") or {}
        raw_l2 = raw_baseline.get("raw_l2_selection") if isinstance(raw_baseline, dict) else None
        raw_l2_monte_carlo = (
            raw_baseline.get("raw_l2_monte_carlo_selection") if isinstance(raw_baseline, dict) else None
        )
        uncoded_l2 = raw_baseline.get("uncoded_l2_selection") if isinstance(raw_baseline, dict) else None
        uncoded_l3 = raw_baseline.get("uncoded_l3_selection") if isinstance(raw_baseline, dict) else None
        samples = list(self.samples)
        dataset_names = sorted({str(sample.dataset) for sample in samples if str(sample.dataset)})
        dataset_splits = sorted({str(sample.split) for sample in samples if str(sample.split)})
        modalities = sorted({str(sample.modality) for sample in samples if str(sample.modality)})
        sample_paths = [str(sample.path) for sample in samples[:5]]
        passing = [row for row in summary.get("candidate_results", []) if bool(row.get("pass_alpha"))]
        passing.sort(key=lambda row: (int(row.get("k", 0)), -float(row.get("identity_failure_rate", 1.0))), reverse=True)
        lines = [
            f"# Identity Capacity Run: {summary['run_name']}",
            "",
            "## Experiment Setup",
            "",
            f"- Adapter: `{summary['adapter']}`",
            f"- Dataset sample count: `{len(samples)}`",
            f"- Dataset names: `{', '.join(dataset_names) if dataset_names else 'unknown'}`",
            f"- Dataset splits: `{', '.join(dataset_splits) if dataset_splits else 'unknown'}`",
            f"- Modalities: `{', '.join(modalities) if modalities else 'unknown'}`",
            f"- Trials per sample: `{self.config.trials_per_sample}`",
            f"- Total prepared trials: `{len(samples) * int(self.config.trials_per_sample)}`",
            f"- Alpha: `{summary['alpha']}`",
            f"- Message seed: `{self.config.message_seed}`",
            f"- Message mode: `{self.config.message_mode}`",
            f"- Payload filler: `{self.config.payload_filler}`",
            f"- Trial batch size: `{self.config.trial_batch_size if self.config.trial_batch_size is not None else 'adapter_default'}`",
            f"- Evaluation mode: `{summary.get('evaluation_mode', self.config.evaluation_mode)}`",
            f"- Evaluation target: `{summary.get('evaluation_target', self.config.evaluation_target)}`",
            f"- Raw baseline trial multiplier: `{summary.get('raw_baseline_trial_multiplier', self._raw_baseline_trial_multiplier())}`",
            f"- Persist trial artifacts: `{self.config.persist_trial_artifacts}`",
            f"- Worker count: `{summary.get('parallel', {}).get('worker_count', self.config.worker_count)}`",
            f"- Worker devices: `{', '.join(summary.get('parallel', {}).get('worker_devices', self.config.worker_devices))}`",
            f"- Sample path preview: `{', '.join(sample_paths) if sample_paths else 'none'}`",
            "",
            "## Search Configuration",
            "",
            f"- Candidates evaluated: `{summary['num_candidates']}`",
            f"- Max candidates: `{summary.get('max_candidates', 'all')}`",
            f"- Candidate pool size: `{summary.get('candidate_pool_size', 0)}`",
            f"- Code family: `{summary.get('code_family', self.config.code_family)}`",
            f"- Search min k: `{summary.get('search_min_k', 0)}`",
            f"- Search max k: `{summary.get('search_max_k', 0)}`",
            f"- Search start k: `{summary.get('search_start_k', 'n/a')}`",
            f"- Search strategy: `{summary.get('search_strategy', 'brute_force')}`",
            "",
            "## Channel Configuration",
            "",
            "```json",
            json.dumps(summary.get("channel", {}), indent=2, sort_keys=True),
            "```",
            "",
            "## Method Parameters",
            "",
            "```json",
            json.dumps(self.config.method_params, indent=2, sort_keys=True),
            "```",
            "",
            "## Raw Baseline",
            "",
            f"- Raw bit accuracy: `{float(raw_baseline.get('raw_bit_accuracy', 0.0)):.6f}`",
            f"- Raw bit error rate: `{float(raw_baseline.get('raw_bit_error_rate', 0.0)):.6f}`",
            f"- Raw baseline trials: `{int(raw_baseline.get('num_trials', 0))}`",
            f"- Raw baseline payload length: `{int(raw_baseline.get('payload_length', 0))}`",
            f"- Raw baseline dir: `{raw_baseline.get('baseline_dir', '')}`",
            "",
        ]
        if isinstance(raw_l2, dict):
            lines.extend(
                [
                    "### Raw L2 Selection",
                    "",
                    f"- `log2(M)`: `{raw_l2.get('k', 0)}`",
                    f"- `M`: `{raw_l2.get('identity_count', 0)}`",
                    f"- `(n, k, t)`: `({raw_l2.get('n', 0)}, {raw_l2.get('k', 0)}, {raw_l2.get('t', 0)})`",
                    f"- Raw L2 P fail: `{float(raw_l2.get('failure_probability', 1.0)):.6f}`",
                    "",
                    "```json",
                    json.dumps(raw_l2, indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            )
        if isinstance(raw_l2_monte_carlo, dict):
            lines.extend(
                [
                    "### Raw L2 Monte Carlo Selection",
                    "",
                    f"- `log2(M)`: `{raw_l2_monte_carlo.get('k', 0)}`",
                    f"- `M`: `{raw_l2_monte_carlo.get('identity_count', 0)}`",
                    f"- `(n, k, t)`: `({raw_l2_monte_carlo.get('n', 0)}, {raw_l2_monte_carlo.get('k', 0)}, {raw_l2_monte_carlo.get('t', 0)})`",
                    f"- Raw L2 Monte Carlo P fail: `{float(raw_l2_monte_carlo.get('failure_probability', 1.0)):.6f}`",
                    "",
                    "```json",
                    json.dumps(raw_l2_monte_carlo, indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            )
        if isinstance(uncoded_l2, dict):
            lines.extend(
                [
                    "### Uncoded L2 Selection",
                    "",
                    f"- `log2(M)`: `{uncoded_l2.get('k', 0)}`",
                    f"- `M`: `{uncoded_l2.get('identity_count', 0)}`",
                    f"- `(n, k, t)`: `({uncoded_l2.get('n', 0)}, {uncoded_l2.get('k', 0)}, {uncoded_l2.get('t', 0)})`",
                    f"- Uncoded L2 P fail: `{float(uncoded_l2.get('failure_probability', 1.0)):.6f}`",
                    "",
                    "```json",
                    json.dumps(uncoded_l2, indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            )
        if isinstance(uncoded_l3, dict):
            lines.extend(
                [
                    "### Uncoded L3 Selection",
                    "",
                    f"- `log2(M)`: `{uncoded_l3.get('k', 0)}`",
                    f"- `M`: `{uncoded_l3.get('identity_count', 0)}`",
                    f"- `(n, k, t)`: `({uncoded_l3.get('n', 0)}, {uncoded_l3.get('k', 0)}, {uncoded_l3.get('t', 0)})`",
                    f"- Uncoded L3 P fail: `{float(uncoded_l3.get('failure_rate', 1.0)):.6f}`",
                    "",
                    "```json",
                    json.dumps(uncoded_l3, indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            )
        if best is None:
            lines.append("No candidate satisfied the failure-rate budget.")
        else:
            lines.extend(
                [
                    "## Best Candidate",
                    "",
                    f"- Candidate index: `{best.get('candidate_index', '')}`",
                    f"- `M`: `{best['identity_count']}`",
                    f"- `log2(M)`: `{best['k']}`",
                    f"- `(n, k, t)`: `({best['n']}, {best['k']}, {best['t']})`",
                    f"- Identity failure rate: `{best['identity_failure_rate']:.6f}`",
                    f"- Raw bit accuracy: `{float(best.get('raw_bit_accuracy', 0.0)):.6f}`",
                    f"- Code-aware bit accuracy: `{float(best.get('code_aware_bit_accuracy', 0.0)):.6f}`",
                    f"- Raw L2 P fail: `{float(best.get('raw_l2_p_fail', 0.0)):.6f}`",
                    f"- Code-aware L2 P fail: `{float(best.get('code_aware_l2_p_fail', 0.0)):.6f}`",
                    f"- L3 P fail CI95: `{_format_ci95(best.get('identity_failure_ci95'))}`",
                    f"- Candidate dir: `{best.get('candidate_dir', '')}`",
                    "",
                ]
            )
        lines.extend(["## Passing Candidates", ""])
        if not passing:
            lines.append("No passing candidates.")
        else:
            best_index = None if best is None else int(best.get("candidate_index", -1))
            for row in passing:
                suffix = " (selected)" if best_index is not None and int(row.get("candidate_index", -1)) == best_index else ""
                lines.extend(_candidate_report_lines(row, suffix=suffix))
        (self.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_code_features_wide(self, summary: dict[str, Any]) -> None:
        candidates = list(summary.get("candidate_results", []))
        position_count = _wide_position_count(candidates, payload_length=int(self.adapter.payload_length))
        position_columns = [f"pos_{pos:02d}_bit_acc_mean" for pos in range(position_count)]
        fieldnames = [
            "code",
            "n",
            "k",
            "t",
            "capacity",
            "pass_alpha",
            "raw_bit_acc",
            "code_aware_bit_acc",
            "raw_l2_fail",
            "code_aware_l2_fail",
            "l3_fail",
            "l3_fail_c95",
            "active_positions",
            "active_position_count",
            *position_columns,
        ]
        rows = [_code_features_wide_row(row, position_columns) for row in candidates]
        with (self.output_dir / "code_features_wide.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _format_ci95(value: Any) -> str:
    if not isinstance(value, dict):
        return "n/a"
    return f"[{float(value.get('lower', 0.0)):.6f}, {float(value.get('upper', 1.0)):.6f}]"


def _format_csv_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.9f}"
    except (TypeError, ValueError):
        return ""


def _candidate_code(row: dict[str, Any]) -> str:
    return f"({int(row.get('n', 0))},{int(row.get('k', 0))},{int(row.get('t', 0))})"


def _wide_position_count(candidates: list[dict[str, Any]], *, payload_length: int) -> int:
    max_position = int(payload_length)
    for row in candidates:
        active_positions = row.get("active_positions", [])
        if isinstance(active_positions, list) and active_positions:
            max_position = max(max_position, max(int(pos) for pos in active_positions) + 1)
        diagnostics = row.get("code_aware_bit_diagnostics") or row.get("bit_diagnostics") or {}
        if isinstance(diagnostics, dict):
            per_bit_ber = diagnostics.get("per_bit_ber") or []
            if isinstance(per_bit_ber, list):
                max_position = max(max_position, len(per_bit_ber))
    return max(16, max_position)


def _code_features_wide_row(row: dict[str, Any], position_columns: list[str]) -> dict[str, str]:
    active_positions = [int(pos) for pos in row.get("active_positions", [])]
    diagnostics = row.get("code_aware_bit_diagnostics") or row.get("bit_diagnostics") or {}
    per_bit_ber = diagnostics.get("per_bit_ber") if isinstance(diagnostics, dict) else []
    per_bit_acc = [1.0 - float(value) for value in per_bit_ber] if isinstance(per_bit_ber, list) else []
    active_position_text = " ".join(str(pos) for pos in active_positions)
    csv_row = {
        "code": _candidate_code(row),
        "n": str(int(row.get("n", 0))),
        "k": str(int(row.get("k", 0))),
        "t": str(int(row.get("t", 0))),
        "capacity": str(int(row.get("identity_count", 2 ** int(row.get("k", 0))))),
        "pass_alpha": str(bool(row.get("pass_alpha", False))),
        "raw_bit_acc": _format_csv_float(row.get("raw_bit_accuracy")),
        "code_aware_bit_acc": _format_csv_float(row.get("code_aware_bit_accuracy", row.get("raw_bit_accuracy"))),
        "raw_l2_fail": _format_csv_float(row.get("raw_l2_p_fail")),
        "code_aware_l2_fail": _format_csv_float(row.get("code_aware_l2_p_fail", row.get("l2_theoretical_p_fail"))),
        "l3_fail": _format_csv_float(row.get("identity_failure_rate")),
        "l3_fail_c95": _format_ci95(row.get("identity_failure_ci95")),
        "active_positions": active_position_text,
        "active_position_count": str(len(active_positions)),
    }
    active_index_by_position = {position: idx for idx, position in enumerate(active_positions)}
    for column in position_columns:
        position = int(column.split("_")[1])
        active_index = active_index_by_position.get(position)
        value = per_bit_acc[active_index] if active_index is not None and active_index < len(per_bit_acc) else 0.0
        csv_row[column] = _format_csv_float(value)
    return csv_row


def _format_optional_int(value: int | None) -> str:
    return "n/a" if value is None else str(int(value))


def _stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    value = 0x811C9DC5
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return int(value)


def _candidate_report_lines(row: dict[str, Any], *, suffix: str = "") -> list[str]:
    active_positions = row.get("active_positions", [])
    mother = row.get("mother") if isinstance(row.get("mother"), dict) else {}
    return [
        f"### Candidate {int(row.get('candidate_index', -1))}{suffix}",
        "",
        f"- Pass alpha: `{bool(row.get('pass_alpha'))}`",
        f"- `M`: `{int(row.get('identity_count', 0))}`",
        f"- `log2(M)`: `{int(row.get('k', 0))}`",
        f"- `(n, k, t)`: `({int(row.get('n', 0))}, {int(row.get('k', 0))}, {int(row.get('t', 0))})`",
        f"- Redundancy: `{int(row.get('redundancy', 0))}`",
        f"- Mother BCH: `(n={mother.get('n', 'n/a')}, k={mother.get('k', 'n/a')}, t={mother.get('t', 'n/a')}, m={mother.get('m', 'n/a')})`",
        f"- Raw bit accuracy: `{float(row.get('raw_bit_accuracy', 0.0)):.6f}`",
        f"- Code-aware bit accuracy: `{float(row.get('code_aware_bit_accuracy', row.get('raw_bit_accuracy', 0.0))):.6f}`",
        f"- Raw L2 P fail: `{float(row.get('raw_l2_p_fail', 0.0)):.6f}`",
        f"- Code-aware L2 P fail: `{float(row.get('code_aware_l2_p_fail', row.get('l2_theoretical_p_fail', 0.0))):.6f}`",
        f"- L3 P fail: `{float(row.get('identity_failure_rate', 0.0)):.6f}`",
        f"- L3 P fail CI95: `{_format_ci95(row.get('identity_failure_ci95'))}`",
        f"- Reject rate: `{float(row.get('reject_rate', 0.0)):.6f}`",
        f"- Misattribution rate: `{float(row.get('misattribution_rate', 0.0)):.6f}`",
        f"- Num trials: `{int(row.get('num_trials', 0))}`",
        f"- Worker: `{row.get('worker_index', 0)}` / `{row.get('worker_device', '')}`",
        f"- Candidate dir: `{row.get('candidate_dir', '')}`",
        f"- Active positions: `{active_positions}`",
        "",
        "```json",
        json.dumps(row, indent=2, sort_keys=True),
        "```",
        "",
    ]


def _stack_or_empty(rows: list[np.ndarray], width: int) -> np.ndarray:
    if not rows:
        return np.zeros((0, int(width)), dtype=np.uint8)
    return np.stack([np.asarray(row, dtype=np.uint8).reshape(-1) for row in rows], axis=0)
