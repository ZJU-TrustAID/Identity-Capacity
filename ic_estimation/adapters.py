from __future__ import annotations

import atexit
import dataclasses
import os
import json
import shutil
import subprocess
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ic_estimation.utils import normalize_binary_vector, read_json, save_bits, write_json


@dataclass(frozen=True)
class EmbedResult:
    status: str
    watermarked_path: Path
    metadata_path: Path
    response: dict[str, Any]


@dataclass(frozen=True)
class DecodeResult:
    status: str
    decoded_bits: np.ndarray
    soft_scores: np.ndarray | None
    detect_prob: float | None
    decoded_bits_path: Path
    soft_scores_path: Path | None
    response: dict[str, Any]


class WatermarkAdapter(ABC):
    name: str
    modality: str
    payload_length: int
    native_coded_output: bool = False
    default_trial_batch_size: int = 1

    @abstractmethod
    def embed(
        self,
        *,
        sample_path: Path,
        payload_bits: np.ndarray,
        work_dir: Path,
        method_params: dict[str, Any],
    ) -> EmbedResult:
        raise NotImplementedError

    @abstractmethod
    def decode(
        self,
        *,
        sample_path: Path,
        metadata_path: Path,
        work_dir: Path,
        method_params: dict[str, Any],
    ) -> DecodeResult:
        raise NotImplementedError

    def embed_batch(
        self,
        jobs: list[dict[str, Any]],
        *,
        method_params: dict[str, Any],
    ) -> list[EmbedResult]:
        return [
            self.embed(
                sample_path=Path(job["sample_path"]),
                payload_bits=np.asarray(job["payload_bits"], dtype=np.uint8),
                work_dir=Path(job["work_dir"]),
                method_params=method_params,
            )
            for job in jobs
        ]

    def decode_batch(
        self,
        jobs: list[dict[str, Any]],
        *,
        method_params: dict[str, Any],
    ) -> list[DecodeResult]:
        return [
            self.decode(
                sample_path=Path(job["sample_path"]),
                metadata_path=Path(job["metadata_path"]),
                work_dir=Path(job["work_dir"]),
                method_params=method_params,
            )
            for job in jobs
        ]

    def clone_for_worker(self, worker_index: int) -> "WatermarkAdapter":
        return self


def _env_override(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _verbose_workers_enabled() -> bool:
    return str(os.environ.get("IC_VERBOSE_WORKERS", "")).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class EchoWatermarkAdapter(WatermarkAdapter):
    name: str = "echo"
    modality: str = "image"
    payload_length: int = 8
    native_coded_output: bool = False
    default_trial_batch_size: int = 1

    def embed(
        self,
        *,
        sample_path: Path,
        payload_bits: np.ndarray,
        work_dir: Path,
        method_params: dict[str, Any],
    ) -> EmbedResult:
        work_dir.mkdir(parents=True, exist_ok=True)
        payload = normalize_binary_vector(payload_bits, self.payload_length)
        payload_path = work_dir / "payload_bits.npy"
        save_bits(payload_path, payload)
        output_path = work_dir / "watermarked.png"
        shutil.copyfile(sample_path, output_path)
        metadata_path = work_dir / "embed_metadata.json"
        write_json(
            metadata_path,
            {
                "payload_bits_path": str(payload_path),
                "payload_length": int(self.payload_length),
            },
        )
        return EmbedResult(
            status="ok",
            watermarked_path=output_path,
            metadata_path=metadata_path,
            response={"watermarked_image_path": str(output_path), "metadata_path": str(metadata_path)},
        )

    def decode(
        self,
        *,
        sample_path: Path,
        metadata_path: Path,
        work_dir: Path,
        method_params: dict[str, Any],
    ) -> DecodeResult:
        work_dir.mkdir(parents=True, exist_ok=True)
        metadata = dict(read_json(metadata_path))
        decoded_bits = normalize_binary_vector(np.load(Path(metadata["payload_bits_path"])), self.payload_length)
        decoded_bits_path = work_dir / "decoded_bits.npy"
        soft_scores_path = work_dir / "soft_scores.npy"
        save_bits(decoded_bits_path, decoded_bits)
        np.save(soft_scores_path, decoded_bits.astype(np.float32))
        return DecodeResult(
            status="decoded",
            decoded_bits=decoded_bits,
            soft_scores=decoded_bits.astype(np.float32),
            detect_prob=1.0,
            decoded_bits_path=decoded_bits_path,
            soft_scores_path=soft_scores_path,
            response={
                "decoded_bits_path": str(decoded_bits_path),
                "soft_scores_path": str(soft_scores_path),
                "detect_prob": 1.0,
            },
        )


@dataclass
class SubprocessWatermarkAdapter(WatermarkAdapter):
    name: str
    modality: str
    payload_length: int
    worker_path: Path
    python_executable: str = "python"
    conda_env: str | None = None
    request_style: str = "json_flags"
    input_key: str = "input_image_path"
    output_key: str = "output_image_path"
    metadata_key: str = "metadata_output_path"
    decode_bits_key: str = "decoded_bits_output_path"
    soft_scores_key: str = "soft_scores_output_path"
    watermarked_response_key: str = "watermarked_image_path"
    native_coded_output: bool = False
    raw_bit_channel: bool = True
    persistent_mode: bool = False
    session_file: Path | None = None
    supports_native_batch: bool = False
    default_trial_batch_size: int = 1
    default_method_params: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        self._persistent_proc: subprocess.Popen[str] | None = None
        self._stderr_reader: threading.Thread | None = None

    def build_command(self, operation: str, request_path: Path, response_path: Path) -> list[str]:
        executable = [self.python_executable, str(self.worker_path)]
        if self.conda_env:
            executable = [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                self.conda_env,
                self.python_executable,
                str(self.worker_path),
            ]
        if self.request_style == "trustmark":
            return executable + [operation, "--request", str(request_path), "--response", str(response_path)]
        if self.request_style == "json_flags":
            return executable + [operation, "--request-json", str(request_path), "--response-json", str(response_path)]
        raise ValueError(f"Unsupported request style: {self.request_style}")

    def build_persistent_command(self) -> list[str]:
        executable = [self.python_executable, str(self.worker_path)]
        if self.conda_env:
            executable = [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                self.conda_env,
                self.python_executable,
                str(self.worker_path),
            ]
        cmd = executable + ["serve-json"]
        if self.default_method_params.get("method"):
            cmd += ["--method", str(self.default_method_params["method"])]
        if self.session_file is not None:
            cmd += ["--session-file", str(self.session_file)]
        return cmd

    def _run_worker(self, operation: str, request_path: Path, response_path: Path) -> dict[str, Any]:
        if self.persistent_mode:
            return self._run_persistent_worker(operation, request_path, response_path)
        cmd = self.build_command(operation, request_path, response_path)
        env = os.environ.copy()
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        if _verbose_workers_enabled():
            print(f"[{self.name} {operation}] command: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, cwd=str(self._worker_cwd()), env=env, text=True, capture_output=True)
        if _verbose_workers_enabled():
            _forward_worker_stream(self.name, operation, "stdout", proc.stdout)
            _forward_worker_stream(self.name, operation, "stderr", proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{self.name} worker {operation} failed with exit code {proc.returncode}\n"
                f"command: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        return dict(read_json(response_path))

    def _run_persistent_worker(self, operation: str, request_path: Path, response_path: Path) -> dict[str, Any]:
        proc = self._ensure_persistent_worker()
        request = dict(read_json(request_path))
        request["operation"] = operation
        request["response_path"] = str(response_path)
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
        response = self._read_persistent_response(proc, operation)
        if response_path.exists():
            return dict(read_json(response_path))
        return dict(response)

    def _read_persistent_response(self, proc: subprocess.Popen[str], operation: str) -> dict[str, Any]:
        assert proc.stdout is not None
        while True:
            response_line = proc.stdout.readline()
            if not response_line:
                raise RuntimeError(f"{self.name} persistent worker {operation} exited unexpectedly")
            try:
                response = json.loads(response_line)
            except json.JSONDecodeError:
                _forward_worker_stream(self.name, operation, "stdout", response_line)
                continue
            if isinstance(response, dict):
                if response.get("status") == "error":
                    raise RuntimeError(
                        f"{self.name} persistent worker {operation} returned error: "
                        f"{response.get('message', response)}"
                    )
                return response
            _forward_worker_stream(self.name, operation, "stdout", str(response))

    def _ensure_persistent_worker(self) -> subprocess.Popen[str]:
        if self._persistent_proc is not None and self._persistent_proc.poll() is None:
            return self._persistent_proc
        cmd = self.build_persistent_command()
        env = os.environ.copy()
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        if _verbose_workers_enabled():
            print(f"[{self.name} serve] command: {' '.join(cmd)}", flush=True)
        self._persistent_proc = subprocess.Popen(
            cmd,
            cwd=str(self._worker_cwd()),
            env=env,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        if self._persistent_proc.stderr is not None:
            self._stderr_reader = threading.Thread(
                target=_drain_worker_stderr,
                args=(self.name, self._persistent_proc.stderr),
                name=f"{self.name}-worker-stderr",
                daemon=True,
            )
            self._stderr_reader.start()
        atexit.register(self.close)
        return self._persistent_proc

    def _worker_cwd(self) -> Path:
        return _measurement_root_for_worker(self.worker_path)

    def close(self) -> None:
        proc = self._persistent_proc
        if proc is None:
            return
        if proc.poll() is None:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        if self._stderr_reader is not None and self._stderr_reader.is_alive():
            self._stderr_reader.join(timeout=1)
        self._stderr_reader = None
        self._persistent_proc = None

    def clone_for_worker(self, worker_index: int) -> "SubprocessWatermarkAdapter":
        clone = dataclasses.replace(self)
        clone.__post_init__()
        return clone

    def _embed_result_from_response(
        self,
        response: dict[str, Any],
        *,
        output_path: Path,
        metadata_path: Path,
    ) -> EmbedResult:
        return EmbedResult(
            status=str(response.get("status", "ok")),
            watermarked_path=Path(response.get(self.watermarked_response_key, output_path)),
            metadata_path=Path(response.get("metadata_path", metadata_path)),
            response=response,
        )

    def _decode_result_from_response(
        self,
        response: dict[str, Any],
        *,
        decoded_bits_path: Path,
        soft_scores_path: Path | None,
    ) -> DecodeResult:
        bit_path = Path(response.get("decoded_bits_path", decoded_bits_path))
        score_value = response.get("soft_scores_path", soft_scores_path)
        score_path = Path(score_value) if score_value else None
        decoded_bits = normalize_binary_vector(np.load(bit_path), self.payload_length)
        soft_scores = None
        if score_path is not None and score_path.exists():
            soft_scores = np.asarray(np.load(score_path), dtype=np.float32).reshape(-1)
        detect_prob = response.get("detect_prob")
        return DecodeResult(
            status=str(response.get("status", "decoded")),
            decoded_bits=decoded_bits,
            soft_scores=soft_scores,
            detect_prob=None if detect_prob is None else float(detect_prob),
            decoded_bits_path=bit_path,
            soft_scores_path=score_path,
            response=response,
        )

    def embed(
        self,
        *,
        sample_path: Path,
        payload_bits: np.ndarray,
        work_dir: Path,
        method_params: dict[str, Any],
    ) -> EmbedResult:
        work_dir.mkdir(parents=True, exist_ok=True)
        payload = normalize_binary_vector(payload_bits, self.payload_length)
        merged_method_params = {**dict(self.default_method_params or {}), **dict(method_params or {})}
        payload_path = work_dir / "payload_bits.npy"
        save_bits(payload_path, payload)
        output_suffix = ".wav" if self.modality == "audio" else ".png"
        output_path = work_dir / f"watermarked{output_suffix}"
        metadata_path = work_dir / "embed_metadata.json"
        request_path = work_dir / "embed_request.json"
        response_path = work_dir / "embed_response.json"
        request = {
            self.input_key: str(sample_path),
            "payload_bits_path": str(payload_path),
            self.output_key: str(output_path),
            self.metadata_key: str(metadata_path),
            "method_params": merged_method_params,
        }
        write_json(request_path, request)
        response = self._run_worker("embed", request_path, response_path)
        return self._embed_result_from_response(response, output_path=output_path, metadata_path=metadata_path)

    def embed_batch(
        self,
        jobs: list[dict[str, Any]],
        *,
        method_params: dict[str, Any],
    ) -> list[EmbedResult]:
        if not self.supports_native_batch:
            return super().embed_batch(jobs, method_params=method_params)
        if not jobs:
            return []
        merged_method_params = {**dict(self.default_method_params or {}), **dict(method_params or {})}

        request_jobs: list[dict[str, str]] = []
        result_contexts: list[dict[str, Path]] = []
        for index, job in enumerate(jobs):
            work_dir = Path(job["work_dir"])
            work_dir.mkdir(parents=True, exist_ok=True)
            payload = normalize_binary_vector(job["payload_bits"], self.payload_length)
            payload_path = work_dir / "payload_bits.npy"
            save_bits(payload_path, payload)
            output_suffix = ".wav" if self.modality == "audio" else ".png"
            output_path = work_dir / f"watermarked{output_suffix}"
            metadata_path = work_dir / "embed_metadata.json"
            request_jobs.append(
                {
                    self.input_key: str(Path(job["sample_path"])),
                    "payload_bits_path": str(payload_path),
                    self.output_key: str(output_path),
                    self.metadata_key: str(metadata_path),
                }
            )
            result_contexts.append(
                {
                    "output_path": output_path,
                    "metadata_path": metadata_path,
                    "response_path": work_dir / f"embed_response_{index:06d}.json",
                }
            )

        batch_work_dir = Path(jobs[0]["work_dir"]) / "_batch"
        batch_work_dir.mkdir(parents=True, exist_ok=True)
        request_path = batch_work_dir / "embed_batch_request.json"
        response_path = batch_work_dir / "embed_batch_response.json"
        write_json(request_path, {"jobs": request_jobs, "method_params": merged_method_params})
        response = self._run_worker("embed_batch", request_path, response_path)
        results = list(response.get("results") or [])
        if len(results) != len(result_contexts):
            raise RuntimeError(f"{self.name} embed_batch returned {len(results)} results for {len(result_contexts)} jobs")

        outputs: list[EmbedResult] = []
        for result, context in zip(results, result_contexts, strict=True):
            result_payload = dict(result)
            result_payload.setdefault("batch_status", response.get("status", "ok"))
            if "operation_counts" in response:
                result_payload["operation_counts"] = response["operation_counts"]
            write_json(context["response_path"], result_payload)
            outputs.append(
                self._embed_result_from_response(
                    result_payload,
                    output_path=context["output_path"],
                    metadata_path=context["metadata_path"],
                )
            )
        return outputs

    def decode(
        self,
        *,
        sample_path: Path,
        metadata_path: Path,
        work_dir: Path,
        method_params: dict[str, Any],
    ) -> DecodeResult:
        work_dir.mkdir(parents=True, exist_ok=True)
        merged_method_params = {**dict(self.default_method_params or {}), **dict(method_params or {})}
        decoded_bits_path = work_dir / "decoded_bits.npy"
        soft_scores_path = work_dir / "soft_scores.npy"
        request_path = work_dir / "decode_request.json"
        response_path = work_dir / "decode_response.json"
        request = {
            self.input_key: str(sample_path),
            "metadata_path": str(metadata_path),
            self.decode_bits_key: str(decoded_bits_path),
            self.soft_scores_key: str(soft_scores_path),
            "method_params": merged_method_params,
        }
        write_json(request_path, request)
        response = self._run_worker("decode", request_path, response_path)
        return self._decode_result_from_response(
            response,
            decoded_bits_path=decoded_bits_path,
            soft_scores_path=soft_scores_path,
        )

    def decode_batch(
        self,
        jobs: list[dict[str, Any]],
        *,
        method_params: dict[str, Any],
    ) -> list[DecodeResult]:
        if not self.supports_native_batch:
            return super().decode_batch(jobs, method_params=method_params)
        if not jobs:
            return []
        merged_method_params = {**dict(self.default_method_params or {}), **dict(method_params or {})}

        request_jobs: list[dict[str, str]] = []
        result_contexts: list[dict[str, Path]] = []
        for index, job in enumerate(jobs):
            work_dir = Path(job["work_dir"])
            work_dir.mkdir(parents=True, exist_ok=True)
            decoded_bits_path = work_dir / "decoded_bits.npy"
            soft_scores_path = work_dir / "soft_scores.npy"
            request_jobs.append(
                {
                    self.input_key: str(Path(job["sample_path"])),
                    "metadata_path": str(Path(job["metadata_path"])),
                    self.decode_bits_key: str(decoded_bits_path),
                    self.soft_scores_key: str(soft_scores_path),
                }
            )
            result_contexts.append(
                {
                    "decoded_bits_path": decoded_bits_path,
                    "soft_scores_path": soft_scores_path,
                    "response_path": work_dir / f"decode_response_{index:06d}.json",
                }
            )

        batch_work_dir = Path(jobs[0]["work_dir"]) / "_batch"
        batch_work_dir.mkdir(parents=True, exist_ok=True)
        request_path = batch_work_dir / "decode_batch_request.json"
        response_path = batch_work_dir / "decode_batch_response.json"
        write_json(request_path, {"jobs": request_jobs, "method_params": merged_method_params})
        response = self._run_worker("decode_batch", request_path, response_path)
        results = list(response.get("results") or [])
        if len(results) != len(result_contexts):
            raise RuntimeError(f"{self.name} decode_batch returned {len(results)} results for {len(result_contexts)} jobs")

        outputs: list[DecodeResult] = []
        for result, context in zip(results, result_contexts, strict=True):
            result_payload = dict(result)
            result_payload.setdefault("batch_status", response.get("status", "ok"))
            if "operation_counts" in response:
                result_payload["operation_counts"] = response["operation_counts"]
            write_json(context["response_path"], result_payload)
            outputs.append(
                self._decode_result_from_response(
                    result_payload,
                    decoded_bits_path=context["decoded_bits_path"],
                    soft_scores_path=context["soft_scores_path"],
                )
            )
        return outputs


def make_audioseal_adapter(repo_root: str | Path) -> SubprocessWatermarkAdapter:
    root = Path(repo_root)
    return SubprocessWatermarkAdapter(
        name="audioseal",
        modality="audio",
        payload_length=16,
        worker_path=root / "workers" / "wm-audioseal" / "audioseal_worker.py",
        conda_env=_env_override("IC_AUDIOSEAL_CONDA_ENV", "wm-audioseal"),
        input_key="input_audio_path",
        output_key="output_audio_path",
        watermarked_response_key="watermarked_audio_path",
        persistent_mode=True,
        supports_native_batch=True,
        default_trial_batch_size=4,
    )


def make_voicemark_adapter(repo_root: str | Path) -> SubprocessWatermarkAdapter:
    root = Path(repo_root)
    return SubprocessWatermarkAdapter(
        name="voicemark",
        modality="audio",
        payload_length=16,
        worker_path=root / "workers" / "wm-voicemark" / "voicemark_worker.py",
        conda_env=_env_override("IC_VOICEMARK_CONDA_ENV", "wm-voicemark"),
        input_key="input_audio_path",
        output_key="output_audio_path",
        watermarked_response_key="watermarked_audio_path",
        persistent_mode=True,
        supports_native_batch=True,
        default_trial_batch_size=4,
    )


def make_hidden_adapter(repo_root: str | Path) -> SubprocessWatermarkAdapter:
    root = Path(repo_root)
    return SubprocessWatermarkAdapter(
        name="hidden",
        modality="image",
        payload_length=30,
        worker_path=root / "workers" / "wm-hidden" / "hidden_worker.py",
        conda_env=_env_override("IC_HIDDEN_CONDA_ENV", "wm-hidden"),
        input_key="input_image_path",
        output_key="output_image_path",
        watermarked_response_key="watermarked_image_path",
        persistent_mode=True,
        supports_native_batch=True,
        default_trial_batch_size=4,
    )


def make_trustmark_adapter(repo_root: str | Path) -> SubprocessWatermarkAdapter:
    root = Path(repo_root)
    return SubprocessWatermarkAdapter(
        name="trustmark",
        modality="image",
        payload_length=100,
        worker_path=root / "workers" / "wm-trustmark" / "worker.py",
        conda_env=_env_override("IC_TRUSTMARK_CONDA_ENV", "wm-trustmark"),
        request_style="trustmark",
        input_key="input_image_path",
        output_key="output_image_path",
        metadata_key="metadata_path",
        decode_bits_key="output_bits_path",
        soft_scores_key="output_scores_path",
        watermarked_response_key="watermarked_image_path",
        native_coded_output=False,
        raw_bit_channel=True,
        persistent_mode=True,
        supports_native_batch=True,
    )


def make_dwt_dct_adapter(repo_root: str | Path) -> SubprocessWatermarkAdapter:
    root = Path(repo_root)
    return SubprocessWatermarkAdapter(
        name="dwt_dct",
        modality="image",
        payload_length=64,
        worker_path=root / "workers" / "wm-invisible" / "worker.py",
        conda_env=_env_override("IC_DWT_DCT_CONDA_ENV", _env_override("IC_HIDDEN_CONDA_ENV", "wm-hidden")),
        input_key="input_image_path",
        output_key="output_image_path",
        watermarked_response_key="watermarked_image_path",
        persistent_mode=True,
        supports_native_batch=True,
        default_trial_batch_size=4,
        default_method_params={"method": "dwtDct"},
    )


def make_svd_chang_adapter(repo_root: str | Path) -> SubprocessWatermarkAdapter:
    root = Path(repo_root)
    return SubprocessWatermarkAdapter(
        name="svd_chang",
        modality="image",
        payload_length=64,
        worker_path=root / "workers" / "wm-svd-traditional" / "svd_chang_worker.py",
        conda_env=_env_override("IC_SVD_CHANG_CONDA_ENV", _env_override("IC_HIDDEN_CONDA_ENV", "wm-hidden")),
        input_key="input_image_path",
        output_key="output_image_path",
        watermarked_response_key="watermarked_image_path",
        persistent_mode=True,
        supports_native_batch=True,
        default_trial_batch_size=4,
        default_method_params={"method": "svd_chang"},
    )


def make_patchwork_adapter(repo_root: str | Path) -> SubprocessWatermarkAdapter:
    return _make_traditional_audio_adapter(
        repo_root,
        name="patchwork",
        payload_length=40,
        default_method_params={"method": "patchwork"},
    )


def make_fsvc_adapter(repo_root: str | Path) -> SubprocessWatermarkAdapter:
    return _make_traditional_audio_adapter(
        repo_root,
        name="fsvc",
        payload_length=40,
        default_method_params={"method": "fsvc"},
    )


def _make_traditional_audio_adapter(
    repo_root: str | Path,
    *,
    name: str,
    payload_length: int,
    default_method_params: dict[str, Any],
) -> SubprocessWatermarkAdapter:
    root = Path(repo_root)
    default_conda_env = _env_override("IC_AUDIOSEAL_CONDA_ENV", "wm-audioseal")
    return SubprocessWatermarkAdapter(
        name=name,
        modality="audio",
        payload_length=int(payload_length),
        worker_path=root / "workers" / "wm-audio-traditional" / "traditional_audio_worker.py",
        conda_env=_env_override(f"IC_{name.upper()}_CONDA_ENV", default_conda_env),
        input_key="input_audio_path",
        output_key="output_audio_path",
        watermarked_response_key="watermarked_audio_path",
        persistent_mode=True,
        supports_native_batch=True,
        default_trial_batch_size=4,
        default_method_params=default_method_params,
    )


def make_echo_adapter(repo_root: str | Path) -> EchoWatermarkAdapter:
    return EchoWatermarkAdapter()


def make_builtin_adapters(repo_root: str | Path) -> dict[str, WatermarkAdapter]:
    return {
        "echo": make_echo_adapter(repo_root),
        "audioseal": make_audioseal_adapter(repo_root),
        "voicemark": make_voicemark_adapter(repo_root),
        "hidden": make_hidden_adapter(repo_root),
        "trustmark": make_trustmark_adapter(repo_root),
        "dwt_dct": make_dwt_dct_adapter(repo_root),
        "svd_chang": make_svd_chang_adapter(repo_root),
        "patchwork": make_patchwork_adapter(repo_root),
        "fsvc": make_fsvc_adapter(repo_root),
    }


def _measurement_root_for_worker(worker_path: Path) -> Path:
    resolved = worker_path.resolve()
    for parent in resolved.parents:
        if parent.name == "workers" and (parent.parent / "ic_estimation").exists():
            return parent.parent
    return resolved.parent


def _forward_worker_stream(adapter_name: str, operation: str, stream_name: str, output: str | None) -> None:
    if not output:
        return
    prefix = f"[{adapter_name} {operation} {stream_name}]"
    for line in output.rstrip("\n").splitlines():
        print(f"{prefix} {line}", flush=True)


def _drain_worker_stderr(adapter_name: str, stderr) -> None:
    for line in stderr:
        if _verbose_workers_enabled():
            _forward_worker_stream(adapter_name, "serve", "stderr", line)
