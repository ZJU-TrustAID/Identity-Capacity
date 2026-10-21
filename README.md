# IC Estimation

This toolkit estimates identity capacity through complete watermark pipelines. The core package handles candidate generation, identity coding, channel application, empirical metrics, and search. Watermark implementations run behind file-based adapters.

## Install

```bash
pip install -e .
```

After installation, the `ic-estimate` command points to this checkout. If another editable checkout was installed earlier, either reinstall from this directory or use `python -m ic_estimation.cli`.

## Local Paths

Default paths are relative to this directory:

- outputs: `runs/`
- manifests: `manifests/`
- datasets: `resources/datasets/`
- workers: `workers/`
- method source trees and checkpoints: `third_party/` and `resources/models/`

You can override paths with environment variables:

```bash
export IC_REPO_ROOT=/path/to/ic-estimation
export IC_OUTPUTS_ROOT=/path/to/ic-runs
export IC_MANIFESTS_ROOT=/path/to/ic-manifests
export IC_VCTK_ROOT=/path/to/VCTK
export IC_COCO_VAL_ROOT=/path/to/COCO/val2017
```

You can also copy `configs/paths.example.jsonc` to `configs/paths.jsonc` and edit local paths there.

For full method runs, place method source trees, checkpoints, and datasets inside this checkout unless you override them in your private local config:

```text
ic-estimation/
  third_party/
    audioseal/
    VoiceMark/
    audio-watermarking/
    HiDDeN-Pytorch/
    invisible-watermark/
    trustmark/
  resources/
    models/
      audioseal/
        generator_base.pth
        detector_base.pth
      VoiceMark/
        voicemark.pth
    datasets/
      VCTK/
      COCO/
        val2017/
```

The workers derive these paths from the IC Estimation root. For example, the AudioSeal worker adds `third_party/audioseal/src` to `sys.path` and looks for `resources/models/audioseal/generator_base.pth` and `resources/models/audioseal/detector_base.pth` by default.

Each worker conda environment must contain the normal Python dependencies for that method. The method source code must either be installed into the environment or placed under `third_party/` so the worker can add it to `sys.path`. For example, to check whether AudioSeal is installed directly in the conda environment, run:

```bash
conda run -n wm-audioseal python -c "from audioseal import AudioSeal; print(AudioSeal)"
```

If this import fails, either install the package in that environment:

```bash
conda run -n wm-audioseal python -m pip install -e third_party/audioseal
```

or keep the source tree at `third_party/audioseal/`. The worker adds `third_party/audioseal/src` to `sys.path` before importing. You can check the worker-style import with:

```bash
conda run -n wm-audioseal python -c "import sys; sys.path.insert(0, 'third_party/audioseal/src'); from audioseal import AudioSeal; print(AudioSeal)"
```

## Worker Conda Environments

Subprocess workers are launched with `conda run -n <env> ...`. The default names are set in `ic_estimation/adapters.py`.

| Method | Default env | Override variable |
| --- | --- | --- |
| AudioSeal | `wm-audioseal` | `IC_AUDIOSEAL_CONDA_ENV` |
| VoiceMark | `wm-voicemark` | `IC_VOICEMARK_CONDA_ENV` |
| Patchwork | `wm-audioseal` | `IC_PATCHWORK_CONDA_ENV` |
| FSVC | `wm-audioseal` | `IC_FSVC_CONDA_ENV` |
| HiDDeN | `wm-hidden` | `IC_HIDDEN_CONDA_ENV` |
| DWT-DCT | `wm-hidden` | `IC_DWT_DCT_CONDA_ENV` |
| SVD-Chang | `wm-hidden` | `IC_SVD_CHANG_CONDA_ENV` |
| TrustMark | `wm-trustmark` | `IC_TRUSTMARK_CONDA_ENV` |

Patchwork and FSVC fall back to `IC_AUDIOSEAL_CONDA_ENV` if their own override is unset. DWT-DCT and SVD-Chang fall back to `IC_HIDDEN_CONDA_ENV` if their own override is unset.

If your local environment uses a different name, export the override before running:

```bash
export IC_AUDIOSEAL_CONDA_ENV=wm_audioseal
ic-estimate run --config configs/audioseal_default.jsonc
```

The same keys can be edited in a private `configs/paths.jsonc`.

## Quick Smoke Example

This command runs without external watermark packages, model checkpoints, or datasets:

```bash
python -m ic_estimation.cli run --config configs/echo_smoke.jsonc
```

It uses `examples/sample.png` and the built-in `echo` adapter. `echo` is not a watermarking method. It copies the input image to the trial output, stores the generated payload bits in metadata, and returns the same bits during decode. This checks that everything works in a clean checkout without relying on any external watermark implementation. The result is written to `runs/echo-smoke/`.

## Full Method Runs

The method configs below require the corresponding worker dependencies. Remember to put upstream source trees under `third_party/`, model checkpoints under `resources/models/`, and datasets under `resources/datasets/`, or override the paths with the `IC_*` variables above before running these commands.

```bash
ic-estimate run --config configs/audioseal_default.jsonc
ic-estimate run --config configs/voicemark_default.jsonc
ic-estimate run --config configs/patchwork_default.jsonc
ic-estimate run --config configs/fsvc_default.jsonc
ic-estimate run --config configs/dwt_dct_default.jsonc
ic-estimate run --config configs/svd_chang_default.jsonc
ic-estimate run --config configs/hidden_default.jsonc
ic-estimate run --config configs/trustmark_default.jsonc
```

Worker stdout/stderr is quiet by default so dependency warnings do not obscure the IC Estimation log. Set `IC_VERBOSE_WORKERS=1` to show raw worker commands and streams during debugging. Worker failure messages still include captured stdout/stderr.

Set `code_family` in a config to choose the candidate pool:

```jsonc
"code_family": "uncoded_bch"  // No-ECC with shortened BCH
"code_family": "bch"          // shortened BCH only
"code_family": "uncoded_rm"   // No-ECC with shortened RM
"code_family": "rm"           // shortened RM only
```

## Built-In Adapters

| Method | Adapter name | Worker entry | Source |
| --- | --- | --- | --- |
| Echo Smoke | `echo` | built in | no external dependency |
| AudioSeal | `audioseal` | `workers/wm-audioseal/audioseal_worker.py` | `https://github.com/facebookresearch/audioseal` |
| VoiceMark | `voicemark` | `workers/wm-voicemark/voicemark_worker.py` | `https://huggingface.co/spaces/haiyunli/VoiceMark` |
| Patchwork | `patchwork` | `workers/wm-audio-traditional/traditional_audio_worker.py` | `https://github.com/kosta-pmf/audio-watermarking` |
| FSVC | `fsvc` | `workers/wm-audio-traditional/traditional_audio_worker.py` | `https://github.com/kosta-pmf/audio-watermarking` |
| DWT-DCT | `dwt_dct` | `workers/wm-invisible/worker.py` | `https://github.com/ShieldMnt/invisible-watermark` |
| SVD-Chang | `svd_chang` | `workers/wm-svd-traditional/svd_chang_worker.py` | Chang, Tsai, and Lin style SVD watermark reimplementation |
| HiDDeN | `hidden` | `workers/wm-hidden/hidden_worker.py` | `https://github.com/ando-khachatryan/HiDDeN` |
| TrustMark | `trustmark` | `workers/wm-trustmark/worker.py` | `https://github.com/adobe/trustmark` |

The built-ins are configured in `ic_estimation/adapters.py`.

## Standard Adapter Contract

Future watermark methods only need to implement `WatermarkAdapter` or configure `SubprocessWatermarkAdapter`.

```python
from pathlib import Path
import numpy as np

from ic_estimation.adapters import DecodeResult, EmbedResult, WatermarkAdapter


class MyWatermarkAdapter(WatermarkAdapter):
    name = "my_method"
    modality = "image"  # or "audio"
    payload_length = 64
    native_coded_output = False
    default_trial_batch_size = 1

    def embed(
        self,
        *,
        sample_path: Path,
        payload_bits: np.ndarray,
        work_dir: Path,
        method_params: dict,
    ) -> EmbedResult:
        ...

    def decode(
        self,
        *,
        sample_path: Path,
        metadata_path: Path,
        work_dir: Path,
        method_params: dict,
    ) -> DecodeResult:
        ...
```

An adapter accepts a binary payload of length `payload_length`, writes a watermarked artifact, and returns decoded bits of the same length. Optional `embed_batch` and `decode_batch` methods can be added for resident-worker execution.
