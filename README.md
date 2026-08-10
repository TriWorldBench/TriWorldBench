<h1 align="center">TriWorldBench</h1>

<h2 align="center">A Benchmark Evaluating Triple-View Embodied World Models</h2>

<a href="docs/assets/wechat_group.jpg" target="_blank">
  <img src="https://cdn.simpleicons.org/wechat/07C160" width="20" style="vertical-align: middle; margin-right: 5px;"> 
  <b>WeChat Group</b>
</a>

## News

- 2026-07-28: We released the TriWorldBench code and valid set with 100 episodes and testset with 500 episodes.
- 2026-07-28: We launched the TriWorldBench Challenge.


## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Environment Setup](#environment-setup)
  - [Dataset Preparation](#dataset-preparation)
    - [Quick Test Dataset](#quick-test-dataset)
    - [Prepare Your Own Test Dataset](#prepare-your-own-test-dataset)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [Leaderboard & Submission](#leaderboard--submission)
- [Acknowledgement](#acknowledgement)

---

## Overview

TriWorldBench is a unified benchmark designed to systematically evaluate embodied world models from three synchronized robot views: head, left wrist, and right wrist. It reports 19 evaluation signals across six dimensions: tri-view consistency, task alignment, physical and 3D coherence, motion quality, temporal consistency, and visual quality. Tri-view consistency is the central dimension. It combines reference-anchored signals, which compare every generated camera with its corresponding view from a consistent ground-truth triplet, with direct semantic judgments of head-wrist and joint tri-view agreement. The remaining dimensions test whether the prediction follows the task, preserves plausible interaction and geometry, exhibits the expected motion in each camera, remains stable over time, and produces usable images. Trajectory-derived STATE annotations connect these evaluations by identifying the action phase, active arm, and expected moving views. TriWorldBench retains per-view, head-wrist pair, and joint tri-view results, additionally reports visual-quality scores grounded by task and consistency performance, and names its final overall aggregate TWB-Score. Together, these dimensions track progress toward embodied world models that generate one consistent robot world across multiple cameras.

![TriWorldBench evaluation overview](docs/assets/overview.png)

---

## Getting Started

### Environment Setup

**a. Create the conda virtual environment**

```bash
conda create -n triworldbench python=3.10.0 -y
conda activate triworldbench

cd Triworld
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install --no-deps -r requirements_no_deps.txt
```

System dependencies: `ffmpeg`, `ffprobe`, CUDA-capable GPU(s).

**b. Download model weights**

Follow [DOWNLOAD_LINKS.md](./DOWNLOAD_LINKS.md) and place all checkpoints under:

```text
./weights/
```

Set `weights_root` in `config/config.yaml` to this directory.

`TORCH_HOME` is set automatically to `{weights_root}/torch_home` during evaluation (ResNet34 for photometric_smoothness).

---



### Dataset Preparation



#### Quick Test Dataset

We provide a 100-episode validation set derived from RoboTwin2.0, the corresponding `gt_dataset`, and processed `STATE` annotations. Users only need to prepare inference results on the validation set, then update a small amount of configuration before running evaluation.

The validation dataset is released at [TriWorldBench/Dataset](https://huggingface.co/datasets/TriWorldBench/Dataset). Download `val_dataset` from this Hugging Face dataset page.

After extraction, the validation bundle contains:

```text
val_dataset/
├── test_dataset/
│   ├── data/
│   ├── first_frame/
│   └── instructions/
├── STATE/
│   └── episode*.json
└── gt_dataset/
    └── episode*/
```

`gt_dataset` and `STATE` already use the format required by this evaluation project. The matching VQA question annotations for this validation set are included in this repository under `metrics/VQA/qa_val/`; they are used by the VQA metric to score task and view-specific question answering. Users only need to run inference on the inputs from `test_dataset`, then point the evaluation config to the resulting outputs for a quick validation run. See the Inference and Evaluation sections for how to use these paths.

#### Prepare Your Own Test Dataset

If you want to build your own test dataset, you can generate or download RoboTwin2.0 data from [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0). See [test_data/data_readme.md](test_data/data_readme.md) for how to prepare `gt_dataset`, `STATE`, and matching VQA QA files for evaluation.

---



### Inference

**a. Run inference and place generated videos**

Use the inputs from the quick `val_dataset/test_dataset` bundle, or from your own prepared test dataset, to run your model inference. Place per-method source data under `input/<your_method>/`:

```text
input/<your_method>/
└── episodeN/
    ├── head.mp4
    ├── left.mp4
    └── right.mp4
```

Example: `input/<your_method>/episode1/{head,left,right}.mp4`

**b. Preprocess inference outputs**

Convert raw MP4 inputs into GT-aligned frame trees.

Before preprocessing, place the corresponding `gt_dataset` under the default `test_data/gt_dataset`, or set `gt_dataset` in `config/config.yaml` to your custom GT path.

```bash
cd Triworld
./run_preprocess.sh <your_method>
# optional: --limit 10 --workers 8 --overwrite --dry-run
```


| Step            | Behavior                                                            |
| --------------- | ------------------------------------------------------------------- |
| Frame alignment | Uniform resampling to match GT frame count (`align_mode: match_gt`) |
| Resolution      | Resize to 320×240 (configurable in `config/config.yaml`)            |
| Scope           | Only episodes present under `input/<your_method>/`                  |


**Output location and naming:**

```text
generate/<your_method>_<input_ts>_<run_ts>/
```

- `input_ts` — batch id for the same inference input (reuse with `--input-ts YYYYMMDD_HHMMSS`)
- `run_ts` — timestamp of this preprocess run

A pointer file is written:

```text
generate/<your_method>_<input_ts>_latest.path   # text file containing latest run dir name
```

Set `eval_input` in config to this directory (or the `_latest` alias).

---



### Evaluation



#### a. Config

Edit `config/config.yaml`:

```yaml
eval_input: generate/<your_method>_latest  # preprocessed data
metrics: all                                       # or [0, 3, 12]
output_dir: output
parallel_per_gpu: 4
metric_parallel_per_gpu:
  background_consistency: 1
  semantic_alignment: 1
  subject_consistency: 2
  trajectory_accuracy: 4
gpu_list: "0,1,2,3,4,5,6,7"
python: python
weights_root: weights
gt_dataset: test_data/gt_dataset
STATE: test_data/STATE
hyperparameters: config/hyperparameters.yaml
vlm_reference_db: test_data/vlm_data_base
vlm_max_new_tokens: 512
episode_weights: false    # false=equal mean; true=frame-bucket weights
resume: true               # skip completed episodes / finished metrics
```

`python: python` assumes that commands are launched after `conda activate triworldbench`; you can also set it to the absolute Python path of your TriWorldBench environment.

**Configurable fields:**


| Key                       | Description                                                             |
| ------------------------- | ----------------------------------------------------------------------- |
| `eval_input`              | Preprocessed `generate/` directory; supports `_latest` pointer aliases   |
| `metrics`                 | Metric index list or `all` for the 19 metrics listed below              |
| `output_dir`              | Root directory for evaluation run outputs                               |
| `parallel_per_gpu`        | Default max concurrent episode workers per GPU                          |
| `metric_parallel_per_gpu` | Optional per-metric worker caps, never above `parallel_per_gpu`          |
| `gpu_list`                | Comma-separated visible GPU device IDs                                  |
| `python`                  | Python interpreter used to run metric workers                           |
| `weights_root`            | Root directory for model checkpoints                                    |
| `gt_dataset` / `STATE`    | GT frame root and flat per-episode STATE JSON root                      |
| `hyperparameters`         | Path to metric aggregation and penalty hyperparameters                  |
| `vlm_reference_db`        | Few-shot reference DB for `vlm_consistency01`                           |
| `vlm_max_new_tokens`      | Max new tokens for VLM consistency generation                           |
| `episode_weights`         | Summary aggregation mode: equal mean or frame-bucket weighted mean      |
| `resume`                  | Reuse the latest compatible run and skip completed metric score files   |


**Metric index reference:**


| Index | Metric                                                  |
| ----- | ------------------------------------------------------- |
| 0     | Instruction_Following                                   |
| 1     | Interaction_Quality                                     |
| 2     | Perspectivity                                           |
| 3     | background_consistency                                  |
| 4     | dynamic_state_alignment                                 |
| 5     | flow_score                                              |
| 6     | jepa_similarity                                         |
| 7     | photometric_smoothness                                  |
| 8     | psnr                                                    |
| 9     | semantic_alignment                                      |
| 10    | ssim                                                    |
| 11    | subject_consistency                                     |
| 12    | trajectory_accuracy                                     |
| 13–15 | vlm_consistency01/02/03                                  |
| 16    | VQA                                                     |
| 17    | aesthetic_quality (penalized score in CSV; raw in log)  |
| 18    | image_quality (penalized score in CSV; raw in log)      |




#### b. Run

```bash
conda activate triworldbench
cd Triworld
./run_eval.sh
# dry run: ./run_eval.sh --dry-run
# custom config: CONFIG=config/config.yaml ./run_eval.sh
```

Final CSV results are written under:

```text
output/<your_method>_<input_ts>_<run_ts>/
└── final_results/
    ├── episode_metrics.csv
    └── summary_metrics.csv
```

`episode_metrics.csv` reports each metric score per episode. `summary_metrics.csv` reports aggregate scores for each metric across the evaluated episodes.

**Reference resource usage for val100:**

On our 100-episode validation run with NVIDIA A100 40GB x 8 GPUs, the full evaluation finished within 2 hours. Peak GPU memory was about 160 GB in total, with the highest single-GPU peak around 23 GB. Peak system memory was about 170 GB. GPU utilization may reach 100% during model-heavy metrics, so adjust `gpu_list`, `parallel_per_gpu`, and `metric_parallel_per_gpu` if running on fewer or smaller GPUs.

---



## Leaderboard & Submission

Submit your results on the [TriWorldBench leaderboard](https://triworldbench-triworldbench-space.hf.space/#top).  
Welcome to join our WeChat Group for further discussion and communication.
![TriWorldBench WeChat Group](docs/assets/wechat_group.jpg)

---



## Acknowledgement

We acknowledge [RoboTwin 2.0](https://robotwin-platform.github.io/) for providing the dataset and simulation platform support that enables embodied task evaluation.

For video quality evaluation, TriWorldBench also benefits from the open-source evaluation community, including [WorldArena](https://github.com/tsinghua-fib-lab/WorldArena), [VBench](https://github.com/Vchitect/VBench), [EWMBench](https://github.com/AgibotTech/EWMBench), [WorldScore](https://github.com/haoyi-duan/WorldScore), [EvalCrafter](https://github.com/evalcrafter/EvalCrafter), and [JEDI](https://github.com/oooolga/JEDi).

---
