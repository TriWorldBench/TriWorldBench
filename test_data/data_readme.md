# Custom Test Dataset Preparation

This note is for users who want to build their own test dataset instead of using the released quick `val_dataset`. Starting from RoboTwin2.0 task-level data, the two commands below generate the two core folders required by TriWorldBench:

- `gt_dataset/`
- `STATE/`

If you also run the VQA metric, prepare matching QA files in the same episode order. The released quick validation set already uses the bundled `metrics/VQA/qa_val/` files as its VQA annotations.

The two scripts share the same flattening rule, so `gt_dataset/episodeN` and `STATE/episodeN.json` always refer to the same RoboTwin2.0 source episode.

---

## 1. Prepare RoboTwin2.0 Data

You can generate your own episodes with [RoboTwin 2.0](https://robotwin-platform.github.io/), or directly download RoboTwin2.0 data from [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0).

The expected RoboTwin2.0 source layout is:

```text
data/
└── <task_name>/
    └── <config_name>/
        ├── data/
        ├── _traj_data/
        ├── instructions/
        └── video/
```

The scripts sort task directories by name and source episodes by numeric episode id, then assign flat ids from `episode1`. The same mapping is reused for both `gt_dataset` and `STATE`.

---

## 2. Step 1: Generate `gt_dataset` and Mapping

Set the source and output roots:

```bash
cd Triworld
conda activate triworldbench

PY=python
ROBOTWIN_ROOT=/path/to/data
CONFIG_NAME=<config_name>
OUT_ROOT=/path/to/output_bundle
```

Generate flat `gt_dataset`:

```bash
$PY scripts/prepare_gt_dataset_from_robotwin.py \
  --source-root "$ROBOTWIN_ROOT" \
  --config-name "$CONFIG_NAME" \
  --target-root "$OUT_ROOT/gt_dataset" \
  --mapping-output "$OUT_ROOT/robotwin_episode_mapping.json" \
  --overwrite
```

The script reads images from `<task_name>/<config_name>/data/episode*.hdf5`, extracts head/left/right JPG frames, and reads the first available instruction from `<task_name>/<config_name>/instructions/episode*.json`.

Outputs:

```text
<output_bundle>/
├── gt_dataset/
│   └── episodeN/
│       ├── episodeN.json
│       ├── head/frames/frame_*.jpg
│       ├── left/frames/frame_*.jpg
│       └── right/frames/frame_*.jpg
└── robotwin_episode_mapping.json
```

`robotwin_episode_mapping.json` records each flat episode id, source task, source episode, source HDF5 path, and instruction. Use the same file to generate STATE.

---

## 3. Step 2: Generate `STATE`

Run:

```bash
$PY scripts/generate_state.py \
  --mapping-json "$OUT_ROOT/robotwin_episode_mapping.json" \
  --gt-root "$OUT_ROOT/gt_dataset" \
  --output-root "$OUT_ROOT/STATE" \
  --overwrite
```

`generate_state.py` uses `robotwin_episode_mapping.json` to find the original HDF5 files and keep the same `episodeN` order as `gt_dataset`. The generated output is flat:

```text
<output_bundle>/STATE/
├── episode1.json
├── episode2.json
├── ...
└── manifest.json
```

Do not add task-level subdirectories under `STATE`; evaluation expects `STATE/episodeN.json`.

The final custom test bundle should be:

```text
<output_bundle>/
├── gt_dataset/
├── STATE/
└── robotwin_episode_mapping.json
```

---

## 4. Step 3: Prepare VQA QA Files

The VQA metric reads per-episode question files from `metrics/VQA/qa_val/` in the default evaluation pipeline. For the released quick `val_dataset`, this folder is already included in the code repository and matches `episode1` through `episode100`.

For a custom test dataset, prepare the same layout and replace `metrics/VQA/qa_val/` with QA files aligned to your flattened episode ids:

```text
metrics/VQA/qa_val/
├── episode1/
│   └── qa.json
├── episode2/
│   └── qa.json
└── ...
```

Each `qa.json` should follow the bundled examples, with a top-level `episode` field and a `Q&A` array containing `questions`, `selections`, and `answers`. Window-based questions should also include `question_type: "window"`, `frame_window`, and `gt_episode_length` so the evaluator can align the question to the episode frames.

Make sure `metrics/VQA/qa_val/episodeN/qa.json`, `gt_dataset/episodeN`, `STATE/episodeN.json`, and the inference output for `episodeN` all refer to the same source episode.

---

## 5. Continue with Inference and Evaluation

After `gt_dataset`, `STATE`, and the required VQA QA files are ready, return to the root [README.md](../README.md):

- run inference and place generated videos under `input/<your_method>/`
- run preprocessing with `./run_preprocess.sh <your_method>`
- set `gt_dataset`, `STATE`, and `eval_input` in `config/config.yaml`
- run `./run_eval.sh`

If you want to use the default config values from the root README, set `OUT_ROOT` to `test_data` so the generated folders are `test_data/gt_dataset` and `test_data/STATE`. Otherwise, point `gt_dataset` and `STATE` in `config/config.yaml` to your custom output bundle before preprocessing and evaluation.
