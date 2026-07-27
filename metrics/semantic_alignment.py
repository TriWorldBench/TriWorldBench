"""Semantic alignment: Qwen-VL captioning + CLIP text similarity."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import json_repair
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizerFast

from metrics._common.episode_sort import episode_sort_key, natural_key
from metrics._common.utils import load_dimension_info, save_json

device = "cuda" if torch.cuda.is_available() else "cpu"

results_list: dict[str, Any] = {}


def _inference(model, processor, video_path, prompt, max_new_tokens=2048, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28):
    from submodel.qwen_vl_utils import process_vision_info

    messages = [
        {"role": "system", "content": "You are a helpful assistant in analyzing videos."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"video": video_path, "total_pixels": total_pixels, "min_pixels": min_pixels},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs["fps"]
    if isinstance(fps_inputs, (list, tuple)):
        fps_inputs = fps_inputs[0] if len(fps_inputs) > 0 else None

    try:
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=fps_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        output_text = "Error: " + str(e)

    return output_text[0]


def _prepare_prompt(video_path):
    return """
   Analyze the video captured by the overhead camera mounted on a robotic and perform the following tasks:
   
   NOTED: There should be **no human hands** in the video. These are often wrongly generated from the robot's gripper. If human hands appear, note them as **abnormal** and a **violation of logical constraints**.

   1. Describe the video in general:
      - Provide a brief description on what tasks the robotic is performing. When something anomaly happens, pay special attention to it.

   2. Describe the events:
      - Provide a detailed, step-by-step description of all actions in the video, focusing on their sequence.

   3. Identify key events with logical constraints:
      - Extract critical actions subject to logical rules, such as:
      - Prerequisites (e.g., opening a door before accessing items).
      - Avoiding physical violations (e.g., objects passing through barriers).
      - Maintaining logical task order.
      - Identify any violations of these constraints.
      
   Provide the result in json format with each key representing a task.

   {
   "General": Brief description on tasks the robotic is performing.
   "Events": Chronological list of all actions.
   "Logical_Constraints": Key actions, their constraints, and whether they are satisfied.
   "Overall_Constraints": True if the logical constraints are satisfied, false otherwise.
   }

   """


VIEW_ORDER = ("head", "left", "right")


def resolve_semantic_root(path: Path) -> tuple[Path, str]:
    """Return episode root and view name for head-only semantic datasets."""
    root = path.resolve()
    if root.name in VIEW_ORDER:
        return root, root.name
    for view in VIEW_ORDER:
        candidate = root / view
        if candidate.is_dir() or candidate.is_symlink():
            return candidate.resolve(), view
    return root, "head"


def load_qwen_captioner(model_path: str):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def build_gen_caption_key(view: str, episode_id: str, gid: str) -> str:
    return f"generated_dataset_{view}_{episode_id}_{gid}"


def build_gt_caption_key(view: str, episode_id: str) -> str:
    return f"gt_dataset_{view}_{episode_id}"


def _caption_one(model, processor, video_path: str) -> Any:
    prompt = _prepare_prompt(video_path)
    try:
        return json_repair.loads(_inference(model, processor, video_path, prompt))
    except Exception as exc:
        print(f"Error: {exc}")
        return f"Error: {exc}"


def caption_video_entries(model, processor, entries: list[dict[str, str]]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for entry in entries:
        key = str(entry["key"])
        updates[key] = _caption_one(model, processor, str(entry["video_path"]))
    return updates


def list_gen_episode_entries(gen_root: Path, episode_id: str, view: str | None = None) -> list[dict[str, str]]:
    resolved_root, resolved_view = resolve_semantic_root(gen_root)
    view = view or resolved_view
    episode_path = resolved_root / episode_id
    if not episode_path.is_dir():
        return []
    entries: list[dict[str, str]] = []
    for gid_dir in sorted(episode_path.iterdir(), key=lambda p: natural_key(p.name)):
        if not gid_dir.is_dir():
            continue
        video_path = gid_dir / "video"
        if not video_path.is_dir():
            continue
        entries.append({
            "key": build_gen_caption_key(view, episode_id, gid_dir.name),
            "video_path": str(video_path),
        })
    return entries


def list_gt_caption_entries(gt_root: Path, view: str | None = None) -> list[dict[str, str]]:
    resolved_root, resolved_view = resolve_semantic_root(gt_root)
    view = view or resolved_view
    entries: list[dict[str, str]] = []
    for episode_dir in sorted(resolved_root.iterdir(), key=lambda p: episode_sort_key(p.name)):
        if not episode_dir.is_dir():
            continue
        video_path = episode_dir / "video"
        if not video_path.is_dir():
            continue
        entries.append({
            "key": build_gt_caption_key(view, episode_dir.name),
            "video_path": str(video_path),
        })
    return entries


def ensure_gt_captions(
    model_path: str,
    gt_root: Path,
    save_path: Path,
    *,
    view: str = "head",
    resume: bool = True,
) -> Path:
    gt_json = save_path / "gt_caption_responses.json"
    if gt_json.is_file() and resume:
        print(f"[skip] GT captions exist: {gt_json}")
        return gt_json

    entries = list_gt_caption_entries(gt_root, view)
    if not entries:
        raise RuntimeError(f"no GT caption entries under {gt_root}")

    model, processor = load_qwen_captioner(model_path)
    payload = caption_video_entries(model, processor, entries)
    save_path.mkdir(parents=True, exist_ok=True)
    gt_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] GT captions -> {gt_json} ({len(payload)} entries)")
    return gt_json


def load_merged_caption_shards(shard_dir: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if not shard_dir.is_dir():
        return merged
    for shard_file in sorted(shard_dir.glob("gpu*.json")):
        payload = json.loads(shard_file.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            merged.update(payload)
    return merged


def pending_caption_entries(
    entries: list[dict[str, str]],
    shard_dir: Path,
) -> list[dict[str, str]]:
    merged = load_merged_caption_shards(shard_dir)
    return [entry for entry in entries if str(entry["key"]) not in merged]


def merge_caption_shards(shard_dir: Path, output_path: Path) -> dict[str, Any]:
    merged = load_merged_caption_shards(shard_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


def parse_generated_id(gen_id: str) -> tuple[str, str, str]:
    rest = gen_id.split("generated_dataset_")[-1]
    try:
        task_name, episode_id, gid = rest.rsplit("_", 2)
    except ValueError:
        parts = rest.split("_")
        if len(parts) >= 3:
            task_name = "_".join(parts[:-2])
            episode_id = parts[-2]
            gid = parts[-1]
        elif len(parts) == 2:
            task_name = parts[0]
            episode_id = parts[1]
            gid = ""
        else:
            task_name = rest
            episode_id = ""
            gid = ""
    return task_name, episode_id, gid


def resolve_gt_caption_key(gen_id: str, gt_idxs: list[str]) -> str:
    task_name, episode_id, _gid = parse_generated_id(gen_id)
    candidate_with_episode = f"gt_dataset_{task_name}_{episode_id}" if episode_id else None
    candidate_no_episode = f"gt_dataset_{task_name}"

    if candidate_with_episode and candidate_with_episode in gt_idxs:
        return candidate_with_episode
    if candidate_no_episode in gt_idxs:
        return candidate_no_episode

    sample = gt_idxs[:50]
    raise ValueError(
        f"GT key not found for generated id '{gen_id}'.\n"
        f"Tried: {candidate_with_episode}, {candidate_no_episode}.\n"
        f"Available GT keys (first 50): {sample}"
    )


def build_clip_score_jobs(
    gen_json_path: str | Path,
    gt_json_path: str | Path,
    *,
    key: str = "General",
) -> list[dict[str, str]]:
    gen_idxs, gen_strings = get_strings(str(gen_json_path), key)
    gt_idxs, gt_strings = get_strings(str(gt_json_path), key)
    jobs: list[dict[str, str]] = []
    for gen_id, gen_string in zip(gen_idxs, gen_strings):
        gt_key = resolve_gt_caption_key(gen_id, gt_idxs)
        gt_string = gt_strings[gt_idxs.index(gt_key)]
        jobs.append({
            "gen_id": gen_id,
            "gen_text": gen_string,
            "gt_text": gt_string,
        })
    return jobs


def clip_scores_to_results_list(scores_by_gen_id: dict[str, float]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for gen_id, score in scores_by_gen_id.items():
        task_name, episode_id, gid = parse_generated_id(gen_id)
        results.setdefault(task_name, {}).setdefault(episode_id, {}).setdefault(gid, {})[
            "CLIPScore"
        ] = float(np.around(score, decimals=6))
    return results


def _caption_videos(model_name, model_path, video_folder_root, save_path, **kwargs):
    model, processor = load_qwen_captioner(model_path)

    os.makedirs(save_path, exist_ok=True)
    json_file_name = os.path.join(save_path, f"{model_name}_caption_responses.json")
    if os.path.exists(json_file_name):
        return

    all_mp4_files = load_dimension_info(video_folder_root, dimension="semantic_alignment")
    all_responses = {}
    for mp4_file in tqdm(all_mp4_files):
        response = _caption_one(model, processor, mp4_file)
        parts = mp4_file.split("/")
        model_dataset = f"{model_name}_dataset"
        try:
            start_index = parts.index(model_dataset) + 1
            video_index = parts.index("video")
            selected_parts = parts[start_index:video_index]
            mp4_file_name = "_".join([model_dataset] + selected_parts)
        except ValueError:
            mp4_file_name = "error_in_filename_construction"
        all_responses[mp4_file_name] = response

    with open(json_file_name, "w") as f:
        json.dump(all_responses, f, indent=4)


def _build_full_info_json(output_path: Path, data_base: str, data_name: str, dimension_list: list[str]) -> str:
    cur_full_info_list = []
    for view_id in sorted(os.listdir(data_base)):
        view_path = os.path.join(data_base, view_id)
        for episode_id in sorted(os.listdir(view_path), key=episode_sort_key):
            if episode_id.endswith((".png", ".json")):
                continue
            episode_path = os.path.join(view_path, episode_id)
            for gid in sorted(os.listdir(episode_path), key=natural_key):
                video_path = os.path.join(episode_path, gid, "video")
                cur_full_info_list.append({"dimension": dimension_list, "video_list": [video_path]})
    cur_full_info_path = output_path / f"{data_name}_full_info.json"
    save_json(cur_full_info_list, str(cur_full_info_path))
    return str(cur_full_info_path)


def _build_full_gt_info_json(output_path: Path, gt_path: str, data_name: str) -> str:
    cur_full_info_list = []
    for view_id in sorted(os.listdir(gt_path)):
        view_path = os.path.join(gt_path, view_id)
        for episode_id in sorted(os.listdir(view_path), key=episode_sort_key):
            if episode_id.endswith((".png", ".json")):
                continue
            video_path = os.path.join(view_path, episode_id, "video")
            cur_full_info_list.append({"video_list": [video_path]})
    cur_full_info_path = output_path / f"{data_name}_full_info.json"
    save_json(cur_full_info_list, str(cur_full_info_path))
    return str(cur_full_info_path)


def load_clip_text_encoder(clip_model_path: str):
    """Load CLIP text encoder + tokenizer once per process."""
    model = CLIPModel.from_pretrained(clip_model_path, torch_dtype=torch.float16 if device == "cuda" else None)
    model.to(device)
    model.eval()
    tokenizer = CLIPTokenizerFast.from_pretrained(clip_model_path)
    return model, tokenizer


def run_clip_text_similarity(model, tokenizer, gen_text: str, gt_text: str) -> float:
    """Compute cosine similarity between two texts using CLIP text encoder."""
    with torch.no_grad():
        inputs_gen = tokenizer(gen_text, return_tensors="pt", truncation=True, max_length=77).to(device)
        inputs_gt = tokenizer(gt_text, return_tensors="pt", truncation=True, max_length=77).to(device)
        gen_out = model.get_text_features(**inputs_gen)
        gt_out = model.get_text_features(**inputs_gt)

        def extract_feat(out):
            if torch.is_tensor(out):
                return out
            if hasattr(out, "text_embeds") and out.text_embeds is not None:
                return out.text_embeds
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                return out.pooler_output
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                return out.last_hidden_state[:, 0]
            raise ValueError("Cannot extract text features from CLIP output")

        gen_feat = extract_feat(gen_out)
        gt_feat = extract_feat(gt_out)
        gen_feat = gen_feat / gen_feat.norm(dim=-1, keepdim=True)
        gt_feat = gt_feat / gt_feat.norm(dim=-1, keepdim=True)
        score = (gen_feat * gt_feat).sum(dim=-1).item()
    return float(score)


def get_strings(json_path, key):
    with open(json_path, "r") as f:
        data = json.load(f)
    all_idxs = []
    all_strings = []
    for idx, value in data.items():
        all_idxs.append(idx)
        if key == "General":
            if isinstance(value, dict) and "General" in value:
                all_strings.append(value["General"])
            else:
                all_strings.append(str(value))
        elif key == "Events":
            event_strings = ""
            current_events = value["Events"]
            for event in current_events:
                if len(event) < 2:
                    break
                event_strings += event + " "
            all_strings.append(event_strings)
        else:
            raise ValueError("Invalid key")
    return all_idxs, all_strings


def compute_metric_scores(semantics_model, metric_type, list_gen_strings, list_gt_strings):
    if metric_type != "CLIPScore":
        raise ValueError("Invalid metric type")

    clip_model, clip_tokenizer = load_clip_text_encoder(semantics_model)
    scores = [
        run_clip_text_similarity(clip_model, clip_tokenizer, gen_strings, gt_strings)
        for gen_strings, gt_strings in tqdm(
            list(zip(list_gen_strings, list_gt_strings)), desc="Compute metric scores"
        )
    ]
    return scores


def evaluate_run(semantics_model, eval_config, dt_json, gt_json):
    metric_type = eval_config["metric_type"]
    key = eval_config["key"]

    gen_idxs, gen_strings = get_strings(dt_json, key)
    gt_idxs, gt_strings = get_strings(gt_json, key)
    expanded_gt_strings = []

    for gen_id, gen_string in zip(gen_idxs, gen_strings):
        gt_idx = resolve_gt_caption_key(gen_id, gt_idxs)
        gt_string = gt_strings[gt_idxs.index(gt_idx)]
        expanded_gt_strings.append(gt_string)

    gt_strings = expanded_gt_strings
    assert len(gen_strings) == len(gt_strings), "Number of generated and ground-truth strings do not match"
    print(f"the number of generated strings: {len(gen_strings)}")

    scores = compute_metric_scores(semantics_model, metric_type, gen_strings, gt_strings)
    scores_list = [np.around(score, decimals=6).tolist() for score in scores]

    for idx, gen_idx in enumerate(gen_idxs):
        task_name, episode_id, gid = parse_generated_id(gen_idx)
        task_id = task_name
        if task_id not in results_list:
            results_list[task_id] = {}
        if episode_id not in results_list[task_id]:
            results_list[task_id][episode_id] = {}
        if gid not in results_list[task_id][episode_id]:
            results_list[task_id][episode_id][gid] = {}

        results_list[task_id][episode_id][gid][metric_type] = scores_list[idx]

    return scores


def evaluate_runs_single_config(eval_config, json_path, gt_json, semantics_model):
    assert gt_json is not None, "Ground-truth JSON not found"
    scores_list = {}
    if json_path != gt_json:
        dt_dataset_name = os.path.basename(json_path).split("_dataset")[0]
        scores = evaluate_run(semantics_model, eval_config, json_path, gt_json)
        avg_score = np.mean(scores)
        print(f"{dt_dataset_name}: {avg_score:.6f}")
        scores_list[dt_dataset_name] = float(avg_score)
    return scores_list


def evaluate_runs_configs(eval_configs, json_path, gt_path, semantics_model):
    eval_results = {}
    for eval_config in tqdm(eval_configs, desc="Eval different configs"):
        exp_name = eval_config["metric_type"]
        print(f"Evaluating {exp_name}...")
        scores_list = evaluate_runs_single_config(eval_config, json_path, gt_path, semantics_model)
        print(f"Scores: {scores_list}")
        eval_results[exp_name] = scores_list
    return eval_results


def score_from_caption_json(json_path: str, gt_path: str, semantics_model: str):
    global results_list
    results_list = {}
    eval_configs = [{"metric_type": "CLIPScore", "key": "General"}]
    evaluate_runs_configs(eval_configs, json_path, gt_path, semantics_model)
    return results_list


def _score_from_caption_json(json_path: str, gt_path: str, semantics_model: str):
    return score_from_caption_json(json_path, gt_path, semantics_model)


def compute_semantic_alignment(
    data_name: str,
    data_base: str,
    gt_path: str,
    output_path: Path | str,
    submodules_list: dict[str, str],
    **kwargs,
):
    """Run caption generation (Qwen-VL) then CLIP text similarity scoring."""
    save_path = Path(output_path)
    save_path.mkdir(parents=True, exist_ok=True)

    caption_model = submodules_list["caption_model"]
    semantics_model = submodules_list["clip_model"]

    cur_full_info_path = _build_full_info_json(save_path, data_base, data_name, ["semantic_alignment"])
    _caption_videos(
        model_name=data_name,
        model_path=caption_model,
        video_folder_root=cur_full_info_path,
        save_path=str(save_path),
        **kwargs,
    )

    caption_json = save_path / f"{data_name}_caption_responses.json"
    gt_caption_json = save_path / "gt_caption_responses.json"
    if not gt_caption_json.is_file():
        gt_full_info_path = _build_full_gt_info_json(save_path, gt_path, "gt")
        _caption_videos(
            model_name="gt",
            model_path=caption_model,
            video_folder_root=gt_full_info_path,
            save_path=str(save_path),
            **kwargs,
        )

    return _score_from_caption_json(str(caption_json), str(gt_caption_json), semantics_model)
