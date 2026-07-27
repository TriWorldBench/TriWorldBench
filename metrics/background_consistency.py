import os
import json
import logging
import numpy as np
import clip
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics._common.utils import load_video, load_dimension_info, clip_transform
from tqdm import tqdm
from metrics._common.progress import progress_iter
from metrics.dynamic_degree import DynamicDegree
from easydict import EasyDict as edict
from metrics._common.distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)

MAX_CLIP_FRAMES = 800


def _score_frame_features(
    image_features,
    *,
    chunk_start: int,
    first_image_feature,
    former_image_feature,
):
    """Roll cosine-sim across one encoded chunk; carry state for next chunk."""
    video_sim = 0.0
    cnt_per_video = 0
    for local_i in range(len(image_features)):
        global_i = chunk_start + local_i
        image_feature = image_features[local_i].unsqueeze(0)
        if global_i == 0:
            first_image_feature = image_feature
        else:
            sim_pre = max(0.0, F.cosine_similarity(former_image_feature, image_feature).item())
            sim_fir = max(0.0, F.cosine_similarity(first_image_feature, image_feature).item())
            cur_sim = (sim_pre + sim_fir) / 2
            video_sim += cur_sim
            cnt_per_video += 1
        former_image_feature = image_feature
    return video_sim, cnt_per_video, first_image_feature, former_image_feature


def _encode_video_features(clip_model, images, device, *, image_transform=None):
    """Encode one video; chunk when frame count exceeds MAX_CLIP_FRAMES."""
    num_frames = len(images)
    if num_frames <= MAX_CLIP_FRAMES:
        chunk = image_transform(images) if image_transform is not None else images
        chunk = chunk.to(device)
        chunk_features = clip_model.encode_image(chunk)
        chunk_features = F.normalize(chunk_features, dim=-1, p=2)
        video_sim, cnt_per_video, _, _ = _score_frame_features(
            chunk_features,
            chunk_start=0,
            first_image_feature=None,
            former_image_feature=None,
        )
        return video_sim, cnt_per_video, num_frames

    first_image_feature = None
    former_image_feature = None
    video_sim = 0.0
    cnt_per_video = 0

    for chunk_start in range(0, num_frames, MAX_CLIP_FRAMES):
        chunk = images[chunk_start:chunk_start + MAX_CLIP_FRAMES]
        if image_transform is not None:
            chunk = image_transform(chunk)
        chunk = chunk.to(device)
        chunk_features = clip_model.encode_image(chunk)
        chunk_features = F.normalize(chunk_features, dim=-1, p=2)
        chunk_sim, chunk_cnt, first_image_feature, former_image_feature = _score_frame_features(
            chunk_features,
            chunk_start=chunk_start,
            first_image_feature=first_image_feature,
            former_image_feature=former_image_feature,
        )
        video_sim += chunk_sim
        cnt_per_video += chunk_cnt
        del chunk, chunk_features

    return video_sim, cnt_per_video, num_frames


def background_consistency(clip_model, preprocess, video_list, device, read_frame, raft_model_path):
    sim = 0.0
    cnt = 0
    args_new = edict({
        "model": raft_model_path,
        "small": False,
        "mixed_precision": False,
        "alternate_corr": False
    })
    dynamic = DynamicDegree(args_new, device)
    video_results = []
    image_transform = clip_transform(224)
    for video_path in progress_iter(video_list, disable=get_rank() > 0):
        video_sim = 0.0
        cnt_per_video = 0
        if read_frame:
            video_path = video_path[:-4].replace('videos', 'frames').replace(' ', '_')
            tmp_paths = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path))]
            images = []
            for tmp_path in tmp_paths:
                images.append(preprocess(Image.open(tmp_path)))
            images = torch.stack(images)
            video_sim, cnt_per_video, num_frames = _encode_video_features(
                clip_model, images, device,
            )
        else:
            images = load_video(video_path)
            video_sim, cnt_per_video, num_frames = _encode_video_features(
                clip_model, images, device, image_transform=image_transform,
            )
        cnt += cnt_per_video
        sim_per_image = video_sim / (num_frames - 1) if num_frames > 1 else 0.0
        dynamic_score = dynamic.infer(video_path)
        if dynamic_score <= 0.1213:
            sim_per_image = sim_per_image * dynamic_score

        sim += sim_per_image
        video_results.append({
            'video_path': video_path, 
            'video_results': sim_per_image,
            'video_sim': video_sim,
            'cnt_per_video': cnt_per_video})
    # sim_per_video = sim / (len(video_list) - 1)
    sim_per_frame = sim / cnt
    return sim_per_frame, video_results


def compute_background_consistency(json_dir, submodules_list, **kwargs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vit_path = submodules_list.get('clip_model')
    raft_model_path = submodules_list.get('raft_model')
    read_frame = submodules_list.get('read_frame', False)

    if vit_path is None or raft_model_path is None:
        raise ValueError("background_consistency requires clip_model and raft_model checkpoints from config")

    clip_model, preprocess = clip.load(vit_path, device=device)
    video_list, _ = load_dimension_info(json_dir, dimension='background_consistency', lang='en')
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = background_consistency(
        clip_model,
        preprocess,
        video_list,
        device,
        read_frame,
        raft_model_path,
    )
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        sim = sum([d['video_sim'] for d in video_results])
        cnt = sum([d['cnt_per_video'] for d in video_results])
        all_results = sim / cnt
    return all_results, video_results
