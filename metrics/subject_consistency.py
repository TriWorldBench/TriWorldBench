import io
import os
import cv2
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from metrics._common.progress import progress_iter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from metrics.dynamic_degree import DynamicDegree
from easydict import EasyDict as edict
from metrics._common.utils import load_video, load_dimension_info, dino_transform, dino_transform_Image, CACHE_DIR
import logging

from metrics._common.distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def subject_consistency(model, video_list, device, read_frame, raft_model_path):
    sim = 0.0
    cnt = 0
    video_results = []
    args_new = edict({
        "model": raft_model_path,
        "small": False,
        "mixed_precision": False,
        "alternate_corr": False
    })
    dynamic = DynamicDegree(args_new, device)
    if read_frame:
        image_transform = dino_transform_Image(224)
    else:
        image_transform = dino_transform(224)

    for video_path in progress_iter(video_list, disable=get_rank() > 0):
        video_sim = 0.0
        if read_frame:
            video_path = video_path[:-4].replace('videos', 'frames').replace(' ', '_')
            tmp_paths = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path))]
            images = []
            for tmp_path in tmp_paths:
                images.append(image_transform(Image.open(tmp_path)))
        else:
            images = load_video(video_path)
            images = image_transform(images)
        for i in range(len(images)):
            with torch.no_grad():
                image = images[i].unsqueeze(0)
                image = image.to(device)
                image_features = model(image)
                image_features = F.normalize(image_features, dim=-1, p=2)
                if i == 0:
                    first_image_features = image_features
                else:
                    sim_pre = max(0.0, F.cosine_similarity(former_image_features, image_features).item())
                    sim_fir = max(0.0, F.cosine_similarity(first_image_features, image_features).item())
                    cur_sim = (sim_pre + sim_fir) / 2
                    video_sim += cur_sim
                    cnt += 1
            former_image_features = image_features
        sim_per_images = video_sim / (len(images) - 1)
        dynamic_score = dynamic.infer(video_path)

        # ===== 唯一耦合点：阈值判断 =====
        if dynamic_score <= 0.1213:
            sim_per_images = sim_per_images * dynamic_score

        sim += sim_per_images
        video_results.append({
            'video_path': video_path,
            'video_results': sim_per_images
        })
    # sim_per_video = sim / (len(video_list) - 1)
    sim_per_frame = sim / cnt
    return sim_per_frame, video_results


def compute_subject_consistency(json_dir, submodules_list, **kwargs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    submodules_kwargs = dict(submodules_list)
    read_frame = submodules_kwargs.pop('read_frame', False)
    raft_model_path = submodules_kwargs.pop('raft_model', None)
    dino_weight_path = submodules_kwargs.pop('path', None)
    if raft_model_path is None:
        raise ValueError("subject_consistency requires raft_model checkpoint from config")

    # Always construct model from local repo without triggering hub URL download.
    # Then load checkpoint weights from config ckpt.subject_consistency.weight.
    if dino_weight_path is None:
        raise ValueError("subject_consistency requires local dino weight path from config")
    if not os.path.isfile(dino_weight_path):
        raise FileNotFoundError(f"subject_consistency dino weight not found: {dino_weight_path}")

    dino_model = torch.hub.load(pretrained=False, **submodules_kwargs).to(device)

    ckpt = torch.load(dino_weight_path, map_location='cpu')
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            state_dict = ckpt['state_dict']
        elif 'teacher' in ckpt and isinstance(ckpt['teacher'], dict):
            state_dict = ckpt['teacher']
        elif 'model' in ckpt and isinstance(ckpt['model'], dict):
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # remove possible wrappers from different training/export styles
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith('module.'):
            nk = nk[len('module.'):]
        if nk.startswith('backbone.'):
            nk = nk[len('backbone.'):]
        cleaned_state_dict[nk] = v

    missing, unexpected = dino_model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        logger.warning(f"DINO missing keys when loading local ckpt: {len(missing)}")
    if unexpected:
        logger.warning(f"DINO unexpected keys when loading local ckpt: {len(unexpected)}")

    dino_model.eval()
    logger.info("Initialize DINO success")
    video_list, _ = load_dimension_info(json_dir, dimension='subject_consistency', lang='en')
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = subject_consistency(
        dino_model,
        video_list,
        device,
        read_frame,
        raft_model_path,
    )
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results