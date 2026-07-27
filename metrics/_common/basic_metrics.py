import math
import os

import cv2
import numpy as np
from metrics._common.episode_sort import episode_sort_key, natural_key
from metrics._common.progress import progress_iter


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_frames(video_dir):
    if not os.path.isdir(video_dir):
        return []
    return sorted(
        (
            os.path.join(video_dir, name)
            for name in os.listdir(video_dir)
            if name.lower().endswith(IMAGE_EXTS)
        ),
        key=lambda path: natural_key(os.path.basename(path)),
    )


def count_episode_frames(task_path, episode_id):
    episode_path = os.path.join(task_path, episode_id)
    if not os.path.isdir(episode_path):
        return 1
    total = 0
    for gid in sorted(os.listdir(episode_path), key=natural_key):
        video_dir = os.path.join(episode_path, gid, "video")
        total += len(list_frames(video_dir))
    return max(1, total)


def decode_adapter_view(view_id, gt_path):
    if view_id in ("head", "left", "right"):
        return view_id, os.path.join(gt_path, view_id, "__episode__", "video")
    if "__" in view_id:
        _legacy, view = view_id.rsplit("__", 1)
        if view in ("head", "left", "right"):
            return view, os.path.join(gt_path, view_id, "__episode__", "video")
    return view_id, os.path.join(gt_path, view_id, "__episode__", "video")


def resolve_gt_video_dir(gt_path, view_id, episode_id, gid=None):
    view, template = decode_adapter_view(view_id, gt_path)
    candidates = []
    if gid:
        candidates.append(os.path.join(gt_path, view_id, episode_id, gid, "video"))
    candidates.extend([
        os.path.join(gt_path, view_id, episode_id, "video"),
        os.path.join(gt_path, view, episode_id, "video"),
        template.replace("__episode__", episode_id),
        os.path.join(gt_path, episode_id, view, "video"),
    ])

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def read_frame(path):
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Failed to read frame: {path}")
    return frame


def resize_to_gt(pd_img, gt_img):
    gt_h, gt_w = gt_img.shape[:2]
    pd_h, pd_w = pd_img.shape[:2]
    if (pd_h, pd_w) == (gt_h, gt_w):
        return pd_img
    return cv2.resize(pd_img, (gt_w, gt_h), interpolation=cv2.INTER_AREA)


def cal_psnr(gt_img, pd_img):
    mse = np.mean((gt_img.astype(np.float64) - pd_img.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def ssim_single_channel(gt_channel, pd_channel):
    gt = gt_channel.astype(np.float64)
    pd = pd_channel.astype(np.float64)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    kernel = (11, 11)
    sigma = 1.5

    mu_gt = cv2.GaussianBlur(gt, kernel, sigma)
    mu_pd = cv2.GaussianBlur(pd, kernel, sigma)
    mu_gt_sq = mu_gt * mu_gt
    mu_pd_sq = mu_pd * mu_pd
    mu_gt_pd = mu_gt * mu_pd

    sigma_gt_sq = cv2.GaussianBlur(gt * gt, kernel, sigma) - mu_gt_sq
    sigma_pd_sq = cv2.GaussianBlur(pd * pd, kernel, sigma) - mu_pd_sq
    sigma_gt_pd = cv2.GaussianBlur(gt * pd, kernel, sigma) - mu_gt_pd

    numerator = (2.0 * mu_gt_pd + c1) * (2.0 * sigma_gt_pd + c2)
    denominator = (mu_gt_sq + mu_pd_sq + c1) * (sigma_gt_sq + sigma_pd_sq + c2)
    return float(np.mean(numerator / denominator))


def cal_ssim(gt_img, pd_img):
    scores = [
        ssim_single_channel(gt_img[:, :, channel], pd_img[:, :, channel])
        for channel in range(gt_img.shape[2])
    ]
    return float(np.mean(scores))


def finite_mean(values):
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    return float(np.mean(finite))


def metric_average(values):
    finite_avg = finite_mean(values)
    if finite_avg is not None:
        return finite_avg
    if values and all(math.isinf(value) for value in values):
        return float("inf")
    return None


def paired_frame_indices(pd_count, gt_count):
    pair_count = min(pd_count, gt_count)
    if pair_count <= 0:
        return []
    if pair_count == 1:
        return [(0, 0)]
    return [
        (
            round(idx * (pd_count - 1) / (pair_count - 1)),
            round(idx * (gt_count - 1) / (pair_count - 1)),
        )
        for idx in range(pair_count)
    ]


def compute_video_metrics(gt_video_dir, pd_video_dir, metric_names):
    gt_frames = list_frames(gt_video_dir)
    pd_frames = list_frames(pd_video_dir)
    pairs = paired_frame_indices(len(pd_frames), len(gt_frames))
    if not pairs:
        return None, len(gt_frames), len(pd_frames), 0

    cur_metrics = {metric: [] for metric in metric_names}
    for pd_idx, gt_idx in pairs:
        pd_img = read_frame(pd_frames[pd_idx])
        gt_img = read_frame(gt_frames[gt_idx])
        pd_img = resize_to_gt(pd_img, gt_img)
        for metric in metric_names:
            if metric == "psnr":
                cur_metrics[metric].append(cal_psnr(gt_img, pd_img))
            elif metric == "ssim":
                cur_metrics[metric].append(cal_ssim(gt_img, pd_img))
            else:
                raise ValueError(f"Unsupported basic metric: {metric}")

    averaged = {metric: metric_average(values) for metric, values in cur_metrics.items()}
    return averaged, len(gt_frames), len(pd_frames), len(pairs)


def build_metric_entries(results_by_metric):
    output = {}
    for metric, entries in results_by_metric.items():
        values = [entry["video_results"] for entry in entries if entry["video_results"] is not None]
        output[metric] = [metric_average(values), entries]
    return output


def compute_basic_metrics(gt_path, pd_path, metric_names=("psnr", "ssim")):
    metric_names = list(metric_names)
    for metric in metric_names:
        if metric not in {"psnr", "ssim"}:
            raise ValueError(f"Unsupported basic metric: {metric}")

    results_by_metric = {metric: [] for metric in metric_names}

    for view_id in sorted(os.listdir(pd_path), key=natural_key):
        view_path = os.path.join(pd_path, view_id)
        if not os.path.isdir(view_path):
            continue

        episodes = sorted(os.listdir(view_path), key=episode_sort_key)
        for episode_id in progress_iter(
            episodes,
            count_fn=lambda ep: count_episode_frames(view_path, ep),
            desc=f"basic_metrics:{view_id}",
        ):
            if episode_id.endswith((".png", ".json")):
                continue
            episode_path = os.path.join(view_path, episode_id)
            if not os.path.isdir(episode_path):
                continue

            for gid in sorted(os.listdir(episode_path), key=natural_key):
                pd_video_dir = os.path.join(episode_path, gid, "video")
                if not os.path.isdir(pd_video_dir):
                    continue

                gt_video_dir = resolve_gt_video_dir(gt_path, view_id, episode_id, gid=gid)
                video_metrics, gt_count, pd_count, compared_count = compute_video_metrics(
                    gt_video_dir, pd_video_dir, metric_names
                )
                if video_metrics is None:
                    print(
                        f"[basic_metrics] missing frames: view={view_id} episode={episode_id} "
                        f"gid={gid} gt={gt_count} pred={pd_count}"
                    )
                    continue
                if gt_count != pd_count:
                    print(
                        f"[basic_metrics] frame-count mismatch: view={view_id} episode={episode_id} "
                        f"gid={gid} gt={gt_count} pred={pd_count}; compared={compared_count}"
                    )

                video_path = os.path.join(pd_video_dir)
                for metric in metric_names:
                    item = {
                        "video_path": video_path,
                        "video_results": video_metrics[metric],
                        "gt_video_path": gt_video_dir,
                        "compared_frames": compared_count,
                        "gt_frames": gt_count,
                        "pred_frames": pd_count,
                    }
                    results_by_metric[metric].append(item)

    return build_metric_entries(results_by_metric)
