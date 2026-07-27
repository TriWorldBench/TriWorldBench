# Metric Weights Download Guide

All weights must be placed under:

```text
./weights/
```

The evaluation pipeline reads model paths from `config/config.yaml` → `weights_root`. Internal sub-path names must match the table below.

## Full Model Repositories

| Local directory | Used by | Hugging Face repository |
|---|---|---|
| `qwenvl3` | VLM judge, VLM consistency (01/02/03), VQA | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| `clip-vit-base-patch16` | semantic_alignment (CLIP text encoder) | [openai/clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16) |
| `depth-anything` | depth_accuracy | [depth-anything/Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) |
| `sam3` | trajectory_accuracy (detection) | [1038lab/sam3](https://huggingface.co/1038lab/sam3) public mirror |

> **SAM3 note:** The official `facebook/sam3` repository is gated. Use the public `1038lab/sam3` mirror.

> **Qwen3 note:** `qwenvl3` is the local folder name for `Qwen3-VL-8B-Instruct`. The deprecated `Qwen2.5-VL-7B-Instruct` checkpoint is no longer required.

### Example: download Qwen3-VL

```bash
export WEIGHTS=./weights
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir "$WEIGHTS/qwenvl3"
```

## VBench and WorldScore Checkpoints

Source dataset: [videogenevalkit/checkpoints](https://huggingface.co/datasets/videogenevalkit/checkpoints)

| Local path | Direct download |
|---|---|
| `clip_model/ViT-B-32.pt` | [ViT-B-32.pt](https://huggingface.co/datasets/videogenevalkit/checkpoints/resolve/main/vbench/pretrained/clip_model/ViT-B-32.pt) |
| `clip_model/ViT-L-14.pt` | [ViT-L-14.pt](https://huggingface.co/datasets/videogenevalkit/checkpoints/resolve/main/vbench/pretrained/clip_model/ViT-L-14.pt) |
| `dino_model/dino_vitbase16_pretrain.pth` | [dino_vitbase16_pretrain.pth](https://huggingface.co/datasets/videogenevalkit/checkpoints/resolve/main/vbench/pretrained/dino_model/dino_vitbase16_pretrain.pth) |
| `pyiqa_model/musiq_spaq_ckpt-358bb6af.pth` | [musiq_spaq_ckpt-358bb6af.pth](https://huggingface.co/datasets/videogenevalkit/checkpoints/resolve/main/vbench/pretrained/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth) |
| `_hf_downloads/videogenevalkit-checkpoints/worldscore/Tartan-C-T-TSKH-spring540x960-M.pth` | [Tartan-C-T-TSKH-spring540x960-M.pth](https://huggingface.co/datasets/videogenevalkit/checkpoints/resolve/main/worldscore/Tartan-C-T-TSKH-spring540x960-M.pth) |
| `VFIMamba/VFIMamba.pkl` | [VFIMamba.pkl](https://huggingface.co/datasets/videogenevalkit/checkpoints/resolve/main/worldscore/VFIMamba.pkl) |
| `raft_model/RAFT/models/raft-things.pth` | [raft-things.pth](https://huggingface.co/datasets/videogenevalkit/checkpoints/resolve/main/worldscore/raft-things.pth) |
| `sam/sam_vit_h_4b8939.pth` | [sam_vit_h_4b8939.pth](https://huggingface.co/datasets/videogenevalkit/checkpoints/resolve/main/worldscore/sam_vit_h_4b8939.pth) |
| `sam/sam2.1_hiera_base_plus.pt` | [sam2.1_hiera_base_plus.pt](https://huggingface.co/datasets/videogenevalkit/checkpoints/resolve/main/worldscore/sam2.1_hiera_base_plus.pt) |
| `sam/sam2.1_hiera_large.pt` | [sam2.1_hiera_large.pt](https://huggingface.co/datasets/videogenevalkit/checkpoints/resolve/main/worldscore/sam2.1_hiera_large.pt) |

## Other Checkpoints

| Local path | Direct download |
|---|---|
| `aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth` | [sa_0_4_vit_l_14_linear.pth](https://huggingface.co/Kurt232/vbench/resolve/main/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth) |
| `vjepa/vith16.pth.tar` | [V-JEPA ViT-H/16](https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar) |
| `vjepa/ssv2-probe.pth.tar` | [V-JEPA SSV2 probe](https://dl.fbaipublicfiles.com/jepa/vith16/ssv2-probe.pth.tar) |
| `torch_home/hub/checkpoints/resnet34-b627a593.pth` | [ResNet-34](https://download.pytorch.org/models/resnet34-b627a593.pth) |

Expected SHA256 for ResNet-34:

```text
b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f
```

## Expected Directory Layout

```text
weights/
├── qwenvl3/                         # Qwen3-VL-8B-Instruct (VLM metrics)
├── clip-vit-base-patch16/
├── clip_model/
├── depth-anything/
├── sam3/
├── sam/
├── raft_model/RAFT/models/
├── VFIMamba/
├── pyiqa_model/
├── aesthetic_model/emb_reader/
├── dino_model/
├── vjepa/
├── torch_home/hub/checkpoints/
└── _hf_downloads/videogenevalkit-checkpoints/worldscore/
```

## Notes

- `_hf_downloads/` holds source copies for checkpoints also exposed through top-level paths.
- `dino_model/facebookresearch_dino_minimal/` is a DINO source checkout, not a downloadable checkpoint.
- Full file inventory: [`weights/METRIC_WEIGHTS_MANIFEST.txt`](weights/METRIC_WEIGHTS_MANIFEST.txt).
