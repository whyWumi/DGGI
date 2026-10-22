# DGGI

**Deep Generative Gradient Inversion with diffusion model**  
Liwen Wu, Zhizhi Liu, Bin Pu, Kang Wei, Hangcheng Cao, Shaowen Yao  
*Information Fusion*, 113 (2025), 102620. Available online August 2024.

[Paper](https://doi.org/10.1016/j.inffus.2024.102620) · [中文说明](README.zh-CN.md) · 

DGGI studies reconstruction of private training images from shared gradients using diffusion model priors and group consistency regularization.



## Layout

```text
DDGI/
├── run.py                          # Launch reconstruction from the repository root
├── gradientinversion-master/
│   ├── reconstruct_image.py        # Original experiment entry point
│   ├── project_paths.py            # Resource and output locations
│   ├── inversefed/                 # Modified gradient inversion library
│   └── guided_diffusion/           # Guided diffusion components
├── scripts/check_environment.py    # Prerequisite inspection
├── models/                        # Local pretrained weights, ignored by Git
├── data/                          # Local datasets, ignored by Git
├── images/                        # Generated images, ignored by Git
├── tables/                        # Generated metrics, ignored by Git
└── docs/                          # Reproduction and environment records
```

## Environment

The supplied Windows export used Python 3.9.16, PyTorch 2.0.0 with CUDA 11.7, and torchvision 0.15.1. The code includes explicit CUDA operations; experiments require a compatible NVIDIA GPU. The paper reports NVIDIA RTX 3090 GPUs.

A portable starting environment for Linux is provided:

```bash
conda env create -f environment.yml
conda activate dggi
```

The package versions come from the supplied export. This new environment has not been installed and validated on a separate GPU host. It replaces Windows build identifiers and the machine-specific prefix, while the sanitized original export is retained in [`docs/environment.original-windows.yml`](docs/environment.original-windows.yml).

For an existing Python 3.9 environment, install a suitable MPI runtime and a compatible PyTorch 2.0.0 / torchvision 0.15.1 pair, then run `python -m pip install -r requirements.txt`. CUDA wheel instructions are available in the [official PyTorch version documentation](https://pytorch.org/get-started/previous-versions/).

## Resources

Prepare class-organized images under `data/myimage/`:

```text
data/myimage/
├── class_000/
│   ├── image_001.jpg
│   └── image_002.jpg
└── class_001/
    └── image_003.jpg
```

The `myimagenet` loader uses torchvision `ImageFolder` and an 80/20 split with seed 0. Folder labels must match the class-index convention of both the target classifier and the conditional diffusion model; alphabetical folder order alone does not establish compatibility.

Place these weights in `models/`:

- `moco_v2_800ep_pretrain.pth.tar`: MoCo v2 ResNet-50 checkpoint.
- `256x256_diffusion.pt`: conditional diffusion model checkpoint.
- `256x256_classifier.pt`: diffusion guidance classifier checkpoint.

None of these weights were supplied with this snapshot. See [`models/README.md`](models/README.md) for upstream references and compatibility notes. Datasets and generated outputs are excluded from Git.

## Run

From the repository root, inspect prerequisites:

```bash
python run.py --help
python scripts/check_environment.py --runtime
```

The first command is available without PyTorch. The second reports missing default resources or runtime dependencies and returns a nonzero status when prerequisites are missing. It does not validate checkpoint contents, dataset labels, or numerical results.

For the Adam/cosine reconstruction branch described in the paper:

```bash
python run.py --optim ours --tv 0.015 --n_seed 4 --num_images 8
```

This selects the supplied `ours` branch and the paper's TV weight. It is a starting configuration, not a guarantee of paper-matching results. In particular, the supplied final reconstruction does not pass `x_max`, while the group-loss condition requires it. Resolve this documented algorithm issue before treating a run as complete DGGI reproduction.

The original parser default `--optim zhu` selects an L-BFGS/L2 branch with 300 iterations; that default was preserved. The `ours` branch specifies Adam, cosine distance, learning rate 0.1, and 24,000 iterations per reconstruction call. Diffusion initialization and several filename/batch assumptions remain fixed at eight images. Changing `--num_images` alone does not configure every component.

`run.py` creates default directories and launches the original entry point from the repository root. `--dryrun` is inherited from the research code; it still requires datasets, checkpoints, CUDA, and diffusion initialization.

## Paths

Defaults can be overridden before launching:

| Environment variable | Default |
| --- | --- |
| `DGGI_DATA_DIR` | `data/` |
| `DGGI_MODELS_DIR` | `models/` |
| `DGGI_IMAGES_DIR` | `images/` |
| `DGGI_TABLES_DIR` | `tables/` |
| `DGGI_PRETRAINED_CHECKPOINT` | `models/moco_v2_800ep_pretrain.pth.tar` |
| `DGGI_DIFFUSION_CHECKPOINT` | `models/256x256_diffusion.pt` |
| `DGGI_CLASSIFIER_CHECKPOINT` | `models/256x256_classifier.pt` |

Relative environment overrides resolve against the repository root. CLI paths such as `--data_path` still work. Diffusion sampler output uses `DGGI_IMAGES_DIR`; overriding only `--image_path` changes the final reconstruction output location, not that sampler location.

## Development

```bash
python -m pip install -r requirements-dev.txt
python -m black --check .
python -m compileall -q .
```

GitHub Actions checks formatting for the entry point and new utilities, Python 3.9 syntax for all source files, and launcher help. Vendored libraries retain their original formatting. The workflow does not execute reconstruction.

## Citation

```bibtex
@article{wu2025dggi,
  title   = {DGGI: Deep Generative Gradient Inversion with diffusion model},
  author  = {Wu, Liwen and Liu, Zhizhi and Pu, Bin and Wei, Kang and Cao, Hangcheng and Yao, Shaowen},
  journal = {Information Fusion},
  volume  = {113},
  pages   = {102620},
  year    = {2025},
  doi     = {10.1016/j.inffus.2024.102620}
}
```

