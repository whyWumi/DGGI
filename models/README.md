# Local pretrained models

Place these files here, or configure their paths using `project_paths.py` environment variables:

| File | Purpose | Source reference |
| --- | --- | --- |
| `moco_v2_800ep_pretrain.pth.tar` | MoCo v2 ResNet-50 classifier initialization | [facebookresearch/moco](https://github.com/facebookresearch/moco) |
| `256x256_diffusion.pt` | Class-conditional image generation | [openai/guided-diffusion](https://github.com/openai/guided-diffusion) |
| `256x256_classifier.pt` | Guidance classifier | [openai/guided-diffusion](https://github.com/openai/guided-diffusion) |

No pretrained weights were present in the provided folder, so hashes and compatibility cannot be independently confirmed. These upstream references describe model sources, not a validated reproduction bundle.

The original entry point also writes `model.pth` here as a temporary target-model checkpoint. All model binaries are ignored by Git. Distribute large model files separately and add verified URLs and hashes when available.
