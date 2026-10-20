"""Resource locations for the DGGI research code."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _path(variable, default):
    value = Path(os.environ.get(variable, str(default))).expanduser()
    return value if value.is_absolute() else ROOT / value


DATA_DIR = _path("DGGI_DATA_DIR", ROOT / "data")
MODELS_DIR = _path("DGGI_MODELS_DIR", ROOT / "models")
IMAGES_DIR = _path("DGGI_IMAGES_DIR", ROOT / "images")
TABLES_DIR = _path("DGGI_TABLES_DIR", ROOT / "tables")
PRETRAINED_CHECKPOINT = _path(
    "DGGI_PRETRAINED_CHECKPOINT", MODELS_DIR / "moco_v2_800ep_pretrain.pth.tar"
)
DIFFUSION_CHECKPOINT = _path(
    "DGGI_DIFFUSION_CHECKPOINT", MODELS_DIR / "256x256_diffusion.pt"
)
CLASSIFIER_CHECKPOINT = _path(
    "DGGI_CLASSIFIER_CHECKPOINT", MODELS_DIR / "256x256_classifier.pt"
)


def prepare_directories():
    """Create local resource/output folders when explicitly starting a run."""
    for folder in (DATA_DIR, MODELS_DIR, IMAGES_DIR, TABLES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
