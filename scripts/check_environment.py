"""Inspect DGGI prerequisites without importing models or running reconstruction."""

import argparse
import importlib.metadata
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gradientinversion-master"))
from project_paths import (
    DATA_DIR,
    PRETRAINED_CHECKPOINT,
    DIFFUSION_CHECKPOINT,
    CLASSIFIER_CHECKPOINT,
)

PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "PIL": "Pillow",
    "matplotlib": "matplotlib",
    "cv2": "opencv-python",
    "skimage": "scikit-image",
    "scipy": "scipy",
    "lpips": "lpips",
    "blobfile": "blobfile",
    "mpi4py": "mpi4py",
    "tqdm": "tqdm",
    "six": "six",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Import PyTorch and test CUDA availability.",
    )
    args = parser.parse_args()
    failures = 0
    print(f"Python: {sys.version.split()[0]}")
    for module, distribution in PACKAGES.items():
        present = importlib.util.find_spec(module) is not None
        try:
            version = importlib.metadata.version(distribution) if present else "missing"
        except importlib.metadata.PackageNotFoundError:
            version = "version unknown"
        print(f"{'OK' if present else 'MISSING'} {distribution}: {version}")
        failures += not present
    for path in [PRETRAINED_CHECKPOINT, DIFFUSION_CHECKPOINT, CLASSIFIER_CHECKPOINT]:
        present = path.is_file() and path.stat().st_size > 0
        print(f"{'OK' if present else 'MISSING'} checkpoint {path}")
        failures += not present
    path = DATA_DIR / "myimage"
    present = path.is_dir() and any(p.is_dir() for p in path.iterdir())
    print(f"{'OK' if present else 'MISSING'} ImageFolder root {path}")
    failures += not present
    if args.runtime:
        try:
            import torch

            available = torch.cuda.is_available()
            print(f"CUDA available: {available}")
            failures += not available
        except Exception as error:
            print(f"PyTorch runtime error: {error}")
            failures += 1
    print(
        "Checks use default paths. Checkpoint/data compatibility and numerical results are not validated."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
