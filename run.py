"""Launch DGGI from the repository root; --help does not require PyTorch."""

from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "gradientinversion-master"
sys.path.insert(0, str(CODE))
from project_paths import prepare_directories


def main():
    options = runpy.run_path(str(CODE / "inversefed" / "options.py"))["options"]
    options().parse_args()  # Validate arguments and serve help before importing ML packages.
    prepare_directories()
    return subprocess.call(
        [sys.executable, str(CODE / "reconstruct_image.py"), *sys.argv[1:]], cwd=ROOT
    )


if __name__ == "__main__":
    raise SystemExit(main())
