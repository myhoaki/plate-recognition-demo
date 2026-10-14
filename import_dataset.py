"""Copy images from an extracted dataset into the app's rawimages folder."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_DESTINATION = Path(__file__).parent / "rawimages"


def unique_destination(destination: Path) -> Path:
    """Choose a filename that will not overwrite an existing raw image."""
    if not destination.exists():
        return destination
    index = 2
    while True:
        candidate = destination.with_name(f"{destination.stem}_{index}{destination.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy supported images from an extracted dataset into rawimages/."
    )
    parser.add_argument("dataset", type=Path, help="Path to the extracted dataset directory")
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"Image folder used by the app (default: {DEFAULT_DESTINATION})",
    )
    args = parser.parse_args()

    source = args.dataset.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if not source.is_dir():
        parser.error(f"dataset directory does not exist: {source}")
    if source == destination or source in destination.parents:
        parser.error("destination cannot be inside the dataset directory")

    images = sorted(
        (path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path).casefold(),
    )
    if not images:
        print(f"No supported images found under {source}")
        return 1

    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for image in images:
        relative = image.relative_to(source)
        # Flatten the Kaggle folder structure because the gallery reads rawimages/ directly.
        flattened_name = "__".join(relative.parts)
        target = unique_destination(destination / flattened_name)
        shutil.copy2(image, target)
        copied += 1

    print(f"Copied {copied} images into {destination}")
    print("Refresh the website and choose Server folder to browse them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
