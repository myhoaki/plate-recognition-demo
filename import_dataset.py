"""Copy images from an extracted dataset into a raw training dataset folder."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REPOSITORY_DIR = Path(__file__).parent
DEFAULT_DESTINATIONS = {
    "detector": REPOSITORY_DIR / "datasets" / "imports",
    "ocr": REPOSITORY_DIR / "raw-dataset-ocr",
}


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


def yolo_label_for(source: Path, image: Path, relative: Path) -> Path | None:
    parts = list(relative.parts)
    image_index = next((index for index, part in enumerate(parts) if part.lower() == "images"), None)
    if image_index is not None:
        parts[image_index] = "labels"
        candidate = source.joinpath(*parts).with_suffix(".txt")
    else:
        candidate = source / "labels" / relative.with_suffix(".txt")
    return candidate if candidate.is_file() else None


def valid_yolo_label(path: Path) -> bool:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            values = [float(value) for value in line.split()]
            if len(values) < 5 or int(values[0]) != 0:
                continue
            coords = values[1:]
            if len(coords) == 4:
                cx, cy, width, height = coords
                if width <= 0 or height <= 0 or not (0 <= cx <= 1 and 0 <= cy <= 1 and width <= 1 and height <= 1):
                    return False
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                if any(value < 0 or value > 1 for value in coords):
                    return False
            else:
                return False
    except (OSError, ValueError):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy supported images into a raw detector or OCR dataset folder.")
    parser.add_argument("dataset", type=Path, help="Path to the extracted dataset directory")
    parser.add_argument("--task", choices=DEFAULT_DESTINATIONS, default="detector", help="Dataset type (default: detector)")
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help="Override the destination (defaults to the selected task's raw dataset folder)",
    )
    args = parser.parse_args()

    source = args.dataset.expanduser().resolve()
    destination = (args.destination or DEFAULT_DESTINATIONS[args.task]).expanduser().resolve()
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
    labels_copied = 0
    for image in images:
        relative = image.relative_to(source)
        try:
            image.resolve(strict=True).relative_to(source)
        except (OSError, ValueError):
            continue
        # Flatten nested dataset folders because the app gallery reads one folder directly.
        flattened_name = "__".join(relative.parts)
        target = unique_destination(destination / flattened_name)
        shutil.copy2(image, target)
        copied += 1
        label = yolo_label_for(source, image, relative) if args.task == "detector" else None
        if label:
            if not valid_yolo_label(label):
                print(f"Skipped malformed YOLO label: {label.relative_to(source)}")
                continue
            label_destination = destination / "labels" / f"{target.name}.txt"
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label, label_destination)
            labels_copied += 1

    print(f"Copied {copied} images into {destination}")
    if args.task == "detector":
        print(f"Imported {labels_copied} YOLO sidecar labels. Refresh Verify Datasets to review them before training.")
    else:
        print("OCR files are staged in raw-dataset-ocr/. Add verified plate text in raw-dataset-ocr/annotations.csv before OCR training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
