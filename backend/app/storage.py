import hashlib
import os
import shutil
from pathlib import Path

from fastapi import HTTPException, UploadFile


DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
UPLOADS_DIR = DATA_DIR / "uploads"
RAW_IMAGES_DIR = Path(os.environ.get("RAW_IMAGES_DIR", "/data/rawimages"))
LEGACY_LABELS_DIRS = [
    Path(item)
    for item in os.environ.get("LEGACY_LABELS_DIRS", "/legacy-labels:/legacy-volume").split(":")
    if item
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 40_000_000


def ensure_dirs():
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_size(path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise HTTPException(status_code=400, detail="The file is not a readable image.")
    height, width = image.shape[:2]
    if width * height > MAX_PIXELS:
        raise HTTPException(status_code=413, detail="Image exceeds the 40 megapixel limit.")
    return width, height


async def save_upload(upload: UploadFile, storage_key: str) -> tuple[Path, str, int, int]:
    extension = Path(upload.filename or "").suffix.lower()
    if extension not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Use JPG, PNG, WebP, or BMP images.")
    ensure_dirs()
    destination = UPLOADS_DIR / storage_key
    total = 0
    digest = hashlib.sha256()
    with destination.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_BYTES:
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Image exceeds the 15 MB limit.")
            digest.update(chunk)
            output.write(chunk)
    width, height = image_size(destination)
    return destination, digest.hexdigest(), width, height


def image_path(source: str, filename: str, storage_key: str) -> Path:
    if storage_key.startswith("uploads/"):
        path = DATA_DIR / storage_key
    elif source == "rawimages":
        path = RAW_IMAGES_DIR / Path(filename).name
    else:
        path = UPLOADS_DIR / Path(storage_key).name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def import_legacy_image(record: dict, labels_dir: Path) -> str | None:
    """Copy a previous labels-folder image into managed filesystem storage."""
    image = record.get("image", {})
    old_name = image.get("file", "")
    if not old_name or Path(old_name).name != old_name:
        return None
    old_path = labels_dir / old_name
    if image.get("source") == "rawimages":
        raw = RAW_IMAGES_DIR / Path(image.get("original_filename", "")).name
        if raw.is_file():
            return f"rawimages/{raw.name}"
    if not old_path.is_file():
        return None
    ensure_dirs()
    target_name = f"legacy-{Path(old_name).name}"
    target = UPLOADS_DIR / target_name
    if not target.exists():
        shutil.copy2(old_path, target)
    return f"uploads/{target_name}"
