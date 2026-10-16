"""Regression tests for read-only snapshot removal and dataset discovery.

Run: python3 test_snapshot_removal.py
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "backend")

from app.datasets import make_readonly, remove_readonly_tree as _remove_readonly_tree


def make_readonly_snapshot(root: Path) -> Path:
    """Create a tree shaped like a real snapshot: 0o444 files, 0o555 dirs."""
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for index in range(3):
            (image_dir / f"sample{index}.png").write_bytes(b"fake-image-bytes")
            (label_dir / f"sample{index}.txt").write_text("0 0.5 0.5 0.2 0.1\n")
    (root / "data.yaml").write_text("path: .\nnames:\n  0: license_plate\n")
    (root / "manifest.json").write_text('{"task": "detector"}')
    make_readonly(root)
    return root


def test_remove_readonly_snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_readonly_snapshot(Path(tmp) / "snapshot")
        files = len([item for item in root.rglob("*") if item.is_file()])
        assert files == 20, files  # 6 split dirs x 3 files + data.yaml + manifest.json
        assert not (os.stat(root).st_mode & stat.S_IWUSR), "root should be read-only before delete"

        _remove_readonly_tree(root)

        assert not root.exists(), "snapshot directory still exists after removal"
        assert list(Path(tmp).iterdir()) == [], "leftover entries in parent directory"
        print(f"PASS test_remove_readonly_snapshot (removed {files} files)")


def test_remove_missing_path_is_noop():
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "does-not-exist"
        _remove_readonly_tree(missing)
        assert not missing.exists()
        print("PASS test_remove_missing_path_is_noop")


def test_remove_plain_writable_tree():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "plain"
        (root / "nested").mkdir(parents=True)
        (root / "nested" / "file.txt").write_text("data")
        _remove_readonly_tree(root)
        assert not root.exists()
        print("PASS test_remove_plain_writable_tree")


def test_snapshot_dirs_are_readonly_before_delete():
    """Confirms the failure mode this fix addresses: parent dir blocks unlink."""
    with tempfile.TemporaryDirectory() as tmp:
        root = make_readonly_snapshot(Path(tmp) / "snapshot")
        label_dir = root / "labels" / "train"
        assert not (os.stat(label_dir).st_mode & stat.S_IWUSR)
        # A child file cannot be unlinked while its parent is read-only.
        blocked = False
        try:
            (label_dir / "sample0.txt").unlink()
        except PermissionError:
            blocked = True
        assert blocked, "expected PermissionError on read-only parent (running as non-root)"
        print("PASS test_snapshot_dirs_are_readonly_before_delete")


if __name__ == "__main__":
    test_remove_readonly_snapshot()
    test_remove_missing_path_is_noop()
    test_remove_plain_writable_tree()
    if os.geteuid() != 0:
        test_snapshot_dirs_are_readonly_before_delete()
    else:
        print("SKIP test_snapshot_dirs_are_readonly_before_delete (running as root)")
    print("\nAll snapshot-removal tests PASSED.")
