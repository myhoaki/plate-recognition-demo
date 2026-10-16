"""Verify dataset discovery, polygon->box conversion, and split balance against real data.

Run from the project root (no DB needed):
    python3 verify_dataset.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "backend")

from app.datasets import (
    GROUP_CHUNK,
    discover_samples,
    group_key,
    label_path_for,
    read_class_names,
    split_for_group,
    yolo_boxes_from_label,
)

IMPORTS_ROOT = Path("/datasets/imports").resolve()


def main() -> int:
    if not IMPORTS_ROOT.is_dir():
        print(f"FAIL: {IMPORTS_ROOT} does not exist")
        return 1

    print(f"Scanning {IMPORTS_ROOT} ...")
    class_count, class_names = read_class_names(IMPORTS_ROOT)
    print(f"Source classes: nc={class_count} names={class_names} -> all collapse to license_plate")
    samples, skipped = discover_samples(IMPORTS_ROOT, class_count)

    print(f"\n=== Discovery ===")
    print(f"Usable samples (image + label with class-0 boxes): {len(samples)}")
    print(f"Skipped: {len(skipped)}")
    if skipped:
        reasons = Counter(row["reason"] for row in skipped)
        for reason, count in reasons.most_common():
            print(f"  {reason}: {count}")

    if not samples:
        print("FAIL: no usable samples found")
        return 1

    total_boxes = sum(len(row["boxes"]) for row in samples)
    print(f"Total class-0 boxes: {total_boxes}")
    print(f"Boxes per image: min={min(len(r['boxes']) for r in samples)}, "
          f"max={max(len(r['boxes']) for r in samples)}, "
          f"avg={total_boxes / len(samples):.2f}")

    print(f"\n=== Split simulation (GROUP_CHUNK={GROUP_CHUNK}) ===")
    grouped: dict[str, list] = {}
    for sample in samples:
        grouped.setdefault(group_key(sample["relative"]), []).append(sample)
    print(f"Frame groups: {len(grouped)}")

    split_rows: dict[str, list] = {"train": [], "val": [], "test": []}
    for key in sorted(grouped):
        split_rows[split_for_group(key)].extend(grouped[key])

    total = len(samples)
    for split, rows in split_rows.items():
        boxes = sum(len(row["boxes"]) for row in rows)
        print(f"  {split:5s}: {len(rows):5d} images ({len(rows)/total*100:5.1f}%), {boxes:6d} boxes")

    empty = [split for split, rows in split_rows.items() if not rows]
    if empty:
        print(f"  WARNING: empty splits -> {', '.join(empty)}")
    else:
        print("  All three splits are populated. OK")

    # Leakage check: no group may appear in more than one split.
    group_splits: dict[str, set[str]] = {}
    for key in sorted(grouped):
        group_splits.setdefault(split_for_group(key), set()).add(key)
    all_groups = set()
    for splits in group_splits.values():
        all_groups |= splits
    print(f"  Every one of {len(all_groups)} groups maps to exactly one split: "
          f"{'YES' if sum(len(v) for v in group_splits.values()) == len(all_groups) else 'NO'}")

    print(f"\n=== Grouping examples (leakage guard) ===")
    for name in ("carlong_0001", "carlong_0051", "greenpack_0001", "Tgmt_0925", "Dieu_0100"):
        candidate = Path(f"{name}.png")
        print(f"  {candidate} -> group={group_key(candidate)} split={split_for_group(group_key(candidate))}")

    print(f"\n=== Polygon -> box conversion sample ===")
    for row in samples[:3]:
        print(f"  {row['relative']}: {row['boxes']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
