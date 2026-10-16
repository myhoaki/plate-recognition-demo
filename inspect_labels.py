"""Inspect why label files are skipped: class ids, coordinate counts, empty files."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

IMPORTS_ROOT = Path("/datasets/imports").resolve()
LABELS_ROOT = IMPORTS_ROOT / "raw-dataset" / "labels"

shape_counter: Counter[str] = Counter()
class_counter: Counter[str] = Counter()
empty = 0
total = 0
examples: dict[str, list[str]] = {}

for label_path in sorted(LABELS_ROOT.rglob("*.txt")):
    total += 1
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not lines:
        empty += 1
        shape_counter["EMPTY FILE"] += 1
        continue
    for line in lines:
        try:
            values = [float(v) for v in line.split()]
        except ValueError:
            shape_counter["NON-NUMERIC"] += 1
            examples.setdefault("NON-NUMERIC", []).append(f"{label_path.name}: {line[:80]}")
            continue
        class_ids = {int(v) for v in values[:1]}
        coords = values[1:]
        key = f"class={sorted(class_ids)} coords={len(coords)}"
        shape_counter[key] += 1
        examples.setdefault(key, []).append(f"{label_path.name}: {line[:100]}")

print(f"Total label files: {total}")
print(f"Empty label files: {empty}")
print(f"\nLabel line shapes (file-level counts):")
for shape, count in shape_counter.most_common(15):
    print(f"  {shape:35s} {count}")
print(f"\nExamples per shape:")
for shape, rows in list(examples.items())[:6]:
    print(f"\n  [{shape}]")
    for row in rows[:2]:
        print(f"    {row}")
