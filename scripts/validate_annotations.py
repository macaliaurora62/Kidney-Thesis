"""Validate the versioned YOLO annotations and the group-aware split."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_IMAGES = {"train": 411, "val": 93, "test": 96}
EXPECTED_BOXES = {"train": 782, "val": 179, "test": 185}
EXPECTED_GROUPS = {"train": 150, "val": 33, "test": 34}
EXPECTED_CATEGORIES = {"Cyst": 150, "Normal": 150, "Stone": 150, "Tumor": 150}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/annotations"),
        help="Directory containing labels/ and metadata/split_manifest.csv.",
    )
    return parser.parse_args()


def validate(root: Path) -> None:
    manifest = root / "metadata" / "split_manifest.csv"
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    required = {"category", "group", "filename", "split", "boxes"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest assente o colonne mancanti: {sorted(required)}")

    image_counts: Counter[str] = Counter()
    box_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    group_splits: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    manifest_labels: set[Path] = set()

    for row in rows:
        split = row["split"]
        if split not in EXPECTED_IMAGES:
            raise ValueError(f"Split non riconosciuto: {split!r}")

        expected_boxes = int(row["boxes"])
        label_path = root / "labels" / split / Path(row["filename"]).with_suffix(".txt").name
        manifest_labels.add(label_path.resolve())
        if not label_path.is_file():
            raise FileNotFoundError(f"Label mancante: {label_path}")

        raw = label_path.read_text(encoding="utf-8").strip()
        lines = raw.splitlines() if raw else []
        if len(lines) != expected_boxes:
            raise ValueError(
                f"Numero di box non coerente in {label_path}: "
                f"manifest={expected_boxes}, label={len(lines)}"
            )

        for line_number, line in enumerate(lines, start=1):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Formato YOLO non valido in {label_path}:{line_number}")
            class_id = int(fields[0])
            x_center, y_center, width, height = map(float, fields[1:])
            if class_id != 0 or not all(
                0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)
            ):
                raise ValueError(f"Valori YOLO non validi in {label_path}:{line_number}")
            if width <= 0.0 or height <= 0.0:
                raise ValueError(f"Box degenere in {label_path}:{line_number}")

        image_counts[split] += 1
        box_counts[split] += len(lines)
        category_counts[row["category"]] += 1
        group_splits[(row["category"], row["group"])].add(split)

    actual_labels = {path.resolve() for path in (root / "labels").glob("*/*.txt")}
    extra_labels = sorted(actual_labels - manifest_labels)
    if extra_labels:
        raise ValueError(f"Label non presenti nel manifest: {extra_labels[:5]}")

    leakage = {key: splits for key, splits in group_splits.items() if len(splits) > 1}
    groups_per_split = Counter(next(iter(splits)) for splits in group_splits.values())

    checks = {
        "immagini per split": (dict(image_counts), EXPECTED_IMAGES),
        "box per split": (dict(box_counts), EXPECTED_BOXES),
        "gruppi per split": (dict(groups_per_split), EXPECTED_GROUPS),
        "immagini per categoria": (dict(category_counts), EXPECTED_CATEGORIES),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"{name}: ottenuto {actual}, atteso {expected}")
    if leakage:
        raise ValueError(f"Leakage tra split: {leakage}")

    empty_labels = sum(int(row["boxes"]) == 0 for row in rows)
    print("Annotazioni valide")
    print(f"Immagini: {len(rows)} | Box: {sum(box_counts.values())}")
    print(f"Gruppi category/group: {len(group_splits)} | Leakage: 0")
    print(f"Label vuote dichiarate nel manifest: {empty_labels}")


if __name__ == "__main__":
    validate(parse_args().root)
