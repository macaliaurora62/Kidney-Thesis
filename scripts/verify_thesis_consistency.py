#!/usr/bin/env python3
"""Verifica la coerenza tra repository finale e valori riportati nella tesi."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ERRORS: list[str] = []
PASSED: list[str] = []


def record(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        PASSED.append(label)
        return
    message = label if not detail else f"{label}: {detail}"
    ERRORS.append(message)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def rounded_equal(actual: float, expected: float, digits: int = 4) -> bool:
    return round(float(actual), digits) == expected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


required_paths = [
    ROOT / "configs/experiment.yaml",
    ROOT / "notebooks/complete_pipeline_colab.ipynb",
    RESULTS / "dataset_distribution.csv",
    RESULTS / "final_metrics.json",
    RESULTS / "model_comparison.csv",
    RESULTS / "baseline/test_metrics.json",
    RESULTS / "two_branch/test_metrics.json",
    RESULTS / "yolo/test_metrics.json",
    RESULTS / "yolo/confusion_matrix.png",
    RESULTS / "yolo/confusion_matrix_normalized.png",
]

for required_path in required_paths:
    record(
        required_path.is_file(),
        f"file richiesto {required_path.relative_to(ROOT)}",
    )

if ERRORS:
    for error in ERRORS:
        print(f"[ERRORE] {error}")
    raise SystemExit(1)


summary = load_json(RESULTS / "final_metrics.json")
yolo_raw = load_json(RESULTS / "yolo/test_metrics.json")
baseline_raw = load_json(RESULTS / "baseline/test_metrics.json")
two_branch_raw = load_json(RESULTS / "two_branch/test_metrics.json")


thesis_values = {
    ("yolo11n_test_annotated", "precision"): 0.9820,
    ("yolo11n_test_annotated", "recall"): 0.9459,
    ("yolo11n_test_annotated", "map50"): 0.9619,
    ("yolo11n_test_annotated", "map50_95"): 0.7692,
    ("resnet50_baseline_test", "best_validation_macro_f1"): 0.8389,
    ("resnet50_baseline_test", "accuracy"): 0.8216,
    ("resnet50_baseline_test", "macro_f1"): 0.7652,
    ("resnet50_two_branch_test", "best_validation_macro_f1"): 0.8937,
    ("resnet50_two_branch_test", "accuracy"): 0.9129,
    ("resnet50_two_branch_test", "macro_f1"): 0.8991,
}

for (experiment, metric), expected in thesis_values.items():
    actual = summary[experiment][metric]
    record(
        rounded_equal(actual, expected),
        f"tesi {experiment}.{metric}",
        f"atteso {expected:.4f}, trovato {actual:.10f}",
    )


raw_mappings = [
    (yolo_raw, "Precision", summary["yolo11n_test_annotated"], "precision"),
    (yolo_raw, "Recall", summary["yolo11n_test_annotated"], "recall"),
    (yolo_raw, "mAP@0.50", summary["yolo11n_test_annotated"], "map50"),
    (yolo_raw, "mAP@0.50:0.95", summary["yolo11n_test_annotated"], "map50_95"),
    (baseline_raw, "best_validation_macro_f1", summary["resnet50_baseline_test"], "best_validation_macro_f1"),
    (baseline_raw, "test_accuracy", summary["resnet50_baseline_test"], "accuracy"),
    (baseline_raw, "test_macro_f1", summary["resnet50_baseline_test"], "macro_f1"),
    (two_branch_raw, "best_validation_macro_f1", summary["resnet50_two_branch_test"], "best_validation_macro_f1"),
    (two_branch_raw, "test_accuracy", summary["resnet50_two_branch_test"], "accuracy"),
    (two_branch_raw, "test_macro_f1", summary["resnet50_two_branch_test"], "macro_f1"),
]

for raw, raw_key, normalized, normalized_key in raw_mappings:
    record(
        math.isclose(
            float(raw[raw_key]),
            float(normalized[normalized_key]),
            rel_tol=0,
            abs_tol=1e-12,
        ),
        f"dato grezzo {raw_key}",
    )

record(
    baseline_raw.get("test_samples") == 1917,
    "test baseline di 1.917 immagini",
    f"trovato {baseline_raw.get('test_samples')}",
)
record(
    two_branch_raw.get("test_samples") == 1917,
    "test due rami di 1.917 immagini",
    f"trovato {two_branch_raw.get('test_samples')}",
)
record(
    summary["yolo11n_test_annotated"].get("images") == 96
    and summary["yolo11n_test_annotated"].get("instances") == 185,
    "test YOLO di 96 immagini e 185 istanze",
)


expected_distribution = {
    "Cyst": (2521, 577, 561, 3659, 49),
    "Normal": (3588, 766, 718, 5072, 5),
    "Stone": (964, 191, 220, 1375, 1),
    "Tumor": (1425, 358, 418, 2201, 79),
    "Total": (8498, 1892, 1917, 12307, 134),
}

with (RESULTS / "dataset_distribution.csv").open(
    encoding="utf-8", newline=""
) as stream:
    distribution_rows = list(csv.DictReader(stream))

actual_distribution = {
    row["class"]: tuple(
        int(row[column])
        for column in ("train", "validation", "test", "total", "excluded")
    )
    for row in distribution_rows
}
record(
    actual_distribution == expected_distribution,
    "distribuzione del dataset",
    f"trovato {actual_distribution}",
)


with (ROOT / "configs/experiment.yaml").open(encoding="utf-8") as stream:
    config = yaml.safe_load(stream)

config_checks = [
    (config["seed"] == 42, "seed 42"),
    (config["yolo"]["architecture"] == "yolo11n", "architettura YOLO11n"),
    (config["yolo"]["epochs"] == 100, "YOLO 100 epoche"),
    (config["yolo"]["image_size"] == 640, "YOLO immagini 640 px"),
    (config["yolo"]["batch_size"] == 16, "YOLO batch size 16"),
    (config["yolo"]["confidence_threshold"] == 0.25, "confidence 0,25"),
    (config["yolo"]["crop_nms_iou_threshold"] == 0.70, "NMS IoU 0,70"),
    (config["cropping"]["maximum_detections"] == 2, "massimo due detection"),
    (config["cropping"]["margin_ratio"] == 0.05, "margine crop 5%"),
    (config["cropping"]["kidney_panel_size"] == 224, "pannelli 224 px"),
    (config["cropping"]["combined_width"] == 448, "input largo 448 px"),
    (config["classification"]["architecture"] == "resnet50", "ResNet50"),
    (config["classification"]["dropout"] == 0.35, "dropout 0,35"),
    (config["two_branch"]["shared_encoder"] is True, "encoder condiviso"),
    (config["two_branch"]["fusion"] == "masked_max_pooling", "masked max pooling"),
]

for condition, label in config_checks:
    record(condition, f"configurazione {label}")


expected_hashes = {
    "confusion_matrix.png": "162196d91cd147362bef805000112409faf24bfa9d1871c85fb805cfa8c6a39d",
    "confusion_matrix_normalized.png": "fe5e655fa364f87fb8b78591d6c0c4a0d21c906140fa5d0b28ec07dcebe56b57",
}

for filename, expected_hash in expected_hashes.items():
    path = RESULTS / "yolo" / filename
    record(
        sha256(path) == expected_hash,
        f"artefatto YOLO finale {filename}",
    )


for prediction_file in RESULTS.rglob("test_predictions.csv"):
    with prediction_file.open(encoding="utf-8", newline="") as stream:
        row_count = sum(1 for _ in csv.DictReader(stream))
    record(
        row_count == 1917,
        f"numerosità {prediction_file.relative_to(ROOT)}",
        f"attese 1.917 righe, trovate {row_count}",
    )


notebook_text = (ROOT / "notebooks/complete_pipeline_colab.ipynb").read_text(
    encoding="utf-8"
)
record(
    notebook_text.count("assert len(test_dataset) == 1917") == 1,
    "vincolo test baseline nel notebook",
)
record(
    notebook_text.count("assert len(test_loader_two.dataset) == 1917") == 1,
    "vincolo test due rami nel notebook",
)


annotation_check = subprocess.run(
    [sys.executable, str(ROOT / "scripts/validate_annotations.py")],
    cwd=ROOT,
    text=True,
    capture_output=True,
    check=False,
)
record(
    annotation_check.returncode == 0,
    "annotazioni, split e assenza di leakage",
    annotation_check.stderr.strip() or annotation_check.stdout.strip(),
)


if ERRORS:
    print("Confronto tesi-repository NON superato.\n")
    for error in ERRORS:
        print(f"[ERRORE] {error}")
    raise SystemExit(1)

print("Confronto tesi-repository superato.")
print(f"Controlli superati: {len(PASSED)}")
print("Classificatori: test finale di 1.917 immagini.")
print("YOLO: test annotato di 96 immagini e 185 istanze.")
