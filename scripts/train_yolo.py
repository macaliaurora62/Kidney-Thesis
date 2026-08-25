"""Addestra YOLO11n con la configurazione utilizzata nella tesi."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Percorso di data.yaml")
    parser.add_argument("--output", type=Path, default=Path("runs/yolo"))
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(args.data)

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO("yolo11n.pt")
    result = model.train(
        data=str(args.data),
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        seed=42,
        deterministic=True,
        device=device,
        project=str(args.output),
        name="yolo11n_kidney",
        exist_ok=True,
    )

    best_checkpoint = Path(result.save_dir) / "weights" / "best.pt"
    print(best_checkpoint)


if __name__ == "__main__":
    main()
