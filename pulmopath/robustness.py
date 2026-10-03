"""Run predeclared perturbation checks on the held-out test set."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .constants import CLASS_NAMES, IMAGE_SIZE
from .data import attach_absolute_paths


def _perturbed(path: str, size: int, name: str) -> np.ndarray:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    if name == "brightness_minus_8pct":
        image = ImageEnhance.Brightness(image).enhance(0.92)
    elif name == "brightness_plus_8pct":
        image = ImageEnhance.Brightness(image).enhance(1.08)
    elif name == "contrast_minus_10pct":
        image = ImageEnhance.Contrast(image).enhance(0.90)
    elif name == "contrast_plus_10pct":
        image = ImageEnhance.Contrast(image).enhance(1.10)
    elif name == "color_gain_warm":
        arr = np.asarray(image, dtype=np.float32) * np.array([1.04, 1.00, 0.96], dtype=np.float32)
        image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    elif name == "color_gain_cool":
        arr = np.asarray(image, dtype=np.float32) * np.array([0.96, 1.00, 1.04], dtype=np.float32)
        image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    elif name == "jpeg_quality_85":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
    resampling = {
        "resize_nearest": Image.Resampling.NEAREST,
        "resize_bilinear": Image.Resampling.BILINEAR,
        "resize_bicubic": Image.Resampling.BICUBIC,
    }.get(name)
    if resampling is not None:
        # Inputs already match the model's 128x128 baseline size. Downsample
        # to 75% and restore to model size so interpolation variants actually
        # perturb pixels instead of resizing an image to its existing size.
        reduced_size = max(1, round(size * 0.75))
        image = image.resize((reduced_size, reduced_size), resampling)
        image = image.resize((size, size), resampling)
        return np.asarray(image, dtype=np.float32)
    return np.asarray(image.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32)


def _metrics(y_true, probabilities):
    y_pred = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(CLASS_NAMES)), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=range(len(CLASS_NAMES)), zero_division=0)),
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES))).tolist(),
    }


def run(model_path: Path, data_dir: Path, split_manifest: Path, output: Path, batch_size: int = 32):
    import tensorflow as tf

    metadata = json.loads(model_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    size = int(metadata.get("image_size", IMAGE_SIZE))
    model = tf.keras.models.load_model(model_path)
    frame = attach_absolute_paths(pd.read_csv(split_manifest), data_dir.resolve())
    test = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    y_true = test["label_id"].to_numpy(dtype=int)
    perturbations = [
        "baseline_bicubic", "brightness_minus_8pct", "brightness_plus_8pct",
        "contrast_minus_10pct", "contrast_plus_10pct", "resize_nearest",
        "resize_bilinear", "resize_bicubic", "jpeg_quality_85", "color_gain_warm", "color_gain_cool",
    ]
    result = {}
    for name in perturbations:
        arrays = np.stack([_perturbed(path, size, name) for path in test["absolute_path"]])
        probabilities = model.predict(arrays, batch_size=batch_size, verbose=0)
        result[name] = _metrics(y_true, probabilities)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "model": model_path.name,
        "test_examples": len(test),
        "perturbation_definitions": {
            "brightness": "Pillow brightness factor 0.92 or 1.08",
            "contrast": "Pillow contrast factor 0.90 or 1.10",
            "resizing": "downsample to 75% of model input size, then resize back using nearest, bilinear, or bicubic interpolation",
            "compression": "JPEG quality 85",
            "color": "RGB channel gain of +/-4% for red and blue",
        },
        "results": result,
        "caveat": "These controlled perturbations do not establish clinical robustness or external generalization.",
    }, indent=2), encoding="utf-8")
    print(f"Saved robustness results to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/lung_subset_small"))
    parser.add_argument("--split-manifest", type=Path, default=Path("reports/splits.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/robustness.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run(args.model, args.data_dir, args.split_manifest, args.output, args.batch_size)


if __name__ == "__main__":
    main()
