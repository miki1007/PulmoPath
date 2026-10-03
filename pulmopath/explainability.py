"""Export reproducible Grad-CAM examples for a selected checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .constants import CLASS_NAMES, IMAGE_SIZE, RANDOM_SEED
from .data import attach_absolute_paths, load_rgb_array
from .gradcam import gradcam_heatmap, overlay_heatmap


def export_examples(
    model_path: Path,
    data_dir: Path,
    split_manifest: Path,
    output_dir: Path,
    split: str = "test",
    samples_per_class: int = 1,
) -> Path:
    import tensorflow as tf
    import matplotlib.pyplot as plt

    model_path = model_path.resolve()
    metadata_path = model_path.with_suffix(".metadata.json")
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Checkpoint and metadata are required: {model_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame = attach_absolute_paths(pd.read_csv(split_manifest), data_dir.resolve())
    subset = frame.loc[frame["split"] == split].copy()
    if subset.empty:
        raise ValueError(f"No rows found for split={split!r}.")
    model = tf.keras.models.load_model(model_path)
    size = int(metadata.get("image_size", IMAGE_SIZE))
    layer_name = metadata.get("gradcam_layer", "last_conv_features")
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = []

    for class_name in CLASS_NAMES:
        candidates = subset.loc[subset["label"] == class_name].sort_values("relative_path").head(samples_per_class)
        for _, row in candidates.iterrows():
            pixels = load_rgb_array(row["absolute_path"], size=size)
            scores = model.predict(pixels[None, ...], verbose=0)[0]
            predicted_index = int(np.argmax(scores))
            predicted_name = CLASS_NAMES[predicted_index]
            heatmap = gradcam_heatmap(model, pixels, predicted_index, layer_name)
            overlay = overlay_heatmap(pixels, heatmap)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(pixels.astype(np.uint8))
            axes[0].set_title("Original model input")
            axes[1].imshow(overlay)
            axes[1].set_title(f"Grad-CAM for predicted label: {predicted_name}")
            for axis in axes:
                axis.axis("off")
            fig.suptitle(f"Dataset label: {row['label']} · Model output: {predicted_name}")
            fig.text(0.5, 0.015, "Model-sensitivity visualization; not proof of biological reasoning or a confirmed lesion.", ha="center", fontsize=8)
            fig.tight_layout(rect=(0, 0.04, 1, 0.93))
            safe_name = Path(str(row["relative_path"])).stem
            image_path = output_dir / f"{class_name}_{safe_name}_gradcam.png"
            fig.savefig(image_path, dpi=160)
            plt.close(fig)
            examples.append({
                "file": str(row["relative_path"]),
                "true_dataset_label": class_name,
                "predicted_dataset_label": predicted_name,
                "class_scores": {name: float(scores[i]) for i, name in enumerate(CLASS_NAMES)},
                "heatmap_target": predicted_name,
                "visualization": image_path.name,
                "caveat": "Grad-CAM is a model-sensitivity visualization, not proof of biological reasoning or a confirmed lesion.",
            })

    summary = {
        "model": model_path.name,
        "split": split,
        "samples_per_true_class": samples_per_class,
        "class_names": CLASS_NAMES,
        "examples": examples,
        "random_seed": RANDOM_SEED,
        "patient_independent": bool(subset["patient_independent"].iloc[0]) if "patient_independent" in subset else False,
        "grouping_level": str(subset["grouping_level"].iloc[0]) if "grouping_level" in subset else "unknown",
    }
    summary_path = output_dir / "examples.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/lung_subset_small"))
    parser.add_argument("--split-manifest", type=Path, default=Path("reports/splits.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/explainability"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--samples-per-class", type=int, default=1)
    args = parser.parse_args()
    output = export_examples(args.model, args.data_dir, args.split_manifest, args.output_dir, args.split, args.samples_per_class)
    print(f"Saved Grad-CAM examples and summary to {output}")


if __name__ == "__main__":
    main()
