"""Train one model using the training partition and validation-only selection."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import CLASS_NAMES, IMAGE_SIZE, RANDOM_SEED
from .data import attach_absolute_paths, make_dataset
from .models import build_model


def _history_json(history, stage: str) -> dict:
    return {"stage": stage, **{key: [float(value) for value in values] for key, values in history.history.items()}}


def _class_weights(labels: np.ndarray) -> dict[int, float] | None:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    positive = counts[counts > 0]
    if len(positive) != len(CLASS_NAMES) or positive.max() / positive.min() < 1.5:
        return None
    total = counts.sum()
    return {i: float(total / (len(CLASS_NAMES) * count)) for i, count in enumerate(counts) if count > 0}


def train(args: argparse.Namespace) -> Path:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    data_root = args.data_dir.resolve()
    frame = pd.read_csv(args.split_manifest)
    frame = attach_absolute_paths(frame, data_root)
    training = frame.loc[frame["split"] == "train"]
    validation = frame.loc[frame["split"] == "validation"]
    if training.empty or validation.empty:
        raise ValueError("Split manifest must contain both training and validation rows.")
    train_ds = make_dataset(frame, "train", args.batch_size, args.image_size, augment=True, seed=args.seed)
    val_ds = make_dataset(frame, "validation", args.batch_size, args.image_size, augment=False, seed=args.seed)
    class_weights = _class_weights(training["label_id"].to_numpy(dtype=int))

    weight_source = "imagenet" if args.model == "mobilenetv2" and args.pretrained == "imagenet" else None
    model, gradcam_layer = build_model(args.model, args.image_size, len(CLASS_NAMES), weights=weight_source)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output), monitor="val_loss", mode="min", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=args.patience, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=3, min_lr=1e-7, verbose=1
        ),
    ]
    history_parts = []
    training_started = time.perf_counter()
    print("Training class counts:", training["label"].value_counts().reindex(CLASS_NAMES, fill_value=0).to_dict())
    print("Class weighting:", class_weights if class_weights else "not applied; class balance does not meet 1.5x rule")
    first = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )
    history_parts.append(_history_json(first, "head_or_baseline"))
    best_val_loss = float(np.min(first.history["val_loss"]))
    best_epoch = int(np.argmin(first.history["val_loss"])) + 1

    if args.fine_tune and args.model == "mobilenetv2":
        backbone = model.get_layer("mobilenetv2_backbone")
        backbone.trainable = True
        cutoff = max(0, len(backbone.layers) - args.unfreeze_last_layers)
        for layer in backbone.layers[:cutoff]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        candidate = output.with_name(output.stem + ".fine_tune_candidate.keras")
        fine_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(candidate), monitor="val_loss", mode="min", save_best_only=True, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", patience=args.patience, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", mode="min", factor=0.5, patience=2, min_lr=1e-7, verbose=1
            ),
        ]
        second = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=args.epochs,
            epochs=args.epochs + args.fine_tune_epochs,
            callbacks=fine_callbacks,
            class_weight=class_weights,
            verbose=1,
        )
        history_parts.append(_history_json(second, "fine_tuning"))
        second_val_loss = float(np.min(second.history["val_loss"]))
        if second_val_loss < best_val_loss and candidate.exists():
            candidate.replace(output)
            best_val_loss = second_val_loss
            best_epoch = args.epochs + int(np.argmin(second.history["val_loss"])) + 1
        elif candidate.exists():
            candidate.unlink()

    training_seconds = time.perf_counter() - training_started

    # Reload the selected checkpoint to verify it is loadable before writing metadata.
    selected = tf.keras.models.load_model(output)
    if selected.output_shape[-1] != len(CLASS_NAMES):
        raise RuntimeError("Saved checkpoint has an unexpected number of output classes.")
    metadata = {
        "project": "WLC PulmoPath AI",
        "model_version": "0.1.0",
        "model_name": args.model,
        "class_names": CLASS_NAMES,
        "class_to_id": {name: index for index, name in enumerate(CLASS_NAMES)},
        "image_size": args.image_size,
        "input": "RGB pixels in [0,255]; preprocessing is embedded in the model",
        "gradcam_layer": gradcam_layer,
        "pretrained_weights": weight_source,
        "selected_epoch_by_validation_loss": best_epoch,
        "selected_validation_loss": best_val_loss,
        "training_seconds": round(training_seconds, 3),
        "random_seed": args.seed,
        "split_manifest": str(args.split_manifest),
        "training_examples": len(training),
        "validation_examples": len(validation),
        "patient_independent_split": bool(training["patient_independent"].iloc[0])
        if "patient_independent" in training.columns else False,
        "grouping_level": str(training["grouping_level"].iloc[0])
        if "grouping_level" in training.columns else "unknown",
        "software": {"python": platform.python_version(), "tensorflow": tf.__version__},
        "hardware": {
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "tensorflow_devices": [device.name for device in tf.config.list_physical_devices()],
        },
        "medical_use": "Educational research only; not for clinical diagnosis or treatment.",
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    output.with_suffix(".history.json").write_text(json.dumps(history_parts, indent=2), encoding="utf-8")
    print(f"Saved and reloaded selected checkpoint: {output}")
    print("Selected epoch (validation loss):", best_epoch)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/lung_subset_small"))
    parser.add_argument("--split-manifest", type=Path, default=Path("reports/splits.csv"))
    parser.add_argument("--model", choices=("baseline", "mobilenetv2"), default="baseline")
    parser.add_argument("--pretrained", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--fine-tune-epochs", type=int, default=15)
    parser.add_argument("--unfreeze-last-layers", type=int, default=30)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(f"artifacts/models/{args.model}.keras")
    if args.fine_tune and args.model != "mobilenetv2":
        parser.error("--fine-tune is available only with --model mobilenetv2")
    train(args)


if __name__ == "__main__":
    main()
