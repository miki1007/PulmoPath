"""Evaluate a selected model once on the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from .constants import CLASS_NAMES, IMAGE_SIZE, RANDOM_SEED
from .data import attach_absolute_paths, make_dataset


def _group_bootstrap(y_true, y_pred, groups, grouping_level="unknown group", seed=RANDOM_SEED, draws=1000):
    unique = np.unique(groups)
    if len(unique) < 10:
        return {"status": "not calculated: fewer than 10 independent groups", "groups": int(len(unique))}
    rng = np.random.default_rng(seed)
    group_rows = {group: np.flatnonzero(groups == group) for group in unique}
    accuracy_values, macro_f1_values = [], []
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([group_rows[group] for group in sampled])
        accuracy_values.append(accuracy_score(y_true[indices], y_pred[indices]))
        macro_f1_values.append(f1_score(y_true[indices], y_pred[indices], average="macro", labels=range(len(CLASS_NAMES)), zero_division=0))
    def interval(values):
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
    return {
        "status": "group bootstrap percentile intervals",
        "groups": int(len(unique)),
        "draws": draws,
        "accuracy_95_percent_interval": interval(accuracy_values),
        "macro_f1_95_percent_interval": interval(macro_f1_values),
        "unit": str(grouping_level),
    }


def evaluate(model_path: Path, data_dir: Path, split_manifest: Path, output_dir: Path, batch_size=32, seed=RANDOM_SEED):
    import tensorflow as tf

    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}. Run the training command first.")
    metadata_path = model_path.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame = attach_absolute_paths(pd.read_csv(split_manifest), data_dir.resolve())
    test = frame.loc[frame["split"] == "test"].copy().reset_index(drop=True)
    if test.empty:
        raise ValueError("No test rows found in split manifest.")
    model = tf.keras.models.load_model(model_path)
    dataset = make_dataset(frame, "test", batch_size, int(metadata.get("image_size", IMAGE_SIZE)), augment=False)
    x_only = dataset.map(lambda images, labels: images)
    probabilities = model.predict(x_only, verbose=1)
    y_true = test["label_id"].to_numpy(dtype=int)
    y_pred = probabilities.argmax(axis=1)
    labels = list(range(len(CLASS_NAMES)))
    per_class = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    one_hot = np.eye(len(CLASS_NAMES))[y_true]
    roc_auc, average_precision = {}, {}
    curves = {}
    for index, class_name in enumerate(CLASS_NAMES):
        if len(np.unique(one_hot[:, index])) == 2:
            roc_auc[class_name] = float(roc_auc_score(one_hot[:, index], probabilities[:, index]))
            average_precision[class_name] = float(average_precision_score(one_hot[:, index], probabilities[:, index]))
            curves[class_name] = {
                "roc": roc_curve(one_hot[:, index], probabilities[:, index]),
                "pr": precision_recall_curve(one_hot[:, index], probabilities[:, index]),
            }
        else:
            roc_auc[class_name] = None
            average_precision[class_name] = None
    groups = test.get("split_group", pd.Series([f"image_{i}" for i in range(len(test))])).astype(str).to_numpy()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model": model_path.name,
        "test_examples": int(len(test)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            class_name: {
                "precision": float(per_class[0][i]),
                "recall": float(per_class[1][i]),
                "f1": float(per_class[2][i]),
                "support": int(per_class[3][i]),
                "roc_auc_ovr": roc_auc[class_name],
                "average_precision_ovr": average_precision[class_name],
            }
            for i, class_name in enumerate(CLASS_NAMES)
        },
        "macro_average": report["macro avg"],
        "weighted_average": report["weighted avg"],
        "confusion_matrix_rows_true_columns_predicted": matrix.tolist(),
        "normalized_confusion_matrix_rows_true_columns_predicted": normalized.round(6).tolist(),
        "group_bootstrap": _group_bootstrap(
            y_true,
            y_pred,
            groups,
            grouping_level=str(test["grouping_level"].iloc[0])
            if "grouping_level" in test.columns else "unknown group",
            seed=seed,
        ),
        "limitations": [
            "This is an internal held-out dataset estimate, not external or clinical validation.",
            "Model scores are not calibrated disease probabilities.",
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(output_dir / "confusion_matrix.csv")
    pd.DataFrame(normalized, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(output_dir / "confusion_matrix_normalized.csv")
    errors = test.copy()
    errors["true_class"] = [CLASS_NAMES[i] for i in y_true]
    errors["predicted_class"] = [CLASS_NAMES[i] for i in y_pred]
    errors["predicted_score"] = probabilities.max(axis=1)
    errors["correct"] = y_true == y_pred
    errors.loc[~errors["correct"]].sort_values("predicted_score", ascending=False).to_csv(output_dir / "misclassified_examples.csv", index=False)

    import matplotlib.pyplot as plt
    if curves:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for class_name, values in curves.items():
            fpr, tpr, _ = values["roc"]
            precision_curve, recall_curve, _ = values["pr"]
            axes[0].plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc[class_name]:.3f})")
            axes[1].plot(recall_curve, precision_curve, label=f"{class_name} (AP={average_precision[class_name]:.3f})")
        axes[0].plot([0, 1], [0, 1], linestyle=":", color="gray")
        axes[0].set(title="One-vs-rest ROC", xlabel="False positive rate", ylabel="True positive rate")
        axes[1].set(title="One-vs-rest precision–recall", xlabel="Recall", ylabel="Precision")
        for axis in axes:
            axis.legend(fontsize=7)
            axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "roc_precision_recall.png", dpi=160)
        plt.close(fig)

    wrong = np.flatnonzero(y_true != y_pred)[:12]
    if len(wrong):
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        for axis, i in zip(axes.flat, wrong):
            with Image.open(test.iloc[i]["absolute_path"]) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(f"true: {CLASS_NAMES[y_true[i]]}\npred: {CLASS_NAMES[y_pred[i]]} ({probabilities[i, y_pred[i]]:.2f})", fontsize=8)
            axis.axis("off")
        for axis in list(axes.flat)[len(wrong):]:
            axis.axis("off")
        fig.suptitle("Representative misclassified held-out examples")
        fig.tight_layout()
        fig.savefig(output_dir / "error_examples.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=range(len(CLASS_NAMES)), xticklabels=CLASS_NAMES, yticks=range(len(CLASS_NAMES)), yticklabels=CLASS_NAMES,
           xlabel="Predicted class", ylabel="True class", title="Normalized test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for row in range(len(CLASS_NAMES)):
        for col in range(len(CLASS_NAMES)):
            ax.text(col, row, f"{normalized[row, col]:.2f}\n(n={matrix[row, col]})", ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_normalized.png", dpi=160)
    plt.close(fig)

    history_path = model_path.with_suffix(".history.json")
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for stage in history:
            epochs = range(1, len(stage.get("loss", [])) + 1)
            if stage.get("loss"):
                axes[0].plot(epochs, stage["loss"], label=f"train {stage['stage']}")
            if stage.get("val_loss"):
                axes[0].plot(epochs, stage["val_loss"], linestyle="--", label=f"validation {stage['stage']}")
            if stage.get("accuracy"):
                axes[1].plot(epochs, stage["accuracy"], label=f"train {stage['stage']}")
            if stage.get("val_accuracy"):
                axes[1].plot(epochs, stage["val_accuracy"], linestyle="--", label=f"validation {stage['stage']}")
        axes[0].set(title="Training and validation loss", xlabel="Epoch", ylabel="Loss")
        axes[1].set(title="Training and validation accuracy", xlabel="Epoch", ylabel="Accuracy")
        for axis in axes:
            axis.legend(fontsize=7)
            axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "training_curves.png", dpi=160)
        plt.close(fig)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/lung_subset_small"))
    parser.add_argument("--split-manifest", type=Path, default=Path("reports/splits.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/evaluation"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    evaluate(args.model, args.data_dir, args.split_manifest, args.output_dir, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
