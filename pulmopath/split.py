"""Create deterministic, group-aware train/validation/test partitions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .constants import CLASS_TO_ID, RANDOM_SEED


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def _score_partition(y: np.ndarray, indices: np.ndarray, target: float) -> float:
    all_counts = np.bincount(y, minlength=len(CLASS_TO_ID)).astype(float)
    selected = np.bincount(y[indices], minlength=len(CLASS_TO_ID)).astype(float)
    overall_error = abs(len(indices) / max(len(y), 1) - target)
    class_ratios = selected / np.maximum(all_counts, 1)
    return overall_error + float(np.mean(np.abs(class_ratios - target)))


def _best_group_holdout(frame: pd.DataFrame, target: float, seed: int, attempts: int = 400):
    y = frame["label_id"].to_numpy(dtype=int)
    groups = frame["split_group"].astype(str).to_numpy()
    best = None
    for offset in range(attempts):
        splitter = GroupShuffleSplit(n_splits=1, test_size=target, random_state=seed + offset)
        train_i, holdout_i = next(splitter.split(np.zeros(len(frame)), y, groups))
        score = _score_partition(y, holdout_i, target)
        if best is None or score < best[0]:
            best = (score, train_i, holdout_i)
    assert best is not None
    return best[1], best[2], best[0]


def build_split(
    inventory: Path,
    output: Path,
    group_manifest: Path | None = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    frame = pd.read_csv(inventory)
    frame = frame.loc[frame["usable"].astype(bool)].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError("No usable images in inventory. Run the dataset audit first.")
    if set(frame["label"].unique()) != set(CLASS_TO_ID):
        raise ValueError("Every one of lung_n, lung_aca, and lung_scc must contain usable images.")
    if frame.groupby("sha256")["label"].nunique().max() > 1:
        raise ValueError("An exact duplicate has conflicting labels. Resolve the labels before splitting.")

    frame["split_group"] = frame["duplicate_group_id"].fillna("").astype(str)
    mixed_duplicate_groups = frame.groupby("duplicate_group_id")["label"].nunique()
    mixed_duplicate_groups = mixed_duplicate_groups[mixed_duplicate_groups > 1].index.tolist()
    if mixed_duplicate_groups:
        print(
            f"Review {len(mixed_duplicate_groups)} cross-class near-duplicate groups in "
            "reports/data_audit.json; related images are kept in the same split."
        )
    grouping_level = "duplicate_cluster"
    patient_independent = False
    if group_manifest is not None:
        groups = pd.read_csv(group_manifest)
        required = {"relative_path", "group_id", "group_type"}
        if not required.issubset(groups.columns):
            raise ValueError("Group manifest must include columns: relative_path, group_id, group_type")
        group_types = set(groups["group_type"].astype(str).str.lower())
        if not group_types.issubset({"patient", "slide"}) or len(group_types) != 1:
            raise ValueError("group_type must be consistently 'patient' or 'slide' throughout the manifest.")
        grouping_level = next(iter(group_types))
        patient_independent = grouping_level == "patient"
        group_lookup = dict(zip(groups["relative_path"].astype(str), groups["group_id"].astype(str)))
        missing = [p for p in frame["relative_path"] if p not in group_lookup]
        if missing:
            raise ValueError(f"Group manifest is missing {len(missing)} image paths; first missing: {missing[0]}")
        uf = UnionFind(len(frame))
        first_for_duplicate, first_for_patient = {}, {}
        for i, row in frame.iterrows():
            dup = str(row["duplicate_group_id"])
            patient = group_lookup[str(row["relative_path"])]
            for value, table in ((dup, first_for_duplicate), (patient, first_for_patient)):
                if value in table:
                    uf.union(i, table[value])
                else:
                    table[value] = i
        frame["split_group"] = [f"group_{uf.find(i):06d}" for i in range(len(frame))]
    elif frame["split_group"].eq("").any():
        raise ValueError("Missing duplicate-group identifiers. Re-run the current audit script.")
    frame["grouping_level"] = grouping_level
    frame["patient_independent"] = patient_independent

    # The first holdout is the test set; only the remaining groups are used to choose validation.
    trainval_idx, test_idx, test_score = _best_group_holdout(frame, 0.15, seed)
    trainval = frame.iloc[trainval_idx].reset_index(drop=True)
    validation_fraction_of_trainval = 0.15 / 0.85
    train_idx, val_idx, val_score = _best_group_holdout(
        trainval, validation_fraction_of_trainval, seed + 10_000
    )
    frame["split"] = ""
    frame.loc[trainval_idx, "split"] = "train"
    frame.loc[test_idx, "split"] = "test"
    frame.loc[trainval_idx[val_idx], "split"] = "validation"

    missing_by_split = {
        name: sorted(set(CLASS_TO_ID) - set(frame.loc[frame["split"] == name, "label"]))
        for name in ("train", "validation", "test")
    }
    missing_by_split = {name: missing for name, missing in missing_by_split.items() if missing}
    if missing_by_split:
        raise ValueError(
            "A fixed three-way split cannot represent every class while keeping groups intact: "
            f"{missing_by_split}. Review group counts and use a repeated grouped cross-validation "
            "protocol before reporting model performance."
        )

    ratios = frame["split"].value_counts(normalize=True).to_dict()
    per_class = frame.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(orient="index")
    frame["split_seed"] = seed
    frame["split_method"] = "repeated_group_shuffle_search"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print("Split ratios:", {key: round(value, 4) for key, value in ratios.items()})
    print("Class counts by split:", per_class)
    print("Group-shuffle balance scores:", {"test": round(test_score, 5), "validation": round(val_score, 5)})
    if group_manifest is None:
        print("Patient/slide independence cannot be guaranteed: no group manifest was supplied.")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path("reports/image_inventory.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/splits.csv"))
    parser.add_argument("--group-manifest", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    build_split(args.inventory, args.output, args.group_manifest, args.seed)


if __name__ == "__main__":
    main()
