"""Audit a class-folder histology dataset without training a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .constants import CLASS_TO_ID, SUPPORTED_EXTENSIONS


def _phash(image: Image.Image) -> int:
    """Return a compact 64-bit perceptual hash using a small DCT."""
    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    n = 32
    basis = np.cos((np.pi / n) * (np.arange(n)[:, None] + 0.5) * np.arange(n)[None, :])
    basis[:, 0] *= 1 / math.sqrt(2)
    basis *= math.sqrt(2 / n)
    dct = basis.T @ gray @ basis
    low = dct[:8, :8].flatten()
    median = float(np.median(low[1:]))
    bits = (low > median).astype(np.uint8)
    return int.from_bytes(np.packbits(bits, bitorder="big").tobytes(), "big")


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def _record_image(path: Path, root: Path, label: str) -> dict:
    record = {
        "relative_path": path.relative_to(root).as_posix(),
        "label": label,
        "label_id": CLASS_TO_ID[label],
        "bytes": path.stat().st_size,
        "extension": path.suffix.lower(),
        "mode": "unknown",
        "width": None,
        "height": None,
        "sha256": "",
        "phash": "",
        "grayscale": False,
        "blank": False,
        "usable": False,
        "status": "unreadable",
    }
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            record["mode"] = image.mode
            record["width"], record["height"] = image.size
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            record["grayscale"] = bool(
                image.mode in {"1", "L", "I", "F", "I;16"}
                or np.max(np.abs(rgb.astype(np.int16) - rgb[..., :1].astype(np.int16))) <= 1
            )
            record["blank"] = bool(float(rgb.std()) < 1.0)
            record["sha256"] = digest
            record["phash"] = f"{_phash(image):016x}"
            record["usable"] = not record["blank"]
            record["status"] = "usable" if record["usable"] else "blank_excluded"
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        record["status"] = f"unreadable:{type(exc).__name__}"
    return record


def audit_dataset(root: Path, near_duplicate_distance: int = 4) -> tuple[dict, list[dict]]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}\n"
            "Download the histology subset linked in docs/MATERIAL_AUDIT.md, then place "
            "lung_n/, lung_aca/, and lung_scc/ directly inside the configured data folder."
        )

    missing = [name for name in CLASS_TO_ID if not (root / name).is_dir()]
    if missing:
        raise ValueError(f"Missing required class folders under {root}: {', '.join(missing)}")

    records: list[dict] = []
    non_image_files = []
    for label in CLASS_TO_ID:
        for path in sorted((root / label).rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                non_image_files.append(path.relative_to(root).as_posix())
                continue
            records.append(_record_image(path, root, label))

    usable_indices = [i for i, row in enumerate(records) if row["usable"]]
    uf = _UnionFind(len(records))
    by_sha: dict[str, list[int]] = defaultdict(list)
    for i in usable_indices:
        by_sha[records[i]["sha256"]].append(i)
    exact_pairs = 0
    exact_cross_label = []
    for indices in by_sha.values():
        for i in indices[1:]:
            uf.union(indices[0], i)
            exact_pairs += 1
        if len({records[i]["label"] for i in indices}) > 1:
            exact_cross_label.append([records[i]["relative_path"] for i in indices])

    hashes = [(i, int(records[i]["phash"], 16)) for i in usable_indices]
    near_pairs = 0
    for offset, (i, hi) in enumerate(hashes):
        for j, hj in hashes[offset + 1 :]:
            if records[i]["sha256"] == records[j]["sha256"]:
                continue
            if (hi ^ hj).bit_count() <= near_duplicate_distance:
                uf.union(i, j)
                near_pairs += 1

    component_sizes: Counter = Counter()
    component_labels: dict[int, set[str]] = defaultdict(set)
    for i in usable_indices:
        root_id = uf.find(i)
        component_sizes[root_id] += 1
        component_labels[root_id].add(records[i]["label"])
    component_to_name = {}
    for number, component in enumerate(sorted(component_sizes, key=lambda x: min(
        records[i]["relative_path"] for i in usable_indices if uf.find(i) == x
    )), start=1):
        component_to_name[component] = f"dup_{number:05d}"
    for i in usable_indices:
        records[i]["duplicate_group_id"] = component_to_name[uf.find(i)]
    for i in set(range(len(records))) - set(usable_indices):
        records[i]["duplicate_group_id"] = ""

    counts = Counter(row["label"] for row in records)
    usable_counts = Counter(row["label"] for row in records if row["usable"])
    grayscale_counts = Counter(row["label"] for row in records if row["grayscale"])
    blank_counts = Counter(row["label"] for row in records if row["blank"])
    unreadable_counts = Counter(row["label"] for row in records if row["status"].startswith("unreadable"))
    summary = {
        "dataset_root": str(root),
        "class_to_id": CLASS_TO_ID,
        "counts_before_cleaning": {label: counts[label] for label in CLASS_TO_ID},
        "counts_after_excluding_blank_and_unreadable": {label: usable_counts[label] for label in CLASS_TO_ID},
        "grayscale_or_nearly_grayscale": {label: grayscale_counts[label] for label in CLASS_TO_ID},
        "blank_excluded": {label: blank_counts[label] for label in CLASS_TO_ID},
        "unreadable": {label: unreadable_counts[label] for label in CLASS_TO_ID},
        "non_image_files_ignored": non_image_files,
        "exact_duplicate_links": exact_pairs,
        "near_duplicate_links_at_phash_distance": {
            "threshold": near_duplicate_distance,
            "links": near_pairs,
        },
        "duplicate_components": sum(size > 1 for size in component_sizes.values()),
        "cross_class_exact_duplicate_groups": exact_cross_label,
        "cross_class_near_duplicate_groups": [
            component_to_name[group]
            for group, labels in component_labels.items()
            if len(labels) > 1
        ],
        "image_dimensions_and_modes": {
            "dimension_counts": dict(Counter(
                f"{row['width']}x{row['height']}" for row in records if row["usable"]
            )),
            "mode_counts": dict(Counter(row["mode"] for row in records if row["usable"])),
        },
        "notes": [
            "Near duplicates are grouped by perceptual-hash distance for splitting; review cross-class groups.",
            "Grayscale files are flagged for review but are not automatically excluded.",
            "Only blank or unreadable images are excluded automatically.",
        ],
    }
    return summary, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/lung_subset_small"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--near-duplicate-distance", type=int, default=4)
    args = parser.parse_args()
    summary, records = audit_dataset(args.data_dir, args.near_duplicate_distance)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "data_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if records:
        with (args.reports_dir / "image_inventory.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps(summary, indent=2))
    if summary["cross_class_exact_duplicate_groups"]:
        raise SystemExit("Exact byte-identical images have conflicting labels. Resolve these labels before splitting.")


if __name__ == "__main__":
    main()
