"""Image loading and training-only augmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .constants import IMAGE_SIZE


def load_rgb_array(path: str | Path, size: int = IMAGE_SIZE, resample=Image.Resampling.BICUBIC) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size), resample=resample)
        return np.asarray(image, dtype=np.float32)


def make_dataset(
    frame: pd.DataFrame,
    split: str,
    batch_size: int,
    size: int = IMAGE_SIZE,
    augment: bool = False,
    seed: int = 42,
):
    """Build a deterministic tf.data dataset. Images remain in [0,255] until model preprocessing."""
    import tensorflow as tf

    subset = frame.loc[frame["split"] == split].copy()
    if subset.empty:
        raise ValueError(f"No rows found for split={split!r} in split manifest.")
    paths = subset["absolute_path"].astype(str).tolist() if "absolute_path" in subset else None
    if paths is None:
        raise ValueError("Split manifest needs absolute_path. Re-run split preparation from the project root.")
    labels = subset["label_id"].to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if split == "train":
        ds = ds.shuffle(len(subset), seed=seed, reshuffle_each_iteration=True)

    def decode(path, label):
        def _read(p):
            p = p.decode("utf-8")
            return load_rgb_array(p, size=size)
        image = tf.numpy_function(_read, [path], Tout=tf.float32)
        image.set_shape([size, size, 3])
        return image, label

    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    if augment:
        rotation = tf.keras.layers.RandomRotation(
            factor=0.04, fill_mode="reflect", seed=seed
        )
        def _augment(image, label):
            image = rotation(image, training=True)
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.random_brightness(image, max_delta=12.0)
            image = tf.image.random_contrast(image, lower=0.90, upper=1.10)
            return tf.clip_by_value(image, 0.0, 255.0), label
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def attach_absolute_paths(frame: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    output = frame.copy()
    output["absolute_path"] = output["relative_path"].map(lambda p: str((data_root / p).resolve()))
    missing = [p for p in output["absolute_path"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"Image referenced by split manifest is missing: {missing[0]}")
    return output
