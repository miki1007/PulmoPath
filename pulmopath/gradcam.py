"""Grad-CAM visualization helper; maps are not biological explanations."""

from __future__ import annotations

import numpy as np


def gradcam_heatmap(model, image_array: np.ndarray, class_index: int, layer_name: str = "last_conv_features") -> np.ndarray:
    import tensorflow as tf

    feature_layer = model.get_layer(layer_name)
    grad_model = tf.keras.Model(model.inputs, [feature_layer.output, model.output])
    batch = tf.convert_to_tensor(image_array[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        feature_maps, predictions = grad_model(batch, training=False)
        score = predictions[:, class_index]
    gradients = tape.gradient(score, feature_maps)
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(weights * feature_maps, axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, tf.zeros_like(heatmap))
    heatmap = tf.image.resize(heatmap[..., None], image_array.shape[:2], method="bilinear")[..., 0]
    return heatmap.numpy()


def overlay_heatmap(image_array: np.ndarray, heatmap: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    import matplotlib

    image = np.clip(image_array, 0, 255).astype(np.uint8)
    color = (matplotlib.colormaps["jet"](np.clip(heatmap, 0, 1))[..., :3] * 255).astype(np.uint8)
    return np.clip((1 - alpha) * image + alpha * color, 0, 255).astype(np.uint8)
