"""CNN baselines and transfer-learning model definitions."""

from __future__ import annotations


def build_baseline(image_size: int = 128, num_classes: int = 3):
    import tensorflow as tf

    layers = tf.keras.layers
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="rgb_pixels_0_255")
    x = layers.Rescaling(1.0 / 255.0, name="scale_0_1")(inputs)
    for filters, block in ((32, 1), (64, 2), (128, 3)):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{block}")(x)
        # Group normalization uses per-example statistics at train and inference
        # time. This avoids the batch-statistic mismatch observed in the first
        # CPU run of the 128x128 baseline on this small dataset.
        x = layers.GroupNormalization(groups=8, axis=-1, name=f"gn{block}")(x)
        x = layers.Activation("relu", name=f"relu{block}")(x)
        if block == 3:
            x = layers.Activation("linear", name="last_conv_features")(x)
        x = layers.MaxPooling2D(2, name=f"pool{block}")(x)
        x = layers.Dropout(0.15 + 0.05 * block, name=f"dropout{block}")(x)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.LayerNormalization(name="dense_norm")(x)
    x = layers.Dropout(0.35, name="dense_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    return tf.keras.Model(inputs, outputs, name="pulmopath_baseline_cnn"), "last_conv_features"


def build_mobilenetv2(image_size: int = 128, num_classes: int = 3, weights: str | None = "imagenet"):
    import tensorflow as tf

    layers = tf.keras.layers
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="rgb_pixels_0_255")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(inputs)
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=weights,
        input_shape=(image_size, image_size, 3),
        name="mobilenetv2_backbone",
    )
    # Freeze an ImageNet backbone for the initial head-training stage. When
    # ImageNet weights are unavailable, train the randomly initialized backbone.
    # ImageNet initialization starts with a frozen feature extractor. A model
    # without pretrained weights is trained end-to-end from random initialization.
    backbone.trainable = weights is None
    x = backbone(x, training=False)
    x = layers.Activation("linear", name="last_conv_features")(x)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(0.25, name="feature_dropout")(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.35, name="dense_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    return tf.keras.Model(inputs, outputs, name="pulmopath_mobilenetv2"), "last_conv_features"


def build_model(model_name: str, image_size: int = 128, num_classes: int = 3, weights: str | None = "imagenet"):
    if model_name == "baseline":
        return build_baseline(image_size, num_classes)
    if model_name == "mobilenetv2":
        return build_mobilenetv2(image_size, num_classes, weights)
    raise ValueError(f"Unsupported model {model_name!r}; choose baseline or mobilenetv2.")
