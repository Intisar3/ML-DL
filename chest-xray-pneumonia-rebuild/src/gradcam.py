"""Grad-CAM: shows which lung regions drove a prediction."""
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .model import BACKBONES


def gradcam(model, images, backbone: str, class_index: int | None = None) -> np.ndarray:
    """images: float array (n, H, W, 3) in 0..255. Returns heatmaps (n, h, w) in 0..1."""
    last_conv = BACKBONES[backbone][2]
    grad_model = keras.Model(model.input, [model.get_layer(last_conv).output, model.output])
    with tf.GradientTape() as tape:
        conv, preds = grad_model(tf.convert_to_tensor(images, tf.float32), training=False)
        score = preds[:, 0] if preds.shape[-1] == 1 else preds[:, class_index or 0]
    grads = tf.cast(tape.gradient(score, conv), tf.float32)
    conv = tf.cast(conv, tf.float32)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(conv * weights, axis=-1))
    cam /= tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-8
    return cam.numpy()


def show_gradcam(model, images, backbone, titles=None, class_index=None, path=None):
    cams = gradcam(model, images, backbone, class_index)
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False)
    for i, ax in enumerate(axes[0]):
        img = np.asarray(images[i]).astype("uint8")
        heat = tf.image.resize(cams[i][..., None], img.shape[:2]).numpy()[..., 0]
        ax.imshow(img)
        ax.imshow(heat, cmap="jet", alpha=0.35)
        ax.set_title(titles[i] if titles else "", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
