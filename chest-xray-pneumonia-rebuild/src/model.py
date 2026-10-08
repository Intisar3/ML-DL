"""Transfer-learning models with the correct preprocessing built into the graph."""
import keras
from keras import layers, ops

# name -> (constructor, preprocessing style, last conv layer for Grad-CAM)
BACKBONES = {
    "resnet50": (keras.applications.ResNet50, "caffe", "conv5_block3_out"),
    "inceptionv3": (keras.applications.InceptionV3, "tf", "mixed10"),
}


@keras.saving.register_keras_serializable(package="cxr")
class CaffePreprocess(layers.Layer):
    """What ResNet50's ImageNet weights expect: RGB to BGR, then subtract the channel means."""
    MEAN = (103.939, 116.779, 123.68)

    def call(self, x):
        x = ops.flip(x, axis=-1)
        return x - ops.convert_to_tensor(self.MEAN, dtype=x.dtype)


def build_model(backbone: str, n_classes: int, img_size: int, weights="imagenet", dropout=0.3,
                seed=42):
    """Returns (model, backbone_layer_names). Input: RGB images, float, range 0..255."""
    ctor, style, _ = BACKBONES[backbone]
    inputs = keras.Input((img_size, img_size, 3), name="image")

    # Augmentation runs only during training. Chest X-rays are always upright,
    # so the changes stay small and realistic (about 10 degrees, 5% shift, 10% zoom).
    x = layers.RandomRotation(10 / 360, fill_mode="constant", seed=seed, name="aug_rotate")(inputs)
    x = layers.RandomTranslation(0.05, 0.05, fill_mode="constant", seed=seed, name="aug_shift")(x)
    x = layers.RandomZoom(0.1, fill_mode="constant", seed=seed, name="aug_zoom")(x)
    x = layers.RandomContrast(0.1, seed=seed, name="aug_contrast")(x)

    if style == "caffe":
        x = CaffePreprocess(name="preprocess")(x)
    else:  # InceptionV3 expects -1..1
        x = layers.Rescaling(1 / 127.5, offset=-1.0, name="preprocess")(x)
    own = {"image", "aug_rotate", "aug_shift", "aug_zoom", "aug_contrast", "preprocess"}

    base = ctor(include_top=False, weights=weights, input_tensor=x)
    backbone_names = [l.name for l in base.layers if l.name not in own]

    x = layers.GlobalAveragePooling2D(name="pool")(base.output)
    x = layers.Dropout(dropout, name="dropout")(x)
    if n_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid", dtype="float32", name="prob")(x)
    else:
        outputs = layers.Dense(n_classes, activation="softmax", dtype="float32", name="probs")(x)
    return keras.Model(inputs, outputs, name=f"cxr_{backbone}"), backbone_names


def set_trainable(model, backbone_names, unfreeze_last: int = 0):
    """Freeze the backbone, then open its last N layers. BatchNorm stays frozen so its
    ImageNet statistics survive fine-tuning on a small dataset."""
    open_names = set(backbone_names[-unfreeze_last:]) if unfreeze_last else set()
    for name in backbone_names:
        layer = model.get_layer(name)
        layer.trainable = name in open_names and not isinstance(layer, layers.BatchNormalization)
