"""Dataset index, patient-grouped splits, and the tf.data input pipeline."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold

from .config import TASKS, Config

IMAGE_SUFFIXES = {".jpeg", ".jpg"}


def find_data_root(path) -> Path:
    """Locate the folder that holds train/ val/ test/ (the Kaggle zip nests it twice)."""
    path = Path(path)
    candidates = [path] + [p.parent for p in path.rglob("train") if p.is_dir()]
    good = [
        c for c in candidates
        if "__MACOSX" not in c.parts
        and (c / "train" / "NORMAL").is_dir()
        and (c / "test" / "PNEUMONIA").is_dir()
    ]
    if not good:
        raise FileNotFoundError(f"No train/NORMAL and test/PNEUMONIA folders under {path}")
    return min(good, key=lambda c: len(c.parts))


def build_index(data_root) -> pd.DataFrame:
    """One row per image: path, original split, 3-class label, patient id."""
    root = find_data_root(data_root)
    rows = []
    for split in ("train", "val", "test"):
        pool = "test" if split == "test" else "dev"
        for folder in ("NORMAL", "PNEUMONIA"):
            for f in sorted((root / split / folder).iterdir()):
                if f.suffix.lower() not in IMAGE_SUFFIXES or f.name.startswith("."):
                    continue
                name = f.name.lower()
                if folder == "NORMAL":
                    label = "normal"
                    # IM-0115-0001.jpeg -> IM-0115 (the trailing number is the image index)
                    patient = re.sub(r"-\d+$", "", f.stem)
                else:
                    label = "bacteria" if "bacteria" in name else "virus" if "virus" in name else None
                    if label is None:
                        continue
                    m = re.match(r"(person\d+)_", name)
                    patient = m.group(1) if m else f.stem
                rows.append({"path": str(f), "split": split, "label": label,
                             "patient": f"{pool}_{patient}"})
    return pd.DataFrame(rows)


def make_splits(df: pd.DataFrame, cfg: Config):
    """Merge train+val and re-split by patient. The official test set stays untouched."""
    dev = df[df["split"] != "test"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    folds = StratifiedGroupKFold(n_splits=round(1 / cfg.val_fraction), shuffle=True,
                                 random_state=cfg.seed)
    train_idx, val_idx = next(folds.split(dev, dev["label"], groups=dev["patient"]))
    train, val = dev.iloc[train_idx].reset_index(drop=True), dev.iloc[val_idx].reset_index(drop=True)
    assert set(train["patient"]).isdisjoint(val["patient"]), "patient leakage between train and val"
    return train, val, test


def task_frame(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Rows and integer labels (column y) for one task."""
    classes = TASKS[task]["classes"]
    out = df.copy()
    if task == "stage1":
        out["y"] = (out["label"] != "normal").astype(int)
    else:
        out = out[out["label"].isin(classes)].copy()
        out["y"] = out["label"].map(classes.index).astype(int)
    return out.reset_index(drop=True)


def class_weights(y) -> dict:
    counts = np.bincount(y)
    return {i: len(y) / (len(counts) * c) for i, c in enumerate(counts)}


def make_dataset(frame: pd.DataFrame, task: str, cfg: Config, training: bool) -> tf.data.Dataset:
    """Images come out as float32 in 0..255. Augmentation and preprocessing live inside the model."""
    binary = len(TASKS[task]["classes"]) == 2
    labels = frame["y"].to_numpy().astype("float32" if binary else "int32")

    def load(path, y):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, (cfg.img_size, cfg.img_size))
        return tf.cast(tf.round(img), tf.uint8), y

    ds = tf.data.Dataset.from_tensor_slices((frame["path"].to_numpy(), labels))
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if cfg.cache:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(len(frame), seed=cfg.seed, reshuffle_each_iteration=True)
    ds = ds.batch(cfg.batch_size)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
