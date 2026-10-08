"""Two-phase training: new head first, then fine-tune the top of the backbone.

    python -m src.train --task stage1 --data-root /path/to/chest_xray --out outputs
"""
import argparse
import json
from pathlib import Path

import keras
import tensorflow as tf

from .config import TASKS, Config
from .data import build_index, class_weights, make_dataset, make_splits, task_frame
from .model import build_model, set_trainable


def _compile(model, lr, binary):
    if binary:
        loss, metrics = "binary_crossentropy", [keras.metrics.BinaryAccuracy(name="acc"),
                                                keras.metrics.AUC(name="auc")]
    else:
        loss, metrics = "sparse_categorical_crossentropy", [
            keras.metrics.SparseCategoricalAccuracy(name="acc")]
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss, metrics=metrics)


def train_task(task: str, data_root, out_dir="outputs", cfg: Config | None = None):
    cfg = cfg or Config()
    keras.utils.set_random_seed(cfg.seed)
    if cfg.mixed_precision and tf.config.list_physical_devices("GPU"):
        keras.mixed_precision.set_global_policy("mixed_float16")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = TASKS[task]["classes"]
    binary = len(classes) == 2

    train_df, val_df, _ = make_splits(build_index(data_root), cfg)
    train_f, val_f = task_frame(train_df, task), task_frame(val_df, task)
    train_ds = make_dataset(train_f, task, cfg, training=True)
    val_ds = make_dataset(val_f, task, cfg, training=False)
    weights = class_weights(train_f["y"].to_numpy())
    print(f"[{task}] train={len(train_f)} val={len(val_f)} class weights={weights}")

    model, backbone_names = build_model(TASKS[task]["backbone"], len(classes), cfg.img_size,
                                        cfg.weights, cfg.dropout, cfg.seed)
    monitor, mode = ("val_auc", "max") if binary else ("val_loss", "min")
    ckpt = out_dir / f"{task}.keras"
    history = {}

    phases = [("head", cfg.head_epochs, cfg.head_lr, 0),
              ("finetune", cfg.finetune_epochs, cfg.finetune_lr, cfg.unfreeze_layers)]
    for name, epochs, lr, unfreeze in phases:
        if epochs <= 0:
            continue
        set_trainable(model, backbone_names, unfreeze)
        _compile(model, lr, binary)
        callbacks = [
            keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=cfg.patience,
                                          restore_best_weights=True),
        ]
        h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=weights,
                      callbacks=callbacks, verbose=2)
        history[name] = {k: [float(v) for v in vals] for k, vals in h.history.items()}
        model.save(ckpt)  # best weights of the phase are restored before saving

    (out_dir / f"{task}_history.json").write_text(json.dumps(history, indent=2))
    print(f"[{task}] saved {ckpt}")
    return model, history


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASKS), required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", default="outputs")
    a = p.parse_args()
    train_task(a.task, a.data_root, a.out)
