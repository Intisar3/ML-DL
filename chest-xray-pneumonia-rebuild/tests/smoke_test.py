"""End-to-end check on tiny synthetic images. Runs on CPU in about a minute.

    python -m tests.smoke_test
"""
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config import TASKS, Config
from src.data import build_index, make_dataset, make_splits, task_frame
from src.evaluate import evaluate_all
from src.gradcam import gradcam
from src.train import train_task


def fake_dataset(root: Path, rng):
    def write(path, bright):
        img = rng.integers(0, 120, (64, 64, 1), dtype=np.uint8) + bright
        path.parent.mkdir(parents=True, exist_ok=True)
        tf.io.write_file(str(path), tf.io.encode_jpeg(img))

    for split, n in (("train", 24), ("val", 2), ("test", 8)):
        for i in range(n):
            write(root / split / "NORMAL" / f"IM-{i:04d}-0001.jpeg", 0)
            write(root / split / "NORMAL" / f"NORMAL2-IM-{i:04d}-0001.jpeg", 0)
            for k, kind in enumerate(("bacteria", "virus")):
                for j in range(2):  # two images per patient, to exercise the grouping
                    write(root / split / "PNEUMONIA" / f"person{i}_{kind}_{i * 10 + j}.jpeg", 60 + 60 * k)


def main():
    cfg = Config(img_size=96, batch_size=8, weights=None, head_epochs=1, finetune_epochs=1,
                 unfreeze_layers=10, mixed_precision=False)
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp) / "nested" / "chest_xray", Path(tmp) / "out"
        fake_dataset(root, np.random.default_rng(0))

        df = build_index(tmp)
        train, val, test = make_splits(df, cfg)
        assert len(df) == 34 * 6 and len(test) == 8 * 6
        assert set(train["patient"]).isdisjoint(val["patient"])
        assert set(df["label"]) == {"normal", "bacteria", "virus"}

        for task in TASKS:
            train_task(task, tmp, out, cfg)
        results = evaluate_all(tmp, out, cfg)
        assert {"stage1", "stage2", "two_stage_pipeline", "three_class_baseline"} <= set(results)
        assert (out / "test_predictions.csv").exists()

        import keras
        model = keras.models.load_model(out / "stage1.keras")
        x, _ = next(iter(make_dataset(task_frame(test, "stage1"), "stage1", cfg, training=False)))
        cam = gradcam(model, x[:2], "resnet50")
        assert cam.shape[0] == 2 and 0 <= cam.min() and cam.max() <= 1 + 1e-6
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
