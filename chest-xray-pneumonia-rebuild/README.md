# Chest X-ray Pneumonia Classification (rebuild)

A rebuild of my deep learning capstone from the Makeen (مكين) bootcamp, Data Science and AI stream. The original notebook is [`../chest-xray-pneumonia.ipynb`](../chest-xray-pneumonia.ipynb) and stays in the repo as the "before" version.

**Purpose:** this is a practical, educational deep learning project. Its goal is to practice transfer learning, honest evaluation, and reproducible project structure. Its outputs are learning material, and medical decisions belong to clinicians.

## The task

Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), 5,856 pediatric chest X-rays (ages 1 to 5, one hospital in Guangzhou).

| Model | Backbone | Classes |
|---|---|---|
| Stage 1 | ResNet50 | normal vs pneumonia |
| Stage 2 | InceptionV3 | bacterial vs viral |
| Baseline | ResNet50 | normal, bacterial, viral in one model |

The baseline answers a design question: do two stages perform better than one simple 3-class model?

## What changed from the original, and why

| Original | Rebuild | The lesson |
|---|---|---|
| `rescale=1/255` on frozen ImageNet backbones | Each backbone's own preprocessing, built into the model | Pretrained weights expect the exact input format they were trained with. The original stayed at the majority-class baseline (about 75%) for this reason. |
| Validation on the 16 official val images | Train and val merged, then re-split by patient (about 12.5% for validation) | Early stopping needs a validation set large enough to give a stable signal. |
| Random image-level folders | Splits grouped by patient ID from the filename | Images of one patient in both train and val inflate validation scores. |
| Stage 2 trained and predicted on the test images | Stage 2 trains on training-set pneumonia images; the test set is used once, for the final numbers | A model's score on data it trained on says little about new data. |
| 360° rotation, shear, heavy zoom | About 10° rotation, 5% shift, 10% zoom, small contrast changes | Augmentation should produce images that could really occur. Chest X-rays are always upright. |
| Frozen backbone only | Head training, then fine-tuning the top backbone layers at a low learning rate (BatchNorm frozen) | Fine-tuning adapts ImageNet features to X-rays. |
| No class weighting | Class weights from training frequencies | The classes are imbalanced (about 74% pneumonia). |
| No test metrics | Sensitivity, specificity, AUROC, PR-AUC, confusion matrices, threshold chosen on validation | Accuracy alone hides majority-class behavior. |
| Hardcoded drug recommendations | Predictions CSV with probabilities and an educational note | A classifier output is one input to a clinician's decision. |
| Colab-only cells, no seeds | `src/` package, config file, fixed seeds, smoke test, requirements | Anyone can rerun the project and get comparable results. |
| 20 minutes per epoch | Cached `tf.data` pipeline and mixed precision on GPU | The input pipeline is often the real bottleneck. |

## Project layout

```
chest-xray-pneumonia-rebuild/
├── chest_xray_pneumonia_rebuild.ipynb   # run this in Colab
├── src/
│   ├── config.py     # every setting in one place
│   ├── data.py       # index, patient-grouped split, tf.data pipeline
│   ├── model.py      # backbones with built-in augmentation and preprocessing
│   ├── train.py      # two-phase training
│   ├── evaluate.py   # test metrics, confusion matrices, pipeline comparison
│   └── gradcam.py    # heatmaps of what the model looks at
├── tests/smoke_test.py   # whole pipeline on synthetic images, about a minute
└── requirements.txt
```

## How to run (Google Colab)

1. Push this folder to GitHub.
2. Open `chest_xray_pneumonia_rebuild.ipynb` in Colab and select a T4 GPU runtime.
3. Run all cells. The smoke test runs first, then the data download, training, evaluation, and Grad-CAM.

From a terminal:

```bash
pip install -r requirements.txt
python -m tests.smoke_test
python -m src.train --task stage1 --data-root /path/to/chest_xray
python -m src.train --task stage2 --data-root /path/to/chest_xray
python -m src.train --task three_class --data-root /path/to/chest_xray
python -m src.evaluate --data-root /path/to/chest_xray
```

## Results

Fill in from `outputs/metrics.json` after the Colab run.

| Model | Sensitivity | Specificity | AUROC | Accuracy | Macro F1 |
|---|---|---|---|---|---|
| Stage 1 (normal vs pneumonia) | | | | | n/a |
| Stage 2 (bacterial vs viral) | | | | | n/a |
| Two-stage pipeline, 3 classes | n/a | n/a | n/a | | |
| Single 3-class baseline | n/a | n/a | n/a | | |

Original notebook, for comparison: stage 1 training accuracy 75 to 77% (majority-class level), no test metrics recorded.

## Limits to keep in mind

- The data comes from children aged 1 to 5 at one hospital, so results describe that population.
- Bacterial vs viral is hard to separate on X-ray alone, even for radiologists. Expect stage 2 to score well below stage 1.
- Patient IDs for normal images are inferred from filenames, which is an approximation.
- The official test set has a different class balance from the training set (62% pneumonia vs 74%).
