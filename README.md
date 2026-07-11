# Lung Cancer Prediction using CNN and Transfer Learning

Classifies chest CT scan images into four classes — **adenocarcinoma**, **large cell carcinoma**, **squamous cell carcinoma**, and **normal** — by fine-tuning a ResNet50 pretrained on ImageNet.

## Results (held-out test set, 315 scans)

| Metric | Score |
|---|---|
| Test Accuracy | **81.3%** |
| Weighted AUC-ROC (one-vs-rest) | **0.946** |
| Cohen's Kappa | 0.74 |
| Matthews Correlation Coefficient | 0.75 |
| Normal vs. malignant precision/recall | **1.00 / 1.00** |

Per-class specificity ranges from 0.83 to 1.00. The model never misclassified a malignant scan as normal (or vice versa) on the test set.

## Approach

- **Backbone:** ResNet50 (ImageNet weights), all layers frozen except `layer4`
- **Head:** Linear(2048→512) → ReLU → Dropout(0.5) → Linear(512→4)
- **Discriminative learning rates:** 1e-5 for `layer4`, 1e-4 for the new head (Adam)
- **Class-weighted cross-entropy loss** to counter class imbalance
- **`ReduceLROnPlateau` scheduler** on validation loss
- **Augmentation:** rotation, horizontal flip, small affine translate/scale, brightness/contrast jitter
- **Model selection:** checkpoint saved at lowest validation loss (early-stopping style)
- **Evaluation:** accuracy, per-class precision/recall/F1/specificity, confusion matrix, Cohen's Kappa, MCC, one-vs-rest ROC curves

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the [Chest CT-Scan images dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) from Kaggle and place it as:

```
dataset/
├── train/
├── valid/
└── test/
```

Then run `Lung Cancer Pred.ipynb` top to bottom (seeded for reproducibility).

## Disclaimer

Educational project — not a medical device and not validated for clinical use.
