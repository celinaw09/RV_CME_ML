# CME Classification from OCT Imaging

Deep learning framework for patient-level classification of Cystoid Macular Edema (CME) versus Non-CME from retinal OCT images.

---

## Overview

This repository implements a binary image classification pipeline designed to evaluate CME detection under a strict patient-level cross-validation protocol.

The primary goal is to estimate real-world generalization performance while preventing information leakage between training and evaluation datasets.

Key characteristics of the framework include:

- Patient-level stratified cross-validation
- Validation-driven threshold optimization
- Independent held-out test evaluation
- Fold-specific checkpointing
- Complete experiment logging
- Reproducible train/validation/test splits
- Aggregate performance analysis across folds

---

## Methodology

### Cross-Validation Strategy

Evaluation is performed using 5-fold patient-level stratified cross-validation.

For each fold:

1. Patients are split into training, validation, and test partitions.
2. Model weights are optimized using only the training set.
3. Classification threshold is selected using validation data.
4. The selected threshold is frozen.
5. Final performance is computed on the held-out test set.

Because splitting occurs at the patient level, images from the same patient never appear in multiple partitions.

This prevents train-test leakage and provides a more realistic estimate of deployment performance.

---

### Threshold Selection

Threshold optimization is performed exclusively on validation predictions.

The chosen threshold is then applied unchanged to the corresponding test set.

No test data are used during threshold selection.

---

## Repository Structure

text . ├── main.py ├── utils/ │   ├── train.py │   └── misc_utils.py │ ├── checkpoint_dir/ │   └── best_model.pth │ └── experiment_logs/     └── run_20260621_083308/         ├── console_output.txt         ├── cv_summary.json         ├── fold_summary.csv         ├── cross_validation_results.csv         ├── cv_confusion_matrix.png         ├── cv_roc_curve.pdf         ├── cv_pr_curve.pdf         │         ├── fold_1/         ├── fold_2/         ├── fold_3/         ├── fold_4/         └── fold_5/ 

---

## Experimental Run

Run ID: run_20260621_083308

Date: 2026-06-21

### Aggregate Test Performance

| Metric | Mean ± Std |
|----------|----------:|
| Accuracy | 73.05 ± 4.08% |
| AUROC | 77.99 ± 7.35% |
| AUPRC | 76.01 ± 7.85% |
| Sensitivity | 55.96 ± 21.51% |
| Specificity | 84.45 ± 10.15% |
| F1 Score | 60.95 ± 13.94% |

---

## Confusion Matrix

<p align="center">
  <img src="experiment_logs/run_20260621_083308/cv_confusion_matrix.png" width="700">
</p>

Aggregate confusion matrix computed from predictions across all held-out test folds.

---

## Results Summary

The model demonstrates:

- Good discriminative performance (AUROC ≈ 0.78)
- Strong precision-recall behavior (AUPRC ≈ 0.76)
- High specificity (84.45%)
- Moderate sensitivity (55.96%)

The current operating point prioritizes reduction of false-positive CME predictions while maintaining clinically useful detection performance.

---

## Reproducibility

Every experiment stores:

- Exact train splits
- Exact validation splits
- Exact test splits
- Fold-specific metrics
- Aggregate metrics
- Best model checkpoints
- Training logs
- Prediction outputs
- Evaluation visualizations

This enables complete reproduction of all reported results.

---

## Logged Artifacts

| Artifact | Description |
|-----------|-------------|
| cv_summary.json | Aggregate cross-validation metrics |
| fold_summary.csv | Fold-level performance summary |
| cross_validation_results.csv | Test predictions across all folds |
| console_output.txt | Full training and evaluation logs |
| cv_confusion_matrix.png | Aggregate confusion matrix |
| cv_roc_curve.pdf | Aggregate ROC curve |
| cv_pr_curve.pdf | Aggregate Precision-Recall curve |
| fold_*/best_model.pth | Best checkpoint for each fold |

---

## Citation

If this repository contributes to your research, please cite the associated publication when available.