# Patient-level CME classification from fluorescein angiography

This directory contains the code and reviewer-facing reproducibility artifacts for the final 117-patient analysis of concurrent cystoid macular edema (CME) in retinal vasculitis.

The primary endpoint is patient-level CME. Every available eye image is encoded by a shared ImageNet-pretrained ResNet-18. The replacement head consists of dropout (`p=0.5`) and a two-logit linear layer. For each eye, CME log-odds are calculated as the positive logit minus the negative logit. The patient's score is the maximum image-level CME log-odds across the available eyes, so training and evaluation produce one label, loss, probability, and prediction per patient.

## Final analysis

- 207 late-phase FA images from 117 patients
- 49 CME-positive and 68 CME-negative patients
- Five-fold outer and four-fold inner patient-level nested cross-validation
- 144 candidate configurations per outer-development cohort
- Configuration selection by mean inner-validation AUROC
- Fold-specific thresholds selected by Youden's J on final-validation data
- One evaluation on each untouched outer-test fold
- Patient-level percentile bootstrap confidence intervals from 5,000 resamples (seed 951)

Pooled untouched outer-test performance:

| Metric | Estimate | 95% CI |
|---|---:|---:|
| AUROC | 0.708 | 0.610–0.802 |
| AUPRC | 0.626 | 0.493–0.755 |
| Accuracy | 0.726 | 0.641–0.803 |
| Sensitivity | 0.694 | 0.558–0.822 |
| Specificity | 0.750 | 0.643–0.853 |
| Precision | 0.667 | 0.532–0.796 |
| F1 | 0.680 | 0.563–0.780 |

The pooled confusion matrix is TN=51, FP=17, FN=15, TP=34.

## Repository layout

```text
Root/
├── configs/
│   └── patient_mil_primary.json
├── reproducibility/patient_mil_v1/
│   ├── outer_test_predictions_deidentified.csv
│   ├── metrics/
│   ├── model_selection/
│   ├── internal_benchmark/
│   └── checkpoints/
├── scripts/
│   └── run_patient_mil_primary.sh
├── src/
│   ├── nested_cv_pipeline.py
│   ├── recalculate_patient_metrics.py
│   └── tests/test_nested_cv_pipeline.py
└── requirements.txt
```

Files outside the paths listed above are legacy materials from earlier repository versions and must not be used to reproduce or report the final 117-patient analysis.

## Environment

The finalized run used Python 3.11.8, PyTorch 2.3.0, torchvision 0.18.0, scikit-learn 1.4.0, NumPy 1.26.2, pandas 2.2.0, Pillow 10.1.0, and macOS 15.6.1 arm64. Training used Apple Metal Performance Shaders in one sequential process; CUDA and cuDNN were not available. The exact Mac chip and RAM were not recorded.

Recreate the Conda environment with:

```bash
conda env create -f environment.yml
conda activate rv-cme-patient-mil
```

Alternatively, install the Python dependencies into an existing Python 3.11.8 environment with:

```bash
python -m pip install -r requirements.txt
```

## Recalculate the published patient-level metrics

From `Root/`:

```bash
python src/recalculate_patient_metrics.py \
  --predictions reproducibility/patient_mil_v1/outer_test_predictions_deidentified.csv \
  --output-dir reproducibility/patient_mil_v1/recalculated_metrics
```

This calculation does not require the clinical image dataset. It reproduces the eight point estimates, the confusion matrix, and the saved 5,000-resample confidence intervals.

## Re-run nested cross-validation

The clinical images are not distributed in this repository. Arrange authorized data as:

```text
allpatients_resized/
├── CME/
└── non_CME/
```

Then run:

```bash
bash scripts/run_patient_mil_primary.sh /absolute/path/to/allpatients_resized
```

The complete run is computationally expensive because it evaluates 720 candidate–outer-cohort combinations, each by four inner folds, before final fitting and the development-only internal benchmarks.

## Privacy

The released outer-test table contains study-specific identifiers `P001`–`P117` only. Raw images, original filenames, identifier mappings, and patient-name-bearing split or XAI logs are not part of the final reproducibility package.

## Versioning

The reviewer response should cite the immutable commit created after these files are committed and pushed. A GitHub release tag such as `patient-mil-v1.0.0` may also be attached to that commit.
