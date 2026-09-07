# Final patient-level MIL reproducibility package

This directory contains the public audit package for the final 117-patient analysis.

## Primary outer-test records

`outer_test_predictions_deidentified.csv` contains exactly one row per patient with these fields:

- `patient_id`: study-specific identifier (`P001`–`P117`)
- `outer_fold`: untouched outer-test fold
- `true_label`: patient-level reference label
- `prob_CME`: patient-level probability after max-pooling image-level CME log-odds
- `threshold`: threshold selected only from the corresponding outer-development final-validation subset
- `pred_label`: `1` when `prob_CME >= threshold`, otherwise `0`

The row order is unchanged from the finalized analysis because the saved bootstrap intervals depend on the recorded pseudorandom resampling sequence.

## Metrics

`metrics/recalculated_pooled_metrics.json` and `metrics/recalculated_patient_bootstrap_95ci.csv` were generated from the de-identified records by `src/recalculate_patient_metrics.py`. `metrics/outer_fold_metrics.csv` contains the five fold-specific outer-test evaluations and selected inner-validation AUROCs.

## Model selection

`model_selection/Supplementary_Table_S1.xlsx` contains all 720 candidate–outer-development-cohort records. Its five selected rows reproduce the configurations and inner-validation AUROCs reported in manuscript Table 4. Its marginal checks reproduce manuscript Table 6. `model_selection/all_inner_fold_metrics.csv` provides the underlying 2,880 inner-fold metric records.

## Internal benchmark

`internal_benchmark/all_inner_benchmark_summaries.csv` contains the 15 cohort-by-model results from the separate development-only benchmark: five repeated ImageNet-pretrained ResNet-18 fits, five randomly initialized ResNet-18 fits, and five simple CNN fits.

The selection-stage pretrained AUROCs were 0.902, 0.898, 0.900, 0.941, and 0.910 (mean ± sample SD, 0.910 ± 0.018). The separately repeated pretrained benchmark values were 0.826, 0.893, 0.829, 0.910, and 0.888 (0.869 ± 0.039). The difference reflects separate training executions. The same partitions and nominal seeds were reused, but strict deterministic PyTorch algorithms were not enforced on Apple MPS.

## Configuration and seeds

The complete declared configuration is `../../configs/patient_mil_primary.json`.

- Outer partition seed: 42
- Inner split seed: `42 + 1000 * outer_fold`
- Inner training seed: `42 + 10000 * outer_fold + 100 * inner_fold`
- Final-validation split seed: `42 + 1000 * outer_fold + 77`
- Final fit seed: `42 + 100000 * outer_fold + 999`
- Bootstrap seed: 951

## Checkpoints

`checkpoints/` contains the fold-specific final models. Each checkpoint applies only to its corresponding outer fold and must be paired with the configuration and validation-derived threshold recorded in `configs/patient_mil_primary.json`.

## Data availability

The clinical FA images and original patient identifiers are not distributed here. Access remains subject to the study's governance and ethics approvals.
