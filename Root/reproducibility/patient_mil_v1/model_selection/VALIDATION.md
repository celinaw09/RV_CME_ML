# Supplementary Table S1 validation

Source: `all_inner_fold_metrics.csv` plus each outer cohort's saved configuration ranking from the finalized `RV_CME_PATIENT_MIL_PRIMARY_MPS` run.

- PASS — 720 candidate–outer-development-cohort records
- PASS — 144 candidate configurations in each of five outer cohorts
- PASS — Four unique inner-fold AUROCs for every candidate
- PASS — One rank-1 selected configuration per outer cohort
- PASS — Selected mean inner-validation AUROCs: 0.901551, 0.898295, 0.899691, 0.941012, and 0.910056
- PASS — Selected values summarize to 0.910 ± 0.018 using sample SD across the five cohorts
- PASS — Selected configurations and values reproduce manuscript Table 4
- PASS — Marginal candidate–outer-cohort summaries reproduce manuscript Table 6 after rounding to three decimals
- PASS — No outer-test predictions are included in S1 or used for model selection

The workbook's `Selected Check` and `Marginal Check` worksheets retain the numerical reconciliation alongside the 720-record `S1 All Candidates` worksheet.
