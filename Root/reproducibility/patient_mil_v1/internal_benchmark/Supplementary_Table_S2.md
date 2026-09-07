# Supplementary Table S2. Reconciliation of selection-stage and repeated internal-benchmark AUROC

## A. Fold-level inner-validation AUROC

Each value is the mean AUROC across the four inner-validation folds within the indicated outer-development cohort.

| Outer-development cohort | Selection-stage pretrained ResNet-18 | Repeated pretrained ResNet-18 benchmark | Randomly initialized ResNet-18 | Simple CNN benchmark |
|---:|---:|---:|---:|---:|
| 1 | 0.902 | 0.826 | 0.828 | 0.778 |
| 2 | 0.898 | 0.893 | 0.802 | 0.777 |
| 3 | 0.900 | 0.829 | 0.784 | 0.787 |
| 4 | 0.941 | 0.910 | 0.841 | 0.752 |
| 5 | 0.910 | 0.888 | 0.833 | 0.765 |
| Mean ± sample SD | 0.910 ± 0.018 | 0.869 ± 0.039 | 0.818 ± 0.024 | 0.772 ± 0.013 |

## B. Fold-specific configurations and random seeds

| Cohort | Learning rate | Scheduler | Input | Pretrained fine-tuning | Loss | Inner-split seed | Inner-training seeds |
|---:|---:|---|---:|---|---|---:|---|
| 1 | 10⁻³ | Step decay | 320 px | Layer3 + Layer4 + FC | Weighted cross-entropy | 1042 | 10142, 10242, 10342, 10442 |
| 2 | 10⁻³ | None | 320 px | Layer3 + Layer4 + FC | Weighted cross-entropy | 2042 | 20142, 20242, 20342, 20442 |
| 3 | 10⁻³ | Cosine annealing | 224 px | Layer3 + Layer4 + FC | Weighted cross-entropy | 3042 | 30142, 30242, 30342, 30442 |
| 4 | 10⁻³ | Step decay | 224 px | Layer4 + FC | Weighted focal loss | 4042 | 40142, 40242, 40342, 40442 |
| 5 | 10⁻⁴ | None | 320 px | Full network | Weighted cross-entropy | 5042 | 50142, 50242, 50342, 50442 |

The selection-stage values came from the original 144-configuration search. The repeated pretrained values came from fitting the selected configurations again during a separate development-only benchmark. The same patient partitions and nominal random-seed schedule were reused. Strict deterministic PyTorch algorithms were not enforced for Apple MPS, so separate executions were not guaranteed to be bitwise identical. No outer-test predictions were used in this comparison.
