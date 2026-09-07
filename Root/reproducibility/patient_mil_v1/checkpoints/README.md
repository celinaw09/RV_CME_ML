# Fold-specific checkpoints

These are the final ResNet-18 multiple-instance model checkpoints used for the five untouched outer-test evaluations. They are not one globally trained deployment model.

| Outer fold | Checkpoint | Input | Fine-tuning | Threshold |
|---:|---|---:|---|---:|
| 1 | `outer_1_best_model.pth` | 320 px | Layer3 + Layer4 + FC | 0.3760009706020355 |
| 2 | `outer_2_best_model.pth` | 320 px | Layer3 + Layer4 + FC | 0.7818639874458313 |
| 3 | `outer_3_best_model.pth` | 224 px | Layer3 + Layer4 + FC | 0.31667959690093994 |
| 4 | `outer_4_best_model.pth` | 224 px | Layer4 + FC | 0.8076737523078918 |
| 5 | `outer_5_best_model.pth` | 320 px | Full network | 0.32119518518447876 |

SHA-256 checksums are provided in `SHA256SUMS`.
