# Model comparison report

## Experimental setup

Both models used the same seed-42 duplicate-cluster split, 128 × 128 RGB inputs, training-only augmentation, and the same validation and test evaluation code. The test set was not examined during training or model selection. Validation macro-F1 was used as the primary model-selection criterion because each class is represented at similar support; validation loss and recorded CPU fit time were also considered. On that basis, MobileNetV2 was selected for the demo. Test metrics below compare the two previously trained checkpoints; they were not used to change the selection.

The scratch CNN used three convolutional blocks with GroupNormalization, ReLU, max-pooling, dropout, global average pooling, a normalized dense layer, and three-class softmax. GroupNormalization replaced BatchNormalization after the first run's saved model predicted only `lung_scc` for every input in inference mode. That failed checkpoint and its history are retained in `artifacts/diagnostics/baseline_batchnorm_failure/` and are not included as a valid model result.

MobileNetV2 used published ImageNet weights from Keras Applications, a frozen backbone, and a trained classification head. Fine-tuning was not run. ImageNet pretraining is generic initialization and provides no medical validation. The published weight file was downloaded before training; fit time below excludes model construction and the one-time weight download.

## Validation model selection

| Model | Selected epoch by validation loss | Validation loss | Validation accuracy | Validation macro-F1 | CPU fit time |
|---|---:|---:|---:|---:|---:|
| Scratch CNN | 33 | 0.14883 | 0.9335 | 0.9333 | 577.9 s |
| MobileNetV2 | 16 | 0.08923 | 0.9612 | 0.9611 | 145.2 s |

MobileNetV2 had the higher validation macro-F1 and lower validation loss, and its recorded fit time was shorter in this CPU-only run. The selected demo model is therefore MobileNetV2. See `reports/model_selection.json` and each checkpoint's metadata/history for the machine-readable selection evidence.

## Held-out test results

Each comparison checkpoint was evaluated once on the untouched 359-image test partition. Counts were 120 `lung_n`, 119 `lung_aca`, and 120 `lung_scc`.

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Scratch CNN | 0.9499 | 0.9506 | 0.9497 | 0.9497 | 0.9498 |
| MobileNetV2 (selected from validation) | 0.9471 | 0.9475 | 0.9469 | 0.9470 | 0.9470 |

### Per-class test metrics

| Model | Class | Precision | Recall | F1 | Support | ROC-AUC OVR | Average precision OVR |
|---|---|---:|---:|---:|---:|---:|---:|
| Scratch CNN | `lung_n` | 1.0000 | 1.0000 | 1.0000 | 120 | 1.0000 | 1.0000 |
| Scratch CNN | `lung_aca` | 0.9469 | 0.8992 | 0.9224 | 119 | 0.9819 | 0.9620 |
| Scratch CNN | `lung_scc` | 0.9048 | 0.9500 | 0.9268 | 120 | 0.9828 | 0.9687 |
| MobileNetV2 | `lung_n` | 1.0000 | 1.0000 | 1.0000 | 120 | 1.0000 | 1.0000 |
| MobileNetV2 | `lung_aca` | 0.9386 | 0.8992 | 0.9185 | 119 | 0.9913 | 0.9850 |
| MobileNetV2 | `lung_scc` | 0.9040 | 0.9417 | 0.9224 | 120 | 0.9915 | 0.9819 |

### Confusion matrices

Rows are true labels and columns are predicted labels in the order `lung_n`, `lung_aca`, `lung_scc`.

**Scratch CNN**

```text
[[120,   0,   0],
 [  0, 107,  12],
 [  0,   6, 114]]
```

**MobileNetV2**

```text
[[120,   0,   0],
 [  0, 107,  12],
 [  0,   7, 113]]
```

Both models classified the normal-label test patches correctly in this partition. The most frequent error for each model was `lung_aca` predicted as `lung_scc` (12 cases); the reverse confusion occurred in 6 scratch-CNN and 7 MobileNetV2 cases. These are dataset-label disagreements, not clinical false diagnoses.

## Uncertainty and interpretation

The evaluator used 1,000 percentile bootstrap draws over the 352 duplicate-cluster groups in the test set. The scratch CNN's 95% intervals were 0.9274–0.9697 for accuracy and 0.9278–0.9698 for macro-F1. MobileNetV2's were 0.9218–0.9668 and 0.9221–0.9670, respectively. These intervals overlap. Since the groups are duplicate clusters rather than verified patients or slides, the resampling does not provide patient-level uncertainty, and unknown within-patient dependence may make the intervals too narrow.

The one-image difference in test accuracy does not establish that the scratch CNN is better. Model selection remains based on validation only. The small subset, unverified possible LC25000 lineage, unknown creator of this custom subset, absent patient/slide/source-image identifiers, and lack of external evaluation prevent claims about patient-independent performance or generalization to other institutions, scanners, staining protocols, or populations. If the likely LC25000 naming lineage is correct, the upstream release includes augmented variants but the subset has no parent-image mapping; source-image leakage may therefore remain beyond the detected duplicate clusters. Softmax scores are not calibrated disease probabilities. Reuse rights also remain unverified.

## Generated artifacts

- `reports/evaluation/baseline/metrics.json` and `reports/evaluation/mobilenetv2/metrics.json` contain all per-class, macro, weighted, ROC-AUC, average-precision, and bootstrap values.
- Each evaluation directory contains confusion-matrix CSV/PNG files, ROC/precision-recall plots, actual training/validation curves, misclassified paths, and representative test error images.
- The selected MobileNetV2 robustness results are in `reports/evaluation/mobilenetv2/robustness.json`.
- Actual test-image Grad-CAM examples and their scores are in `reports/evaluation/mobilenetv2/gradcam_examples/`.
