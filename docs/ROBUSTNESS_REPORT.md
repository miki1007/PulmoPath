# Robustness experiment report

The selected MobileNetV2 checkpoint was evaluated on the same 359-image held-out test partition under the predeclared pixel perturbations below. The baseline is ordinary RGB loading with bicubic resizing to 128 × 128. Accuracy and macro-F1 changes are percentage-point differences from that baseline. Per-class F1 values are reported in `reports/evaluation/mobilenetv2/robustness.json`.

| Test condition | Definition | Accuracy | Macro-F1 | Accuracy change (pp) | Macro-F1 change (pp) |
|---|---|---:|---:|---:|---:|
| Baseline | Bicubic 128 × 128 | 0.9471 | 0.9470 | 0.00 | 0.00 |
| Brightness −8% | Pillow factor 0.92 | 0.9499 | 0.9497 | +0.28 | +0.28 |
| Brightness +8% | Pillow factor 1.08 | 0.9526 | 0.9526 | +0.56 | +0.56 |
| Contrast −10% | Pillow factor 0.90 | 0.9499 | 0.9496 | +0.28 | +0.27 |
| Contrast +10% | Pillow factor 1.10 | 0.9499 | 0.9498 | +0.28 | +0.28 |
| Resize, nearest | Downsample to 96 × 96 then restore to 128 × 128 using nearest interpolation | 0.9304 | 0.9297 | −1.67 | −1.73 |
| Resize, bilinear | Downsample to 96 × 96 then restore to 128 × 128 using bilinear interpolation | 0.9526 | 0.9525 | +0.56 | +0.55 |
| Resize, bicubic | Downsample to 96 × 96 then restore to 128 × 128 using bicubic interpolation | 0.9554 | 0.9553 | +0.84 | +0.83 |
| JPEG quality 85 | Re-encode each patch at JPEG quality 85 | 0.9554 | 0.9554 | +0.84 | +0.84 |
| Warm color gain | Multiply red by 1.04 and blue by 0.96 | 0.9415 | 0.9414 | −0.56 | −0.56 |
| Cool color gain | Multiply red by 0.96 and blue by 1.04 | 0.9554 | 0.9554 | +0.84 | +0.84 |

The largest observed decline was with nearest-neighbor down/up-sampling: accuracy fell 1.67 percentage points and macro-F1 fell 1.73 points. The warm color shift reduced both measures by 0.56 points. Mild brightness/contrast changes, bilinear/bicubic resizing, quality-85 compression, and cool color gain did not lower aggregate metrics in this test set; some produced small apparent gains. These gains should not be interpreted as evidence that the perturbations improve the model. Under all conditions, the most persistent classification difficulty remained the distinction between `lung_aca` and `lung_scc`; see class-wise results in the JSON output.

These are controlled transformations of the same small test set, not independent replications. No separate out-of-distribution lung set or external institution dataset was available. The Streamlit app's file-type and color warnings are heuristics and do not constitute a measured OOD detector. Generalization across institutions, populations, scanners, tissue preparation, and staining protocols remains untested; these experiments do not establish clinical robustness.
