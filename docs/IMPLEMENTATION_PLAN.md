# Implementation plan and stage status

| Stage | Work | Status |
|---|---|---|
| 1. Material audit | Inspect article/notebook task, supplied ZIP contents and models, image references, labels, and logo. Keep histology, CT, and tabular tasks separate. | Complete; see `MATERIAL_AUDIT.md`. |
| 2. Dataset audit and split | Verify class folders, image integrity/modes, file sizes, exact/near duplicates, and patient/slide identifiers when documented; prepare approximately 70/15/15 partitions. | Complete for the supplied files. 2,400 images passed basic checks; duplicate groups were held intact. Patient/slide IDs were not supplied, so patient independence is unverified. |
| 3. Train comparable models | Train the 128×128 scratch CNN and ImageNet MobileNetV2 on the same split. | Complete. Both checkpoints load in inference mode; run metadata and histories are saved. The initial BatchNormalization run predicted only `lung_scc` in inference mode and is retained as a failed diagnostic, not a model result. |
| 4. Select and evaluate | Use validation results and computational cost for model selection, then evaluate the predeclared model comparison on the held-out test split. | Complete. MobileNetV2 was selected using validation-only comparison. Both models were then evaluated once on the test set; see `MODEL_COMPARISON.md`. |
| 5. Robustness and explanation | Run controlled image perturbations and create Grad-CAM overlays from verified checkpoints. | Complete for the selected model and available subset. Results are in `ROBUSTNESS_REPORT.md`; three test-patch Grad-CAM figures were generated. External and formal OOD evaluation remain unavailable. |
| 6. Local demo | Run Streamlit with the supplied WLC logo, supported-input guidance, class scores, limitations, and optional Grad-CAM. | Source compiles; localhost health and page checks pass. The app defaults to the validation-selected MobileNetV2 checkpoint. |

## Current measured data facts

- Class folders contain 800 images each (2,400 total), all 128 × 128 RGB JPEGs.
- No blank, grayscale, or unreadable images were found.
- Exact and near-duplicate components were grouped before splitting; no cross-class duplicate components were found.
- Seed 42 produced 1,680 training, 361 validation, and 359 test images.
- Only duplicate-group separation is supported by the supplied metadata. Patient-independent evaluation cannot be claimed.
- The article-linked Drive acquisition route is confirmed by the team. The original dataset creator/custodian, license, exact download date, original collection dates, and permitted academic use remain unverified.
- Results are internal to this supplied subset and do not establish external or clinical generalization.

Do not treat validation performance as test performance, or any model score as a clinical result.
