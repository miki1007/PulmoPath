# Remaining-work checklist

## Completed in this worker run

- [x] Stage three supplied archives into `data/raw/lung_subset_small/{lung_n,lung_aca,lung_scc}/`.
- [x] Audit class counts, file integrity, dimensions, channels, file sizes, and exact/near duplicates.
- [x] Prepare the seed-42 duplicate-cluster grouped 70/15/15 split; record its counts.
- [x] Install the project requirements, including TensorFlow 2.21.0, in the isolated `.venv`.
- [x] Train MobileNetV2 with an ImageNet-pretrained frozen backbone and select its checkpoint by validation loss.
- [x] Train the corrected GroupNormalization scratch CNN on the same split.
- [x] Select the demo model using validation macro-F1, validation loss, and recorded CPU fit time; retain test data out of model selection.
- [x] Evaluate both predeclared models once on the held-out test partition with common evaluation code.
- [x] Generate test metrics, class-wise error summaries, confusion matrices, ROC/PR plots, training curves, and representative errors.
- [x] Run the documented image perturbations on the selected MobileNetV2 model and save per-class results.
- [x] Generate and inspect original/Grad-CAM side-by-side examples from the selected model.
- [x] Start the local Streamlit app and verify its health endpoint and page response; test model loading/inference through evaluation and Grad-CAM runs.
- [x] Preserve and document the initial scratch-CNN run, which failed an inference-mode sanity check.

## Still required

- [x] Record the team-confirmed acquisition route: GeeksforGeeks article → linked Google Drive folder → supplied class archives.
- [x] Review archive filenames against the documented LC25000 release: record LC25000 as a probable but unverified lineage candidate and record its named authors conditionally. The custom subset creator remains unknown.
- [x] Check supplied archives, article, and official candidate-source repository for license notices. No explicit dataset license was found; reuse and redistribution rights remain unverified.
- [x] Inspect filenames, image metadata, manifests, and split metadata for patient/slide IDs. None were found; numeric filename suffixes are not treated as biological identifiers.
- [x] Preserve the current evaluation label as duplicate-group-separated, not patient-independent.
- [ ] Obtain written source/rights-holder confirmation of the subset's lineage, creator, license, version, and permitted reuse before redistribution or broader reuse.
- [ ] If reliable patient/slide or source-image parent IDs become available, create a verified group manifest, rebuild the split, retrain both models, and rerun evaluation on the new held-out test set.
- [ ] Obtain a separate compatible external dataset before making any external-generalization claim.
- [ ] If an OOD dataset becomes available, conduct a formal OOD evaluation; current app warnings are heuristic only.

No unmeasured result is filled in as if it were known. If provenance or identifiers cannot be verified, retain that as an explicit limitation.
