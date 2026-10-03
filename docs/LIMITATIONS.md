# Limitations and ethical use

- Educational research prototype only. Not for clinical diagnosis, screening, treatment, or patient decisions.
- A softmax score is a model output for one of the dataset labels, not a calibrated probability of disease.
- A Grad-CAM map is a model-sensitivity visualization; it is not proof of biological reasoning and does not identify a confirmed lesion.
- The team confirms that the supplied histology archives came from the Drive folder linked by the GeeksforGeeks article. The filenames suggest possible LC25000 lineage, but this is unverified; the creator of the 2,400-image subset is unknown. No explicit license was found, so reuse and redistribution rights remain unverified. Exact download date, original collection period, patient/slide counts, and annotation process remain unknown.
- The reference notebook uses an image-level train/validation split, has no held-out test set, and has a potential prediction/label ordering issue because its validation generator shuffles by default.
- The current split keeps exact/near-duplicate components together but has no reliable patient, slide, or source-image parent identifiers; its results are duplicate-group-separated, not patient-independent. If reliable grouping metadata becomes available, the split, both training runs, and evaluation must be repeated.
- If the likely LC25000 filename lineage is correct, the upstream dataset includes augmented variants of a smaller set of original tissue images. The supplied subset has no mapping to those originals, so source-image leakage beyond the duplicate clusters cannot be ruled out.
- The audited set contains 2,400 images from one supplied subset and has no external evaluation cohort. Results cannot establish generalization to other institutions, scanners, staining protocols, or patient populations.
- Duplicate removal and controlled perturbation checks do not establish external generalization or clinical robustness.
- The app's low-color heuristic cannot reliably identify CT scans, non-lung images, or all unsupported inputs. Users must provide documented histology patches.
- The supplied CT and tabular materials belong to different tasks and are excluded from this model.

Do not enter personally identifying information or upload patient images to the local demo. The current app performs local inference and does not intentionally save or transmit uploaded files.
