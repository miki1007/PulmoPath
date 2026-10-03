# Data Card — `lung_subset_small`

## Dataset identity and provenance

| Field | Audited information |
|---|---|
| Dataset name | `lung_subset_small`, the name used by the GeeksforGeeks article and notebook. The team confirms that the three supplied class archives came from the Drive folder linked by that article. The subset creator/custodian is not identified. |
| Confirmed acquisition folder | The article links to [this Google Drive folder](https://drive.google.com/drive/folders/1vcw8JRQr8-62IVvix4rK4rDXrFnlap8e?usp=sharing), and the team confirms the supplied archives came from it. The folder documents where the files were obtained, not who originally created or collected the images. |
| Version or release date | Not stated in the article/notebook materials or the supplied archives. |
| Team download date | Exact Drive download date/time was not separately recorded. The archives were handed off to the project workspace on 2026-10-03, and their filenames carry timestamps from that date; neither value is treated as a verified download timestamp. |
| Original image collection period | Unknown; no source documentation establishes clinical collection dates. |
| Candidate upstream dataset | **Probable but unverified LC25000 lineage.** The class-plus-number filename pattern and three lung labels match the official LC25000 release. The supplied 128 × 128, 800-per-class subset differs from the documented 768 × 768, 5,000-per-class release, and no manifest or image-parent mapping confirms derivation. The 2,400-image subset creator remains unknown. See `docs/PROVENANCE_AND_GROUP_ID_REVIEW.md`. |
| Candidate upstream creators | If LC25000 lineage is confirmed: Andrew A. Borkowski, Marilyn M. Bui, L. Brannon Thomas, Catherine P. Wilson, Lauren A. DeLand, and Stephen M. Mastorides (2019). This is a qualified attribution to a possible upstream dataset, not attribution of the custom subset. |
| License and permitted academic use | No license notice was found in the supplied archives or the GeeksforGeeks article. The LC25000 paper says the dataset is freely available to AI researchers, but does not give an explicit dataset license; its official repository page has no dataset license file. The lineage is unverified. **Reuse and redistribution rights remain unverified.** |
| Supplied archives | `lung_n-20261003T175532Z-1-001.zip`, `lung_aca-20261003T175529Z-1-001.zip`, `lung_scc-20261003T175534Z-1-001.zip`. SHA-256 digests are recorded in `reports/archive_checksums.json`. |
| Machine-readable provenance | `reports/provenance.json` records the confirmed acquisition route, qualified LC25000 lineage assessment, archive audit facts, and unresolved ownership/license fields. |

## Image audit

The article describes approximately 2,400 images; the supplied notebook output also shows 2,400 examples for its run. The project audit found exactly 2,400 files in the supplied archives:

| Folder/class | Image files before cleaning | Usable after blank/unreadable checks |
|---|---:|---:|
| `lung_n` | 800 | 800 |
| `lung_aca` | 800 | 800 |
| `lung_scc` | 800 | 800 |
| **Total** | **2,400** | **2,400** |

All 2,400 files are 128 × 128 RGB JPEG images. The audit found no unreadable, blank, or grayscale images and no non-image files in the class directories. It recorded 22 exact-duplicate links and 32 perceptual-hash links at Hamming distance ≤4, forming 54 connected duplicate components. No duplicate component crossed class labels. These checks reduce obvious image-level leakage but do not establish patient or slide independence.

Compressed file sizes ranged from 1,570 to 5,319 bytes. By class, the minimum/median/maximum were 1,570/3,569/4,832 bytes for `lung_n`, 2,472/3,888/5,319 bytes for `lung_aca`, and 2,313/3,787/5,016 bytes for `lung_scc`. ZIP byte counts and SHA-256 digests are recorded in `reports/archive_checksums.json`.

The data are patch images according to the article/notebook task and folder structure; source-level documentation was not supplied to confirm how patches were created. Patient count, slide count, patient/slide identifiers, staining protocol, scanner, sampling process, and annotation procedure are unknown. Filenames contain a class label and numeric sample suffix only; no patient/slide manifest or EXIF metadata was found. Do not infer identifiers from filenames or folder names.

## Classes and intended use

| Label | Dataset folder meaning |
|---|---|
| `lung_n` | Normal lung tissue |
| `lung_aca` | Lung adenocarcinoma tissue |
| `lung_scc` | Lung squamous-cell carcinoma tissue |

Intended use is educational research on image classification, reproducibility, error analysis, interpretability, and robustness. The model's output is a dataset-label score; it is not a calibrated probability of disease and must not be used for diagnosis, screening, triage, treatment, or patient advice.

## Ethical and research limitations

The acquisition route is documented as the article-linked Drive folder. The filename convention suggests possible LC25000 lineage, but the connection is unverified and does not identify who assembled the 2,400-image subset. No explicit license was found; reuse and redistribution rights remain unverified. No patient/slide metadata or original acquisition dates were supplied. The present split is grouped by exact/near-duplicate components only, so patient-independent evaluation cannot be claimed. If the LC25000 lineage is correct, the upstream release was expanded from a smaller set of original images by augmentation; the supplied subset has no parent-image map, leaving possible source-image leakage unassessed. The dataset's size and unknown original-source provenance limit external validity. Results do not establish performance on other institutions, populations, scanners, or staining protocols.
