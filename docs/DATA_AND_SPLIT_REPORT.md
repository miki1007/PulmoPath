# Data audit and split report

## Audit findings

The supplied class ZIPs were extracted into `data/raw/lung_subset_small/{lung_n,lung_aca,lung_scc}/`. The machine-readable image inventory and summary are `reports/image_inventory.csv` and `reports/data_audit.json`.

The audit found 800 files per class (2,400 total), all 128 × 128 RGB JPEGs. All files passed decoding and blank-image checks; none were grayscale, blank, unreadable, or excluded. There were 22 exact-duplicate links and 32 perceptual-hash links at distance ≤4, with 54 connected duplicate components. No duplicate group crossed class labels. The audit does not infer patient identity from filenames.

The team confirms the supplied archives were obtained from the Google Drive folder linked by the GeeksforGeeks article. The class-plus-number filenames match the LC25000 naming convention, suggesting a possible LC25000 lineage, but the archive lacks a source manifest, checksums against an upstream copy, or image-parent map; the relationship and creator of this smaller subset remain unverified. No license notice was found in the archives or article. Although the LC25000 paper describes its dataset as freely available to AI researchers, it does not provide an explicit dataset license, and its connection to this subset is unconfirmed. Reuse and redistribution rights remain unverified. No patient/slide identifiers or EXIF metadata were found. The 2026-10-03 workspace handoff and archive filename timestamps do not establish the exact Drive download date; original collection dates remain unknown. See `docs/PROVENANCE_AND_GROUP_ID_REVIEW.md` and `reports/provenance.json` for the evidence record.

## Split method and observed partition

The fixed seed was 42. Perceptual-hash duplicate components were kept intact, and repeated group-shuffle search selected partitions close to 70/15/15 while maintaining all three labels in each partition. No patient/slide/source-image group manifest was available, so grouping is only by duplicate component. This is duplicate-group separation, not patient-independent splitting. If reliable patient/slide or source-image lineage metadata becomes available, regenerate the split and rerun both training and held-out evaluation; current metrics do not transfer to a revised split.

| Partition | Images | `lung_n` | `lung_aca` | `lung_scc` | Share |
|---|---:|---:|---:|---:|---:|
| Training | 1,680 | 559 | 561 | 560 | 70.00% |
| Validation | 361 | 121 | 120 | 120 | 15.04% |
| Test | 359 | 120 | 119 | 120 | 14.96% |
| **Total** | **2,400** | **800** | **800** | **800** | **100%** |

`reports/splits.csv` records each image, label, duplicate group, partition, seed, and grouping method. All exact/near-duplicate components remain within one partition. Because patient or slide identifiers are absent, patient-level independence cannot be guaranteed; this is an internal duplicate-group-separated evaluation, not external or clinical validation.

## Preprocessing choices

- The explicit class map is `lung_n` → 0, `lung_aca` → 1, and `lung_scc` → 2.
- Images are decoded as RGB, resized to 128 × 128 using bicubic interpolation, and passed to the model as float pixel values in [0,255]. Model-specific scaling is embedded in each architecture.
- Training augmentation consists of modest rotation (factor 0.04), horizontal/vertical flips, brightness delta up to 12/255, and contrast range 0.90–1.10. Augmentation is applied only to training data.
- Validation and test data are not augmented.
- No class weights are applied because the observed training-class ratio is below the documented 1.5× threshold.
- Stain/color normalization has not been applied. Its effects require a separate, controlled comparison.

Training, validation, and test rows are stored in one manifest. Training fits weights; validation loss selects checkpoints and controls early stopping; the test partition is held out from that model-selection process.
