# Provenance and patient/slide identifier review

**Review date:** 4 October 2026

## Findings

The team confirms that the three supplied class archives came from the Google Drive folder linked by the GeeksforGeeks article. The article describes the folder contents as a custom subset of approximately 2,400 images in `lung_n`, `lung_aca`, and `lung_scc`; it does not name the subset creator, identify an upstream dataset, or state a dataset license. This confirms the acquisition route, not ownership or reuse permission.

The archived filenames provide a strong but unverified clue about lineage. The 2,400 files use names such as `lungn1.jpeg`, `lungaca1001.jpeg`, and `lungscc1001.jpeg`. The official LC25000 repository documents the same class names and filename pattern. The supplied subset contains 800 files per class at 128 × 128 pixels, while the LC25000 release is documented as 5,000 per class at 768 × 768 pixels. Those matches are consistent with a resized LC25000 subset, but the Drive folder contains no manifest, derivation note, source checksums, or image-parent map that proves the relationship. We therefore identify LC25000 as the **probable candidate upstream**, not as confirmed provenance. The creator of this 2,400-image subset remains unidentified.

If the candidate lineage is correct, the upstream LC25000 paper names Andrew A. Borkowski, Marilyn M. Bui, L. Brannon Thomas, Catherine P. Wilson, Lauren A. DeLand, and Stephen M. Mastorides as authors. It reports 250 original lung images per class and expansion to 5,000 images per class by augmentation. The subset has no mapping from its images to those source images, slides, or patients. This conditional lineage increases concern that transformed variants of the same source tissue could remain in separate duplicate groups even after the current near-duplicate check.

## License and reuse status

No license or reuse terms were found in the three supplied archives or the GeeksforGeeks article. The LC25000 paper describes the images as freely available for download to AI researchers, but that statement is not an explicit license grant. The official repository page lists a README but no dataset license file. As the LC25000 relationship is not confirmed, its availability statement cannot clear the rights for this Drive subset in any case. **Reuse and redistribution rights remain unverified.** Obtain a written license or permission from the responsible rights holder before further reuse, publication of images, or redistribution of the archives.

## Identifier and split review

The archives contain no patient/slide manifest or other metadata file. Image files are JPEGs with class-plus-number names; the number is not documented as a patient or slide identifier. The image files contain no EXIF metadata. We found no reliable patient IDs, slide IDs, or source-image parent IDs, so a patient/slide-level split cannot currently be constructed.

The saved split remains seed-42, stratified by class, and grouped by exact/near-duplicate component. Its partitions are approximately 70/15/15 (1,680 train, 361 validation, 359 test); all 54 duplicate components stay within one partition. Every row in `reports/splits.csv` has `grouping_level=duplicate_cluster` and `patient_independent=false`. Describe the current results only as **duplicate-group-separated**. They are not patient-independent, external, or clinical validation.

If reliable identifiers or a source-image parent mapping become available, create `data/group_manifest.csv` with `relative_path,group_id,group_type`, rerun the audit and group split, train both declared models on the new training partition, select by validation only, then rerun the held-out test evaluation once. Do not carry the current metrics over to a changed split. If no such mapping can be obtained, preserve the present limitation and result label.

## Evidence checked

| Evidence | What it supports | What it does not establish |
|---|---|---|
| GeeksforGeeks article and its linked Drive folder | The article calls this a custom approximately 2,400-image, three-class subset and links to the folder; the team confirms that as the acquisition route. | Subset creator, upstream source, license, collection dates, or patient/slide identity. |
| Supplied image files and archive contents | 800 JPEGs per class, 128 × 128 RGB, class-plus-index filenames, no EXIF, and no manifests. | That filenames encode biological identifiers or establish LC25000 lineage. |
| LC25000 official repository and 2019 paper | Same class labels and filename pattern; LC25000 authors, documented release dimensions/counts, and augmentation method. | That the Drive subset is definitively derived from LC25000, or that its license permits reuse/redistribution. |

## Sources

- [GeeksforGeeks: Lung Cancer Detection using CNN](https://www.geeksforgeeks.org/deep-learning/lung-cancer-detection-using-convolutional-neural-network-cnn/)
- [Article-linked Drive folder](https://drive.google.com/drive/folders/1vcw8JRQr8-62IVvix4rK4rDXrFnlap8e?usp=sharing)
- [LC25000 official repository](https://github.com/tampapath/lung_colon_image_set)
- [Borkowski et al. (2019), LC25000 paper](https://arxiv.org/abs/1912.12142)
