# Dataset setup

The three class folders are staged here from the archives the team confirms were obtained from the Google Drive folder linked by the GeeksforGeeks article (see `docs/DATA_CARD.md`). This is the acquisition route; the original dataset creator and license remain unverified. If restoring the data from the source folder, first confirm reuse terms, then extract only the three class directories here:

```text
data/raw/lung_subset_small/lung_n/
data/raw/lung_subset_small/lung_aca/
data/raw/lung_subset_small/lung_scc/
```

Do not place the CT-scan folders or tabular CSV files from the reference ZIP here. Do not commit the raw dataset. If verified patient or slide identifiers are available, create `data/group_manifest.csv` with columns `relative_path,group_id,group_type`, where `group_type` is consistently `patient` or `slide`, and use it during splitting.
