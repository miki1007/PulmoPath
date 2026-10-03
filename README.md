# WLC PulmoPath AI

**CNN-Based Lung Histology Image Classification**  
World Link College · Educational research prototype

WLC PulmoPath AI is a local research prototype for classifying a single lung histopathology image patch into one of the dataset labels `lung_n`, `lung_aca`, or `lung_scc`. It includes an auditable data workflow, a CNN baseline, a MobileNetV2 transfer-learning comparison, held-out evaluation, Grad-CAM visualization, controlled robustness checks, and a Streamlit interface.

> **Educational research prototype — not for clinical diagnosis or treatment.**

This project is not a diagnostic or screening device and is not a substitute for a pathologist. A predicted dataset label and its score do not describe a person's health. Test-set results are reported only after model training and held-out evaluation complete; validation results are not presented as test performance.

## GitHub repository contents

This repository contains the source code, documentation, college logo, aggregate evaluation metrics, and plots. It intentionally excludes the raw histology images, trained model checkpoints, training logs, and evaluation examples that contain image patches. The dataset creator and reuse license are unverified, so these files should not be redistributed from this repository. A fresh clone needs a lawfully obtained dataset and newly trained checkpoints before it can make predictions; the app will show a setup message while checkpoints are missing.

## Project material audit

Read [Project Material Audit](docs/MATERIAL_AUDIT.md) before using the files. The article/notebook and three supplied class archives concern three-class lung histology. The separate Streamlit ZIP contains CT-image and tabular risk-factor components; those tasks, labels, and model weights are not mixed with histology.

The staged work and current completion status are in [Implementation Plan](docs/IMPLEMENTATION_PLAN.md).

## Data status and provenance

In the local audited worktree, the three class archives were staged under `data/raw/lung_subset_small/`; the audit found 800 files per class. The team confirms that they came from the Google Drive folder linked by the GeeksforGeeks article ([folder](https://drive.google.com/drive/folders/1vcw8JRQr8-62IVvix4rK4rDXrFnlap8e?usp=sharing)). Their filenames suggest possible LC25000 lineage, but the link is unverified and the creator of this smaller subset is unknown. No explicit license was found; reuse and redistribution rights remain unverified. Patient/slide/source-image identifiers are absent. The current split and results are duplicate-group-separated, not patient-independent. See the [data card](docs/DATA_CARD.md), [provenance and identifier review](docs/PROVENANCE_AND_GROUP_ID_REVIEW.md), [audit report](docs/DATA_AND_SPLIT_REPORT.md), and machine-readable [`reports/provenance.json`](reports/provenance.json). Raw archives are intentionally absent from this GitHub repository.

The current project data tree is:

```text
data/raw/lung_subset_small/
├── lung_n/
├── lung_aca/
└── lung_scc/
```

Do not mix CT scans or tabular records into these folders. `.gitignore` excludes raw data and generated model artifacts.

If patient or slide IDs are available, create a CSV at `data/group_manifest.csv` with these columns:

```csv
relative_path,group_id,group_type
lung_n/example.png,REPLACE_WITH_REAL_GROUP_ID,patient
```

Replace the example row with actual file paths and identifiers from the dataset. Set `group_type` consistently to `patient` or `slide`. Do not infer patient IDs from filenames unless the dataset documentation supports that interpretation. Only a verified patient-level manifest supports the `patient_independent` metadata flag. A slide-level split prevents slide leakage but may still place slides from one patient in different splits. Without a valid manifest, the pipeline groups exact/near-duplicate images only.

## Setup

Use Python 3.11 or 3.12. **Do not use Python 3.13 with the supplied worker lock file:** it pins NumPy 1.26.4, which conflicts with the Python 3.13 requirement from `ml-dtypes`. Confirm the interpreter version before creating the environment.

### macOS or Linux

From the project root, use the explicit Python 3.12 command (or replace it with `python3.11` if that is the version installed):

```bash
python3.12 --version
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-worker-lock.txt
python -m streamlit run app.py
```

If `python3.12` is not installed, install Python 3.12 first. With Homebrew, run `brew install python@3.12`. On Apple Silicon, TensorFlow 2.21 has a macOS ARM64 wheel for macOS 12 or later. An Intel Mac may need the Docker/Linux alternative because the pinned TensorFlow version does not publish a macOS Intel wheel.

### Windows PowerShell

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-worker-lock.txt
python -m streamlit run app.py
```

`requirements-worker-lock.txt` records the exact package versions used in the completed runs. `requirements.txt` is the less restrictive alternative if a different supported Python 3.11/3.12 environment is needed. This GitHub repository does not include trained checkpoints; train a model before making predictions in the demo.

For MobileNetV2 ImageNet initialization, TensorFlow may need internet access once to retrieve the published weights. If unavailable, use `--pretrained none` and document that this run is not ImageNet transfer learning.

## Reproducible workflow

Run commands from the project root.

### 1. Audit data

```bash
python -m pulmopath.audit --data-dir data/raw/lung_subset_small --reports-dir reports
```

This checks the explicit class folders; inventories image dimensions, modes, file sizes, and formats; flags corrupt, blank, and grayscale files; computes SHA-256 and perceptual hashes; and writes `reports/data_audit.json` and `reports/image_inventory.csv`. Blank and unreadable files are excluded from subsequent splits. Grayscale inputs are flagged for review, not silently discarded. Exact duplicate images with conflicting class labels stop the workflow for manual resolution.

### 2. Make group-aware train/validation/test splits

When patient or slide identifiers are available:

```bash
python -m pulmopath.split --inventory reports/image_inventory.csv --output reports/splits.csv --group-manifest data/group_manifest.csv --seed 42
```

Without those identifiers:

```bash
python -m pulmopath.split --inventory reports/image_inventory.csv --output reports/splits.csv --seed 42
```

The script makes a deterministic, group-aware approximately 70/15/15 partition and searches multiple group holdouts for class balance. Exact ratios can vary because groups stay intact. Without patient/slide IDs, describe the evaluation as duplicate-group separated only, not patient-independent. Inspect `reports/splits.csv` and the split counts before training. Keep the test rows untouched until model selection is complete.

### 3. Train the baseline CNN

```bash
python -m pulmopath.train --model baseline --epochs 40 --patience 7 --split-manifest reports/splits.csv --output artifacts/models/baseline.keras --seed 42
```

### 4. Train a transfer-learning model

```bash
python -m pulmopath.train --model mobilenetv2 --pretrained imagenet --epochs 25 --patience 6 --split-manifest reports/splits.csv --output artifacts/models/mobilenetv2.keras --seed 42
```

With ImageNet weights, the MobileNetV2 classification head is trained first with the backbone frozen. Optional fine-tuning unfreezes only selected upper layers and keeps batch-normalization layers frozen. If ImageNet weights are unavailable, `--pretrained none` trains the randomly initialized backbone and is not transfer learning. Both model runs use the same split manifest and image size. Training uses only the training split; validation loss selects checkpoints and early stopping. Class weights are applied only when the training class ratio meets the documented 1.5× imbalance rule. The completed run used the frozen-backbone head stage without fine-tuning.

Each successful training run creates a genuine `.keras` checkpoint, `.metadata.json`, and `.history.json`. Training refuses to proceed if the split manifest or images are missing. The code verifies that a saved checkpoint can be loaded before writing metadata.

### 5. Select one model, then evaluate the held-out test set

Choose the model using validation results and computational cost first. After recording that choice, evaluate each predeclared comparison model once on the held-out test split for reporting; do not use test results to select, tune, or retrain a model:

```bash
python -m pulmopath.evaluate --model artifacts/models/baseline.keras --split-manifest reports/splits.csv --output-dir reports/evaluation/baseline
python -m pulmopath.evaluate --model artifacts/models/mobilenetv2.keras --split-manifest reports/splits.csv --output-dir reports/evaluation/mobilenetv2
```

Evaluation writes accuracy; per-class, macro, and weighted precision/recall/F1; confusion matrices; one-vs-rest ROC-AUC and average precision when meaningful; group-bootstrap intervals where feasible; misclassified image paths; and actual training/validation curves. It does not fill in results before a real training/evaluation run.

### 6. Run predefined robustness checks

```bash
python -m pulmopath.robustness --model artifacts/models/mobilenetv2.keras --split-manifest reports/splits.csv --output reports/evaluation/mobilenetv2/robustness.json
python -m pulmopath.explainability --model artifacts/models/mobilenetv2.keras --split test --output-dir reports/evaluation/mobilenetv2/gradcam_examples
```

The robustness script evaluates brightness ±8%, contrast ±10%, 96×96-to-128×128 resizing with nearest/bilinear/bicubic interpolation, JPEG quality 85, and ±4% red/blue channel gains. The explainability command saves side-by-side original/Grad-CAM images for one test example per true class. These controlled perturbations and attention maps do not establish clinical robustness or biological reasoning.

### 7. Launch the local demo

```bash
python -m streamlit run app.py
```

The app displays the supplied World Link College logo, supports one PNG/JPEG/TIFF image at a time, and includes bundled held-out test examples for quick testing. Its result card explains the three model scores in plain language and offers a Grad-CAM overlay with an opacity control and limitations. The app does not save or transmit uploaded images. It warns that CT scans and ordinary images are unsupported; its color check is only a heuristic and cannot reliably detect every out-of-distribution image. See [Local demo guide](docs/APP_GUIDE.md) for score and heatmap interpretation.

## Project structure

```text
app.py                         Local Streamlit interface
.streamlit/config.toml         Light theme and local server defaults
assets/world_link_college_logo.jpeg
pulmopath/audit.py             Image and duplicate audit
pulmopath/split.py             Group-aware data partitioning
pulmopath/data.py              RGB loading and training-only augmentation
pulmopath/models.py            Baseline CNN and MobileNetV2
pulmopath/train.py             Training and validation checkpoint selection
pulmopath/evaluate.py          Held-out evaluation and plots
pulmopath/gradcam.py           Grad-CAM overlay helper
pulmopath/explainability.py    Reproducible side-by-side Grad-CAM exports
pulmopath/robustness.py        Controlled perturbation checks
docs/                          Audit, data card, methods, and limitations
reports/evaluation/             Measured metrics, curves, errors, robustness, and Grad-CAM outputs
```

## Current status

- Source code and documentation are implemented.
- The provided World Link College logo is included without alteration.
- The supplied archives are staged and audited; see `reports/data_audit.json`, `reports/image_inventory.csv`, and `reports/splits.csv`.
- Both comparison checkpoints were trained and inference-checked. MobileNetV2 was selected for the demo from validation results; test metrics and robustness results are recorded in `docs/MODEL_COMPARISON.md` and `docs/ROBUSTNESS_REPORT.md`.
- A failed initial scratch-CNN run is retained under `artifacts/diagnostics/baseline_batchnorm_failure/`; it predicted only one class in inference mode and is not reported as a valid model.
- Test results are internal to this supplied subset and use duplicate-cluster-separated—not patient-independent—partitions. The article-linked Drive route is team-confirmed. LC25000 is a probable but unverified upstream candidate; the custom-subset creator and explicit license remain unverified. Reliable patient/slide/source-image identifiers, exact Drive download date, and original collection period are unavailable.

See [the remaining-work checklist](docs/REMAINING_CHECKLIST.md) before presenting results.
