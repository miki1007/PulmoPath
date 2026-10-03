"""Polished local Streamlit interface for the WLC PulmoPath AI research prototype."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from pulmopath.constants import CLASS_NAMES, GFG_ARTICLE_URL, GFG_DATASET_URL

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "artifacts" / "models"
LOGO_PATH = ROOT / "assets" / "world_link_college_logo.jpeg"
DATA_DIR = ROOT / "data" / "raw" / "lung_subset_small"
SPLIT_PATH = ROOT / "reports" / "splits.csv"
DISCLAIMER = "Educational research prototype — not for clinical diagnosis or treatment."

CLASS_INFO = {
    "lung_n": {
        "title": "Normal lung tissue",
        "short": "Normal tissue",
        "description": "The dataset's normal-lung tissue category.",
        "color": "#25816F",
        "light": "#E7F4F0",
    },
    "lung_aca": {
        "title": "Lung adenocarcinoma tissue",
        "short": "Adenocarcinoma",
        "description": "The dataset's lung adenocarcinoma category.",
        "color": "#B45D70",
        "light": "#F9EDF0",
    },
    "lung_scc": {
        "title": "Lung squamous-cell carcinoma tissue",
        "short": "Squamous-cell carcinoma",
        "description": "The dataset's lung squamous-cell carcinoma category.",
        "color": "#685BB2",
        "light": "#F0EEFA",
    },
}

st.set_page_config(
    page_title="WLC PulmoPath AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #182b40;
          --muted: #64758a;
          --navy: #142a43;
          --teal: #16877f;
          --line: #e2e9f0;
          --canvas: #f3f6fa;
          --white: #ffffff;
        }
        .stApp { background: var(--canvas); color: var(--ink); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stMainBlockContainer"] {
          max-width: 1500px;
          padding-top: 1.25rem;
          padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
          background: #12263d;
          border-right: 1px solid rgba(255,255,255,.08);
        }
        [data-testid="stSidebar"] * { color: #eef4fa; }
        [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.16); }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
          padding: .45rem .55rem;
          border-radius: 10px;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
          color: #b9c9d9;
        }
        h1, h2, h3 { color: var(--ink); letter-spacing: -.025em; }
        h2 { margin-top: .25rem; }
        p, li { color: #34495f; }
        .hero {
          background: linear-gradient(115deg, #142a43 0%, #1c4058 70%, #205866 100%);
          border-radius: 16px;
          padding: 1.05rem 1.45rem;
          margin: .1rem 0 .8rem 0;
          color: white;
        }
        .hero-kicker {
          color: #a9e5d9;
          text-transform: uppercase;
          font-size: .72rem;
          font-weight: 800;
          letter-spacing: .14em;
          margin-bottom: .25rem;
        }
        .hero-title {
          color: #ffffff;
          font-size: clamp(1.45rem, 2.5vw, 2rem);
          line-height: 1.12;
          font-weight: 780;
          letter-spacing: -.035em;
          margin: 0;
        }
        .hero-copy {
          color: #d7e6ef;
          font-size: .91rem;
          max-width: 760px;
          line-height: 1.55;
          margin: .3rem 0 0 0;
        }
        .notice {
          background: #fff7e9;
          border: 1px solid #f0d49e;
          border-left: 4px solid #d99a2b;
          color: #5f481e;
          border-radius: 9px;
          padding: .55rem .8rem;
          margin: .65rem 0 .9rem;
          line-height: 1.3;
          font-size: .86rem;
        }
        .notice strong { color: #553c0d; }
        .section-kicker {
          color: #18857d;
          text-transform: uppercase;
          font-size: .7rem;
          font-weight: 800;
          letter-spacing: .13em;
          margin: 0 0 .25rem 0;
        }
        .panel-title {
          color: var(--ink);
          font-size: 1.12rem;
          font-weight: 760;
          margin: 0 0 .3rem 0;
        }
        .panel-subtitle {
          color: var(--muted);
          font-size: .89rem;
          margin: 0 0 .85rem 0;
          line-height: 1.45;
        }
        .top-match {
          border-radius: 17px;
          border: 1px solid #e5eaf0;
          padding: 1rem 1.05rem;
          margin: .1rem 0 1rem 0;
        }
        .top-match-kicker {
          text-transform: uppercase;
          letter-spacing: .11em;
          font-size: .68rem;
          font-weight: 850;
          opacity: .76;
        }
        .top-match-title {
          font-size: 1.25rem;
          font-weight: 800;
          margin: .35rem 0 .2rem 0;
          line-height: 1.25;
        }
        .top-match-copy {
          font-size: .84rem;
          opacity: .83;
          line-height: 1.45;
        }
        .top-match-score {
          display: inline-block;
          margin-top: .7rem;
          padding: .35rem .65rem;
          border-radius: 999px;
          background: rgba(255,255,255,.78);
          font-weight: 800;
          font-size: .87rem;
        }
        .score-row { margin: .75rem 0; }
        .score-heading {
          display: flex;
          justify-content: space-between;
          gap: 1rem;
          align-items: baseline;
          margin-bottom: .35rem;
        }
        .score-name { color: #24394f; font-size: .88rem; font-weight: 700; }
        .score-label { color: #738197; font-size: .75rem; margin-left: .35rem; }
        .score-number { color: #22364b; font-variant-numeric: tabular-nums; font-size: .88rem; font-weight: 800; }
        .score-track {
          height: 9px;
          background: #edf1f5;
          border-radius: 99px;
          overflow: hidden;
        }
        .score-fill { height: 100%; border-radius: 99px; }
        .score-note {
          padding: .7rem .8rem;
          margin-top: .85rem;
          border-radius: 10px;
          background: #f2f6fa;
          color: #53677d;
          font-size: .8rem;
          line-height: 1.48;
        }
        .privacy-note {
          border-radius: 10px;
          background: #edf7f4;
          color: #30665e;
          padding: .65rem .75rem;
          font-size: .8rem;
          line-height: 1.4;
          margin-top: .7rem;
        }
        .explain-step {
          border: 1px solid var(--line);
          background: #ffffff;
          border-radius: 13px;
          padding: .9rem;
          min-height: 122px;
        }
        .explain-step-number {
          color: #14877c;
          font-size: .7rem;
          font-weight: 850;
          letter-spacing: .1em;
        }
        .explain-step-title {
          color: var(--ink);
          font-size: .95rem;
          font-weight: 750;
          margin: .3rem 0 .25rem 0;
        }
        .explain-step-copy {
          color: var(--muted);
          font-size: .8rem;
          line-height: 1.45;
        }
        .class-card {
          border: 1px solid var(--line);
          background: #ffffff;
          border-radius: 14px;
          padding: 1rem;
          height: 100%;
        }
        .class-code {
          display: inline-block;
          font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
          background: #f2f5f8;
          color: #344a60;
          border-radius: 6px;
          padding: .16rem .4rem;
          font-size: .78rem;
          margin-bottom: .55rem;
        }
        .class-title {
          color: var(--ink);
          font-weight: 760;
          font-size: .94rem;
          margin-bottom: .28rem;
        }
        .class-copy {
          color: var(--muted);
          font-size: .8rem;
          line-height: 1.45;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
          border-color: var(--line);
          border-radius: 17px;
          background: #ffffff;
          box-shadow: 0 4px 16px rgba(30, 54, 78, .035);
        }
        div[data-testid="stFileUploader"] {
          border-radius: 13px;
        }
        div[data-testid="stFileUploader"] section {
          border-radius: 12px;
          border-color: #cfdae5;
          background: #f8fafc;
        }
        div[data-testid="stButton"] button {
          border-radius: 10px;
          font-weight: 700;
        }
        @media (max-width: 850px) {
          [data-testid="stMainBlockContainer"] { padding: .9rem 1rem 2rem; }
          .hero { padding: 1.25rem; border-radius: 17px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading the selected model…")
def load_model_cached(model_file: str):
    import tensorflow as tf

    return tf.keras.models.load_model(model_file)


def _available_models() -> list[Path]:
    return sorted(MODEL_DIR.glob("*.keras")) if MODEL_DIR.is_dir() else []


def _metadata_for(model_file: Path) -> dict:
    path = model_file.with_suffix(".metadata.json")
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _selected_model_stem() -> str | None:
    selection_path = ROOT / "reports" / "model_selection.json"
    try:
        if selection_path.is_file():
            return json.loads(selection_path.read_text(encoding="utf-8")).get("selected_model")
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _hero(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">World Link College · research sandbox</div>
          <div class="hero-title">{html.escape(title)}</div>
          <p class="hero-copy">{html.escape(copy)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _notice() -> None:
    st.markdown(
        f'<div class="notice"><strong>{html.escape(DISCLAIMER)}</strong></div>',
        unsafe_allow_html=True,
    )


def _sidebar() -> tuple[str, Path | None, dict]:
    if LOGO_PATH.is_file():
        st.sidebar.image(str(LOGO_PATH), width=100)
    st.sidebar.markdown("## WLC PulmoPath AI")
    st.sidebar.caption("Lung histology image research demo")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Workspace",
        ["Explore a patch", "About & methods"],
        label_visibility="visible",
    )
    models = _available_models()
    selected_model = None
    metadata: dict = {}
    if models and page == "Explore a patch":
        selected_stem = _selected_model_stem()
        default_index = next((i for i, path in enumerate(models) if path.stem == selected_stem), 0)
        with st.sidebar.expander("Model settings", expanded=False):
            selected_model = st.selectbox(
                "Checkpoint",
                models,
                index=default_index,
                format_func=lambda path: path.stem.replace("_", " ").title(),
            )
            metadata = _metadata_for(selected_model)
            st.caption(
                f"Version {metadata.get('model_version', 'not recorded')} · "
                f"{metadata.get('image_size', 128)} px input"
            )
            if metadata.get("grouping_level"):
                st.caption(
                    "Evaluation split: duplicate-group-separated; "
                    "not patient-independent."
                )
    st.sidebar.markdown("---")
    st.sidebar.caption("Images are processed locally and are not saved or transmitted by this app.")
    return page, selected_model, metadata


def _bundled_test_examples() -> dict[str, Path]:
    examples: dict[str, Path] = {}
    try:
        split = pd.read_csv(SPLIT_PATH)
        for label in CLASS_NAMES:
            candidates = split[(split["label"] == label) & (split["split"] == "test")]
            if candidates.empty:
                continue
            relative = Path(str(candidates.iloc[0]["relative_path"]))
            path = DATA_DIR / relative
            if path.is_file():
                examples[label] = path
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return {}
    return examples


def _load_pil_image(source) -> tuple[Image.Image | None, str | None, str, tuple[int, int] | None]:
    try:
        with Image.open(source) as image:
            image.load()
            original_mode = image.mode
            original_size = image.size
            original = image.convert("RGB").copy()
        if min(original_size) < 64:
            return None, "This image is too small for the pipeline (minimum side: 64 pixels).", original_mode, original_size
        return original, None, original_mode, original_size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return None, f"The file could not be decoded as an image: {exc}", "", None


def _render_score_rows(scores: np.ndarray, names: list[str]) -> None:
    rows = []
    for name, value in zip(names, scores):
        info = CLASS_INFO.get(name, {"short": name, "color": "#58708a"})
        safe_name = html.escape(name)
        safe_short = html.escape(info["short"])
        percent = float(np.clip(value, 0, 1)) * 100
        rows.append(
            f"""
            <div class="score-row">
              <div class="score-heading">
                <div><span class="score-name">{safe_name}</span>
                     <span class="score-label">{safe_short}</span></div>
                <span class="score-number">{percent:.1f}%</span>
              </div>
              <div class="score-track"><div class="score-fill" style="width:{percent:.2f}%;background:{info['color']}"></div></div>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def _prediction_page(selected_model: Path | None, metadata: dict) -> None:
    _hero(
        "Lung histology image classifier",
        "Choose one tissue patch to view the model’s three class scores.",
    )
    _notice()

    models = _available_models()
    if not models:
        st.warning(
            "No trained checkpoints were found. The project ZIP normally includes the baseline and MobileNetV2 weights. "
            "Check that the artifacts/models folder was kept beside app.py."
        )

    input_column, result_column = st.columns([1.02, 0.98], gap="large")
    original: Image.Image | None = None
    original_mode = ""
    original_size: tuple[int, int] | None = None
    reference_label: str | None = None
    source_name = ""

    with input_column:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Input image</div>', unsafe_allow_html=True)
            st.caption("H&E-stained lung tissue · PNG, JPEG or TIFF")
            source_mode = st.radio(
                "Image source",
                ["Upload a patch", "Try a bundled test example"],
                horizontal=True,
                label_visibility="collapsed",
            )
            source = None
            if source_mode == "Upload a patch":
                source = st.file_uploader(
                    "Drop one image here or browse files",
                    type=["png", "jpg", "jpeg", "tif", "tiff"],
                    accept_multiple_files=False,
                    help="Supported formats: PNG, JPEG, TIFF. One image at a time.",
                )
                if source is not None:
                    source_name = source.name
                    original, error, original_mode, original_size = _load_pil_image(source)
                    if error:
                        st.error(error)
            else:
                examples = _bundled_test_examples()
                if examples:
                    selected_label = st.selectbox(
                        "Included held-out test samples",
                        list(examples),
                        format_func=lambda label: f"{label} · {CLASS_INFO[label]['title']}",
                    )
                    source = examples[selected_label]
                    reference_label = selected_label
                    source_name = source.name
                    original, error, original_mode, original_size = _load_pil_image(source)
                    if error:
                        st.error(error)
                    else:
                        st.caption(f"Included test patch · reference dataset label: {selected_label}")
                else:
                    st.info("Bundled test examples are unavailable. Use the upload option or check the data folders.")
            if original is not None:
                st.image(original, caption="Image preview", width="stretch")
                st.caption(f"{source_name} · {original_size[0]} × {original_size[1]} px · RGB model input")
                if original_mode in {"1", "L", "I", "F", "I;16"}:
                    st.warning("The source file is grayscale. Verify that it is a supported stained tissue patch.")
                rgb = np.asarray(original, dtype=np.uint8)
                saturation_proxy = float((rgb.max(axis=2).astype(float) - rgb.min(axis=2).astype(float)).mean())
                if saturation_proxy < 8:
                    st.warning("Very little color was detected. This simple check cannot confirm image type.")
            st.markdown(
                '<div class="privacy-note">🔒 Processed locally · image is not saved or transmitted.</div>',
                unsafe_allow_html=True,
            )

    scores = None
    predicted_name = None
    predicted_index = None
    model = None
    pixels = None
    prediction_error = None
    if original is not None and selected_model is not None:
        try:
            model = load_model_cached(str(selected_model))
            image_size = int(metadata.get("image_size", 128))
            pixels = load_rgb_array_from_pil(original, image_size)
            with st.spinner("Scoring this patch against the three dataset classes…"):
                scores = np.asarray(model.predict(pixels[None, ...], verbose=0)[0], dtype=float)
            names = metadata.get("class_names", CLASS_NAMES)
            if len(scores) != len(names) or len(names) != 3 or not np.all(np.isfinite(scores)):
                raise ValueError("The checkpoint output does not match the expected three-class mapping.")
            predicted_index = int(np.argmax(scores))
            predicted_name = str(names[predicted_index])
        except Exception as exc:
            prediction_error = str(exc)

    with result_column:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Model result</div>', unsafe_allow_html=True)
            if original is None:
                st.caption("Select an image to see its predicted dataset label and class scores.")
            elif selected_model is None:
                st.warning("No model checkpoint is available. Keep the artifacts/models folder with the project.")
            elif prediction_error:
                st.error(f"Could not make a prediction: {prediction_error}")
                st.caption("Check the TensorFlow installation and verify that the checkpoint belongs to this project.")
            elif scores is not None and predicted_name:
                info = CLASS_INFO.get(predicted_name, {"title": predicted_name, "color": "#58708a", "light": "#eff3f7"})
                safe_label = html.escape(predicted_name)
                safe_title = html.escape(info["title"])
                st.markdown(
                    f"""
                    <div class="top-match" style="background:{info['light']};border-color:{info['color']}44">
                      <div class="top-match-kicker" style="color:{info['color']}">Top-scoring dataset label</div>
                      <div class="top-match-title" style="color:{info['color']}">{safe_title}</div>
                      <div class="top-match-score" style="color:{info['color']}">{float(scores[predicted_index]):.1%} model score · {safe_label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if reference_label:
                    st.caption(f"Bundled example reference label: {reference_label}")
                st.markdown("#### Class scores")
                _render_score_rows(scores, list(metadata.get("class_names", CLASS_NAMES)))
                with st.expander("About these scores"):
                    st.caption("Scores compare the three dataset labels. They are not calibrated disease probabilities or a diagnosis.")
                st.caption(
                    f"Model: {selected_model.stem} · input: {metadata.get('image_size', 128)} px RGB"
                )

    if original is None or scores is None or model is None or pixels is None or predicted_name is None:
        return

    with st.expander("View Grad-CAM heatmap", expanded=False):
        st.caption("Highlights regions influencing the selected score; it is not a tumor outline or clinical explanation.")
        alpha = st.slider("Overlay strength", min_value=0.15, max_value=0.60, value=0.35, step=0.05)
        try:
            from pulmopath.gradcam import gradcam_heatmap, overlay_heatmap

            heatmap = gradcam_heatmap(
                model,
                pixels,
                predicted_index,
                metadata.get("gradcam_layer", "last_conv_features"),
            )
            overlay = overlay_heatmap(pixels, heatmap, alpha=alpha)
            original_small = Image.fromarray(pixels.astype(np.uint8))
            original_column, overlay_column = st.columns(2, gap="large")
            original_column.image(original_small, caption="Input", width="stretch")
            overlay_column.image(overlay, caption=f"Grad-CAM · {predicted_name}", width="stretch")
        except Exception as exc:
            st.warning(f"Grad-CAM unavailable: {exc}")


def _class_card(label: str) -> str:
    info = CLASS_INFO[label]
    return (
        '<div class="class-card">'
        f'<div class="class-code">{html.escape(label)}</div>'
        f'<div class="class-title">{html.escape(info["title"])}</div>'
        f'<div class="class-copy">{html.escape(info["description"])}</div>'
        "</div>"
    )


def _about_page(models: list[Path]) -> None:
    _hero(
        "About the research prototype",
        "A reproducible classroom workflow for comparing CNNs on labeled lung-tissue image patches.",
    )
    _notice()

    overview = st.columns(3, gap="medium")
    for column, value, label in zip(
        overview,
        ["2,400", "3", "128 × 128"],
        ["audited image patches", "dataset classes", "baseline input size"],
    ):
        with column:
            with st.container(border=True):
                st.metric(label, value)

    st.markdown("### The three dataset labels")
    class_columns = st.columns(3, gap="medium")
    for column, label in zip(class_columns, CLASS_NAMES):
        with column:
            st.markdown(_class_card(label), unsafe_allow_html=True)

    st.markdown("### How a prediction is produced")
    steps = st.columns(3, gap="medium")
    step_content = [
        ("01", "Prepare", "Convert the image to RGB and resize it to the model’s documented input size."),
        ("02", "Compare", "The CNN returns one model score for each of the three dataset categories."),
        ("03", "Inspect", "Optionally view Grad-CAM to examine spatial activations related to the top score."),
    ]
    for column, (number, title, copy) in zip(steps, step_content):
        with column:
            st.markdown(
                f"""
                <div class="explain-step">
                  <div class="explain-step-number">{number}</div>
                  <div class="explain-step-title">{html.escape(title)}</div>
                  <div class="explain-step-copy">{html.escape(copy)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown("### Data and provenance")
            st.markdown(
                f"Primary reference: [GeeksforGeeks CNN article]({GFG_ARTICLE_URL})  \n"
                f"Article-linked folder: [histology subset]({GFG_DATASET_URL})"
            )
            st.write(
                "The team confirms the archives came from the article-linked folder. The filename pattern suggests possible "
                "LC25000 lineage, but the link and the creator of this smaller subset are unverified. No explicit dataset "
                "license was found; reuse and redistribution rights remain unverified."
            )
            st.warning(
                "No reliable patient, slide, or source-image identifiers were found. Current results use a duplicate-group-separated "
                "split and are not patient-independent."
            )
    with right:
        with st.container(border=True):
            st.markdown("### Models in this project")
            if not models:
                st.info("No trained checkpoints are installed.")
            else:
                selected_stem = _selected_model_stem()
                for model_path in models:
                    model_metadata = _metadata_for(model_path)
                    model_title = "MobileNetV2 transfer learning" if model_path.stem == "mobilenetv2" else "Scratch CNN baseline"
                    selected_note = " · selected for demo using validation data" if model_path.stem == selected_stem else ""
                    st.markdown(
                        f"**{model_title}**{selected_note}  \n"
                        f"Checkpoint version {model_metadata.get('model_version', 'not recorded')}; "
                        f"selected epoch {model_metadata.get('selected_epoch_by_validation_loss', 'not recorded')}."
                    )
            st.write(
                "ImageNet weights are generic pretraining, not medical validation. The project has no external evaluation cohort. "
                "These outputs are educational research results only."
            )

    st.markdown("### Responsible use")
    st.info(
        "This demo is not a diagnostic device, screening service, clinical decision system, or substitute for a pathologist. "
        "Do not use its output for health decisions, treatment, or patient advice."
    )


def load_rgb_array_from_pil(image: Image.Image, size: int) -> np.ndarray:
    from PIL import Image as PILImage

    return np.asarray(
        image.convert("RGB").resize((size, size), PILImage.Resampling.BICUBIC),
        dtype=np.float32,
    )


def main() -> None:
    _inject_styles()
    page, selected_model, metadata = _sidebar()
    if page == "About & methods":
        _about_page(_available_models())
    else:
        _prediction_page(selected_model, metadata)


if __name__ == "__main__":
    main()
