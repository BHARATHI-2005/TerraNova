from __future__ import annotations

import html
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
CLASS_NAMES_PATH = ROOT / "config" / "class_names.json"
MODEL_CANDIDATES = (
    ROOT / "models" / "mobilenetv2_finetuned_best.keras",
    ROOT / "models" / "mobilenetv2_transfer_best.keras",
)
IMAGE_SIZE = (224, 224)
CHART_COLORS = ("#ff4fd8", "#a855f7", "#27d9ff")


st.set_page_config(
    page_title="Cosmic Land Explorer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_class_names() -> list[str]:
    try:
        names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not read the notebook-derived class configuration at "
            f"{CLASS_NAMES_PATH.relative_to(ROOT)}."
        ) from exc
    if not isinstance(names, list) or not names or not all(
        isinstance(name, str) and name for name in names
    ):
        raise RuntimeError("The class configuration must be a non-empty list of names.")
    return names


CLASS_NAMES = load_class_names()


def find_model_path() -> Path | None:
    return next((path for path in MODEL_CANDIDATES if path.is_file()), None)


@st.cache_resource(show_spinner="Loading your fine-tuned satellite model…")
def load_model(model_path: str) -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed in this Python environment. Install the "
            "project requirements using the instructions in README.md."
        ) from exc

    model = tf.keras.models.load_model(model_path)
    input_shape = model.input_shape
    output_shape = model.output_shape
    if isinstance(input_shape, list) or len(input_shape) != 4:
        raise ValueError(f"Expected one image input, but model input is {input_shape}.")
    if tuple(input_shape[1:]) != (*IMAGE_SIZE, 3):
        raise ValueError(
            f"Expected model input (None, 224, 224, 3), found {input_shape}."
        )
    if isinstance(output_shape, list) or len(output_shape) != 2:
        raise ValueError(f"Expected one classification output, found {output_shape}.")
    if output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"Model returns {output_shape[-1]} scores, but the class configuration "
            f"contains {len(CLASS_NAMES)} labels."
        )
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    import tensorflow as tf

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    resized = tf.image.resize(
        rgb,
        IMAGE_SIZE,
        method="bilinear",
        antialias=True,
    )
    # Training resized uint8 images and cast them to float32 without scaling.
    # The saved model performs the MobileNetV2 rescaling internally.
    batch = tf.expand_dims(tf.cast(resized, tf.float32), axis=0)
    return batch.numpy()


def to_probabilities(raw_output: Any, model: Any) -> np.ndarray:
    values = np.asarray(raw_output, dtype=np.float64).squeeze()
    if values.ndim != 1 or values.size != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} class scores, received shape "
            f"{np.asarray(raw_output).shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("The model returned non-finite prediction scores.")

    final_activation = getattr(getattr(model.layers[-1], "activation", None), "__name__", "")
    looks_like_probabilities = (
        np.all(values >= 0)
        and np.all(values <= 1)
        and math.isclose(float(values.sum()), 1.0, rel_tol=1e-4, abs_tol=1e-4)
    )
    if final_activation == "softmax" or looks_like_probabilities:
        probabilities = values
    else:
        shifted = values - np.max(values)
        exponentials = np.exp(shifted)
        probabilities = exponentials / exponentials.sum()
    return probabilities


def predict(image: Image.Image, model: Any) -> list[tuple[str, float]]:
    batch = preprocess_image(image)
    raw_output = model.predict(batch, verbose=0)
    probabilities = to_probabilities(raw_output, model)
    best_ids = np.argsort(probabilities)[::-1][: min(3, len(CLASS_NAMES))]
    return [(CLASS_NAMES[int(index)], float(probabilities[index])) for index in best_ids]


def render_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #f7f3ff;
            --muted: #d3caef;
            --violet: #b45cff;
            --pink: #ff4fd8;
            --cyan: #31ddff;
            --yellow: #ffe66d;
        }
        .stApp {
            color: var(--ink);
            background:
                radial-gradient(ellipse at 12% 9%, rgba(108, 48, 177, .46), transparent 34%),
                radial-gradient(ellipse at 90% 20%, rgba(10, 112, 173, .35), transparent 32%),
                radial-gradient(ellipse at 58% 92%, rgba(100, 34, 128, .28), transparent 36%),
                radial-gradient(1.5px 1.5px at 8% 32%, #fff 50%, transparent 100%),
                radial-gradient(2px 2px at 82% 9%, #ffe66d 50%, transparent 100%),
                radial-gradient(1px 1px at 66% 33%, #fff 50%, transparent 100%),
                radial-gradient(1.5px 1.5px at 94% 60%, #31ddff 50%, transparent 100%),
                linear-gradient(145deg, #09051b 0%, #100a2e 54%, #080719 100%);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stAppViewContainer"] { background: transparent; }
        [data-testid="stMain"] { color: var(--ink); }
        h1, h2, h3, p, label, li, span, div { color: inherit; }
        .block-container { max-width: 1180px; padding-top: 2rem; }
        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.8rem 2.5rem 2.5rem;
            border: 1px solid rgba(187, 120, 255, .42);
            border-radius: 30px;
            background: linear-gradient(115deg, rgba(33, 18, 72, .96), rgba(12, 35, 75, .91));
            box-shadow: 0 0 42px rgba(142, 65, 255, .22), inset 0 0 30px rgba(49, 221, 255, .06);
            animation: float-in .8s ease-out both;
        }
        .hero:after {
            content: "🪐";
            position: absolute;
            right: 4%;
            top: 4%;
            font-size: clamp(4rem, 11vw, 8rem);
            filter: drop-shadow(0 0 22px rgba(255, 79, 216, .5));
            animation: orbit 9s ease-in-out infinite;
        }
        .eyebrow {
            color: var(--yellow) !important;
            text-transform: uppercase;
            letter-spacing: .18em;
            font-size: .82rem;
            font-weight: 800;
        }
        .hero h1 {
            max-width: 720px;
            margin: .45rem 0 .65rem;
            color: #fff !important;
            font-size: clamp(2.4rem, 6vw, 4.5rem);
            line-height: 1.06;
            text-shadow: 0 0 24px rgba(180, 92, 255, .34);
        }
        .hero p { max-width: 680px; color: #e9e3ff !important; font-size: 1.12rem; }
        .hero-badges { display: flex; flex-wrap: wrap; gap: .6rem; margin-top: 1.4rem; }
        .badge {
            padding: .48rem .8rem;
            border: 1px solid rgba(49, 221, 255, .42);
            border-radius: 999px;
            background: rgba(15, 25, 58, .78);
            color: #c9f8ff !important;
            font-size: .88rem;
        }
        .section-title {
            color: #fff !important;
            font-size: 1.65rem;
            font-weight: 800;
            margin: 2rem 0 .8rem;
        }
        .glass-card {
            padding: 1.3rem 1.4rem;
            border: 1px solid rgba(180, 92, 255, .36);
            border-radius: 22px;
            background: linear-gradient(145deg, rgba(30, 21, 64, .94), rgba(13, 25, 54, .92));
            box-shadow: 0 12px 32px rgba(3, 2, 18, .24);
        }
        .prediction-card {
            border-color: rgba(49, 221, 255, .55);
            box-shadow: 0 0 26px rgba(49, 221, 255, .12);
        }
        .prediction-label { color: #a9efff !important; font-size: .92rem; font-weight: 700; }
        .prediction-class {
            margin: .35rem 0;
            color: #fff !important;
            font-size: clamp(1.8rem, 4vw, 2.9rem);
            font-weight: 900;
            overflow-wrap: anywhere;
            text-shadow: 0 0 22px rgba(49, 221, 255, .35);
        }
        .confidence { color: var(--yellow) !important; font-size: 1.3rem; font-weight: 800; }
        .chart-row { display: grid; grid-template-columns: minmax(130px, 1fr) 3fr 72px; gap: .75rem; align-items: center; margin: .95rem 0; }
        .chart-label { color: #f4edff !important; font-weight: 650; overflow-wrap: anywhere; }
        .chart-track { height: 14px; border-radius: 99px; background: rgba(255,255,255,.11); overflow: hidden; }
        .chart-fill { height: 100%; border-radius: inherit; box-shadow: 0 0 14px currentColor; }
        .chart-value { color: #fff !important; text-align: right; font-variant-numeric: tabular-nums; }
        .fine-print { color: var(--muted) !important; font-size: .9rem; }
        .mascot-line { color: #e5dcff !important; text-align: center; margin: 1.7rem 0 .6rem; }
        .footer {
            margin-top: 2rem;
            padding: 1.2rem .8rem;
            border-top: 1px solid rgba(180, 92, 255, .26);
            color: #d3caef !important;
            text-align: center;
            font-size: .9rem;
        }
        .stButton > button {
            min-height: 3.1rem;
            border: 0;
            border-radius: 999px;
            background: linear-gradient(100deg, #a846ff, #ed45d0 54%, #ff8a45);
            color: #fff !important;
            font-weight: 850;
            box-shadow: 0 0 22px rgba(207, 63, 241, .35);
            transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
        }
        .stButton > button:hover { transform: translateY(-2px); filter: brightness(1.12); box-shadow: 0 0 30px rgba(49, 221, 255, .38); }
        [data-testid="stFileUploader"] section {
            border: 1px dashed rgba(49, 221, 255, .65);
            border-radius: 20px;
            background: rgba(16, 23, 55, .7);
        }
        [data-testid="stFileUploader"] * { color: #f7f3ff !important; }
        [data-testid="stAlert"] { border-radius: 16px; }
        @keyframes float-in { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes orbit { 0%, 100% { transform: translateY(0) rotate(-8deg); } 50% { transform: translateY(-10px) rotate(7deg); } }
        @media (max-width: 650px) {
            .hero { padding: 2rem 1.25rem; }
            .hero:after { right: 2%; top: 0; opacity: .7; }
            .chart-row { grid-template-columns: minmax(100px, 1fr) 2fr 60px; gap: .45rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_chart(predictions: list[tuple[str, float]]) -> None:
    rows = [
        '<div class="glass-card"><div class="prediction-label">📊 TOP PREDICTIONS</div>'
    ]
    for index, (class_name, confidence) in enumerate(predictions):
        label = html.escape(class_name)
        width = max(0.0, min(100.0, confidence * 100))
        color = CHART_COLORS[index % len(CHART_COLORS)]
        rows.append(
            f'<div class="chart-row"><div class="chart-label">{label}</div>'
            f'<div class="chart-track"><div class="chart-fill" '
            f'style="width:{width:.2f}%;background:linear-gradient(90deg,{color},#ffe66d);'
            f'color:{color};"></div></div>'
            f'<div class="chart-value">{confidence:.1%}</div></div>'
        )
    rows.append("</div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


def image_fingerprint(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def main() -> None:
    render_styles()
    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">A tiny AI mission across a very big planet</div>
          <h1>Cosmic Land Explorer 🚀</h1>
          <p>Launch a satellite image and let our fine-tuned AI scout its land-cover
          landscape. One image, one predicted class, and a whole lot of cosmic curiosity.</p>
          <div class="hero-badges">
            <span class="badge">🛰️ MobileNetV2</span>
            <span class="badge">🌍 EuroSAT RGB</span>
            <span class="badge">✨ 10 land-cover classes</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title">🪐 Prepare your image</div>', unsafe_allow_html=True)

    if "upload_generation" not in st.session_state:
        st.session_state.upload_generation = 0
    upload_key = f"satellite_upload_{st.session_state.upload_generation}"
    uploaded_file = st.file_uploader(
        "Choose a satellite image",
        type=("jpg", "jpeg", "png"),
        help="Upload one JPG, JPEG, or PNG image.",
        key=upload_key,
    )

    model_path = find_model_path()
    if model_path is None:
        st.warning(
            "The trained model is not in this workspace yet. The notebook saved "
            "`models/mobilenetv2_finetuned_best.keras` in Google Drive. Copy that "
            "file into the `models` folder beside `app.py` to enable predictions."
        )

    image = None
    image_bytes = None
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        fingerprint = image_fingerprint(image_bytes)
        try:
            image = Image.open(uploaded_file).convert("RGB")
            image.load()
        except (UnidentifiedImageError, OSError, ValueError):
            st.error("That file could not be opened as an image. Try a valid JPG, JPEG, or PNG.")
        if image is not None:
            left, right = st.columns([1.05, 1], gap="large")
            with left:
                st.image(image, caption="Your satellite image preview", width="stretch")
            with right:
                st.markdown(
                    '<div class="glass-card"><span class="prediction-label">MISSION CHECKLIST</span>'
                    '<p style="color:#f7f3ff;margin:.7rem 0 0">🌟 Image received<br>'
                    '🛰️ RGB view prepared<br>🧭 Ready for a single-class land-cover prediction</p></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("")
            classify_col, clear_col, _ = st.columns([1.25, 1, 2.8])
            with classify_col:
                classify = st.button(
                    "Classify Image 🚀",
                    type="primary",
                    width="stretch",
                    disabled=model_path is None,
                )
            with clear_col:
                clear = st.button("Clear image ✨", width="stretch")

            if clear:
                st.session_state.upload_generation += 1
                st.session_state.pop("prediction_result", None)
                st.rerun()

            if classify and model_path is not None:
                try:
                    model = load_model(str(model_path))
                    with st.spinner("Scanning the landscape…"):
                        ranked_predictions = predict(image, model)
                    st.session_state.prediction_result = {
                        "fingerprint": fingerprint,
                        "predictions": ranked_predictions,
                    }
                except Exception as exc:
                    st.error(
                        "The image could not be classified. Check that the model file "
                        "is valid and that the required TensorFlow environment is installed."
                    )
                    with st.expander("Technical details"):
                        st.code(f"{type(exc).__name__}: {exc}")

            saved = st.session_state.get("prediction_result")
            if (
                saved
                and saved["fingerprint"] == fingerprint
                and saved["predictions"]
            ):
                top_class, confidence = saved["predictions"][0]
                result_col, chart_col = st.columns([1, 1.2], gap="large")
                with result_col:
                    st.markdown(
                        f'<div class="glass-card prediction-card">'
                        f'<div class="prediction-label">🌠 MOST LIKELY CLASS</div>'
                        f'<div class="prediction-class">{html.escape(top_class)}</div>'
                        f'<div class="confidence">{confidence:.1%} confidence</div>'
                        f'<p class="fine-print">One image, one predicted land-cover class.</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with chart_col:
                    display_chart(saved["predictions"])
                st.info(
                    "Confidence is the model’s score, not a guarantee of correctness. "
                    "Image quality, resolution, and visually similar landscapes can affect predictions."
                )
    else:
        st.markdown(
            '<div class="glass-card"><span class="prediction-label">🧑‍🚀 Your co-pilot is ready!</span>'
            '<p style="color:#e9e3ff;margin:.55rem 0 0">Drop in a satellite snapshot to begin. '
            'Supported formats: JPG, JPEG, and PNG.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="mascot-line">👩‍🚀 &nbsp; 🌙 &nbsp; 🛸 &nbsp; ✨ &nbsp; 🛰️</div>'
        '<div class="footer">Satellite Image Classification for Smart Land Monitoring'
        '<br><span class="fine-print">Built for curious explorers of our changing planet 🌎</span></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
