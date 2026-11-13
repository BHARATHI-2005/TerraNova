import os

import pandas as pd
import streamlit as st
from huggingface_hub.errors import HfHubHTTPError
from PIL import Image
from streamlit.errors import StreamlitSecretNotFoundError

st.set_page_config(page_title="TerraNova | Satellite Image Intelligence", layout="wide")

try:
    secrets = dict(st.secrets)
except StreamlitSecretNotFoundError:
    secrets = {}

for secret_name in ("TERRANOVA_ADAPTER", "TERRANOVA_BASE_MODEL", "HF_TOKEN"):
    if secret_name in secrets and not os.getenv(secret_name):
        os.environ[secret_name] = str(secrets[secret_name])

import terranova_app as core


def add_history(record):
    st.session_state.history = core.append_history(
        st.session_state.history,
        record,
    )


if "history" not in st.session_state:
    st.session_state.history = []
if "single_result" not in st.session_state:
    st.session_state.single_result = None
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []

st.title("TerraNova / Image Intelligence")
st.caption("Satellite land-cover analysis with Qwen2.5-VL and EuroSAT classes.")
st.info("Built with Qwen. Model-generated reports are interpretations, not verified ground truth.")
adapter_setting = os.getenv(
    "TERRANOVA_ADAPTER",
    "BK0822/terranova-qwen25vl-eurosat",
).strip()
if adapter_setting.casefold() == "bk0822/teranova":
    st.warning(
        "`TERRANOVA_ADAPTER` points to `BK0822/teranova`, which is a dataset repo. "
        "Publish the trained adapter as the model repo "
        "`BK0822/terranova-qwen25vl-eurosat`, then update this Streamlit Secret."
    )
else:
    st.caption(
        "The fine-tuned adapter must be available at "
        f"`{adapter_setting}`. Make the Hugging Face repository public or add "
        "`HF_TOKEN` to Streamlit Secrets."
    )


def show_model_loading_error(error):
    status_code = getattr(getattr(error, "response", None), "status_code", None)
    if status_code in (401, 403):
        st.error(
            "Hugging Face denied access to the model. Confirm the adapter repository "
            "exists and is public, or add a valid read token as `HF_TOKEN` in "
            "Streamlit Cloud → App settings → Secrets. If the repository is private, "
            "also set `TERRANOVA_ADAPTER` to its exact repository ID."
        )
    else:
        st.error(f"Could not download the model from Hugging Face: {error}")

with st.sidebar:
    st.header("Analysis settings")
    language_label = st.selectbox(
        "Report language",
        [label for label, _ in core.language_choices],
    )
    language = dict(core.language_choices)[language_label]
    max_tokens = st.slider("Maximum output length", 30, 160, 80, step=10)
    temperature = st.slider("Sampling temperature", 0.0, 1.0, 0.0, step=0.05)

single_tab, batch_tab, history_tab = st.tabs(
    ["Single image", "Batch classification", "Prediction history"]
)

with single_tab:
    uploaded_image = st.file_uploader(
        "Upload a satellite image",
        type=["png", "jpg", "jpeg", "webp", "tif", "tiff"],
        key="single_image",
    )
    if uploaded_image is not None:
        st.image(uploaded_image, caption=uploaded_image.name, use_container_width=True)

    if st.button("Analyze image", type="primary"):
        if uploaded_image is None:
            st.warning("Upload a satellite image to begin.")
        else:
            try:
                uploaded_image.seek(0)
                image = Image.open(uploaded_image).convert("RGB")
                with st.spinner("Loading model and analyzing image…"):
                    predicted_class, confidence, report = core.run_image_inference(
                        image, language, max_tokens, temperature
                    )
                result = {
                    "image_name": uploaded_image.name,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "report": report,
                }
                st.session_state.single_result = result
                add_history(result)
            except core.gr.Error as error:
                st.error(str(error))
            except core.ModelRepositoryAccessError as error:
                st.error(str(error))
            except HfHubHTTPError as error:
                show_model_loading_error(error)

    if st.session_state.single_result:
        result = st.session_state.single_result
        st.subheader("Analysis")
        st.metric("Predicted class", result["predicted_class"])
        st.metric("Confidence", result["confidence"])
        st.write(result["report"])
        st.download_button(
            "Download report",
            data=result["report"],
            file_name="terranova-report.txt",
            mime="text/plain",
        )

with batch_tab:
    uploaded_images = st.file_uploader(
        "Upload multiple satellite images",
        type=["png", "jpg", "jpeg", "webp", "tif", "tiff"],
        accept_multiple_files=True,
        key="batch_images",
    )
    if st.button("Classify batch", type="primary"):
        if not uploaded_images:
            st.warning("Upload one or more images to classify.")
        else:
            results = []
            progress = st.progress(0, text="Preparing batch…")
            try:
                for index, uploaded in enumerate(uploaded_images):
                    uploaded.seek(0)
                    image = Image.open(uploaded).convert("RGB")
                    with st.spinner(f"Analyzing {uploaded.name}…"):
                        predicted_class, confidence, report = core.run_image_inference(
                            image, language, max_tokens, temperature
                        )
                    result = {
                        "image_name": uploaded.name,
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "report": report,
                    }
                    results.append(result)
                    add_history(result)
                    progress.progress(
                        (index + 1) / len(uploaded_images),
                        text=f"Analyzed {index + 1} of {len(uploaded_images)} images",
                    )
                st.session_state.batch_results = results
            except core.gr.Error as error:
                st.error(str(error))
            except core.ModelRepositoryAccessError as error:
                st.error(str(error))
            except HfHubHTTPError as error:
                show_model_loading_error(error)

    if st.session_state.batch_results:
        st.dataframe(
            pd.DataFrame(st.session_state.batch_results),
            use_container_width=True,
            hide_index=True,
        )

with history_tab:
    if st.session_state.history:
        st.dataframe(
            pd.DataFrame(st.session_state.history),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Your predictions will appear here during this session.")
    if st.button("Clear history"):
        st.session_state.history = []
        st.session_state.single_result = None
        st.session_state.batch_results = []
        st.rerun()
