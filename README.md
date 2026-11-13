<div align="center">
  <img src="assets/terranova-mark.svg" width="112" alt="TerraNova satellite globe mark">
  <h1>TerraNova</h1>
  <p><strong>Satellite image intelligence, powered by a fine-tuned vision-language model.</strong></p>
  <p>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
    <img alt="Gradio" src="https://img.shields.io/badge/UI-Gradio-orange?logo=gradio&logoColor=white">
    <img alt="Model" src="https://img.shields.io/badge/Model-Qwen2.5--VL-6EDFC8">
    <img alt="Task" src="https://img.shields.io/badge/Task-EuroSAT%20land%20cover-8AA63B">
  </p>
</div>

---

TerraNova interprets uploaded satellite imagery and produces concise land-cover reports. Its Gradio interface supports single-image analysis, batch classification, multilingual reports, and a session prediction history.

> **Built with Qwen.** Model-generated reports are image interpretations, not verified ground truth.

## What it does

- Classifies imagery into the 10 EuroSAT land-cover classes.
- Generates short reports in more than 40 selectable languages.
- Analyzes one image or a batch of images.
- Keeps a session history and lets you download single-image reports.
- Runs locally or can be configured for a model hosted on Hugging Face.

## Preview

The app opens with three workspaces: **Single image**, **Batch classification**, and **Prediction history**.

## Quick start

### 1. Install dependencies

Python 3.10 or newer is recommended. Install a PyTorch build suitable for your hardware by following the [PyTorch installation guide](https://pytorch.org/get-started/locally/), then install the app dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

python -m pip install -r requirements.txt
```

### 2. Provide model assets

For a local run, place the Qwen2.5-VL-3B-Instruct model and the trained adapter in:

```text
models/
├── Qwen2.5-VL-3B-Instruct/
└── qwen25vl_eurosat_qlora/
```

These large model files are intentionally excluded from this Git repository. The base model can also be downloaded from Hugging Face automatically. The adapter is loaded from the local folder when present; otherwise the app uses `BK0822/teranova`.

### 3. Start the app

```bash
streamlit run app.py
```

Open the URL printed by Streamlit. The model loads when you run the first analysis, and may download several gigabytes of base-model weights. GPU acceleration is recommended; CPU inference may be slow. To run the original Gradio interface locally, use `python terranova_app.py`.

## Deploy

The repository includes `app.py` as the Streamlit Cloud entry point (it launches `streamlit_app.py`) and a Gradio entry point (`terranova_app.py`). For Streamlit Community Cloud, select `app.py` as the app file. A GPU-backed host is recommended.

1. Run the Hugging Face upload section in the Colab notebook. It publishes the adapter and processor files to `BK0822/teranova` and includes the required Qwen license and attribution.
2. Deploy this repository as a Streamlit app using `app.py`.
3. In Streamlit Cloud, add these values under **App settings → Secrets**. The app reads these secrets before loading the model:

   ```toml
   TERRANOVA_ADAPTER = "BK0822/teranova"
   # Add only if the model repository is private:
   HF_TOKEN = "hf_your_read_token"
   ```

   | Variable | Required | Default | Description |
   | --- | --- | --- | --- |
   | `TERRANOVA_ADAPTER` | No, if using the configured public adapter | Local adapter if present; otherwise `BK0822/teranova` | Adapter directory or Hugging Face model repository ID |
   | `TERRANOVA_BASE_MODEL` | No | Local base model if present; otherwise `Qwen/Qwen2.5-VL-3B-Instruct` | Base model directory or Hugging Face model repository ID |
   | `HF_TOKEN` | Only if the adapter repository is private | Not set | Hugging Face read token; keep it in Secrets and never commit it |

4. Ensure the hosting plan has enough RAM and storage for the model download and inference. The 3B model may exceed the memory available on small or free CPU-only instances.

The GitHub repository contains application code and setup instructions, not the model checkpoints. A deployment therefore needs access to a compatible adapter repository. If you prefer the Gradio interface, use a Gradio-compatible host and the `terranova_app.py` entry point instead.

If the adapter repository is private, add both `TERRANOVA_ADAPTER` and `HF_TOKEN` to Streamlit Secrets. The app passes `HF_TOKEN` to the Hugging Face model loaders and displays an actionable message for access-denied responses. A `401` or repository access error means the repo is private or unavailable, or the token lacks read access; verify the exact repo ID and visibility on Hugging Face.

## Training

[`TerraNova_Qwen2_5_VL_QLoRA_EuroSAT.ipynb`](TerraNova_Qwen2_5_VL_QLoRA_EuroSAT.ipynb) documents QLoRA fine-tuning with the `timm/eurosat-rgb` dataset. The notebook is designed for Google Colab with a GPU; training data, checkpoints, and model weights are stored separately and are not committed here.

## Model, data, and limitations

- Base model: [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- Training dataset: [EuroSAT RGB](https://huggingface.co/datasets/timm/eurosat-rgb)
- The Qwen model is provided under the [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE), which limits use to non-commercial purposes unless a separate license is obtained. Review the license before using or redistributing the base model or derivatives.
- Model predictions can be incorrect or incomplete. Do not treat generated descriptions as authoritative geospatial data or use them as a substitute for expert review.
- This repository does not include a project-wide software license. The model and dataset have their own terms; consult those terms independently.

## Repository layout

```text
.
├── assets/                                  # Project branding
├── terranova_app.py                         # Gradio application
├── requirements.txt                         # Python dependencies
└── TerraNova_Qwen2_5_VL_QLoRA_EuroSAT.ipynb # Fine-tuning notebook
```

## Acknowledgements

TerraNova is built with [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [EuroSAT](https://github.com/phelber/EuroSAT), [Hugging Face Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), and [Gradio](https://github.com/gradio-app/gradio).
