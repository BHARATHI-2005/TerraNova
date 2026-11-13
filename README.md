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

These large model files are intentionally excluded from this Git repository. The base model can also be downloaded from Hugging Face automatically. The adapter must be available locally or hosted in a Hugging Face model repository.

### 3. Start the app

```bash
python terranova_app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). The first launch may download several gigabytes of base-model weights if they are not already cached. GPU acceleration is recommended; CPU inference may be slow.

## Deploy

The app can run on a Gradio-compatible host with Python dependencies and enough memory for Qwen2.5-VL-3B. A GPU-backed host is recommended.

1. Run the Hugging Face upload section in the Colab notebook. It publishes the adapter and processor files to `BK0822/terranova-qwen25vl-eurosat` and includes the required Qwen license and attribution.
2. Deploy this repository as a Gradio app on a compatible host. This project is a Gradio app, not a Streamlit app; configure environment variables or secrets using your host's settings.
3. Configure these environment variables:

   | Variable | Required | Default | Description |
   | --- | --- | --- | --- |
   | `TERRANOVA_ADAPTER` | If no local adapter is present | Local `models/qwen25vl_eurosat_qlora` | Adapter directory or Hugging Face model repository ID; use `BK0822/terranova-qwen25vl-eurosat` after uploading |
   | `TERRANOVA_BASE_MODEL` | No | Local base model if present; otherwise `Qwen/Qwen2.5-VL-3B-Instruct` | Base model directory or Hugging Face model repository ID |
   | `HF_TOKEN` | Only if the adapter repository is private | Not set | Hugging Face access token; never commit it to the repository |
   | `GRADIO_SERVER_NAME` | No | `127.0.0.1` | Set to `0.0.0.0` on hosts that require external binding |
   | `PORT` | No | `7860` | Port supplied by the deployment host |

4. Ensure the hosting plan has enough RAM/VRAM and storage for the model download and inference.

The GitHub repository contains application code and setup instructions, not the model checkpoints. A deployment therefore needs access to a compatible adapter repository.

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
