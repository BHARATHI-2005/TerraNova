"""
TerraNova core module
Qwen2.5-VL + PEFT/LoRA inference for EuroSAT satellite-image classification.

This module is imported by streamlit_app.py. It intentionally does not launch
a Gradio interface; the user-facing interface is Streamlit.
"""

import os
import re
from pathlib import Path
from threading import Lock
from typing import Any

import gradio as gr
import torch
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_secret(name: str, default: str = "") -> str:
    """Read a Streamlit Secret when running on Streamlit Cloud, else env var."""
    try:
        import streamlit as st

        value = st.secrets.get(name, default)
        if value:
            return str(value).strip()
    except Exception:
        # Streamlit may not be installed in a local inference-only environment,
        # or no secrets.toml may exist. Environment variables remain available.
        pass

    return os.getenv(name, default).strip()


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
BASE_MODEL_PATH = MODEL_DIR / "Qwen2.5-VL-3B-Instruct"
ADAPTER_PATH = MODEL_DIR / "qwen25vl_eurosat_qlora"

BASE_MODEL_SOURCE = get_secret("TERRANOVA_BASE_MODEL") or (
    str(BASE_MODEL_PATH)
    if (BASE_MODEL_PATH / "config.json").is_file()
    else "Qwen/Qwen2.5-VL-3B-Instruct"
)

ADAPTER_SOURCE = get_secret("TERRANOVA_ADAPTER") or (
    str(ADAPTER_PATH)
    if (
        (ADAPTER_PATH / "adapter_config.json").is_file()
        and (
            (ADAPTER_PATH / "adapter_model.safetensors").is_file()
            or (ADAPTER_PATH / "adapter_model.bin").is_file()
        )
    )
    else "BK0822/teranova"
)

HF_TOKEN = get_secret("HF_TOKEN") or None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

if DEVICE == "cpu":
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


# EuroSAT RGB class labels
EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# The second value is the language name passed into the prompt.
language_choices = [
    ("English", "English"),
    ("Hindi", "Hindi"),
    ("Telugu", "Telugu"),
    ("Tamil", "Tamil"),
    ("Kannada", "Kannada"),
