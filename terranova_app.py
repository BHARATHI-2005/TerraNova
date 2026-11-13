import json
import os
import tempfile
from pathlib import Path
from threading import Lock

import gradio as gr
import pandas as pd
import torch
from huggingface_hub.errors import HfHubHTTPError
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Read Streamlit Cloud Secrets when available; otherwise use environment variables.
def get_secret(name: str, default: str = "") -> str:
    try:
        import streamlit as st
        value = st.secrets.get(name, default)
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.getenv(name, default).strip()


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
BASE_MODEL_PATH = MODEL_DIR / "Qwen2.5-VL-3B-Instruct"
ADAPTER_PATH = MODEL_DIR / "qwen25vl_eurosat_qlora"
BASE_MODEL_SOURCE = get_secret("TERRANOVA_BASE_MODEL") or (
    str(BASE_MODEL_PATH)
    if (BASE_MODEL_PATH / "config.json").exists()
    else "Qwen/Qwen2.5-VL-3B-Instruct"
)
ADAPTER_SOURCE = get_secret("TERRANOVA_ADAPTER") or (
    str(ADAPTER_PATH)
    if (
        (ADAPTER_PATH / "adapter_config.json").exists()
        and (
            (ADAPTER_PATH / "adapter_model.safetensors").exists()
            or (ADAPTER_PATH / "adapter_model.bin").exists()
        )
    )
    else "BK0822/teranova"
)
HF_TOKEN = get_secret("HF_TOKEN") or None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

if DEVICE == "cpu":
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

processor = None
model = None
_model_load_lock = Lock()


class ModelRepositoryAccessError(RuntimeError):
    pass


def _caused_by_hub_error(error: BaseException) -> bool:
    seen = set()
    current = error.__cause__ or error.__context__
    while current is not None and id(current) not in seen:
        if isinstance(current, HfHubHTTPError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def load_model():
    global processor, model

    if model is not None and processor is not None:
        return model, processor

    with _model_load_lock:
        if model is not None and processor is not None:
            return model, processor

        try:
            loaded_processor = AutoProcessor.from_pretrained(
                BASE_MODEL_SOURCE,
                use_fast=True,
                token=HF_TOKEN,
            )
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                BASE_MODEL_SOURCE,
                torch_dtype=DTYPE,
                low_cpu_mem_usage=True,
                device_map=None if DEVICE == "cpu" else {"": DEVICE},
                token=HF_TOKEN,
