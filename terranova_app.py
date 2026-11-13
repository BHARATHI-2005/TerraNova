
import json
import os
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
BASE_MODEL_PATH = MODEL_DIR / "Qwen2.5-VL-3B-Instruct"
ADAPTER_PATH = MODEL_DIR / "qwen25vl_eurosat_qlora"

BASE_MODEL_SOURCE = os.getenv(
    "TERRANOVA_BASE_MODEL",
    str(BASE_MODEL_PATH)
    if (BASE_MODEL_PATH / "config.json").exists()
    else "Qwen/Qwen2.5-VL-3B-Instruct",
)

# Load the adapter from an environment variable,
# a local folder, or Hugging Face.
ADAPTER_SOURCE = os.getenv("TERRANOVA_ADAPTER", "").strip()

if not ADAPTER_SOURCE:
    local_adapter_files_exist = (
        (ADAPTER_PATH / "adapter_config.json").exists()
        and (
            (ADAPTER_PATH / "adapter_model.safetensors").exists()
            or (ADAPTER_PATH / "adapter_model.bin").exists()
        )
    )

    if local_adapter_files_exist:
        ADAPTER_SOURCE = str(ADAPTER_PATH)
    else:
        ADAPTER_SOURCE = "BK0822/terranova-qwen25vl-eurosat"

if not ADAPTER_SOURCE:
    raise RuntimeError(
        "LoRA adapter not found. Set TERRANOVA_ADAPTER "
        "to a local adapter directory or Hugging Face repository ID."
    )

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

if DEVICE == "cpu":
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

# Load processor
processor = AutoProcessor.from_pretrained(
    ADAPTER_SOURCE,
    use_fast=True
)

# Load base model
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_SOURCE,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
    device_map=None if DEVICE == "cpu" else {"": DEVICE},
)

if DEVICE == "cuda":
    base_model.to("cuda")
else:
    base_model.to("cpu")

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_SOURCE,
    device_map=None if DEVICE == "cpu" else {"": DEVICE},
    low_cpu_mem_usage=False,
)

model.eval()
model.config.use_cache = True

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

# EuroSAT classes
app_classes = [
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

# Language choices
language_choices = [
    ("English", "en-US"),
    ("Arabic", "ar-SA"),
    ("Bengali", "bn-BD"),
    ("Chinese (Simplified)", "zh-CN"),
    ("Chinese (Traditional)", "zh-TW"),
    ("Czech", "cs-CZ"),
    ("Danish", "da-DK"),
]
