"""
TerraNova core module
Qwen2.5-VL + PEFT/LoRA inference for EuroSAT satellite-image classification.

This module is imported by streamlit_app.py. It intentionally does not launch
a Gradio interface; the user-facing interface is Streamlit.
"""

import os
from pathlib import Path
from threading import Lock

import gradio as gr
import torch
from huggingface_hub.errors import HfHubHTTPError
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
BASE_MODEL_PATH = MODEL_DIR / "Qwen2.5-VL-3B-Instruct"
ADAPTER_PATH = MODEL_DIR / "qwen25vl_eurosat_qlora"
BASE_MODEL_SOURCE = os.getenv("TERRANOVA_BASE_MODEL") or (
    str(BASE_MODEL_PATH)
    if (BASE_MODEL_PATH / "config.json").is_file()
    else "Qwen/Qwen2.5-VL-3B-Instruct"
)

ADAPTER_SOURCE = os.getenv("TERRANOVA_ADAPTER") or (
    str(ADAPTER_PATH)
    if (
        (ADAPTER_PATH / "adapter_config.json").is_file()
        and (
            (ADAPTER_PATH / "adapter_model.safetensors").is_file()
            or (ADAPTER_PATH / "adapter_model.bin").is_file()
        )
    )
    else "BK0822/terranova-qwen25vl-eurosat"
)

HF_TOKEN = os.getenv("HF_TOKEN") or None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

if DEVICE == "cpu":
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

processor = None
model = None
_model_load_lock = Lock()


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
]

app_classes = EUROSAT_CLASSES


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
                ADAPTER_SOURCE,
                use_fast=True,
                token=HF_TOKEN,
            )
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                BASE_MODEL_SOURCE,
                torch_dtype=DTYPE,
                low_cpu_mem_usage=True,
                device_map=None if DEVICE == "cpu" else {"": DEVICE},
                token=HF_TOKEN,
            )
            base_model.to(DEVICE)
            loaded_model = PeftModel.from_pretrained(
                base_model,
                ADAPTER_SOURCE,
                device_map=None if DEVICE == "cpu" else {"": DEVICE},
                low_cpu_mem_usage=False,
                token=HF_TOKEN,
            )
        except HfHubHTTPError as error:
            raise ModelRepositoryAccessError(
                f"Cannot access Hugging Face model '{ADAPTER_SOURCE}'. Verify the "
                "repository ID and ensure it is public or that HF_TOKEN has read access."
            ) from error
        except OSError as error:
            if not _caused_by_hub_error(error):
                raise
            raise ModelRepositoryAccessError(
                f"Cannot access Hugging Face model '{ADAPTER_SOURCE}'. Verify the "
                "repository ID and ensure it is public or that HF_TOKEN has read access."
            ) from error

        loaded_model.eval()
        loaded_model.config.use_cache = True
        if hasattr(loaded_model, "enable_input_require_grads"):
            loaded_model.enable_input_require_grads()

        processor = loaded_processor
        model = loaded_model
        return model, processor


def run_image_inference(
    image,
    language: str,
    max_tokens: int = 80,
    temperature: float = 0.0,
):
    if image is None:
        raise gr.Error("Upload a satellite image to begin.")

    active_model, active_processor = load_model()
    image = image.convert("RGB")
    image.thumbnail((448, 448), Image.Resampling.BILINEAR)
    prompt = (
        "Analyze this satellite image. First select its dominant land-cover class "
        f"from exactly this list: {', '.join(app_classes)}. Then write at most two concise sentences "
        f"in {language}, naming the class and one or two visible clues. "
        "Base the report only on visible image evidence. Do not infer an exact "
        "location, date, sensor, resolution, or facts not shown in the image. "
        "Mention that this is an image interpretation, not verified ground truth."
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    formatted_prompt = active_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = active_processor(
        text=[formatted_prompt],
        images=[image],
        return_tensors="pt",
    )
    inputs = {
        key: value.to(DEVICE) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }

    generation_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0.0,
    }
    if temperature > 0.0:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        generated = active_model.generate(**inputs, **generation_kwargs)

    prompt_length = inputs["input_ids"].shape[1]
    report = active_processor.batch_decode(
        generated[:, prompt_length:],
        skip_special_tokens=True,
    )[0].strip()
    if not report:
        raise gr.Error("The model returned an empty report. Please try another image.")

    predicted_class = next(
        (name for name in app_classes if name.lower() in report.lower()),
        "Unknown",
    )
    if predicted_class == "Unknown":
        confidence = "Low"
    elif report.lower().startswith(predicted_class.lower()):
        confidence = "High"
    else:
        confidence = "Medium"
    return predicted_class, confidence, report


def append_history(history_state, record):
    history = list(history_state or [])
    history.insert(0, record)
    return history[:50]
