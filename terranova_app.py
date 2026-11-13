import json
import os
import tempfile
from pathlib import Path
from threading import Lock

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
BASE_MODEL_SOURCE = os.getenv("TERRANOVA_BASE_MODEL") or (
    str(BASE_MODEL_PATH)
    if (BASE_MODEL_PATH / "config.json").exists()
    else "Qwen/Qwen2.5-VL-3B-Instruct"
)
ADAPTER_SOURCE = os.getenv("TERRANOVA_ADAPTER") or (
    str(ADAPTER_PATH)
    if (ADAPTER_PATH / "adapter_model.safetensors").exists()
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


def load_model():
    global processor, model

    if model is not None and processor is not None:
        return model, processor

    with _model_load_lock:
        if model is not None and processor is not None:
            return model, processor

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
        loaded_model.eval()
        loaded_model.config.use_cache = True
        if hasattr(loaded_model, "enable_input_require_grads"):
            loaded_model.enable_input_require_grads()

        processor = loaded_processor
        model = loaded_model
        return model, processor

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

language_choices = [
    ("English", "en-US"),
    ("Arabic", "ar-SA"),
    ("Bengali", "bn-BD"),
    ("Chinese (Simplified)", "zh-CN"),
    ("Chinese (Traditional)", "zh-TW"),
    ("Czech", "cs-CZ"),
    ("Danish", "da-DK"),
    ("Dutch", "nl-NL"),
    ("Filipino", "fil-PH"),
    ("Finnish", "fi-FI"),
    ("French", "fr-FR"),
    ("German", "de-DE"),
    ("Greek", "el-GR"),
    ("Gujarati", "gu-IN"),
    ("Hebrew", "he-IL"),
    ("Hindi", "hi-IN"),
    ("Hungarian", "hu-HU"),
    ("Indonesian", "id-ID"),
    ("Italian", "it-IT"),
    ("Japanese", "ja-JP"),
    ("Kannada", "kn-IN"),
    ("Korean", "ko-KR"),
    ("Malay", "ms-MY"),
    ("Malayalam", "ml-IN"),
    ("Marathi", "mr-IN"),
    ("Norwegian", "no-NO"),
    ("Persian", "fa-IR"),
    ("Polish", "pl-PL"),
    ("Portuguese", "pt-BR"),
    ("Punjabi", "pa-IN"),
    ("Romanian", "ro-RO"),
    ("Russian", "ru-RU"),
    ("Spanish", "es-ES"),
    ("Swahili", "sw-KE"),
    ("Swedish", "sv-SE"),
    ("Tamil", "ta-IN"),
    ("Telugu", "te-IN"),
    ("Thai", "th-TH"),
    ("Turkish", "tr-TR"),
    ("Ukrainian", "uk-UA"),
    ("Urdu", "ur-PK"),
    ("Vietnamese", "vi-VN"),
]


def normalize_prediction(report_text: str):
    text = report_text.strip()
    if not text:
        return "Unknown", "Low"
    lowered = text.lower()
    for class_name in app_classes:
        if class_name.lower() in lowered:
            return class_name, "High" if lowered.startswith(class_name.lower()) else "Medium"
    for class_name in app_classes:
        if class_name.lower() in lowered:
            return class_name, "Medium"
    return "Unknown", "Low"


def write_report_file(report_text: str) -> str:
    file_handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", delete=False)
    with file_handle:
        file_handle.write(report_text)
    return file_handle.name


def run_image_inference(image, language: str, max_tokens: int = 80, temperature: float = 0.0):
    if image is None:
        raise gr.Error("Upload a satellite image to begin.")

    active_model, active_processor = load_model()
    image = image.convert("RGB")
    image.thumbnail((448, 448), Image.Resampling.BILINEAR)
    prompt = (
        "Analyze this satellite image. First select its dominant land-cover class "
        f"from exactly this list: {', '.join(app_classes)}. Then write at most two concise sentences "
        f"in the language with locale {language}, naming the class and one or two visible clues. "
        "Base the report only on visible image evidence. "
        "Do not infer an exact location, date, sensor, resolution, or facts not shown in the image. "
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
    inputs = active_processor(text=[formatted_prompt], images=[image], return_tensors="pt")
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

    predicted_class, confidence = normalize_prediction(report)
    return predicted_class, confidence, report


def append_history(history_state, record):
    history = list(history_state or [])
    history.insert(0, record)
    return history[:50]


def history_dataframe(history):
    if not history:
        return pd.DataFrame(columns=["image_name", "predicted_class", "confidence", "report"])
    frame = pd.DataFrame(history)
    return frame[["image_name", "predicted_class", "confidence", "report"]]


def handle_single_prediction(image, language, max_tokens, temperature, history_state):
    predicted_class, confidence, report = run_image_inference(image, language, max_tokens, temperature)
    image_name = getattr(image, "name", "uploaded-image") or "uploaded-image"
    record = {
        "image_name": image_name,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "report": report,
    }
    updated_history = append_history(history_state, record)
    report_path = write_report_file(report)
    return (
        predicted_class,
        confidence,
        report,
        history_dataframe(updated_history),
        updated_history,
        report_path,
    )


def handle_batch_prediction(files, language, max_tokens, temperature, history_state):
    if files is None:
        raise gr.Error("Upload one or more images to classify.")

    normalized_files = files if isinstance(files, list) else [files]
    rows = []
    working_history = list(history_state or [])
    for file_ref in normalized_files:
        if isinstance(file_ref, str):
            path = Path(file_ref)
        elif hasattr(file_ref, "name"):
            path = Path(str(file_ref.name))
        else:
            path = Path(str(file_ref))

        if not path.exists():
            continue

        image = Image.open(path).convert("RGB")
        predicted_class, confidence, report = run_image_inference(image, language, max_tokens, temperature)
        response_entry = {
            "image_name": path.name,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "report": report,
        }
        rows.append(response_entry)
        working_history = append_history(working_history, response_entry)

    if not rows:
        raise gr.Error("No valid images were found in the uploaded batch.")

    summary = " | ".join(f"{row['image_name']}: {row['predicted_class']} ({row['confidence']})" for row in rows)
    return summary, rows, history_dataframe(working_history), working_history


satellite_css = """
:root {
    --bg: #0f1718;
    --panel: #122022;
    --panel-2: #162c2d;
    --card: #1b3032;
    --line: #32575a;
    --text: #edf5f3;
    --muted: #b8c7c1;
    --primary: #d0f160;
    --secondary: #6edfc8;
}
.gradio-container {
    background: linear-gradient(180deg, #0d1517, #112123);
    color: var(--text);
}
#mission-header {
    padding: 16px 8px 18px;
    border-bottom: 1px solid var(--line);
    margin-bottom: 18px;
}
#mission-header h1 {
    margin: 0;
    font-size: 2.1rem;
    letter-spacing: 0.08em;
    color: #f5f8ed;
}
#mission-header p {
    margin: 8px 0 0;
    color: var(--muted);
}
#info-card {
    background: rgba(17, 35, 37, 0.9);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1rem 1.1rem;
}
.block-label {
    color: var(--secondary);
    font-weight: 600;
}
#report-output {
    min-height: 240px;
}
button.primary {
    background: linear-gradient(135deg, var(--primary), #7bd6b5);
    color: #0a1717;
    font-weight: 700;
}
"""

with gr.Blocks(title="TerraNova | Satellite Image Intelligence", css=satellite_css) as app:
    gr.HTML("""
    <header id="mission-header">
      <h1>TERRANOVA <span style="color:#d0f160">/</span> IMAGE INTELLIGENCE</h1>
      <p>Satellite land-cover understanding · Qwen2.5-VL · EuroSAT classification</p>
    </header>
    """)
    shared_history_state = gr.State(value=[])

    with gr.Tabs():
        with gr.TabItem("Single image"):
            with gr.Row():
                with gr.Column(scale=5):
                    image_input = gr.Image(label="Satellite image", type="pil", sources=["upload"], height=360)
                    language_input = gr.Dropdown(choices=language_choices, value="en-US", label="Report language", allow_custom_value=True)
                    max_tokens_input = gr.Slider(minimum=30, maximum=160, value=80, step=10, label="Max output length")
                    temperature_input = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.05, label="Sampling temperature")
                    analyze_button = gr.Button("Analyze image", variant="primary")
                with gr.Column(scale=6):
                    prediction_output = gr.Textbox(label="Predicted class", lines=1, interactive=False)
                    confidence_output = gr.Textbox(label="Confidence", lines=1, interactive=False)
                    report_output = gr.Textbox(label="Image report", lines=12, max_lines=20, interactive=False)
                    download_output = gr.File(label="Download report")

        with gr.TabItem("Batch classification"):
            with gr.Row():
                with gr.Column(scale=5):
                    batch_files = gr.File(label="Upload multiple satellite images", file_count="multiple", type="filepath")
                    batch_language = gr.Dropdown(choices=language_choices, value="en-US", label="Report language", allow_custom_value=True)
                    batch_max_tokens = gr.Slider(minimum=30, maximum=120, value=70, step=10, label="Max output length")
                    batch_temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.05, label="Sampling temperature")
                    batch_button = gr.Button("Classify batch", variant="primary")
                with gr.Column(scale=6):
                    batch_summary = gr.Textbox(label="Batch summary", lines=6, interactive=False)
                    batch_results = gr.JSON(label="Per-image results")

        with gr.TabItem("Prediction history"):
            history_dataframe_output = gr.DataFrame(
                headers=["image_name", "predicted_class", "confidence", "report"],
                interactive=False,
                datatype=["str", "str", "str", "str"],
                row_count=(0, "dynamic"),
            )
            clear_history_button = gr.Button("Clear history")

    analyze_button.click(
        fn=handle_single_prediction,
        inputs=[image_input, language_input, max_tokens_input, temperature_input, shared_history_state],
        outputs=[prediction_output, confidence_output, report_output, history_dataframe_output, shared_history_state, download_output],
    )

    batch_button.click(
        fn=handle_batch_prediction,
        inputs=[batch_files, batch_language, batch_max_tokens, batch_temperature, shared_history_state],
        outputs=[batch_summary, batch_results, history_dataframe_output, shared_history_state],
    )

    clear_history_button.click(
        fn=lambda: ([], pd.DataFrame(columns=["image_name", "predicted_class", "confidence", "report"])),
        outputs=[shared_history_state, history_dataframe_output],
    )

if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("PORT", "7860"))
    print(f"Starting TerraNova app on http://{server_name}:{server_port}")
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        show_error=True,
        quiet=True,
    )
