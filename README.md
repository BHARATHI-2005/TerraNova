# Cosmic Land Explorer

A Streamlit interface for the notebook's fine-tuned MobileNetV2 model for
**Satellite Image Classification for Smart Land Monitoring**.

## Model and prediction contract

The notebook trains a MobileNetV2 classifier with a `(224, 224, 3)` RGB input,
resizes EuroSAT images to 224 × 224 using antialiased bilinear resizing, and
passes float32 pixels in the original `0–255` range. The saved model contains
the `Rescaling(scale=1/127.5, offset=-1)` layer, so `app.py` does not normalize
the image a second time. Its output is a ten-class softmax in the order listed
in [`config/class_names.json`](config/class_names.json).

The notebook's evaluation selected the fine-tuned best checkpoint:
`models/mobilenetv2_finetuned_best.keras`. If only the transfer-learning best
checkpoint is present, the app can use
`models/mobilenetv2_transfer_best.keras` as the notebook's documented fallback.
No new or substitute model is used.

The notebook output confirms that the fine-tuned best checkpoint was saved in
Google Drive under
`Satellite_Land_Monitoring/models/mobilenetv2_finetuned_best.keras`. The
checkpoint is included at `models/mobilenetv2_finetuned_best.keras`; no
retraining or separate download is needed for this repository.

Expected project layout:

```text
finetuneproject/
├── app.py
├── 10 (1).ipynb
├── requirements.txt
├── models/
│   └── mobilenetv2_finetuned_best.keras
├── config/
│   └── class_names.json
└── .streamlit/
    └── config.toml
```

## Install

Use Python 3.10–3.13 and a virtual environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned TensorFlow 2.21 wheel is available for this workspace's Windows
Python 3.13 environment. Use the virtual environment above to avoid conflicts
with unrelated packages already installed globally. The saved `.keras` file
remains unchanged.

## Run

From the project directory:

```powershell
python -m streamlit run app.py
```

Upload one JPG, JPEG, or PNG image and choose **Classify Image 🚀**. The app
reports the most likely single land-cover class, confidence, and up to three
class scores. Confidence is not a guarantee of a correct prediction.

If the model is missing or cannot be loaded, the app shows an explicit message
and does not substitute another network. The original notebook and model
weights are not modified by the app.

## Repository contents

The trained checkpoint is included so the app is ready to run after installing
requirements. The training dataset, local virtual environment, secrets, and
runtime caches are not included.
