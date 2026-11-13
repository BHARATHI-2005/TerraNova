import runpy
from pathlib import Path


runpy.run_path(str(Path(__file__).resolve().with_name("streamlit_app.py")))
