"""Streamlit Community Cloud entrypoint: installs demo/requirements.txt (light) instead of the root one (torch)."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "research_demo_app.py"), run_name="__main__")
