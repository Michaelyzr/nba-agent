"""Chat-model access. Gemini when a key is set; otherwise the agent uses its offline rules."""
import json
import os
import re

from nba_agent_paths import ROOT

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / "backend" / ".env")
    load_dotenv(ROOT / ".env")  # Keep existing root-level configuration working.
except ImportError:
    pass

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def available():
    if os.getenv("NBA_AGENT_OFFLINE") == "1":
        return False
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def ask(prompt, want_json=False):
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    config = {"response_mime_type": "application/json"} if want_json else None
    text = client.models.generate_content(model=MODEL, contents=prompt, config=config).text or ""
    if not want_json:
        return text.strip()
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group(0) if match else text)


def strip_fences(code):
    m = re.search(r"```(?:python)?\n(.*?)```", code, re.S)
    return (m.group(1) if m else code).strip()
