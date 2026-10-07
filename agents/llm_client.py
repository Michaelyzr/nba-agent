"""Cached, accounted LLM calls for the tool agent and its evaluation.

Every request is keyed by a SHA-256 of (model, temperature, role, system
prompt, user prompt); the user prompt already contains every tool output the
model has seen. Responses are stored one file per key under the git-ignored
runs/llm_cache/, so a re-run replays the same decisions for free. Errors are
never cached.

Backends are plain objects with generate(system, prompt, meta) -> (text, usage).
GeminiBackend uses google-genai with temperature 0, JSON output and thinking
off; tests and dry runs pass a stub. The key comes from GEMINI_API_KEY in .env.
DeepSeekBackend posts to DeepSeek's OpenAI-compatible /chat/completions with
httpx (temperature 0, JSON mode); the key comes from DEEPSEEK_API_KEY.
"""
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "runs" / "llm_cache"
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# USD per million tokens (input, output), Google AI Studio paid tier list prices; an estimate only.
PRICES = {"gemini-3.8-flash": (0.30, 2.50), "gemini-2.5-flash": (0.30, 2.50), "gemini-2.5-flash-lite": (0.10, 0.40),
          "gemini-3.5-flash-lite": (0.10, 0.40),   # assumed equal to 2.5 Flash-Lite
          "gemini-2.5-pro": (1.25, 10.0),
          "deepseek-chat": (0.27, 1.10)}           # approximate DeepSeek list price, cache-miss input
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_URL = "https://api.deepseek.com"
MAX_WAIT = 300                 # seconds; a longer retry hint means the daily quota is spent


def parse_json(text: str):
    """The first JSON object in a reply, or None."""
    if not text:
        return None
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    if model.startswith("gemini"):
        price_in, price_out = PRICES.get(model, PRICES["gemini-3.8-flash"])
    elif model.startswith("deepseek"):
        price_in, price_out = PRICES.get(model, PRICES["deepseek-chat"])
    else:
        return 0.0
    return (tokens_in * price_in + tokens_out * price_out) / 1e6


class GeminiBackend:
    """google-genai client: temperature 0, JSON mime type, thinking budget 0, retries on 429/5xx."""

    def __init__(self, model: str = DEFAULT_MODEL, max_retries: int = 8, min_interval: float = 0.0):
        from dotenv import load_dotenv
        from google import genai

        load_dotenv(ROOT / ".env")
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set (add it to .env)")
        self.model, self.max_retries, self.min_interval = model, max_retries, min_interval
        self.client = genai.Client(api_key=key)
        self._last = 0.0
        self._lock = threading.Lock()

    def _config(self, system: str):
        from google.genai import types
        kwargs = {"temperature": 0.0, "response_mime_type": "application/json", "system_instruction": system}
        if "2.5" in self.model:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0 if "pro" not in self.model else 128)
        return types.GenerateContentConfig(**kwargs)

    def generate(self, system: str, prompt: str, meta=None):
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            with self._lock:
                wait = self._last + self.min_interval - time.time()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.time()
            try:
                r = self.client.models.generate_content(model=self.model, contents=prompt, config=self._config(system))
                u = r.usage_metadata
                usage = {"tokens_in": int(getattr(u, "prompt_token_count", 0) or 0),
                         "tokens_out": int((getattr(u, "candidates_token_count", 0) or 0)
                                           + (getattr(u, "thoughts_token_count", 0) or 0))}
                return r.text or "", usage
            except Exception as exc:
                code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                retryable = code in (429, 500, 502, 503, 504) or "timed out" in str(exc).lower()
                if not retryable or attempt == self.max_retries:
                    raise
                hint = re.search(r"retry in ([\d.]+)s", str(exc)) or re.search(r"'retryDelay': '(\d+)s'", str(exc))
                if re.search(r"retry in \d+h", str(exc)) or (hint and float(hint.group(1)) > MAX_WAIT):
                    raise                                  # daily quota spent: fail this call, do not sleep for hours
                time.sleep(max(delay, float(hint.group(1)) + 1) if hint else delay)
                delay = min(delay * 2, 60)


class DeepSeekBackend:
    """DeepSeek chat completions over httpx: temperature 0, JSON mode, retries on 429/5xx and timeouts."""

    def __init__(self, model: str = DEEPSEEK_MODEL, max_retries: int = 8, min_interval: float = 0.0,
                 base_url: str = DEEPSEEK_URL, timeout: float = 120.0, client=None):
        import httpx
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key and client is None:
            raise RuntimeError("DEEPSEEK_API_KEY is not set (add it to .env)")
        self.model, self.max_retries, self.min_interval = model, max_retries, min_interval
        self.client = client or httpx.Client(base_url=base_url, timeout=timeout,
                                             headers={"Authorization": f"Bearer {key}"})
        self._last = 0.0
        self._lock = threading.Lock()

    def _body(self, system: str, prompt: str) -> dict:
        return {"model": self.model, "temperature": 0.0, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}

    def generate(self, system: str, prompt: str, meta=None):
        import httpx
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            with self._lock:
                wait = self._last + self.min_interval - time.time()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.time()
            try:
                r = self.client.post("/chat/completions", json=self._body(system, prompt))
                code = r.status_code
                if code == 200:
                    data = r.json()
                    u = data.get("usage") or {}
                    text = (data["choices"][0]["message"].get("content") or "")
                    return text, {"tokens_in": int(u.get("prompt_tokens", 0) or 0),
                                  "tokens_out": int(u.get("completion_tokens", 0) or 0)}
                err = f"HTTP {code}: {r.text[:200]}"
                retry_after = r.headers.get("retry-after")
            except httpx.TransportError as exc:          # timeouts, dropped connections
                code, err, retry_after = None, f"{exc.__class__.__name__}: {exc}", None
            if (code is not None and code not in (429, 500, 502, 503, 504)) or attempt == self.max_retries:
                raise RuntimeError(err)
            hint = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else None
            if hint is not None and hint > MAX_WAIT:
                raise RuntimeError(err)
            time.sleep(max(delay, hint + 1) if hint is not None else delay)
            delay = min(delay * 2, 60)


class StubBackend:
    """Test and dry-run backend: fn(role, system, prompt, meta) -> reply text (or a dict, dumped as JSON)."""

    def __init__(self, fn, model: str = "stub"):
        self.fn, self.model = fn, model
        self.calls = 0

    def generate(self, system: str, prompt: str, meta=None):
        self.calls += 1
        out = self.fn((meta or {}).get("role"), system, prompt, meta or {})
        text = out if isinstance(out, str) else json.dumps(out)
        return text, {"tokens_in": len(system + prompt) // 4, "tokens_out": len(text) // 4}


class CachedLLM:
    def __init__(self, backend, cache_dir: Path | None = CACHE_DIR, temperature: float = 0.0):
        self.backend = backend
        self.model = getattr(backend, "model", "unknown")
        self.temperature = temperature
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.log = []              # one entry per request in this process

    def key(self, role: str, system: str, prompt: str) -> str:
        blob = json.dumps({"model": self.model, "temperature": self.temperature, "role": role,
                           "system": system, "prompt": prompt}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def ask(self, role: str, system: str, prompt: str, meta: dict | None = None) -> dict:
        """{"json", "text", "cached", "tokens_in", "tokens_out", "cost", "latency", "error", "key"}."""
        key = self.key(role, system, prompt)
        path = self._path(key) if self.cache_dir else None
        if path is not None and path.exists():
            hit = json.loads(path.read_text())
            out = {"text": hit["text"], "cached": True, "tokens_in": hit["tokens_in"], "tokens_out": hit["tokens_out"],
                   "latency": hit["latency"], "error": None}
        else:
            t0 = time.time()
            try:
                text, usage = self.backend.generate(system, prompt, {**(meta or {}), "role": role})
            except Exception as exc:
                out = {"text": "", "cached": False, "tokens_in": 0, "tokens_out": 0, "latency": time.time() - t0,
                       "error": f"{exc.__class__.__name__}: {str(exc)[:200]}"}
                out.update(json=None, cost=0.0, key=key, role=role)
                self.log.append({k: v for k, v in out.items() if k not in ("json", "text")})
                return out
            out = {"text": text, "cached": False, "tokens_in": usage["tokens_in"], "tokens_out": usage["tokens_out"],
                   "latency": time.time() - t0, "error": None}
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
                tmp.write_text(json.dumps({"model": self.model, "temperature": self.temperature, "role": role,
                                           "system": system, "prompt": prompt, "text": text,
                                           "tokens_in": out["tokens_in"], "tokens_out": out["tokens_out"],
                                           "latency": out["latency"], "created": time.time()}))
                os.replace(tmp, path)
        out.update(json=parse_json(out["text"]), cost=cost_usd(self.model, out["tokens_in"], out["tokens_out"]),
                   key=key, role=role)
        self.log.append({k: v for k, v in out.items() if k not in ("json", "text")})
        return out
