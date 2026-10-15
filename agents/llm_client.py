"""Cached, accounted LLM calls for the tool agent and its evaluation.

Every request is keyed by a SHA-256 of (model, temperature, role, system
prompt, user prompt); the user prompt already contains every tool output the
model has seen. Responses are stored one file per key under the git-ignored
runs/llm_cache/, so a re-run replays the same decisions for free. Errors are
never cached.

Backends are plain objects with generate(system, prompt, meta) -> (text, usage).
GeminiBackend uses google-genai with temperature 0, JSON output and thinking
off; tests and dry runs pass a stub. The key comes from GEMINI_API_KEY in .env.
DeepSeekBackend streams DeepSeek's OpenAI-compatible /chat/completions over
httpx in JSON mode; the key comes from DEEPSEEK_API_KEY. Thinking is an explicit
flag sent as {"thinking": {"type": "enabled" | "disabled"}}: off means
temperature 0, on drops temperature (ignored in thinking mode) and allows
REASONER_MAX_TOKENS; reasoning_content is kept out of the answer and only its
length is recorded. The legacy names deepseek-chat / deepseek-reasoner resolve to
deepseek-flash with thinking off / on.

DeepSeek cache keys hash the resolved model id and the thinking flag. A request
made under a legacy name first looks up the key it had before the migration, so
the cached published runs replay without API calls.
"""
import hashlib
import json
import os
import random
import re
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "runs" / "llm_cache"
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# USD per million tokens (input, output), list prices; an estimate only. DeepSeek input is the cache-miss price.
PRICES = {"gemini-3.8-flash": (0.30, 2.50), "gemini-2.5-flash": (0.30, 2.50), "gemini-2.5-flash-lite": (0.10, 0.40),
          "gemini-3.5-flash-lite": (0.10, 0.40),   # assumed equal to 2.5 Flash-Lite
          "gemini-2.5-pro": (1.25, 10.0),
          "deepseek-chat": (0.27, 1.10),           # legacy name; the estimate used for the published runs
          "deepseek-reasoner": (0.27, 1.10),       # legacy name, thinking on; same estimate
          "deepseek-flash": (0.15, 0.60),          # off-peak; PEAK_FACTOR applies in DeepSeek peak hours
          "deepseek-v4-pro": (0.66, 1.98)}
PEAK_FACTOR = 2.0                                  # DeepSeek peak prices are twice the off-peak ones
PEAK_HOURS_UTC = ((1, 4), (6, 10))                 # [start, end) hours, Monday - Friday (Chinese holidays ignored)
PEAK_PRICED = ("deepseek-flash", "deepseek-v4-pro")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
DEEPSEEK_MODELS = ("deepseek-flash", "deepseek-v4-pro")
LEGACY_DEEPSEEK = {"deepseek-chat": ("deepseek-flash", False), "deepseek-reasoner": ("deepseek-flash", True)}
DEEPSEEK_URL = "https://api.deepseek.com"
REASONER_MAX_TOKENS = int(os.getenv("DEEPSEEK_REASONING_MAX_TOKENS", "32768"))  # reasoning counts against it
LEGACY_REASONER_MAX_TOKENS = 8192  # the budget of the published deepseek-reasoner runs
CONNECT_S, FIRST_TOKEN_S, ATTEMPT_S = 10.0, 60.0, 180.0   # per attempt; keep-alives do not count as tokens
DEEPSEEK_RETRIES = 5
BACKOFF_MIN, BACKOFF_MAX = 2.0, 60.0
RETRY_CODES = (429, 500, 502, 503, 504)
MAX_WAIT = 300                 # seconds; a longer retry hint means the daily quota is spent


def resolve_deepseek(model: str, thinking: bool | None = None) -> tuple:
    """(model id sent to the API, thinking flag); legacy names map to deepseek-flash, an explicit flag wins."""
    resolved, default = LEGACY_DEEPSEEK.get(model, (model, False))
    return resolved, default if thinking is None else bool(thinking)


def is_peak(ts: float | None = None) -> bool:
    t = time.gmtime(time.time() if ts is None else ts)
    return t.tm_wday < 5 and any(a <= t.tm_hour < b for a, b in PEAK_HOURS_UTC)


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
        pass
    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            out, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(out, dict):
            return out
    return None


def cost_usd(model: str, tokens_in: int, tokens_out: int, at: float | None = None) -> float:
    """List-price estimate; deepseek-flash / v4-pro cost PEAK_FACTOR more in peak hours (at = unix time, default now)."""
    if model.startswith("gemini"):
        price_in, price_out = PRICES.get(model, PRICES["gemini-3.8-flash"])
    elif model.startswith("deepseek"):
        price_in, price_out = PRICES.get(model, PRICES["deepseek-flash"])
        if model in PEAK_PRICED and is_peak(at):
            price_in, price_out = price_in * PEAK_FACTOR, price_out * PEAK_FACTOR
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


class TruncatedError(RuntimeError):
    """The reply spent the whole max_tokens budget (usually on reasoning) and has no answer."""


class _Retry(Exception):
    def __init__(self, code, msg, retry_after=None):
        super().__init__(msg)
        self.code, self.retry_after = code, retry_after


class DeepSeekBackend:
    """Streamed DeepSeek chat completions over httpx, JSON mode, explicit thinking flag.

    An attempt fails if no content or reasoning token arrives within first_token_s (DeepSeek answers 200 and then
    sends only keep-alives when overloaded, which defeats a plain read timeout) or it runs past attempt_s. Failed
    attempts, 429 and 5xx are retried up to max_retries times with jittered exponential backoff (2 - 60 s)."""

    def __init__(self, model: str = DEEPSEEK_MODEL, thinking: bool | None = None, max_tokens: int | None = None,
                 max_retries: int = DEEPSEEK_RETRIES, min_interval: float = 0.0, base_url: str = DEEPSEEK_URL,
                 first_token_s: float = FIRST_TOKEN_S, attempt_s: float = ATTEMPT_S, client=None,
                 clock=time.monotonic, temperature: float | None = None):
        import httpx
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key and client is None:
            raise RuntimeError("DEEPSEEK_API_KEY is not set (add it to .env)")
        self.model = model                     # the requested name; legacy names keep their old cache keys
        self.resolved_model, self.thinking = resolve_deepseek(model, thinking)
        # Temperature only applies with thinking off; default 0. Opt-in sampling (e.g. majority-vote sceptic).
        self.temperature = 0.0 if self.thinking else (0.0 if temperature is None else float(temperature))
        self.max_tokens = max_tokens or (REASONER_MAX_TOKENS if self.thinking else None)
        self.max_retries, self.min_interval = max_retries, min_interval
        self.first_token_s, self.attempt_s, self.clock = first_token_s, attempt_s, clock
        self.client = client or httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {key}"})
        self.client.timeout = httpx.Timeout(first_token_s, connect=CONNECT_S, write=30.0, pool=30.0)
        self._last = 0.0
        self._lock = threading.Lock()

    def _body(self, system: str, prompt: str) -> dict:
        body = {"model": self.resolved_model, "messages": [{"role": "system", "content": system},
                                                           {"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "thinking": {"type": "enabled" if self.thinking else "disabled"},
                "stream": True, "stream_options": {"include_usage": True}}
        if not self.thinking:
            body["temperature"] = self.temperature
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        return body

    def _attempt(self, body: dict):
        t0, first, content, reasoning, usage, finish = self.clock(), None, [], 0, {}, None
        with self.client.stream("POST", "/chat/completions", json=body) as r:
            if r.status_code != 200:
                r.read()
                raise _Retry(r.status_code, f"HTTP {r.status_code}: {r.text[:200]}", r.headers.get("retry-after"))
            for line in r.iter_lines():
                elapsed = self.clock() - t0
                if elapsed > self.attempt_s or (first is None and elapsed > self.first_token_s):
                    raise _Retry(None, f"stalled after {elapsed:.0f} s ({'no token' if first is None else 'slow'})")
                if not line.startswith("data:") or line.strip() == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                usage = chunk.get("usage") or usage
                for ch in chunk.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content"):
                        first = elapsed if first is None else first
                    content.append(d.get("content") or "")
                    reasoning += len(d.get("reasoning_content") or "")
                    finish = ch.get("finish_reason") or finish
        if first is None:
            raise _Retry(None, "stream ended without tokens")
        text = "".join(content)
        out = {"tokens_in": int(usage.get("prompt_tokens", 0) or 0),
               "tokens_out": int(usage.get("completion_tokens", 0) or 0)}
        if self.thinking:
            details = usage.get("completion_tokens_details") or {}
            out.update(reasoning_tokens=int(details.get("reasoning_tokens", 0) or 0), reasoning_chars=reasoning)
        if not text.strip() and (finish == "length" or (self.max_tokens and out["tokens_out"] >= self.max_tokens)):
            raise TruncatedError(f"truncated: no answer after {out['tokens_out']} tokens "
                                 f"(max_tokens {self.max_tokens}, reasoning {out.get('reasoning_tokens', 0)})")
        return text, out

    def generate(self, system: str, prompt: str, meta=None):
        import httpx
        body = self._body(system, prompt)
        for attempt in range(self.max_retries + 1):
            with self._lock:
                wait = self._last + self.min_interval - time.time()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.time()
            retry_after = None
            try:
                return self._attempt(body)
            except _Retry as exc:
                code, err, retry_after = exc.code, str(exc), exc.retry_after
            except httpx.TransportError as exc:          # timeouts, dropped connections
                code, err = None, f"{exc.__class__.__name__}: {exc}"
            if (code is not None and code not in RETRY_CODES) or attempt == self.max_retries:
                raise RuntimeError(err)
            hint = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else None
            if hint is not None and hint > MAX_WAIT:
                raise RuntimeError(err)
            delay = min(BACKOFF_MAX, max(BACKOFF_MIN, BACKOFF_MIN * 2 ** attempt * random.uniform(0.5, 1.5)))
            time.sleep(max(delay, hint + 1) if hint is not None else delay)


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
    def __init__(self, backend, cache_dir: Path | None = CACHE_DIR, temperature: float | None = None):
        self.backend = backend
        self.model = getattr(backend, "model", "unknown")
        # DeepSeekBackend: the model id actually called and the thinking flag, both part of the cache key.
        self.resolved_model = getattr(backend, "resolved_model", None)
        self.thinking = getattr(backend, "thinking", None) if self.resolved_model else None
        # Prefer the backend's temperature (DeepSeek sampling) when the caller does not override.
        self.temperature = float(getattr(backend, "temperature", 0.0) if temperature is None else temperature)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.log = []              # one entry per request in this process

    def _hash(self, **fields) -> str:
        blob = json.dumps({"temperature": self.temperature, **fields}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def key(self, role: str, system: str, prompt: str) -> str:
        """Where a new reply is stored: (resolved model, thinking) for DeepSeekBackend, else the requested model."""
        if self.resolved_model is None:
            return self._hash(model=self.model, role=role, system=system, prompt=prompt)
        return self._hash(model=self.resolved_model, thinking=bool(self.thinking), role=role, system=system,
                          prompt=prompt)

    def legacy_key(self, role: str, system: str, prompt: str) -> str | None:
        """The pre-migration key of a legacy deepseek-chat / deepseek-reasoner request (the published runs)."""
        if self.resolved_model is None or self.model not in LEGACY_DEEPSEEK:
            return None
        if resolve_deepseek(self.model) != (self.resolved_model, bool(self.thinking)):
            return None
        return self._hash(model=self.model, role=role, system=system, prompt=prompt)

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def ask(self, role: str, system: str, prompt: str, meta: dict | None = None) -> dict:
        """{"json", "text", "cached", "tokens_in", "tokens_out", "cost", "latency", "error", "truncated", "key"}."""
        key = self.key(role, system, prompt)
        path = self._path(key) if self.cache_dir else None
        hit = None
        if self.cache_dir:
            for k in (self.legacy_key(role, system, prompt), key):
                if k is not None and self._path(k).exists():
                    hit, key = json.loads(self._path(k).read_text()), k
                    break
        model = self.resolved_model or self.model
        if hit is not None:
            out = {"text": hit["text"], "cached": True, "tokens_in": hit["tokens_in"], "tokens_out": hit["tokens_out"],
                   "reasoning_tokens": hit.get("reasoning_tokens", 0), "reasoning_chars": hit.get("reasoning_chars", 0),
                   "latency": hit["latency"], "error": None, "truncated": False}
            model, created = hit.get("model", self.model), hit.get("created")
        else:
            t0 = time.time()
            try:
                text, usage = self.backend.generate(system, prompt, {**(meta or {}), "role": role})
            except Exception as exc:
                out = {"text": "", "cached": False, "tokens_in": 0, "tokens_out": 0, "reasoning_tokens": 0,
                       "reasoning_chars": 0, "latency": time.time() - t0,
                       "error": f"{exc.__class__.__name__}: {str(exc)[:200]}",
                       "truncated": isinstance(exc, TruncatedError)}
                out.update(json=None, cost=0.0, key=key, role=role)
                self.log.append({k: v for k, v in out.items() if k not in ("json", "text")})
                return out
            out = {"text": text, "cached": False, "tokens_in": usage["tokens_in"], "tokens_out": usage["tokens_out"],
                   "reasoning_tokens": usage.get("reasoning_tokens", 0),
                   "reasoning_chars": usage.get("reasoning_chars", 0), "latency": time.time() - t0, "error": None,
                   "truncated": False}
            created = time.time()
            if path is not None:
                entry = {"model": model, "temperature": self.temperature, "role": role, "system": system,
                         "prompt": prompt, "text": text, "tokens_in": out["tokens_in"], "tokens_out": out["tokens_out"],
                         "reasoning_tokens": out["reasoning_tokens"], "reasoning_chars": out["reasoning_chars"],
                         "latency": out["latency"], "created": created}
                if self.resolved_model is not None:
                    entry.update(thinking=bool(self.thinking), requested_model=self.model)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
                tmp.write_text(json.dumps(entry))
                os.replace(tmp, path)
        out.update(json=parse_json(out["text"]), cost=cost_usd(model, out["tokens_in"], out["tokens_out"], created),
                   key=key, role=role)
        self.log.append({k: v for k, v in out.items() if k not in ("json", "text")})
        return out
