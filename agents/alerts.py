"""Deterministic pregame alerts and optional webhook delivery.

Alerts are derived from two consecutive pregame snapshots.  The detector never
calls an LLM and never changes a forecast or an order.  Replay runs always keep
local alert history, while external delivery is allowed only when the caller
explicitly selects live mode and enables notifications.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import threading
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import requests


SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}
VALID_SEVERITIES = tuple(SEVERITY_RANK)
STATUS_VALUES = {"out", "doubtful", "questionable", "probable", "available", "active"}


def _utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _finite(value, default: float, *, minimum: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) and result >= minimum else default


def _bool(value, default=False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _json_default(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


@dataclass
class AlertConfig:
    """Thresholds and delivery settings for pregame alerts."""

    enabled: bool = False
    notify_requested: bool = False
    webhook_url: str = ""
    webhook_secret: str = ""
    min_severity: str = "medium"
    probability_shift_pp: float = 3.0
    high_probability_shift_pp: float = 5.0
    market_move_pp: float = 3.0
    high_market_move_pp: float = 5.0
    market_window_minutes: float = 5.0
    quote_max_age_seconds: float = 90.0
    webhook_timeout_seconds: float = 3.0
    webhook_retries: int = 2
    cooldowns: dict[str, float] = field(default_factory=lambda: {
        "injury_status_change": 600.0,
        "probability_shift": 300.0,
        "market_move": 300.0,
        "news_conflict": 600.0,
        "source_degraded": 600.0,
        "quote_stale": 600.0,
        "source_recovered": 0.0,
        "quote_recovered": 0.0,
    })

    def __post_init__(self):
        if self.min_severity not in SEVERITY_RANK:
            self.min_severity = "medium"
        self.probability_shift_pp = _finite(self.probability_shift_pp, 3.0)
        self.high_probability_shift_pp = max(
            self.probability_shift_pp, _finite(self.high_probability_shift_pp, 5.0)
        )
        self.market_move_pp = _finite(self.market_move_pp, 3.0)
        self.high_market_move_pp = max(self.market_move_pp, _finite(self.high_market_move_pp, 5.0))
        self.market_window_minutes = _finite(self.market_window_minutes, 5.0)
        self.quote_max_age_seconds = _finite(self.quote_max_age_seconds, 90.0)
        self.webhook_timeout_seconds = _finite(self.webhook_timeout_seconds, 3.0, minimum=0.1)
        try:
            self.webhook_retries = max(0, min(5, int(self.webhook_retries)))
        except (TypeError, ValueError):
            self.webhook_retries = 2

    @classmethod
    def from_env(cls, path: Path | str | None = None) -> "AlertConfig":
        """Load defaults, optional JSON config, then environment overrides."""
        values: dict[str, Any] = {}
        if path:
            config_path = Path(path)
            try:
                values.update(json.loads(config_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid alert config {config_path}: {exc}") from exc
        env_map = {
            "enabled": ("ALERT_ENABLED", _bool),
            "webhook_url": ("ALERT_WEBHOOK_URL", str),
            "webhook_secret": ("ALERT_WEBHOOK_SECRET", str),
            "min_severity": ("ALERT_MIN_SEVERITY", str),
            "probability_shift_pp": ("ALERT_PROBABILITY_SHIFT_PP", float),
            "high_probability_shift_pp": ("ALERT_HIGH_PROBABILITY_SHIFT_PP", float),
            "market_move_pp": ("ALERT_MARKET_MOVE_PP", float),
            "high_market_move_pp": ("ALERT_HIGH_MARKET_MOVE_PP", float),
            "market_window_minutes": ("ALERT_MARKET_WINDOW_MINUTES", float),
            "quote_max_age_seconds": ("ALERT_QUOTE_MAX_AGE_SECONDS", float),
            "webhook_timeout_seconds": ("ALERT_WEBHOOK_TIMEOUT_SECONDS", float),
            "webhook_retries": ("ALERT_WEBHOOK_RETRIES", int),
        }
        for key, (env_name, converter) in env_map.items():
            if env_name in os.environ:
                try:
                    values[key] = converter(os.environ[env_name])
                except (TypeError, ValueError):
                    continue
        cooldowns = {**cls().cooldowns, **values.get("cooldowns", {})}
        for name, default in cls().cooldowns.items():
            env_name = "ALERT_COOLDOWN_" + name.upper()
            if env_name in os.environ:
                cooldowns[name] = _finite(os.environ[env_name], default)
        values["cooldowns"] = cooldowns
        return cls(**{k: v for k, v in values.items() if k in cls.__dataclass_fields__})

    def for_mode(self, mode: str, notify_requested: bool | None = None) -> "AlertConfig":
        return replace(self, notify_requested=self.notify_requested if notify_requested is None else notify_requested)

    def can_notify(self, mode: str) -> bool:
        return mode == "live" and self.enabled and self.notify_requested and bool(self.webhook_url)

    def cooldown_for(self, alert_type: str) -> float:
        return float(self.cooldowns.get(alert_type, self.cooldowns.get("source_degraded", 600.0)))


class AlertStore:
    """Durable event history and idempotency state for one run directory."""

    def __init__(self, output: Path | str):
        self.output = Path(output)
        self.events_path = self.output / "alerts.jsonl"
        self.state_path = self.output / "alert_state.json"
        self._lock = threading.RLock()
        self.events = self._read_events()
        self.state = self._read_state()

    def _read_events(self) -> list[dict]:
        if not self.events_path.exists():
            return []
        out = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if isinstance(row, dict) and row.get("event_id"):
                    out.append(row)
            except json.JSONDecodeError:
                continue
        return out

    def _read_state(self) -> dict:
        if not self.state_path.exists():
            return {}
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_events(self):
        self.output.mkdir(parents=True, exist_ok=True)
        temporary = self.events_path.with_suffix(".tmp")
        text = "".join(json.dumps(row, ensure_ascii=False, default=_json_default, allow_nan=False) + "\n"
                       for row in self.events)
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(self.events_path)

    def _write_state(self):
        self.output.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, default=_json_default,
                                        allow_nan=False, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    def emit(self, event: dict, cooldown_seconds: float) -> dict | None:
        """Persist an event unless its dedupe key is still cooling down."""
        key = event["dedupe_key"]
        now = _utc(event["observed_at"])
        with self._lock:
            previous = self.state.get(key)
            if previous:
                elapsed = (now - _utc(previous["observed_at"])).total_seconds()
                old_rank = SEVERITY_RANK.get(previous.get("severity", "low"), 0)
                new_rank = SEVERITY_RANK.get(event.get("severity", "low"), 0)
                if elapsed < cooldown_seconds and new_rank <= old_rank:
                    return None
            event = json.loads(json.dumps(event, default=_json_default))
            event["acknowledged"] = False
            self.events.append(event)
            self.state[key] = {"event_id": event["event_id"], "observed_at": event["observed_at"],
                               "severity": event["severity"]}
            self._write_events()
            self._write_state()
            return event

    def update_delivery(self, event_id: str, status: str, *, attempts: int = 0, error: str | None = None):
        with self._lock:
            for event in reversed(self.events):
                if event.get("event_id") == event_id:
                    event.setdefault("delivery", {}).update(status=status, attempts=attempts,
                                                             last_error=error)
                    self._write_events()
                    return event
        return None

    def latest(self, limit: int | None = None) -> list[dict]:
        with self._lock:
            rows = deepcopy(self.events)
        return rows[-limit:] if limit else rows

    def acknowledge(self, event_ids):
        wanted = {str(value) for value in event_ids}
        if not wanted:
            return 0
        changed = 0
        with self._lock:
            for event in self.events:
                if str(event.get("event_id")) in wanted and not event.get("acknowledged", False):
                    event["acknowledged"] = True
                    changed += 1
            if changed:
                self._write_events()
        return changed


def read_alerts(path: Path | str) -> list[dict]:
    """Read alert events for UI consumers, tolerating a partially written file."""
    return AlertStore(Path(path).parent).events if Path(path).exists() else []


def _player_names(players) -> dict[int, str]:
    if players is None:
        return {}
    if isinstance(players, Mapping):
        return {int(k): str(v) for k, v in players.items()}
    try:
        return {int(row.player_id): str(row.player_name) for row in players.itertuples()}
    except (AttributeError, TypeError, ValueError):
        return {}


def _num(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


class AlertDetector:
    """Compare adjacent pregame snapshots and return deterministic alert dictionaries."""

    def __init__(self, config: AlertConfig | None = None, players=None):
        self.config = config or AlertConfig()
        self.names = _player_names(players)

    def _event(self, current, alert_type, severity, dedupe_key, title, message, payload):
        observed = _utc(current.get("as_of"))
        game_id = str(current.get("game_id", "unknown"))
        digest = hashlib.sha256(f"{game_id}|{alert_type}|{dedupe_key}|{observed.isoformat()}".encode()).hexdigest()[:12]
        return {"event_id": f"{game_id}:{alert_type}:{digest}", "run_id": current.get("run_id"),
                "game_id": game_id, "phase": "pregame", "type": alert_type, "severity": severity,
                "observed_at": observed, "title": title, "message": message, "payload": payload,
                "dedupe_key": f"{game_id}:{alert_type}:{dedupe_key}",
                "delivery": {"status": "pending", "attempts": 0, "last_error": None}}

    def _factors(self, snapshot):
        return {str(row.get("player_id")): row for row in (snapshot or {}).get("factors", [])
                if row.get("player_id") is not None}

    def _market_rows(self, snapshot):
        markets = ((snapshot or {}).get("polymarket") or {}).get("markets") or []
        return {str(row.get("market_ticker") or row.get("token_id")): row for row in markets
                if row.get("market_ticker") or row.get("token_id")}

    def _market_mid(self, row):
        mid = _num(row.get("midpoint"))
        if mid is not None:
            return mid
        bid, ask = _num(row.get("bid")), _num(row.get("ask"))
        return (bid + ask) / 2 if bid is not None and ask is not None else _num(row.get("probability"))

    def detect(self, previous: dict | None, current: dict) -> list[dict]:
        if not current or current.get("phase") != "update":
            return []
        previous = previous or {}
        events = []
        game_id = str(current.get("game_id", "unknown"))
        new_ids = set(current.get("new_news_ids", []))
        old_factors, new_factors = self._factors(previous), self._factors(current)
        for pid, factor in new_factors.items():
            news_id = factor.get("news_id")
            status = str(factor.get("status", "")).lower()
            if news_id not in new_ids or status not in STATUS_VALUES:
                continue
            old = old_factors.get(pid)
            if old and str(old.get("status", "")).lower() == status:
                continue
            name = self.names.get(int(pid), f"Player {pid}")
            severity = "high" if status in {"out", "doubtful"} else "medium"
            events.append(self._event(
                current, "injury_status_change", severity, str(news_id),
                f"{name} status changed to {status.upper()}",
                f"{name} is now listed as {status}; the pregame forecast was updated.",
                {"player_id": int(pid), "player": name, "status": status, "news_id": news_id,
                 "source": factor.get("source"), "published_at": factor.get("published_at"),
                 "thresholds": {"status_values": sorted(STATUS_VALUES)}}))

        old_p, new_p = _num(previous.get("p_home")), _num(current.get("p_home"))
        if old_p is not None and new_p is not None:
            delta_pp = (new_p - old_p) * 100
            magnitude = abs(delta_pp)
            if magnitude >= self.config.probability_shift_pp:
                severity = "high" if magnitude >= self.config.high_probability_shift_pp else "medium"
                direction = "up" if delta_pp > 0 else "down"
                events.append(self._event(
                    current, "probability_shift", severity, direction,
                    f"Home win probability moved {delta_pp:+.1f} points",
                    f"Home win probability changed from {old_p:.1%} to {new_p:.1%}.",
                    {"p_before": old_p, "p_after": new_p, "delta_pp": delta_pp,
                     "thresholds": {"medium_pp": self.config.probability_shift_pp,
                                    "high_pp": self.config.high_probability_shift_pp}}))

        old_conflicts = {str(c.get("selected_news_id")) for c in previous.get("conflicts", [])}
        for conflict in current.get("conflicts", []):
            key = str(conflict.get("selected_news_id") or conflict.get("factor") or "unknown")
            if key in old_conflicts:
                continue
            events.append(self._event(
                current, "news_conflict", "medium", key,
                "Conflicting pregame reports retained",
                "Sources disagree about a player status; source priority was applied and alternatives were retained.",
                {"conflict": conflict, "thresholds": {"new_conflict": True}}))

        old_health = previous.get("news_health")
        new_health = current.get("news_health")
        if new_health == "degraded" and old_health != "degraded":
            errors = current.get("errors", [])
            source = str(errors[0].get("source", "news") if errors and isinstance(errors[0], dict) else "news")
            events.append(self._event(
                current, "source_degraded", "medium", f"{source}:degraded",
                "Pregame news coverage degraded",
                "At least one news source failed or returned invalid data; the previous factors were retained.",
                {"source": source, "errors": errors, "thresholds": {"state": "degraded"}}))
        elif old_health == "degraded" and new_health == "ok":
            events.append(self._event(
                current, "source_recovered", "low", "news:ok",
                "Pregame news coverage recovered",
                "The latest poll completed without news-source errors.",
                {"thresholds": {"state": "ok"}}))

        old_poly, new_poly = previous.get("polymarket") or {}, current.get("polymarket") or {}
        # A replay without a market provider has no quote state to alert on.
        if new_poly:
            old_age, new_age = _num(old_poly.get("age_seconds")), _num(new_poly.get("age_seconds"))
            old_stale = bool(old_poly) and (old_age is None or old_age > self.config.quote_max_age_seconds
                                          or old_poly.get("status") in {"WARNING", "UNAVAILABLE"})
            new_stale = (new_age is None or new_age > self.config.quote_max_age_seconds
                         or new_poly.get("status") in {"WARNING", "UNAVAILABLE"})
            if new_stale and not old_stale:
                events.append(self._event(
                    current, "quote_stale", "medium", "polymarket:stale",
                    "Polymarket quote is stale or unavailable",
                    "Current market data is too old or unavailable for a fresh comparison.",
                    {"age_seconds": new_age, "status": new_poly.get("status"),
                     "thresholds": {"max_age_seconds": self.config.quote_max_age_seconds}}))
            elif old_stale and not new_stale:
                events.append(self._event(
                    current, "quote_recovered", "low", "polymarket:fresh",
                    "Polymarket quote recovered",
                    "A fresh Polymarket quote is available again.",
                    {"age_seconds": new_age, "thresholds": {"max_age_seconds": self.config.quote_max_age_seconds}}))
        else:
            new_stale = False

        old_markets, new_markets = self._market_rows(previous), self._market_rows(current)
        old_as_of, new_as_of = previous.get("as_of"), current.get("as_of")
        within_window = True
        if old_as_of and new_as_of:
            try:
                within_window = 0 <= (_utc(new_as_of) - _utc(old_as_of)).total_seconds() <= self.config.market_window_minutes * 60
            except (TypeError, ValueError):
                within_window = False
        if within_window and not new_stale:
            for ticker, row in new_markets.items():
                before = self._market_mid(old_markets.get(ticker, {}))
                after = self._market_mid(row)
                if before is None or after is None:
                    continue
                move_pp = (after - before) * 100
                magnitude = abs(move_pp)
                if magnitude < self.config.market_move_pp:
                    continue
                severity = "high" if magnitude >= self.config.high_market_move_pp else "medium"
                direction = "up" if move_pp > 0 else "down"
                events.append(self._event(
                    current, "market_move", severity, f"{ticker}:{direction}",
                    f"Market price moved {move_pp:+.1f} points",
                    f"{row.get('team') or ticker} midpoint changed from {before:.1%} to {after:.1%}.",
                    {"market_ticker": ticker, "team": row.get("team"), "mid_before": before,
                     "mid_after": after, "move_pp": move_pp,
                     "thresholds": {"window_minutes": self.config.market_window_minutes,
                                    "medium_pp": self.config.market_move_pp,
                                    "high_pp": self.config.high_market_move_pp}}))
        return events


class WebhookNotifier:
    def __init__(self, config: AlertConfig):
        self.config = config

    def send(self, event: dict) -> tuple[str, int, str | None]:
        """Return (status, attempts, error); never raise delivery errors."""
        body = json.dumps(event, ensure_ascii=False, default=_json_default, allow_nan=False,
                          separators=(",", ":"))
        headers = {"Content-Type": "application/json", "User-Agent": "nba-agent-alerts/1"}
        if self.config.webhook_secret:
            signature = hmac.new(self.config.webhook_secret.encode(), body.encode(), hashlib.sha256).hexdigest()
            headers["X-NBA-Agent-Signature"] = f"sha256={signature}"
        last_error = None
        total = self.config.webhook_retries + 1
        for attempt in range(1, total + 1):
            try:
                response = requests.post(self.config.webhook_url, data=body.encode("utf-8"),
                                         headers=headers, timeout=self.config.webhook_timeout_seconds)
                if 200 <= response.status_code < 300:
                    return "sent", attempt, None
                last_error = f"HTTP {response.status_code}"
            except requests.RequestException as exc:
                # Do not persist webhook URLs, credentials or query-string secrets.
                last_error = type(exc).__name__
            if attempt < total:
                time.sleep(min(1.0, 0.25 * attempt))
        return "failed", total, last_error or "webhook delivery failed"


class AlertManager:
    def __init__(self, output: Path | str, *, config: AlertConfig | None = None, mode: str = "replay",
                 run_id: str | None = None, players=None, notifier: WebhookNotifier | None = None,
                 background_delivery: bool = False):
        self.output = Path(output)
        self.config = config or AlertConfig.from_env()
        self.mode = mode
        self.run_id = run_id or self.output.name
        self.store = AlertStore(self.output)
        self.detector = AlertDetector(self.config, players)
        self.notifier = notifier or WebhookNotifier(self.config)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="alerts") if background_delivery else None

    def _deliver(self, event):
        try:
            status, attempts, error = self.notifier.send(event)
        except Exception as exc:
            status, attempts, error = "failed", 1, type(exc).__name__
        self.store.update_delivery(event["event_id"], status, attempts=attempts, error=error)

    def close(self):
        if self.executor:
            self.executor.shutdown(wait=True)

    def process(self, previous: dict | None, current: dict, mode: str | None = None) -> list[dict]:
        mode = mode or self.mode
        current = dict(current)
        current["run_id"] = self.run_id
        emitted = []
        for event in self.detector.detect(previous, current):
            event["run_id"] = self.run_id
            saved = self.store.emit(event, self.config.cooldown_for(event["type"]))
            if saved is None:
                continue
            if self.config.can_notify(mode) and SEVERITY_RANK[event["severity"]] >= SEVERITY_RANK[self.config.min_severity]:
                if self.executor:
                    self.executor.submit(self._deliver, deepcopy(saved))
                else:
                    self._deliver(saved)
            else:
                self.store.update_delivery(saved["event_id"], "not_sent", attempts=0, error=None)
                saved["delivery"].update(status="not_sent", attempts=0, last_error=None)
            emitted.append(saved)
        return emitted

