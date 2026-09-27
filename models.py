"""The three trained models, with labeled stand-ins until training has been run.

Ranker:  fine-tuned cross-encoder in models/ranker, else BM25 (stand-in).
Support: fine-tuned cross-encoder in models/support, else a number-match rule (stand-in).
Signing: one-hidden-layer network in models/signing (run train_signing.py first).
"""
import json
import re
from functools import lru_cache

import numpy as np

from tables import MODELS

SIGNING_FEATURES = ["MPG", "PPG", "TS", "USG", "GP", "AGE", "PRIOR_PAY", "HAS_PRIOR"]


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=1)
def _cross_encoder(name):
    path = MODELS / name
    if not (path / "config.json").exists():
        return None
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return None
    return CrossEncoder(str(path))


def bm25_scores(query, texts):
    from rank_bm25 import BM25Okapi

    if not texts:
        return []
    bm = BM25Okapi([_tokens(t) for t in texts])
    return [float(s) for s in bm.get_scores(_tokens(query))]


def rank(query, texts, k=5):
    """Return (indices of the top k texts, scores, which model scored them)."""
    model = _cross_encoder("ranker")
    if model is not None and texts:
        scores = [float(s) for s in model.predict([(query, t) for t in texts])]
        source = "cross-encoder ranker"
    else:
        scores = bm25_scores(query, texts)
        source = "BM25 stand-in (run train_ranker.py)"
    order = list(np.argsort(scores)[::-1][:k])
    return [int(i) for i in order], [scores[i] for i in order], source


def _numbers(text):
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def evidence_text(values):
    """Checked figures written out at the roundings a note may use, for the support check."""
    parts = []
    for v in values:
        if abs(v - round(v)) < 1e-9:
            parts.append(str(int(round(v))))
        parts += [f"{v:.1f}", f"{v:.2f}", f"{v:.3f}"]
    return " ".join(parts)


def support(sentence, evidence):
    """Return (supported: bool, score, which model judged it)."""
    model = _cross_encoder("support")
    if model is not None:
        score = float(model.predict([(sentence, evidence)])[0])
        return score >= 0.5, score, "cross-encoder support check"
    claimed, available = _numbers(sentence), _numbers(evidence)
    ok = claimed <= available
    return ok, float(ok), "number-match stand-in (run train_support.py)"


class SigningNet:
    def __init__(self):
        import torch

        path = MODELS / "signing"
        if not (path / "model.pt").exists():
            raise FileNotFoundError("models/signing/model.pt is missing. Run `python train_signing.py`.")
        meta = json.loads((path / "metrics.json").read_text())
        self.mean = np.array(meta["feature_mean"], dtype=np.float32)
        self.std = np.array(meta["feature_std"], dtype=np.float32)
        self.y_scale = meta["y_scale"]
        self.heldout_mae = meta["mlp_mae"]
        self.net = build_mlp(len(SIGNING_FEATURES), meta["hidden"])
        self.net.load_state_dict(torch.load(path / "model.pt", weights_only=True))
        self.net.eval()

    def predict(self, features):
        import torch

        x = np.array([[features[k] for k in SIGNING_FEATURES]], dtype=np.float32)
        x = (x - self.mean) / self.std
        with torch.no_grad():
            y = self.net(torch.from_numpy(x)).item()
        return float(y * self.y_scale)


def build_mlp(n_in, hidden):
    import torch.nn as nn

    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, 1))


@lru_cache(maxsize=1)
def signing_net():
    return SigningNet()
