# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liceron
"""The model being taught, and how it finds its own blind spots.

Data: Banking77 (PolyAI, CC-BY-4.0), 77 customer-support intents. Splits, fixed and recorded:
- pool:       train messages that can be bought (labels hidden from the agent until purchased)
- validation: 1,000 train messages, labelled, used by the agent to measure progress and decide purchases
- sealed:     the official 3,080-message test split; never used for any decision; reported only
Model: frozen sentence embeddings (all-MiniLM-L6-v2) + multinomial logistic regression (retrains in about a second on CPU).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VALIDATION_SIZE = 1000
SPLIT_SEED = 77


def pretty(intent: str) -> str:
    return intent.replace("_", " ").replace("?", "").strip().capitalize()


@dataclass
class Corpus:
    texts: list[str]  # pool texts
    labels: np.ndarray  # pool gold labels (hidden from the agent; used only to simulate/verify annotators)
    X: np.ndarray  # pool embeddings
    val_X: np.ndarray
    val_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    intents: list[str]

    @property
    def n_classes(self) -> int:
        return len(self.intents)


def build_cache(cache: Path) -> None:
    """One-off: download Banking77, embed everything, save arrays and texts."""
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    cache.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("PolyAI/banking77")
    intents = ds["train"].features["label"].names
    train_text, train_y = list(ds["train"]["text"]), np.array(ds["train"]["label"])
    test_text, test_y = list(ds["test"]["text"]), np.array(ds["test"]["label"])
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(train_text))
    val_idx, pool_idx = order[:VALIDATION_SIZE], order[VALIDATION_SIZE:]
    enc = SentenceTransformer(EMBED_MODEL, device="cpu")
    emb = lambda t: enc.encode(t, batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    train_X = emb(train_text)
    np.save(cache / "pool_X.npy", train_X[pool_idx])
    np.save(cache / "pool_y.npy", train_y[pool_idx])
    np.save(cache / "val_X.npy", train_X[val_idx])
    np.save(cache / "val_y.npy", train_y[val_idx])
    np.save(cache / "test_X.npy", emb(test_text))
    np.save(cache / "test_y.npy", test_y)
    (cache / "pool_texts.json").write_text(json.dumps([train_text[i] for i in pool_idx]))
    (cache / "meta.json").write_text(
        json.dumps(
            {
                "intents": intents,
                "embed_model": EMBED_MODEL,
                "validation_size": VALIDATION_SIZE,
                "split_seed": SPLIT_SEED,
                "dataset": "PolyAI/banking77 (CC-BY-4.0)",
            }
        )
    )


def load_corpus(cache: Path) -> Corpus:
    if not (cache / "meta.json").exists():
        raise SystemExit(
            f"no dataset cache in {cache}. It is committed in the repository (data/); to rebuild it from Banking77: "
            "uv sync --extra build && uv run python -c \"from pathlib import Path; "
            "from blindspot.learner import build_cache; build_cache(Path('data'))\""
        )
    meta = json.loads((cache / "meta.json").read_text())
    return Corpus(
        texts=json.loads((cache / "pool_texts.json").read_text()),
        labels=np.load(cache / "pool_y.npy"),
        X=np.load(cache / "pool_X.npy"),
        val_X=np.load(cache / "val_X.npy"),
        val_y=np.load(cache / "val_y.npy"),
        test_X=np.load(cache / "test_X.npy"),
        test_y=np.load(cache / "test_y.npy"),
        intents=meta["intents"],
    )


class Model:
    def __init__(self, corpus: Corpus) -> None:
        self.c = corpus
        self.clf: LogisticRegression | None = None

    def fit(self, pool_idx: np.ndarray, labels: np.ndarray) -> None:
        clf = LogisticRegression(max_iter=2000, C=10.0)
        clf.fit(self.c.X[pool_idx], labels)
        self.clf = clf

    def _proba(self, X: np.ndarray) -> np.ndarray:
        assert self.clf is not None
        p = np.zeros((len(X), self.c.n_classes), dtype=np.float64)
        p[:, self.clf.classes_] = self.clf.predict_proba(X)
        return p

    def accuracy(self, split: str) -> float:
        X, y = (self.c.val_X, self.c.val_y) if split == "validation" else (self.c.test_X, self.c.test_y)
        return float((self._proba(X).argmax(1) == y).mean())

    def per_class_validation(self) -> dict[int, float]:
        pred = self._proba(self.c.val_X).argmax(1)
        out = {}
        for k in range(self.c.n_classes):
            m = self.c.val_y == k
            if m.any():
                out[k] = float((pred[m] == k).mean())
        return out

    def predict(self, pool_idx: np.ndarray, top: int = 5) -> list[list[tuple[int, float]]]:
        p = self._proba(self.c.X[pool_idx])
        order = np.argsort(-p, axis=1)[:, :top]
        return [[(int(k), float(p[i, k])) for k in row] for i, row in enumerate(order)]

    def blind_spots(self, available: np.ndarray, n: int, seed: int) -> list[dict]:
        """Group the model's most uncertain unlabelled messages into confusion areas and pick one message per area.

        Returns a list of picks: {pool_index, margin, confused: (intent_a, intent_b), cluster}.
        """
        p = self._proba(self.c.X[available])
        top2 = np.argsort(-p, axis=1)[:, :2]
        margin = p[np.arange(len(p)), top2[:, 0]] - p[np.arange(len(p)), top2[:, 1]]
        k = min(n * 6, len(available))
        cand = np.argsort(margin)[:k]
        clusters = min(n, len(cand))
        km = KMeans(n_clusters=clusters, n_init=2, random_state=seed).fit(self.c.X[available[cand]])
        picks: list[dict] = []
        for c in range(clusters):
            members = cand[km.labels_ == c]
            if len(members):
                best = members[np.argmin(margin[members])]
                picks.append(
                    {
                        "pool_index": int(available[best]),
                        "margin": float(margin[best]),
                        "confused": (int(top2[best, 0]), int(top2[best, 1])),
                        "cluster": c,
                    }
                )
        picks.sort(key=lambda d: d["margin"])
        return picks[:n]

    def random_picks(self, available: np.ndarray, n: int, rng: np.random.Generator) -> list[int]:
        return [int(i) for i in rng.choice(available, min(n, len(available)), replace=False)]
