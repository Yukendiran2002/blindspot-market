# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liceron
"""Blindspot go/no-go experiment: does buying labels where the model is weak beat buying random labels?

Data: Banking77 (PolyAI, CC-BY-4.0): 10,003 train messages = the unlabeled pool, 3,080 test = sealed test set,
never used for selection. Model: frozen sentence embeddings + logistic regression (retrains in seconds on CPU).
Each "purchase round" buys BATCH labels at $0.50 each. Strategies:
  random     - buy random pool messages (the baseline a normal labeling vendor gives you)
  uncertain  - buy the messages with the smallest top-2 probability margin
  blindspot  - find the model's confusion clusters among uncertain messages (k-means), buy across clusters
Labels come from the dataset's gold labels (standing in for human annotators). Several random seeds.
"""

import json
import sys
import time

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

SEEDS = [0, 1, 2, 3, 4]
START = 200          # labels the customer already has
BATCH = 100          # labels bought per round
ROUNDS = 12          # up to 1,400 labels total
PRICE = 0.50         # $ per label
EMBED = sys.argv[1] if len(sys.argv) > 1 else "sentence-transformers/all-MiniLM-L6-v2"

t0 = time.time()
ds = load_dataset("PolyAI/banking77")
Xtxt, y = ds["train"]["text"], np.array(ds["train"]["label"])
Ttxt, ty = ds["test"]["text"], np.array(ds["test"]["label"])
enc = SentenceTransformer(EMBED, device="cpu")
X = enc.encode(Xtxt, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
T = enc.encode(Ttxt, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
print(f"embedded {len(X)}+{len(T)} with {EMBED} in {time.time() - t0:.0f}s", flush=True)


def fit(idx):
    clf = LogisticRegression(max_iter=2000, C=10.0)
    clf.fit(X[idx], y[idx])
    return clf


def accuracy(clf):
    return float((clf.predict(T) == ty).mean())


def margins(clf, pool):
    p = clf.predict_proba(X[pool])
    top2 = np.sort(p, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def choose(strategy, clf, labeled, rng):
    pool = np.setdiff1d(np.arange(len(X)), labeled)
    if strategy == "random":
        return rng.choice(pool, BATCH, replace=False)
    m = margins(clf, pool)
    if strategy == "uncertain":
        return pool[np.argsort(m)[:BATCH]]
    # blindspot: take the 6x most uncertain, cluster them into confusion areas, buy the most uncertain per cluster
    cand = pool[np.argsort(m)[: BATCH * 6]]
    cm = np.sort(m)[: BATCH * 6]
    km = KMeans(n_clusters=BATCH, n_init=2, random_state=int(rng.integers(1e9))).fit(X[cand])
    picks = []
    for c in range(BATCH):
        members = np.where(km.labels_ == c)[0]
        if len(members):
            picks.append(cand[members[np.argmin(cm[members])]])
    if len(picks) < BATCH:  # fill from remaining most-uncertain
        rest = [i for i in cand if i not in set(picks)]
        picks += rest[: BATCH - len(picks)]
    return np.array(picks[:BATCH])


results = {s: [] for s in ("random", "uncertain", "blindspot")}
for seed in SEEDS:
    rng0 = np.random.default_rng(seed)
    start = rng0.choice(len(X), START, replace=False)
    for strategy in results:
        rng = np.random.default_rng(seed + 1000)
        labeled = start.copy()
        clf = fit(labeled)
        curve = [accuracy(clf)]
        for _ in range(ROUNDS):
            labeled = np.concatenate([labeled, choose(strategy, clf, labeled, rng)])
            clf = fit(labeled)
            curve.append(accuracy(clf))
        results[strategy].append(curve)
    print(f"seed {seed} done ({time.time() - t0:.0f}s): " +
          " | ".join(f"{s} {results[s][-1][-1]:.3f}" for s in results), flush=True)

labels = [START + BATCH * r for r in range(ROUNDS + 1)]
summary = {}
print("\nlabels bought -> sealed-test accuracy (mean over seeds, ± sd)")
print("labels  spent   " + "  ".join(f"{s:>16}" for s in results))
for r, n in enumerate(labels):
    row = []
    for s in results:
        a = np.array([c[r] for c in results[s]])
        row.append(f"{a.mean():.3f} ± {a.std():.3f}")
    print(f"{n:6} ${(n - START) * PRICE:6.0f}  " + "  ".join(f"{x:>16}" for x in row))
for s in results:
    summary[s] = np.mean(results[s], axis=0).tolist()
rand = np.array(summary["random"])
for s in ("uncertain", "blindspot"):
    curve = np.array(summary[s])
    # labels random needs to reach this strategy's final accuracy
    target = curve[-1]
    reach = next((labels[i] for i, a in enumerate(rand) if a >= target), None)
    print(f"\n{s}: final {curve[-1]:.3f} vs random {rand[-1]:.3f} (+{(curve[-1] - rand[-1]) * 100:.1f} pts). "
          f"Random reaches {target:.3f} at: {reach if reach else 'not within budget'} labels.")
json.dump({"embed": EMBED, "labels": labels, "price": PRICE, "seeds": SEEDS, "curves": results, "mean": summary},
          open("results.json", "w"), indent=1)
print(f"\ntotal time {time.time() - t0:.0f}s")
