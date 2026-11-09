# data/: precomputed Banking77 embeddings

The server reads these files and never downloads or embeds anything at runtime.

| File | Contents |
|---|---|
| `pool_texts.json`, `pool_y.npy`, `pool_X.npy` | 9,003 training messages that can be bought (texts, intent ids, embeddings) |
| `val_y.npy`, `val_X.npy` | 1,000 training messages held out as the customer's validation set |
| `test_y.npy`, `test_X.npy` | the official 3,080-message test split ("sealed"), only reported, never used for decisions |
| `meta.json` | intent names, embedding model, split seed |

How they were made (`blindspot.learner.build_cache`):

- dataset: PolyAI/banking77 from the Hugging Face Hub
- embeddings: `sentence-transformers/all-MiniLM-L6-v2`, normalised, float32, 384 dimensions
- validation split: 1,000 training messages drawn with seed 77

To rebuild:

```
uv sync --extra build && uv run python -c "from pathlib import Path; from blindspot.learner import build_cache; build_cache(Path('data'))"
```

Attribution: **Banking77** by PolyAI (Casanueva et al., 2020, "Efficient Intent Detection with Dual Sentence
Encoders"), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source:
https://huggingface.co/datasets/PolyAI/banking77.

Changes from the original: the training split is divided into a purchasable pool and a validation set (seed 77). The
texts are stored together with sentence embeddings computed from them.

`SHA256SUMS` lists checksums of the files as committed.
