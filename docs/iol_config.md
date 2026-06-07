# IOLConfig — PRISM configuration reference

This document explains each field in `IOLConfig` (see `config.py`) and gives short examples and tuning tips.

## Overview
`IOLConfig` contains runtime parameters that control deduplication, noise filtering, contradiction detection and MMR compression. Tune these to trade off compactness vs. coverage.

## Fields

- `exact_dedup` (bool, default: `True`)
  - If true, identical sentences (after normalization) are removed before embedding.
  - When to disable: only for debugging or if you want to preserve exact duplicates for provenance checks.

- `near_dedup_threshold` (float, default: `0.92`)
  - Cosine similarity threshold used when removing semantically near-duplicate sentences.
  - Lower values (e.g. `0.85`) remove more sentences; higher values keep more.
  - Tuning tip: if you see many near-duplicates in output, lower this; if you lose necessary nuance, raise it.

- `min_sentence_length` (int, default: `20`)
  - Sentences shorter than this character length are considered noise and removed.
  - Useful to remove UI fragments like "Subscribe now" or short boilerplate lines.

- `remove_boilerplate` (bool, default: `True`)
  - Toggle heuristics that match common boilerplate/SEO fragments (newsletter, cookie policy, etc.).

- `mmr_lambda` (float, default: `0.6`)
  - Controls MMR balance: `1.0` → pure relevance (greedy top-k), `0.0` → pure diversity (spread).
  - `0.6` is a practical default. Lower values increase diversity and reduce redundancy.

- `target_token_budget` (Optional[int], default: `None`)
  - If set, this is the token budget used by the compressor. Otherwise the budget is derived from `consumer_type` and `CONSUMER_BUDGETS`.
  - Example: set to `800` to force output to fit a small LLM.

- `consumer_type` (str, default: `large_llm`)
  - Selects a budget preset: `large_llm` (3000), `small_llm` (800), `classifier` (400).
  - Use `consumer_type` when you want consistent defaults without overriding `target_token_budget`.

- `detect_contradictions` (bool, default: `True`)
  - Toggles lightweight numeric-claim contradiction detection. When enabled, conflicting numeric claims are flagged and summary notes are appended.

## Example: programmatic config

```python
from pipeline import IOLPipeline
from config import PipelineConfig, IOLConfig

cfg = PipelineConfig(
    iol=IOLConfig(
        near_dedup_threshold=0.90,
        mmr_lambda=0.5,
        consumer_type="small_llm",
    ),
    embedding_model="all-MiniLM-L6-v2",
    retrieval_top_k=12,
)

pipeline = IOLPipeline(cfg)
```

## Tuning checklist
- If outputs contain many near-duplicates: lower `near_dedup_threshold`.
- If outputs drop short but relevant facts: lower `min_sentence_length`.
- If outputs lack coverage (too diverse): increase `mmr_lambda` toward `1.0`.
- If you need to hard-limit output size: set `target_token_budget` explicitly.

## Notes & best practices
- Start with defaults and run `smoke_test.py` to inspect behavior quickly.
- Changes to the embedding model (`PipelineConfig.embedding_model`) can substantially change dedup and MMR behavior — re-tune thresholds if you switch models.
- Use `consumer_type` to pick sensible default budgets per downstream consumer.

---

If you'd like, I can generate a short example notebook showing how different `mmr_lambda` and `near_dedup_threshold` values affect the resulting optimized context.