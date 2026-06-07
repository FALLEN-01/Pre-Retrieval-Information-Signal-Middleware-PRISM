# PRISM — Pre-Retrieval Information Signal Middleware

PRISM provides an Information Optimization Layer (IOL) that sits between retrieval and any downstream consumer. The IOL performs light-weight, explainable preprocessing (noise filtering, deduplication, contradiction detection and extractive compression) before sending a compact, high-density context to a model.

**This README explains how to install, run and configure PRISM.**

**Files to know**
- `pipeline.py` — orchestration entry point and demo ([pipeline.py](pipeline.py))
- `config.py` — all runtime tunables: `IOLConfig` and `PipelineConfig` ([config.py](config.py))
- `consumer/llm_consumer.py` — optional LLM consumer wrapper ([consumer/llm_consumer.py](consumer/llm_consumer.py))

## Quickstart

1. Install core dependencies:

```powershell
py -m pip install -r requirements.txt
```

2. Run the smoke test (no external API keys required):

```powershell
py smoke_test.py
```

3. Run the pipeline demo (uses the embedding model and Chroma):

```powershell
py pipeline.py
```

Notes:
- On Windows use `py` to ensure the correct Python launcher is used. On POSIX systems use `python3`.
- If you plan to call a real LLM consumer, see the Consumer section below.

## Configuration

All runtime settings live in `config.py` via two dataclasses: `IOLConfig` and `PipelineConfig`.

- `IOLConfig` (important fields):
    - `near_dedup_threshold` (float, default 0.92): cosine similarity ceiling for removing semantically duplicate sentences. Lower → more aggressive dedup.
    - `min_sentence_length` (int, default 20): sentences shorter than this are treated as noise and dropped.
    - `mmr_lambda` (float, default 0.6): controls MMR tradeoff (1.0 relevance, 0.0 diversity).
    - `target_token_budget` (Optional[int]): override consumer token budget for extractive compression.
    - `consumer_type` (str): one of `large_llm` / `small_llm` / `classifier` used to pick default budgets.
    - `detect_contradictions` (bool): toggle simple numeric-claim contradiction detection.

- `PipelineConfig` (important fields):
    - `embedding_model` (str): default embedding model name (`all-MiniLM-L6-v2`). Change to a different model if needed.
    - `retrieval_top_k` (int): how many chunks to retrieve from the vector store per query.
    - `chroma_collection` (str): Chroma collection name (default `prism`).

To change settings programmatically, pass a custom `PipelineConfig` to `IOLPipeline`:

```python
from pipeline import IOLPipeline
from config import PipelineConfig, IOLConfig

cfg = PipelineConfig(iol=IOLConfig(near_dedup_threshold=0.90, consumer_type="small_llm"))
pipeline = IOLPipeline(cfg)
```

## Consumer / LLM integration

The project separates core IOL code from consumer-specific dependencies. If you want to enable the `consumer/llm_consumer.py` wrapper (Anthropic/Claude), install the consumer requirements and set an API key.

Install consumer deps (if provided):

```powershell
py -m pip install -r requirements.txt
# If there is a separate consumer requirements file, install it too (example):
# py -m pip install -r requirements-consumer-anthropic.txt
```

Set the API key (PowerShell):

```powershell
$env:ANTHROPIC_API_KEY = "sk-..."
```

Or (cmd.exe):

```bat
setx ANTHROPIC_API_KEY "sk-..."
```

Then run `py pipeline.py` to exercise the full pipeline including the consumer.

## Sources and retrieval

- `sources/` contains lightweight adapters:
    - `web_source.py` — `trafilatura` based HTML extraction.
    - `pdf_source.py` — `pdfplumber` based PDF extraction.
    - `db_source.py` — `SQLAlchemy` adapter returning rows as text.

- `retrieval/vectorstore.py` uses `chromadb` + the embedding model supplied by `sentence-transformers`. By default it creates a collection named `prism`.

Customizing Chroma/embedding behavior:
- Change `PipelineConfig.embedding_model` to use a different `sentence-transformers` model.
- To adjust Chroma persistence or advanced settings, configure the `chromadb` client via environment variables or the `chromadb` API — consult the `chromadb` docs for advanced options.

## Troubleshooting & tips

- If embeddings fail to load, ensure `sentence-transformers` and its dependencies are installed (this may download models).
- If `chromadb` reports no documents after indexing, ensure `vectorstore.add_documents()` was called and check the collection name in `PipelineConfig.chroma_collection`.
- For offline development, use `smoke_test.py` which avoids heavy deps by using a `FakeModel`.

## Attribution

All developer-facing docstrings and comments were generated with the assistance of GitHub Copilot and consolidated in this README to keep source files concise.

---

If you'd like, I can:
- convert this README to include quick copyable commands for WSL/UNIX as well; or
- create a short `docs/` page that documents each `IOLConfig` field with examples.

---

## Attribution

All docstrings and developer-facing comments previously distributed across source files have been consolidated here. These were generated by GitHub Copilot and are presented centrally to avoid repeated in-file attributions.
