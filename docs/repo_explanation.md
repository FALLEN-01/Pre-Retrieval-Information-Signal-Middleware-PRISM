# PRISM Repository File Guide

This document explains the purpose of each project file in the current repository. It is organized by file and focuses on what each file does, how it fits into the system, and what to know when editing it.

## Project Overview

PRISM is built as a retrieval pipeline with an Information Optimization Layer (IOL) in the middle:

- ingest content from web pages, PDFs, and databases
- store and retrieve chunks with ChromaDB and sentence embeddings
- run the retrieved text through the IOL
- optionally send the optimized context to an LLM consumer

## Top-Level Files

### `.env`
Local environment variables for development. This file is meant for machine-specific values such as `ANTHROPIC_API_KEY`, `CHROMA_DB_DIR`, and the optional Chroma Cloud settings (`CHROMA_TENANT`, `CHROMA_DATABASE`, `CHROMA_API_KEY`). The current default is local Chroma; Cloud remains available if you re-enable `CHROMA_MODE=cloud`.

### `.gitignore`
Lists files and folders that Git should ignore. In this repository it excludes Python caches, virtual environments, build artifacts, test caches, `.env`, and common editor or tool state.

### `.vscode/settings.json`
VS Code workspace settings. This repository uses it to auto-approve a terminal command related to archive extraction.

### `LICENSE`
The project license. This repository uses the MIT License, which allows reuse with attribution and the usual warranty disclaimer.

### `README.md`
The user-facing project overview. It explains what PRISM does, how to install dependencies, how to run the pipeline, how to configure the system, and how the consumer integration works.

### `config.py`
Defines the configuration dataclasses used across the pipeline:

- `IOLConfig` controls deduplication, noise filtering, contradiction detection, and MMR compression.
- `CONSUMER_BUDGETS` maps consumer types to default token budgets.
- `PipelineConfig` groups pipeline-wide settings like the embedding model, retrieval depth, and Chroma collection name.

### `requirements.txt`
Python dependency list for the core system. It includes the retrieval stack, vector database, web/PDF extraction libraries, numeric libraries, SQLAlchemy, and related dependencies.

### `scripts/migrate_chroma_cloud.py`
Utility script for copying an existing local Chroma collection into Chroma Cloud. It reuses the current embeddings when available and falls back to re-embedding documents with the configured sentence-transformer model when needed.

## Documentation Files

### `docs/iol_config.md`
A deeper reference for `IOLConfig`. It explains each IOL setting, gives tuning tips, and shows how to configure the pipeline programmatically.

### `docs/repo_explanation.md`
This file. It is a repository map for quickly understanding what every file does.

## Runtime Entry Point

### `pipeline.py`
The main orchestrator for the whole application. It wires together:

- the embedding model (`SentenceTransformer`)
- source adapters (`WebSource`, `PDFSource`, `DBSource`)
- retrieval (`VectorStore`)
- the IOL (`InformationOptimizationLayer`)
- the optional consumer (`LLMConsumer`)

It provides methods to ingest content from web pages, PDFs, and databases, and a `query()` method that runs retrieval, optimization, and optional consumer answering. The `__main__` block runs a demo using a live Wikipedia page.

## Consumer Package

### `consumer/__init__.py`
Package entry point for the consumer package. It re-exports `LLMConsumer` so other modules can import it from `consumer` directly.

### `consumer/llm_consumer.py`
Implements the optional LLM consumer wrapper around Anthropic's client. It builds the prompt from the optimized context, applies a system instruction, sends the request, and returns the model's response text.

This file is optional in practice: `pipeline.py` now guards the import so the rest of the project can run without `anthropic` installed.

## Retrieval Package

### `retrieval/__init__.py`
Package entry point for retrieval. It re-exports `VectorStore`.

### `retrieval/vectorstore.py`
Implements the ChromaDB-backed vector store.

Main responsibilities:

- chunk incoming text into overlapping windows
- embed chunks using the configured sentence-transformer model
- add chunks and metadata to ChromaDB
- retrieve top-k matching chunks for a query

This is the retrieval layer between the source adapters and the IOL.

## PRISM / IOL Package

### `prism/__init__.py`
Package entry point for the IOL package. It re-exports `InformationOptimizationLayer`.

### `prism/optimizer.py`
The core IOL controller. It orchestrates the optimization stages in order:

1. split documents into sentences
2. filter noise
3. exact deduplicate
4. embed sentences and query
5. near-deduplicate semantically similar sentences
6. detect contradictions in numeric claims
7. compress using MMR
8. return the optimized context and metadata

It is the main business logic of PRISM.

### `prism/noise_filter.py`
Contains simple heuristics for removing low-value text. It uses regex patterns and sentence-length checks to filter boilerplate, SEO filler, navigation fragments, and other noise.

### `prism/deduplicator.py`
Contains two deduplication strategies:

- exact deduplication using normalized MD5 hashes
- near-deduplication using cosine similarity over sentence embeddings

This module removes repeated or near-repeated content before compression.

### `prism/contradiction_detector.py`
Implements lightweight contradiction detection for simple numeric claims. It looks for claims that match the same subject and verb but use different numeric values, then flags them and appends conflict notes.

### `prism/compressor.py`
Implements MMR (Maximal Marginal Relevance) compression. It selects sentences that are both relevant to the query and diverse with respect to already selected sentences, until the token budget is reached.

## Sources Package

### `sources/__init__.py`
Package entry point for source adapters. It re-exports the web, PDF, and DB source classes.

### `sources/web_source.py`
Fetches and cleans text from web URLs using `trafilatura`.

Main behavior:

- downloads a web page
- extracts readable text
- returns a document dictionary with `source`, `source_type`, and `content`

This is the live web ingestion path.

### `sources/pdf_source.py`
Extracts text from PDF files using `pdfplumber`.

Main behavior:

- opens a PDF file
- extracts text from each page
- combines page text into a single content string
- returns document dictionaries for valid PDFs

### `sources/db_source.py`
Queries SQL databases using SQLAlchemy and converts results to readable text.

Main behavior:

- builds a SQLAlchemy engine from a connection string
- runs a SQL query
- formats rows as text
- returns a document dictionary when rows are found

This is the database ingestion path.

## Generated / Local-Only Files

### `__pycache__/`
Python bytecode cache directory. It is generated automatically and should not be edited by hand.

## How the files work together

The flow across files is:

1. `sources/*` fetch input data from web, PDF, or SQL sources.
2. `retrieval/vectorstore.py` chunks and indexes the content.
3. `pipeline.py` retrieves relevant chunks for a question.
4. `prism/optimizer.py` runs the IOL stages.
5. `consumer/llm_consumer.py` optionally turns the optimized context into a final model answer.
6. `config.py` controls the defaults used at every stage.

## Practical editing notes

- Change pipeline-wide behavior in `config.py` first.
- Change source ingestion behavior in the relevant file under `sources/`.
- Change retrieval behavior in `retrieval/vectorstore.py`.
- Change optimization behavior in `prism/`.
- Change consumer behavior in `consumer/llm_consumer.py`.

If you want, I can also create a second version of this file that is more concise and formatted as a table.