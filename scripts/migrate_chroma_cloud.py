"""Migrate a local Chroma collection into Chroma Cloud.

This script copies chunk documents, ids, metadatas, and embeddings from a local
persistent Chroma database into Chroma Cloud. If embeddings are missing in the
source collection, it re-embeds the documents with the configured sentence-
transformer model before upload.

Usage example:
    py scripts/migrate_chroma_cloud.py \
        --source-dir .chroma \
        --source-collection prism \
        --target-collection prism
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

# Ensure the repository root is on the import path when this script is run from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PipelineConfig


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


def migrate_collection(
    source_dir: str,
    source_collection_name: str,
    target_collection_name: str,
    batch_size: int,
    config: PipelineConfig,
) -> None:
    source_client = chromadb.PersistentClient(path=source_dir)
    target_client = chromadb.CloudClient(
        api_key=config.chroma_api_key,
        tenant=config.chroma_tenant,
        database=config.chroma_database,
        cloud_host=config.chroma_host,
        cloud_port=config.chroma_port,
        enable_ssl=config.chroma_ssl,
    )

    source_collection = source_client.get_or_create_collection(source_collection_name)
    target_collection = target_client.get_or_create_collection(target_collection_name)
    embedder = SentenceTransformer(config.embedding_model)

    total = source_collection.count()
    print(f"Source collection: {source_collection_name} ({total} records)")
    print(f"Target collection: {target_collection_name}")

    offset = 0
    copied = 0
    while offset < total:
        batch = source_collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )

        ids = _to_list(batch.get("ids"))
        documents = _to_list(batch.get("documents"))
        metadatas = _to_list(batch.get("metadatas"))
        embeddings = batch.get("embeddings")

        if not ids:
            break

        if embeddings is None or len(_to_list(embeddings)) != len(ids):
            print(f"Re-embedding batch at offset {offset} (missing embeddings in source)")
            embeddings = embedder.encode(documents, convert_to_numpy=True, show_progress_bar=False).tolist()
        else:
            embeddings = _to_list(embeddings)

        target_collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        copied += len(ids)
        offset += len(ids)
        print(f"Copied {copied}/{total} records")

    print("Migration complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy a local Chroma collection into Chroma Cloud.")
    parser.add_argument("--source-dir", default=".chroma", help="Path to the local persistent Chroma directory")
    parser.add_argument("--source-collection", default="prism", help="Local collection name to copy from")
    parser.add_argument("--target-collection", default="prism", help="Cloud collection name to copy into")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of records to copy per batch")
    args = parser.parse_args()

    config = PipelineConfig()
    if not config.chroma_tenant or not config.chroma_database or not config.chroma_api_key:
        raise SystemExit(
            "Missing Chroma Cloud credentials. Set CHROMA_TENANT, CHROMA_DATABASE, and CHROMA_API_KEY."
        )

    migrate_collection(
        source_dir=args.source_dir,
        source_collection_name=args.source_collection,
        target_collection_name=args.target_collection,
        batch_size=args.batch_size,
        config=config,
    )


if __name__ == "__main__":
    main()
