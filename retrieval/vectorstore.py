from typing import List, Dict, Any
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:
    """ChromaDB-backed vector store with sliding-window chunking.
    """

    def __init__(self, collection_name: str, embedding_model: SentenceTransformer):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)
        self.model = embedding_model

    # Ingestion helpers for indexing documents

    def add_documents(self, documents: List[Dict[str, Any]]):
        texts, ids, metadatas = [], [], []

        for doc in documents:
            chunks = self._chunk(doc["content"])
            for i, chunk in enumerate(chunks):
                doc_id = f"{doc['source']}__chunk_{i}"
                texts.append(chunk)
                ids.append(doc_id)
                metadatas.append({
                    "source": doc["source"],
                    "source_type": doc["source_type"],
                })

        if not texts:
            return

        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            ids=ids,
            metadatas=metadatas,
        )
        print(f"[VectorStore] Indexed {len(texts)} chunks.")

    # Retrieval helpers for querying the vector index

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        count = self.collection.count()
        if count == 0:
            return []

        query_emb = self.model.encode([query], convert_to_numpy=True)
        results = self.collection.query(
            query_embeddings=query_emb.tolist(),
            n_results=min(top_k, count),
        )

        docs = []
        for i, text in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            docs.append({
                "source": meta["source"],
                "source_type": meta["source_type"],
                "content": text,
            })
        return docs

    # Helper utilities

    def _chunk(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """Sliding-window word chunking."""
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunks.append(" ".join(words[i : i + chunk_size]))
            i += chunk_size - overlap
        return chunks
