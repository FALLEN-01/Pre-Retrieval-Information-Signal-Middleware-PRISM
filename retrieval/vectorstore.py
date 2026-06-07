from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:
    """ChromaDB-backed vector store with sliding-window chunking.
    """

    def __init__(
        self,
        collection_name: str,
        embedding_model: SentenceTransformer,
        persist_directory: Optional[str] = None,
        chroma_mode: str = "local",
        chroma_tenant: Optional[str] = None,
        chroma_database: Optional[str] = None,
        chroma_api_key: Optional[str] = None,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_ssl: bool = False,
    ):
        if chroma_mode == "cloud" or (chroma_tenant and chroma_database and chroma_api_key):
            self.client = chromadb.CloudClient(
                api_key=chroma_api_key,
                tenant=chroma_tenant,
                database=chroma_database,
                cloud_host=chroma_host,
                cloud_port=chroma_port,
                enable_ssl=chroma_ssl,
            )
        elif chroma_mode == "http":
            self.client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                ssl=chroma_ssl,
            )
        elif persist_directory:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)
        self.model = embedding_model

    # Ingestion helpers for indexing documents

    def add_documents(self, documents: List[Dict[str, Any]]):
        texts, ids, metadatas = [], [], []

        for doc in documents:
            chunks = self._chunk(doc["content"])
            source_doc_id = doc.get("source_doc_id", doc["source"])
            for i, chunk in enumerate(chunks):
                doc_id = f"{doc['source']}__chunk_{i}"
                texts.append(chunk)
                ids.append(doc_id)
                metadatas.append({
                    "source": doc["source"],
                    "source_type": doc["source_type"],
                    "source_doc_id": source_doc_id,
                    "chunk_index": i,
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
