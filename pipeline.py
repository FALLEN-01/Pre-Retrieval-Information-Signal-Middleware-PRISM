"""
PRISM Pipeline
==============

End-to-end pipeline:

    Ingest (Web / PDF / DB)
        ↓
    VectorStore (ChromaDB + sentence-transformers)
        ↓
    IOL (noise → dedup → contradiction detect → MMR compress)
        ↓
    LLM Consumer (Claude)

Usage
-----
    pipeline = IOLPipeline()

    # Ingest sources
    pipeline.ingest_web(["https://example.com/article"])
    pipeline.ingest_pdfs(["report.pdf"])
    pipeline.ingest_db("sqlite:///data.db", "SELECT * FROM facts", label="facts")

    # Query
    answer = pipeline.query("What is the revenue growth?")
    print(answer)

Note: PRISM Pipeline documentation.
"""

# Comments in this file are for maintainers.

from sentence_transformers import SentenceTransformer

from sources import WebSource, PDFSource, DBSource
from retrieval import VectorStore
from prism import InformationOptimizationLayer
try:
    from consumer import LLMConsumer
except Exception:
    LLMConsumer = None
from config import PipelineConfig


class IOLPipeline:
    """Orchestrates ingestion, retrieval, IOL, and LLM response.

    
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()

        print("[Pipeline] Loading embedding model...")
        self.embed_model = SentenceTransformer(self.config.embedding_model)

        self.vectorstore = VectorStore(
            self.config.chroma_collection,
            self.embed_model,
            persist_directory=self.config.chroma_persist_directory,
            chroma_mode=self.config.chroma_mode,
            chroma_tenant=self.config.chroma_tenant,
            chroma_database=self.config.chroma_database,
            chroma_api_key=self.config.chroma_api_key,
            chroma_host=self.config.chroma_host,
            chroma_port=self.config.chroma_port,
            chroma_ssl=self.config.chroma_ssl,
        )
        self.iol         = InformationOptimizationLayer(self.config.iol, self.embed_model)
        self.consumer    = LLMConsumer() if LLMConsumer is not None else None

        self.web_source = WebSource()
        self.pdf_source = PDFSource()

    # Ingestion methods (web, pdf, db)

    def ingest_web(self, urls: list[str]):
        docs = self.web_source.fetch_many(urls)
        self.vectorstore.add_documents(docs)
        print(f"[Pipeline] Web: ingested {len(docs)}/{len(urls)} pages.")

    def ingest_pdfs(self, paths: list[str]):
        docs = self.pdf_source.fetch_many(paths)
        self.vectorstore.add_documents(docs)
        print(f"[Pipeline] PDF: ingested {len(docs)}/{len(paths)} files.")

    def ingest_db(self, connection_string: str, sql: str, label: str = "database"):
        db  = DBSource(connection_string)
        doc = db.query_to_doc(sql, label)
        if doc:
            self.vectorstore.add_documents([doc])
            print(f"[Pipeline] DB: ingested '{label}'.")
        else:
            print(f"[Pipeline] DB: no results for '{label}'.")

    # Query handling and orchestration

    def query(self, question: str, verbose: bool = True) -> str:
        # Step 1 — retrieve relevant chunks from the vector store
        retrieved = self.vectorstore.retrieve(question, top_k=self.config.retrieval_top_k)

        if not retrieved:
            return "No relevant documents found in the index."

        # Step 2 — run retrieved chunks through the Information Optimization Layer (IOL)
        result = self.iol.optimize(retrieved, question)

        if verbose:
            m = result["metadata"]
            print(
                f"\n[IOL] {m['input_tokens']} → {m['output_tokens']} tokens "
                f"({m['compression_ratio']}× compression)"
            )
            print(
                f"      noise removed: {m['noise_removed']}  |  "
                f"duplicates removed: {m['duplicates_removed']}  |  "
                f"conflicts: {m['conflicts_detected']}"
            )

        if not result["optimized_context"]:
            return "IOL produced an empty context after optimization."

        # Step 3 — send the optimized context to the LLM consumer (optional)
        if self.consumer:
            return self.consumer.answer(question, result["optimized_context"], result["metadata"])

        # Consumer disabled: return the optimized context directly
        return result["optimized_context"]


# Quick demo / example usage when running as __main__
if __name__ == "__main__":
    pipeline = IOLPipeline()

    # ── Ingest ───────────────────────────────────────────────────────────────
    pipeline.ingest_web([
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    ])

    # Uncomment to ingest a PDF:
    # pipeline.ingest_pdfs(["my_report.pdf"])

    # Uncomment to ingest from a SQLite DB:
    # pipeline.ingest_db(
    #     "sqlite:///data.db",
    #     "SELECT product, revenue FROM sales WHERE year = 2024",
    #     label="sales_2024",
    # )

    # ── Query ─────────────────────────────────────────────────────────────────
    question = "What is retrieval-augmented generation and how does it work?"
    print(f"\nQ: {question}")

    answer = pipeline.query(question)
    print(f"\nA: {answer}")
