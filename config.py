from dataclasses import dataclass, field
from typing import Optional

@dataclass
class IOLConfig:
    # Deduplication settings
    exact_dedup: bool = True
    near_dedup_threshold: float = 0.92   # cosine similarity ceiling for near-duplicate removal

    # Noise filtering parameters
    min_sentence_length: int = 20        # minimum sentence length in characters
    remove_boilerplate: bool = True

    # Compression parameters for MMR
    target_token_budget: Optional[int] = None  # override per-consumer token budget
    mmr_lambda: float = 0.6             # λ for MMR (1.0 relevance, 0.0 diversity)

    # Contradiction detection toggle
    detect_contradictions: bool = True

    # Target consumer type for budget defaults
    consumer_type: str = "large_llm"    # large_llm | small_llm | classifier


# Default token budgets per consumer type
CONSUMER_BUDGETS = {
    "large_llm":  3000,
    "small_llm":   800,
    "classifier":  400,
}


@dataclass
class PipelineConfig:
    iol: IOLConfig = field(default_factory=IOLConfig)
    embedding_model: str = "all-MiniLM-L6-v2"
    retrieval_top_k: int = 10
    chroma_collection: str = "prism"
