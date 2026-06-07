import os
from dataclasses import dataclass, field
from pathlib import Path
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


def _env_value(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is not None:
        return value

    dotenv_path = Path.cwd() / ".env"
    if not dotenv_path.exists():
        return None

    try:
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            if key.strip() == name:
                return raw_value.strip().strip('"').strip("'")
    except Exception:
        return None

    return None


def _env_bool(name: str, default: bool) -> bool:
    value = _env_value(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = _env_value(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class PipelineConfig:
    iol: IOLConfig = field(default_factory=IOLConfig)
    embedding_model: str = "all-MiniLM-L6-v2"
    retrieval_top_k: int = 10
    chroma_collection: str = "prism"
    chroma_persist_directory: str = ".chroma"
    chroma_mode: str = "local"  # local | cloud | http
    chroma_tenant: Optional[str] = None
    chroma_database: Optional[str] = None
    chroma_api_key: Optional[str] = None
    chroma_host: str = "api.trychroma.com"
    chroma_port: int = 443
    chroma_ssl: bool = True

    def __post_init__(self):
        self.chroma_persist_directory = _env_value("CHROMA_DB_DIR") or self.chroma_persist_directory
        self.chroma_mode = _env_value("CHROMA_MODE") or self.chroma_mode
        self.chroma_tenant = _env_value("CHROMA_TENANT") or self.chroma_tenant
        self.chroma_database = _env_value("CHROMA_DATABASE") or self.chroma_database
        self.chroma_api_key = _env_value("CHROMA_API_KEY") or self.chroma_api_key
        self.chroma_host = _env_value("CHROMA_HOST") or self.chroma_host
        self.chroma_port = _env_int("CHROMA_PORT", self.chroma_port)
        self.chroma_ssl = _env_bool("CHROMA_SSL", self.chroma_ssl)
