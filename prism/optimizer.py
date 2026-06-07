import re
from typing import List, Dict, Any

import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - allow running without the package for smoke tests
    SentenceTransformer = object

from config import IOLConfig, CONSUMER_BUDGETS
from .deduplicator import Deduplicator
from .noise_filter import NoiseFilter
from .compressor import MMRCompressor
from .contradiction_detector import ContradictionDetector

# Comments in this file are for maintainers.


class InformationOptimizationLayer:
    """
    PRISM's Information Optimization Layer (IOL).

    Orchestrates all IOL steps in order:

        raw sentences
            ↓  noise filter        (remove boilerplate / short lines)
            ↓  exact dedup         (hash-based)
            ↓  embed
            ↓  near dedup          (cosine similarity)
            ↓  contradiction detect (flag conflicting numeric claims)
            ↓  MMR compress        (relevance + diversity, token budget)
            ↓
        optimised context

    No large LLM required for any of these steps.

    
    """

    def __init__(self, config: IOLConfig, embedding_model: SentenceTransformer):
        self.config = config
        self.model  = embedding_model

        self.noise_filter  = NoiseFilter(min_length=config.min_sentence_length)
        self.deduplicator  = Deduplicator(near_dup_threshold=config.near_dedup_threshold)
        self.compressor    = MMRCompressor(mmr_lambda=config.mmr_lambda)
        self.contradiction = ContradictionDetector()

    # Public API methods

    def optimize(
        self,
        documents: List[Dict[str, Any]],
        query: str,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        documents : list of {"source": str, "source_type": str, "content": str}
        query     : user question (used to score relevance)

        Returns
        -------
        {
          "optimized_context": str,
          "metadata":          dict,
          "conflicts":         list,
        }
        """
        # Step 1 — split documents into sentences
        raw = []
        for doc in documents:
            raw.extend(self._split(doc["content"]))

        input_tokens = self._token_count(raw)

        if not raw:
            return self._empty(input_tokens)

        # Step 2 — filter out low-value/noise sentences
        filtered = self.noise_filter.filter(raw)
        noise_removed = len(raw) - len(filtered)

        # Step 3 — exact deduplication based on normalized-hash
        deduped = self.deduplicator.exact_dedup(filtered)

        if not deduped:
            return self._empty(input_tokens)

        # Step 4 — compute sentence embeddings
        embeddings     = self.model.encode(deduped,  convert_to_numpy=True, show_progress_bar=False)
        query_emb      = self.model.encode([query],  convert_to_numpy=True)[0]

        # Step 5 — remove near-duplicates using cosine similarity
        near_deduped, near_embs = self.deduplicator.near_dedup(deduped, embeddings)
        dupes_removed = len(deduped) - len(near_deduped)

        # Step 6 — detect conflicting numeric claims
        if self.config.detect_contradictions:
            clean, conflicts = self.contradiction.detect(near_deduped)
            # Re-embed only if conflict notes were added
            if conflicts:
                clean_embs = self.model.encode(clean, convert_to_numpy=True, show_progress_bar=False)
            else:
                clean_embs = near_embs
        else:
            clean, conflicts, clean_embs = near_deduped, [], near_embs

        # Step 7 — compress using Maximal Marginal Relevance (MMR)
        budget   = self._budget()
        selected = self.compressor.compress(clean, clean_embs, query_emb, budget)

        # Step 8 — assemble final optimized context and metadata
        output_text   = "\n".join(selected)
        output_tokens = len(output_text) // 4
        ratio         = round(input_tokens / max(output_tokens, 1), 2)

        return {
            "optimized_context": output_text,
            "metadata": {
                "input_tokens":       input_tokens,
                "output_tokens":      output_tokens,
                "compression_ratio":  ratio,
                "sentences_in":       len(raw),
                "sentences_out":      len(selected),
                "noise_removed":      noise_removed,
                "duplicates_removed": dupes_removed,
                "conflicts_detected": len(conflicts),
            },
            "conflicts": conflicts,
        }

    # Helper methods

    def _split(self, text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def _token_count(self, sentences: List[str]) -> int:
        return sum(len(s) // 4 for s in sentences)

    def _budget(self) -> int:
        if self.config.target_token_budget:
            return self.config.target_token_budget
        return CONSUMER_BUDGETS.get(self.config.consumer_type, 2000)

    def _empty(self, input_tokens: int) -> Dict[str, Any]:
        return {
            "optimized_context": "",
            "metadata": {
                "input_tokens":       input_tokens,
                "output_tokens":      0,
                "compression_ratio":  0,
                "sentences_in":       0,
                "sentences_out":      0,
                "noise_removed":      0,
                "duplicates_removed": 0,
                "conflicts_detected": 0,
            },
            "conflicts": [],
        }
