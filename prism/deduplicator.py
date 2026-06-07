import hashlib
from typing import List, Tuple
import numpy as np


class Deduplicator:
    """
        Two-stage deduplication used by PRISM:
            1. Exact dedup  — MD5 hash of normalised text
            2. Near dedup   — cosine similarity on sentence embeddings

        """

    def __init__(self, near_dup_threshold: float = 0.92):
        self.threshold = near_dup_threshold

    # Stage 1 — exact deduplication using MD5 hashes

    def exact_dedup(self, sentences: List[str]) -> List[str]:
        seen, unique = set(), []
        for s in sentences:
            h = self._hash(s)
            if h not in seen:
                seen.add(h)
                unique.append(s)
        return unique

    # Stage 2 — near-duplicate removal using embedding cosine similarity

    def near_dedup(
        self, sentences: List[str], embeddings: np.ndarray
    ) -> Tuple[List[str], np.ndarray]:
        """
        Greedy: keep a sentence only if its cosine similarity
        to every already-kept sentence is below the threshold.
        """
        if not sentences:
            return sentences, embeddings

        kept_idx = [0]
        kept_embs = [embeddings[0]]

        for i in range(1, len(sentences)):
            emb = embeddings[i]
            sims = [
                self._cosine(emb, k) for k in kept_embs
            ]
            if max(sims) < self.threshold:
                kept_idx.append(i)
                kept_embs.append(emb)

        return (
            [sentences[i] for i in kept_idx],
            np.array(kept_embs),
        )

    # Helper utility methods

    @staticmethod
    def _hash(text: str) -> str:
        normalised = " ".join(text.lower().split())
        return hashlib.md5(normalised.encode()).hexdigest()

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
