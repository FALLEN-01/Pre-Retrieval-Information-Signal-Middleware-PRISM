from typing import List
import numpy as np


class MMRCompressor:
    """
    Maximal Marginal Relevance (MMR) extractive compression used by PRISM.

    At each step, picks the sentence that maximises:

        MMR = λ · relevance(s, query)  −  (1−λ) · max_sim(s, already_selected)

    This single pass handles:
      • compression      — stops at token budget
      • relevance        — high-relevance sentences selected first
      • diversity        — penalises sentences too similar to what's already selected
      • implicit dedup   — near-duplicates score poorly on the second term

    λ = 1.0  →  pure relevance (greedy top-k)
    λ = 0.0  →  pure diversity (maximum spread)
    λ = 0.6  →  recommended balance

    
    """

    def __init__(self, mmr_lambda: float = 0.6):
        self.lam = mmr_lambda

    def compress(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        query_embedding: np.ndarray,
        token_budget: int = 2000,
    ) -> List[str]:
        if not sentences:
            return []

        # Pre-compute relevance of each sentence to the query embedding
        relevance = np.array([self._cos(embeddings[i], query_embedding) for i in range(len(sentences))])

        selected: List[str] = []
        selected_embs: List[np.ndarray] = []
        remaining = list(range(len(sentences)))
        token_count = 0

        while remaining:
            best_idx, best_score = None, -float("inf")

            for i in remaining:
                rel = relevance[i]
                max_sim = (
                    max(self._cos(embeddings[i], s) for s in selected_embs)
                    if selected_embs
                    else 0.0
                )
                score = self.lam * rel - (1 - self.lam) * max_sim

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx is None:
                break

            # approximate token count (1 token ≈ 4 characters)
            sentence_tokens = len(sentences[best_idx]) // 4
            if token_count + sentence_tokens > token_budget:
                break

            selected.append(sentences[best_idx])
            selected_embs.append(embeddings[best_idx])
            token_count += sentence_tokens
            remaining.remove(best_idx)

        return selected

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
