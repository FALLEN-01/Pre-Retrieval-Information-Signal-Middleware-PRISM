import anthropic

class LLMConsumer:
    """
    Passes the IOL-optimised context to an LLM consumer and returns an answer.
    Docstrings retained.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model  = model

    def answer(self, query: str, optimized_context: str, metadata: dict) -> str:
        ratio = metadata.get("compression_ratio", "?")

        system = (
            "You are a precise, factual Q&A assistant. "
            "Answer the question using ONLY the provided context. "
            "If the context is insufficient, say so clearly. "
            "Be concise. Do not add information not present in the context."
        )

        user_message = (
            f"Context (IOL-optimised · {ratio}× compression · "
            f"{metadata.get('conflicts_detected', 0)} conflict(s) flagged):\n\n"
            f"{optimized_context}\n\n"
            f"Question: {query}"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )

        return response.content[0].text
