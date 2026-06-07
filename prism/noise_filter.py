import re
from typing import List
from typing import List

# Regex patterns that indicate low-value or boilerplate content
_NOISE_PATTERNS = [
    r"(subscribe|newsletter|sign[\s-]?up|log[\s-]?in|cookie policy|privacy policy|terms of service)",
    r"(click here|read more|learn more|find out more|see more|show more)",
    r"(advertisement|sponsored content?|follow us|share this|retweet|like us on)",
    r"(all rights reserved|copyright\s*©?\s*\d{4}|©\s*\d{4})",
    r"^(home|about|contact|blog|news|search|menu|navigation|sitemap)\s*$",
    r"^\s*[\d\W]+\s*$",   # lines that are just numbers or punctuation
]


class NoiseFilter:
    """
    Removes boilerplate, SEO filler, and structurally low-value sentences.
    Operates on plain heuristics — no model needed.
    This is the same noise filtering used by PRISM's IOL.
    """
    
    def __init__(self, min_length: int = 20):
        self.min_length = min_length
        self._patterns = [re.compile(p, re.IGNORECASE) for p in _NOISE_PATTERNS]

    def is_noise(self, sentence: str) -> bool:
        s = sentence.strip()

        # treat very short sentences as noise
        if len(s) < self.min_length:
            return True

        # match against known low-value patterns
        if any(p.search(s) for p in self._patterns):
            return True

        # detect mostly-nonalphabetic garbage lines
        alpha_ratio = sum(c.isalpha() for c in s) / max(len(s), 1)
        if alpha_ratio < 0.40:
            return True

        return False

    def filter(self, sentences: List[str]) -> List[str]:
        return [s for s in sentences if not self.is_noise(s)]
