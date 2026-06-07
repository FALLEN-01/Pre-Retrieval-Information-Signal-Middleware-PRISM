import re
from typing import List, Tuple, Dict, Any

# Regex captures simple numeric claims of the form 'subject verb numeric_value'
# Example matches: "Revenue grew 10%", "Temperature is 42 degrees"
_CLAIM_RE = re.compile(
    r"([a-zA-Z][\w\s]{2,29}?)\s+"
    r"(is|was|were|grew|increased|decreased|fell|dropped|rose|stands?|reached?|hit)\s+"
    r"([+-]?\d[\d,]*\.?\d*\s*(?:%|percent|degrees?|units?|million|billion|trillion)?)",
    re.IGNORECASE,
)


class ContradictionDetector:
    """
    Lightweight pattern-based contradiction detection used in PRISM.
    Identifies sentences that make conflicting numeric claims about the same subject.

    
    """

    def detect(
        self, sentences: List[str]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Returns
        -------
        clean_sentences : original sentences minus those involved in conflicts,
                          plus appended conflict-note strings.
        conflicts       : list of conflict dicts for downstream inspection.
        """
        claims: Dict[str, Tuple[str, int]] = {}  # key -> (value, sentence_index)
        flagged: set[int] = set()
        conflicts: List[Dict[str, Any]] = []

        for i, sentence in enumerate(sentences):
            for m in _CLAIM_RE.finditer(sentence):
                subject = m.group(1).strip().lower()
                verb    = m.group(2).strip().lower()
                value   = m.group(3).strip()
                key     = f"{subject}__{verb}"

                if key in claims:
                    prev_value, prev_idx = claims[key]
                    # flag only when numeric values differ for the same subject+verb
                    if self._normalise(prev_value) != self._normalise(value):
                        conflicts.append({
                            "subject":    subject,
                            "sentence_a": sentences[prev_idx],
                            "sentence_b": sentence,
                            "value_a":    prev_value,
                            "value_b":    value,
                        })
                        flagged.add(i)
                        flagged.add(prev_idx)
                else:
                    claims[key] = (value, i)

        # Build output — keep non-flagged sentences and append conflict notes
        clean = [s for i, s in enumerate(sentences) if i not in flagged]
        notes = [
            f"[CONFLICT — confidence reduced] "
            f"'{c['subject']}': one source says {c['value_a']}, another says {c['value_b']}."
            for c in conflicts
        ]

        return clean + notes, conflicts

    @staticmethod
    def _normalise(value: str) -> str:
        """Strip whitespace and commas for comparison."""
        return value.lower().replace(",", "").replace(" ", "")
