"""
Closed label vocabulary and the explicit synonym map.

The map is a constant, not a heuristic, and is logged in full at the start of
every inference run so that any normalisation applied to a cell is visible in
its log rather than inferred.

Design rule: normalisation may only map a raw string onto the closed set. It
may never invent a label from a probability, a prose fragment, or a value that
does not name one of the three classes. Anything not in the map after
normalisation is NOT coerced — it becomes the sentinel.
"""
from typing import Optional
import re
import unicodedata

# The closed set. Order is the positivity ordering used by LABEL_PRIORITY.
CANONICAL_LABELS = ("Probable ILD", "Possible ILD", "Not ILD")

# Explicit synonym map: normalised key -> canonical label.
# Keys are lowercased, punctuation-stripped, whitespace-collapsed.
# Every entry names a class. None is a guess from surrounding evidence.
SYNONYM_MAP = {
    # exact canonical forms
    "probable ild":            "Probable ILD",
    "possible ild":            "Possible ILD",
    "not ild":                 "Not ILD",
    # bare degree words
    "probable":                "Probable ILD",
    "possible":                "Possible ILD",
    # common re-orderings / expansions observed in raw completions
    "ild probable":            "Probable ILD",
    "ild possible":            "Possible ILD",
    "probable interstitial lung disease": "Probable ILD",
    "possible interstitial lung disease": "Possible ILD",
    "no interstitial lung disease":       "Not ILD",
    "not interstitial lung disease":      "Not ILD",
    "no ild":                  "Not ILD",
    "ild not present":         "Not ILD",
    "ild absent":              "Not ILD",
    "absent":                  "Not ILD",
    "none":                    "Not ILD",
    "negative":                "Not ILD",
}

# Strings that unambiguously indicate the model declined to classify. These are
# recorded distinctly from "unrecognised" so v2 can report them separately, but
# they are treated identically: sentinel, never coerced.
REFUSAL_TOKENS = {"uncertain", "unknown", "indeterminate", "insufficient",
                  "insufficient evidence", "cannot determine", "unable to determine"}


def _normalise(s: str) -> str:
    """Case-fold, strip accents and punctuation, collapse whitespace."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.strip().lower()
    s = re.sub(r"[^a-z ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalise_label(raw) -> Optional[str]:
    """Map a raw label string onto the closed set, or return None.

    None means 'not recognised' and MUST become the sentinel upstream. This
    function never returns a label that was not named in the input.
    """
    key = _normalise(raw)
    if not key:
        return None
    return SYNONYM_MAP.get(key)


def is_refusal(raw) -> bool:
    return _normalise(raw) in REFUSAL_TOKENS


def vocabulary_report() -> str:
    """Full dump of the vocabulary, logged at inference startup."""
    lines = ["Label vocabulary (closed set, no coercion outside it):",
             f"  canonical: {list(CANONICAL_LABELS)}",
             f"  synonym map ({len(SYNONYM_MAP)} entries):"]
    for k in sorted(SYNONYM_MAP):
        lines.append(f"    {k!r:<40} -> {SYNONYM_MAP[k]}")
    lines.append(f"  refusal tokens (-> sentinel, never coerced): {sorted(REFUSAL_TOKENS)}")
    return "\n".join(lines)
