"""
The hardened parser. Five rungs, deterministic, no resampling.

  rung 1  guided       constrained decoding produced valid JSON (s3_infer)
  rung 2  strict       json.loads on the raw completion succeeded
  rung 3  repaired     bounded DETERMINISTIC document repair, then json.loads
  rung 4  normalised   parsed, but the label needed vocabulary normalisation
  rung 5  sentinel     unrecoverable -> SENTINEL_LABEL, full raw retained

The rung that produced each record is written to `parse_path`. With rung 1
active, rungs 2-5 should be nearly dead code; they are kept as the net.

TWO PROHIBITIONS:

  * NEVER assemble a record field by field from fragments. Scraping
    `probability` with a regex independent of the label can produce records
    such as predicted_label="Not ILD" with ild_probability=0.998. Rung 3
    repairs the DOCUMENT or fails.

  * NEVER cap the retained raw completion: a truncated raw response destroys
    exactly the evidence needed to diagnose a failure. `raw_response` is
    stored in full.

There is no temperature escalation and no retry of any kind. Remediation for a
sentinel is a deterministic re-run of the affected batches, never resampling.
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, Optional

from .vocabulary import CANONICAL_LABELS, normalise_label, is_refusal

SENTINEL_LABEL = "UNPARSEABLE"

PARSE_PATHS = ("guided", "strict", "repaired", "normalised", "sentinel")

# Accepted spellings of the label key. The guided schema emits the long form
# ("ILD on the index date"); "label" is also read so any raw completion can be
# re-parsed by this module without modification.
_LABEL_KEYS = ("label", "ILD on the index date", "ild_on_the_index_date")
_PROB_KEYS  = ("probability", "ild_probability")
_CONF_KEYS  = ("confidence", "confidence_score")
_REASON_KEYS = ("reasoning", "primary_reason", "reason")
_EVIDENCE_KEYS = ("key_evidence", "evidence")
_CONCORDANCE_KEYS = ("modality_concordance",)
_VALID_CONCORDANCE = {"concordant", "discordant", "single_modality"}


# ── rung 3 helpers: document-level repair only ────────────────────────────────
def _strip_wrappers(text: str) -> str:
    """Remove code fences and reasoning-trace tags. Whole-document operations."""
    t = text.strip()
    t = re.sub(r"^\s*```(?:json)?\s*", "", t)
    t = re.sub(r"```\s*$", "", t)
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL)
    t = re.sub(r"<thought>.*?</thought>", "", t, flags=re.DOTALL)
    t = re.sub(r"<\|.*?\|>", "", t)
    return t.strip()


def _outermost_object(text: str) -> Optional[str]:
    """Extract the outermost balanced {...}, respecting strings and escapes.

    A brace-counting scan, not a greedy regex: a greedy `\\{.*\\}` would
    swallow trailing prose and any second object.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _repair_document(blob: str) -> Optional[dict]:
    """Bounded deterministic repairs, tried in order. Whole-document only."""
    candidates = [
        blob,
        re.sub(r",\s*([}\]])", r"\1", blob),                       # trailing commas
        re.sub(r",\s*([}\]])", r"\1", blob).replace("'", '"'),     # single quotes
        unicodedata.normalize("NFKC", re.sub(r",\s*([}\]])", r"\1", blob)),
    ]
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


# ── field extraction from an ALREADY-PARSED object ────────────────────────────
def _first_key(obj: dict, keys) -> Any:
    for k in keys:
        if k in obj:
            return obj[k]
    return None


def _clip_unit(v, default: Optional[float]) -> Optional[float]:
    try:
        return round(max(0.0, min(1.0, float(v))), 4)
    except (TypeError, ValueError):
        return default


def _coerce_evidence(v) -> list:
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        return []
    return [str(x)[:300] for x in v[:5]]


def _record_from_object(obj: dict, config: str) -> Optional[Dict[str, Any]]:
    """Build a record from one parsed JSON object, or None if the label is not
    recoverable. Fields are read from a SINGLE object -- never merged across
    sources -- so label and probability always come from the same completion."""
    raw_label = _first_key(obj, _LABEL_KEYS)
    if raw_label is None:
        return None

    exact = str(raw_label).strip() if isinstance(raw_label, str) else None
    if exact in CANONICAL_LABELS:
        label, needed_norm = exact, False
    else:
        label = normalise_label(raw_label)
        needed_norm = True
        if label is None:
            return None            # includes refusals; never coerced

    prob = _clip_unit(_first_key(obj, _PROB_KEYS), None)
    conf = _clip_unit(_first_key(obj, _CONF_KEYS), 0.0)
    if prob is None:
        # A record with a label but no usable probability is incomplete.
        # No imputation: fall to the sentinel.
        return None

    rec = {
        "predicted_label": label,
        "ild_probability": prob,
        "confidence_score": conf,
        "reasoning": str(_first_key(obj, _REASON_KEYS) or "")[:4000],
        "key_evidence": _coerce_evidence(_first_key(obj, _EVIDENCE_KEYS)),
        "_needed_normalisation": needed_norm,
        "_raw_label": str(raw_label)[:200],
    }
    if config == "fused":
        mc = str(_first_key(obj, _CONCORDANCE_KEYS) or "").strip().lower()
        rec["modality_concordance"] = mc if mc in _VALID_CONCORDANCE else ""
    return rec


def sentinel_record(raw: str, config: str, reason: str) -> Dict[str, Any]:
    """Rung 5: explicit sentinel. Propagates into aggregation and metrics
    and is counted; never filled, guessed or dropped."""
    rec = {
        "predicted_label": SENTINEL_LABEL,
        "ild_probability": None,
        "confidence_score": None,
        "reasoning": "",
        "key_evidence": [],
        "_needed_normalisation": False,
        "_raw_label": "",
        "parse_path": "sentinel",
        "parse_ok": False,
        "sentinel_reason": reason,
    }
    if config == "fused":
        rec["modality_concordance"] = ""
    return rec


def parse_completion(raw: str, config: str, guided: bool) -> Dict[str, Any]:
    """Parse one raw completion. Returns a record carrying `parse_path`.

    `guided=True` only affects the LABEL of the successful path (rung 1 vs 2):
    the arithmetic is identical, so a guided and an unguided run of the same
    completion produce the same record.
    """
    if raw is None or not str(raw).strip():
        return sentinel_record(raw or "", config, "empty completion")

    # rungs 1/2 -- strict parse of the raw text
    obj, path = None, None
    try:
        cand = json.loads(raw)
        if isinstance(cand, dict):
            obj, path = cand, ("guided" if guided else "strict")
    except (json.JSONDecodeError, ValueError):
        obj = None

    # rung 3 -- bounded deterministic document repair
    if obj is None:
        stripped = _strip_wrappers(raw)
        blob = _outermost_object(stripped)
        if blob is not None:
            obj = _repair_document(blob)
            path = "repaired"

    if obj is None:
        return sentinel_record(raw, config, "no parseable JSON object")

    rec = _record_from_object(obj, config)
    if rec is None:
        reason = ("refusal label" if is_refusal(_first_key(obj, _LABEL_KEYS))
                  else "label or probability not recoverable")
        return sentinel_record(raw, config, reason)

    # rung 4 -- record that vocabulary normalisation was needed
    if rec.pop("_needed_normalisation"):
        path = "normalised"

    rec["parse_path"] = path
    rec["parse_ok"] = True
    rec["sentinel_reason"] = ""
    return rec
