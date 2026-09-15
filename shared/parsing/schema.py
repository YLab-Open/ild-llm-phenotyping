"""
JSON schema for vLLM guided (constrained) decoding.

Constrained decoding masks any token that would make the output invalid, so
malformed JSON and out-of-vocabulary labels become UNREACHABLE rather than
merely improbable. This addresses the dominant failure mode: small models do
not fail to follow instructions, they lack the capacity to hold the format.

The schema is the response contract of the prompts, expressed as a grammar.

Three properties are deliberate:

1. The DECISION FIELDS COME FIRST -- label, probability, confidence. An
   unbounded free-text field placed first invites a degenerate repetition loop
   under greedy decoding at temperature 0 with no repetition penalty; if that
   loop consumes the whole token budget, the label and probability are never
   emitted and the record is unrecoverable by construction. With the decision
   fields first, a generation cut short still carries the label, and rung 3
   can close the object and recover the record.

2. `label` is an ENUM over exactly the three canonical strings, so "Uncertain"
   and free-text variants cannot be produced at all. The key is the long form,
   "ILD on the index date".

3. EVERY free-text field is BOUNDED. xgrammar in this build enforces
   `maxLength` on strings (verified: a 40-character string is rejected against
   maxLength 10), so runaway generation is structurally impossible rather than
   merely unlikely.

Per-configuration field sets follow the prompts and are NOT uniform: claims
has no `key_evidence`; only fused has `modality_concordance`.

Guided decoding is all-models-or-no-models (config.GUIDED_DECODING).
s3_infer refuses to run a cell if the installed vLLM does not expose it.
"""
from .vocabulary import CANONICAL_LABELS

# Long-form label key. parser._LABEL_KEYS already accepts it.
LABEL_KEY = "ILD on the index date"

# Bounds. The fused prompt caps key_evidence at 5 in prose; that cap is
# enforced by the grammar. The notes prompt states no cap, but leaving it
# unbounded would leave the repetition-loop failure mode open, so the cap is
# applied to notes as well — the one place the enforced schema is stricter
# than the prompt prose.
MAX_KEY_EVIDENCE      = 5
MAX_EVIDENCE_CHARS    = 200      # per item
MAX_PRIMARY_REASON    = 400      # "<one sentence>"

CONCORDANCE_VALUES = ["concordant", "discordant", "single_modality"]

_LABEL_PROP = {
    "type": "string",
    "enum": list(CANONICAL_LABELS),
    "description": "ILD status on the index date.",
}
_PROBABILITY_PROP = {
    "type": "number", "minimum": 0.0, "maximum": 1.0,
    "description": "Probability the patient has ILD on the index date.",
}
_CONFIDENCE_PROP = {
    "type": "number", "minimum": 0.0, "maximum": 1.0,
    "description": "Confidence in the stated label.",
}
_KEY_EVIDENCE_PROP = {
    "type": "array",
    "items": {"type": "string", "maxLength": MAX_EVIDENCE_CHARS},
    "maxItems": MAX_KEY_EVIDENCE,
    "description": "At most 5 decisive findings, each stated once.",
}
_PRIMARY_REASON_PROP = {
    "type": "string", "maxLength": MAX_PRIMARY_REASON,
    "description": "One sentence.",
}
_CONCORDANCE_PROP = {
    "type": "string", "enum": CONCORDANCE_VALUES,
    "description": "Agreement between claims and notes.",
}

# Field ORDER per configuration, matching the prompts exactly.
_FIELD_ORDER = {
    "claims": [LABEL_KEY, "probability", "confidence", "primary_reason"],
    "notes":  [LABEL_KEY, "probability", "confidence", "key_evidence",
               "primary_reason"],
    "fused":  [LABEL_KEY, "probability", "confidence", "key_evidence",
               "primary_reason", "modality_concordance"],
}

_PROP_BY_NAME = {
    LABEL_KEY:               _LABEL_PROP,
    "probability":           _PROBABILITY_PROP,
    "confidence":            _CONFIDENCE_PROP,
    "key_evidence":          _KEY_EVIDENCE_PROP,
    "primary_reason":        _PRIMARY_REASON_PROP,
    "modality_concordance":  _CONCORDANCE_PROP,
}


def response_schema(config: str) -> dict:
    """JSON schema for one configuration. `config` in {claims, notes, fused}."""
    if config not in _FIELD_ORDER:
        raise ValueError(f"unknown config {config!r}; expected one of "
                         f"{list(_FIELD_ORDER)}")
    order = list(_FIELD_ORDER[config])
    return {
        "type": "object",
        "properties": {name: dict(_PROP_BY_NAME[name]) for name in order},
        # `required` fixes the emitted key order under most guided backends and
        # documents the intended order regardless.
        "required": order,
        "additionalProperties": False,
    }


def schema_field_order(config: str) -> list:
    return list(response_schema(config)["required"])