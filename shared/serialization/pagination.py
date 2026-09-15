"""
Multi-pass note batcher, used by BOTH the notes and fused configurations.

Selection order:
  1. Each note's ild_priority maps to a packing rank via C.TIER_PRIORITY.
  2. Notes are sorted (tier_priority ASC, rel_day DESC) — high-priority tier
     first, then most-recent within tier.
  3. Greedy pack into batches with budget_per_pass tokens each.
  4. Display order WITHIN a batch is chronological (build_notes_block).

Tokenizer-fair costing: `annotate_notes` costs each note under EVERY registry
tokenizer and stores the per-tokenizer counts plus their maximum. Pagination
packs against the maximum, so a batch that fits the budget fits every model.
Costing under a single tokenizer instead would silently disadvantage models
whose tokenizer is more expensive — a confound along exactly the cross-family
axis the study compares. There is no truncation path at inference time at all:
a batch that would overflow for any model is a serialization bug for v1 to
catch, not a runtime condition to absorb.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared import config as C

log = logging.getLogger("PAGINATION")

# Fixed per-note rendering overhead (the "--- <type> (<n> days ...) ---" header
# and blank lines).
NOTE_OVERHEAD_TOKENS = 50


def days_from_ced(event_date, ced) -> Optional[int]:
    try:
        return (datetime.fromisoformat(str(event_date)[:10]) -
                datetime.fromisoformat(str(ced)[:10])).days
    except Exception:
        return None


def annotate_notes(notes_df: pd.DataFrame, ced: str,
                   text_col: str, date_col: str,
                   tokenizers: Dict[str, object]) -> pd.DataFrame:
    """Adds rel_day, tier_priority, per-tokenizer token counts and note_tokens.

    Parameters
    ----------
    tokenizers : {model_name: tokenizer}
        ALL registry tokenizers, not one. `note_tokens` becomes the
        per-note MAXIMUM across them, which is what pagination packs against.
        Per-tokenizer columns `note_tokens__<model>` are retained so
        serialization_stats.csv can prove zero projected truncation (v1).

    Drops rows with unparseable dates or empty text. Returns a copy.
    """
    if not tokenizers:
        raise ValueError("annotate_notes requires at least one tokenizer")

    df = notes_df.copy()
    df["rel_day"] = df[date_col].apply(lambda d: days_from_ced(d, ced))
    df = df[df["rel_day"].notna()].copy()
    df["rel_day"] = df["rel_day"].astype(int)
    df["tier_priority"] = df["ild_priority"].map(C.TIER_PRIORITY).fillna(99).astype(int)
    df = df[df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")].copy()
    if len(df) == 0:
        return df

    texts = df[text_col].astype(str).tolist()
    for name, tok in tokenizers.items():
        counts = [len(tok.encode(t)) + NOTE_OVERHEAD_TOKENS for t in texts]
        df[f"note_tokens__{name}"] = counts

    cols = [f"note_tokens__{n}" for n in tokenizers]
    df["note_tokens"] = df[cols].max(axis=1).astype(int)          # worst case
    df["note_tokens_argmax"] = df[cols].idxmax(axis=1).str.replace(
        "note_tokens__", "", regex=False)
    return df


def prefix_tokens_max(text: str, tokenizers: Dict[str, object]) -> Dict[str, int]:
    """Cost a fixed prefix under every tokenizer.

    Returns {'max': int, 'argmax': model, 'per_model': {...}}. The budget must
    be reduced by the MAXIMUM, otherwise a patient whose claims prefix is
    cheap under one tokenizer but expensive under another overflows for the
    latter only.
    """
    per = {name: len(tok.encode(text)) for name, tok in tokenizers.items()}
    argmax = max(per, key=per.get)
    return {"max": per[argmax], "argmax": argmax, "per_model": per}


def paginate(notes_df: pd.DataFrame, budget_per_pass: int,
             patient_id: str = "") -> List[List]:
    """Greedy multi-pass batcher.

    Parameters
    ----------
    notes_df          must have tier_priority, rel_day, note_tokens
                      (note_tokens = worst-case cost, from annotate_notes)
    budget_per_pass   token budget for notes in one batch; the caller has
                      already subtracted the worst-case fixed prefix
    patient_id        for logging only

    Returns a list of lists of DataFrame index values, in packing order.
    """
    if len(notes_df) == 0:
        return []
    if budget_per_pass <= 0:
        log.warning(f"  [{patient_id}] budget_per_pass={budget_per_pass} <= 0 — no batches")
        return []

    df = notes_df.sort_values(["tier_priority", "rel_day"], ascending=[True, False])

    batches: List[List] = []
    current: List = []
    current_tokens = 0

    for idx, row in df.iterrows():
        nt = int(row["note_tokens"])
        if nt > budget_per_pass:
            # An oversize single note is still placed alone, but it is a HARD
            # ERROR for v1 rather than something inference absorbs by slicing
            # characters. See v1_serialization.
            log.error(
                f"  [{patient_id}] single note costs {nt} tokens under "
                f"{row.get('note_tokens_argmax', '?')}, exceeding the per-pass "
                f"budget ({budget_per_pass}). Placed alone; v1 WILL FAIL on this."
            )
            if current:
                batches.append(current)
                current, current_tokens = [], 0
            batches.append([idx])
            continue
        if current_tokens + nt > budget_per_pass:
            batches.append(current)
            current = [idx]
            current_tokens = nt
        else:
            current.append(idx)
            current_tokens += nt
    if current:
        batches.append(current)
    return batches
