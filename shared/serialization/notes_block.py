"""
Clinical-notes section formatter for one batch.


Used identically by the notes and fused configurations — the same
selected notes produce the same rendered text in both.


Format
------
=== CLINICAL NOTES ===

Clinical notes recorded prior to the index date:
--- <note_type> (<N> days before index date) ---
<note text>

Clinical notes recorded on the index date:
  None

Clinical notes recorded after the index date:
--- <note_type> (<N> days after index date) ---
<note text>

Within each section notes are shown in chronological order.
"""
from __future__ import annotations
import pandas as pd


def _fmt_day(d: int) -> str:
    if d < 0:  return f"{abs(d)} days before index date"
    if d == 0: return "on index date"
    return f"{d} days after index date"


def build_notes_block(
    batch_df: pd.DataFrame,
    text_col:    str = "text_clean",
    type_col:    str = "note_type",
    rel_day_col: str = "rel_day",
) -> str:
    """Render one batch's notes as before/on/after sections, chronological within each."""
    if len(batch_df) == 0:
        return "=== CLINICAL NOTES ===\n\n(no notes in this batch)\n"

    before = batch_df[batch_df[rel_day_col]  < 0].sort_values(rel_day_col, ascending=False)
    on     = batch_df[batch_df[rel_day_col] == 0]
    after  = batch_df[batch_df[rel_day_col]  > 0].sort_values(rel_day_col)

    lines = ["=== CLINICAL NOTES ===", ""]

    def _section(df_sec: pd.DataFrame, label: str):
        lines.append(f"Clinical notes recorded {label} the index date:")
        if len(df_sec) == 0:
            lines.append("  None")
            lines.append("")
            return
        for _, row in df_sec.iterrows():
            ntype = str(row.get(type_col, "Clinical Note"))
            day   = int(row[rel_day_col])
            text  = str(row[text_col]).strip()
            lines.append(f"--- {ntype} ({_fmt_day(day)}) ---")
            lines.append(text)
            lines.append("")

    _section(before, "prior to")
    _section(on,     "on")
    _section(after,  "after")

    return "\n".join(lines)