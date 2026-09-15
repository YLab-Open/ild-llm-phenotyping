"""
analysis/decomposition.py — CPU only. Per-patient modality decomposition and
head-to-head comparison against the published rules, for EVERY model.

    python analysis/decomposition.py

Two tables:

  1. modality decomposition — for each model, how the claims and notes calls
     combine: caught by both / claims only / notes only / neither, and how many
     patients the fused configuration rescues or loses relative to each single
     modality.

  2. head-to-head vs the 20 rules — per model x config x rule, on the identical
     309 patients joined by EMPI, with a McNemar exact test.

Outputs: analysis/results/decomposition/{modality_decomposition,head_to_head}.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import config as C
from shared.evaluation.metrics import mcnemar_exact, binary_metrics

OUT = C.CLEAN_ROOT / "analysis" / "results" / "decomposition"


def load_cell(model, config, tier):
    name = config if tier is None else f"{config}_tier{tier}"
    p = C.RESULTS_ROOT / model / name / "patient_predictions.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d["empi"] = d["empi"].astype(str).str.strip()
    d["pos"] = d["predicted_label"].isin(C.POSITIVE_LABELS)
    d["gold_pos"] = d["gold_label"].isin(C.POSITIVE_LABELS)
    return d.set_index("empi")


def main(tier):
    OUT.mkdir(parents=True, exist_ok=True)

    # ── 1. modality decomposition ─────────────────────────────────────────────
    rows = []
    for model in C.ALL_MODELS:
        cells = {cfg: load_cell(model, cfg, tier) for cfg in C.CONFIGS}
        if any(v is None for v in cells.values()):
            continue
        idx = cells["claims"].index
        cl, no, fu = (cells["claims"].loc[idx], cells["notes"].loc[idx],
                      cells["fused"].loc[idx])
        gold = cl["gold_pos"]
        both = (cl["pos"] & no["pos"])
        c_only = (cl["pos"] & ~no["pos"])
        n_only = (~cl["pos"] & no["pos"])
        neither = (~cl["pos"] & ~no["pos"])
        rows.append({
            "model": model, "reported": C.MODELS[model]["reported"],
            "n_patients": len(idx),
            "flagged_by_both": int(both.sum()),
            "flagged_claims_only": int(c_only.sum()),
            "flagged_notes_only": int(n_only.sum()),
            "flagged_by_neither": int(neither.sum()),
            "tp_both": int((both & gold).sum()),
            "tp_claims_only": int((c_only & gold).sum()),
            "tp_notes_only": int((n_only & gold).sum()),
            "fused_rescues_vs_claims": int((fu["pos"] & ~cl["pos"] & gold).sum()),
            "fused_rescues_vs_notes": int((fu["pos"] & ~no["pos"] & gold).sum()),
            "fused_loses_vs_claims": int((~fu["pos"] & cl["pos"] & gold).sum()),
            "fused_loses_vs_notes": int((~fu["pos"] & no["pos"] & gold).sum()),
            "tp_claims": int((cl["pos"] & gold).sum()),
            "tp_notes": int((no["pos"] & gold).sum()),
            "tp_fused": int((fu["pos"] & gold).sum()),
            "n_gold_positive": int(gold.sum()),
        })
    dec = pd.DataFrame(rows)
    dec.to_csv(OUT / "modality_decomposition.csv", index=False)

    # ── 2. head-to-head vs every published rule ───────────────────────────────
    meta = pd.read_excel(C.SHEN_XLSX)
    meta["EMPI"] = meta["EMPI"].astype(str).str.strip()
    meta = meta.set_index("EMPI")

    h2h = []
    for model in C.ALL_MODELS:
        for cfg in C.CONFIGS:
            d = load_cell(model, cfg, tier)
            if d is None:
                continue
            idx = d.index.intersection(meta.index)
            gold = d.loc[idx, "gold_pos"].values
            llm = d.loc[idx, "pos"].values
            for algo in [C.SHEN_SYNTHESIZED_ALGO] + C.SHEN_MEASURED_ALGOS:
                if algo == C.SHEN_SYNTHESIZED_ALGO:
                    rule = pd.Series(True, index=idx).values      # flag-all
                elif algo in meta.columns:
                    rule = (meta.loc[idx, algo].fillna(0).astype(int) == 1).values
                else:
                    continue
                b = int(((llm == gold) & (rule != gold)).sum())
                c = int(((rule == gold) & (llm != gold)).sum())
                net, half, p = mcnemar_exact(b, c)
                lm = binary_metrics(int((llm & gold).sum()), int((llm & ~gold).sum()),
                                    int((~llm & ~gold).sum()), int((~llm & gold).sum()))
                rm = binary_metrics(int((rule & gold).sum()), int((rule & ~gold).sum()),
                                    int((~rule & ~gold).sum()), int((~rule & gold).sum()))
                h2h.append({
                    "model": model, "config": cfg, "rule": algo,
                    "rule_is_synthesized": algo == C.SHEN_SYNTHESIZED_ALGO,
                    "n": len(idx),
                    "llm_ppv": lm["ppv"], "llm_yield": lm["yield"],
                    "rule_ppv": rm["ppv"], "rule_yield": rm["yield"],
                    "ppv_ratio": (lm["ppv"] / rm["ppv"]) if rm["ppv"] else float("nan"),
                    "yield_ratio": (lm["yield"] / rm["yield"]) if rm["yield"] else float("nan"),
                    "llm_dominates": bool(lm["ppv"] >= rm["ppv"] and lm["yield"] >= rm["yield"]),
                    "mcnemar_b_llm_right_rule_wrong": b,
                    "mcnemar_c_rule_right_llm_wrong": c,
                    "mcnemar_net": net, "mcnemar_ci_half_width": half,
                    "mcnemar_exact_p": p,
                })
    pd.DataFrame(h2h).to_csv(OUT / "head_to_head.csv", index=False)

    print(f"modality decomposition: {len(dec)} models -> "
          f"{OUT/'modality_decomposition.csv'}")
    if len(dec):
        print(dec[["model", "tp_claims", "tp_notes", "tp_fused",
                   "fused_rescues_vs_claims", "fused_rescues_vs_notes",
                   "n_gold_positive"]].to_string(index=False))
    print(f"\nhead-to-head: {len(h2h)} model x config x rule rows -> "
          f"{OUT/'head_to_head.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=None)
    main(ap.parse_args().tier)
