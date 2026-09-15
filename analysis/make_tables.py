#!/usr/bin/env python3
"""
analysis/make_tables.py — CPU only. Regenerates the manuscript's data tables as
LaTeX fragments.

    python analysis/make_tables.py                 # all tables
    python analysis/make_tables.py --table S4      # one
    python analysis/make_tables.py --model Qwen3-32B --table S3 S5

Writes analysis/results/tables/<name>.tex, each a complete table environment with
the same structure, column spec and caption wording as the current main.tex, so a
fragment can be pasted over the old block or pulled in with \\input{}.

Covered here:
    S3  three-way confusion matrices          (illustrative model)
    S4  full performance: rules + LLM grid
    S5  matched-cohort head-to-head by note tier
    S6  patient-level aggregation strategies
    S7  system prompts, verbatim
    S8  per-patient modality decomposition, all models

NOT covered (unchanged, or not data-driven):
    Table 1  cohort characteristics  — reproduces exactly; emitted with --table T1
             if you want it rebuilt from core_cohort.csv
    S1  J-codes         — static, from config.SHEN/J-code list
    S2  note classes    — static, from config.TIER_PRIORITY / TIER_LABEL

TWO QUANTITIES ARE COMPUTED HERE rather than read from an analysis CSV, because
no analysis script currently writes them:
  * S5 needs each published rule's PPV and yield ON EACH TIER SUBCOHORT, which
    no other script saves.
  * S8 needs, among the gold-positive cases that NEITHER single modality caught,
    how many the fused configuration recovered. modality_decomposition.csv
    carries fused_rescues_vs_claims / _vs_notes, which are different quantities.

REPORTED METRICS: on a code-screened cohort, sensitivity and specificity do not
carry their usual population meaning, so no table here reports them. PPV and
yield only, with Wilson intervals throughout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import config as C

OUT = C.CLEAN_ROOT / "analysis" / "results" / "tables"
SHEN_METRICS = (C.CLEAN_ROOT / "analysis" / "results" / "shen_evaluation" /
                "shen_metrics.csv")
GRID = C.RESULTS_ROOT / "grid_metrics.csv"
AGG = C.CLEAN_ROOT / "analysis" / "results" / "aggregation" / "aggregation_rules.csv"

CONFIG_LABEL = {"claims": "Claims", "notes": "Notes", "fused": "Fused"}
TIER_ROWS = [  # (tier_id, display name used in the manuscript)
    (1, "Chest CT"), (5, "Pulmonary function tests"), (3, "Discharge summaries"),
    (2, "Chest X-ray"), (6, "Progress notes"),
]
MIN_FLAG_TIER = 11          # Shen et al. mask sparser variants
LABELS3 = ["Probable ILD", "Possible ILD", "Not ILD"]


_DISP_OVERRIDE = {
    "medgemma-27b-text-it":   "MedGemma-27B",
    "medgemma-4b-it":         "MedGemma-4B",
    "gemma-3-27b-it":         "Gemma-3-27B",
    "Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "Qwen2.5-72B-Instruct":   "Qwen2.5-72B",
}


def _disp(model):
    """Explicit map rather than chained replaces, so one model's substring can
    never corrupt another model's display name."""
    m = str(model)
    return _DISP_OVERRIDE.get(m, m.replace("-Instruct", "").replace("-it", ""))


def _algo_disp(a):
    return str(a).replace("Algorithm_", "Algorithm ")


def _preds(model, cfg, tier=None):
    name = cfg if tier is None else f"{cfg}_tier{tier}"
    p = C.RESULTS_ROOT / model / name / "patient_predictions.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d["empi"] = d["empi"].astype(str).str.strip()
    d["pos"] = d["predicted_label"].isin(C.POSITIVE_LABELS)
    d["gold_pos"] = d["gold_label"].isin(C.POSITIVE_LABELS)
    return d


def _flags():
    m = pd.read_excel(C.SHEN_XLSX)
    m["EMPI"] = m["EMPI"].astype(str).str.strip()
    return m.set_index("EMPI")


def _wilson(k, n):
    if n <= 0:
        return (float("nan"), float("nan"))
    z = C.Z_975
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _pct(x):
    return "--" if x != x else f"{100 * x:.1f}"


def _ci(lo, hi):
    return "--" if lo != lo else f"[{100 * lo:.1f}, {100 * hi:.1f}]"


def _write(name, text):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(text)
    print(f"  wrote {OUT / (name + '.tex')}")


# ============================================================ S3
def table_S3(model):
    rows = []
    for cfg in ("claims", "notes", "fused"):
        d = _preds(model, cfg)
        if d is None:
            print(f"  [skip] S3: {model}/{cfg} absent")
            return
        rows.append((CONFIG_LABEL[cfg], d))
    body = []
    for label, d in rows:
        body.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{label}}}}} \\\\")
        for gold in LABELS3:
            sub = d[d["gold_label"] == gold]
            cells = [int((sub["predicted_label"] == p).sum()) for p in LABELS3]
            body.append(f"{gold} ($n={len(sub)}$) & " +
                        " & ".join(f"{c}" for c in cells) + " \\\\")
        body.append("\\midrule")
    body = body[:-1]

    _write("S3_threeway", f"""\\begin{{table}}[H]
\\centering
\\caption{{\\textbf{{Three-way classification confusion matrices.}} Full $3\\times3$
(gold $\\times$ predicted) confusion matrices for the \\textit{{Claims only}}, \\textit{{Notes only}}, and \\textit{{Fused}} experiments ({_disp(model)}) on the 309-patient core cohort, prior to binary aggregation into the (Probable~$\\cup$~Possible) positive class used throughout the rest of the manuscript. Row counts sum to the gold-label totals.}}
\\label{{si:tab:threeway}}
\\begin{{tabular}}{{lccc}}
\\toprule
 & \\multicolumn{{3}}{{c}}{{Predicted}} \\\\
Gold & Probable ILD & Possible ILD  & Not ILD \\\\
\\midrule
{chr(10).join(body)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


# ============================================================ S4
def table_S4():
    shen = pd.read_csv(SHEN_METRICS)
    grid = pd.read_csv(GRID)
    grid = grid[grid["reported"].astype(str).str.lower().isin(("true", "1"))]

    lines = []
    for cfg in ("claims", "notes", "fused"):
        sub = grid[grid["config"] == cfg].sort_values("ppv", ascending=False)
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{CONFIG_LABEL[cfg]}}}}} \\\\")
        for _, r in sub.iterrows():
            lines.append(
                f"\\quad {_disp(r['model'])} & "
                f"{int(r['n_true_positive'])}/{int(r['n_false_positive'])}/"
                f"{int(r['n_true_negative'])}/{int(r['n_false_negative'])} & "
                f"{_pct(r['ppv'])} {_ci(r['ppv_wilson_lo'], r['ppv_wilson_hi'])} & "
                f"{_pct(r['yield'])} {_ci(r['yield_wilson_lo'], r['yield_wilson_hi'])} \\\\")
        lines.append("\\midrule")

    lines.append("\\multicolumn{4}{l}{\\emph{Published claims-based algorithms}} \\\\")
    for _, r in shen.iterrows():
        if r.get("tp") != r.get("tp"):
            continue
        nm = _algo_disp(r["algorithm"])
        if bool(r.get("is_synthesized", False)):
            nm += " (flag-all)"
        lines.append(
            f"\\quad {nm} & "
            f"{int(r['tp'])}/{int(r['fp'])}/{int(r['tn'])}/{int(r['fn'])} & "
            f"{_pct(r['ppv'])} {_ci(r['ppv_wilson_lo'], r['ppv_wilson_hi'])} & "
            f"{_pct(r['yield'])} {_ci(r['yield_wilson_lo'], r['yield_wilson_hi'])} \\\\")

    n_models = grid["model"].nunique()
    _write("S4_allalgos", f"""\\begin{{longtable}}{{l c c c}}
\\caption{{\\textbf{{Full performance table: all 20 published algorithms and every
model $\\times$ configuration cell on the 309-patient core cohort.}} For every Shen
algorithm variant and for each of the {n_models} models under Claims, Notes and Fused, the
table reports the confusion matrix (TP/FP/TN/FN), PPV with Wilson 95\\% confidence
interval, and yield (the fraction of the cohort flagged positive) on the identical
patient set. Sensitivity and specificity are not reported: on a code-screened
cohort they do not carry their usual population interpretation. Extended companion
to Fig.~\\ref{{fig:pareto}}.}}
\\label{{si:tab:allalgos}}\\\\
\\toprule
Method & TP/FP/TN/FN & PPV (95\\% CI), \\% & Yield (95\\% CI), \\% \\\\
\\midrule
\\endfirsthead
\\multicolumn{{4}}{{l}}{{\\small\\itshape Supplementary Table~\\ref{{si:tab:allalgos}} (continued)}}\\\\
\\toprule
Method & TP/FP/TN/FN & PPV (95\\% CI), \\% & Yield (95\\% CI), \\% \\\\
\\midrule
\\endhead
\\midrule
\\multicolumn{{4}}{{r}}{{\\small\\itshape continued on next page}}\\\\
\\endfoot
\\bottomrule
\\endlastfoot
{chr(10).join(lines)}
\\end{{longtable}}
""")


# ============================================================ S5
def table_S5(model):
    flags = _flags()
    algos = [C.SHEN_SYNTHESIZED_ALGO] + C.SHEN_MEASURED_ALGOS
    out = []
    for tier_id, tier_name in TIER_ROWS:
        notes = _preds(model, "notes", tier_id)
        fused = _preds(model, "fused", tier_id)
        if notes is None or fused is None:
            print(f"  [skip] S5: {model} tier {tier_id} absent")
            continue
        idx = notes["empi"]
        n = len(notes)
        npos = int(notes["gold_pos"].sum())
        out.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{tier_name}}} - subcohort "
                   f"$n={n}$ ({npos} ILD-positive, {n - npos} negative)}} \\\\")
        out.append("\\midrule")
        for lab, d in (("LLM - Notes", notes), ("LLM - Fused", fused)):
            k = int(d["pos"].sum()); tp = int((d["pos"] & d["gold_pos"]).sum())
            out.append(f"{lab} & {len(d)} & {_pct(tp / k) if k else '--'} "
                       f"& {_pct(k / len(d))} \\\\")
        sub_flags = flags.reindex(idx)
        gold = notes["gold_pos"].values
        for a in algos:
            if a == C.SHEN_SYNTHESIZED_ALGO:
                rule = np.ones(n, dtype=bool)
            elif a in sub_flags.columns:
                rule = (sub_flags[a].fillna(0).astype(int) == 1).values
            else:
                continue
            k = int(rule.sum())
            if k < MIN_FLAG_TIER:
                continue
            tp = int((rule & gold).sum())
            out.append(f"{_algo_disp(a)} & {n} & {_pct(tp / k)} & {_pct(k / n)} \\\\")
        out.append("\\addlinespace")
    if not out:
        return
    out = out[:-1]

    _write("S5_pertier", f"""\\begin{{longtable}}{{l r r r}}
\\caption{{\\textbf{{Matched-cohort head-to-head, grouped by note type.}} For each of the five evaluable note tiers, every method is evaluated on the subcohort of patients
possessing that tier ($n$, with its ILD-positive/negative split, given in each group header). Rows: the Notes and Fused LLM configurations ({_disp(model)}), followed by each published Shen algorithm variant\\cite{{shen2025}} that flagged $\\geq {MIN_FLAG_TIER}$ patients in
that subcohort (sparser variants masked, per Shen et al.). PPV and yield (the fraction of the subcohort flagged positive) are computed on the full
subcohort. PPV values underlie the ratio heatmaps in Fig.~\\ref{{fig:perAlgo}}c,d; Algorithm~1 flags every patient by construction (yield 100\\%).}}
\\label{{si:tab:pertier}}\\\\
\\toprule
Method & $n$ & PPV \\% & Yield \\% \\\\
\\midrule
\\endfirsthead
\\multicolumn{{4}}{{l}}{{\\small\\itshape Supplementary Table~\\ref{{si:tab:pertier}} (continued)}}\\\\
\\toprule
Method & $n$ & PPV \\% & Yield \\% \\\\
\\midrule
\\endhead
\\midrule
\\multicolumn{{4}}{{r}}{{\\small\\itshape continued on next page}}\\\\
\\endfoot
\\bottomrule
\\endlastfoot
{chr(10).join(out)}
\\end{{longtable}}
""")


# ============================================================ S6
def table_S6(model):
    agg = pd.read_csv(AGG)
    agg = agg[agg["model"] == model]
    if not len(agg):
        print(f"  [skip] S6: no aggregation rows for {model}")
        return
    RULE_LABEL = [("or_logic", "OR-logic\\,$^{*}$"), ("first_batch", "First batch"),
                  ("majority", "Majority vote"), ("max_confidence", "Max confidence")]
    lines = []
    for i, cfg in enumerate(("notes", "fused")):
        sub = agg[agg["config"] == cfg].set_index("rule")
        for j, (rule, label) in enumerate(RULE_LABEL):
            if rule not in sub.index:
                continue
            r = sub.loc[rule]
            head = CONFIG_LABEL[cfg] if j == 0 else ""
            lines.append(
                f"{head} & {label} & "
                f"{int(r['n_true_positive'])}/{int(r['n_false_positive'])}/"
                f"{int(r['n_true_negative'])}/{int(r['n_false_negative'])} & "
                f"{_pct(r['ppv'])} {_ci(r['ppv_wilson_lo'], r['ppv_wilson_hi'])} & "
                f"{_pct(r['yield'])} {_ci(r['yield_wilson_lo'], r['yield_wilson_hi'])} \\\\")
        if i == 0:
            lines.append("\\midrule")

    _write("S6_aggregation", f"""\\begin{{table}}[H]
\\centering
\\caption{{\\textbf{{Patient-level aggregation strategy comparison for the Notes and Fused configurations on the 309-patient core cohort ({_disp(model)}).}} Each row re-aggregates the same per-batch predictions under a different deterministic rule. OR-logic (any batch positive) is the deployed default and preserves the highest yield. \\textit{{First batch}} uses the predicted label of the first batch as the patient final label. \\textit{{Majority vote}} uses the most present label across a patient batch as the patient label. \\textit{{Max Confidence}} uses the batch label with the highest model-reported confidence as the patient label. More restrictive rules trade yield for PPV. All intervals are Wilson score intervals.}}
\\label{{si:tab:aggregation}}
\\setlength{{\\tabcolsep}}{{4pt}}
\\footnotesize
\\begin{{tabular}}{{@{{}}lllcc@{{}}}}
\\toprule
Configuration & Strategy & TP/FP/TN/FN & PPV [95\\% CI] (\\%) & Yield [95\\% CI] (\\%) \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\par\\vspace{{2pt}}
{{\\footnotesize $^{{*}}$Deployed configuration (plain OR aggregation, $\\tau = 0$).}}
\\end{{table}}
""")


# ============================================================ S7
def table_S7():
    import yaml
    blocks = []
    for cfg, title in (("claims", "Claims only"), ("notes", "Notes only"),
                       ("fused", "Fusion (claims + notes)")):
        p = C.PROMPTS_DIR / f"{cfg}.yaml"
        y = yaml.safe_load(p.read_text())
        body = y["SYSTEM_PROMPT"].strip()
        blocks.append(f"""\\subsection*{{{title}}}
\\begin{{Verbatim}}[fontsize=\\scriptsize, breaklines=true, breakanywhere=true]
[SYSTEM PROMPT]
{body}
\\end{{Verbatim}}""")

    _write("S7_prompts", f"""\\par\\vspace{{1em}}
\\captionof{{table}}{{\\textbf{{System prompts used for LLM inference, by configuration.}}
Reproduced verbatim from \\texttt{{shared/prompts/}}. The response contract is
additionally enforced at decoding time by a JSON schema (constrained decoding),
so the label field is a closed enumeration of the three classes and every
free-text field is length-bounded; malformed output is unreachable rather than
repaired after the fact.}}
\\label{{si:tab:sysprompt}}

{chr(10).join(blocks)}
""")


# ============================================================ S8
def table_S8():
    grid = pd.read_csv(GRID)
    grid = grid[grid["reported"].astype(str).str.lower().isin(("true", "1"))]
    order = list(dict.fromkeys(grid["model"]))
    lines, n_pos_ref = [], None
    for m in order:
        cl, no, fu = (_preds(m, "claims"), _preds(m, "notes"), _preds(m, "fused"))
        if cl is None or no is None or fu is None:
            print(f"  [skip] S8: {m} incomplete")
            continue
        cl = cl.set_index("empi"); no = no.set_index("empi"); fu = fu.set_index("empi")
        idx = cl.index.intersection(no.index).intersection(fu.index)
        gold = cl.loc[idx, "gold_pos"].values
        a = cl.loc[idx, "pos"].values
        b = no.loc[idx, "pos"].values
        f = fu.loc[idx, "pos"].values
        n_pos_ref = int(gold.sum())
        both = int((a & b & gold).sum())
        c_only = int((a & ~b & gold).sum())
        n_only = int((~a & b & gold).sum())
        neither = int((~a & ~b & gold).sum())
        # cases neither single modality caught, that fusion recovered
        rescues = int((~a & ~b & f & gold).sum())
        tag = " (illustrative)" if m == C.ILLUSTRATIVE_MODEL else ""
        lines.append(f"{_disp(m)}{tag} & {both} & {c_only} & {n_only} & "
                     f"{neither} & {rescues} \\\\")

    _write("S8_decomposition", f"""\\begin{{table}}[H]
\\centering
\\caption{{\\textbf{{Per-patient modality decomposition across models, on the {n_pos_ref} true ILD cases.}} For each model, every true-positive patient is classified by which single-modality configuration detected it: both Claims and Notes, Claims only, Notes only, or neither (the four counts sum to {n_pos_ref}). ``Fusion rescues'' is the number of the ``neither'' cases subsequently recovered by the Fused configuration. Companion to Fig.~\\ref{{fig:models}}d and Supplementary Fig.~\\ref{{si:fig:complementarity}}.}}
\\label{{si:tab:decomp}}
\\setlength{{\\tabcolsep}}{{4pt}}
\\small
\\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\\toprule
Model & Claims $\\cap$ Notes & Claims only & Notes only & Neither & Fusion rescues \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


# ============================================================ Table 1
def table_T1():
    d = pd.read_csv(C.COHORT_CSV)
    n = len(d)

    def row(label, k, indent=False):
        pre = "\\quad " if indent else ""
        return f"{pre}{label} & {k} & {100 * k / n:.1f} \\\\"

    lines = [row("Core cohort", n), "\\midrule",
             "\\multicolumn{3}{l}{\\emph{Adjudicated label}} \\\\"]
    for lab in LABELS3:
        lines.append(row(lab, int((d["gold_label"] == lab).sum()), True))
    lines.append(row("ILD-positive (Probable + Possible)",
                     int(d["gold_label"].isin(C.POSITIVE_LABELS).sum()), True))
    lines.append(row("ILD-negative (Not ILD)",
                     int((d["gold_label"] == "Not ILD").sum()), True))
    for col, title in (("sex", "Sex"), ("race", "Race"),
                       ("cancer_group", "Cancer type"), ("insurance", "Insurance")):
        if col not in d.columns:
            continue
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{3}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for v, k in d[col].value_counts().items():
            lines.append(row(str(v), int(k), True))
    if "age" in d.columns:
        a = pd.to_numeric(d["age"], errors="coerce").dropna()
        lines += ["\\midrule",
                  "\\multicolumn{3}{l}{\\emph{Age, years}} \\\\",
                  f"\\quad Mean (SD) & {a.mean():.1f} ({a.std():.1f}) & \\\\",
                  f"\\quad Median [IQR] & {a.median():.0f} "
                  f"[{a.quantile(.25):.0f}-{a.quantile(.75):.0f}] & \\\\",
                  f"\\quad Range & {a.min():.0f}-{a.max():.0f} & \\\\"]

    _write("T1_cohort", f"""\\begin{{table}}[H]
\\centering
\\caption{{\\textbf{{Cohort characteristics (core cohort, $n={n}$).}}}}
\\label{{tab:cohort}}
\\begin{{tabular}}{{lrr}}
\\toprule
 & Patients & \\% \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


TABLES = {"T1": lambda m: table_T1(), "S3": table_S3, "S4": lambda m: table_S4(),
          "S5": table_S5, "S6": table_S6, "S7": lambda m: table_S7(),
          "S8": lambda m: table_S8()}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", nargs="*", default=["S3", "S4", "S5", "S6", "S7", "S8"],
                    choices=list(TABLES))
    ap.add_argument("--model", default=C.ILLUSTRATIVE_MODEL,
                    help="model for the single-model tables (S3, S5, S6)")
    a = ap.parse_args()
    print(f"Tables -> {OUT}   (single-model tables use {a.model})")
    for t in a.table:
        print(f"\n{t}:")
        TABLES[t](a.model)
    print("\nPaste each fragment over its block in main.tex, or \\input{} it.")
