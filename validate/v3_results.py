"""
validate/v3_results.py — CPU only. Gate after s5_metrics.

    python validate/v3_results.py

Checks (all gating):
  * confusion cells sum to 309 for every cell
  * gold labels agree with the Shen table on ALL rows
  * no NaN in any reported metric
  * Wilson intervals bracket every point estimate
  * no column named `tp` anywhere in grid_metrics.csv (reserved for the
    true-positive count downstream; the tensor-parallel size has its own column)
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import config as C
from validate._gate import Gate

POINT_INTERVAL = [("ppv", "ppv_wilson_lo", "ppv_wilson_hi"),
                  ("yield", "yield_wilson_lo", "yield_wilson_hi"),
                  ("sensitivity", "sensitivity_wilson_lo", "sensitivity_wilson_hi"),
                  ("specificity", "specificity_wilson_lo", "specificity_wilson_hi"),
                  ("npv", "npv_wilson_lo", "npv_wilson_hi")]
REPORTED = ["ppv", "yield", "sensitivity", "specificity", "npv", "f1", "accuracy",
            "auroc", "ece_discrete"]


def main(tier):
    suffix = "" if tier is None else f"_tier{tier}"
    gp = C.RESULTS_ROOT / f"grid_metrics{suffix}.csv"
    g = Gate(f"v3 RESULTS — {gp.name}")
    if not gp.exists():
        g.check(False, "grid_metrics.csv exists", str(gp))
        g.report()

    grid = pd.read_csv(gp)
    n_expected = len(pd.read_csv(C.COHORT_CSV))

    g.check("tp" not in grid.columns,
            "no column named 'tp' (ambiguous between true positives and tensor-parallel size)",
            f"columns: {list(grid.columns)[:12]}...")

    sums = (grid["n_true_positive"] + grid["n_false_positive"] +
            grid["n_true_negative"] + grid["n_false_negative"])
    bad = grid.loc[sums != n_expected, ["model", "config"]].values.tolist()
    g.check(bad == [], f"confusion cells sum to {n_expected} in every cell", f"{bad[:10]}")

    g.check(int(grid["n_sentinel_patients"].sum()) == 0,
            "zero sentinel patients across the grid",
            grid.loc[grid["n_sentinel_patients"] > 0,
                     ["model", "config", "n_sentinel_patients"]].to_string(index=False)
            if grid["n_sentinel_patients"].sum() else "")

    nan_cells = []
    for _, r in grid.iterrows():
        for m in REPORTED:
            if m not in grid.columns:
                continue
            v = r[m]
            is_nan = v is None or (isinstance(v, float) and math.isnan(v))
            if not is_nan:
                continue
            # A metric with an empty denominator is undefined, not missing.
            # gemma-3-27b-it flags every patient in claims, so TN+FN = 0 and NPV
            # is 0/0. Report it; do not gate on it.
            undefined = (
                (m == "npv" and r["n_true_negative"] + r["n_false_negative"] == 0) or
                (m == "ppv" and r["n_flagged"] == 0) or
                (m == "specificity" and r["n_true_negative"] + r["n_false_positive"] == 0) or
                (m == "sensitivity" and r["n_true_positive"] + r["n_false_negative"] == 0)
            )
            (g.note if undefined else nan_cells.append)(
                f"{r['model']}/{r['config']}: {m} undefined (empty denominator)"
                if undefined else (r["model"], r["config"], m))
    g.check(not nan_cells, "no NaN in any reported metric (empty denominators "
            "reported separately)", f"{nan_cells[:10]}")

    viol = []
    for _, r in grid.iterrows():
        for p, lo, hi in POINT_INTERVAL:
            if lo not in grid.columns:
                continue
            if any(isinstance(x, float) and math.isnan(x) for x in (r[p], r[lo], r[hi])):
                continue
            if lo in grid.columns and not (r[lo] - 1e-9 <= r[p] <= r[hi] + 1e-9):
                viol.append((r["model"], r["config"], p, r[lo], r[p], r[hi]))
    g.check(not viol, "Wilson intervals bracket every point estimate", f"{viol[:10]}")

    # gold-label agreement with the Shen table — must be exact
    meta = pd.read_excel(C.SHEN_XLSX)
    meta["EMPI"] = meta["EMPI"].astype(str).str.strip()
    gold_x = meta.set_index("EMPI")["Final_diagnosis"]
    disagree, unmatched = [], []
    for _, r in grid.iterrows():
        name = r["config"] if tier is None else f"{r['config']}_tier{tier}"
        pp = C.RESULTS_ROOT / r["model"] / name / "patient_predictions.csv"
        if not pp.exists():
            continue
        d = pd.read_csv(pp)
        d["empi"] = d["empi"].astype(str).str.strip()
        j = d.join(gold_x, on="empi", how="left")
        nm = int(j["Final_diagnosis"].isna().sum())
        if nm:
            unmatched.append((r["model"], r["config"], nm))
        jj = j[j["Final_diagnosis"].notna()]
        n_bad = int((jj["final_diagnosis"].astype(int)
                     != jj["Final_diagnosis"].astype(int)).sum())
        if n_bad:
            disagree.append((r["model"], r["config"], n_bad))
    g.check(not unmatched, "every EMPI joins to the Shen table", f"{unmatched[:10]}")
    g.check(not disagree,
            "gold labels agree with the Shen table on all rows",
            f"{disagree[:10]}")

    g.note(f"cells: {len(grid)}  reported models: "
           f"{grid[grid['reported']]['model'].nunique()}")
    if "n_label_probability_inconsistent" in grid.columns:
        tot = int(grid["n_label_probability_inconsistent"].sum())
        g.note(f"label/probability disagreement (model's own, reported not repaired): "
               f"{tot} patient-rows across the grid")
    if "n_distinct_probabilities" in grid.columns:
        g.note(f"distinct emitted probabilities per cell: "
               f"min {int(grid['n_distinct_probabilities'].min())}, "
               f"max {int(grid['n_distinct_probabilities'].max())} "
               f"(no calibration claim is readable without this)")
    g.report()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=None)
    main(ap.parse_args().tier)
