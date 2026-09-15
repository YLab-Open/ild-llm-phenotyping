"""
analysis/power_projection.py — CPU only. Copeland misclassification + power,
for every model x config and every published rule.

    python analysis/power_projection.py

The model lives in shared/evaluation/metrics.py, with constants from
config.py, and is evaluated over the whole model x config grid plus every
published rule.

WHICH QUANTITY DRIVES THE MODEL: sensitivity and specificity, NOT yield.
_logor_V(se, sp) reconstructs the misclassified 2x2 from se/sp alone, and
mc_power draws Beta(tp+.5, fn+.5) and Beta(tn+.5, fp+.5). Yield never enters the
arithmetic; it is carried in the output as reporting context because it governs
how many patients a real study could assemble, which is a separate matter from
how much signal each assembled patient retains.

NOTE for Figure 5: the manuscript's power panels keep MedGemma-27B as the
displayed classifier alongside Algorithm 1, Algorithm 4 and the Gemma-3-27B
control. This full table is a SUPPLEMENTARY output, not a substitute for them.

Outputs: analysis/results/power/power_projection.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import config as C
from shared.evaluation.metrics import mc_power, obs_or_pstar, attenuation_pct

OUT = C.CLEAN_ROOT / "analysis" / "results" / "power"


def row_for(name, model, config, tp, fn, tn, fp, kind):
    s = mc_power(tp, fn, tn, fp)
    oro = obs_or_pstar(s["se_pt"], s["sp_pt"])
    n = tp + fn + tn + fp
    return {
        "classifier": name, "model": model, "config": config, "kind": kind,
        "n_true_positive": tp, "n_false_negative": fn,
        "n_true_negative": tn, "n_false_positive": fp,
        "sensitivity": s["se_pt"], "specificity": s["sp_pt"],
        "ppv": tp / (tp + fp) if (tp + fp) else float("nan"),
        "yield": (tp + fp) / n if n else float("nan"),
        "observed_or": oro,
        "attenuation_pct": attenuation_pct(C.OR_TRUE, oro),
        "required_N_estimable": s["required_N_estimable"],
        "required_N_80pct": s["N_pt"],
        "required_N_lo": s["N_lo"], "required_N_hi": s["N_hi"],
        "power_at_N_2000": s["power2000"],
    }


def main(tier):
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "" if tier is None else f"_tier{tier}"
    rows = []

    shen = pd.read_csv(C.CLEAN_ROOT / "analysis" / "results" /
                       "shen_evaluation" / "shen_metrics.csv")
    for _, r in shen.iterrows():
        if pd.isna(r.get("tp")):
            continue
        kind = ("published rule" if not r.get("is_synthesized", False)
                else "SYNTHESIZED inclusion criterion (flag-all)")
        rows.append(row_for(f"Rule {r['algorithm']}", r["algorithm"], "-",
                            int(r["tp"]), int(r["fn"]), int(r["tn"]), int(r["fp"]),
                            kind))

    grid = pd.read_csv(C.RESULTS_ROOT / f"grid_metrics{suffix}.csv")
    for _, r in grid.iterrows():
        if not r["reported"]:
            continue
        rows.append(row_for(f"{r['model']} [{r['config']}]", r["model"], r["config"],
                            int(r["n_true_positive"]), int(r["n_false_negative"]),
                            int(r["n_true_negative"]), int(r["n_false_positive"]),
                            "LLM"))

    out = pd.DataFrame(rows).sort_values(
        ["required_N_estimable", "required_N_80pct"],
        ascending=[False, True]).reset_index(drop=True)
    out.to_csv(OUT / f"power_projection{suffix}.csv", index=False)

    print(f"{'classifier':<44}{'PPV':>7}{'Sens':>7}{'Spec':>7}{'OR':>7}"
          f"{'req N':>11}{'pw@2k':>7}")
    print("-" * 92)
    for _, r in out.iterrows():
        nn = f"{r['required_N_80pct']:>11,.0f}" if r["required_N_estimable"] else f"{'off scale':>11}"
        print(f"{r['classifier']:<44}{r['ppv']*100:>6.1f}%{r['sensitivity']*100:>6.1f}%"
              f"{r['specificity']*100:>6.1f}%{r['observed_or']:>7.2f}{nn}"
              f"{r['power_at_N_2000']*100:>6.0f}%")
    ill = out[(out["model"] == C.ILLUSTRATIVE_MODEL) & out["required_N_estimable"]]
    if len(ill):
        best = out[(out["kind"] == "LLM") & out["required_N_estimable"]].iloc[0]
        print(f"\nIllustrative model best cell: {ill['required_N_80pct'].min():,.0f}")
        print(f"Best cell overall: {best['classifier']} at {best['required_N_80pct']:,.0f} "
              f"({ill['required_N_80pct'].min()/best['required_N_80pct']:.1f}x fewer patients)")
    print(f"\nSaved -> {OUT/('power_projection'+suffix+'.csv')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=None)
    main(ap.parse_args().tier)
