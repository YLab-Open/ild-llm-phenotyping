"""
analysis/frontier.py — CPU only. PPV-yield Pareto frontier of the published
rules, and where each model x config sits relative to it.

    python analysis/frontier.py

Three criteria are reported because they disagree, and the difference matters
for what the manuscript can claim:
  strictly_above_frontier  no rule has both PPV >= and yield >= this point
  coincident_with_rule     the point matches a rule on BOTH coordinates (1e-9):
                           it lies ON the frontier, not above it
  above_interpolated       PPV exceeds the frontier linearly interpolated at
                           this point's yield -- the stricter reading, and the
                           one a "lies above the traced curve" claim needs

Frontier computed three ways: all 20 rules; excluding the synthesized
Algorithm_1; and additionally excluding rules flagging < 10 patients.

Outputs: analysis/results/frontier/frontier_dominance.csv, frontier_summary.json
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

OUT = C.CLEAN_ROOT / "analysis" / "results" / "frontier"
TOL = 1e-9
MIN_FLAG = 10


def pareto_front(x, y):
    """Indices of the non-dominated (max-x, max-y) set, sorted by x.
    Generic 2-D non-domination; the x axis here is yield."""
    x = np.asarray(x); y = np.asarray(y)
    keep = []
    for i in range(len(x)):
        dominated = np.any((x >= x[i]) & (y >= y[i]) & ((x > x[i]) | (y > y[i])))
        if not dominated:
            keep.append(i)
    return sorted(keep, key=lambda i: x[i])


def load_rules():
    p = C.CLEAN_ROOT / "analysis" / "results" / "shen_evaluation" / "shen_metrics.csv"
    df = pd.read_csv(p)
    df = df[df["algorithm"].astype(str).str.startswith("Algorithm_")].copy()
    df = df[df["tp"].notna()].copy()
    if "yield" not in df.columns:
        tot = df[["tp", "fp", "tn", "fn"]].sum(axis=1)
        df["yield"] = (df["tp"] + df["fp"]) / tot
    df["n_flagged_calc"] = (df["tp"] + df["fp"]).astype(int)
    return df.reset_index(drop=True)


def frontier_points(rules):
    idx = pareto_front(rules["yield"].values, rules["ppv"].values)
    return rules.iloc[idx].sort_values("yield").reset_index(drop=True)


def interp_ppv(front, y):
    ys, ps = front["yield"].values, front["ppv"].values
    if y < ys.min() or y > ys.max():
        return float("nan")          # no extrapolation
    return float(np.interp(y, ys, ps))


def main(tier):
    OUT.mkdir(parents=True, exist_ok=True)
    rules = load_rules()
    suffix = "" if tier is None else f"_tier{tier}"
    grid = pd.read_csv(C.RESULTS_ROOT / f"grid_metrics{suffix}.csv")
    grid = grid[grid["reported"]].copy()

    variants = {
        "all20": rules,
        "no_algo1": rules[rules["algorithm"] != C.SHEN_SYNTHESIZED_ALGO],
        "ge10flag": rules[(rules["algorithm"] != C.SHEN_SYNTHESIZED_ALGO) &
                          (rules["n_flagged_calc"] >= MIN_FLAG)],
    }
    fronts = {k: frontier_points(v.reset_index(drop=True)) for k, v in variants.items()}

    rows = []
    for vname, rset in variants.items():
        F = fronts[vname]
        for _, m in grid.iterrows():
            y, p = float(m["yield"]), float(m["ppv"])
            dom = rset[(rset["ppv"] >= p - TOL) & (rset["yield"] >= y - TOL)]
            tie = dom[(np.abs(dom["ppv"] - p) <= TOL) & (np.abs(dom["yield"] - y) <= TOL)]
            coincident = len(tie) > 0
            dom = dom[~((np.abs(dom["ppv"] - p) <= TOL) &
                        (np.abs(dom["yield"] - y) <= TOL))]
            fi = interp_ppv(F, y)
            rows.append({
                "frontier_variant": vname, "model": m["model"], "config": m["config"],
                "ppv": p, "yield": y,
                "n_dominating_rules": int(len(dom)),
                "dominated_by": "|".join(sorted(dom["algorithm"])),
                "coincident_with_rule": coincident,
                "coincident_rules": "|".join(sorted(tie["algorithm"])) if coincident else "",
                "strictly_above_frontier": bool(len(dom) == 0 and not coincident),
                "ppv_at_matched_yield": fi,
                "ppv_margin": (p - fi) if fi == fi else float("nan"),
                "above_interpolated": bool(fi == fi and (p - fi) > TOL),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"frontier_dominance{suffix}.csv", index=False)

    summary = {}
    for vname in variants:
        sub = out[out["frontier_variant"] == vname]
        v = {"n_rules_in_pool": int(len(variants[vname])),
             "frontier": fronts[vname][["algorithm", "yield", "ppv"]].round(4)
                         .to_dict("records"),
             "per_config": {}}
        for cfg in C.CONFIGS:
            s = sub[sub["config"] == cfg]
            v["per_config"][cfg] = {
                "n_models": int(len(s)),
                "n_strictly_non_dominated": int(s["strictly_above_frontier"].sum()),
                "n_above_interpolated": int(s["above_interpolated"].sum()),
                "n_above_by_gt5pp": int((s["ppv_margin"] > 0.05).sum()),
                "n_above_by_gt10pp": int((s["ppv_margin"] > 0.10).sum()),
                "n_coincident_with_a_rule": int(s["coincident_with_rule"].sum()),
                "coincident_models": sorted(s[s["coincident_with_rule"]]["model"]),
            }
        summary[vname] = v
    summary["_near_degenerate_rules"] = rules[rules["n_flagged_calc"] < MIN_FLAG][
        ["algorithm", "n_flagged_calc", "ppv", "yield"]].round(4).to_dict("records")
    (OUT / f"frontier_summary{suffix}.json").write_text(json.dumps(summary, indent=2, default=float))

    for vname in variants:
        v = summary[vname]
        print(f"\n{'='*80}\nFRONTIER: {vname}  ({v['n_rules_in_pool']} rules, "
              f"{len(v['frontier'])} on the frontier)")
        for cfg in C.CONFIGS:
            c = v["per_config"][cfg]
            print(f"  {cfg:<8} non-dominated {c['n_strictly_non_dominated']:>2}/{c['n_models']} | "
                  f"above interp {c['n_above_interpolated']:>2} | >5pp {c['n_above_by_gt5pp']:>2} | "
                  f">10pp {c['n_above_by_gt10pp']:>2} | coincident {c['n_coincident_with_a_rule']}")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=None)
    main(ap.parse_args().tier)
