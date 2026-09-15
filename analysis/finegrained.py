"""
analysis/finegrained.py — the note-tier ablation.

  python analysis/finegrained.py --plan                      # CPU: print the matrix
  python analysis/finegrained.py --build-subcohorts          # CPU
  python analysis/finegrained.py --summarise                 # CPU: after inference

Serialization and inference are NOT re-implemented here. A tier variant is just
the ordinary pipeline with a --tier flag, so the ablation runs through the same
s2/s3/s4/s5 as everything else:

    python stages/s2_serialize.py --config notes --tier 1
    python validate/v1_serialization.py --config notes --tier 1
    python stages/s3_infer.py --model <M> --config notes --tier 1
    ...

Because the ablation shares the serializer and the inference driver with the
main pipeline, it cannot drift from the main path.

Tier 7 (Pulmonology Dept) has zero notes in the cohort and is skipped. The
`priority` in a variant slug is the PACKING rank from config.TIER_PRIORITY,
not the classifier tier — e.g. notes_p2_path is packing rank 2, classifier
tier 4.

The subcohort for a variant is the patients with >=1 note of that tier; s2
produces exactly those patients because a patient with no tier-N note yields no
batches. build_subcohorts records the expected membership so v1 can check it.

Outputs: analysis/results/finegrained/{subcohort_summary,finegrained_summary}.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import config as C

OUT = C.CLEAN_ROOT / "analysis" / "results" / "finegrained"


def variants():
    for t in C.TIER_VARIANTS:
        for cfg in C.TIER_ABLATION_CONFIGS:
            yield {"name": f"{cfg}_p{t['priority']}_{t['slug']}", "config": cfg,
                   **t}


def plan():
    print(f"{'variant':<28}{'config':<8}{'tier_id':>8}{'packing rank':>14}  desc")
    print("-" * 78)
    for v in variants():
        print(f"{v['name']:<28}{v['config']:<8}{v['tier_id']:>8}{v['priority']:>14}  {v['desc']}")
    n = len(list(variants()))
    print(f"\n{n} variants. Commands per variant (illustrative model only):")
    print("  python stages/s2_serialize.py      --config <cfg> --tier <tier_id>")
    print("  python validate/v1_serialization.py --config <cfg> --tier <tier_id>")
    print(f"  python stages/s3_infer.py          --model {C.ILLUSTRATIVE_MODEL} "
          f"--config <cfg> --tier <tier_id>")


def build_subcohorts():
    """Expected membership per variant, from the typed EHR parquet."""
    OUT.mkdir(parents=True, exist_ok=True)
    core = pd.read_csv(C.COHORT_CSV)
    core["patient_id"] = core["patient_id"].astype(str)
    ehr = pd.read_parquet(C.EHR_TYPED)
    ehr["Patient_Num_New"] = ehr["Patient_Num_New"].astype(str)
    ehr = ehr[ehr["Patient_Num_New"].isin(set(core["patient_id"]))]

    rows = []
    for v in variants():
        pts = set(ehr[ehr["ild_priority"] == v["tier_id"]]["Patient_Num_New"].unique())
        sub = core[core["patient_id"].isin(pts)]
        d = OUT / v["name"]
        d.mkdir(parents=True, exist_ok=True)
        sub.to_csv(d / "expected_cohort.csv", index=False)
        rows.append({"variant": v["name"], "config": v["config"],
                     "tier_id": v["tier_id"], "packing_rank": v["priority"],
                     "desc": v["desc"], "n": len(sub),
                     "n_probable": int((sub["final_diagnosis"] == 1).sum()),
                     "n_possible": int((sub["final_diagnosis"] == 2).sum()),
                     "n_not_ild": int((sub["final_diagnosis"] == 0).sum())})
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "subcohort_summary.csv", index=False)
    print(s.to_string(index=False))
    small = s[s["n"] < 10]
    if len(small):
        print(f"\n  NOTE: {len(small)} variant(s) have <10 patients and will not "
              f"support a stable estimate:\n{small[['variant','n']].to_string(index=False)}")
    return s


def summarise(model):
    """Tier results beside the full-notes baseline for the SAME model."""
    rows = []
    for v in variants():
        d = C.RESULTS_ROOT / model / f"{v['config']}_tier{v['tier_id']}"
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        import json
        m = json.loads(mp.read_text())
        base_p = C.RESULTS_ROOT / model / v["config"] / "metrics.json"
        base = json.loads(base_p.read_text()) if base_p.exists() else {}
        rows.append({
            "variant": v["name"], "model": model, "config": v["config"],
            "tier_id": v["tier_id"], "desc": v["desc"],
            "n": m["n_patients"], "ppv": m["ppv"],
            "ppv_wilson_lo": m["ppv_wilson_lo"], "ppv_wilson_hi": m["ppv_wilson_hi"],
            "yield": m["yield"], "sensitivity": m["sensitivity"],
            "specificity": m["specificity"], "f1": m["f1"],
            "baseline_all_notes_ppv": base.get("ppv"),
            "baseline_all_notes_n": base.get("n_patients"),
            "ppv_delta_vs_baseline": (m["ppv"] - base["ppv"]) if base else None,
        })
    if not rows:
        print("No tier results found. Run the variants through s2/s3/s4/s5 first.")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["config", "tier_id"])
    df.to_csv(OUT / "finegrained_summary.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nSaved -> {OUT/'finegrained_summary.csv'}")
    print("\nNOTE: baseline_all_notes_* comes from the SAME pipeline as the "
          "tier cells, so Figure 3 compares like with like.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--build-subcohorts", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--model", default=C.ILLUSTRATIVE_MODEL, choices=C.ALL_MODELS)
    a = ap.parse_args()
    if a.plan:              plan()
    elif a.build_subcohorts: build_subcohorts()
    elif a.summarise:       summarise(a.model)
    else:                   ap.error("choose --plan, --build-subcohorts or --summarise")
