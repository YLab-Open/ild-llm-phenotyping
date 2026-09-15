"""
analysis/shen_evaluation.py — CPU only. The 20 published Shen rules on the cohort.

    python analysis/shen_evaluation.py

Notes:

  * Algorithm_1 is labelled as SYNTHESIZED in the output. It is not a measured
    rule: it is the inclusion criterion (>=2 ILD codes 7-180 days apart), met
    by every cohort patient by construction, so shen_evaluation sets
    y_pred = ones and its "PPV" is the cohort base rate (59.2%) and its yield
    is 1.000 by definition. The xlsx carries only 19 Algorithm_* columns.
    Columns `is_synthesized` and `rule_kind` say so in the file itself.
  * Wilson intervals are emitted alongside the seeded bootstrap ones, which
    exactly reproduce the originally published values.
  * `yield` is a first-class column.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import config as C
from shared.evaluation.metrics import bootstrap_ci, binary_metrics

OUT_DIR = C.CLEAN_ROOT / "analysis" / "results" / "shen_evaluation"


def _eval_one(name: str, y_true: np.ndarray, y_pred: np.ndarray,
              synthesized: bool) -> dict:
    n_flag = int(y_pred.sum())
    row = {"algorithm": name,
           "is_synthesized": synthesized,
           "rule_kind": ("inclusion criterion (flag-all, synthesized in code)"
                         if synthesized else "published rule (column in xlsx)"),
           "n_flagged": n_flag}
    if n_flag == 0:
        row["note"] = "no flagged patients"
        return row
    b = bootstrap_ci(y_true, y_pred, n_iter=C.BOOTSTRAP_N, seed=C.BOOTSTRAP_SEED)
    w = binary_metrics(b["tp"], b["fp"], b["tn"], b["fn"])
    row.update({
        "tp": b["tp"], "fp": b["fp"], "tn": b["tn"], "fn": b["fn"],
        "ppv": b["ppv"]["point"], "ppv_lo": b["ppv"]["lo"], "ppv_hi": b["ppv"]["hi"],
        "sensitivity": b["sensitivity"]["point"], "sens_lo": b["sensitivity"]["lo"],
        "sens_hi": b["sensitivity"]["hi"],
        "specificity": b["specificity"]["point"], "spec_lo": b["specificity"]["lo"],
        "spec_hi": b["specificity"]["hi"],
        "f1": b["f1"]["point"], "f1_lo": b["f1"]["lo"], "f1_hi": b["f1"]["hi"],
        "accuracy": b["accuracy"]["point"], "acc_lo": b["accuracy"]["lo"],
        "acc_hi": b["accuracy"]["hi"],
        # additions
        "yield": w["yield"],
        "yield_wilson_lo": w["yield_wilson_lo"], "yield_wilson_hi": w["yield_wilson_hi"],
        "ppv_wilson_lo": w["ppv_wilson_lo"], "ppv_wilson_hi": w["ppv_wilson_hi"],
        "sensitivity_wilson_lo": w["sensitivity_wilson_lo"],
        "sensitivity_wilson_hi": w["sensitivity_wilson_hi"],
        "specificity_wilson_lo": w["specificity_wilson_lo"],
        "specificity_wilson_hi": w["specificity_wilson_hi"],
    })
    return row


def compute_shen_metrics(cohort_csv: Path | None = None,
                         write: bool = True) -> pd.DataFrame:
    shen = pd.read_excel(C.SHEN_XLSX)
    cohort = pd.read_csv(cohort_csv or C.COHORT_CSV)
    shen["EMPI"] = shen["EMPI"].astype(str).str.strip()
    cohort["empi"] = cohort["empi"].astype(str).str.strip()

    merged = cohort.merge(shen, left_on="empi", right_on="EMPI", how="inner")
    if len(merged) != len(cohort):
        raise RuntimeError(f"EMPI join produced {len(merged)}, expected {len(cohort)}")

    y_true = merged["final_diagnosis"].isin(C.GOLD_POSITIVE_CODES).astype(int).values

    rows = [_eval_one(C.SHEN_SYNTHESIZED_ALGO, y_true,
                      np.ones(len(merged), dtype=int), synthesized=True)]
    for algo in C.SHEN_MEASURED_ALGOS:
        if algo not in merged.columns:
            print(f"  WARN: {algo} not in xlsx — skipping")
            continue
        rows.append(_eval_one(algo, y_true,
                              merged[algo].fillna(0).astype(int).values,
                              synthesized=False))
    df = pd.DataFrame(rows)

    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_DIR / "shen_metrics.csv", index=False)
        lines = [
            "=" * 112,
            f"Published claims-based rules on the core cohort (n = {len(merged)})",
            "  Binary positive = gold in {Probable ILD, Possible ILD}",
            f"  {C.SHEN_SYNTHESIZED_ALGO} is SYNTHESIZED: the inclusion criterion, "
            f"flagging all {len(merged)} patients by construction.",
            "    Its PPV is therefore the cohort base rate and its yield is 1.000 "
            "by definition — it is not a measured comparator.",
            "=" * 112,
            f"{'Algorithm':<16}{'kind':<10}{'n_flag':>7}{'yield':>8}"
            f"{'PPV':>24}{'Sensitivity':>24}",
            "-" * 112,
        ]
        for r in rows:
            kind = "SYNTH" if r["is_synthesized"] else "published"
            if "ppv" not in r:
                lines.append(f"{r['algorithm']:<16}{kind:<10}{r['n_flagged']:>7}"
                             f"   (no flagged patients)")
                continue
            lines.append(
                f"{r['algorithm']:<16}{kind:<10}{r['n_flagged']:>7}"
                f"{r['yield']*100:7.1f}%"
                f"{r['ppv']*100:12.1f}% [{r['ppv_wilson_lo']*100:4.1f},{r['ppv_wilson_hi']*100:5.1f}]"
                f"{r['sensitivity']*100:12.1f}% [{r['sensitivity_wilson_lo']*100:4.1f},"
                f"{r['sensitivity_wilson_hi']*100:5.1f}]")
        lines.append("=" * 112)
        lines.append("  Intervals shown are Wilson (canonical); the file also carries "
                     "the seeded bootstrap intervals for comparison.")
        report = "\n".join(lines)
        (OUT_DIR / "shen_metrics_report.txt").write_text(report + "\n")
        print(report)
        print(f"\nSaved -> {OUT_DIR/'shen_metrics.csv'}")
    return df


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(
        description="Evaluate the 20 published claims-based rules on the core "
                    "cohort (CPU only). Algorithm_1 is synthesized and labelled "
                    "as such in the output."
    ).parse_args()
    compute_shen_metrics()
