"""
stages/s5_metrics.py — CPU only. Per-cell metrics + the unified grid table.

    python stages/s5_metrics.py
    python stages/s5_metrics.py --tier 1

Conventions:
  * Wilson intervals are canonical and are what the figures read. Bootstrap is
    still emitted (ppv_bootstrap_lo/hi) for comparison, but is never the
    reported interval: the percentile bootstrap degenerates to [1.000, 1.000]
    at PPV = 1.
  * No column is named `tp`. Counts are n_true_positive / n_false_positive /
    n_true_negative / n_false_negative; the tensor-parallel size is
    tensor_parallel_size.
  * `yield` is a first-class column, not derived downstream.
  * sensitivity and specificity are carried explicitly — the Copeland power
    model consumes those, not yield.
  * AUROC and ECE per cell from the winning-batch probability, with the count of
    DISTINCT emitted probability values, without which no calibration claim is
    readable.

Outputs:
  results/<model>/<config>/metrics.json, metrics_report.txt
  results/grid_metrics.csv      one row per model x config
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
from shared import runtime as RT
from shared.evaluation.metrics import (binary_metrics, confusion_from_frame,
                                       bootstrap_ci, auroc, ece_discrete,
                                       print_report, json_safe)


def cell_metrics(model, config, tier, d):
    pred = pd.read_csv(d / "patient_predictions.csv")
    conf = confusion_from_frame(pred)
    m = binary_metrics(conf["tp"], conf["fp"], conf["tn"], conf["fn"])

    # bootstrap, for comparability only
    y_true = pred["gold_label"].isin(C.POSITIVE_LABELS).astype(int).values
    y_pred = pred["predicted_label"].isin(C.POSITIVE_LABELS).astype(int).values
    b = bootstrap_ci(y_true, y_pred)

    # discrimination + calibration on the winning-batch probability
    prob = pd.to_numeric(pred["ild_probability"], errors="coerce").values
    valid = ~np.isnan(prob)
    au = auroc(prob, y_true.astype(bool))
    ece, n_ece, table = ece_discrete(prob, y_true.astype(float), valid_mask=valid)
    distinct = int(len(np.unique(prob[valid]))) if valid.any() else 0

    # label/probability disagreement: reported, never repaired. It can only
    # come from the model itself, since label and probability always originate
    # in the same batch and the same JSON object.
    lab_pos = pred["predicted_label"].isin(C.POSITIVE_LABELS).values
    inconsistent = int(np.sum(valid & (lab_pos != (prob > 0.5))))

    n_sent = int(pred["is_sentinel"].sum()) if "is_sentinel" in pred else 0
    n_sent_batches = (int(pred["n_sentinel_batches"].sum())
                      if "n_sentinel_batches" in pred else 0)
    manifest = {}
    mp = d / "run_manifest.json"
    if mp.exists():
        manifest = json.loads(mp.read_text())

    out = {
        "model": model, "config": config, "tier": tier,
        "params_b": C.MODELS[model]["params_b"],
        "family": C.MODELS[model]["family"],
        "reported": C.MODELS[model]["reported"],
        **m,
        "ppv_bootstrap_lo": b["ppv"]["lo"], "ppv_bootstrap_hi": b["ppv"]["hi"],
        "ci_method_canonical": C.CI_METHOD_CANONICAL,
        "auroc": au, "ece_discrete": ece, "n_ece": n_ece,
        "n_distinct_probabilities": distinct,
        "n_label_probability_inconsistent": inconsistent,
        "n_sentinel_patients": n_sent,
        "n_sentinel_batches": n_sent_batches,
        "tensor_parallel_size": manifest.get("tensor_parallel_size", C.TENSOR_PARALLEL_SIZE),
        "prompt_sha256": manifest.get("prompt_sha256", ""),
        "vllm_version": manifest.get("vllm_version", ""),
        "guided_decoding": manifest.get("decoding", {}).get("guided_decoding", ""),
        "inference_min": manifest.get("run_minutes"),
        "parse_path_census": manifest.get("parse_path_census_this_run", {}),
        "n_generations_hitting_token_cap": manifest.get("n_generations_hitting_token_cap"),
        "calibration_table": table,
    }
    (d / "metrics.json").write_text(json.dumps(json_safe(out), indent=2))
    rep = print_report(m, title=f"{model} — {C.CONFIG_TITLE[config]}"
                                + (f" (tier {tier})" if tier is not None else ""))
    extra = (f"  AUROC {au:.3f} | ECE {ece:.3f} over {distinct} distinct probabilities"
             f" | label/prob inconsistent {inconsistent}"
             f" | sentinels {n_sent} | sentinel batches {n_sent_batches}")
    (d / "metrics_report.txt").write_text(rep + "\n" + extra + "\n")
    print(extra)
    return out


def main(tier):
    log = RT.setup_logging("S5_METRICS", C.LOGS_DIR / "s5_metrics.log")
    rows = []
    for model in C.ALL_MODELS:
        for config in C.CONFIGS:
            name = config if tier is None else f"{config}_tier{tier}"
            d = C.RESULTS_ROOT / model / name
            if not (d / "patient_predictions.csv").exists():
                continue
            rows.append(cell_metrics(model, config, tier, d))
    if not rows:
        log.error("No aggregated cells found. Run s4_aggregate first.")
        sys.exit(2)

    grid = pd.DataFrame(rows)
    grid = grid.drop(columns=["calibration_table", "parse_path_census"])
    order = ["model", "config", "tier", "params_b", "family", "reported",
             "n_patients", "n_flagged",
             "n_true_positive", "n_false_positive", "n_true_negative", "n_false_negative",
             "ppv", "ppv_wilson_lo", "ppv_wilson_hi", "ppv_bootstrap_lo", "ppv_bootstrap_hi",
             "yield", "yield_wilson_lo", "yield_wilson_hi",
             "sensitivity", "sensitivity_wilson_lo", "sensitivity_wilson_hi",
             "specificity", "specificity_wilson_lo", "specificity_wilson_hi",
             "npv", "npv_wilson_lo", "npv_wilson_hi", "f1", "accuracy",
             "auroc", "ece_discrete", "n_ece", "n_distinct_probabilities",
             "n_label_probability_inconsistent", "n_sentinel_patients", "n_sentinel_batches",
             "tensor_parallel_size", "guided_decoding", "prompt_sha256",
             "vllm_version", "inference_min", "n_generations_hitting_token_cap",
             "ci_method_canonical"]
    grid = grid[[c for c in order if c in grid.columns]]
    suffix = "" if tier is None else f"_tier{tier}"
    out = C.RESULTS_ROOT / f"grid_metrics{suffix}.csv"
    grid.to_csv(out, index=False)

    assert "tp" not in grid.columns, "no column may be named 'tp'"
    log.info(f"\nGrid -> {out}  ({len(grid)} cells)")
    print()
    print(f"{'model':<24}{'config':<9}{'PPV':>7}{'  Wilson 95%':>18}{'yield':>8}"
          f"{'sens':>7}{'spec':>7}{'AUROC':>7}{'sent':>6}")
    print("-" * 96)
    for _, r in grid.sort_values(["config", "ppv"], ascending=[True, False]).iterrows():
        flag = "" if r["reported"] else "  (not reported)"
        print(f"{r['model']:<24}{r['config']:<9}{r['ppv']*100:6.1f}%"
              f"  [{r['ppv_wilson_lo']*100:5.1f},{r['ppv_wilson_hi']*100:5.1f}]"
              f"{r['yield']*100:7.1f}%{r['sensitivity']*100:6.1f}%"
              f"{r['specificity']*100:6.1f}%{r['auroc']:7.3f}"
              f"{int(r['n_sentinel_patients']):6d}{flag}")
    log.info("\nNext: python validate/v3_results.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=None)
    main(ap.parse_args().tier)
