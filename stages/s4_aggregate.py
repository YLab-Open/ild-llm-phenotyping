"""
stages/s4_aggregate.py — CPU only. Per-batch records -> per-patient predictions.

    python stages/s4_aggregate.py                      # every complete cell
    python stages/s4_aggregate.py --model Qwen3-8B --config notes

The patient label is the highest-positivity batch label, and the reported
probability is THAT batch's own, not the maximum across batches.
ild_probability_max carries the across-batch maximum separately.
per_batch_labels / per_batch_probabilities / per_batch_parse_paths are written
for EVERY model.

Also emits the alternative aggregation rules (or_logic / first_batch /
majority / max_confidence) per cell, for every model.

Outputs per cell: patient_predictions.csv, aggregation_rules.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import config as C
from shared import runtime as RT
from shared.evaluation.metrics import (aggregate_batches, AGGREGATION_RULES,
                                       binary_metrics)


def iter_cells(models, configs, tier):
    for m in models:
        for cfg in configs:
            name = cfg if tier is None else f"{cfg}_tier{tier}"
            d = C.RESULTS_ROOT / m / name
            if (d / "batch_predictions.jsonl").exists():
                yield m, cfg, tier, d


def aggregate_cell(model, config, tier, d, log):
    recs = [json.loads(l) for l in open(d / "batch_predictions.jsonl") if l.strip()]
    by_pid = defaultdict(list)
    for r in recs:
        by_pid[r["patient_id"]].append(r)

    rows = []
    for pid, batches in by_pid.items():
        agg = aggregate_batches(batches)
        meta = sorted(batches, key=lambda b: b["batch_idx"])[0]
        rows.append({
            "patient_id": pid, "empi": str(meta.get("empi", "")),
            "model": model, "config": config, "tier": tier,
            "cancer_group": meta["cancer_group"],
            "final_diagnosis": meta["final_diagnosis"],
            "gold_label": meta["gold_label"],
            "predicted_label": agg["predicted_label"],
            "ild_probability": agg["ild_probability"],           # winning batch's own
            "ild_probability_max": agg["ild_probability_max"],   # across-batch maximum
            "confidence_score": agg["confidence_score"],
            "modality_concordance": agg.get("modality_concordance", ""),
            "n_batches": agg["n_batches"],
            "winning_batch_idx": agg.get("winning_batch_idx"),
            "n_sentinel_batches": agg["n_sentinel_batches"],
            "is_sentinel": agg["predicted_label"] == C.SENTINEL_LABEL,
            "per_batch_labels": json.dumps(agg["per_batch_labels"]),
            "per_batch_probabilities": json.dumps(agg["per_batch_probabilities"]),
            "per_batch_parse_paths": json.dumps(agg["per_batch_parse_paths"]),
            "reasoning": agg.get("reasoning", "")[:2000],
            "key_evidence": json.dumps(agg.get("key_evidence", [])),
        })
    pred = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    pred.to_csv(d / "patient_predictions.csv", index=False)

    # alternative aggregation rules, all models
    alt = []
    for rule, rule_fn in AGGREGATION_RULES.items():
        n_tp = n_fp = n_tn = n_fn = 0
        for pid, batches in by_pid.items():
            gold = batches[0]["final_diagnosis"] in C.GOLD_POSITIVE_CODES
            pos = rule_fn(batches) in C.POSITIVE_LABELS
            n_tp += gold and pos
            n_fp += (not gold) and pos
            n_tn += (not gold) and (not pos)
            n_fn += gold and (not pos)
        m = binary_metrics(int(n_tp), int(n_fp), int(n_tn), int(n_fn))
        alt.append({"model": model, "config": config, "rule": rule,
                    "n_multi_batch": sum(1 for b in by_pid.values() if len(b) > 1),
                    **m})
    pd.DataFrame(alt).to_csv(d / "aggregation_rules.csv", index=False)

    n_sent = int(pred["is_sentinel"].sum())
    log.info(f"  {model:<24}{(config if tier is None else config+'/t'+str(tier)):<14}"
             f"n={len(pred):<5} sentinels={n_sent:<4} "
             f"batches={len(recs)}"
             + ("   <-- v2 WILL FAIL (sentinels present)" if n_sent else ""))
    return pred


def main(models, configs, tier):
    log = RT.setup_logging("S4_AGG", C.LOGS_DIR / "s4_aggregate.log")
    log.info("=" * 78)
    log.info("s4_aggregate — probability from the winning batch")
    log.info("=" * 78)
    n = 0
    for m, cfg, t, d in iter_cells(models, configs, tier):
        aggregate_cell(m, cfg, t, d, log)
        n += 1
    if n == 0:
        log.error("No cells found with batch_predictions.jsonl. Run s3_infer first.")
        sys.exit(2)
    log.info(f"\nAggregated {n} cell(s). Next: python stages/s5_metrics.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", choices=C.ALL_MODELS)
    ap.add_argument("--config", action="append", choices=C.CONFIGS)
    ap.add_argument("--tier", type=int, default=None)
    a = ap.parse_args()
    main(a.model or C.ALL_MODELS, a.config or C.CONFIGS, a.tier)
