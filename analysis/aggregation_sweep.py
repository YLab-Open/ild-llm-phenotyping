"""
analysis/aggregation_sweep.py — CPU only. Aggregation-rule and probability /
confidence threshold sweeps, for EVERY model x config.

    python analysis/aggregation_sweep.py

Both sweeps run over every cell that exists on disk, reading per-batch
records.

Sentinels: a sentinel batch is excluded from every rule's vote (it is not a
label), and a patient whose batches are all sentinels stays a sentinel. Since
v2 gates on zero sentinels, this should never engage in a released grid.

Outputs: analysis/results/aggregation/{aggregation_rules,threshold_sweep}.csv
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
from shared.evaluation.metrics import AGGREGATION_RULES, binary_metrics

OUT = C.CLEAN_ROOT / "analysis" / "results" / "aggregation"
THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def cells(tier):
    for m in C.ALL_MODELS:
        for cfg in C.CONFIGS:
            name = cfg if tier is None else f"{cfg}_tier{tier}"
            p = C.RESULTS_ROOT / m / name / "batch_predictions.jsonl"
            if p.exists():
                yield m, cfg, p


def confusion(by_pid, decide):
    tp = fp = tn = fn = 0
    for pid, batches in by_pid.items():
        gold = batches[0]["final_diagnosis"] in C.GOLD_POSITIVE_CODES
        pos = decide(batches)
        tp += gold and pos; fp += (not gold) and pos
        tn += (not gold) and (not pos); fn += gold and (not pos)
    return int(tp), int(fp), int(tn), int(fn)


def main(tier):
    OUT.mkdir(parents=True, exist_ok=True)
    agg_rows, thr_rows = [], []

    for model, cfg, p in cells(tier):
        recs = [json.loads(l) for l in open(p) if l.strip()]
        by_pid = defaultdict(list)
        for r in recs:
            by_pid[r["patient_id"]].append(r)
        n_multi = sum(1 for b in by_pid.values() if len(b) > 1)

        for rule, fn_rule in AGGREGATION_RULES.items():
            m = binary_metrics(*confusion(
                by_pid, lambda bs, f=fn_rule: f(bs) in C.POSITIVE_LABELS))
            agg_rows.append({"model": model, "config": cfg, "rule": rule,
                             "n_multi_batch": n_multi, **m})

        for gate, getter in (("prob", lambda b: b.get("ild_probability") or 0.0),
                             ("conf", lambda b: b.get("confidence_score") or 0.0)):
            for tau in THRESHOLDS:
                def decide(batches, g=getter, t=tau):
                    return any(b.get("predicted_label") in C.POSITIVE_LABELS
                               and g(b) >= t for b in batches)
                m = binary_metrics(*confusion(by_pid, decide))
                thr_rows.append({"model": model, "config": cfg,
                                 "gate": gate, "tau": tau, **m})

    if not agg_rows:
        print("No cells found. Run s3_infer/s4_aggregate first.")
        sys.exit(2)
    pd.DataFrame(agg_rows).to_csv(OUT / "aggregation_rules.csv", index=False)
    pd.DataFrame(thr_rows).to_csv(OUT / "threshold_sweep.csv", index=False)
    print(f"aggregation rules : {len(agg_rows)} rows -> {OUT/'aggregation_rules.csv'}")
    print(f"threshold sweep   : {len(thr_rows)} rows -> {OUT/'threshold_sweep.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=None)
    main(ap.parse_args().tier)
