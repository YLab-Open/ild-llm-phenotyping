"""
validate/v2_inference.py — CPU only. Gate after EACH inference cell.

    python validate/v2_inference.py --model Qwen3-8B --config notes
    python validate/v2_inference.py --all

Prints a one-page pass/fail summary to read in the terminal before releasing
the GPUs.

Checks (all gating):
  * batch count matches serialization exactly; no missing or duplicate
    (patient_id, batch_idx)
  * all 309 patients present after aggregation
  * ZERO sentinel patients
  * every label in the closed set
  * every probability in [0,1] and non-null
  * per-batch and per-patient records mutually consistent
  * run manifest complete, and its decoding parameters match config.py — a cell
    run under different settings cannot silently enter the grid

Remediation for a sentinel is a DETERMINISTIC re-run of the affected batches
(delete those lines from batch_predictions.jsonl and re-run the same s3_infer
command, which resumes). Never resampling, never temperature escalation.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import config as C
from validate._gate import Gate


def check_cell(model, config, tier, exit_on_fail=True):
    name = config if tier is None else f"{config}_tier{tier}"
    d = C.RESULTS_ROOT / model / name
    sd = C.RESULTS_ROOT / "_serialized" / name
    g = Gate(f"v2 INFERENCE — {model} / {name}")

    bp = d / "batch_predictions.jsonl"
    if not bp.exists():
        g.check(False, "batch_predictions.jsonl exists", str(bp))
        return g.report(exit_on_fail)

    recs = [json.loads(l) for l in open(bp) if l.strip()]
    exp = [json.loads(l) for l in open(sd / "serialized_prompts.jsonl") if l.strip()]
    exp_keys = {(r["patient_id"], int(r["batch_idx"])) for r in exp}
    got_keys = [(r["patient_id"], int(r["batch_idx"])) for r in recs]
    got_set = set(got_keys)

    dups = [k for k, n in Counter(got_keys).items() if n > 1]
    g.check(not dups, "no duplicate (patient_id, batch_idx)", f"{dups[:10]}")
    g.check(got_set == exp_keys,
            f"batch count matches serialization exactly ({len(exp_keys)})",
            f"missing={sorted(exp_keys - got_set)[:10]} "
            f"unexpected={sorted(got_set - exp_keys)[:10]}"
            if got_set != exp_keys else f"{len(got_set)} batches")
    
    # Sentinel batches are REPORTED, not gating. s4_aggregate drops a sentinel
    # batch unless EVERY batch for that patient failed, so one unparseable batch
    # among several costs a slice of evidence, not a label. The invariant that
    # actually matters -- no patient left without a usable label -- is checked
    # here from the batch records (so it holds even before s4 has run) and again
    # at patient level below.
    #
    # Re-running is NOT remediation for a model-side failure: decoding is
    # deterministic and chunk composition is fixed, so an identical re-run
    # reproduces the same malformed completion. A sentinel batch belonging to a
    # patient whose other batches parsed is left in place by design.
    by_pid_recs = defaultdict(list)
    for r in recs:
        by_pid_recs[r["patient_id"]].append(r)
    sent = [r for r in recs if r["predicted_label"] == C.SENTINEL_LABEL]
    all_sent = sorted(pid for pid, rs in by_pid_recs.items()
                      if all(x["predicted_label"] == C.SENTINEL_LABEL for x in rs))
    g.check(not all_sent,
            "no patient has ALL batches sentinel (none left without a label)",
            "\n".join(f"  {pid}: {len(by_pid_recs[pid])}/{len(by_pid_recs[pid])} "
                      f"batches unparseable" for pid in all_sent[:15]))
    if sent:
        g.note(f"{len(sent)} sentinel batch(es) — reported, not gating:\n" +
               "\n".join(f"  {r['patient_id']}/b{r['batch_idx']} "
                         f"(1 of {len(by_pid_recs[r['patient_id']])} batches for "
                         f"this patient): {r.get('sentinel_reason','')}"
                         for r in sent[:15]))
    
    bad_lab = {r["predicted_label"] for r in recs} - set(C.ALL_LABELS) - {C.SENTINEL_LABEL}
    g.check(not bad_lab, "every batch label in the closed set", f"{sorted(bad_lab)}")
    bad_p = [(r["patient_id"], r["batch_idx"], r.get("ild_probability"))
             for r in recs
             if r["predicted_label"] != C.SENTINEL_LABEL          # sentinels carry None by design
             and (r.get("ild_probability") is None
                  or not (0.0 <= float(r["ild_probability"]) <= 1.0))]
    g.check(not bad_p, "every batch probability non-null and in [0,1]", f"{bad_p[:10]}")

    census = Counter(r.get("parse_path") for r in recs)
    g.note(f"parse_path census: {dict(census)}")
    capped = sum(1 for r in recs if r.get("finish_reason") == "length")
    if capped:
        g.note(f"{capped} generation(s) hit the {C.GEN_MAX_TOKENS}-token cap "
               f"(finish_reason='length') — inspect raw_response.")

    # patient level
    pp = d / "patient_predictions.csv"
    if pp.exists():
        pred = pd.read_csv(pp)
        # As in v1: a tier variant holds only the patients possessing that note
        # tier. Gate against the variant's expected membership, derived
        # independently by finegrained.build_subcohorts.
        if tier is not None:
            base = C.CLEAN_ROOT / "analysis" / "results" / "finegrained"
            var = next((base / f"{config}_p{t['priority']}_{t['slug']}" /
                        "expected_cohort.csv"
                        for t in C.TIER_VARIANTS if t["tier_id"] == tier), None)
            if var is None or not var.exists():
                g.check(False, "expected_cohort.csv for this tier variant exists",
                        "run: python analysis/finegrained.py --build-subcohorts")
                return g.report(exit_on_fail)
            cohort = pd.read_csv(var)
        else:
            cohort = pd.read_csv(C.COHORT_CSV)
        cohort["patient_id"] = cohort["patient_id"].astype(str)
        pred["patient_id"] = pred["patient_id"].astype(str)
        g.check(len(pred) == len(cohort) and
                set(pred["patient_id"]) == set(cohort["patient_id"]),
                f"all {len(cohort)} patients present after aggregation",
                f"n={len(pred)}")
        
        sent_pids = (pred.loc[pred.get("is_sentinel", False) == True, "patient_id"]
                     .tolist() if "is_sentinel" in pred else [])
        g.check(not sent_pids, "ZERO sentinel patients", f"{sent_pids[:10]}")
        if "n_sentinel_batches" in pred:
            nb = int(pred["n_sentinel_batches"].sum())
            if nb:
                g.note(f"{nb} sentinel batch(es) across "
                       f"{int((pred['n_sentinel_batches'] > 0).sum())} patient(s), "
                       f"all of whom retained a label from their other batches")
        bad = pred[pred["ild_probability"].isna() |
                   (pd.to_numeric(pred["ild_probability"], errors="coerce") < 0) |
                   (pd.to_numeric(pred["ild_probability"], errors="coerce") > 1)]
        g.check(len(bad) == 0, "every patient probability non-null and in [0,1]",
                f"{bad['patient_id'].tolist()[:10]}")

        # per-batch <-> per-patient consistency
        from shared.evaluation.metrics import aggregate_batches
        by = defaultdict(list)
        for r in recs:
            by[r["patient_id"]].append(r)
        bad_rows = []
        for _, row in pred.iterrows():
            a = aggregate_batches(by[row["patient_id"]])
            if a["predicted_label"] != row["predicted_label"]:
                bad_rows.append((row["patient_id"], row["predicted_label"],
                                 a["predicted_label"]))
            elif a["ild_probability"] is not None and not (
                    abs(float(a["ild_probability"]) - float(row["ild_probability"])) < 1e-9):
                bad_rows.append((row["patient_id"], "prob", a["ild_probability"],
                                 row["ild_probability"]))
        g.check(not bad_rows,
                "per-patient rows reproduce from per-batch records", f"{bad_rows[:10]}")
    else:
        g.note("patient_predictions.csv not present yet — run s4_aggregate, then "
               "re-run v2 to gate the patient-level checks.")

    # manifest
    mp = d / "run_manifest.json"
    if not mp.exists():
        g.check(False, "run_manifest.json exists", str(mp))
    else:
        man = json.loads(mp.read_text())
        need = ["git_commit", "prompt_sha256", "tokenizer", "vllm_version",
                "decoding", "tensor_parallel_size", "gpu_indices",
                "batches_expected", "resume_count", "finished_utc"]
        miss = [k for k in need if k not in man or man[k] in (None, "")]
        g.check(not miss, "run manifest complete", f"missing {miss}")
        dec = man.get("decoding", {})
        drift = []
        for k, want in (("temperature", C.TEMPERATURE), ("seed", C.SEED),
                        ("max_tokens", C.GEN_MAX_TOKENS),
                        ("repetition_penalty", C.REPETITION_PENALTY)):
            if dec.get(k) != want:
                drift.append(f"{k}: manifest={dec.get(k)!r} config={want!r}")
        if man.get("tensor_parallel_size") != C.TENSOR_PARALLEL_SIZE:
            drift.append(f"tensor_parallel_size: {man.get('tensor_parallel_size')} "
                         f"!= {C.TENSOR_PARALLEL_SIZE}")
        g.check(not drift, "decoding parameters match config.py",
                "\n".join(drift))
        g.note(f"resume_count={man.get('resume_count')} "
               f"inference_min={man.get('run_minutes')} "
               f"guided={dec.get('guided_decoding')} "
               f"prompt_sha={str(man.get('prompt_sha256'))[:16]}")

    return g.report(exit_on_fail)


def main(models, configs, tier, run_all):
    if run_all:
        results = []
        for m in C.ALL_MODELS:
            for cfg in C.CONFIGS:
                name = cfg if tier is None else f"{cfg}_tier{tier}"
                if (C.RESULTS_ROOT / m / name / "batch_predictions.jsonl").exists():
                    results.append(((m, cfg), check_cell(m, cfg, tier, exit_on_fail=False)))
        bad = [k for k, ok in results if not ok]
        print(f"\n{'='*78}\nv2 SUMMARY: {len(results)-len(bad)}/{len(results)} cells pass")
        if bad:
            print("  FAILING: " + ", ".join(f"{m}/{c}" for m, c in bad))
            sys.exit(1)
        return
    for m in models:
        for cfg in configs:
            check_cell(m, cfg, tier)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", choices=C.ALL_MODELS)
    ap.add_argument("--config", action="append", choices=C.CONFIGS)
    ap.add_argument("--tier", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="check every cell on disk")
    a = ap.parse_args()
    if not a.all and not (a.model and a.config):
        ap.error("give --model and --config, or --all")
    main(a.model or [], a.config or [], a.tier, a.all)
