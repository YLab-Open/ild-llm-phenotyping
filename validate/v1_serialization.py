"""
validate/v1_serialization.py — CPU only. Gate after s2_serialize.

    python validate/v1_serialization.py --config notes
    python validate/v1_serialization.py --config fused --tier 1

Checks (all gating):
  1. every cohort patient present; no empty prompt
  2. every batch fits EVERY model's context with CONTEXT_MARGIN_TOKENS to
     spare, measured with each model's real chat template. This is the check
     that proves zero projected truncation before a GPU is touched.
  3. every note in the window appears in exactly one batch, counts reconciling
     against serialization_stats
  4. day offsets within -365..+180 (the observation window)
  5. the fused claims prefix is byte-identical across a patient's batches
  6. the notes shown for a patient are identical between notes and fused,
     differing only in partitioning

On PASS writes v1_PASSED.json into the serialized directory as a record that
the configuration cleared this gate.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import config as C
from shared import runtime as RT
from validate._gate import Gate

NOTE_HDR = re.compile(r"^--- (.*?) \((?:(\d+) days (before|after) index date|on index date)\) ---$", re.M)


def sdir(config, tier):
    return C.RESULTS_ROOT / "_serialized" / (config if tier is None else f"{config}_tier{tier}")

def _variant_dirs(config, tier):
    """expected_cohort.csv for the (config, tier) variant. The directory is
    named by PACKING rank while --tier is the CLASSIFIER tier, so resolve
    through config.TIER_VARIANTS rather than formatting a name."""
    base = C.CLEAN_ROOT / "analysis" / "results" / "finegrained"
    for t in C.TIER_VARIANTS:
        if t["tier_id"] == tier:
            yield base / f"{config}_p{t['priority']}_{t['slug']}" / "expected_cohort.csv"


def note_headers(prompt):
    out = []
    for m in NOTE_HDR.finditer(prompt):
        n, direction = m.group(2), m.group(3)
        day = 0 if n is None else (int(n) if direction == "after" else -int(n))
        out.append((m.group(1), day))
    return out


def main(config, tier, skip_context):
    d = sdir(config, tier)
    g = Gate(f"v1 SERIALIZATION — {d.name}")
    pj = d / "serialized_prompts.jsonl"
    if not pj.exists():
        g.check(False, "serialized_prompts.jsonl exists", str(pj))
        g.report()

    recs = [json.loads(l) for l in open(pj) if l.strip()]
    stats = pd.read_csv(d / "serialization_stats.csv")
    # A tier variant legitimately contains only the patients possessing that
    # note tier -- a patient with no tier-N note yields no batches, by design
    # (analysis/finegrained.py). Gate against the variant's expected membership,
    # which build_subcohorts derived independently from the typed EHR parquet,
    # so the check is still exact: it catches a patient silently dropped for
    # any OTHER reason.
    cohort_src = "core cohort"
    if tier is not None:
        var = next((v for v in _variant_dirs(config, tier)), None)
        if var is None or not var.exists():
            g.check(False, "expected_cohort.csv for this tier variant exists",
                    f"run: python analysis/finegrained.py --build-subcohorts")
            g.report()
        cohort = pd.read_csv(var)
        cohort_src = f"tier-{tier} subcohort ({var.parent.name})"
    else:
        cohort = pd.read_csv(C.COHORT_CSV)
    cohort["patient_id"] = cohort["patient_id"].astype(str)

    # 1 — coverage
    pids = {r["patient_id"] for r in recs}
    expect = set(cohort["patient_id"])
    missing, extra = expect - pids, pids - expect
    g.check(not missing and not extra,
            f"all {len(expect)} patients present ({cohort_src}), no extras",
            f"missing={sorted(missing)[:10]} extra={sorted(extra)[:10]}"
            if (missing or extra) else f"{len(pids)} patients, {len(recs)} batches")
    empt = [r["patient_id"] for r in recs if not r["prompt"].strip()]
    g.check(not empt, "no empty prompt", f"empty for {empt[:10]}")

    # batch index integrity
    bad_idx = []
    by_pid = defaultdict(list)
    for r in recs:
        by_pid[r["patient_id"]].append(r)
    for pid, rs in by_pid.items():
        idxs = sorted(r["batch_idx"] for r in rs)
        if idxs != list(range(len(rs))) or any(r["n_batches"] != len(rs) for r in rs):
            bad_idx.append(pid)
    g.check(not bad_idx, "batch_idx contiguous from 0 and n_batches consistent",
            f"bad: {bad_idx[:10]}")

    # 2 — every batch fits every model's context, with margin
    if skip_context:
        g.note("--skip-context given: the per-tokenizer context check was SKIPPED. "
               "This is the check that proves zero projected truncation; do not "
               "skip it before a real run.")
    else:
        toks = RT.load_all_tokenizers(required=True)
        sysp, usert, _ = RT.load_prompt(config)
        cap = C.MAX_MODEL_LEN - C.GEN_MAX_TOKENS - C.CONTEXT_MARGIN_TOKENS
        worst = {}
        overflow = []
        for r in recs:
            for mname, tk in toks.items():
                chat = RT.chat_wrap(tk, sysp, usert, r["prompt"],
                                    C.MODELS[mname]["is_qwen3"])
                n = len(tk.encode(chat))
                if n > worst.get(mname, (0, None))[0]:
                    worst[mname] = (n, f"{r['patient_id']}/b{r['batch_idx']}")
                if n > cap:
                    overflow.append((mname, r["patient_id"], r["batch_idx"], n))
        detail = "\n".join(f"{m:<24} worst {v[0]:>7} tok  ({v[1]})  headroom "
                           f"{cap - v[0]:>6}" for m, v in sorted(worst.items()))
        g.check(not overflow,
                f"every batch fits every model within {cap} tokens "
                f"({C.MAX_MODEL_LEN} - {C.GEN_MAX_TOKENS} gen - "
                f"{C.CONTEXT_MARGIN_TOKENS} margin)",
                detail if not overflow else
                "OVERFLOW:\n" + "\n".join(f"  {m} {p}/b{b}: {n} tok" for m, p, b, n in overflow[:20]))

    # 3 — note reconciliation
    nm_p = d / "note_manifest.csv"
    if C.CONFIGS.index(config) >= 0 and nm_p.exists():
        nm = pd.read_csv(nm_p)
        dup = nm.duplicated(["patient_id", "note_index"]).sum()
        g.check(dup == 0, "every note lands in exactly one batch",
                f"{dup} notes appear in more than one batch")
        per_pt = nm.groupby("patient_id").size()
        s_idx = stats.set_index("patient_id")["n_notes_total"]
        common = per_pt.index.intersection(s_idx.index)
        mismatch = [(p, int(per_pt[p]), int(s_idx[p])) for p in common
                    if int(per_pt[p]) != int(s_idx[p])]
        g.check(not mismatch, "note counts reconcile with serialization_stats",
                f"{mismatch[:10]}")
        # 4 — window
        lo, hi = int(nm["rel_day"].min()), int(nm["rel_day"].max())
        g.check(-365 <= lo and hi <= 180,
                "note day offsets within the -365..+180 window",
                f"observed {lo} .. {hi}")
    elif config != "claims":
        g.check(False, "note_manifest.csv exists", str(nm_p))

    # 5 — fused prefix stability within a patient
    if config == "fused":
        unstable = []
        for pid, rs in by_pid.items():
            if len(rs) < 2:
                continue
            prefixes = {r["prompt"].split("=== CLINICAL NOTES ===")[0] for r in rs}
            if len(prefixes) != 1:
                unstable.append(pid)
        g.check(not unstable,
                "claims prefix byte-identical across a patient's batches",
                f"unstable for {unstable[:10]}")

    # 6 — notes set identical between notes and fused
    if config in ("notes", "fused"):
        other = "fused" if config == "notes" else "notes"
        od = sdir(other, tier)
        opj = od / "serialized_prompts.jsonl"
        if not opj.exists():
            g.note(f"'{other}' not serialized yet — the cross-config note-set "
                   f"comparison will run when it is.")
        else:
            orecs = [json.loads(l) for l in open(opj) if l.strip()]
            oby = defaultdict(list)
            for r in orecs:
                oby[r["patient_id"]].append(r)
            diff = []
            for pid in set(by_pid) & set(oby):
                a = sorted(h for r in by_pid[pid] for h in note_headers(r["prompt"]))
                b = sorted(h for r in oby[pid] for h in note_headers(r["prompt"]))
                if a != b:
                    diff.append((pid, len(a), len(b)))
            g.check(not diff,
                    f"notes shown are identical between '{config}' and '{other}' "
                    f"(partitioning may differ)",
                    f"{diff[:10]}")

    ok = g.report(exit_on_fail=False)
    if ok:
        (d / "v1_PASSED.json").write_text(json.dumps({
            "config": config, "tier": tier, "n_patients": len(pids),
            "n_batches": len(recs), "context_check_skipped": bool(skip_context),
            "passed_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": RT.git_commit(),
        }, indent=2))
        print(f"\nWrote {d/'v1_PASSED.json'} — this configuration is cleared for s3_infer.")
    else:
        (d / "v1_PASSED.json").unlink(missing_ok=True)
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=C.CONFIGS)
    ap.add_argument("--tier", type=int, default=None)
    ap.add_argument("--skip-context", action="store_true",
                    help="skip the per-tokenizer context check (development only)")
    a = ap.parse_args()
    main(a.config, a.tier, a.skip_context)
