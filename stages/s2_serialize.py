"""
stages/s2_serialize.py — CPU only. Builds the prompts every model will see.

    python stages/s2_serialize.py --config claims
    python stages/s2_serialize.py --config notes
    python stages/s2_serialize.py --config fused
    python stages/s2_serialize.py --config notes --tier 1     # ablation variant
    python stages/s2_serialize.py --config claims --limit 5   # smoke test

One parameterised serializer covers all three configurations; they differ only
in which blocks form the fixed prefix and whether notes are included.

  claims : prefix = demographics + claims;   no notes;  exactly 1 batch/patient
  notes  : prefix = demographics;            notes;     paginated
  fused  : prefix = demographics + claims;   notes;     paginated
  --tier N restricts notes to ild_priority == N (the ablation)

Every note and every fixed prefix is costed under EVERY registry tokenizer and
packed against the per-note MAXIMUM, so one pagination serves all models and no
model can overflow. There is no truncation path here or in s3_infer.
serialization_stats.csv records per-tokenizer costs so v1 can prove zero
projected truncation before any GPU is touched.

Outputs (per config, under results/_serialized/<config>[_tier<N>]/):
  serialized_prompts.jsonl   one record per (patient, batch)
  serialization_stats.csv    per-patient token accounting, all tokenizers
  note_manifest.csv          every note -> the batch it landed in (v1 reconciles)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import config as C
from shared import runtime as RT
from shared.serialization.demographics import build_demographics_block
from shared.serialization.claims_block import build_claims_block
from shared.serialization.notes_block import build_notes_block
from shared.serialization.pagination import annotate_notes, paginate, prefix_tokens_max

USES_CLAIMS = {"claims": True, "notes": False, "fused": True}
USES_NOTES = {"claims": False, "notes": True, "fused": True}


def out_dir(config: str, tier: int | None) -> Path:
    name = config if tier is None else f"{config}_tier{tier}"
    return C.RESULTS_ROOT / "_serialized" / name


def load_lookups(log):
    """ICD/CPT description maps and the drug-exposure table."""
    dx_desc_map = {}
    if C.DX_MAP_PATH.exists():
        df = pd.read_csv(C.DX_MAP_PATH, low_memory=False)
        if {"dx", "description"} <= set(df.columns):
            dx_desc_map = dict(zip(df["dx"].astype(str), df["description"].astype(str)))
    px_desc_map = {}
    if C.PX_MAP_PATH.exists():
        df = pd.read_csv(C.PX_MAP_PATH, low_memory=False)
        if {"px", "description"} <= set(df.columns):
            px_desc_map = dict(zip(df["px"].astype(str), df["description"].astype(str)))
    drug_df = pd.read_csv(C.DRUG_EXP)
    drug_df["Patient_Num"] = drug_df["Patient_Num"].astype(str)
    drug_exp_map = drug_df.set_index("Patient_Num").to_dict("index")
    log.info(f"  lookups: {len(dx_desc_map):,} ICD, {len(px_desc_map):,} CPT, "
             f"{len(drug_exp_map):,} drug rows")
    return dx_desc_map, px_desc_map, drug_exp_map


def main(config: str, tier: int | None, limit: int | None):
    od = out_dir(config, tier)
    od.mkdir(parents=True, exist_ok=True)
    log = RT.setup_logging("S2_SERIALIZE", C.LOGS_DIR / f"s2_serialize_{od.name}.log")
    log.info("=" * 72)
    log.info(f"s2_serialize — config={config} tier={tier} "
             f"(prefix: demographics{'+claims' if USES_CLAIMS[config] else ''}; "
             f"notes: {'yes' if USES_NOTES[config] else 'no'})")
    log.info("=" * 72)

    cohort = pd.read_csv(C.COHORT_CSV)
    cohort["patient_id"] = cohort["patient_id"].astype(str)
    if limit:
        cohort = cohort.head(limit)
        log.info(f"  --limit {limit}: smoke test on the first {limit} patients")
    log.info(f"Cohort: {len(cohort)} patients")

    # every tokenizer, mandatory: pagination packs against the worst case
    log.info(f"Loading {len(C.ALL_MODELS)} tokenizers for worst-case costing...")
    tokenizers = RT.load_all_tokenizers(required=True)
    log.info(f"  loaded: {', '.join(tokenizers)}")

    ehr = None
    if USES_NOTES[config]:
        log.info(f"Loading EHR: {C.EHR_TYPED}")
        ehr = pd.read_parquet(C.EHR_TYPED)
        ehr["Patient_Num_New"] = ehr["Patient_Num_New"].astype(str)
        ehr = ehr[ehr["Patient_Num_New"].isin(set(cohort["patient_id"]))].copy()
        for c in ("text_clean", "UPDATE_DATE", "note_type", "ild_priority"):
            if c not in ehr.columns:
                log.error(f"Missing EHR column {c}; have {list(ehr.columns)}")
                sys.exit(1)
        if tier is not None:
            before = len(ehr)
            ehr = ehr[ehr["ild_priority"] == tier].copy()
            log.info(f"  tier {tier} filter: {before:,} -> {len(ehr):,} notes")
        log.info(f"  EHR rows in cohort: {len(ehr):,} "
                 f"({ehr['Patient_Num_New'].nunique()} patients)")

    dx_map = px_map = drug_map = None
    if USES_CLAIMS[config]:
        dx_map, px_map, drug_map = load_lookups(log)

    text_col, date_col, type_col = "text_clean", "UPDATE_DATE", "note_type"
    rows, stats_rows, note_rows, skipped = [], [], [], []

    for n_done, (_, pt) in enumerate(cohort.iterrows(), start=1):
        pid = pt["patient_id"]
        jpath = C.PATIENT_WINDOWS_DIR / f"{pid}.json"
        if not jpath.exists():
            skipped.append((pid, "no patient_windows JSON"))
            continue
        patient = json.loads(jpath.read_text())
        ced = patient.get("ced") or patient.get("index_date", "")

        prefix = build_demographics_block(patient)
        claims_meta = {}
        if USES_CLAIMS[config]:
            claims_text, claims_meta = build_claims_block(patient, dx_map, drug_map, px_map)
            prefix += claims_text
        pref = prefix_tokens_max(prefix, tokenizers)

        common = {
            "patient_id": pid, "empi": str(pt.get("empi", "")),
            "cancer_group": pt["cancer_group"],
            "final_diagnosis": int(pt["final_diagnosis"]),
            "gold_label": pt["gold_label"], "ced": ced, "config": config,
            "tier": tier,
        }

        # ── claims: single pass, no notes ──────────────────────────────────────
        if not USES_NOTES[config]:
            rows.append({**common, "batch_idx": 0, "n_batches": 1,
                         "prompt": prefix,
                         "prefix_tokens_max": pref["max"],
                         "prefix_tokens_argmax": pref["argmax"],
                         "body_tokens_max": pref["max"],
                         "n_notes_in_batch": 0, "tiers_in_batch": "[]",
                         **claims_meta})
            stats_rows.append({**{k: common[k] for k in
                                  ("patient_id", "gold_label", "cancer_group")},
                               "n_batches": 1, "n_notes_total": 0,
                               "total_note_tokens_max": 0,
                               "prefix_tokens_max": pref["max"],
                               "prefix_tokens_argmax": pref["argmax"],
                               **{f"prefix_tokens__{m}": v
                                  for m, v in pref["per_model"].items()}})
            continue

        # ── notes / fused ─────────────────────────────────────────────────────
        pnotes = ehr[ehr["Patient_Num_New"] == pid].copy()
        if len(pnotes) == 0:
            skipped.append((pid, "no EHR notes in window"))
            continue
        annotated = annotate_notes(pnotes, ced, text_col, date_col, tokenizers)
        if len(annotated) == 0:
            skipped.append((pid, "all notes dropped (bad dates / empty text)"))
            continue

        budget = C.MAX_TOKENS_PER_PASS - pref["max"] - C.SECTION_OVERHEAD
        if budget <= 0:
            skipped.append((pid, f"prefix {pref['max']} exhausts the per-pass budget"))
            continue

        batches = paginate(annotated, budget_per_pass=budget, patient_id=pid)
        if not batches:
            skipped.append((pid, "pagination produced no batches"))
            continue

        for bidx, idxs in enumerate(batches):
            bdf = annotated.loc[idxs]
            notes_section = build_notes_block(bdf, text_col=text_col,
                                              type_col=type_col, rel_day_col="rel_day")
            prompt = prefix + notes_section
            body_max = {m: len(t.encode(prompt)) for m, t in tokenizers.items()}
            worst = max(body_max, key=body_max.get)
            rows.append({**common, "batch_idx": bidx, "n_batches": len(batches),
                         "prompt": prompt,
                         "prefix_tokens_max": pref["max"],
                         "prefix_tokens_argmax": pref["argmax"],
                         "body_tokens_max": body_max[worst],
                         "body_tokens_argmax": worst,
                         "body_tokens_per_model": json.dumps(body_max),
                         "n_notes_in_batch": len(idxs),
                         "note_tokens_in_batch": int(bdf["note_tokens"].sum()),
                         "tiers_in_batch": json.dumps(
                             sorted({int(t) for t in bdf["ild_priority"]}))})
            # note -> batch manifest, so v1 can reconcile counts against source
            for ix in idxs:
                r = annotated.loc[ix]
                note_rows.append({"patient_id": pid, "batch_idx": bidx,
                                  "note_index": int(ix), "rel_day": int(r["rel_day"]),
                                  "ild_priority": int(r["ild_priority"]),
                                  "note_tokens_max": int(r["note_tokens"]),
                                  "note_tokens_argmax": r["note_tokens_argmax"]})

        srow = {"patient_id": pid, "gold_label": pt["gold_label"],
                "cancer_group": pt["cancer_group"],
                "n_batches": len(batches), "n_notes_total": len(annotated),
                "total_note_tokens_max": int(annotated["note_tokens"].sum()),
                "prefix_tokens_max": pref["max"],
                "prefix_tokens_argmax": pref["argmax"],
                "budget_per_pass": budget}
        srow.update({f"prefix_tokens__{m}": v for m, v in pref["per_model"].items()})
        srow.update({f"note_tokens_sum__{m}": int(annotated[f"note_tokens__{m}"].sum())
                     for m in tokenizers})
        stats_rows.append(srow)

        if n_done % 25 == 0:
            log.info(f"  ...{n_done}/{len(cohort)} patients, {len(rows)} batches so far")

    # ── write ─────────────────────────────────────────────────────────────────
    with open(od / "serialized_prompts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    pd.DataFrame(stats_rows).to_csv(od / "serialization_stats.csv", index=False)
    if note_rows:
        pd.DataFrame(note_rows).to_csv(od / "note_manifest.csv", index=False)

    log.info(f"\nSerialized {len(rows)} batches over {len(stats_rows)} patients -> {od}")
    if skipped:
        log.warning(f"  SKIPPED {len(skipped)} patients:")
        for pid, why in skipped[:20]:
            log.warning(f"    {pid}: {why}")
        log.warning("  v1 will FAIL unless every cohort patient is present.")
    if stats_rows:
        st = pd.DataFrame(stats_rows)
        log.info(f"  batches/patient — median {st['n_batches'].median():.0f}  "
                 f"max {st['n_batches'].max()}  total {int(st['n_batches'].sum())}")
        log.info(f"  worst-case prefix tokens — median {st['prefix_tokens_max'].median():.0f}  "
                 f"max {st['prefix_tokens_max'].max()}")
        if "prefix_tokens_argmax" in st:
            log.info(f"  most expensive tokenizer for the prefix: "
                     f"{st['prefix_tokens_argmax'].value_counts().head(3).to_dict()}")
    log.info("\nNext: python validate/v1_serialization.py --config "
             f"{config}" + (f" --tier {tier}" if tier is not None else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=C.CONFIGS)
    ap.add_argument("--tier", type=int, default=None,
                    help="restrict notes to this ild_priority (ablation variant)")
    ap.add_argument("--limit", type=int, default=None, help="smoke test on N patients")
    a = ap.parse_args()
    if a.tier is not None and a.config == "claims":
        ap.error("--tier applies to the notes and fused configurations only")
    main(a.config, a.tier, a.limit)
