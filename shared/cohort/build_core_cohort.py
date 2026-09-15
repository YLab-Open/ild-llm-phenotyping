"""
Builds the canonical core cohort.

Inclusion criteria
------------------
  1. Final adjudicated diagnosis in {0 = Not ILD, 1 = Probable, 2 = Possible}
     (drop 9 = Insufficient evidence — the Shen PPV-evaluable set)
  2. At least one EHR note in the observation window (has_ehr == True)

Derivation, from window_stats.csv:
  360 -> drop 43 with final_diagnosis == 9 -> 317
      -> drop 8 without retrievable EHR (5 Not ILD, 2 Probable, 1 Possible)
      -> 309   (126 Not ILD, 97 Possible, 86 Probable; 183 positives = 59.2%)

The observation window is -365 -> +180 days around the index date, for BOTH
claims and notes.

Outputs
-------
  shared/cohort/core_cohort.csv          — one row per patient
  shared/cohort/core_cohort_summary.txt  — human-readable summary
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared import config as C


def build(strict: bool = True) -> pd.DataFrame:
    ws = pd.read_csv(C.WINDOW_STATS)
    ws["patient_id"] = ws["patient_id"].astype(str)
    print(f"window_stats.csv: {len(ws)} patients")
    print(f"  final_diagnosis: {ws['final_diagnosis'].value_counts().to_dict()}")
    print(f"  has_ehr:         {ws['has_ehr'].value_counts().to_dict()}")

    eligible = C.GOLD_POSITIVE_CODES | C.GOLD_NEGATIVE_CODES          # {0,1,2}
    n_drop9 = int((~ws["final_diagnosis"].isin(eligible)).sum())
    mask = ws["final_diagnosis"].isin(eligible) & ws["has_ehr"]
    core = ws[mask].copy()
    n_drop_ehr = int(ws["final_diagnosis"].isin(eligible).sum()) - len(core)
    print(f"\nDerivation: {len(ws)} - {n_drop9} (final_diagnosis==9) "
          f"- {n_drop_ehr} (no retrievable EHR) = {len(core)}")

    enriched, missing = [], []
    for _, r in core.iterrows():
        pid = str(r["patient_id"])
        jpath = C.PATIENT_WINDOWS_DIR / f"{pid}.json"
        if not jpath.exists():
            missing.append(pid)
            continue
        with open(jpath) as f:
            pt = json.load(f)
        demo = pt.get("demographics", {}) or {}
        enriched.append({
            "patient_id":      pid,
            "empi":            str(pt.get("empi", "")),
            "final_diagnosis": int(r["final_diagnosis"]),
            "gold_label":      r["gold_label"],
            "cancer_group":    r["cancer_group"],
            "age_at_index":    r["age_at_index"],
            "sex":             demo.get("sex", ""),
            "race":            demo.get("race", ""),
            "insurance":       pt.get("insurance", ""),
            "n_ehr_notes":     int(r["n_ehr_notes"]),
        })
    if missing:
        msg = f"{len(missing)} cohort patients have no patient_windows JSON: {missing[:5]}"
        if strict:
            raise RuntimeError(msg + "  (rerun with strict=False only to inspect)")
        print(f"  WARNING: {msg}")

    return pd.DataFrame(enriched).sort_values("patient_id").reset_index(drop=True)


def main():
    C.COHORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    core_df = build()

    gold = pd.read_excel(C.SHEN_XLSX)
    xlsx_empis = set(gold["EMPI"].astype(str).str.strip())
    unmatched = [e for e in core_df["empi"] if e not in xlsx_empis]
    print(f"\nEMPI join vs Shen xlsx: {len(core_df) - len(unmatched)}/{len(core_df)} matched")
    if unmatched:
        raise RuntimeError(f"UNMATCHED EMPIs: {unmatched[:5]} — the join must be 309/309")

    core_df.to_csv(C.COHORT_CSV, index=False)
    print(f"\nSaved -> {C.COHORT_CSV}")

    lines = [
        "Core Cohort Summary",
        "=" * 60,
        f"Total patients: {len(core_df)}",
        "", "By gold label:", core_df["gold_label"].value_counts().to_string(),
        "", "By cancer group:", core_df["cancer_group"].value_counts().to_string(),
        "", "By insurance:", core_df["insurance"].value_counts().to_string(),
        "", "By sex:", core_df["sex"].value_counts().to_string(),
        "",
        f"Age — median: {core_df['age_at_index'].median():.1f}  "
        f"mean: {core_df['age_at_index'].mean():.1f}  "
        f"range: {core_df['age_at_index'].min():.0f}–{core_df['age_at_index'].max():.0f}",
        f"Notes/patient — median: {core_df['n_ehr_notes'].median():.0f}  "
        f"max: {core_df['n_ehr_notes'].max()}  "
        f"95th pct: {core_df['n_ehr_notes'].quantile(0.95):.0f}",
    ]
    (C.COHORT_CSV.parent / "core_cohort_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
