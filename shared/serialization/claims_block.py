"""
Builds the structured-claims prompt section for a single patient.

Used by the claims configuration (entire prompt body besides
demographics) and the fused configuration (fixed prefix re-shown in
every batch).

Design notes. Do not change the rendering without re-running the
claims cells: the exact rendered text is part of what the reported
results measured.


  - Events organized as before / on / after the index date
  - Clinic location stripped from procedure labels (same CPT at different
    clinics is the same procedure, not new information)
  - Routine lab CPTs (800xx–899xx, Q99x, EPRC:*) compacted to a summary
    block — temporal range + unique-lab count kept; daily detail dropped
  - ILD-relevant procedures kept in full detail (CT chest, PFT, biopsy,
    bronchoscopy, echo, RHC, plus HCPCS J-codes for ICIs/ADCs/chemo)
  - ILD-risk medications flagged inline ([HIGH ILD risk — ICI] etc.)
"""
from __future__ import annotations
from datetime import datetime
from typing import Tuple, Dict, Any, Optional


# ── ILD-relevant code references ──────────────────────────────────────────────
ILD_CODES = {"J84.113", "J84.114", "J84.89", "J84.9",
             "J70.2", "J70.3", "J70.4", "J70.8", "J70.9", "J18.9"}

ILD_CODE_NAMES = {
    "J18.9":   "Pneumonia, unspecified organism",
    "J84.9":   "Interstitial pulmonary disease, unspecified",
    "J84.89":  "Other specified interstitial pulmonary diseases",
    "J70.4":   "Drug-induced interstitial lung disorders, unspecified",
    "J70.2":   "Acute drug-induced interstitial pneumonitis",
    "J70.3":   "Chronic drug-induced interstitial pneumonitis",
    "J84.113": "Idiopathic non-specific interstitial pneumonitis",
    "J84.114": "Acute interstitial pneumonitis",
}

ILD_RELEVANT_CPT = {
    # Chest CT
    "71250", "71260", "71270", "71275",
    # Chest X-ray
    "71045", "71046", "71047", "71048",
    # PFT
    "94010", "94060", "94070", "94150", "94200", "94375", "94620", "94621",
    "94640", "94660", "94664", "94726", "94727", "94728", "94729",
    # Bronchoscopy / lung biopsy
    "31622", "31623", "31624", "31625", "31626", "31628", "31629", "31632",
    "31633", "31635", "31636", "31638", "31640", "31641", "31643",
    "32607", "32608", "32609", "32650", "32651", "32652", "32653", "32654",
    # Echo (pulm HTN)
    "93303", "93304", "93306", "93307", "93308",
    # Right heart cath
    "93451", "93452", "93453", "93456", "93457", "93458", "93459", "93460", "93461",
    # CT-guided / open lung biopsy
    "32400", "32405",
    # PET
    "78814", "78815", "78816",
    # BAL / thoracentesis
    "32554", "32555", "32556", "32557",
    # 6-minute walk
    "94618",
    # HCPCS J-codes — ILD-risk Part B systemic therapies
    "J9271", "J9299", "J9022", "J9173", "J9023", "J9228", "J9119", "J9272", "J9347",   # ICIs
    "J9358", "J9317", "J9354", "J9359", "J9309", "J9357",                              # ADCs
    "J9355", "J9306",                                                                   # HER2 mAbs
    "J9035", "J9308",                                                                   # anti-VEGF
    "J9201", "J9040", "J9250", "J9260", "J9070", "J9080", "J9090", "J9091",            # cytotoxic
    "J9092", "J0594", "J9050", "J9280", "J9171", "J9267", "J9264", "J9305",
    "J9206", "J9263", "J9190",
    "J9330",                                                                            # mTOR
    "J9310", "J9312", "J1745",                                                          # other
}

ROUTINE_LAB_PREFIXES = tuple(f"{i:03d}" for i in range(800, 900)) + ("Q99",)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _days_from_ced(event_date, ced) -> Optional[int]:
    try:
        return (datetime.fromisoformat(str(event_date)[:10]) -
                datetime.fromisoformat(str(ced)[:10])).days
    except Exception:
        return None


def _is_routine_lab(code: str) -> bool:
    if str(code).startswith("EPRC:"):
        return True
    if code in ILD_RELEVANT_CPT:
        return False
    return any(code.startswith(p) for p in ROUTINE_LAB_PREFIXES)


def _fmt_day(d: int) -> str:
    return f"day {d}" if d < 0 else ("day 0" if d == 0 else f"day +{d}")


def _group_events(events):
    b, o, a = {}, {}, {}
    for day, label in events:
        target = b if day < 0 else (o if day == 0 else a)
        target.setdefault(label, []).append(day)
    return {"before": b, "on": o, "after": a}


def _fmt_days(days, section: str) -> str:
    ds = sorted(days, reverse=(section == "before"))
    shown = ds[:8]
    suffix = f" [×{len(ds)} total]" if len(ds) > 8 else ""
    return "(" + ", ".join(_fmt_day(d) for d in shown) + ")" + suffix


def _render_section(buckets, section: str):
    items = buckets.get(section, {})
    if not items:
        return ["  None"]
    return [f"  {label} {_fmt_days(items[label], section)}"
            for label in sorted(items)]


# ── Public API ─────────────────────────────────────────────────────────────────
def build_claims_block(
    patient: Dict[str, Any],
    dx_desc_map: Dict[str, str],
    drug_exp_map: Dict[str, Any],
    px_desc_map: Dict[str, str],
) -> Tuple[str, Dict[str, int]]:
    """
    Returns (claims_block_text, metadata).
    Does NOT include demographics — concatenate with demographics_block upstream.
    """
    pid = patient["patient_id"]
    ced = patient.get("ced") or patient.get("index_date", "")

    # ── Index ILD codes ───────────────────────────────────────────────────────
    index_codes = []
    for c in [str(patient.get("claims_dx1", "")), str(patient.get("claims_dx2", ""))]:
        c = c.strip()
        if c and c != "nan":
            name = ILD_CODE_NAMES.get(c, dx_desc_map.get(c, ""))
            index_codes.append(f"{c} — {name}" if name else c)

    # ── Drug risk lookup ──────────────────────────────────────────────────────
    p_drug = drug_exp_map.get(pid, {})
    try:
        ild_drugs = eval(p_drug.get("ild_risk_drugs_json", "[]"))
    except Exception:
        ild_drugs = []
    risk_map = {
        d.get("ild_matched_drug", "").lower():
            (d.get("ild_risk", ""), d.get("ild_drug_class", ""))
        for d in ild_drugs
    }

    # ── Diagnoses (excluding the index ILD codes themselves at day 0) ─────────
    dx_events = set()
    ild_norm = {c.replace(".", "") for c in ILD_CODES}
    for d in patient.get("diagnosis_codes", []):
        day = _days_from_ced(d.get("date", ""), ced)
        if day is None:
            continue
        code = str(d.get("dx_code", "")).strip()
        if not code or code == "nan":
            continue
        if code.replace(".", "") in ild_norm and abs(day) <= 1:
            continue
        desc = dx_desc_map.get(code, "")
        dx_events.add((day, f"{code} — {desc[:55]}" if desc else code))
    dx_b = _group_events(dx_events)

    # ── Procedures: split ILD-relevant vs routine labs ────────────────────────
    px_ild_events, px_lab_events = set(), set()
    for p in patient.get("procedures", []):
        day = _days_from_ced(p.get("date", ""), ced)
        if day is None:
            continue
        code = str(p.get("px_code", "")).strip()
        if not code or code == "nan":
            continue
        desc = px_desc_map.get(code, "")
        label = f"{code} — {desc[:45]}" if desc else code
        if _is_routine_lab(code):
            px_lab_events.add((day, label))
        else:
            px_ild_events.add((day, label))
    px_ild_b = _group_events(px_ild_events)

    lab_summary_lines = []
    if px_lab_events:
        lab_by_name = {}
        for day, label in px_lab_events:
            lab_by_name.setdefault(label, []).append(day)
        total_orders = sum(len(v) for v in lab_by_name.values())
        unique_labs  = len(lab_by_name)
        all_days     = [d for days in lab_by_name.values() for d in days]
        lab_summary_lines.append(
            f"  [{unique_labs} unique routine lab tests, {total_orders} total orders, "
            f"spanning {_fmt_day(min(all_days))} to {_fmt_day(max(all_days))}]"
        )
        for label in sorted(lab_by_name.keys())[:30]:
            lab_summary_lines.append(f"  {label} (×{len(lab_by_name[label])} orders)")
        if unique_labs > 30:
            lab_summary_lines.append(f"  ... and {unique_labs - 30} additional lab test types")

    # ── Medications with ILD-risk flagging ────────────────────────────────────
    rx_events = set()
    for r in patient.get("medications", []):
        day = _days_from_ced(r.get("date", ""), ced)
        if day is None:
            continue
        med = str(r.get("medication", "")).strip()
        if not med or med == "nan":
            continue
        risk = next((v for k, v in risk_map.items() if k and k in med.lower()), None)
        label = f"{med} [{risk[0]} ILD risk — {risk[1]}]" if risk else med
        rx_events.add((day, label))
    rx_b = _group_events(rx_events)

    # ── Assemble ──────────────────────────────────────────────────────────────
    lines = ["ILD diagnosis codes triggering cohort entry (on index date):"]
    lines.extend([f"  {c}" for c in index_codes] if index_codes else ["  Not available"])
    lines.append("")

    for section, lbl in [("before", "prior to"), ("on", "on"), ("after", "after")]:
        lines.append(f"Diagnoses recorded {lbl} the index date:")
        lines.extend(_render_section(dx_b, section))
        lines.append("")

    for section, lbl in [("before", "prior to"), ("on", "on"), ("after", "after")]:
        lines.append(f"ILD-relevant diagnostic procedures recorded {lbl} the index date:")
        lines.extend(_render_section(px_ild_b, section))
        lines.append("")

    lines.append("Routine laboratory monitoring (compacted summary):")
    lines.extend(lab_summary_lines if lab_summary_lines else ["  None"])
    lines.append("")

    for section, lbl in [("before", "prior to"), ("on", "on"), ("after", "after")]:
        lines.append(f"Medications recorded {lbl} the index date:")
        lines.extend(_render_section(rx_b, section))
        lines.append("")

    return "\n".join(lines), {
        "n_ild_procedures": len(px_ild_events),
        "n_routine_labs":   len(px_lab_events),
        "n_ild_risk_drugs": len(ild_drugs),
    }