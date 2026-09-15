"""
Demographics header block.

Used identically by all three configurations — the same patient context
regardless of which modality follows.
"""
from typing import Dict, Any


RACE_MAP = {
    "1":  "American Indian/Alaska Native",  "01": "American Indian/Alaska Native",
    "2":  "Asian",                          "02": "Asian",
    "3":  "White",                          "03": "White",
    "4":  "Black/African American",         "04": "Black/African American",
    "5":  "Hispanic",                       "05": "Hispanic",
    "6":  "Native Hawaiian/Pacific Islander","06":"Native Hawaiian/Pacific Islander",
    "7":  "Other",                          "07": "Other",
    "8":  "Unknown",                        "08": "Unknown",
}


def build_demographics_block(patient: Dict[str, Any]) -> str:
    """Lines: CED → Age, Sex, Race, Insurance, Cancer type."""
    ced       = patient.get("ced") or patient.get("index_date", "")
    cancer    = patient.get("cancer_group", "").replace("_", " ")
    age       = patient.get("age_at_index")
    insurance = patient.get("insurance", "")
    demo      = patient.get("demographics", {}) or {}

    age_str = f"{age:.0f}" if age else "unknown"
    sex     = demo.get("sex", "unknown")
    race    = str(demo.get("race", "")).strip()
    race_label = RACE_MAP.get(race, race)

    lines = [
        f"Patient's index date (CED): {ced}",
        "",
        "Demographics and details about the patient on the index date:",
        f"Age: {age_str}",
        f"Sex: {sex}",
    ]
    if race_label and race_label not in ("nan", ""):
        lines.append(f"Race: {race_label}")
    lines += [
        f"Insurance: {insurance}",
        f"Cancer type: {cancer}",
        "",
    ]
    return "\n".join(lines)