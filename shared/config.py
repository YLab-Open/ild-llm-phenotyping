"""
Single source of truth for paths, budgets, tier priorities, label priorities
and the model registry. Imported by every script in the pipeline.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
# CLEAN_ROOT is the repository checkout itself, derived from this file's own
# location so the repository can be cloned under any name. ILD_ROOT overrides it.
CLEAN_ROOT    = Path(os.environ.get("ILD_ROOT",
                                    Path(__file__).resolve().parents[1]))
SHARED_DIR    = CLEAN_ROOT / "shared"

COHORT_CSV    = SHARED_DIR / "cohort" / "core_cohort.csv"
# DATA_DIR holds symlinks to the shared inputs (created by scaffold.sh).
# ILD_DATA_DIR overrides it when the inputs live elsewhere.
DATA_DIR      = Path(os.environ.get("ILD_DATA_DIR", str(SHARED_DIR / "data")))
PROMPTS_DIR   = SHARED_DIR / "prompts"
LOGS_DIR      = CLEAN_ROOT / "logs"
RESULTS_ROOT  = CLEAN_ROOT / "results"

# Symlinks into DATA_DIR (created by scaffold.sh)
PATIENT_WINDOWS_DIR = DATA_DIR / "patient_windows"
EHR_TYPED           = DATA_DIR / "ehr_windowed_typed.parquet"
DRUG_EXP            = DATA_DIR / "patient_drug_exposure.csv"
WINDOW_STATS        = DATA_DIR / "window_stats.csv"
SHEN_XLSX           = DATA_DIR / "ILD_01_EMPI_Final_Diagnosis_360.xlsx"

# ICD/CPT code-mapping CSVs
DX_MAP_PATH = Path(os.environ.get("ILD_DX_MAP", str(DATA_DIR / "df_dx_mapping.csv")))
PX_MAP_PATH = Path(os.environ.get("ILD_PX_MAP", str(DATA_DIR / "df_px_mapping.csv")))

BASE_MODEL_DIR  = Path(os.environ.get("ILD_MODEL_SHARE", "./models"))

# ── Configurations ─────────────────────────────────────────────────────────────
CONFIGS        = ["claims", "notes", "fused"]
CONFIG_TITLE   = {"claims": "Claims only", "notes": "Notes only", "fused": "Claims + notes (fused)"}

# ── Model registry ─────────────────────────────────────────────────────────────
# `tp` (tensor-parallel size) is NOT stored here. It is fixed globally at
# TENSOR_PARALLEL_SIZE for every model: TP changes the order of tensor-parallel
# reductions, which changes floating-point accumulation, so varying it across
# models would introduce a cross-model confound.
#
# `reported` marks the models in the manuscript's reported set.
MODELS = {
    "medgemma-27b-text-it":   {"params_b": 27,  "family": "gemma",  "is_qwen3": False, "reported": True},
    "medgemma-4b-it":         {"params_b": 4,   "family": "gemma",  "is_qwen3": False, "reported": True},
    "gemma-3-27b-it":         {"params_b": 27,  "family": "gemma",  "is_qwen3": False, "reported": True},
    "HuatuoGPT-o1-72B":       {"params_b": 72,  "family": "qwen",   "is_qwen3": False, "reported": True},
    "HuatuoGPT-o1-7B":        {"params_b": 7,   "family": "qwen",   "is_qwen3": False, "reported": True},
    "Llama-3.3-70B-Instruct": {"params_b": 70,  "family": "llama",  "is_qwen3": False, "reported": True},
    "Qwen2.5-72B-Instruct":   {"params_b": 72,  "family": "qwen",   "is_qwen3": False, "reported": True},
    "Qwen3-1.7B":             {"params_b": 1.7, "family": "qwen",   "is_qwen3": True,  "reported": True},
    "Qwen3-4B":               {"params_b": 4,   "family": "qwen",   "is_qwen3": True,  "reported": True},
    "Qwen3-8B":               {"params_b": 8,   "family": "qwen",   "is_qwen3": True,  "reported": True},
    "Qwen3-14B":              {"params_b": 14,  "family": "qwen",   "is_qwen3": True,  "reported": True},
    "Qwen3-32B":              {"params_b": 32,  "family": "qwen",   "is_qwen3": True,  "reported": True},
    "Qwen3-0.6B":             {"params_b": 0.6, "family": "qwen",   "is_qwen3": True,  "reported": False},
}
ALL_MODELS      = list(MODELS)
REPORTED_MODELS = [m for m, c in MODELS.items() if c["reported"]]

# The illustrative model retained in the manuscript's headline narrative and in
# the Figure 5 power panels. Not privileged anywhere in the pipeline.
ILLUSTRATIVE_MODEL = "medgemma-27b-text-it"
CONTRAST_MODEL     = "gemma-3-27b-it"      # matched-architecture control


def model_path(name: str) -> Path:
    """Where the pipeline loads this model from: directly under the model
    share (ILD_MODEL_SHARE)."""
    return BASE_MODEL_DIR / name


# ── Inference / decoding ───────────────────────────────────────────────────────
# One decoding configuration for every model x config. No per-model branch
# exists anywhere in the pipeline.
MAX_MODEL_LEN        = 128_000
TENSOR_PARALLEL_SIZE = 4          # fixed for every model; jobs refuse < 4 free GPUs
GPU_MEM_UTIL         = 0.85
DTYPE                = "bfloat16"
MIN_FREE_MIB         = 70_000

TEMPERATURE        = 0.0
SEED               = 42
GEN_MAX_TOKENS     = 1536
REPETITION_PENALTY = None         # never set
STOP_TOKENS        = ["<end_of_turn>", "</s>", "<|im_end|>", "<|endoftext|>"]
GUIDED_DECODING    = True         # constrained decoding; all models or none

# Generation is issued in chunks so results can be flushed incrementally and a
# killed job resumes without repeating work. Fixed for every model so chunk
# composition cannot differ across models.
CHUNK_SIZE = 16

# ── Token budget per pass ──────────────────────────────────────────────────────
# The budget must hold for EVERY model's tokenizer. Pagination costs each note
# at the MAXIMUM token count across all registry tokenizers, and the prefix
# likewise, so a batch that fits under this budget fits every model. There is no
# truncation path at inference time.
SYSTEM_PROMPT_RESERVE = 1_500     # system prompt + chat-template overhead
SAFETY_BUFFER         = 1_000     # budget must cover every registry tokenizer
SECTION_OVERHEAD      = 200       # notes-block section headers
MAX_TOKENS_PER_PASS   = MAX_MODEL_LEN - GEN_MAX_TOKENS - SYSTEM_PROMPT_RESERVE - SAFETY_BUFFER
# = 123_964 tokens of patient content per pass, under the worst-case tokenizer

# v1 asserts every rendered prompt fits with at least this much room to spare,
# measured with each model's real chat template.
CONTEXT_MARGIN_TOKENS = 512

# ── Tier priority ordering (lower = packed first) ─────────────────────────────
# Maps the ild_priority assigned by the upstream note classifier to the packing
# rank. Same order for notes and fused, by design.
TIER_PRIORITY = {
    1:  1,    # CT Chest             — gold-standard ILD imaging
    4:  2,    # Pathology/Biopsy     — histologic confirmation
    5:  3,    # PFT Report           — physiologic confirmation
    3:  4,    # Discharge Summary    — longitudinal clinical synthesis
    2:  5,    # Chest X-Ray          — lower-sensitivity imaging
    7:  6,    # Pulmonology Dept     — specialist encounters
    6:  7,    # Progress Note        — general clinical reasoning
    9:  8,    # H&P / Patient Note   — initial workup
    8:  9,    # EPIC Ambulatory      — outpatient touchpoints
    10: 10,   # Other                — residual
}

TIER_LABEL = {
    1: "CT Chest", 2: "Chest X-Ray", 3: "Discharge Summary",
    4: "Pathology/Biopsy", 5: "PFT Report", 6: "Progress Note",
    7: "Pulmonology Dept", 8: "EPIC Ambulatory",
    9: "H&P/Patient Note", 10: "Other",
}

# ── Labels ─────────────────────────────────────────────────────────────────────
# Closed three-way set. "Uncertain" does not exist as a label.
LABEL_PRIORITY = {
    "Probable ILD": 3,
    "Possible ILD": 2,
    "Not ILD":      1,
}
ALL_LABELS = list(LABEL_PRIORITY)

# Sentinel for an unrecoverable completion. Never imputed, never dropped;
# propagates into aggregation and metrics and is counted. v2 fails any cell that
# retains one.
SENTINEL_LABEL = "UNPARSEABLE"

POSITIVE_LABELS     = {"Probable ILD", "Possible ILD"}
GOLD_POSITIVE_CODES = {1, 2}   # 1 = Probable, 2 = Possible
GOLD_NEGATIVE_CODES = {0}      # 0 = Not ILD
GOLD_DROP_CODES     = {9}      # 9 = Insufficient evidence — excluded from core cohort
GOLD_LABEL_MAP      = {1: "Probable ILD", 2: "Possible ILD", 0: "Not ILD", 9: "Insufficient"}

# ── Shen published-rule columns ────────────────────────────────────────────────
# The xlsx carries 19 Algorithm_* columns. Algorithm_1 is NOT among them: it is
# the inclusion criterion (>=2 ILD codes 7-180 days apart), met by every patient
# in the cohort by construction, and is synthesized as a flag-everyone rule.
# It is labelled as synthesized in every output that reports it.
SHEN_MEASURED_ALGOS = [
    "Algorithm_1a", "Algorithm_1b", "Algorithm_1c", "Algorithm_1d",
    "Algorithm_2",  "Algorithm_2a", "Algorithm_2b", "Algorithm_2c", "Algorithm_2d",
    "Algorithm_3",  "Algorithm_3a", "Algorithm_3b", "Algorithm_3c", "Algorithm_3d",
    "Algorithm_4",  "Algorithm_4a", "Algorithm_4b", "Algorithm_4c", "Algorithm_4d",
]
SHEN_SYNTHESIZED_ALGO = "Algorithm_1"
SHEN_HEADLINE_ALGOS   = ["Algorithm_1", "Algorithm_4"]

# ── Statistics ─────────────────────────────────────────────────────────────────
# Wilson is the canonical interval everywhere, including figures. Bootstrap is
# still emitted for comparison but is never the reported interval.
CI_METHOD_CANONICAL = "wilson"
Z_975          = 1.959963984540054
BOOTSTRAP_N    = 1000
BOOTSTRAP_SEED = 42

# Copeland misclassification / power model constants.
OR_TRUE      = 3.0
P0_PREV      = 0.10
ALPHA        = 0.05
TARGET_POWER = 0.80
N_MC         = 2000
EPS          = 1e-9

# ── Note-tier ablation (finegrained) ───────────────────────────────────────────
# Tier 7 (Pulmonology Dept) has zero notes in the cohort and is skipped.
# `priority` is the PACKING rank from TIER_PRIORITY, not the classifier tier;
# variant slugs use the packing rank.
TIER_VARIANTS = [
    {"priority": 1, "tier_id": 1, "slug": "ct",        "desc": "CT Chest"},
    {"priority": 2, "tier_id": 4, "slug": "path",      "desc": "Pathology/Biopsy"},
    {"priority": 3, "tier_id": 5, "slug": "pft",       "desc": "PFT Report"},
    {"priority": 4, "tier_id": 3, "slug": "discharge", "desc": "Discharge Summary"},
    {"priority": 5, "tier_id": 2, "slug": "cxr",       "desc": "Chest X-Ray"},
    {"priority": 7, "tier_id": 6, "slug": "progress",  "desc": "Progress Note"},
]
TIER_ABLATION_CONFIGS = ["notes", "fused"]
