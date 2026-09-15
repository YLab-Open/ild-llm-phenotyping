"""
Aggregation + evaluation, shared by every configuration: patient-level
aggregation of per-batch predictions, Wilson intervals, bootstrap CIs,
AUROC, discrete ECE, McNemar's exact test and the Copeland power model.

Conventions:
  * aggregate_batches: label from the highest-positivity batch, and the
    probability from THAT SAME batch, so label and probability can never
    disagree by construction. The across-batch maximum is still emitted as
    ild_probability_max.
  * Wilson is canonical. Bootstrap is still computed for comparison but is
    never the reported interval.
  * Sentinels propagate: a sentinel batch can never win aggregation, and a
    patient whose batches are ALL sentinels becomes a sentinel patient,
    counted and never imputed.
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared import config as C

try:
    from scipy.stats import rankdata, norm, binomtest
    HAVE_SCIPY = True
except Exception:                                     # pragma: no cover
    HAVE_SCIPY = False

SENTINEL = C.SENTINEL_LABEL


# ══ Aggregation ═══════════════════════════════════════════════════════════════
def _positivity(label: str) -> int:
    """Positivity rank. The sentinel sorts BELOW every real label so it can only
    win when every batch is a sentinel."""
    return C.LABEL_PRIORITY.get(label, -1)


def aggregate_batches(batch_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Patient-level rollup from per-batch records.

    Rule: the patient label is the highest-positivity batch label under
    C.LABEL_PRIORITY, ties broken by that batch's probability.

    The reported ild_probability is the WINNING batch's own probability.
    ild_probability_max carries the across-batch maximum.
    """
    if not batch_predictions:
        return {
            "predicted_label": SENTINEL, "ild_probability": None,
            "ild_probability_max": None, "confidence_score": None,
            "reasoning": "", "key_evidence": [], "modality_concordance": "",
            "n_batches": 0, "n_sentinel_batches": 0, "all_batches_sentinel": True,
            "per_batch_labels": [], "per_batch_probabilities": [],
            "per_batch_parse_paths": [],
        }

    ordered = sorted(batch_predictions, key=lambda b: b.get("batch_idx", 0))
    real = [b for b in ordered if b.get("predicted_label") != SENTINEL]
    n_sent = len(ordered) - len(real)

    if not real:
        # Every batch failed: explicit sentinel, never imputed.
        return {
            "predicted_label": SENTINEL, "ild_probability": None,
            "ild_probability_max": None, "confidence_score": None,
            "reasoning": "", "key_evidence": [], "modality_concordance": "",
            "n_batches": len(ordered), "n_sentinel_batches": n_sent,
            "all_batches_sentinel": True,
            "per_batch_labels": [b.get("predicted_label") for b in ordered],
            "per_batch_probabilities": [b.get("ild_probability") for b in ordered],
            "per_batch_parse_paths": [b.get("parse_path") for b in ordered],
        }

    def key(b):
        return (_positivity(b["predicted_label"]), b.get("ild_probability") or 0.0)

    best = max(real, key=key)
    probs = [b.get("ild_probability") for b in real if b.get("ild_probability") is not None]

    return {
        "predicted_label":      best["predicted_label"],
        "ild_probability":      best.get("ild_probability"),          # winner's own
        "ild_probability_max":  (max(probs) if probs else None),      # across-batch max
        "confidence_score":     best.get("confidence_score"),
        "reasoning":            best.get("reasoning", ""),
        "key_evidence":         best.get("key_evidence", []),
        "modality_concordance": best.get("modality_concordance", ""),
        "n_batches":            len(ordered),
        "n_sentinel_batches":   n_sent,
        "all_batches_sentinel": False,
        "winning_batch_idx":    best.get("batch_idx"),
        # retained for EVERY model
        "per_batch_labels":        [b.get("predicted_label") for b in ordered],
        "per_batch_probabilities": [b.get("ild_probability") for b in ordered],
        "per_batch_parse_paths":   [b.get("parse_path") for b in ordered],
    }


# ── Alternative aggregation rules ──────────────────────────────────────────────
def _lab(b):  return b.get("predicted_label", SENTINEL)
def _prob(b): return b.get("ild_probability") or 0.0
def _conf(b): return b.get("confidence_score") or 0.0


def agg_or_logic(batches):
    real = [b for b in batches if _lab(b) != SENTINEL]
    if not real:
        return SENTINEL
    return _lab(max(real, key=lambda b: (_positivity(_lab(b)), _prob(b))))


def agg_first_batch(batches):
    real = [b for b in batches if _lab(b) != SENTINEL]
    if not real:
        return SENTINEL
    return _lab(min(real, key=lambda b: b.get("batch_idx", 0)))


def agg_majority(batches):
    real = [b for b in batches if _lab(b) != SENTINEL]
    if not real:
        return SENTINEL
    counts = Counter(_lab(b) for b in real)
    top = max(counts.values())
    tied = [l for l, c in counts.items() if c == top]
    return max(tied, key=_positivity)


def agg_max_confidence(batches):
    real = [b for b in batches if _lab(b) != SENTINEL]
    if not real:
        return SENTINEL
    return _lab(max(real, key=_conf))


AGGREGATION_RULES = {
    "or_logic":       agg_or_logic,      # deployed default
    "first_batch":    agg_first_batch,
    "majority":       agg_majority,
    "max_confidence": agg_max_confidence,
}


# ══ Interval estimation ═══════════════════════════════════════════════════════
def wilson(k: int, n: int, z: float = C.Z_975):
    """Wilson score interval. Canonical everywhere."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _binary_metrics_one(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = tp + fp + tn + fn
    safe = lambda a, b: a / b if b > 0 else 0.0
    ppv = safe(tp, tp + fp)
    npv = safe(tn, tn + fn)
    sens = safe(tp, tp + fn)
    spec = safe(tn, tn + fp)
    f1 = safe(2 * ppv * sens, ppv + sens)
    acc = safe(tp + tn, n)
    return {"n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "ppv": ppv, "npv": npv, "sensitivity": sens,
            "specificity": spec, "f1": f1, "accuracy": acc}


def bootstrap_ci(y_true, y_pred, n_iter=C.BOOTSTRAP_N, alpha=0.05, seed=C.BOOTSTRAP_SEED):
    """Seeded percentile bootstrap, kept for exact reproducibility of the
    published rule metrics. NOT the reported interval: it degenerates to
    [1.000, 1.000] at PPV = 1."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = ["ppv", "npv", "sensitivity", "specificity", "f1", "accuracy"]
    samples = {k: [] for k in keys}
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        m = _binary_metrics_one(y_true[idx], y_pred[idx])
        for k in keys:
            samples[k].append(m[k])
    point = _binary_metrics_one(y_true, y_pred)
    lo_p, hi_p = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {k: {"point": point[k],
               "lo": float(np.percentile(v, lo_p)),
               "hi": float(np.percentile(v, hi_p))}
           for k, v in samples.items()}
    for k in ["n", "tp", "fp", "tn", "fn"]:
        out[k] = int(point[k])
    return out


def binary_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, Any]:
    """Confusion cells -> full metric block with Wilson intervals.

    The counts are named n_true_positive / n_false_positive /
    n_true_negative / n_false_negative and no field is named `tp`, which
    would be ambiguous with tensor-parallel size. Yield is first-class,
    not derived downstream.
    """
    n = tp + fp + tn + fn
    flagged = tp + fp
    safe = lambda a, b: (a / b if b > 0 else float("nan"))
    ppv, sens = safe(tp, flagged), safe(tp, tp + fn)
    spec, npv = safe(tn, tn + fp), safe(tn, tn + fn)
    f1 = (2 * ppv * sens / (ppv + sens)) if (ppv == ppv and sens == sens and (ppv + sens) > 0) else float("nan")
    out = {
        "n_patients": n, "n_flagged": flagged,
        "n_true_positive": tp, "n_false_positive": fp,
        "n_true_negative": tn, "n_false_negative": fn,
        "ppv": ppv, "sensitivity": sens, "specificity": spec,
        "npv": npv, "f1": f1, "accuracy": safe(tp + tn, n),
        "yield": safe(flagged, n),
    }
    for name, k, d in (("ppv", tp, flagged), ("sensitivity", tp, tp + fn),
                       ("specificity", tn, tn + fp), ("npv", tn, tn + fn),
                       ("yield", flagged, n)):
        lo, hi = wilson(k, d)
        out[f"{name}_wilson_lo"], out[f"{name}_wilson_hi"] = lo, hi
    return out


def confusion_from_frame(df: pd.DataFrame) -> Dict[str, int]:
    """Confusion cells from a patient-level frame. Sentinels are NOT dropped:
    a sentinel is not in POSITIVE_LABELS, so it counts as a negative call, and
    v3 refuses any cell that still contains one."""
    gold = df["gold_label"].isin(C.POSITIVE_LABELS)
    pred = df["predicted_label"].isin(C.POSITIVE_LABELS)
    return {"tp": int((gold & pred).sum()), "fp": int((~gold & pred).sum()),
            "tn": int((~gold & ~pred).sum()), "fn": int((gold & ~pred).sum())}


# ══ Discrimination and calibration ════════════════════════════════════════════
def auroc(scores, labels) -> float:
    """AUROC via the rank (Mann-Whitney) statistic with tie correction."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    ok = ~np.isnan(scores)
    scores, labels = scores[ok], labels[ok]
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    if HAVE_SCIPY:
        r = rankdata(scores)
    else:
        order = np.argsort(scores, kind="mergesort")
        sr = scores[order]
        ranks = np.empty(len(sr), float)
        i = 0
        while i < len(sr):
            j = i
            while j + 1 < len(sr) and sr[j + 1] == sr[i]:
                j += 1
            ranks[i:j + 1] = (i + j) / 2.0 + 1
            i = j + 1
        r = np.empty(len(scores), float)
        r[order] = ranks
    return float((r[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def ece_discrete(prob, labels, valid_mask=None):
    """ECE with one bin per DISTINCT emitted probability value: fixed-width
    bins would merge adjacent emitted values (e.g. 0.9 and 0.95).
    Returns (ece, n_used, table[(value, count, observed_rate)])."""
    prob = np.asarray(prob, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if valid_mask is not None:
        m = np.asarray(valid_mask, dtype=bool)
        prob, labels = prob[m], labels[m]
    ok = ~np.isnan(prob)
    prob, labels = prob[ok], labels[ok]
    N = len(prob)
    if N == 0:
        return float("nan"), 0, []
    e, table = 0.0, []
    for v in np.unique(prob):
        sel = prob == v
        obs = float(labels[sel].mean())
        e += sel.sum() / N * abs(obs - v)
        table.append((float(v), int(sel.sum()), obs))
    return e, N, table


def mcnemar_exact(b: int, c: int):
    """(net, Wald half-width, two-sided exact p)."""
    b, c = int(b), int(c)
    net, nn = b - c, b + c
    if nn == 0:
        return net, float("nan"), float("nan")
    z = C.Z_975
    half = z * 2.0 * math.sqrt(b * c / nn) if (b * c) else z * math.sqrt(nn)
    if HAVE_SCIPY:
        p = float(binomtest(min(b, c), nn, p=0.5, alternative="two-sided").pvalue)
    else:
        k = min(b, c)
        p = min(1.0, 2.0 * sum(math.comb(nn, i) for i in range(0, k + 1)) * (0.5 ** nn))
    return net, half, p


# ══ Copeland misclassification / power model ══════════════════════════════════
# Constants as set in config.py; parameterised over any classifier's
# confusion cells.
def _logor_V(se, sp, p0=C.P0_PREV, true_or=C.OR_TRUE):
    """(log observed OR, per-group Wald variance unit V) under non-differential
    OUTCOME misclassification with sensitivity se and specificity sp."""
    se = np.asarray(se, float); sp = np.asarray(sp, float)
    p1 = true_or * p0 / (1 - p0 + true_or * p0)
    p0s = np.clip(se * p0 + (1 - sp) * (1 - p0), C.EPS, 1 - C.EPS)
    p1s = np.clip(se * p1 + (1 - sp) * (1 - p1), C.EPS, 1 - C.EPS)
    log_or = np.log((p1s / (1 - p1s)) / (p0s / (1 - p0s)))
    V = 1.0 / (p0s * (1 - p0s)) + 1.0 / (p1s * (1 - p1s))
    return log_or, V


def obs_or_pstar(se, sp, p0=C.P0_PREV, true_or=C.OR_TRUE) -> float:
    """Observed OR under non-differential misclassification (Copeland 1977)."""
    log_or, _ = _logor_V(se, sp, p0, true_or)
    return float(np.exp(np.asarray(log_or)))


def attenuation_pct(or_true, or_observed) -> float:
    if or_true == 1:
        return float("nan")
    return 100.0 * (or_true - or_observed) / (or_true - 1.0)


def required_N(se, sp, p0=C.P0_PREV, true_or=C.OR_TRUE):
    """Total N (cases + controls, 1:1) for TARGET_POWER at level ALPHA."""
    if not HAVE_SCIPY:
        return np.nan
    z_a = norm.ppf(1 - C.ALPHA / 2.0)
    z_b = norm.ppf(C.TARGET_POWER)
    log_or, V = _logor_V(se, sp, p0, true_or)
    log_or_abs = np.maximum(np.abs(log_or), C.EPS)
    return 2.0 * V * ((z_a + z_b) / log_or_abs) ** 2


def power_at_N(se, sp, N, p0=C.P0_PREV, true_or=C.OR_TRUE):
    if not HAVE_SCIPY:
        return np.nan
    z_a = norm.ppf(1 - C.ALPHA / 2.0)
    log_or, V = _logor_V(se, sp, p0, true_or)
    n_per = np.asarray(N, float) / 2.0
    return norm.cdf(np.abs(log_or) * np.sqrt(n_per / V) - z_a)


def mc_power(tp, fn, tn, fp, p0=C.P0_PREV, true_or=C.OR_TRUE,
             n=C.N_MC, seed=C.BOOTSTRAP_SEED) -> Dict[str, Any]:
    """Point estimate + Jeffreys-Beta Monte-Carlo CI for required N.

    A flag-all (or flag-none) classifier has specificity 0, the misclassified
    log-OR collapses to 0 and required_N diverges against the EPS floor. That is
    reported as required_N_estimable=False, never as a number.
    """
    rng = np.random.default_rng(seed)
    se_s = rng.beta(tp + 0.5, fn + 0.5, n)
    sp_s = rng.beta(tn + 0.5, fp + 0.5, n)
    N_s = required_N(se_s, sp_s, p0, true_or)
    se_pt = tp / (tp + fn) if (tp + fn) else 0.0
    sp_pt = tn / (tn + fp) if (tn + fp) else 0.0
    N_pt = float(required_N(np.array([se_pt]), np.array([sp_pt]), p0, true_or)[0]) \
        if HAVE_SCIPY else float("nan")
    obs_or = float(np.exp(_logor_V(np.array([se_pt]), np.array([sp_pt]), p0, true_or)[0][0]))
    pw2000 = float(power_at_N(se_pt, sp_pt, 2000, p0, true_or)) if HAVE_SCIPY else float("nan")
    estimable = bool(np.isfinite(N_pt) and N_pt < 1e9)
    return {
        "se_pt": se_pt, "sp_pt": sp_pt, "obs_or": obs_or,
        "required_N_estimable": estimable,
        "N_pt": N_pt if estimable else float("nan"),
        "N_lo": float(np.nanpercentile(N_s, 2.5)) if (HAVE_SCIPY and estimable) else float("nan"),
        "N_hi": float(np.nanpercentile(N_s, 97.5)) if (HAVE_SCIPY and estimable) else float("nan"),
        "power2000": pw2000,
        "_N_pt_raw": N_pt,
    }


# ══ Reporting ═════════════════════════════════════════════════════════════════
def fmt_ci(point, lo, hi) -> str:
    return f"{point*100:5.1f}% ({lo*100:5.1f}-{hi*100:5.1f}%)"


def print_report(m: Dict[str, Any], title: str = "Evaluation") -> str:
    lines = [
        "=" * 68,
        f"{title}  |  n = {m['n_patients']}",
        "=" * 68,
        f"  Counts (TP/FP/TN/FN) : {m['n_true_positive']}/{m['n_false_positive']}/"
        f"{m['n_true_negative']}/{m['n_false_negative']}",
        f"  Yield               : {fmt_ci(m['yield'], m['yield_wilson_lo'], m['yield_wilson_hi'])}",
        f"  PPV                 : {fmt_ci(m['ppv'], m['ppv_wilson_lo'], m['ppv_wilson_hi'])}",
        f"  Sensitivity         : {fmt_ci(m['sensitivity'], m['sensitivity_wilson_lo'], m['sensitivity_wilson_hi'])}",
        f"  Specificity         : {fmt_ci(m['specificity'], m['specificity_wilson_lo'], m['specificity_wilson_hi'])}",
        f"  NPV                 : {fmt_ci(m['npv'], m['npv_wilson_lo'], m['npv_wilson_hi'])}",
        f"  F1                  : {m['f1']*100:5.1f}%",
        f"  Accuracy            : {m['accuracy']*100:5.1f}%",
        "  (intervals are Wilson score — canonical throughout the pipeline)",
        "=" * 68,
    ]
    s = "\n".join(lines)
    print(s)
    return s


def json_safe(o):
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)):    return bool(o)
    if isinstance(o, dict):  return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [json_safe(v) for v in o]
    return o
