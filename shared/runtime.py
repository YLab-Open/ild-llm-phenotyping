"""
Small runtime helpers shared by the stages: prompt loading with content
hashing, tokenizer loading, git provenance and logging setup.

Prompt YAMLs are content-hashed at load and the hash is written into every
output record, so a changed prompt is visible in the results rather than
inferred from file mtimes.
"""
from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import config as C


# ── Prompts ───────────────────────────────────────────────────────────────────
def prompt_path(config: str) -> Path:
    return C.PROMPTS_DIR / f"{config}.yaml"


def load_prompt(config: str) -> Tuple[str, str, str]:
    """Returns (system_prompt, user_template, sha256_of_file)."""
    import re
    import yaml
    p = prompt_path(config)
    raw = p.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    y = yaml.safe_load(raw.decode())
    sysp = y["SYSTEM_PROMPT"].strip()
    usert = y["USER_TEMPLATE"].strip()
    body = sysp + usert
    # The closed label set is enforced structurally by the guided schema; this
    # is a cheap guard against a prompt being edited back to a 4-way set.
    #
    # The notes/fused prompts EXPLICITLY FORBID the fourth label, in the
    # sentence '"Uncertain" is NOT an allowed answer.' The guard must not fire
    # on that sentence: only an OFFER of "Uncertain" is rejected.
    probe = re.sub(r'"?Uncertain"?\s+is\s+NOT\s+an\s+allowed\s+answer',
                   "", body, flags=re.IGNORECASE)
    if "uncertain" in probe.lower():
        raise ValueError(f"{p} offers 'Uncertain'; the label set is closed to three.")
    return sysp, usert, digest


def prompt_hashes() -> Dict[str, str]:
    out = {}
    for cfg in C.CONFIGS:
        p = prompt_path(cfg)
        if p.exists():
            out[cfg] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ── Tokenizers ────────────────────────────────────────────────────────────────
_TOKENIZER_CACHE: Dict[str, object] = {}


def load_tokenizer(model: str):
    if model not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer
        _TOKENIZER_CACHE[model] = AutoTokenizer.from_pretrained(
            str(C.model_path(model)), trust_remote_code=True)
    return _TOKENIZER_CACHE[model]


def load_all_tokenizers(models=None, required: bool = True) -> Dict[str, object]:
    """Every registry tokenizer, for worst-case token costing.

    `required=True` (serialization) raises if any tokenizer is unavailable: a
    missing tokenizer would silently narrow the worst case and permit
    per-model context overflow.
    """
    models = models or C.ALL_MODELS
    toks, missing = {}, []
    for m in models:
        try:
            toks[m] = load_tokenizer(m)
        except Exception as e:                       # pragma: no cover
            missing.append((m, repr(e)[:200]))
    if missing and required:
        raise RuntimeError(
            "Tokenizers unavailable for: " + ", ".join(m for m, _ in missing) +
            ". Serialization must cost every model or the worst-case budget is wrong."
        )
    return toks


def chat_wrap(tokenizer, sys_prompt: str, user_template: str,
              body: str, is_qwen3: bool) -> str:
    """Render one full chat-templated prompt exactly as s3_infer will."""
    msg = sys_prompt + "\n\n" + user_template.format(patient_summary=body)
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if is_qwen3:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template([{"role": "user", "content": msg}], **kwargs)


# ── Provenance ────────────────────────────────────────────────────────────────
def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              cwd=str(C.CLEAN_ROOT), capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def vllm_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "unknown")
    except Exception:
        return "not-installed"


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(name: str, logfile: Path | None = None) -> logging.Logger:
    C.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout)]
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, mode="a"))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        handlers=handlers, force=True)
    return logging.getLogger(name)
