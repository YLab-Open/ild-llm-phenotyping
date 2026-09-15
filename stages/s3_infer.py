"""
stages/s3_infer.py — THE ONLY GPU STAGE. One model x one configuration per call.

    python stages/s3_infer.py --model Qwen3-8B --config notes
    python stages/s3_infer.py --model Qwen3-8B --config notes --limit 5
    python stages/s3_infer.py --model medgemma-27b-text-it --config fused --tier 1

One call is one `screen` session. Kill it at any time (Ctrl-C, SIGTERM, or the
node going away) and re-run the IDENTICAL command: it resumes.

There is exactly one inference path: no per-model branch anywhere. Every cell
runs with temperature 0, seed 42, no repetition penalty, 1536 output tokens,
the 3-way closed label set, tensor_parallel_size 4 and a 128K context. The
ONLY thing that varies across cells is the model weights and the prompt file
for the configuration.

Guided (constrained) JSON decoding is the primary defence against malformed
output, on for every model or none. The parser's lower rungs remain as the net.

Results append incrementally, flushed per chunk. SIGINT/SIGTERM finish the
in-flight chunk, flush, write a resume marker and exit 0 -- a killed job never
leaves a half-written record.

Chunk boundaries are FIXED over the full serialized prompt list, not over the
outstanding work. vLLM's greedy decoding is deterministic only given identical
batch composition: batch size changes the floating-point reduction order in
the attention and GEMM kernels, which can flip a token at a near-tie. Chunks
are therefore indexed off `prompts`; a chunk is skipped only if EVERY batch in
it is already done, and a partially-complete chunk is regenerated whole.
Composition then depends only on the serialized order and CHUNK_SIZE, never on
interruption history, so a resumed cell is bit-identical to a clean one.

Outputs (results/<model>/<config>[_tier<N>]/):
  batch_predictions.jsonl   one record per (patient_id, batch_idx), appended
  run_manifest.json         provenance + parse_path census + resume history
  heartbeat.json            live progress, safe to `cat` from another shell
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import config as C
from shared import runtime as RT
from shared.parsing import (parse_completion, response_schema, sentinel_record,
                            vocabulary_report, PARSE_PATHS)

_STOP = {"requested": False, "signal": None}


def _handle_signal(signum, frame):
    """Cooperative shutdown: never interrupt a write."""
    _STOP["requested"] = True
    _STOP["signal"] = signal.Signals(signum).name
    print(f"\n[s3_infer] {signal.Signals(signum).name} received — finishing the "
          f"in-flight chunk, flushing, then exiting cleanly. Re-run the same "
          f"command to resume.", flush=True)


def cell_dir(model: str, config: str, tier: int | None) -> Path:
    name = config if tier is None else f"{config}_tier{tier}"
    return C.RESULTS_ROOT / model / name


def serialized_dir(config: str, tier: int | None) -> Path:
    name = config if tier is None else f"{config}_tier{tier}"
    return C.RESULTS_ROOT / "_serialized" / name


def load_done_keys(path: Path):
    """Existing (patient_id, batch_idx) keys. Tolerates a truncated final
    line from a hard kill: that record is treated as not-done and redone."""
    done, bad = set(), 0
    if not path.exists():
        return done, bad
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["patient_id"], int(r["batch_idx"])))
            except Exception:
                bad += 1
    return done, bad


def rewrite_without_partial(path: Path):
    """Drop any unparseable trailing line left by a hard kill."""
    if not path.exists():
        return 0
    lines = path.read_text().splitlines()
    good = []
    dropped = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            json.loads(line)
            good.append(line)
        except Exception:
            dropped += 1
    if dropped:
        path.write_text("\n".join(good) + ("\n" if good else ""))
    return dropped


def drop_keys(path: Path, keys: set) -> int:
    """Remove records whose (patient_id, batch_idx) is in `keys`.

    Used for the already-done members of a partially-complete chunk: that chunk
    is regenerated whole, so its stale records must go first or the file would
    carry duplicates."""
    if not path.exists() or not keys:
        return 0
    kept, dropped = [], 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                dropped += 1
                continue
            if (r["patient_id"], int(r["batch_idx"])) in keys:
                dropped += 1
            else:
                kept.append(line)
    path.write_text("\n".join(kept) + ("\n" if kept else ""))
    return dropped


def plan_chunks(prompts: list, chunk_size: int, done: set):
    """Partition the FULL prompt list at fixed index boundaries, then
    decide which chunks still need work.

    Returns (todo_chunks, stale_keys, n_resumed):
      todo_chunks  chunks with at least one outstanding batch, regenerated whole
      stale_keys   already-done batches inside those chunks, to be dropped first
      n_resumed    batches in fully-complete chunks, skipped entirely
    """
    all_chunks = [prompts[i:i + chunk_size]
                  for i in range(0, len(prompts), chunk_size)]
    todo_chunks, stale_keys, n_resumed = [], set(), 0
    for ch in all_chunks:
        keys = {(p["patient_id"], int(p["batch_idx"])) for p in ch}
        if keys <= done:
            n_resumed += len(ch)
            continue
        todo_chunks.append(ch)
        stale_keys |= (keys & done)
    return todo_chunks, stale_keys, n_resumed


def main(model: str, config: str, tier: int | None, limit: int | None,
         chunk_size: int, dry_run: bool):
    od = cell_dir(model, config, tier)
    od.mkdir(parents=True, exist_ok=True)
    log = RT.setup_logging("S3_INFER", C.LOGS_DIR / f"s3_{model}_{od.name}.log")

    batch_jsonl = od / "batch_predictions.jsonl"
    manifest_p = od / "run_manifest.json"
    heartbeat_p = od / "heartbeat.json"

    sd = serialized_dir(config, tier)
    prompts_p = sd / "serialized_prompts.jsonl"
    if not prompts_p.exists():
        log.error(f"Missing {prompts_p} — run s2_serialize (and v1) first.")
        sys.exit(2)

    prompts = [json.loads(l) for l in open(prompts_p) if l.strip()]
    if limit:
        keep = {p["patient_id"] for p in prompts[:limit * 8]}
        keep = set(sorted(keep)[:limit])
        prompts = [p for p in prompts if p["patient_id"] in keep]
        log.info(f"  --limit {limit}: {len(prompts)} batches over {len(keep)} patients")

    sys_prompt, user_template, prompt_sha = RT.load_prompt(config)
    schema = response_schema(config)

    log.info("=" * 78)
    log.info(f"s3_infer  model={model}  config={config}"
             + (f"  tier={tier}" if tier is not None else ""))
    log.info(f"  prompt {config}.yaml sha256={prompt_sha[:16]}  batches={len(prompts)}")
    log.info(f"  decoding: temp={C.TEMPERATURE} seed={C.SEED} "
             f"max_tokens={C.GEN_MAX_TOKENS} repetition_penalty={C.REPETITION_PENALTY} "
             f"guided={C.GUIDED_DECODING} tp={C.TENSOR_PARALLEL_SIZE}")
    log.info(f"  chunk_size={chunk_size} alignment=fixed_index")
    log.info("=" * 78)
    log.info(vocabulary_report())

    dropped = rewrite_without_partial(batch_jsonl)
    if dropped:
        log.warning(f"  dropped {dropped} truncated trailing line(s) from a previous kill")

    done, _ = load_done_keys(batch_jsonl)
    todo_chunks, stale_keys, resumed = plan_chunks(prompts, chunk_size, done)
    n_todo = sum(len(ch) for ch in todo_chunks)
    if resumed:
        log.info(f"  RESUMING: {resumed}/{len(prompts)} batches in complete chunks, "
                 f"{n_todo} to do across {len(todo_chunks)} chunk(s)")
    if stale_keys:
        log.info(f"  {len(stale_keys)} batch(es) sit in partially-complete "
                 f"chunks and will be REGENERATED so chunk composition matches a "
                 f"clean run")
    if not todo_chunks:
        log.info("  nothing to do — cell already complete. Run v2 to verify.")
        return

    if dry_run:
        log.info(f"  --dry-run: would generate {n_todo} batches in "
                 f"{len(todo_chunks)} chunk(s). No GPU touched.")
        return

    n_stale_dropped = drop_keys(batch_jsonl, stale_keys)
    if n_stale_dropped:
        log.info(f"  dropped {n_stale_dropped} stale record(s) before regeneration")

    # ── GPU ───────────────────────────────────────────────────────────────────
    from shared.inference.gpu_select import pick_gpus
    gpus = pick_gpus(n=C.TENSOR_PARALLEL_SIZE, min_free_mib=C.MIN_FREE_MIB)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    log.info(f"CUDA_VISIBLE_DEVICES = {gpus}")

    tok = RT.load_tokenizer(model)
    is_qwen3 = C.MODELS[model]["is_qwen3"]

    from vllm import LLM, SamplingParams
    so_config = None
    if C.GUIDED_DECODING:
        try:
            from vllm.config.structured_outputs import StructuredOutputsConfig
        except ImportError:
            from vllm.config import StructuredOutputsConfig
        cfg_kwargs = {"backend": "xgrammar", "disable_any_whitespace": True}
        try:
            so_config = StructuredOutputsConfig(**cfg_kwargs, disable_fallback=True)
            cfg_kwargs["disable_fallback"] = True
        except Exception:          # pydantic ValidationError on older vLLM
            so_config = StructuredOutputsConfig(**cfg_kwargs)
        log.info(f"  structured outputs engine config: {cfg_kwargs}")

    t_load = time.time()
    llm = LLM(model=str(C.model_path(model)),
              tensor_parallel_size=C.TENSOR_PARALLEL_SIZE,
              max_model_len=C.MAX_MODEL_LEN,
              gpu_memory_utilization=C.GPU_MEM_UTIL,
              dtype=C.DTYPE, trust_remote_code=True, seed=C.SEED,
              structured_outputs_config=so_config)
    load_min = (time.time() - t_load) / 60
    log.info(f"  model loaded in {load_min:.1f} min")

    sp_kwargs = dict(temperature=C.TEMPERATURE, max_tokens=C.GEN_MAX_TOKENS,
                     seed=C.SEED, stop=C.STOP_TOKENS)
    if C.REPETITION_PENALTY is not None:      # never, by config; guard kept explicit
        sp_kwargs["repetition_penalty"] = C.REPETITION_PENALTY
    guided_mode = "off"
    if C.GUIDED_DECODING:
        try:                                   # vLLM >= 0.19 structured outputs
            from vllm.sampling_params import StructuredOutputsParams
            sp_kwargs["structured_outputs"] = StructuredOutputsParams(
                json=schema, disable_any_whitespace=True)
            guided_mode = "structured_outputs"
        except Exception as e:
            # The older GuidedDecodingParams API has no disable_any_whitespace,
            # so a cell run through it would permit an unbounded inter-token
            # whitespace loop under greedy decoding. Mixing the two APIs would
            # be a per-model decoding asymmetry. Refuse rather than fall back.
            log.error(f"  StructuredOutputsParams unavailable ({e!r}). "
                      f"disable_any_whitespace is required and the older "
                      f"GuidedDecodingParams API does not support it — refusing "
                      f"this cell. Install a vLLM build that exposes it.")
            sys.exit(3)
    sampling = SamplingParams(**sp_kwargs)
    log.info(f"  guided decoding: {guided_mode}")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    census = {p: 0 for p in PARSE_PATHS}
    n_len_capped = 0
    t0 = time.time()
    n_done_this_run = 0

    with open(batch_jsonl, "a") as fout:
        for chunk in todo_chunks:
            if _STOP["requested"]:
                break
            chats = [RT.chat_wrap(tok, sys_prompt, user_template, p["prompt"], is_qwen3)
                     for p in chunk]
            outs = llm.generate(chats, sampling)

            for p, out in zip(chunk, outs):
                o = out.outputs[0]
                raw = o.text
                finish = getattr(o, "finish_reason", None)
                if finish == "length":
                    n_len_capped += 1
                rec = parse_completion(raw, config, guided=(guided_mode != "off"))
                census[rec["parse_path"]] += 1
                fout.write(json.dumps({
                    "patient_id": p["patient_id"], "empi": str(p.get("empi", "")),
                    "batch_idx": int(p["batch_idx"]), "n_batches": int(p["n_batches"]),
                    "cancer_group": p["cancer_group"],
                    "final_diagnosis": p["final_diagnosis"],
                    "gold_label": p["gold_label"],
                    "model": model, "config": config, "tier": tier,
                    "prompt_sha256": prompt_sha,
                    **{k: rec[k] for k in rec if not k.startswith("_")},
                    "_raw_label": rec.get("_raw_label", ""),
                    "finish_reason": finish,
                    "n_output_tokens": len(getattr(o, "token_ids", []) or []),
                    # FULL raw completion, no character cap: the raw text is
                    # the evidence needed to diagnose any parse failure.
                    "raw_response": raw,
                }) + "\n")
            fout.flush()
            os.fsync(fout.fileno())            # durable before we proceed
            n_done_this_run += len(chunk)

            el = (time.time() - t0) / 60
            rate = n_done_this_run / max(el, 1e-9)
            remaining = (n_todo - n_done_this_run) / max(rate, 1e-9)
            hb = {"model": model, "config": od.name,
                  "batches_total": len(prompts),
                  "batches_done": resumed + n_done_this_run,
                  "batches_remaining": n_todo - n_done_this_run,
                  "elapsed_min": round(el, 1),
                  "projected_remaining_min": round(remaining, 1),
                  "rate_batches_per_min": round(rate, 2),
                  "parse_paths": dict(census),
                  "updated": datetime.now(timezone.utc).isoformat()}
            heartbeat_p.write_text(json.dumps(hb, indent=2))
            log.info(f"  [{resumed + n_done_this_run}/{len(prompts)}] "
                     f"{el:.1f} min elapsed, ~{remaining:.1f} min remaining, "
                     f"paths={ {k: v for k, v in census.items() if v} }")

    interrupted = _STOP["requested"]
    manifest = {
        "model": model, "params_b": C.MODELS[model]["params_b"],
        "config": config, "tier": tier,
        "git_commit": RT.git_commit(),
        "prompt_file": f"{config}.yaml", "prompt_sha256": prompt_sha,
        "tokenizer": str(C.model_path(model)),
        "vllm_version": RT.vllm_version(),
        "decoding": {"temperature": C.TEMPERATURE, "seed": C.SEED,
                     "max_tokens": C.GEN_MAX_TOKENS,
                     "repetition_penalty": C.REPETITION_PENALTY,
                     "stop": C.STOP_TOKENS, "guided_decoding": guided_mode,
                     "chunk_size": chunk_size,
                     "chunk_alignment": "fixed_index",
                     "disable_any_whitespace": True,
                     "structured_outputs_backend": "xgrammar"},
        "tensor_parallel_size": C.TENSOR_PARALLEL_SIZE,
        "max_model_len": C.MAX_MODEL_LEN,
        "gpu_indices": gpus, "dtype": C.DTYPE,
        "gpu_name": __import__("subprocess").run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip().splitlines()[0],
        "hostname": __import__("socket").gethostname(),
        "gpu_memory_utilization": C.GPU_MEM_UTIL,
        "batches_expected": len(prompts),
        "batches_generated_this_run": n_done_this_run,
        "batches_resumed": resumed,
        "batches_regenerated_for_chunk_alignment": n_stale_dropped,
        "n_generations_hitting_token_cap": n_len_capped,
        "parse_path_census_this_run": dict(census),
        "model_load_min": round(load_min, 1),
        "run_minutes": round((time.time() - t0) / 60, 1),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "complete": (not interrupted) and (resumed + n_done_this_run == len(prompts)),
        "interrupted_by": _STOP["signal"],
    }
    prev = []
    if manifest_p.exists():
        try:
            old = json.loads(manifest_p.read_text())
            prev = old.get("previous_runs", []) + [{k: old.get(k) for k in
                     ("finished_utc", "batches_generated_this_run", "interrupted_by",
                      "run_minutes", "git_commit")}]
        except Exception:
            pass
    manifest["previous_runs"] = prev
    manifest["resume_count"] = len(prev)
    manifest_p.write_text(json.dumps(manifest, indent=2))

    if n_len_capped:
        log.warning(f"  {n_len_capped} generation(s) hit the {C.GEN_MAX_TOKENS}-token "
                    f"cap (finish_reason='length'). Inspect raw_response.")
    if interrupted:
        log.info(f"  stopped cleanly on {_STOP['signal']}. "
                 f"{resumed + n_done_this_run}/{len(prompts)} done. "
                 f"Re-run the identical command to resume.")
        sys.exit(0)
    log.info(f"  CELL COMPLETE — {len(prompts)} batches, "
             f"{manifest['run_minutes']:.1f} min. parse paths: {census}")
    log.info(f"\nNext: python validate/v2_inference.py --model {model} "
             f"--config {config}" + (f" --tier {tier}" if tier is not None else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=C.ALL_MODELS)
    ap.add_argument("--config", required=True, choices=C.CONFIGS)
    ap.add_argument("--tier", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke test on N patients")
    ap.add_argument("--chunk-size", type=int, default=C.CHUNK_SIZE)
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve everything and report the work, touch no GPU")
    a = ap.parse_args()
    main(a.model, a.config, a.tier, a.limit, a.chunk_size, a.dry_run)