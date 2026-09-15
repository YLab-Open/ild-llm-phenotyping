"""
gpu_select.py
=============
Dynamic GPU detection for vLLM jobs on shared clusters.
Scans nvidia-smi and returns CUDA_VISIBLE_DEVICES for the first N free GPUs.

Usage from a script:
    from shared.inference.gpu_select import pick_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = pick_gpus(n=4)

Usage from the shell:
    CUDA_VISIBLE_DEVICES=$(python -m shared.inference.gpu_select --n 4) python infer.py
"""
import subprocess
from typing import List

# 70 GB free required per GPU for medgemma-27b @ 128K context with tp=4.
# Empirical: weights ~14 GB/GPU + KV cache ~12 GB/GPU + buffer.
MIN_FREE_MIB = 70_000


def free_gpus(min_free_mib: int = MIN_FREE_MIB) -> List[int]:
    """Sorted list of GPU indices with ≥min_free_mib MiB free memory."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free",
         "--format=csv,nounits,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout
    free = []
    for line in out.strip().splitlines():
        idx, mem = (s.strip() for s in line.split(","))
        if int(mem) >= min_free_mib:
            free.append(int(idx))
    return sorted(free)


def pick_gpus(n: int = 4, min_free_mib: int = MIN_FREE_MIB) -> str:
    """CUDA_VISIBLE_DEVICES string for the first n free GPUs. Raises if not enough."""
    free = free_gpus(min_free_mib)
    if len(free) < n:
        raise RuntimeError(
            f"Only {len(free)} GPUs have ≥{min_free_mib} MiB free; need {n}. "
            f"Free GPUs: {free}. Free more memory or lower --n / --min-free-mib."
        )
    chosen = free[:n]
    return ",".join(str(i) for i in chosen)


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--min-free-mib", type=int, default=MIN_FREE_MIB)
    ap.add_argument("--list", action="store_true",
                    help="Print all free GPUs and exit (don't enforce n)")
    args = ap.parse_args()
    if args.list:
        print(free_gpus(args.min_free_mib))
        sys.exit(0)
    print(pick_gpus(args.n, args.min_free_mib))