"""
stages/s1_build_cohort.py — CPU only.

Thin driver over shared/cohort/build_core_cohort.py. Kept as a stage so the
run order in the README is uniform.

    python stages/s1_build_cohort.py

The build itself asserts the 360 -> 317 -> 309 derivation and a complete
EMPI join against the reference-standard xlsx.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.cohort import build_core_cohort

if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Build the 309-patient core cohort (CPU only)."
    ).parse_args()
    build_core_cohort.main()
