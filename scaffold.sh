#!/usr/bin/env bash
# scaffold.sh — one-time setup: link the shared inputs into shared/data/.
#   bash scaffold.sh
#
# Idempotent. Reads cohort inputs from shared storage and writes nothing
# outside this repository.
#
# Overrides:
#   ILD_ROOT   the repository checkout (defaults to the current directory)
set -euo pipefail

ROOT="${ILD_ROOT:-.}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$HERE/shared/data"

mkdir -p "$DATA" "$HERE/results" "$HERE/logs"

fail () { echo "" >&2; echo "SCAFFOLD FAILED: $*" >&2; exit 1; }

link () {  # link <target> <name>
  [ -e "$1" ] || fail "missing source: $1
  Expected under ILD_ROOT=$ROOT. Set ILD_ROOT if the project lives elsewhere."
  ln -sfn "$1" "$DATA/$2"; echo "  $2 -> $1"
}

echo "Linking shared inputs into $DATA"
link "$ROOT/shared/data/patient_windows"             patient_windows
link "$ROOT/shared/data/window_stats.csv"            window_stats.csv
link "$ROOT/shared/data/patient_drug_exposure.csv"   patient_drug_exposure.csv
link "$ROOT/shared/data/ehr_windowed_typed.parquet"  ehr_windowed_typed.parquet
link "$ROOT/ILD_01_EMPI_Final_Diagnosis_360.xlsx"    ILD_01_EMPI_Final_Diagnosis_360.xlsx

echo ""
echo "Scaffold complete."
echo "Next:"
echo "  python stages/s1_build_cohort.py"
echo "  python stages/s2_serialize.py --config claims"
echo "  python validate/v1_serialization.py --config claims"
