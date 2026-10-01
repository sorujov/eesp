#!/bin/bash
# Part 1f (heavy-tailed errors): the standard chunk (Python, Octave, R) plus the Laplace mixture.
set -u
CODE=$(cd "$(dirname "$0")" && pwd); cd "$CODE"
export V4_STUDY=1f UNITS_CSV=units_v4f.csv
bash run_chunk.sh || exit 1
S=${UNIT_START:-0}; E=${UNIT_END:-300}; C=${CHUNK_ID:-0}
OUT=${RESULTS_DIR:-$CODE/../results_v4}; DATA=${TMPDIR:-/tmp}/lpf_${C}_$$; mkdir -p "$DATA"
env -u UNIT_START -u UNIT_END python simv4.py "$DATA" > /dev/null
UNIT_START=$S UNIT_END=$E Rscript r_laplace.R "$DATA" "$OUT/lp_${C}.csv" units_v4f.csv > "$OUT/lp_${C}.log" 2>&1
rm -rf "$DATA"
echo "UNITS_DONE=$((E-S))"
