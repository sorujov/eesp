#!/bin/bash
# Laplace mixture, 100 starts, on the four-group design E4 (part 1c): one Rscript per unit,
# WORKERS at a time, over units [UNIT_START, UNIT_END) of the E4 list; one CSV per unit.
set -u
CODE=$(cd "$(dirname "$0")" && pwd); cd "$CODE"
W=${WORKERS:-1}; A=${UNIT_START:-0}; B=${UNIT_END:-200}
OUT=${RESULTS_DIR:-$CODE/results}/lp100_1c; mkdir -p "$OUT"
DATA=${TMPDIR:-/tmp}/lp100e4_$$; mkdir -p "$DATA"
U=$DATA/units.csv; head -1 units_v4c.csv > "$U"; grep -E '^E4-' units_v4c.csv >> "$U"
env -u UNIT_START -u UNIT_END V4_STUDY=1c python simv4.py "$DATA" > /dev/null || exit 1
n=0
for ((u=A; u<B; u++)); do
  ( UNIT_START=$u UNIT_END=$((u+1)) LP_NIT=100 Rscript r_laplace.R "$DATA" "$OUT/lp_u$u.csv" "$U" > "$OUT/lp_u$u.log" 2>&1; echo "unit $u done" ) &
  n=$((n+1)); if [ $n -ge $W ]; then wait -n; n=$((n-1)); fi
done
wait
rm -rf "$DATA"
echo "UNITS_DONE=$((B-A))"
