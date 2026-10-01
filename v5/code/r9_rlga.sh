#!/bin/bash
# RLGA on all 7,750 study units (global list as in oct_long.sh), W workers strided.
set -u
CODE=$(cd "$(dirname "$0")" && pwd); cd "$CODE"
W=${WORKERS:-1}; OUT=${RESULTS_DIR:-$CODE/results}; mkdir -p "$OUT"
DATA=${TMPDIR:-/tmp}/rlga_$$; mkdir -p "$DATA"
U=$DATA/units.csv; echo "name,K,rseed" > "$U"
# each study in its own folder: unit names repeat across studies (parts 1 and 1b share e30 names)
for st in "" b c d e f g h; do
  mkdir -p "$DATA/s1$st"
  env -u UNIT_START -u UNIT_END V4_STUDY=1$st python simv4.py "$DATA/s1$st" > /dev/null
  tail -n +2 units_v4$st.csv | sed "s|^|s1$st/|" >> "$U"
done
pids=()
for ((k=0; k<W; k++)); do
  ( UNIT_START=$k UNIT_END=7750 UNIT_STRIDE=$W Rscript r9_rlga.R "$DATA" "$OUT/rlga_$k.csv" "$U" > "$OUT/rlga_$k.log" 2>&1 ) &
  pids+=($!); done
for p in "${pids[@]}"; do wait $p; done
rm -rf "$DATA"
echo "UNITS_DONE=7750"
