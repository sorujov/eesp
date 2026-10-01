#!/bin/bash
# Laplace mixture with 100 starts (LP_NIT=100) on the designs with three or four groups
# (D2 in parts 1 and 1b, E4 in part 1c); task TASK_ID picks the study.
set -u
CODE=$(cd "$(dirname "$0")" && pwd); cd "$CODE"
STUDIES=(1 1b 1c); UNITSF=(units_v4.csv units_v4b.csv units_v4c.csv)
T=${TASK_ID:-0}; ST=${STUDIES[$T]}; W=${WORKERS:-1}
OUT=${RESULTS_DIR:-$CODE/results}/lp100_$ST; mkdir -p "$OUT"
DATA=${TMPDIR:-/tmp}/lp100data_${ST}_$$; mkdir -p "$DATA"
U=$DATA/units.csv
head -1 "${UNITSF[$T]}" > "$U"; grep -E '^(D2-|E4-)' "${UNITSF[$T]}" >> "$U"
env -u UNIT_START -u UNIT_END V4_STUDY=$ST python simv4.py "$DATA" > /dev/null || exit 1
N=$(($(wc -l < "$U") - 1)); STEP=$(( (N + W - 1) / W )); pids=()
for ((k=0; k<W; k++)); do a=$((k*STEP)); b=$((a+STEP)); [ $b -gt $N ] && b=$N; [ $a -ge $b ] && continue
  ( UNIT_START=$a UNIT_END=$b LP_NIT=100 Rscript r_laplace.R "$DATA" "$OUT/lp_$k.csv" "$U" > "$OUT/lp_$k.log" 2>&1 ) &
  pids+=($!); done
f=0; for p in "${pids[@]}"; do wait $p || f=1; done
rm -rf "$DATA"
[ $f -eq 0 ] && echo "UNITS_DONE=$N"
