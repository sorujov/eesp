#!/bin/bash
# Laplace mixture (RobMixReg::mixLp) on every data set of one study; task TASK_ID picks the study.
set -u
CODE=$(cd "$(dirname "$0")" && pwd); cd "$CODE"
STUDIES=(1 1b 1c 1d 1e); UNITSF=(units_v4.csv units_v4b.csv units_v4c.csv units_v4d.csv units_v4e.csv)
T=${TASK_ID:-0}; ST=${STUDIES[$T]}; U=${UNITSF[$T]}; W=${WORKERS:-1}
OUT=${RESULTS_DIR:-$CODE/results}/lp_$ST; mkdir -p "$OUT"
DATA=${TMPDIR:-/tmp}/lpdata_${ST}_$$; mkdir -p "$DATA"
env -u UNIT_START -u UNIT_END V4_STUDY=$ST python simv4.py "$DATA" > /dev/null || exit 1
N=$(($(wc -l < "$U") - 1)); STEP=$(( (N + W - 1) / W )); pids=()
for ((k=0; k<W; k++)); do a=$((k*STEP)); b=$((a+STEP)); [ $b -gt $N ] && b=$N; [ $a -ge $b ] && continue
  ( UNIT_START=$a UNIT_END=$b Rscript r_laplace.R "$DATA" "$OUT/lp_$k.csv" "$U" > "$OUT/lp_$k.log" 2>&1 ) &
  pids+=($!); done
f=0; for p in "${pids[@]}"; do wait $p || f=1; done
rm -rf "$DATA"
[ $f -eq 0 ] && echo "UNITS_DONE=$N"
