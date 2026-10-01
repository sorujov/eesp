#!/bin/bash
# Study 1c only, split over tasks: task TASK_ID of N_TASKS takes a contiguous slice of units.
set -u
CODE=$(cd "$(dirname "$0")" && pwd); cd "$CODE"
T=${TASK_ID:-0}; NT=${N_TASKS:-1}; W=${WORKERS:-1}; U=units_v4c.csv
OUT=${RESULTS_DIR:-$CODE/results}/lp_1c_split; mkdir -p "$OUT"
DATA=${TMPDIR:-/tmp}/lpdata_1c_${T}_$$; mkdir -p "$DATA"
env -u UNIT_START -u UNIT_END V4_STUDY=1c python simv4.py "$DATA" > /dev/null || exit 1
N=$(($(wc -l < "$U") - 1)); PER=$(( (N + NT - 1) / NT )); S=$((T*PER)); E=$((S+PER)); [ $E -gt $N ] && E=$N
STEP=$(( (E - S + W - 1) / W )); pids=()
for ((k=0; k<W; k++)); do a=$((S + k*STEP)); b=$((a+STEP)); [ $b -gt $E ] && b=$E; [ $a -ge $b ] && continue
  ( UNIT_START=$a UNIT_END=$b Rscript r_laplace.R "$DATA" "$OUT/lp_${T}_$k.csv" "$U" > "$OUT/lp_${T}_$k.log" 2>&1 ) &
  pids+=($!); done
f=0; for p in "${pids[@]}"; do wait $p || f=1; done
rm -rf "$DATA"
[ $f -eq 0 ] && echo "UNITS_DONE=$((E-S))"
