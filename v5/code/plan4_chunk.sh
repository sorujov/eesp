#!/bin/bash
# Plan 4 (study 1g): all methods on units [UNIT_START, UNIT_END) of units_v4g.csv.
set -u
CODE=$(cd "$(dirname "$0")" && pwd); cd "$CODE"
export V4_STUDY=1g UNITS_CSV=units_v4g.csv V4_TA=0
S=${UNIT_START:-0}; E=${UNIT_END:-600}; W=${WORKERS:-1}; C=${CHUNK_ID:-0}
OUT=${RESULTS_DIR:-$CODE/results}; mkdir -p "$OUT"
bash run_chunk.sh || exit 1
DATA=${TMPDIR:-/tmp}/g1lp_$$; mkdir -p "$DATA"
env -u UNIT_START -u UNIT_END python simv4.py "$DATA" > /dev/null
N=$((E-S)); pids=()
for ((u=S; u<E; u++)); do
  ( UNIT_START=$u UNIT_END=$((u+1)) Rscript r_laplace.R "$DATA" "$OUT/lp_u$u.csv" units_v4g.csv > "$OUT/lp_u$u.log" 2>&1 ) &
  pids+=($!)
  if [ ${#pids[@]} -ge $W ]; then wait "${pids[0]}"; pids=("${pids[@]:1}"); fi
done
wait
rm -rf "$DATA"
echo "UNITS_DONE=$N"
