#!/bin/bash
# TCLUST-REG long search (1000 starts, 50 refinement steps) at 0.25 and 0.40 on the 20 taxi samples.
set -u
CODE=$(cd "$(dirname "$0")" && pwd); cd "$CODE"
export FSDA=${FSDA:-$HOME/FSDA/toolbox} SHIM=${SHIM:-$CODE/fsda_shim}
W=${WORKERS:-1}; OUT=${RESULTS_DIR:-$CODE/results}; mkdir -p "$OUT"
DATA=${TMPDIR:-/tmp}/octtaxi_$$; mkdir -p "$DATA"
env -u UNIT_START -u UNIT_END V4_STUDY=taxi python simv4.py "$DATA" > /dev/null
pids=()
for ((k=0; k<W; k++)); do [ $k -ge 20 ] && continue
  ( UNIT_START=$k UNIT_END=20 UNIT_STRIDE=$W DATA_DIR="$DATA" UNITS_FILE=units_taxi.csv OUT_FILE="$OUT/octtaxi_$k.csv" octave --no-gui -q oct_long.m > "$OUT/octtaxi_$k.log" 2>&1 ) &
  pids+=($!); done
for p in "${pids[@]}"; do wait $p; done
rm -rf "$DATA"
cd ../realdata 2>/dev/null || cd "$CODE/realdata"
octave --no-gui -q realrival.m > "$OUT/realrival.log" 2>&1
cp realrival.csv "$OUT/"
echo "UNITS_DONE=20"
