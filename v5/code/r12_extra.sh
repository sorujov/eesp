#!/bin/bash
# Round 12: the 1,650 further data sets of r12_extra.py: ESF, then TCLUST-REG with the long search
# at 0.25 and 0.40 (oct_long.m) and at 0.30, 0.35 and with equal weights at 0.25, 0.40 (oct_r9.m,
# R9_MODE=r10).  Worker k of a chunk takes units UNIT_START+k, UNIT_START+k+W, ...
set -u
CODE=$(cd "$(dirname "$0")" && pwd); cd "$CODE"
export FSDA=${FSDA:-$HOME/FSDA/toolbox} SHIM=${SHIM:-$CODE/fsda_shim}
S=${UNIT_START:-0}; E=${UNIT_END:-1650}; W=${WORKERS:-1}; C=${CHUNK_ID:-0}
OUT=${RESULTS_DIR:-$CODE/results}; mkdir -p "$OUT"
DATA=${TMPDIR:-/tmp}/r12x_$$; mkdir -p "$DATA"
env -u UNIT_START -u UNIT_END python r12_extra.py units "$DATA" > /dev/null
python r12_extra.py esf "$OUT"
pids=()
for ((k=0; k<W; k++)); do a=$((S + k)); [ $a -ge $E ] && continue
  ( UNIT_START=$a UNIT_END=$E UNIT_STRIDE=$W DATA_DIR="$DATA" UNITS_FILE="$DATA/units.csv" OUT_FILE="$OUT/octlong_${C}_$k.csv" octave --no-gui -q oct_long.m > "$OUT/octlong_${C}_$k.log" 2>&1 ;
    R9_MODE=r10 UNIT_START=$a UNIT_END=$E UNIT_STRIDE=$W DATA_DIR="$DATA" UNITS_FILE="$DATA/units.csv" OUT_FILE="$OUT/octr9r10_${C}_$k.csv" octave --no-gui -q oct_r9.m > "$OUT/octr9r10_${C}_$k.log" 2>&1 ) &
  pids+=($!); done
for p in "${pids[@]}"; do wait $p; done
rm -rf "$DATA"
echo "UNITS_DONE=$((E-S))"
