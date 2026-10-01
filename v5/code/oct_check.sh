#!/bin/bash
# TCLUST-REG settings check on D2 (parts 1 and 1b): units [UNIT_START, UNIT_END) of the joint D2 list
set -u
CODE=$(cd "$(dirname "$0")" && pwd); cd "$CODE"
export FSDA=${FSDA:-$HOME/FSDA/toolbox} SHIM=${SHIM:-$CODE/fsda_shim}
S=${UNIT_START:-0}; E=${UNIT_END:-500}; W=${WORKERS:-1}; C=${CHUNK_ID:-0}
OUT=${RESULTS_DIR:-$CODE/results}; mkdir -p "$OUT"
DATA=${TMPDIR:-/tmp}/octchk_$$; mkdir -p "$DATA"
env -u UNIT_START -u UNIT_END V4_STUDY=1 python simv4.py "$DATA" > /dev/null
env -u UNIT_START -u UNIT_END V4_STUDY=1b python simv4.py "$DATA" > /dev/null
U=$DATA/units.csv; head -1 units_v4.csv > "$U"; grep '^D2-' units_v4.csv | grep -v '_e30_' >> "$U"; grep '^D2-' units_v4b.csv >> "$U"
N=$((E-S)); STEP=$(( (N + W - 1) / W )); pids=()
for ((k=0; k<W; k++)); do a=$((S + k*STEP)); b=$((a+STEP)); [ $b -gt $E ] && b=$E; [ $a -ge $b ] && continue
  ( UNIT_START=$a UNIT_END=$b DATA_DIR="$DATA" UNITS_FILE="$U" OUT_FILE="$OUT/octchk_${C}_$k.csv" octave --no-gui -q oct_check.m > "$OUT/octchk_${C}_$k.log" 2>&1 ) &
  pids+=($!); done
for p in "${pids[@]}"; do wait $p; done
rm -rf "$DATA"
echo "UNITS_DONE=$N"
