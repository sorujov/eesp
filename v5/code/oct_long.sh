#!/bin/bash
# TCLUST-REG long search (1000 starts, 50 refinement steps) at levels 0.25 and 0.40 on all
# 7,750 study units. Global unit list = units_v4{,b,c,d,e,f,g,h}.csv in that order.
# Worker k of a chunk takes units UNIT_START+k, UNIT_START+k+W, ... (strided, for balance).
set -u
CODE=$(cd "$(dirname "$0")" && pwd); cd "$CODE"
export FSDA=${FSDA:-$HOME/FSDA/toolbox} SHIM=${SHIM:-$CODE/fsda_shim}
S=${UNIT_START:-0}; E=${UNIT_END:-7750}; W=${WORKERS:-1}; C=${CHUNK_ID:-0}
OUT=${RESULTS_DIR:-$CODE/results}; mkdir -p "$OUT"
DATA=${TMPDIR:-/tmp}/octlong_$$; mkdir -p "$DATA"
U=$DATA/units.csv; echo "name,K,rseed" > "$U"
off=0
for st in "" b c d e f g h; do
  f=units_v4$st.csv; n=$(($(wc -l < $f) - 1))
  if [ $((off+n)) -gt $S ] && [ $off -lt $E ]; then
    env -u UNIT_START -u UNIT_END V4_STUDY=1$st python simv4.py "$DATA" > /dev/null
  fi
  tail -n +2 $f >> "$U"; off=$((off+n))
done
pids=()
for ((k=0; k<W; k++)); do a=$((S + k)); [ $a -ge $E ] && continue
  ( REVERSE=${REVERSE:-} UNIT_START=$a UNIT_END=$E UNIT_STRIDE=$W DATA_DIR="$DATA" UNITS_FILE="$U" OUT_FILE="$OUT/octlong_${C}_$k.csv" octave --no-gui -q oct_long.m > "$OUT/octlong_${C}_$k.log" 2>&1 ) &
  pids+=($!); done
for p in "${pids[@]}"; do wait $p; done
rm -rf "$DATA"
echo "UNITS_DONE=$((E-S))"
