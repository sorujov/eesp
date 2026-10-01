#!/bin/bash
# TCLUST-REG long search (1000 starts, 50 refinement steps) at levels 0.25 and 0.40 on all
# 7,750 study units. Global unit list = units_v4{,b,c,d,e,f,g,h}.csv in that order.
# Worker k of a chunk takes units UNIT_START+k, UNIT_START+k+W, ... (strided, for balance).
set -u
CODE=$(cd "$(dirname "$0")" && pwd); cd "$CODE"
export FSDA=${FSDA:-$HOME/FSDA/toolbox} SHIM=${SHIM:-$CODE/fsda_shim}
S=${UNIT_START:-0}; E=${UNIT_END:-7750}; W=${WORKERS:-1}; C=${CHUNK_ID:-0}
OUT=${RESULTS_DIR:-$CODE/results}; mkdir -p "$OUT"
DATA=${TMPDIR:-/tmp}/octr9_$$; mkdir -p "$DATA"
U=$DATA/units.csv; echo "name,K,rseed" > "$U"
# each study in its own folder: unit names repeat across studies (parts 1 and 1b share e30 names)
for st in "" b c d e f g h; do
  mkdir -p "$DATA/s1$st"
  env -u UNIT_START -u UNIT_END V4_STUDY=1$st python simv4.py "$DATA/s1$st" > /dev/null
  tail -n +2 units_v4$st.csv | sed "s|^|s1$st/|" >> "$U"
done
pids=()
for ((k=0; k<W; k++)); do a=$((S + k)); [ $a -ge $E ] && continue
  ( REVERSE=${REVERSE:-} R9_MODE=$R9_MODE UNIT_START=$a UNIT_END=$E UNIT_STRIDE=$W DATA_DIR="$DATA" UNITS_FILE="$U" OUT_FILE="$OUT/octr9${R9_MODE}_${C}_$k.csv" octave --no-gui -q oct_r9.m > "$OUT/octr9${R9_MODE}_${C}_$k.log" 2>&1 ) &
  pids+=($!); done
for p in "${pids[@]}"; do wait $p; done
rm -rf "$DATA"
echo "UNITS_DONE=$((E-S))"
