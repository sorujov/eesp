#!/bin/bash
# Round 12: TCLUST-REG long search at 0.25 and 0.40 with its own assignment (out.idx) saved, on the
# three-group (D2), heteroscedastic (D4) and four-group (E4) designs of all parts; same seeds as
# oct_long.sh.  Unit names are prefixed by the study folder (parts 1 and 1b share e30 names).
set -u
CODE=$(cd "$(dirname "$0")" && pwd); cd "$CODE"
export FSDA=${FSDA:-$HOME/FSDA/toolbox} SHIM=${SHIM:-$CODE/fsda_shim}
S=${UNIT_START:-0}; E=${UNIT_END:-100000}; W=${WORKERS:-1}; C=${CHUNK_ID:-0}
OUT=${RESULTS_DIR:-$CODE/results}; mkdir -p "$OUT"
DATA=${TMPDIR:-/tmp}/octidx_$$; mkdir -p "$DATA"
U=$DATA/units.csv; echo "name,K,rseed" > "$U"
for st in "" b c; do
  mkdir -p "$DATA/s1$st"
  env -u UNIT_START -u UNIT_END V4_STUDY=1$st python simv4.py "$DATA/s1$st" > /dev/null
  tail -n +2 units_v4$st.csv | grep -E '^(D2-K3par|D4-hetero|E4-K4par)_' | sed "s|^|s1$st/|" >> "$U"
done
N=$(($(wc -l < "$U") - 1)); [ $E -gt $N ] && E=$N
pids=()
for ((k=0; k<W; k++)); do a=$((S + k)); [ $a -ge $E ] && continue
  ( UNIT_START=$a UNIT_END=$E UNIT_STRIDE=$W DATA_DIR="$DATA" UNITS_FILE="$U" OUT_FILE="$OUT/octidx_${C}_$k.csv" octave --no-gui -q oct_idx.m > "$OUT/octidx_${C}_$k.log" 2>&1 ) &
  pids+=($!); done
for p in "${pids[@]}"; do wait $p; done
rm -rf "$DATA"
echo "UNITS_DONE=$((E-S))"
