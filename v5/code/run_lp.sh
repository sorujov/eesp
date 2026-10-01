#!/bin/bash
# Laplace mixture on every data set of one study: run_lp.sh STUDY RESULTS_DIR UNITS_CSV
set -u
cd "$(dirname "$0")"
ST=$1; OUT=$2; U=$3; W=${WORKERS:-2}
DATA=${TMPDIR:-/tmp}/lpdata_$ST; mkdir -p "$DATA"
V4_STUDY=$ST python3 simv4.py "$DATA" > /dev/null
N=$(($(wc -l < "$U") - 1)); STEP=$(( (N + W - 1) / W ))
for ((k=0; k<W; k++)); do a=$((k*STEP)); b=$((a+STEP)); [ $b -gt $N ] && b=$N
  ( UNIT_START=$a UNIT_END=$b Rscript r_laplace.R "$DATA" "$OUT/lp_$k.csv" "$U" > "$OUT/lp_$k.log" 2>&1 ) & done
wait
V4_STUDY=$ST python3 evaluate.py "$OUT" "$OUT/laplace.csv" lp
rm -rf "$DATA"
echo done $ST
