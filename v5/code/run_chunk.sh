#!/bin/bash
# One chunk of the v4 study: data, Python methods (process pool), R and Octave
# methods (WORKERS parallel sub-ranges each).  Env from the cluster job.
set -u
CODE=$(cd "$(dirname "$0")" && pwd)
S=${UNIT_START:-0}; E=${UNIT_END:-2000}; W=${WORKERS:-1}; C=${CHUNK_ID:-0}
OUT=${RESULTS_DIR:-$CODE/../results_v4}; mkdir -p "$OUT"
DATA=${TMPDIR:-/tmp}/v4data_${C}_$$; mkdir -p "$DATA"
export FSDA=${FSDA:-$HOME/FSDA/toolbox} SHIM=${SHIM:-$CODE/fsda_shim}
cd "$CODE"
UNITS_CSV=${UNITS_CSV:-units_v4.csv}
UNIT_START=$S UNIT_END=$E python simv4.py "$DATA" || exit 1
UNIT_START=$S UNIT_END=$E WORKERS=$W RESULTS_DIR=$OUT CHUNK_ID=$C python run_py.py || exit 1
N=$((E-S))
# phase 2: Octave (TCLUST-REG) and phase 3: R (RobMixReg), each over W parallel sub-ranges,
# so that every reserved CPU stays busy
run_phase() {   # $1 = oct | r
  local STEP=$(( (N + W - 1) / W )); local pids=(); local k a b
  for ((k=0; k<W; k++)); do
    a=$((S + k*STEP)); b=$((a + STEP)); [ $b -gt $E ] && b=$E; [ $a -ge $b ] && continue
    if [ "$1" = oct ]; then
      ( UNIT_START=$a UNIT_END=$b DATA_DIR="$DATA" UNITS_FILE="$UNITS_CSV" OUT_FILE="$OUT/oct_${C}_$k.csv" \
          octave --no-gui -q oct_methods.m > "$OUT/oct_${C}_$k.log" 2>&1 ) &
    else
      ( UNIT_START=$a UNIT_END=$b Rscript r_methods.R "$DATA" "$OUT/r_${C}_$k.csv" "$UNITS_CSV" > "$OUT/r_${C}_$k.log" 2>&1 ) &
    fi
    pids+=($!)
  done
  local f=0; for p in "${pids[@]}"; do wait $p || f=1; done; return $f
}
fail=0
run_phase oct || fail=1
run_phase r || fail=1
rm -rf "$DATA"
[ $fail -eq 0 ] && echo "UNITS_DONE=$N"
