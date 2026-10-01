#!/bin/bash
# Round 8: the 60 real-data TCLUST-REG long-search fits in parallel (one Octave process each).
CODE=$(cd "$(dirname "$0")" && pwd); cd "$CODE/realdata"
export FSDA=${FSDA:-$HOME/FSDA/toolbox} SHIM=${SHIM:-$CODE/fsda_shim}
OUT=${RESULTS_DIR:-$CODE/results}; mkdir -p "$OUT"
for f in fishery_log tone_clean tone_out; do for a in 0.25 0.40; do for s in 1 2 3 4 5 6 7 8 9 10; do
  RR_DATA=$f RR_ALPHA=$a RR_SEED=$s RR_OUT="$OUT/rr_${f}_${a}_$s.csv" octave --no-gui -q realrival1.m > /dev/null 2>&1 &
done; done; done
wait
echo "data,alpha,seed,b0_1,b1_1,b0_2,b1_2,trimmed" > "$OUT/realrival.csv"; cat "$OUT"/rr_*.csv >> "$OUT/realrival.csv"
echo "UNITS_DONE=60"
