#!/bin/bash
# B=500 variant over all units of one study; chunk ranges from the plan; study from BV_STUDY.
set -u
CODE=$(cd "$(dirname "$0")" && pwd); cd "$CODE"
OUT=${RESULTS_DIR:-$CODE/results}; mkdir -p "$OUT"
V4_STUDY=$BV_STUDY python bvar.py "$OUT/bvar_${BV_STUDY}_${CHUNK_ID:-0}.csv"
