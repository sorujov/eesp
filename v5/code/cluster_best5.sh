#!/bin/bash
set -u
CODE=$(cd "$(dirname "$0")" && pwd); cd "$CODE"
STUDIES=(1 1b 1c 1e 1d 1f); T=${TASK_ID:-0}; ST=${STUDIES[$T]}
OUT=${RESULTS_DIR:-$CODE/results}; mkdir -p "$OUT"
env -u UNIT_START -u UNIT_END V4_STUDY=$ST python best5.py "$OUT/best5_$ST.csv"
