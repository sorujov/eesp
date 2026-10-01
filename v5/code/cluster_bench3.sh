#!/bin/bash
set -u
CODE=$(cd "$(dirname "$0")" && pwd); cd "$CODE"
env -u UNIT_START -u UNIT_END python solver_bench3.py
