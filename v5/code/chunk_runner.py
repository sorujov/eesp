"""Cluster chunk runner.  python3 chunk_runner.py [e3|e4|all]
Reads UNIT_START, UNIT_END, WORKERS, RESULTS_DIR, CHUNK_ID from the
environment (defaults run everything serially) and writes one CSV per chunk.
Units are seeded by their identity, so results do not depend on the split."""
import os
import sys
import time
from multiprocessing import get_context

import pandas as pd

import numpy as np

import exp_contam as C
import exp_optim as O

exp = sys.argv[1] if len(sys.argv) > 1 else "all"
MODS = {"e3": [O], "e4": [C], "all": [O, C]}[exp]
UNITS = [(k, u) for k, M in enumerate(MODS) for u in M.units()]
# deterministic shuffle so that every chunk mixes cheap and expensive cells
UNITS = [UNITS[i] for i in np.random.default_rng(20260917).permutation(len(UNITS))]


def work(u):
    k, unit = UNITS[u]
    tag = "e3" if MODS[k] is O else "e4"
    try:
        return [dict(r, exp=tag, unit=u) for r in MODS[k].run_unit(unit)]
    except Exception as exc:              # record, do not lose the chunk
        return [dict(exp=tag, unit=u, error=repr(exc), unit_desc=repr(unit)[:200])]


if __name__ == "__main__":
    start = int(os.environ.get("UNIT_START", 0))
    end = int(os.environ.get("UNIT_END", len(UNITS)))
    end = min(end, len(UNITS))
    workers = int(os.environ.get("WORKERS", 1))
    out_dir = os.environ.get("RESULTS_DIR", "results_chunks")
    chunk = os.environ.get("CHUNK_ID", "0")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    for M in MODS:
        M.warmup()                      # compile once in the parent; children inherit
    idx = list(range(start, end))
    rows = []
    if workers > 1:
        with get_context("fork").Pool(workers) as pool:
            for r in pool.imap_unordered(work, idx, chunksize=1):
                rows.extend(r)
    else:
        for u in idx:
            rows.extend(work(u))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{exp}_chunk_{chunk}.csv"), index=False)
    print(f"N_UNITS={len(UNITS)} elapsed={time.time()-t0:.1f}s")
    print(f"UNITS_DONE={end - start}")
