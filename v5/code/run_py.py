"""Python methods over a unit range; writes unit, method, time, alpha_hat, b1..b12.
Env: UNIT_START, UNIT_END, RESULTS_DIR, CHUNK_ID, WORKERS; argv[1] = units file."""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

import simv4
import py_methods as P


def run_unit(i):
    u = simv4.units()[i]
    X, y, z0, out = simv4.generate(u)
    D = simv4.DESIGNS[u[0]]
    rng = np.random.default_rng(simv4.seed_for("v4py", *u))
    res = P.run_all(X, y, D["K"], D["m"], rng, with_ta=os.environ.get("V4_TA", "1") == "1")
    rows = []
    for meth, (th, ah, t) in res.items():
        b = np.full(12, np.nan)
        if th is not None and np.isfinite(th).all():
            v = np.asarray(th).reshape(-1)          # component-major: b_1_1..b_1_d, b_2_1..
            b[:len(v)] = v
        rows.append(dict(unit=simv4.unit_name(u), method=meth, time=t, alpha_hat=ah,
                         **{f"b{j+1}": b[j] for j in range(12)}))
    return rows


def main():
    N = len(simv4.units())
    a = int(os.environ.get("UNIT_START", 0))
    b = int(os.environ.get("UNIT_END", N))
    w = int(os.environ.get("WORKERS", 1))
    out = os.environ.get("RESULTS_DIR", "results")
    os.makedirs(out, exist_ok=True)
    chunk = os.environ.get("CHUNK_ID", "0")
    # warm up numba once per process
    X, y, _, _ = simv4.generate(simv4.units()[0])
    P.run_all(X[:80], y[:80], 2, 8, np.random.default_rng(0), B=10, with_ta=False)
    if w > 1:
        with Pool(w) as pool:
            parts = pool.map(run_unit, range(a, b), chunksize=1)
    else:
        parts = [run_unit(i) for i in range(a, b)]
    rows = [r for p in parts for r in p]
    pd.DataFrame(rows).to_csv(os.path.join(out, f"py_{chunk}.csv"), index=False)
    print(f"UNITS_DONE={b - a}")


if __name__ == "__main__":
    main()
