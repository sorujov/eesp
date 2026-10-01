"""E3: aggregation versus selection versus alternation, as optimisers of the
least-squares criterion.

Reference value: the exact minimiser (O(n^3) oracle) when K = 2 with one
covariate; otherwise the best value found by any method or by 2000 starts of
hard alternation ('best known', flagged in the output).

Usage: python3 exp_optim.py JOB NJOBS
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import eespcore as E
from simcommon import OUT, generate, seed_for

REPS = int(os.environ.get("E3_REPS", "200"))
B = 100

# (label, K, geom, delta, n, m, extra)
CELLS = [
    ("K2-par-m8", 2, "par", 3.0, 200, 8, {}),
    ("K2-par-m16", 2, "par", 3.0, 200, 16, {}),
    ("K2-cross-m8", 2, "cross", 9.0, 200, 8, {}),
    ("K2-cross-m16", 2, "cross", 9.0, 200, 16, {}),
    ("K2-par-d2", 2, "par", 2.0, 200, 8, {}),
    ("K2-par-d4", 2, "par", 4.0, 200, 8, {}),
    ("K2-cross-d6", 2, "cross", 6.0, 200, 8, {}),
    ("K2-cross-d12", 2, "cross", 12.0, 200, 8, {}),
    ("K2-cross-imb", 2, "cross", 9.0, 200, 16, {"pis": [0.8, 0.2]}),
    ("K2-cross-t3", 2, "cross", 9.0, 200, 8, {"err": "t3"}),
    ("K2-cross-n100", 2, "cross", 9.0, 100, 8, {}),
    ("K2-cross-n500", 2, "cross", 9.0, 500, 8, {}),
    ("K2-cross-n1000", 2, "cross", 9.0, 1000, 8, {"reps": 40}),
    ("K2-par-p2", 2, "par", 3.0, 200, 14, {"p": 2}),
    ("K3-par", 3, "par", 3.0, 300, 12, {}),
    ("K3-cross", 3, "cross", 9.0, 300, 12, {}),
    ("K4-par", 4, "par", 3.0, 400, 16, {}),
    ("K4-cross", 4, "cross", 9.0, 400, 16, {}),
]


def run_methods(X, y, K, m, rng):
    res = {}
    t = time.perf_counter(); f = E.eesp(X, y, K, m, B, rng); tv = time.perf_counter() - t
    res["vote"] = (f.z, f.F, tv)
    t = time.perf_counter(); g = E.polish(X, y, K, f); res["vote+pol"] = (g.z, g.F, tv + time.perf_counter() - t)
    t_votepol = res["vote+pol"][2]
    t = time.perf_counter(); f = E.polish(X, y, K, E.eesp(X, y, K, m, B, rng, agg="param"))
    res["param+pol"] = (f.z, f.F, time.perf_counter() - t)
    t = time.perf_counter(); f = E.best_of_B(X, y, K, m, B, rng); tb = time.perf_counter() - t
    res["bestB"] = (f.z, f.F, tb)
    t = time.perf_counter(); g = E.polish(X, y, K, f); res["bestB+pol"] = (g.z, g.F, tb + time.perf_counter() - t)
    t = time.perf_counter(); f = E.polish(X, y, K, E.eesp(X, y, K, m, B, rng, solver="local"))
    res["local-vote+pol"] = (f.z, f.F, time.perf_counter() - t)
    t = time.perf_counter(); f = E.polish(X, y, K, E.eesp(X, y, K, m, B, rng, solver="multistart", R_sub=5))
    res["ms-vote+pol"] = (f.z, f.F, time.perf_counter() - t)
    t = time.perf_counter(); z, _, F, _ = E.hard_alternation(X, y, K, 10, rng)
    res["HA10"] = (z, F, time.perf_counter() - t)
    t = time.perf_counter(); z, _, F, _ = E.hard_alternation(X, y, K, 10, rng, init="labels")
    res["HA10-part"] = (z, F, time.perf_counter() - t)
    # time-matched multistart alternation (budget = time of vote+pol)
    t = time.perf_counter(); R = 0; bz, bF = None, np.inf
    while R == 0 or time.perf_counter() - t < t_votepol:
        z, _, F, _ = E.hard_alternation(X, y, K, 1, rng)
        R += 1
        if F < bF:
            bz, bF = z, F
    res["HA-tm"] = (bz, bF, time.perf_counter() - t)
    t = time.perf_counter(); z, _, F, _, _ = E.em_mixreg(X, y, K, 10, rng, init="labels", short=10)
    res["emEM10"] = (z, F, time.perf_counter() - t)
    return res, R


def units():
    """Global list of independent units (cell, replication)."""
    out = []
    for c in CELLS:
        for r in range(c[6].get("reps", REPS)):
            out.append((c, r))
    return out


def warmup():
    Xw, yw, _, _ = generate(np.random.default_rng(0), 60, 2, "cross", 9.0)
    run_methods(Xw, yw, 2, 8, np.random.default_rng(0))


def run_unit(unit):
    (lab, K, geom, delta, n, m, extra), r = unit
    kw = {k: v for k, v in extra.items() if k in ("pis", "err", "p")}
    X, y, z0, _ = generate(np.random.default_rng(seed_for("E3data", lab, r)), n, K, geom, delta, **kw)
    rng = np.random.default_rng(seed_for("E3", lab, r))
    res, R_tm = run_methods(X, y, K, m, rng)
    exact = (K == 2 and X.shape[1] == 2)
    if exact:
        zref, Fref = E.oracle_k2_line(X[:, 1], y)
    else:
        zref, _, Fref, _ = E.hard_alternation(X, y, K, 2000, rng)
        for v in res.values():
            if v[1] < Fref:
                zref, Fref = v[0], v[1]
    rows = []
    for meth, (z, F, tm) in res.items():
        rows.append(dict(cell=lab, K=K, geom=geom, delta=delta, n=n, m=m, rep=r,
                         method=meth, F=F, Fref=Fref, ref_exact=exact,
                         hit=bool(F <= Fref * (1 + 1e-9) + 1e-12),
                         gap=F / Fref - 1,
                         agree_ref=E.agreement(z, zref, K),
                         acc0=E.agreement(z, z0, K), ari0=E.ari(z, z0),
                         time=tm, R_tm=R_tm))
    return rows


def main():
    job, njobs = int(sys.argv[1]), int(sys.argv[2])
    fn = os.path.join(OUT, f"e3_rows_{job}.csv")
    rows = pd.read_csv(fn).to_dict("records") if os.path.exists(fn) else []
    done = {(r["cell"], r["rep"]) for r in rows}
    cells = CELLS[job::njobs]
    # warm-up: compile every kernel before any timing
    Xw, yw, _, _ = generate(np.random.default_rng(0), 60, 2, "cross", 9.0)
    run_methods(Xw, yw, 2, 8, np.random.default_rng(0))
    for (lab, K, geom, delta, n, m, extra) in cells:
        reps = extra.get("reps", REPS)
        kw = {k: v for k, v in extra.items() if k in ("pis", "err", "p")}
        for r in range(reps):
            if (lab, r) in done:
                continue
            X, y, z0, _ = generate(np.random.default_rng(seed_for("E3data", lab, r)), n, K, geom, delta, **kw)
            rng = np.random.default_rng(seed_for("E3", lab, r))
            res, R_tm = run_methods(X, y, K, m, rng)
            exact = (K == 2 and X.shape[1] == 2)
            if exact:
                zref, Fref = E.oracle_k2_line(X[:, 1], y)
            else:
                zref, _, Fref, _ = E.hard_alternation(X, y, K, 2000, rng)
                for v in res.values():
                    if v[1] < Fref:
                        zref, Fref = v[0], v[1]
            for meth, (z, F, tm) in res.items():
                rows.append(dict(cell=lab, K=K, geom=geom, delta=delta, n=n, m=m, rep=r,
                                 method=meth, F=F, Fref=Fref, ref_exact=exact,
                                 hit=bool(F <= Fref * (1 + 1e-9) + 1e-12),
                                 gap=F / Fref - 1,
                                 agree_ref=E.agreement(z, zref, K),
                                 acc0=E.agreement(z, z0, K), ari0=E.ari(z, z0),
                                 time=tm, R_tm=R_tm))
            if r % 10 == 9 or r == reps - 1:
                pd.DataFrame(rows).to_csv(fn, index=False)
                print(time.strftime("%H:%M:%S"), lab, r, flush=True)
    pd.DataFrame(rows).to_csv(fn, index=False)


if __name__ == "__main__":
    main()
