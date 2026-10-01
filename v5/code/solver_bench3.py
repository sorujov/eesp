"""Solver benchmark, additions of review round 3 (descriptive).
(1) rbba_full: repetitive branch and bound with the incumbents of Brusco (2006): each nested
    problem on units j..m-1 starts from the optimal labelling of units j+1..m-1 extended by the
    best label of unit j, and the full problem from ten starts of hard alternation, if admissible.
(2) Larger K and d: five parallel lines (K=5, one covariate, m = 20, 25) and two crossing
    groups with four covariates (K=2, d=5, m = 12, 16, 20), from simcommon.generate with 10%
    outliers in the response.
All other solvers as in solver_bench2.py.  Usage: TASK_ID=k python solver_bench3.py"""
import os, time
import numpy as np, pandas as pd
import eespcore as E
import simcommon as SC
import solver_bench2 as S2


def rbba_full(X, y, K, rng):
    m, d = X.shape
    lbs = np.zeros(m + 1)
    nodes, cap = 0, False
    lab_next = np.zeros(0, dtype=np.int64)
    for j in range(m - 1, 0, -1):
        Xs, ys = X[j:], y[j:]
        # incumbent: previous optimum extended by the best label of unit j
        best_v, best_l = np.inf, np.zeros(m - j, dtype=np.int64)
        if len(lab_next):
            for k in range(K):
                l = np.concatenate([[k], lab_next]).astype(np.int64)
                l = E._canonical(l).astype(np.int64)
                v = E._labels_value(Xs, ys, l, K) if hasattr(E, "_labels_value") else np.inf
                if v < best_v:
                    best_v, best_l = v, l
        v, lab, nd, c = S2._bb(Xs, ys, K, best_l, best_v * (1 + 1e-12) + 1e-12 if np.isfinite(best_v) else np.inf,
                               lbs[j:].copy(), 0, S2.MAXNODES)
        lbs[j] = v
        lab_next = lab.copy()
        nodes += nd
        cap = cap or c
    z, _, f, _ = E.hard_alternation(X, y, K, 10, rng)
    z = E._canonical(z).astype(np.int64)
    if np.bincount(z, minlength=K).min() >= d + 1:
        z0, v0 = z, f * (1 + 1e-12) + 1e-12
    else:
        z0, v0 = np.zeros(m, dtype=np.int64), np.inf
    v, lab, nd, c = S2._bb(X, y, K, z0, v0, lbs.copy(), d + 1, S2.MAXNODES)
    return v, nodes + nd, not (cap or c)


CASES = [("K5", 5, "par", 4.0, 1, m) for m in [20, 25]] + [("p4", 2, "cross", 9.0, 4, m) for m in [12, 16, 20]] + \
        [("D1", 2, "cross", 9.0, 1, m) for m in [12, 20, 28]] + [("D2", 3, "par", 3.0, 1, m) for m in [12, 20]]

if __name__ == "__main__":
    reps = int(os.environ.get("BENCH_REPS", "30"))
    tasks = CASES if "TASK_ID" not in os.environ else [CASES[int(os.environ["TASK_ID"])]]
    Xw = np.column_stack([np.ones(12), np.arange(12.0)]); yw = np.sin(np.arange(12.0))
    S2.bb(Xw, yw, 2); S2.bb_inc(Xw, yw, 2, np.random.default_rng(0)); S2.rbba(Xw, yw, 2); rbba_full(Xw, yw, 2, np.random.default_rng(0))
    rows = []
    for name, K, geom, delta, p, m in tasks:
        for j in range(reps):
            g = np.random.default_rng([2026, j, m, K, p])
            X, y, z0, out = SC.generate(g, 300, K, geom=geom, delta=delta, eps=0.10, p=p)
            S = g.choice(300, m, replace=False)
            XS, yS = np.ascontiguousarray(X[S]), np.ascontiguousarray(y[S])
            res = {}
            t = time.perf_counter(); v, n, ok = S2.bb(XS, yS, K); res["bb"] = (v, time.perf_counter() - t, n, ok)
            t = time.perf_counter(); v, n, ok = S2.bb_inc(XS, yS, K, np.random.default_rng(1)); res["bb_inc"] = (v, time.perf_counter() - t, n, ok)
            t = time.perf_counter(); v, n, ok = S2.rbba(XS, yS, K); res["rbba"] = (v, time.perf_counter() - t, n, ok)
            t = time.perf_counter(); v, n, ok = rbba_full(XS, yS, K, np.random.default_rng(1)); res["rbba_full"] = (v, time.perf_counter() - t, n, ok)
            if name in ("K5", "p4"):
                res["miqp_g"] = S2.miqp_gurobi(XS, yS, K)
            fin = [r[0] for r in res.values() if r[3] and np.isfinite(r[0])]
            ref = min(fin) if fin else np.nan
            for s, (v, t, n, ok) in res.items():
                rows.append(dict(design=name, K=K, d=XS.shape[1], m=m, rep=j, solver=s, value=v, time=t,
                                 nodes=n, optimal=ok, gap=(v - ref) / max(ref, 1e-12)))
            print(name, m, j, {s: (round(r[1], 3), r[3]) for s, r in res.items()}, flush=True)
    d = os.environ.get("RESULTS_DIR", ".")
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(d, "bench3_%s.csv" % os.environ.get("TASK_ID", "all")), index=False)
    print("UNITS_DONE=1")
