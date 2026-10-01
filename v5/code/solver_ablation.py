"""Ablation (descriptive, after the pre-registered study): Algorithm 1 with the exact
subsample fit replaced by (a) one start of hard alternation on the subsample, (b) ten starts,
(c) elemental fits: K*d units drawn and split at random into K groups of d, each group fitted
exactly through its d points (RANSAC-style).  Everything else unchanged.
Usage: V4_STUDY=... python solver_ablation.py OUT.csv DESIGNS EPS"""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
import simv4, py_methods as P, evaluate as EV
import eespcore as E, eesp_variants as V

_orig = V._replicate


def make_rep(solver):
    def rep(X, y, K, m, rng, probs=None, max_redraw=10000):
        n, d = X.shape
        for r in range(1, max_redraw + 1):
            if solver == "elemental":
                S = rng.choice(n, size=K * d, replace=False, p=probs)
                lab = np.repeat(np.arange(K), d)
                th = E.refit(X[S], y[S], lab, K)
                if np.isfinite(th).all():
                    return th, S
                continue
            S = rng.choice(n, size=m, replace=False, p=probs)
            XS, yS = X[S], y[S]
            if solver == "exact":
                lab, _, _ = E.exact_solve(XS, yS, K)
            else:
                lab, _, _, _ = E.hard_alternation(XS, yS, K, 1 if solver == "local" else 10, rng)
            if np.bincount(lab, minlength=K).min() < d + 1:
                continue
            th = E.refit(XS, yS, lab, K)
            if np.isfinite(th).all():
                return th, S
        raise RuntimeError("no admissible subsample")
    return rep


SOLVERS = ["exact", "local", "multistart", "elemental"]


def run(u):
    X, y, z0, out = simv4.generate(u)
    D = simv4.DESIGNS[u[0]]
    rows = []
    for s in SOLVERS:
        V._replicate = make_rep(s)
        rng = np.random.default_rng(simv4.seed_for("v4solver", *u))
        try:
            th, ah, t = P.eesp_adaptive_x(X, y, D["K"], D["m"], 100, rng, xtrim=True)
            a = EV.acc_clean(X, y, th, z0, ~out)
        except Exception:
            a, ah, t = 0.0, np.nan, np.nan
        rows.append(dict(design=u[0], eps=u[1], rep=u[2], solver=s, acc=a, ahat=ah, time=t))
    V._replicate = _orig
    return rows


if __name__ == "__main__":
    des = sys.argv[2].split(","); eps = [float(e) for e in sys.argv[3].split(",")]
    U = [u for u in simv4.units() if u[0] in des and any(abs(u[1] - e) < 1e-9 for e in eps) and u[2] < 50]
    with Pool(int(os.environ.get("WORKERS", "2"))) as p:
        R = p.map(run, U, chunksize=2)
    df = pd.DataFrame([r for rr in R for r in rr])
    df.to_csv(sys.argv[1], index=False)
    print(df.groupby(["design", "eps", "solver"]).acc.mean().unstack().round(3))
    print(df.groupby(["solver"]).time.median().round(3))
