"""E4: gross contamination.  Accuracy on the clean units, parameter error,
estimated contamination.

Part (a), methods (n = 300, B = 100):
  no protection  : HA50 (least-squares alternation, 50 starts), emEM10;
  EESP, no tuning: vote, median (coordinatewise median of aligned parameters);
  EESP adaptive  : adapt (sequential, 10 stages, weighted vote, lam = 3),
                   adapt-sel (lam = infinity: best-scoring replicate only),
                   adapt-1st (one stage, no sequential resampling),
                   adapt-lam0 (plain vote inside the adaptive scheme);
  trimmed        : TA-rw(alpha) = multistart trimmed alternation at a fixed
                   trimming level alpha followed by the same reweighting step,
                   given the wall time of `adapt`; alpha in {eps (oracle),
                   0.25, 0.40}; bestB-trim(0.25).
Cells: K = 2 crossing lines (m = 8), K = 3 parallel lines (m = 12), K = 2 with
unequal weights (0.7, 0.3), K = 2 with high-leverage outliers.

Part (b): the subsample-size window of the plain vote at eps = 0.1, 0.2.

Usage: python3 exp_contam.py JOB NJOBS
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import eespcore as E
import eesp_variants as V
from simcommon import OUT, generate, seed_for, truth

REPS = int(os.environ.get("E4_REPS", "200"))
B = 100
EPS = [0.0, 0.05, 0.10, 0.20, 0.30]
# label, K, geom, delta, n, m, extra
CELLS = [("K2-cross", 2, "cross", 9.0, 300, 8, {}),
         ("K3-par", 3, "par", 3.0, 300, 12, {}),
         ("K2-cross-unequal", 2, "cross", 9.0, 300, 10, {"pis": [0.7, 0.3], "reps": 100}),
         ("K2-cross-leverage", 2, "cross", 9.0, 300, 8, {"leverage": True, "reps": 100}),
         ("K2-cross-hetero", 2, "cross", 9.0, 300, 8, {"sigmas": [0.5, 1.5], "reps": 100})]
MWIN = [6, 8, 10, 12, 16, 20, 24]   # m >= 6: two groups of >= d+1 = 3 units


def make_data(rng, lab, K, geom, delta, n, eps, extra):
    X, y, z0, out = generate(rng, n, K, geom, delta, pis=extra.get("pis"), eps=eps,
                             sigma=extra.get("sigmas", 1.0))
    if extra.get("leverage") and out.any():
        k = out.sum()
        X[out, 1] = rng.uniform(6, 9, k)          # outside the design range
        y[out] = rng.uniform(-3, 3, k)            # far from every line there
    return X, y, z0, out


def ta_rw_timed(X, y, K, alpha, budget, rng, c=2.5):
    """Trimmed alternation at level alpha, restarted until `budget` seconds,
    then the reweighting step of V.ta_adaptive."""
    t0 = time.perf_counter()
    best = None
    R = 0
    while R == 0 or time.perf_counter() - t0 < budget:
        f = E.trimmed_alternation(X, y, K, 5, alpha, rng)
        R += 5
        if best is None or f.F < best.F:
            best = f
    n, d = X.shape
    theta = best.theta.copy()
    for _ in range(2):
        flagged, s, z = V._flag(X, y, theta, c)
        th2, cnt = V._refit_unflagged(X, y, z, flagged, K)
        for k in range(K):
            if cnt[k] >= d:
                theta[k] = th2[k]
    flagged, _, z = V._flag(X, y, theta, c)
    return z, theta, float(flagged.mean()), time.perf_counter() - t0, R


def perr(theta, K, geom, delta):
    th0 = truth(K, geom, delta)
    if theta is None or not np.isfinite(theta).all():
        return np.nan
    a, _ = E.align(theta, th0)
    return float(np.abs(a - th0).max())


def methods(X, y, K, m, eps, rng, geom, delta):
    res = {}

    def rec(name, fn):
        t = time.perf_counter()
        z, th, ah = fn()
        res[name] = (np.asarray(z), th, ah, time.perf_counter() - t)

    def ha():
        z, th, _, _ = E.hard_alternation(X, y, K, 50, rng)
        return z, th, np.nan
    rec("HA50", ha)

    def em():
        z, th, _, _, _ = E.em_mixreg(X, y, K, 10, rng, short=10)
        return z, th, np.nan
    rec("emEM10", em)
    rec("vote", lambda: (lambda f: (f.z, f.theta, np.nan))(E.eesp(X, y, K, m, B, rng)))
    rec("median", lambda: (lambda f: (f.z, f.theta, np.nan))(E.eesp(X, y, K, m, B, rng, agg="median")))
    rec("adapt", lambda: (lambda f: (f.z, f.theta, f.alpha_hat))(V.eesp_adaptive(X, y, K, m, B, rng)))
    budget = res["adapt"][3]
    rec("adapt-sel", lambda: (lambda f: (f.z, f.theta, f.alpha_hat))(V.eesp_adaptive(X, y, K, m, B, rng, lam=1e9)))
    rec("adapt-lam0", lambda: (lambda f: (f.z, f.theta, f.alpha_hat))(V.eesp_adaptive(X, y, K, m, B, rng, lam=0.0)))
    rec("adapt-1st", lambda: (lambda f: (f.z, f.theta, f.alpha_hat))(V.eesp_adaptive(X, y, K, m, B, rng, stages=1)))
    for tag, a in [("oracle", eps), ("0.25", 0.25), ("0.40", 0.40)]:
        def ta(a=a):
            z, th, ah, _, _ = ta_rw_timed(X, y, K, a, budget, rng)
            return z, th, ah
        rec(f"TA-rw({tag})", ta)
    rec("bestB-trim(0.25)", lambda: (lambda f: (f.z, f.theta, np.nan))(
        E.best_of_B_trimmed(X, y, K, m, B, 0.25, rng)))
    return res


def units():
    tasks = [("a", cc, e, cc[5]) for cc in CELLS for e in EPS] + \
            [("b", CELLS[0], e, mm) for e in (0.10, 0.20) for mm in MWIN]
    out = []
    for t in tasks:
        part, cc = t[0], t[1]
        reps = cc[6].get("reps", REPS) if part == "a" else REPS // 2
        for r in range(reps):
            out.append((t, r))
    return out


def warmup():
    Xw, yw, _, _ = generate(np.random.default_rng(0), 80, 2, "cross", 9.0, eps=0.1)
    methods(Xw, yw, 2, 8, 0.1, np.random.default_rng(0), "cross", 9.0)


def run_unit(unit):
    (part, (lab, K, geom, delta, n, m0, extra), eps, m), r = unit
    X, y, z0, out = make_data(np.random.default_rng(seed_for("E4data", lab, eps, r)),
                              lab, K, geom, delta, n, eps, extra)
    X = np.ascontiguousarray(X)
    rng = np.random.default_rng(seed_for("E4", part, lab, eps, m, r))
    cl = ~out
    if part == "a":
        res = methods(X, y, K, m, eps, rng, geom, delta)
    else:
        res = {}
        t = time.perf_counter()
        f = E.eesp(X, y, K, m, B, rng)
        res["vote"] = (f.z, f.theta, np.nan, time.perf_counter() - t)
        g = E.eesp(X, y, K, m, B, rng, agg="median")
        res["median"] = (g.z, g.theta, np.nan, np.nan)
    rows = []
    for meth, (z, th, ah, tm) in res.items():
        rows.append(dict(part=part, cell=lab, K=K, n=n, eps=eps, m=m, rep=r,
                         method=meth, n_out=int(out.sum()),
                         acc_clean=E.agreement(z[cl], z0[cl], K),
                         ari_clean=E.ari(z[cl], z0[cl]),
                         perr=perr(th, K, geom, delta), alpha_hat=ah, time=tm,
                         p_clean_sub=(1 - eps) ** m))
    return rows


def main():
    job, njobs = int(sys.argv[1]), int(sys.argv[2])
    fn = os.path.join(OUT, f"e4_rows_{job}.csv")
    rows = pd.read_csv(fn).to_dict("records") if os.path.exists(fn) else []
    done = {(r["part"], r["cell"], r["eps"], r["m"], r["rep"]) for r in rows}
    Xw, yw, _, _ = generate(np.random.default_rng(0), 80, 2, "cross", 9.0, eps=0.1)
    methods(Xw, yw, 2, 8, 0.1, np.random.default_rng(0), "cross", 9.0)
    tasks = [("a", c, e, c[5]) for c in CELLS for e in EPS] + \
            [("b", CELLS[0], e, mm) for e in (0.10, 0.20) for mm in MWIN]
    tasks = tasks[job::njobs]
    for (part, (lab, K, geom, delta, n, m0, extra), eps, m) in tasks:
        reps = extra.get("reps", REPS) if part == "a" else REPS // 2
        for r in range(reps):
            key = (part, lab, eps, m, r)
            if key in done:
                continue
            X, y, z0, out = make_data(np.random.default_rng(seed_for("E4data", lab, eps, r)),
                                      lab, K, geom, delta, n, eps, extra)
            X = np.ascontiguousarray(X)
            rng = np.random.default_rng(seed_for("E4", part, lab, eps, m, r))
            cl = ~out
            if part == "a":
                res = methods(X, y, K, m, eps, rng, geom, delta)
            else:
                res = {}
                t = time.perf_counter()
                f = E.eesp(X, y, K, m, B, rng)
                res["vote"] = (f.z, f.theta, np.nan, time.perf_counter() - t)
                g = E.eesp(X, y, K, m, B, rng, agg="median")
                res["median"] = (g.z, g.theta, np.nan, np.nan)
            for meth, (z, th, ah, tm) in res.items():
                rows.append(dict(part=part, cell=lab, K=K, n=n, eps=eps, m=m, rep=r,
                                 method=meth, n_out=int(out.sum()),
                                 acc_clean=E.agreement(z[cl], z0[cl], K),
                                 ari_clean=E.ari(z[cl], z0[cl]),
                                 perr=perr(th, K, geom, delta), alpha_hat=ah, time=tm,
                                 p_clean_sub=(1 - eps) ** m))
            if r % 20 == 19:
                pd.DataFrame(rows).to_csv(fn, index=False)
        pd.DataFrame(rows).to_csv(fn, index=False)
        print(time.strftime("%H:%M:%S"), part, lab, eps, m, flush=True)


if __name__ == "__main__":
    main()
