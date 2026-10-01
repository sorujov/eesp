"""Sequential RANSAC (descriptive competitor from multi-model fitting, added after the external
reading): K lines extracted one at a time, each the elemental line (500 draws) with most remaining
units within 2.5 true noise scales, refitted three times by least squares on those units, which
are then removed.  The true noise scale is given to it, an advantage no other method has.
Usage: V4_STUDY=... python seqransac.py OUT"""
import sys
from multiprocessing import Pool
import numpy as np, pandas as pd
import simv4, evaluate as EV


def seq_ransac(X, y, K, sigma, rng, draws=500, c=2.5):
    n, d = X.shape
    left = np.ones(n, dtype=bool)
    th = np.full((K, d), np.nan)
    for k in range(K):
        idx = np.where(left)[0]
        if len(idx) <= d:
            break
        best = (-1, None)
        for _ in range(draws):
            S = rng.choice(idx, d, replace=False)
            try:
                b = np.linalg.solve(X[S], y[S])
            except np.linalg.LinAlgError:
                continue
            cnt = int((np.abs(y[idx] - X[idx] @ b) <= c * sigma).sum())
            if cnt > best[0]:
                best = (cnt, b)
        b = best[1]
        for _ in range(3):
            inl = idx[np.abs(y[idx] - X[idx] @ b) <= c * sigma]
            if len(inl) <= d:
                break
            b = np.linalg.lstsq(X[inl], y[inl], rcond=None)[0]
        th[k] = b
        left[idx[np.abs(y[idx] - X[idx] @ b) <= c * sigma]] = False
    return th, float(left.mean())


def run(u):
    X, y, z0, out = simv4.generate(u)
    D = simv4.DESIGNS[u[0]]
    sigma = float(np.mean(D.get("sigmas", 1.0)))
    th, ah = seq_ransac(X, y, D["K"], sigma, np.random.default_rng(simv4.seed_for("v4sr", *u)))
    ok = np.isfinite(th).all()
    a = EV.acc_clean(X, y, th, z0, ~out) if ok else 0.0
    return dict(design=u[0], eps=u[1], rep=u[2], unit=simv4.unit_name(u), method="SeqRANSAC", acc=a,
                perr=EV.align_err(th, EV.true_theta(D)) if ok else np.nan, ahat=ah)


if __name__ == "__main__":
    with Pool(2) as p:
        R = p.map(run, simv4.units(), chunksize=20)
    pd.DataFrame(R).to_csv(sys.argv[1], index=False)
    print(len(R))
