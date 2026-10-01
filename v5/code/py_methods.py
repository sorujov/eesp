"""Python methods for the v4 study.

adapt   : Algorithm 2 of v3 (eesp_variants.eesp_adaptive), unchanged.
adaptX  : Algorithm 2 with covariate trimming (W2): at every stage, and in the
          final step, a unit is also flagged when its covariates are outlying
          within its current group (robust distance above the 0.975 chi-square
          quantile), in the manner of the second trimming of TCLUST-REG.
TA(a)   : multistart trimmed alternation at fixed level a + reweighting,
          given the wall time of adapt (home-made; supplement only).
HA50    : least-squares hard alternation, 50 starts (no protection).
"""
import time

import numpy as np
from scipy.stats import chi2

import eespcore as E
import eesp_variants as V


def _xflag(X, z, K, q=0.975, min_group=10):
    """Covariate outlyingness within groups.  p = 1: median/MAD; p > 1: MCD."""
    Z = X[:, np.ptp(X, axis=0) > 0]      # the covariates (drop the intercept column, if any)
    n, p = Z.shape
    cut = chi2.ppf(q, p)
    fl = np.zeros(n, dtype=bool)
    for k in range(K):
        mk = np.where(z == k)[0]
        if len(mk) < max(min_group, 5 * p):
            continue
        Zk = Z[mk]
        if p == 1:
            med = np.median(Zk[:, 0])
            s = 1.4826 * np.median(np.abs(Zk[:, 0] - med))
            if s <= 0:
                continue
            d2 = ((Zk[:, 0] - med) / s) ** 2
        else:
            from sklearn.covariance import MinCovDet
            mcd = MinCovDet(random_state=0).fit(Zk)
            d2 = mcd.mahalanobis(Zk)
        fl[mk] = d2 > cut
    return fl


def _flag_all(X, y, theta, c, xtrim, xq=0.975):
    fl, s, z = V._flag(X, y, theta, c)
    if xtrim:
        fl = fl | _xflag(X, z, theta.shape[0], q=xq)
    return fl, s, z


def eesp_adaptive_x(X, y, K, m, B, rng, stages=10, lam=3.0, c=2.5, c_bound=3.0,
                    xtrim=True, xq=0.975, scale_q=None):
    """Algorithm 2 with covariate trimming; identical to
    eesp_variants.eesp_adaptive(select='best', csteps=True) when xtrim=False."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    t0 = time.perf_counter()
    n, d = X.shape
    Bs = [B // stages + (1 if s < B % stages else 0) for s in range(stages)]
    thetas = np.empty((0, K, d))
    R = np.empty((0, n))
    probs, ref = None, None
    flagged = np.zeros(n, dtype=bool)
    if xtrim:
        # before any fit, covariates are screened as one group
        flagged = _xflag(X, np.zeros(n, dtype=int), 1, q=xq)
        probs = (~flagged).astype(float)
        probs /= probs.sum()
    for Bb in Bs:
        th, ref = V.collect(X, y, K, m, Bb, rng, probs=probs, ref=ref)
        thetas = np.concatenate([thetas, th])
        Rb = np.empty((Bb, n))
        for b in range(Bb):
            Rb[b], _ = E._min_resid(X, y, th[b])
        R = np.concatenate([R, Rb])
        keep = ~flagged if xtrim else np.ones(n, dtype=bool)
        # guard for data in which a group lies exactly on a line (e.g. a fixed tariff): the
        # median is taken over the units a replicate does not fit exactly, so that the scale
        # is not zero; with continuous errors no residual is exactly zero and nothing changes
        Rk = R[:, keep]
        tol = 1e-12 * float(np.var(y))
        pos = Rk > tol
        if pos.all() and scale_q is not None:
            # round-9 sensitivity (descriptive): a quantile of the replicates' medians instead
            # of their minimum, which is biased downwards and more so for larger B
            s2 = float(np.quantile(np.median(Rk, axis=1), scale_q)) / 0.4549
        elif pos.all():
            s2 = np.min(np.median(Rk, axis=1)) / 0.4549
        else:
            med = np.array([np.median(r[q]) if q.any() else np.inf for r, q in zip(Rk, pos)])
            s2 = max(float(np.min(med)), tol) / 0.4549
        F = np.minimum(R[:, keep], c_bound * c_bound * s2).sum(axis=1)
        w = np.exp(-lam * (F - F.min()) / (keep.sum() * s2)) if lam > 0 else np.ones(len(F))
        th_ref = thetas[int(np.argmin(F))]
        flagged, s, _ = _flag_all(X, y, th_ref, c, xtrim, xq)
        probs = (~flagged).astype(float)
        if probs.sum() < m + 1:
            probs = np.ones(n)
        probs /= probs.sum()
    z, Vv = V.weighted_vote(X, y, thetas, w, rng)
    theta, cnt = V._refit_unflagged(X, y, z, flagged, K)
    for k in range(K):
        if cnt[k] < d:
            theta[k] = th_ref[k]
    h = n - int(flagged.sum())
    theta, z, kept, v = E._cstep(X, y, K, theta.copy(), h, 200)
    flagged, s, z = _flag_all(X, y, theta, c, xtrim, xq)
    theta2, cnt = V._refit_unflagged(X, y, z, flagged, K)
    for k in range(K):
        if cnt[k] >= d:
            theta[k] = theta2[k]
    flagged, _, z = _flag_all(X, y, theta, c, xtrim, xq)
    return theta, float(flagged.mean()), time.perf_counter() - t0


def reweight(X, y, theta, c=2.5, xtrim=False, steps=2, xq=0.975):
    """The reweighting step shared by Algorithm 2 and the TA/TCLUST-REG + rw
    competitors: flag at c robust scales (and in the covariates if xtrim),
    refit on the unflagged units."""
    K, d = theta.shape
    theta = theta.copy()
    for _ in range(steps):
        fl, _, z = _flag_all(X, y, theta, c, xtrim, xq)
        th2, cnt = V._refit_unflagged(X, y, z, fl, K)
        for k in range(K):
            if cnt[k] >= d:
                theta[k] = th2[k]
    fl, _, _ = _flag_all(X, y, theta, c, xtrim, xq)
    return theta, float(fl.mean())


def ta_timed(X, y, K, alpha, budget, rng):
    t0 = time.perf_counter()
    best, R = None, 0
    while R == 0 or time.perf_counter() - t0 < budget:
        f = E.trimmed_alternation(X, y, K, 5, alpha, rng)
        R += 5
        if best is None or f.F < best.F:
            best = f
    theta, ah = reweight(X, y, best.theta)
    return theta, ah, time.perf_counter() - t0


TA_LEVELS = [0.05, 0.10, 0.15, 0.25, 0.40]


def run_all(X, y, K, m, rng, B=100, with_ta=True):
    """Returns {method: (theta, alpha_hat, time)}."""
    res = {}
    th, ah, t = eesp_adaptive_x(X, y, K, m, B, rng, xtrim=False)
    res["adapt"] = (th, ah, t)
    th, ah, t = eesp_adaptive_x(X, y, K, m, B, rng, xtrim=True)
    res["adaptX"] = (th, ah, t)
    t0 = time.perf_counter()
    z, th, _, _ = E.hard_alternation(X, y, K, 50, rng)
    res["HA50"] = (th, np.nan, time.perf_counter() - t0)
    if with_ta:
        budget = res["adapt"][2]
        for a in TA_LEVELS:
            th, ah, t = ta_timed(X, y, K, a, budget, rng)
            res[f"TA({a:.2f})"] = (th, ah, t)
    return res


def noise_mix(X, y, K, rng, starts=10, short=10, max_iter=1000, tol=1e-9, restr=12.0,
              init="partition", z_init=None):
    """Mixture of K Gaussian linear regressions plus a uniform noise component for the
    response (Banfield & Raftery 1993, in regression form): p(y|x) = sum_k pi_k N(y; x'b_k,
    s_k^2) + pi_0 / R, with R 1.1 times the range of y.  emEM: `starts` random-partition
    starts run for `short` iterations, the best continued to convergence.  Returns
    (theta (K,d), fraction of units whose posterior of the noise component exceeds 1/2, time)."""
    t0 = time.perf_counter()
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, d = X.shape
    R = 1.1 * (y.max() - y.min())
    vfloor = 1e-6 * np.var(y)

    def mstep(W):
        th = np.empty((K, d)); s2 = np.empty(K)
        for k in range(K):
            w = W[:, k] + 1e-12
            Xw = X * w[:, None]
            A = X.T @ Xw + 1e-9 * np.eye(d)
            th[k] = np.linalg.solve(A, Xw.T @ y)
            e = y - X @ th[k]
            s2[k] = max(float(w @ e ** 2) / float(w.sum()), vfloor)
        # variance-ratio restriction, as in TCLUST-REG (factor `restr`), to avoid
        # degenerate components
        s2 = np.maximum(s2, s2.max() / restr)
        return th, s2

    def estep(th, s2, pis, p0):
        E = y[:, None] - X @ th.T
        dens = pis[None, :] * np.exp(-0.5 * E ** 2 / s2[None, :]) / np.sqrt(2 * np.pi * s2[None, :])
        d0 = np.full(n, p0 / R)
        tot = dens.sum(axis=1) + d0
        tot = np.maximum(tot, 1e-300)
        return dens / tot[:, None], d0 / tot, float(np.log(tot).sum())

    def run(th, s2, pis, p0, iters):
        ll_old = -np.inf
        for _ in range(iters):
            Wk, W0, ll = estep(th, s2, pis, p0)
            th, s2 = mstep(Wk)
            pis = Wk.mean(axis=0); p0 = float(W0.mean())
            if abs(ll - ll_old) < tol * max(1.0, abs(ll)):
                break
            ll_old = ll
        Wk, W0, ll = estep(th, s2, pis, p0)
        return th, s2, pis, p0, ll, W0

    best = None
    for st in range(starts + (z_init is not None)):
        if st == starts:
            # an extra start from a given partition (e.g. least squares), after the usual ones
            z = np.asarray(z_init)
        elif init == "partition" or (init == "mixed" and st % 2 == 0):
            # random partition (the flexmix default)
            z = rng.integers(0, K, n)
        else:
            # elemental start: each line fitted to d random units, units assigned to the nearest
            th0 = np.empty((K, d))
            for k in range(K):
                S = rng.choice(n, size=d, replace=False)
                th0[k] = np.linalg.lstsq(X[S], y[S], rcond=None)[0]
            z = np.argmin((y[:, None] - X @ th0.T) ** 2, axis=1)
        W = np.zeros((n, K)); W[np.arange(n), z] = 1.0
        th, s2 = mstep(W)
        out = run(th, s2, np.full(K, 0.9 / K), 0.1, short)
        if best is None or out[4] > best[4]:
            best = out
    th, s2, pis, p0, ll, W0 = run(*best[:4], max_iter)
    return th, float((W0 > 0.5).mean()), time.perf_counter() - t0
