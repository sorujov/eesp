"""Mixture of contaminated normal regressions (Mazza and Punzo 2020, univariate response).

Group k: y | x ~ alpha_k N(x'b_k, s_k^2) + (1 - alpha_k) N(x'b_k, eta_k s_k^2), with
alpha_k >= 1/2 (the proportion of good points) and eta_k > 1 (the inflation of the bad ones).
Fitted by ECM; a unit is flagged (bad) when its posterior probability of being bad exceeds 1/2.
No trimming level: the proportions of bad points are estimated.  Our implementation (no package
for the regression version is on CRAN); emEM starts from random partitions as for the noise
mixture, and the ratio of the group variances is bounded by 12 as in TCLUST-REG.
"""
import time

import numpy as np

LOG2PI = np.log(2 * np.pi)


def _dens(r2, s2, alpha, eta):
    g = np.exp(-0.5 * r2 / s2) / np.sqrt(s2)
    b = np.exp(-0.5 * r2 / (eta * s2)) / np.sqrt(eta * s2)
    return alpha * g / np.sqrt(2 * np.pi), (1 - alpha) * b / np.sqrt(2 * np.pi)


def _em(X, y, K, z0, iters, restr, alpha_min, tol=1e-9, eta0=4.0, V0=None):
    n, d = X.shape
    Z = np.zeros((n, K))
    Z[np.arange(n), z0] = 1.0
    V = np.ones((n, K)) if V0 is None else np.repeat(V0[:, None], K, axis=1)
    eta = np.full(K, eta0)
    alpha = np.full(K, 0.9)
    b = np.zeros((K, d))
    s2 = np.ones(K)
    pi = Z.mean(0) + 1e-12
    ll_old = -np.inf
    ll = -np.inf
    for it in range(iters):
        # CM-step 1
        pi = np.maximum(Z.mean(0), 1e-6)
        for k in range(K):
            w = Z[:, k] * (V[:, k] + (1 - V[:, k]) / eta[k]) + 1e-12
            Xw = X * w[:, None]
            try:
                b[k] = np.linalg.solve(X.T @ Xw + 1e-10 * np.eye(d), Xw.T @ y)
            except np.linalg.LinAlgError:
                pass
            r2 = (y - X @ b[k]) ** 2
            s2[k] = max((w * r2).sum() / max(Z[:, k].sum(), 1e-12), 1e-8)
            zk = Z[:, k].sum()
            alpha[k] = min(max((Z[:, k] * V[:, k]).sum() / max(zk, 1e-12), alpha_min), 0.9999)
            bad = (Z[:, k] * (1 - V[:, k])).sum()
            if bad > 1e-8:
                eta[k] = max((Z[:, k] * (1 - V[:, k]) * r2).sum() / (s2[k] * bad), 1.001)
        s2 = np.maximum(s2, s2.max() / restr)
        # E-step
        R2 = (y[:, None] - X @ b.T) ** 2
        G, Bd = _dens(R2, s2[None, :], alpha[None, :], eta[None, :])
        F = pi[None, :] * (G + Bd)
        tot = F.sum(1) + 1e-300
        ll = np.log(tot).sum()
        Z = F / tot[:, None]
        V = G / (G + Bd + 1e-300)
        if abs(ll - ll_old) < tol * max(1.0, abs(ll)):
            break
        ll_old = ll
    return dict(b=b.copy(), ll=ll, Z=Z, V=V)


def cn_mix(X, y, K, rng, starts=50, short=10, max_iter=1000, restr=12.0, alpha_min=0.5, eta0=20.0, init="elemental",
           z_init=None, th_init=None):
    """Returns (theta, flagged fraction, time)."""
    t0 = time.perf_counter()
    n = len(y)
    best = None
    d = X.shape[1]
    for j in range(starts):
        if init == "elemental" or (init == "mixed" and j % 2 == 1):
            # K random lines through d units each; units to their nearest line
            th = np.array([np.linalg.lstsq(X[S], y[S], rcond=None)[0]
                           for S in [rng.choice(n, d, replace=False) for _ in range(K)]])
            r = np.min(np.abs(y[:, None] - X @ th.T), axis=1)
            z0 = np.argmin((y[:, None] - X @ th.T) ** 2, axis=1)
            # units far from every starting line start as bad points
            V0 = np.where(r <= 2.5 * 1.4826 * np.median(r), 1.0, 0.0)
        else:
            z0 = rng.integers(0, K, n)
            V0 = None
        f = _em(X, y, K, z0, short, restr, alpha_min, eta0=eta0, V0=V0)
        if best is None or f["ll"] > best[0]["ll"]:
            best = (f, z0, V0)
    if z_init is not None:
        # an extra start from a given partition and its lines (e.g. least squares)
        r = np.min(np.abs(y[:, None] - X @ np.asarray(th_init).T), axis=1)
        V0 = np.where(r <= 2.5 * 1.4826 * np.median(r), 1.0, 0.0)
        z0 = np.asarray(z_init)
        f = _em(X, y, K, z0, short, restr, alpha_min, eta0=eta0, V0=V0)
        if f["ll"] > best[0]["ll"]:
            best = (f, z0, V0)
    f = _em(X, y, K, best[1], max_iter, restr, alpha_min, eta0=eta0, V0=best[2])
    bad = (f["Z"] * (1 - f["V"])).sum(1) > 0.5
    return f["b"], float(bad.mean()), time.perf_counter() - t0
