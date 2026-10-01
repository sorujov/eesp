"""Mixture of t regressions (Yao, Wei and Yu 2014), round-9 competitor (descriptive).
EM with the scale-mixture representation; a common degrees-of-freedom parameter chosen at every
iteration from a grid by the observed likelihood; group variances with ratio bounded by 12, as
for the other mixtures of the study; 20 starts, each from K random elemental lines."""
import numpy as np
from scipy.special import gammaln

NU_GRID = np.array([1, 1.5, 2, 3, 4, 6, 8, 12, 20, 35, 60, 100], float)


def _logt(r, s, nu):
    z2 = (r / s) ** 2
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(s)
            - (nu + 1) / 2 * np.log1p(z2 / nu))


def _bound(s2, ratio=12.0):
    lo = s2.max() / ratio
    return np.maximum(s2, lo)


def fit_once(X, y, K, rng, iters=200, tol=1e-7):
    n, d = X.shape
    th = np.empty((K, d))
    for k in range(K):
        for _ in range(20):
            S = rng.choice(n, d, replace=False)
            try:
                th[k] = np.linalg.solve(X[S], y[S]); break
            except np.linalg.LinAlgError:
                continue
    R = y[:, None] - X @ th.T
    s2 = np.full(K, np.median(np.min(R ** 2, axis=1)) / 0.4549 + 1e-12)
    pi = np.full(K, 1.0 / K); nu = 4.0; ll_old = -np.inf
    for it in range(iters):
        R = y[:, None] - X @ th.T
        s = np.sqrt(s2)
        L = np.log(pi)[None, :] + _logt(R, s[None, :], nu)
        m = L.max(axis=1, keepdims=True)
        ll = float((m[:, 0] + np.log(np.exp(L - m).sum(axis=1))).sum())
        tau = np.exp(L - m); tau /= tau.sum(axis=1, keepdims=True)
        u = (nu + 1) / (nu + (R / s[None, :]) ** 2)
        for k in range(K):
            w = tau[:, k] * u[:, k]
            if w.sum() < 1e-8:
                continue
            Xw = X * w[:, None]
            try:
                th[k] = np.linalg.solve(X.T @ Xw + 1e-10 * np.eye(d), Xw.T @ y)
            except np.linalg.LinAlgError:
                pass
            r = y - X @ th[k]
            s2[k] = (w * r * r).sum() / max(tau[:, k].sum(), 1e-8)
        s2 = _bound(s2 + 1e-12)
        pi = np.clip(tau.mean(axis=0), 1e-6, None); pi /= pi.sum()
        R = y[:, None] - X @ th.T; s = np.sqrt(s2)
        best = (-np.inf, nu)
        for v in NU_GRID:
            Lv = np.log(pi)[None, :] + _logt(R, s[None, :], v)
            mv = Lv.max(axis=1, keepdims=True)
            lv = float((mv[:, 0] + np.log(np.exp(Lv - mv).sum(axis=1))).sum())
            if lv > best[0]:
                best = (lv, v)
        nu = best[1]
        if abs(best[0] - ll_old) < tol * (1 + abs(best[0])):
            ll_old = best[0]; break
        ll_old = best[0]
    return th, ll_old, nu


def tmix(X, y, K, rng, starts=20):
    best = (None, -np.inf, None)
    for _ in range(starts):
        th, ll, nu = fit_once(X, y, K, rng)
        if np.isfinite(ll) and ll > best[1]:
            best = (th.copy(), ll, nu)
    return best
