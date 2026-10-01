"""Development of a group-recovery step for Algorithm 1 (after Gate 1e).

Idea: Algorithm 1 flags units that its fit cannot explain.  If a sizeable part of the
flagged set lies close to one line, with a scale comparable to the fitted groups and a
covariate spread comparable to the data, it is a candidate group, not contamination.  The
candidate line is added to the K fitted lines, every choice of K lines out of the K + 1 is
refitted by the reweighting step, and the choice with the smallest flagged fraction is kept
(the original fit competes as well).
"""
import itertools

import numpy as np

import eespcore as E
import eesp_variants as V
import py_methods as P

TRUNC = 0.1426   # variance of N(0,1) truncated to its central 50%


def line_lts(X, y, rng, draws=300, csteps=20):
    """Least trimmed squares line (h = half) by elemental starts and concentration."""
    n, d = X.shape
    h = max(d + 1, (n + 1) // 2)
    best = (np.inf, None)
    for _ in range(draws):
        S = rng.choice(n, d, replace=False)
        try:
            b = np.linalg.solve(X[S], y[S])
        except np.linalg.LinAlgError:
            continue
        for _ in range(csteps):
            r2 = (y - X @ b) ** 2
            keep = np.argsort(r2)[:h]
            b_new, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
            if np.allclose(b_new, b):
                break
            b = b_new
        f = np.sort((y - X @ b) ** 2)[:h].sum()
        if f < best[0]:
            best = (f, b)
    b = best[1]
    s = np.sqrt(best[0] / h / TRUNC)
    return b, s


def line_consensus(X, y, thr, rng, draws=500, iters=3):
    """RANSAC line: the elemental line with most units within thr, refitted on them."""
    n, d = X.shape
    best = (-1, None)
    for _ in range(draws):
        S = rng.choice(n, d, replace=False)
        try:
            b = np.linalg.solve(X[S], y[S])
        except np.linalg.LinAlgError:
            continue
        cnt = int((np.abs(y - X @ b) <= thr).sum())
        if cnt > best[0]:
            best = (cnt, b)
    b = best[1]
    for _ in range(iters):
        inl = np.abs(y - X @ b) <= thr
        if inl.sum() <= d:
            break
        b, *_ = np.linalg.lstsq(X[inl], y[inl], rcond=None)
    return b


def group_scales(X, y, theta):
    r, z = E._min_resid(X, y, theta)
    K = theta.shape[0]
    s = np.array([V.robust_scale(r[z == k]) if (z == k).sum() >= 10 else V.robust_scale(r)
                  for k in range(K)])
    return s


def recover(X, y, theta, rng, c=2.5, xq=0.975, min_frac=0.03, restr=12.0, xspread=0.25,
            rho=1.5, tight=99.0, contrast=10.0, redundant=0.6, min_alpha=0.01, max_alpha=1/3, min_gain=0.05, verbose=False):
    """Returns (theta, flagged fraction, info)."""
    n, d = X.shape
    K = theta.shape[0]
    fl, _, z = P._flag_all(X, y, theta, c, True, xq)
    base_a = float(fl.mean())
    info = dict(tried=False, accepted=False)
    # a missing group has ordinary covariates: units flagged by the covariate screen are left out
    # (the screen is applied to all units as one group: within the fitted groups, the units of a
    # missing group can look outlying in their covariates)
    xf = P._xflag(X, np.zeros(n, dtype=int), 1, q=xq)
    F = np.where(fl & ~xf)[0]
    # when a third of the data or more is flagged, the fit itself is in doubt (the procedure
    # breaks down beyond about 30% of outliers), and no group is sought among the flagged units
    if base_a > max_alpha:
        info["reason"] = "too many flagged"
        return theta, base_a, info
    if len(F) < max(2 * (d + 1), int(np.ceil(min_frac * n))):
        return theta, base_a, info
    s_k = group_scales(X, y, theta)
    info["tried"] = True
    thr = c * np.median(s_k)
    b = line_consensus(X[F], y[F], thr, rng)
    s_new = 0.0
    S = F[np.abs(y[F] - X[F] @ b) <= thr]
    if len(S) < max(2 * (d + 1), int(np.ceil(min_frac * n))):
        info["reason"] = "support"
        return theta, base_a, info
    # a group stands out from its surroundings: the band around the line must hold at least
    # `contrast` times as many flagged units as parallel bands beside it
    rF = y[F] - X[F] @ b
    side = np.mean([np.sum(np.abs(rF - o) <= thr) for o in (-4 * thr, -2 * thr, 2 * thr, 4 * thr)])
    info["contrast"] = len(S) / max(side, 0.5)
    if len(S) < contrast * max(side, 0.5):
        info["reason"] = "contrast"
        return theta, base_a, info
    # a group is concentrated around its line; points scattered through the band are not:
    # uniform residuals in [-thr, thr] have robust scale 0.74 thr = 1.85 s, a Gaussian group about s
    s_cand = V.robust_scale((y[S] - X[S] @ b) ** 2)
    info["s_ratio"] = s_cand / np.median(s_k)
    if s_cand > tight * np.median(s_k):
        info["reason"] = "loose"
        return theta, base_a, info
    cov = [j for j in range(d) if np.ptp(X[:, j]) > 0]
    # a missing group lives where the data live: most of its units must have covariates inside
    # the range of the units the fit explains (leverage points lie outside it)
    ok = ~fl
    inside = np.all([(X[S, j] >= X[ok, j].min()) & (X[S, j] <= X[ok, j].max()) for j in cov], axis=0)
    if inside.mean() < 0.5:
        info["reason"] = "outside"
        return theta, base_a, info
    if any(X[S, j].std() < xspread * X[:, j].std() for j in cov):
        info["reason"] = "spread"
        return theta, base_a, info
    lines = np.vstack([theta, b[None, :]])

    def outside(th, t):
        r, _ = E._min_resid(X, y, th)
        return float((np.sqrt(r) > t).mean())

    def refit_excluding(th, excl, steps=3):
        # the reweighting step, with the units of the global covariate screen kept out of every
        # group, so that a candidate cannot turn a cloud of leverage points into a group
        th = th.copy()
        for _ in range(steps):
            f, _, zz = P._flag_all(X, y, th, c, True, xq)
            th2, cnt = V._refit_unflagged(X, y, zz, f | excl, K)
            for k in range(K):
                if cnt[k] >= d:
                    th[k] = th2[k]
        return th

    def spread_ok(th):
        # every group must be a line over the covariate range, not a tight cloud, and its
        # units within the threshold must be concentrated around it, not spread over the band
        r, zz = E._min_resid(X, y, th)
        inl = np.sqrt(r) <= thr
        for k in range(K):
            mk = inl & (zz == k)
            if mk.sum() <= d:
                return False
            if any(X[mk, j].std() < xspread * X[:, j].std() for j in cov):
                return False
            if V.robust_scale(r[mk]) > tight * np.median(s_k):
                return False
        return True

    # candidates compete on a common threshold, fixed from the original fit
    best = (0.0, theta)   # gain over the original fit
    r_old, z_old = E._min_resid(X, y, theta)
    inl_old = np.sqrt(r_old) <= thr
    for drop in range(K + 1):
        cand = np.delete(lines, drop, axis=0)
        if drop < K:
            # only a redundant line may go: most of its units must be within the threshold of
            # one of the lines that remain, as when two lines share one group
            mk = inl_old & (z_old == drop)
            rr = np.min(np.abs(y[mk, None] - X[mk] @ cand.T), axis=1)
            if mk.sum() and np.mean(rr <= thr) < redundant:
                info.setdefault("log", []).append((drop, "redundant", round(float(np.mean(rr <= thr)), 2)))
                continue
        th = refit_excluding(cand, xf)
        if not spread_ok(th):
            info.setdefault("log", []).append((drop, "spread"))
            continue
        # the original and the candidate are compared at the smaller of their two thresholds
        t = c * min(np.median(s_k), np.median(group_scales(X, y, th)))
        # a fit that flags almost nothing has absorbed the contamination: below the rate
        # 2 Phi(-c) at which Gaussian errors alone are flagged, the candidate is refused
        a_new = float(P._flag_all(X, y, th, c, True, xq)[0].mean())
        if a_new < min_alpha:
            info.setdefault("log", []).append((drop, "alpha", round(a_new, 3)))
            continue
        gain = outside(theta, t) - outside(th, t)
        info.setdefault("log", []).append((drop, "gain", round(gain, 3)))
        # the recovered group must explain at least min_gain n more units than the original fit
        if gain >= min_gain and gain > best[0] + 1e-12:
            best = (gain, th)
    info["accepted"] = best[1] is not theta
    if not info["accepted"]:
        return theta, base_a, info
    fl, _, _ = P._flag_all(X, y, best[1], c, True, xq)
    return best[1], float(fl.mean()), info


def algorithm1_r(X, y, K, m, B, rng, rounds=3, **kw):
    th, ah, t = P.eesp_adaptive_x(X, y, K, m, B, rng, xtrim=True)
    th2, ah2, info = th, ah, dict(tried=False, accepted=False)
    for _ in range(rounds):
        th3, ah3, inf = recover(X, y, th2, rng, **kw)
        if not inf["accepted"]:
            break
        th2, ah2 = th3, ah3
        info = inf
    return th2, ah2, info, th, ah
