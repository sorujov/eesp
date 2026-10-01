"""Group recovery for Algorithm 1 by a likelihood comparison (recover2, 27 Sep 2026).

Algorithm 1 flags the units its K lines cannot explain.  When a small group has been taken
for contamination, the fit typically spends two lines on one large group and flags the small
one.  This step looks for a line among the flagged units, adds it to the K fitted lines and
asks which of the K + 1 lines should go.

The rule, in three sentences.
1. Candidate.  A RANSAC line is fitted to the flagged units (those of the global covariate
   screen excluded) at the threshold c s, with s the median robust scale of the fitted
   groups; it is a candidate group if, among the flagged units within 3 c s of it, the
   majority belong to a peak of scale s rather than to a uniform spread over that band (the
   local two-component model w N(0, s^2) + (1 - w) Uniform, fitted in w; w >= 1/2).  A slice
   of a band of gross outliers, which RANSAC can always find, has w near 1/3.
2. Score.  Every K-line fit is scored by the log-likelihood of the model behind the flagging
   rule: K Gaussian lines with their own scales and proportions plus a uniform noise
   component on the range of y (the noise component of Banfield and Raftery, the spurious
   outliers of TCLUST), with the parameters read off the fit itself (lines; robust group
   scales; proportions of the unflagged units; the flagged fraction as the noise weight;
   units of the covariate screen always in the noise).  All fits compared have K lines, so
   no model-size penalty is needed; but the candidate line was searched for, so a candidate
   fit must beat the original by the BIC cost of one line, (d + 1)/2 log n.
3. Admissibility.  The candidate fits are the candidate line plus all but one of the
   original lines, each refitted by the reweighting step; a fit is admissible only if all its
   group scales are within the TCLUST factor sqrt(12) of the smallest scale of the original
   fit (the noise scale established by Algorithm 1).  The best admissible candidate replaces
   the original when it passes the margin; the step is repeated up to three times.

Why the likelihood separates the cases.  Two lines through one Gaussian group give almost the
same mixture density as one line (their mixture is nearly the group's own distribution), so
un-splitting is free, while explaining a flagged group of n_1 units gains about
n_1 (log R - log s - 1.4 + log pi_1) nats.  Dropping a line that carries a group of its own
sends the group to the noise component or to a widened neighbour, which costs at least the
Kullback-Leibler distance between the group and its new description.  What the likelihood
cannot see is a slice of a dense band of gross outliers, which it values like a group; that is
the job of the local peak test and of the scale restriction.
"""
import numpy as np
from scipy.stats import norm

import eespcore as E
import eesp_variants as V
import py_methods as P

LOG2PI = float(np.log(2 * np.pi))


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
    if b is None:
        return None
    for _ in range(iters):
        inl = np.abs(y - X @ b) <= thr
        if inl.sum() <= d:
            break
        b, *_ = np.linalg.lstsq(X[inl], y[inl], rcond=None)
    return b


def fit_params(X, y, theta, c=2.5, xq=0.975, min_group=10):
    """Parameters of the Gaussian-lines-plus-uniform model read off a fit: flags, labels,
    group scales (robust, from the unflagged units of each group; the pooled scale for groups
    below min_group units), proportions of the unflagged units, flagged fraction."""
    n = len(y)
    K = theta.shape[0]
    fl, s_pool, z = P._flag_all(X, y, theta, c, True, xq)
    r, _ = E._min_resid(X, y, theta)
    s = np.full(K, s_pool)
    nk = np.zeros(K)
    for k in range(K):
        mk = (z == k) & ~fl
        nk[k] = mk.sum()
        if nk[k] >= min_group:
            sk = V.robust_scale(r[mk])
            if sk > 0:
                s[k] = sk
    return fl, z, s, nk / n, float(fl.mean())


def loglik(X, y, theta, R, c=2.5, xq=0.975, excl=None):
    """Log-likelihood of K Gaussian lines + uniform noise on a range R, at the parameters
    read off the fit.  Units in `excl` (the global covariate screen) are noise in every fit.
    Returns (log-likelihood, flags, group scales)."""
    fl, z, s, pi, pi0 = fit_params(X, y, theta, c, xq)
    Ez = y[:, None] - X @ theta.T
    logd = np.log(np.maximum(pi, 1e-300))[None, :] - np.log(s)[None, :] - 0.5 * LOG2PI \
        - 0.5 * Ez ** 2 / s[None, :] ** 2
    logd = np.column_stack([logd, np.full(len(y), np.log(max(pi0, 1e-300)) - np.log(R))])
    if excl is not None:
        logd[excl, :-1] = -np.inf
    m = logd.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logd - m).sum(axis=1))
    return float(lse.sum()), fl, s


def peak_weight(r, thr, s, wide=3.0, iters=50):
    """Local test of a candidate line: among the units within `wide` thr of it (residuals r),
    fit w N(0, s^2) + (1 - w) Uniform(-wide thr, wide thr) by EM in w alone.
    Returns (w, log-likelihood gain over the uniform alone, number of units)."""
    rw = r[np.abs(r) <= wide * thr]
    N = len(rw)
    if N == 0:
        return 0.0, 0.0, 0
    u = 1.0 / (2 * wide * thr)
    g = norm.pdf(rw / s) / s
    w = 0.5
    for _ in range(iters):
        t = w * g / (w * g + (1 - w) * u)
        w = float(t.mean())
    gain = float(np.log(w * g + (1 - w) * u).sum() + N * np.log(2 * wide * thr))
    return w, gain, N


def reweight_excluding(X, y, theta, excl, c=2.5, xq=0.975, steps=3):
    """The reweighting step of Algorithm 1, with the units of the global covariate screen
    kept out of every refit."""
    K, d = theta.shape
    th = theta.copy()
    for _ in range(steps):
        f, _, zz = P._flag_all(X, y, th, c, True, xq)
        th2, cnt = V._refit_unflagged(X, y, zz, f | excl, K)
        for k in range(K):
            if cnt[k] >= d:
                th[k] = th2[k]
    return th


def recover(X, y, theta, rng, c=2.5, xq=0.975, restr=12.0, min_frac=0.03, max_alpha=1 / 3,
            min_w=0.5, wide=3.0, min_ll_gain=None, noise_range="full"):
    """One recovery step.  Returns (theta, flagged fraction, info).
    noise_range="unflagged" (round-9 sensitivity, descriptive) takes the range of the response
    over the units the fit does not flag, so that the outliers do not set it; a number f multiplies
    the default range (round-12 sensitivity, descriptive)."""
    n, d = X.shape
    K = theta.shape[0]
    R = 1.1 * float(np.ptp(y))
    fl, z, s_k, pi, pi0 = fit_params(X, y, theta, c, xq)
    if noise_range == "unflagged" and (~fl).sum() > 1:
        R = 1.1 * float(np.ptp(y[~fl]))
    elif isinstance(noise_range, (int, float)) and not isinstance(noise_range, bool):
        R = float(noise_range) * R    # round 12 sensitivity: as if one outlier lay farther out
    elif noise_range == "scale20":    # round 13 (descriptive): a range tied to the fitted scales
        R = 20.0 * float(np.median(s_k))
    elif noise_range == "unflagged10" and (~fl).sum() > 1:   # round 13: unflagged range + 10 scales
        R = 1.1 * float(np.ptp(y[~fl])) + 10.0 * float(np.median(s_k))
    base_a = float(fl.mean())
    info = dict(tried=False, accepted=False)
    # beyond a third of flagged units the fit itself is in doubt (the breakdown region of the
    # procedure) and no group is sought
    if base_a > max_alpha:
        info["reason"] = "too many flagged"
        return theta, base_a, info
    # a missing group has ordinary covariates: units of the global covariate screen are left out
    xf = P._xflag(X, np.zeros(n, dtype=int), 1, q=xq)
    F = np.where(fl & ~xf)[0]
    min_support = max(2 * (d + 1), int(np.ceil(min_frac * n)))
    if len(F) < min_support:
        info["reason"] = "few flagged"
        return theta, base_a, info
    info["tried"] = True
    # the noise scale of the data, as established by the fitted groups
    s_ref = float(np.median(s_k))
    thr = c * s_ref
    b = line_consensus(X[F], y[F], thr, rng)
    if b is None:
        info["reason"] = "no line"
        return theta, base_a, info
    rF = y[F] - X[F] @ b
    if int((np.abs(rF) <= thr).sum()) < min_support:
        info["reason"] = "support"
        return theta, base_a, info
    # 1. a group, not a slice of a band: the majority of the flagged units in the neighbourhood
    #    of the line belong to a peak of the noise scale of the fitted groups
    w, lg, NW = peak_weight(rF, thr, s_ref, wide)
    info["peak_w"] = round(w, 2)
    if w < min_w:
        info["reason"] = "band"
        return theta, base_a, info
    # 2. likelihood comparison, with the BIC cost of the searched line as the margin
    if min_ll_gain is None:
        min_ll_gain = 0.5 * (d + 1) * np.log(n)
    ll0, _, _ = loglik(X, y, theta, R, c, xq, xf)
    info["ll0"] = ll0
    # 3. admissible scales: within the TCLUST factor of the smallest scale of the original fit
    s_max = float(np.sqrt(restr)) * float(s_k.min())
    lines = np.vstack([theta, b[None, :]])
    best = (ll0 + min_ll_gain, None)
    log = []
    for drop in range(K):
        th = reweight_excluding(X, y, np.delete(lines, drop, axis=0), xf, c, xq)
        ll, f, s = loglik(X, y, th, R, c, xq, xf)
        log.append((drop, round(ll - ll0, 1), round(float(s.max() / s_k.min()), 2)))
        if s.max() <= s_max and ll > best[0]:
            best = (ll, th)
    info["log"] = log
    info["gain"] = max(g for _, g, _ in log)
    info["accepted"] = best[1] is not None
    if not info["accepted"]:
        info["reason"] = "likelihood"
        return theta, base_a, info
    f, _, _ = P._flag_all(X, y, best[1], c, True, xq)
    return best[1], float(f.mean()), info


def algorithm1_r(X, y, K, m, B, rng, rounds=3, **kw):
    """Algorithm 1 followed by up to `rounds` recovery steps.
    Returns (theta, alpha_hat, info, theta of Algorithm 1, its alpha_hat)."""
    th, ah, t = P.eesp_adaptive_x(X, y, K, m, B, rng, xtrim=True)
    th2, ah2 = th, ah
    tried, accepted, gains = False, False, []
    info = dict(tried=False, accepted=False)
    for _ in range(rounds):
        th3, ah3, info = recover(X, y, th2, rng, **kw)
        tried = tried or info["tried"]
        if not info["accepted"]:
            break
        accepted = True
        gains.append(info["gain"])
        th2, ah2 = th3, ah3
    info["tried"], info["accepted"], info["gains"] = tried, accepted, gains
    return th2, ah2, info, th, ah
