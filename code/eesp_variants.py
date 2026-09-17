"""
eesp_variants.py -- weighted-vote and adaptive-trimming variants of EESP
(pilot study, 2026-09-16).  Import as:  import eesp_variants as V

Functions (numpy Generators throughout; X includes the intercept column):

  collect(X, y, K, m, B, rng, probs=None, ref=None)
      B admissible exact-subsample replicates, aligned to a pilot replicate
      (or to `ref`); `probs` are per-unit inclusion probabilities for the
      subsample draw (uniform if None).  Returns (thetas (B,K,d), ref).
  rep_scores(X, y, thetas, crit)
      Per-replicate full-sample scores F_b: 'ls' (least-squares criterion of
      the induced partition), 'lms' (median over units of the minimum squared
      residual), 'bounded' (sum over units of min(r_i, (3 s)^2), s the
      ensemble robust scale).
  weights(F, lam)          w_b = exp(-lam (F_b - min F) / MAD(F)).
  weighted_vote(X, y, thetas, w, rng)
      Weighted plurality vote of the plug-in classifications; returns (z, V).
  eesp_weighted(X, y, K, m, B, rng, lam, crit)
      Weighted-vote EESP (lam = 0 plain vote, lam -> inf selection).
      Finding: with crit='ls' this never improves on vote+polish / best-of-B
      +polish on clean data and collapses like least squares under gross
      contamination; with a robust crit it is accurate in labels but its
      least-squares refit is biased (no trimming).  Kept for reference.
  eesp_adaptive(X, y, K, m, B, rng, stages=10, lam=3, c=2.5)
      THE RECOMMENDED VARIANT: adaptive trimming via sequential ensemble
      subsampling.  No trimming level is supplied; alpha_hat is estimated
      from the ensemble (see the function docstring).
  ta_adaptive(X, y, K, R, rng, c=2.5, alpha0=0.25)
      Reference competitor: trimmed alternation at a fixed conservative
      alpha0 followed by the same reweighting step (RTCLUST-type).
"""
import time

import numpy as np

import eespcore as E


def _replicate(X, y, K, m, rng, probs=None, max_redraw=10000):
    n, d = X.shape
    for r in range(1, max_redraw + 1):
        S = rng.choice(n, size=m, replace=False, p=probs)
        XS, yS = X[S], y[S]
        lab, _, _ = E.exact_solve(XS, yS, K)
        if np.bincount(lab, minlength=K).min() < d + 1:
            continue
        th = E.refit(XS, yS, lab, K)
        if np.isfinite(th).all():
            return th, S
    raise RuntimeError("no admissible subsample")


def collect(X, y, K, m, B, rng, probs=None, ref=None):
    """B admissible replicates, aligned to a pilot replicate (or `ref`)."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    d = X.shape[1]
    if ref is None:
        # robust pilot: best of 10 replicates by the median minimum squared residual
        best = (np.inf, None)
        for _ in range(10):
            th0, _ = _replicate(X, y, K, m, rng, probs)
            rr, _ = E._min_resid(X, y, th0)
            f = float(np.median(rr))
            if f < best[0]:
                best = (f, th0)
        ref = best[1]
    thetas = np.empty((B, K, d))
    for b in range(B):
        th, _ = _replicate(X, y, K, m, rng, probs)
        thetas[b], _ = E.align(th, ref)
    return thetas, ref


def robust_scale(r):
    """1.4826 * median absolute residual, r = squared residuals."""
    return 1.4826 * float(np.sqrt(np.median(r)))


def rep_scores(X, y, thetas, crit="ls", c=3.0):
    B = thetas.shape[0]
    n = X.shape[0]
    K = thetas.shape[1]
    F = np.empty(B)
    if crit == "ls":
        for b in range(B):
            _, z = E._min_resid(X, y, thetas[b])
            F[b] = E._labels_value(X, y, z, K)
        return F
    R = np.empty((B, n))
    for b in range(B):
        R[b], _ = E._min_resid(X, y, thetas[b])
    if crit == "lms":
        return np.median(R, axis=1)
    if crit == "bounded":
        # ensemble robust scale: best LMedS replicate, median of chi2_1 = 0.4549
        s2 = np.min(np.median(R, axis=1)) / 0.4549
        return np.minimum(R, c * c * s2).sum(axis=1)
    raise ValueError(crit)


def weights(F, lam):
    F = np.asarray(F, float)
    if lam <= 0:
        return np.ones_like(F)
    scale = np.median(np.abs(F - np.median(F)))
    if scale <= 0:
        scale = F.std()
    if scale <= 0:
        return np.ones_like(F)
    w = np.exp(-lam * (F - F.min()) / scale)
    return w / w.sum() * len(F)


def weighted_vote(X, y, thetas, w, rng):
    n = X.shape[0]
    B, K, d = thetas.shape
    V = np.zeros((n, K))
    idx = np.arange(n)
    for b in range(B):
        _, z = E._min_resid(X, y, thetas[b])
        V[idx, z] += w[b]
    tie = V >= V.max(axis=1, keepdims=True)
    z = np.argmax(rng.random(V.shape) * tie, axis=1)
    return z, V / w.sum()


def eesp_weighted(X, y, K, m, B, rng, lam=3.0, crit="ls"):
    """Weighted-vote EESP.  Returns Fit(z, theta, F, V, w, thetas, time)."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    t0 = time.perf_counter()
    thetas, _ = collect(X, y, K, m, B, rng)
    F = rep_scores(X, y, thetas, crit)
    w = weights(F, lam)
    z, V = weighted_vote(X, y, thetas, w, rng)
    return E.Fit(z=z, theta=E.refit(X, y, z, K), F=E.criterion(X, y, z, K),
                 V=V, w=w, thetas=thetas, time=time.perf_counter() - t0)


def _flag(X, y, theta, c, min_group=10):
    """Flag units whose residual under their nearest surface exceeds c robust
    scale units.  The scale is estimated separately within each group
    (1.4826 times the median absolute residual of the units assigned to it),
    so that groups with different noise levels are treated alike; groups
    with fewer than min_group units use the pooled scale.  Returns
    (flags, pooled scale, labels)."""
    r, z = E._min_resid(X, y, theta)
    s_pool = robust_scale(r)
    s_unit = np.full(len(r), s_pool)
    for k in range(theta.shape[0]):
        mk = z == k
        if mk.sum() >= min_group:
            sk = robust_scale(r[mk])
            if sk > 0:
                s_unit[mk] = sk
    return np.sqrt(r) > c * s_unit, s_pool, z


def _refit_unflagged(X, y, z, flagged, K):
    zk = z.copy()
    zk[flagged] = K
    th, cnt = E._refit(X, y, zk, K + 1)
    return th[:K], cnt[:K]


def eesp_adaptive(X, y, K, m, B, rng, stages=10, lam=3.0, c=2.5, c_bound=3.0,
                  select="best", csteps=True):
    """Adaptive trimming via sequential ensemble subsampling (no alpha).

    Defaults: stages = 10 batches (B/stages = 10 replicates each for
    B = 100), lam = 3, c = 2.5 (flagging), c_bound = 3 (bounded loss).
    Robustness window (n = 300): contamination up to about 0.30 for K = 2,
    m = 8 and up to about 0.20-0.25 for K = 3, m = 12; beyond that the
    bounded criterion itself is fooled (outlier clouds become groups).

    stages : number of sequential batches of B/stages replicates.  After
             each batch, every replicate so far is scored by the bounded
             criterion F_b = sum_i min(r_ib, (c_bound s)^2), with the
             ensemble scale s^2 = min_b median_i r_ib / 0.4549; units whose
             minimum residual under the reference fit (the best-scoring
             replicate, or the weighted mean if select='wmean') exceeds c s
             are flagged; the next batch is drawn from unflagged units only.
    lam    : vote weights w_b = exp(-lam (F_b - min F) / (n s^2)); lam = 0 is
             the plain vote, lam -> infinity is selection.
    Final step: weighted vote -> least-squares refit on the unflagged units of
    each vote group -> concentration steps with h = n - #flagged -> one
    reweighting step (re-estimate s, re-flag at c s, refit on unflagged) ->
    nearest-surface assignment of every unit.
    Returns Fit(z, theta, F (trimmed value), alpha_hat, flagged, scale, V, w,
    thetas, time).  stages=1 gives the non-sequential version."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    t0 = time.perf_counter()
    n, d = X.shape
    Bs = [B // stages + (1 if s < B % stages else 0) for s in range(stages)]
    thetas = np.empty((0, K, d))
    R = np.empty((0, n))
    probs, ref = None, None
    flagged = np.zeros(n, dtype=bool)
    for s_idx, Bb in enumerate(Bs):
        th, ref = collect(X, y, K, m, Bb, rng, probs=probs, ref=ref)
        thetas = np.concatenate([thetas, th])
        Rb = np.empty((Bb, n))
        for b in range(Bb):
            Rb[b], _ = E._min_resid(X, y, th[b])
        R = np.concatenate([R, Rb])
        s2 = np.min(np.median(R, axis=1)) / 0.4549
        F = np.minimum(R, c_bound * c_bound * s2).sum(axis=1)
        w = np.exp(-lam * (F - F.min()) / (n * s2)) if lam > 0 else np.ones(len(F))
        if select == "best":
            th_ref = thetas[int(np.argmin(F))]
        else:
            th_ref = np.tensordot(w / w.sum(), thetas, axes=(0, 0))
        flagged, s, _ = _flag(X, y, th_ref, c)
        probs = (~flagged).astype(float)
        if probs.sum() < m + 1:
            probs = np.ones(n)
        probs /= probs.sum()
    z, V = weighted_vote(X, y, thetas, w, rng)
    theta, cnt = _refit_unflagged(X, y, z, flagged, K)
    for k in range(K):
        if cnt[k] < d:
            theta[k] = th_ref[k]
    h = n - int(flagged.sum())
    if csteps:
        theta, z, kept, v = E._cstep(X, y, K, theta.copy(), h, 200)
    flagged, s, z = _flag(X, y, theta, c)
    theta2, cnt = _refit_unflagged(X, y, z, flagged, K)
    for k in range(K):
        if cnt[k] >= d:
            theta[k] = theta2[k]
    flagged, _, z = _flag(X, y, theta, c)
    r, _ = E._min_resid(X, y, theta)
    v = float(np.sort(r)[: n - int(flagged.sum())].sum())
    return E.Fit(z=z, theta=theta, F=v, alpha_hat=float(flagged.mean()),
                 flagged=flagged, scale=s, V=V, w=w, thetas=thetas,
                 time=time.perf_counter() - t0)


def ta_adaptive(X, y, K, R, rng, c=2.5, alpha0=0.25):
    """Reference: trimmed alternation at a conservative alpha0 followed by the
    same reweighting step (RTCLUST-type reweighting, no ensemble)."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    t0 = time.perf_counter()
    n, d = X.shape
    f = E.trimmed_alternation(X, y, K, R, alpha0, rng)
    theta = f.theta.copy()
    for _ in range(2):
        flagged, s, z = _flag(X, y, theta, c)
        th2, cnt = _refit_unflagged(X, y, z, flagged, K)
        for k in range(K):
            if cnt[k] >= d:
                theta[k] = th2[k]
    flagged, _, z = _flag(X, y, theta, c)
    r, _ = E._min_resid(X, y, theta)
    return E.Fit(z=z, theta=theta, F=float(np.sort(r)[: n - int(flagged.sum())].sum()),
                 alpha_hat=float(flagged.mean()), flagged=flagged,
                 time=time.perf_counter() - t0)
