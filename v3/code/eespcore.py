"""
eespcore.py -- Ensemble Exhaustive-Subsample Partitioning (EESP), reference
implementation accompanying the paper.

Model.  Units u_i = (x_i, y_i) with x_i in R^d (the design row, intercept
included) and loss g(u; theta_k) = (y - x' theta_k)^2.  The location model
(k-means on a line) is the case d = 1, x_i = 1.

Contents
  * exact_solve        exact global minimiser of the clusterwise least-squares
                       criterion on a small index set (depth-first branch and
                       bound over label vectors in restricted-growth form)
  * eesp               Algorithm 1 of the paper, exactly as specified:
                       data-measurable alignment reference, fitted-label
                       feasibility rule with redraw, Hungarian alignment,
                       vote or parameter aggregation, exact refit, monotone
                       safeguard, random tie-breaking
  * best_of_B          the selection analogue (CLARA-type): same micro-solves,
                       keep the single replicate with the lowest full-sample
                       criterion
  * hard_alternation   multistart hard alternation (clusterwise k-means)
  * em_mixreg          multistart EM for Gaussian mixtures of regressions
  * oracle_k2_line     exact global minimiser of the full-sample criterion for
                       K = 2 and one covariate, O(n^3) candidates
  * oracle_location    exact global minimiser for the location model, any K,
                       by dynamic programming on the sorted sample
  * metrics            criterion, accuracy / agreement modulo relabelling, ARI

All random draws go through numpy Generators passed by the caller.
"""

import itertools
import math
import time

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Small linear-algebra kernels
# ---------------------------------------------------------------------------


@njit(cache=True)
def _group_sse(nk, XX, Xy, yy, d):
    """Residual sum of squares of the least-squares fit of a group, from its
    sufficient statistics (n_k, X'X, X'y, y'y).  Exact also for rank-deficient
    groups (ties in x, n_k <= d): there the minimum-norm solution
    beta = (X'X)^+ X'y is used, and the residual sum is y'y - beta'X'y.
    Returns (sse, beta)."""
    beta = np.zeros(d)
    if nk <= 0:
        return 0.0, beta
    if nk > d:
        # Cholesky; falls through to the pseudo-inverse if not positive definite
        L = np.zeros((d, d))
        ok = True
        for i in range(d):
            for j in range(i + 1):
                s = XX[i, j]
                for k in range(j):
                    s -= L[i, k] * L[j, k]
                if i == j:
                    if s <= 1e-10 * max(1.0, XX[i, i]):
                        ok = False
                        break
                    L[i, i] = math.sqrt(s)
                else:
                    L[i, j] = s / L[j, j]
            if not ok:
                break
        if ok:
            w = np.zeros(d)
            for i in range(d):
                s = Xy[i]
                for k in range(i):
                    s -= L[i, k] * w[k]
                w[i] = s / L[i, i]
            for i in range(d - 1, -1, -1):
                s = w[i]
                for k in range(i + 1, d):
                    s -= L[k, i] * beta[k]
                beta[i] = s / L[i, i]
            sse = yy
            for i in range(d):
                sse -= beta[i] * Xy[i]
            if sse < 0.0:
                sse = 0.0
            return sse, beta
    # rank-deficient or small group: pseudo-inverse via symmetric eigen-decomposition
    ev, V = np.linalg.eigh(XX)
    tol = 1e-10 * max(1.0, ev[d - 1])
    sse = yy
    for a in range(d):
        if ev[a] > tol:
            proj = 0.0
            for b in range(d):
                proj += V[b, a] * Xy[b]
            sse -= proj * proj / ev[a]
            for b in range(d):
                beta[b] += V[b, a] * proj / ev[a]
    if sse < 0.0:
        sse = 0.0
    # guard against round-off in exactly interpolated groups
    if sse < 1e-12 * max(1.0, yy):
        sse = 0.0
    return sse, beta


@njit(cache=True)
def _bb_exact(X, y, K, init_labels, init_value):
    """Exact minimisation of sum_i (y_i - x_i' theta_{z_i})^2 over all label
    vectors z in {0..K-1}^m, by depth-first branch and bound.

    Label vectors are enumerated in restricted-growth form (the first unit of
    each group opens it), which removes the K! relabellings.  The bound is the
    current sum of group residual sums of squares, valid because a group's
    residual sum of squares cannot decrease when a unit is added.  An incumbent
    (init_labels, init_value) may be supplied; the search then proves that no
    strictly better labelling exists or finds one.  Returns (value, labels,
    nodes)."""
    m, d = X.shape
    best = init_value
    best_lab = init_labels.copy()
    # per-group sufficient statistics
    gn = np.zeros(K, dtype=np.int64)
    gXX = np.zeros((K, d, d))
    gXy = np.zeros((K, d))
    gyy = np.zeros(K)
    gsse = np.zeros(K)
    lab = -np.ones(m, dtype=np.int64)
    used = np.zeros(m + 1, dtype=np.int64)
    saved_sse = np.zeros(m)
    total = 0.0
    nodes = 0
    i = 0
    used[0] = 0
    while i >= 0:
        # undo the current assignment of unit i, if any
        k = lab[i]
        if k >= 0:
            gn[k] -= 1
            for a in range(d):
                gXy[k, a] -= X[i, a] * y[i]
                for b in range(d):
                    gXX[k, a, b] -= X[i, a] * X[i, b]
            gyy[k] -= y[i] * y[i]
            total -= gsse[k]
            gsse[k] = saved_sse[i]
            total += gsse[k]
        k += 1
        maxlab = used[i]
        if maxlab > K - 1:
            maxlab = K - 1
        if k > maxlab:
            lab[i] = -1
            i -= 1
            continue
        lab[i] = k
        nodes += 1
        # apply assignment i -> k
        gn[k] += 1
        for a in range(d):
            gXy[k, a] += X[i, a] * y[i]
            for b in range(d):
                gXX[k, a, b] += X[i, a] * X[i, b]
        gyy[k] += y[i] * y[i]
        saved_sse[i] = gsse[k]
        s_new, _ = _group_sse(gn[k], gXX[k], gXy[k], gyy[k], d)
        total += s_new - gsse[k]
        gsse[k] = s_new
        if total >= best:
            continue  # prune; the undo happens at the top of the loop
        if i == m - 1:
            best = total
            for j in range(m):
                best_lab[j] = lab[j]
            continue
        nu = used[i]
        if k + 1 > nu:
            nu = k + 1
        used[i + 1] = nu
        lab[i + 1] = -1
        i += 1
    return best, best_lab, nodes


@njit(cache=True)
def _labels_value(X, y, z, K):
    m, d = X.shape
    tot = 0.0
    for k in range(K):
        nk = 0
        XX = np.zeros((d, d))
        Xy = np.zeros(d)
        yy = 0.0
        for i in range(m):
            if z[i] == k:
                nk += 1
                yy += y[i] * y[i]
                for a in range(d):
                    Xy[a] += X[i, a] * y[i]
                    for b in range(d):
                        XX[a, b] += X[i, a] * X[i, b]
        s, _ = _group_sse(nk, XX, Xy, yy, d)
        tot += s
    return tot


# ---------------------------------------------------------------------------
# Compiled kernels: refit, classification, alternation, EM
# ---------------------------------------------------------------------------


@njit(cache=True)
def _refit(X, y, z, K):
    """Least-squares parameters of each group (minimum-norm if rank
    deficient) and the group sizes."""
    n, d = X.shape
    theta = np.zeros((K, d))
    cnt = np.zeros(K, dtype=np.int64)
    XX = np.zeros((K, d, d))
    Xy = np.zeros((K, d))
    yy = np.zeros(K)
    for i in range(n):
        k = z[i]
        cnt[k] += 1
        yy[k] += y[i] * y[i]
        for a in range(d):
            Xy[k, a] += X[i, a] * y[i]
            for c in range(d):
                XX[k, a, c] += X[i, a] * X[i, c]
    for k in range(K):
        _, beta = _group_sse(cnt[k], XX[k], Xy[k], yy[k], d)
        theta[k] = beta
    return theta, cnt


@njit(cache=True)
def _assign(X, y, theta, z):
    """Nearest-surface assignment in place; returns the number of changes."""
    n, d = X.shape
    K = theta.shape[0]
    ch = 0
    for i in range(n):
        bk = 0
        bv = 1e300
        for k in range(K):
            f = 0.0
            for a in range(d):
                f += X[i, a] * theta[k, a]
            r = (y[i] - f) ** 2
            if r < bv:
                bv = r
                bk = k
        if bk != z[i]:
            ch += 1
            z[i] = bk
    return ch


@njit(cache=True)
def _alternate_from_labels(X, y, K, z, max_iter):
    """Hard alternation started from a labelling.  A group that falls below
    d units keeps its previous parameters.  Returns (z, theta, iterations)."""
    n, d = X.shape
    theta, cnt = _refit(X, y, z, K)
    it = 0
    for it in range(max_iter):
        ch = _assign(X, y, theta, z)
        if ch == 0 and it > 0:
            break
        th2, cnt = _refit(X, y, z, K)
        for k in range(K):
            if cnt[k] >= d:
                theta[k] = th2[k]
    return z, theta, it + 1


@njit(cache=True)
def _alternate_from_theta(X, y, K, theta, max_iter):
    n, d = X.shape
    z = -np.ones(n, dtype=np.int64)
    _assign(X, y, theta, z)
    it = 0
    for it in range(max_iter):
        th2, cnt = _refit(X, y, z, K)
        for k in range(K):
            if cnt[k] >= d:
                theta[k] = th2[k]
        ch = _assign(X, y, theta, z)
        if ch == 0:
            break
    return z, theta, it + 1


def refit(X, y, z, K):
    """Least-squares parameters per group; NaN for groups with fewer than d
    units or a rank-deficient design."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.int64)
    th, cnt = _refit(X, y, z, K)
    d = X.shape[1]
    for k in range(K):
        if cnt[k] < d or np.linalg.matrix_rank(X[z == k]) < d:
            th[k] = np.nan
    return th


def resid2(X, y, theta):
    return (y[:, None] - X @ theta.T) ** 2


def classify(X, y, theta, rng=None):
    """Plug-in rule h_theta: nearest surface; exact ties broken uniformly at
    random when rng is given (ties have probability zero for continuous
    data)."""
    r = resid2(X, y, theta)
    r = np.where(np.isfinite(r), r, np.inf)
    z = np.argmin(r, axis=1)
    if rng is not None:
        tie = r <= r[np.arange(len(z)), z][:, None]
        if (tie.sum(axis=1) > 1).any():
            z = np.argmax(rng.random(r.shape) * tie, axis=1)
    return z


def criterion(X, y, z, K):
    """min_theta F(z, theta), the least-squares criterion of a partition."""
    return float(_labels_value(np.ascontiguousarray(X, dtype=np.float64),
                               np.ascontiguousarray(y, dtype=np.float64),
                               np.asarray(z, dtype=np.int64), K))


def random_labels(n, K, rng):
    """Uniform random partition with every group non-empty (flexmix default
    initialisation)."""
    while True:
        z = rng.integers(0, K, n)
        if np.bincount(z, minlength=K).min() > 0:
            return z.astype(np.int64)


def random_theta(X, y, K, rng):
    """For each group an exact fit to d random units."""
    n, d = X.shape
    theta = np.empty((K, d))
    for k in range(K):
        for _ in range(50):
            idx = rng.choice(n, size=d, replace=False)
            if abs(np.linalg.det(X[idx])) > 1e-10:
                theta[k] = np.linalg.solve(X[idx], y[idx])
                break
        else:
            theta[k] = np.linalg.lstsq(X, y, rcond=None)[0] + rng.normal(size=d)
    return theta


def hard_alternation(X, y, K, R, rng, init="theta", max_iter=500):
    """Multistart hard alternation.  init = 'theta' (exact fits to d random
    units per group) or 'labels' (random partition).  Returns (z, theta, F,
    number of least-squares refits performed)."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n = X.shape[0]
    best = (np.inf, None)
    work = 0
    for r in range(R):
        if init == "labels":
            z, th, it = _alternate_from_labels(X, y, K, random_labels(n, K, rng), max_iter)
        else:
            z, th, it = _alternate_from_theta(X, y, K, random_theta(X, y, K, rng), max_iter)
        work += it
        f = _labels_value(X, y, z, K)
        if f < best[0]:
            best = (f, z.copy())
    z = best[1]
    return z, refit(X, y, z, K), float(best[0]), work


@njit(cache=True)
def _em(X, y, K, G0, max_iter, tol, var_floor):
    """EM for a Gaussian mixture of linear regressions, started from a
    responsibility matrix G0.  Returns (loglik, G, theta, s2, w, iters)."""
    n, d = X.shape
    G = G0.copy()
    theta = np.zeros((K, d))
    s2 = np.ones(K)
    w = np.ones(K) / K
    ll_old = -1e300
    ll = -1e300
    it = 0
    logp = np.zeros((n, K))
    for it in range(max_iter):
        # M-step
        for k in range(K):
            nk = 0.0
            XX = np.zeros((d, d))
            Xy = np.zeros(d)
            yy = 0.0
            for i in range(n):
                g = G[i, k]
                nk += g
                yy += g * y[i] * y[i]
                for a in range(d):
                    Xy[a] += g * X[i, a] * y[i]
                    for c in range(d):
                        XX[a, c] += g * X[i, a] * X[i, c]
            if nk < d + 1e-6:
                return -1e300, G, theta, s2, w, it
            sse, beta = _group_sse(n, XX, Xy, yy, d)
            theta[k] = beta
            s2[k] = max(sse / nk, var_floor)
            w[k] = nk / n
        # E-step
        ll = 0.0
        for i in range(n):
            mx = -1e300
            for k in range(K):
                f = 0.0
                for a in range(d):
                    f += X[i, a] * theta[k, a]
                r = y[i] - f
                lp = math.log(w[k]) - 0.5 * math.log(2 * math.pi * s2[k]) - 0.5 * r * r / s2[k]
                logp[i, k] = lp
                if lp > mx:
                    mx = lp
            s = 0.0
            for k in range(K):
                s += math.exp(logp[i, k] - mx)
            lse = mx + math.log(s)
            ll += lse
            for k in range(K):
                G[i, k] = math.exp(logp[i, k] - lse)
        if abs(ll - ll_old) < tol * (1.0 + abs(ll)):
            break
        ll_old = ll
    return ll, G, theta, s2, w, it + 1


def em_mixreg(X, y, K, R, rng, init="labels", short=0, max_iter=1000, tol=1e-9):
    """Multistart EM for Gaussian mixtures of regressions.

    init  : 'labels' (random hard partition, the flexmix default) or 'theta'
            (exact fits to d random units, equal weights, pooled variance).
    short : if > 0, the emEM strategy of Biernacki, Celeux and Govaert (2003):
            R short runs of `short` iterations, the best of which is run to
            convergence.
    The variance floor is 1e-6 times the pooled residual variance.
    Returns (MAP labels, theta, F, loglik, work)."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n, d = X.shape
    s2p = float(np.var(y - X @ np.linalg.lstsq(X, y, rcond=None)[0]))
    floor = 1e-6 * max(s2p, 1e-12)
    best = (-np.inf, None)
    work = 0
    for r in range(R):
        if init == "labels":
            z0 = random_labels(n, K, rng)
            G0 = np.zeros((n, K))
            G0[np.arange(n), z0] = 1.0
        else:
            th = random_theta(X, y, K, rng)
            r2 = resid2(X, y, th)
            lp = -0.5 * r2 / max(s2p, 1e-12)
            lp -= lp.max(axis=1, keepdims=True)
            G0 = np.exp(lp)
            G0 /= G0.sum(axis=1, keepdims=True)
        ll, G, _, _, _, it = _em(X, y, K, G0, short if short > 0 else max_iter, tol, floor)
        work += it
        if ll > best[0]:
            best = (ll, G.copy())
    if best[1] is None:
        z = random_labels(n, K, rng)
        return z, refit(X, y, z, K), criterion(X, y, z, K), -np.inf, work
    G = best[1]
    if short > 0:
        ll, G, _, _, _, it = _em(X, y, K, G, max_iter, tol, floor)
        work += it
        if ll <= -1e299:
            G = best[1]
    z = np.argmax(G, axis=1).astype(np.int64)
    return z, refit(X, y, z, K), criterion(X, y, z, K), float(best[0]), work


# ---------------------------------------------------------------------------
# Exact micro-solve
# ---------------------------------------------------------------------------


def exact_solve(X, y, K, rng=None, n_inc=0):
    """Exact global minimiser of the clusterwise least-squares criterion on the
    rows of (X, y).  With n_inc > 0 an incumbent from alternation starts is
    supplied to the branch and bound; this changes the running time, not the
    result (up to ties).  Returns (labels, value, nodes)."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    m = X.shape[0]
    if n_inc > 0:
        z0, _, f0, _ = hard_alternation(X, y, K, n_inc, rng)
        z0 = _canonical(z0).astype(np.int64)
        init_v = f0 * (1 + 1e-12) + 1e-12
    else:
        z0 = np.zeros(m, dtype=np.int64)
        init_v = np.inf
    val, lab, nodes = _bb_exact(X, y, K, z0, init_v)
    return lab, float(val), int(nodes)


def _canonical(z):
    """Relabel so that groups appear in order of first occurrence."""
    z = np.asarray(z)
    out = np.empty_like(z)
    mp = {}
    for i, v in enumerate(z):
        if v not in mp:
            mp[v] = len(mp)
        out[i] = mp[v]
    return out


def brute_force(X, y, K):
    """Plain enumeration of all K^m label vectors (testing only)."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    m = X.shape[0]
    best, bl = np.inf, None
    for z in itertools.product(range(K), repeat=m):
        z = np.array(z, dtype=np.int64)
        v = _labels_value(X, y, z, K)
        if v < best:
            best, bl = v, z
    return bl, best


# ---------------------------------------------------------------------------
# Label alignment
# ---------------------------------------------------------------------------

_PERMS = {K: np.array(list(itertools.permutations(range(K)))) for K in range(1, 7)}


def align(theta, ref):
    """Relabel theta by the permutation tau minimising
    sum_k ||theta_k - ref_{tau(k)}||^2 (exact; enumeration for K <= 6,
    Hungarian algorithm otherwise).  Returns (relabelled theta, tau)."""
    K = theta.shape[0]
    C = ((theta[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
    if K <= 6:
        P = _PERMS[K]
        tau = P[int(np.argmin(C[np.arange(K)[None, :], P].sum(axis=1)))]
    else:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(C)
        tau = np.empty(K, dtype=int)
        tau[r] = c
    out = np.empty_like(theta)
    out[tau] = theta
    return out, tau


# ---------------------------------------------------------------------------
# Replicates and EESP (Algorithm 1)
# ---------------------------------------------------------------------------


class Fit:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def replicate(X, y, K, m, rng, solver="exact", max_redraw=10000, R_sub=10):
    """One admissible replicate.  Draw S uniformly without replacement from
    {1..n}, solve the restricted problem, and redraw if the fitted labelling
    leaves a group with fewer than d + 1 units or a rank-deficient design
    (fitted-label feasibility rule).  solver = 'exact' (branch and bound) or
    'local' (one start of hard alternation, the ablation).
    Returns (theta, S, number of draws)."""
    n, d = X.shape
    for r in range(1, max_redraw + 1):
        S = rng.choice(n, size=m, replace=False)
        XS, yS = X[S], y[S]
        if solver == "exact":
            lab, _, _ = exact_solve(XS, yS, K)
        elif solver == "multistart":
            lab, _, _, _ = hard_alternation(XS, yS, K, R_sub, rng)
        else:
            lab, _, _, _ = hard_alternation(XS, yS, K, 1, rng)
        cnt = np.bincount(lab, minlength=K)
        if cnt.min() < d + 1:
            continue
        th = refit(XS, yS, lab, K)
        if np.isfinite(th).all():
            return th, S, r
    raise RuntimeError("no admissible subsample")


def pilot_reference(X, y, K, m, rng, solver="exact", P=10, crit="ls", R_sub=10):
    """Alignment reference: the best of P independent replicates, scored by the
    full-sample least-squares criterion of its plug-in partition (crit='ls')
    or by the median minimum squared residual (crit='lms', for contaminated
    data).  The pilot replicates are not aggregated, so the aggregated
    replicates remain conditionally i.i.d. given the data and the pilot.
    Returns (theta_ref, number of draws)."""
    best, draws = (np.inf, None), 0
    for _ in range(P):
        th, _, r = replicate(X, y, K, m, rng, solver, R_sub=R_sub)
        draws += r
        if crit == "ls":
            f = criterion(X, y, classify(X, y, th), K)
        else:
            rr, _ = _min_resid(X, y, th)
            f = float(np.median(rr))
        if f < best[0]:
            best = (f, th)
    return best[1], draws


def eesp(X, y, K, m, B, rng, agg="vote", T=1, solver="exact", ref=None,
         keep=False, R_sub=10, P=10):
    """Algorithm 1.

    Pass 1: the best of P pilot replicates (pilot_reference) supplies the
    alignment reference (pilots are not voted); B further replicates are drawn,
    aligned to the reference, extended to all n units by the plug-in rule and
    aggregated by plurality vote ('vote') or by averaging the aligned
    parameters and classifying against the average ('param'); the aggregated
    partition is refitted.  Passes t >= 2 (optional) use the previous accepted
    parameters as reference and are accepted only if they lower the criterion
    (monotone safeguard).  Returns a Fit."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n, d = X.shape
    t0 = time.perf_counter()
    draws = 0
    if ref is None:
        ref, r0 = pilot_reference(X, y, K, m, rng, solver, P=P, R_sub=R_sub)
        draws += r0
    F_cur, z_cur, th_cur = np.inf, None, None
    accepted = 0
    trace = []
    V_keep = th_keep = None
    for t in range(T):
        V = np.zeros((n, K))
        thetas = np.empty((B, K, d))
        idx = np.arange(n)
        for b in range(B):
            th, _, r = replicate(X, y, K, m, rng, solver, R_sub=R_sub)
            draws += r
            th, _ = align(th, ref)
            thetas[b] = th
            V[idx, classify(X, y, th, rng)] += 1.0
        if agg == "vote":
            tie = V >= V.max(axis=1, keepdims=True)
            z_new = np.argmax(rng.random(V.shape) * tie, axis=1)
        elif agg == "median":
            z_new = classify(X, y, np.median(thetas, axis=0), rng)
        else:
            z_new = classify(X, y, thetas.mean(axis=0), rng)
        F_new = criterion(X, y, z_new, K)
        trace.append(F_new)
        if t == 0 or keep:
            V_keep, th_keep = V / B, thetas
        if F_new >= F_cur:
            break
        z_cur, F_cur = z_new, F_new
        th_cur = refit(X, y, z_cur, K)
        accepted += 1
        if np.isfinite(th_cur).all():
            ref = th_cur
    out = Fit(z=z_cur, theta=th_cur, F=F_cur, accepted=accepted, draws=draws,
              trace=trace, time=time.perf_counter() - t0)
    out.V, out.thetas = V_keep, th_keep
    return out


def best_of_B(X, y, K, m, B, rng, solver="exact", R_sub=10):
    """Selection analogue (CLARA-type): identical replicates, each extended to
    the full sample; the one whose induced partition has the lowest criterion
    is returned."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    t0 = time.perf_counter()
    best = (np.inf, None)
    draws = 0
    for b in range(B):
        th, _, r = replicate(X, y, K, m, rng, solver, R_sub=R_sub)
        draws += r
        z = classify(X, y, th, rng)
        f = criterion(X, y, z, K)
        if f < best[0]:
            best = (f, z)
    z = best[1]
    return Fit(z=z, theta=refit(X, y, z, K), F=best[0], draws=draws,
               time=time.perf_counter() - t0)


def polish(X, y, K, fit, max_iter=500):
    """Hard alternation started from the parameters of a fit.  The criterion
    cannot increase."""
    t0 = time.perf_counter()
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    z0 = np.asarray(fit.z, dtype=np.int64).copy()
    z, th, it = _alternate_from_labels(X, y, K, z0, max_iter)
    f = _labels_value(X, y, z, K)
    if f > fit.F:
        z, f = np.asarray(fit.z, dtype=np.int64), fit.F
    return Fit(z=z, theta=refit(X, y, z, K), F=float(f),
               time=getattr(fit, "time", 0.0) + time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Trimmed criterion (contamination experiments)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _min_resid(X, y, theta):
    n, d = X.shape
    K = theta.shape[0]
    r = np.empty(n)
    z = np.empty(n, dtype=np.int64)
    for i in range(n):
        bv = 1e300
        bk = 0
        for k in range(K):
            f = 0.0
            for a in range(d):
                f += X[i, a] * theta[k, a]
            v = (y[i] - f) ** 2
            if v < bv:
                bv = v
                bk = k
        r[i] = bv
        z[i] = bk
    return r, z


@njit(cache=True)
def _cstep(X, y, K, theta, h, max_iter):
    """Concentration steps for trimmed clusterwise least squares: assign each
    unit to its nearest surface, keep the h units with the smallest residuals,
    refit each group on its kept units; repeat until the trimmed criterion
    stops decreasing.  Returns (theta, labels, kept mask, trimmed value)."""
    n, d = X.shape
    best_v = 1e300
    best_th = theta.copy()
    for it in range(max_iter):
        r, z = _min_resid(X, y, theta)
        order = np.argsort(r)
        v = 0.0
        for j in range(h):
            v += r[order[j]]
        if v >= best_v * (1 - 1e-12):
            break
        best_v = v
        best_th[:, :] = theta
        zk = z.copy()
        for j in range(h, n):
            zk[order[j]] = K  # trimmed units go to a dummy group
        th2, cnt = _refit(X, y, zk, K + 1)
        for k in range(K):
            if cnt[k] >= d:
                theta[k] = th2[k]
    r, z = _min_resid(X, y, best_th)
    order = np.argsort(r)
    kept = np.zeros(n, dtype=np.bool_)
    for j in range(h):
        kept[order[j]] = True
    return best_th, z, kept, best_v


def trimmed_value(X, y, theta, alpha):
    r, _ = _min_resid(np.ascontiguousarray(X, dtype=np.float64),
                      np.ascontiguousarray(y, dtype=np.float64), theta)
    h = int(np.ceil((1 - alpha) * len(y)))
    return float(np.sort(r)[:h].sum())


def trimmed_alternation(X, y, K, R, alpha, rng, max_iter=200):
    """Multistart trimmed hard alternation (clusterwise regression with
    trimming; the equal-scale, equal-weight case of TCLUST-REG): R random
    elemental starts, each followed by concentration steps, best trimmed
    criterion kept."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    t0 = time.perf_counter()
    h = int(np.ceil((1 - alpha) * len(y)))
    best = (np.inf, None, None)
    for r in range(R):
        th, z, kept, v = _cstep(X, y, K, random_theta(X, y, K, rng), h, max_iter)
        if v < best[0]:
            best = (v, th.copy(), z.copy())
    return Fit(z=best[2], theta=best[1], F=best[0], time=time.perf_counter() - t0)


def best_of_B_trimmed(X, y, K, m, B, alpha, rng, solver="exact", R_sub=10,
                      csteps=True):
    """Selection among replicates by the trimmed criterion, followed by
    concentration steps: the elemental-subset strategy of Rousseeuw (1984)
    with an exact K-group micro-solve."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    t0 = time.perf_counter()
    h = int(np.ceil((1 - alpha) * len(y)))
    best = (np.inf, None)
    for b in range(B):
        th, _, _ = replicate(X, y, K, m, rng, solver, R_sub=R_sub)
        r, _ = _min_resid(X, y, th)
        v = np.sort(r)[:h].sum()
        if v < best[0]:
            best = (v, th)
    th = best[1]
    if csteps:
        th, z, kept, v = _cstep(X, y, K, th.copy(), h, 200)
    else:
        v = best[0]
        _, z = _min_resid(X, y, th)
    return Fit(z=z, theta=th, F=float(v), time=time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Exact oracles for the full-sample criterion
# ---------------------------------------------------------------------------


@njit(cache=True)
def _sse_from(n, sx, sy, sxx, sxy, syy):
    """Residual sum of squares of a simple regression from sums; exact for
    groups whose x values all coincide (then the fit is the group mean)."""
    if n <= 1.5:
        return 0.0
    den = n * sxx - sx * sx
    if den <= 1e-10 * max(1.0, n * sxx):
        v = syy - sy * sy / n
    else:
        b = (n * sxy - sx * sy) / den
        a = (sy - b * sx) / n
        v = syy - a * sy - b * sxy
    if v < 1e-12 * max(1.0, syy):
        v = 0.0
    return v


@njit(cache=True)
def _oracle_k2_line(xs, ys):
    """xs sorted increasingly.  Candidate partitions: group 1 = units left of
    a vertical split that lie above a line, together with units right of the
    split that lie below it; the line passes through two units, each of which
    is placed on either side.  Every optimal partition is of this form."""
    n = xs.shape[0]
    best = 1e300
    best_i = -1
    best_j = -1
    best_c = -1
    best_s = -1
    T = np.zeros(6)
    T[0] = n
    for k in range(n):
        T[1] += xs[k]
        T[2] += ys[k]
        T[3] += xs[k] * xs[k]
        T[4] += xs[k] * ys[k]
        T[5] += ys[k] * ys[k]
    above = np.zeros(n, dtype=np.int8)
    pre = np.zeros((n + 1, 6))    # prefix sums over units with above == 1
    suf = np.zeros((n + 1, 6))    # suffix sums over units with above == 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[j] - xs[i]
            if dx == 0.0:
                continue
            slope = (ys[j] - ys[i]) / dx
            icpt = ys[i] - slope * xs[i]
            for k in range(n):
                above[k] = 1 if ys[k] - icpt - slope * xs[k] > 0 else 0
            for c in range(4):
                above[i] = c & 1
                above[j] = (c >> 1) & 1
                for q in range(6):
                    pre[0, q] = 0.0
                    suf[n, q] = 0.0
                for k in range(n):
                    a = above[k]
                    x = xs[k]
                    yv = ys[k]
                    pre[k + 1, 0] = pre[k, 0] + a
                    pre[k + 1, 1] = pre[k, 1] + a * x
                    pre[k + 1, 2] = pre[k, 2] + a * yv
                    pre[k + 1, 3] = pre[k, 3] + a * x * x
                    pre[k + 1, 4] = pre[k, 4] + a * x * yv
                    pre[k + 1, 5] = pre[k, 5] + a * yv * yv
                for k in range(n - 1, -1, -1):
                    a = 1 - above[k]
                    x = xs[k]
                    yv = ys[k]
                    suf[k, 0] = suf[k + 1, 0] + a
                    suf[k, 1] = suf[k + 1, 1] + a * x
                    suf[k, 2] = suf[k + 1, 2] + a * yv
                    suf[k, 3] = suf[k + 1, 3] + a * x * x
                    suf[k, 4] = suf[k + 1, 4] + a * x * yv
                    suf[k, 5] = suf[k + 1, 5] + a * yv * yv
                for s in range(n + 1):
                    g0 = pre[s, 0] + suf[s, 0]
                    g1 = pre[s, 1] + suf[s, 1]
                    g2 = pre[s, 2] + suf[s, 2]
                    g3 = pre[s, 3] + suf[s, 3]
                    g4 = pre[s, 4] + suf[s, 4]
                    g5 = pre[s, 5] + suf[s, 5]
                    v = _sse_from(g0, g1, g2, g3, g4, g5) + \
                        _sse_from(T[0] - g0, T[1] - g1, T[2] - g2,
                                  T[3] - g3, T[4] - g4, T[5] - g5)
                    if v < best:
                        best = v
                        best_i = i
                        best_j = j
                        best_c = c
                        best_s = s
    # reconstruct
    z = np.zeros(n, dtype=np.int64)
    if best_i >= 0:
        dx = xs[best_j] - xs[best_i]
        slope = (ys[best_j] - ys[best_i]) / dx
        icpt = ys[best_i] - slope * xs[best_i]
        for k in range(n):
            above[k] = 1 if ys[k] - icpt - slope * xs[k] > 0 else 0
        above[best_i] = best_c & 1
        above[best_j] = (best_c >> 1) & 1
        for k in range(n):
            if k < best_s:
                z[k] = 1 if above[k] == 1 else 0
            else:
                z[k] = 1 if above[k] == 0 else 0
    return best, z


def oracle_k2_line(x, y):
    """Exact global minimiser of the two-group clusterwise simple-regression
    criterion.  x, y 1-d arrays.  Returns (labels in the input order, value)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x, kind="mergesort")
    v, zs = _oracle_k2_line(x[order].copy(), y[order].copy())
    z = np.empty_like(zs)
    z[order] = zs
    return z, float(v)


@njit(cache=True)
def _oracle_location(ys, K):
    n = ys.shape[0]
    c1 = np.zeros(n + 1)
    c2 = np.zeros(n + 1)
    for i in range(n):
        c1[i + 1] = c1[i] + ys[i]
        c2[i + 1] = c2[i] + ys[i] * ys[i]
    INF = 1e300
    D = np.full((K + 1, n + 1), INF)
    arg = np.zeros((K + 1, n + 1), dtype=np.int64)
    D[0, 0] = 0.0
    for k in range(1, K + 1):
        for j in range(1, n + 1):
            bestv = INF
            besta = 0
            for a in range(k - 1, j):
                if D[k - 1, a] >= INF:
                    continue
                cnt = j - a
                s1 = c1[j] - c1[a]
                s2 = c2[j] - c2[a]
                v = D[k - 1, a] + s2 - s1 * s1 / cnt
                if v < bestv:
                    bestv = v
                    besta = a
            D[k, j] = bestv
            arg[k, j] = besta
    z = np.zeros(n, dtype=np.int64)
    j = n
    for k in range(K, 0, -1):
        a = arg[k, j]
        for t in range(a, j):
            z[t] = k - 1
        j = a
    return D[K, n], z


def oracle_location(y, K):
    y = np.asarray(y, dtype=float)
    order = np.argsort(y, kind="mergesort")
    v, zs = _oracle_location(y[order].copy(), K)
    z = np.empty_like(zs)
    z[order] = zs
    return z, float(v)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def agreement(z1, z2, K):
    """Largest fraction of units on which z1 and tau(z2) agree, over tau."""
    z1 = np.asarray(z1)
    z2 = np.asarray(z2)
    C = np.zeros((K, K))
    np.add.at(C, (z1, z2), 1)
    P = _PERMS[K]
    return float(C[P, np.arange(K)[None, :]].sum(axis=1).max() / len(z1))


def ari(z1, z2):
    z1 = np.asarray(z1)
    z2 = np.asarray(z2)
    n = len(z1)
    _, a = np.unique(z1, return_inverse=True)
    _, b = np.unique(z2, return_inverse=True)
    C = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(C, (a, b), 1)
    comb = lambda v: v * (v - 1) / 2.0
    sij = comb(C).sum()
    si = comb(C.sum(axis=1)).sum()
    sj = comb(C.sum(axis=0)).sum()
    exp = si * sj / comb(n)
    mx = 0.5 * (si + sj)
    return float((sij - exp) / (mx - exp)) if mx != exp else 1.0


def same_partition(z1, z2, K):
    return agreement(z1, z2, K) == 1.0
