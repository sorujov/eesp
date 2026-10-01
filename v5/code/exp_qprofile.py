"""E1/E2/E5: the replicate law against the exact criterion minimiser.

For K = 2 simple clusterwise regression the exact minimiser z_hat of the
full-sample criterion is computed by the O(n^3) oracle.  For each data set a
large number Bbig of admissible replicates is drawn, and from them

  * Q_i = P(h_b(u_i) = z_hat_i | D_n), estimated by the replicate frequency
    (Lemma 4.5 makes this an average of Bbig i.i.d. Bernoulli(Q_i));
  * the boundary set {Q_i < 1/2} and the margin sets {Q_i >= 1/2 + gamma};
  * the B-curve: probability, over subsets of B replicates, that the vote
    agrees with z_hat on the margin set, against the union of exact binomial
    tails and against the Hoeffding union bound;
  * a batching check of conditional independence;
  * the realised floor: units whose plurality label differs from the
    population quantizer partition z*;
  * vote shares versus the plug-in margin as indicators of units that z_hat
    misclassifies.

Usage: python3 exp_qprofile.py JOB NJOBS   (cells are split across jobs)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import eespcore as E
from simcommon import OUT, generate, seed_for, bayes_labels, truth

N = 200
K = 2
BBIG = 2000
NDATA = int(os.environ.get("E1_NDATA", "50"))
BLIST = [5, 10, 25, 50, 100, 200, 400, 1000]
NRES = 200
GAMMAS = [0.05, 0.10, 0.20]
GEOMS = {"par": 3.0, "cross": 9.0}
MS = [6, 8, 12, 16, 24]
SOLVERS = ["exact", "local"]


def population_theta(geom, delta, rng):
    X, y, z0, _ = generate(rng, 400_000, K, geom, delta)
    z, th, it = E._alternate_from_theta(X, y, K, truth(K, geom, delta).copy(), 500)
    return th


def auc(score, label):
    """P(score of a positive > score of a negative), ties counted 1/2."""
    pos = score[label]
    neg = score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().values
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def binom_lower_tail(B, q):
    """P(Binomial(B, q) <= B/2) with ties at B/2 counted as a half error."""
    from scipy.stats import binom
    q = np.clip(q, 0, 1)
    if B % 2 == 0:
        return binom.cdf(B // 2 - 1, B, q) + 0.5 * binom.pmf(B // 2, B, q)
    return binom.cdf(B // 2, B, q)


def one(geom, m, solver, r, thstar):
    delta = GEOMS[geom]
    rng = np.random.default_rng(seed_for("E1", geom, m, solver, r))
    X, y, z0, _ = generate(np.random.default_rng(seed_for("E1data", geom, r)), N, K, geom, delta)
    zhat, Fhat = E.oracle_k2_line(X[:, 1], y)
    t0 = time.perf_counter()
    ref, _ = E.pilot_reference(X, y, K, m, rng, solver)
    H = np.empty((BBIG, N), dtype=np.int8)
    draws = 0
    for b in range(BBIG):
        th, _, d_ = E.replicate(X, y, K, m, rng, solver)
        draws += d_
        th, _ = E.align(th, ref)
        H[b] = E.classify(X, y, th, rng)
    t_rep = (time.perf_counter() - t0) / (BBIG + 1)
    # label z_hat and z* consistently with the reference
    th_hat = E.refit(X, y, zhat, K)
    _, tau = E.align(th_hat, ref)
    zh = tau[zhat]
    ths, _ = E.align(thstar, ref)
    zstar = E.classify(X, y, ths)
    Q = (H == zh[None, :]).mean(axis=0)
    allright = (H == zh[None, :]).all(axis=1)
    # basin: does one alternation polish from a replicate's extension reach z_hat?
    nb_basin = min(BBIG, 300)
    Xc = np.ascontiguousarray(X)
    yc = np.ascontiguousarray(y)
    inbasin = 0
    for b in range(nb_basin):
        zz, _, _ = E._alternate_from_labels(Xc, yc, K, H[b].astype(np.int64), 500)
        inbasin += E.same_partition(zz, zhat, K)
    share1 = (H == 1).mean(axis=0)
    plural = np.where(share1 > 0.5, 1, np.where(share1 < 0.5, 0, -1))
    # vote with all replicates (ties at random)
    V = np.stack([1 - share1, share1], axis=1)
    tie = V >= V.max(axis=1, keepdims=True)
    zvote = np.argmax(rng.random(V.shape) * tie, axis=1)
    fv = E.Fit(z=zvote, theta=E.refit(X, y, zvote, K), F=E.criterion(X, y, zvote, K))
    fp = E.polish(X, y, K, fv)
    zb, _ = bayes_labels(X, y, K, geom, delta)
    # plug-in margin of z_hat
    r2 = E.resid2(X, y, th_hat)
    plug_amb = -np.abs(r2[:, 0] - r2[:, 1])
    wrong = E.agreement(zhat, z0, K) >= 0  # placeholder to get orientation
    # orientation of z_hat relative to z0
    a_same = (zhat == z0).mean()
    zhat0 = zhat if a_same >= 0.5 else 1 - zhat
    miss = zhat0 != z0
    row = dict(geom=geom, m=m, solver=solver, rep=r, n=N, B=BBIG,
               F_hat=Fhat, t_rep=t_rep, draws_per_rep=draws / BBIG,
               p_zhat=float(allright.mean()), p_basin=inbasin / nb_basin,
               frac_Q_below_half=float((Q < 0.5).mean()),
               n_Q_below_half=int((Q < 0.5).sum()),
               min_Q=float(Q.min()), mean_Q=float(Q.mean()),
               q10=float(np.quantile(Q, 0.10)),
               vote_agree_zhat=E.agreement(zvote, zhat, K),
               vote_equals_zhat=bool(E.same_partition(zvote, zhat, K)),
               vote_gap=fv.F / Fhat - 1,
               votepol_equals_zhat=bool(E.same_partition(fp.z, zhat, K)),
               votepol_gap=fp.F / Fhat - 1,
               plural_ne_zstar=float(((plural != zstar) & (plural >= 0)).mean()),
               zhat_ne_zstar=float((zh != zstar).mean()),
               acc0_zhat=E.agreement(zhat, z0, K), acc0_vote=E.agreement(zvote, z0, K),
               acc0_bayes=E.agreement(zb, z0, K),
               auc_vote_share=auc(1 - Q, miss), auc_plugin=auc(plug_amb, miss),
               n_zhat_wrong=int(miss.sum()),
               corr_Q_margin=float(np.corrcoef(Q, -plug_amb)[0, 1]))
    for g in GAMMAS:
        row[f"n_margin_{g}"] = int((Q >= 0.5 + g).sum())
    # batching check: 40 batches of 50
    nb, bs = 40, BBIG // 40
    C = (H[: nb * bs] == zh[None, :]).reshape(nb, bs, N).sum(axis=1)
    sel = (Q > 0.05) & (Q < 0.95)
    if sel.any():
        ratio = C[:, sel].var(axis=0, ddof=1) / (bs * Q[sel] * (1 - Q[sel]))
        row["batch_var_ratio"] = float(ratio.mean())
        row["batch_var_ratio_n"] = int(sel.sum())
    # B-curve
    curve = []
    for B in BLIST:
        agree = {g: 0 for g in GAMMAS}
        agree_all = 0
        for _ in range(NRES):
            idx = rng.choice(BBIG, size=B, replace=False)
            s1 = (H[idx] == 1).sum(axis=0)
            zv = np.where(2 * s1 > B, 1, np.where(2 * s1 < B, 0, rng.integers(0, 2, N)))
            ok = zv == zh
            agree_all += ok.all()
            for g in GAMMAS:
                Mg = Q >= 0.5 + g
                agree[g] += ok[Mg].all()
        for g in GAMMAS:
            Mg = Q >= 0.5 + g
            tails = binom_lower_tail(B, Q[Mg])
            curve.append(dict(geom=geom, m=m, solver=solver, rep=r, B=B, gamma=g,
                              n_margin=int(Mg.sum()),
                              p_miss_emp=1 - agree[g] / NRES,
                              union_binom=float(min(1.0, tails.sum())),
                              union_hoeff=float(min(1.0, (np.exp(-2 * B * (Q[Mg] - 0.5) ** 2)).sum())),
                              p_miss_all_emp=1 - agree_all / NRES))
    prof = dict(Q=Q.astype(np.float32), margin=(-plug_amb).astype(np.float32),
                miss=miss, x=X[:, 1].astype(np.float32), y=y.astype(np.float32),
                zhat=zh.astype(np.int8))
    return row, curve, prof


def main():
    job, njobs = int(sys.argv[1]), int(sys.argv[2])
    cells = [(g, m, s) for g in GEOMS for m in MS for s in SOLVERS]
    cells = cells[job::njobs]
    thstar = {}
    for g in GEOMS:
        thstar[g] = population_theta(g, GEOMS[g], np.random.default_rng(seed_for("pop", g)))
    rows, curves = [], []
    fn_rows = os.path.join(OUT, f"e1_rows_{job}.csv")
    fn_curve = os.path.join(OUT, f"e1_curve_{job}.csv")
    done = set()
    if os.path.exists(fn_rows):
        old = pd.read_csv(fn_rows)
        rows = old.to_dict("records")
        curves = pd.read_csv(fn_curve).to_dict("records")
        done = {(d["geom"], d["m"], d["solver"], d["rep"]) for d in rows}
    profs = {}
    for (g, m, s) in cells:
        for r in range(NDATA):
            if (g, m, s, r) in done:
                continue
            row, curve, prof = one(g, m, s, r, thstar[g])
            rows.append(row)
            curves.extend(curve)
            if r < 3:
                profs[f"{g}_{m}_{s}_{r}"] = prof
            if r % 5 == 4 or r == NDATA - 1:
                pd.DataFrame(rows).to_csv(fn_rows, index=False)
                pd.DataFrame(curves).to_csv(fn_curve, index=False)
                print(time.strftime("%H:%M:%S"), g, m, s, r, flush=True)
        np.savez_compressed(os.path.join(OUT, f"e1_prof_{job}_{g}_{m}_{s}.npz"),
                            **{k: v for p in [profs] for kk, vv in p.items() for k, v in
                               [(kk + "_" + a, b) for a, b in vv.items()]})
        profs = {}
    with open(os.path.join(OUT, f"e1_thstar_{job}.txt"), "w") as f:
        for g in thstar:
            f.write(f"{g} {thstar[g].tolist()}\n")


if __name__ == "__main__":
    main()
