"""Merge Python, R and Octave outputs and compute metrics per (unit, method).

Every method is scored the same way, from its K coefficient vectors alone:
  acc   : agreement (best label permutation) between the nearest-surface
          labels of the clean units and their true labels;
  perr  : max abs coefficient error after optimal alignment to the truth;
  fail  : acc < 0.7 or no fit;
  ahat  : trimmed/flagged fraction reported by the method.
Also derives '+rw' versions (the reweighting step of Algorithm 2) of the fixed-
level TCLUST-REG fits, TCR(a)+rw, which are level-free apart from a.
"""
import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd

import simv4
import py_methods as P
from simcommon import truth


def true_theta(D):
    if "theta0" in D:
        return np.asarray(D["theta0"], float)
    th = truth(D["K"], D["geom"], D["delta"])
    if D["p"] > 1:
        th = np.column_stack([th, np.ones((D["K"], D["p"] - 1))])
    return th


def align_err(th, th0):
    K = th0.shape[0]
    best = np.inf
    for perm in itertools.permutations(range(K)):
        best = min(best, float(np.abs(th[list(perm)] - th0).max()))
    return best


def acc_clean(X, y, th, z0, clean):
    K = th.shape[0]
    r = (y[:, None] - X @ th.T) ** 2
    z = np.argmin(r, axis=1)
    best = 0.0
    for perm in itertools.permutations(range(K)):
        best = max(best, float(np.mean(np.asarray(perm)[z[clean]] == z0[clean])))
    return best


def main(folder, out_csv, add_rw=True, prefixes=("py", "r", "oct")):
    frames = [pd.read_csv(f) for f in glob.glob(os.path.join(folder, "*.csv"))
              if os.path.basename(f).split("_")[0] in prefixes]
    R = pd.concat(frames, ignore_index=True)
    R["method"] = R["method"].astype(str)
    units = {simv4.unit_name(u): u for u in simv4.units()}
    rows = []
    for name, g in R.groupby("unit"):
        u = units[name]
        D = simv4.DESIGNS[u[0]]
        X, y, z0, out = simv4.generate(u)
        K, d = D["K"], X.shape[1]
        th0 = true_theta(D)
        clean = ~out
        recs = list(g.itertuples(index=False))
        extra = []
        if add_rw:
            for rec in recs:
                if rec.method.startswith("TCR"):
                    b = np.array([getattr(rec, f"b{j+1}") for j in range(K * d)], float)
                    if np.isfinite(b).all():
                        th, ah = P.reweight(X, y, b.reshape(K, d), xtrim=rec.method.startswith("TCRx"))
                        extra.append((rec.method + "+rw", rec.time, ah, th))
        for rec in recs:
            b = np.array([getattr(rec, f"b{j+1}") for j in range(K * d)], float)
            th = b.reshape(K, d) if np.isfinite(b).all() else None
            extra.append((rec.method, rec.time, rec.alpha_hat, th))
        for meth, t, ah, th in extra:
            if th is None:
                a, pe = np.nan, np.nan
            else:
                a, pe = acc_clean(X, y, th, z0, clean), align_err(th, th0)
            rows.append(dict(design=u[0], eps=u[1], rep=u[2], unit=name, method=meth,
                             acc=a, perr=pe, fail=bool(not np.isfinite(a) or a < 0.7),
                             ahat=ah, time=t))
    M = pd.DataFrame(rows)
    M.to_csv(out_csv, index=False)
    return M


if __name__ == "__main__":
    if len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], add_rw=False, prefixes=tuple(sys.argv[3].split(",")))
    else:
        main(sys.argv[1], sys.argv[2])
