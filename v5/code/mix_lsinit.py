"""Mixtures given one extra start from the least-squares partition (descriptive, added after
the external reading of the study).  In the designs where the level-free mixtures fail (D2, D8,
four groups) the noise-component and contaminated normal mixtures are rerun with their usual
starts plus one start from least squares with 50 random starts; the start of highest likelihood
after the short runs is continued, as before.  Usage: V4_STUDY=... python mix_lsinit.py OUT"""
import sys
from multiprocessing import Pool
import numpy as np, pandas as pd
import simv4, evaluate as EV, eespcore as E, py_methods as P, cn_mix as C

KEEP = ("D2-", "D8-", "E4-")


def run(u):
    X, y, z0, out = simv4.generate(u)
    D = simv4.DESIGNS[u[0]]; K = D["K"]
    rng = np.random.default_rng(simv4.seed_for("v4lsinit", *u))
    z_ls, th_ls, f, _ = E.hard_alternation(X, y, K, 50, rng)
    fits = [("NoiseMixLS",) + tuple(P.noise_mix(X, y, K, np.random.default_rng(simv4.seed_for("v4nm", *u)), z_init=z_ls)[:2]),
            ("CNmixLS",) + tuple(C.cn_mix(X, y, K, np.random.default_rng(simv4.seed_for("v4cn", *u)), z_init=z_ls, th_init=th_ls)[:2])]
    res = []
    for meth, th, ah in fits:
        ok = np.isfinite(th).all()
        a = EV.acc_clean(X, y, th, z0, ~out) if ok else 0.0
        res.append(dict(design=u[0], eps=u[1], rep=u[2], unit=simv4.unit_name(u), method=meth, acc=a,
                        perr=EV.align_err(th, EV.true_theta(D)) if ok else np.nan, ahat=ah))
    return res


if __name__ == "__main__":
    us = [u for u in simv4.units() if u[0].startswith(KEEP)]
    with Pool(2) as p:
        R = p.map(run, us, chunksize=4)
    pd.DataFrame([r for rr in R for r in rr]).to_csv(sys.argv[1], index=False)
    print(len(R))
