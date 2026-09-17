"""E6: real clusterwise-regression data with the exact criterion minimiser.

Data: tone perception (Cohen 1984; n = 150), NO / equivalence ratio
(Brinkman 1981, via Hurn, Justel and Robert 2003; n = 88), CO2 emission
against GNP (n = 28), all as shipped with the R package mixtools; and United
States real GNP growth on its own lag (FRED GNPC96, 1947Q3-2026Q2, n = 316),
a two-regime switching AR(1) written as a clusterwise regression.

Responses are perturbed by 1e-7 standard deviations so that the data are in
general position (tone contains one duplicated point); all methods see the
perturbed data.  Each stochastic method is run from NSEED seeds.  A second
block plants 10% gross outliers in the tone data and measures agreement
with the clean-data minimiser on the clean units.

Usage: python3 exp_real.py
"""
import os
import time

import numpy as np
import pandas as pd
import pyreadr

import eespcore as E
import eesp_variants as V
from simcommon import OUT, ROOT, seed_for

NSEED = int(os.environ.get("E6_NSEED", "50"))
B = 200
RDATA = os.environ.get("RDATA", "/home/claude/rdata/mixtools/data")


def load():
    ds = {}
    for f, xc, yc, lab in [("tonedata", "stretchratio", "tuned", "tone"),
                           ("NOdata", "Equivalence", "NO", "ethanol"),
                           ("CO2data", "GNP", "CO2", "co2")]:
        d = pyreadr.read_r(os.path.join(RDATA, f"{f}.RData"))[f]
        ds[lab] = (d[xc].values.astype(float), d[yc].values.astype(float))
    g = pd.read_csv(os.path.join(ROOT, "data", "gnp.csv"), parse_dates=["observation_date"])
    g["growth"] = 100 * np.log(g["GNPC96"]).diff()
    g["lag"] = g["growth"].shift(1)
    g = g[(g.observation_date >= "1947-07-01") & (g.observation_date <= "2026-04-01")].dropna()
    ds["gnp-ar1"] = (g["lag"].values, g["growth"].values)
    return ds, g


def jitter(y, key):
    rng = np.random.default_rng(seed_for("jitter", key))
    return y + 1e-7 * np.std(y) * rng.normal(size=len(y))


def run(X, y, K, m, rng, zhat):
    res = {}
    t = time.perf_counter(); f = E.eesp(X, y, K, m, B, rng, keep=True); tv = time.perf_counter() - t
    res["vote"] = (f.z, f.F, tv)
    g = E.polish(X, y, K, f); res["vote+pol"] = (g.z, g.F, g.time)
    budget = g.time
    t = time.perf_counter(); h = E.polish(X, y, K, E.eesp(X, y, K, m, B, rng, agg="param"))
    res["param+pol"] = (h.z, h.F, time.perf_counter() - t)
    t = time.perf_counter(); h = E.polish(X, y, K, E.best_of_B(X, y, K, m, B, rng))
    res["bestB+pol"] = (h.z, h.F, time.perf_counter() - t)
    t = time.perf_counter(); h = E.polish(X, y, K, E.eesp(X, y, K, m, B, rng, solver="local"))
    res["local-vote+pol"] = (h.z, h.F, time.perf_counter() - t)
    t = time.perf_counter(); h = E.polish(X, y, K, E.eesp(X, y, K, m, B, rng, solver="multistart", R_sub=5))
    res["ms-vote+pol"] = (h.z, h.F, time.perf_counter() - t)
    t = time.perf_counter(); z, _, F, _ = E.hard_alternation(X, y, K, 10, rng)
    res["HA10"] = (z, F, time.perf_counter() - t)
    t = time.perf_counter(); R = 0; bF, bz = np.inf, None
    while R == 0 or time.perf_counter() - t < budget:
        z, _, F, _ = E.hard_alternation(X, y, K, 1, rng); R += 1
        if F < bF:
            bF, bz = F, z
    res["HA-tm"] = (bz, bF, time.perf_counter() - t)
    t = time.perf_counter(); z, _, F, _, _ = E.em_mixreg(X, y, K, 10, rng, short=10)
    res["emEM10"] = (z, F, time.perf_counter() - t)
    # replicate law against z_hat, for the Q profile of real data
    th_hat = E.refit(X, y, zhat, K)
    Q = None
    if f.V is not None:
        # f.V are vote shares in the labelling of the pilot reference; relabel z_hat
        ref = f.thetas.mean(axis=0)
        _, tau = E.align(th_hat, ref)
        Q = f.V[np.arange(len(y)), tau[zhat]]
    return res, R, Q


def main():
    ds, g = load()
    rows, qrows = [], []
    K = 2
    design = {"tone": 12, "ethanol": 12, "co2": 10, "gnp-ar1": 16}
    for lab, (x, y) in ds.items():
        y = jitter(y, lab)
        X = np.column_stack([np.ones(len(x)), x])
        zhat, Fhat = E.oracle_k2_line(x, y)
        n = len(y)
        for mm in sorted({design[lab], 20}):
            for s in range(NSEED):
                rng = np.random.default_rng(seed_for("E6", lab, mm, s))
                res, R, Q = run(X, y, K, mm, rng, zhat)
                for meth, (z, F, tm) in res.items():
                    rows.append(dict(data=lab, n=n, m=mm, seed=s, method=meth, F=F, Fhat=Fhat,
                                     hit=bool(F <= Fhat * (1 + 1e-9)), gap=F / Fhat - 1,
                                     agree_zhat=E.agreement(z, zhat, K), time=tm, R_tm=R))
                if Q is not None and s < 5:
                    qrows.append(dict(data=lab, m=mm, seed=s, frac_Q_below_half=float((Q < .5).mean()),
                                      n_Q_below_half=int((Q < .5).sum()), min_Q=float(Q.min())))
            print(time.strftime("%H:%M:%S"), lab, mm, flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "e6_rows.csv"), index=False)
        pd.DataFrame(qrows).to_csv(os.path.join(OUT, "e6_q.csv"), index=False)
        if lab == "gnp-ar1":
            nber = pd.read_csv(os.path.join(ROOT, "data", "usrec.csv"), parse_dates=["observation_date"])
            nber["q"] = nber.observation_date.dt.to_period("Q")
            rq = nber.groupby("q").USREC.mean().round().astype(int)
            gg = g.copy()
            gg["q"] = gg.observation_date.dt.to_period("Q")
            rec = gg["q"].map(rq).values
            th = E.refit(X, y, zhat, K)
            info = dict(n=n, Fhat=Fhat, theta=th.tolist(), sizes=np.bincount(zhat).tolist(),
                        recession_share=float(np.nanmean(rec)))
            # which group is the low-growth regime: lower fitted mean
            mu = [y[zhat == k].mean() for k in range(K)]
            low = int(np.argmin(mu))
            pred = (zhat == low).astype(int)
            sens = float(((pred == 1) & (rec == 1)).sum() / (rec == 1).sum())
            spec = float(((pred == 0) & (rec == 0)).sum() / (rec == 0).sum())
            info.update(group_means=mu, sens=sens, spec=spec, bal_acc=(sens + spec) / 2)
            with open(os.path.join(OUT, "e6_gnp_zhat.txt"), "w") as fh:
                fh.write(repr(info) + "\n")
    run_contam(ds)


def run_contam(ds):
    """Planted gross outliers in the tone data.  Each method is compared with
    its own output on the clean data (same seed), on the clean units: this
    measures how far the contamination moves the method.  Agreement with the
    clean least-squares minimiser is recorded as well."""
    K = 2
    x, y = ds["tone"]
    y = jitter(y, "tone")
    X = np.column_stack([np.ones(len(x)), x])
    zhat, _ = E.oracle_k2_line(x, y)

    def meths(yy, rng):
        return {
            "HA50": lambda: E.hard_alternation(X, yy, K, 50, rng)[0],
            "emEM10": lambda: E.em_mixreg(X, yy, K, 10, rng, short=10)[0],
            "vote": lambda: E.eesp(X, yy, K, 12, B, rng).z,
            "median": lambda: E.eesp(X, yy, K, 12, B, rng, agg="median").z,
            "adapt": lambda: V.eesp_adaptive(X, yy, K, 12, B, rng).z,
            "TA-rw(0.25)": lambda: V.ta_adaptive(X, yy, K, 50, rng, alpha0=0.25).z,
            "TA50(a=0.05)": lambda: E.trimmed_alternation(X, yy, K, 50, 0.05, rng).z,
            "TA50(a=0.10)": lambda: E.trimmed_alternation(X, yy, K, 50, 0.10, rng).z,
            "TA50(a=0.25)": lambda: E.trimmed_alternation(X, yy, K, 50, 0.25, rng).z,
            "bestB-trim(a=0.10)": lambda: E.best_of_B_trimmed(X, yy, K, 12, B, 0.10, rng).z,
        }
    ref = {}
    for meth, fn in meths(y, np.random.default_rng(seed_for("E6clean", 0))).items():
        ref[meth] = np.asarray(fn())
    crow = []
    for s in range(NSEED):
        rng = np.random.default_rng(seed_for("E6c", s))
        yc = y.copy()
        out = np.zeros(len(y), dtype=bool)
        out[rng.choice(len(y), size=15, replace=False)] = True
        yc[out] += rng.choice([-1, 1], out.sum()) * rng.uniform(1.0, 2.0, out.sum())
        cl = ~out
        for meth, fn in meths(yc, rng).items():
            z = np.asarray(fn())
            crow.append(dict(seed=s, method=meth,
                             agree_own_clean=E.agreement(z[cl], ref[meth][cl], K),
                             agree_zhat_clean=E.agreement(z[cl], zhat[cl], K),
                             ref_agree_zhat=E.agreement(ref[meth], zhat, K)))
    pd.DataFrame(crow).to_csv(os.path.join(OUT, "e6_contam.csv"), index=False)
    print("done", flush=True)


def run_flags(ds):
    """Adaptive procedure on the tone data: estimated trimming level on the
    clean data and on the 50 contamination patterns of run_contam, and the
    fraction of planted outliers flagged."""
    K = 2
    x, y = ds["tone"]
    y = jitter(y, "tone")
    X = np.column_stack([np.ones(len(x)), x])
    f = V.eesp_adaptive(X, y, K, 12, B, np.random.default_rng(seed_for("E6clean", 0)))
    rows = [dict(seed=-1, alpha_hat=f.alpha_hat, recall=np.nan)]
    for s in range(NSEED):
        rng = np.random.default_rng(seed_for("E6c", s))
        yc = y.copy()
        out = np.zeros(len(y), dtype=bool)
        out[rng.choice(len(y), size=15, replace=False)] = True
        yc[out] += rng.choice([-1, 1], out.sum()) * rng.uniform(1.0, 2.0, out.sum())
        f = V.eesp_adaptive(X, yc, K, 12, B, np.random.default_rng(seed_for("E6flags", s)))
        rows.append(dict(seed=s, alpha_hat=f.alpha_hat, recall=float(f.flagged[out].mean())))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "e6_flags.csv"), index=False)
    print(d.describe(), flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "flags":
        run_flags(load()[0])
    elif len(sys.argv) > 1 and sys.argv[1] == "contam":
        run_contam(load()[0])
    else:
        main()
