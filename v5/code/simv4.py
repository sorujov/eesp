"""v4 Gate-1 designs: data generation shared by the Python, R and Octave runners.

A unit is one data set, identified by (design, eps, rep).  Data are generated
deterministically from the unit, written to CSV (columns x1..xp, y, z0, out),
and every method in every language reads the same file.
"""
import os
import zlib

import numpy as np

from simcommon import truth

STUDY = os.environ.get("V4_STUDY", "1")
if STUDY == "1h":
    # Plan 5 (pre-registered 27 Sep 2026): skewed errors, three covariates with unequal groups,
    # and error scale growing with the covariate
    SEED0 = 20261006
    EPS = [0.0, 0.10, 0.20]
    REPS = int(os.environ.get("V4_REPS", "50"))
elif STUDY == "1g":
    # Plan 4 (pre-registered 27 Sep 2026, review round 3): fresh-data confirmation of the
    # recovery step on the small-group designs and two designs not tried before
    SEED0 = 20261005
    EPS = [0.0, 0.10, 0.20]
    REPS = int(os.environ.get("V4_REPS", "50"))
elif STUDY == "taxi":
    # real data (added in review round 3): random subsamples of the taxi trips from JFK airport
    SEED0 = 20261004
    EPS = [0.0]
    REPS = int(os.environ.get("V4_REPS", "20"))
elif STUDY == "1b":
    # Gate 1b (pre-registered 27 Sept 2026 after Gate 1): fresh seeds,
    # contamination at and beyond the conservative level 0.25
    SEED0 = 20260928
    EPS = [0.25, 0.30, 0.35]
    REPS = int(os.environ.get("V4_REPS", "100"))
elif STUDY == "1f":
    # part 3, continued (descriptive, after Gate 2): heavy-tailed errors
    SEED0 = 20261003
    EPS = [0.0, 0.10, 0.20]
    REPS = int(os.environ.get("V4_REPS", "50"))
elif STUDY == "1e":
    # part 3, continued (descriptive): small groups; outliers concentrated in a cluster
    SEED0 = 20261001
    EPS = [0.0, 0.10, 0.20, 0.30]
    REPS = int(os.environ.get("V4_REPS", "50"))
elif STUDY == "1d":
    # part 3, continued (descriptive): moderate outliers in the response
    SEED0 = 20260930
    EPS = [0.05, 0.10, 0.20, 0.30]
    REPS = int(os.environ.get("V4_REPS", "50"))
elif STUDY == "1c":
    # Gate 1c (27 Sept 2026, descriptive): sample size, four groups, correlated covariates
    SEED0 = 20260929
    EPS = [0.0, 0.10, 0.20, 0.30]
    REPS = int(os.environ.get("V4_REPS", "50"))
else:
    SEED0 = 20260927
    EPS = [0.0, 0.05, 0.10, 0.20, 0.30]
    REPS = int(os.environ.get("V4_REPS", "50"))

# name: K, geom, delta, n, p, m, extra
DESIGNS = {
    "D1-K2cross":    dict(K=2, geom="cross", delta=9.0, n=300, p=1, m=8),
    "D2-K3par":      dict(K=3, geom="par", delta=3.0, n=300, p=1, m=12),
    "D3-unequal":    dict(K=2, geom="cross", delta=9.0, n=300, p=1, m=10, pis=[0.7, 0.3]),
    "D4-hetero":     dict(K=2, geom="cross", delta=9.0, n=300, p=1, m=8, sigmas=[0.5, 1.5]),
    "D5-leverage":   dict(K=2, geom="cross", delta=9.0, n=300, p=1, m=8, otype="leverage"),
    "D6-uniform":    dict(K=2, geom="cross", delta=9.0, n=300, p=1, m=8, otype="uniform"),
    "D7-p3":         dict(K=2, geom="cross", delta=9.0, n=300, p=3, m=12),
    "D8-K2par-sep":  dict(K=2, geom="par", delta=6.0, n=300, p=1, m=8),
}


if STUDY == "1c":
    DESIGNS = {
        "E1-K2cross-n100":  dict(K=2, geom="cross", delta=9.0, n=100, p=1, m=8),
        "E2-K2cross-n1000": dict(K=2, geom="cross", delta=9.0, n=1000, p=1, m=8),
        "E3-unequal-n1000": dict(K=2, geom="cross", delta=9.0, n=1000, p=1, m=10, pis=[0.7, 0.3]),
        "E4-K4par":         dict(K=4, geom="par", delta=4.0, n=400, p=1, m=16),
        "E5-p3corr":        dict(K=2, geom="cross", delta=9.0, n=300, p=3, m=12, corr=0.5),
    }


if STUDY == "1e":
    DESIGNS = {
        "G1-small20": dict(K=2, geom="cross", delta=9.0, n=300, p=1, m=16, pis=[0.8, 0.2]),
        "G2-small15": dict(K=2, geom="cross", delta=9.0, n=300, p=1, m=20, pis=[0.85, 0.15]),
        "G3-cluster": dict(K=2, geom="cross", delta=9.0, n=300, p=1, m=8, otype="cluster"),
    }

if STUDY == "1f":
    DESIGNS = {
        "H1-K2cross-t3": dict(K=2, geom="cross", delta=9.0, n=300, p=1, m=8, err="t3"),
        "H2-unequal-t3": dict(K=2, geom="cross", delta=9.0, n=300, p=1, m=10, pis=[0.7, 0.3], err="t3"),
    }

if STUDY == "1h":
    DESIGNS = {
        "S1-skew": dict(K=2, geom="cross", delta=9.0, n=300, p=1, m=8, err="chi3"),
        "S2-p3-unequal": dict(K=2, geom="cross", delta=9.0, n=300, p=3, m=16, pis=[0.7, 0.3]),
        "S3-xhet": dict(K=2, geom="cross", delta=9.0, n=300, p=1, m=8, xhet=True),
    }

if STUDY == "1g":
    DESIGNS = {
        "G1f-small20": dict(K=2, geom="cross", delta=9.0, n=300, p=1, m=16, pis=[0.8, 0.2]),
        "G2f-small15": dict(K=2, geom="cross", delta=9.0, n=300, p=1, m=20, pis=[0.85, 0.15]),
        "G4-small20-p3": dict(K=2, geom="cross", delta=9.0, n=300, p=3, m=20, pis=[0.8, 0.2]),
        "G5-small15-par": dict(K=2, geom="par", delta=6.0, n=300, p=1, m=20, pis=[0.85, 0.15]),
    }

if STUDY == "taxi":
    # fare on trip distance and duration; group 0 metered fare (rate code 1), group 1 the flat
    # fare to Manhattan (rate code 2); reference lines: flat 70, metered from a robust fit to all
    # rate-code-1 trips of the month (fit_taxi_reference in taxi_data.py)
    DESIGNS = {"T1-taxi": dict(K=2, geom=None, delta=None, n=2000, p=2, m=10, taxi=True,
                               theta0=[[2.398, 3.064, 0.334], [70.0, 0.0, 0.0]])}

if STUDY == "1d":
    DESIGNS = {
        "F1-K2cross-moderate": dict(K=2, geom="cross", delta=9.0, n=300, p=1, m=8, otype="moderate"),
        "F2-unequal-moderate": dict(K=2, geom="cross", delta=9.0, n=300, p=1, m=10, pis=[0.7, 0.3],
                                    otype="moderate"),
    }


def seed_for(*parts):
    s = "|".join(str(p) for p in parts) + f"|{SEED0}"
    return zlib.crc32(s.encode()) & 0x7FFFFFFF


def units():
    # ordered by replicate first, so that any contiguous range of units mixes
    # all designs (and the costly D5/D6 units are spread over the chunks)
    return [(d, e, r) for r in range(REPS) for d in DESIGNS for e in EPS]


def unit_name(u):
    d, e, r = u
    return f"{d}_e{int(round(100 * e)):02d}_r{r:03d}"


def generate(u):
    """Returns X (with intercept), y, z0, out (bool)."""
    d, eps, r = u
    D = DESIGNS[d]
    rng = np.random.default_rng(seed_for("v4data", d, eps, r))
    K, n, p = D["K"], D["n"], D["p"]
    if D.get("taxi"):
        import taxi_data
        return taxi_data.sample(rng, n)
    if D.get("corr"):
        # uniform marginals on [-3, 3] with a Gaussian copula of equicorrelation rho
        from scipy.stats import norm
        rho = D["corr"]
        C = np.full((p, p), rho) + (1 - rho) * np.eye(p)
        zz = rng.multivariate_normal(np.zeros(p), C, size=n)
        x = 6.0 * norm.cdf(zz) - 3.0
    else:
        x = rng.uniform(-3, 3, size=(n, p))
    pis = np.full(K, 1.0 / K) if "pis" not in D else np.asarray(D["pis"], float)
    z0 = rng.choice(K, size=n, p=pis)
    th = truth(K, D["geom"], D["delta"])
    mean = th[z0, 0] + th[z0, 1] * x[:, 0]
    if p > 1:
        mean = mean + x[:, 1:].sum(axis=1)
    sig = np.asarray(D.get("sigmas", 1.0), float)
    s_unit = sig[z0] if sig.ndim == 1 else np.full(n, float(sig))
    if D.get("xhet"):
        # error scale growing with the first covariate: 0.5 at x = 0, 2 at |x| = 3
        s_unit = s_unit * (0.5 + 0.5 * np.abs(x[:, 0]))
    if D.get("err") == "chi3":
        # skewed errors: chi-square with 3 degrees of freedom, centred and scaled to unit
        # variance, built from normal draws only
        ch = (rng.normal(size=(n, 3)) ** 2).sum(axis=1)
        y = mean + s_unit * (ch - 3.0) / np.sqrt(6.0)
    elif D.get("err") == "t3":
        # Student t with 3 degrees of freedom, scaled to unit variance, built from normal
        # draws only (the normal generator is identical across numpy versions)
        e0 = rng.normal(size=n)
        ch = (rng.normal(size=(n, 3)) ** 2).sum(axis=1)
        y = mean + s_unit * e0 / np.sqrt(ch / 3.0) / np.sqrt(3.0)
    else:
        y = mean + s_unit * rng.normal(size=n)
    out = np.zeros(n, dtype=bool)
    if eps > 0:
        out = rng.random(n) < eps
        k = int(out.sum())
        ot = D.get("otype", "vertical")
        if ot == "vertical":
            y[out] += rng.choice([-1.0, 1.0], k) * rng.uniform(15, 30, k) * float(np.mean(sig))
        elif ot == "moderate":
            # moderate outliers: shifted by 4 to 8 noise scales, beyond the flagging cut-off
            # but within reach of a fitted line
            y[out] += rng.choice([-1.0, 1.0], k) * rng.uniform(4, 8, k) * float(np.mean(sig))
        elif ot == "cluster":
            # outliers concentrated in a tight cloud away from both lines
            x[out, 0] = rng.normal(2.0, 0.3, k)
            y[out] = rng.normal(20.0, 1.0, k) * float(np.mean(sig))
        elif ot == "leverage":
            x[out, 0] = rng.uniform(6, 9, k)
            y[out] = rng.uniform(-3, 3, k)
        elif ot == "uniform":
            # uniform noise over the bounding box of the clean data, enlarged by 20%;
            # points closer than 3 sigma to a true line are redrawn, so every
            # outlier is a genuine outlier
            cl = ~out
            lo, hi = y[cl].min(), y[cl].max()
            w = hi - lo
            idx = np.where(out)[0]
            for i in idx:
                while True:
                    xi = rng.uniform(-3.6, 3.6)
                    yi = rng.uniform(lo - 0.2 * w, hi + 0.2 * w)
                    dist = np.min(np.abs(yi - (th[:, 0] + th[:, 1] * xi)))
                    if dist > 3.0 * float(np.max(sig)):
                        break
                x[i, 0], y[i] = xi, yi
    X = np.column_stack([np.ones(n), x])
    return X, y, z0, out


def write_csv(u, folder):
    X, y, z0, out = generate(u)
    p = X.shape[1] - 1
    arr = np.column_stack([X[:, 1:], y, z0, out.astype(int)])
    hdr = ",".join([f"x{j+1}" for j in range(p)] + ["y", "z0", "out"])
    fn = os.path.join(folder, unit_name(u) + ".csv")
    np.savetxt(fn, arr, delimiter=",", header=hdr, comments="", fmt="%.10g")
    return fn


if __name__ == "__main__":
    import sys
    folder = sys.argv[1]
    os.makedirs(folder, exist_ok=True)
    U = units()
    a = int(os.environ.get("UNIT_START", 0))
    b = int(os.environ.get("UNIT_END", len(U)))
    for u in U[a:b]:
        write_csv(u, folder)
    print(f"wrote {b - a} files")
