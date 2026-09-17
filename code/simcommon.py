"""Shared data generators, seeding and method registry for the experiments."""
import os
import time
import zlib

import numpy as np

import eespcore as E

ROOT = os.environ.get("EESP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results_v3")
os.makedirs(OUT, exist_ok=True)
SEED0 = int(os.environ.get("EESP_SEED0", "20260916"))


def seed_for(*parts):
    """Deterministic seed from a canonical string (stable across processes)."""
    s = "|".join(str(p) for p in parts) + f"|{SEED0}"
    return zlib.crc32(s.encode()) & 0x7FFFFFFF


def truth(K, geom, delta):
    """Intercepts and slopes of the K generating lines (sigma = 1).
    'par'  : common slope 1, intercepts delta apart;
    'cross': common intercept 0, slopes delta/3 apart (vertical gap delta at
             |x| = 3, the edge of the design)."""
    c = np.arange(K) - (K - 1) / 2
    if geom == "par":
        return np.column_stack([delta * c, np.ones(K)])
    return np.column_stack([np.zeros(K), (delta / 3.0) * c])


def generate(rng, n, K, geom="cross", delta=3.0, sigma=1.0, pis=None,
             err="normal", eps=0.0, p=1):
    """Clusterwise linear regression data.  x ~ U(-3,3)^p; the lines differ in
    the first covariate only (further covariates have coefficient 1 in every
    group).  sigma: common noise scale, or one scale per group.  eps:
    fraction of gross vertical outliers, shifted by +-U(15, 30) times the
    (mean) noise scale.  Returns X (with intercept), y, z0, outlier mask."""
    x = rng.uniform(-3, 3, size=(n, p))
    pis = np.full(K, 1.0 / K) if pis is None else np.asarray(pis, float)
    z0 = rng.choice(K, size=n, p=pis)
    th = truth(K, geom, delta)
    mean = th[z0, 0] + th[z0, 1] * x[:, 0]
    if p > 1:
        mean = mean + x[:, 1:].sum(axis=1)
    if err == "normal":
        e = rng.normal(size=n)
    else:
        e = rng.standard_t(3, size=n) / np.sqrt(3.0)
    sig = np.asarray(sigma, float)
    s_unit = sig[z0] if sig.ndim == 1 else np.full(n, float(sig))
    y = mean + s_unit * e
    out = np.zeros(n, dtype=bool)
    if eps > 0:
        out = rng.random(n) < eps
        scale = float(np.mean(sig))
        y[out] += rng.choice([-1.0, 1.0], out.sum()) * rng.uniform(15, 30, out.sum()) * scale
    X = np.column_stack([np.ones(n), x])
    return X, y, z0, out


def bayes_labels(X, y, K, geom, delta, sigma=1.0, pis=None):
    """Bayes classifier for the Gaussian model (known parameters)."""
    th = truth(K, geom, delta)
    pis = np.full(K, 1.0 / K) if pis is None else np.asarray(pis, float)
    mean = th[:, 0][None, :] + np.outer(X[:, 1], th[:, 1])
    if X.shape[1] > 2:
        mean = mean + X[:, 2:].sum(axis=1)[:, None]
    lp = np.log(pis)[None, :] - 0.5 * (y[:, None] - mean) ** 2 / sigma ** 2
    return np.argmax(lp, axis=1), lp


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def hardware():
    import platform
    try:
        with open("/proc/cpuinfo") as f:
            model = [l.split(":")[1].strip() for l in f if l.startswith("model name")][0]
    except Exception:
        model = platform.processor()
    return f"{model}; Python {platform.python_version()}; numba"
