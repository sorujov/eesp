"""Taxi trips from JFK airport, NYC TLC yellow taxi records, March 2024 (public data).

data/taxi_jfk_2024-03.csv: all 157,264 trips picked up in zone 132 (JFK) with a rate code;
columns distance (miles), duration (minutes), fare (fare_amount, USD), ratecode.
Model: fare = b0 + b1 distance + b2 duration, two groups: rate code 1 (standard metered fare)
and rate code 2 (flat fare between JFK and Manhattan, 70 USD).  Contamination: the other rate
codes (Nassau/Westchester, negotiated, Newark, unknown) and invalid records (fare <= 0,
distance <= 0, duration <= 0 or above 180 minutes).  No record is removed."""
import os

import numpy as np
import pandas as pd

_F = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "taxi_jfk_2024-03.csv")
_D = None


def load():
    global _D
    if _D is None:
        _D = pd.read_csv(_F)
    return _D


def arrays(D):
    X = np.column_stack([np.ones(len(D)), D.distance.values, D.duration.values])
    y = D.fare.values.astype(float)
    code = D.ratecode.values
    out = (~np.isin(code, [1, 2])) | (D.fare.values <= 0) | (D.distance.values <= 0) | \
        (D.duration.values <= 0) | (D.duration.values > 180)
    z0 = np.where(code == 2, 1, 0)
    return X, y, z0, out


def sample(rng, n):
    D = load()
    idx = rng.choice(len(D), size=n, replace=False)
    return arrays(D.iloc[np.sort(idx)])


def fit_taxi_reference():
    """Reference metered line: least squares on rate-code-1 trips with valid records, iteratively
    trimmed at three robust scales."""
    X, y, z0, out = arrays(load())
    k = (~out) & (z0 == 0)
    Xk, yk = X[k], y[k]
    b = np.linalg.lstsq(Xk, yk, rcond=None)[0]
    for _ in range(10):
        e = yk - Xk @ b
        s = 1.4826 * np.median(np.abs(e))
        keep = np.abs(e) < 3 * s
        b = np.linalg.lstsq(Xk[keep], yk[keep], rcond=None)[0]
    return b, s, keep.mean()


if __name__ == "__main__":
    b, s, f = fit_taxi_reference()
    print("metered reference line", np.round(b, 3), "robust scale", round(s, 3), "kept", round(f, 3))
    X, y, z0, out = arrays(load())
    print("n", len(y), "contaminated", round(out.mean(), 4), "flat among clean", round(z0[~out].mean(), 3))
