"""Unit tests for eespcore.  Run:  python3 test_eespcore.py"""
import itertools
import sys

import numpy as np

import eespcore as E


def lstsq_value(X, y, z, K):
    v = 0.0
    for k in range(K):
        mk = z == k
        if mk.sum() == 0:
            continue
        b = np.linalg.lstsq(X[mk], y[mk], rcond=None)[0]
        v += float(((y[mk] - X[mk] @ b) ** 2).sum())
    return v


def lstsq_brute(X, y, K):
    m = len(y)
    best = np.inf
    for z in itertools.product(range(K), repeat=m):
        best = min(best, lstsq_value(X, y, np.array(z), K))
    return best


def test_branch_and_bound(n_inst=200):
    rng = np.random.default_rng(1)
    cases = [(2, 1, 10), (2, 2, 10), (3, 2, 8), (2, 3, 9)]
    bad = 0
    for t in range(n_inst):
        K, d, m = cases[t % 4]
        ties = t % 3 == 0
        if ties:
            cols = [rng.integers(0, 3, m).astype(float) for _ in range(d - 1)]
            y = rng.normal(size=m).round(1)
        else:
            cols = [rng.normal(size=m) for _ in range(d - 1)]
            y = rng.normal(size=m)
        X = np.column_stack([np.ones(m)] + cols)
        ref = lstsq_brute(X, y, K) if m <= 9 else E.brute_force(X, y, K)[1]
        lab, v, _ = E.exact_solve(X, y, K)
        lab2, v2, _ = E.exact_solve(X, y, K, rng=rng, n_inc=3)
        vv = lstsq_value(X, y, lab, K)
        if not (abs(v - ref) < 1e-7 * (1 + ref) and abs(v2 - ref) < 1e-7 * (1 + ref)
                and abs(vv - ref) < 1e-7 * (1 + ref)):
            bad += 1
            print("B&B mismatch", K, d, m, ties, v, v2, vv, ref)
    assert bad == 0, bad
    print("branch and bound: ok on", n_inst, "instances (one third with tied x)")


def test_oracle_line(n_inst=40):
    rng = np.random.default_rng(2)
    bad = 0
    for t in range(n_inst):
        m = 11 + t % 3
        if t % 4 == 0:
            x = rng.integers(0, 4, m).astype(float)       # tied x
            y = rng.normal(size=m).round(2) + 1e-7 * rng.normal(size=m)
        else:
            x = rng.uniform(-3, 3, m)
            y = rng.normal(size=m) + np.where(rng.random(m) < .5, x, -x)
        X = np.column_stack([np.ones(m), x])
        ref = E.brute_force(X, y, 2)[1]
        z, v = E.oracle_k2_line(x, y)
        vz = lstsq_value(X, y, z, 2)
        if not (abs(v - ref) < 1e-7 * (1 + ref) and abs(vz - ref) < 1e-7 * (1 + ref)):
            bad += 1
            print("oracle mismatch", t, v, vz, ref)
    assert bad == 0
    print("line oracle: ok on", n_inst, "instances")


def test_oracle_location(n_inst=30):
    rng = np.random.default_rng(3)
    for t in range(n_inst):
        K = 2 + t % 3
        m = 9
        y = rng.normal(size=m) + rng.integers(0, K, m) * 3
        X = np.ones((m, 1))
        ref = E.brute_force(X, y, K)[1]
        z, v = E.oracle_location(y, K)
        assert abs(v - ref) < 1e-8 * (1 + ref), (v, ref)
        assert abs(E.criterion(X, y, z, K) - ref) < 1e-8 * (1 + ref)
    print("location oracle: ok on", n_inst, "instances")


def test_align():
    rng = np.random.default_rng(4)
    for K in (2, 3, 4, 5):
        th = rng.normal(size=(K, 2))
        P = rng.permutation(K)
        out, tau = E.align(th[P], th)
        assert np.allclose(out, th)
    print("alignment: ok")


def test_metrics():
    z = np.array([0, 0, 1, 1, 2])
    assert E.agreement(z, (z + 1) % 3, 3) == 1.0
    assert abs(E.ari(z, (z + 1) % 3) - 1.0) < 1e-12
    print("metrics: ok")


def test_polish_monotone(n_inst=50):
    rng = np.random.default_rng(5)
    for t in range(n_inst):
        n = 120
        x = rng.uniform(-3, 3, n)
        y = np.where(rng.random(n) < .5, x, -x) + rng.normal(size=n)
        X = np.column_stack([np.ones(n), x])
        f = E.eesp(X, y, 2, 8, 20, rng)
        g = E.polish(X, y, 2, f)
        assert g.F <= f.F + 1e-9
        assert abs(g.F - E.criterion(X, y, g.z, 2)) < 1e-8 * (1 + g.F)
    print("polish monotone: ok")


def test_eesp_runs():
    rng = np.random.default_rng(6)
    n = 150
    x = rng.uniform(-3, 3, n)
    z0 = rng.integers(0, 2, n)
    y = np.where(z0 == 0, 2 + x, -2 + x) + 0.5 * rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    for agg in ("vote", "param", "median"):
        for solver in ("exact", "multistart", "local"):
            f = E.eesp(X, y, 2, 8, 30, rng, agg=agg, solver=solver)
            a = E.agreement(f.z, z0, 2)
            # averaging one-start local solves is not expected to work
            if not (agg != "vote" and solver == "local"):
                assert a > 0.9, (agg, solver, a)
    f = E.eesp(X, y, 2, 8, 30, rng, T=4)
    assert all(np.diff(f.trace[:f.accepted]) < 0)
    print("eesp variants: ok")


def test_em():
    rng = np.random.default_rng(7)
    n = 300
    x = rng.uniform(-3, 3, n)
    z0 = rng.integers(0, 2, n)
    y = np.where(z0 == 0, 2 + x, -2 - x) + 0.5 * rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    for init in ("labels", "theta"):
        z, th, F, ll, w = E.em_mixreg(X, y, 2, 10, rng, init=init, short=10)
        assert E.agreement(z, z0, 2) > 0.9, init
    print("EM: ok")


if __name__ == "__main__":
    test_metrics()
    test_align()
    test_oracle_location()
    test_oracle_line()
    test_branch_and_bound()
    test_polish_monotone()
    test_eesp_runs()
    test_em()
    print("ALL TESTS PASSED")
