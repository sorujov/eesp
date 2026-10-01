"""Exact solvers for the subsample problem (descriptive, added after the review of 27 Sep 2026).

The subsample problem is the clusterwise least-squares fit of m units into K groups.  Solvers:
  bb      depth-first branch and bound of Algorithm 1 (eespcore._bb_exact), no incumbent
  bb_inc  the same with an incumbent from ten starts of hard alternation
  rbba    repetitive branch and bound (Brusco 2006; Carbonneau, Caporossi and Hansen 2012):
          the problems on the last 1, 2, ..., m units are solved in turn, and the optimal value
          of the units not yet labelled is added to the bound
  miqp_g  mixed-integer quadratic programme with indicator constraints (the mixed
          logical-quadratic formulation of Carbonneau, Caporossi and Hansen 2011), Gurobi 13
  miqp_s  the same programme with big-M constraints, SCIP 10 (open source)
All solvers are run on the same subsamples of the simulated data (10% outliers); values are
checked against each other.  Time limit 60 s per MIQP solve.
Usage: python solver_bench.py OUT.csv"""
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from numba import njit

import eespcore as E

TL = 60.0


@njit(cache=True)
def _bb_rb(X, y, K, init_labels, init_value, lb):
    """eespcore._bb_exact with the repetitive bound: a partial labelling of units 0..i is
    pruned when its value plus lb[i + 1], the optimal value of units i+1..m-1 alone, reaches
    the incumbent (the value of a group is superadditive over disjoint sets of its units)."""
    m, d = X.shape
    best = init_value
    best_lab = init_labels.copy()
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
    while i >= 0:
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
        gn[k] += 1
        for a in range(d):
            gXy[k, a] += X[i, a] * y[i]
            for b in range(d):
                gXX[k, a, b] += X[i, a] * X[i, b]
        gyy[k] += y[i] * y[i]
        saved_sse[i] = gsse[k]
        s_new, _ = E._group_sse(gn[k], gXX[k], gXy[k], gyy[k], d)
        total += s_new - gsse[k]
        gsse[k] = s_new
        if total + lb[i + 1] >= best:
            continue
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


def rbba(X, y, K):
    m = len(y)
    lbs = np.zeros(m + 1)      # lbs[j]: optimal value of units j..m-1
    nodes = 0
    for j in range(m - 1, -1, -1):
        sub = lbs[j:] - 0.0
        v, lab, nd = _bb_rb(X[j:], y[j:], K, np.zeros(m - j, dtype=np.int64), np.inf, sub.copy())
        # the bound array for the subproblem is indexed from its own first unit
        lbs[j] = v
        nodes += nd
    return lab, lbs[0], nodes


def _bigM(X, y, bmax):
    return np.abs(y) + np.abs(X).sum(1) * bmax


def miqp_gurobi(X, y, K, bmax):
    import gurobipy as gp
    from gurobipy import GRB
    m, d = X.shape
    env = gp.Env(params={"OutputFlag": 0, "Threads": 1, "TimeLimit": TL})
    M = gp.Model(env=env)
    b = M.addVars(K, d, lb=-bmax, ub=bmax)
    e = M.addVars(m, lb=-GRB.INFINITY)
    z = M.addVars(m, K, vtype=GRB.BINARY)
    for i in range(m):
        M.addConstr(z.sum(i, "*") == 1)
        for k in range(K):
            if k > i:
                M.addConstr(z[i, k] == 0)      # symmetry: unit i opens at most group i
            M.addGenConstrIndicator(z[i, k], True,
                                    e[i] == y[i] - gp.quicksum(X[i, a] * b[k, a] for a in range(d)))
    M.setObjective(gp.quicksum(e[i] * e[i] for i in range(m)), GRB.MINIMIZE)
    t = time.perf_counter()
    M.optimize()
    t = time.perf_counter() - t
    ok = M.Status == GRB.OPTIMAL
    val = M.ObjVal if M.SolCount else np.nan
    nodes = M.NodeCount
    M.dispose(); env.dispose()
    return val, t, nodes, ok


def miqp_scip(X, y, K, bmax):
    from pyscipopt import Model, quicksum
    m, d = X.shape
    Mv = _bigM(X, y, bmax)
    M = Model()
    M.hideOutput()
    M.setParam("limits/time", TL)
    M.setParam("parallel/maxnthreads", 1)
    b = {(k, a): M.addVar(lb=-bmax, ub=bmax) for k in range(K) for a in range(d)}
    e = {i: M.addVar(lb=0) for i in range(m)}
    z = {(i, k): M.addVar(vtype="B") for i in range(m) for k in range(K)}
    t_ = M.addVar(lb=0)
    for i in range(m):
        M.addCons(quicksum(z[i, k] for k in range(K)) == 1)
        for k in range(K):
            if k > i:
                M.addCons(z[i, k] == 0)
            r = y[i] - quicksum(X[i, a] * b[k, a] for a in range(d))
            M.addCons(e[i] >= r - Mv[i] * (1 - z[i, k]))
            M.addCons(e[i] >= -r - Mv[i] * (1 - z[i, k]))
    M.addCons(t_ >= quicksum(e[i] * e[i] for i in range(m)))
    M.setObjective(t_, "minimize")
    t = time.perf_counter()
    M.optimize()
    t = time.perf_counter() - t
    ok = M.getStatus() == "optimal"
    val = M.getObjVal() if M.getNSols() else np.nan
    return val, t, M.getNNodes(), ok


if __name__ == "__main__":
    import importlib
    rows = []
    cases = [("1", "D1-K2cross", 2, [8, 12, 16, 20, 24]), ("1", "D2-K3par", 3, [12, 16, 20]),
             ("1c", "E4-K4par", 4, [16, 20]), ("1", "D7-p3", 2, [12, 16])]
    reps = int(os.environ.get("BENCH_REPS", "10"))
    tasks = [(st, des, K, m) for st, des, K, ms in cases for m in ms]
    if "TASK_ID" in os.environ:
        tasks = [tasks[int(os.environ["TASK_ID"])]]
    # compile the numba kernels before timing
    Xw = np.column_stack([np.ones(12), np.arange(12.0)]); yw = np.sin(np.arange(12.0))
    E.exact_solve(Xw, yw, 2); rbba(Xw, yw, 2)
    for st, des, K, m0 in tasks:
        ms = [m0]
        os.environ["V4_STUDY"] = st
        import simv4
        importlib.reload(simv4)
        us = [u for u in simv4.units() if u[0] == des and abs(u[1] - 0.1) < 1e-9][:reps]
        for m in ms:
            for j, u in enumerate(us):
                X, y, z, o = simv4.generate(u)
                rng = np.random.default_rng([7, j, m])
                S = rng.choice(len(y), m, replace=False)
                XS, yS = np.ascontiguousarray(X[S]), np.ascontiguousarray(y[S])
                bmax = 50.0
                res = {}
                t = time.perf_counter(); lab, v, n = E.exact_solve(XS, yS, K); res["bb"] = (v, time.perf_counter() - t, n, True)
                t = time.perf_counter(); lab, v, n = E.exact_solve(XS, yS, K, np.random.default_rng(1), n_inc=10); res["bb_inc"] = (v, time.perf_counter() - t, n, True)
                t = time.perf_counter(); lab, v, n = rbba(XS, yS, K); res["rbba"] = (v, time.perf_counter() - t, n, True)
                if m <= 20:
                    res["miqp_g"] = miqp_gurobi(XS, yS, K, bmax)
                    res["miqp_s"] = miqp_scip(XS, yS, K, bmax)
                for s, (v, t, n, ok) in res.items():
                    rows.append(dict(design=des, K=K, d=XS.shape[1], m=m, rep=j, solver=s, value=v,
                                     time=t, nodes=n, optimal=ok,
                                     gap=(v - res["bb"][0]) / max(res["bb"][0], 1e-12)))
                print(des, m, j, {s: (round(r[1], 3), r[3]) for s, r in res.items()}, flush=True)
    out = sys.argv[1]
    if "TASK_ID" in os.environ:
        d = os.environ.get("RESULTS_DIR", "."); os.makedirs(d, exist_ok=True)
        out = os.path.join(d, "bench_%s.csv" % os.environ["TASK_ID"])
    pd.DataFrame(rows).to_csv(out, index=False)
    print("UNITS_DONE=1")
