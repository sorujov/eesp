"""Exact solvers for the admissible subsample problem (descriptive; review round 3, 27 Sep 2026).

Problem: minimise the clusterwise residual sum of squares of m units over labellings into K
groups, each group with at least d + 1 units (the admissibility condition of Algorithm 1).
Solvers, all on one thread and on the same subsamples:
  bb      depth-first branch and bound over restricted-growth labellings; bound = current sum of
          group residual sums of squares; a partial labelling is also pruned when the units left
          cannot bring every group up to d + 1 units
  bb_inc  the same with an incumbent from ten starts of hard alternation (used if admissible)
  rbba    repetitive branch and bound (Brusco 2006; Carbonneau, Caporossi and Hansen 2012): the
          unconstrained problems on the last 1, ..., m-1 units are solved in turn (each with the
          same repetitive bound), and the optimal value of the units not yet labelled is added
          to the bound of the admissible search
  miqp_g  mixed logical-quadratic programme (Carbonneau, Caporossi and Hansen 2011): binary
          labels, free coefficients, indicator constraints z_ik = 1 => e_i = y_i - x_i'b_k,
          quadratic objective, symmetry breaking z_ik = 0 for k > i, group sizes >= d + 1;
          Gurobi 13
  miqp_s  the same with big-M constraints (coefficients bounded by 50, M_i = |y_i| + 50 sum|x_i|)
          and the quadratic objective as an epigraph constraint; SCIP 10 via PySCIPOpt
Time limit 60 s per solve (a node limit of 2e8 for the branch and bound, about a minute).
Usage: TASK_ID=k python solver_bench2.py  (writes RESULTS_DIR/bench2_k.csv)"""
import os
import sys
import time

import numpy as np
import pandas as pd
from numba import njit

import eespcore as E

TL = 60.0
MAXNODES = 200_000_000


@njit(cache=True)
def _bb(X, y, K, init_labels, init_value, lb, min_size, max_nodes):
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
    capped = False
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
        if nodes > max_nodes:
            capped = True
            break
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
        if min_size > 0:
            nu = used[i]
            if k + 1 > nu:
                nu = k + 1
            deficit = (K - nu) * min_size
            for g in range(nu):
                if gn[g] < min_size:
                    deficit += min_size - gn[g]
            if deficit > m - 1 - i:
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
    # restore the state is not needed: arrays are local
    return best, best_lab, nodes, capped


def bb(X, y, K, init=None):
    m, d = X.shape
    z0 = np.zeros(m, dtype=np.int64)
    v0 = np.inf
    if init is not None:
        z0, v0 = init
    v, lab, n, cap = _bb(X, y, K, z0, v0, np.zeros(m + 1), d + 1, MAXNODES)
    return v, n, not cap


def bb_inc(X, y, K, rng):
    m, d = X.shape
    z, _, f, _ = E.hard_alternation(X, y, K, 10, rng)
    z = E._canonical(z).astype(np.int64)
    if np.bincount(z, minlength=K).min() >= d + 1:
        return bb(X, y, K, (z, f * (1 + 1e-12) + 1e-12))
    return bb(X, y, K)


def rbba(X, y, K):
    m, d = X.shape
    lbs = np.zeros(m + 1)
    nodes = 0
    cap = False
    for j in range(m - 1, 0, -1):
        v, lab, nd, c = _bb(X[j:], y[j:], K, np.zeros(m - j, dtype=np.int64), np.inf, lbs[j:].copy(), 0,
                            MAXNODES)
        lbs[j] = v
        nodes += nd
        cap = cap or c
    v, lab, nd, c = _bb(X, y, K, np.zeros(m, dtype=np.int64), np.inf, lbs.copy(), d + 1, MAXNODES)
    return v, nodes + nd, not (cap or c)


def miqp_gurobi(X, y, K):
    import gurobipy as gp
    from gurobipy import GRB
    m, d = X.shape
    env = gp.Env(params={"OutputFlag": 0, "Threads": 1, "TimeLimit": TL})
    M = gp.Model(env=env)
    b = M.addVars(K, d, lb=-GRB.INFINITY)
    e = M.addVars(m, lb=-GRB.INFINITY)
    z = M.addVars(m, K, vtype=GRB.BINARY)
    for i in range(m):
        M.addConstr(z.sum(i, "*") == 1)
        for k in range(K):
            if k > i:
                M.addConstr(z[i, k] == 0)
            M.addGenConstrIndicator(z[i, k], True,
                                    e[i] == y[i] - gp.quicksum(X[i, a] * b[k, a] for a in range(d)))
    for k in range(K):
        M.addConstr(z.sum("*", k) >= d + 1)
    M.setObjective(gp.quicksum(e[i] * e[i] for i in range(m)), GRB.MINIMIZE)
    t = time.perf_counter()
    M.optimize()
    t = time.perf_counter() - t
    ok = M.Status == GRB.OPTIMAL
    val = M.ObjVal if M.SolCount else np.nan
    nodes = M.NodeCount
    M.dispose(); env.dispose()
    return val, t, nodes, ok


def miqp_scip(X, y, K, bmax=50.0):
    from pyscipopt import Model, quicksum
    m, d = X.shape
    Mv = np.abs(y) + np.abs(X).sum(1) * bmax
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
    for k in range(K):
        M.addCons(quicksum(z[i, k] for i in range(m)) >= d + 1)
    M.addCons(t_ >= quicksum(e[i] * e[i] for i in range(m)))
    M.setObjective(t_, "minimize")
    t = time.perf_counter()
    M.optimize()
    t = time.perf_counter() - t
    ok = M.getStatus() == "optimal"
    val = M.getObjVal() if M.getNSols() else np.nan
    return val, t, M.getNNodes(), ok


CASES = [("1", "D1-K2cross", 2, m) for m in [8, 12, 16, 20, 24, 28, 32, 40]] + \
        [("1", "D2-K3par", 3, m) for m in [12, 16, 20, 24]] + \
        [("1c", "E4-K4par", 4, m) for m in [16, 20]] + \
        [("1", "D7-p3", 2, m) for m in [12, 16]]

if __name__ == "__main__":
    import importlib
    reps = int(os.environ.get("BENCH_REPS", "30"))
    tasks = CASES if "TASK_ID" not in os.environ else [CASES[int(os.environ["TASK_ID"])]]
    # compile every numba kernel before timing
    Xw = np.column_stack([np.ones(12), np.arange(12.0)]); yw = np.sin(np.arange(12.0))
    bb(Xw, yw, 2); bb_inc(Xw, yw, 2, np.random.default_rng(0)); rbba(Xw, yw, 2)
    rows = []
    for st, des, K, m in tasks:
        os.environ["V4_STUDY"] = st
        import simv4
        importlib.reload(simv4)
        us = [u for u in simv4.units() if u[0] == des and abs(u[1] - 0.1) < 1e-9]
        for j in range(reps):
            u = us[j % len(us)]
            X, y, z, o = simv4.generate(u)
            rng = np.random.default_rng([11, j, m])
            S = rng.choice(len(y), m, replace=False)
            XS, yS = np.ascontiguousarray(X[S]), np.ascontiguousarray(y[S])
            res = {}
            t = time.perf_counter(); v, n, ok = bb(XS, yS, K); res["bb"] = (v, time.perf_counter() - t, n, ok)
            t = time.perf_counter(); v, n, ok = bb_inc(XS, yS, K, np.random.default_rng(1)); res["bb_inc"] = (v, time.perf_counter() - t, n, ok)
            t = time.perf_counter(); v, n, ok = rbba(XS, yS, K); res["rbba"] = (v, time.perf_counter() - t, n, ok)
            res["miqp_g"] = miqp_gurobi(XS, yS, K)
            res["miqp_s"] = miqp_scip(XS, yS, K)
            ref = min(r[0] for r in res.values() if r[3] and np.isfinite(r[0]))
            for s, (v, t, n, ok) in res.items():
                rows.append(dict(design=des, K=K, d=XS.shape[1], m=m, rep=j, solver=s, value=v, time=t,
                                 nodes=n, optimal=ok, gap=(v - ref) / max(ref, 1e-12)))
            print(des, m, j, {s: (round(r[1], 3), r[3]) for s, r in res.items()}, flush=True)
    d = os.environ.get("RESULTS_DIR", ".")
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(d, "bench2_%s.csv" % os.environ.get("TASK_ID", "all")), index=False)
    print("UNITS_DONE=1")
