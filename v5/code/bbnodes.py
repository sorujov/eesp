"""Branch-and-bound cost on subsamples of the simulated data (eps = 0.10): nodes and time."""
import os, time, math
import numpy as np, pandas as pd
import eespcore as E
rows = []
for st, des, K, ms in [("1", "D1-K2cross", 2, [8, 12, 16]), ("1", "D2-K3par", 3, [12, 16]), ("1c", "E4-K4par", 4, [16, 20]), ("1", "D7-p3", 2, [12])]:
    os.environ["V4_STUDY"] = st
    import importlib, simv4; importlib.reload(simv4)
    us = [u for u in simv4.units() if u[0] == des and abs(u[1] - 0.1) < 1e-9][:10]
    for m in ms:
        N, T = [], []
        for u in us:
            X, y, z, o = simv4.generate(u); rng = np.random.default_rng(7)
            for r in range(20):
                S = rng.choice(len(y), m, replace=False)
                t = time.perf_counter(); lab, v, n = E.exact_solve(X[S], y[S], K, rng); T.append(time.perf_counter() - t); N.append(n)
        rows.append(dict(design=des, K=K, d=X.shape[1], m=m, nodes_med=np.median(N), nodes_p90=np.percentile(N, 90),
                         time_med=np.median(T), time_p90=np.percentile(T, 90), labellings=K ** m / math.factorial(K)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("../results_v4/bbnodes.csv", index=False)
