"""Agreement check of TCLUST-REG between MATLAB (FSDA in MATLAB Online, eesp_matlab_check.m) and
the Octave runs of the paper (oct_long.m, 1000 starts, 50 refinement steps, restriction factor 12),
on 80 units of parts 1, 1b and 1c.  Descriptive.  Usage:
OCT_DIR=../results_tcrlong/octlong_all python mlcompare.py ../results_matlab/results_*.csv"""
import glob, importlib, itertools, os, sys
import numpy as np, pandas as pd
import tcrlong_post as TP            # rebuilds the Octave fits (TP.O) with their study labels

M = pd.concat([pd.read_csv(f) for f in sys.argv[1:]], ignore_index=True)
M["study"] = M.unit.str.split("/").str[0].str[1:]
M["name"] = M.unit.str.split("/").str[1]
O = TP.O
rows = []
for st, G in M.groupby("study"):
    os.environ["V4_STUDY"] = st
    import simv4, evaluate as EV, recover2 as R, py_methods as P
    importlib.reload(simv4)
    U = {simv4.unit_name(u): u for u in simv4.units()}
    for _, m in G.iterrows():
        u = U[m["name"]]
        X, y, z0, out = simv4.generate(u)
        K, d = simv4.DESIGNS[u[0]]["K"], X.shape[1]
        o = O[(O.unit == m["name"]) & (O.study == st) & (O.method == m.method)].iloc[0]
        res = dict(study=st, design=u[0], eps=u[1], rep=u[2], method=m.method,
                   time_matlab=m.time, time_octave=o.time, ahat_matlab=m.alpha_hat, ahat_octave=o.alpha_hat)
        ths = {}
        for tag, r in (("matlab", m), ("octave", o)):
            b = np.array([r[f"b{j+1}"] for j in range(K * d)], float)
            ths[tag] = b.reshape(K, d)
        A, B = ths["matlab"], ths["octave"]
        scale = np.abs(B).max()
        res["coef_diff"] = min(np.abs(A[list(p)] - B).max() for p in itertools.permutations(range(K))) / scale
        for tag, th0 in ths.items():
            res[f"acc_{tag}"] = EV.acc_clean(X, y, th0, z0, ~out)
            th, ah = P.reweight(X, y, th0, xtrim=False)
            res[f"accrw_{tag}"] = EV.acc_clean(X, y, th, z0, ~out)
            rng = np.random.default_rng(simv4.seed_for("v4rec_tcrlong", m.method, *u))
            th2 = th
            for _ in range(3):
                th3, a3, info = R.recover(X, y, th2, rng)
                if not info["accepted"]:
                    break
                th2 = th3
            res[f"accA2_{tag}"] = EV.acc_clean(X, y, th2, z0, ~out)
        rows.append(res)
D = pd.DataFrame(rows)
os.makedirs("../results_matlab", exist_ok=True)
D.to_csv("../results_matlab/compare.csv", index=False)
D["same"] = D.coef_diff < 1e-3
g = D.groupby(["study", "design", "eps", "method"])
S = g.agg(n=("same", "size"), same=("same", "sum"), acc_m=("acc_matlab", "mean"), acc_o=("acc_octave", "mean"),
          A2_m=("accA2_matlab", "mean"), A2_o=("accA2_octave", "mean")).reset_index()
print(S.to_string())
for c in ("acc", "accrw", "accA2"):
    dd = D[f"{c}_matlab"] - D[f"{c}_octave"]
    print(c, "mean diff", round(dd.mean(), 4), "max |diff|", round(dd.abs().max(), 4))
print("same fit (rel. coef diff < 1e-3):", int(D.same.sum()), "of", len(D))
print("identical to written digits:", int((D.coef_diff == 0).sum()), " smallest nonzero rel. diff", round(D.coef_diff[D.coef_diff > 0].min(), 4))
