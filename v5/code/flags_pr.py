"""Precision and recall of the flagged set of the procedure (Algorithm 2 fit), by cell."""
import os, sys
import numpy as np, pandas as pd
import simv4, py_methods as P
FOLD = sys.argv[1]
A = pd.read_csv(os.path.join(FOLD, "alg2.csv")).set_index("unit")
rows = []
for u in simv4.units():
    name = simv4.unit_name(u); r = A.loc[name]
    X, y, z0, out = simv4.generate(u); K, d = simv4.DESIGNS[u[0]]["K"], X.shape[1]
    th = np.array([r[f"t{j+1}"] for j in range(K * d)], float).reshape(K, d)
    fl, _, _ = P._flag_all(X, y, th, 2.5, True, 0.975)
    rows.append(dict(design=u[0], eps=u[1], rep=u[2], clean_flagged=fl[~out].mean(),
                     recall=fl[out].mean() if out.any() else np.nan,
                     precision=out[fl].mean() if fl.any() else np.nan))
D = pd.DataFrame(rows); D.to_csv(os.path.join(FOLD, "flags_pr.csv"), index=False)
print(D.groupby(["design", "eps"])[["clean_flagged", "recall", "precision"]].mean().round(3).to_string())
