"""Round 12 (descriptive): TCLUST-REG's own assignment against the paper's criterion.  For the
long-search fits at 0.25 and 0.40 in D2, D4 and the four-group design (oct_idx.sh), the accuracy on
clean units (a) by the paper's rule (nearest fitted line) and (b) by TCLUST-REG's assignment
out.idx, the trimmed clean units taking the label of their nearest line; and the fraction of clean
units that TCLUST-REG trims.  Usage: python r12_idx_post.py 'octidx_*.csv' OUT.csv"""
import glob, importlib, itertools, os, sys
import numpy as np, pandas as pd


def best(z, z0, clean, K):
    return max(float(np.mean(np.asarray(p)[z[clean]] == z0[clean])) for p in itertools.permutations(range(K)))


F = pd.concat([pd.read_csv(f) for f in glob.glob(sys.argv[1])], ignore_index=True).drop_duplicates(["unit", "method"])
rows = []
for st in sorted({u.split("/")[0][1:] for u in F.unit}):
    os.environ["V4_STUDY"] = st
    import simv4
    importlib.reload(simv4)
    U = {simv4.unit_name(u): u for u in simv4.units()}
    for _, r in F[F.unit.str.startswith("s" + st + "/")].iterrows():
        u = U[r.unit.split("/")[1]]
        X, y, z0, out = simv4.generate(u)
        K, d = simv4.DESIGNS[u[0]]["K"], X.shape[1]
        b = np.array([r[f"b{j+1}"] for j in range(K * d)], float)
        if not np.isfinite(b).all():
            continue
        th = b.reshape(K, d)
        idx = np.array([int(v) for v in str(r.idx).split(";")])
        near = np.argmin((y[:, None] - X @ th.T) ** 2, axis=1)
        own = np.where(idx > 0, idx - 1, near)
        clean = ~out
        rows.append(dict(study=st, design=u[0], eps=u[1], rep=u[2], method=r.method,
                         acc_nearest=best(near, z0, clean, K), acc_own=best(own, z0, clean, K),
                         clean_trimmed=float(np.mean(idx[clean] <= 0))))
pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
print(len(rows))
