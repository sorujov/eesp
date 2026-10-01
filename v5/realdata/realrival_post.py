"""Round 8: TCLUST-REG long search on fishery and tone data, then reweighting and Algorithm 2."""
import sys; sys.path.insert(0, "../code")
import numpy as np, pandas as pd
import py_methods as P, recover2 as R
T = pd.read_csv("realrival.csv")
data = {}
D = pd.read_csv("fishery_log.csv"); data["fishery_log"] = (np.column_stack([np.ones(len(D)), D.x1.values]), D.y.values)
for f in ["tone_clean", "tone_out"]:
    Q = pd.read_csv(f + ".csv"); data[f] = (np.column_stack([np.ones(len(Q)), Q.stretchratio.values]), Q.tuned.values)
rows = []
for _, r in T.iterrows():
    X, y = data[r.data]
    th0 = np.array([[r.b0_1, r.b1_1], [r.b0_2, r.b1_2]])
    th, ah = P.reweight(X, y, th0, xtrim=False)
    th2, a2, acc = th, ah, 0
    rng = np.random.default_rng([int(r.seed), int(r.alpha * 100)])
    for _ in range(3):
        th3, a3, info = R.recover(X, y, th2, rng)
        if not info["accepted"]:
            break
        th2, a2, acc = th3, a3, acc + 1
    key = 0 if r.data == "fishery_log" else 1
    for lab, t, a in (("rw", th, ah), ("rw+A2", th2, a2)):
        o = np.argsort(t[:, key]); t = t[o]
        rows.append(dict(data=r.data, alpha=r.alpha, seed=r.seed, step=lab, b0_1=t[0, 0], b1_1=t[0, 1], b0_2=t[1, 0], b1_2=t[1, 1], flagged=a, accepted=acc))
out = pd.DataFrame(rows); out.to_csv("realrival_post.csv", index=False)
print(out.groupby(["data", "alpha", "step"])[["b0_1", "b1_1", "b0_2", "b1_2", "flagged", "accepted"]].agg(["median", "min", "max"]).round(3).to_string())
