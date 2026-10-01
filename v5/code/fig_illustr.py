"""Illustrative figure: one data set with groups of 85% and 15% and 10% outliers
(G2-small15_e10_r008, third part of the study).  Stored fits of Algorithm 1 (py_*.csv) and of
the procedure (alg2.csv); flags by the rule of the reweighting step (_flag_all, c = 2.5)."""
import glob, os
os.environ["V4_STUDY"] = "1e"
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import simv4, evaluate as EV, py_methods as P
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
F = os.path.join(ROOT, "results_v4e")
NAME = "G2-small15_e10_r008"
u = [v for v in simv4.units() if simv4.unit_name(v) == NAME][0]
X, y, z0, out = simv4.generate(u)
K, d = 2, X.shape[1]
A = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(F, "py_*.csv"))])
r1 = A[(A.unit == NAME) & (A.method == "adaptX")].iloc[0]
th1 = np.array([r1[f"b{j+1}"] for j in range(K * d)], float).reshape(K, d)
r2 = pd.read_csv(os.path.join(F, "alg2.csv")).set_index("unit").loc[NAME]
th2 = np.array([r2[f"t{j+1}"] for j in range(K * d)], float).reshape(K, d)
x = X[:, 1]; xs = np.linspace(x.min(), x.max(), 50)
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
ylim = (np.percentile(y, 0.5) - 1, np.percentile(y, 99.5) + 1)
big = (~out) & (z0 == np.bincount(z0[~out]).argmax()); small = (~out) & ~big
ax[0].scatter(x[big], y[big], s=6, c="0.55", label="large group")
ax[0].scatter(x[small], y[small], s=10, c="tab:blue", marker="^", label="small group")
ax[0].scatter(x[out], y[out], s=10, c="tab:red", marker="x", label="outliers")
ax[0].set_title("(a) data and true groups\n", fontsize=9); ax[0].legend(fontsize=7, loc="upper left")
for j, (th, title) in enumerate([(th1, "(b) ESF alone"), (th2, "(c) ESF (after the recovery step)")]):
    fl, _, _ = P._flag_all(X, y, th, 2.5, False, 0.975)
    a = ax[j + 1]
    a.scatter(x[~fl], y[~fl], s=6, c="0.55", label="not flagged")
    a.scatter(x[fl], y[fl], s=10, facecolors="none", edgecolors="k", label="flagged")
    for k in range(K):
        a.plot(xs, th[k, 0] + th[k, 1] * xs, lw=1.6)
    acc = EV.acc_clean(X, y, th, z0, ~out)
    a.set_title(f"{title}\naccuracy {acc:.2f}, flagged fraction {fl.mean():.2f}", fontsize=9)
    a.legend(fontsize=7, loc="upper left")
for a in ax:
    a.set_xlabel("$x$"); a.set_ylim(*ylim)
ax[0].set_ylabel("$y$")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "paper", "figures", "illustr.pdf"))
print(th1.round(2), th2.round(2), r1.alpha_hat, r2.ahat)
