"""Fishery data (FSDA; Riani et al. 2008), log scale, K = 2.
Algorithm 1 over 20 seeds; TCLUST-REG over trimming levels (fish_tcr.csv, from
fish_tcr.m); RobMixReg methods (fish_r.csv, from fish_r.R).  Writes the table and figure
of Section 5."""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import py_methods as P  # noqa: E402
import recover2 as RC  # noqa: E402

PAPER = os.path.join(HERE, "..", "paper")
D = pd.read_csv(os.path.join(HERE, "fishery_log.csv"))
lx, ly = D.x1.values, D.y.values
X = np.column_stack([np.ones(len(lx)), lx])

rows = []
fits = []
rows2 = []
fits2 = []
for s in range(20):
    th2, ah2, info, th, ah = RC.algorithm1_r(X, ly, 2, 8, 100, np.random.default_rng(s))
    o = np.argsort(th[:, 0])
    th = th[o]
    fits.append(th)
    rows.append(dict(seed=s, b0_1=th[0, 0], b1_1=th[0, 1], b0_2=th[1, 0], b1_2=th[1, 1],
                     flagged=ah))
    o = np.argsort(th2[:, 0])
    th2 = th2[o]
    fits2.append(th2)
    rows2.append(dict(seed=s, b0_1=th2[0, 0], b1_1=th2[0, 1], b0_2=th2[1, 0], b1_2=th2[1, 1],
                      flagged=ah2, changed=info["accepted"]))
A2 = pd.DataFrame(rows2)
A2.to_csv(os.path.join(HERE, "fish_alg2.csv"), index=False)
# three groups, m = 16
rows3 = []
for s in range(20):
    th3, ah3, info3, _, _ = RC.algorithm1_r(X, ly, 3, 16, 100, np.random.default_rng(100 + s))
    o = np.argsort(th3[:, 0])
    th3 = th3[o]
    rows3.append(dict(seed=s, **{f"b{j}_{k}": th3[k - 1, j] for k in (1, 2, 3) for j in (0, 1)},
                      flagged=ah3, changed=info3["accepted"]))
A3 = pd.DataFrame(rows3)
A3.to_csv(os.path.join(HERE, "fish_k3.csv"), index=False)
print(A3.round(3).to_string())
A = pd.DataFrame(rows)
A.to_csv(os.path.join(HERE, "fish_alg1.csv"), index=False)
print(A.describe().round(3))

T = pd.read_csv(os.path.join(HERE, "fish_tcr.csv"))
R = pd.read_csv(os.path.join(HERE, "fish_r.csv")) if os.path.exists(os.path.join(HERE, "fish_r.csv")) else None


def rng_str(v):
    lo, hi = v.min(), v.max()
    return f"{np.median(v):.2f}" if hi - lo < 0.005 else f"{np.median(v):.2f} [{lo:.2f}, {hi:.2f}]"


lines = ["\\begin{tabular}{lccc}", "\\toprule",
         "Method & Group 1: intercept, slope & Group 2: intercept, slope & Trimmed or flagged \\\\",
         "\\midrule"]
lines.append("Procedure (20 seeds) & " + rng_str(A2.b0_1) + ", " + rng_str(A2.b1_1) + " & " +
             rng_str(A2.b0_2) + ", " + rng_str(A2.b1_2) + " & " + rng_str(A2.flagged) + " \\\\")
lines.append("Algorithm 1 alone (20 seeds) & " + rng_str(A.b0_1) + ", " + rng_str(A.b1_1) + " & " +
             rng_str(A.b0_2) + ", " + rng_str(A.b1_2) + " & " + rng_str(A.flagged) + " \\\\")
for a in [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]:
    g = T[np.isclose(T.alpha, a)]
    lines.append(f"TCLUST-REG, $\\alpha={a:.2f}$ (10 seeds) & " + rng_str(g.b0_1) + ", " +
                 rng_str(g.b1_1) + " & " + rng_str(g.b0_2) + ", " + rng_str(g.b1_2) +
                 f" & {a:.2f} \\\\")
if R is not None:
    for m, g in R.groupby("method"):
        g = g.copy()
        cols = [c for c in g.columns if c.startswith("X")]
        b = g[cols].values
        # order the two groups by intercept
        b01 = np.where(b[:, 0] <= b[:, 2], b[:, 0], b[:, 2])
        b11 = np.where(b[:, 0] <= b[:, 2], b[:, 1], b[:, 3])
        b02 = np.where(b[:, 0] <= b[:, 2], b[:, 2], b[:, 0])
        b12 = np.where(b[:, 0] <= b[:, 2], b[:, 3], b[:, 1])
        name = {"CTLE": "CTLE"}.get(m, m.replace("tcwm(", "Trimmed CWM, $\\alpha=").replace(")", "$"))
        lines.append(f"{name} ({len(g)} seeds) & " + rng_str(pd.Series(b01)) + ", " +
                     rng_str(pd.Series(b11)) + " & " + rng_str(pd.Series(b02)) + ", " +
                     rng_str(pd.Series(b12)) + " & " + rng_str(g.trimmed) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(PAPER, "tables", "fishery.tex"), "w").write("\n".join(lines))

# figure: data, Algorithm 1 fit with flagged units, TCLUST-REG at 0.04 and 0.25
th = fits2[0]
fl, s, z = P._flag_all(X, ly, th, 2.5, True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
xx = np.linspace(lx.min(), lx.max(), 10)
for ax, (title, a) in zip(axes, [("Algorithm 1", None), ("TCLUST-REG", None)]):
    ax.scatter(lx[~fl], ly[~fl], s=7, c=np.where(z[~fl] == 0, "#1f77b4", "#ff7f0e"), lw=0)
    ax.scatter(lx[fl], ly[fl], s=16, marker="x", c="k", lw=0.8)
    ax.set_xlabel("log quantity", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("log value", fontsize=9)
for k in range(2):
    axes[0].plot(xx, th[k, 0] + th[k, 1] * xx, "k-", lw=1.2)
axes[0].set_title("ESF (flagged units as crosses)", fontsize=9)
for a, ls in [(0.04, "--"), (0.25, "-")]:
    g = T[np.isclose(T.alpha, a)].iloc[0]
    for b0, b1 in [(g.b0_1, g.b1_1), (g.b0_2, g.b1_2)]:
        axes[1].plot(xx, b0 + b1 * xx, color="k", ls=ls, lw=1.2,
                     label=f"$\\alpha={a:.2f}$" if b0 == g.b0_1 else None)
axes[1].legend(fontsize=8, frameon=False)
axes[1].set_title("TCLUST-REG at two trimming levels (points as left)", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(PAPER, "figures", "fishery.pdf"))
print("flagged at seed 0:", fl.mean(), " cheap trades among flagged (residual<0):",
      np.mean((ly - (X @ th.T).max(axis=1))[fl] < 0))
