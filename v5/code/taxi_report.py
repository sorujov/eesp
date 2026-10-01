"""Taxi application (20 random samples of 2,000 trips from JFK): table tables/taxi.tex."""
import os, glob
import numpy as np, pandas as pd
import make_v4_outputs as O
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
F = os.path.join(ROOT, "results_taxi")
M = pd.concat([pd.read_csv(os.path.join(F, f)) for f in ["metrics.csv", "alg2.csv", "laplace.csv", "cnmix.csv", "noisemix.csv"]])
b5 = pd.read_csv(os.path.join(F, "best5.csv"))
M = pd.concat([M, b5[["design", "eps", "rep", "unit"]].assign(method="Alg2b5", acc=b5.acc_best5)])
TL = pd.read_csv(os.path.join(F, "long", "tcrlong_taxi.csv"))
M = pd.concat([M, TL[TL.method.str.contains("rw")]])
M.loc[M.acc.isna(), "acc"] = 0.0
if "ahat" not in M.columns:
    M["ahat"] = np.nan
M["ahat"] = M["ahat"].fillna(M.get("alpha_hat"))
G = M.groupby("method").agg(acc=("acc", "mean"), accmin=("acc", "min"), perr=("perr", "median"),
                            ahat=("ahat", "mean"), time=("time", "median"))
LONG = ["TCRlong(0.25)+rw+A2", "TCRlong(0.40)+rw+A2", "TCRlong(0.25)+rw", "TCRlong(0.40)+rw"]
rows = [m for m in O.ROWS if m in G.index and m not in ("adapt", "TA(0.05)")]
k = rows.index("adaptX") + 1
rows = rows[:k] + LONG + rows[k:]
L = ["\\begin{tabular}{lccccc}", "\\toprule",
     "Method & Accuracy, mean & Accuracy, worst & Coefficient error & Flagged or trimmed & Time (s) \\\\", "\\midrule"]
for m in rows:
    r = G.loc[m]
    pe = "--" if not np.isfinite(r.perr) else (f"{r.perr:.2f}" if r.perr < 100 else f"{r.perr:.0f}")
    ah = "--" if not np.isfinite(r.ahat) else f"{r.ahat:.2f}"
    tm = "--" if not np.isfinite(r.time) else (f"{r.time:.2f}" if r.time < 10 else f"{r.time:.0f}")
    L.append(f"{O.label(m)} & {r.acc:.3f} & {r.accmin:.3f} & {pe} & {ah} & {tm} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(ROOT, "paper", "tables", "taxi.tex"), "w").write("\n".join(L))
print(G.loc[rows].round(3).to_string())
A = pd.read_csv(os.path.join(F, "alg2.csv"))
th = A[[f"t{j}" for j in range(1, 7)]].values.reshape(-1, 2, 3)
flat = np.argmin(np.abs(th[:, :, 1]) + np.abs(th[:, :, 2]), axis=1)
met = th[np.arange(len(th)), 1 - flat]; fl = th[np.arange(len(th)), flat]
print("metered line median", np.median(met, 0).round(3), "range", met.min(0).round(2), met.max(0).round(2))
print("flat line median", np.median(fl, 0).round(3))
