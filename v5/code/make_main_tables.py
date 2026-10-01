"""Main comparison tables after round 8 (all parts of the study, long-search TCLUST-REG rivals).
Writes paper/tables/main_low.tex, main_high.tex, levelchoice.tex and results_v4/allcells.csv."""
import glob, os
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TAB = os.path.join(ROOT, "paper", "tables")
FOLD = {"1": "results_v4", "1b": "results_v4b", "1c": "results_v4c", "1d": "results_v4d",
        "1e": "results_v4e", "1f": "results_v4f", "1g": "results_v4g", "1h": "results_v4h"}
EXTRA = ["alg2.csv", "laplace.csv", "cnmix.csv", "noisemix.csv", "seqransac.csv"]
KEEP = ["Alg2", "ESF12", "adaptX", "TCRlong(0.25)+rw+A2", "TCRlong(0.40)+rw+A2", "TCRlong(0.30)+rw+A2", "TCReqwL(0.25)+rw+A2", "TCRlong(0.25)+rw",
        "TCRlong(0.40)+rw", "CTLE", "bisq", "Laplace", "CNmix", "NoiseMix", "SeqRANSAC", "HA50"]
R10 = ["TCRlong(0.30)+rw+A2", "TCRlong(0.35)+rw+A2", "TCReqwL(0.25)+rw+A2", "TCReqwL(0.40)+rw+A2"]
LABEL = {"Alg2": "ESF (with the recovery step)", "adaptX": "ESF alone", "ESF12": "ESF with $m=12$ in every design",
         "TCRlong(0.25)+rw+A2": "TCLUST-REG, $\\alpha=0.25$, reweighted, then the recovery step",
         "TCRlong(0.40)+rw+A2": "TCLUST-REG, $\\alpha=0.40$, reweighted, then the recovery step",
         "TCRlong(0.30)+rw+A2": "TCLUST-REG, $\\alpha=0.30$, reweighted, then the recovery step",
         "TCReqwL(0.25)+rw+A2": "TCLUST-REG, $\\alpha=0.25$, equal weights, reweighted, then the recovery step",
         "TCRlong(0.25)+rw": "TCLUST-REG, $\\alpha=0.25$, reweighted",
         "TCRlong(0.40)+rw": "TCLUST-REG, $\\alpha=0.40$, reweighted",
         "CTLE": "CTLE", "bisq": "Bisquare mixture", "Laplace": "Laplace mixture",
         "CNmix": "Contaminated normal mixture", "NoiseMix": "Noise-component mixture",
         "SeqRANSAC": "Sequential RANSAC (oracle: true noise scale given)", "HA50": "Least squares (50 starts)"}


def load():
    parts = []
    for st, f in FOLD.items():
        for fn in ["metrics.csv"] + EXTRA:
            p = os.path.join(ROOT, f, fn)
            if os.path.exists(p):
                d = pd.read_csv(p)[["design", "eps", "rep", "unit", "method", "acc"]]
                d["study"] = st
                parts.append(d)
    T = pd.read_csv(os.path.join(ROOT, "results_tcrlong", "tcrlong_0.csv"))
    parts.append(T[["study", "design", "eps", "rep", "unit", "method", "acc"]])
    for f in sorted(glob.glob(os.path.join(ROOT, "results_r13", "r13_*.csv"))):   # round 13: m = 12
        R = pd.read_csv(f); R["study"] = R.study.astype(str)
        parts.append(R[R.method == "ESF12"][["study", "design", "eps", "rep", "unit", "method", "acc"]])
    for f in sorted(glob.glob(os.path.join(ROOT, "results_r10", "fitspost_*.csv"))):   # round 10, descriptive
        R = pd.read_csv(f); R["study"] = R.study.astype(str)
        parts.append(R[["study", "design", "eps", "rep", "unit", "method", "acc"]])
    M = pd.concat(parts, ignore_index=True)
    M = M[M.method.isin(KEEP + R10)]
    M = M[~((M.study == "1") & (M.eps.round(2) == 0.30))]      # superseded by part two
    M["eps"] = M.eps.round(2)
    M.loc[M.acc.isna(), "acc"] = 0.0
    return M


LOWCOL = [("D1", ["D1-K2cross"]), ("D2", ["D2-K3par"]), ("D3", ["D3-unequal"]), ("D4", ["D4-hetero"]),
          ("D6", ["D6-uniform"]), ("D7", ["D7-p3"]), ("D8", ["D8-K2par-sep"]), ("D5", ["D5-leverage"]),
          ("$K=4$", ["E4-K4par"]),
          ("Small", ["G1-small20", "G2-small15", "G1f-small20", "G2f-small15", "G4-small20-p3", "G5-small15-par"]),
          ("Clust.", ["G3-cluster"]),
          ("Other", ["E1-K2cross-n100", "E2-K2cross-n1000", "E3-unequal-n1000", "E5-p3corr",
                     "F1-K2cross-moderate", "F2-unequal-moderate", "H1-K2cross-t3", "H2-unequal-t3",
                     "S1-skew", "S2-p3-unequal", "S3-xhet"])]


def table(C, lo, fname, caption, label):
    sub = C[(C.eps <= 0.2001) if lo else (C.eps >= 0.2499)]
    cols = [(h, ds) for h, ds in LOWCOL if sub.design.isin(ds).any()]
    L = ["\\begin{table}[!tb]", "\\centering", "\\footnotesize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
         "Method & " + " & ".join(h for h, _ in cols) + " \\\\", "\\midrule"]
    for i, m in enumerate(KEEP):
        if i in (3, 9):
            L.append("\\addlinespace")
        vals = []
        for h, ds in cols:
            v = sub[(sub.method == m) & sub.design.isin(ds)].acc
            s = "--" if v.empty else f"{v.min():.2f}"
            if not v.empty and v.min() < 0.85:
                s = f"\\textit{{{s}}}"
            vals.append(s)
        L.append(LABEL[m] + " & " + " & ".join(vals) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    open(os.path.join(TAB, fname), "w").write("\n".join(L))


def levelchoice(C):
    P = C.pivot_table(index=["study", "design", "eps"], columns="method", values="acc").reset_index()
    P["oracle"] = P[["TCRlong(0.25)+rw+A2", "TCRlong(0.40)+rw+A2"]].max(axis=1)
    P["wrong"] = P[["TCRlong(0.25)+rw+A2", "TCRlong(0.40)+rw+A2"]].min(axis=1)
    P.to_csv(os.path.join(ROOT, "results_v4", "allcells.csv"), index=False)
    rows = [("Alg2", "ESF"), ("ESF12", "ESF, $m=12$ in every design"), ("TCRlong(0.25)+rw+A2", "TCLUST-REG, $\\alpha=0.25$, rw, recovery"),
            ("TCRlong(0.40)+rw+A2", "TCLUST-REG, $\\alpha=0.40$, rw, recovery"),
            ("oracle", "Better of the two levels, chosen afterwards"), ("wrong", "Worse of the two levels"),
            ("TCRlong(0.25)+rw", "TCLUST-REG, $\\alpha=0.25$, rw, no recovery"),
            ("TCRlong(0.40)+rw", "TCLUST-REG, $\\alpha=0.40$, rw, no recovery"),
            ("TCRlong(0.30)+rw+A2", "TCLUST-REG, $\\alpha=0.30$, rw, recovery"),
            ("TCRlong(0.35)+rw+A2", "TCLUST-REG, $\\alpha=0.35$, rw, recovery"),
            ("TCReqwL(0.25)+rw+A2", "TCLUST-REG, $\\alpha=0.25$, equal weights, rw, recovery"),
            ("TCReqwL(0.40)+rw+A2", "TCLUST-REG, $\\alpha=0.40$, equal weights, rw, recovery")]
    lo, hi = P[P.eps <= 0.2001], P[P.eps >= 0.2499]
    L = ["\\begin{tabular}{lcccccc}", "\\toprule",
         " & \\multicolumn{3}{c}{$\\varepsilon\\le0.20$ (" + str(len(lo)) + " cells)} & \\multicolumn{3}{c}{$\\varepsilon\\ge0.25$ (" + str(len(hi)) + " cells)} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "Method & Mean & Below $0.85$ & Worst & Mean & Below $0.85$ & Worst \\\\", "\\midrule"]
    for k, lab in rows:
        if k == "TCRlong(0.30)+rw+A2":
            L.append("\\addlinespace")
        f = lambda Q: f"{Q[k].mean():.3f} & {int((Q[k] < 0.85).sum())} & {Q[k].min():.2f}"
        L.append(f"{lab} & {f(lo)} & {f(hi)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "levelchoice.tex"), "w").write("\n".join(L))
    d = lo["Alg2"] - lo["oracle"]
    print("low cells", len(lo), "high", len(hi))
    print("proc vs oracle low: within 0.03:", int((d.abs() <= 0.03).sum()), "above by >0.03:", int((d > 0.03).sum()),
          "below by >0.03:", int((d < -0.03).sum()))
    dh = hi["Alg2"] - hi["TCRlong(0.40)+rw+A2"]
    print("high: proc below 0.40 rival by >0.03:", int((dh < -0.03).sum()), " above by >0.03:", int((dh > 0.03).sum()))
    print(lo[(d < -0.03) | (d > 0.03)][["study", "design", "eps", "Alg2", "oracle"]].round(3).to_string(index=False))
    return P


if __name__ == "__main__":
    M = load()
    C = M.groupby(["study", "design", "eps", "method"]).acc.mean().reset_index()
    n = M.groupby(["study", "design", "eps", "method"]).size()
    print("cells", C.groupby("method").size().to_dict())
    table(C, True, "main_low.tex",
          "Worst mean accuracy on clean units (the lowest of the cell means) with up to $20\\%$ of outliers ($\\varepsilon\\in\\{0,0.05,0.10,0.20\\}$), by design. D1 to D8: first part ($50$ data sets per cell). $K=4$, small groups (80/20 and 85/15, including the fresh data of Plan~4), clustered outliers and other designs (sample size, moderate outliers, dependent covariates, $t_3$, skewed and heteroscedastic errors): third part and Plans~4 and~5. Column entries are minima over different sets of $\\varepsilon$: D1 to D8 over the four values above, the other columns over the values of $\\varepsilon\\le0.20$ run in the third part and Plans~4 and~5. TCLUST-REG was run with $1000$ random starts and $50$ refinement steps (the rows with the level $0.30$ and with equal weights were run after the others); ``reweighted'' is our two-pass rule ($c=2.5$, median absolute deviation per group). Sequential RANSAC is given the true noise scale and is an oracle benchmark, not a competitor that can be run on data. CTLE and the mixtures were run with the starts of Supplement Section~S1. In D5, TCLUST-REG at level $0.25$ with $5\\%$ second-level trimming in the covariates and reweighting ($300$ starts) reaches $0.86$ (Supplement, Table~S37). Values below $0.85$ in italics.",
          "tab:mainlow")
    table(C, False, "main_high.tex",
          "Worst mean accuracy on clean units with $25\\%$ to $35\\%$ of outliers. D1 to D8: second part, new data ($100$ data sets per cell); the other columns: third part at $\\varepsilon=0.30$ only ($50$ data sets per cell), so column entries are minima over different sets of $\\varepsilon$. Rows and conventions as in Table~\\ref{tab:mainlow}. Values below $0.85$ in italics.",
          "tab:mainhigh")
    levelchoice(C)


def figure(C):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [("Alg2", "ESF", "k", "-", "o"),
            ("adaptX", "ESF alone", "#7f7f7f", "--", "o"),
            ("TCRlong(0.25)+rw+A2", r"TCLUST-REG $\alpha=0.25$, rw, recovery", "#ff7f0e", "-.", "^"),
            ("TCRlong(0.40)+rw+A2", r"TCLUST-REG $\alpha=0.40$, rw, recovery", "#2ca02c", "-", "v"),
            ("TCRlong(0.40)+rw", r"TCLUST-REG $\alpha=0.40$, rw", "#2ca02c", ":", "v"),
            ("Laplace", "Laplace mixture", "#17becf", "--", "*"),
            ("CNmix", "Contaminated normal mixture", "#bcbd22", "-.", "p")]
    ds = [("D2-K3par", "D2: three parallel lines"), ("E4-K4par", "four parallel lines"),
          ("D6-uniform", "D6: uniform background noise"), ("G2-small15", "groups of 85% and 15%")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.6), sharey=True)
    for ax, (d, title) in zip(axes.ravel(), ds):
        for m, lab, col, ls, mk in show:
            s = C[(C.design == d) & (C.method == m)].sort_values("eps")
            ax.plot(s.eps, s.acc, color=col, ls=ls, marker=mk, ms=5, lw=1.5, label=lab)
        ax.set_title(title, fontsize=12); ax.set_xlabel(r"fraction of outliers $\varepsilon$", fontsize=11)
        ax.set_ylim(0.3, 1.0); ax.grid(alpha=0.3); ax.tick_params(labelsize=10)
    axes[0, 0].set_ylabel("accuracy on clean units", fontsize=11); axes[1, 0].set_ylabel("accuracy on clean units", fontsize=11)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=10.5, frameon=False)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(os.path.join(ROOT, "paper", "figures", "curves2.pdf"))


if __name__ == "__main__":
    figure(load().groupby(["study", "design", "eps", "method"]).acc.mean().reset_index())
