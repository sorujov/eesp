"""Tables and figures of the v4 paper from results_v4/metrics.csv (Gate 1) and
results_v4b/metrics.csv (Gate 1b).  Writes LaTeX tables to paper/tables/ and figures to
paper/figures/.  Usage: python make_v4_outputs.py ROOT"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "paper", "tables")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

DESIGNS = ["D1-K2cross", "D2-K3par", "D3-unequal", "D4-hetero", "D6-uniform", "D7-p3",
           "D8-K2par-sep", "D5-leverage"]
SHORT = {"D1-K2cross": "D1", "D2-K3par": "D2", "D3-unequal": "D3", "D4-hetero": "D4",
         "D5-leverage": "D5", "D6-uniform": "D6", "D7-p3": "D7", "D8-K2par-sep": "D8"}
LEVELS = ["0.05", "0.10", "0.15", "0.25", "0.40"]
ROWS = (["Alg2", "adaptX", "adapt"] +
        [f"TCR({a})" for a in LEVELS] + ["TCR(0.25)+rw", "TCR(0.40)+rw", "TCR(0.40)+rw+A2", "TA(0.40)+A2", "TCRx(0.10)+rw", "TCRx(0.25)+rw"] +
        [f"tcwm({a})" for a in LEVELS] + [f"TLE({a})" for a in LEVELS] + ["CTLE", "bisq", "Laplace", "CNmix", "NoiseMix", "SeqRANSAC", "HA50"])
LABEL = {"Alg2": "Proposed procedure (Algorithms 1 and 2)", "Alg2b5": "Procedure, best of five runs", "TCR(0.40)+rw+A2": "TCLUST-REG, $\\alpha=0.40$, reweighted, then Algorithm 2", "TA(0.40)+A2": "Trimmed alternation, $\\alpha=0.40$, reweighted, then Algorithm 2", "adaptX": "Algorithm 1 alone (ablation)", "adapt": "Earlier version, no covariate screen (ablation)",
         "CTLE": "CTLE", "bisq": "Bisquare mixture", "NoiseMix": "Noise-component mixture", "SeqRANSAC": "Sequential RANSAC (true noise scale given)", "Laplace": "Laplace mixture", "CNmix": "Contaminated normal mixture", "HA50": "Least squares (50 starts)"}


def label(m):
    if m in LABEL:
        return LABEL[m]
    fam, rest = m.split("(")
    a = rest.split(")")[0]
    rw = "+rw" in m
    name = {"TCR": "TCLUST-REG", "TCRlong": "TCLUST-REG (long search)", "TCRx": "TCLUST-REG$^{\\ast}$", "tcwm": "Trimmed CWM",
            "TLE": "TLE", "TA": "Trimmed alternation (supplement)"}[fam]
    return f"{name}, $\\alpha={a}$" + (", reweighted" if rw else "") + (", then Alg.~2" if "+A2" in m else "")


def load(keep_gate1_030=False):
    """Gate 1 (eps <= 0.30) and Gate 1b (eps 0.25-0.35, fresh data).  The Gate 1 cells at
    eps = 0.30 were outside its pre-registered criteria and are superseded by Gate 1b; they
    are dropped from the main tables and kept in the supplement."""
    M = pd.read_csv(os.path.join(ROOT, "results_v4", "metrics.csv"))
    M["study"] = "1"
    fb = os.path.join(ROOT, "results_v4b", "metrics.csv")
    if os.path.exists(fb):
        B = pd.read_csv(fb)
        B["study"] = "1b"
        M = pd.concat([M, B], ignore_index=True)
    for d, st in (("results_v4", "1"), ("results_v4b", "1b")):
        for extra in ("noisemix.csv", "laplace.csv", "cnmix.csv", "alg2.csv", "seqransac.csv"):
            fn = os.path.join(ROOT, d, extra)
            if os.path.exists(fn):
                N = pd.read_csv(fn)
                N["study"] = st
                M = pd.concat([M, N], ignore_index=True)
        fn = os.path.join(ROOT, d, "rec_after_tcr.csv")
        if os.path.exists(fn):
            N = pd.read_csv(fn)
            N = N[N.method == "TCR(0.40)+rw"].assign(method="TCR(0.40)+rw+A2", acc=lambda q: q.acc_rec, study=st)
            N["unit"] = N.design + "_e" + (N.eps * 100).round().astype(int).astype(str).str.zfill(2) + "_r" + N.rep.astype(str).str.zfill(3)
            M = pd.concat([M, N.drop(columns=["acc_rec"])], ignore_index=True)
        fn = os.path.join(ROOT, d, "ta_rec.csv")
        if os.path.exists(fn):
            N = pd.read_csv(fn)
            N = N[["design", "eps", "rep", "unit"]].assign(method="TA(0.40)+A2", acc=N.acc_rec, study=st)
            M = pd.concat([M, N], ignore_index=True)
        fn = os.path.join(ROOT, d, "best5.csv")
        if os.path.exists(fn):
            N = pd.read_csv(fn)
            N = N[["design", "eps", "rep", "unit"]].assign(method="Alg2b5", acc=N.acc_best5, study=st)
            M = pd.concat([M, N], ignore_index=True)
    if not keep_gate1_030:
        M = M[~((M.study == "1") & (M.eps.round(2) == 0.30))]
    M = M.copy()
    M.loc[M.acc.isna(), "acc"] = 0.0          # a failed fit counts as accuracy 0
    return M


def worst_table(M, eps_set, fname, caption, lab):
    g = M[M.eps.round(2).isin(eps_set)]
    cell = g.groupby(["method", "design", "eps"]).acc.mean().reset_index()
    worst = cell.groupby(["method", "design"]).acc.min().unstack()
    best_by_design = cell.groupby(["design", "eps"]).acc.max().groupby("design").min()
    cols = [d for d in DESIGNS if d in worst.columns]
    lines = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize",
             f"\\caption{{{caption}}}", f"\\label{{{lab}}}",
             "\\resizebox{\\textwidth}{!}{\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             "Method & " + " & ".join(SHORT[c] for c in cols) + " \\\\", "\\midrule"]
    for m in ROWS:
        if m not in worst.index:
            continue
        vals = []
        for c in cols:
            v = worst.loc[m, c]
            s = "--" if not np.isfinite(v) else f"{v:.2f}"
            if np.isfinite(v) and v < 0.85:
                s = f"\\textit{{{s}}}"
            vals.append(s)
        lines.append(label(m) + " & " + " & ".join(vals) + " \\\\")
        if m in ("adapt", "TCRx(0.25)+rw", "TLE(0.40)", "tcwm(0.40)", "NoiseMix"):
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    open(os.path.join(TAB, fname), "w").write("\n".join(lines))
    return worst


def ahat_table(M):
    g = M[M.method == "Alg2"]
    t = g.groupby(["design", "eps"]).ahat.mean().unstack()
    eps = sorted(t.columns)
    lines = ["\\begin{tabular}{l" + "c" * len(eps) + "}", "\\toprule",
             "Design & " + " & ".join(f"{e:.2f}" for e in eps) + " \\\\", "\\midrule"]
    for d in DESIGNS:
        if d in t.index:
            lines.append(SHORT[d] + " & " + " & ".join(
                "--" if not np.isfinite(t.loc[d, e]) else f"{t.loc[d, e]:.3f}" for e in eps) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "ahat.tex"), "w").write("\n".join(lines))
    return t


def time_table(M):
    fams = {"Proposed procedure": ["Alg2"], "Algorithm 1 alone": ["adaptX"], "TCLUST-REG": [f"TCR({a})" for a in LEVELS],
            "TCLUST-REG, second trimming": ["TCRx(0.10)", "TCRx(0.25)"],
            "Trimmed CWM": [f"tcwm({a})" for a in LEVELS], "TLE": [f"TLE({a})" for a in LEVELS],
            "CTLE": ["CTLE"], "Bisquare mixture": ["bisq"], "Laplace mixture": ["Laplace"], "Contaminated normal mixture": ["CNmix"],
            "Noise-component mixture": ["NoiseMix"]}
    lines = ["\\begin{tabular}{lcc}", "\\toprule", "Method & Median & 90th percentile \\\\",
             "\\midrule"]
    out = {}
    for f, ms in fams.items():
        v = M[M.method.isin(ms)].time
        if len(v):
            out[f] = (v.median(), v.quantile(0.9))
            lines.append(f"{f} & {v.median():.2f} & {v.quantile(0.9):.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "time.tex"), "w").write("\n".join(lines))
    return out


def figure_curves(M):
    show = [("Alg2", "Proposed procedure", "k", "-", "o"),
            ("adaptX", "Algorithm 1 alone", "#7f7f7f", "--", "o"),
            ("TCR(0.10)", r"TCLUST-REG $\alpha=0.10$", "#1f77b4", "--", "s"),
            ("TCR(0.25)+rw", r"TCLUST-REG $\alpha=0.25$, reweighted", "#ff7f0e", "-.", "^"),
            ("TCR(0.40)+rw", r"TCLUST-REG $\alpha=0.40$, reweighted", "#2ca02c", ":", "v"),
            ("CTLE", "CTLE", "#9467bd", "--", "d"),
            ("bisq", "Bisquare mixture", "#8c564b", ":", "x"),
            ("Laplace", "Laplace mixture", "#17becf", "--", "*"),
            ("CNmix", "Contaminated normal mixture", "#bcbd22", "-.", "p"),
            ("NoiseMix", "Noise-component mixture", "#e377c2", "-", "+")]
    ds = ["D1-K2cross", "D2-K3par", "D3-unequal", "D6-uniform"]
    titles = {"D1-K2cross": "D1: two crossing lines", "D2-K3par": "D2: three parallel lines",
              "D3-unequal": "D3: groups of 70% and 30%", "D6-uniform": "D6: uniform background noise"}
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.6), sharey=True)
    axes = axes.ravel()
    cell = M.groupby(["design", "method", "eps"]).acc.mean()
    for ax, d in zip(axes, ds):
        for m, lab, col, ls, mk in show:
            try:
                s = cell.loc[(d, m)].sort_index()
            except KeyError:
                continue
            ax.plot(s.index, s.values, color=col, ls=ls, marker=mk, ms=5, lw=1.5, label=lab)
        ax.set_title(titles[d], fontsize=12)
        ax.set_xlabel(r"fraction of outliers $\varepsilon$", fontsize=11)
        ax.set_ylim(0.3, 1.0)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy on clean units", fontsize=11)
    axes[2].set_ylabel("accuracy on clean units", fontsize=11)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=11, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(os.path.join(FIG, "curves.pdf"))


def supplement_tables(M):
    """Full per-cell tables: mean acc (SE) for every method, one table per design."""
    out = []
    for d in DESIGNS:
        g = M[M.design == d].copy()
        if g.empty:
            continue
        g["eps"] = np.where((g.study == "1") & (g.eps.round(2) == 0.30), 0.301, g.eps)
        mean = g.groupby(["method", "eps"]).acc.mean().unstack()
        se = g.groupby(["method", "eps"]).acc.agg(lambda v: v.std(ddof=1) / np.sqrt(len(v))).unstack()
        eps = sorted(mean.columns)
        out += ["\\begin{table}[htbp]", "\\centering", "\\scriptsize",
                f"\\caption{{Design {SHORT[d]}: mean accuracy on clean units (Monte Carlo standard error).}}",
                "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{l" + "c" * len(eps) + "}", "\\toprule",
                "Method & " + " & ".join(("0.30 (part 1)" if e == 0.301 else f"{e:.2f}") for e in eps) + " \\\\", "\\midrule"]
        order = [m for m in ROWS if m in mean.index] + sorted(
            m for m in mean.index if m not in ROWS)
        for m in order:
            out.append(label(m) + " & " + " & ".join(
                "--" if not np.isfinite(mean.loc[m, e]) else f"{mean.loc[m, e]:.3f} ({se.loc[m, e]:.3f})"
                for e in eps) + " \\\\")
        out += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    open(os.path.join(TAB, "supp_full.tex"), "w").write("\n".join(out))


if __name__ == "__main__":
    M = load()
    worst_table(M, [0.0, 0.05, 0.1, 0.2], "worst_low.tex",
                "Pre-registered competitor settings (TCLUST-REG with $300$ random starts and $10$ refinement steps): worst mean accuracy on clean units (the lowest of the cell means) over $\\varepsilon\\in\\{0,0.05,0.10,0.20\\}$, "
                "by design, first part of the study. Values below $0.85$ in italics. Columns: the seven designs without high-leverage points, then D5. $^{\\ast}$With $5\\%$ second trimming in the covariates (run in D5 and D6 only). The Monte Carlo standard error of a cell mean is about $0.003$ near the ceiling and up to $0.05$ where a method fails (Tables~S4--S11).", "tab:worstlow")
    worst_table(M, [0.25, 0.3, 0.35], "worst_high.tex",
                "Pre-registered competitor settings: worst mean accuracy on clean units (the lowest of the cell means) over $\\varepsilon\\in\\{0.25,0.30,0.35\\}$ "
                "(second part of the study, new data), by design. Values below $0.85$ in italics. Columns: the seven designs without high-leverage points, then D5. $^{\\ast}$With $5\\%$ second trimming in the covariates (run in D5 and D6 only). Standard errors as in Table~S37.", "tab:worsthigh")
    print(ahat_table(M).round(3))
    print(time_table(M))
    figure_curves(M)
    supplement_tables(load(keep_gate1_030=True))
    print("done")
