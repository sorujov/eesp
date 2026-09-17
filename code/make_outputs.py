"""Tables (LaTeX) and figures (PDF) for Sections 5-6, generated from the raw
experiment output in results_v3/.  Also writes results_v3/summary.md with the
numbers quoted in the text.

Usage: python3 make_outputs.py [e1] [e3] [e4] [e6]   (default: all available)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simcommon import OUT

TAB = os.path.join(OUT, "tables")
FIG = os.path.join(OUT, "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# validated categorical palette (fixed order) and marker shapes for print
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5,
                     "lines.linewidth": 1.6, "lines.markersize": 5, "legend.frameon": False,
                     "savefig.bbox": "tight"})
SUMMARY = []


def note(s):
    SUMMARY.append(s)
    print(s)


def load(pattern):
    fs = sorted(glob.glob(os.path.join(OUT, pattern)))
    if not fs:
        return None
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def mse(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return x.mean(), (x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan)


def fmt(v, d=3):
    return "--" if not np.isfinite(v) else f"{v:.{d}f}"


# ---------------------------------------------------------------------------
def e1():
    d = load("e1_rows_*.csv")
    c = load("e1_curve_*.csv")
    if d is None:
        return
    # permutation-invariant floor quantities (K = 2)
    d["floor"] = np.minimum(d.plural_ne_zstar, 1 - d.plural_ne_zstar)
    d["zhat_zstar"] = np.minimum(d.zhat_ne_zstar, 1 - d.zhat_ne_zstar)
    d["align_fail"] = d.mean_Q < 0.6
    g = d.groupby(["geom", "solver", "m"])
    note(f"E1: {len(d)} data sets; per cell {g.size().min()}-{g.size().max()}")
    note(f"E1 batch variance ratio: mean {d.batch_var_ratio.mean():.3f}, "
         f"range of cell means {g.batch_var_ratio.mean().min():.3f}-{g.batch_var_ratio.mean().max():.3f}")
    note(f"E1 alignment failures (mean Q<0.6): exact {d[d.solver=='exact'].align_fail.sum()}, "
         f"local {d[d.solver=='local'].align_fail.sum()} of {len(d)//2} each")
    rows = []
    for geom, gname in [("par", "Parallel lines"), ("cross", "Crossing lines")]:
        for m in sorted(d.m.unique()):
            ex = d[(d.geom == geom) & (d.m == m) & (d.solver == "exact")]
            lo = d[(d.geom == geom) & (d.m == m) & (d.solver == "local")]
            r = dict(geom=gname, m=m)
            r["bex"], r["bex_se"] = mse(100 * ex.frac_Q_below_half)
            r["blo"], r["blo_se"] = mse(100 * lo.frac_Q_below_half)
            r["agree"] = 100 * ex.vote_agree_zhat.mean()
            r["veq"] = 100 * ex.vote_equals_zhat.mean()
            r["vpeq"] = 100 * ex.votepol_equals_zhat.mean()
            r["pz"] = 100 * ex.p_zhat.mean()
            r["pb"] = 100 * ex.p_basin.mean()
            r["floor"] = 100 * ex.floor.mean()
            r["zz"] = 100 * ex.zhat_zstar.mean()
            r["tms"] = 1000 * ex.t_rep.mean()
            rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(TAB, "e1_table.csv"), index=False)
    lines = [r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{boundary set (\%)} & agreement & \multicolumn{2}{c}{vote $=\hat z$ (\%)}"
             r" & \multicolumn{2}{c}{one replicate (\%)} & floor & time \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
             r" & $m$ & exact & local & vote--$\hat z$ (\%) & vote & $+$pol. & $=\hat z$ & $\to\hat z$ & (\%) & (ms) \\",
             r"\midrule"]
    for gname in ["Parallel lines", "Crossing lines"]:
        sub = T[T.geom == gname]
        for j, r in enumerate(sub.itertuples()):
            lab = rf"\multirow{{{len(sub)}}}{{*}}{{{gname}}}" if j == 0 else ""
            lines.append(f"{lab} & {r.m} & {r.bex:.1f} ({r.bex_se:.1f}) & {r.blo:.1f} ({r.blo_se:.1f}) & "
                         f"{r.agree:.1f} & {r.veq:.0f} & {r.vpeq:.0f} & {r.pz:.1f} & {r.pb:.0f} & "
                         f"{r.floor:.1f} & {r.tms:.2f} \\\\")
        lines.append(r"\midrule" if gname == "Parallel lines" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    open(os.path.join(TAB, "e1_table.tex"), "w").write("\n".join(lines) + "\n")
    for r in T.itertuples():
        note(f"E1 {r.geom} m={r.m}: boundary exact {r.bex:.2f}+-{r.bex_se:.2f} local {r.blo:.2f}+-{r.blo_se:.2f}; "
             f"agree {r.agree:.2f}; vote=zhat {r.veq:.0f}%; vote+pol {r.vpeq:.0f}%; p_zhat {r.pz:.2f}%; "
             f"p_basin {r.pb:.1f}%; floor {r.floor:.2f}%; zhat!=z* {r.zz:.2f}%; t {r.tms:.2f}ms")
    for (geom, solver), s in d.groupby(["geom", "solver"]):
        note(f"E1 AUC {geom} {solver}: vote-share {s.auc_vote_share.mean():.3f}, plug-in {s.auc_plugin.mean():.3f}; "
             f"acc0 zhat {s.acc0_zhat.mean():.3f} bayes {s.acc0_bayes.mean():.3f}")
    ex = d[d.solver == "exact"]
    note(f"E1 AUC exact overall: vote {ex.auc_vote_share.mean():.3f} plug-in {ex.auc_plugin.mean():.3f}; "
         f"corr(Q, margin) {ex.corr_Q_margin.mean():.3f}")
    for (geom, m), s in ex.groupby(["geom", "m"]):
        note(f"E1 acc0 {geom} m={m}: vote {s.acc0_vote.mean():.3f} zhat {s.acc0_zhat.mean():.3f} bayes {s.acc0_bayes.mean():.3f}")

    # Figure: boundary set vs m
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, (geom, gname) in zip(axes, [("par", "Parallel lines"), ("cross", "Crossing lines")]):
        for j, (solver, lab) in enumerate([("exact", "exact micro-solve"), ("local", "one-start local solve")]):
            s = d[(d.geom == geom) & (d.solver == solver)].groupby("m").frac_Q_below_half
            mu, se = 100 * s.mean(), 100 * s.std() / np.sqrt(s.count())
            ax.errorbar(mu.index, mu.values, yerr=1.96 * se.values, color=PAL[j], marker=MARK[j],
                        capsize=2, label=lab)
        ax.set_title(gname)
        ax.set_xlabel("subsample size $m$")
        ax.set_xticks(sorted(d.m.unique()))
    axes[0].set_ylabel(r"boundary set $\{Q_i<1/2\}$ (% of units)")
    axes[1].legend(loc="upper right")
    fig.savefig(os.path.join(FIG, "fig_boundary_vs_m.pdf"))
    plt.close(fig)

    # Figure: B-curve
    if c is not None:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
        cc = c[(c.solver == "exact") & (c.gamma == 0.1) & (c.B <= 400)]
        for ax, (geom, gname) in zip(axes, [("par", "Parallel lines"), ("cross", "Crossing lines")]):
            for j, m in enumerate([8, 16]):
                s = cc[(cc.geom == geom) & (cc.m == m)].groupby("B")
                ax.plot(s.p_miss_emp.mean().index, s.p_miss_emp.mean().values, color=PAL[j],
                        marker=MARK[j], label=f"observed, $m={m}$")
                ax.plot(s.union_binom.mean().index, s.union_binom.mean().values, color=PAL[j],
                        ls="--", label=f"binomial union bound, $m={m}$")
                ax.plot(s.union_hoeff.mean().index, s.union_hoeff.mean().values, color=PAL[j],
                        ls=":", label=f"Hoeffding union bound, $m={m}$")
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_ylim(0, 1.05)
            ax.set_title(gname)
            ax.set_xlabel("ensemble size $B$")
        axes[0].set_ylabel(r"P(vote $\ne\hat z$ on $\mathfrak{M}_{0.1}$)")
        h, l = axes[1].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.18))
        fig.savefig(os.path.join(FIG, "fig_bcurve.pdf"))
        plt.close(fig)
        s = c[(c.solver == "exact") & (c.gamma == 0.1)]
        viol = (s.p_miss_emp > s.union_binom + 0.02).mean()
        note(f"E1 B-curve: fraction of (dataset,B) with observed > binomial bound + 0.02: {viol:.4f}")
        for (geom, m), t in s[s.B.isin([100, 200, 400])].groupby(["geom", "m"]):
            note(f"E1 Bcurve {geom} m={m}: " + "; ".join(
                f"B={B}: obs {u.p_miss_emp.mean():.3f} binom {u.union_binom.mean():.3f} hoeff {u.union_hoeff.mean():.3f} all {u.p_miss_all_emp.mean():.3f}"
                for B, u in t.groupby("B")))

    # Figure: Q profile for one data set
    prof = sorted(glob.glob(os.path.join(OUT, "e1_prof_*_cross_8_*.npz")))
    if prof:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
        for ax, solver in zip(axes, ["exact", "local"]):
            f = [p for p in prof if p.endswith(f"_{solver}.npz")
                 and f"cross_8_{solver}_0_Q" in np.load(p).files]
            if not f:
                continue
            z = np.load(f[0])
            key = f"cross_8_{solver}_0_"
            Q = z[key + "Q"]
            mg = z[key + "margin"]
            miss = z[key + "miss"]
            ax.scatter(np.log10(mg + 1e-3), Q, s=8, color=PAL[0], label=r"$\hat z_i=z^0_i$")
            ax.scatter(np.log10(mg[miss] + 1e-3), Q[miss], s=14, color=PAL[1], marker="x",
                       label=r"$\hat z_i\ne z^0_i$")
            ax.axhline(0.5, color="#666666", lw=0.8, ls="--")
            ax.set_title(("exact" if solver == "exact" else "one-start local") + " micro-solve, $m=8$")
            ax.set_xlabel(r"$\log_{10}$ plug-in margin of $\hat z$")
        axes[0].set_ylabel(r"$Q_i$")
        axes[0].legend(loc="upper left")
        fig.savefig(os.path.join(FIG, "fig_qprofile.pdf"))
        plt.close(fig)


# ---------------------------------------------------------------------------
E3_METHODS = [("vote", "vote"), ("vote+pol", r"\shortstack{vote\\+pol}"), ("param+pol", r"\shortstack{mean\\+pol}"),
              ("bestB+pol", r"\shortstack{select\\+pol}"), ("local-vote+pol", r"\shortstack{local\\vote+pol}"),
              ("ms-vote+pol", r"\shortstack{ms\\vote+pol}"), ("HA10", "HA-10"), ("HA-tm", "HA-time"),
              ("emEM10", "emEM")]
E3_LABELS = {
    "K2-par-m8": (r"2, par., $m=8$", 2), "K2-par-m16": (r"2, par., $m=16$", 2),
    "K2-par-d2": (r"2, par., $\Delta=2$", 2), "K2-par-d4": (r"2, par., $\Delta=4$", 2),
    "K2-cross-m8": (r"2, cross, $m=8$", 2), "K2-cross-m16": (r"2, cross, $m=16$", 2),
    "K2-cross-d6": (r"2, cross, $\Delta=6$", 2), "K2-cross-d12": (r"2, cross, $\Delta=12$", 2),
    "K2-cross-imb": (r"2, cross, $\pi=(.8,.2)$", 2), "K2-cross-t3": (r"2, cross, $t_3$", 2),
    "K2-cross-n100": (r"2, cross, $n=100$", 2), "K2-cross-n500": (r"2, cross, $n=500$", 2),
    "K2-cross-n1000": (r"2, cross, $n=1000$", 2), "K2-par-p2": (r"2, par., $p=2$", 2),
    "K3-par": (r"3, par.", 3), "K3-cross": (r"3, cross", 3),
    "K4-par": (r"4, par.", 4), "K4-cross": (r"4, cross", 4)}


def e3():
    d = load("e3_rows_*.csv")
    if d is None:
        return
    order = [k for k in E3_LABELS if k in set(d.cell)]
    note(f"E3: {d.groupby('cell').rep.nunique().to_dict()}")
    lines = [r"\begin{tabular}{l" + "r" * len(E3_METHODS) + "}", r"\toprule",
             "$K$, design & " + " & ".join(lab for _, lab in E3_METHODS) + r" \\", r"\midrule"]
    for cell in order:
        s = d[d.cell == cell]
        vals = []
        for meth, _ in E3_METHODS:
            vals.append(100 * s[s.method == meth].hit.mean())
        best = max(vals)
        cells = [(rf"\textbf{{{v:.0f}}}" if v >= best - 0.5 else f"{v:.0f}") for v in vals]
        flag = "" if s.ref_exact.iloc[0] else r"$^{\dagger}$"
        lines.append(E3_LABELS[cell][0] + flag + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    tm = [1000 * d[d.method == meth].time.median() for meth, _ in E3_METHODS]
    lines.append(r"median time (ms) & " + " & ".join(f"{t:.0f}" if t >= 1 else f"{t:.1f}" for t in tm) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "e3_hit.tex"), "w").write("\n".join(lines) + "\n")
    # gap table (mean relative gap x 1000) for the supplement
    lines = [r"\begin{tabular}{l" + "r" * len(E3_METHODS) + "}", r"\toprule",
             "$K$, design & " + " & ".join(lab for _, lab in E3_METHODS) + r" \\", r"\midrule"]
    for cell in order:
        s = d[d.cell == cell]
        vals = [abs(v) if abs(v) < 0.05 else v for v in (1000 * s[s.method == meth].gap.mean() for meth, _ in E3_METHODS)]
        lines.append(E3_LABELS[cell][0] + " & " + " & ".join(f"{v:.1f}" for v in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "e3_gap.tex"), "w").write("\n".join(lines) + "\n")
    lines = [r"\begin{tabular}{l" + "r" * len(E3_METHODS) + "}", r"\toprule",
             "$K$, design & " + " & ".join(lab for _, lab in E3_METHODS) + r" \\", r"\midrule"]
    for cell in order:
        s = d[d.cell == cell]
        vals = [s[s.method == meth].acc0.mean() for meth, _ in E3_METHODS]
        lines.append(E3_LABELS[cell][0] + " & " + " & ".join(f"{v:.3f}" for v in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "e3_acc0.tex"), "w").write("\n".join(lines) + "\n")
    for cell in order:
        s = d[d.cell == cell]
        parts = []
        for meth, _ in E3_METHODS:
            u = s[s.method == meth]
            h, hse = mse(u.hit.astype(float))
            parts.append(f"{meth} hit {100*h:.1f}({100*hse:.1f}) gap {1000*u.gap.mean():.2f} acc0 {u.acc0.mean():.3f} t {1000*u.time.median():.1f}ms")
        note(f"E3 {cell} (n reps {s.rep.nunique()}, R_tm median {s.R_tm.median():.0f}): " + "; ".join(parts))


# ---------------------------------------------------------------------------
E4_METHODS = [("HA50", "HA"), ("emEM10", "EM"), ("vote", "vote"), ("median", "med."),
              ("adapt", r"Ad$_{3}$"), ("adapt-sel", r"Ad$_{\infty}$"),
              ("adapt-lam0", r"Ad$_{0}$"), ("adapt-1st", r"Ad$_{3}^{(1)}$"),
              ("TA-rw(oracle)", r"TA$_{\varepsilon}$"), ("TA-rw(0.25)", r"TA$_{.25}$"),
              ("TA-rw(0.40)", r"TA$_{.40}$"), ("bestB-trim(0.25)", r"ST$_{.25}$")]
E4_MAIN = [m for m in E4_METHODS if m[0] not in ("adapt-lam0", "adapt-1st")]
E4_CELLS = [("K2-cross", r"$K{=}2$, crossing"), ("K3-par", r"$K{=}3$, parallel"),
            ("K2-cross-unequal", r"$K{=}2$, $\pi{=}(.7,.3)$"),
            ("K2-cross-leverage", r"$K{=}2$, leverage"),
            ("K2-cross-hetero", r"$K{=}2$, $\sigma{=}(.5,1.5)$")]


def e4():
    d = load("e4_rows_*.csv")
    if d is None:
        return
    a = d[d.part == "a"]
    e4_table(a, E4_METHODS, "e4_acc_full.tex", worst=False)
    e4_table(a, E4_MAIN, "e4_acc.tex", worst=True)
    _e4_rest(a, d)


def e4_table(a, METHODS, fname, worst):
    lines = [r"\begin{tabular}{ll" + "r" * len(METHODS) + "}", r"\toprule",
             r" & $\varepsilon$ & " + " & ".join(lab for _, lab in METHODS) + r" \\", r"\midrule"]
    for cell, cname in E4_CELLS:
        s = a[a.cell == cell]
        if s.empty:
            continue
        epss = sorted(s.eps.unique())
        for j, eps in enumerate(epss):
            u = s[s.eps == eps]
            vals = [u[u.method == meth].acc_clean.mean() for meth, _ in METHODS]
            best = np.nanmax(vals)
            cells = [(rf"\textbf{{{v:.3f}}}" if v >= best - 0.005 else f"{v:.3f}") if np.isfinite(v) else "--"
                     for v in vals]
            lab = cname if j == 0 else ""
            lines.append(f"{lab} & {eps:.2f} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    if worst:
        vert = a[a.cell.isin(["K2-cross", "K3-par", "K2-cross-unequal", "K2-cross-hetero"])]
        g = vert.groupby(["cell", "eps", "method"]).acc_clean.mean()
        for lab, emax in [(r"\multicolumn{2}{l}{worst, $\varepsilon\le0.2$}", 0.2),
                          (r"\multicolumn{2}{l}{worst, $\varepsilon\le0.3$}", 0.3)]:
            gg = g[g.index.get_level_values(1) <= emax + 1e-9]
            vals = [gg.xs(meth, level=2).min() for meth, _ in METHODS]
            best = np.nanmax(vals)
            cells = [rf"\textbf{{{v:.3f}}}" if v >= best - 0.005 else f"{v:.3f}" for v in vals]
            lines.append(lab + " & " + " & ".join(cells) + r" \\")
            note(f"E4 worst over vertical designs eps<={emax}: " + ", ".join(f"{m} {v:.3f}" for (m, _), v in zip(METHODS, vals)))
        lines.append(r"\bottomrule")
    else:
        lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    open(os.path.join(TAB, fname), "w").write("\n".join(lines) + "\n")


def _e4_rest(a, d):
    # supplement: estimated trimming level and median parameter error
    AM = [m for m in E4_METHODS if m[0].startswith(("adapt", "TA-rw"))]
    lines = [r"\begin{tabular}{ll" + "r" * len(AM) + "}", r"\toprule",
             r" & $\varepsilon$ & " + " & ".join(lab for _, lab in AM) + r" \\", r"\midrule"]
    PM = E4_METHODS
    lines2 = [r"\begin{tabular}{ll" + "r" * len(PM) + "}", r"\toprule",
              r" & $\varepsilon$ & " + " & ".join(lab for _, lab in PM) + r" \\", r"\midrule"]
    for cell, cname in E4_CELLS:
        s = a[a.cell == cell]
        for j, eps in enumerate(sorted(s.eps.unique())):
            u = s[s.eps == eps]
            lab = cname if j == 0 else ""
            lines.append(f"{lab} & {eps:.2f} & " + " & ".join(
                f"{u[u.method == m].alpha_hat.mean():.3f}" for m, _ in AM) + r" \\")
            lines2.append(f"{lab} & {eps:.2f} & " + " & ".join(
                (f"{v:.2f}" if v < 10 else f"{v:.1f}") for v in (u[u.method == m].perr.median() for m, _ in PM)) + r" \\")
        lines.append(r"\midrule")
        lines2.append(r"\midrule")
    for L, f in [(lines, "e4_alpha.tex"), (lines2, "e4_perr.tex")]:
        L[-1] = r"\bottomrule"
        L.append(r"\end{tabular}")
        open(os.path.join(TAB, f), "w").write("\n".join(L) + "\n")
    for cell, cname in E4_CELLS:
        s = a[a.cell == cell]
        for eps, u in s.groupby("eps"):
            parts = []
            for meth, _ in E4_METHODS:
                v = u[u.method == meth]
                m_, se = mse(v.acc_clean)
                fails = (v.acc_clean < 0.75).mean()
                parts.append(f"{meth} {m_:.3f}({se:.3f}) fail {100*fails:.0f}% perr {np.nanmedian(v.perr):.3f}"
                             + (f" ahat {np.nanmean(v.alpha_hat):.3f}" if v.alpha_hat.notna().any() else "")
                             + f" t {1000*np.nanmedian(v.time):.0f}ms")
            note(f"E4 {cell} eps={eps} (reps {u.rep.nunique()}, n_out mean {u.n_out.mean():.1f}): " + "; ".join(parts))
    # figure: accuracy vs eps
    sel = [("HA50", 0), ("vote", 1), ("median", 2), ("adapt", 3), ("TA-rw(oracle)", 4),
           ("TA-rw(0.25)", 5), ("TA-rw(0.40)", 6)]
    labs = dict(E4_METHODS)
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.6), sharey=True)
    for ax, (cell, cname) in zip(axes, E4_CELLS):
        s = a[a.cell == cell]
        for meth, j in sel:
            u = s[s.method == meth].groupby("eps").acc_clean.mean()
            ax.plot(u.index, u.values, color=PAL[j], marker=MARK[j], label=labs[meth])
        ax.set_title(cname, fontsize=8)
        ax.set_xlabel(r"contamination $\varepsilon$")
    axes[0].set_ylabel("accuracy on clean units")
    axes[-1].legend(loc="lower left", fontsize=6.5)
    fig.savefig(os.path.join(FIG, "fig_contam.pdf"))
    plt.close(fig)
    # window figure
    b = d[d.part == "b"]
    if not b.empty:
        fig, ax = plt.subplots(figsize=(3.6, 2.6))
        for j, eps in enumerate(sorted(b.eps.unique())):
            u = b[(b.eps == eps) & (b.method == "vote")].groupby("m").acc_clean
            ax.errorbar(u.mean().index, u.mean().values, yerr=1.96 * u.std() / np.sqrt(u.count()),
                        color=PAL[j], marker=MARK[j], capsize=2, label=rf"vote, $\varepsilon={eps}$")
            u = b[(b.eps == eps) & (b.method == "median")].groupby("m").acc_clean.mean()
            ax.plot(u.index, u.values, color=PAL[j], ls="--", label=rf"median, $\varepsilon={eps}$")
        ax.set_xlabel("subsample size $m$")
        ax.set_ylabel("accuracy on clean units")
        ax.legend(fontsize=7)
        fig.savefig(os.path.join(FIG, "fig_window.pdf"))
        plt.close(fig)
        for (eps, m), u in b[b.method == "vote"].groupby(["eps", "m"]):
            note(f"E4b eps={eps} m={m}: vote {u.acc_clean.mean():.3f} median "
                 f"{b[(b.eps==eps)&(b.m==m)&(b.method=='median')].acc_clean.mean():.3f} p_clean {(1-eps)**m:.3f}")


# ---------------------------------------------------------------------------
E6_METHODS = [("vote", "vote"), ("vote+pol", r"\shortstack{vote\\+pol}"), ("param+pol", r"\shortstack{mean\\+pol}"),
              ("bestB+pol", r"\shortstack{select\\+pol}"), ("local-vote+pol", r"\shortstack{local\\vote+pol}"),
              ("ms-vote+pol", r"\shortstack{ms\\vote+pol}"), ("HA10", "HA-10"), ("HA-tm", "HA-time"),
              ("emEM10", "emEM")]
E6_DATA = [("tone", "Tone", 150), ("ethanol", "NO", 88),
           ("co2", r"CO$_2$", 28), ("gnp-ar1", "GNP", 316)]


def e6():
    d = load("e6_rows.csv")
    if d is None:
        return
    lines = [r"\begin{tabular}{llr" + "r" * len(E6_METHODS) + "}", r"\toprule",
             r"data & $n$ & $m$ & " + " & ".join(lab for _, lab in E6_METHODS) + r" \\", r"\midrule"]
    for key, name, n in E6_DATA:
        s = d[d.data == key]
        for j, m in enumerate(sorted(s.m.unique())):
            u = s[s.m == m]
            vals = [100 * u[u.method == meth].hit.mean() for meth, _ in E6_METHODS]
            lab = (rf"\multirow{{2}}{{*}}{{{name}}} & \multirow{{2}}{{*}}{{{n}}}" if j == 0 else " & ")
            lines.append(f"{lab} & {m} & " + " & ".join(f"{v:.0f}" for v in vals) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    open(os.path.join(TAB, "e6_hit.tex"), "w").write("\n".join(lines) + "\n")
    for key, name, n in E6_DATA:
        s = d[d.data == key]
        note(f"E6 {key}: Fhat {s.Fhat.iloc[0]:.6g}")
        for m, u in s.groupby("m"):
            note(f"E6 {key} m={m}: " + "; ".join(
                f"{meth} hit {100*u[u.method==meth].hit.mean():.0f}% agree {u[u.method==meth].agree_zhat.mean():.3f} "
                f"minF/Fhat-1 {u[u.method==meth].gap.min():.2e} t {1000*u[u.method==meth].time.median():.0f}ms"
                for meth, _ in E6_METHODS) + f"; R_tm {u.R_tm.median():.0f}")
    q = load("e6_q.csv")
    if q is not None:
        for (key, m), u in q.groupby(["data", "m"]):
            note(f"E6 Q {key} m={m}: boundary {100*u.frac_Q_below_half.mean():.1f}% (units {u.n_Q_below_half.mean():.1f}), min Q {u.min_Q.mean():.3f}")
    c = load("e6_contam.csv")
    if c is not None:
        g = c.groupby("method")
        note("E6 contaminated tone: " + "; ".join(
            f"{k} own {v.agree_own_clean.mean():.3f} (runs<0.95: {100*(v.agree_own_clean<0.95).mean():.0f}%) "
            f"zhat {v.agree_zhat_clean.mean():.3f} clean-vs-zhat {v.ref_agree_zhat.iloc[0]:.3f}" for k, v in g))
        order = ["HA50", "emEM10", "vote", "median", "adapt", "TA-rw(0.25)", "TA50(a=0.05)",
                 "TA50(a=0.10)", "TA50(a=0.25)", "bestB-trim(a=0.10)"]
        names = {"HA50": "HA-50", "emEM10": "emEM", "vote": "vote", "median": "median",
                 "adapt": "adaptive", "TA-rw(0.25)": r"TA ($.25$) + reweighting",
                 "TA50(a=0.05)": r"TA ($\alpha=.05$)", "TA50(a=0.10)": r"TA ($\alpha=.10$)",
                 "TA50(a=0.25)": r"TA ($\alpha=.25$)", "bestB-trim(a=0.10)": r"select-trim ($\alpha=.10$)"}
        lines = [r"\begin{tabular}{lrrr}", r"\toprule",
                 r"method & clean fit vs.\ $\hat z$ & contaminated vs.\ own clean fit & runs below $0.95$ (\%) \\",
                 r"\midrule"]
        for k in order:
            v = c[c.method == k]
            if v.empty:
                continue
            lines.append(f"{names[k]} & {v.ref_agree_zhat.iloc[0]:.3f} & {v.agree_own_clean.mean():.3f} & "
                         f"{100*(v.agree_own_clean<0.95).mean():.0f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(TAB, "e6_contam.tex"), "w").write("\n".join(lines) + "\n")
    p = os.path.join(OUT, "e6_gnp_zhat.txt")
    if os.path.exists(p):
        note("E6 GNP zhat: " + open(p).read().strip())


if __name__ == "__main__":
    which = sys.argv[1:] or ["e1", "e3", "e4", "e6"]
    for w in which:
        globals()[w]()
    with open(os.path.join(OUT, "summary_" + "_".join(which) + ".md"), "w") as f:
        f.write("\n".join(SUMMARY) + "\n")
