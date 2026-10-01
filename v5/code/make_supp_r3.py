"""Supplement tables added in review round 3: coefficient error (S19), flagged fraction of the
methods that estimate the contamination (S20), mixtures with 100 starts (S21), cost of the
branch and bound (S22), sensitivity of Algorithm 2 (S23), best-of-five rule (S24)."""
import os
import numpy as np, pandas as pd
import make_v4_outputs as O

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TAB = os.path.join(ROOT, "paper", "tables")
SH = {**O.SHORT, "E1-K2cross-n100": "D1, $n=100$", "E2-K2cross-n1000": "D1, $n=1000$",
      "E3-unequal-n1000": "D3, $n=1000$", "E4-K4par": "four groups", "E5-p3corr": "D7, dependent",
      "F1-K2cross-moderate": "D1, moderate", "F2-unequal-moderate": "D3, moderate",
      "G1-small20": "groups 80/20", "G2-small15": "groups 85/15", "G3-cluster": "clustered",
      "H1-K2cross-t3": "D1, $t_3$", "H2-unequal-t3": "D3, $t_3$"}
DIRS = [("results_v4", "1"), ("results_v4b", "1b"), ("results_v4c", "1c"), ("results_v4d", "1d"),
        ("results_v4e", "1e"), ("results_v4f", "1f")]


def table(name, caption, label, header, rows, colspec, wide=False):
    L = ["\\begin{table}[htbp]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         ("\\resizebox{\\textwidth}{!}{" if wide else "") + f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"] + \
        [r + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}" + ("}" if wide else ""), "\\end{table}"]
    open(os.path.join(TAB, name), "w").write("\n".join(L))


M = O.load()
# S19: median coefficient error
meth = ["Alg2", "TCR(0.25)+rw", "TCR(0.40)+rw", "tcwm(0.25)", "CTLE", "bisq", "Laplace", "CNmix", "NoiseMix"]
lab = {"Alg2": "Procedure", "TCR(0.25)+rw": "TCLUST-REG 0.25 rw", "TCR(0.40)+rw": "TCLUST-REG 0.40 rw",
       "tcwm(0.25)": "Trimmed CWM 0.25", "CTLE": "CTLE", "bisq": "Bisquare", "Laplace": "Laplace",
       "CNmix": "Contam.\\ normal", "NoiseMix": "Noise comp."}
g = M[M.method.isin(meth) & M.eps.round(2).isin([0.0, 0.1, 0.2, 0.3])]
P = g.groupby(["design", "eps", "method"]).perr.median().unstack("method")
rows = []
for des in O.DESIGNS:
    for e in [0.0, 0.1, 0.2, 0.3]:
        if (des, e) not in P.index:
            continue
        r = P.loc[(des, e)]
        rows.append(f"{SH[des]} & {e:.2f} & " + " & ".join(
            ("{:.2f}".format(r[m]) if r[m] < 10 else "{:.0f}".format(r[m])) if pd.notna(r[m]) else "--" for m in meth))
table("supp_S19.tex", "Median coefficient error $\\min_{\\text{relabelling}}\\max_{k,j}|\\hat\\theta_{kj}-\\theta^{0}_{kj}|$ over the data sets of each cell (parts one and two; $\\varepsilon=0.30$ from the second part).",
      "tab:S19", "Design & $\\varepsilon$ & " + " & ".join(lab[m] for m in meth), rows, "ll" + "r" * len(meth), wide=True)

# S20: flagged fraction of the methods that estimate the contamination
rows = []
A = []
for d, st in DIRS:
    for f in ["alg2.csv", "cnmix.csv", "noisemix.csv"]:
        fn = os.path.join(ROOT, d, f)
        if os.path.exists(fn):
            x = pd.read_csv(fn); x["study"] = st; A.append(x)
A = pd.concat(A)
A = A[~((A.study == "1") & (A.eps.round(2) == 0.30))]
Q = A.groupby(["design", "eps", "method"]).ahat.mean().unstack("method")
for (des, e), r in Q.iterrows():
    rows.append(f"{SH.get(des, des)} & {e:.2f} & {r.get('Alg2', np.nan):.3f} & {r.get('CNmix', np.nan):.3f} & {r.get('NoiseMix', np.nan):.3f}")
table("supp_S20.tex", "Mean flagged fraction of the three methods that estimate the contamination: the procedure, the contaminated normal mixture (posterior probability of a bad point above one half) and the noise-component mixture (posterior probability of the noise component above one half).",
      "tab:S20", "Design & $\\varepsilon$ & Procedure & Contaminated normal & Noise component", rows, "llccc")

# S21: mixtures with 100 starts
rows = []
for d, st in DIRS[:3]:
    parts = {}
    for f, m in [("noisemix.csv", "NM10"), ("noisemix100.csv", "NM100"), ("cnmix.csv", "CN50"), ("cnmix100.csv", "CN100"),
                 ("laplace.csv", "LP20"), ("laplace100.csv", "LP100")]:
        fn = os.path.join(ROOT, d, f)
        if os.path.exists(fn):
            x = pd.read_csv(fn); x.loc[x.acc.isna(), "acc"] = 0.0
            parts[m] = x.groupby(["design", "eps"]).acc.mean()
    T = pd.DataFrame(parts)
    for (des, e), r in T.iterrows():
        if des not in ("D2-K3par", "E4-K4par") and st != "1":
            continue
        if st == "1" and e > 0.2:
            continue
        rows.append(f"{SH.get(des, des)} & {e:.2f} & " + " & ".join(
            f"{r[c]:.3f}" if c in r and pd.notna(r[c]) else "--" for c in ["NM10", "NM100", "CN50", "CN100", "LP20", "LP100"]))
table("supp_S21.tex", "Mixtures with more random starts: mean accuracy on clean units of the noise-component mixture with $10$ and $100$ starts (the latter half from random partitions, half from elemental lines), of the contaminated normal mixture with $50$ elemental and $100$ mixed starts, and of the Laplace mixture with $20$ and $100$ starts (\\texttt{mixLp}, argument \\texttt{nit}). The last two were run with more starts in the designs with three and four groups only.",
      "tab:S21", "Design & $\\varepsilon$ & Noise, 10 & Noise, 100 & CN, 50 & CN, 100 & Laplace, 20 & Laplace, 100", rows, "llcccccc")

# S22: cost of the branch and bound (bbnodes.py)
B = pd.read_csv(os.path.join(ROOT, "results_v4", "bbnodes.csv"))
def sci(v):
    if v < 1000:
        return f"${v:.0f}$"
    e = int(np.floor(np.log10(v))); return f"${v / 10 ** e:.1f}\\times10^{{{e}}}$"
rows = [f"{SH.get(r.design, 'four lines' if r.design.startswith('E4') else r.design)} & {r.K} & {r.d} & {r.m} & {sci(r.nodes_med)} & {sci(r.nodes_p90)} & {1000 * r.time_med:.2f} & {sci(r.labellings)}" for r in B.itertuples()]
table("supp_S22.tex", "Cost of the exact fit of one subsample by our branch and bound, without the admissibility constraint and on other subsamples than Table~S30: nodes visited and time in milliseconds (median and $90$th percentile over $200$ subsamples of ten simulated data sets with $10\\%$ of outliers), and the number $K^{m}/K!$ of labellings that plain enumeration would list.",
      "tab:S22", "Design & $K$ & $d$ & $m$ & Nodes, median & Nodes, $90\\%$ & Time (ms), median & $K^{m}/K!$", rows, "lccccccc")

# S23: sensitivity of Algorithm 2 to its constants
R = []
for d, st in DIRS:
    fn = os.path.join(ROOT, d, "rec_sens.csv")
    if os.path.exists(fn):
        x = pd.read_csv(fn); x["study"] = st; R.append(x)
if R:
    R = pd.concat(R)
    R = R[~((R.study == "1") & (R.eps.round(2) == 0.30))]
    cfg = [c for c in R.columns if c not in ("design", "eps", "rep", "unit", "study", "alg1")]
    C = R.groupby(["design", "eps"])[["alg1"] + cfg].mean()
    dev = C[cfg].sub(C["base"], axis=0)
    keep = dev.abs().max(axis=1) > 0.005
    cols = [c for c in cfg if c != "base"]
    rows = []
    for (des, e), r in C[keep].iterrows():
        rows.append(f"{SH.get(des, des)} & {e:.2f} & {r['alg1']:.3f} & {r['base']:.3f} & " + " & ".join(f"{r[c] - r['base']:+.3f}" for c in cols))
    hdr = "Design & $\\varepsilon$ & Alg.~1 & Proc. & " + " & ".join(c.replace("min_w", "$w$").replace("wide", "window").replace("min_frac", "support").replace("restr", "restr.").replace("max_alpha", "limit").replace("margin", "margin") for c in cols)
    table("supp_S23.tex", "Sensitivity of Algorithm~2 to its constants: mean accuracy of Algorithm~1 alone and of the procedure, and the change in the mean accuracy of the procedure when one constant is moved (peak weight $w$ from $\\tfrac12$ to $0.4$ or $0.6$; window from $3$ to $2$ or $4$ cut-offs; minimum support from $0.03n$ to $0.02n$ or $0.05n$; restriction factor from $12$ to $8$ or $20$; margin halved or doubled; limit on the flagged fraction from $\\tfrac13$ to $0.25$ or $0.40$), in the cells where some change exceeds $0.005$. All data sets of the study.",
          "tab:S23", hdr, rows, "ll" + "c" * (2 + len(cols)), wide=True)
    print("S23: max |change| over all cells:", dev.abs().max().round(3).to_dict())
    print("cells <= 0.2 max |change|", dev[dev.index.get_level_values(1) <= 0.2].abs().max().round(3).to_dict())

# S24: best of five
F = []
for d, st in DIRS:
    fn = os.path.join(ROOT, d, "best5.csv")
    if os.path.exists(fn):
        x = pd.read_csv(fn); x["study"] = st; F.append(x)
if F:
    F = pd.concat(F)
    F = F[~((F.study == "1") & (F.eps.round(2) == 0.30))]
    C = F.groupby(["design", "eps"])[["acc_mean", "acc_best5", "acc_min", "acc_max"]].mean()
    rows = [f"{SH.get(des, des)} & {e:.2f} & {r.acc_mean:.3f} & {r.acc_best5:.3f} & {r.acc_best5 - r.acc_mean:+.3f}" for (des, e), r in C.iterrows()
            if abs(r.acc_best5 - r.acc_mean) > 0.005]
    table("supp_S24.tex", "The best-of-five rule: mean accuracy of the procedure averaged over five runs with different seeds, and of the fit of highest likelihood among the five, in the cells where they differ by more than $0.005$. All data sets of the study.",
          "tab:S24", "Design & $\\varepsilon$ & One run & Best of five & Difference", rows, "llccc")
    dd = C.acc_best5 - C.acc_mean
    print("S24 diff range", dd.min().round(3), dd.idxmin(), dd.max().round(3), dd.idxmax())

# S26: solver benchmark, additions (solver_bench3.py)
fn = os.path.join(ROOT, "results_v4", "solver_bench3.csv")
if os.path.exists(fn):
    B3 = pd.read_csv(fn).drop_duplicates(subset=["design", "m", "solver", "rep"])
    lab3 = {"D1": "as D1", "D2": "as D2", "K5": "five parallel lines", "p4": "two groups, four covariates"}
    def ms3(v):
        v = 1000 * v
        s = f"{v:.2f}" if v < 10 else (f"{v:.1f}" if v < 100 else f"{v:,.0f}")
        return s.replace(",", "\\,")
    rows = []
    for (des, K, dd, m), g in B3.groupby(["design", "K", "d", "m"]):
        med = g.groupby("solver").time.median()
        cells = []
        for s_ in ["bb", "bb_inc", "rbba", "rbba_full", "miqp_g"]:
            if s_ in med.index:
                tl = int((~g[g.solver == s_].optimal.astype(bool)).sum())
                cells.append(ms3(med[s_]) + (f"$^{{{tl}}}$" if tl else ""))
            else:
                cells.append("--")
        rows.append(f"{lab3[des]} & {K} & {dd} & {m} & " + " & ".join(cells))
    table("supp_S26.tex", "Exact solution of the admissible subsample problem, further sizes (synthetic data sets of $300$ units with $10\\%$ of outliers, $30$ subsamples per row): median time in milliseconds of our depth-first search without and with an incumbent from alternation, of repetitive branch and bound as in the paper and with the incumbents of \\citet{brusco2006repetitive} (each nested problem started from the optimum of the next smaller one, the full problem from alternation), and of Gurobi. A superscript counts the solves that reached the limit ($60$ seconds or $2\\times10^{8}$ nodes).",
          "tab:S26", "Design & $K$ & $d$ & $m$ & Ours & With incumbent & Repetitive & Repetitive, incumbents & Gurobi", rows, "lcccccccc", wide=True)

# S27: mixtures with one extra start from least squares (mix_lsinit.py)
rows = []
for d, st in DIRS[:3]:
    fn = os.path.join(ROOT, d, "mix_lsinit.csv")
    if not os.path.exists(fn):
        continue
    Lx = pd.read_csv(fn)
    base = {b: pd.read_csv(os.path.join(ROOT, d, b + ".csv")) for b in ["noisemix", "cnmix"]}
    T = pd.DataFrame({"NM": base["noisemix"][base["noisemix"].unit.isin(Lx.unit)].groupby(["design", "eps"]).acc.mean(),
                      "NMLS": Lx[Lx.method == "NoiseMixLS"].groupby(["design", "eps"]).acc.mean(),
                      "CN": base["cnmix"][base["cnmix"].unit.isin(Lx.unit)].groupby(["design", "eps"]).acc.mean(),
                      "CNLS": Lx[Lx.method == "CNmixLS"].groupby(["design", "eps"]).acc.mean()})
    for (des, e), r in T.iterrows():
        if st == "1" and e > 0.29:
            continue
        rows.append(f"{SH.get(des, des)} & {e:.2f} & {r.NM:.3f} & {r.NMLS:.3f} & {r.CN:.3f} & {r.CNLS:.3f}")
if rows:
    table("supp_S27.tex", "Mixtures given one extra start from the partition of least squares with $50$ random starts, in the designs where they fail: mean accuracy on clean units of the noise-component and contaminated normal mixtures with their usual starts and with the extra start (the fit of highest likelihood after the short runs is continued, as before).",
          "tab:S27", "Design & $\\varepsilon$ & Noise, usual & Noise, with LS start & CN, usual & CN, with LS start", rows, "llcccc")

# S28: small-group designs with m = 12 (msmall.py)
simv4_m = {"G1-small20": 16, "G2-small15": 20, "G1f-small20": 16, "G2f-small15": 20, "G4-small20-p3": 20, "G5-small15-par": 20}
rows = []
for d, st in [("results_v4e", "1e"), ("results_v4g", "1g")]:
    fn = os.path.join(ROOT, d, "msmall.csv")
    if not os.path.exists(fn):
        continue
    Ms = pd.read_csv(fn)
    A2 = pd.read_csv(os.path.join(ROOT, d, "alg2.csv"))
    T = pd.DataFrame({"m12": Ms.groupby(["design", "eps"]).acc.mean(), "a1": Ms.groupby(["design", "eps"]).acc_alg1.mean(),
                      "base": A2[A2.design.isin(Ms.design.unique())].groupby(["design", "eps"]).acc.mean()})
    NM = {"G1f-small20": "groups 80/20 (fresh)", "G2f-small15": "groups 85/15 (fresh)", "G4-small20-p3": "80/20, three covariates", "G5-small15-par": "85/15, parallel lines"}
    for (des, e), r in T.iterrows():
        rows.append(f"{NM.get(des, SH.get(des, des))} & {e:.2f} & {simv4_m.get(des, '')} & {r.base:.3f} & {r.a1:.3f} & {r.m12:.3f}")
if rows:
    table("supp_S28.tex", "Small-group designs with one subsample size, $m=12$, instead of the size set from the smallest group proportion: mean accuracy on clean units of the procedure with the size used in the paper, and of Algorithm~1 alone and the procedure with $m=12$ (parts three and Plan~4).",
          "tab:S28", "Design & $\\varepsilon$ & $m$ in the paper & Procedure & Algorithm~1, $m=12$ & Procedure, $m=12$", rows, "llcccc")

# S29: B = 500 instead of 100 (bvar_all.py)
fn = os.path.join(ROOT, "results_v4", "bvar_all.csv")
if os.path.exists(fn):
    Bv = pd.read_csv(fn)
    A = []
    for d, st in [("results_v4", "1"), ("results_v4b", "1b"), ("results_v4e", "1e"), ("results_v4g", "1g")]:
        a = pd.read_csv(os.path.join(ROOT, d, "alg2.csv")); a["study"] = st; A.append(a)
    A = pd.concat(A)
    mm = Bv.merge(A[["study", "unit", "acc"]], on=["study", "unit"], suffixes=("_500", "_100"))
    mm = mm[~((mm.study == "1") & (mm.eps.round(2) == 0.30))]
    g = mm.groupby(["design", "eps"])[["acc_100", "acc_500"]].mean()
    g["d"] = g.acc_500 - g.acc_100
    NMg = {"G1f-small20": "groups 80/20 (fresh)", "G2f-small15": "groups 85/15 (fresh)", "G4-small20-p3": "80/20, three covariates", "G5-small15-par": "85/15, parallel lines"}
    rows = [f"{NMg.get(des, SH.get(des, des))} & {e:.2f} & {r.acc_100:.3f} & {r.acc_500:.3f} & {r.d:+.3f}" for (des, e), r in g.iterrows() if abs(r.d) > 0.005]
    table("supp_S29.tex", "The procedure with $B=500$ replicates (ten stages of $50$) instead of $B=100$: mean accuracy on clean units, in the cells where the two differ by more than $0.005$ ($80$ cells of parts one to three and Plan~4 examined; new random seeds for $B=500$).",
          "tab:S29", "Design & $\\varepsilon$ & $B=100$ & $B=500$ & Difference", rows, "llccc")
    print("S29 range", g.d.min().round(3), g.d.max().round(3), (g.d > 0.005).sum(), (g.d < -0.005).sum(), len(g))
import longtab
