"""Supplement Table S39: round-9 competitors and checks (descriptive), summarised over the 89 cells
with eps <= 0.20 and the 34 with eps >= 0.25, next to ESF and the long-search rival."""
import glob, os
import numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = pd.read_csv(os.path.join(ROOT, "results_v4", "allcells.csv")); P["study"] = P.study.astype(str); P["eps"] = P.eps.round(2)
parts = [pd.read_csv(f) for f in glob.glob(os.path.join(ROOT, "results_r9", "*.csv")) + glob.glob(os.path.join(ROOT, "results_r10", "fitspost_*.csv")) + glob.glob(os.path.join(ROOT, "results_r13", "r13_*.csv")) if "scale" not in f]
R = pd.concat(parts, ignore_index=True)
R = R[~R.method.str.endswith("_level")]
R["study"] = R.study.astype(str); R["eps"] = R.eps.round(2)
R = R[~((R.study == "1") & (R.eps == 0.30))]
R.loc[R.acc.isna(), "acc"] = 0.0
C = R.groupby(["study", "design", "eps", "method"]).acc.mean().unstack("method").reset_index()
C = C[[c for c in C.columns if c in ("study", "design", "eps") or c not in P.columns]]
M = P.merge(C, on=["study", "design", "eps"], how="left")
M.to_csv(os.path.join(ROOT, "results_v4", "allcells_r9.csv"), index=False)
ROWS = [("Alg2", "ESF"), ("ESF12", "ESF, $m=12$ in every design"), ("adaptX", "ESF alone"), ("ESF12alone", "ESF alone, $m=12$ in every design"), ("TCRlong(0.40)+rw+A2", "TCLUST-REG $0.40$, rw, recovery (main rival)"),
        ("TCRlong(0.25)+rw+A2", "TCLUST-REG $0.25$, rw, recovery (main rival)"),
        ("TCRlong(0.30)+rw+A2", "TCLUST-REG $0.30$, rw, recovery"),
        ("TCRlong(0.35)+rw+A2", "TCLUST-REG $0.35$, rw, recovery"),
        ("TCReqwL(0.25)+rw+A2", "TCLUST-REG $0.25$, equal weights, rw, recovery"),
        ("TCReqwL(0.40)+rw", "TCLUST-REG $0.40$, equal weights, rw"),
        ("TCReqwL(0.40)+rw+A2", "TCLUST-REG $0.40$, equal weights, rw, recovery"),
        ("TCRx99(0.25)+rw+A2", "TCLUST-REG $0.25$, adaptive second trimming, rw, recovery"),
        ("TCRx99(0.40)+rw+A2", "TCLUST-REG $0.40$, adaptive second trimming, rw, recovery"),
        ("TCRr1(0.40)+rw+A2", "TCLUST-REG $0.40$, equal scales, rw, recovery"),
        ("MONsel+rw+A2", "TCLUST-REG, level from the monitoring path (first stable pair), rw, recovery"),
        ("MONrest9+rw+A2", "TCLUST-REG, level from the monitoring path (stable from there on), rw, recovery"),
        ("TCRlong(0.40)+irw", "TCLUST-REG $0.40$, incremental reweighting"),
        ("TCRlong(0.40)+irw+A2", "TCLUST-REG $0.40$, incremental reweighting, recovery"),
        ("tcwm(0.25)+rw", "Trimmed CWM $0.25$, rw"), ("tcwm(0.25)+rw+A2", "Trimmed CWM $0.25$, rw, recovery"),
        ("tcwm(0.40)+rw", "Trimmed CWM $0.40$, rw"), ("tcwm(0.40)+rw+A2", "Trimmed CWM $0.40$, rw, recovery"),
        ("TMIX", "Mixture of $t$ regressions"),
        ("RLGA(0.25)+rw+A2", "RLGA, trimming $0.25$, rw, recovery"), ("RLGA(0.40)+rw+A2", "RLGA, trimming $0.40$, rw, recovery"),
        ("TCRlong(0.40)+rw+A2R", "TCLUST-REG $0.40$, rw, recovery with robust noise range"),
        ("Alg1+A2R", "ESF with robust noise range in the recovery step")]
lo, hi = M[M.eps <= 0.2001], M[M.eps >= 0.2499]
L = ["\\begin{tabular}{lcccccc}", "\\toprule",
     " & \\multicolumn{3}{c}{$\\varepsilon\\le0.20$ (" + str(len(lo)) + " cells)} & \\multicolumn{3}{c}{$\\varepsilon\\ge0.25$ (" + str(len(hi)) + " cells)} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}", "Method & Mean & Below $0.85$ & Worst & Mean & Below $0.85$ & Worst \\\\", "\\midrule"]
for k, lab in ROWS:
    if k not in M.columns or M[k].isna().all():
        continue
    def f(Q):
        v = Q[k].dropna()
        return f"{v.mean():.3f} & {int((v < 0.85).sum())} & {v.min():.2f}" + ("" if len(v) == len(Q) else f"$^{{({len(v)})}}$")
    L.append(f"{lab} & {f(lo)} & {f(hi)} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(ROOT, "paper", "tables", "supp_r9.tex"), "w").write("\n".join(L))
print("\n".join(L))
