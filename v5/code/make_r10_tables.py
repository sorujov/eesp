"""Round 10 (descriptive) supplement tables.
supp_ari.tex: adjusted Rand index on all units (outliers as a class of their own, flagged or trimmed
units labelled as outliers) and misclassification rate of the clean units counting a flagged clean
unit as an error, for ESF and the long-search TCLUST-REG pipelines (results_ari/, ari_all.py).
supp_tcwm.tex: the trimmed cluster-weighted model at levels 0.25 and 0.40 (RobMixReg, 50 starts),
alone, reweighted, and followed by the recovery step (results_r9/fitspost_tcwm.csv, fits_post.py).
Cells as in the main tables: the eps = 0.30 cells of the first part are left out."""
import glob, os
import pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
T = os.path.join(ROOT, "paper", "tables")


def cells(d, val):
    d = d.copy(); d["study"] = d.study.astype(str); d["eps"] = d.eps.round(2)
    d = d[~((d.study == "1") & (d.eps >= 0.30))]
    return d.groupby(["study", "design", "eps", "method"])[val].mean().reset_index()


A = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, "results_ari", "ari_*.csv")))])
C = cells(A, ["ari", "mis"])
lo, hi = C[C.eps <= 0.2001], C[C.eps >= 0.2499]
ROWS = [("Alg2", "ESF"), ("TCRlong(0.25)+rw+A2", "TCLUST-REG, $\\alpha=0.25$, rw, recovery"),
        ("TCRlong(0.40)+rw+A2", "TCLUST-REG, $\\alpha=0.40$, rw, recovery"),
        ("TCRlong(0.25)+rw", "TCLUST-REG, $\\alpha=0.25$, rw"), ("TCRlong(0.40)+rw", "TCLUST-REG, $\\alpha=0.40$, rw")]
L = ["\\begin{tabular}{lcccccccc}", "\\toprule",
     f" & \\multicolumn{{4}}{{c}}{{$\\varepsilon\\le0.20$ ({lo.groupby(['study','design','eps']).ngroups} cells)}} & \\multicolumn{{4}}{{c}}{{$\\varepsilon\\ge0.25$ ({hi.groupby(['study','design','eps']).ngroups} cells)}} \\\\",
     "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
     "Method & ARI mean & ARI worst & Misc.\\ mean & Misc.\\ worst & ARI mean & ARI worst & Misc.\\ mean & Misc.\\ worst \\\\", "\\midrule"]
for k, lab in ROWS:
    def f(Q):
        q = Q[Q.method == k]
        return f"{q.ari.mean():.3f} & {q.ari.min():.2f} & {q.mis.mean():.3f} & {q.mis.max():.2f}"
    L.append(f"{lab} & {f(lo)} & {f(hi)} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(T, "supp_ari.tex"), "w").write("\n".join(L))
print("\n".join(L))

W = cells(pd.read_csv(os.path.join(ROOT, "results_r9", "fitspost_tcwm.csv")), "acc")
lo, hi = W[W.eps <= 0.2001], W[W.eps >= 0.2499]
L = ["\\begin{tabular}{lcccccc}", "\\toprule",
     f" & \\multicolumn{{3}}{{c}}{{$\\varepsilon\\le0.20$ ({lo.groupby(['study','design','eps']).ngroups} cells)}} & \\multicolumn{{3}}{{c}}{{$\\varepsilon\\ge0.25$ ({hi.groupby(['study','design','eps']).ngroups} cells)}} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}", "Method & Mean & Below $0.85$ & Worst & Mean & Below $0.85$ & Worst \\\\", "\\midrule"]
for a in ("0.25", "0.40"):
    for s, lab in (("", "alone"), ("+rw", "rw"), ("+rw+A2", "rw, recovery")):
        k = f"tcwm({a}){s}"
        def f(Q):
            v = Q[Q.method == k].acc
            return f"{v.mean():.3f} & {int((v < 0.85).sum())} & {v.min():.2f}"
        L.append(f"Trimmed CWM, $\\alpha={a}$, {lab} & {f(lo)} & {f(hi)} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(T, "supp_tcwm.tex"), "w").write("\n".join(L))
print("\n".join(L))
P = W.pivot_table(index=["study", "design", "eps"], columns="method", values="acc")
for a in ("0.25", "0.40"):
    d = P[f"tcwm({a})+rw+A2"] - P[f"tcwm({a})+rw"]
    print(a, "gain>0.02", int((d > 0.02).sum()), "loss>0.02", int((d < -0.02).sum()), "worst", round(d.min(), 3),
          d[d < -0.02].round(3).to_dict())

# Round 12: sensitivity of the recovery step to the noise range (results_r12/r12range_*.csv, r12_range.py)
G = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, "results_r12", "r12range_*.csv")))]
              + [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, "results_r13", "r13_*.csv")))])
G = G[~G.method.str.startswith("ESF12")]
G = cells(G, "acc")
Mb = pd.read_csv(os.path.join(ROOT, "results_v4", "allcells.csv")); Mb["study"] = Mb.study.astype(str); Mb["eps"] = Mb.eps.round(2)
B0 = Mb.melt(id_vars=["study", "design", "eps"], value_vars=["Alg2", "TCRlong(0.40)+rw+A2"], var_name="method", value_name="acc")
G = pd.concat([B0, G], ignore_index=True)
lo, hi = G[G.eps <= 0.2001], G[G.eps >= 0.2499]
L = ["\\begin{tabular}{lcccccc}", "\\toprule",
     f" & \\multicolumn{{3}}{{c}}{{$\\varepsilon\\le0.20$ ({lo.groupby(['study','design','eps']).ngroups} cells)}} & \\multicolumn{{3}}{{c}}{{$\\varepsilon\\ge0.25$ ({hi.groupby(['study','design','eps']).ngroups} cells)}} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}", "Method & Mean & Below $0.85$ & Worst & Mean & Below $0.85$ & Worst \\\\", "\\midrule"]
for k, lab in (("Alg2", "ESF, range $R$ as in the paper"), ("Alg2_R2", "ESF, range $2R$"), ("Alg2_R5", "ESF, range $5R$"),
               ("Alg2_Rs20", "ESF, range $20$ median scales"), ("Alg2_Ru10", "ESF, unflagged range plus $10$ median scales"),
               ("TCRlong(0.40)+rw+A2", "TCLUST-REG $0.40$, rw, recovery, range $R$"),
               ("TCRlong(0.40)+rw+A2_R2", "TCLUST-REG $0.40$, rw, recovery, range $2R$"),
               ("TCRlong(0.40)+rw+A2_R5", "TCLUST-REG $0.40$, rw, recovery, range $5R$"),
               ("TCRlong(0.40)+rw+A2_Rs20", "TCLUST-REG $0.40$, rw, recovery, $20$ median scales"),
               ("TCRlong(0.40)+rw+A2_Ru10", "TCLUST-REG $0.40$, rw, recovery, unflagged range plus $10$ scales")):
    def f(Q):
        v = Q[Q.method == k].acc
        return f"{v.mean():.3f} & {int((v < 0.85).sum())} & {v.min():.2f}"
    L.append(f"{lab} & {f(lo)} & {f(hi)} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(T, "supp_range.tex"), "w").write("\n".join(L))
print("\n".join(L))
P = G.pivot_table(index=["study", "design", "eps"], columns="method", values="acc")
for base, alts in (("Alg2", ("Alg2_R2", "Alg2_R5")), ("TCRlong(0.40)+rw+A2", ("TCRlong(0.40)+rw+A2_R2", "TCRlong(0.40)+rw+A2_R5"))):
    for a in alts:
        d = P[a] - P[base]
        for nm, sel in (("low", d.index.get_level_values("eps") <= 0.2001), ("high", d.index.get_level_values("eps") >= 0.2499)):
            x = d[sel]
            print(a, nm, "max loss", round(-x.min(), 3), "at", x.idxmin(), "cells lost >0.02:", int((x < -0.02).sum()), "gain>0.005:", int((x > 0.005).sum()))

# Round 12: TCLUST-REG's own assignment against the paper's criterion (results_r12/idx_post.csv, r12_idx_post.py)
I = pd.read_csv(os.path.join(ROOT, "results_r12", "idx_post.csv"))
I = cells(I, ["acc_nearest", "acc_own", "clean_trimmed"])
DN = {"D2-K3par": "D2", "D4-hetero": "D4", "E4-K4par": "$K=4$"}
L = ["\\begin{tabular}{llcccccc}", "\\toprule",
     " & & \\multicolumn{3}{c}{$\\alpha=0.25$} & \\multicolumn{3}{c}{$\\alpha=0.40$} \\\\",
     "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
     "Design & $\\varepsilon$ & Nearest line & Own assignment & Clean trimmed & Nearest line & Own assignment & Clean trimmed \\\\", "\\midrule"]
for (st, d, e), g in I.groupby(["study", "design", "eps"]):
    g = g.set_index("method")
    v = lambda m: f"{g.loc[m, 'acc_nearest']:.3f} & {g.loc[m, 'acc_own']:.3f} & {g.loc[m, 'clean_trimmed']:.3f}"
    L.append(f"{DN[d]} & {e:.2f} & {v('TCRlong(0.25)')} & {v('TCRlong(0.40)')} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(T, "supp_idx.tex"), "w").write("\n".join(L))
dd = I.acc_own - I.acc_nearest
print("own - nearest: min", round(dd.min(), 3), "max", round(dd.max(), 3), "| eps<=0.20:", round(dd[I.eps <= 0.2001].min(), 3), round(dd[I.eps <= 0.2001].max(), 3))
