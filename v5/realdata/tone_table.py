"""Tone perception data (mixtools; De Veaux 1989), K = 2, with and without ten added points at (0,4).
Reads tone_py.csv, tone_tcr.csv, tone_r.csv; writes paper/tables/tone.tex."""
import numpy as np, pandas as pd
P = pd.concat([pd.read_csv("tone_py.csv"), pd.read_csv("tone_cn.csv")])
T = pd.read_csv("tone_tcr.csv")
R = pd.concat([pd.read_csv("tone_r.csv"), pd.read_csv("tone_lp.csv").assign(trimmed=float("nan"))]).rename(columns={"X1": "a", "X2": "b", "X3": "c", "X4": "d"})
rows = []
for _, r in P.iterrows():
    rows.append(dict(data=r.data, method={"alg1": "Procedure", "nm": "Noise-component mixture", "cn": "Contaminated normal mixture"}[r.method],
                     co=[(r.b0_1, r.b1_1), (r.b0_2, r.b1_2)], frac=r.flagged))
for _, r in T.iterrows():
    rows.append(dict(data=r.data, method=f"TCLUST-REG, $\\alpha={r.alpha:.2f}$", co=[(r.b0_1, r.b1_1), (r.b0_2, r.b1_2)], frac=r.trimmed))
for _, r in R.iterrows():
    # RobMixReg compcoef is (intercept, slope) x K, stored column-wise
    m = {"CTLE": "CTLE", "Laplace": "Laplace mixture"}[r.method] if r.method in ("CTLE", "Laplace") else "Trimmed CWM, $\\alpha=" + r.method[5:9] + "$"
    rows.append(dict(data=r.data, method=m, co=[(r.a, r.b), (r.c, r.d)], frac=r.trimmed))
for q in rows:
    q["co"] = sorted(q["co"], key=lambda t: t[1])            # flat line first
    (a0, a1), (c0, c1) = q["co"]
    # the two lines of the data: a nearly flat line near 2 and a line of slope near one
    q["bd"] = not (1.8 < a0 < 2.1 and abs(a1) < 0.1 and abs(c1 - 1) < 0.1)
D = pd.DataFrame(rows)
def f(v):
    v = np.asarray(v, float)
    if len(v) == 0 or np.isnan(v).all():
        return "--"
    md = np.median(v)
    return f"{md:.2f}" + (f" [{v.min():.2f}, {v.max():.2f}]" if v.max() - v.min() > 0.005 else "")
order = ["Procedure", "Contaminated normal mixture", "Noise-component mixture"] + [f"TCLUST-REG, $\\alpha={a:.2f}$" for a in (0.02, 0.05, 0.10, 0.15, 0.25, 0.40)] + \
        ["CTLE", "Laplace mixture"] + [f"Trimmed CWM, $\\alpha={a}$" for a in ("0.05", "0.10", "0.25")]
L = ["\\begin{tabular}{lccccc}", "\\toprule",
     " & \\multicolumn{3}{c}{Original data ($n=150$)} & \\multicolumn{2}{c}{With ten points at $(0,4)$} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}",
     "Method & Flat line & Other fits & Fraction & Other fits & Fraction \\\\", "\\midrule"]
for m in order:
    a = D[(D.data == "tone_clean") & (D.method == m)]; b = D[(D.data == "tone_out") & (D.method == m)]
    if a.empty: continue
    oa, ob = a[~a.bd], b[~b.bd]
    fl = f"{f([c[0][0] for c in oa.co])}, {f([c[0][1] for c in oa.co])}"
    L.append(f"{m} & {fl} & {int(a.bd.sum())}/{len(a)} & {f(oa.frac)} & {int(b.bd.sum())}/{len(b)} & {f(ob.frac) if len(ob) else '--'} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open("../paper/tables/tone.tex", "w").write("\n".join(L))
print("\n".join(L))
for m in order:
    a = D[(D.data == "tone_clean") & (D.method == m) & ~D.bd]
    print(m, f([c[1][0] for c in a.co]), f([c[1][1] for c in a.co]))
