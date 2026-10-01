"""Gate 1b report, as pre-registered in PREREG_GATE1B.md."""
import sys

import numpy as np
import pandas as pd

RESP = ["D1-K2cross", "D2-K3par", "D3-unequal", "D4-hetero", "D6-uniform", "D7-p3", "D8-K2par-sep"]
PROP = "adaptX"


def paired(g, a, b):
    A = g[g.method == a].set_index("rep").acc
    B = g[g.method == b].set_index("rep").acc
    idx = A.index.intersection(B.index)
    d = A[idx] - B[idx]
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), len(d)


def main(metrics_csv, out_md):
    M = pd.read_csv(metrics_csv)
    M.loc[M.acc.isna(), "acc"] = 0.0
    L = ["# Gate 1b report", ""]
    # (b')
    L += ["## (b') TCR(0.25)+rw against adaptX, response designs", "",
          "| cell | adaptX | TCR(0.25)+rw | diff | SE | n | lower 95% |", "|---|---|---|---|---|---|---|"]
    ok_b = False
    for d in RESP:
        for e in sorted(M.eps.unique()):
            g = M[(M.design == d) & (M.eps == e)]
            if g.empty:
                continue
            diff, se, n = paired(g, PROP, "TCR(0.25)+rw")
            lo = diff - 1.96 * se
            if diff >= 0.05 and lo > 0:
                ok_b = True
            ma = g[g.method == PROP].acc.mean()
            mt = g[g.method == "TCR(0.25)+rw"].acc.mean()
            L.append(f"| {d} ε={e:.2f} | {ma:.3f} | {mt:.3f} | {diff:+.3f} | {se:.3f} | {n} | {lo:+.3f} |")
    L.insert(2, f"**(b') {'HOLDS' if ok_b else 'FAILS'}.**\n")
    # (d) limits
    L += ["", "## (d) cells where adaptX mean accuracy < 0.85", ""]
    cell = M.groupby(["design", "eps", "method"]).acc.mean()
    for d in sorted(M.design.unique()):
        for e in sorted(M.eps.unique()):
            try:
                v = cell.loc[(d, e, PROP)]
            except KeyError:
                continue
            if v < 0.85:
                L.append(f"- {d} ε={e:.2f}: {v:.3f}")
    L += ["", "## (d) cells where a competitor beats adaptX by > 0.03 (diff + 1.96 SE < 0)", ""]
    for d in sorted(M.design.unique()):
        for e in sorted(M.eps.unique()):
            g = M[(M.design == d) & (M.eps == e)]
            if g.empty:
                continue
            for m in sorted(g.method.unique()):
                if m in (PROP, "adapt"):
                    continue
                diff, se, n = paired(g, PROP, m)
                if diff < -0.03 and diff + 1.96 * se < 0:
                    L.append(f"- {d} ε={e:.2f}: {m} ahead by {-diff:.3f} (SE {se:.3f})")
    L += ["", "## Mean accuracy, selected methods", ""]
    sel = ["adaptX", "adapt", "TCR(0.25)+rw", "TCR(0.40)+rw", "TCR(0.25)", "TCR(0.40)", "TCRx(0.25)+rw",
           "tcwm(0.40)", "TLE(0.40)", "CTLE", "bisq"]
    t = M[M.method.isin(sel)].pivot_table(index=["design", "eps"], columns="method", values="acc").round(3)
    L += [t[[c for c in sel if c in t.columns]].to_markdown()]
    open(out_md, "w").write("\n".join(L))
    print("(b')", "HOLDS" if ok_b else "FAILS")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
