"""Gate 1 report, exactly as pre-registered in PREREG_GATE1.md."""
import re
import sys

import numpy as np
import pandas as pd

RESP = ["D1-K2cross", "D2-K3par", "D3-unequal", "D4-hetero", "D6-uniform", "D7-p3", "D8-K2par-sep"]
PROP = "adaptX"
FIXED_FAM = ("TCR(", "tcwm(", "TLE(")          # TCR( covers TCR(a) and TCR(a)+rw


def paired(M, cell, a, b):
    g = M[(M.design == cell[0]) & (M.eps == cell[1])]
    A = g[g.method == a].set_index("rep").acc
    Bm = g[g.method == b].set_index("rep").acc
    idx = A.index.intersection(Bm.index)
    d = (A[idx] - Bm[idx]).fillna(-1.0)          # a missing fit counts as a total failure
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), len(d)


def main(metrics_csv, out_md):
    M = pd.read_csv(metrics_csv)
    M.loc[M.acc.isna(), "acc"] = 0.0
    lines = ["# Gate 1 report", ""]
    competitors = sorted(m for m in M.method.unique() if m not in (PROP, "adapt"))
    cells = [(d, e) for d in RESP for e in (0.0, 0.05, 0.10, 0.20)]
    # (a)
    lines += ["## (a) adaptX against the best competitor in each cell", "",
              "| cell | adaptX | best competitor | its acc | diff | SE | verdict |",
              "|---|---|---|---|---|---|---|"]
    fail_a = []
    for c in cells:
        g = M[(M.design == c[0]) & (M.eps == c[1])]
        if g.empty:
            continue
        means = g.groupby("method").acc.mean()
        best = means[[c_ for c_ in competitors if c_ in means.index]].idxmax()
        d, se, n = paired(M, c, PROP, best)
        se = 0.0 if not np.isfinite(se) else se
        bad = d < -0.03 or d + 1.96 * se < -0.02
        if bad:
            fail_a.append(c)
        lines.append(f"| {c[0]} ε={c[1]:.2f} | {means[PROP]:.3f} | {best} | {means[best]:.3f} | "
                     f"{d:+.3f} | {se:.3f} | {'FAIL' if bad else 'ok'} |")
    # (b)
    lines += ["", "## (b) every fixed-level competitor loses somewhere", "",
              "| competitor | worst cell for it | diff (adaptX − it) | SE | verdict |",
              "|---|---|---|---|---|"]
    fail_b = []
    fixed = [m for m in competitors if m.startswith(FIXED_FAM) and not m.startswith("TCRx")]
    for m in fixed:
        best_c, best_d, best_se = None, -np.inf, 0
        for c in cells:
            if M[(M.design == c[0]) & (M.eps == c[1])].empty:
                continue
            d, se, n = paired(M, c, PROP, m)
            se = 0.0 if not np.isfinite(se) else se
            if d - 1.96 * se > best_d - 1.96 * best_se:
                best_c, best_d, best_se = c, d, se
        ok = best_d >= 0.05 and best_d - 1.96 * best_se > 0
        if not ok:
            fail_b.append(m)
        lines.append(f"| {m} | {best_c[0]} ε={best_c[1]:.2f} | {best_d:+.3f} | {best_se:.3f} | "
                     f"{'ok' if ok else 'FAIL'} |")
    # (c) leverage
    lines += ["", "## (c) leverage design D5 (reported, not gated)", ""]
    g = M[M.design == "D5-leverage"]
    tab = g.pivot_table(index="method", columns="eps", values="acc", aggfunc="mean").round(3)
    lines += [tab.to_markdown(), ""]
    # overall table
    lines += ["## Mean acc by cell (all methods)", ""]
    for dsg in sorted(M.design.unique()):
        g = M[M.design == dsg]
        tab = g.pivot_table(index="method", columns="eps", values="acc", aggfunc="mean").round(3)
        lines += [f"### {dsg}", "", tab.to_markdown(), ""]
    lines += ["## Failure rate (acc < 0.7), ε ≤ 0.2, response designs", ""]
    g = M[M.design.isin(RESP) & (M.eps <= 0.2)]
    lines += [g.groupby("method").fail.mean().round(3).sort_values().to_markdown(), ""]
    lines += ["## Mean time (s)", "", M.groupby("method").time.mean().round(3).sort_values().to_markdown(), ""]
    verdict = "PASS" if not fail_a and not fail_b else "FAIL"
    lines.insert(2, f"**Verdict: {verdict}.** (a) failing cells: {fail_a or 'none'}; "
                    f"(b) failing competitors: {fail_b or 'none'}.\n")
    open(out_md, "w").write("\n".join(lines))
    print(verdict, fail_a, fail_b)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
