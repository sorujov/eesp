"""Turn supplement tables too long for one page into longtable (repeated header)."""
import glob, os, re
TAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "tables")
for fn in sorted(glob.glob(os.path.join(TAB, "supp_S*.tex"))):
    L = open(fn).read().split("\n")
    if "\\begin{longtable}" in "\n".join(L) or sum(l.rstrip().endswith("\\\\") for l in L) <= 45:
        continue
    cap = next(l for l in L if l.startswith("\\caption"))
    lab = next(l for l in L if l.startswith("\\label"))
    i_tab = next(i for i, l in enumerate(L) if "\\begin{tabular}" in l)
    spec = re.search(r"\\begin\{tabular\}\{([^}]*)\}", L[i_tab]).group(1)
    i_mid = next(i for i, l in enumerate(L) if l.startswith("\\midrule"))
    i_bot = max(i for i, l in enumerate(L) if l.startswith("\\bottomrule"))
    head = L[i_tab + 1:i_mid]
    size = "\\small" if "\\small" in L else ("\\footnotesize" if "\\footnotesize" in L else "")
    body = L[i_mid + 1:i_bot]
    ncol = spec.count("l") + spec.count("c") + spec.count("r")
    out = ["{" + size, f"\\begin{{longtable}}{{{spec}}}", cap + lab + " \\\\"] + head + ["\\midrule", "\\endfirsthead"] + \
          [f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{Table \\thetable, continued}}}} \\\\"] + head + ["\\midrule", "\\endhead"] + \
          body + ["\\bottomrule", "\\end{longtable}", "}"]
    open(fn, "w").write("\n".join(out))
    print("longtable:", os.path.basename(fn))
