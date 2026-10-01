"""Round 10: method names in the generated tables.  ESF = the exact-subsample flagging procedure
(Algorithm 1 followed by the recovery step); ESF alone = Algorithm 1; the recovery step =
Algorithm 2.  Run after the table scripts: python relabel.py ../paper/tables/*.tex (idempotent)."""
import re, sys

R = [
    (r"Proposed procedure \(Algorithms 1 and 2\)", "ESF"),
    (r"(?:the )?proposed procedure \(Algorithms~1 and~2\)", "ESF"),
    (r"& Proposed &", "& ESF &"),
    (r"Algorithm[ ~]1 alone \(ablation\)", "ESF alone"),
    (r"Algorithm[ ~]1 alone", "ESF alone"),
    (r"Alg\.~1 alone", "ESF alone"),
    (r"Proposed procedure", "ESF"),
    (r"\b[Tt]he (?:proposed )?procedure\b", "ESF"),
    (r"\bProcedure\b", "ESF"),
    (r"\bProc\.", "ESF"),
    (r", then Alg\.~2", ", then the recovery step"),
    (r"then Algorithm[ ~]2", "then the recovery step"),
    (r"no Alg\.~2", "no recovery"),
    (r"Alg\.~2 \(main rival\)", "recovery (main rival)"),
    (r"in Alg\.~2", "in the recovery step"),
    (r"A2 \(diff\.\)", "rec.\\ (diff.)"),
    (r"(^|[.:] |\\caption\{)Algorithm[ ~]2", r"\1The recovery step"),
    (r"Algorithm[ ~]2", "the recovery step"),
    (r"Alg\.~2", "recovery"),
    (r"(^|[.:] |\\caption\{)Algorithm[ ~]1\b", r"\1ESF alone"),
    (r"Algorithm[ ~]1\b", "ESF alone"),
    (r"Alg\.~1", "ESF alone"),
    (r"Pre-registered comparison", "Comparison of Plan~1"),
    (r"Pre-registered competitor settings", "Competitor settings as first planned"),
    (r"Pre-registration files", "Analysis plan files"),
    (r"run after the reviews of an earlier version", "decided after the analysis plans"),
    (r"ESF alone alone", "ESF alone"),
]

for f in sys.argv[1:]:
    s = open(f).read()
    t = s
    for a, b in R:
        t = re.sub(a, b, t, flags=re.M)
    if t != s:
        open(f, "w").write(t)
        print("relabelled", f)
