"""Check that the zero-scale guard in py_methods.eesp_adaptive_x changes nothing on simulated
data: Algorithm 1 with and without the guard (py_methods_plan3.py = the Plan 3 version) on the
first 40 data sets of part 1, same seeds.  Writes results_v4/guard_check.csv."""
import os
import numpy as np, pandas as pd
os.environ.setdefault("V4_STUDY", "1")
import simv4, py_methods as P, py_methods_plan3 as P3
rows = []
for u in simv4.units()[:40]:
    X, y, z, o = simv4.generate(u); D = simv4.DESIGNS[u[0]]
    a = P.eesp_adaptive_x(X, y, D["K"], D["m"], 100, np.random.default_rng(5))[0]
    b = P3.eesp_adaptive_x(X, y, D["K"], D["m"], 100, np.random.default_rng(5))[0]
    rows.append(dict(unit=simv4.unit_name(u), max_abs_diff=float(np.abs(a - b).max())))
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(__file__), "..", "results_v4", "guard_check.csv"), index=False)
print(max(r["max_abs_diff"] for r in rows))
