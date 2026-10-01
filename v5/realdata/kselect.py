import sys; sys.path.insert(0,'../code')
import numpy as np, pandas as pd, recover2 as R, py_methods as P
def bic(X, y, K, m, seed):
    n, d = X.shape
    th2, a2, info, th, a = R.algorithm1_r(X, y, K, m, 100, np.random.default_rng(seed))
    xf = P._xflag(X, np.zeros(n, dtype=int), 1)
    ll, fl, s = R.loglik(X, y, th2, 1.1 * np.ptp(y), excl=xf)
    p = K * (d + 1) + K     # lines, scales, K-1 proportions + noise weight
    return -2 * ll + p * np.log(n), th2, a2
if __name__ == "__main__":
    D = pd.read_csv("fishery_log.csv"); X = np.column_stack([np.ones(len(D)), D.x1.values]); y = D.y.values
    for K, m in [(1, 4), (2, 8), (3, 16), (4, 20)]:
        b = [bic(X, y, K, m, s)[0] for s in range(5)]
        print(K, m, np.round(b, 1))
