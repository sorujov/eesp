import os, sys, numpy as np, pandas as pd
sys.path.insert(0, "../code")
import py_methods as P
rows=[]
for f in ["tone_clean","tone_out"]:
    D=pd.read_csv(f+".csv"); x=D.stretchratio.values; y=D.tuned.values
    X=np.column_stack([np.ones(len(x)),x])
    for s in range(20):
        for meth in ["alg1","nm"]:
            rng=np.random.default_rng(s)
            if meth=="alg1": th,ah,t=P.eesp_adaptive_x(X,y,2,8,100,rng,xtrim=True)
            else: th,ah,t=P.noise_mix(X,y,2,rng)
            o=np.argsort(th[:,1]); th=th[o]
            rows.append(dict(data=f,method=meth,seed=s,b0_1=th[0,0],b1_1=th[0,1],b0_2=th[1,0],b1_2=th[1,1],flagged=ah,time=t))
R=pd.DataFrame(rows); R.to_csv("tone_py.csv",index=False)
print(R.groupby(["data","method"])[["b0_1","b1_1","b0_2","b1_2","flagged"]].agg(["mean","min","max"]).round(3).T.to_string())
