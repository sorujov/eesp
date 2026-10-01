import pandas as pd, numpy as np
D=pd.read_csv("tone.csv")
O=pd.DataFrame({"stretchratio":[0.0]*10,"tuned":[4.0]*10})
D.to_csv("tone_clean.csv",index=False)
pd.concat([D,O],ignore_index=True).to_csv("tone_out.csv",index=False)
