# /// script
# dependencies = ["numpy","pandas","scikit-learn"]
# ///
import numpy as np, pandas as pd, json
from sklearn.linear_model import LogisticRegression
S=["en","hi_deva","hi_rom","pa_guru","pa_rom"]
TAG="tiny-aya-global"
df=pd.read_json(f"runs/baseline/{TAG}/gen_{TAG}_judged.jsonl",lines=True)
H=np.load(f"runs/baseline/{TAG}/hidden_{TAG}.npy").astype(np.float32)
fit=set(json.load(open("fit_ids.json"))); df["fit"]=df.id.isin(fit)
y=df.label.values; sc=df.script.values; ev=~df.fit.values
def report(name,p,t):
    print(f"{name:28s} t={t} "+"  ".join(f"{s}:{(p[ev&(sc==s)&(y==1)]>t).mean():.2f}/{(p[ev&(sc==s)&(y==0)]>t).mean():.2f}" for s in S))
for L in [8,10,12,14]:
    trn=df.fit.values
    X=H[:,L]; mu=X[trn].mean(0); sd=X[trn].std(0)+1e-6
    p=LogisticRegression(max_iter=3000,C=0.05).fit((X[trn]-mu)/sd,y[trn]).predict_proba((X-mu)/sd)[:,1]
    for t in [0.5,0.7]: report(f"all-scripts L{L}",p,t)
# English-only ensemble
trn=(df.fit&(df.script=="en")).values; ps=[]
for L in [10,11,12,13,14]:
    X=H[:,L]; mu=X[trn].mean(0); sd=X[trn].std(0)+1e-6
    ps.append(LogisticRegression(max_iter=3000,C=0.05).fit((X[trn]-mu)/sd,y[trn]).predict_proba((X-mu)/sd)[:,1])
p=np.mean(ps,0)
for t in [0.5,0.3,0.2]: report("en-only ens L10-14",p,t)
