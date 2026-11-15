# /// script
# dependencies = ["numpy","pandas","scikit-learn"]
# ///
import numpy as np, pandas as pd, json
from sklearn.linear_model import LogisticRegression
S=["en","hi_deva","hi_rom","pa_guru","pa_rom"]
for TAG in ["tiny-aya-global"]:
    df=pd.read_json(f"runs/baseline/{TAG}/gen_{TAG}_judged.jsonl",lines=True)
    H=np.load(f"runs/baseline/{TAG}/hidden_{TAG}.npy").astype(np.float32)
    ids=sorted(df.id.unique()); rng=np.random.default_rng(0); fit=set(rng.choice(ids,len(ids)//2,replace=False))
    df["fit"]=df.id.isin(fit); json.dump(sorted(fit),open("fit_ids.json","w"))
    y=df.label.values; sc=df.script.values
    trn=(df.fit&(df.script=="en")).values
    print("fit ids",len(fit),"| eval rows",(~df.fit).sum())
    for L in [4,6,8,10,12,14]:
        X=H[:,L]; mu=X[trn].mean(0); sd=X[trn].std(0)+1e-6
        clf=LogisticRegression(max_iter=3000,C=0.05).fit((X[trn]-mu)/sd,y[trn])
        p=clf.predict_proba((X-mu)/sd)[:,1]
        for t in [0.5,0.3]:
            row=[]
            for s in S:
                m=(~df.fit).values&(sc==s)
                row.append(f"{s}:TPR{(p[m&(y==1)]>t).mean():.2f}/FPR{(p[m&(y==0)]>t).mean():.2f}")
            print(f"L{L:2d} t={t}  "+"  ".join(row))
