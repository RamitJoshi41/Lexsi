# /// script
# dependencies = ["numpy", "pandas", "scikit-learn"]
# ///
"""Retrain the harm gate with everyday benign negatives; pick layer/threshold on held-out data; save gate params."""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

S = ["en", "hi_deva", "hi_rom", "pa_guru", "pa_rom"]; TAG = "tiny-aya-global"
df = pd.read_json(f"runs/baseline/{TAG}/gen_{TAG}_judged.jsonl", lines=True)
H = np.load(f"runs/baseline/{TAG}/hidden_{TAG}.npy").astype(np.float32)
B = np.load(f"runs/benign/{TAG}/benign_feats.npy").astype(np.float32); LB = list(range(6, 15))
bm = pd.read_json(f"runs/benign/{TAG}/benign_meta.jsonl", lines=True)
fit = set(json.load(open("fit_ids.json"))); isfit = df.id.isin(fit).values; y = df.label.values; sc = df.script.values
rng = np.random.default_rng(0)
btest = np.zeros(len(bm), bool)
for lang in bm.lang.unique():
    ix = np.where(bm.lang == lang)[0]; btest[rng.choice(ix, 200, replace=False)] = True
best = None
for L in LB:
    Xo = H[:, L]; Xb = B[:, LB.index(L)]
    Xtr = np.concatenate([Xo[isfit], Xb[~btest]]); ytr = np.concatenate([y[isfit], np.zeros((~btest).sum())])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    wt = np.where(ytr == 1, (ytr == 0).sum() / (ytr == 1).sum(), 1.0)   # balance classes
    clf = LogisticRegression(max_iter=4000, C=0.05).fit((Xtr - mu) / sd, ytr, sample_weight=wt)
    po = clf.predict_proba((Xo - mu) / sd)[:, 1]; pb = clf.predict_proba((Xb - mu) / sd)[:, 1]
    for t in [0.5, 0.7, 0.8, 0.9, 0.95]:
        ev = ~isfit
        tpr = {s: (po[ev & (sc == s) & (y == 1)] > t).mean() for s in S}
        fpr_x = {s: (po[ev & (sc == s) & (y == 0)] > t).mean() for s in S}
        fpr_b = {l: (pb[btest & (bm.lang == l).values] > t).mean() for l in bm.lang.unique()}
        score = np.mean(list(tpr.values())) - 2 * max(fpr_b.values()) - np.mean(list(fpr_x.values()))
        print(f"L{L:2d} t={t:.2f} TPR " + " ".join(f"{s}:{v:.2f}" for s, v in tpr.items()) +
              " | FPR-xstest " + " ".join(f"{v:.2f}" for v in fpr_x.values()) +
              " | FPR-everyday " + " ".join(f"{k[:2]}:{v:.3f}" for k, v in fpr_b.items()))
        if best is None or score > best[0]:
            best = (score, L, t, clf, mu, sd, tpr, fpr_x, fpr_b)
score, L, t, clf, mu, sd, tpr, fpr_x, fpr_b = best
print(f"\nBEST: layer {L}, threshold {t}\n TPR {tpr}\n FPR xstest {fpr_x}\n FPR everyday (held-out) {fpr_b}")
w_raw = clf.coef_[0] / sd; b_raw = clf.intercept_[0] - (clf.coef_[0] * mu / sd).sum()
np.savez("gate_v2.npz", w=w_raw.astype(np.float32), b=np.float32(b_raw), layer=L, thr=t)
json.dump({"layer": L, "threshold": t, "tpr_heldout": tpr, "fpr_xstest_heldout": fpr_x, "fpr_everyday_heldout": fpr_b},
          open("results/gate_v2.json", "w"), indent=1, default=float)
