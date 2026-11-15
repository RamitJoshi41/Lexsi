# /// script
# dependencies = ["numpy", "pandas", "scikit-learn"]
# ///
"""Gate v3.1 = gate v3 + 940 fit-set prompts wrapped in 12 extra chatty filler templates (aug_fit2). Evaluated on
filler_test: HELD-OUT prompts x 5 UNSEEN templates (+ template 5 = the v3 test wrapper). Writes gate_v3_1.npz, results/gate_v3_1.json.
Original doc: Gate v3: v2's data + (train halves of) new-category pairs, romanized everyday prompts (orig + ASCII-folded),
and noisy/filler-wrapped FIT-set prompts. Layer + threshold chosen on a validation split of the TRAIN data only;
everything reported on untouched TEST sets, next to gate v2. Writes gate_v3.npz, results/gate_v3.json."""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
TAG = "tiny-aya-global"; S = ["en", "hi_deva", "hi_rom", "pa_guru", "pa_rom"]; LB = list(range(6, 15)); rng = np.random.default_rng(0)
# --- codemix (baseline hidden, all 36 layers) ---
cm = pd.read_json(f"runs/baseline/{TAG}/gen_{TAG}_judged4.jsonl", lines=True)
Hc = np.load(f"runs/baseline/{TAG}/hidden_{TAG}.npy", mmap_mode="r")
fit = set(json.load(open("fit_ids.json"))); cm["split"] = np.where(cm.id.isin(fit), "train", "test"); cm["src"] = "codemix"
# --- everyday native/English (benign_feats: layers 6..14), v2's exact held-out 600 ---
bm = pd.read_json(f"runs/benign/{TAG}/benign_meta.jsonl", lines=True); Hb = np.load(f"runs/benign/{TAG}/benign_feats.npy", mmap_mode="r")
bt = np.zeros(len(bm), bool)
for lang in bm.lang.unique():
    ix = np.where(bm.lang == lang)[0]; bt[np.random.default_rng(0).choice(ix, 200, replace=False)] = True
bm["split"] = np.where(bt, "test", "train"); bm["label"] = 0; bm["src"] = "everyday"; bm["script"] = bm.lang
# --- extend job (layers 6..18) ---
ex = pd.read_json(f"runs/extend/{TAG}/extend_rows.jsonl", lines=True); He = np.load(f"runs/extend/{TAG}/hidden_extend.npy", mmap_mode="r")
ex["src"] = ex.part
nc = ex.part == "newcats"
pairs = sorted(ex[nc].pair.unique()); trp = {p for c in ex[nc].category.unique() for p in sorted(ex[nc & (ex.category == c)].pair.unique())[::2]}
ex.loc[nc, "split"] = np.where(ex[nc].pair.isin(trp), "train", "test")
ex.loc[ex.part == "perturb", "split"] = "test"
ex.loc[ex.part.str.startswith("everyday"), "split"] = "skip"          # replaced by feats/everyday_rom (has ASCII copies)
# --- feats job (layers 6..18) ---
ag = pd.read_json(f"runs/feats/{TAG}/aug_fit.jsonl", lines=True); Ha = np.load(f"runs/feats/{TAG}/aug_fit.npy", mmap_mode="r")
ag["split"] = "train"; ag["src"] = "aug_fit"
er = pd.read_json(f"runs/feats/{TAG}/everyday_rom.jsonl", lines=True); Hr = np.load(f"runs/feats/{TAG}/everyday_rom.npy", mmap_mode="r")
base = er.id.str.replace("_ascii", "", regex=False); ub = base.unique(); trb = set(rng.choice(ub, len(ub) // 2, replace=False))
a2 = pd.read_json(f"runs/feats/{TAG}/aug_fit2.jsonl", lines=True); Ha2 = np.load(f"runs/feats/{TAG}/aug_fit2.npy", mmap_mode="r")
a2["split"] = "train"; a2["src"] = "aug_fit2"
ft = pd.read_json(f"runs/feats/{TAG}/filler_test.jsonl", lines=True); Hf = np.load(f"runs/feats/{TAG}/filler_test.npy", mmap_mode="r")
ft["split"] = "test"; ft["src"] = "filler_t" + ft.template.astype(str)
er["split"] = np.where(base.isin(trb), "train", "test"); er["src"] = np.where(er.id.str.endswith("_ascii"), "everyday_rom_ascii", "everyday_rom")
def X(L):   # stacked features at layer L for every source
    parts = [(cm, Hc[:, L]), (bm, Hb[:, LB.index(L)]), (ex, He[:, L - 6]), (ag, Ha[:, L - 6]), (er, Hr[:, L - 6]), (a2, Ha2[:, L - 6]), (ft, Hf[:, L - 6])]
    return np.concatenate([np.asarray(h, np.float32) for _, h in parts])
META = pd.concat([d[["src", "script", "label", "split"] + [c for c in ["perturb", "category"] if c in d]].assign(_k=i) for i, (d) in enumerate([cm, bm, ex, ag, er, a2, ft])], ignore_index=True)
tr = (META.split == "train").values; te = (META.split == "test").values; y = META.label.values.astype(int)
val = tr & (rng.random(len(META)) < 0.2); fitm = tr & ~val
def train(L, mask):
    Xa = X(L); mu, sd = Xa[mask].mean(0), Xa[mask].std(0) + 1e-6
    w = np.where(y[mask] == 1, (y[mask] == 0).sum() / max(1, (y[mask] == 1).sum()), 1.0)
    clf = LogisticRegression(max_iter=4000, C=0.05).fit((Xa[mask] - mu) / sd, y[mask], sample_weight=w)
    return clf, mu, sd, clf.predict_proba((Xa - mu) / sd)[:, 1]
def report(p, t, mask):
    M = META[mask].assign(f=p[mask] > t); out = {}
    for (src, lab), g in M.groupby(["src", "label"]):
        out[f"{src}|{'harm' if lab else 'safe'}"] = {s: round(float(v), 3) for s, v in g.groupby("script").f.mean().items()} | {"ALL": round(float(g.f.mean()), 3), "n": int(len(g))}
    if "perturb" in M:
        for (k, lab), g in M[M.src == "perturb"].groupby(["perturb", "label"]): out[f"perturb_{k}|{'harm' if lab else 'safe'}"] = round(float(g.f.mean()), 3)
    return out
def score(p, t, mask):
    M = META[mask].assign(f=p[mask] > t)
    tpr = M[(M.label == 1) & M.src.isin(["codemix", "newcats", "aug_fit", "aug_fit2"])].groupby("src").f.mean().mean()
    fpr_every = M[(M.label == 0) & M.src.str.startswith("everyday")].groupby("src").f.mean().max()
    fpr_other = M[(M.label == 0) & ~M.src.str.startswith("everyday")].f.mean()
    return tpr - 2 * fpr_every - fpr_other
best = None
for L in LB:
    clf, mu, sd, p = train(L, fitm)
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        sc = score(p, t, val)
        if best is None or sc > best[0]: best = (sc, L, t)
    print("L", L, "best-so-far", best, flush=True)
_, L, t = best
clf, mu, sd, p = train(L, tr)
w_raw = clf.coef_[0] / sd; b_raw = clf.intercept_[0] - (clf.coef_[0] * mu / sd).sum()
np.savez("gate_v3_1.npz", w=w_raw.astype(np.float32), b=np.float32(b_raw), layer=L, thr=t)
g2 = np.load("gate_v3.npz"); p2 = 1 / (1 + np.exp(-(X(int(g2["layer"])) @ g2["w"] + float(g2["b"]))))
res = {"layer": L, "threshold": t, "selected_on": "validation split of TRAIN data", "v3_1_test": report(p, t, te), "v3_test": report(p2, float(g2["thr"]), te)}
fm = te & META.src.str.startswith("filler").values; F = META[fm].assign(v31=p[fm] > t, v3=p2[fm] > float(g2["thr"]))
F["tmpl"] = np.where(F.src == "filler_t5", "seen-by-v3-test wrapper (t5)", "unseen templates t0-t4")
res["filler_summary"] = {f"{k}|{'harm' if lab else 'safe'}": {"v3": round(float(g.v3.mean()), 3), "v3_1": round(float(g.v31.mean()), 3), "n": int(len(g))}
                         for (k, lab), g in F.groupby(["tmpl", "label"])}
res["filler_by_template"] = {f"{k}|{'harm' if lab else 'safe'}": {"v3": round(float(g.v3.mean()), 3), "v3_1": round(float(g.v31.mean()), 3)} for (k, lab), g in F.groupby(["src", "label"])}
json.dump(res, open("results/gate_v3_1.json", "w"), indent=1)
print("layer", L, "thr", t); print(json.dumps(res["filler_summary"], indent=1)); print(json.dumps(res["filler_by_template"], indent=1))
for k in res["v3_1_test"]:
    if not k.startswith("filler"): print(f"{k:32s} v3  ={res['v3_test'].get(k)}\n{'':32s} v3.1={res['v3_1_test'][k]}")
