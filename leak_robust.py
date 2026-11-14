"""Headline re-scored on held-out prompt ids with NO near-duplicate in the guard's fitting data
(drop every id with any script flagged by MinHash >= 0.5 or LaBSE cosine >= 0.85). Cluster bootstrap as bootstrap_ci.py."""
import json, numpy as np, pandas as pd, importlib.util
spec = importlib.util.spec_from_file_location("b", "bootstrap_ci.py")
src = open("bootstrap_ci.py").read().split("rng = np.random")[0]; ns = {}; exec(src, ns)
load, NONEN = ns["load"], ns["NONEN"]
ex = pd.read_json("results/leak_check_examples.jsonl", lines=True); flagged = set(ex[ex.src == "benchmark_heldout"].id)
runs = {"control": "gen_control_a0", "cast_all": "gen_safetune_cast_all_a1", "gate_v3": "gen_g3_14"}
D = {k: load(v) for k, v in runs.items()}
rng = np.random.default_rng(0); B = 10000; out = {"dropped_ids": sorted(flagged)}
for name, keep in [("all held-out", lambda d: d), ("near-duplicates removed", lambda d: d[~d.id.isin(flagged)])]:
    E = {k: keep(v) for k, v in D.items()}; ids = E["control"].id.unique(); n = len(ids); dr = rng.integers(0, n, (B, n))
    def pid(d):
        x = d[(d.label == 1) & d.script.isin(NONEN)].groupby("id").usable.agg(["sum", "count"]).reindex(ids).fillna(0); return x["sum"].values, x["count"].values
    r = {"n_ids": int(n), "n_harmful_ids": int(E["control"][E["control"].label == 1].id.nunique())}
    for k, d in E.items():
        s, c = pid(d); r[k] = round(float(s.sum() / c.sum()), 3)
    (sa, ca), (sb, cb) = pid(E["gate_v3"]), pid(E["cast_all"])
    x = sa[dr].sum(1) / ca[dr].sum(1) - sb[dr].sum(1) / cb[dr].sum(1)
    r["gate_v3 - cast_all"] = [round(float(np.percentile(x, q)), 3) for q in (2.5, 50, 97.5)]
    out[name] = r; print(name, r)
json.dump(out, open("results/leak_robust.json", "w"), indent=1)
