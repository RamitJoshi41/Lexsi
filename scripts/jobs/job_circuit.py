# /// script
# requires-python = ">=3.11"
# dependencies = ["circuitkit>=0.2.4", "torch", "transformers>=4.56", "accelerate", "pandas", "numpy", "huggingface_hub"]
# ///
"""Job 21 (Day 4): CircuitKIT refusal circuits of Tiny Aya per language/script.

CircuitKIT `multilingual_refusal` task (EAP-IG, node level): clean = harmful request, corrupt = minimal-pair
harmless request; metric = logit diff between the refusal opener and the compliance opener at the first reply token.
Languages: en (bundled pairs), hi (Devanagari), hi-Latn (romanized Hindi), pa-Latn (romanized Punjabi);
non-English pairs are ours (data/minimal_pairs/pairs_<lang>.csv, template x noun swaps).
Outputs per language: node scores, top-k node sets, per-layer attribution mass; plus Jaccard overlaps between
languages. Usage: job_circuit.py <work_repo> <model_id> [n_examples]"""
import os, sys, time, json, traceback
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

REPO, MODEL_ID = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 64
TAG = MODEL_ID.split("/")[-1]
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
import circuitkit as ck
log("circuitkit", getattr(ck, "__version__", "?"))

LANGS = {
    "en": {"language": "en"},
    "hi": {"language": "hi", "pairs_path": hf_hub_download(REPO, "data/minimal_pairs/pairs_hi.csv", repo_type="dataset")},
    "hi-Latn": {"language": "hi-Latn", "pairs_path": hf_hub_download(REPO, "data/minimal_pairs/pairs_hi-Latn.csv", repo_type="dataset"),
                "refuse": "Main", "comply": "Zaroor"},
    "pa-Latn": {"language": "pa-Latn", "pairs_path": hf_hub_download(REPO, "data/minimal_pairs/pairs_pa-Latn.csv", repo_type="dataset"),
                "refuse": "Main", "comply": "Zaroor"},
}
model = ck.load_model(MODEL_ID, dtype="bfloat16")
log("model loaded", model.cfg.n_layers, "layers", model.cfg.n_heads, "heads")
os.makedirs("out", exist_ok=True); api = HfApi()
res = {}
for lang, dp in LANGS.items():
    try:
        circ = ck.discover(model, "multilingual_refusal", algorithm="eap-ig", level="node", n_examples=N, batch_size=1,
                           sparsity=0.9, scope="both", output_path=f"out/circuit_{lang}.pt", ig_steps=5, signed=True,
                           data_params={"num_examples": N, "batch_size": 1, **dp})
        sc = {k: float(v) for k, v in (circ.scores or {}).items()}
        signed = {k: float(v) for k, v in (getattr(circ, "signed_scores", None) or {}).items()}
        res[lang] = {"n_nodes": len(circ), "scores": sc, "signed": signed, "top20": list(circ.top_nodes(20).keys()),
                     "meta": {k: str(v) for k, v in (getattr(circ, "metadata", {}) or {}).items()}}
        log(lang, "nodes", len(circ), "top10", list(circ.top_nodes(10).keys()))
    except Exception as e:
        traceback.print_exc(); res[lang] = {"error": repr(e)}
        log(lang, "FAILED", repr(e)[:300])
    import gc, torch; gc.collect(); torch.cuda.empty_cache()

def layer_of(n):
    import re
    m = re.search(r"(?:a|m|blocks\.|L)(\d+)", n); return int(m.group(1)) if m else -1
summary = {}
for lang, r in res.items():
    if "scores" not in r: continue
    s = r["scores"]; tot = sum(abs(v) for v in s.values()) or 1.0
    by_layer = {}
    for n, v in s.items(): by_layer[layer_of(n)] = by_layer.get(layer_of(n), 0) + abs(v) / tot
    summary[lang] = {"by_layer": dict(sorted(by_layer.items())), "top20": r["top20"]}
def topk(lang, k):
    s = res[lang]["scores"]; return set(sorted(s, key=lambda n: -abs(s[n]))[:k])
ok = [l for l in res if "scores" in res[l]]
jac = {}
for k in (20, 50, 100):
    for i, a in enumerate(ok):
        for b in ok[i + 1:]:
            A, B = topk(a, k), topk(b, k); jac[f"{a}|{b}|top{k}"] = len(A & B) / max(1, len(A | B))
summary["jaccard"] = jac
log("jaccard", json.dumps(jac, indent=0))
json.dump(res, open("out/circuit_scores.json", "w"), ensure_ascii=False)
json.dump(summary, open("out/circuit_summary.json", "w"), ensure_ascii=False, indent=1)
for f in os.listdir("out"):
    if f.endswith((".json", ".pt")):
        api.upload_file(path_or_fileobj=f"out/{f}", path_in_repo=f"runs/circuit/{TAG}/{f}", repo_id=REPO, repo_type="dataset")
log("DONE")
