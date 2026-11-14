# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=4.56", "accelerate", "pandas", "numpy", "scikit-learn", "huggingface_hub", "pyarrow"]
# ///
"""Job 27 (Day 4): necessity test, directional ablation at EVERY layer (Arditi et al. 2024).
Single-layer ablation of the refusal direction barely moved English refusal (89% -> 83-87%): later layers rebuild it.
Here one unit direction u (fit at layer 14 from English fit prompts) is projected out of the residual stream after every
decoder layer, at every position, during prefill and generation:
  r  = refusal direction (diff in means, harmful - harmless)
  p  = English harm-probe direction, orthogonalised to r
  q  = random unit direction orthogonal to r
Held-out harmful: English (46) and non-English (184); held-out harmless non-English (192) for r only.
Usage: job_ablate_all.py <work_repo> <model_id>"""
import os, sys, time, json
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download
REPO, MODEL_ID = sys.argv[1], sys.argv[2]; TAG = MODEL_ID.split("/")[-1]; L0, BS, MAX_NEW = 14, 16, 64
torch.set_grad_enabled(False); t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
dl = lambda p: hf_hub_download(REPO, p, repo_type="dataset")
df = pd.read_json(dl("codemix_v1.jsonl"), lines=True); fit = set(json.load(open(dl("fit_ids.json"))))
H = np.load(dl(f"runs/baseline/{TAG}/hidden_{TAG}.npy")).astype(np.float32)
y = df.label.values; isfit = df.id.isin(fit).values; en = (df.script == "en").values
v = H[isfit & en & (y == 1), L0].mean(0) - H[isfit & en & (y == 0), L0].mean(0); r = v / np.linalg.norm(v)
X = H[isfit & en, L0]; mu, sd = X.mean(0), X.std(0) + 1e-6
w = LogisticRegression(max_iter=3000, C=0.05).fit((X - mu) / sd, y[isfit & en]).coef_[0] / sd
p = w / np.linalg.norm(w); p = p - (p @ r) * r; p /= np.linalg.norm(p)
q = np.random.default_rng(0).standard_normal(len(r)); q -= (q @ r) * r; q /= np.linalg.norm(q)
ho = df[~df.id.isin(fit)]
sets = {"en_h": ho[(ho.script == "en") & (ho.label == 1)], "cm_h": ho[(ho.script != "en") & (ho.label == 1)], "cm_s": ho[(ho.script != "en") & (ho.label == 0)]}
tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
st = {"u": None}
def hook(mod, inp, out):
    if st["u"] is None: return out
    h = out[0] if isinstance(out, tuple) else out; u = st["u"]
    h = h - (h.float() @ u)[..., None].to(h.dtype) * u.to(h.dtype)
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
for lyr in model.model.layers: lyr.register_forward_hook(hook)
chat = lambda t: tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False)
def run(rows, u):
    st["u"] = None if u is None else torch.tensor(u, dtype=torch.float32, device="cuda")
    ps = [chat(t) for t in rows.text]; order = np.argsort([len(x) for x in ps]); resp = [None] * len(ps)
    for b in range(0, len(order), BS):
        idx = order[b:b + BS]
        enc = tok([ps[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx): resp[i] = tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    st["u"] = None; return resp
out = []
for name, u in [("none", None), ("ablate_r_all", r), ("ablate_p_all", p), ("ablate_q_all", q)]:
    for sname, rows in sets.items():
        if sname == "cm_s" and name not in ("none", "ablate_r_all"): continue
        f = rows[["id", "script", "label", "category", "text"]].copy(); f["response"] = run(rows, u); f["cond"] = name; out.append(f); log(name, sname)
D = pd.concat(out, ignore_index=True); os.makedirs("out", exist_ok=True); path = f"runs/causal/{TAG}/gen_ablate_all.jsonl"
D.to_json("out/a.jsonl", orient="records", lines=True, force_ascii=False)
HfApi().upload_file(path_or_fileobj="out/a.jsonl", path_in_repo=path, repo_id=REPO, repo_type="dataset"); log("DONE", len(D))
