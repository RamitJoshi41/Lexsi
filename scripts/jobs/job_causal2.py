# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=4.56", "accelerate", "pandas", "numpy", "scikit-learn", "huggingface_hub", "pyarrow"]
# ///
"""Job 24 (Day 4): causal test v2 — direction-level interventions at ALL positions (prefill + generated tokens),
sized per prompt by its own English twin (no tuned alpha).

v1 (job_causal.py) patched only the last prompt token at one layer: even the full English state moved refusal
only 17% -> ~24% at layers 6-18, so refusal is not read from that single position. Here, at layer L, every
position gets the same added vector (like steering), with the size taken from the English twin's last-token gap:

  c_r = (h_en - h_cm) . r      c_p = (h_en - h_cm) . p      (r refusal dir, p harm-probe dir orthogonalised to r)

  code-mixed harmful:
    refusal_twin   + c_r * r          (give it the refusal component English has)
    harm_twin      + c_p * p          (give it the harm component English has)
    harm_matched   + |c_r| * p        (harm direction at the refusal gap's size: rules out "any big push works")
    random_matched + |c_r| * q        (random unit q ⊥ r, same size)
  English harmful (necessity):
    en_ablate_r    h - (h . r) r      (directional ablation of r at every position, Arditi et al. 2024)
    en_ablate_p    h - (h . p) p      (same for the harm-probe direction)
    en_minus_gap   - c_r(own twin) * r   (remove the English-vs-code-mixed refusal gap)
  code-mixed harmless (specificity): refusal_avg  + mean(c_r) * r   (does the same push make it refuse harmless?)
Usage: job_causal2.py <work_repo> <model_id> <layers comma>"""
import os, sys, time, json
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download

REPO, MODEL_ID = sys.argv[1], sys.argv[2]; LAYERS = [int(x) for x in sys.argv[3].split(",")]
TAG = MODEL_ID.split("/")[-1]; BS, MAX_NEW = 16, 64
torch.set_grad_enabled(False); t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
dl = lambda p: hf_hub_download(REPO, p, repo_type="dataset")
df = pd.read_json(dl("codemix_v1.jsonl"), lines=True); fit = set(json.load(open(dl("fit_ids.json"))))
H = np.load(dl(f"runs/baseline/{TAG}/hidden_{TAG}.npy")).astype(np.float32)
y = df.label.values; isfit = df.id.isin(fit).values; en = (df.script == "en").values
R, P, Q = {}, {}, {}
rng = np.random.default_rng(0)
for l in LAYERS:
    v = H[isfit & en & (y == 1), l].mean(0) - H[isfit & en & (y == 0), l].mean(0); r = v / np.linalg.norm(v)
    X = H[isfit & en, l]; mu, sd = X.mean(0), X.std(0) + 1e-6
    w = LogisticRegression(max_iter=3000, C=0.05).fit((X - mu) / sd, y[isfit & en]).coef_[0] / sd
    p = w / np.linalg.norm(w); p = p - (p @ r) * r; p /= np.linalg.norm(p)
    q = rng.standard_normal(len(r)); q -= (q @ r) * r; q /= np.linalg.norm(q)
    R[l], P[l], Q[l] = r, p, q

# last-token states straight from the stored baseline pass (same chat template, bf16): output of layer l = H[:, l]
ho = ~isfit
cm_h = df[ho & ~en & (y == 1)].reset_index(); cm_s = df[ho & ~en & (y == 0)].reset_index(); en_h = df[ho & en & (y == 1)].reset_index()
en_idx = df[en].reset_index().set_index("id")["index"]
def gaps(rows, l):
    hc = H[rows["index"].values, l]; he = H[en_idx[rows.id].values, l]
    return (he - hc) @ R[l], (he - hc) @ P[l]
# English necessity: code-mixed mean state of the same id
def en_gap(l):
    cmm = {i: H[df.index[(df.id == i) & ~en].values, l].mean(0) for i in en_h.id}
    return np.array([(cmm[i] - H[j, l]) @ R[l] for i, j in zip(en_h.id, en_h["index"])])

tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
chat = lambda t: tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False)
st = {"add": None, "ablate": None}
def hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if st["ablate"] is not None:
        u = st["ablate"]; h = h - (h.float() @ u)[..., None].to(h.dtype) * u.to(h.dtype)
    if st["add"] is not None: h = h + st["add"][:, None, :].to(h.dtype)
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

def run(rows, l=None, add=None, ablate=None):
    texts = rows.text.tolist(); ps = [chat(t) for t in texts]; order = np.argsort([len(p) for p in ps]); resp = [None] * len(ps)
    hk = model.model.layers[l].register_forward_hook(hook) if l is not None else None
    st["ablate"] = torch.tensor(ablate, dtype=torch.float32, device="cuda") if ablate is not None else None
    for b in range(0, len(order), BS):
        idx = order[b:b + BS]
        st["add"] = torch.tensor(add[idx], dtype=torch.float32, device="cuda") if add is not None else None
        enc = tok([ps[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx): resp[i] = tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    if hk: hk.remove()
    st["add"] = st["ablate"] = None; return resp

out = []
def add_rows(rows, resp, cond, l, extra=None):
    f = rows[["id", "script", "label", "category", "text"]].copy(); f["response"] = resp; f["cond"] = cond; f["layer"] = l
    if extra is not None: f["coef"] = extra
    out.append(f)
add_rows(cm_h, run(cm_h), "none", -1); add_rows(en_h, run(en_h), "none", -1); add_rows(cm_s, run(cm_s), "none", -1)
log("baselines")
for l in LAYERS:
    cr, cp = gaps(cm_h, l); r, p, q = R[l], P[l], Q[l]
    log(f"L{l}: c_r mean {cr.mean():.2f} (sd {cr.std():.2f}), c_p mean {cp.mean():.2f} (sd {cp.std():.2f})")
    add_rows(cm_h, run(cm_h, l, add=cr[:, None] * r[None]), "refusal_twin", l, cr)
    add_rows(cm_h, run(cm_h, l, add=cp[:, None] * p[None]), "harm_twin", l, cp)
    add_rows(cm_h, run(cm_h, l, add=np.abs(cr)[:, None] * p[None]), "harm_matched", l, np.abs(cr))
    add_rows(cm_h, run(cm_h, l, add=np.abs(cr)[:, None] * q[None]), "random_matched", l, np.abs(cr))
    add_rows(en_h, run(en_h, l, ablate=r), "en_ablate_r", l)
    add_rows(en_h, run(en_h, l, ablate=p), "en_ablate_p", l)
    eg = en_gap(l); add_rows(en_h, run(en_h, l, add=eg[:, None] * r[None]), "en_minus_gap", l, eg)
    add_rows(cm_s, run(cm_s, l, add=np.full((len(cm_s), 1), cr.mean()) * r[None]), "harmless_refusal_avg", l)
    log(f"L{l} done")
D = pd.concat(out, ignore_index=True); os.makedirs("out", exist_ok=True)
path = f"runs/causal/{TAG}/gen_causal2.jsonl"; D.to_json("out/c2.jsonl", orient="records", lines=True, force_ascii=False)
HfApi().upload_file(path_or_fileobj="out/c2.jsonl", path_in_repo=path, repo_id=REPO, repo_type="dataset"); log("DONE", len(D), path)
