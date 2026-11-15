# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=4.56", "accelerate", "pandas", "numpy", "scikit-learn", "huggingface_hub", "pyarrow"]
# ///
"""Job 04: probe-gated, layer-targeted refusal steering on held-out prompts.
Usage: hf jobs uv run --flavor a10g-small --secrets HF_TOKEN job_steer.py <work_repo> <model_id> <configs>
configs: semicolon list of name:gate:add_layer:alpha  (gate = 'gated' | 'always' )
"""
import os, sys, time, json
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download

REPO, MODEL_ID, CONFIGS = sys.argv[1], sys.argv[2], sys.argv[3]
TAG = MODEL_ID.split("/")[-1]
GATE_L, GATE_T, BS, MAX_NEW = 8, 0.7, 16, 160
torch.set_grad_enabled(False)
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
dl = lambda p: hf_hub_download(REPO, p, repo_type="dataset")

df = pd.read_json(dl("codemix_v1.jsonl"), lines=True)
fit = set(json.load(open(dl("fit_ids.json"))))
H = np.load(dl(f"runs/baseline/{TAG}/hidden_{TAG}.npy")).astype(np.float32)
y = df.label.values; isfit = df.id.isin(fit).values; en = (df.script == "en").values

# gate probe: layer GATE_L, trained on fit ids, all scripts
X = H[:, GATE_L]; mu = X[isfit].mean(0); sd = X[isfit].std(0) + 1e-6
clf = LogisticRegression(max_iter=3000, C=0.05).fit((X[isfit] - mu) / sd, y[isfit])
w_raw = clf.coef_[0] / sd; b_raw = clf.intercept_[0] - (clf.coef_[0] * mu / sd).sum()
# refusal vectors (diff in means, English fit ids only) for every layer
V = {l: H[isfit & en & (y == 1), l].mean(0) - H[isfit & en & (y == 0), l].mean(0) for l in range(H.shape[1])}
if os.environ.get("GATE_FILE"):   # gate v2: retrained with everyday benign negatives
    g = np.load(dl(os.environ["GATE_FILE"]))
    w_raw, b_raw, GATE_L, GATE_T = g["w"], float(g["b"]), int(g["layer"]), float(g["thr"])
    log("loaded gate", os.environ["GATE_FILE"], "layer", GATE_L, "thr", GATE_T)
log("probe + vectors ready", {l: round(float(np.linalg.norm(V[l])), 1) for l in (8, 14, 18, 23)})

ev = df[~isfit].reset_index(drop=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
layers = model.model.layers
W = torch.tensor(w_raw, dtype=torch.float32, device="cuda"); B = float(b_raw)

state = {"mask": None, "p": None}
def gate_hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if h.shape[1] > 1:   # prefill: decide per example from the last prompt token
        p = torch.sigmoid(h[:, -1].float() @ W + B)
        state["p"] = p.detach().cpu().numpy()
        state["mask"] = (p > GATE_T).to(h.dtype) if state["mode"] == "gated" else torch.ones_like(p, dtype=h.dtype)
    return out
def make_add_hook(vec, alpha):
    v = torch.tensor(vec, dtype=torch.bfloat16, device="cuda")
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + alpha * state["mask"][:, None, None] * v
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return hook

prompts = [tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False) for t in ev.text]
order = np.argsort([len(p) for p in prompts])
api = HfApi(); os.makedirs("out", exist_ok=True)
for cfg in CONFIGS.split(";"):
    name, mode, add_l, alpha = cfg.split(":"); add_l = [int(x) for x in add_l.split(",")]; alpha = float(alpha)
    state["mode"] = mode
    hs = [layers[GATE_L].register_forward_hook(gate_hook)] + [layers[l].register_forward_hook(make_add_hook(V[l], alpha)) for l in add_l]
    resp, gate_p = [None] * len(ev), [None] * len(ev)
    for b in range(0, len(order), BS):
        idx = order[b:b + BS]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            resp[i] = tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True); gate_p[i] = float(state["p"][j])
    for h in hs: h.remove()
    out = ev.copy(); out["response"] = resp; out["gate_p"] = gate_p; out["config"] = name
    path = f"runs/steer/{TAG}/gen_{name}.jsonl"
    os.makedirs(os.path.dirname("out/" + path), exist_ok=True)
    out.to_json("out/" + path, orient="records", lines=True, force_ascii=False)
    api.upload_file(path_or_fileobj="out/" + path, path_in_repo=path, repo_id=REPO, repo_type="dataset")
    fired = out.groupby(["script", "label"]).gate_p.apply(lambda s: (s > GATE_T).mean()).unstack()
    log(name, "done; gate fire rate (label 0/1):\n", fired.round(2).to_string())
log("DONE")
