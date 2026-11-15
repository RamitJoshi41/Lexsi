# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=4.56", "accelerate", "pandas", "numpy", "scikit-learn", "huggingface_hub", "pyarrow"]
# ///
"""Job 20 (Day 4): causal test of "knows, but doesn't refuse" by component-wise activation patching.

For each held-out harmful prompt in a non-English script we also run its parallel English prompt.
At one layer L, on the last prompt token only (prefill), we move the code-mixed residual h_cm toward the
English residual h_en along chosen directions and let the model generate:

  full        h_cm := h_en                                   (positive control: everything English)
  refusal     h_cm += ((h_en - h_cm) . r) r                   (only the refusal-direction component)
  harm        h_cm += ((h_en - h_cm) . p) p                   (only the harm-probe component, p orthogonalised to r)
  allbutref   h_cm := h_en - ((h_en - h_cm) . r) r            (everything English EXCEPT the refusal component)
  random      h_cm += ((h_en - h_cm) . q) q                   (random unit direction q, control)

and the necessity test on English prompts:
  en_ablate   h_en += ((h_cm - h_en) . r) r                   (give English the code-mixed refusal component)

r = unit diff-in-means refusal direction (English fit prompts) at L; p = unit weight of an English-only
logistic harmfulness probe at L (the "knows" probe of Exp 2). Directions come from the fit split only;
patched prompts are held-out. Outputs are judged afterwards with job_judge4.py.
Usage: job_causal.py <work_repo> <model_id> <layers comma> [max_new]"""
import os, sys, time, json
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download

REPO, MODEL_ID = sys.argv[1], sys.argv[2]
LAYERS = [int(x) for x in sys.argv[3].split(",")]
MAX_NEW = int(sys.argv[4]) if len(sys.argv) > 4 else 64
TAG = MODEL_ID.split("/")[-1]; BS = 16
torch.set_grad_enabled(False); t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
dl = lambda p: hf_hub_download(REPO, p, repo_type="dataset")

df = pd.read_json(dl("codemix_v1.jsonl"), lines=True)
fit = set(json.load(open(dl("fit_ids.json"))))
H = np.load(dl(f"runs/baseline/{TAG}/hidden_{TAG}.npy")).astype(np.float32)
y = df.label.values; isfit = df.id.isin(fit).values; en = (df.script == "en").values

R, P = {}, {}
rng = np.random.default_rng(0)
Q = {}
for l in LAYERS:
    v = H[isfit & en & (y == 1), l].mean(0) - H[isfit & en & (y == 0), l].mean(0); r = v / np.linalg.norm(v)
    X = H[isfit & en, l]; mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(max_iter=3000, C=0.05).fit((X - mu) / sd, y[isfit & en])
    w = clf.coef_[0] / sd; w = w / np.linalg.norm(w)
    cos_rp = float(w @ r); p = w - cos_rp * r; p /= np.linalg.norm(p)
    q = rng.standard_normal(len(r)); q -= (q @ r) * r; q /= np.linalg.norm(q)
    R[l], P[l], Q[l] = r, p, q
    # how well does the English probe 'know' on held-out code-mixed harmful vs harmless at this layer?
    ho = ~isfit & ~en
    acc = float(((clf.predict((H[ho, l] - mu) / sd)) == y[ho]).mean())
    log(f"L{l}: cos(probe, refusal)={cos_rp:.3f}  EN-probe acc on held-out non-EN={acc:.3f}")

tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
layers = model.model.layers
chat = lambda t: tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False)

ho_h = df[~isfit & (y == 1)]
en_txt = ho_h[ho_h.script == "en"].set_index("id").text
cm = ho_h[ho_h.script != "en"].reset_index(drop=True)
ens = ho_h[ho_h.script == "en"].reset_index(drop=True)
log("held-out harmful: non-EN", len(cm), "EN", len(ens))

def last_states(texts):
    """residual (output of every patched layer) at the last prompt token, fp32 numpy [n, len(LAYERS), d]"""
    out = np.zeros((len(texts), len(LAYERS), model.config.hidden_size), np.float32)
    ps = [chat(t) for t in texts]
    for b in range(0, len(ps), BS):
        enc = tok(ps[b:b + BS], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        hs = model.model(**enc, output_hidden_states=True).hidden_states
        out[b:b + BS] = torch.stack([hs[l + 1][:, -1].float() for l in LAYERS], 1).cpu().numpy()
    return out

S_cm = last_states(cm.text.tolist())                         # code-mixed states
S_en_for_cm = last_states([en_txt[i] for i in cm.id])        # parallel English states
S_en = last_states(ens.text.tolist())
# for EN necessity: mean code-mixed state of the same id (average over the 4 scripts)
cm_mean = {i: S_cm[(cm.id == i).values].mean(0) for i in ens.id}
S_cm_for_en = np.stack([cm_mean[i] for i in ens.id])
log("cached states")

state = {"delta": None}
def make_hook():
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1 and state["delta"] is not None:   # prefill only, last prompt token
            h = h.clone(); h[:, -1] = h[:, -1] + state["delta"].to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return hook

def delta(kind, s_src, s_tgt, li, l):
    """change to add to source state s_src (n,d) to move it toward target s_tgt along `kind`."""
    d = s_tgt - s_src
    proj = lambda u: (d @ u)[:, None] * u[None]
    if kind in ("full",): return d
    if kind in ("refusal", "en_ablate"): return proj(R[l])
    if kind == "harm": return proj(P[l])
    if kind == "random": return proj(Q[l])
    if kind == "allbutref": return d - proj(R[l])
    if kind == "none": return np.zeros_like(d)
    raise ValueError(kind)

def run(texts, deltas, l):
    ps = [chat(t) for t in texts]; order = np.argsort([len(p) for p in ps]); resp = [None] * len(ps)
    h = layers[l].register_forward_hook(make_hook()) if l is not None else None
    for b in range(0, len(order), BS):
        idx = order[b:b + BS]
        state["delta"] = torch.tensor(deltas[idx], device="cuda") if deltas is not None else None
        enc = tok([ps[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx): resp[i] = tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    if h: h.remove()
    state["delta"] = None
    return resp

rows = []
def add(frame, resp, cond, l):
    f = frame[["id", "script", "label", "category", "text"]].copy(); f["response"] = resp; f["cond"] = cond; f["layer"] = l; rows.append(f)

add(cm, run(cm.text.tolist(), None, None), "none", -1)
add(ens, run(ens.text.tolist(), None, None), "none", -1)
log("baselines done")
for li, l in enumerate(LAYERS):
    for kind in ["full", "refusal", "harm", "allbutref", "random"]:
        D = delta(kind, S_cm[:, li], S_en_for_cm[:, li], li, l)
        add(cm, run(cm.text.tolist(), D, l), kind, l)
    D = delta("en_ablate", S_en[:, li], S_cm_for_en[:, li], li, l)
    add(ens, run(ens.text.tolist(), D, l), "en_ablate", l)
    # magnitude of the gap along r and p (in units of the English harmful-harmless separation)
    log(f"L{l} done; mean |gap| along r={np.abs((S_en_for_cm[:, li]-S_cm[:, li]) @ R[l]).mean():.2f} "
        f"along p={np.abs((S_en_for_cm[:, li]-S_cm[:, li]) @ P[l]).mean():.2f}")

out = pd.concat(rows, ignore_index=True)
os.makedirs("out", exist_ok=True); path = f"runs/causal/{TAG}/gen_causal.jsonl"
os.makedirs(os.path.dirname("out/" + path), exist_ok=True)
out.to_json("out/" + path, orient="records", lines=True, force_ascii=False)
HfApi().upload_file(path_or_fileobj="out/" + path, path_in_repo=path, repo_id=REPO, repo_type="dataset")
log("DONE", len(out), "rows ->", path)
