# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=4.56", "accelerate", "pandas", "numpy", "huggingface_hub", "pyarrow"]
# ///
"""Job 22 (Day 4): teacher data for distilling the guard into the weights (AlignTune SFT, LoRA).

TRAIN prompts only (no held-out prompt is ever used): codemix fit ids (all 5 scripts), aug_fit v1+v2 (perturbed fit
ids), the train half of new-category pairs (same split as gate v3), and 150/language of the everyday benign
train pool (the 600 everyday test prompts are excluded with the same seed as gate v2/v3).
Teacher:
  harmful  -> base model + refusal steering at layer 14 (always on, alpha 1, English diff-in-means vector), with a
              system prompt asking it to answer in the user's own language/script. (The student never sees it:
              context distillation.)
  harmless -> the base model itself, no system prompt (self-distillation keeps its normal behaviour).
Filtering (judge + LID + loop check) happens afterwards. Usage: job_distill_data.py <work_repo> <model_id>"""
import os, sys, time, json
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download

REPO, MODEL_ID = sys.argv[1], sys.argv[2]; TAG = MODEL_ID.split("/")[-1]
STEER_L, ALPHA, BS, MAX_NEW = 14, 1.0, 16, 160
SYS = ("Always reply in exactly the same language and the same script as the user's message. If the user writes "
       "Hindi or Punjabi in Latin letters (romanized), reply in romanized Hindi or Punjabi in Latin letters.")
torch.set_grad_enabled(False); t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
dl = lambda p: hf_hub_download(REPO, p, repo_type="dataset")

cm = pd.read_json(dl("codemix_v1.jsonl"), lines=True); fit = set(json.load(open(dl("fit_ids.json"))))
H = np.load(dl(f"runs/baseline/{TAG}/hidden_{TAG}.npy")).astype(np.float32)
y = cm.label.values; isfit = cm.id.isin(fit).values; en = (cm.script == "en").values
V = H[isfit & en & (y == 1), STEER_L].mean(0) - H[isfit & en & (y == 0), STEER_L].mean(0)

parts = [cm[isfit].assign(src="codemix_fit")]
for f in ["aug_fit_v1.jsonl", "aug_fit_v2.jsonl"]:
    a = pd.read_json(dl(f), lines=True); assert a.base_id.isin(fit).all(); parts.append(a.assign(src=f.replace(".jsonl", "")))
nc = pd.read_json(dl("newcats_v1.jsonl"), lines=True)
trp = {p for c in nc.category.unique() for p in sorted(nc[nc.category == c].pair.unique())[::2]}
parts.append(nc[nc.pair.isin(trp)].assign(src="newcats_train"))
bm = pd.read_json(dl(f"runs/benign/{TAG}/benign_meta.jsonl"), lines=True)
bt = np.zeros(len(bm), bool)
for lang in bm.lang.unique():
    ix = np.where(bm.lang == lang)[0]; bt[np.random.default_rng(0).choice(ix, 200, replace=False)] = True
tr = bm[~bt].copy(); tr = pd.concat([tr[tr.lang == lg].sample(150, random_state=0) for lg in sorted(tr.lang.unique())])
tr["script"] = tr.lang.map({"English": "en", "Hindi": "hi_deva", "Panjabi": "pa_guru"}); tr["label"] = 0
tr["id"] = ["everyday_tr_%04d" % i for i in range(len(tr))]; parts.append(tr.assign(src="everyday_train"))
D = pd.concat([p[["id", "label", "script", "text", "src"] + [c for c in ["category"] if c in p]] for p in parts], ignore_index=True)
D = D.drop_duplicates("text").reset_index(drop=True)
log("train prompts", len(D), D.groupby(["src", "label"]).size().to_dict())

tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
v = torch.tensor(V, dtype=torch.bfloat16, device="cuda"); state = {"on": False}
def hook(mod, inp, out):
    if not state["on"]: return out
    h = out[0] if isinstance(out, tuple) else out; h = h + ALPHA * v
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
model.model.layers[STEER_L].register_forward_hook(hook)

def gen(texts, system, steer):
    msgs = lambda t: ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": t}]
    ps = [tok.apply_chat_template(msgs(t), add_generation_prompt=True, tokenize=False) for t in texts]
    order = np.argsort([len(p) for p in ps]); out = [None] * len(ps); state["on"] = steer
    for b in range(0, len(order), BS):
        idx = order[b:b + BS]
        enc = tok([ps[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx): out[i] = tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    state["on"] = False; return out

D["response"] = None
h = D.label == 1
D.loc[h, "response"] = gen(D[h].text.tolist(), SYS, True); log("harmful teacher done", int(h.sum()))
D.loc[~h, "response"] = gen(D[~h].text.tolist(), None, False); log("harmless teacher done", int((~h).sum()))
os.makedirs("out", exist_ok=True); path = f"runs/distill/{TAG}/teacher.jsonl"
D.to_json("out/teacher.jsonl", orient="records", lines=True, force_ascii=False)
HfApi().upload_file(path_or_fileobj="out/teacher.jsonl", path_in_repo=path, repo_id=REPO, repo_type="dataset")
log("DONE ->", path)
