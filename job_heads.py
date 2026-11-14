# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=4.56", "accelerate", "pandas", "numpy", "huggingface_hub", "pyarrow"]
# ///
"""Job 25 (Day 4): circuit-level test of the CircuitKIT refusal heads.

CircuitKIT (EAP-IG, multilingual_refusal) put attention heads A14.13, A16.2, A20.4 in the top-20 refusal nodes for
English and Devanagari Hindi (and A16.2, A20.4 for romanized Hindi) but none of them for romanized Punjabi.
Here we intervene on those heads' outputs (the per-head slice of the o_proj input) at every position:
  English harmful   : ablate the 3 heads (zero) | ablate 3 random heads at the same layers | ablate the 3 top
                      pa-Latn-only heads (A31.11, A25.3, A29.13) as a control
  non-EN harmful    : scale the 3 heads x2, x4 | scale 3 random same-layer heads x4
  non-EN harmless   : scale the 3 heads x4 (over-refusal check)
Usage: job_heads.py <work_repo> <model_id>"""
import os, sys, time, json
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download
REPO, MODEL_ID = sys.argv[1], sys.argv[2]; TAG = MODEL_ID.split("/")[-1]; BS, MAX_NEW = 16, 64
torch.set_grad_enabled(False); t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
dl = lambda p: hf_hub_download(REPO, p, repo_type="dataset")
df = pd.read_json(dl("codemix_v1.jsonl"), lines=True); fit = set(json.load(open(dl("fit_ids.json"))))
ho = df[~df.id.isin(fit)]
en_h = ho[(ho.script == "en") & (ho.label == 1)]; cm_h = ho[(ho.script != "en") & (ho.label == 1)]; cm_s = ho[(ho.script != "en") & (ho.label == 0)]
tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
cfg = model.config; NH = cfg.num_attention_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH
log("heads", NH, "head_dim", HD)
REF = [(14, 13), (16, 2), (20, 4)]; PA = [(31, 11), (25, 3), (29, 13)]
rng = np.random.default_rng(0)
RND = [(l, int(rng.choice([h for h in range(NH) if (l, h) not in REF]))) for l, _ in REF]
log("random control heads", RND)
cur = {}
def make_pre(layer):
    def pre(mod, args):
        x = args[0]
        heads = cur.get(layer)
        if not heads: return None
        x = x.clone()
        for h, s in heads: x[..., h * HD:(h + 1) * HD] *= s
        return (x,) + tuple(args[1:])
    return pre
for l in range(cfg.num_hidden_layers): model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(make_pre(l))
chat = lambda t: tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False)
def run(rows, heads, scale):
    cur.clear()
    for l, h in heads: cur.setdefault(l, []).append((h, scale))
    ps = [chat(t) for t in rows.text]; order = np.argsort([len(p) for p in ps]); resp = [None] * len(ps)
    for b in range(0, len(order), BS):
        idx = order[b:b + BS]
        enc = tok([ps[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx): resp[i] = tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    cur.clear(); return resp
out = []
def add(rows, cond, heads, scale):
    f = rows[["id", "script", "label", "category", "text"]].copy(); f["response"] = run(rows, heads, scale); f["cond"] = cond; out.append(f); log(cond, len(f))
add(en_h, "none", [], 1); add(cm_h, "none", [], 1); add(cm_s, "none", [], 1)
add(en_h, "ablate_ref_heads", REF, 0.0); add(en_h, "ablate_random_heads", RND, 0.0); add(en_h, "ablate_pa_heads", PA, 0.0)
add(cm_h, "amp2_ref_heads", REF, 2.0); add(cm_h, "amp4_ref_heads", REF, 4.0); add(cm_h, "amp4_random_heads", RND, 4.0)
add(cm_s, "amp4_ref_heads", REF, 4.0)
D = pd.concat(out, ignore_index=True); D["random_heads"] = json.dumps(RND)
os.makedirs("out", exist_ok=True); p = f"runs/circuit/{TAG}/gen_heads.jsonl"; D.to_json("out/h.jsonl", orient="records", lines=True, force_ascii=False)
HfApi().upload_file(path_or_fileobj="out/h.jsonl", path_in_repo=p, repo_id=REPO, repo_type="dataset"); log("DONE", len(D), p)
