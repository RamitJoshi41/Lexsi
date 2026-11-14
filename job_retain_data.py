# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=4.56", "accelerate", "pandas", "numpy", "datasets", "huggingface_hub", "pyarrow"]
# ///
"""Job 29 (Day 4): capability-retention data for distillation v2 (self-distillation of the BASE model).
- 400 multiple-choice questions from cais/mmlu `auxiliary_train` (never the test split used by AuditKIT), in the same
  "Reply with only the letter" format, answered by the base model (kept as-is, right or wrong: we preserve behaviour,
  we do not teach answers).
- 300 more everyday benign prompts from the TRAIN pool (same exclusion of the 600 test prompts as gate v2/v3), not used in v1.
Usage: job_retain_data.py <work_repo> <model_id>"""
import os, sys, time, json
import numpy as np, pandas as pd, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download
REPO, MODEL_ID = sys.argv[1], sys.argv[2]; TAG = MODEL_ID.split("/")[-1]; BS = 16
torch.set_grad_enabled(False); t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
aux = load_dataset("cais/mmlu", "auxiliary_train", split="train")
ix = np.random.default_rng(0).choice(len(aux), 400, replace=False)
mc = []
for i in ix:
    q = aux[int(i)]; q = q.get("train", q)
    mc.append({"id": f"mmlu_aux_{i}", "label": 0, "script": "en", "src": "mmlu_aux", "max_new": 8,
               "text": q["question"] + "\n" + "\n".join(f"{'ABCD'[k]}. {c}" for k, c in enumerate(q["choices"])) + "\nReply with only the letter (A, B, C or D) of the correct option."})
bm = pd.read_json(hf_hub_download(REPO, f"runs/benign/{TAG}/benign_meta.jsonl", repo_type="dataset"), lines=True); bt = np.zeros(len(bm), bool)
for lang in bm.lang.unique():
    w = np.where(bm.lang == lang)[0]; bt[np.random.default_rng(0).choice(w, 200, replace=False)] = True
tr = bm[~bt].copy(); used = pd.concat([tr[tr.lang == lg].sample(150, random_state=0) for lg in sorted(tr.lang.unique())])
rest = tr.drop(used.index); ev = pd.concat([rest[rest.lang == lg].sample(100, random_state=1) for lg in sorted(rest.lang.unique())])
ev = [{"id": f"everyday_tr2_{k:04d}", "label": 0, "script": {"English": "en", "Hindi": "hi_deva", "Panjabi": "pa_guru"}[r.lang],
       "src": "everyday_train2", "max_new": 160, "text": r.text} for k, r in enumerate(ev.itertuples())]
D = pd.DataFrame(mc + ev); log("retain prompts", len(D), D.src.value_counts().to_dict())
tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
resp = [None] * len(D)
for mx in sorted(D.max_new.unique()):
    sub = np.where(D.max_new.values == mx)[0]
    ps = [tok.apply_chat_template([{"role": "user", "content": D.text[i]}], add_generation_prompt=True, tokenize=False) for i in sub]
    order = np.argsort([len(p) for p in ps])
    for b in range(0, len(order), BS):
        idx = order[b:b + BS]
        enc = tok([ps[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=int(mx), do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx): resp[sub[i]] = tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
D["response"] = resp
os.makedirs("out", exist_ok=True); p = f"runs/distill/{TAG}/retain.jsonl"; D.to_json("out/r.jsonl", orient="records", lines=True, force_ascii=False)
HfApi().upload_file(path_or_fileobj="out/r.jsonl", path_in_repo=p, repo_id=REPO, repo_type="dataset"); log("DONE", len(D), p)
