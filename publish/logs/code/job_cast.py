# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=4.56", "accelerate", "pandas", "numpy", "huggingface_hub", "pyarrow", "safetune>=0.1.13"]
# ///
"""Job 07: SafeTune CASTTrainer baseline (conditional activation steering) on the same held-out prompts.
Usage: job_cast.py <work_repo> <model_id> <calib: en|all> [alpha]"""
import os, sys, time, json
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download
from safetune.runner.steer import CASTTrainer

REPO, MODEL_ID, CALIB = sys.argv[1], sys.argv[2], sys.argv[3]
ALPHA = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
TAG = MODEL_ID.split("/")[-1]; BS, MAX_NEW = 16, 160
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
dl = lambda p: hf_hub_download(REPO, p, repo_type="dataset")
df = pd.read_json(dl("codemix_v1.jsonl"), lines=True); fit = set(json.load(open(dl("fit_ids.json"))))
cal = df[df.id.isin(fit) & ((df.script == "en") if CALIB == "en" else True)]
harmful, harmless = cal[cal.label == 1].text.tolist(), cal[cal.label == 0].text.tolist()
ev = df[~df.id.isin(fit)].reset_index(drop=True)

tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
trainer = CASTTrainer(model, tok, alpha=ALPHA, per_prompt_gate=True)
cast, _ = trainer.calibrate(harmful=harmful, harmless=harmless)
log("CAST calibrated on", CALIB, len(harmful), len(harmless), "| threshold", getattr(cast, "threshold", None),
    "| condition layer", getattr(getattr(cast, "condition", None), "layer", None))

prompts = [tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False) for t in ev.text]
order = np.argsort([len(p) for p in prompts]); resp = [None] * len(ev)
with torch.no_grad():
    for b in range(0, len(order), BS):
        idx = order[b:b + BS]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = cast.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                            max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            resp[i] = tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
name = f"safetune_cast_{CALIB}_a{ALPHA:g}"
out = ev.copy(); out["response"] = resp; out["config"] = name
path = f"runs/steer/{TAG}/gen_{name}.jsonl"; os.makedirs(os.path.dirname("out/" + path), exist_ok=True)
out.to_json("out/" + path, orient="records", lines=True, force_ascii=False)
HfApi().upload_file(path_or_fileobj="out/" + path, path_in_repo=path, repo_id=REPO, repo_type="dataset")
log("DONE", path)
