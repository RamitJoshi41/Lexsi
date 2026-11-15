# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=4.56", "accelerate", "peft", "pandas", "numpy", "huggingface_hub", "pyarrow"]
# ///
"""Job 30 (Day 4): generate with base + merged LoRA adapter (plain model, no hooks) on any prompt files of the work repo.
Usage: job_eval_adapter.py <work_repo> <model_id> <adapter_repo> <run_name> <file.jsonl> [...]
-> runs/lora/<tag>/gen_<run_name>_<file stem>.jsonl"""
import os, sys, time
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, hf_hub_download
REPO, MODEL_ID, ADAPTER, NAME = sys.argv[1:5]; FILES = sys.argv[5:]; TAG = MODEL_ID.split("/")[-1]
torch.set_grad_enabled(False); t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda"), ADAPTER).merge_and_unload().eval()
api = HfApi(); os.makedirs("out", exist_ok=True)
for f in FILES:
    df = pd.read_json(hf_hub_download(REPO, f, repo_type="dataset"), lines=True).reset_index(drop=True)
    ps = [tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False) for t in df.text]
    order = np.argsort([len(p) for p in ps]); resp = [None] * len(ps)
    for b in range(0, len(order), 24):
        idx = order[b:b + 24]
        enc = tok([ps[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=160, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx): resp[i] = tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    out = df.copy(); out["response"] = resp; out["config"] = NAME
    p = f"runs/lora/{TAG}/gen_{NAME}_{os.path.basename(f).replace('.jsonl', '')}.jsonl"
    out.to_json("out/x.jsonl", orient="records", lines=True, force_ascii=False)
    api.upload_file(path_or_fileobj="out/x.jsonl", path_in_repo=p, repo_id=REPO, repo_type="dataset"); log("done", p, len(out))
log("DONE")
