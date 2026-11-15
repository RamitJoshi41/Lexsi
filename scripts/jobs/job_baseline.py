# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=4.56", "accelerate", "pandas", "numpy", "huggingface_hub", "pyarrow"]
# ///
"""Job 02: batched generations + last-token hidden states for every prompt in codemix_v1.
Usage: hf jobs uv run --flavor l4x1 --secrets HF_TOKEN job_baseline.py <work_repo> <model_id>
"""
import os, sys, time, json
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download

REPO, MODEL_ID = sys.argv[1], sys.argv[2]
DATA = sys.argv[3] if len(sys.argv) > 3 else "codemix_v1.jsonl"
TAG = MODEL_ID.split("/")[-1]
BS, MAX_NEW = 16, 160
torch.set_grad_enabled(False)
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

df = pd.read_json(hf_hub_download(REPO, DATA, repo_type="dataset"), lines=True)
log("rows", len(df))
tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
L = model.config.num_hidden_layers; log("loaded", MODEL_ID, L)

prompts = [tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False) for t in df.text]
order = np.argsort([len(p) for p in prompts])
responses = [None] * len(df)
H = np.zeros((len(df), L, model.config.hidden_size), dtype=np.float16)

for b in range(0, len(order), BS):
    idx = order[b:b + BS]
    enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    out = model.model(**enc, output_hidden_states=True)
    hs = torch.stack([h[:, -1].float() for h in out.hidden_states[1:]], 1).cpu().numpy()   # [B, L, d]
    H[idx] = hs.astype(np.float16)
    gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
    for j, i in enumerate(idx):
        responses[i] = tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    if (b // BS) % 5 == 0: log(f"batch {b//BS+1}/{(len(order)+BS-1)//BS}")

df["model"] = MODEL_ID; df["response"] = responses
os.makedirs("out", exist_ok=True)
df.to_json(f"out/gen_{TAG}.jsonl", orient="records", lines=True, force_ascii=False)
np.save(f"out/hidden_{TAG}.npy", H)
api = HfApi()
api.upload_folder(folder_path="out", path_in_repo=f"runs/baseline/{TAG}", repo_id=REPO, repo_type="dataset")
log("DONE")
