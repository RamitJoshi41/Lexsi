# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=4.56", "accelerate", "pandas", "numpy", "huggingface_hub", "pyarrow"]
# ///
"""Job 14: last-token hidden states (layers 6-18, chat template, same as baseline) for any jsonl with a `text` column.
Usage: job_feats.py <work_repo> <model_id> <name>=<path_in_repo> [...]  -> runs/feats/<tag>/<name>.npy + <name>.jsonl
A jsonl row with "dediacritize": true also gets an ASCII-folded copy (e.g. 'āndolān' -> 'andolan')."""
import os, sys, time, unicodedata
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download
REPO, MODEL_ID = sys.argv[1], sys.argv[2]; TAG = MODEL_ID.split("/")[-1]; KEEP = list(range(6, 19))
torch.set_grad_enabled(False); t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
fold = lambda t: unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
os.makedirs("out", exist_ok=True); api = HfApi()
for spec in sys.argv[3:]:
    name, path = spec.split("=", 1)
    df = pd.read_json(hf_hub_download(REPO, path, repo_type="dataset"), lines=True)
    if "dediacritize" in df:
        extra = df[df.dediacritize.fillna(False)].copy(); extra["text"] = extra.text.map(fold); extra["id"] = extra.id + "_ascii"
        df = pd.concat([df, extra], ignore_index=True)
    prompts = [tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False) for t in df.text]
    order = np.argsort([len(p) for p in prompts]); H = np.zeros((len(df), len(KEEP), model.config.hidden_size), np.float16)
    for b in range(0, len(order), 16):
        idx = order[b:b+16]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        hs = model.model(**enc, output_hidden_states=True).hidden_states[1:]
        H[idx] = torch.stack([hs[l][:, -1].float() for l in KEEP], 1).cpu().numpy().astype(np.float16)
    np.save(f"out/{name}.npy", H); df.to_json(f"out/{name}.jsonl", orient="records", lines=True, force_ascii=False)
    log(name, H.shape)
api.upload_folder(folder_path="out", path_in_repo=f"runs/feats/{TAG}", repo_id=REPO, repo_type="dataset")
log("DONE")
