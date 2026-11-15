# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=4.56", "accelerate", "pandas", "numpy", "huggingface_hub", "pyarrow", "datasets"]
# ///
"""Job 06: layer-wise last-token features for everyday instructions (aya_dataset: English/Hindi/Panjabi).
Saves layers 6..14 so the gate can be retrained with benign everyday negatives."""
import sys, time
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
from datasets import load_dataset

REPO, MODEL_ID = sys.argv[1], sys.argv[2]
TAG = MODEL_ID.split("/")[-1]; N = 700; LAYERS = list(range(6, 15))
torch.set_grad_enabled(False); t0 = time.time()
ad = load_dataset("CohereLabs/aya_dataset", split="train").to_pandas()
rows = []
for lang in ["English", "Hindi", "Panjabi"]:
    sub = ad[ad.language == lang].drop_duplicates("inputs")
    sub = sub[sub.inputs.str.len().between(10, 1500)].sample(min(N, len(sub)), random_state=1)
    rows += [{"lang": lang, "text": t} for t in sub.inputs]
meta = pd.DataFrame(rows)
tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
prompts = [tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False) for t in meta.text]
order = np.argsort([len(p) for p in prompts]); F = np.zeros((len(meta), len(LAYERS), model.config.hidden_size), np.float16)
for b in range(0, len(order), 16):
    idx = order[b:b+16]
    enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    hs = model.model(**enc, output_hidden_states=True).hidden_states
    F[idx] = np.stack([hs[l + 1][:, -1].float().cpu().numpy() for l in LAYERS], 1).astype(np.float16)
print(f"[{time.time()-t0:.0f}s] features", F.shape, meta.lang.value_counts().to_dict(), flush=True)
np.save("benign_feats.npy", F); meta.to_json("benign_meta.jsonl", orient="records", lines=True, force_ascii=False)
api = HfApi()
for f in ["benign_feats.npy", "benign_meta.jsonl"]:
    api.upload_file(path_or_fileobj=f, path_in_repo=f"runs/benign/{TAG}/{f}", repo_id=REPO, repo_type="dataset")
print("DONE layers", LAYERS, flush=True)
