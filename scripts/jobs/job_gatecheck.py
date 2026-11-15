# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=4.56", "accelerate", "pandas", "numpy", "scikit-learn", "huggingface_hub", "pyarrow", "datasets"]
# ///
"""Job 05: gate false-positive rate on everyday instructions (CohereLabs/aya_dataset) - forward passes only."""
import sys, json, time
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from datasets import load_dataset

REPO, MODEL_ID = sys.argv[1], sys.argv[2]
TAG = MODEL_ID.split("/")[-1]; GATE_L, GATE_T, N = 8, 0.7, 200
torch.set_grad_enabled(False)
dl = lambda p: hf_hub_download(REPO, p, repo_type="dataset")
df = pd.read_json(dl("codemix_v1.jsonl"), lines=True); fit = set(json.load(open(dl("fit_ids.json"))))
H = np.load(dl(f"runs/baseline/{TAG}/hidden_{TAG}.npy")).astype(np.float32)
y = df.label.values; isfit = df.id.isin(fit).values
X = H[:, GATE_L]; mu = X[isfit].mean(0); sd = X[isfit].std(0) + 1e-6
clf = LogisticRegression(max_iter=3000, C=0.05).fit((X[isfit] - mu) / sd, y[isfit])

ad = load_dataset("CohereLabs/aya_dataset", split="train").to_pandas()
print(ad.language.value_counts().head(40).to_string(), flush=True)
langs = [l for l in ["English", "Hindi", "Panjabi", "Punjabi"] if l in set(ad.language)]
tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
res = {}
for lang in langs:
    texts = ad[ad.language == lang].inputs.sample(min(N, (ad.language == lang).sum()), random_state=0).str[:1500].tolist()
    feats = []
    for b in range(0, len(texts), 16):
        p = [tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False) for t in texts[b:b+16]]
        enc = tok(p, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        hs = model.model(**enc, output_hidden_states=True).hidden_states[GATE_L + 1][:, -1].float().cpu().numpy()
        feats.append(hs)
    F = np.concatenate(feats)
    pr = clf.predict_proba((F - mu) / sd)[:, 1]
    fired = [t for t, q in zip(texts, pr) if q > GATE_T]
    res[lang] = {"n": len(texts), "fire_rate": float((pr > GATE_T).mean()), "examples_fired": [t[:200] for t in fired[:8]]}
    print(lang, res[lang], flush=True)
from huggingface_hub import HfApi
json.dump(res, open("gatecheck.json", "w"), ensure_ascii=False, indent=1)
HfApi().upload_file(path_or_fileobj="gatecheck.json", path_in_repo=f"runs/gatecheck_{TAG}.json", repo_id=REPO, repo_type="dataset")
print("DONE", flush=True)
