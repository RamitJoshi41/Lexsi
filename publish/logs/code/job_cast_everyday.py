# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=4.56", "accelerate", "pandas", "numpy", "huggingface_hub", "pyarrow", "safetune>=0.1.13"]
# ///
"""Gate false-trigger rate of SafeTune CAST (all-script calibration) on the SAME 600 held-out everyday prompts used for gate v2."""
import json, time, numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download
from safetune.runner.steer import CASTTrainer
REPO, MID = "sunit12/aya-codemix-work", "CohereLabs/tiny-aya-global"
dl = lambda p: hf_hub_download(REPO, p, repo_type="dataset")
df = pd.read_json(dl("codemix_v1.jsonl"), lines=True); fit = set(json.load(open(dl("fit_ids.json"))))
cal = df[df.id.isin(fit)]
bm = pd.read_json(dl("runs/benign/tiny-aya-global/benign_meta.jsonl"), lines=True)
rng = np.random.default_rng(0); btest = np.zeros(len(bm), bool)
for lang in bm.lang.unique():
    ix = np.where(bm.lang == lang)[0]; btest[rng.choice(ix, 200, replace=False)] = True
ev = bm[btest].reset_index(drop=True)
tok = AutoTokenizer.from_pretrained(MID); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda").eval()
cast, _ = CASTTrainer(model, tok, alpha=1.0, per_prompt_gate=True).calibrate(
    harmful=cal[cal.label == 1].text.tolist(), harmless=cal[cal.label == 0].text.tolist())
fires = []
with torch.no_grad():
    for t in ev.text:
        enc = tok(tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False),
                  return_tensors="pt", add_special_tokens=False).to("cuda")
        f, s = cast._gate_fires(enc["input_ids"], attention_mask=enc["attention_mask"])
        fires.append(bool(f[0]) if isinstance(f, (list, tuple)) else bool(f))
ev["cast_fired"] = fires
res = ev.groupby("lang").cast_fired.agg(["mean", "sum", "count"]).to_dict(orient="index")
print("CAST(all-script) everyday false-trigger rate:", res, flush=True)
print("examples:", ev[ev.cast_fired].text.str[:120].head(8).tolist(), flush=True)
json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in res.items()}, open("cast_everyday.json", "w"), indent=1)
HfApi().upload_file(path_or_fileobj="cast_everyday.json", path_in_repo="runs/cast_everyday_tiny-aya-global.json", repo_id=REPO, repo_type="dataset")
print("DONE", flush=True)
