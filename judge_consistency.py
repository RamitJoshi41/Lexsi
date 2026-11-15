"""Agreement between the A10G-bf16 judge labels and the Kaggle-T4-fp16 judge labels on the same 470 replies (LoRA v2c held-out).
Writes results/judge_consistency.json"""
import json, pandas as pd, numpy as np
a = pd.read_json('runs/lora/tiny-aya-global/gen_lora_v2c_heldout_judged4.jsonl', lines=True)
b = pd.read_json('runs/lora/tiny-aya-global/kagglecheck_v2c_heldout_judged4.jsonl', lines=True)
m = a[['id', 'script', 'label', 'judge4']].merge(b[['id', 'script', 'judge4']], on=['id', 'script'], suffixes=('_a10g', '_t4'))
agree4 = float((m.judge4_a10g == m.judge4_t4).mean())
ra, rb = m.judge4_a10g == 'REFUSAL', m.judge4_t4 == 'REFUSAL'
po = float((ra == rb).mean()); pe = float(ra.mean() * rb.mean() + (1 - ra.mean()) * (1 - rb.mean())); kappa = (po - pe) / (1 - pe)
h = m.label == 1
res = {"n": len(m), "agreement_4way": round(agree4, 4), "agreement_refusal_binary": round(po, 4), "cohen_kappa_refusal": round(kappa, 4),
       "refusal_rate_harmful_a10g": round(float(ra[h].mean()), 4), "refusal_rate_harmful_t4": round(float(rb[h].mean()), 4),
       "refusal_rate_harmless_a10g": round(float(ra[~h].mean()), 4), "refusal_rate_harmless_t4": round(float(rb[~h].mean()), 4),
       "confusion": pd.crosstab(m.judge4_a10g, m.judge4_t4).to_dict()}
json.dump(res, open('results/judge_consistency.json', 'w'), indent=1); print(json.dumps({k: v for k, v in res.items() if k != 'confusion'}, indent=1))
