# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=4.56", "accelerate", "peft", "datasets", "numpy", "huggingface_hub"]
# ///
"""Diagnose capability loss: print base vs LoRA answers on 40 MMLU questions (same prompt as job_capability.py).
Usage: job_mmlu_diag.py <model_id> <adapter_repo> [adapter_repo2 ...]"""
import sys, re, numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
MODEL_ID, ADS = sys.argv[1], sys.argv[2:]; torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
ds = load_dataset("cais/mmlu", "all", split="test"); idx = np.random.default_rng(0).choice(len(ds), 40, replace=False)
qs = [ds[int(i)] for i in idx]
P = [tok.apply_chat_template([{"role": "user", "content": q["question"] + "\n" + "\n".join(f"{'ABCD'[k]}. {c}" for k, c in enumerate(q["choices"])) +
      "\nReply with only the letter (A, B, C or D) of the correct option."}], add_generation_prompt=True, tokenize=False) for q in qs]
def gen(m):
    out = []
    for b in range(0, len(P), 20):
        enc = tok(P[b:b+20], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = m.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
        out += [tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True) for j in range(g.shape[0])]
    return out
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
R = {"base": gen(base)}
for a in ADS:
    m = PeftModel.from_pretrained(base, a).eval(); R[a] = gen(m); base = m.unload()
acc = lambda outs: np.mean([bool(re.search(r"\b([ABCD])\b", o.strip().upper())) and re.search(r"\b([ABCD])\b", o.strip().upper()).group(1) == "ABCD"[q["answer"]] for o, q in zip(outs, qs)])
for k, v in R.items(): print("ACC", k, round(float(acc(v)), 3))
for i in range(40):
    print(f"--- Q{i} gold={'ABCD'[qs[i]['answer']]}")
    for k, v in R.items(): print(f"  {k.split('/')[-1][:28]:28s} | {v[i][:90]!r}")
