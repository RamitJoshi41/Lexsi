# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.56", "accelerate", "peft", "pandas", "numpy", "pyarrow", "huggingface_hub"]
# ///
"""Capability in the TARGET languages: does Fix 2 (the LoRA) cost Hindi / Punjabi knowledge? MMLU (job_capability.py)
is English only. This runs MILU (ai4bharat/MILU, CC-BY-4.0: Indian exam-style multiple choice, native script)
for Hindi and Punjabi, plain model vs plain + LoRA, same prompts, greedy, one letter.

    python scripts/jobs/job_capability_indic.py [N_PER_LANG=500] [--upload]      # any >=16 GB GPU, ~10 min
    env: MODEL (default CohereLabs/tiny-aya-global), ADAPTER (default sunit12/tiny-aya-global-codemix-refusal-lora)

MILU is gated (automatic approval): accept its terms on the Hub and set HF_TOKEN. One model is loaded; the plain answers
come from the same weights with the adapter switched off (PeftModel.disable_adapter). Writes
results/capability_indic_<adapter>.json (accuracy per language, answers changed, newly right/wrong, paired bootstrap
95% CI of the difference) and runs/capability_indic_<adapter>_per_question.jsonl. Romanized Hindi/Punjabi has no
standard benchmark; romanized harmless traffic is covered by the everyday over-refusal set instead."""
import sys, os, re, json, time
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import hf_hub_download, HfApi

MODEL = os.environ.get("MODEL", "CohereLabs/tiny-aya-global")
ADAPTER = os.environ.get("ADAPTER", "sunit12/tiny-aya-global-codemix-refusal-lora")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
N = int(args[0]) if args else 500
LANGS, BS, SEED = ["Hindi", "Punjabi"], 16, 0
LOGS = "sunit12/codemix-safety-logs"
INSTR = "\nReply with only the letter (A, B, C or D) of the correct option."     # same wording as the MMLU job
torch.set_grad_enabled(False); t0 = time.time(); log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

def load_milu(lang):
    df = pd.read_parquet(hf_hub_download("ai4bharat/MILU", f"{lang}/test-00000-of-00001.parquet", repo_type="dataset"))
    opts = [f"option{i}" for i in range(1, 5)]
    if not {"question", "target", *opts} <= set(df.columns):
        sys.exit(f"MILU {lang}: unexpected columns {df.columns.tolist()}; adapt load_milu()")
    def letter(t):  # MILU's target names the option ("option2"); a bare letter is accepted too
        t = str(t).strip().lower()
        if re.fullmatch(r"option[1-4]", t): return "ABCD"[int(t[-1]) - 1]
        if t in ("a", "b", "c", "d"): return t.upper()
        sys.exit(f"MILU {lang}: unexpected target {t!r}; adapt letter()")
    df = df.sample(min(N, len(df)), random_state=SEED).reset_index(drop=True)
    return pd.DataFrame({"lang": lang, "subject": df.get("subject", ""), "target": df.target.map(letter),
                         "prompt": [q + "\n" + "\n".join(f"{'ABCD'[k]}. {r[o]}" for k, o in enumerate(opts)) + INSTR
                                    for q, (_, r) in zip(df.question, df.iterrows())]})

def is_correct(target, output):
    m = re.search(r"\b([ABCD])\b", (output or "").strip().upper())
    return m is not None and m.group(1) == target

tok = AutoTokenizer.from_pretrained(ADAPTER); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda"), ADAPTER).eval()

def answer(prompts):
    out = []
    for b in range(0, len(prompts), BS):
        chunk = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
                 for p in prompts[b:b + BS]]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
        out += [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True) for g in gen]
    return out

qs = pd.concat([load_milu(l) for l in LANGS], ignore_index=True); log(len(qs), "questions", qs.groupby("lang").size().to_dict())
with model.disable_adapter():
    qs["base"] = answer(qs.prompt.tolist()); log("plain done")
qs["lora"] = answer(qs.prompt.tolist()); log("LoRA done")
for c in ["base", "lora"]:
    qs[c + "_correct"] = [float(is_correct(t, o)) for t, o in zip(qs.target, qs[c])]

rng = np.random.default_rng(SEED); res = {"model": MODEL, "adapter": ADAPTER, "n_per_lang": N, "by_lang": {}}
for lang, g in [*qs.groupby("lang"), ("all", qs)]:
    d = (g.lora_correct - g.base_correct).values
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    res["by_lang"][lang] = {"n": len(g), "acc_base": g.base_correct.mean(), "acc_lora": g.lora_correct.mean(),
                            "delta": d.mean(), "delta_ci95": [np.percentile(boot, 2.5), np.percentile(boot, 97.5)],
                            "answers_changed": int((g.base.str.strip() != g.lora.str.strip()).sum()),
                            "newly_correct": int(((g.base_correct == 0) & (g.lora_correct == 1)).sum()),
                            "newly_wrong": int(((g.base_correct == 1) & (g.lora_correct == 0)).sum()),
                            "no_letter_base": int(sum(not re.search(r"\b[ABCD]\b", (o or "").upper()) for o in g.base)),
                            "no_letter_lora": int(sum(not re.search(r"\b[ABCD]\b", (o or "").upper()) for o in g.lora))}
    r = res["by_lang"][lang]
    log(f"{lang}: plain {r['acc_base']:.1%} -> LoRA {r['acc_lora']:.1%} ({r['delta']:+.1%} "
        f"[{r['delta_ci95'][0]:+.1%}, {r['delta_ci95'][1]:+.1%}]); {r['answers_changed']} answers changed")

tag = ADAPTER.split("/")[-1]
os.makedirs("results", exist_ok=True); os.makedirs("runs", exist_ok=True)
paths = [f"results/capability_indic_{tag}.json", f"runs/capability_indic_{tag}_per_question.jsonl"]
json.dump(res, open(paths[0], "w"), indent=1, default=float)
qs.drop(columns="prompt").to_json(paths[1], orient="records", lines=True, force_ascii=False)
if "--upload" in sys.argv:
    for p in paths: HfApi().upload_file(path_or_fileobj=p, path_in_repo=p, repo_id=LOGS, repo_type="dataset")
log("DONE", paths)
