# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=4.56", "accelerate", "pandas", "numpy", "huggingface_hub", "pyarrow", "indic-transliteration"]
# ///
"""Job 15: transliterate romanized Hindi/Punjabi prompts into Devanagari/Gurmukhi before they reach Tiny Aya.
Primary: Aya Expanse 8B with a strict transliteration prompt. A row falls back to rule-based ITRANS transliteration
(indic_transliteration) if the LLM output is not mostly in the target script, has an implausible length, or is a refusal.
Inputs: codemix held-out romanized rows + newcats_test romanized rows. Writes translit_heldout.jsonl, translit_newcats.jsonl.
Usage: job_translit.py <work_repo>"""
import os, sys, re, json, time, unicodedata
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download
from indic_transliteration import sanscript
REPO = sys.argv[1]; JUDGE = "CohereLabs/aya-expanse-8b"
torch.set_grad_enabled(False); t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
dl = lambda p: hf_hub_download(REPO, p, repo_type="dataset")
bench = pd.read_json(dl("codemix_v1.jsonl"), lines=True); fit = set(json.load(open(dl("fit_ids.json"))))
ho = bench[~bench.id.isin(fit) & bench.script.isin(["hi_rom", "pa_rom"])].copy(); ho["set"] = "heldout"
nc = pd.read_json(dl("newcats_test.jsonl"), lines=True); nc = nc[nc.script.isin(["hi_rom", "pa_rom"])].copy(); nc["set"] = "newcats"
rows = pd.concat([ho, nc], ignore_index=True); log("rows", len(rows))
tok = AutoTokenizer.from_pretrained(JUDGE); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(JUDGE, dtype=torch.bfloat16, device_map="cuda").eval()
P = {"hi": ("Hindi", "Devanagari", r"[ऀ-ॿ]"), "pa": ("Punjabi", "Gurmukhi", r"[਀-੿]")}
TPL = ("You are a transliteration tool, not an assistant. Rewrite the {lang} text below, which is typed in Latin letters, "
       "in {script} script. Do not translate, answer, judge, shorten or comment on it: only change the script, word for word. "
       "English words may stay in Latin letters. Output only the transliterated text.\n\nText: {t}")
def gen(msgs, bs=16):
    prompts = [tok.apply_chat_template([{"role": "user", "content": m}], add_generation_prompt=True, tokenize=False) for m in msgs]
    out = [None] * len(prompts); order = np.argsort([len(p) for p in prompts])
    for b in range(0, len(order), bs):
        idx = order[b:b+bs]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=160, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx): out[i] = tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return out
lang = rows.script.str[:2]
llm = gen([TPL.format(lang=P[l][0], script=P[l][1], t=t) for l, t in zip(lang, rows.text)])
REFUSE = re.compile(r"sorry|cannot|can't|माफ़|क्षमा|नहीं कर सकत|ਮਾਫ਼|ਨਹੀਂ ਕਰ ਸਕ|unable", re.I)
def ok(src, out, l):
    out = (out or "").split("\n")[0].strip().strip('"“”')
    letters = [c for c in out if c.isalpha()]
    frac = sum(bool(re.match(P[l][2], c)) for c in letters) / max(1, len(letters))
    return out, frac >= 0.6 and 0.5 <= len(out) / max(1, len(src)) <= 2.5 and not REFUSE.search(out)
def rule(t, l):   # informal romanization -> lower-case ITRANS-ish -> script
    s = t.lower().replace("ee", "I").replace("oo", "U").replace("aa", "A")
    return sanscript.transliterate(s, sanscript.ITRANS, sanscript.DEVANAGARI if l == "hi" else sanscript.GURMUKHI)
texts, methods = [], []
for src, out, l in zip(rows.text, llm, lang):
    o, good = ok(src, out, l)
    if good: texts.append(o); methods.append("aya_expanse")
    else: texts.append(rule(src, l)); methods.append("rule_fallback" if not REFUSE.search(out or "") else "rule_fallback_llm_refused")
rows["orig_text"] = rows.text; rows["text"] = texts; rows["translit_method"] = methods; rows["llm_raw"] = llm
rows["script"] = rows.script.map({"hi_rom": "hi_rom2deva", "pa_rom": "pa_rom2guru"})
log(pd.crosstab([rows.set, rows.script, rows.label], rows.translit_method).to_string())
log(rows[["orig_text", "text", "translit_method"]].head(8).to_string())
os.makedirs("out", exist_ok=True); api = HfApi()
for s in ["heldout", "newcats"]:
    f = f"out/translit_{s}.jsonl"; rows[rows.set == s].to_json(f, orient="records", lines=True, force_ascii=False)
    api.upload_file(path_or_fileobj=f, path_in_repo=f"translit_{s}.jsonl", repo_id=REPO, repo_type="dataset")
log("DONE")
