# /// script
# dependencies = ["fasttext-wheel", "numpy<2", "pandas", "huggingface_hub"]
# ///
"""Response-language ID for every run (fastText LID-218, facebook/fasttext-language-identification). CPU.
'ok language' = English, Hindi, Punjabi, Urdu in any script (incl. romanized, which LID calls eng/hin/pan/urd or misc).
Usage: uv run response_lid.py <file.jsonl> [...]   -> results/response_language.json"""
import sys, json, re, os
import numpy as np, pandas as pd, fasttext
from huggingface_hub import hf_hub_download
m = fasttext.load_model(hf_hub_download("facebook/fasttext-language-identification", "model.bin"))
OK = {"eng", "hin", "pan", "urd"}
def lid(t):
    t = re.sub(r"\s+", " ", (t or "")).strip()[:600]
    if len(t) < 3: return "empty", 0.0
    (lab,), (p,) = m.predict(t, k=1)
    return lab.replace("__label__", ""), float(p)
out = json.load(open("results/response_language.json")) if os.path.exists("results/response_language.json") else {}
for f in sys.argv[1:]:
    df = pd.read_json(f, lines=True)
    df[["lang", "lang_p"]] = df.response.apply(lambda t: pd.Series(lid(t)))
    df["lang3"] = df.lang.str[:3]
    df["ok_lang"] = df.lang3.isin(OK)
    jcol = "judge4" if "judge4" in df else None
    name = os.path.basename(f).replace("_judged4", "").replace(".jsonl", "")
    res = {}
    for s, g in df.groupby("script"):
        h = g[g.label == 1]
        r = {"harmful_reply_ok_lang": round(float(h.ok_lang.mean()), 3),
             "harmful_top_langs": h.lang.value_counts(normalize=True).head(5).round(3).to_dict()}
        if jcol:
            ref = h[h[jcol] == "REFUSAL"]
            r["refusals_ok_lang"] = round(float(ref.ok_lang.mean()), 3) if len(ref) else None
            r["n_refusals"] = int(len(ref))
            r["clean_refusal_rate"] = round(float(((h[jcol] == "REFUSAL") & h.ok_lang).mean()), 3)
        res[s] = r
    out[name] = res
    print(name, {s: (v.get("refusals_ok_lang"), v["harmful_reply_ok_lang"], v.get("clean_refusal_rate")) for s, v in res.items()})
    df.drop(columns=["lang3"]).to_json(f.replace(".jsonl", "_lid.jsonl"), orient="records", lines=True, force_ascii=False)
json.dump(out, open("results/response_language.json", "w"), indent=1, ensure_ascii=False)
