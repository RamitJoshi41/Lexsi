# /// script
# dependencies = ["pandas", "numpy", "datasets", "huggingface_hub"]
# ///
"""Build the AlignTune SFT set from teacher generations (job_distill_data.py -> judge4 -> response_lid).

harmful prompt  -> keep the steered teacher reply if it is a judged REFUSAL, not a degenerate loop, and written in
                   the user's own script family (Latin for en/romanized, Devanagari for hi_deva, Gurmukhi for
                   pa_guru); otherwise use a short hand-written refusal in the user's script (counted and reported).
harmless prompt -> keep the base model's own reply (self-distillation) unless the judge calls it a REFUSAL or it loops.
Usage: uv run build_distill_sft.py <teacher_judged4_lid.jsonl> <dataset_repo>"""
import sys, re, zlib, json, collections
import pandas as pd, numpy as np
from datasets import Dataset

src, DREPO = sys.argv[1], sys.argv[2]
d = pd.read_json(src, lines=True)
deg = lambda t: len((t or "").strip().encode()) > 40 and len(zlib.compress((t or "").strip().encode())) / len((t or "").strip().encode()) < 0.15
def script_of(t):
    t = t or ""; c = collections.Counter("deva" if "ऀ" <= ch <= "ॿ" else "guru" if "਀" <= ch <= "੿"
                                         else "latin" if ch.isascii() and ch.isalpha() else None for ch in t)
    c.pop(None, None); return c.most_common(1)[0][0] if c else "none"
WANT = {"en": "latin", "hi_rom": "latin", "pa_rom": "latin", "hi_deva": "deva", "pa_guru": "guru"}
TPL = {
 "en": "I'm sorry, but I can't help with that. It could cause real harm to people. If you'd like, I'm happy to help with something else.",
 "hi_rom": "Maaf kijiye, main is mein madad nahi kar sakta. Isse kisi ko sach mein nuksaan pahunch sakta hai. Agar aap kisi aur cheez ke baare mein baat karna chahein, toh main madad ke liye taiyaar hoon.",
 "pa_rom": "Maaf karna, main is vich madad nahi kar sakda. Is naal kise nu sach vich nuksaan ho sakda hai. Je tusi kise hor gall bare gall karni chaho, taan main madad layi tiyaar haan.",
 "hi_deva": "माफ़ कीजिए, मैं इसमें मदद नहीं कर सकता। इससे किसी को सच में नुकसान पहुँच सकता है। अगर आप किसी और विषय पर बात करना चाहें, तो मैं मदद के लिए तैयार हूँ।",
 "pa_guru": "ਮਾਫ਼ ਕਰਨਾ, ਮੈਂ ਇਸ ਵਿੱਚ ਮਦਦ ਨਹੀਂ ਕਰ ਸਕਦਾ। ਇਸ ਨਾਲ ਕਿਸੇ ਨੂੰ ਸੱਚਮੁੱਚ ਨੁਕਸਾਨ ਹੋ ਸਕਦਾ ਹੈ। ਜੇ ਤੁਸੀਂ ਕਿਸੇ ਹੋਰ ਵਿਸ਼ੇ ਬਾਰੇ ਗੱਲ ਕਰਨੀ ਚਾਹੋ, ਤਾਂ ਮੈਂ ਮਦਦ ਲਈ ਤਿਆਰ ਹਾਂ।",
}
HELP = {  # appended for self-harm prompts
 "en": " If you're going through a hard time, please talk to someone you trust or call a local helpline.",
 "hi_rom": " Agar aap mushkil waqt se guzar rahe hain, toh kisi bharosemand insaan se ya helpline par baat zaroor karein.",
 "pa_rom": " Je tusi aukhe samay vichon langh rahe ho, taan kise bharosemand insaan naal jaan helpline te gall zaroor karo.",
 "hi_deva": " अगर आप मुश्किल समय से गुज़र रहे हैं, तो किसी भरोसेमंद इंसान से या हेल्पलाइन पर बात ज़रूर करें।",
 "pa_guru": " ਜੇ ਤੁਸੀਂ ਔਖੇ ਸਮੇਂ ਵਿੱਚੋਂ ਲੰਘ ਰਹੇ ਹੋ, ਤਾਂ ਕਿਸੇ ਭਰੋਸੇਮੰਦ ਇਨਸਾਨ ਨਾਲ ਜਾਂ ਹੈਲਪਲਾਈਨ ਤੇ ਗੱਲ ਜ਼ਰੂਰ ਕਰੋ।",
}
rows, stats = [], collections.Counter()
for r in d.itertuples():
    sc = r.script if r.script in WANT else "en"
    resp = (r.response or "").strip()
    if r.label == 1:
        good = r.judge4 == "REFUSAL" and not deg(resp) and script_of(resp) == WANT[sc] and len(resp) > 20
        if good: comp, kind = resp, "teacher_refusal"
        else:
            comp = TPL[sc] + (HELP[sc] if "self" in str(getattr(r, "category", "")).lower() else ""); kind = "template_refusal"
    else:
        if r.judge4 == "REFUSAL" or deg(resp) or len(resp) < 2: stats[f"drop_harmless_{r.script}"] += 1; continue
        comp, kind = resp, "self_distill"
    stats[f"{kind}_{sc}"] += 1
    rows.append({"prompt": r.text, "completion": comp, "label": int(r.label), "script": r.script, "src": r.src, "kind": kind})
D = pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)
print(json.dumps(dict(sorted(stats.items())), indent=1)); print("total", len(D), D.groupby(["label", "kind"]).size().to_dict())
json.dump({"stats": dict(stats), "n": len(D)}, open("results/distill_sft_stats.json", "w"), indent=1)
Dataset.from_pandas(D).push_to_hub(DREPO, split="train", private=False)
print("pushed", DREPO)
