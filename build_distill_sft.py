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
TPL = {  # several phrasings per script so the student does not memorise one canned sentence
 "en": ["I'm sorry, but I can't help with that. It could cause real harm to people. If you'd like, I'm happy to help with something else.",
        "I can't help with this request, because it could hurt someone. Is there something else I can do for you?",
        "Sorry, I won't do that. It promotes harm against people. I'm glad to help with a different question.",
        "I'm not able to help with that, since it could be used to harm others. Let me know if there's anything else you need.",
        "That's not something I can help with. It could lead to real harm. I'd be happy to help with something else, though."],
 "hi_rom": ["Maaf kijiye, main is mein madad nahi kar sakta. Isse kisi ko sach mein nuksaan pahunch sakta hai. Agar aap kisi aur cheez ke baare mein baat karna chahein, toh main madad ke liye taiyaar hoon.",
            "Main is request mein aapki madad nahi kar sakta, kyunki isse kisi ko nuksaan ho sakta hai. Kya main kisi aur cheez mein aapki madad kar sakta hoon?",
            "Sorry, main yeh nahi kar sakta. Yeh logon ke khilaaf nuksaan ko badhava deta hai. Aap koi aur sawaal poochna chahein toh zaroor poochiye.",
            "Yeh cheez main nahi bata sakta, kyunki iska istemaal doosron ko nuksaan pahunchane ke liye ho sakta hai. Kisi aur cheez mein madad chahiye toh batayiye.",
            "Mujhe maaf karein, is mein main madad nahi karunga. Isse kisi ki jaan ya izzat ko khatra ho sakta hai. Main kisi aur tarah se aapki madad kar sakta hoon."],
 "pa_rom": ["Maaf karna, main is vich madad nahi kar sakda. Is naal kise nu sach vich nuksaan ho sakda hai. Je tusi kise hor gall bare gall karni chaho, taan main madad layi tiyaar haan.",
            "Main is request vich tuhadi madad nahi kar sakda, kyunki is naal kise da nuksaan ho sakda hai. Ki main kise hor cheez vich madad kar sakda haan?",
            "Sorry, main eh nahi kar sakda. Eh lokan de khilaaf nuksaan nu vadhava dinda hai. Tusi koi hor sawaal puchhna chaho taan zaroor puchho.",
            "Eh gall main nahi dass sakda, kyunki is da istemaal dujiyan nu nuksaan pahunchaun layi ho sakda hai. Kise hor cheez vich madad chahidi hai taan dasso.",
            "Mainu maaf karna, is vich main madad nahi karanga. Is naal kise di jaan jaan izzat nu khatra ho sakda hai. Main kise hor tarike naal tuhadi madad kar sakda haan."],
 "hi_deva": ["माफ़ कीजिए, मैं इसमें मदद नहीं कर सकता। इससे किसी को सच में नुकसान पहुँच सकता है। अगर आप किसी और विषय पर बात करना चाहें, तो मैं मदद के लिए तैयार हूँ।",
             "मैं इस अनुरोध में आपकी मदद नहीं कर सकता, क्योंकि इससे किसी को नुकसान हो सकता है। क्या मैं किसी और चीज़ में आपकी मदद कर सकता हूँ?",
             "क्षमा करें, मैं यह नहीं कर सकता। यह लोगों के खिलाफ़ नुकसान को बढ़ावा देता है। आप कोई और सवाल पूछना चाहें तो ज़रूर पूछिए।"],
 "pa_guru": ["ਮਾਫ਼ ਕਰਨਾ, ਮੈਂ ਇਸ ਵਿੱਚ ਮਦਦ ਨਹੀਂ ਕਰ ਸਕਦਾ। ਇਸ ਨਾਲ ਕਿਸੇ ਨੂੰ ਸੱਚਮੁੱਚ ਨੁਕਸਾਨ ਹੋ ਸਕਦਾ ਹੈ। ਜੇ ਤੁਸੀਂ ਕਿਸੇ ਹੋਰ ਵਿਸ਼ੇ ਬਾਰੇ ਗੱਲ ਕਰਨੀ ਚਾਹੋ, ਤਾਂ ਮੈਂ ਮਦਦ ਲਈ ਤਿਆਰ ਹਾਂ।",
             "ਮੈਂ ਇਸ ਬੇਨਤੀ ਵਿੱਚ ਤੁਹਾਡੀ ਮਦਦ ਨਹੀਂ ਕਰ ਸਕਦਾ, ਕਿਉਂਕਿ ਇਸ ਨਾਲ ਕਿਸੇ ਦਾ ਨੁਕਸਾਨ ਹੋ ਸਕਦਾ ਹੈ। ਕੀ ਮੈਂ ਕਿਸੇ ਹੋਰ ਚੀਜ਼ ਵਿੱਚ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?",
             "ਮਾਫ਼ ਕਰਨਾ, ਮੈਂ ਇਹ ਨਹੀਂ ਕਰ ਸਕਦਾ। ਇਹ ਲੋਕਾਂ ਦੇ ਖ਼ਿਲਾਫ਼ ਨੁਕਸਾਨ ਨੂੰ ਵਧਾਵਾ ਦਿੰਦਾ ਹੈ। ਤੁਸੀਂ ਕੋਈ ਹੋਰ ਸਵਾਲ ਪੁੱਛਣਾ ਚਾਹੋ ਤਾਂ ਜ਼ਰੂਰ ਪੁੱਛੋ।"],
}
HELP = {  # appended for self-harm prompts
 "en": " If you're going through a hard time, please talk to someone you trust or call a local helpline.",
 "hi_rom": " Agar aap mushkil waqt se guzar rahe hain, toh kisi bharosemand insaan se ya helpline par baat zaroor karein.",
 "pa_rom": " Je tusi aukhe samay vichon langh rahe ho, taan kise bharosemand insaan naal jaan helpline te gall zaroor karo.",
 "hi_deva": " अगर आप मुश्किल समय से गुज़र रहे हैं, तो किसी भरोसेमंद इंसान से या हेल्पलाइन पर बात ज़रूर करें।",
 "pa_guru": " ਜੇ ਤੁਸੀਂ ਔਖੇ ਸਮੇਂ ਵਿੱਚੋਂ ਲੰਘ ਰਹੇ ਹੋ, ਤਾਂ ਕਿਸੇ ਭਰੋਸੇਮੰਦ ਇਨਸਾਨ ਨਾਲ ਜਾਂ ਹੈਲਪਲਾਈਨ ਤੇ ਗੱਲ ਜ਼ਰੂਰ ਕਰੋ।",
}
rows, stats = [], collections.Counter(); rng = np.random.default_rng(0)
for r in d.itertuples():
    sc = r.script if r.script in WANT else "en"
    resp = (r.response or "").strip()
    if r.label == 1:
        okl = bool(getattr(r, "ok_lang", True))   # fastText LID: English/Hindi/Punjabi/Urdu (drops e.g. Indonesian "Maaf, tapi aku...")
        good = r.judge4 == "REFUSAL" and not deg(resp) and script_of(resp) == WANT[sc] and okl and len(resp) > 20
        if good: comp, kind = resp, "teacher_refusal"
        else:
            comp = TPL[sc][rng.integers(len(TPL[sc]))] + (HELP[sc] if "self" in str(getattr(r, "category", "")).lower() else ""); kind = "template_refusal"
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
