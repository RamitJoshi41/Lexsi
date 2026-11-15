# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=4.56", "accelerate", "pandas", "numpy", "scikit-learn", "huggingface_hub", "pyarrow", "matplotlib"]
# ///
"""Job 12: what language does Tiny Aya *think* romanized Hindi / Punjabi is?
Per layer: mean-pooled hidden states of reference sentences in ~20 languages (CohereLabs/aya_dataset) -> multinomial
language-ID probe; apply it to our 940 benchmark prompts and report, per script, the predicted language distribution.
Also: nearest language centroid (cosine) per layer.
Usage: hf jobs uv run --flavor a10g-small --secrets HF_TOKEN job_langid.py <work_repo> <model_id>
"""
import os, sys, time, json
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO, MODEL_ID = sys.argv[1], sys.argv[2]
TAG = MODEL_ID.split("/")[-1]; N_REF, BS = 120, 32
WANT = ["English", "Hindi", "Panjabi", "Urdu", "Marathi", "Nepali", "Indonesian", "Javanese", "Malay", "Sundanese", "Tagalog",
        "Swahili", "Xhosa", "Zulu", "Wolof", "Igbo", "Yoruba", "Hausa", "Somali", "Spanish", "French", "Portuguese", "Turkish", "Vietnamese"]
torch.set_grad_enabled(False); t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
rng = np.random.default_rng(0)

# reference sentences
aya = pd.read_parquet(hf_hub_download("CohereLabs/aya_dataset", "data/train-00000-of-00001.parquet", repo_type="dataset"))
log("aya_dataset languages:", aya.language.value_counts().head(80).to_dict())
refs = []
for lang in WANT:
    s = aya[(aya.language == lang) & aya.inputs.str.len().between(30, 400)]
    if len(s) < 60: log("skip", lang, len(s)); continue
    refs.append(s.sample(min(N_REF, len(s)), random_state=0).assign(lang=lang)[["lang", "inputs"]].rename(columns={"inputs": "text"}))
refs = pd.concat(refs, ignore_index=True); LANGS = sorted(refs.lang.unique()); log("ref langs", LANGS, len(refs))
bench = pd.read_json(hf_hub_download(REPO, "codemix_v1.jsonl", repo_type="dataset"), lines=True)

tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "right"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
L = model.config.num_hidden_layers

def pooled(texts):
    out = np.zeros((len(texts), L, model.config.hidden_size), np.float32)
    for b in range(0, len(texts), BS):
        enc = tok(texts[b:b+BS], return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
        hs = model.model(**enc, output_hidden_states=True).hidden_states[1:]
        m = enc["attention_mask"].unsqueeze(-1).float()
        for l, h in enumerate(hs):
            out[b:b+len(hs[0]), l] = ((h.float() * m).sum(1) / m.sum(1)).cpu().numpy()
    return out
R = pooled(refs.text.tolist()); log("refs pooled", R.shape)
Bh = pooled(bench.text.tolist()); log("bench pooled", Bh.shape)

S = ["en", "hi_deva", "hi_rom", "pa_guru", "pa_rom"]
res = {"langs": LANGS, "layers": {}}
yref = refs.lang.values
for l in range(L):
    X = R[:, l]; mu, sd = X.mean(0), X.std(0) + 1e-6
    # held-out accuracy of the probe on references (sanity)
    idx = rng.permutation(len(X)); tr, te = idx[: int(.8 * len(X))], idx[int(.8 * len(X)):]
    clf = LogisticRegression(max_iter=2000, C=0.1).fit((X[tr] - mu) / sd, yref[tr])
    acc = float((clf.predict((X[te] - mu) / sd) == yref[te]).mean())
    clf = LogisticRegression(max_iter=2000, C=0.1).fit((X - mu) / sd, yref)
    P = clf.predict_proba((Bh[:, l] - mu) / sd)
    cent = np.stack([X[yref == g].mean(0) for g in clf.classes_]); cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    Z = Bh[:, l] / np.linalg.norm(Bh[:, l], axis=1, keepdims=True); near = np.array(clf.classes_)[(Z @ cent.T).argmax(1)]
    row = {"probe_acc": acc}
    for s in S:
        m = (bench.script == s).values
        row[s] = {"probe_mean_prob": dict(zip(clf.classes_, P[m].mean(0).round(4).tolist())),
                  "probe_top1_share": pd.Series(np.array(clf.classes_)[P[m].argmax(1)]).value_counts(normalize=True).round(3).to_dict(),
                  "nearest_centroid_share": pd.Series(near[m]).value_counts(normalize=True).round(3).to_dict()}
    res["layers"][l] = row
    if l % 4 == 0 or l == L - 1:
        log(f"L{l} acc={acc:.2f} | " + " | ".join(f"{s}: " + ", ".join(f"{k}:{v:.2f}" for k, v in list(row[s]['probe_top1_share'].items())[:3]) for s in S))

os.makedirs("out", exist_ok=True)
json.dump(res, open(f"out/langid_{TAG}.json", "w"), indent=1)
# figure: per script, probability mass on the top languages across layers
fig, axes = plt.subplots(1, 4, figsize=(20, 4.2), sharey=True)
for ax, s in zip(axes, ["hi_deva", "hi_rom", "pa_guru", "pa_rom"]):
    M = pd.DataFrame({l: res["layers"][l][s]["probe_mean_prob"] for l in range(L)}).T
    top = M.mean().sort_values(ascending=False).index[:6]
    for g in top: ax.plot(M.index, M[g], label=g, lw=2)
    ax.set_title({"hi_deva": "Hindi (Devanagari)", "hi_rom": "Hindi (romanized)", "pa_guru": "Punjabi (Gurmukhi)", "pa_rom": "Punjabi (romanized)"}[s])
    ax.set_xlabel("layer"); ax.axvline(14, color="grey", ls=":"); ax.legend(fontsize=8)
axes[0].set_ylabel("language-ID probe probability")
plt.suptitle(f"What language does {TAG} think the prompt is in? (probe trained on aya_dataset sentences)")
plt.tight_layout(); plt.savefig(f"out/langid_{TAG}.png", dpi=140)
HfApi().upload_folder(folder_path="out", path_in_repo=f"runs/langid/{TAG}", repo_id=REPO, repo_type="dataset")
log("DONE")
