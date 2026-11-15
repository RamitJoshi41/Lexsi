# /// script
# dependencies = ["pandas", "numpy", "datasketch", "sentence-transformers", "huggingface_hub"]
# ///
"""Leak check between everything used to FIT the guard (gate v3 training data, refusal-vector prompts) and every
TEST set. (1) exact match after normalisation, (2) MinHash (char 5-gram shingles, 128 perms) Jaccard >= 0.5 via LSH,
(3) multilingual embeddings (paraphrase-multilingual-mpnet-base-v2) max cosine >= 0.90.
Writes results/leak_check.json + results/leak_check_examples.jsonl."""
import json, os, re, sys, unicodedata, numpy as np, pandas as pd
from huggingface_hub import hf_hub_download, HfApi
WORK = "sunit12/aya-codemix-work"
def P(local, remote):   # local file if present, else from the work repo (HF Jobs)
    return local if os.path.exists(local) else hf_hub_download(WORK, remote, repo_type="dataset")
os.makedirs("results", exist_ok=True)
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer
rj = lambda p: pd.read_json(p, lines=True)
norm = lambda t: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", unicodedata.normalize("NFKC", str(t)).lower())).strip()
cm = rj(P("data/codemix_v1.jsonl", "codemix_v1.jsonl")); fit = set(json.load(open(P("fit_ids.json", "fit_ids.json"))))
nc = rj(P("data/newcats_v1.jsonl", "newcats_v1.jsonl")); test_pairs = set(rj(P("data/newcats_test.jsonl", "newcats_test.jsonl")).pair)
train = pd.concat([
    cm[cm.id.isin(fit)].assign(src="benchmark_fit"),
    nc[~nc.pair.isin(test_pairs)].assign(src="newcats_train"),
    rj(P("data/aug_fit_v1.jsonl", "aug_fit_v1.jsonl")).assign(src="aug_fit_v1"), rj(P("data/aug_fit_v2.jsonl", "aug_fit_v2.jsonl")).assign(src="aug_fit_v2"),
    rj(P("runs/benign/tiny-aya-global/benign_meta.jsonl", "runs/benign/tiny-aya-global/benign_meta.jsonl")).assign(src="everyday_benign", label=0, script="native"),
    rj(P("data/everyday_rom_v1.jsonl", "everyday_rom_v1.jsonl")).assign(src="everyday_rom")], ignore_index=True)[["src", "label", "script", "text"]]
test = pd.concat([cm[~cm.id.isin(fit)].assign(src="benchmark_heldout"),
                  nc[nc.pair.isin(test_pairs)].assign(src="newcats_test")], ignore_index=True)[["src", "id", "label", "script", "text"]]
print("train", train.src.value_counts().to_dict(), "\ntest", test.src.value_counts().to_dict(), flush=True)
train["n"] = train.text.map(norm); test["n"] = test.text.map(norm)
exact = test.n.isin(set(train.n)); print("exact matches:", int(exact.sum()), flush=True)
def mh(t):
    m = MinHash(num_perm=128); s = t if len(t) >= 5 else t + "     "
    for i in range(len(s) - 4): m.update(s[i:i + 5].encode())
    return m
lsh = MinHashLSH(threshold=0.5, num_perm=128); TM = [mh(t) for t in train.n]
for i, m in enumerate(TM): lsh.insert(str(i), m)
mh_hits = []
for i, t in enumerate(test.n):
    m = mh(t); c = lsh.query(m)
    if c:
        j = max(c, key=lambda k: m.jaccard(TM[int(k)])); mh_hits.append((i, int(j), float(m.jaccard(TM[int(j)]))))
print("minhash >=0.5:", len(mh_hits), flush=True)
emb = SentenceTransformer("sentence-transformers/LaBSE")
E_tr = emb.encode(train.text.tolist(), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
E_te = emb.encode(test.text.tolist(), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
S = E_te @ E_tr.T; best = S.argmax(1); bc = S.max(1)
test["max_cos"] = bc; test["nn_src"] = train.src.values[best]; test["nn_text"] = train.text.values[best]; test["nn_label"] = train.label.values[best]
test["minhash"] = 0.0
for i, j, s in mh_hits: test.loc[i, "minhash"] = s
res = {"n_train": len(train), "n_test": len(test), "train_sources": train.src.value_counts().to_dict(),
       "exact_matches": int(exact.sum()), "minhash_ge_0.5": len(mh_hits)}
for thr in [0.95, 0.90, 0.85]:
    h = test[test.max_cos >= thr]
    res[f"emb_cos_ge_{thr}"] = {"n": len(h), "by_test_set": h.src.value_counts().to_dict(), "by_nn_source": h.nn_src.value_counts().to_dict(),
                                "same_label_as_nn": int((h.label == h.nn_label).sum())}
res["max_cos_quantiles"] = {q: round(float(np.quantile(bc, q)), 3) for q in [0.5, 0.9, 0.99, 1.0]}
json.dump(res, open("results/leak_check.json", "w"), indent=1)
test[(test.max_cos >= 0.85) | (test.minhash > 0) | exact].sort_values("max_cos", ascending=False)[
    ["src", "id", "script", "label", "text", "max_cos", "minhash", "nn_src", "nn_label", "nn_text"]].to_json(
    "results/leak_check_examples.jsonl", orient="records", lines=True, force_ascii=False)
print(json.dumps(res, indent=1))

if os.environ.get("UPLOAD"):
    for f in ["results/leak_check.json", "results/leak_check_examples.jsonl"]:
        HfApi().upload_file(path_or_fileobj=f, path_in_repo=f, repo_id=WORK, repo_type="dataset")
