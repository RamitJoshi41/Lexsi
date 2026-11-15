# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "matplotlib"]
# ///
"""Baseline analysis: behaviour table + knows-vs-acts per layer. Runs on CPU.
Usage: uv run analyze.py <run_dir> <tag> [<tag> ...]   e.g. uv run analyze.py runs/baseline tiny-aya-global tiny-aya-fire
"""
import sys, os, re, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

RUN, TAGS = sys.argv[1], sys.argv[2:]
SCRIPTS = ["en", "hi_deva", "hi_rom", "pa_guru", "pa_rom"]
NAMES = {"en": "English", "hi_deva": "Hindi (Devanagari)", "hi_rom": "Hindi (romanized)",
         "pa_guru": "Punjabi (Gurmukhi)", "pa_rom": "Punjabi (romanized)"}
COL = {"en": "#222222", "hi_deva": "#1f77b4", "hi_rom": "#aec7e8", "pa_guru": "#d62728", "pa_rom": "#ff9896"}
NGUNI = re.compile(r"\b(ndiya|ngiya|andina|angikwazi|angikwaz|ukukunceda|ukukusiza|kodwa|ngaloo|nalokho|kwaye|akulunganga|ngoloo)\w*", re.I)
os.makedirs("results", exist_ok=True)
summary = {}

for TAG in TAGS:
    df = pd.read_json(f"{RUN}/{TAG}/gen_{TAG}_judged4.jsonl", lines=True); df["judge"] = df["judge4"]
    H = np.load(f"{RUN}/{TAG}/hidden_{TAG}.npy").astype(np.float32)   # [N, L, d]
    N, L, D = H.shape
    df["nguni"] = df.response.fillna("").str.contains(NGUNI)

    # ---------- 1. behaviour ----------
    def rates(g):
        h, s = g[g.label == 1], g[g.label == 0]
        return pd.Series({
            "harmful_refusal": (h.judge == "REFUSAL").mean(),
            "harmful_compliance": (h.judge == "COMPLIANCE").mean(),
            "harmful_partial": (h.judge == "PARTIAL").mean(),
            "harmless_over_refusal": (s.judge == "REFUSAL").mean(),
            "reply_in_xhosa_zulu": g.nguni.mean(),
        })
    beh = df.groupby("script").apply(rates).reindex(SCRIPTS)
    # bootstrap 95% CI on harmful refusal (resample prompts)
    rng = np.random.default_rng(0); ci = {}
    for s in SCRIPTS:
        v = (df[(df.script == s) & (df.label == 1)].judge == "REFUSAL").values.astype(float)
        bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        ci[s] = (np.percentile(bs, 2.5), np.percentile(bs, 97.5))
    beh["refusal_ci"] = [f"[{ci[s][0]:.2f}, {ci[s][1]:.2f}]" for s in SCRIPTS]
    print(f"\n===== {TAG} behaviour =====\n", beh.round(3).to_string(), flush=True)
    beh.to_csv(f"results/behaviour_{TAG}.csv")

    # ---------- 2. knows vs acts per layer ----------
    y = df.label.values; sc = df.script.values; en = sc == "en"
    knows = {s: [] for s in SCRIPTS}; acts = {s: [] for s in SCRIPTS}
    for l in range(L):
        X = H[:, l]
        mu, sd = X[en].mean(0), X[en].std(0) + 1e-6
        Z = (X - mu) / sd
        d = X[en & (y == 1)].mean(0) - X[en & (y == 0)].mean(0); d /= np.linalg.norm(d)
        p = X @ d
        base = p[en & (y == 0)].mean(); top = p[en & (y == 1)].mean()
        clf = LogisticRegression(max_iter=3000, C=0.05).fit(Z[en], y[en])
        for s in SCRIPTS:
            m = sc == s
            if s == "en":   # out-of-fold for English
                sc_en = np.zeros(m.sum()); idx = np.where(m)[0]
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(idx, y[idx]):
                    c = LogisticRegression(max_iter=3000, C=0.05).fit(Z[idx[tr]], y[idx[tr]])
                    sc_en[te] = c.decision_function(Z[idx[te]])
                knows[s].append(roc_auc_score(y[idx], sc_en))
            else:
                knows[s].append(roc_auc_score(y[m], clf.decision_function(Z[m])))
            # refusal signal on harmful prompts of this script, relative to English (0 = English harmless level, 1 = English harmful level)
            acts[s].append(float((p[m & (y == 1)].mean() - base) / (top - base + 1e-9)))

    # ---------- 3. does the refusal signal predict behaviour? ----------
    best = int(np.argmax([np.mean([knows[s][l] for s in SCRIPTS]) for l in range(L)]))
    mid = list(range(L // 3, 2 * L // 3))
    act_layer = mid[int(np.argmax([acts["en"][l] - np.mean([acts[s][l] for s in SCRIPTS[1:]]) for l in mid]))]
    X = H[:, act_layer]
    d = X[en & (y == 1)].mean(0) - X[en & (y == 0)].mean(0); d /= np.linalg.norm(d)
    harm = df.label == 1
    beh_auc = roc_auc_score((df[harm].judge == "REFUSAL").astype(int), X[harm.values] @ d)
    print(f"[{TAG}] layer with largest English-vs-other refusal-signal gap: {act_layer};  "
          f"refusal-direction projection predicts actual refusal on harmful prompts: AUROC={beh_auc:.3f}", flush=True)

    # ---------- plot ----------
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    for s in SCRIPTS:
        ax[0].plot(knows[s], color=COL[s], label=NAMES[s], lw=2)
        ax[1].plot(acts[s], color=COL[s], label=NAMES[s], lw=2)
    ax[0].set(title="KNOWS — English-trained harm probe (AUROC)", xlabel="layer", ylim=(0.4, 1.02))
    ax[0].axhline(0.5, ls=":", c="gray")
    ax[1].set(title="ACTS — refusal-direction signal on harmful prompts\n(0 = English harmless, 1 = English harmful)", xlabel="layer")
    ax[1].axhline(1, ls="--", c="gray"); ax[1].axhline(0, ls=":", c="gray"); ax[1].axvline(act_layer, ls=":", c="purple")
    ax[0].legend(fontsize=8)
    xs = np.arange(len(SCRIPTS))
    ax[2].bar(xs - 0.2, beh.harmful_refusal, 0.4, label="harmful refused", color="#2ca02c")
    ax[2].bar(xs + 0.2, beh.harmless_over_refusal, 0.4, label="harmless refused (over-refusal)", color="#9467bd")
    ax[2].set_xticks(xs, [NAMES[s].replace(" (", "\n(") for s in SCRIPTS], fontsize=8); ax[2].set_ylim(0, 1.05)
    ax[2].set(title="BEHAVIOUR (Aya Expanse 8B, 4-way judge)"); ax[2].legend(fontsize=8)
    fig.suptitle(f"{TAG}: knows vs. acts across scripts", fontsize=13); plt.tight_layout()
    plt.savefig(f"results/knows_vs_acts_{TAG}.png", dpi=150)
    summary[TAG] = {"behaviour": beh.drop(columns="refusal_ci").round(4).to_dict(), "refusal_ci": ci,
                    "knows_auroc": {s: [round(v, 4) for v in knows[s]] for s in SCRIPTS},
                    "acts": {s: [round(v, 4) for v in acts[s]] for s in SCRIPTS},
                    "act_layer": act_layer, "refusal_dir_predicts_behaviour_auroc": round(beh_auc, 4)}
    # compact per-layer print at a few layers
    for l in [6, 12, 18, act_layer, 24, 30, L - 1]:
        print(f"  L{l:2d} knows " + " ".join(f"{s}:{knows[s][l]:.2f}" for s in SCRIPTS) +
              " | acts " + " ".join(f"{s}:{acts[s][l]:.2f}" for s in SCRIPTS), flush=True)

json.dump(summary, open("results/summary.json", "w"), indent=1, default=float)
print("saved results/")
