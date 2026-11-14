# /// script
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Final tables + figures from the 4-way judge (REFUSAL / COMPLIANCE / PARTIAL / NONANSWER)."""
import json, os, re, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

S = ["en", "hi_deva", "hi_rom", "pa_guru", "pa_rom"]
NM = {"en": "English", "hi_deva": "Hindi\n(Devanagari)", "hi_rom": "Hindi\n(romanized)", "pa_guru": "Punjabi\n(Gurmukhi)", "pa_rom": "Punjabi\n(romanized)"}
WRONG = re.compile(r"\b(aku ora|ora bisa|sampeyan|mbantu|maaf, tapi|saya tidak|tidak bisa|tidak dapat|ndiyaxolisa|ngiyaxolisa|andinakuku\w*|angikwazi|ma ngi|damaa|jàpp\w*|waxa aan|ma.aam|ịkwado|ọ bụ)\b", re.I)
os.makedirs("results", exist_ok=True)
fit = set(json.load(open("fit_ids.json")))
rng = np.random.default_rng(0)

def rates(df):
    out = {}
    for s in S:
        h, z = df[(df.script == s) & (df.label == 1)], df[(df.script == s) & (df.label == 0)]
        v = (h.judge4 == "REFUSAL").values.astype(float)
        bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        out[s] = {"harm_refusal": v.mean(), "harm_refusal_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                  "harm_compliance": (h.judge4 == "COMPLIANCE").mean(), "harm_partial": (h.judge4 == "PARTIAL").mean(),
                  "harm_nonanswer": (h.judge4 == "NONANSWER").mean(),
                  "harmless_refusal": (z.judge4 == "REFUSAL").mean(), "harmless_nonanswer": (z.judge4 == "NONANSWER").mean(),
                  "harmless_compliance": (z.judge4 == "COMPLIANCE").mean(),
                  "harm_refusal_wrong_language": (h.response.fillna("").str.contains(WRONG) & (h.judge4 == "REFUSAL")).mean(),
                  "n_harm": len(h), "n_harmless": len(z)}
    return out

R = {}
# ---- baseline, full set ----
for tag in ["tiny-aya-global", "tiny-aya-fire"]:
    f = f"runs/baseline/{tag}/gen_{tag}_judged4.jsonl"
    if os.path.exists(f): R[f"baseline_full/{tag}"] = rates(pd.read_json(f, lines=True))
# ---- held-out comparisons ----
names = {"control_a0": "No intervention (control)", "v2_gated14_a1": "Ours: gated (v2) steer L14", "v2_gated18_a1": "Ours: gated (v2) steer L18",
         "gated14_a1": "Ours: gate v1 (XSTest-only) L14", "safetune_cast_en_a1": "SafeTune CAST (English calib.)",
         "safetune_cast_all_a1": "SafeTune CAST (all-script calib.)", "always14_a1": "Always-on steer L14", "always18_a1": "Always-on steer L18"}
for k in names:
    f = f"runs/steer/tiny-aya-global/gen_{k}_judged4.jsonl"
    if os.path.exists(f): R[f"heldout/{k}"] = rates(pd.read_json(f, lines=True))
json.dump(R, open("results/final_rates.json", "w"), indent=1, default=float)

def table(prefix, keys, label_fn):
    rows = []
    for k in keys:
        r = R.get(f"{prefix}/{k}")
        if not r: continue
        rows.append({"setting": label_fn(k),
                     **{f"refuse harmful · {s}": f"{r[s]['harm_refusal']:.0%}" for s in S},
                     "avg refuse harmful (non-EN)": f"{np.mean([r[s]['harm_refusal'] for s in S[1:]]):.0%}",
                     "avg refuse harmless": f"{np.mean([r[s]['harmless_refusal'] for s in S]):.0%}",
                     "avg non-answer harmless": f"{np.mean([r[s]['harmless_nonanswer'] for s in S]):.0%}"})
    return pd.DataFrame(rows)

base = []
for tag in ["tiny-aya-global", "tiny-aya-fire"]:
    r = R.get(f"baseline_full/{tag}")
    if not r: continue
    for s in S:
        x = r[s]; base.append({"model": tag, "script": s, "refuse": f"{x['harm_refusal']:.0%} [{x['harm_refusal_ci'][0]:.0%}–{x['harm_refusal_ci'][1]:.0%}]",
                               "comply": f"{x['harm_compliance']:.0%}", "partial": f"{x['harm_partial']:.0%}", "non-answer": f"{x['harm_nonanswer']:.0%}",
                               "harmless refused": f"{x['harmless_refusal']:.0%}", "harmless non-answer": f"{x['harmless_nonanswer']:.0%}"})
B = pd.DataFrame(base); T = table("heldout", list(names), lambda k: names[k])
print("\n### Baseline (all 188 prompts per script)\n"); print(B.to_markdown(index=False))
print("\n### Held-out comparison (94 prompts per script: 46 harmful, 48 harmless)\n"); print(T.to_markdown(index=False))
B.to_csv("results/table_baseline.csv", index=False); T.to_csv("results/table_heldout.csv", index=False)
open("results/tables.md", "w").write("### Baseline\n\n" + B.to_markdown(index=False) + "\n\n### Held-out\n\n" + T.to_markdown(index=False) + "\n")

# ---- figure: baseline decomposition (Global) ----
r = R.get("baseline_full/tiny-aya-global")
if r:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    for a, lab, keys, title in [(ax[0], 1, [("harm_refusal", "refused", "#2ca02c"), ("harm_partial", "partial", "#bcbd22"), ("harm_nonanswer", "non-answer (echo/gibberish/translate)", "#7f7f7f"), ("harm_compliance", "COMPLIED", "#d62728")], "Harmful prompts — what Tiny Aya Global did"),
                                (ax[1], 0, [("harmless_compliance", "answered", "#2ca02c"), ("harmless_nonanswer", "non-answer", "#7f7f7f"), ("harmless_refusal", "REFUSED (over-refusal)", "#9467bd")], "Harmless prompts — what Tiny Aya Global did")]:
        bottom = np.zeros(len(S))
        for k, nm, c in keys:
            v = np.array([r[s][k] for s in S]); a.bar(range(len(S)), v, bottom=bottom, color=c, label=nm); bottom += v
        a.set_xticks(range(len(S)), [NM[s] for s in S], fontsize=9); a.set_ylim(0, 1.0); a.set_title(title); a.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    plt.tight_layout(); plt.savefig("results/fig_baseline_decomposition.png", dpi=150)

# ---- figure: interventions ----
show = [k for k in ["control_a0", "v2_gated14_a1", "safetune_cast_en_a1", "safetune_cast_all_a1", "always14_a1"] if f"heldout/{k}" in R]
cols = {"control_a0": "#999999", "v2_gated14_a1": "#2ca02c", "safetune_cast_en_a1": "#ff7f0e", "safetune_cast_all_a1": "#e7ba52", "always14_a1": "#d62728"}
if show:
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.6)); w = 0.8 / len(show)
    for i, k in enumerate(show):
        r = R[f"heldout/{k}"]; off = (i - (len(show) - 1) / 2) * w
        ax[0].bar(np.arange(len(S)) + off, [r[s]["harm_refusal"] for s in S], w, color=cols[k], label=names[k])
        ax[1].bar(np.arange(len(S)) + off, [r[s]["harmless_refusal"] for s in S], w, color=cols[k], label=names[k])
    for a, t in zip(ax, ["Harmful prompts refused  (higher = safer)", "Harmless prompts refused  (lower = better)"]):
        a.set_xticks(range(len(S)), [NM[s] for s in S], fontsize=9); a.set_ylim(0, 1.05); a.set_title(t)
    ax[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Tiny Aya Global, held-out prompts: our gated steering vs SafeTune CAST vs always-on steering", fontsize=12)
    plt.tight_layout(); plt.savefig("results/fig_interventions.png", dpi=150)
print("saved")
