# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
"""Day-4 article figures from results/*.json:
 fig_circuits.png  — CircuitKIT: circuit overlap with English and total refusal attribution, per script
 fig_causal2.png   — causal v2: twin-sized additions at every position, by layer (+ all-layer ablation panel)
 fig_robustness.png — unseen languages and adaptive attacks, base vs hook guard (vs LoRA when present)"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29d", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#d6d5d0", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
J = lambda p: json.load(open(p))

# ---- 1. circuits
s = J("runs/circuit/tiny-aya-global/circuit_summary.json"); sc = J("runs/circuit/tiny-aya-global/circuit_scores.json")
langs = ["en", "hi", "hi-Latn", "pa-Latn"]; names = ["English", "Hindi\n(Devanagari)", "Hinglish\n(romanized)", "Punjabi\n(romanized)"]
jac = [1.0] + [s["jaccard"].get(f"en|{l}|top20", 0) for l in langs[1:]]
mass = [sum(abs(v) for v in sc[l]["scores"].values()) for l in langs]
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
for a, vals, t, fmt in [(ax[0], jac, "Top-20 refusal-circuit overlap with English (Jaccard)", "{:.2f}"),
                        (ax[1], mass, "Total refusal attribution (EAP-IG, |score| sum)", "{:.0f}")]:
    b = a.bar(names, vals, color=[GRAY] + [BLUE] * 3, width=0.6)
    for r, v in zip(b, vals): a.text(r.get_x() + r.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=9, color=INK)
    a.set_title(t, fontsize=10, color=INK, loc="left"); a.grid(axis="x", visible=False)
ax[0].set_ylim(0, 1.12); ax[1].set_ylim(0, max(mass) * 1.15)
fig.suptitle("CircuitKIT: the refusal circuit fades where refusal fails (Tiny Aya Global, 64 minimal pairs per language)", fontsize=10.5, x=0.01, ha="left", color=INK)
plt.tight_layout(); plt.savefig("results/fig_circuits.png", dpi=170); plt.close()

# ---- 2. causal v2 + all-layer ablation
c = J("results/causal2.json"); ab = J("results/ablate_all.json") if os.path.exists("results/ablate_all.json") else None
L = [10, 13, 14, 18]; g = c["non-EN harmful"]; base = g["none"]["-1"]["refusal"]
fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.9), gridspec_kw={"width_ratios": [1.6, 1]})
a = ax[0]; a.axhspan(base[1], base[2], color=GRAY, alpha=0.25, lw=0); a.axhline(base[0], color=GRAY, lw=1)
a.text(9.7, 0.325, f"grey band: no intervention, {base[0]:.0%} [{base[1]:.0%}, {base[2]:.0%}]", fontsize=8.5, color=INK2)
for k, lab, col, ls in [("refusal_twin", "+ refusal component (twin-sized)", BLUE, "-"), ("harm_twin", "+ harm component (twin-sized)", ORANGE, "-"),
                        ("harm_matched", "+ harm direction, refusal-sized", ORANGE, "--"), ("random_matched", "+ random direction, refusal-sized", AQUA, ":")]:
    y = [g[k][str(l)]["refusal"] for l in L]
    a.errorbar(L, [v[0] for v in y], yerr=[[v[0] - v[1] for v in y], [v[2] - v[0] for v in y]], color=col, ls=ls, marker="o", ms=5, lw=2, capsize=3)
    a.text(L[-1] + 0.25, y[-1][0], lab, fontsize=8.5, color=INK, va="center")
a.set_xlim(9.5, 23.8); a.set_ylim(0.3, 0.92); a.set_xticks(L); a.set_xlabel("layer of the intervention (all positions)")
a.set_ylabel("judged refusal, code-mixed harmful"); a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
a.set_title("Sufficiency: only the refusal direction restores refusal", fontsize=10, loc="left", color=INK)
if ab:
    a = ax[1]; conds = [("none", "nothing"), ("ablate_r_all", "refusal dir."), ("ablate_p_all", "harm dir."), ("ablate_q_all", "random dir.")]
    x = np.arange(len(conds)); w = 0.38
    for off, grp, col, lab in [(-w / 2, "EN harmful", GRAY, "English"), (w / 2, "non-EN harmful", BLUE, "code-mixed")]:
        v = [ab[grp][k]["refusal"] for k, _ in conds]
        bars = a.bar(x + off, [t[0] for t in v], w * 0.95, color=col, label=lab)
        for r, t in zip(bars, v): a.text(r.get_x() + r.get_width() / 2, t[0] + 0.01, f"{t[0]:.0%}", ha="center", fontsize=8, color=INK)
    a.set_xticks(x); a.set_xticklabels([n for _, n in conds], fontsize=8.5); a.set_ylim(0, 1.22); a.grid(axis="x", visible=False)
    a.set_yticks([0, .2, .4, .6, .8, 1.0]); a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); a.legend(fontsize=8, frameon=False, loc="upper center", ncol=2)
    a.set_title("Necessity: direction removed at every layer", fontsize=10, loc="left", color=INK)
plt.tight_layout(); plt.savefig("results/fig_causal2.png", dpi=170); plt.close()

# ---- 3. robustness
u = J("results/unseen_lora.json") if os.path.exists("results/unseen_lora.json") else J("results/unseen_langs.json")
at = J("results/attacks_lora.json") if os.path.exists("results/attacks_lora.json") else J("results/attacks.json")
KEEP = ["control", "guard_v3", "lora_v2c"]
def panel(a, res, groups, gl, title):
    sysl = [k for k in KEEP if k in res]; x = np.arange(len(groups)); w = 0.8 / len(sysl)
    cols = {"control": GRAY, "guard_v3": BLUE, "lora_v2c": ORANGE}; lab = {"control": "plain Tiny Aya", "guard_v3": "fix 1: gated steer", "lora_v2c": "fix 2: circuit-guided LoRA"}
    for i, k in enumerate(sysl):
        v = [res[k]["by_group"][gg]["usable"] for gg in groups]
        a.bar(x - 0.4 + w * (i + 0.5), v, w * 0.92, color=cols.get(k, AQUA), label=lab.get(k, k.replace("_", " ")))
    a.set_xticks(x); a.set_xticklabels(gl, fontsize=8.5); a.set_ylim(0, 1.05); a.grid(axis="x", visible=False)
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); a.set_title(title, fontsize=10, loc="left", color=INK)
fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.7))
panel(ax[0], u, ["bn_rom", "mr_rom", "ta_rom", "ur_rom"], ["Bengali", "Marathi", "Tamil", "Urdu"], "Languages the guard never saw (romanized)")
panel(ax[1], at, ["scriptmix", "roleplay", "translate", "prefix", "fiction"], ["script-mix", "role-play", "translate", "prefix", "fiction"], "Jailbreak wrappers (Hinglish)")
ax[0].set_ylabel("usable refusal, harmful prompts"); ax[1].legend(fontsize=8, frameon=False, loc="upper right")
plt.tight_layout(); plt.savefig("results/fig_robustness.png", dpi=170); plt.close()
# ---- 4. v4 headline: plain vs guard vs LoRA on the metrics that matter
h = J("results/heldout_lora.json"); NONEN = ["hi_deva", "hi_rom", "pa_guru", "pa_rom"]
nonen = lambda k: np.mean([h[k]["by_group"][g]["usable"] for g in NONEN])
harmless = lambda k: np.mean([h[k]["by_group"][g]["refused_harmless"] for g in NONEN])
mm = {"control": 0.505, "guard_v3": 0.505}
for k, f in [("lora_v2c", "runs/capability_tiny-aya-global_tiny-aya-global-codemix-refusal-lora-v2-circuit.json")]:
    if os.path.exists(f): mm[k] = J(f)["guarded"]["mcq_acc"]
metrics = [("usable refusal,\nheld-out non-English", lambda k: nonen(k)), ("usable refusal,\n4 unseen languages", lambda k: u[k]["usable_all"]),
           ("harmless prompts\nrefused (non-English)", lambda k: harmless(k)), ("MMLU\n(capability)", lambda k: mm.get(k, np.nan))]
fig, ax = plt.subplots(figsize=(10.2, 3.9)); x = np.arange(len(metrics)); w = 0.26
for i, (k, col, lab) in enumerate([("control", GRAY, "plain Tiny Aya Global"), ("guard_v3", BLUE, "fix 1: gated steer (no training)"), ("lora_v2c", ORANGE, "fix 2: circuit-guided LoRA (no hooks)")]):
    v = [f(k) for _, f in metrics]; bars = ax.bar(x + (i - 1) * w, v, w * 0.92, color=col, label=lab)
    for r, t in zip(bars, v): ax.text(r.get_x() + r.get_width() / 2, t + 0.012, f"{t:.0%}", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(x); ax.set_xticklabels([m for m, _ in metrics], fontsize=9); ax.set_ylim(0, 1.08); ax.grid(axis="x", visible=False)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.legend(fontsize=8.5, frameon=False, loc="upper right", ncol=1)
ax.set_title("Knows, But Doesn't Refuse — Tiny Aya Global on romanized / code-mixed Hindi & Punjabi (held-out)", fontsize=10.5, loc="left", color=INK)
plt.tight_layout(); plt.savefig("results/fig_headline_v4.png", dpi=170); plt.close()
print("saved fig_circuits, fig_causal2, fig_robustness, fig_headline_v4")
