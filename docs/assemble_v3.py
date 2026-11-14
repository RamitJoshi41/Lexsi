import re
v2 = open("docs/hf-article-v2.md").read()
def sec(text, start, end=None):
    i = text.index(start); j = text.index(end, i + len(start)) if end else len(text)
    return text[i:j].rstrip() + "\n\n"
front = open("docs/v3_front.md").read().rstrip() + "\n\n"
draft = open("docs/new_sections_draft.md").read()
back = open("docs/v3_back.md").read()
judges = sec(draft, "## For judges: one minute", "## Threat model").replace(
    "survives 4-bit quantization (TBD)", "same gain on the 4-bit model (39% → 76%) and on Tiny Aya Fire (47% → 76%)")
threat = sec(draft, "## Threat model", "## Over-refusal test sets").replace("## Threat model", "## 2. Threat model")
overref = sec(draft, "## Over-refusal test sets", "## Train / test splits and leak check").replace("## Over-refusal test sets", "### Over-refusal and retention test sets")
splits = sec(draft, "## Train / test splits and leak check", "## Compute and cost").replace("## Train / test splits and leak check", "### Train / test splits and leak check")
compute = sec(draft, "## Compute and cost", "## Confidence intervals").replace("## Compute and cost", "## 11. Compute, cost and ROI")
cis = sec(draft, "## Confidence intervals", "## 6.x Is it the script").replace("## Confidence intervals", "### 10.5 Confidence intervals")
translit = sec(draft, "## 6.x Is it the script", "## 6.x Filler-wrapper").replace("## 6.x Is it the script or the words? (transliteration test)", "### 10.9 Is it the script or the words? (transliteration test)")
filler = sec(draft, "## 6.x Filler-wrapper").replace("## 6.x Filler-wrapper robustness (gate v3.1 ablation)", "### 10.10 Filler-wrapper robustness (gate v3.1 ablation)")

p1 = sec(v2, "## 1. Problem and use case", "## 2. Approach in one picture").replace("## 1. Problem and use case", "## 1. Problem, use case and research questions")
p2 = sec(v2, "## 2. Approach in one picture", "## 3. Hackathon repos used").replace("## 2. Approach in one picture", "## 3. Approach in one picture")
p3 = sec(v2, "## 3. Hackathon repos used", "## 4. Models and licences").replace("## 3. Hackathon repos used — and what for", "## 5. Hackathon repos (kits) used and how they connect").replace("AuditKIT** 1.1.11", "AuditKIT** 1.1.12")
p3 += """**How they connect.** CuratorKIT gates and documents the benchmark that every later stage reads. Our HF-Jobs scripts generate replies and hidden states; the hidden states feed the knows-vs-acts analysis and the gate, and the same prompts are run through SafeTune's `CASTTrainer` so the baseline and our guard see identical inputs and decoding. AuditKIT closes the loop twice: its MMLU scenario checks the guarded model (bf16 and 4-bit) for capability loss, and our `codemix_safety` scenario plus multilingual scorers (`auditkit_codemix/`) let anyone re-run the safety evaluation through AuditKIT. Its built-in refusal detector was itself audited against our judge (0% recall on Indic scripts), which is why we added the multilingual detector.

"""
p4 = sec(v2, "## 4. Models and licences", "## 5. Datasets").replace("## 4. Models and licences", "## 6. Models and licences")
p4 = p4.replace("| `facebook/fasttext-language-identification` (LID-218) | Language of each reply | CC-BY-NC-4.0 |",
 "| `facebook/fasttext-language-identification` (LID-218) | Language of each reply | CC-BY-NC-4.0 |\n| `sentence-transformers/LaBSE` | Embedding-based leak check | Apache-2.0 |\n\nBase-model configs evaluated: Tiny Aya Global **bf16** (main), **nf4 4-bit** via `bitsandbytes` (deployment check, same guard files), and Tiny Aya Fire bf16 (transfer). No weights were modified.")
p5 = sec(v2, "## 5. Datasets", "## 6. Experiments and findings").replace("## 5. Datasets / benchmarks and licences", "## 7. Datasets, benchmarks and licences")
p5 += "**Languages covered:** English, Hindi and Punjabi, each in native script (Devanagari / Gurmukhi) and romanized Latin script (code-mixed with English loan-words); the language-ID probe uses reference sentences in 22 languages.\n\n"
p5 += "**CuratorKIT filtering:** every script of the benchmark went 188 → 188 rows through schema gate, exact dedup, MinHash dedup (0.85) and text cleaning — nothing removed, because parallel versions of a prompt are curated per script. Provenance files ship with the dataset.\n\n"
p5 += "**Attack prompt set** — grouped by attack family in §2 (script switch, romanization, transliteration, SMS spelling, typos, chatty wrappers, unseen harm categories).\n\n"
p5 += overref + splits
metrics = open("docs/v3_metrics.md").read().replace("## Metrics, and why these", "## 8. Metrics, and why these").rstrip() + "\n\n"
hp = "## 9. Method and experimental setup\n\n" + open("docs/v3_hparams.md").read().replace("### Setup and hyperparameters, per repo", "### Setup and hyperparameters, per repo").rstrip() + "\n\n**Seeds and determinism.** Greedy decoding everywhere; data splits, bootstrap and MMLU sampling use seed 0. **Ablations:** gate v2 vs v3 vs v3.1, layer and threshold sweep, gated vs always-on, English vs in-language vectors, English vs all-script CAST calibration, bf16 vs nf4, Global vs Fire.\n\n"
res = sec(v2, "## 6. Experiments and findings", "## 7. Limitations").replace("## 6. Experiments and findings", "## 10. Results and findings")
for a, b in [("### 6.1 ", "### 10.1 "), ("### 6.2 ", "### 10.2 "), ("### 6.3 ", "### 10.3 "), ("### 6.4 ", "### 10.4 "), ("### 6.5 ", "### 10.6 "), ("### 6.6 ", "### 10.7 "), ("### 6.7 ", "### 10.8 "), ("### 6.8 ", "### 10.14 ")]:
    res = res.replace(a, b)
res = re.sub(r"§6\.(\d)", lambda m: "§10." + {"1": "1", "2": "2", "3": "3", "4": "4", "5": "6", "6": "7", "7": "8", "8": "14"}[m.group(1)], res)
i = res.index("### 10.6 "); res = res[:i] + cis + res[i:]
i = res.index("### 10.14 ")
newres = translit + filler + """### 10.11 Does it survive deployment? (4-bit quantization)

We loaded Tiny Aya Global in 4-bit (`bitsandbytes` nf4, bf16 compute) and applied **the same gate and vector fit on bf16 hidden states** — no refit (`job_steer2.py` with `QUANT=nf4`).

| Held-out | bf16 | **nf4** |
|---|---|---|
| Plain: refused EN / romanized HI / romanized PA | 74% / 30% / 20% | 74% / 30% / 13% |
| Usable refusal, non-EN harmful, plain → guarded | 38% → 76% | **39% → 76%** [68, 83] (gain +37 [+28, +46]) |
| Gate fires, harmful / harmless | 93% / 2% | 91% / 1.7% |
| Harmless refused (/240), plain → guarded | 32 → 35 | 31 → 35 |
| MMLU (400), plain → guarded | 50.5% → 50.5% | 48.0% → 48.0% (fired 0 / 400) |
| Peak memory / throughput (A10G, batch 16) | 7.7 GB / 330 tok/s | **3.7 GB** / 238 tok/s |

The romanization gap is the same (or slightly worse for Punjabi) after quantization, and the guard's effect is unchanged: the gate reads a direction that 4-bit weights preserve.

### 10.12 Does it transfer? (Tiny Aya Fire)

We re-ran the identical recipe on Tiny Aya Fire (`gate_v3_fire.py`: same data, splits and selection rule; Fire's own diff-in-means vector). The selection picked **layer 13** again.

| Tiny Aya Fire, held-out | Plain | + guard |
|---|---|---|
| Refused: EN / romanized HI / romanized PA | 85% / 35% / 30% | 96% / 89% / 78% |
| **Usable refusal, non-EN harmful (184)** | 47% [39, 54] | **76% [69, 82]** (+29 [+20, +38]) |
| Complied, non-EN harmful | 30% | 7% |
| India-specific harms, usable (80) | 60% [49, 71] | **84% [73, 93]** (+24 [+11, +37]) |
| Harmless refused | 35 / 240 | 38 / 240 |
| Gate: held-out harmful / harmless / everyday (600) / romanized everyday | — | 92% / 1.7% / 0.3% / 2.7% |
| MMLU (400) | 51.0% | 51.0% (fired 0 / 400) |

Global's gate applied to Fire **without refitting** still catches 90% of held-out harmful prompts (66% of unseen categories vs 77% refit): the harm representation the gate reads is largely shared between the two models. The guard is released as `sunit12/tiny-aya-fire-codemix-refusal-guard`. (SafeTune CAST was not re-run on Fire.)

### 10.13 The evaluation as a reusable AuditKIT component

`auditkit_codemix/` packages the benchmark as an AuditKIT `Scenario` (`codemix_safety`) with a `MultilingualRefusalDetector` and judge-free scorers (`usable_refusal`, `any_refusal`, `harmless_refusal`, `degenerate`). Replaying our saved held-out generations through `ak.evaluate` on CPU (`example_replay.py`): usable refusal (non-EN harmful) plain 27%, guard **63%**, CAST 38%; degenerate replies plain 4%, guard 2%, **CAST 46%**. The keyword proxy is stricter than the LLM judge but preserves the ranking — whereas a plain refusal rate (CAST 84%, guard 70%) would rank CAST first. It is a cheap, offline regression test for multilingual safety.

"""
res = res[:i] + newres + res[i:]
lim = sec(v2, "## 7. Limitations", "## 8. With more credits").replace("## 7. Limitations", "## 12. Limitations (known weaknesses)")
lim = lim.replace("- The gate falsely flags 44% of harmless romanized prompts wrapped in an unseen filler template — it partly keys on chatty \"please help\" framing.",
 "- Chatty wrappers still cause false triggers: gate v3 flags 44% of harmless romanized prompts in the *\"bhai plz help…\"* wrapper and 17.5% across five other unseen wrappers (v3.1: 26% / 11.5%, at a small recall cost).")
lim = lim.replace("- The judge is an Aya model; validated against native-speaker labels on 29 items (§6.7).", "- The judge is an Aya model; validated against native-speaker labels on only 29 items (§10.8).")
lim = lim.replace("about a quarter of the refusals it restores on romanized Punjabi", "about a quarter of the refusals it restores on romanized Punjabi")
lim += "- SafeTune CAST was compared on Tiny Aya Global (bf16) only, not on Fire or the 4-bit model.\n- The Punjabi transliteration result is inconclusive because informal romanized Punjabi has no standard spelling (§10.9).\n- No adaptive attacker: a white-box attacker who knows the gate could likely evade a linear probe (§2).\n\n"
lim = lim.replace("\n\n- SafeTune", "\n- SafeTune")
credits = sec(back, "## With more credits", "## Documentation and reproduction").replace("## With more credits", "## 13. With more credits: plan and expected results")
art = sec(v2, "## 9. Artifacts", "## 10. Team").replace("## 9. Artifacts", "## 14. Artifacts")
art = art.replace("`sunit12/tiny-aya-global-codemix-refusal-guard` (gate v3 + v2 weights, refusal vectors for all 36 layers, `guard.py`).",
 "`sunit12/tiny-aya-global-codemix-refusal-guard` (gates v3 / v3.1 / v2, refusal vectors for all 36 layers, `guard.py`, 4-bit results) and `sunit12/tiny-aya-fire-codemix-refusal-guard` (the same recipe re-fit on Tiny Aya Fire).")
art += "- **Demo / Space:** explorer of every held-out output (plain vs guard vs CAST, with judge, language and gate scores; filters for fixed / guard-failed / CAST-loop / over-refusal): https://sunit12-knows-but-doesnt-refuse-explorer.static.hf.space\n- **Slides:** 11-slide demo deck with before/after examples; 2-minute demo script in `docs/demo_script_2min.md`.\n- **AuditKIT component:** `auditkit_codemix/` (GitHub and `code/` in the logs repo).\n\n"
repro = sec(back, "## Documentation and reproduction", "## LLM usage statement").replace("## Documentation and reproduction", "## 15. Documentation and reproduction")
llm = sec(back, "## LLM usage statement").replace("## LLM usage statement", "## 17. LLM usage statement")
team = "## 16. Team\n\n{{TEAM}}\n\n**Author contributions:** {{CONTRIBUTIONS}}\n\n**Acknowledgements:** Lexsi Labs and Cohere Labs for the hackathon, kits and models; Hugging Face Jobs for compute.\n\n"
related = """## 4. Related work

- **Multilingual and code-mixed safety.** Safety training transfers unevenly across languages (Aakanksha et al. 2024, *The Multilingual Alignment Prism*); code-mixed and transliterated prompts weaken guardrails (Cohere Expedition Aya, *Mix, Fine-Tune, Break*; Aswal & Jaiswal 2025, *Haet Bhasha aur Diskrimineshun*). These works measure or attack; we localise the failure and repair it.
- **Refusal as a direction.** Refusal is mediated by a single residual-stream direction (Arditi et al. 2024); we use it as a measuring stick (*acts*) and as the steering vector.
- **Conditional steering.** CAST (Lee et al. 2024) gates steering on a condition vector; SafeTune's `CASTTrainer` is our baseline. Our difference: a supervised probe gate trained on everyday and romanized traffic, and evaluation by *usable* refusals.
- **Latent language.** Models route non-English input through English-like latent spaces (Wendler et al. 2024, *Do Llamas Work in English?*); our language-ID probe shows the opposite failure — romanized input is mapped to the *wrong* language.
- **Over-refusal.** XSTest (Röttger et al. 2024) supplies our harmless-but-scary prompts.

"""
refs = sec(v2, "### References").replace("### References", "## References")
appendix = """## Appendix A — Judge prompt (Aya Expanse 8B, greedy, ≤6 new tokens; reply truncated to 1,000 characters)

```
""" + open("job_judge4.py").read().split('INSTR = """')[1].split('"""')[0].strip() + """
```

## Appendix B — Extra tables

Full per-script tables, bootstrap CIs, leak-check examples, transliteration and filler results: `results/final_v2.md`, `results/bootstrap_ci.md`, `results/leak_check_examples.jsonl`, `results/translit.md`, `results/gate_v3_1.json`, `results/gate_v3_fire.json` (logs repo and GitHub).
"""
out = front + judges + p1 + threat + p2 + related + p3 + p4 + p5 + metrics + hp + res + compute + lim + credits + art + repro + team + llm + refs + appendix
open("docs/hf-article-v3.md", "w").write(out)
print(len(out.split()), "words")
