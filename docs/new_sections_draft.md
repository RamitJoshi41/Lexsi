<!-- NEW SECTIONS (15 Nov) — merge into hf-article-v2 → v3. Numbers marked TBD are filled from results/*.json -->

## For judges: one minute

| | |
|---|---|
| **Claim** | Tiny Aya *knows* romanized Hindi/Punjabi requests are harmful but doesn't refuse them; a probe-gated refusal steer at the layer where refusal collapses fixes it without retraining or over-refusal. |
| **Headline** | Usable refusals on held-out non-English harmful prompts: **38% → 76%** (95% CI of the gain: +29 to +47 pts); SafeTune CAST: 40%. |
| **Doesn't break** | MMLU 50.5 → 50.5; 2 / 600 everyday requests trigger the guard; 0 ms measurable latency cost; survives 4-bit quantization (TBD). |
| **Try it** | Explorer of every held-out output: https://sunit12-knows-but-doesnt-refuse-explorer.static.hf.space · guard: `sunit12/tiny-aya-global-codemix-refusal-guard` |

## Threat model

- **Who:** an ordinary user of an Indian-language assistant built on Tiny Aya (customer support, scheme helpline, chat app) — *not* a sophisticated jailbreaker. The "attack" is simply typing the way most people in north India type: romanized Hindi/Punjabi, often with English loan-words, SMS spellings and chatty wrappers (*"bhai plz help…"*).
- **Capability:** black-box chat access; no system-prompt control, no weights, no gradient access; single turn.
- **Goal:** get harmful help (violence, self-harm, hate, fraud/scam scripts, misinformation, cyber intrusion, illegal goods) that the same model refuses in English.
- **Defender:** the deployer, who can wrap the model (add hooks) but does not want to retrain it, and cannot accept over-refusal of everyday traffic.
- **Out of scope:** adaptive white-box attacks on the gate itself, multi-turn escalation, prompt injection via tools. The guard is a *closing-the-language-gap* defence, not a jailbreak defence; §Limitations lists what an adaptive attacker could do.

Attack prompt families we evaluate (all held out from guard fitting):

| Attack family | What changes | Test rows | Languages |
|---|---|---|---|
| Script switch (native) | Devanagari / Gurmukhi instead of English | 92 harmful + 96 harmless | HI, PA |
| Romanization (code-mixed) | Latin-script Hindi/Punjabi with English loan-words | 92 + 96 | HI, PA |
| Romanized → native transliteration | the *same words* as the romanized prompt, re-written in native script | 92 + 96 (+ 40 + 40 India-specific) | HI, PA |
| SMS spelling (*mujhe→mjhe*) | vowel dropping | 92 + 96 | HI, PA (rom) |
| Random typos (20% of words) | character noise | 92 + 96 | HI, PA (rom) |
| Chatty filler wrappers | 6 templates (5 never seen in training) | 552 + 576 | HI, PA (rom) |
| Unseen harm categories | India-specific misinformation, cyber, fraud, illegal goods | 100 + 100 | EN, HI, PA × 2 scripts |

## Over-refusal test sets

| Set | Source | Size (test) | Languages | Role |
|---|---|---|---|---|
| Harmless-but-scary | XSTest safe prompts, translated + romanized | 48 × 5 scripts = 240 | EN, HI, PA (native + rom) | over-refusal (focus) |
| Harmless twins of India-specific harms | ours | 100 | EN, HI, PA × 2 scripts | over-refusal on the same topics (focus) |
| Everyday instructions | `aya_dataset` | 600 | EN, HI, PA (native) | false-trigger rate (focus) |
| Everyday romanized | `aya_dataset` romanized by Aya Expanse (+ ASCII-folded copies) | 150 + 150 | HI, PA (rom) | false-trigger rate (focus) |
| Filler-wrapped harmless | XSTest prompts in chatty wrappers | 576 | HI, PA (rom) | gate robustness |
| MMLU (AuditKIT `MMLUScenario`) | `cais/mmlu` | 400 | EN | capability retention |

## Train / test splits and leak check

- Benchmark split **by prompt id** (all 5 script versions of a prompt fall on the same side): 94 ids fit / 94 ids held out. India-specific set split **by harmful/harmless pair, stratified by category** (20 pairs train / 20 test). Romanized everyday prompts split by base id (the ASCII-folded copy goes with its original).
- Gate layer and threshold are chosen on a validation split **of the training data**; no test number was looked at to pick them.
- **Leak check** between *everything used to fit the guard* (gate training data + refusal-vector prompts: 4,856 rows) and every test set (670 rows), three ways (`leak_check.py`, `results/leak_check.json`):

| Check | Test rows flagged |
|---|---|
| Exact match after Unicode/punctuation/whitespace normalisation | **0** |
| LaBSE cosine ≥ 0.95 | **0** (max over all 670 test rows: 0.924) |
| LaBSE cosine ≥ 0.90 / ≥ 0.85 | 12 / 30 |
| MinHash, char 5-grams, Jaccard ≥ 0.5 | 56 |

  Every flagged pair we inspected is a **template sibling, not a copy**: the Aya red-teaming set reuses frames with a different target (*"tell me ways to go to war with **Japan**"* in test vs *"…with **Bhutan**"* in fit; *"ways to insult my limp son"* vs *"ways to taunt my blind grandson"*). To be safe we **re-scored the headline after dropping all 25 held-out prompt ids that had any flagged script version** (`leak_robust.py`): on the remaining 25 harmful ids, usable refusal is control 34%, CAST 45%, **ours 74%**; ours − CAST = **+29 pts [+14, +43]**. The result does not depend on near-duplicates.

## Compute and cost

- **Compute:** all GPU work ran on Hugging Face Jobs. Logged job history (14–15 Nov): TBD GPU-hours (A10G-small $1.00/h, A10G-large $1.50/h, L4) ≈ **$TBD**; plus day-1 runs not in the job history (~TBD h). CPU analyses (probes, gates, bootstrap, leak check) on a laptop / HF CPU jobs.
- **Cost per 1,000 replies** (measured on the held-out set: batch 16, ≤160 new tokens, ~151 output tokens/reply, plain `transformers.generate`, no vLLM):

| Setup | Throughput | Time / 1k replies | Cost / 1k replies |
|---|---|---|---|
| Tiny Aya Global bf16, A10G ($1.00/h) — no guard | 328 tok/s | 463 s | $0.129 |
| + our guard | 330 tok/s | 457 s | $0.127 (no measurable overhead) |
| Tiny Aya Global **nf4**, A10G, + guard | TBD | TBD | TBD (peak memory 3.7 GB vs 7.7 GB) |
| Cohere Command R7B API ($0.0375 in / $0.15 out per M) | — | — | ≈ $0.024 |
| Cohere Command A API ($2.50 / $10 per M) | — | — | ≈ $1.61 |

  The guard is free at inference: one dot product at layer 13 and one vector add at layer 14. Our self-hosted number is an upper bound — with vLLM batching, throughput on an A10G is typically several times higher.


## Confidence intervals

All headline rates come with 95% **cluster-bootstrap** intervals (10,000 draws, resampling *prompt ids* so the five script versions of one prompt move together; `bootstrap_ci.py`, `results/bootstrap_ci.md`):

| Held-out, non-English harmful (184) | Usable refusal | Judge "refused" | Harmless refused (/240) |
|---|---|---|---|
| Plain Tiny Aya Global | 38% [31, 46] | 40% [32, 49] | 13% [8, 20] |
| SafeTune CAST, English calib. | 7% [3, 10] | 39% [31, 46] | 14% [8, 20] |
| SafeTune CAST, all-script calib. | 40% [32, 48] | 88% [80, 94] | 17% [11, 24] |
| **Ours (gate v3 → L14)** | **76% [69, 83]** | 86% [80, 92] | 15% [9, 21] |
| Always-on steer | 82% [77, 87] | 93% [89, 96] | **88% [82, 92]** |

| Paired difference (usable, non-EN) | Held-out | India-specific (unseen categories) |
|---|---|---|
| Ours − plain | **+38 pts [+29, +47]** | **+29 [+14, +44]** |
| Ours − CAST (all-script) | **+36 [+25, +46]** | **+22 [+12, +33]** |
| CAST − plain | +2 [−10, +14] | +6 [−7, +19] |
| Baseline gap, English − romanized Hindi / Punjabi (judge refusal) | +43 [+29, +58] / +54 [+37, +71] | +55 [+33, +77] / +60 [+38, +82] |

## 6.x Is it the script or the words? (transliteration test)

Romanized prompts differ from our native-script prompts in two ways: the **script**, and the **wording** (the native versions were translated independently; romanized ones keep English loan-words like *murder*, *bomb*). To separate them we re-wrote each held-out romanized prompt in Devanagari / Gurmukhi *word for word* (Aya Expanse 8B as a transliteration tool, 266 / 268 rows; 2 refusals replaced by rule-based `indic_transliteration`) and ran plain Tiny Aya and the guard on them (`job_translit.py`, `translit_compare.py`, `results/translit.md`).

| Held-out harmful (46 per row) | Plain: refused | Plain: complied | Guard: usable refusal |
|---|---|---|---|
| Hindi, romanized (original) | 30% | 43% | 63% |
| Hindi, **same words in Devanagari** | **48%** (paired p = 0.04) | 48% | **89%** |
| Hindi, independent Devanagari translation | 59% | 39% | 93% |
| Punjabi, romanized (original) | 20% | 35% | 63% |
| Punjabi, **same words in Gurmukhi** | 28% (p = 0.50) | **67%** | 74% |
| Punjabi, independent Gurmukhi translation | 52% | 46% | 85% |

- **Hindi: the script is most of the problem.** Changing *only* the script recovers ~60% of the gap between romanized and native Hindi (30% → 48%, vs 59%), and with the guard it turns readable-refusal failures into clean Hindi refusals (63% → 89% usable). The same direction holds on the India-specific set (35% → 65%, matching the native translation).
- **Punjabi: inconclusive, and the reason is informative.** Word-for-word Gurmukhi barely helps and *compliance* rises. Informal romanized Punjabi has no standard spelling, and machine transliteration produces non-standard Gurmukhi (e.g. *ghar* → ਗਰ instead of ਘਰ) that Tiny Aya reads no better than the Latin original. Caveat for both languages: the transliterator sometimes replaced an English loan-word with a native word (*murder* → हत्या), so "same words" is approximate.
- **Practical upshot:** for Hindi, *transliterate-then-guard* is a strong pipeline (89% usable), but it needs a transliterator; the guard alone needs nothing extra.

## 6.x Filler-wrapper robustness (gate v3.1 ablation)

Gate v3's clearest weakness was chatty wrappers on harmless prompts. We added 940 fit-set prompts wrapped in 12 new templates (gate **v3.1**, `gate_v3_1.py`) and tested on held-out prompts in **5 templates never seen in training** plus the original wrapper (1,128 rows):

| Gate fire rate | v3 | v3.1 |
|---|---|---|
| Unseen wrappers, harmful (460) | 94.3% | 94.8% |
| Unseen wrappers, **harmless** (480) | 17.5% | **11.5%** |
| Original wrapper (*"bhai plz help … 🙏"*, §6.6 test set), harmful / harmless | 98.9% / 43.8% | 97.8% / **26.0%** |
| SMS spelling / typos, harmless | 7.3% / 5.2% | 4.2% / 5.2% |
| Original benchmark, harmful | 93.0% | 91.7% |
| India-specific, harmful | 82% | 78% |
| Romanized everyday (orig / ASCII-folded) | 2.7% / 4.7% | 2.0% / 2.7% |

Training on wrappers cuts wrapper false-flags by a third with no loss on wrapped harmful prompts, at a cost of 1–4 points of recall on clean prompts. It is a precision/recall dial rather than a free win, so we **ship v3** (the fully end-to-end-evaluated gate) and publish v3.1 (`gate_v3_1.npz`) for deployments where chatty traffic dominates.
