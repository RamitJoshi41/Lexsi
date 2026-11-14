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
| Romanized → native transliteration | the *same words* as the romanized prompt, re-written in native script | TBD | HI, PA |
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
- **Leak check** between *everything used to fit the guard* (gate training data, refusal-vector prompts: 4,856 rows) and every test set (670 rows): exact match after normalisation **0**; MinHash (char 5-grams, Jaccard ≥ 0.5) TBD; multilingual embeddings (LaBSE, cosine ≥ 0.90) TBD. TBD: re-scoring of the headline with near-duplicates removed.

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
