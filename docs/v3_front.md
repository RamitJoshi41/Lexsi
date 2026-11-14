---
title: "Knows, But Doesn't Refuse: Where Tiny Aya's Safety Breaks in Romanized Hindi & Punjabi — and a Gated Fix That Doesn't Over-Refuse"
thumbnail: https://huggingface.co/datasets/sunit12/codemix-safety-logs/resolve/main/results/fig_headline.png
authors:
- user: sunit12
---

# Knows, But Doesn't Refuse

**One-line claim:** Tiny Aya *recognises* harmful requests typed in romanized Hindi and Punjabi but does not refuse them, because it cannot tell what language it is reading; a 9 KB probe gate that switches on a refusal steer at the layer where refusal collapses doubles usable refusals (38% → 76%) with no retraining, no over-refusal, no capability loss, and it holds on the 4-bit model and on Tiny Aya Fire.

![Headline figure](https://huggingface.co/datasets/sunit12/codemix-safety-logs/resolve/main/results/fig_headline.png)

**Tracks and problem statements**
- **Track 2b**, problem statement **#9**: locating where safety lives inside the model (knows-vs-acts layer analysis, language-ID probe)
- **Track 2b**, problem statement **#8**: inference-time steering (probe-gated, layer-targeted refusal steer vs SafeTune CAST)
- **Track 2a**, problem statement **#4**: multilingual and code-mixed safety (romanized/code-mixed Hindi & Punjabi benchmark and attack families)

> ⚠️ **Content warning.** This write-up quotes harmful requests (violence, hate, self-harm, fraud, misinformation) and short excerpts of model outputs that comply with them. The linked explorer and logs contain full unguarded model outputs.

## Abstract

Hundreds of millions of people in India type Hindi and Punjabi in Latin letters, mixed with English words. We ask whether Cohere's Tiny Aya models, built for these languages, stay safe on such input, and if not, *where* and *why* safety breaks and whether it can be repaired at that point.

We built a parallel benchmark of 188 prompts (harmful requests from Aya red-teaming, harmless-but-scary prompts from XSTest) in five scripts — English, Hindi in Devanagari and romanized, Punjabi in Gurmukhi and romanized — plus 40 India-specific harmful/harmless pairs (fake RBI notices, "digital arrest" scams, election misinformation). Tiny Aya Global refuses 74% of held-out harmful English prompts but only 30% in romanized Hindi and 20% in romanized Punjabi; the gap replicates on the India-specific set (97% vs 40% / 20%). Native scripts sit in between, and a transliteration test shows the script itself explains most of the Hindi gap.

Inside the model, an English-trained harm probe still separates harmful from harmless romanized prompts (AUROC 0.87–0.95): the model *knows*. But the refusal direction reaches only 30% (Hindi) and 6% (Punjabi) of its English strength from layer 14 on: it doesn't *act*. A language-ID probe explains why — the model never represents romanized Punjabi as Punjabi at any layer (it looks like Sundanese, Somali or Wolof), and refuses, when it does, in those languages.

We repair it where it breaks: a logistic probe on the last prompt token at layer 13 gates the addition of the English refusal direction at layer 14. On 184 held-out non-English harmful prompts, usable refusals (coherent, in a language the user reads) rise from 38% to 76% (95% CI of the gain +29 to +47 points), versus 40% for SafeTune's CAST, whose refusals are 46% degenerate loops (paired McNemar p < 1e-10). The gain holds on unseen India-specific harms (+29 points), after removing every test prompt with a near-duplicate in training (+29 over CAST), on a 4-bit model (39% → 76%), and on Tiny Aya Fire. The guard fires on 2 of 600 everyday requests, leaves MMLU unchanged (50.5% → 50.5%) and adds no measurable latency. Limits: the guard restores the refusal decision, not comprehension — a quarter of restored romanized-Punjabi refusals are in a language the user can't read — and chatty wrappers still cause false triggers.

We release the benchmark, every generation and judge label, the guard, an AuditKIT scenario with multilingual refusal scorers, and an explorer of all held-out outputs.

| | |
|---|---|
| **Headline before → after** | Usable refusal on held-out non-English harmful prompts: **38% → 76%** (SafeTune CAST: 40%) |
| **Base model + method in one line** | `CohereLabs/tiny-aya-global` (3.35B, bf16; also nf4 and Tiny Aya Fire) + a probe gate at layer 13 that adds the English diff-in-means refusal vector at layer 14 when it fires — inference-time, no weight changes |
