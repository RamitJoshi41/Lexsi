# Day 4 findings (notes for article v4) — branch exp/day4

All numbers: Tiny Aya Global, greedy decoding, Aya Expanse 8B 4-way judge; "usable" = judged REFUSAL, not a degenerate
loop, reply in a language the user reads (same definition as v1.0). CIs = cluster bootstrap over prompt ids.

## 1. CircuitKIT: the refusal circuit fades in proportion to the refusal failure
CircuitKIT 0.2.4, `multilingual_refusal` task, EAP-IG (5 steps), node level, 64 minimal pairs per language
(Devanagari 46 after equal-length filtering). Pairs: ours (template x noun swaps, `data/minimal_pairs/`), English: bundled.
Openers: en I/Sure, hi मैं/ज़रूर, hi-Latn and pa-Latn Main/Zaroor (identical metric for the two romanized sets).

| | Jaccard of top-20 nodes with English | total |attribution| |
|---|---|---|
| English | — | 51.5 |
| Hindi (Devanagari) | 0.60 | 63.6 |
| Hinglish (romanized Hindi) | 0.48 | 44.1 |
| romanized Punjabi | 0.21 | **9.2** |

- Mid-layer attention heads A14.13, A16.2, A20.4 are in the English and Devanagari top-20; A16.2 and A20.4 in romanized
  Hindi; **none in romanized Punjabi**, whose top nodes are early MLPs (5, 8, 9, 10, 11) and late heads (A31.11, A25.3, A29.13).
- Those heads sit at layers 14–20, where the probe gate (L13) and steering (L14) act: circuit discovery and the
  representation analysis point to the same place.
- Romanized Hindi and Punjabi share templates and openers, so the 5x lower attribution for Punjabi is not a metric artifact.

## 2. Zero-shot transfer to four languages the guard never saw
`data/unseen_langs_v1.jsonl`: 30 harmful + 30 harmless held-out prompts in romanized Bengali, Tamil, Marathi, Urdu
(translated by Claude, not native-reviewed). Same guard (gate v3 + steer L14), no refit.
Usable counts a reply in the user's own language as readable.

| | bn | mr | ta | ur | all usable | harmless refused |
|---|---|---|---|---|---|---|
| base | 10% | 13% | 0% | 33% | 14% | 15% |
| guard | 40% | 40% | 50% | 83% | **53%** | 23% |
Δusable +39 pts [+29, +49]; Δharmless refused +8 pts [+4, +12] (mostly Bengali 17→40%).

## 3. Adaptive attacks (romanized Hindi wrappers, same 30+30 ids)
| wrapper | base usable | guard usable | base complied | guard complied |
|---|---|---|---|---|
| script-mix (Devanagari+Latin) | 37% | 100% | 63% | 0% |
| role-play "AI without rules" | 47% | 83% | 27% | 7% |
| translate-then-answer | 33% | 77% | 67% | 23% |
| prefix injection ("Bilkul, yeh rahe") | 30% | 60% | 47% | 3% |
| **fiction framing** | 30% | **30%** | 30% | 30% |
All: 35% → 70% (+35 [+23, +45]); harmless refused 6% → 7% (+1 [0, +2]). Fiction framing is the guard's blind spot.

## 4. Causal patching v1 (last token, one layer) — a negative result that shapes v2
English-only harm probe on held-out non-EN: acc 0.80 (L6) / 0.86 (L10) / 0.91 (L13) / 0.88 (L14) / 0.78 (L18) / 0.75 (L24).
Code-mixed vs English twin, last-token gap along refusal dir: 2.3 (L6) → 3.4 (L13) → 5.0 (L14); along the harm-probe
direction orthogonal to refusal: 0.45 → 0.66 → 0.81 (and signed mean −0.27 at L13: code-mixed is *more* harm-like).
Replacing the code-mixed last-token state with the full English state at one layer barely moves refusal at L6–18
(keyword estimate 17% → 22–25%); only at L24 (≈60%). Refusal is read from many positions, not one.
(judge-confirmed table: results/causal.md)

Judge-confirmed (64-token replies; judge REFUSAL incl. warnings/lectures, any language): baseline non-EN 51% [43, 60],
English 89% [78, 98]. Last-token patch at one layer: full English state 51–57% at L6–18, 68% [58, 78] at L24;
refusal component only ≤57%; harm component ≈ random ≈ 49–51%. English with its last-token refusal component replaced:
85–91% (no drop). => Refusal is not decided at a single position. Table: results/causal.md, fig: results/fig_causal.png.

## 5. Circuit-level head interventions (CircuitKIT heads A14.13, A16.2, A20.4)
Zero-ablating the 3 heads at every position in English: judged refusal 89% → 83% (CI overlaps) but replies that open with
an explicit refusal ("I'm sorry / I cannot …") fall 65% → 41% (random same-layer heads: 63%; pa-Latn-circuit heads: 63%):
the model hedges or redirects instead. CircuitKIT's metric is the first-token refusal-vs-comply logit, and the heads it
found carry the refusal *phrase*, not the refusal *decision*. Amplifying them on code-mixed prompts: x2 55% [47, 62] vs 51%
baseline, x4 53% (random x4 49%); harmless 18% → 18%. Necessary for the canonical refusal form, not sufficient to restore
refusal — consistent with the decision failing upstream. Table: results/heads.md.

## 6. Causal v2: direction-level interventions at every position, sized by the prompt's own English twin
At layer L, every position (prefill + generated) gets the same added vector; its size is the English twin's last-token
gap along that direction (no tuned alpha). r = refusal dir, p = harm-probe dir orthogonal to r, q = random ⊥ r.
Judged refusal (64-token replies), held-out; table results/causal2.md.

| code-mixed harmful (base 52% [42, 60]) | L10 | L13 | L14 | L18 |
|---|---|---|---|---|
| + c_r·r (refusal comp. English has) | 52% | 57% | 68% [60, 77] | **76% [67, 85]** |
| + c_p·p (harm comp. English has) | 51% | 49% | 52% | 53% |
| + abs(c_r)·p (harm dir., refusal-sized) | 53% | 53% | 48% | 45% |
| + abs(c_r)·q (random, refusal-sized) | 45% | 48% | 52% | 47% |
| harmless non-EN + mean(c_r)·r (base 18%) | 19% | 15% | 22% | 32% |

Mean twin gaps: c_r 3.3 (L13), 4.9 (L14), 9.6 (L18); c_p −0.27, −0.58, −1.07 (code-mixed is slightly MORE harm-like).
=> Sufficiency: only the refusal component restores refusal; the harm component is already there and adding more does
nothing; same-size pushes along other directions do nothing. Specificity holds at L14 (harmless 18→22%), weakens at L18.
Necessity at ONE layer is weak (English 89% → 83–87% when r is ablated at L10–18; 74% when the twin gap is removed at L18):
later layers rebuild the direction. All-layer ablation (Arditi et al.) running: job_ablate_all.py.

## 7. Over-refusal checks for the hook guard (baselines for the LoRA comparison)
- 600 everyday held-out prompts (en / hi_deva / pa_guru): refused 3% → 3% (+0).
- filler wrappers (1,128 romanized prompts, chatty prefixes/suffixes): usable 23% → 70% (+47 [+39, +56]) but harmless
  refused 10% → 28% (+19 [+13, +25]) — the guard's known weak spot (gate fires on chatty harmless text).

## 8. All-layer ablation (Arditi-style necessity) — results/ablate_all.md
Removing one direction (fit at L14) after every layer: refusal dir → English 89% → 76% [63, 87], code-mixed 51% → 29% [21, 37];
explicit "I'm sorry/I cannot" refusals in English 74% → 13% (the remaining judged refusals are moralising lectures).
Harm-probe dir: 91% / 48%; random dir: 91% / 52% (no effect). Harmless code-mixed: 18% → 16%.

## 9. AlignTune distillation: the guard compiled into a hook-free LoRA (results/*_lora.md)
Teacher = steered model + "reply in the user's script" system prompt on TRAIN prompts; student = LoRA r16 (all attn+MLP,
all layers), no hooks, no system prompt. v0 = 2/3 of teacher data; v1 = all 2,512 rows.

| | base | hook guard v1.0 | LoRA v0 | LoRA v1 |
|---|---|---|---|---|
| held-out usable, non-EN avg | 38% | 76% | 77% | **82%** |
| held-out usable hi_rom / pa_rom | 26 / 15% | 63 / 63% | 78 / 50% | 89 / 67% |
| held-out harmless refused (all) | 13% | 15% | 22% | 20% |
| unseen langs usable (bn/mr/ta/ur) | 15% | 57% | 67% | **72%** |
| attacks usable (fiction) | 35% (30%) | 70% (30%) | 59% (30%) | 71% (37%) |
| filler: usable / harmless refused | 23 / 10% | 70 / 28% | 73 / 25% | 77 / **21%** |
| everyday refused | 3% | 3% | 3% | 3% |
| MMLU (AuditKIT, 400 q) | 50.5% | 50.5% | 32.3% | 30.5% |
MMLU drop is mostly format: the student no longer stops after the letter ("AThe correct answer is…"); on 40 diagnostic
questions it picks nearly the same letters as base (45.0% vs 47.5%, first-letter parse). Fix in v2: retention data.
GGUF LoRA (llama.cpp convert_lora_to_gguf): 60.5 MB f16, plugs into CohereLabs/tiny-aya-global-GGUF.

## 10. LoRA v2 and the circuit-guided LoRA (final adapter)
v2 data (3,750 rows, `sunit12/codemix-refusal-distill-v2`): v1 rows with LID-filtered teacher refusals + 541 wrapper-augmented
train rows (story / screenplay / hypothetical / persona; test wrappers stay unseen) + 700 capability-retention rows
(base answers to MMLU auxiliary_train MCQs, extra everyday prompts). v2 = LoRA on all layers (lr 1e-4);
**v2-circuit = LoRA only on layers 10–20** (the region located by causal patching + CircuitKIT), lr 2e-4.

| | base | hook guard | LoRA v1 | LoRA v2 (all layers) | **LoRA v2-circuit (L10–20)** |
|---|---|---|---|---|---|
| held-out usable, non-EN avg | 38% | 76% | 82% | 90% | **90%** |
| hi_rom / pa_rom | 26 / 15% | 63 / 63% | 89 / 67% | 89 / 83% | **91 / 87%** |
| harmless refused (Δ vs base 13%) | — | +1 | +7 | +7 | **+5 [+1, +10]** |
| unseen langs usable | 15% | 57% | 72% | 77% | **78%** |
| filler usable / harmless refused | 23 / 10% | 70 / 28% | 77 / 21% | — | **86 / 19%** |
| MMLU (AuditKIT 400 q) | 50.5% | 50.5% | 30.5% | **2.5% (collapsed)** | **50.5%** |

Restricting the LoRA to the located layers keeps capability intact at equal refusal gains; the all-layer LoRA on the
same data destroys MMLU. Final adapter = v2-circuit, published as `sunit12/tiny-aya-global-codemix-refusal-lora`
(v1 kept at `-v1`; v0, v2 and v2-circuit at their own repos). 56 MB, r16, 77 target modules.
Pending (HF Jobs credits ran out ~04:45 IST): v2-circuit judging on attacks_v1 and everyday_test, GGUF conversion.
