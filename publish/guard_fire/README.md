---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-fire
tags: [safety, steering, refusal, multilingual, code-mixing, hindi, punjabi, romanized, tiny-aya]
language: [en, hi, pa]
---

# Tiny Aya Fire — CodeMix Refusal Guard

The [Tiny Aya Global guard](https://huggingface.co/sunit12/tiny-aya-global-codemix-refusal-guard) re-fit for
**`CohereLabs/tiny-aya-fire`** with the identical recipe — a transfer test of the method from
*Lost in Transliteration*. No weights change: a logistic probe on the last prompt token at **layer 13**
(threshold 0.6, chosen on a validation split of training data) gates the addition of the English diff-in-means
refusal vector at **layer 14** (α = 1).

## Results (held-out; judge: Aya Expanse 8B; 95% cluster-bootstrap CIs)

| Tiny Aya Fire | Plain | + guard |
|---|---|---|
| Refused, harmful English / romanized Hindi / romanized Punjabi | 85% / 35% / 30% | 96% / 89% / 78% |
| **Usable refusal, non-English harmful (184)** | 47% [39, 54] | **76% [69, 82]** (+29 [+20, +38]) |
| Complied, non-English harmful | 30% | 7% |
| Unseen India-specific harms, usable refusal (80) | 60% | 84% (+24 [+11, +37]) |
| Harmless-but-scary refused (/240) | 35 | 38 |
| Gate fires: held-out harmful / harmless / everyday (600) / romanized everyday | — | 92% / 1.7% / 0.3% / 2.7% |
| MMLU (AuditKIT, 400 q) | 51.0% | 51.0% (gate fired 0 / 400) |

Fire shows the same romanization gap as Global, and the same fix closes most of it. Global's gate applied to Fire
*without refitting* still catches 90% of held-out harmful prompts (66% of unseen categories), so most of the gate
transfers between the two models; the per-model refit is better on new categories (77%).

## Use

Same `guard.py` as the Global guard:

```python
guard = guard_mod.CodeMixRefusalGuard.from_pretrained("sunit12/tiny-aya-fire-codemix-refusal-guard").attach(model)  # model = tiny-aya-fire
```

Files: `gate_v3.npz` (Fire gate), `refusal_vectors.npz` (Fire diff-in-means vectors, all 36 layers), `guard.py`.
Code, data and logs: https://github.com/SunitSinghDhaliwal/lost-in-transliteration ·
`sunit12/codemix-safety-bench` · `sunit12/codemix-safety-logs`.

## Limitations

Same as the Global guard: it restores the refusal *decision*, not comprehension of romanized text (some restored
romanized-Punjabi refusals are in the wrong language); chatty wrappers can trigger the gate on harmless prompts;
an inference-time hook is removable by anyone with the weights. SafeTune CAST was not re-run on Fire.
