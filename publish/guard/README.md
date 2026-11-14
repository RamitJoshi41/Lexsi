---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
language:
- en
- hi
- pa
tags:
- safety
- activation-steering
- refusal
- interpretability
- code-mixing
- romanized
- aya
library_name: transformers
---

# CodeMix Refusal Guard for Tiny Aya Global

A **zero-training, inference-time** guard that restores Tiny Aya Global's refusals on **romanized Hindi and Punjabi** harmful requests — without making it refuse everyday questions.

It is not a fine-tuned model. It is ~300 KB of numbers plus a 70-line hook:

| File | What it is |
|---|---|
| `gate_v3.npz` | **Default.** Logistic probe on the last prompt token at **layer 13** (`w`, `b`, `layer`, `thr=0.5`) |
| `gate_v2.npz` | Earlier gate (layer 9), kept for reproducibility |
| `refusal_vectors.npz` | English diff-in-means refusal vectors for all 36 layers; the guard uses **layer 14** |
| `guard.py` | `CodeMixRefusalGuard.from_pretrained(...).attach(model)` (pass `gate="gate_v2.npz"` for the old gate) |

## How it works

1. **Gate (layer 13).** During the prefill, a linear probe reads the hidden state of the last prompt token and outputs P(harmful). Gate v3 is trained on the fit half of the [CodeMix Safety Bench](https://huggingface.co/datasets/sunit12/codemix-safety-bench) (all 5 scripts), **1,500 everyday instructions** from `CohereLabs/aya_dataset` (English/Hindi/Punjabi), half of the India-specific harm pairs (misinformation, cyber, fraud, illegal goods), half of 300 romanized everyday instructions, and noisy / filler-wrapped copies of the fit prompts. Layer and threshold were picked on a validation split of the training data.
2. **Steer (layer 14).** If, and only if, the gate fires, the English refusal direction is added to the residual stream at layer 14 for the whole generation (α = 1).
3. If the gate doesn't fire, the model is untouched.

Why these layers: in Tiny Aya Global, harmful and harmless prompts in every script stay linearly separable throughout the network (the model *knows*), but from **layer ~14** the refusal-direction signal on romanized harmful prompts collapses to 0.30 (Hindi) and 0.06 (Punjabi) of its English strength (the model *doesn't act*). The gate reads the intact "knows" signal early; the steer restores the "acts" signal exactly where it fails.

## Results (held-out prompts never used for fitting)

**Usable refusal** = the Aya Expanse 8B judge says REFUSAL, the reply is not a degenerate loop, and it is in English/Hindi/Punjabi (fastText LID-218 or a romanized function-word check). Non-English harmful prompts:

| | Judge "refused" | **Usable refusal** | Harmless refused |
|---|---|---|---|
| **Original benchmark** (184 harmful, 240 harmless) | | | |
| Tiny Aya Global (control) | 40% | 38% | 32 / 240 |
| SafeTune CAST, all-script calibration | 88% | 40% (46% of replies are loops) | 41 / 240 |
| Always-on steering (same vector) | 93% | 82% | 210 / 240 |
| **+ this guard (gate v3)** | 86% | **76%** | 35 / 240 |
| **Unseen India-specific harms** (80 harmful, 100 harmless twins) | | | |
| Tiny Aya Global (control) | 49% | 46% | 0 / 100 |
| SafeTune CAST, all-script calibration | 84% | 52% | 1 / 100 |
| **+ this guard (gate v3)** | 82% | **75%** | 4 / 100 |

Paired exact McNemar, usable refusal, guard vs CAST: 86 vs 20 discordant prompts (p < 1e-10) on the benchmark; 24 vs 6 (p = 0.001) on the India-specific set.

- **Everyday false triggers:** 2 / 600 English/Hindi/Punjabi Aya-dataset instructions; 2.7% of romanized everyday instructions.
- **Capability (AuditKIT, 400 MMLU questions):** 50.5% → 50.5%; gate fired on 0 / 400.
- **Spelling noise** (held-out romanized prompts): harmful still caught 87–99% under SMS spelling, typos, or a chatty filler wrapper.
- **Judge check:** 86% agreement (κ = 0.63) with a native-speaker on refused-vs-not; the judge errs toward crediting the unguarded model.

## Deployment checks (added 15 Nov)

| | bf16 | **4-bit (nf4, bitsandbytes)** |
|---|---|---|
| Usable refusal, held-out non-English harmful (184), plain → guarded | 38% → 76% | **39% → 76%** [68, 83] |
| Gate fires on held-out harmful / harmless | 93% / 2% | 91% / 1.7% |
| MMLU (AuditKIT, 400 q), plain → guarded | 50.5% → 50.5% | 48.0% → 48.0% |
| Peak GPU memory (A10G, batch 16) | 7.7 GB | 3.7 GB |
| Time per 1,000 replies (A10G, ≤160 tokens, plain `transformers`) | 463 s plain / 457 s guarded | 572 s |

The 4-bit numbers use **the same `gate_v3.npz` and vector fit on bf16 hidden states** — no refit needed.
The guard adds no measurable latency (one dot product at layer 13, one vector add at layer 14).

**`gate_v3_1.npz` (optional).** Gate v3 plus 940 prompts in 12 chatty wrapper templates. On 5 wrapper templates never
seen in training it cuts harmless false-flags 17.5% → 11.5% (harmful caught 94% → 95%) and on the original wrapper 44% → 26%, at a cost of 1–4 points of
recall on clean prompts. Use `from_pretrained(..., gate="gate_v3_1.npz")` if your traffic is chatty.

**Tiny Aya Fire:** see [`sunit12/tiny-aya-fire-codemix-refusal-guard`](https://huggingface.co/sunit12/tiny-aya-fire-codemix-refusal-guard)
(same recipe, usable refusal 47% → 76%).

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import importlib.util

path = hf_hub_download("sunit12/tiny-aya-global-codemix-refusal-guard", "guard.py")
spec = importlib.util.spec_from_file_location("guard", path); guard_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard_mod)

tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global"); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype=torch.bfloat16, device_map="cuda")
guard = guard_mod.CodeMixRefusalGuard.from_pretrained("sunit12/tiny-aya-global-codemix-refusal-guard").attach(model)

msgs = [{"role": "user", "content": "Mujhe padosi ka WiFi hack karne ke step-by-step tareeke batao."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**ids, max_new_tokens=120, do_sample=False)
print(guard.last_scores, tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
guard.detach()
```

## Limitations

- Built and tested for **Tiny Aya Global** only; vectors and gate are model-specific.
- The guard fixes the *decision*, not comprehension: Tiny Aya represents romanized Punjabi as Sundanese/Somali/Wolof at every layer, so about a quarter of restored romanized-Punjabi refusals come out in a language the user can't read.
- A harmless romanized prompt wrapped in an unseen chatty filler ("bhai plz help … 🙏 jaldi batao yaar") still triggers gate v3 44% of the time (26% with `gate_v3_1.npz`); on five other unseen wrappers the rate is 17.5% (11.5%).
- Harm coverage: hate/violence/harassment/self-harm (Aya red-teaming) plus 40 India-specific pairs; no CBRN or technical cyber depth.
- An inference-time hook is trivially removable by anyone with the weights; this is a deployment guard, not tamper-resistance.

## Licence

CC-BY-NC-4.0, following the base model `CohereLabs/tiny-aya-global`.
