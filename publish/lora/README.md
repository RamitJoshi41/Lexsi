---
base_model: CohereLabs/tiny-aya-global
library_name: peft
license: cc-by-nc-4.0
language: [en, hi, pa, bn, ta, mr, ur]
tags: [safety, refusal, code-mixing, hinglish, romanized-punjabi, lora, aligntune, circuitkit, lexsi-hackathon]
datasets: [sunit12/codemix-refusal-distill-v2, sunit12/codemix-safety-bench]
---

# Tiny Aya Global — code-mixed refusal LoRA (circuit-guided, v2)

A 37 MB LoRA for [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) that makes the model refuse
harmful requests written in **romanized / code-mixed Hindi and Punjabi** (and, zero-shot, romanized Bengali, Tamil,
Marathi and Urdu) — **without runtime hooks, without a system prompt, and with MMLU unchanged**.

It is the hook-based guard from *Lost in Transliteration* (probe-gated refusal steering at layer 14,
[`sunit12/tiny-aya-global-codemix-refusal-guard`](https://huggingface.co/sunit12/tiny-aya-global-codemix-refusal-guard))
distilled into the weights with **AlignTune** (TRL SFT backend), with LoRA restricted to **layers 10–20** — the region our
causal patching and **CircuitKIT** circuit analysis located as where the refusal decision fails for code-mixed input.

## Results (held-out prompts, Aya Expanse 8B judge; "usable" = a refusal, not a degenerate loop, in a language the user reads)

| | base | hook guard v1.0 | **this LoRA** |
|---|---|---|---|
| usable refusal, held-out harmful, non-English (avg of 4 scripts) | 38% | 76% | **90%** |
| … romanized Hindi / romanized Punjabi | 26% / 15% | 63% / 63% | **91% / 87%** |
| harmless held-out refused (all scripts) | 13% | 15% | 19% (Δ +5 [+1, +10]) |
| unseen languages (romanized bn / mr / ta / ur), usable | 15% | 57% | **78%** |
| chatty filler wrappers: usable / harmless refused | 23% / 10% | 70% / 28% | **86% / 19%** |
| jailbreak wrappers (romanized Hindi), usable — fiction framing | 35% — 30% | 70% — 30% | **75% — 77%** |
| everyday prompts refused (600, EN / HI / PA native) | 3% | 3% | 3.5% |
| MMLU, 400 q (AuditKIT) | 50.5% | 50.5% | **50.5%** |

95% CIs are cluster-bootstrap over prompt ids; full tables and logs in
[`sunit12/codemix-safety-logs`](https://huggingface.co/datasets/sunit12/codemix-safety-logs).
Same-data LoRA on **all** layers reached the same refusal rates but collapsed MMLU (2.5%); without the capability-retention
rows (v1) MMLU fell to 30.5% because the model stopped ending its turn after a one-letter answer.

## Use

**llama.cpp / LM Studio / Ollama (CPU):** download `codemix-refusal-lora-f16.gguf` from this repo and the official
[`CohereLabs/tiny-aya-global-GGUF`](https://huggingface.co/CohereLabs/tiny-aya-global-GGUF) model, then
`llama-cli -m tiny-aya-global-q4_k_m.gguf --lora codemix-refusal-lora-f16.gguf`.

**transformers + peft:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
model = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(model, "sunit12/tiny-aya-global-codemix-refusal-lora").merge_and_unload()
```

## Training
- Data: [`sunit12/codemix-refusal-distill-v2`](https://huggingface.co/datasets/sunit12/codemix-refusal-distill-v2), 3,750 rows built
  **only from training prompts**: steered-teacher refusals in the user's own script (kept only if judged a refusal, not a loop,
  in the right script and a readable language by fastText LID; otherwise a short hand-written refusal in that script),
  self-distilled base-model answers for harmless prompts, story/screenplay/hypothetical/persona-wrapped versions of
  training prompts, and 700 capability-retention rows (base-model answers to MMLU *auxiliary_train* questions and everyday prompts).
- AlignTune 0.1.35 `create_sft_trainer` (TRL backend; CuratorKIT schema gate), LoRA r=16, α=32, dropout 0.05,
  q/k/v/o/gate/up/down of layers 10–20, lr 2e-4, 2 epochs, bf16, A10G.

## Limits
- Harmless false refusals rise by ~4 points on held-out non-English prompts (XSTest-style look-alikes; CI −1 to +9), more on Bengali; +6 points on jailbreak-wrapped harmless prompts.
- Weaker than the hook guard on role-play (73% vs 83%) and translate-then-answer (67% vs 77%) wrappers.
- Many romanized-Hindi refusals reuse one of five hand-written phrasings.
- The translations of the unseen-language test set were machine-written (Claude) and not native-reviewed.
- Inherits the Tiny Aya license (CC-BY-NC 4.0).

Part of *Lost in Transliteration* (Lexsi Labs hackathon, tracks 2a + 2b).
