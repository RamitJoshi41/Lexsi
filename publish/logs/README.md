---
license: cc-by-nc-4.0
tags:
- safety
- refusal
- activation-steering
- interpretability
- code-mixing
- aya
- evaluation-logs
pretty_name: CodeMix Safety — full experiment logs (Tiny Aya Global / Fire)
---

# CodeMix Safety — full experiment logs

Everything behind the article **"Lost in Transliteration"** — every generation, every judge label, hidden states, result tables, figures and the exact job scripts.

> ⚠️ Contains model responses to harmful requests (hate, violence, self-harm, harassment), including cases where the model complied. Provided for safety research only.

Benchmark: [`sunit12/codemix-safety-bench`](https://huggingface.co/datasets/sunit12/codemix-safety-bench) · Guard: [`sunit12/tiny-aya-global-codemix-refusal-guard`](https://huggingface.co/sunit12/tiny-aya-global-codemix-refusal-guard)

## Layout

| Path | Contents |
|---|---|
| `codemix_v1.jsonl`, `fit_ids.json` | The 940-row benchmark; the 94 prompt ids (of 188) used to *fit* probes/vectors/gates. Everything else is held out. |
| `runs/sanity/` | First 40-prompt run (keyword detector, SafeTune `RefusalDirectionTrainer` check) |
| `runs/baseline/<model>/gen_*.jsonl` | Greedy generations (160 tokens) for all 940 prompts, Tiny Aya Global and Fire |
| `runs/baseline/<model>/hidden_*.npy` | Last-prompt-token hidden states, all 36 layers, float16 `[940, 36, 2048]` |
| `*_judged.jsonl` / `*_judged4.jsonl` | Aya Expanse 8B judge labels — 3-way (superseded) and 4-way (REFUSAL / COMPLIANCE / PARTIAL / NONANSWER, used in the article) |
| `runs/steer/tiny-aya-global/gen_<config>*.jsonl` | Held-out runs: `control_a0` (no steering, identical batching), `v2_gated14_a1`, `v2_gated18_a1`, `gated14_a1` (gate v1), `gated14_a2`, `gated18_a2`, `always14_a1`, `always18_a1`, `safetune_cast_en_a1`, `safetune_cast_all_a1`. Steered runs include `gate_p`. |
| `runs/benign/` | Layer 6–14 features of 2,100 everyday instructions (CohereLabs/aya_dataset) used to retrain the gate |
| `runs/gatecheck_*.json`, `runs/cast_everyday_*.json`, `runs/capability_*.json` | Gate v1 everyday false-trigger check, SafeTune CAST everyday check, AuditKIT MMLU capability check |
| `gate_v2.npz` | Final gate (layer 9 logistic probe) |
| `results/` | `tables.md`, `final_rates.json` (with bootstrap CIs), `summary.json` (per-layer knows/acts curves), detector evaluations, figures |
| `code/` | All scripts. GPU jobs run with `hf jobs uv run --flavor a10g-small --secrets HF_TOKEN code/job_*.py …` |

## Pipeline

1. `curate_with_curatorkit.py` — CuratorKIT clean + dedup + provenance per script.
2. `job_sanity.py` → `job_baseline.py` (generations + hidden states) → `job_judge.py` / `job_judge4.py` (Aya Expanse 8B judge).
3. `analyze.py` — behaviour + knows (English-trained probe AUROC) vs acts (refusal-direction signal) per layer.
4. `gate_offline*.py`, `job_gatecheck.py`, `job_gatefeat.py`, `gate_retrain.py` — gate v1 → everyday-traffic check → gate v2.
5. `job_steer.py` (gated / always-on steering, `GATE_FILE=gate_v2.npz` for v2), `job_cast.py` (SafeTune CASTTrainer baseline), `job_cast_everyday.py`.
6. `job_capability.py` — AuditKIT `evaluate` + `compare` on 400 MMLU questions, base vs guarded.
7. `auditkit_detector_check.py`, `ml_refusal_detector.py` — AuditKIT `RefusalDetector` audit + `MultilingualRefusalDetector`.
8. `final_tables.py` — all tables and figures. `job_guard_demo.py` — end-to-end test of the published guard.

## Licences
Generations are outputs of CC-BY-NC-4.0 models (Tiny Aya, Aya Expanse); this repo is released CC-BY-NC-4.0. Prompts derive from Aya Red-teaming (Apache-2.0), XSTest (CC-BY-4.0), Aya dataset (Apache-2.0); MMLU (MIT).
