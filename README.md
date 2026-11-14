# Knows, But Doesn't Refuse

**Where Tiny Aya's safety breaks in romanized Hindi & Punjabi, and a gated fix that doesn't over-refuse.**
Lexsi Labs × Cohere hackathon (Track 2b #9 / #8, Track 2a #4).

Tiny Aya Global refuses 83% of harmful English requests but only 30% (romanized Hindi) and 22% (romanized Punjabi) of the same requests typed in Latin script. An English-trained harm probe still separates harmful from harmless romanized prompts (AUROC 0.87–0.95), so the model *knows*; the refusal direction collapses from layer ~14, so it doesn't *act*. A probe gate at layer 9 plus a refusal steer at layer 14 restores refusal with no weight changes, 0/600 everyday false triggers, and unchanged MMLU.

Full write-up: [`docs/hf-article-draft.md`](docs/hf-article-draft.md).

## Layout

| Path | What |
|---|---|
| `job_*.py` | HF Jobs scripts (`hf jobs uv run --flavor a10g-small --secrets HF_TOKEN job_x.py ...`): baseline generation + hidden states, judge (3-way, 4-way), steering, CAST baselines, gate features, capability (AuditKIT MMLU), guard demo |
| `analyze.py`, `final_tables.py` | Layer analysis (knows vs acts) and final tables/figures, CPU |
| `gate_*.py`, `gate_v2.npz` | Gate training/retraining; final gate weights |
| `ml_refusal_detector.py`, `auditkit_detector_check.py` | AuditKIT `MultilingualRefusalDetector` and detector audit |
| `data/` | Source seeds, translations, benchmark `codemix_v1.jsonl` |
| `runs/` | Generations + judge labels (hidden-state `.npy` files are excluded here; see the HF logs repo) |
| `results/` | Tables, JSON metrics, figures |
| `publish/` | Exact contents pushed to the three HF repos |
| `docs/` | Article draft, project spec, progress log |

## Hugging Face

- Benchmark: [`sunit12/codemix-safety-bench`](https://huggingface.co/datasets/sunit12/codemix-safety-bench)
- Logs (all generations, labels, hidden states, code): [`sunit12/codemix-safety-logs`](https://huggingface.co/datasets/sunit12/codemix-safety-logs)
- Guard: [`sunit12/tiny-aya-global-codemix-refusal-guard`](https://huggingface.co/sunit12/tiny-aya-global-codemix-refusal-guard)
