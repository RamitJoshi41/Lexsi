# Frozen build: v1.0-submission

Safe fallback snapshot taken Sun 15 Nov 2026, ~2:10 AM IST, before Day-4 experiments.
Everything below is tagged `v1.0-submission` and can be submitted as-is.

| Artifact | Where | Tag |
|---|---|---|
| Code | GitHub `SunitSinghDhaliwal/lost-in-transliteration` @ 3bbd659 | `v1.0-submission` |
| Guard (Global) | HF model `sunit12/tiny-aya-global-codemix-refusal-guard` | `v1.0-submission` |
| Guard (Fire) | HF model `sunit12/tiny-aya-fire-codemix-refusal-guard` | `v1.0-submission` |
| Benchmark | HF dataset `sunit12/codemix-safety-bench` | `v1.0-submission` |
| Logs + article v3 | HF dataset `sunit12/codemix-safety-logs` (`docs/hf-article-v3.md`) | `v1.0-submission` |
| Work repo | HF dataset `sunit12/aya-codemix-work` | `v1.0-submission` |
| Explorer | HF Space `sunit12/knows-but-doesnt-refuse-explorer` | `v1.0-submission` |

Headline (usable refusal, held-out non-EN harmful, 95% cluster-bootstrap CI):
Global 38% -> 76% (+38 [+29,+47]); CAST 40%; Fire 47% -> 76%; 4-bit 39% -> 76%; MMLU unchanged.

Restore: `git checkout v1.0-submission`; on HF use `revision="v1.0-submission"`.
New work lives on branch `exp/day4` and is merged only if it beats v1.0.
