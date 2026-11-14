## With more credits

| Planned work | How | Compute (est.) | Expected before → after |
|---|---|---|---|
| **Teach the model to *read* romanized text**, not just refuse it | LoRA (AlignTune SFT) on ~20k romanized Hindi/Punjabi instructions (aya_dataset romanized with Aya Expanse, as we did for 300), then re-run the language-ID probe and knows/acts analysis | ~2 A10G-h | Probe recognises romanized Punjabi as Punjabi at mid layers: ~0% → >50%. Usable refusal on romanized Punjabi with the guard: 63% → ~80% (fewer wrong-language refusals). Plain-model refusal gap shrinks by ~half. |
| **More languages** | Same pipeline for Tamil, Bengali, Marathi, Gujarati (native vs romanized), one shared gate recipe | ~6 A10G-h + judge | Similar romanized gaps (we forecast 30–50 pts for the lower-resource romanized forms); per-language usable-refusal gains comparable to Hindi (+35–45 pts). |
| **Attack the guard** | Best-of-N wrappers, GCG-style suffixes against the linear gate, multi-turn escalation; report attack-success rate and cost-to-break; adversarially train the gate | ~5 A10G-h | A white-box attacker will evade a linear gate; we expect wrapper-trained gates (v3.1-style) to raise cost-to-break several-fold for black-box attackers. |
| **Scale** | Aya Expanse 8B / 32B as subjects | ~8 A10G-h / 2 A100-h | Test whether the gap shrinks with scale (hypothesis: smaller but present for romanized Punjabi). |
| **Better human evaluation** | 300 items, two native annotators per language, inter-annotator κ | ~6 person-hours | Tighter judge-validity claim (currently 29 items, κ 0.63). |
| **Serving** | Guard as a vLLM forward hook; measure throughput at production batch sizes | ~1 A10G-h | Overhead stays ≈ 0; cost per 1k replies falls several-fold vs our `transformers` number. |

## Documentation and reproduction

**Environment.** Python 3.12, `uv` (every script declares its dependencies inline, PEP 723). GPU work: one NVIDIA A10G (24 GB) per job via `hf jobs uv run` (bf16 needs ~8 GB, nf4 ~4 GB). CPU analyses run on a laptop or an HF CPU job. Main packages: `torch`, `transformers>=4.56`, `scikit-learn`, `safetune==0.1.18`, `auditkit==1.1.12`, `curatorkit==0.1.27`, `bitsandbytes` (4-bit), `fasttext` (LID-218), `datasketch` + `sentence-transformers` (leak check).

**Repo layout** (https://github.com/SunitSinghDhaliwal/knows-but-doesnt-refuse):

| Path | What it does |
|---|---|
| `data/` | benchmark, India-specific pairs, perturbation / filler / transliteration test sets, everyday-traffic prompts |
| `job_baseline.py`, `job_judge4.py` | generations + hidden states for all prompts; 4-way Aya Expanse judge |
| `analyze.py`, `job_langid.py` | knows-vs-acts layer analysis; language-ID probe |
| `gate_v3.py`, `gate_v3_1.py`, `gate_v3_fire.py` | gate training (Global, filler ablation, Fire) |
| `job_steer2.py` | gated / always-on steering; `GATE_FILE`, `DATA_FILE`, `QUANT=nf4` options; logs throughput |
| `job_cast.py` | SafeTune CAST baseline |
| `job_capability.py` | AuditKIT MMLU check (base vs guarded; `QUANT=nf4` option) |
| `job_translit.py`, `translit_compare.py` | transliteration test |
| `response_lid.py`, `compare_final.py`, `bootstrap_ci.py`, `leak_check.py`, `leak_robust.py` | language of replies, final tables + McNemar, bootstrap CIs, leak check |
| `auditkit_codemix/` | AuditKIT scenario + multilingual refusal scorers (+ CPU replay example) |
| `publish/guard/` | the released guard (`guard.py`, gates, vectors, model card) |
| `results/` | every table, JSON and figure in this write-up |

**Reproduce the headline** (≈ 40 A10G-minutes):
```bash
hf jobs uv run --flavor a10g-small --secrets HF_TOKEN -e GATE_FILE=gate_v3.npz job_steer2.py <work_repo> CohereLabs/tiny-aya-global "control:gated:14:0.0:en;g3_14:gated:14:1.0:en"
hf jobs uv run --flavor a10g-small --secrets HF_TOKEN job_cast.py <work_repo> CohereLabs/tiny-aya-global all
hf jobs uv run --flavor a10g-small --secrets HF_TOKEN job_judge4.py <work_repo> runs/steer/tiny-aya-global/gen_*.jsonl
uv run response_lid.py runs/steer/tiny-aya-global/*_judged4.jsonl && uv run compare_final.py && uv run bootstrap_ci.py
```
Greedy decoding everywhere (`do_sample=False`, ≤160 new tokens); random splits use seed 0. Use the guard directly:
```python
# guard.py from sunit12/tiny-aya-global-codemix-refusal-guard (full snippet in the model card)
guard = guard_mod.CodeMixRefusalGuard.from_pretrained("sunit12/tiny-aya-global-codemix-refusal-guard").attach(model)
out = model.generate(**ids, max_new_tokens=120, do_sample=False)   # gate decides per prompt; guard.last_scores holds gate scores
guard.detach()
```

**Declaration of work done outside the hackathon.** None. All data construction, code, experiments and writing were done during the hackathon. We used public datasets and models (listed above) and the hackathon kits.

## LLM usage statement

We used an AI assistant (Claude) heavily for engineering and writing: roughly 80% of the code (job scripts, analysis, guard, explorer) and about 70% of the first-draft prose were AI-generated, then directed, checked and edited by the team. The research question, experiment choices, native-language review and human judge-validation labels are the team's. Hindi/Punjabi versions of the India-specific prompts were drafted with AI assistance (Aya Expanse refused to translate the harmful ones) and reviewed by a native speaker. All reported numbers come from scripts in the repo, not from the assistant.
