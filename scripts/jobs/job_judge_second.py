# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.56", "accelerate", "pandas", "numpy", "huggingface_hub"]
# ///
"""Re-judge the three headline runs (plain / Fix 1 steer / Fix 2 LoRA, 470 held-out replies each) with a judge from
ANOTHER model family, using the same 4-way prompt (codemix_eval.JUDGE_PROMPT). The main judge, Aya Expanse 8B, is a
Cohere model like Tiny Aya; this tests whether the headline survives a judge with no family tie.

    python scripts/jobs/job_judge_second.py [JUDGE_ID] [--upload]          # from the repo root, any >=24 GB GPU
    default JUDGE_ID = Qwen/Qwen2.5-7B-Instruct (Apache-2.0, not gated). Gated alternatives with stronger Indic
    coverage: meta-llama/Llama-3.1-8B-Instruct, google/gemma-2-9b-it (need HF_TOKEN with access).

Writes runs/judge2/<judge>/<run>.jsonl (the logs + column judge4_2nd); --upload also pushes them to the logs dataset.
Compare (CPU): python scripts/analysis/second_judge_compare.py <judge>
On HF Jobs (no repo checkout) it fetches codemix_eval.py from the logs repo's code/ folder, so upload the current one
there first: `hf upload sunit12/codemix-safety-logs codemix_eval.py code/codemix_eval.py --repo-type dataset`."""
import sys, os
sys.path.insert(0, ".")
try:
    import codemix_eval as ce
except ImportError:
    from huggingface_hub import hf_hub_download
    sys.path.insert(0, os.path.dirname(hf_hub_download("sunit12/codemix-safety-logs", "code/codemix_eval.py", repo_type="dataset")))
    import codemix_eval as ce
assert hasattr(ce, "judge_input") and hasattr(ce, "HEADLINE_RUNS"), \
    "this codemix_eval.py predates the second-judge tooling: run from the repo root, or upload the current one to code/"

args = [a for a in sys.argv[1:] if not a.startswith("--")]
JUDGE = args[0] if args else "Qwen/Qwen2.5-7B-Instruct"
TAG = JUDGE.split("/")[-1]
os.makedirs(f"runs/judge2/{TAG}", exist_ok=True)
for run, path in ce.HEADLINE_RUNS.items():
    df = ce.load_log(path)
    print(f"{run}: judging {len(df)} replies with {JUDGE}", flush=True)
    labs = ce.judge(df, judge_id=JUDGE).judge4
    out = df.assign(judge4_2nd=labs.values)
    p = f"runs/judge2/{TAG}/{run}.jsonl"
    out.to_json(p, orient="records", lines=True, force_ascii=False)
    print(f"  wrote {p}; agreement with main judge: {(out.judge4 == out.judge4_2nd).mean():.1%}; "
          f"unparsed: {(out.judge4_2nd == 'UNPARSED').mean():.1%}", flush=True)
    if "--upload" in sys.argv:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=p, path_in_repo=p, repo_id=ce.LOGS, repo_type="dataset")
print("DONE. Next: python scripts/analysis/second_judge_compare.py", TAG)
