"""Does the headline survive a judge from another model family? (CPU; run from the repo root after job_judge_second.py)

    python scripts/analysis/second_judge_compare.py [JUDGE_TAG]     # default Qwen2.5-7B-Instruct

For each headline run: refusal agreement and Cohen's kappa between the main judge (Aya Expanse 8B) and the second judge,
and usable refusal on the 184 non-English harmful prompts under each judge; then the paired cluster-bootstrap gain
(Fix 1 / Fix 2 minus plain) under each judge. Writes results/second_judge_<tag>.{md,json}"""
import sys, json
sys.path.insert(0, ".")
import codemix_eval as ce

TAG = sys.argv[1] if len(sys.argv) > 1 else "Qwen2.5-7B-Instruct"
runs = {r: ce.load_log(f"runs/judge2/{TAG}/{r}.jsonl") for r in ce.HEADLINE_RUNS}
harm = lambda d: d[(d.label == 1) & (d.script != "en")]
second = lambda d: d.assign(judge4=d.judge4_2nd)        # add_usable / summary read the judge4 column

res = {"judge_main": "CohereLabs/aya-expanse-8b", "judge_second": TAG, "runs": {}, "gain_vs_plain": {}}
md = [f"# Second judge: {TAG} vs Aya Expanse 8B", "",
      "| run | refusal agreement | κ (refused vs not) | unparsed (2nd) | usable refusal, non-EN harmful: main / 2nd | harmless refused, non-EN: main / 2nd |",
      "|---|---|---|---|---|---|"]
for r, d in runs.items():
    ra, rb = ce.add_usable(d).ref, ce.add_usable(second(d)).ref
    sa, sb = ce.summary(d).loc["ALL non-English"], ce.summary(second(d)).loc["ALL non-English"]
    v = {"n": len(d), "agree_refusal": float((ra == rb).mean()), "kappa_refusal": ce.kappa(ra, rb),
         "unparsed_second": float((d.judge4_2nd == "UNPARSED").mean()),
         "usable_main": float(sa.harmful_usable_refusal), "usable_second": float(sb.harmful_usable_refusal),
         "harmless_refused_main": float(sa.harmless_refused), "harmless_refused_second": float(sb.harmless_refused)}
    res["runs"][r] = v
    md.append(f"| {r} | {v['agree_refusal']:.0%} | {v['kappa_refusal']:.2f} | {v['unparsed_second']:.1%} | "
              f"{v['usable_main']:.0%} / {v['usable_second']:.0%} | {v['harmless_refused_main']:.1%} / {v['harmless_refused_second']:.1%} |")
md += ["", "| gain vs plain (usable refusal, 184 paired prompts) | main judge | second judge |", "|---|---|---|"]
for r in ["steer", "lora"]:
    g = {}
    for name, f in [("main", lambda d: d), ("second", second)]:
        m, lo, hi = ce.paired_bootstrap(harm(f(runs["base"])), harm(f(runs[r])))
        g[name] = {"delta": float(m), "ci": [float(lo), float(hi)]}
    res["gain_vs_plain"][r] = g
    fmt = lambda x: f"{x['delta']:+.0%} [{x['ci'][0]:+.0%}, {x['ci'][1]:+.0%}]"
    md.append(f"| {r} | {fmt(g['main'])} | {fmt(g['second'])} |")
json.dump(res, open(f"results/second_judge_{TAG}.json", "w"), indent=1)
open(f"results/second_judge_{TAG}.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
