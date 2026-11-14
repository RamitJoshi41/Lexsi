# /// script
# dependencies = ["auditkit", "pandas", "huggingface_hub"]
# ///
"""Replay our saved held-out outputs through AuditKIT with the CodeMix scenario + scorers (CPU only).
Compares plain Tiny Aya, our guard and SafeTune CAST, and checks the judge-free usable_refusal proxy against the
Aya-Expanse-judge usable-refusal labels used in the article. Run from the repo root."""
import json, sys, pandas as pd
sys.path.insert(0, "auditkit_codemix")
import auditkit as ak
from auditkit.model import CallableModel
from codemix_auditkit import (CodeMixSafetyScenario, split_by_label, HARMFUL_SCORERS, HARMLESS_SCORERS,
                              is_usable_refusal, _DET)
samples = CodeMixSafetyScenario(split="heldout", fit_ids=json.load(open("fit_ids.json"))).samples()
harmful, harmless = split_by_label(samples)
R = "runs/steer/tiny-aya-global/"
RUNS = {"plain": "gen_control_a0_judged4_lid.jsonl", "guard": "gen_g3_14_judged4_lid.jsonl",
        "cast_all": "gen_safetune_cast_all_a1_judged4_lid.jsonl"}
by_input = {}
for s in samples: by_input.setdefault(s.input, s.id)
summary, rows = {}, []
for name, f in RUNS.items():
    resp = {f'{r["id"]}:{r["script"]}': r for r in map(json.loads, open(R + f))}
    model = CallableModel(lambda prompts, resp=resp: [resp[by_input[p]]["response"] or "" for p in prompts], name=name)
    nonen = [s for s in harmful if s.metadata["script"] != "en"]
    a = ak.evaluate(nonen, model, scorers=HARMFUL_SCORERS, cache=False).headline
    b = ak.evaluate(harmless, model, scorers=HARMLESS_SCORERS, cache=False).headline
    summary[name] = {"nonEN_harmful_" + k: round(v, 3) for k, v in a.items()} | {"harmless_" + k: round(v, 3) for k, v in b.items()}
    for s in nonen:
        r = resp[s.id]; o = r["response"] or ""
        judge_usable = r["judge4"] == "REFUSAL" and r.get("ok_lang", True) and not r.get("degenerate", False)
        rows.append({"run": name, "script": s.metadata["script"], "proxy": is_usable_refusal(s.input, o),
                     "judge_refusal": r["judge4"] == "REFUSAL"})
print(pd.DataFrame(summary).T.to_string())
df = pd.DataFrame(rows)
agree = (df.proxy <= df.judge_refusal).mean()
print("\nproxy usable vs judge REFUSAL: agreement %.3f, proxy-usable rows the judge did NOT call a refusal: %d / %d"
      % ((df.proxy == df.judge_refusal).mean(), (df.proxy & ~df.judge_refusal).sum(), df.proxy.sum()))
print(df.groupby(["run", "script"]).proxy.mean().unstack().round(2).to_string())
json.dump({"summary": summary, "proxy_vs_judge_agreement": round(float((df.proxy == df.judge_refusal).mean()), 4)},
          open("results/auditkit_codemix_replay.json", "w"), indent=1)
