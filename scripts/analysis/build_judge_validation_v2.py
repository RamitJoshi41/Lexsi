"""Build the blinded native-speaker labelling set for judge validation v2.

v1 (results/judge_validation.json) labelled 29 items from baseline / guard / CAST, before Fix 2 existed.
v2 covers the three headline runs (plain, Fix 1 gated steer, Fix 2 LoRA) and has two strata:
  random   - per run x non-English script x harmful/harmless, N_CELL items at random: estimates judge accuracy (kappa)
  targeted - (flag `flip`, whichever stratum drew them) items whose judge label CHANGED between plain and LoRA: the replies that make up the headline gain
             (harmful: plain not refused -> LoRA refused) and the over-refusal cost (harmless: newly refused).
             Checks whether the gain is real, not whether the judge is accurate on average.

Writes (run from the repo root):
  validation/judge_validation_v2_sheet.csv        what labellers see: no run name, no judge label, shuffled
  validation/private/judge_validation_v2_key.csv  item -> run, id, script, judge label, stratum (git-ignored: keep it
                                                  away from labellers until labelling is done)
Labelling instructions: validation/README.md. Scoring: scripts/analysis/judge_validation_v2.py"""
import sys, os
sys.path.insert(0, ".")
import pandas as pd, numpy as np
import codemix_eval as ce

N_CELL, N_HARM_FLIPS, SEED = 7, 24, 0          # 3 runs x 4 scripts x 2 labels x 7 = 168 random; + targeted
SCRIPTS = ["hi_deva", "hi_rom", "pa_guru", "pa_rom"]
rng = np.random.default_rng(SEED)

runs = {k: ce.add_usable(ce.load_log(p)).assign(run=k) for k, p in ce.HEADLINE_RUNS.items()}
allr = pd.concat(runs.values(), ignore_index=True)
allr = allr[allr.script.isin(SCRIPTS)]

# random stratum
rand = allr.groupby(["run", "script", "label"], group_keys=False).apply(
    lambda g: g.sample(min(N_CELL, len(g)), random_state=int(rng.integers(1 << 31)))).assign(stratum="random")

# targeted stratum: plain vs LoRA judge-label flips, LoRA side
m = runs["base"].merge(runs["lora"], on=["id", "script", "label"], suffixes=("_b", "_l"))
m = m[m.script.isin(SCRIPTS) & ~m.ref_b & m.ref_l]
harm_flips = m[m.label == 1].sample(min(N_HARM_FLIPS, (m.label == 1).sum()), random_state=SEED)
harmless_flips = m[m.label == 0]                                   # all of them (17 on the published logs)
keys = pd.concat([harm_flips, harmless_flips])[["id", "script", "label"]]
targ = runs["lora"].merge(keys, on=["id", "script", "label"]).assign(stratum="targeted")

items = pd.concat([rand, targ]).drop_duplicates(["run", "id", "script"], keep="first")   # in both: stays "random"
flips = set(map(tuple, keys.values))
items["flip"] = (items.run == "lora") & [(i, s, l) in flips for i, s, l in zip(items.id, items.script, items.label)]
items = items.sample(frac=1, random_state=SEED).reset_index(drop=True)
items["item"] = [f"w{i:03d}" for i in range(len(items))]

sheet = pd.DataFrame({"item": items.item, "script": items.script, "harmful_prompt": items.label.map({1: "yes", 0: "no"}),
                      "prompt": items.text, "response": items.response,
                      "your_label": "", "reply_readable": "", "notes": ""})
key = items[["item", "run", "id", "script", "label", "stratum", "flip", "judge4", "lang", "ok_lang", "ref", "usable"]]
os.makedirs("validation/private", exist_ok=True)
sheet.to_csv("validation/judge_validation_v2_sheet.csv", index=False)
key.to_csv("validation/private/judge_validation_v2_key.csv", index=False)
print(f"{len(items)} items -> validation/judge_validation_v2_sheet.csv (key in validation/private/)")
print(key.groupby(["stratum", "run"]).size().unstack(fill_value=0).to_string())
