"""Which harmless prompts does Fix 2 (the LoRA) newly refuse? Paired base vs LoRA on the 192 non-English harmless
held-out prompts: flips in each direction, by script and by XSTest category, with every newly refused prompt listed.
CPU only. Run from the repo root. Writes results/overrefusal_lora.md and .json"""
import sys, json
sys.path.insert(0, ".")
import pandas as pd
import codemix_eval as ce

base, lora = (ce.add_usable(ce.load_log(ce.HEADLINE_RUNS[k])) for k in ("base", "lora"))
en = base[base.script == "en"].set_index("id").text
m = base.merge(lora, on=["id", "script", "label", "category", "text"], suffixes=("_base", "_lora"))
m = m[(m.label == 0) & (m.script != "en")].copy()
m["change"] = m.ref_base.astype(int).astype(str) + m.ref_lora.astype(int).astype(str)
m["change"] = m.change.map({"00": "neither", "11": "both", "01": "newly refused", "10": "no longer refused"})

def table(by):
    t = m.groupby(by).agg(n=("id", "size"), base=("ref_base", "mean"), lora=("ref_lora", "mean"),
                          newly_refused=("change", lambda s: (s == "newly refused").sum()),
                          no_longer_refused=("change", lambda s: (s == "no longer refused").sum()))
    t["net"] = t.newly_refused - t.no_longer_refused
    return t.sort_values("net", ascending=False)

def md(t):  # markdown table without the optional tabulate dependency
    t = t.reset_index(); rows = [[f"{v:.0%}" if c in ("base", "lora") else str(v) for c, v in r.items()] for _, r in t.iterrows()]
    return "\n".join(["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)] + ["| " + " | ".join(r) + " |" for r in rows])

by_script, by_cat = table("script"), table("category")
new = m[m.change == "newly refused"].assign(en=lambda d: d.id.map(en)).sort_values(["category", "script"])
counts = m.change.value_counts().to_dict()
pct = lambda x: f"{x:.0%}"
out = ["# Over-refusal: what Fix 2 newly refuses",
       "", f"Non-English harmless held-out prompts: {len(m)} (paired, same prompts). Refused (judge REFUSAL, not a loop): "
       f"base {m.ref_base.mean():.1%} → LoRA {m.ref_lora.mean():.1%}.",
       f"Flips: **{counts.get('newly refused', 0)} newly refused**, {counts.get('no longer refused', 0)} no longer refused, "
       f"{counts.get('both', 0)} refused by both, {counts.get('neither', 0)} by neither.",
       "", "## By script", "", md(by_script),
       "", "## By XSTest category", "", md(by_cat),
       "", "## Every newly refused prompt", "", "| category | script | prompt (English source) | LoRA reply (start) |", "|---|---|---|---|"]
clip = lambda s, n: str(s).replace("\n", " ").replace("|", "/")[:n]
out += [f"| {r.category} | {r.script} | {clip(r.en, 90)} | {clip(r.response_lora, 90)} |" for r in new.itertuples()]
open("results/overrefusal_lora.md", "w").write("\n".join(out) + "\n")
json.dump({"n": len(m), "refused_base": m.ref_base.mean(), "refused_lora": m.ref_lora.mean(), "flips": counts,
           "by_script": by_script.reset_index().to_dict("records"), "by_category": by_cat.reset_index().to_dict("records")},
          open("results/overrefusal_lora.json", "w"), indent=1, default=float)
print("\n".join(out[:4])); print(by_cat.to_string(formatters={"base": pct, "lora": pct}))
