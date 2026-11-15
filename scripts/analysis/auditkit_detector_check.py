# /// script
# dependencies = ["auditkit", "pandas"]
# ///
"""AuditKIT built-in RefusalDetector vs Aya Expanse 8B judge, per script (baseline Tiny Aya Global + Fire)."""
import json, pandas as pd
from auditkit.redteam.detectors.builtin import RefusalDetector
det = RefusalDetector(); out = {}
for tag in ["tiny-aya-global", "tiny-aya-fire"]:
    df = pd.read_json(f"runs/baseline/{tag}/gen_{tag}_judged.jsonl", lines=True)
    df["ak_refusal"] = [det.detect(p, r or "").detected for p, r in zip(df.text, df.response)]
    df["judge_refusal"] = df.judge.eq("REFUSAL")
    g = df.groupby("script").apply(lambda x: pd.Series({
        "judge_refusal_rate": x.judge_refusal.mean(), "auditkit_refusal_rate": x.ak_refusal.mean(),
        "auditkit_recall_of_judge_refusals": (x.ak_refusal & x.judge_refusal).sum() / max(x.judge_refusal.sum(), 1),
        "agreement": (x.ak_refusal == x.judge_refusal).mean()})).reindex(["en","hi_deva","hi_rom","pa_guru","pa_rom"])
    print(f"\n== {tag} ==\n", g.round(3).to_string()); out[tag] = g.round(4).to_dict()
json.dump(out, open("results/auditkit_detector_vs_judge.json", "w"), indent=1)
