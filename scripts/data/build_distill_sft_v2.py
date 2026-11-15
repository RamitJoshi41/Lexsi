# /// script
# dependencies = ["pandas", "numpy", "datasets", "huggingface_hub"]
# ///
"""Distillation set v2 = v1 rows + wrapper-augmented TRAIN rows (job_wrap_data.py, judged).
harmful wrapped  -> the refusal target of the same unwrapped prompt in v1 (so the student refuses the request, not the frame)
harmless wrapped -> base model's own reply unless judged REFUSAL or a degenerate loop (dropped)
+ capability-retention rows (job_retain_data.py): base model's own answers to MMLU auxiliary_train MCQs (as-is) and to
  extra everyday train prompts (dropped if judged REFUSAL / loop).
Usage: uv run build_distill_sft_v2.py <wrapped_judged4.jsonl> <v1_dataset_repo> <v2_dataset_repo> [retain_judged4.jsonl]"""
import sys, zlib, json, collections
import pandas as pd
from datasets import load_dataset, Dataset
wf, V1, V2 = sys.argv[1:4]
deg = lambda t: len((t or "").strip().encode()) > 40 and len(zlib.compress((t or "").strip().encode())) / len((t or "").strip().encode()) < 0.15
v1 = load_dataset(V1, split="train").to_pandas()
ref = v1[v1.label == 1].drop_duplicates("prompt").set_index("prompt").completion
w = pd.read_json(wf, lines=True); rows, st = [], collections.Counter()
for r in w.itertuples():
    if r.label == 1:
        if r.base_text not in ref.index: st["harm_no_target"] += 1; continue
        rows.append({"prompt": r.text, "completion": ref[r.base_text], "label": 1, "script": r.script, "src": f"wrap{r.wrapper}", "kind": "wrapped_refusal"}); st["wrapped_refusal"] += 1
    else:
        if r.judge4 == "REFUSAL" or deg(r.response) or len((r.response or "").strip()) < 2: st["drop_wrapped_harmless"] += 1; continue
        rows.append({"prompt": r.text, "completion": r.response.strip(), "label": 0, "script": r.script, "src": f"wrap{r.wrapper}", "kind": "wrapped_self_distill"}); st["wrapped_self_distill"] += 1
if len(sys.argv) > 4:
    for r in pd.read_json(sys.argv[4], lines=True).itertuples():
        resp = (r.response or "").strip()
        if r.src != "mmlu_aux" and (getattr(r, "judge4", "") == "REFUSAL" or deg(resp) or len(resp) < 2): st["drop_retain"] += 1; continue
        if not resp: st["drop_retain"] += 1; continue
        rows.append({"prompt": r.text, "completion": resp, "label": 0, "script": r.script, "src": r.src, "kind": "retain_self_distill"}); st[f"retain_{r.src}"] += 1
D = pd.concat([v1, pd.DataFrame(rows)], ignore_index=True).sample(frac=1, random_state=0).reset_index(drop=True)
print(dict(st), "total", len(D), D.groupby(["label", "kind"]).size().to_dict())
json.dump({"stats": dict(st), "n": len(D)}, open("results/distill_sft_v2_stats.json", "w"), indent=1)
Dataset.from_pandas(D).push_to_hub(V2, split="train", private=False); print("pushed", V2)
