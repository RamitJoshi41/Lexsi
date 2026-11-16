# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=4.56", "accelerate", "pandas", "numpy", "huggingface_hub", "pyarrow", "datasets", "auditkit[interop]", "bitsandbytes", "peft"]
# ///
"""Job 08: capability check with AuditKIT — MMLU subset, base model vs base+guard (gate v2 + steering @L14, a=1).
Generation is wrapped as an AuditKIT CallableModel; scored with AuditKIT's evaluate() + compare()."""
import sys, json, re, time, os
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download
import auditkit as ak
from auditkit.model import CallableModel
from auditkit.scenarios.mmlu import MMLUScenario
from auditkit.types import Direction

def is_correct(target, output):
    m = re.search(r"\b([ABCD])\b", (output or "").strip().upper())
    return m is not None and m.group(1) == target

@ak.scorer(direction=Direction.MAXIMIZE, name="mcq_acc")
def mcq_acc(sample, output):
    return float(is_correct(sample.target, output))

REPO, MODEL_ID = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 400
TAG = MODEL_ID.split("/")[-1]; ADD_L, ALPHA, BS = 14, 1.0, 16
torch.set_grad_enabled(False); t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
dl = lambda p: hf_hub_download(REPO, p, repo_type="dataset")
df = pd.read_json(dl("codemix_v1.jsonl"), lines=True); fit = set(json.load(open(dl("fit_ids.json"))))
H = np.load(dl(f"runs/baseline/{TAG}/hidden_{TAG}.npy")).astype(np.float32)
y = df.label.values; m = df.id.isin(fit).values & (df.script == "en").values
v = H[m & (y == 1), ADD_L].mean(0) - H[m & (y == 0), ADD_L].mean(0)
GATE_FILE = os.environ.get("GATE_FILE", "gate_v2.npz"); g = np.load(dl(GATE_FILE)); GL, GT = int(g["layer"]), float(g["thr"])

tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
QUANT = os.environ.get("QUANT", "")
if QUANT:
    from transformers import BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16), dtype=torch.bfloat16, device_map="cuda").eval()
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
ADAPTER = os.environ.get("ADAPTER", "")   # Day 4: compare base vs base+distilled LoRA (no hooks) instead of the hook guard
if ADAPTER:
    from peft import PeftModel
    lora_model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda"), ADAPTER).merge_and_unload().eval()
W = torch.tensor(g["w"], device="cuda"); B = float(g["b"]); V = torch.tensor(v, dtype=torch.bfloat16, device="cuda")
st = {"mask": None, "fired": []}
answers = {"base": {}, "guarded": {}}   # prompt -> raw reply, per side: lets anyone check which answers changed
def gate_hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if h.shape[1] > 1:
        p = torch.sigmoid(h[:, -1].float() @ W + B); st["mask"] = (p > GT).to(h.dtype); st["fired"] += (p > GT).tolist()
    return out
def add_hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    h = h + ALPHA * st["mask"][:, None, None] * V
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

def make_fn(guard):
    def fn(prompts):
        mdl = lora_model if (guard and ADAPTER) else model
        hs = [model.model.layers[GL].register_forward_hook(gate_hook), model.model.layers[ADD_L].register_forward_hook(add_hook)] if (guard and not ADAPTER) else []
        outs = []
        for b in range(0, len(prompts), BS):
            chunk = [tok.apply_chat_template([{"role": "user", "content": p + "\nReply with only the letter (A, B, C or D) of the correct option."}],
                                             add_generation_prompt=True, tokenize=False) for p in prompts[b:b+BS]]
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            gen = mdl.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
            outs += [tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True) for j in range(len(chunk))]
        for h in hs: h.remove()
        answers["guarded" if guard else "base"].update(zip(prompts, outs))
        return outs
    return fn

samples = MMLUScenario("all").samples()
rng = np.random.default_rng(0); samples = [samples[i] for i in rng.choice(len(samples), N, replace=False)]
from auditkit.sample import Sample
samples = [Sample(input=s.input + "\n" + "\n".join(f"{'ABCD'[k]}. {c}" for k, c in enumerate(s.choices)), target="ABCD"[int(s.target)]) for s in samples]
log("mmlu samples", len(samples))
res = {}
for name, guard in [("base", False), ("guarded", True)]:
    st["fired"] = []
    r = ak.evaluate(samples, CallableModel(make_fn(guard), name=f"{TAG}-{name}"), scorers=[mcq_acc], cache=False)
    res[name] = r
    log(name, "headline:", r.headline, "| gate fired on", sum(st["fired"]), "/", len(st["fired"]) if guard else "-")
cmp = ak.compare(res["base"], res["guarded"])

def _try(f):  # the GPU work is done by now: one failing report field must not lose the rest of the summary
    try: return f()
    except Exception as e: return {"error": repr(e)}

# str(cmp) is only the object repr (RunComparison has no __str__), so store its report fields explicitly.
# sample_summary counts answers that flipped between base and guarded; significance is a paired-bootstrap CI.
comparison = {"summary": _try(cmp.summary),
              "metric_deltas": _try(lambda: [{k: getattr(v, "value", v) for k, v in vars(d).items()} for d in cmp.metric_deltas()]),
              "significance": _try(lambda: cmp.significance("mcq_acc")),
              "sample_summary": _try(lambda: cmp.sample_summary("mcq_acc"))}
# Per-question record + our own flip count and paired bootstrap CI (independent of the AuditKIT API)
def answer_for(side, question):  # AuditKIT may wrap the prompt it sends, so fall back to "sent prompt contains the question"
    a = answers[side]
    if question in a: return a[question]
    hits = [v for k, v in a.items() if question in k]
    return hits[0] if len(hits) == 1 else None
pq = pd.DataFrame([{"q": i, "target": smp.target, "base": answer_for("base", smp.input), "guarded": answer_for("guarded", smp.input)}
                   for i, smp in enumerate(samples)])
for c in ["base", "guarded"]:
    pq[c + "_correct"] = [float(is_correct(t, o)) for t, o in zip(pq.target, pq[c])]
mq = pq[pq.base.notna() & pq.guarded.notna()]; matched = len(mq)
d = (mq.guarded_correct - mq.base_correct).values
boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
per_question = {"matched": matched, "answers_changed": int((mq.base.str.strip() != mq.guarded.str.strip()).sum()),
                "newly_correct": int(((mq.base_correct == 0) & (mq.guarded_correct == 1)).sum()),
                "newly_wrong": int(((mq.base_correct == 1) & (mq.guarded_correct == 0)).sum()),
                "delta": float(d.mean()), "delta_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
log("per question:", per_question)
if matched < len(pq): log(f"WARNING: only {matched}/{len(pq)} questions matched to a reply; per-question numbers cover those only")
summary = {"n": N, "base": res["base"].headline, "guarded": res["guarded"].headline,
           "gate_fired": int(sum(st["fired"])), "per_question": per_question, "comparison": comparison}
print(json.dumps(summary, indent=1, default=str), flush=True)
json.dump(summary, open("capability.json", "w"), indent=1, default=str)
OUT = f"runs/capability_{TAG}_{(ADAPTER.split('/')[-1]) if ADAPTER else GATE_FILE.replace('.npz', '')}{'_' + QUANT if QUANT else ''}"
pq.to_json("capability_per_question.jsonl", orient="records", lines=True, force_ascii=False)
HfApi().upload_file(path_or_fileobj="capability.json", path_in_repo=OUT + ".json", repo_id=REPO, repo_type="dataset")
HfApi().upload_file(path_or_fileobj="capability_per_question.jsonl", path_in_repo=OUT + "_per_question.jsonl", repo_id=REPO, repo_type="dataset")
log("DONE")
