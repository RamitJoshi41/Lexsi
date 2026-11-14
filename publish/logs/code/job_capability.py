# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=4.56", "accelerate", "pandas", "numpy", "huggingface_hub", "pyarrow", "datasets", "auditkit[interop]"]
# ///
"""Job 08: capability check with AuditKIT — MMLU subset, base model vs base+guard (gate v2 + steering @L14, a=1).
Generation is wrapped as an AuditKIT CallableModel; scored with AuditKIT's evaluate() + compare()."""
import sys, json, re, time
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download
import auditkit as ak
from auditkit.model import CallableModel
from auditkit.scenarios.mmlu import MMLUScenario
from auditkit.types import Direction

@ak.scorer(direction=Direction.MAXIMIZE, name="mcq_acc")
def mcq_acc(sample, output):
    m = re.search(r"\b([ABCD])\b", (output or "").strip().upper())
    return float(m is not None and m.group(1) == sample.target)

REPO, MODEL_ID = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 400
TAG = MODEL_ID.split("/")[-1]; ADD_L, ALPHA, BS = 14, 1.0, 16
torch.set_grad_enabled(False); t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
dl = lambda p: hf_hub_download(REPO, p, repo_type="dataset")
df = pd.read_json(dl("codemix_v1.jsonl"), lines=True); fit = set(json.load(open(dl("fit_ids.json"))))
H = np.load(dl(f"runs/baseline/{TAG}/hidden_{TAG}.npy")).astype(np.float32)
y = df.label.values; m = df.id.isin(fit).values & (df.script == "en").values
v = H[m & (y == 1), ADD_L].mean(0) - H[m & (y == 0), ADD_L].mean(0)
g = np.load(dl("gate_v2.npz")); GL, GT = int(g["layer"]), float(g["thr"])

tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
W = torch.tensor(g["w"], device="cuda"); B = float(g["b"]); V = torch.tensor(v, dtype=torch.bfloat16, device="cuda")
st = {"mask": None, "fired": []}
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
        hs = [model.model.layers[GL].register_forward_hook(gate_hook), model.model.layers[ADD_L].register_forward_hook(add_hook)] if guard else []
        outs = []
        for b in range(0, len(prompts), BS):
            chunk = [tok.apply_chat_template([{"role": "user", "content": p + "\nReply with only the letter (A, B, C or D) of the correct option."}],
                                             add_generation_prompt=True, tokenize=False) for p in prompts[b:b+BS]]
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            gen = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
            outs += [tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True) for j in range(len(chunk))]
        for h in hs: h.remove()
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
summary = {"n": N, "base": res["base"].headline, "guarded": res["guarded"].headline,
           "gate_fired": int(sum(st["fired"])), "comparison": str(cmp)[:3000]}
print(json.dumps(summary, indent=1, default=str), flush=True)
json.dump(summary, open("capability.json", "w"), indent=1, default=str)
HfApi().upload_file(path_or_fileobj="capability.json", path_in_repo=f"runs/capability_{TAG}.json", repo_id=REPO, repo_type="dataset")
log("DONE")
