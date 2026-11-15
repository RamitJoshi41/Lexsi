# /// script
# requires-python = ">=3.11"
# dependencies = ["aligntune>=0.1.35", "torch", "transformers>=4.56", "accelerate", "peft", "trl", "datasets", "pandas", "numpy", "huggingface_hub", "pyarrow"]
# ///
"""Job 23 (Day 4): distil the probe-gated guard into the weights with AlignTune (SFT + LoRA), then evaluate.

Train: AlignTune `create_sft_trainer` (TRL backend, LoRA r=16 on all attention + MLP projections) on
<distill_repo> (prompt/completion pairs built ONLY from training prompts, see build_distill_sft.py).
Eval (no hooks, plain model): codemix held-out (all 5 scripts), new-category test pairs, unseen-language set
(romanized Bengali/Tamil/Marathi/Urdu), filler-wrapper test, and the 600 everyday held-out prompts. Outputs go
to runs/lora/<tag>/gen_<name>.jsonl in the work repo; judged afterwards with job_judge4.py.
Usage: job_distill_train.py <work_repo> <model_id> <distill_dataset_repo> <adapter_repo> [epochs] [lr]"""
import os, sys, time, json, shutil
import numpy as np, pandas as pd, torch
from huggingface_hub import HfApi, hf_hub_download

REPO, MODEL_ID, DATA_REPO, ADAPTER_REPO = sys.argv[1:5]
EPOCHS = float(sys.argv[5]) if len(sys.argv) > 5 else 2; LR = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-4
TAG = MODEL_ID.split("/")[-1]; NAME = os.environ.get("RUN_NAME", "lora_v1")
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
api = HfApi()

from aligntune.core.backend_factory import create_sft_trainer
MODS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
if os.environ.get("LORA_LAYERS"):   # circuit-guided LoRA: only the layers located by the causal / CircuitKIT analysis
    a, b = map(int, os.environ["LORA_LAYERS"].split("-"))
    TARGETS = [f"layers.{l}.{'self_attn' if m.endswith(('q_proj','k_proj','v_proj','o_proj')) else 'mlp'}.{m}" for l in range(a, b + 1) for m in MODS]
else:
    TARGETS = MODS
log("LoRA targets:", len(TARGETS), TARGETS[:3])
trainer = create_sft_trainer(
    model_name=MODEL_ID, dataset_name=DATA_REPO, backend="trl", split="train", output_dir="out/lora",
    num_epochs=int(EPOCHS), batch_size=8, learning_rate=LR, max_seq_length=512, precision="bf16",
    use_peft=True, lora_r=16, lora_alpha=32, lora_dropout=0.05,
    lora_target_modules=TARGETS,
    task_type="sft", seed=0, gradient_accumulation_steps=2, use_gradient_checkpointing=True, logging_steps=10,
)
log("trainer ready:", type(trainer).__name__)
res = trainer.train(); log("train result:", str(res)[:500])
path = trainer.save_model("out/lora_final"); log("saved to", path)
api.create_repo(ADAPTER_REPO, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=path if isinstance(path, str) and os.path.isdir(path) else "out/lora_final",
                  repo_id=ADAPTER_REPO, repo_type="model", commit_message=f"AlignTune SFT LoRA ({NAME})")
log("adapter pushed to", ADAPTER_REPO)
del trainer; torch.cuda.empty_cache()

# ---------------- evaluation: plain merged model, no hooks ----------------
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(base, ADAPTER_REPO).merge_and_unload().eval()
dl = lambda p: hf_hub_download(REPO, p, repo_type="dataset")
cm = pd.read_json(dl("codemix_v1.jsonl"), lines=True); fit = set(json.load(open(dl("fit_ids.json"))))
nc = pd.read_json(dl("newcats_v1.jsonl"), lines=True)
trp = {p for c in nc.category.unique() for p in sorted(nc[nc.category == c].pair.unique())[::2]}
bm = pd.read_json(dl(f"runs/benign/{TAG}/benign_meta.jsonl"), lines=True); bt = np.zeros(len(bm), bool)
for lang in bm.lang.unique():
    ix = np.where(bm.lang == lang)[0]; bt[np.random.default_rng(0).choice(ix, 200, replace=False)] = True
ev = bm[bt].copy(); ev["script"] = ev.lang.map({"English": "en", "Hindi": "hi_deva", "Panjabi": "pa_guru"}); ev["label"] = 0
ev["id"] = ["everyday_te_%03d" % i for i in range(len(ev))]
SETS = {"heldout": cm[~cm.id.isin(fit)], "newcats_test": nc[~nc.pair.isin(trp)],
        "unseen_langs_v1": pd.read_json(dl("unseen_langs_v1.jsonl"), lines=True),
        "filler_test": pd.read_json(dl("filler_test.jsonl"), lines=True), "everyday_test": ev}
for name, df in SETS.items():
    df = df.reset_index(drop=True)
    ps = [tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False) for t in df.text]
    order = np.argsort([len(p) for p in ps]); resp = [None] * len(ps)
    mx = 160
    for b in range(0, len(order), 24):
        idx = order[b:b + 24]
        enc = tok([ps[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx): resp[i] = tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    out = df.copy(); out["response"] = resp; out["config"] = NAME
    p = f"runs/lora/{TAG}/gen_{NAME}_{name}.jsonl"; os.makedirs(os.path.dirname("out/" + p), exist_ok=True)
    out.to_json("out/" + p, orient="records", lines=True, force_ascii=False)
    api.upload_file(path_or_fileobj="out/" + p, path_in_repo=p, repo_id=REPO, repo_type="dataset"); log("eval", name, len(out))
log("DONE")
