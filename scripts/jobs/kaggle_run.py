"""Kaggle runner (2x T4, fp16) for the Day-4 leftovers after HF Jobs credits ran out.
1. Judge the final adapter (circuit-guided LoRA v2) on attacks_v1 and everyday_test with the same Aya Expanse 8B 4-way judge
   (job_judge4.py, patched only for T4: fp16 instead of bf16, model split over both GPUs).
2. Judge-consistency check: re-judge the already-judged v2c held-out file under a new name, so Kaggle-fp16 labels can be
   compared with the A10G-bf16 labels.
3. Convert the final adapter to a GGUF LoRA (llama.cpp convert_lora_to_gguf.py) and push it to the adapter repo.
Every output is uploaded to the HF work repo by the scripts themselves (HF_TOKEN from Kaggle secrets)."""
import os, sys, subprocess, time
t0 = time.time()
def sh(cmd):
    print(f"\n[{time.time()-t0:6.0f}s] $ {' '.join(cmd)[:200]}", flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if "it/s]" in line or "B/s]" in line: continue
        print(line, end="", flush=True)
    p.wait(); print(f"[{time.time()-t0:6.0f}s] exit {p.returncode}", flush=True); return p.returncode
sh([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.56", "accelerate", "huggingface_hub", "peft", "safetensors", "sentencepiece", "protobuf"])
# everything else runs in a fresh interpreter, so the upgraded packages are imported cleanly
STAGE2 = r'''
import os, sys, subprocess, time
t0 = time.time()
def sh(cmd):
    print(f"\n[{time.time()-t0:6.0f}s] $ {' '.join(cmd)[:200]}", flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if "it/s]" in line or "B/s]" in line: continue
        print(line, end="", flush=True)
    p.wait(); print(f"[{time.time()-t0:6.0f}s] exit {p.returncode}", flush=True); return p.returncode
from huggingface_hub import hf_hub_download, HfApi
W, L = "sunit12/aya-codemix-work", "runs/lora/tiny-aya-global"
j = open(hf_hub_download(W, "scripts/job_judge4.py", repo_type="dataset", force_download=True)).read()
assert 'dtype=torch.bfloat16, device_map="cuda"' in j
open("judge.py", "w").write(j.replace('dtype=torch.bfloat16, device_map="cuda"', 'dtype=torch.float16, device_map="auto"'))
g = hf_hub_download(W, "scripts/job_gguf_lora.py", repo_type="dataset", force_download=True)
import torch; print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)

# 1. final adapter on the two missing test sets
sh([sys.executable, "-u", "judge.py", W, f"{L}/gen_lora_v2c_attacks_v1.jsonl", f"{L}/gen_lora_v2c_everyday_test.jsonl"])
# 2. judge-consistency check (Kaggle fp16 vs A10G bf16 labels on the same 470 replies)
src = hf_hub_download(W, f"{L}/gen_lora_v2c_heldout.jsonl", repo_type="dataset")
HfApi().upload_file(path_or_fileobj=src, path_in_repo=f"{L}/kagglecheck_v2c_heldout.jsonl", repo_id=W, repo_type="dataset")
sh([sys.executable, "-u", "judge.py", W, f"{L}/kagglecheck_v2c_heldout.jsonl"])
# 3. GGUF LoRA of the final adapter
sh([sys.executable, "-u", g, "sunit12/tiny-aya-global-codemix-refusal-lora", "CohereLabs/tiny-aya-global"])
print(f"\nALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)
'''
open('stage2.py', 'w').write(STAGE2)
sh([sys.executable, '-u', 'stage2.py'])
