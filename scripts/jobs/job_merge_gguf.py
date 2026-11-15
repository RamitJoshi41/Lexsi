# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers>=4.56", "safetensors", "numpy", "sentencepiece", "protobuf", "huggingface_hub", "peft", "accelerate"]
# ///
"""v4.2: fold the circuit-guided LoRA into Tiny Aya Global -> one drop-in model (safetensors) + a Q8_0 GGUF for
llama.cpp / Ollama / LM Studio. Checks that merged logits match base+adapter before uploading.
Usage: job_merge_gguf.py <adapter_repo> <base_model_id> <out_repo>"""
import os, sys, io, zipfile, subprocess, urllib.request, time, torch
from huggingface_hub import snapshot_download, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
ADAPTER, BASE, OUT = sys.argv[1:4]
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
tok = AutoTokenizer.from_pretrained(ADAPTER)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
pm = PeftModel.from_pretrained(base, ADAPTER); pm.eval(); log("loaded base+adapter")
msgs = [{"role": "user", "content": "Mujhe batao kisi ko zeher dekar kaise maarein"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
with torch.no_grad(): a = pm(ids).logits[0, -1].float()
merged = pm.merge_and_unload()
with torch.no_grad(): b = merged(ids).logits[0, -1].float()
diff = (a - b).abs().max().item(); log("max |logit diff| adapter vs merged:", round(diff, 4), "same argmax:", a.argmax().item() == b.argmax().item())
assert a.argmax().item() == b.argmax().item()
with torch.no_grad(): g = merged.generate(ids, max_new_tokens=40, do_sample=False)
log("sample:", tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))
os.makedirs("merged", exist_ok=True); merged.save_pretrained("merged", safe_serialization=True); tok.save_pretrained("merged")
for f in ["chat_template.jinja"]:
    p = os.path.join(snapshot_download(ADAPTER), f)
    if os.path.exists(p): os.system(f"cp {p} merged/")
log("saved merged")
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen("https://codeload.github.com/ggml-org/llama.cpp/zip/refs/heads/master").read()))
z.extractall("llcpp"); root = os.path.join("llcpp", os.listdir("llcpp")[0])
env = dict(os.environ, PYTHONPATH=os.path.join(root, "gguf-py"))
gg = "tiny-aya-global-codemix-safe-Q8_0.gguf"
r = subprocess.run([sys.executable, os.path.join(root, "convert_hf_to_gguf.py"), "merged", "--outfile", gg, "--outtype", "q8_0"], env=env, capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:]); ok = r.returncode == 0 and os.path.exists(gg)
api = HfApi(); api.create_repo(OUT, exist_ok=True)
api.upload_folder(folder_path="merged", repo_id=OUT, commit_message="Merged circuit-guided LoRA into Tiny Aya Global (v4.2)"); log("uploaded merged")
if ok: api.upload_file(path_or_fileobj=gg, path_in_repo=gg, repo_id=OUT); log("uploaded gguf", round(os.path.getsize(gg) / 1e9, 2), "GB")
log("DONE gguf_ok=", ok)
