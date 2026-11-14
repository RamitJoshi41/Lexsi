# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers>=4.56", "safetensors", "numpy", "sentencepiece", "protobuf", "huggingface_hub", "peft"]
# ///
"""Job 26 (Day 4): convert the AlignTune LoRA to a GGUF LoRA (llama.cpp convert_lora_to_gguf.py) so it plugs into
the official CohereLabs/tiny-aya-global-GGUF files in llama.cpp / llama-cpp-python / LM Studio / Ollama.
Usage: job_gguf_lora.py <adapter_repo> <base_model_id>"""
import os, sys, io, zipfile, subprocess, urllib.request, time
from huggingface_hub import snapshot_download, HfApi
ADAPTER, BASE = sys.argv[1], sys.argv[2]
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen("https://codeload.github.com/ggml-org/llama.cpp/zip/refs/heads/master").read()))
z.extractall("llcpp"); root = os.path.join("llcpp", os.listdir("llcpp")[0]); log("llama.cpp extracted", root)
ad = snapshot_download(ADAPTER); base = snapshot_download(BASE, allow_patterns=["*.json", "*.model", "tokenizer*"])
out = "codemix-refusal-lora-f16.gguf"
env = dict(os.environ, PYTHONPATH=os.path.join(root, "gguf-py"))
r = subprocess.run([sys.executable, os.path.join(root, "convert_lora_to_gguf.py"), ad, "--base", base, "--outfile", out, "--outtype", "f16"],
                   env=env, capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-3000:]); assert r.returncode == 0 and os.path.exists(out)
log("gguf lora size MB", round(os.path.getsize(out) / 1e6, 1))
HfApi().upload_file(path_or_fileobj=out, path_in_repo=out, repo_id=ADAPTER, repo_type="model"); log("DONE")
