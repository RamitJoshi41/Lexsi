---
title: Knows But Doesn't Refuse – live demo
emoji: 🛡️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-4.0
short_description: Tiny Aya with and without our code-mixed safety LoRA
---

Type a request in Hinglish, romanized Punjabi, Hindi, Punjabi or English and see **Tiny Aya Global** answer twice:
once as released, once with our **circuit-guided code-mixed refusal LoRA** (AlignTune SFT on layers 10–20, no runtime hooks).

Runs on a free CPU: llama.cpp server, official `CohereLabs/tiny-aya-global-GGUF` (Q4_K_M) + our adapter
`sunit12/tiny-aya-global-codemix-refusal-lora` (`codemix-refusal-lora-f16.gguf`), switched per request
(`lora` scale 0 = base, 1 = guarded). Replies are short (max 120 tokens) and take ~15–40 s on 2 vCPUs.

**Run it locally (no GPU, ~3 GB RAM):**
```bash
docker build -t knows-but-doesnt-refuse .
docker run -p 7860:7860 knows-but-doesnt-refuse   # then open http://localhost:7860
```
Or without Docker: `llama-server -m tiny-aya-global-q4_k_m.gguf --lora codemix-refusal-lora-f16.gguf --lora-init-without-apply --path public --port 7860`.

Part of *Knows, But Doesn't Refuse* (Lexsi Labs hackathon). Model license: Tiny Aya (CC-BY-NC 4.0).
