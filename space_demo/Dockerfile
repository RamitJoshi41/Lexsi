FROM ghcr.io/ggml-org/llama.cpp:server
USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# official Tiny Aya Global GGUF (Q4_K_M, 2.1 GB) + our distilled safety LoRA as a GGUF adapter
RUN curl -fL -o /app/base.gguf https://huggingface.co/CohereLabs/tiny-aya-global-GGUF/resolve/main/tiny-aya-global-q4_k_m.gguf
ARG ADAPTER_REPO=sunit12/tiny-aya-global-codemix-refusal-lora-v0
RUN curl -fL -o /app/guard.gguf https://huggingface.co/${ADAPTER_REPO}/resolve/main/codemix-refusal-lora-f16.gguf
COPY public /app/public
EXPOSE 7860
ENTRYPOINT ["/app/llama-server"]
CMD ["-m", "/app/base.gguf", "--lora", "/app/guard.gguf", "--lora-init-without-apply", "--path", "/app/public", "--host", "0.0.0.0", "--port", "7860", "-c", "2048", "-np", "2", "--threads", "2"]
