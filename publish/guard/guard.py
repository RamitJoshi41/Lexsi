"""CodeMix Refusal Guard for Tiny Aya Global.

A probe-gated, layer-targeted refusal-steering hook. No weights are changed.
  1. Gate: a linear probe on the last prompt token at layer 9 decides whether the request is harmful.
  2. Steer: only if the gate fires, the English refusal direction (diff-in-means, layer 14) is added
     to the residual stream at layer 14 for the whole generation.
If the gate does not fire, the model runs exactly as before.

Usage:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from guard import CodeMixRefusalGuard
    tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global"); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype="bfloat16", device_map="cuda")
    guard = CodeMixRefusalGuard.from_pretrained("sunit12/tiny-aya-global-codemix-refusal-guard").attach(model)
    ... model.generate(...) as usual; guard.last_scores holds the gate probabilities of the last batch.
    guard.detach()
"""
from __future__ import annotations
import numpy as np
import torch


class CodeMixRefusalGuard:
    def __init__(self, w, b, gate_layer: int, threshold: float, vector, steer_layer: int, alpha: float = 1.0):
        self.w, self.b = np.asarray(w, np.float32), float(b)
        self.gate_layer, self.threshold = int(gate_layer), float(threshold)
        self.vector, self.steer_layer, self.alpha = np.asarray(vector, np.float32), int(steer_layer), float(alpha)
        self._handles, self._mask, self.last_scores = [], None, None

    @classmethod
    def from_pretrained(cls, repo_or_dir: str, alpha: float = 1.0):
        import os
        if os.path.isdir(repo_or_dir):
            gp, vp = os.path.join(repo_or_dir, "gate_v2.npz"), os.path.join(repo_or_dir, "refusal_vectors.npz")
        else:
            from huggingface_hub import hf_hub_download
            gp, vp = hf_hub_download(repo_or_dir, "gate_v2.npz"), hf_hub_download(repo_or_dir, "refusal_vectors.npz")
        g, v = np.load(gp), np.load(vp)
        L = int(v["steer_layer"])
        return cls(g["w"], g["b"], int(g["layer"]), float(g["thr"]), v["vectors"][L], L, alpha)

    def attach(self, model):
        layers = model.model.layers
        dev = next(model.parameters()).device
        W = torch.tensor(self.w, device=dev)
        V = torch.tensor(self.vector, device=dev, dtype=next(model.parameters()).dtype)

        def gate_hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1:  # prefill: decide once per prompt
                p = torch.sigmoid(h[:, -1].float() @ W + self.b)
                self.last_scores = p.detach().cpu().numpy()
                self._mask = (p > self.threshold).to(h.dtype)
            return out

        def steer_hook(mod, inp, out):
            if self._mask is None or not bool(self._mask.any()):
                return out
            h = out[0] if isinstance(out, tuple) else out
            h = h + self.alpha * self._mask[:, None, None] * V
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

        self._handles = [layers[self.gate_layer].register_forward_hook(gate_hook),
                         layers[self.steer_layer].register_forward_hook(steer_hook)]
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles, self._mask = [], None
