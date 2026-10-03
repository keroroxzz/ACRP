"""DINOv2 perceptual similarity between a render and the reference photo (sky region).
Run with the `dino` conda env python.  Standalone test:  python dino_perceptual.py imgA imgB
"""
from __future__ import annotations
import sys, numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

_M = {}
def load(name="facebook/dinov2-base"):
    if "m" not in _M:
        _M["proc"] = AutoImageProcessor.from_pretrained(name)
        _M["m"] = AutoModel.from_pretrained(name).eval().to("cuda" if torch.cuda.is_available() else "cpu")
        _M["dev"] = next(_M["m"].parameters()).device
    return _M["proc"], _M["m"], _M["dev"]

def sky_crop(pil, top=0.0, bot=0.60):
    w, h = pil.size; return pil.crop((0, int(top*h), w, int(bot*h)))

@torch.no_grad()
def embed(pil):
    proc, m, dev = load()
    x = proc(images=pil, return_tensors="pt").to(dev)
    out = m(**x)
    cls = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0]
    patch = out.last_hidden_state[:, 1:]                       # (1,N,C) spatial tokens
    return cls, patch

def similarity(pilA, pilB):
    ca, pa = embed(pilA); cb, pb = embed(pilB)
    cls = torch.cosine_similarity(ca, cb).item()              # global semantic match
    # style: translation-invariant patch-feature statistics (mean+std) -> texture/"cumulus-ness"
    sa = torch.cat([pa.mean(1), pa.std(1)], -1); sb = torch.cat([pb.mean(1), pb.std(1)], -1)
    style = torch.cosine_similarity(sa, sb).item()
    return 0.5*cls + 0.5*style, cls, style


if __name__ == "__main__":
    a = Image.open(sys.argv[1]).convert("RGB"); b = Image.open(sys.argv[2]).convert("RGB")
    a = sky_crop(a); b = sky_crop(b)
    s, c, st = similarity(a, b)
    print("DINOv2 perceptual similarity: %.4f  (cls %.4f, style %.4f)" % (s, c, st))
