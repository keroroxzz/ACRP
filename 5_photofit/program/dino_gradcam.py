"""GradCAM on the DINOv2 perceptual similarity: which regions of the RENDER most drive (or limit)
its perceptual match to the reference photo. Overlays a heatmap on the render.
Run with the `dino` env:  python dino_gradcam.py render.png [out.png]
"""
from __future__ import annotations
import sys, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PROC = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
MODEL = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(DEV)
import os
HERE = os.path.dirname(os.path.abspath(__file__))
PHOTO = os.path.join(HERE, "reference_photo.jpg")


def sky_crop(pil, top=0.0, bot=0.60):
    w, h = pil.size; return pil.crop((0, int(top*h), w, int(bot*h)))

def to_tensor(pil):
    x = PROC(images=pil, return_tensors="pt")["pixel_values"].to(DEV); return x

@torch.no_grad()
def ref_feats():
    x = to_tensor(sky_crop(Image.open(PHOTO).convert("RGB")))
    o = MODEL(pixel_values=x)
    cls = o.pooler_output if o.pooler_output is not None else o.last_hidden_state[:, 0]
    patch = o.last_hidden_state[:, 1:]
    return cls.detach(), torch.cat([patch.mean(1), patch.std(1)], -1).detach()


def main():
    render_path = sys.argv[1]; out_path = sys.argv[2] if len(sys.argv) > 2 else render_path.replace(".png", "_cam.png")
    cb, sb = ref_feats()
    pil = sky_crop(Image.open(render_path).convert("RGB"))
    x = to_tensor(pil)
    feats = {}
    def hook(m, i, o): feats["A"] = o[0] if isinstance(o, tuple) else o
    h = MODEL.encoder.layer[-1].register_forward_hook(hook)     # last block output (1, 1+N, C)
    o = MODEL(pixel_values=x)
    A = feats["A"]; A.retain_grad()
    cls = o.pooler_output if o.pooler_output is not None else o.last_hidden_state[:, 0]
    patch = o.last_hidden_state[:, 1:]
    sa = torch.cat([patch.mean(1), patch.std(1)], -1)
    S = 0.5*F.cosine_similarity(cls, cb) + 0.5*F.cosine_similarity(sa, sb)
    MODEL.zero_grad(); S.backward()
    g = A.grad[0, 1:]                                           # (N, C) grad on patch tokens
    a = A[0, 1:].detach()
    cam = F.relu((g * a).sum(-1))                               # GradCAM: contribution to similarity
    n = cam.shape[0]; hw = int(round(n**0.5))
    cam = cam.reshape(1, 1, hw, hw)
    cam = (cam - cam.min())/(cam.max()-cam.min()+1e-8)
    W, H = pil.size
    cam = F.interpolate(cam, size=(H, W), mode="bicubic", align_corners=False)[0, 0].cpu().numpy()
    # low CAM = perceptually WEAK/off region -> highlight the deficit (1-cam) in red
    base = np.asarray(pil, float)/255.0
    deficit = (1.0 - cam)[..., None]
    over = base*(1-0.55*deficit) + np.array([1.0, 0.15, 0.1])*0.55*deficit
    h.remove()
    Image.fromarray((np.clip(over, 0, 1)*255).astype(np.uint8)).save(out_path)
    print("perceptual similarity %.4f ; heatmap -> %s (red = perceptually weak/off regions)" % (S.item(), out_path))


if __name__ == "__main__":
    main()
