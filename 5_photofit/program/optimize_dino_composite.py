"""ViT Visual-Scoring Genetic Algorithm, with SKY COMPOSITING (user's idea).

Before scoring, the render's sky is composited INTO the real photo (the photo's ground/buildings/
foreground are kept), so the ViT (DINOv2) sees a complete natural scene that differs from the photo
ONLY in the sky -> the perceptual score isolates sky style & detail, and the global (CLS) feature
becomes meaningful (no longer dragged down by the missing foreground). A GA maximizes the score.

Run with the `dino` env:  python optimize_dino_composite.py [generations]
"""
from __future__ import annotations
import sys, os, json, numpy as np, glfw, torch, torch.nn.functional as F
from OpenGL.GL import *
from PIL import Image
from scipy.ndimage import binary_opening, binary_closing, binary_dilation, gaussian_filter
from transformers import AutoImageProcessor, AutoModel

HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"..","results")
SHADER=os.path.join(HERE,"cumulus_beauty.glsl"); PHOTO=os.path.join(HERE,"reference_photo.jpg")
VS="#version 330 core\nvoid main(){ vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2)); gl_Position=vec4(p*2.0-1.0,0.0,1.0);}\n"
W,H=512,384                                                     # 4:3 to match the photo
DEV="cuda" if torch.cuda.is_available() else "cpu"

# ---- photo + sky mask (feathered) at (W,H) ----
_p=Image.open(PHOTO).convert("RGB").resize((W,H),Image.LANCZOS)
PH=np.asarray(_p,np.float32)/255.0
def _mask():
    r,g,b=PH[:,:,0],PH[:,:,1],PH[:,:,2]; bright=PH.mean(2)
    green=(g>r+0.03)&(g>b+0.03); rows=np.arange(H)[:,None]*np.ones((1,W)); cols=np.ones((H,1))*np.arange(W)[None,:]
    sky=(rows<0.535*H)&(bright>0.34)&(~green); sky&=~((cols<0.10*W)&(rows<0.12*H))
    sky=binary_closing(binary_opening(sky,np.ones((2,2))),np.ones((4,4)))
    blur=gaussian_filter(bright,9); lines=(blur-bright>0.10)&(bright<0.55)&(b>r)&sky
    sky&=~binary_dilation(lines,np.ones((2,2)))
    return gaussian_filter(sky.astype(np.float32),2.0)          # feathered edge for a seamless composite
MASK=_mask()[:,:,None]                                          # (H,W,1) in [0,1]
Image.fromarray((MASK[:,:,0]*255).astype(np.uint8)).save(os.path.join(OUT,"composite_mask.png"))

P=[("pitch",0.08,0.42),("yaw",-3.14,3.14),("height",0.03,0.55),("sunAz",-3.14,3.14),("sunEl",0.55,1.30),
   ("expo",0.40,0.95),("uCbase",0.55,1.20),("uCtop",2.2,4.0),("uCovLo",0.40,0.58),("uCovHi",0.62,0.82),
   ("uCovFreq",0.10,0.34),("uDetAmp",0.35,0.66),("uDens",3.5,7.5),("uSigma",3.0,6.0),
   ("zenR",0.02,0.12),("zenG",0.12,0.30),("zenB",0.50,0.78),("horR",0.40,0.68),("horG",0.55,0.82),("horB",0.72,0.94)]
D=len(P); NAMES=[p[0] for p in P]; LO=np.array([p[1] for p in P]); HI=np.array([p[2] for p in P])
def unpack(x): return LO+np.clip(x,0,1)*(HI-LO)

def sh(src,st):
    s=glCreateShader(st); glShaderSource(s,src); glCompileShader(s)
    if glGetShaderiv(s,GL_COMPILE_STATUS)!=GL_TRUE: raise RuntimeError(glGetShaderInfoLog(s).decode())
    return s
def progc():
    p=glCreateProgram(); glAttachShader(p,sh(VS,GL_VERTEX_SHADER)); glAttachShader(p,sh(open(SHADER).read(),GL_FRAGMENT_SHADER)); glLinkProgram(p)
    if glGetProgramiv(p,GL_LINK_STATUS)!=GL_TRUE: raise RuntimeError(glGetProgramInfoLog(p).decode())
    return p
def cam_basis(eye,ctr):
    f=ctr-eye; f/=np.linalg.norm(f); r=np.cross([0,1,0],f); r/=np.linalg.norm(r); u=np.cross(f,r); return np.stack([r,u,f],1).astype(np.float32)
_g={}
def render(x):
    v=unpack(x); pr=_g["prog"]; glUseProgram(pr); pv={NAMES[i]:float(v[i]) for i in range(D)}
    eye=np.array([0,pv['height'],0],np.float32); d=np.array([np.sin(pv['yaw'])*np.cos(pv['pitch']),np.sin(pv['pitch']),np.cos(pv['yaw'])*np.cos(pv['pitch'])],np.float32); cam=cam_basis(eye,eye+d)
    sd=np.array([np.cos(pv['sunEl'])*np.sin(pv['sunAz']),np.sin(pv['sunEl']),np.cos(pv['sunEl'])*np.cos(pv['sunAz'])],np.float32)
    U=lambda n: glGetUniformLocation(pr,n)
    glUniform2f(U("uRes"),float(W),float(H)); glUniform3f(U("uEye"),*[float(z) for z in eye]); glUniformMatrix3fv(U("uCam"),1,GL_FALSE,cam.T.copy()); glUniform1f(U("uFocal"),1.05)
    glUniform3f(U("uSunDir"),*[float(z) for z in sd]); glUniform1f(U("uExposure"),pv['expo'])
    for n in ["uCbase","uCtop","uCovLo","uCovHi","uCovFreq","uDetAmp","uDens","uSigma"]: glUniform1f(U(n),pv[n])
    glUniform3f(U("uZen"),pv['zenR'],pv['zenG'],pv['zenB']); glUniform3f(U("uHor"),pv['horR'],pv['horG'],pv['horB'])
    glBindVertexArray(_g["vao"]); glDrawArrays(GL_TRIANGLES,0,3); glFinish()
    return np.frombuffer(glReadPixels(0,0,W,H,GL_RGB,GL_FLOAT),np.float32).reshape(H,W,3)[::-1]
def composite(rnd): return (PH*(1-MASK)+rnd*MASK)              # render sky into the real photo

PROC=AutoImageProcessor.from_pretrained("facebook/dinov2-base"); MODEL=AutoModel.from_pretrained("facebook/dinov2-base").eval().to(DEV)
@torch.no_grad()
def feats(np_img):
    x=PROC(images=Image.fromarray((np.clip(np_img,0,1)*255).astype(np.uint8)),return_tensors="pt").to(DEV); o=MODEL(**x)
    cls=o.pooler_output if o.pooler_output is not None else o.last_hidden_state[:,0]; pt=o.last_hidden_state[:,1:]
    return cls, torch.cat([pt.mean(1),pt.std(1)],-1)
CB,SB=feats(PH)
@torch.no_grad()
def score(np_img):
    c,s=feats(np_img); return (0.5*F.cosine_similarity(c,CB)+0.5*F.cosine_similarity(s,SB)).item()


def main():
    G=int(sys.argv[1]) if len(sys.argv)>1 else 30
    glfw.init()
    for k,val in [(glfw.VISIBLE,glfw.FALSE),(glfw.CONTEXT_VERSION_MAJOR,3),(glfw.CONTEXT_VERSION_MINOR,3),(glfw.OPENGL_PROFILE,glfw.OPENGL_CORE_PROFILE)]: glfw.window_hint(k,val)
    win=glfw.create_window(W,H,"o",None,None); glfw.make_context_current(win)
    tex=glGenTextures(1); glBindTexture(GL_TEXTURE_2D,tex); glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,W,H,0,GL_RGBA,GL_FLOAT,None)
    fbo=glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER,fbo); glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tex,0)
    glViewport(0,0,W,H); _g["prog"]=progc(); _g["vao"]=glGenVertexArrays(1)
    rng=np.random.default_rng(0)
    seed=np.full(D,0.5)
    for n,val in [("pitch",0.16),("yaw",1.62),("height",0.06),("sunAz",1.25),("sunEl",1.09),("expo",0.80),
                  ("uCbase",0.84),("uCtop",3.17),("uCovLo",0.43),("uCovHi",0.68),("uCovFreq",0.236),("uDetAmp",0.56),
                  ("uDens",5.76),("uSigma",4.41),("zenR",0.071),("zenG",0.137),("zenB",0.566),("horR",0.42),("horG",0.708),("horB",0.90)]:
        i=NAMES.index(n); seed[i]=(val-LO[i])/(HI[i]-LO[i])
    N,E=36,4
    pop=np.clip(np.concatenate([[seed],np.tile(seed,(N-1,1))+0.15*rng.standard_normal((N-1,D))]),0,1)
    fit=np.array([score(composite(render(x))) for x in pop]); best=pop[fit.argmax()].copy(); bestf=fit.max()
    print("seed composite DINO sim %.4f"%fit[0])
    for gen in range(G):
        order=np.argsort(-fit); pop,fit=pop[order],fit[order]
        if fit[0]>bestf: bestf,best=fit[0],pop[0].copy()
        if gen%5==0 or gen==G-1: print("gen %3d  best composite DINO sim %.4f"%(gen,bestf))
        newpop=[pop[i].copy() for i in range(E)]; newfit=[fit[i] for i in range(E)]; sig=0.11*(1-0.6*gen/G)
        while len(newpop)<N:
            a,b=rng.integers(0,N//2,2); p1=pop[a] if fit[a]>fit[b] else pop[b]
            a,b=rng.integers(0,N//2,2); p2=pop[a] if fit[a]>fit[b] else pop[b]
            m=rng.random(D)<0.5; child=np.clip(np.where(m,p1,p2)+(rng.random(D)<0.5)*sig*rng.standard_normal(D),0,1)
            newpop.append(child); newfit.append(score(composite(render(child))))
        pop=np.array(newpop); fit=np.array(newfit)
    print("FINAL best composite DINO sim %.4f"%bestf)
    rnd=render(best); comp=composite(rnd)
    Image.fromarray((np.clip(rnd,0,1)*255).astype(np.uint8)).save(os.path.join(OUT,"dino_comp_render.png"))
    Image.fromarray((np.clip(comp,0,1)*255).astype(np.uint8)).save(os.path.join(OUT,"dino_comp_composite.png"))
    Image.fromarray((np.clip(np.concatenate([PH,comp],1),0,1)*255).astype(np.uint8)).save(os.path.join(OUT,"dino_comp_vs_photo.png"))
    json.dump({"params":{NAMES[i]:float(unpack(best)[i]) for i in range(D)},"composite_dino_sim":bestf,"x":best.tolist()},open(os.path.join(OUT,"dino_comp_best.json"),"w"),indent=2)
    glfw.terminate()


if __name__=="__main__":
    main()
