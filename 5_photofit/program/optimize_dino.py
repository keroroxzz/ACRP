"""DINOv2-guided optimization of the beauty cumulus shader. Renders (glfw+OpenGL) and scores
(DINOv2 perceptual similarity to the reference photo) in one process; a genetic algorithm maximizes
the perceptual similarity -> the shader is guided by PERCEPTION, not RMSE. Run with the `dino` env.
Usage:  python optimize_dino.py [generations]
"""
from __future__ import annotations
import sys, os, json, numpy as np, glfw, torch, torch.nn.functional as F
from OpenGL.GL import *
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "results")
SHADER = os.path.join(HERE, "cumulus_beauty.glsl"); PHOTO = os.path.join(HERE, "reference_photo.jpg")
VS = ("#version 330 core\nvoid main(){ vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));"
      " gl_Position=vec4(p*2.0-1.0,0.0,1.0); }\n")
W, H = 640, 384                                                # low-res for GA speed (DINO resizes anyway)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

P = [("pitch",0.10,0.45),("yaw",-3.14,3.14),("height",0.05,0.60),("sunAz",-3.14,3.14),("sunEl",0.55,1.30),
     ("expo",0.40,0.90),("uCbase",0.60,1.20),("uCtop",2.2,4.0),("uCovLo",0.42,0.58),("uCovHi",0.62,0.80),
     ("uCovFreq",0.10,0.30),("uDetAmp",0.35,0.65),("uDens",3.5,7.0),("uSigma",3.0,6.0),
     ("zenR",0.02,0.12),("zenG",0.12,0.30),("zenB",0.50,0.75),("horR",0.40,0.66),("horG",0.55,0.80),("horB",0.72,0.92)]
D = len(P); NAMES=[p[0] for p in P]; LO=np.array([p[1] for p in P]); HI=np.array([p[2] for p in P])
def unpack(x): return LO+np.clip(x,0,1)*(HI-LO)

# ---- GL ----
def sh(src,st):
    s=glCreateShader(st); glShaderSource(s,src); glCompileShader(s)
    if glGetShaderiv(s,GL_COMPILE_STATUS)!=GL_TRUE: raise RuntimeError(glGetShaderInfoLog(s).decode())
    return s
def prog():
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
    glUniform2f(U("uRes"),float(W),float(H)); glUniform3f(U("uEye"),*[float(z) for z in eye]); glUniformMatrix3fv(U("uCam"),1,GL_FALSE,cam.T.copy()); glUniform1f(U("uFocal"),1.0)
    glUniform3f(U("uSunDir"),*[float(z) for z in sd]); glUniform1f(U("uExposure"),pv['expo'])
    for n in ["uCbase","uCtop","uCovLo","uCovHi","uCovFreq","uDetAmp","uDens","uSigma"]: glUniform1f(U(n),pv[n])
    glUniform3f(U("uZen"),pv['zenR'],pv['zenG'],pv['zenB']); glUniform3f(U("uHor"),pv['horR'],pv['horG'],pv['horB'])
    glBindVertexArray(_g["vao"]); glDrawArrays(GL_TRIANGLES,0,3); glFinish()
    img=np.frombuffer(glReadPixels(0,0,W,H,GL_RGB,GL_FLOAT),np.float32).reshape(H,W,3)[::-1]
    return Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8))

# ---- DINO ----
PROC=AutoImageProcessor.from_pretrained("facebook/dinov2-base"); MODEL=AutoModel.from_pretrained("facebook/dinov2-base").eval().to(DEV)
def sky_crop(pil,bot=0.60): w,h=pil.size; return pil.crop((0,0,w,int(bot*h)))
@torch.no_grad()
def feats(pil):
    x=PROC(images=sky_crop(pil),return_tensors="pt").to(DEV); o=MODEL(**x)
    cls=o.pooler_output if o.pooler_output is not None else o.last_hidden_state[:,0]
    pt=o.last_hidden_state[:,1:]; return cls, torch.cat([pt.mean(1),pt.std(1)],-1)
CB,SB=feats(Image.open(PHOTO).convert("RGB"))
@torch.no_grad()
def score(pil):
    c,s=feats(pil); return (0.5*F.cosine_similarity(c,CB)+0.5*F.cosine_similarity(s,SB)).item()


def main():
    G=int(sys.argv[1]) if len(sys.argv)>1 else 30
    glfw.init()
    for k,val in [(glfw.VISIBLE,glfw.FALSE),(glfw.CONTEXT_VERSION_MAJOR,3),(glfw.CONTEXT_VERSION_MINOR,3),(glfw.OPENGL_PROFILE,glfw.OPENGL_CORE_PROFILE)]: glfw.window_hint(k,val)
    win=glfw.create_window(W,H,"o",None,None); glfw.make_context_current(win)
    tex=glGenTextures(1); glBindTexture(GL_TEXTURE_2D,tex); glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,W,H,0,GL_RGBA,GL_FLOAT,None)
    fbo=glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER,fbo); glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tex,0)
    glViewport(0,0,W,H); _g["prog"]=prog(); _g["vao"]=glGenVertexArrays(1)
    rng=np.random.default_rng(0)
    # seed = current tuned look (billow)
    seed=np.full(D,0.5)
    for n,val in [("pitch",0.19),("yaw",3.20),("height",0.12),("sunAz",1.20),("sunEl",1.05),("expo",0.60),
                  ("uCbase",0.85),("uCtop",3.35),("uCovLo",0.485),("uCovHi",0.71),("uCovFreq",0.15),("uDetAmp",0.50),
                  ("uDens",5.4),("uSigma",4.2),("zenR",0.045),("zenG",0.18),("zenB",0.60),("horR",0.50),("horG",0.65),("horB",0.83)]:
        i=NAMES.index(n); seed[i]=(val-LO[i])/(HI[i]-LO[i])
    N,E=36,4
    pop=np.clip(np.tile(seed,(N,1))+np.concatenate([[np.zeros(D)],0.15*rng.standard_normal((N-1,D))]),0,1)
    fit=np.array([score(render(x)) for x in pop])                # maximize
    best=pop[fit.argmax()].copy(); bestf=fit.max()
    for gen in range(G):
        order=np.argsort(-fit); pop,fit=pop[order],fit[order]
        if fit[0]>bestf: bestf,best=fit[0],pop[0].copy()
        if gen%5==0 or gen==G-1: print("gen %3d  best DINO sim %.4f"%(gen,bestf))
        newpop=[pop[i].copy() for i in range(E)]; newfit=[fit[i] for i in range(E)]
        sig=0.12*(1-0.6*gen/G)
        while len(newpop)<N:
            a,b=rng.integers(0,N//2,2); p1=pop[a] if fit[a]>fit[b] else pop[b]
            a,b=rng.integers(0,N//2,2); p2=pop[a] if fit[a]>fit[b] else pop[b]
            m=rng.random(D)<0.5; child=np.clip(np.where(m,p1,p2)+(rng.random(D)<0.5)*sig*rng.standard_normal(D),0,1)
            newpop.append(child); newfit.append(score(render(child)))
        pop=np.array(newpop); fit=np.array(newfit)
    print("FINAL best DINO similarity %.4f"%bestf)
    json.dump({"params":{NAMES[i]:float(unpack(best)[i]) for i in range(D)},"dino_sim":bestf,"x":best.tolist()},
              open(os.path.join(OUT,"dino_best.json"),"w"),indent=2)
    render(best).save(os.path.join(OUT,"dino_best_lowres.png"))
    glfw.terminate()


if __name__=="__main__":
    main()
