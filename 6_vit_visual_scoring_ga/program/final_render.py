"""High-res hero render of the VSGA-optimized sky (out/dino_comp_best.json): the standalone sky and
its composite into the reference photo. Run with the `dino` env (or any env with glfw/PyOpenGL/scipy/PIL).
"""
from __future__ import annotations
import os, json, numpy as np, glfw
from OpenGL.GL import *
from PIL import Image
from scipy.ndimage import binary_opening, binary_closing, binary_dilation, gaussian_filter, label

HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"..","results")
SH=open(os.path.join(HERE,"cumulus_beauty.glsl"),encoding="utf-8").read(); PHOTO=os.path.join(HERE,"reference_photo.jpg")
VS="#version 330 core\nvoid main(){ vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2)); gl_Position=vec4(p*2.0-1.0,0.0,1.0);}\n"
W,H=1280,960                                                    # 4:3 hero
pv=json.load(open(os.path.join(OUT,"dino_comp_best.json")))["params"]

PH=np.asarray(Image.open(PHOTO).convert("RGB").resize((W,H),Image.LANCZOS),np.float32)/255.0
def mask():
    # skyline flood-fill mask (same as the optimizer, scaled to hero res): replace ALL sky+clouds
    # incl. the far horizon clouds with render; keep buildings/fields/foreground + photo power lines.
    r,g,b=PH[:,:,0],PH[:,:,1],PH[:,:,2]; bright=PH.mean(2); sat=PH.max(2)-PH.min(2); green=(g>r+0.03)&(g>b+0.03)
    rows=np.arange(H)[:,None]*np.ones((1,W)); cols=np.ones((H,1))*np.arange(W)[None,:]
    building=(rows>0.54*H)&(sat<0.16)&(bright>0.38)&(bright<0.78)
    cand=(bright>0.34)&(~green)&(~building)&(rows<0.63*H); cand&=~((cols<0.10*W)&(rows<0.12*H))
    cand=binary_closing(binary_opening(cand,np.ones((3,3))),np.ones((6,6)))
    lab,_=label(cand); band=lab[:max(3,int(0.05*H))]; tl=set(band[band>0].tolist())  # top-band seed (robust to border erosion)
    sky=binary_closing(np.isin(lab,list(tl)),np.ones((12,12)))
    blur=gaussian_filter(bright,18); lines=binary_dilation((blur-bright>0.09)&(bright<0.6)&sky,np.ones((3,3)))
    sky=sky&~lines
    return gaussian_filter(sky.astype(np.float32),4.0)[:,:,None]
M=mask()

def shc(src,st):
    s=glCreateShader(st); glShaderSource(s,src); glCompileShader(s)
    if glGetShaderiv(s,GL_COMPILE_STATUS)!=GL_TRUE: raise RuntimeError(glGetShaderInfoLog(s).decode())
    return s
glfw.init()
for k,v in [(glfw.VISIBLE,glfw.FALSE),(glfw.CONTEXT_VERSION_MAJOR,3),(glfw.CONTEXT_VERSION_MINOR,3),(glfw.OPENGL_PROFILE,glfw.OPENGL_CORE_PROFILE)]: glfw.window_hint(k,v)
win=glfw.create_window(W,H,"h",None,None); glfw.make_context_current(win)
tex=glGenTextures(1); glBindTexture(GL_TEXTURE_2D,tex); glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,W,H,0,GL_RGBA,GL_FLOAT,None)
fbo=glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER,fbo); glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tex,0)
glViewport(0,0,W,H); pr=glCreateProgram(); glAttachShader(pr,shc(VS,GL_VERTEX_SHADER)); glAttachShader(pr,shc(SH,GL_FRAGMENT_SHADER)); glLinkProgram(pr); glUseProgram(pr); vao=glGenVertexArrays(1)
def cb(e,c): f=c-e; f/=np.linalg.norm(f); r=np.cross([0,1,0],f); r/=np.linalg.norm(r); u=np.cross(f,r); return np.stack([r,u,f],1).astype(np.float32)
eye=np.array([0,pv['height'],0],np.float32); d=np.array([np.sin(pv['yaw'])*np.cos(pv['pitch']),np.sin(pv['pitch']),np.cos(pv['yaw'])*np.cos(pv['pitch'])],np.float32); cam=cb(eye,eye+d)
sd=np.array([np.cos(pv['sunEl'])*np.sin(pv['sunAz']),np.sin(pv['sunEl']),np.cos(pv['sunEl'])*np.cos(pv['sunAz'])],np.float32)
U=lambda n: glGetUniformLocation(pr,n)
glUniform2f(U("uRes"),float(W),float(H)); glUniform3f(U("uEye"),*[float(z) for z in eye]); glUniformMatrix3fv(U("uCam"),1,GL_FALSE,cam.T.copy()); glUniform1f(U("uFocal"),1.05)
glUniform3f(U("uSunDir"),*[float(z) for z in sd]); glUniform1f(U("uExposure"),pv['expo'])
for n in ["uCbase","uCtop","uCovLo","uCovHi","uCovFreq","uDetAmp","uDens","uSigma","uDet2Amp","uDet3Amp"]: glUniform1f(U(n),pv[n])
glUniform3f(U("uZen"),pv['zenR'],pv['zenG'],pv['zenB']); glUniform3f(U("uHor"),pv['horR'],pv['horG'],pv['horB'])
glBindVertexArray(vao); glDrawArrays(GL_TRIANGLES,0,3); glFinish()
rnd=np.frombuffer(glReadPixels(0,0,W,H,GL_RGB,GL_FLOAT),np.float32).reshape(H,W,3)[::-1]
comp=PH*(1-M)+rnd*M
Image.fromarray((np.clip(rnd,0,1)*255).astype(np.uint8)).save(os.path.join(OUT,"hero_sky.png"))
Image.fromarray((np.clip(comp,0,1)*255).astype(np.uint8)).save(os.path.join(OUT,"hero_composite.png"))
Image.fromarray((np.clip(np.concatenate([PH,comp],1),0,1)*255).astype(np.uint8)).save(os.path.join(OUT,"hero_vs_photo.png"))
print("wrote out/hero_sky.png, hero_composite.png, hero_vs_photo.png")
glfw.terminate()
