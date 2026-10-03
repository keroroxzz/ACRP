"""Multi-photo joint ViT Visual-Scoring GA (extends optimize_dino_composite.py to a photo SET).

Each real photo gets its own sky mask (reference_photos/<name>_mask.png, white=sky) and its own set of
PER-PHOTO parameters (camera, sun direction, cloud-type distribution, and the three deck coverages), while
a single set of SHARED parameters (physical atmosphere + cloud scattering/shape + deck altitudes) is
optimized jointly across ALL photos. Rendering uses the 3-deck cumulus_multilayer.glsl (physical Paper-3
sky, so per-photo sun direction reproduces each photo's time-of-day colour).

Fitness = mean DINOv2 composite similarity over a random MINI-BATCH of photos each generation (all
candidates share the same batch for a fair comparison); shared params feel pressure from the whole set,
per-photo params only from their own photo. Global best is re-scored on ALL photos at each checkpoint.

Run with the `dino` env:  python optimize_multiphoto.py [generations] [batch_k]
"""
from __future__ import annotations
import sys, os, glob, json, numpy as np, glfw, torch, torch.nn.functional as F
from OpenGL.GL import *
from PIL import Image
from scipy.ndimage import gaussian_filter
from transformers import AutoModel

HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"..","results")
PHOTODIR=os.path.abspath(os.path.join(HERE,"..","..","..","reference_photos"))
SHADER=os.path.join(HERE,"cumulus_multilayer.glsl")
os.makedirs(OUT,exist_ok=True)
VS="#version 330 core\nvoid main(){ vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2)); gl_Position=vec4(p*2.0-1.0,0.0,1.0);}\n"
W,H=512,384
DEV="cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------ parameters
# SHARED across all photos (physical atmosphere + cloud scattering/shape + deck altitudes).
SHARED=[("uAtmDens",0.6,1.4),("uSunI",2.5,5.5),("uHaze",0.05,0.30),("uAmbStr",0.20,0.80),  # Rayleigh kept weak/dim:
        # too strong scatters the blue out of the view path and the sky turns yellow/olive (see results/rayleigh_sweep.png)
        ("uDens",4.0,20.0),("uSigma",3.0,14.0),("uSilver",0.0,5.0),("uMsBoost",0.0,1.6),
        ("uDetAmp",0.30,0.66),("uDet2Amp",0.05,0.45),("uDet3Amp",0.03,0.30),("uCovFreq",0.10,0.34),
        ("uWarp",0.0,1.2),                                             # flow-field domain-warp (cirrus flow-drag)
        ("uDiffuse",0.0,0.30),("uShear",0.0,3.0),   # wind shear (horizontal offset per unit height)                                         # view-independent multiple-scatter lit face
        # deck altitudes (kept disjoint: max uLowTop <= min uMidBase, etc.); low deck reaches hero uCtop=4.0
        # OCTAVE LADDER. Each band is confined to its OWN octave, and the bounds are adjacent and
        # non-overlapping (c/sqrt2 .. c*sqrt2 about 6.25/12.5/25/50), so the four scales stay distinct
        # and cannot collapse onto one another while the GA places each within its octave.
        ("uFreqB",4.42,8.84),("uFreqD1",8.84,17.68),("uFreqD2",17.68,35.36),("uFreqD3",35.36,70.71),
        ("uLowBase",0.50,1.40),("uLowTop",2.2,4.2),("uMidBase",4.3,5.2),("uMidTop",5.3,6.2),
        ("uHiBase",6.5,7.5),("uHiTop",7.6,8.6)]
# PER-PHOTO OPTIMIZED (only what the ViT can't already pin down): exposure + cloud coverage/type.
# The GEOMETRY (camera pitch, yaw, FoV, sun az/el) is FIXED per photo from a hand-verified pass (FIXED_GEO
# below) — it's directly determinable from each photo, so removing it from the search shrinks the space
# (163 -> ~91 dims) and stops the ViT features from being spent representing geometry we already know.
PERPH=[("expo",0.40,1.10),("uCovLow",0.20,0.85),("uCovMid",0.0,0.75),("uCovHigh",0.0,0.75),
       ("uTypeAmp",0.0,0.20),("uTypeSeed",0.0,8.0)]
SN=[p[0] for p in SHARED]; SLO=np.array([p[1] for p in SHARED]); SHI=np.array([p[2] for p in SHARED]); DS=len(SHARED)
PN=[p[0] for p in PERPH]; PLO=np.array([p[1] for p in PERPH]); PHI=np.array([p[2] for p in PERPH]); DP=len(PERPH)

# FIXED per-photo geometry (yaw=0, height=HEIGHT; pitch/sunAz/sunEl/focal hand-verified against each photo).
HEIGHT=0.06
FIXED_GEO={
 "PXL_20221113_083154230":dict(pitch=0.285,sunAz=0.1145,sunEl=0.130,focal=0.674),
 "PXL_20230218_054142843":dict(pitch=0.690,sunAz=3.000,sunEl=0.700,focal=0.886),
 "PXL_20230529_083824111":dict(pitch=0.599,sunAz=2.600,sunEl=0.800,focal=1.211),
 "PXL_20230916_033807521":dict(pitch=0.399,sunAz=3.000,sunEl=0.850,focal=0.674),
 "PXL_20230916_092533286":dict(pitch=0.392,sunAz=0.2445,sunEl=0.080,focal=0.541),
 "PXL_20230916_092922631":dict(pitch=0.327,sunAz=2.800,sunEl=0.850,focal=1.00),
 "PXL_20230916_092931219":dict(pitch=0.392,sunAz=2.800,sunEl=0.900,focal=0.95),
 "PXL_20231015_091244420":dict(pitch=0.646,sunAz=-0.100,sunEl=0.250,focal=0.850),
 "PXL_20231029_085611625":dict(pitch=0.482,sunAz=0.290,sunEl=0.332,focal=1.00),
 "PXL_20240405_095946838":dict(pitch=0.714,sunAz=0.000,sunEl=0.060,focal=0.541),
 "PXL_20240706_081842778":dict(pitch=0.405,sunAz=0.050,sunEl=0.276,focal=0.702),
 "reference_photo":       dict(pitch=0.146,sunAz=-0.785,sunEl=0.576,focal=1.05)}
_geocfg=os.path.join(HERE,"fixed_geo.json")            # written by geometry_tool.py; overrides the table above
if os.path.exists(_geocfg):
    try:
        # Also ADMIT photos that are only in the json: a newly added reference photo gets its geometry
        # from geometry_tool.py, and keying off the built-in table alone made it a hard KeyError below.
        for _n,_g in json.load(open(_geocfg,encoding="utf-8")).items():
            FIXED_GEO.setdefault(_n,dict(pitch=0.25,sunAz=0.0,sunEl=0.5,focal=1.05))
            FIXED_GEO[_n].update({k:float(v) for k,v in _g.items()})
        print("geometry: fixed_geo.json applied")
    except Exception as _e: print("geometry: fixed_geo.json ignored (%s)"%_e)

def un_shared(xs): return {SN[i]: float(SLO[i]+np.clip(xs[i],0,1)*(SHI[i]-SLO[i])) for i in range(DS)}
def un_photo(xp):  return {PN[i]: float(PLO[i]+np.clip(xp[i],0,1)*(PHI[i]-PLO[i])) for i in range(DP)}

# ------------------------------------------------------------------ photos + masks
# Photos owned by another paper. Paper 6's fitness (and every recorded score) is defined on ITS 12-photo
# set, so a reference photo added for another paper must not silently redefine the metric. Set
# VSGA_ALL=1 to include everything with a mask.
EXCLUDE={"IMG_20200531_141120"}                                      # Research-7 cumulonimbus target
def load_set():
    photos=[]
    keep_all=os.environ.get("VSGA_ALL","")!=""
    for jpg in sorted(glob.glob(os.path.join(PHOTODIR,"*.jpg"))):
        base=os.path.basename(jpg)[:-4]; mp=os.path.join(PHOTODIR,base+"_mask.png")
        if not os.path.exists(mp): continue
        if base in EXCLUDE and not keep_all: continue
        ph=np.asarray(Image.open(jpg).convert("RGB").resize((W,H),Image.LANCZOS),np.float32)/255.0
        mk=np.asarray(Image.open(mp).convert("L").resize((W,H),Image.LANCZOS),np.float32)/255.0
        mk=gaussian_filter(mk,1.5)                                   # soften mask edge for a clean composite
        photos.append((base,ph,mk[:,:,None]))
    only=os.environ.get("VSGA_ONLY","")                             # single-photo mode: optimize shared+per-photo for ONE photo
    if only!="": photos=[photos[int(only)]]
    return photos
PHOTOS=load_set()
assert PHOTOS, f"no photo+mask pairs in {PHOTODIR}"
print("photo set (%d): %s"%(len(PHOTOS),", ".join(b for b,_,_ in PHOTOS)))

# ------------------------------------------------------------------ GL
def sh(src,st):
    s=glCreateShader(st); glShaderSource(s,src); glCompileShader(s)
    if glGetShaderiv(s,GL_COMPILE_STATUS)!=GL_TRUE: raise RuntimeError(glGetShaderInfoLog(s).decode())
    return s
def progc():
    p=glCreateProgram(); glAttachShader(p,sh(VS,GL_VERTEX_SHADER)); glAttachShader(p,sh(open(SHADER,encoding="utf-8").read(),GL_FRAGMENT_SHADER)); glLinkProgram(p)
    if glGetProgramiv(p,GL_LINK_STATUS)!=GL_TRUE: raise RuntimeError(glGetProgramInfoLog(p).decode())
    return p
def cam_basis(eye,ctr):
    f=ctr-eye; f/=np.linalg.norm(f); r=np.cross([0,1,0],f); r/=np.linalg.norm(r); u=np.cross(f,r); return np.stack([r,u,f],1).astype(np.float32)
_g={}
def render(shared,geo,cloud):
    pr=_g["prog"]; glUseProgram(pr); U=lambda n: glGetUniformLocation(pr,n)
    eye=np.array([0,HEIGHT,0],np.float32); yaw=0.0
    d=np.array([np.sin(yaw)*np.cos(geo['pitch']),np.sin(geo['pitch']),np.cos(yaw)*np.cos(geo['pitch'])],np.float32)
    cam=cam_basis(eye,eye+d)
    sd=np.array([np.cos(geo['sunEl'])*np.sin(geo['sunAz']),np.sin(geo['sunEl']),np.cos(geo['sunEl'])*np.cos(geo['sunAz'])],np.float32)
    glUniform2f(U("uRes"),float(W),float(H)); glUniform3f(U("uEye"),*[float(z) for z in eye])
    glUniformMatrix3fv(U("uCam"),1,GL_FALSE,cam.T.copy()); glUniform1f(U("uFocal"),geo['focal'])
    glUniform3f(U("uSunDir"),*[float(z) for z in sd]); glUniform1f(U("uExposure"),cloud['expo'])
    for n in SN: glUniform1f(U(n),shared[n])
    for n in ["uCovLow","uCovMid","uCovHigh","uTypeAmp","uTypeSeed"]: glUniform1f(U(n),cloud[n])
    glBindVertexArray(_g["vao"]); glDrawArrays(GL_TRIANGLES,0,3); glFinish()
    return np.frombuffer(glReadPixels(0,0,W,H,GL_RGB,GL_FLOAT),np.float32).reshape(H,W,3)[::-1]

# ------------------------------------------------------------------ dual-model, multi-depth ViT scoring
# BOTH DINOv2 (informative CLS, good on horizon/structure) AND DINOv3 (better cloud scattering / high cloud)
# score every candidate; their per-photo similarities are averaged. For each model we compare, sky-masked:
#   (a) CLS               -> global semantics
#   (b) FINAL patch style -> whole-image style (mean/std)
#   (c) MID  patch style  -> the "head-input" intermediate features: more SPATIAL DETAIL, less semantic,
#                            which resists the flat-wash gaming that the final-only stats allowed.
# Fitness across photos is 0.5*mean + 0.5*min so the optimizer can't wash one photo flat to lift the average.
SUF="_dual"+(("_p"+os.environ["VSGA_ONLY"]) if os.environ.get("VSGA_ONLY","")!="" else "")  # per-photo outputs don't clobber
V2=AutoModel.from_pretrained("facebook/dinov2-base").eval().to(DEV)
V3DIR=os.path.abspath(os.path.join(HERE,"..","..","..","models"))
V3=AutoModel.from_pretrained(V3DIR,local_files_only=True).eval().to(DEV)
def _meta(name,m):
    L=int(m.config.num_hidden_layers); nreg=int(getattr(m.config,"num_register_tokens",0))
    return dict(name=name,m=m,pstart=1+nreg,mid=max(1,int(round(0.6*L))),L=L)
MODELS=[_meta("v2",V2),_meta("v3",V3)]
IMEAN=torch.tensor([0.485,0.456,0.406],device=DEV).view(1,3,1,1); ISTD=torch.tensor([0.229,0.224,0.225],device=DEV).view(1,3,1,1)
SQ=224; WCLS,WF,WM=0.30,0.30,0.40      # per-model weights: CLS / final-patch style / mid-patch style(detail)
for md in MODELS: print("model %s: hidden=%d patch=%d registers=%d mid-layer=%d/%d"%(md["name"],md["m"].config.hidden_size,md["m"].config.patch_size,md["pstart"]-1,md["mid"],md["L"]))
@torch.no_grad()
def embed(np_img,md):
    t=torch.tensor(np.ascontiguousarray(np_img),device=DEV,dtype=torch.float32).permute(2,0,1)[None]
    t=F.interpolate(t,size=(SQ,SQ),mode="bilinear",align_corners=False); t=(t-IMEAN)/ISTD
    o=md["m"](pixel_values=t,output_hidden_states=True); ps=md["pstart"]
    cls=o.pooler_output if o.pooler_output is not None else o.last_hidden_state[:,0]
    pf=o.last_hidden_state[:,ps:]                              # final-layer patches
    pm=o.hidden_states[md["mid"]][:,ps:]                       # intermediate "head-input" patches (spatial detail)
    return cls,pf,pm
def sky_style(patch,w): m=(patch*w).sum(1); v=((patch-m[:,None])**2*w).sum(1); return torch.cat([m,v.sqrt()],-1)

# ---- cloud-amount term -------------------------------------------------------------------------
# The patch-style score rewards filling the sky with cloud-like texture, and measurement confirmed it:
# the renders averaged +0.24 cloudier than the photos, worst on the CLEAR-BLUE ones (photo 3: 0.10 vs
# 0.77). Add an explicit, interpretable penalty on the sky-region cloud fraction. Blue sky has B well
# above R; cloud is white/grey, so low blue-dominance == cloudy. This cannot be gamed with texture.
CLOUD_W=0.30
def cloudiness(img,mk):
    blueness=np.clip((img[...,2]-img[...,0])*3.0,0.0,1.0)
    w=mk[...,0]; return float(((1.0-blueness)*w).sum()/max(w.sum(),1e-6))
CF_PHOTO=[cloudiness(ph,mk) for _,ph,mk in PHOTOS]

TARGETS=[]                                                     # per photo: [{cls,sf,sm,w} for each model]
for base,ph,mk in PHOTOS:
    per={}
    for md in MODELS:
        cls,pf,pm=embed(ph,md); grid=int(round(pf.shape[1]**0.5))
        w=F.interpolate(torch.tensor(mk[:,:,0],device=DEV)[None,None],size=(grid,grid),mode="area")[0,0].reshape(1,-1,1); w=w/(w.sum()+1e-6)
        per[md["name"]]=dict(cls=cls,sf=sky_style(pf,w),sm=sky_style(pm,w),w=w)
    TARGETS.append(per)
@torch.no_grad()
def score_photo(np_img,i):
    s=0.0
    _,_,mk=PHOTOS[i]
    pen=CLOUD_W*abs(cloudiness(np_img,mk)-CF_PHOTO[i])          # match the photo's cloud amount
    for md in MODELS:
        T=TARGETS[i][md["name"]]; c,pf,pm=embed(np_img,md); w=T["w"]
        s+=(WCLS*F.cosine_similarity(c,T["cls"])+WF*F.cosine_similarity(sky_style(pf,w),T["sf"])
            +WM*F.cosine_similarity(sky_style(pm,w),T["sm"])).item()
    return s/len(MODELS) - pen
def composite(rnd,i): _,ph,mk=PHOTOS[i]; return ph*(1-mk)+rnd*mk
GEO=[FIXED_GEO[b] for b,_,_ in PHOTOS]                             # fixed geometry per photo (index-aligned)
EVALS=[0]                                                           # photo-evaluation counter (fair-budget accounting)
def eval_photo(xs_shared,xp,i):                                     # render photo i (fixed geom + optimized cloud), score
    EVALS[0]+=1
    return score_photo(composite(render(xs_shared,GEO[i],un_photo(xp)),i),i)

# ------------------------------------------------------------------ vector <-> (shared, per-photo) layout
NP=len(PHOTOS); D=DS+NP*DP
def split(x): return x[:DS], [x[DS+i*DP:DS+(i+1)*DP] for i in range(NP)]
def _agg(s): return 0.5*float(np.mean(s))+0.5*float(np.min(s))      # mean + worst-photo (anti-wash)
def full_fitness(x):                                                # over ALL photos
    xs=un_shared(x[:DS]); ps=split(x)[1]; return _agg([eval_photo(xs,ps[i],i) for i in range(NP)])
def batch_fitness(x,batch):                                         # over a photo subset
    xs=un_shared(x[:DS]); ps=split(x)[1]; return _agg([eval_photo(xs,ps[i],i) for i in batch])

# ------------------------------------------------------------------ interleaved per-photo SEED SCAN
# uTypeSeed is a DISCONTINUOUS parameter: nudging it re-rolls the whole cloud layout, so gaussian
# mutation cannot hill-climb it and it only adds noise to the continuous search. Instead we FREEZE it
# during normal generations and, every SEED_EVERY generations, scan it exhaustively per photo. The scan
# is separable because a photo's seed only affects that photo's own score.
SEED_IDX=PN.index("uTypeSeed")
SEED_MASK=np.zeros(D,bool)
for _i in range(NP): SEED_MASK[DS+_i*DP+SEED_IDX]=True
SEED_EVERY=12; SEED_TRIALS=20
def seed_scan(x):
    xs=un_shared(x[:DS]); ps=split(x)[1]; gained=0.0
    for i in range(NP):
        cur=float(ps[i][SEED_IDX]); best_v,best_s=cur,eval_photo(xs,ps[i],i); s0=best_s
        for t in range(SEED_TRIALS):
            v=(t+0.5)/SEED_TRIALS
            xp=ps[i].copy(); xp[SEED_IDX]=v
            sc=eval_photo(xs,xp,i)
            if sc>best_s: best_s,best_v=sc,v
        x[DS+i*DP+SEED_IDX]=best_v; gained+=best_s-s0
    return x,gained/NP

# ------------------------------------------------------------------ BLOCK (alternating-session) mode
# Rationale, and its limits. The joint search carries 93 dims on a population of 32 -- far too few to
# cover that space. Restricting mutation to one small block at a time makes the population adequate for
# the block. NOTE this is block coordinate descent: it CANNOT move along diagonal directions (e.g. raise
# density while lowering extinction to hold opacity), and the parameters here are known to be coupled, so
# it is not obviously better than the joint search -- it is an experiment to be judged on equal budget.
#
# The larger structural win is separability: a photo's expo/coverage/type affect ONLY that photo's score,
# exactly like uTypeSeed. That is 60 of the 93 dims that a joint GA has no reason to search jointly.
BLOCKS=[("shape",  ["uDetAmp","uDet2Amp","uDet3Amp","uCovFreq","uWarp","uShear",
                    "uFreqB","uFreqD1","uFreqD2","uFreqD3",
                    "uLowBase","uLowTop","uMidBase","uMidTop","uHiBase","uHiTop"]),
        ("scatter",["uDens","uSigma","uSilver","uMsBoost","uDiffuse"]),
        ("atmos",  ["uAtmDens","uSunI","uHaze","uAmbStr"])]
BLOCK_IDX=[(n,np.array([SN.index(p) for p in ps],int)) for n,ps in BLOCKS]
PHOTO_CONT=np.array([j for j in range(DP) if j!=SEED_IDX],int)      # expo, uCovLow/Mid/High, uTypeAmp

def block_session(x,idx,rng,gens=8,pop=16,sig=0.14):
    """A small GA that mutates ONLY the given shared-parameter indices."""
    base=x.copy(); popl=[base.copy()]
    for _ in range(pop-1):
        c=base.copy(); c[idx]=np.clip(c[idx]+sig*rng.standard_normal(len(idx)),0,1); popl.append(c)
    popl=np.array(popl)
    for g in range(gens):
        batch=rng.choice(NP,size=min(6,NP),replace=False)
        fit=np.array([batch_fitness(c,batch) for c in popl])
        popl=popl[np.argsort(-fit)]
        s=sig*(1.0-0.6*g/max(gens-1,1)); newp=[popl[0].copy(),popl[1].copy()]
        while len(newp)<pop:
            a,b=rng.integers(0,pop//2,2); child=popl[a].copy()
            child[idx]=np.where(rng.random(len(idx))<0.5,popl[a][idx],popl[b][idx])
            child[idx]=np.clip(child[idx]+s*rng.standard_normal(len(idx)),0,1)
            newp.append(child)
        popl=np.array(newp)
    # The ranking above uses a 6-photo batch but the caller accepts on the FULL set, so the batch winner
    # was often rejected and whole sessions made no progress. Re-score the top few on the full set and
    # hand back the genuine best -- 3 extra full evaluations per session.
    cand=[(full_fitness(popl[i]),i) for i in range(min(3,len(popl)))]
    return popl[max(cand)[1]]

def photo_session(x,rng,iters=5,pop=10,sig=0.11):
    """Per-photo CONTINUOUS params, optimized one photo at a time (they are separable, so this is a
    direct hill-climb on each photo's own score instead of a slice of a 93-dim joint search)."""
    xs=un_shared(x[:DS]); ps=split(x)[1]; gained=0.0
    for i in range(NP):
        cur=ps[i].copy(); cs=eval_photo(xs,cur,i); s0=cs; s=sig
        for _ in range(iters):
            for _k in range(pop):
                c=cur.copy()
                c[PHOTO_CONT]=np.clip(c[PHOTO_CONT]+s*rng.standard_normal(len(PHOTO_CONT)),0,1)
                sc=eval_photo(xs,c,i)
                if sc>cs: cs,cur=sc,c
            s*=0.75
        x[DS+i*DP:DS+(i+1)*DP]=cur; gained+=cs-s0
    return x,gained/NP

def run_blocks(budget,rng,best,bestf):
    """Alternate: shape -> scatter -> atmosphere -> per-photo -> seed, repeated until the photo-evaluation
    BUDGET is spent (same budget the joint run used, so the comparison is fair)."""
    cyc=0
    while EVALS[0]<budget:
        cyc+=1
        for name,idx in BLOCK_IDX:
            cand=block_session(best,idx,rng); f=full_fitness(cand)
            mark="" if f<=bestf else "  <-- accepted"
            if f>bestf: bestf,best=f,cand.copy(); save_best(best,bestf)
            print("  [%-7s] FULL %.4f   best %.4f%s  (evals %d)"%(name,f,bestf,mark,EVALS[0]))
            if EVALS[0]>=budget: break
        if EVALS[0]<budget:
            best,g=photo_session(best.copy(),rng); bestf=full_fitness(best); save_best(best,bestf)
            print("  [photo  ] mean gain %+.4f -> FULL %.4f  (evals %d)"%(g,bestf,EVALS[0]))
        if EVALS[0]<budget:
            best,g=seed_scan(best.copy()); bestf=full_fitness(best); save_best(best,bestf)
            print("  [seed   ] mean gain %+.4f -> FULL %.4f  (evals %d)"%(g,bestf,EVALS[0]))
        print("cycle %d done  FULL %.4f  (evals %d/%d)"%(cyc,bestf,EVALS[0],budget))
    return best,bestf

# ------------------------------------------------------------------ per-parameter step calibration
# A single global mutation sigma implicitly assumes every parameter's sensitivity is proportional to its
# range. Measured at the incumbent, |dS/dx| spans ~380x (uCovHigh 0.20 vs uMidTop 0.0008), so one sigma
# over-perturbs the sensitive parameters and barely explores the dull ones.
#
# CAVEATS, and they matter for reading any experiment built on this:
#  - sensitivity is CONDITIONAL on where every other parameter sits, so these are local slopes at one
#    point, not properties of the parameter. uMidBase/uMidTop look inert only because covMid is currently
#    low; raise mid coverage and they become sensitive. Hence RE-calibration, not a one-off measurement.
#  - it is a DIAGONAL approximation: cross-parameter interaction (off-diagonal curvature) is ignored
#    entirely. Two individually-dull parameters can be sensitive along a combined direction; only
#    something like CMA-ES would see that.
#  - uTypeSeed is discontinuous, so its "derivative" is just the size of a layout re-roll, not a slope.
#    It is excluded (it is frozen and scanned separately anyway).
SIG_EVERY=30; SIG_DAMP=0.5; SIG_CLAMP=(0.3,3.0); SIG_H=0.03
def calibrate_sigma(x):
    sens=np.ones(D)
    xs=lambda v: un_shared(v[:DS])
    def photo_s(v,i): return eval_photo(xs(v),split(v)[1][i],i)
    base_all=[photo_s(x,i) for i in range(NP)]
    for j in range(DS):                                        # shared: affects every photo
        d=[]
        for sgn in (+1,-1):
            v=x.copy(); v[j]=np.clip(v[j]+sgn*SIG_H,0,1)
            d.append(_agg([photo_s(v,i) for i in range(NP)]))
        sens[j]=abs(d[0]-d[1])/(2*SIG_H)
    for k in range(DP):                                        # per-photo: only its own photo moves
        for i in range(NP):
            j=DS+i*DP+k; d=[]
            for sgn in (+1,-1):
                v=x.copy(); v[j]=np.clip(v[j]+sgn*SIG_H,0,1)
                d.append(photo_s(v,i))
            sens[j]=abs(d[0]-d[1])/(2*SIG_H)
    ok=~SEED_MASK
    med=float(np.median(np.maximum(sens[ok],1e-6)))
    sc=(med/np.maximum(sens,1e-6))**SIG_DAMP                   # damped inverse-sensitivity
    sc=np.clip(sc,*SIG_CLAMP); sc[SEED_MASK]=0.0               # seeds never mutate here
    return sc

# ------------------------------------------------------------------ seeds / warm-start
_LADDER_DEF={"uFreqB":6.25,"uFreqD1":12.5,"uFreqD2":25.0,"uFreqD3":50.0}
def seed_vector(rng):
    """Seed from the hero (single-photo VSGA best): carry its cloud shape/scatter + altitude, calibrate the
    physical atmosphere to a deep-blue hero-like sky, and start every photo as a single cumulus deck
    (mid/high OFF) so the baseline reproduces the known-good hero look before the GA diversifies."""
    if os.environ.get("WARM",""):                                     # warm-start: refit the CURRENT best on a
        wp=os.path.join(OUT,"multi_best%s.json"%SUF)                  # (cheaper) shader instead of re-searching
        if os.path.exists(wp):                                        # from the hero. Used after cutting steps.
            j=json.load(open(wp)); xj=j.get("x")
            if xj and len(xj)==D:
                print("WARM-start from",os.path.basename(wp)); return np.array(xj,dtype=float)
            if xj:
                # The parameter set changed. Re-map the SHARED params BY NAME and the per-photo blocks by
                # index; anything new keeps its seed default. Without this a new parameter throws away a
                # perfectly good fit.
                old=j.get("shared",{}); x=None
                print("WARM-start REMAPPED from",os.path.basename(wp),"(%d -> %d dims)"%(len(xj),D))
                x=np.full(D,0.5)
                for i,n in enumerate(SN):
                    if n in old: x[i]=np.clip((old[n]-SLO[i])/(SHI[i]-SLO[i]),0,1)
                    else:        x[i]=np.clip((_LADDER_DEF.get(n,0.5*(SLO[i]+SHI[i]))-SLO[i])/(SHI[i]-SLO[i]),0,1)
                nold=(len(xj)-(len(xj)-DP*NP))//DP if DP else 0
                oldDS=len(xj)-DP*NP
                if oldDS>=0:
                    for i in range(NP): x[DS+i*DP:DS+(i+1)*DP]=np.array(xj[oldDS+i*DP:oldDS+(i+1)*DP])
                return x
    x=np.full(D,0.5); pv={}
    bp_old=os.path.join(OUT,"dino_comp_best.json")
    if os.path.exists(bp_old):
        try: pv=json.load(open(bp_old))["params"]
        except Exception: pv={}
    def sset(name,val):
        if name in SN: i=SN.index(name); x[i]=np.clip((val-SLO[i])/(SHI[i]-SLO[i]),0,1)
    for n in SN:                                                       # shared cloud shape/scatter (matching names)
        if n in pv: sset(n,pv[n])
    if "uCbase" in pv: sset("uLowBase",pv["uCbase"])                   # carry hero cloud altitude into low deck
    if "uCtop"  in pv: sset("uLowTop", pv["uCtop"])
    sset("uAtmDens",1.05); sset("uSunI",4.0); sset("uHaze",0.12); sset("uAmbStr",0.5)  # physical sky ~ hero blue
    sset("uWarp",0.5)                                                 # moderate flow-drag to start
    sset("uDiffuse",0.05); sset("uShear",0.8)
    sset("uFreqB",6.25); sset("uFreqD1",12.5); sset("uFreqD2",25.0); sset("uFreqD3",50.0)   # octave centres                                             # visible lit face from any sun azimuth
    # per-photo OPTIMIZED start: a clean single cumulus deck (uCovLow=0.71 ~ hero coverage; mid/high OFF).
    dflt=dict(expo=pv.get("expo",0.71), uCovLow=0.71, uCovMid=0.0, uCovHigh=0.0,
              uTypeAmp=pv.get("uTypeAmp",0.09), uTypeSeed=1.0)
    for i in range(NP):
        b=DS+i*DP
        for j,n in enumerate(PN):
            x[b+j]=np.clip((dflt[n]-PLO[j])/(PHI[j]-PLO[j]),0,1)
        x[b+PN.index("uTypeSeed")]=rng.random()                       # decorrelate cloud layout per photo
    return x

def save_best(x,f):
    xs=un_shared(x[:DS]); ps=split(x)[1]; tiles=[]; perphoto={}
    for i,(base,ph,mk) in enumerate(PHOTOS):
        comp=composite(render(xs,GEO[i],un_photo(ps[i])),i)
        tiles.append(np.clip(np.concatenate([ph,comp],1),0,1))
        perphoto[base]={**un_photo(ps[i]),**GEO[i]}                    # optimized cloud + fixed geometry
    grid=np.concatenate(tiles,0)
    Image.fromarray((grid*255).astype(np.uint8)).save(os.path.join(OUT,"multi_vs_photo%s.png"%SUF))
    json.dump({"model":"v2+v3 dual, multi-depth, mean+min; geometry fixed per photo","fitness":float(f),
               "shared":xs,"per_photo":perphoto,"x":x.tolist()},
              open(os.path.join(OUT,"multi_best%s.json"%SUF),"w"),indent=2)

# ------------------------------------------------------------------ GA
def main():
    G=int(sys.argv[1]) if len(sys.argv)>1 else 40
    K=int(sys.argv[2]) if len(sys.argv)>2 else min(4,NP)
    glfw.init()
    for k,val in [(glfw.VISIBLE,glfw.FALSE),(glfw.CONTEXT_VERSION_MAJOR,3),(glfw.CONTEXT_VERSION_MINOR,3),(glfw.OPENGL_PROFILE,glfw.OPENGL_CORE_PROFILE)]: glfw.window_hint(k,val)
    win=glfw.create_window(W,H,"o",None,None); glfw.make_context_current(win)
    tex=glGenTextures(1); glBindTexture(GL_TEXTURE_2D,tex); glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,W,H,0,GL_RGBA,GL_FLOAT,None)
    fbo=glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER,fbo); glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tex,0)
    glViewport(0,0,W,H); _g["prog"]=progc(); _g["vao"]=glGenVertexArrays(1)
    rng=np.random.default_rng(0)
    N,E=32,4
    SIGSC=np.ones(D); SIGSC[SEED_MASK]=0.0
    seed=seed_vector(rng)
    pop=np.clip(np.concatenate([[seed],np.tile(seed,(N-1,1))+0.12*rng.standard_normal((N-1,D))]),0,1)
    best=seed.copy(); bestf=full_fitness(seed); save_best(best,bestf)
    if os.environ.get("BLOCK",""):
        # BUDGET is what the joint run of G generations would spend, so both modes are compared on the
        # same number of photo evaluations rather than on wall clock or generation count.
        budget=EVALS[0] + G*(N*K + NP//5)
        print("BLOCK mode: alternating sessions, budget %d photo-evals (= joint %d gens)"%(budget,G))
        best,bestf=run_blocks(budget,rng,best,bestf)
        save_best(best,bestf)
        print("FINAL global best FULL dino sim %.4f  (evals %d)"%(bestf,EVALS[0]))
        glfw.terminate(); return
    print("seed FULL dino sim %.4f  (dims=%d, photos=%d, batch=%d)"%(bestf,D,NP,K))
    for gen in range(G):
        batch=rng.choice(NP,size=K,replace=False)
        fit=np.array([batch_fitness(x,batch) for x in pop])
        order=np.argsort(-fit); pop,fit=pop[order],fit[order]
        if gen%5==0 or gen==G-1:
            f_top=full_fitness(pop[0])
            if f_top>bestf: bestf,best=f_top,pop[0].copy(); save_best(best,bestf)
            print("gen %3d  batch best %.4f  incumbent FULL %.4f  globalbest %.4f"%(gen,fit[0],f_top,bestf))
        sig=0.10*(1-0.6*gen/G)
        if gen%SIG_EVERY==0:
            SIGSC[:]=calibrate_sigma(best)
            print('  [sigma calib] per-param step x%.2f..%.2f (median-normalised)'%(SIGSC[~SEED_MASK].min(),SIGSC[~SEED_MASK].max()))
        newpop=[pop[i].copy() for i in range(E)]
        while len(newpop)<N:
            a,b=rng.integers(0,N//2,2); p1=pop[a] if fit[a]>fit[b] else pop[b]
            a,b=rng.integers(0,N//2,2); p2=pop[a] if fit[a]>fit[b] else pop[b]
            m=rng.random(D)<0.5; child=np.clip(np.where(m,p1,p2)+(rng.random(D)<0.5)*sig*SIGSC*rng.standard_normal(D),0,1)
            child[SEED_MASK]=best[SEED_MASK]            # seeds are frozen outside the scan sessions
            newpop.append(child)
        pop=np.array(newpop)
        if (gen+1)%SEED_EVERY==0 and gen<G-1:          # interleaved seed session
            best,g=seed_scan(best.copy()); bestf=full_fitness(best)
            pop[:, SEED_MASK]=best[SEED_MASK]           # propagate the chosen layouts to everyone
            print('  [seed scan] mean per-photo gain %+.4f -> FULL %.4f'%(g,bestf)); save_best(best,bestf)
    fb=full_fitness(pop[0])
    if fb>bestf: bestf,best=fb,pop[0].copy()
    save_best(best,bestf)
    print("FINAL global best FULL dino sim %.4f -> multi_best.json, multi_vs_photo.png"%bestf)
    glfw.terminate()

if __name__=="__main__": main()
