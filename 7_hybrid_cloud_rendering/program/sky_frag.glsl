#version 330 core
// ============================================================
// Realistic layered sky: low/mid/high cloud decks + a cumulonimbus tower that
// punches through all three layers (base ~0.9km, anvil ~7.5km). One shared
// density; METHOD selects the renderer so the comparison is apples-to-apples:
//   METHOD 0 = A3 (finest DDA + per-cell GL optical depth + 2-octave empty skip
//              + per-pixel jittered sun)
//   METHOD 1 = AAA raymarch baseline, tuned to be STRONG: coarse cloud-presence
//              empty-space skipping (big steps in clear air, fine steps in cloud),
//              cheap few-step sun march + powder term, transmittance early-out,
//              blue-noise jittered start.
// Injected #defines: METHOD, CAM_STEPS, SUN_STEPS, BIG_STEP, A3_CAMF, A3_CAMGL,
//   A3_SUNF, A3_SUNGL, SUN_JITTER, INSTRUMENT.
// ============================================================
out vec4 fragColor;
uniform vec2 uRes;
uniform vec3 uEye;
uniform mat3 uCam;
uniform vec3 uSunDir;
uniform float uFocal;
// Research-7 cumulonimbus morphology controls (fixed defaults supplied by host).
uniform float uCbLean,uCbBaseR,uCbTopR,uCbWarp,uCbDetailWarp;
uniform float uCbMacroScale,uCbBillowScale,uCbRidge,uCbMicro;
uniform float uCbAnvil,uCbAnvilDens,uCbCrown;
uniform float uCbX,uCbZ;
uniform sampler3D uMacroBounds;
uniform sampler3D uMacroCumulants;
uniform vec3 uMacroOrigin,uMacroCellSize,uMacroDims;
uniform float uBoundLayer,uMacroMargin,uClosureK;

#ifndef USE_MACRO_TEX
#define USE_MACRO_TEX 0
#endif
#ifndef BUILD_BOUNDS
#define BUILD_BOUNDS 0
#endif
#ifndef BUILD_CUMULANTS
#define BUILD_CUMULANTS 0
#endif
#ifndef H2_MODE
#define H2_MODE 0
#endif
#ifndef H3_BOUNDARY_REFINE
#define H3_BOUNDARY_REFINE 0
#endif
#ifndef CLOUDWORKS_SHADOW_BUDGET
#define CLOUDWORKS_SHADOW_BUDGET 0
#endif
#ifndef CLOUDWORKS_SHADOW_T
#define CLOUDWORKS_SHADOW_T 0.10
#endif

#define PI 3.14159265358979
#define SIGMA_T 7.0
// per-deck view optics: background decks are optically THIN (fair-weather cloud that must let the blue
// through), the cumulonimbus is DENSE. BG_FADE applies aerial perspective to the background so a wide
// low deck fades into the sky with distance instead of becoming a flat white wall.
#define SIG_BG 2.8
#define SIG_CB 7.0
#define BG_FADE 0.050
#define ALBEDO 0.94
#define HG_G 0.5
#define YMIN 0.5
#define YMAX 8.0
#define CB_X 6.0
#define CB_Z 9.0
#define DENS 2.4
#ifndef SMALL_STEP
#define SMALL_STEP 0.05
#endif
#ifndef BIG_STEP
#define BIG_STEP 0.30
#endif
#ifndef MAXI
#define MAXI 1200
#endif
#ifndef JITTER_ON
#define JITTER_ON 1
#endif
#ifndef SUN_JITTER
#define SUN_JITTER 1.0
#endif

#define SUN normalize(uSunDir)
const vec3 SUNCOL = vec3(1.7, 1.45, 1.15) * 4.2;
const vec3 SKYTOP = vec3(0.20, 0.38, 0.70);
const vec3 SKYHOR = vec3(0.72, 0.80, 0.90);

const float GL2X[2] = float[2](-0.5773502691896257, 0.5773502691896257);
const float GL3X[3] = float[3](-0.7745966692414834, 0.0, 0.7745966692414834);
const float GL3W[3] = float[3](0.5555555555555556, 0.8888888888888888, 0.5555555555555556);
const float GL5X[5] = float[5](-0.906179845938664,-0.538469310105683,0.0,0.538469310105683,0.906179845938664);
const float GL5W[5] = float[5](0.236926885056189,0.478628670499366,0.568888888888889,0.478628670499366,0.236926885056189);

float gJit = 0.0;
float gEvals = 0.0, gFullEvals = 0.0;
float gSunRays = 0.0, gSkipped = 0.0, gSegs = 0.0;
float gActiveIntervals = 0.0, gActiveLength = 0.0;
float gBgD = 0.0, gCbD = 0.0;   // last cloudDensity() split: background decks vs cumulonimbus (per-deck optics)

float hash13(vec3 p){ p = fract(p*vec3(0.1031,0.1030,0.0973)); p += dot(p,p.yzx+33.33); return fract((p.x+p.y)*p.z); }
float noiseCubic(vec3 p){
    vec3 i=floor(p), f=p-i; vec3 u=f*f*(3.0-2.0*f);
    float n000=hash13(i+vec3(0,0,0)),n100=hash13(i+vec3(1,0,0)),n010=hash13(i+vec3(0,1,0)),n110=hash13(i+vec3(1,1,0));
    float n001=hash13(i+vec3(0,0,1)),n101=hash13(i+vec3(1,0,1)),n011=hash13(i+vec3(0,1,1)),n111=hash13(i+vec3(1,1,1));
    float x00=mix(n000,n100,u.x),x10=mix(n010,n110,u.x),x01=mix(n001,n101,u.x),x11=mix(n011,n111,u.x);
    return mix(mix(x00,x10,u.y),mix(x01,x11,u.y),u.z);
}
float fbm(vec3 p,int n){ float d=0.0,a=0.5,f=1.0; for(int o=0;o<n;o++){d+=a*noiseCubic(p*f);a*=0.5;f*=2.0;} return d; }
float billowFbm(vec3 p,int n){
    float d=0.0,a=0.5,f=1.0,w=0.0;
    for(int o=0;o<n;o++){ d+=a*abs(noiseCubic(p*f)*2.0-1.0); w+=a; a*=0.5; f*=2.03; }
    return d/max(w,1e-4);
}
float ridgedFbm(vec3 p,int n){
    float d=0.0,a=0.5,f=1.0,w=0.0;
    for(int o=0;o<n;o++){ float r=1.0-abs(noiseCubic(p*f)*2.0-1.0); d+=a*r*r; w+=a; a*=0.5; f*=2.07; }
    return d/max(w,1e-4);
}
vec3 cloudWarp(vec3 p,float scale,float amount){
    // Low-frequency vector displacement decorrelates the later detail stacks.
    // The vertical component is weaker so buoyant structures remain coherent.
    vec3 w=vec3(fbm(p*scale+vec3(13,1,7),3),
                fbm(p*scale+vec3(3,17,11),3),
                fbm(p*scale+vec3(19,5,2),3))-vec3(0.44);
    w.y*=0.35;
    return p+w*amount;
}
float remapDensity(float x,float edge){
    return clamp((x-edge)/max(1.0-edge,1e-4),0.0,1.0);
}
float hg(float y,float a,float b,float fa,float fb){ return smoothstep(a-fa,a+fa,y)*smoothstep(b+fb,b-fb,y); }

// ---- Paper-6 multilayer low-cumulus morphology (transplanted) -------------------------------
// The background low deck used to be one flat fbm ("deliberately simple"); it read far worse
// than the hero sky. These bring over the multilayer lowDeck shaping: an asymmetric convex
// cauliflower profile, a billow body, three successive-remap erosion passes, and wind shear.
// Params are FIXED at the multilayer best values (this deck is not GA-searched here).
float remapd(float x,float a,float b,float c,float d){ return c+(clamp(x,a,b)-a)*(d-c)/max(b-a,1e-4); }
vec3 flowWarpF(vec3 p,float amp,float freq,int oct){
    if(amp<1e-3) return p;
    float c=fbm(p*freq*0.30 + vec3(13.0,5.0,21.0), 1);
    vec3 f=vec3(1.6*fbm(p*freq+vec3(0.0,20.0,0.0),oct), 0.35*fbm(p*freq+vec3(40.0,0.0,10.0),oct), 1.6*fbm(p*freq+vec3(7.0,60.0,3.0),oct))
          -vec3(0.8,0.17,0.8);
    return p + f*amp*(0.45+0.8*c);
}
const float LOW_DET_FREQ=1.7, LOW_THRESH=0.08;
const float LOW_BASE=0.72, LOW_TOP=2.18, LOW_SHEAR=0.85, LOW_COVFREQ=0.24;
const float LOW_DETAMP=0.22, LOW_DET2AMP=0.045, LOW_DET3AMP=0.10, LOW_WARP=0.48;
const float LOW_COV=0.46, LOW_TYPEAMP=0.12, LOW_TYPESEED=7.8, LOW_GAIN=1.20;
// full beauty-density low cumulus at world point p (returns a density contribution ~[0,1.9])
float lowCumulus(float y, vec3 p){
    float hf=(y-LOW_BASE)/(LOW_TOP-LOW_BASE); if(hf<=0.0||hf>=1.0) return 0.0;
    vec3 sp=p; sp.x += hf*LOW_SHEAR;                                   // wind shear: leaning column
    float ct=fbm(vec3(sp.xz*LOW_COVFREQ*0.34, 20.0+LOW_TYPESEED), 3);  // large-scale type field
    float th=mix(0.72,0.30,LOW_COV) - LOW_TYPEAMP*(ct-0.5);
    // Moist inflow raises the existing continuous weather field near the
    // storm. This creates clustered feeder cumulus without analytic spheres.
    float inflow=smoothstep(5.2,1.7,length(sp.xz-vec2(uCbX,uCbZ)))
                 *smoothstep(1.0,0.15,hf);
    float weather=fbm(vec3(sp.xz*LOW_COVFREQ, 3.7), 3)+0.13*inflow;
    float cov=smoothstep(th, th+0.30,weather);
    float prof=smoothstep(0.0,0.08,hf)*smoothstep(1.0, mix(0.55,0.18,ct), hf);  // asymmetric convex cauliflower
    float base=cov*prof; if(base<=0.001) return 0.0;
    float body=billowFbm(sp*LOW_DET_FREQ + vec3(0.0,2.0,0.0), 4);
    float shape=base*(0.55 + 0.65*body);
    float er1=LOW_DETAMP*mix(0.9,0.5,hf)*ridgedFbm(sp*LOW_DET_FREQ*2.3 + vec3(5.0,0.0,9.0), 3);
    float d=remapd(shape, er1, 1.0, 0.0, 1.0);
    vec3 pw=flowWarpF(sp, LOW_WARP*0.25, 1.2, 1);
    float er2=LOW_DET2AMP*mix(0.40,1.0,hf)*billowFbm(pw*LOW_DET_FREQ*5.1 + vec3(11.0,3.0,2.0), 2);
    d=remapd(d, er2, 1.0, 0.0, 1.0);
    float er3=LOW_DET3AMP*ridgedFbm(pw*LOW_DET_FREQ*11.3 + vec3(1.0,7.0,4.0), 1);
    d -= er3*smoothstep(0.0,0.45,d)*mix(0.7,0.45,hf);
    d -= LOW_THRESH;
    return clamp(d,0.0,1.0)*LOW_GAIN*mix(0.75,1.25,ct);
}
// cheap shadow variant: cauliflower profile + coarse billow only (no fine erosion / warp)
float lowCumulusShadow(float y, vec3 p){
    float hf=(y-LOW_BASE)/(LOW_TOP-LOW_BASE); if(hf<=0.0||hf>=1.0) return 0.0;
    vec3 sp=p; sp.x += hf*LOW_SHEAR;
    float ct=fbm(vec3(sp.xz*LOW_COVFREQ*0.34, 20.0+LOW_TYPESEED), 3);
    float th=mix(0.72,0.30,LOW_COV) - LOW_TYPEAMP*(ct-0.5);
    float inflow=smoothstep(5.2,1.7,length(sp.xz-vec2(uCbX,uCbZ)))
                 *smoothstep(1.0,0.15,hf);
    float cov=smoothstep(th, th+0.30,
                         fbm(vec3(sp.xz*LOW_COVFREQ, 3.7), 2)+0.13*inflow);
    float prof=smoothstep(0.0,0.08,hf)*smoothstep(1.0, mix(0.55,0.18,ct), hf);
    float body=billowFbm(sp*LOW_DET_FREQ + vec3(0.0,2.0,0.0), 3);
    float d=cov*prof*(0.55+0.65*body) - LOW_THRESH;
    return clamp(d,0.0,1.0)*LOW_GAIN*mix(0.75,1.25,ct);
}

// coarse cloud-presence (backbone only, cheap): used for empty-space skipping AND as the A3 empty bound
float presence(vec3 p){
    gEvals+=1.0;
    float y=p.y;
    // conservative bound for the transplanted low deck (0.89-3.02 km, sheared). It MUST use the SAME
    // wind-shear coordinate as lowCumulus, or the empty-space skip judges sheared cloud as empty, the
    // big step jumps through it, and the deck breaks into horizontal streaks and floating blocks.
    float lhf=clamp((y-LOW_BASE)/(LOW_TOP-LOW_BASE),0.0,1.0); vec3 lp=p; lp.x += lhf*LOW_SHEAR;
    float inflow=smoothstep(5.2,1.7,length(lp.xz-vec2(uCbX,uCbZ)))
                 *smoothstep(1.0,0.15,lhf);
    float low = hg(y,LOW_BASE-0.10,LOW_TOP+0.10,0.35,0.5)
                *smoothstep(0.28,0.75,
                            fbm(vec3(lp.xz*LOW_COVFREQ,3.7),2)+0.13*inflow);
    float mid = hg(y,3.0,4.1,0.30,0.4) * smoothstep(0.55,0.82, fbm(vec3(p.xz*0.5,40.0),2));
    float high= hg(y,5.5,6.9,0.25,0.35)* smoothstep(0.60,0.86, fbm(vec3(p.xz*0.7,80.0),2));
    // Conservative support for the asymmetric multi-turret cumulonimbus below.
    // This is deliberately broader than the density silhouette: presence is an
    // empty-space bound and must never cut off a noisy lobe.
    vec2 cbCenter=vec2(uCbX,uCbZ)+vec2(uCbLean*(y-1.0),0.10*sin(y*1.7));
    float cbR=length(p.xz-cbCenter);
    float towerBound=hg(y,0.8,8.1,0.45,0.50)*smoothstep(2.25,1.75,cbR);
    float baseBound=hg(y,0.62,1.95,0.24,0.38)
                    *smoothstep(uCbBaseR*4.2,uCbBaseR*3.7,cbR);
    vec2 aq=p.xz-(vec2(uCbX,uCbZ)+vec2(1.1,0.1));
    float anvilBound=hg(y,6.55,7.75,0.30,0.35)
                    *smoothstep(1.20,0.92,length(aq/vec2(4.9,2.7)));
    return max(max(low,mid),
               max(high,max(max(towerBound,baseBound),anvilBound)));
}

// H1-only conservative macro support. Thresholds are intentionally wider
// than presence(): false positives cost short-march work, while false empty
// cells can remove a silhouette. H0 continues to use the tighter function.
float presenceH1(vec3 p,vec3 rd){
#if USE_MACRO_TEX==1
    gEvals+=1.0;
    vec3 uv=(p-uMacroOrigin)/(uMacroCellSize*uMacroDims);
    if(any(lessThan(uv,vec3(0.0))) ||
       any(greaterThanEqual(uv,vec3(1.0)))) return 0.0;
    ivec3 ts=textureSize(uMacroBounds,0);
    ivec3 c=clamp(ivec3(floor(vec3(uv.x,uv.z,uv.y)*vec3(ts))),
                  ivec3(0),ts-ivec3(1));
    float b=texelFetch(uMacroBounds,c,0).r;
    ivec3 rayStep=ivec3(sign(vec3(rd.x,rd.z,rd.y)));
    b=max(b,texelFetch(uMacroBounds,
        clamp(c-rayStep,ivec3(0),ts-1),0).r);
    return b+uMacroMargin;
#else
    gEvals+=1.0;
    float y=p.y;
    float lhf=clamp((y-LOW_BASE)/(LOW_TOP-LOW_BASE),0.0,1.0);
    vec3 lp=p; lp.x+=lhf*LOW_SHEAR;
    float low=hg(y,LOW_BASE-0.14,LOW_TOP+0.14,0.42,0.58)
              *smoothstep(0.20,0.68,
                  fbm(vec3(lp.xz*LOW_COVFREQ,3.7),2));
    float mid=hg(y,2.90,4.22,0.38,0.48)
              *smoothstep(0.42,0.76,
                  fbm(vec3(p.xz*0.5,40.0),2));
    float high=hg(y,5.38,7.02,0.32,0.42)
               *smoothstep(0.48,0.80,
                   fbm(vec3(p.xz*0.7,80.0),2));
    vec2 c=vec2(uCbX,uCbZ)
           +vec2(uCbLean*(y-1.0),0.10*sin(y*1.7));
    float r=length(p.xz-c);
    float tower=hg(y,0.68,8.18,0.52,0.58)
                *smoothstep(2.55,2.10,r);
    float base=hg(y,0.52,2.04,0.30,0.44)
               *smoothstep(uCbBaseR*4.35,uCbBaseR*3.90,r);
    vec2 aq=p.xz-(vec2(uCbX,uCbZ)+vec2(1.1,0.1));
    float anvil=hg(y,6.42,7.88,0.38,0.42)
                *smoothstep(1.28,0.98,length(aq/vec2(5.1,2.9)));
    return max(max(low,mid),max(high,max(max(tower,base),anvil)));
#endif
}

#ifndef LAYERS
#define LAYERS 1
#endif
float cloudDensity(vec3 p){
    gEvals+=1.0; gFullEvals+=1.0;
    float y=p.y, bg=0.0;                                          // background low/mid/high decks (optically thin)
#if LAYERS==1
    bg += lowCumulus(y, p);                                       // Paper-6 multilayer cauliflower low deck
    float mc=smoothstep(0.50,0.78,fbm(vec3(p.xz*0.45,40.0),3));
    float mb=hg(y,3.1,4.2,0.28,0.35)*mc;
    bg+=max(mb*fbm(p*vec3(1.3,2.0,1.3)+vec3(5,0,5),4)-0.16,0.0)*1.35;
    float hc=smoothstep(0.52,0.80,fbm(vec3(p.xz*0.6,80.0),3));
    float hb=hg(y,5.6,6.7,0.22,0.3)*hc;
    bg+=max(hb*fbm(p*vec3(2.0,0.5,2.0)+vec3(9,0,0),3)-0.18,0.0)*0.9;
#endif
    float d=0.0;                                                  // cumulonimbus accumulates here (optically dense)
    // CUMULONIMBUS: a family of leaning, merging thermals rather than one
    // cylinder. Different lobes dominate at different heights, creating the
    // stepped cauliflower outline seen in a developing convective tower.
    vec2 center=vec2(uCbX,uCbZ)+vec2(uCbLean*(y-1.0),0.10*sin(y*1.7));
    vec2 q=p.xz-center;
    float ql=length(q);
    // TOWER PROFILE: a WIDE, HEAVY BASE that narrows going up (not a constant-radius cylinder). uCbBaseR is
    // the broad cloud-base radius; the column tapers toward uCbTopR near the top; anvil/crown re-spread it.
    // A broad cumuliform base slab sits under the tower so the storm rises from a thick pad of cloud, and
    // the lobe offsets scale with baseR so a wide storm gets correspondingly wider cauliflower cells.
    float cbhf=clamp((y-0.8)/5.8,0.0,1.0);
    float r0=mix(uCbBaseR, uCbTopR, smoothstep(0.04,0.90,cbhf));
    // Break the analytic cylinder at the envelope level, not only after density
    // evaluation. Broad, height-varying billows perturb the support radius so
    // the silhouette can form cauliflower shoulders instead of a smooth stalk.
    float envelopeNoise=fbm(p*vec3(0.34,0.24,0.34)+vec3(31,7,19),3);
    float heightPulse=0.84+0.16*sin(y*2.35+1.1)
                     +0.10*sin(y*4.7-0.6);
    float rEnvelope=r0*heightPulse*mix(0.72,1.34,envelopeNoise);
    float core=smoothstep(rEnvelope, rEnvelope*0.46, ql);
    // Wide storm base: about 3.8 base radii across the inflow direction.
    // A low-frequency edge field breaks the ellipse so it reads as a connected
    // cloud complex rather than an analytic disk.
    vec2 baseQ=q/vec2(uCbBaseR*3.7,uCbBaseR*2.8);
    float baseMetric=length(baseQ);
    float baseSlab=0.0;
    if(y>0.55 && y<1.95 && baseMetric<1.35){
        float baseEdge=fbm(
            p*vec3(0.36,0.62,0.36)+vec3(41,6,17),3)-0.5;
        float basePlan=smoothstep(1.02+0.30*baseEdge,
                                  0.58+0.12*baseEdge,baseMetric);
        float baseCells=0.62*billowFbm(
            p*vec3(0.82,1.05,0.82)+vec3(8,3,29),4)
            +0.38*fbm(
            p*vec3(1.65,0.72,1.65)+vec3(5,17,2),3);
        float baseMod=0.58+0.68*smoothstep(0.28,0.64,baseCells);
        baseSlab=basePlan*hg(y,0.72,1.72,0.20,0.32)*baseMod;
    }
    float s=uCbBaseR;                                             // lobe offsets scale with the base radius
    float lobe1=smoothstep(0.72*s,0.30*s,length(q-vec2(-0.52,0.18)*s))
                *hg(y,2.0,5.9,0.55,0.65);
    float lobe2=smoothstep(0.86*s,0.36*s,length(q-vec2(0.48,-0.28)*s))
                *hg(y,3.0,6.8,0.60,0.55);
    float lobe3=smoothstep(0.72*s,0.28*s,length(q-vec2(-0.25,-0.62)*s))
                *hg(y,4.1,7.25,0.50,0.45);
    float lobe4=smoothstep(0.68*s,0.25*s,length(q-vec2(0.68,0.30)*s))
                *hg(y,5.1,7.55,0.45,0.40);
    float tower=max(max(max(core,baseSlab),lobe1),max(lobe2,max(lobe3,lobe4)));
    tower*=hg(y,0.6,7.75,0.30,0.48);
    // Scale-separated density detail. A broad warp bends whole cauliflower
    // cells; a second, smaller warp prevents the erosion fields from lining up
    // with them. Billow builds convex lobes, while ridged layers carve their
    // seams and micro-fracture only the optically thin boundary.
    if(tower>0.001){
        vec3 towerWarp=cloudWarp(p,0.20,uCbWarp);
        vec3 detailWarp=cloudWarp(towerWarp,0.62,uCbDetailWarp);
        float towerMacro=fbm(towerWarp*uCbMacroScale+vec3(0,1,0),4);
        float towerBillow=billowFbm(detailWarp*uCbBillowScale+vec3(5,2,9),5);
        float towerRidge=ridgedFbm(detailWarp*2.85+vec3(7,2,5),4);
        float towerMicro=billowFbm(detailWarp*7.4+vec3(1,13,4),3);
        float towerBody=tower*(0.12+0.82*towerMacro+0.42*towerBillow);
        float towerBoundary=smoothstep(0.05,0.48,towerBody);
        float towerErode=(uCbRidge*towerRidge+uCbMicro*towerMicro)
                         *(1.0-towerBoundary*0.58);
        d += max(towerBody-0.38-towerErode,0.0)*2.55;
    }

    // Lower-level feeder clouds: broad, asymmetric thermals merge into the
    // rain-free base and fill the otherwise empty middle of the composition.
    // They vanish below 3.4 km, so the established developing crown is not
    // changed. Noise controls their support as well as their surface.
    float feederL=smoothstep(1.0,0.38,length(vec3(
        (q-vec2(-0.82,0.12))/vec2(1.25,0.90),(y-1.75)/1.15)));
    float feederR=smoothstep(1.0,0.38,length(vec3(
        (q-vec2(0.92,-0.18))/vec2(1.05,0.82),(y-2.20)/1.30)));
    float feederEnv=max(feederL,feederR)*hg(y,0.72,3.55,0.22,0.35);
    if(feederEnv>0.001){
        vec3 towerWarp=cloudWarp(p,0.20,uCbWarp);
        vec3 detailWarp=cloudWarp(towerWarp,0.62,uCbDetailWarp);
        float feederNoise=0.58*billowFbm(detailWarp*1.15+vec3(23,5,2),3)
                         +0.42*fbm(towerWarp*0.62+vec3(3,19,7),2);
        float feederCut=0.08*ridgedFbm(detailWarp*3.7+vec3(4,1,15),2);
        d += max(feederEnv*(0.12+0.98*feederNoise)-0.34-feederCut,
                 0.0)*1.85;
    }

    // Mature anvil: wind-sheared, elliptical, vertically rounded and broken by
    // low-frequency billows. The upwind side is compact while the downwind
    // side fans out; there is no rotationally symmetric flat disk.
    vec2 anvilCenter=vec2(uCbX,uCbZ)+vec2(0.75+0.42*(y-6.4),0.08);
    vec2 aq=p.xz-anvilCenter;
    float downwindRadius=(aq.x>0.0)?uCbAnvil:1.55;
    float anvilMetric=length(vec2(aq.x/downwindRadius,aq.y/1.75));
    float anvilVertical=hg(y,6.62,7.55,0.20,0.28);
    if(anvilVertical>0.001 && anvilMetric<1.32){
        vec3 anvilWarp=cloudWarp(p*vec3(0.55,1.0,0.82),0.26,0.85);
        vec3 anvilFine=cloudWarp(anvilWarp,0.75,0.20);
        float anvilEdge=fbm(vec3(anvilWarp.xz*0.48,73.0),4)-0.5;
        float anvilPlan=smoothstep(1.00+0.30*anvilEdge,
                                   0.70+0.14*anvilEdge,anvilMetric);
        float anvilTexture=billowFbm(anvilWarp*vec3(0.74,1.7,0.58)
                                     +vec3(9,0,3),5);
        float anvilWisps=ridgedFbm(anvilFine*vec3(2.2,0.65,1.4)
                                   +vec3(2,6,11),4);
        float anvilMicro=billowFbm(anvilFine*vec3(6.0,1.3,3.8)
                                   +vec3(17,4,8),3);
        d += max(anvilPlan*anvilVertical
                 *(0.62*anvilTexture+0.26*anvilWisps+0.12*anvilMicro)
                 -0.49,0.0)*uCbAnvilDens;
    }

    // A few rising turrets remain visible above the spreading shoulder, which
    // prevents the crown from becoming a single smooth horizontal plate.
    float crown1=smoothstep(1.05,0.34,length(q-vec2(-0.42,0.05)))
                 *hg(y,6.0,8.05,0.45,0.32);
    float crown2=smoothstep(0.92,0.30,length(q-vec2(0.72,-0.28)))
                 *hg(y,5.85,7.85,0.48,0.36);
    float crown3=smoothstep(0.76,0.24,length(q-vec2(0.08,0.78)))
                 *hg(y,5.5,7.55,0.45,0.38);
    float crown=max(crown1,max(crown2,crown3));
    if(crown>0.001){
        vec3 towerWarp=cloudWarp(p,0.20,uCbWarp);
        vec3 detailWarp=cloudWarp(towerWarp,0.62,uCbDetailWarp);
        float crownBillow=billowFbm(detailWarp*1.45+vec3(4,8,1),5);
        float crownErode=ridgedFbm(detailWarp*4.8+vec3(12,3,6),3);
        d += max(crown*(0.16+0.92*crownBillow)-0.31-0.08*crownErode,
                 0.0)*uCbCrown;
    }
    gBgD = bg; gCbD = d;                                         // publish the split for per-deck view optics
    return clamp((bg+d)*DENS, 0.0, 12.0);
}

// Lighting shadows need the same macro extinction support, but not every fine
// erosion layer. This follows the Paper-6 multilayer optimization: preserve the
// large density and deck identity while avoiding full beauty-density work at
// every sun sample.
float shadowDensity(vec3 p){
    float y=p.y,d=0.0;
#if LAYERS==1
    d += lowCumulusShadow(y, p);                                 // Paper-6 multilayer low deck (cheap shadow)
    float mid=hg(y,3.1,4.2,0.28,0.35)
              *smoothstep(0.52,0.72,
                          billowFbm(vec3(p.xz*vec2(0.58,1.0)*1.85,8.0),2));
    d+=mid*0.42;
    float high=hg(y,5.6,6.7,0.22,0.3)
               *smoothstep(0.60,0.78,
                           ridgedFbm(vec3(p.xz*vec2(0.12,1.0)*1.25,14.0),2));
    d+=high*0.16;
#endif
    vec2 c=vec2(uCbX,uCbZ)+vec2(uCbLean*(y-1.0),0.10*sin(y*1.7));
    float tower=hg(y,0.8,8.05,0.38,0.40)
                *smoothstep(mix(1.05,1.55,smoothstep(1.0,6.5,y)),
                            mix(0.45,0.75,smoothstep(1.0,6.5,y)),
                            length(p.xz-c));
    float coarse=fbm(p*0.48+vec3(0,1,0),3);
    d+=max(tower*(0.20+0.95*coarse)-0.32,0.0)*2.2;
    vec2 aq=p.xz-(vec2(uCbX,uCbZ)+vec2(0.9,0.1));
    float ar=(aq.x>0.0)?4.5:1.6;
    float anvil=hg(y,6.62,7.55,0.20,0.28)
                *smoothstep(1.0,0.70,length(vec2(aq.x/ar,aq.y/1.75)));
    d+=max(anvil*(0.35+0.75*fbm(p*vec3(0.40,0.9,0.55),3))-0.42,0.0);
    return clamp(d*DENS,0.0,12.0);
}

float phaseHG(float c){ float g2=HG_G*HG_G; return (1.0-g2)/(4.0*PI*pow(1.0+g2-2.0*HG_G*c,1.5)); }
float phaseHGG(float c,float g){
    float g2=g*g;
    return (1.0-g2)/(4.0*PI*pow(1.0+g2-2.0*g*c,1.5));
}
// cheap multi-scatter approx (Wrenninge octave beer-powder): decreasing extinction per octave
// lets light "leak" deeper -> bright cloud cores instead of black silhouettes.
vec3 lightCloud(float sunT, float mu, float dtau, float y){
    // Dual-lobe water-cloud phase: strong forward scattering plus a softer
    // back lobe. This is the useful physically-inspired part of the local
    // procedural-clouds skill; it replaces the single symmetric-looking lobe.
    float ph=0.72*phaseHGG(mu,0.62)+0.28*phaseHGG(mu,-0.22);
    vec3 acc = vec3(0.0); float a=1.0, b=1.0;
    for(int n=0;n<3;n++){ acc += a*ph*exp(-b*sunT)*SUNCOL; a*=0.5; b*=0.35; }
    float powder = 1.0 - exp(-dtau*2.5);
    acc *= mix(0.55, 1.0, powder);
    vec3 amb = mix(SKYHOR, SKYTOP, clamp(y/YMAX,0.0,1.0))*0.85 + vec3(0.12,0.13,0.15);
    // Convective clouds need a visibly darker base. Apply the vertical bias to
    // both direct and ambient terms, but keep enough blue fill to avoid black.
    float baseLight=mix(0.48,1.0,smoothstep(0.65,5.8,y));
    return ALBEDO*(acc*baseLight + amb*mix(0.62,1.0,baseLight));
}
vec3 sky(vec3 rd){ float h=clamp(rd.y*0.5+0.5,0.0,1.0); vec3 c=mix(SKYHOR,SKYTOP,h);
    float s=max(dot(rd,SUN),0.0); c+=SUNCOL*0.10*pow(s,300.0)+SUNCOL*0.03*pow(s,8.0); return c; }
vec3 tonemap(vec3 c){ c*=0.5; c=(c*(2.51*c+0.03))/(c*(2.43*c+0.59)+0.14); return pow(clamp(c,0.0,1.0),vec3(1.0/2.2)); }

// altitude-slab entry/exit
bool slabHit(vec3 ro,vec3 rd,out float t0,out float t1){
    float a=(YMIN-ro.y)/rd.y, b=(YMAX-ro.y)/rd.y; t0=min(a,b); t1=max(a,b);
    t0=max(t0,0.0); t1=min(t1,60.0); return t1>t0;
}

// ---- cheap sun optical depth via a few long steps + presence gating ----
float sunTau(vec3 x){
    gSunRays+=1.0;
    float t0,t1; if(!slabHit(x,SUN,t0,t1)) return 0.0; t1=min(t1-t0,10.0);
    float tau=0.0, t=0.02, st=t1/float(SUN_STEPS);
    for(int i=0;i<SUN_STEPS;i++){
        vec3 p=x+SUN*t;
        if(presence(p)>0.02) tau += shadowDensity(p)*SIGMA_T*st;
        if(tau>7.0) break; t += st*(1.0+0.25*float(i));   // stretch sun steps with distance
    }
    return tau;
}

// ============================ AAA raymarch ============================
// Fixed WORLD-space steps (ray-angle independent): big steps skip clear air (capped below the
// thinnest layer so we never jump a deck), fine steps integrate cloud. Jitter only the fine step
// (small offset) to break residual banding without under-sampling. Transmittance early-out.
vec3 renderMarch(vec3 ro,vec3 rd,out float alpha){
    alpha=0.0; float t0,t1; if(!slabHit(ro,rd,t0,t1)) return vec3(0.0);
    float small=SMALL_STEP, big=BIG_STEP;
    float t=t0 + gJit*small; float T=1.0; vec3 L=vec3(0.0);
    for(int i=0;i<MAXI;i++){
        if(t>t1 || T<0.01) break;
        vec3 p=ro+rd*t; gSegs+=1.0;
        if(presence(p) <= 0.02){ gSkipped+=1.0; t += big; continue; }   // empty-space skip
        float dens=cloudDensity(p);                            // fills gBgD (thin bg) + gCbD (dense cumulonimbus)
        if(dens>1e-3){
            float aer=exp(-t*BG_FADE);                          // aerial perspective fades distant bg into sky
            float dtau=(gBgD*SIG_BG*aer + gCbD*SIG_CB)*DENS*small;
            float ltau;
#if CLOUDWORKS_SHADOW_BUDGET==1
            // CloudWorks stops shadow marching once accumulated camera density
            // is high enough that direct-light detail is no longer visible.
            ltau=(T>CLOUDWORKS_SHADOW_T)?sunTau(p):8.0;
#else
            ltau=sunTau(p);
#endif
            vec3 S = lightCloud(ltau, dot(rd,SUN), dtau, p.y);
            L += T*S*(1.0-exp(-dtau)); T *= exp(-dtau);
        }
        t += small;
    }
    alpha=1.0-T; return L;
}

// ============================ A3 ============================
float glTau(vec3 ro,vec3 rd,float ta,float tb,int order){
    float h=0.5*(tb-ta),m=0.5*(tb+ta),s=0.0;
    if(order==2){ for(int k=0;k<2;k++) s+=cloudDensity(ro+rd*(m+h*GL2X[k])); }
    else if(order==3){ for(int k=0;k<3;k++) s+=GL3W[k]*cloudDensity(ro+rd*(m+h*GL3X[k])); }
    else { for(int k=0;k<5;k++) s+=GL5W[k]*cloudDensity(ro+rd*(m+h*GL5X[k])); }
    return SIGMA_T*h*s;
}
bool occ(vec3 p){ return presence(p) > 0.02; }
vec3 renderA3(vec3 ro,vec3 rd,out float alpha){
    alpha=0.0; float t0,t1; if(!slabHit(ro,rd,t0,t1)) return vec3(0.0);
    float cs=1.0/float(A3_CAMF); vec3 inv=1.0/(rd+(1.0-step(vec3(1e-6),abs(rd)))*1e-6);
    float Tc=1.0; vec3 L=vec3(0.0); float ph=phaseHG(dot(rd,SUN)); float t=t0;
    for(int i=0;i<MAXI;i++){
        vec3 p=ro+rd*t; vec3 nb=(floor(p/cs)+max(sign(rd),0.0))*cs; vec3 tn=(nb-p)*inv;
        float tnext=t+min(tn.x,min(tn.y,tn.z))+1e-5; float tb=min(tnext,t1);
        if(tb>t){ gSegs+=1.0; vec3 c=ro+rd*(0.5*(t+tb));
            if(occ(c)){
                float dtau=glTau(ro,rd,t,tb,A3_CAMGL);
                if(dtau>1e-4){
                    float jf=0.5+SUN_JITTER*(gJit-0.5);
                    vec3 xc=ro+rd*mix(t,tb,jf);
                    vec3 S = lightCloud(sunTau(xc), dot(rd,SUN), dtau, xc.y);
                    float sc=1.0-exp(-dtau);
                    L += Tc*S*sc; Tc *= exp(-dtau);
                }
            } else gSkipped+=1.0;
        }
        if(tnext>=t1 || Tc<0.01) break; t=tnext;
    }
    alpha=1.0-Tc; return L;
}

// ============================ A3 HIERARCHICAL (two-level) ============================
// Coarse presence-march (BIG steps) blasts through empty air like the AAA baseline; when a
// coarse sample hits cloud we rewind one BIG step (to catch the near edge) and run the FINE
// DDA + exact per-cell GL until presence drops, then pop back to the coarse march. A3's exact
// integration is spent only inside clouds; empty air is skipped as cheaply as the march.
vec3 renderA3H(vec3 ro,vec3 rd,out float alpha){
    alpha=0.0; float t0,t1; if(!slabHit(ro,rd,t0,t1)) return vec3(0.0);
    float fine=max(SMALL_STEP*3.0,0.15);
    if(ro.y>5.0) fine*=1.35;
    float Tc=1.0; vec3 L=vec3(0.0);
    float t=t0 + gJit*BIG_STEP;
    bool inActive=false;
    bool previousDense=false;
    int refinedEntries=0;
    float inactiveGap=0.0;
    for(int i=0;i<MAXI;i++){
        if(t>t1 || Tc<0.01) break;
        vec3 p=ro+rd*t;
        float support=presenceH1(p,rd);
#if USE_MACRO_TEX==1
        float emptyCut=uMacroMargin+0.001;
#else
        float emptyCut=0.02;
#endif
        if(support<=emptyCut){
            gSkipped+=1.0;
            if(inActive){
                inactiveGap+=BIG_STEP;
                if(inactiveGap>4.00) inActive=false;
            }
            t+=BIG_STEP; continue;
        }
        if(!inActive){ gActiveIntervals+=1.0; inActive=true; }
        inactiveGap=0.0;
        gSegs+=1.0;
        float localStep=fine;
        float tb=min(t+localStep,t1);
        float ds=tb-t;
        gActiveLength+=ds;
        vec3 xc=ro+rd*(t+(0.35+0.30*gJit)*ds);
        float dens=cloudDensity(xc);
#if H3_BOUNDARY_REFINE==1
        bool dense=dens>1e-3;
        bool refineBoundary=dense && !previousDense;
        if(refineBoundary){
            float centerBg=gBgD,centerCb=gCbD;
            float boundaryTau=0.0;
            for(int j=0;j<3;j++){
                vec3 xj=ro+rd*(t+(float(j)+0.5)*ds/3.0);
                float dj;
                if(j==1){ dj=dens; gBgD=centerBg; gCbD=centerCb; }
                else dj=cloudDensity(xj);
                if(dj>1e-3){
                    float sj=ds/3.0;
                    float tj=t+(float(j)+0.5)*sj;
                    float aer=exp(-tj*BG_FADE);
                    boundaryTau+=(gBgD*SIG_BG*aer+gCbD*SIG_CB)*DENS*sj;
                }
            }
            if(boundaryTau>1e-4){
                vec3 S=lightCloud(sunTau(xc),dot(rd,SUN),boundaryTau,xc.y);
                L+=Tc*S*(1.0-exp(-boundaryTau));
                Tc*=exp(-boundaryTau);
            }
        } else
#elif H3_BOUNDARY_REFINE==2
        bool dense=dens>1e-3;
        bool refineBoundary=dense && !previousDense && refinedEntries<2;
        if(refineBoundary){
            refinedEntries++;
            float bestPresence=-1.0,occupied=0.0;
            vec3 bestPoint=xc;
            for(int j=0;j<2;j++){
                vec3 xj=ro+rd*(t+(float(j)+0.5)*ds/2.0);
                float pj=presence(xj);
                if(pj>0.02) occupied+=0.5;
                if(pj>bestPresence){
                    bestPresence=pj; bestPoint=xj;
                }
            }
            float bd;
            if(dense){ bd=dens; bestPoint=xc; }
            else bd=cloudDensity(bestPoint);
            if(bd>1e-3 && occupied>0.0){
                float aer=exp(-(t+0.5*ds)*BG_FADE);
                float dtau=(gBgD*SIG_BG*aer+gCbD*SIG_CB)*DENS*ds*occupied;
                vec3 S=lightCloud(sunTau(bestPoint),dot(rd,SUN),dtau,bestPoint.y);
                L+=Tc*S*(1.0-exp(-dtau));
                Tc*=exp(-dtau);
            }
        } else
#endif
        if(dens>1e-3){
            float aer=exp(-(t+0.5*ds)*BG_FADE);
            float dtau=(gBgD*SIG_BG*aer+gCbD*SIG_CB)*DENS*ds;
            vec3 S=lightCloud(sunTau(xc),dot(rd,SUN),dtau,xc.y);
            L+=Tc*S*(1.0-exp(-dtau));
            Tc*=exp(-dtau);
        }
#if H3_BOUNDARY_REFINE>=1
        previousDense=dense;
#endif
        t=tb+1e-5;
    }
    alpha=1.0-Tc; return L;
}

// H2 transport-only validation. This deliberately uses one uniform synthetic
// extinction for raw cloudDensity so deck lighting/aerial terms cannot hide
// closure error. Boundary and out-of-grid segments use the fine reference.
float h2FineTau(vec3 ro,vec3 rd,float ta,float tb){
    float tau=0.0;
    for(int j=0;j<64;j++){
        float t=ta+(float(j)+0.5)*SMALL_STEP;
        if(t>=tb) break;
        tau+=cloudDensity(ro+rd*t)*SIGMA_T*DENS*min(SMALL_STEP,tb-t);
    }
    return tau;
}
bool h2Cell(vec3 p,out ivec3 c,out vec4 s,out vec4 k){
    vec3 q=(p-uMacroOrigin)/uMacroCellSize;
    ivec3 xyz=ivec3(floor(q));
    if(any(lessThan(xyz,ivec3(0))) ||
       any(greaterThanEqual(xyz,ivec3(uMacroDims)))) return false;
    ivec3 tc=ivec3(xyz.x,xyz.z,xyz.y);
    c=xyz; s=texelFetch(uMacroBounds,tc,0);
    k=texelFetch(uMacroCumulants,tc,0);
    return true;
}
float renderH2T(vec3 ro,vec3 rd){
    float t0,t1; if(!slabHit(ro,rd,t0,t1)) return 1.0;
    float tau=0.0,t=t0;
    for(int i=0;i<MAXI;i++){
        if(t>=t1 || tau>12.0) break;
        vec3 p=ro+rd*(t+1e-4); ivec3 c; vec4 s,k;
        if(!h2Cell(p,c,s,k)){
            float tb=min(t+BIG_STEP,t1);
            tau+=h2FineTau(ro,rd,t,tb); t=tb+1e-5; continue;
        }
        vec3 edge=uMacroOrigin+(vec3(c)+max(sign(rd),vec3(0.0)))*uMacroCellSize;
        vec3 hit=(edge-p)/(rd+(1.0-step(vec3(1e-6),abs(rd)))*1e-6);
        float ds=max(min(hit.x,min(hit.y,hit.z)),1e-4);
        float tb=min(t+ds,t1);
        ds=tb-t;
        gSegs+=ds;
        float fill=s.y/max(s.x,1e-6);
        float nv=s.z/max(s.y*s.y,1e-6);
        float scale=SIGMA_T*DENS*ds;
        float risk=0.5*scale*scale*s.z;
        bool analytic=s.x>0.02 && fill>=0.25 && nv<=2.0 && risk<=0.1;
        if(analytic){ gActiveLength+=ds; gActiveIntervals+=1.0; }
#if H2_MODE==0
        tau+=h2FineTau(ro,rd,t,tb);
#else
        if(!analytic) tau+=h2FineTau(ro,rd,t,tb);
        else {
            float logT=-scale*s.y;
#if H2_MODE>=2
            logT+=0.5*scale*scale*s.z;
#endif
#if H2_MODE>=3
            logT-=scale*scale*scale*k.z/6.0;
#endif
#if H2_MODE>=4
            logT+=scale*scale*scale*scale*k.w/24.0;
#endif
            tau+=max(-logT,0.0);
        }
#endif
        t=tb+1e-5;
    }
    return exp(-tau);
}

void main(){
#if BUILD_BOUNDS==1
    vec3 cell=uMacroOrigin+vec3(
        gl_FragCoord.x-0.5,uBoundLayer,gl_FragCoord.y-0.5
    )*uMacroCellSize;
    float upper=0.0,sum=0.0,sum2=0.0,sum3=0.0,sum4=0.0,sumExp=0.0;
    for(int iz=0;iz<4;iz++) for(int iy=0;iy<4;iy++)
    for(int ix=0;ix<4;ix++){
        vec3 f=vec3(float(ix),float(iy),float(iz))/3.0;
        float d=cloudDensity(cell+f*uMacroCellSize);
        upper=max(upper,d);
        float d2=d*d;
        sum+=d; sum2+=d2; sum3+=d2*d; sum4+=d2*d2;
        sumExp+=exp(-uClosureK*d);
    }
    float mean=sum/64.0;
    float raw2=sum2/64.0;
    float variance=max(raw2-mean*mean,0.0);
#if BUILD_CUMULANTS==1
    float raw3=sum3/64.0;
    float raw4=sum4/64.0;
    float kappa3=raw3-3.0*mean*raw2+2.0*mean*mean*mean;
    float central4=raw4-4.0*mean*raw3+6.0*mean*mean*raw2
                   -3.0*mean*mean*mean*mean;
    float kappa4=central4-3.0*variance*variance;
    fragColor=vec4(mean,variance,kappa3,kappa4);
#else
    fragColor=vec4(upper,mean,variance,sumExp/64.0);
#endif
#else
    vec2 uv=(gl_FragCoord.xy-0.5*uRes)/uRes.y;
    gJit = float(JITTER_ON) * hash13(vec3(gl_FragCoord.xy,0.5));
    vec3 rd=uCam*normalize(vec3(uv,uFocal)); vec3 ro=uEye;
#if INSTRUMENT==4
    float h2t=renderH2T(ro,rd);
    fragColor=vec4(h2t,gActiveLength,gSegs,gActiveIntervals);
#else
    float alpha; vec3 cloud;
#if METHOD==0
    cloud=renderA3(ro,rd,alpha);
#elif METHOD==2
    cloud=renderA3H(ro,rd,alpha);
#else
    cloud=renderMarch(ro,rd,alpha);
#endif
#if INSTRUMENT==1
    fragColor=vec4(gFullEvals,gActiveIntervals,gActiveLength,gEvals);
#elif INSTRUMENT==2
    fragColor=vec4(vec3(alpha),1.0);
#elif INSTRUMENT==3
    fragColor=vec4(vec3(gActiveIntervals>0.0?1.0:0.0),1.0);
#else
    fragColor=vec4(tonemap(sky(rd)*(1.0-alpha)+cloud),1.0);
#endif
#endif
#endif
}
