#version 330 core
// Multi-layer cumulus sky for the MULTI-PHOTO VSGA -- CLOUDWORKS-DERIVED REWRITE.
//
// The density model and the marcher are rebuilt from the hand-tuned CloudWorks shader
// (articles/0_cloudworks/shader.glsl, RTU 2021), which is both better looking and far cheaper than the
// octave-stacked model this replaces. The five ideas taken from it:
//   1. PREDICTIVE ADAPTIVE STEP. The step length is a continuous function of a linear extrapolation of
//      the density field (2*df - df_prev), so the marcher slows down BEFORE it reaches cloud and runs
//      long steps through clear air -- instead of a binary coarse/fine skip with hysteresis. The step
//      also grows with distance (LOD), so far cloud costs a fraction of near cloud.
//   2. DENSITY-DRIVEN DISTORTION. The detail coordinate is displaced by the MACRO density itself
//      (lateral cos term + a downward bump). Detail therefore bulges out of the macro lump instead of
//      sitting on top of it -- this is what makes cauliflower billows follow the cloud's own shape.
//      DetailB nests a second displacement driven by the first detail octave.
//   3. MULTIPLICATIVE construction everywhere. The macro field is noiseA*(noiseB*B+A) (2 lookups, not a
//      5-octave fbm), and detail MODULATES the macro (df = lump*(1+detail)) rather than being subtracted
//      from it, so detail adds texture without eroding the silhouette into confetti.
//      Net effect: ~5 noise lookups per full density sample instead of ~24.
//   4. HEIGHT-VARYING THRESHOLD BAND. The density cut is a band (thr +/- soft) whose width shrinks with
//      height, so tops taper naturally and bases stay flat.
//   5. HEIGHT-VARYING SOLIDNESS. Density scales from ~0 at the deck base to full at the top: flat
//      translucent cumulus base, dense bright crown.
//
// KEPT from the previous Paper-6 shader (these were hard-won and are better than the CloudWorks
// equivalents, which use a hand-tuned gradient sky and a single-scatter cloud):
//   - the Paper-3 atmosphere() (Chapman/erfcx, Rayleigh + ozone Chappuis + grey aerosol) as the sky AND
//     as the source of cloud sun colour, so each photo's time of day is physical, not hand-keyed;
//   - the 1x1 ambient pre-pass hoist (uAmbMode/uProbe) -- the probes are view-independent;
//   - the view-INDEPENDENT multiple-scatter lit-face term (uDiffuse), the multi-tap MS approximation,
//     powder, and the zenith-biased ambient mix.
//
// The uniform names are unchanged, so optimize_multiphoto.py / perf_quality.py drive this file as-is.
out vec4 fragColor;
uniform vec2 uRes;
uniform vec3 uEye;
uniform mat3 uCam;
uniform float uFocal;
uniform vec3 uSunDir;
uniform float uExposure;
// shared atmosphere
uniform float uAtmDens, uSunI, uHaze, uAmbStr;
// shared cloud scattering / shape
uniform float uDens, uSigma, uSilver, uMsBoost, uDetAmp, uDet2Amp, uDet3Amp, uCovFreq;
// Octave ladder multipliers, GA-tuned but each CONFINED TO ITS OWN OCTAVE with adjacent, non-overlapping
// bounds (c/sqrt2 .. c*sqrt2 around 6.25 / 12.5 / 25 / 50). The bands therefore stay distinct scales and
// cannot collapse onto each other, while the GA still places each one within its octave.
uniform float uFreqB, uFreqD1, uFreqD2, uFreqD3;
uniform float uWarp;                             // density-driven distortion strength (billow character)
uniform float uDiffuse;                          // view-INDEPENDENT multiple-scatter lit-face strength
uniform float uShear;                            // wind shear: horizontal offset per unit height
// shared layer altitudes (require uLowTop < uMidBase < uMidTop < uHiBase)
uniform float uLowBase, uLowTop, uMidBase, uMidTop, uHiBase, uHiTop;
// per-photo coverage + cloud-type distribution
uniform float uCovLow, uCovMid, uCovHigh, uTypeAmp, uTypeSeed;
// The two ambient probes are VIEW-INDEPENDENT (constant per frame) but were recomputed per pixel.
// uAmbMode>0.5 => host supplies them (filled by a 1x1 pre-pass driven by uProbe).
uniform vec3 uAmbZen, uAmbHor; uniform float uAmbMode, uProbe;

#define PI 3.14159265358979
#define SUN_STEPS 5
#define SUNCOL (vec3(1.9,1.75,1.50)*3.4)
#define T_EXIT 0.04              // early-out transmittance (CloudWorks stops well before true opacity)

// ================= noise =================
float hash13(vec3 p){ p=fract(p*vec3(0.1031,0.1030,0.0973)); p+=dot(p,p.yzx+33.33); return fract((p.x+p.y)*p.z); }
float vnoise(vec3 p){
    vec3 i=floor(p), f=p-i; vec3 u=f*f*(3.0-2.0*f);
    float n000=hash13(i+vec3(0,0,0)),n100=hash13(i+vec3(1,0,0)),n010=hash13(i+vec3(0,1,0)),n110=hash13(i+vec3(1,1,0));
    float n001=hash13(i+vec3(0,0,1)),n101=hash13(i+vec3(1,0,1)),n011=hash13(i+vec3(0,1,1)),n111=hash13(i+vec3(1,1,1));
    float x00=mix(n000,n100,u.x),x10=mix(n010,n110,u.x),x01=mix(n001,n101,u.x),x11=mix(n011,n111,u.x);
    return mix(mix(x00,x10,u.y),mix(x01,x11,u.y),u.z);
}
float fbm(vec3 p,int oct){ float s=0.0,a=0.5,f=1.0; for(int o=0;o<oct;o++){ s+=a*vnoise(p*f); a*=0.5; f*=2.02; } return s; }
// CloudWorks map helpers: `a` maps to `c`, `b` maps to `d`.
float cmap (float x,float a,float b,float c,float d){ return (x-b)/(a-b)*(c-d)+d; }
float ccmap(float x,float a,float b,float c,float d){ return clamp((x-b)/(a-b),0.0,1.0)*(c-d)+d; }

// ================= Paper-3 atmosphere (+ grey aerosol) =================
const float planetRadius=6.371, atmRadius=6.471, Hatm=0.008, Csw=1.12;
const float Haer=0.0016;
const float CLOUD_KM=0.001;   // 1 local cloud unit == 1 km of real altitude
const vec3 betaR=vec3(5.802,13.558,33.10), betaO=vec3(3.426,8.298,0.356);
const float ozC=0.025, ozW=0.030, ozA=0.060, fDay=0.55, MS_STR=0.40;  // ozA was 0.30 = 50x too much ozone
                                                                       // -> Chappuis absorption turned low sun purple/crimson
float erfcx(float y){ if(y<1.5){ float y2=y*y,t=y,s=y; for(int n=1;n<10;n++){t*=-y2/float(n); s+=t/float(2*n+1);} return exp(y2)*(1.0-1.12837916709551*s);} float f=y; for(int k=12;k>=1;k--) f=y+0.5*float(k)/f; return 0.564189583547756/f; }
float chapPos(float r0,float mu){ float s=sqrt(max(0.0,1.0-mu*mu)); float X=r0/Hatm,h0=r0-planetRadius,eps=1.0/X; if(s<1e-4) return Hatm*exp(-h0/Hatm);
    if(1.0/s>Csw){ float m2=mu*mu,c1=1.0-1.0/m2,c2=3.0*(1.0-m2)/(m2*m2),c3=3.0*(m2*m2+4.0*m2-5.0)/(m2*m2*m2); return Hatm*exp(-h0/Hatm)*(1.0+eps*(c1+eps*(c2+eps*c3)))/mu; }
    float z=X*s,p=r0*s,eta=sqrt(2.0*X*mu*mu/(1.0+s)),iz=1.0/z;
    float P=1.0+iz*(0.375+iz*(-0.1171875+iz*(0.1025390625+iz*(-0.144195556640625)))); float e2=eta*eta;
    float Q=iz*(0.375*eta+iz*(-0.0390625*(e2*eta+3.0*eta)+iz*(0.0068359375*(e2*e2*eta+5.0*e2*eta+15.0*eta)+iz*(-0.001373291015625*(e2*e2*e2*eta+7.0*e2*e2*eta+35.0*e2*eta+105.0*eta)))));
    return sqrt(p*Hatm)*exp(-h0/Hatm)*(P*1.25331413731550*erfcx(eta*0.707106781186548)+Q); }
float chapInf(float r0,float mu){ if(mu<0.0){ float p=r0*sqrt(max(0.0,1.0-mu*mu)); if(p<planetRadius) return 1e20; return 2.0*chapPos(p,0.0)-chapPos(r0,-mu);} return chapPos(r0,mu); }
vec2 raySphere(vec3 ro,vec3 rd,float ra){ float b=dot(ro,rd),c=dot(ro,ro)-ra*ra,d=b*b-c; if(d<0.0) return vec2(1e9,-1e9); d=sqrt(d); return vec2(-b-d,-b+d); }
float ozone(float h){ return max(0.0,1.0-abs(h-ozC)/ozW)*ozA; }
float hgHenyey(float c,float g){ float g2=g*g; return (1.0-g2)/(4.0*PI*pow(1.0+g2-2.0*g*c,1.5)); }
vec3 sunTrans(vec3 pos,float r,vec3 L){ float sunOD=chapInf(r,dot(L,pos/r)); float am=sunOD/max(Hatm*exp(-(r-planetRadius)/Hatm),1e-4); return exp(-(betaR*sunOD+betaO*(ozA*ozW)*am)*uAtmDens); }
vec3 gather(vec3 P,vec3 up,vec3 L){ vec3 sp=L-dot(L,up)*up; vec3 t1=(length(sp)>1e-4)?normalize(sp):normalize(cross(up,vec3(0,0,1))); vec3 t2=cross(up,t1); vec3 J=vec3(0.0);
    for(int ie=0;ie<2;ie++){ float e=0.35+float(ie)*0.65,ce=cos(e),se=sin(e); for(int ia=0;ia<1;ia++){ float a=float(ia)*PI; vec3 d=ce*(cos(a)*t1+sin(a)*t2)+se*up; vec2 hit=raySphere(P,d,atmRadius); float Ld=max(0.0,hit.y)*0.4; vec3 Q=P+d*Ld; float rq=length(Q),hq=rq-planetRadius; if(hq<0.0) continue; float dq=exp(-hq/Hatm); float cph=dot(d,L); J+=betaR*dq*sunTrans(Q,rq,L)*0.75*(1.0+cph*cph)*(Ld+1e-4);} } return J*0.5; }   // half the azimuth samples, compensated
vec3 atmosphere(vec3 cam,vec3 ray,vec3 L,float sunDisk){ vec2 atm=raySphere(cam,ray,atmRadius); if(atm.y<0.0||atm.x>atm.y) return vec3(0.0);
    float tS=max(atm.x,0.0),tE=atm.y; vec2 pl=raySphere(cam,ray,planetRadius); if(pl.x>tS&&pl.x<tE) tE=pl.x;
    float msG=1.0/(1.0-fDay); float c=dot(ray,L),phR=0.75*(1.0+c*c),phM=hgHenyey(c,0.76);
    vec3 betaM=vec3(21.0)*uHaze;                                  // grey aerosol extinction
    float ds=(tE-tS)/6.0,vOD=0.0,vOz=0.0,vAe=0.0; vec3 single=vec3(0.0),multi=vec3(0.0),singleM=vec3(0.0);
    for(int i=0;i<6;i++){ vec3 pos=cam+ray*(tS+(float(i)+0.5)*ds); float r=length(pos),h=r-planetRadius,dens=exp(-h/Hatm),ozo=ozone(h),dae=exp(-h/Haer); vOD+=dens*ds; vOz+=ozo*ds; vAe+=dae*ds;
        vec3 Tv=exp(-(betaR*vOD+betaO*vOz+betaM*vAe)*uAtmDens); vec3 up=pos/r; vec3 st=sunTrans(pos,r,L);
        single+=dens*Tv*st*ds; multi+=dens*Tv*gather(pos,up,L)*msG*ds; singleM+=dae*Tv*st*ds; }
    vec3 col=uSunI*(betaR*(single*phR+multi*MS_STR) + betaM*singleM*phM); if(sunDisk>0.5 && c>0.9997) col+=vec3(1.0,0.95,0.9)*45.0; return col; }

// ========== Large-Scale Hashed Spatiotemporal cloud-top events (HSE, from shader_HSE.glsl) ==========
// The low deck used to have a FLAT top at uLowTop, so every cumulus in the field topped out at the same
// height -- the classic "sliced off" procedural look. Instead the deck's TOP SURFACE is a field of large
// compact convective events: one per coarse cell, each with its own lifecycle. Height develops FAST and
// the radius expands LATER, which is how real convection grows. A 3x3 max over cells with compact support
// (bump = (1-r^2)^3) keeps it seam-free, and no trigonometry or texture is involved.
// Even in a STILL frame this pays off: the phase is u = frac(uTime*rate + hash(cell)), so with uTime=0
// every cell sits at a different point in its life -- some just rising, some mature towers, some decaying.
#define HSE_CELL 5.2                       // km, coarse cell size
// Radii must be matched to the RISE or the events read as flat domes rather than towers: 8.5 km wide
// against a 2.4 km rise is 1:3.5. These give roughly 1:1 to 1:1.5, i.e. convective towers. Keeping
// RMAX below HSE_CELL also makes the 3x3 neighbourhood strictly sufficient.
#define HSE_RMIN 1.3
#define HSE_RMAX 4.0
#define HSE_RATE 0.08
uniform float uTime;                       // 0 for stills (the per-cell phase still varies); animates if driven

float hseHash12(vec2 p){ vec3 q=fract(vec3(p.x,p.y,p.x)*0.1031); q+=dot(q,q.yzx+33.33); return fract((q.x+q.y)*q.z); }
vec2  hseHash22(vec2 p){ vec3 q=fract(vec3(p.x,p.y,p.x)*vec3(0.1031,0.1030,0.0973)); q+=dot(q,q.yzx+33.33);
                         return fract(vec2((q.x+q.y)*q.z,(q.x+q.z)*q.y)); }
float hseSmooth(float x){ x=clamp(x,0.0,1.0); return x*x*x*(x*(x*6.0-15.0)+10.0); }
float hseLife(float u){                    // fast rise, long mature plateau, smooth dissipation
    return hseSmooth(u/0.20)*(1.0-hseSmooth((u-0.70)/0.30)); }
float hseRadiusLife(float u){ return mix(0.52,1.0,hseSmooth((u-0.08)/0.55)); }   // width lags height
float hseEventRise(vec2 p, vec2 cell, float maxRise){
    vec2 c=cell+uTypeSeed*3.7;                                   // per-photo layout comes from the seed
    if(hseHash12(c+19.71) < 0.38) return 0.0;                    // most cells hold no event
    vec2 r0=hseHash22(c+7.13), r1=hseHash22(c+23.91), r2=hseHash22(c+51.37);
    vec2 centre=(cell+mix(vec2(0.25),vec2(0.75),r0))*HSE_CELL;
    float u=fract(uTime*HSE_RATE + hseHash12(c+37.11));
    float life=hseLife(u); if(life<=1e-4) return 0.0;
    centre += (r1*2.0-1.0)*(0.65*(u*2.0-1.0));                   // small bounded drift, no trig
    float radius=mix(HSE_RMIN,HSE_RMAX,r1.y)*hseRadiusLife(u);
    float aspect=mix(0.68,1.0,r2.y);
    vec2 radii=(r0.x<0.5)?vec2(radius,radius*aspect):vec2(radius*aspect,radius);   // axis swap without trig
    vec2 q=(p-centre)/max(radii,vec2(1e-3));
    float r2s=dot(q,q); if(r2s>=1.0) return 0.0;                 // compact support -> no cell seams
    float s=1.0-r2s;
    return mix(0.22,1.0,r2.x)*maxRise*life*(s*s*s);
}
float hseTop(vec2 p, float baseTop, float maxRise){
    vec2 base=floor(p/HSE_CELL); float rise=0.0;
    for(int j=-1;j<=1;j++) for(int i=-1;i<=1;i++)
        rise=max(rise, hseEventRise(p, base+vec2(float(i),float(j)), maxRise));
    float broad=(vnoise(vec3(p*0.055,uTime*0.02))-0.5)*0.09;     // stops the untouched roof being dead flat
    return baseTop + max(0.0, rise + broad);
}

// ================= CloudWorks density core =================
struct Deck {
    float base, top;                  // march slab (also the solidness ramp)
    float sBot, sMid;                 // vertical envelope: rises from sBot, peaks at sMid, falls off at top
    float thr;                        // density threshold CENTRE (this is the coverage control)
    float rMid, rTop;                 // threshold half-width at sMid / at top (narrows upward -> tapered tops)
    float solBot, solTop;             // density scale at base / top (flat translucent base, solid crown)
    float edgePow;                    // >1 = harder silhouette via 1-(1-t)^k; 1 = the old linear ramp
    float hseMax;                     // max HSE cloud-top rise (km); 0 disables vertical development
    vec2  dChunk;                     // macro density A, B
    vec3  dDet;                       // detail density C, D, E
    vec3  sChunk;                     // macro scale A, B + vertical stretch
    vec3  sDet;                       // detail scale C, D, E
    vec4  dist;                       // distortion: angle, lateral strength, bump, fine bump
    vec3  shift;                      // wind shear per unit height
    vec3  oA, oB, oC, oD;             // decorrelation offsets (carry the per-photo seed)
    float gain;                       // density gain
    float ice;                        // 0 = water lighting, 1 = thin ice lighting
    float dtMin, dtMax, grow, fade;   // adaptive march: min/max step, distance step growth, cutoff range
    float shStep, shDetail, shExpand, shStr;   // shadow march
};

// vertical envelope + the height-varying threshold band. .x = envelope, .y = high cut, .z = low cut.
// lTop/lMid are the LOCAL top and mid for this column (equal to dk.top/dk.sMid when HSE is off).
vec3 deckShapeL(float y, Deck dk, float lTop, float lMid){
    float shape = smoothstep(dk.sBot, lMid, y) * smoothstep(lTop, lMid, y);
    // Clamped at rMid: extrapolating below sMid would widen the band and flood the deck base with cloud.
    float soft  = clamp(cmap(y, lMid, lTop, dk.rMid, dk.rTop), dk.rTop*0.5, dk.rMid);
    return vec3(shape, dk.thr + soft, dk.thr - soft);
}
vec3 deckShape(float y, Deck dk){ return deckShapeL(y, dk, dk.top, dk.sMid); }
// Local top for this column: a raised HSE event lifts the top, and the soft/mid transition follows at
// about half the rise so a growing tower stays convex instead of turning into a flat-topped column.
void deckLocal(vec3 p, Deck dk, out float lTop, out float lMid){
    lTop=dk.top; lMid=dk.sMid;
    if(dk.hseMax>0.001){
        lTop=hseTop(p.xz, dk.top, dk.hseMax);
        lMid=min(lTop-0.04, dk.sMid + (lTop-dk.top)*0.52);
    }
}
// macro cloud mass: TWO value-noise fields combined MULTIPLICATIVELY. Large coherent masses with clean
// gaps for 2 lookups -- the single biggest saving over a 5-6 octave fbm coverage field.
float chunk(vec3 p, Deck dk, float shape){
    p.y /= dk.sChunk.z;                       // vertical stretch
    p   += dk.shift * p.y;                    // wind shear grows with height -> leaning masses
    float a = vnoise((p + dk.oA)*dk.sChunk.x);
    float b = vnoise((p + dk.oB)*dk.sChunk.y);
    return clamp(a*(b*dk.dChunk.y + dk.dChunk.x)*shape, 0.0, 1.0);
}
// Displacement driven by the MACRO density: lateral cos sweep + a downward push proportional to the
// lump. Sampling lower where the lump is dense makes the detail bulge UPWARD out of the mass, so
// billows follow the cloud's own shape instead of being pasted flat across it.
vec3 distortion(float lump, vec4 ds){ return vec3(cos(lump*ds.x)*ds.y, -lump*ds.z, 0.0); }
float detailA(vec3 p, Deck dk, vec3 disp){ return dk.dDet.x * vnoise((p + dk.oC + disp)*dk.sDet.x); }
// three detail octaves with NESTED displacement; the last is multiplicative so it textures what is
// already there instead of punching new holes.
// `lod` (1 near -> 0 far) fades the fine bands out with distance. The march step GROWS with distance
// (the LOD in the stepping), so without a matching detail LOD the far field samples ~100 m features with
// ~250 m steps and aliases into speckle -- the same salt-and-pepper the old shader had, arriving by a
// different route. Fading them is also cheaper: the last band is skipped entirely once it is invisible.
float detailB(float lump, vec3 p, Deck dk, float lod){
    vec3 d  = distortion(lump, dk.dist);
    vec3 pD = p + dk.oD;
    float nc = detailA(p, dk, d) * mix(0.55,1.0,lod);
    d.y -= nc * dk.dist.w;                    // nested: fine scale displaced by the coarse detail
    float dens = nc + dk.dDet.y*lod * vnoise((pD + d/3.0)*dk.sDet.y);
    if(lod>0.12) dens += dens * dk.dDet.z*lod*lod * vnoise((pD + d*8.0)*dk.sDet.z);
    return dens;
}
// CONCAVE density remap. A linear ramp lets density crawl up from the threshold, and because the macro
// field is ~5 km scale with a gradient of only ~0.11/km, a 0.20-wide band smears the silhouette over
// ~1.8 km -- the clouds read soft no matter how dense the core is. pow(t, edgePow<1) lifts density
// immediately past the threshold, so the boundary is short and the interior stays solid. It is also
// CHEAPER: the higher density near the edge reaches the transmittance early-out sooner.
float densityMix(float df, float y, Deck dk, float lo, float hi, float lTop){
    float sol = ccmap(y, dk.base, lTop, dk.solBot, dk.solTop);
    float t   = clamp((df-lo)/max(hi-lo,1e-4), 0.0, 1.0);
    // 1-(1-t)^k, NOT pow(t,1/k). Both are concave and harden the silhouette, but pow(t,0.3) has an
    // INFINITE derivative at t=0, so a voxel barely over the threshold jumps to ~13% density and the
    // sample-to-sample noise there explodes into speckle in thin cloud. This form has slope k at t=0.
    return sol*(1.0 - pow(1.0-t, dk.edgePow));
}

// ---------------- deck builders ----------------
vec3 seedOff(float k){ float s=uTypeSeed; return vec3(s*13.1+k*7.0, s*3.3, s*9.7-k*5.0); }

Deck lowDeck(){                                    // cumulus: flat base, cauliflower crown
    Deck dk;
    dk.base=uLowBase; dk.top=uLowTop;
    float H=max(dk.top-dk.base,0.05);
    dk.sBot=dk.base-0.85*H; dk.sMid=dk.base+0.38*H;
    // Thresholds are set from the MEASURED distribution of df = lump*(1+detail) for this construction
    // (mean 0.24, sd 0.16, p90 0.46, p99.5 0.67). The low cut (thr - soft) selects the volume fraction
    // that becomes cloud: 0.49 -> ~6% (sparse), 0.39 -> ~18%, 0.31 -> ~33% (near overcast).
    dk.thr = mix(0.67, 0.43, uCovLow) - uTypeAmp*0.05;   // +0.03: trims the dominant cloud masses back
    // Band width and solidness are coupled: df only reaches ~0.67 at p99.5, so a wide band would leave
    // almost every voxel at a fraction of full density (translucent haze). CloudWorks solves this with a
    // solidness well above 1 (theirs is 3.0), so mid-band samples are already optically thick.
    dk.rMid=0.06; dk.rTop=0.03;
    dk.solBot=0.03; dk.solTop=2.6; dk.edgePow=3.6;
    dk.dChunk=vec2(0.30,0.50);
    // Detail carries MORE of the field than the macro mass now: stronger fine texture, and because the
    // threshold crossing becomes more detail-driven, the small-scale structure at the silhouette reads
    // crisper. Paired with a slightly higher thr below so the main masses do not also grow.
    // Boost only the RESOLVABLE bands. Tripling the finest one brought the speckle back: its ~65 m
    // features alias against the 35 m step once the amplitude makes the threshold crossing steep.
    dk.dDet=vec3(uDetAmp*1.30, uDet2Amp*1.9, uDet3Amp*1.5);   // roughly decaying, GA still scales each
    // OCTAVE LADDER. The bands must be successive octaves of the macro field, not a jump straight to
    // high frequency: at 6.25x then 18x there was a 2.6-octave hole between the cloud mass (5.8 km) and
    // the first detail band (325 m), so the detail bore no visual relation to the shape and read as
    // sprinkled noise. Near-uniform ~1.2-octave steps give structure at every scale instead.
    dk.sChunk=vec3(uCovFreq, uCovFreq*uFreqB, 1.0);
    // Successive DOUBLINGS of the second macro band, not a jump past it. Previously the detail sat at
    // 18/34/58 -- 1.5 octaves above the macro with nothing in between -- so it bore no relation to the
    // shape and read as sprinkled noise however hard it was driven.
    dk.sDet=dk.sChunk.x*vec3(uFreqD1,uFreqD2,uFreqD3);
    dk.dist=vec4(3.0, 0.040+0.052*uWarp, 0.060+0.085*uWarp, 0.040+0.040*uWarp);
    dk.shift=vec3(uShear*0.35,0.0,0.0);
    dk.oA=seedOff(0.0); dk.oB=seedOff(1.0); dk.oC=seedOff(2.0); dk.oD=seedOff(3.0);
    dk.gain=uDens*0.055; dk.ice=0.0;
    // Vertical development is a PER-PHOTO property (flat stratocumulus vs towering cumulus), so scale it
    // by uTypeAmp and cap it at the headroom below the mid deck so the decks stay disjoint.
    // Scales from ZERO so a photo with flat stratocumulus disables the field entirely and pays none of
    // its cost. It is deliberately NOT capped by uMidBase: tying it to the gap below the mid deck made
    // the two height mechanisms compete for the same space, and when the GA raised uLowTop to thicken
    // the deck it starved the towers to 0.4 km. Real towering cumulus DO punch through the mid level,
    // so the decks are allowed to overlap here.
    dk.hseMax=clamp(uMidBase-0.15-uLowTop,0.0,1.8)*smoothstep(0.02,0.16,uTypeAmp);
    // Stronger detail needs finer sampling: the step floor is path/maxStep, and HSE lengthens the marched
    // span by up to 2.4 km, which coarsened the floor and aliased the detail into speckle. More steps and
    // a gentler distance growth keep the fine bands resolved.
    dk.dtMin=0.035; dk.dtMax=0.55; dk.grow=7.0; dk.fade=70.0;
    dk.shStep=0.22; dk.shDetail=1.4; dk.shExpand=0.06; dk.shStr=1.0;
    return dk;
}
Deck midDeck(){                                    // altocumulus: thin granular cellular slab
    Deck dk;
    dk.base=uMidBase; dk.top=uMidTop;
    float H=max(dk.top-dk.base,0.05);
    dk.sBot=dk.base-0.35*H; dk.sMid=dk.base+0.45*H;
    dk.thr = mix(0.62, 0.42, uCovMid);
    dk.rMid=0.07; dk.rTop=0.04;
    dk.solBot=0.10; dk.solTop=1.20; dk.edgePow=2.0;
    dk.dChunk=vec2(0.28,0.55);
    dk.dDet=vec3(uDetAmp*0.8, uDet2Amp*1.2, uDet3Amp);
    dk.sChunk=vec3(uCovFreq*1.7, uCovFreq*1.7*uFreqB, 1.6);   // finer + vertically stretched -> mackerel rows
    dk.sDet=dk.sChunk.x*vec3(uFreqD1,uFreqD2,uFreqD3);
    dk.dist=vec4(5.0, 0.018+0.020*uWarp, 0.010, 0.018);
    dk.shift=vec3(uShear*0.20,0.0,0.0);
    dk.oA=seedOff(7.0); dk.oB=seedOff(8.0); dk.oC=seedOff(9.0); dk.oD=seedOff(10.0);
    dk.gain=uDens*0.020; dk.ice=1.0; dk.hseMax=0.0;
    dk.dtMin=0.055; dk.dtMax=0.45; dk.grow=8.0; dk.fade=90.0;
    dk.shStep=0.30; dk.shDetail=0.8; dk.shExpand=0.05; dk.shStr=0.45;
    return dk;
}
Deck hiDeck(){                                     // cirrus: very thin, wind-stretched fibres
    Deck dk;
    dk.base=uHiBase; dk.top=uHiTop;
    float H=max(dk.top-dk.base,0.05);
    dk.sBot=dk.base-0.30*H; dk.sMid=dk.base+0.50*H;
    dk.thr = mix(0.60, 0.42, uCovHigh);
    dk.rMid=0.08; dk.rTop=0.05;
    dk.solBot=0.20; dk.solTop=0.90; dk.edgePow=1.3;
    dk.dChunk=vec2(0.35,0.45);
    dk.dDet=vec3(uDetAmp*0.7, uDet2Amp, uDet3Amp*0.8);
    // strong horizontal anisotropy: fibres are drawn out along x by the shear-dominated chunk scale
    dk.sChunk=vec3(uCovFreq*0.9, uCovFreq*0.9*uFreqB, 2.2);
    dk.sDet=dk.sChunk.x*vec3(uFreqD1,uFreqD2,uFreqD3);
    dk.dist=vec4(2.5, 0.10+0.30*uWarp, 0.004, 0.010);   // large lateral drag = fallstreak filaments
    dk.shift=vec3(uShear*0.9,0.0,0.0);
    dk.oA=seedOff(14.0); dk.oB=seedOff(15.0); dk.oC=seedOff(16.0); dk.oD=seedOff(17.0);
    dk.gain=uDens*0.012; dk.ice=1.0; dk.hseMax=0.0;
    dk.dtMin=0.09; dk.dtMax=0.60; dk.grow=9.0; dk.fade=140.0;
    dk.shStep=0.40; dk.shDetail=0.5; dk.shExpand=0.04; dk.shStr=0.25;
    return dk;
}

// cheap shadow density: macro + ONE detail octave only (no nested displacement, no fine octaves).
// lTop/lMid are REUSED from the camera sample rather than recomputed per shadow step: the sun march
// only walks ~1 km, far less than the 5.2 km HSE cell, so the local cloud top barely changes along it.
// This removes 6 of every 7 hseTop evaluations and was the whole cost of the HSE field.
float shadowMarch(vec3 p, vec3 L, Deck dk, float lTop, float lMid){
    float d=0.0;
    for(int i=0;i<6;i++){
        p += L*dk.shStep;
        if(p.y>dk.top+dk.hseMax || p.y<dk.base) break;
        if(p.y>lTop) continue;
        vec3 cs = deckShapeL(p.y, dk, lTop, lMid);
        if(cs.x<=0.001) continue;
        float d1 = chunk(p, dk, cs.x);
        if(d1<=0.001) continue;
        float d2 = detailA(p, dk, distortion(d1, dk.dist))*dk.shDetail;
        d += densityMix(d1*(1.0+d2), p.y, dk, cs.z-dk.shExpand, cs.y, lTop)*dk.gain;
        if(d>3.0) break;
    }
    return d*dk.shStr*dk.shStep*uSigma;
}

vec3 tonemap(vec3 c){ c=1.0-exp(-c*uExposure); return pow(clamp(c,0.0,1.0),vec3(1.0/2.2)); }

// ================= march =================
void marchDeck(vec3 ro,vec3 rd,vec3 L, Deck dk, int maxStep, float mu,float sunMag,vec3 zenAmb,vec3 horAmb,
               inout float T, inout vec3 acc, inout float meanT, inout float wsum){
    if(T<0.02) return;
    // March up to the HIGHEST the HSE top can reach; columns without an event simply return no density
    // above their own local top, so the extra span costs only the adaptive step's empty-space skipping.
    float topEx=dk.top+dk.hseMax;
    float t0=(dk.base-ro.y)/rd.y, t1=(topEx-ro.y)/rd.y; if(t0>t1){ float tmp=t0; t0=t1; t1=tmp; }
    t0=max(t0,0.0); t1=min(t1, dk.fade); if(t1<=t0) return;
    // Sunlight reaching THIS deck is attenuated only by the air ABOVE the deck, not the whole column.
    float rC=planetRadius+0.5*(dk.base+topEx)*CLOUD_KM;
    vec3 sunT=sunTrans(vec3(0.0,rC,0.0),rC,L);
    vec3 sunCol=SUNCOL*sunMag*(sunT/max(sunT.r,1e-3));
    // CloudWorks marches in ABSOLUTE step lengths, which assumes a short path through a thin game-scale
    // deck. Here a grazing ray crosses a 2 km deck over 10+ km, so a fixed dtMin would run out of steps
    // and truncate the cloud. Floor the step at path/maxStep: short (upward) paths keep the fine step,
    // long (grazing) paths coarsen just enough to always finish -- bounded cost, no truncation.
    // Floor the step on the BASE deck's crossing length, not the HSE-extended one: the extra span is
    // mostly empty (only ~38% of cells hold an event) so the adaptive step blasts through it at dtMax,
    // and letting it coarsen the floor would undersample the detail everywhere and bring back speckle.
    float basePath = (dk.top-dk.base)/max(abs(rd.y),1e-3);
    float dtMin = max(dk.dtMin, min(basePath, t1-t0)/float(maxStep));
    float dtMax = max(dk.dtMax, dtMin*2.0);
    float t=t0 + dtMin*hash13(vec3(gl_FragCoord.xy, dk.base));
    float pdf=0.0;                                   // previous density field, for the step extrapolation
    for(int i=0;i<maxStep;i++){
        if(t>t1 || T<T_EXIT) break;
        vec3 p=ro+rd*t;
        float lod=1.0-smoothstep(5.0,26.0,t);          // detail LOD, matched to the step LOD
        float lTop,lMid; deckLocal(p, dk, lTop, lMid);
        vec3 cs=deckShapeL(p.y, dk, lTop, lMid);
        float df=0.0, d1=0.0;
        if(p.y<=lTop && cs.x>0.001){
            d1=chunk(p, dk, cs.x);
            if(d1>0.001) df = d1*(1.0 + detailB(d1, p, dk, lod));   // detail MODULATES the macro
        }
        // PREDICTIVE ADAPTIVE STEP: extrapolate the field one step ahead so the marcher decelerates
        // BEFORE the cloud edge, then grow the step with distance (LOD).
        float dt = ccmap(2.0*df - pdf, cs.z*0.85, 0.0, dtMin, dtMax);
        dt *= ccmap(t, 0.0, dk.fade, 1.0, dk.grow);
        pdf = df;
        if(df>cs.z){
            float dens=densityMix(df, p.y, dk, cs.z, cs.y, lTop)*dk.gain;
            if(dens>1e-4){
                float hf=clamp((p.y-dk.base)/max(lTop-dk.base,1e-3),0.0,1.0);
                float silver=1.0+uSilver*pow(max(mu,0.0),9.0);
                // CloudWorks only lights while the pixel is still translucent (its fx.y<2.3 gate) and
                // approximates the shadow for near-empty samples. A sample hidden behind opaque cloud, or
                // too thin to resolve a shadow, cannot show one -- skipping both saves the 6-step,
                // 3-noise-per-step sun march, which otherwise dominates the frame.
                bool cheap = (T<0.22) || (dens < 0.03*dk.solTop*dk.gain);
                float ss = cheap ? 2.2 : shadowMarch(p, L, dk, lTop, lMid);
                vec3 Lc; float dtau;
                if(dk.ice>0.5){
                    // thin translucent ice: transmitted forward-scatter + sky ambient, faint self-shadow.
                    float silverIce=1.0+0.5*uSilver*pow(max(mu,0.0),5.0);
                    float fwd=hgHenyey(mu,0.45)*silverIce + 0.35*hgHenyey(mu,-0.10);
                    float sun=exp(-ss)*fwd*(1.0+0.5*uMsBoost) + 0.5*uDiffuse*exp(-ss*0.5);
                    vec3 amb=mix(horAmb,zenAmb,clamp(0.80+0.20*hf,0.0,1.0))*uAmbStr*(0.55+0.45*hf);
                    dtau=dens*uSigma*dt*0.45;
                    Lc=sunCol*sun + amb;
                } else {
                    float sun=0.0,a=1.0,b=1.0,cc=1.0;
                    for(int n=0;n<4;n++){ float ph=mix(hgHenyey(mu,0.80*cc),hgHenyey(mu,-0.12),0.28)*silver;
                        sun+=a*exp(-b*ss)*ph*(n>=2?(1.0+uMsBoost):1.0); a*=0.45; b*=(n<2?0.5:0.32); cc*=0.82; }
                    // VIEW-INDEPENDENT multiple-scatter term: a thick cloud's sun-facing side is bright from
                    // ANY direction (near-Lambertian), which a pure HG(mu) phase cannot express.
                    sun += uDiffuse*exp(-ss*0.7)*(1.0+0.5*uMsBoost);
                    float powder=1.0-exp(-dens*3.0);
                    vec3 amb=mix(horAmb,zenAmb,clamp(0.80+0.20*hf,0.0,1.0))*uAmbStr*(0.35+0.65*hf);
                    dtau=dens*uSigma*dt;
                    Lc=sunCol*sun*mix(0.4,1.0,powder) + amb;
                }
                float ai=1.0-exp(-dtau);
                acc+=T*ai*Lc; T*=exp(-dtau); meanT+=t*ai; wsum+=ai;
            }
        }
        t += dt + dtMin*0.35*hash13(vec3(gl_FragCoord.xy, t));   // small jitter breaks residual banding
    }
}

void main(){
    vec2 uv=(gl_FragCoord.xy-0.5*uRes)/uRes.y;
    vec3 rd=uCam*normalize(vec3(uv,uFocal)); vec3 ro=uEye; vec3 L=normalize(uSunDir);
    // atmosphere() lives in PLANET space (observer at r~=planetRadius); clouds/camera live in a local flat
    // frame with the eye near the origin. Evaluate the sky with an observer placed on the globe, same up-axis.
    vec3 gro=vec3(0.0, planetRadius+0.0005, 0.0);
    vec3 sky=atmosphere(gro,rd,L,1.0);
    float mu0=max(dot(rd,L),0.0);
    sky+=vec3(1.0,0.80,0.52)*0.6*pow(mu0,40.0);                   // tight aerosol forward glow
    vec3 zenAmb, horAmb;
    if(uAmbMode>0.5){ zenAmb=uAmbZen; horAmb=uAmbHor; } else {
    zenAmb=atmosphere(gro,vec3(0,1,0),L,0.0);
    // Horizon ambient probe must NOT depend on the view direction. Sample at the SUN's azimuth, and mix
    // the anti-solar side in heavily: a cloud's shadow side faces away from the sun and sees the cooler
    // hemisphere, not the bright patch beside the sun.
    vec3 shz=vec3(L.x,0.0,L.z); vec3 hdir=(length(shz)>1e-4)?normalize(shz):vec3(0.0,0.0,1.0);
    horAmb=0.30*atmosphere(gro,normalize( hdir+vec3(0.0,0.10,0.0)),L,0.0)
               +0.70*atmosphere(gro,normalize(-hdir+vec3(0.0,0.10,0.0)),L,0.0); }
    if(uProbe>0.5){ fragColor=vec4(uProbe<1.5?zenAmb:horAmb,1.0); return; }   // 1x1 pre-pass
    // sunlight reaching the clouds is REDDENED by its atmospheric path; magnitude SATURATES well before
    // zenith so the clouds do not blow out to white at noon.
    float sunMag=0.14+0.56*smoothstep(0.0,0.90,L.y);

    vec3 col=sky;
    // March the decks from ANY camera height, near->far (order reverses when looking down).
    if(abs(rd.y) > 0.002){
        float mu=dot(rd,L);
        float T=1.0; vec3 acc=vec3(0.0); float meanT=0.0, wsum=0.0;
        Deck lo=lowDeck(), md=midDeck(), hi=hiDeck();
        if(rd.y > 0.0){
            if(uCovLow >0.02) marchDeck(ro,rd,L, lo, 72, mu,sunMag,zenAmb,horAmb, T,acc,meanT,wsum);
            if(uCovMid >0.02) marchDeck(ro,rd,L, md, 28, mu,sunMag,zenAmb,horAmb, T,acc,meanT,wsum);
            if(uCovHigh>0.02) marchDeck(ro,rd,L, hi, 20, mu,sunMag,zenAmb,horAmb, T,acc,meanT,wsum);
        } else {
            if(uCovHigh>0.02) marchDeck(ro,rd,L, hi, 20, mu,sunMag,zenAmb,horAmb, T,acc,meanT,wsum);
            if(uCovMid >0.02) marchDeck(ro,rd,L, md, 28, mu,sunMag,zenAmb,horAmb, T,acc,meanT,wsum);
            if(uCovLow >0.02) marchDeck(ro,rd,L, lo, 72, mu,sunMag,zenAmb,horAmb, T,acc,meanT,wsum);
        }
        float md2=(wsum>0.0)?meanT/wsum:1.0; float fog=exp(-max(md2-6.0,0.0)*0.006);
        vec3 cloudCol=mix(sky, acc, fog);
        col=sky*T + cloudCol*(1.0-T);
    }
    fragColor=vec4(tonemap(col),1.0);
}
