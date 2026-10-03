#version 330 core
// Multi-layer cumulus sky for the MULTI-PHOTO VSGA.
//  - Physically-driven sky: the Paper-3 atmosphere() (Rayleigh + ozone + grey aerosol) is the sky colour,
//    so a per-photo SUN direction + SHARED atmosphere params reproduce each photo's time-of-day colour.
//  - THREE cloud decks, each with its own altitude band and a per-photo coverage knob:
//      low  = cumulus     (thick, billowy, cauliflower tops)
//      mid  = altocumulus (thin "mackerel" cellular slab)
//      high = cirrus      (very thin wind-stretched wisps)
//  - One front-to-back march per deck (low -> mid -> high) carrying transmittance.
// Cloud/atmosphere params are UNIFORMS; the optimizer splits them into shared vs per-photo (see optimize_multiphoto.py).
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
uniform float uWarp;                             // flow-field domain-warp strength (cirrus flow-drag)
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
#define DET_FREQ 1.6
#define COARSE 8.0        // empty-space skip multiplier
#define THRESH 0.10
#define SUN_STEPS 3
#define SUNCOL (vec3(1.9,1.75,1.50)*3.4)

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
float billowFbm(vec3 p,int oct){ float s=0.0,a=0.5,f=1.0,n=0.0; for(int o=0;o<oct;o++){ s+=a*abs(vnoise(p*f)*2.0-1.0); n+=a; a*=0.5; f*=2.02; } return s/n; }
float ridgedFbm(vec3 p,int oct){ float s=0.0,a=0.5,f=1.0,n=0.0; for(int o=0;o<oct;o++){ float r=1.0-abs(vnoise(p*f)*2.0-1.0); s+=a*r*r; n+=a; a*=0.5; f*=2.02; } return s/n; }
float remap(float x,float a,float b,float c,float d){ return c+(clamp(x,a,b)-a)*(d-c)/max(b-a,1e-4); }
// continuous flow-field domain warp: drags DETAIL sample coords along a smooth flow, modulated by a coarse
// field, so wisps look torn along atmospheric flow (strongest for cirrus). NB: domain-warp defeats A3.
vec3 flowWarpF(vec3 p,float amp,float freq,int oct){
    if(amp<1e-3) return p;
    float c=fbm(p*freq*0.30 + vec3(13.0,5.0,21.0), 1);                        // coarse strength field
    // two-octave flow, biased HORIZONTAL (wind). `freq` sets the SOURCE NOISE SCALE: low freq + large amp
    // is what drags detail into long filaments; high freq only tears locally.
    vec3 f=vec3(1.6*fbm(p*freq+vec3(0.0,20.0,0.0),oct), 0.35*fbm(p*freq+vec3(40.0,0.0,10.0),oct), 1.6*fbm(p*freq+vec3(7.0,60.0,3.0),oct))
          -vec3(0.8,0.17,0.8);
    if(oct>2) f += 0.5*(vec3(fbm(p*freq*2.2+vec3(5.0),2),0.0,fbm(p*freq*2.2+vec3(31.0),2))-0.5);
    return p + f*amp*(0.45+0.8*c);
}
vec3 flowWarp(vec3 p,float amp){ return flowWarpF(p,amp,1.2,3); }               // cumulus/altocumulus detail


// ================= Paper-3 atmosphere (+ grey aerosol) =================
const float planetRadius=6.371, atmRadius=6.471, Hatm=0.008, Csw=1.12;
const float Haer=0.0016;
const float CLOUD_KM=0.001;   // 1 local cloud unit == 1 km of real altitude
const float CIRRUS_ARC=12.0;
const float CIRRUS_WARP_AMP=35.0;  // cirrus filament drag: LARGE amplitude
const float CIRRUS_WARP_FREQ=0.05; // ...over a LARGE source-noise scale   // lateral sweep amplitude of the large-scale cirrus arcs
const float CIRRUS_SHEAR=4.0; // fallstreak lean across the deck thickness   // 1 local cloud unit == 1 km of real altitude                         // aerosol scale height (~1.6 units)
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

// ================= cloud decks =================
// low: cumulus (thick, billowy) ------------------------------------------------
float lowDeck(vec3 p){
    float hf=(p.y-uLowBase)/(uLowTop-uLowBase); if(hf<0.0||hf>1.0) return 0.0;
    // WIND SHEAR: displace the sample horizontally in proportion to height, so the column leans and the
    // deck develops on a slant instead of growing as vertical pillars. Very cheap, and it is what gives
    // real cumulus fields their tilted, wind-blown look.
    vec3 sp=p; sp.x += hf*uShear;
    float ct=fbm(vec3(sp.xz*uCovFreq*0.34, 20.0+uTypeSeed), 3);          // large-scale type field (per-photo seed)
    float th=mix(0.72,0.30,uCovLow) - uTypeAmp*(ct-0.5);
    float cov=smoothstep(th, th+0.30, fbm(vec3(sp.xz*uCovFreq, 3.7), 5));  // width ~ beauty's uCovHi-uCovLo
    // ASYMMETRIC CONVEX profile: flat cumulus base, heavy lower body, convex narrowing toward the top.
    // (The old symmetric smoothstep taper gave a slab-like body with no weight at the base.)
    // Rounded cauliflower TOP: a gentle smoothstep shoulder (not pow(), whose steep top-edge collapse
    // let the erosion layers chop the top into confetti). Erosion is also weakened near the top so the
    // billow lumps survive as rounded cauliflower rather than being cut apart.
    float prof=smoothstep(0.0,0.08,hf)*smoothstep(1.0, mix(0.55,0.18,ct), hf);
    float base=cov*prof; if(base<=0.001) return 0.0;
    float body=billowFbm(sp*DET_FREQ + vec3(0.0,2.0,0.0), 6);
    float shape=base*(0.55 + 0.65*body);
    float er1=uDetAmp *mix(0.9,0.5,hf)*ridgedFbm(sp*DET_FREQ*2.3  + vec3(5.0,0.0,9.0), 5);
    float d=remap(shape, er1, 1.0, 0.0, 1.0);
    vec3 pw=flowWarpF(sp, uWarp*0.25, 1.2, 1);   // tiny displacement: 1 octave is enough                                     // light flow-drag on fine detail
    float er2=uDet2Amp*mix(0.40,1.0,hf)*billowFbm(pw*DET_FREQ*5.1  + vec3(11.0,3.0,2.0), 3);
    d=remap(d, er2, 1.0, 0.0, 1.0);
    float er3=uDet3Amp*ridgedFbm(pw*DET_FREQ*11.3 + vec3(1.0,7.0,4.0), 2);
    d -= er3*smoothstep(0.0,0.45,d)*mix(0.7,0.45,hf);
    d -= THRESH;
    return clamp(d,0.0,1.0)*uDens*mix(0.75,1.25,ct);
}
// mid: cirrocumulus / altocumulus -- THIN, translucent, granular ripples in rows (NOT a flat slab).
// Small rounded granules (billow), row anisotropy, plus a 3-D (p.y) variation so the layer has internal
// structure and soft edges instead of reading as one flat sheet.
float midDeck(vec3 p){
    if(uCovMid<0.02) return 0.0;                                        // deck off (per-photo)
    float hf=(p.y-uMidBase)/(uMidTop-uMidBase); if(hf<0.0||hf>1.0) return 0.0;
    vec3 sp=p; sp.x += hf*uShear*0.5;
    vec3 pw=flowWarpF(sp, uWarp*0.6, 1.2, 2);
    vec3 q=pw; q.x*=0.55;                                               // slight row alignment (ripples)
    float gran=billowFbm(vec3(q.xz*uCovFreq*4.2, 8.0+uTypeSeed), 4);    // small rounded granules
    float vary=0.55+0.45*vnoise(pw*uCovFreq*3.0 + vec3(0.0,40.0,0.0));  // 3-D break-up -> not a flat sheet
    float th=mix(0.60,0.32,uCovMid);
    float cov=smoothstep(th, th+0.22, gran*vary);        // wider = softer edges (was 0.12)
    float prof=smoothstep(0.0,0.35,hf)*smoothstep(1.0,0.45,hf);         // soft thin vertical envelope
    float d=cov*prof;
    d -= uDet2Amp*0.5*ridgedFbm(pw*DET_FREQ*10.0 + vec3(2.0,5.0,7.0), 3);
    d -= 0.04;
    return clamp(d,0.0,1.0)*uDens*0.28;                                 // THIN (was 0.55)
}
// high: cirrus -- VERY thin, translucent, wind-stretched ice wisps.
float hiDeck(vec3 p){
    if(uCovHigh<0.02) return 0.0;                                       // deck off (per-photo)
    float hf=(p.y-uHiBase)/(uHiTop-uHiBase); if(hf<0.0||hf>1.0) return 0.0;
    // Large-scale FLOWING BANDS, not confetti. Three things make cirrus read as long streamlines:
    //  - extreme anisotropy (~16:1) so features elongate into bands,
    //  - a LOW base frequency with FEW octaves so a band stays coherent across the sky,
    //  - fine fibres that MULTIPLY inside the band (streaky texture) instead of being subtracted,
    //    which is what used to chop the deck into small fragments.
    vec3 pw=flowWarpF(p, uWarp*CIRRUS_WARP_AMP, CIRRUS_WARP_FREQ, 2);
    // Large-scale sweeping ARC: a low-frequency, high-amplitude lateral displacement, so bands curve
    // right across the sky. The isotropic flowWarp alone only tears locally; it cannot make long arcs.
    float arc=fbm(vec3(p.xz*uCovFreq*0.22, 3.0+uTypeSeed), 2)-0.5;
    pw.z += arc*CIRRUS_ARC*uWarp;
    // FALLSTREAK shear: ice crystals fall and lag behind the wind, so the lower part of a band trails
    // the upper part -- this is what makes the hooks/tails of cirrus uncinus.
    pw.x += (1.0-hf)*CIRRUS_SHEAR*uWarp;
    vec3 q=pw; q.x*=0.12;                                              // ~8:1 elongation
    float band=ridgedFbm(vec3(q.xz*uCovFreq*1.80, 14.0+uTypeSeed), 4);   // enough octaves to keep fibre
    float th=mix(0.66,0.30,uCovHigh);
    float cov=smoothstep(th, th+0.14, band);                            // defined, not washed flat
    // Fine fibre needs its OWN lightly-warped coordinate. Sharing the band's huge low-frequency warp
    // (tens of units of displacement) stretched the fine noise along the flow and smeared it away, so
    // raising its frequency did nothing and the deck read as big distorted blocks.
    vec3 fq=flowWarpF(p, uWarp*3.0, 0.55, 2); fq.x*=0.35;
    float fibN=0.20+0.80*ridgedFbm(vec3(fq.xz*uCovFreq*26.0, 31.0+uTypeSeed), 4);
    float prof=smoothstep(0.0,0.45,hf)*smoothstep(1.0,0.35,hf);
    float d=cov*prof*mix(1.0, fibN, clamp(0.45+uDet3Amp*1.8,0.0,1.0));
    d -= 0.02;
    return clamp(d,0.0,1.0)*uDens*0.10;                                 // VERY thin
}
float cloudDensity(vec3 p){
    if(p.y>=uLowBase && p.y<=uLowTop) return lowDeck(p);
    if(p.y>=uMidBase && p.y<=uMidTop) return midDeck(p);
    if(p.y>=uHiBase  && p.y<=uHiTop ) return hiDeck(p);
    return 0.0;
}
// Self-shadow must use the SAME extinction as the view ray. It used to ignore uSigma entirely, so the
// optimizer could buy opacity with uSigma without ever paying the shadow cost -- and self-shadowing is
// what sculpts cloud form. Normalised by a reference sigma of 5 so the established look is preserved.
// Shadow density: shadows are a LOW-FREQUENCY quantity, so the fine erosion/fibre layers that
// dominate cloudDensity cost are skipped here. Visually near-identical, roughly half the work.
float shadowDensity(vec3 p){
    if(p.y>=uLowBase && p.y<=uLowTop){
        float hf=(p.y-uLowBase)/(uLowTop-uLowBase);
        vec3 sp=p; sp.x += hf*uShear;
        float ct=fbm(vec3(sp.xz*uCovFreq*0.34, 20.0+uTypeSeed), 2);
        float th=mix(0.72,0.30,uCovLow) - uTypeAmp*(ct-0.5);
        float cov=smoothstep(th, th+0.30, fbm(vec3(sp.xz*uCovFreq, 3.7), 3));
        float prof=smoothstep(0.0,0.08,hf)*smoothstep(1.0, mix(0.55,0.18,ct), hf);
        float d=cov*prof*(0.55+0.65*billowFbm(sp*DET_FREQ + vec3(0.0,2.0,0.0), 3));
        return clamp(d-THRESH,0.0,1.0)*uDens*mix(0.75,1.25,ct);
    }
    // The ICE decks need cheap shadow variants too. They used to call the FULL midDeck/hiDeck, so every
    // lit sample's shadow march paid two flowWarpF calls (7 vnoise each) plus the 26x fine-fibre field.
    // Ablation: that single defect was 37% of the whole frame -- far more than the cirrus deck's own
    // 16-step primary march. Shadows only need the coarse band shape, so drop the warps and the fibre.
    if(p.y>=uMidBase && p.y<=uMidTop){
        if(uCovMid<0.02) return 0.0;
        float hf=(p.y-uMidBase)/(uMidTop-uMidBase);
        float gran=billowFbm(vec3(p.xz*uCovFreq*4.2*vec2(0.55,1.0), 8.0+uTypeSeed), 2);
        float cov=smoothstep(mix(0.60,0.32,uCovMid), mix(0.60,0.32,uCovMid)+0.12, gran);
        float prof=smoothstep(0.0,0.35,hf)*smoothstep(1.0,0.45,hf);
        return clamp(cov*prof-0.04,0.0,1.0)*uDens*0.28;
    }
    if(p.y>=uHiBase  && p.y<=uHiTop ){
        if(uCovHigh<0.02) return 0.0;
        float hf=(p.y-uHiBase)/(uHiTop-uHiBase);
        float band=ridgedFbm(vec3(p.xz*vec2(0.12,1.0)*uCovFreq*1.80, 14.0+uTypeSeed), 2);
        float cov=smoothstep(mix(0.66,0.30,uCovHigh), mix(0.66,0.30,uCovHigh)+0.14, band);
        float prof=smoothstep(0.0,0.45,hf)*smoothstep(1.0,0.35,hf);
        return clamp(cov*prof-0.02,0.0,1.0)*uDens*0.10;
    }
    return 0.0;
}
float sunShadow(vec3 p){ float t=0.05,tau=0.0; float sg=uSigma*0.2;
    for(int i=0;i<SUN_STEPS;i++){ float st=0.24*(1.0+0.8*float(i)); tau+=shadowDensity(p+uSunDir*(t))*st*sg; t+=st; if(tau>10.0) break; }
    return tau; }
vec3 tonemap(vec3 c){ c=1.0-exp(-c*uExposure); return pow(clamp(c,0.0,1.0),vec3(1.0/2.2)); }

// march one deck's altitude band, front-to-back, carrying transmittance.
// ice>0.5 selects a THIN ICE-CLOUD lighting model (translucent, forward-scatter dominant, only faint
// self-shadow) so cirro/cirrus read as bright wisps lit THROUGH, not a uniformly-shadowed board.
void marchDeck(vec3 ro,vec3 rd,vec3 L,float base,float top,int steps,float mu,float sunMag,vec3 zenAmb,vec3 horAmb,float ice,float cov,
               inout float T, inout vec3 acc, inout float meanT, inout float wsum){
    if(T<0.02||cov<0.02) return;                 // deck switched off for this photo: do not march it at all
    // Slab intersection + empty-deck early-out FIRST. sunTrans below is a Chapman evaluation; computing it
    // before finding out the ray misses the slab cost ~2x once downward rays (which mostly miss) began
    // entering marchDeck.
    float t0=(base-ro.y)/rd.y, t1=(top-ro.y)/rd.y; if(t0>t1){ float tmp=t0; t0=t1; t1=tmp; }
    t0=max(t0,0.0); t1=min(t1, t0+90.0); if(t1<=t0) return;
    // Sunlight reaching THIS deck is attenuated only by the air ABOVE the deck, not the whole column.
    float rC=planetRadius+0.5*(base+top)*CLOUD_KM;
    vec3 sunT=sunTrans(vec3(0.0,rC,0.0),rC,L);
    vec3 sunCol=SUNCOL*sunMag*(sunT/max(sunT.r,1e-3));
    float dt=(t1-t0)/float(steps);
    float t=t0+dt*hash13(vec3(gl_FragCoord.xy, base));
    // Empty-space skipping: most of a sparse deck is air, so march it with COARSE steps and drop to the
    // fine step only after a hit (backing up one coarse step so nothing is missed).
    float step=dt*COARSE; int miss=99;                 // miss = consecutive empty FINE samples
    for(int i=0;i<steps;i++){
        if(t>t1||T<0.05) break;
        vec3 p=ro+rd*t; float d=cloudDensity(p);
        if(d<=0.001){
            miss++;
            if(miss>=3){ step=dt*COARSE; }            // only revert to coarse after leaving the cloud;
            t+=step; continue;                        // reverting on the FIRST empty made it oscillate
        }
        if(step>dt){ t=max(t0,t-step); step=dt; miss=0; continue; }   // first hit: back up, go fine
        miss=0;
        {
            float silver=1.0+uSilver*pow(max(mu,0.0),9.0);
            float hf=clamp((p.y-base)/(top-base),0.0,1.0);
            vec3 Lc; float dtau;
            // Samples that are nearly invisible (deep behind cloud) or very thin cannot resolve a
            // shadow the eye will see, so skip the whole shadow march for them.
            // Shadow is what sculpts cloud FORM, so the water deck keeps an accurate shadow while the thin ice
            // decks (which carry no modelling) stay on the wide skip.
            bool cheap = (ice>0.5) ? ((T<0.25)||(d<0.04)) : ((T<0.08)||(d<0.012));
            if(ice>0.5){
                // thin translucent ice: transmitted forward-scatter + sky ambient, only a faint self-shadow.
                // The forward lobe is deliberately SOFT (broad g, gentle silver) - the old sharp
                // pow(mu,9) spike made the thin decks show hard, defect-looking scattering edges.
                float ss=(cheap?2.0:sunShadow(p))*0.30;
                float silverIce=1.0+0.5*uSilver*pow(max(mu,0.0),5.0);
                float fwd=hgHenyey(mu,0.45)*silverIce + 0.35*hgHenyey(mu,-0.10);
                float sun=exp(-ss)*fwd*(1.0+0.5*uMsBoost) + 0.5*uDiffuse*exp(-ss*0.5);
                vec3 amb=mix(horAmb,zenAmb,clamp(0.80+0.20*hf,0.0,1.0))*uAmbStr*(0.55+0.45*hf);
                dtau=d*uSigma*dt*0.45;                              // low optical depth -> see-through
                Lc=sunCol*sun + amb;
            } else {
                float ss=cheap?3.0:sunShadow(p);
                float sun=0.0,a=1.0,b=1.0,cc=1.0;
                for(int n=0;n<4;n++){ float ph=mix(hgHenyey(mu,0.80*cc),hgHenyey(mu,-0.12),0.28)*silver;
                    sun+=a*exp(-b*ss)*ph*(n>=2?(1.0+uMsBoost):1.0); a*=0.45; b*=(n<2?0.5:0.32); cc*=0.82; }
                // VIEW-INDEPENDENT multiple-scatter term: a thick cloud's sun-facing side is bright from ANY
                // direction (near-Lambertian), which a pure HG(mu) phase cannot express - without this the
                // clouds look unlit whenever the sun is off to the side (azimuth ~ +-90 deg).
                sun += uDiffuse*exp(-ss*0.7)*(1.0+0.5*uMsBoost);
                float powder=1.0-exp(-d*3.0);
                vec3 amb=mix(horAmb,zenAmb,clamp(0.80+0.20*hf,0.0,1.0))*uAmbStr*(0.35+0.65*hf);  // >=80% zenith. The horizon probe is 10-15x BRIGHTER than the zenith, so a nominal 40% still let it
            // supply ~93% of the ambient; only a heavy zenith bias actually cools the shadow sides.
                dtau=d*uSigma*dt;
                Lc=sunCol*sun*mix(0.4,1.0,powder) + amb;
            }
            float ai=1.0-exp(-dtau);
            acc+=T*ai*Lc; T*=exp(-dtau); meanT+=t*ai; wsum+=ai;
        }
        t+=step;
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
    // Horizon ambient probe must NOT depend on the view direction: normalize(vec3(rd.x,0.08,rd.z))
    // degenerates to straight up as the ray nears the zenith, so at low sun (dark zenith) the cloud
    // ambient collapsed into a dark halo. Using the view azimuth instead just moves the problem to a
    // radial singularity. Sample the horizon at the SUN's azimuth: view-independent, and physically the
    // bright side of the horizon at sunset.
    vec3 shz=vec3(L.x,0.0,L.z); vec3 hdir=(length(shz)>1e-4)?normalize(shz):vec3(0.0,0.0,1.0);
    // Average the sun-side AND anti-solar horizon: a cloud's shadow side is lit by the whole sky dome,
    // which is dominated by blue sky, not by the small bright patch of horizon near the sun.
    // Weighted toward the ANTI-SOLAR horizon: the shadow side of a cloud faces away from the sun, so the
    // sky it actually sees is the cooler anti-solar hemisphere, not the bright horizon beside the sun.
    horAmb=0.30*atmosphere(gro,normalize( hdir+vec3(0.0,0.10,0.0)),L,0.0)
               +0.70*atmosphere(gro,normalize(-hdir+vec3(0.0,0.10,0.0)),L,0.0); }
    if(uProbe>0.5){ fragColor=vec4(uProbe<1.5?zenAmb:horAmb,1.0); return; }   // 1x1 pre-pass
    // sunlight that reaches the clouds is REDDENED by its atmospheric path (Chapman transmittance toward the
    // sun): white-ish at noon, warm/orange at low sun -> cloud lit faces warm at sunset instead of staying white.
    // magnitude SATURATES well before zenith (the old clamp(L.y*1.4+0.15) reached 1.55x at the zenith and
    // blew the clouds out to white, and double-counted the elevation dependence already in sunT).
    float sunMag=0.14+0.56*smoothstep(0.0,0.90,L.y);   // gentler ramp, saturates below 1 -> no zenith blow-out

    // NO ground: the render is only ever composited into a photo's SKY (mask), the ground is always the
    // real photo. Below the horizon we just continue the atmosphere so nothing green bleeds into the seam.
    vec3 col=sky;
    // March the decks from ANY camera height. The slab intersection already handles the eye being below,
    // inside, or above a deck; what mattered was (a) not gating on rd.y>0 (that dropped all downward rays,
    // so looking down from above the clouds rendered nothing) and (b) ordering the decks near->far, which
    // reverses when looking down. Skip only near-horizontal rays, where 1/rd.y explodes.
    if(abs(rd.y) > 0.002){
        float mu=dot(rd,L);
        float T=1.0; vec3 acc=vec3(0.0); float meanT=0.0, wsum=0.0;
        if(rd.y > 0.0){                                        // below/within, looking up: low -> mid -> high
            marchDeck(ro,rd,L, uLowBase,uLowTop, 24, mu,sunMag,zenAmb,horAmb, 0.0,uCovLow, T,acc,meanT,wsum);
            marchDeck(ro,rd,L, uMidBase,uMidTop,  10, mu,sunMag,zenAmb,horAmb, 1.0,uCovMid, T,acc,meanT,wsum);
            marchDeck(ro,rd,L, uHiBase, uHiTop,    4, mu,sunMag,zenAmb,horAmb, 1.0,uCovHigh, T,acc,meanT,wsum);
        } else {                                               // above/within, looking down: high -> mid -> low
            marchDeck(ro,rd,L, uHiBase, uHiTop,    4, mu,sunMag,zenAmb,horAmb, 1.0,uCovHigh, T,acc,meanT,wsum);
            marchDeck(ro,rd,L, uMidBase,uMidTop,  10, mu,sunMag,zenAmb,horAmb, 1.0,uCovMid, T,acc,meanT,wsum);
            marchDeck(ro,rd,L, uLowBase,uLowTop, 24, mu,sunMag,zenAmb,horAmb, 0.0,uCovLow, T,acc,meanT,wsum);
        }
        float md=(wsum>0.0)?meanT/wsum:1.0; float fog=exp(-max(md-6.0,0.0)*0.006);
        vec3 cloudCol=mix(sky, acc, fog);
        col=sky*T + cloudCol*(1.0-T);
    }
    fragColor=vec4(tonemap(col),1.0);
}
