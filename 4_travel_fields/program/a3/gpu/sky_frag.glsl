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

#define PI 3.14159265358979
#define SIGMA_T 7.0
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

const vec3 SUN = vec3(0.60, 0.62, 0.50);        // afternoon sun
const vec3 SUNCOL = vec3(1.7, 1.45, 1.15) * 4.2;
const vec3 SKYTOP = vec3(0.20, 0.38, 0.70);
const vec3 SKYHOR = vec3(0.72, 0.80, 0.90);

const float GL2X[2] = float[2](-0.5773502691896257, 0.5773502691896257);
const float GL3X[3] = float[3](-0.7745966692414834, 0.0, 0.7745966692414834);
const float GL3W[3] = float[3](0.5555555555555556, 0.8888888888888888, 0.5555555555555556);
const float GL5X[5] = float[5](-0.906179845938664,-0.538469310105683,0.0,0.538469310105683,0.906179845938664);
const float GL5W[5] = float[5](0.236926885056189,0.478628670499366,0.568888888888889,0.478628670499366,0.236926885056189);

float gJit = 0.0;
float gEvals = 0.0, gSunRays = 0.0, gSkipped = 0.0, gSegs = 0.0;

float hash13(vec3 p){ p = fract(p*vec3(0.1031,0.1030,0.0973)); p += dot(p,p.yzx+33.33); return fract((p.x+p.y)*p.z); }
float noiseCubic(vec3 p){
    vec3 i=floor(p), f=p-i; vec3 u=f*f*(3.0-2.0*f);
    float n000=hash13(i+vec3(0,0,0)),n100=hash13(i+vec3(1,0,0)),n010=hash13(i+vec3(0,1,0)),n110=hash13(i+vec3(1,1,0));
    float n001=hash13(i+vec3(0,0,1)),n101=hash13(i+vec3(1,0,1)),n011=hash13(i+vec3(0,1,1)),n111=hash13(i+vec3(1,1,1));
    float x00=mix(n000,n100,u.x),x10=mix(n010,n110,u.x),x01=mix(n001,n101,u.x),x11=mix(n011,n111,u.x);
    return mix(mix(x00,x10,u.y),mix(x01,x11,u.y),u.z);
}
float fbm(vec3 p,int n){ float d=0.0,a=0.5,f=1.0; for(int o=0;o<n;o++){d+=a*noiseCubic(p*f);a*=0.5;f*=2.0;} return d; }
float hg(float y,float a,float b,float fa,float fb){ return smoothstep(a-fa,a+fa,y)*smoothstep(b+fb,b-fb,y); }

// coarse cloud-presence (backbone only, cheap): used for empty-space skipping AND as the A3 empty bound
float presence(vec3 p){
    gEvals+=1.0;
    float y=p.y;
    float low = hg(y,0.9,2.2,0.35,0.6) * smoothstep(0.48,0.80, fbm(vec3(p.xz*0.33,11.0),2));
    float mid = hg(y,3.0,4.1,0.30,0.4) * smoothstep(0.55,0.82, fbm(vec3(p.xz*0.5,40.0),2));
    float high= hg(y,5.5,6.9,0.25,0.35)* smoothstep(0.60,0.86, fbm(vec3(p.xz*0.7,80.0),2));
    float r=length(p.xz-vec2(CB_X,CB_Z));
    float trunkR=mix(0.55,0.95,smoothstep(0.9,4.5,y));
    float anvilR=mix(0.0,3.4,smoothstep(6.0,7.4,y));
    float rad=max(trunkR,anvilR);
    float cb = hg(y,0.9,7.7,0.35,0.5)*smoothstep(rad,rad*0.4,r);
    float cap= hg(y,6.9,7.7,0.25,0.35)*smoothstep(3.4,1.0,r);
    return max(max(low,mid),max(max(high,cb*1.4),cap*1.4));
}

#ifndef LAYERS
#define LAYERS 1
#endif
float cloudDensity(vec3 p){
    gEvals+=1.0;
    float y=p.y, d=0.0;
#if LAYERS==1
    // LOW cumulus (puffy)
    float lc = smoothstep(0.48,0.80, fbm(vec3(p.xz*0.33,11.0),3));
    float lb = hg(y,0.9,2.2,0.35,0.6)*lc;
    d += max(lb*fbm(p*0.95+vec3(0,3,0),4) - 0.14, 0.0)*1.7;
    // MID altocumulus (sheet-like, dappled)
    float mc = smoothstep(0.50,0.78, fbm(vec3(p.xz*0.45,40.0),3));
    float mb = hg(y,3.1,4.2,0.28,0.35)*mc;
    d += max(mb*fbm(p*vec3(1.3,2.0,1.3)+vec3(5,0,5),4) - 0.16, 0.0)*1.35;
    // HIGH cirrus (thin, wind-stretched sheets)
    float hc = smoothstep(0.52,0.80, fbm(vec3(p.xz*0.6,80.0),3));
    float hb = hg(y,5.6,6.7,0.22,0.3)*hc;
    d += max(hb*fbm(p*vec3(2.0,0.5,2.0)+vec3(9,0,0),3) - 0.18, 0.0)*0.9;
#endif
    // CUMULONIMBUS: narrow trunk (r~0.6) that flares into a wide anvil near the top
    float r=length(p.xz-vec2(CB_X,CB_Z));
    float trunkR = mix(0.55, 0.95, smoothstep(0.9,4.5,y));          // slightly widening trunk
    float anvilR = mix(0.0, 3.4, smoothstep(6.0,7.4,y));            // anvil spread only near the top
    float rad = max(trunkR, anvilR);
    float col=smoothstep(rad, rad*0.4, r);
    float tb=hg(y,0.9,7.7,0.35,0.5)*col;
    d += max(tb*(0.55+fbm(p*0.7+vec3(0,1,0),5)) - 0.18, 0.0)*2.3;
    // flat anvil cap (spreads horizontally, thin in height)
    float cap = hg(y,6.9,7.7,0.25,0.35)*smoothstep(3.4,1.0,r);
    d += max(cap*(0.5+fbm(p*vec3(1.1,0.5,1.1),4)) - 0.2, 0.0)*1.4;
    return clamp(d*DENS, 0.0, 12.0);
}

float phaseHG(float c){ float g2=HG_G*HG_G; return (1.0-g2)/(4.0*PI*pow(1.0+g2-2.0*HG_G*c,1.5)); }
// cheap multi-scatter approx (Wrenninge octave beer-powder): decreasing extinction per octave
// lets light "leak" deeper -> bright cloud cores instead of black silhouettes.
vec3 lightCloud(float sunT, float mu, float dtau, float y){
    float ph = phaseHG(mu); vec3 acc = vec3(0.0); float a=1.0, b=1.0;
    for(int n=0;n<3;n++){ acc += a*ph*exp(-b*sunT)*SUNCOL; a*=0.5; b*=0.35; }
    float powder = 1.0 - exp(-dtau*2.5);
    acc *= mix(0.55, 1.0, powder);
    vec3 amb = mix(SKYHOR, SKYTOP, clamp(y/YMAX,0.0,1.0))*0.85 + vec3(0.12,0.13,0.15);
    return ALBEDO*(acc + amb);
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
        if(presence(p)>0.02) tau += cloudDensity(p)*SIGMA_T*st;
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
        float dens=cloudDensity(p);
        if(dens>1e-3){
            float dtau=dens*SIGMA_T*small;
            vec3 S = lightCloud(sunTau(p), dot(rd,SUN), dtau, p.y);
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
    float cs=1.0/float(A3_CAMF); vec3 inv=1.0/(rd+(1.0-step(vec3(1e-6),abs(rd)))*1e-6);
    float Tc=1.0; vec3 L=vec3(0.0); float ph=phaseHG(dot(rd,SUN));
    float t=t0 + gJit*BIG_STEP;
    for(int i=0;i<MAXI;i++){                       // per-iteration adaptive: empty->big jump, cloud->fine cell
        if(t>t1 || Tc<0.01) break;
        vec3 p=ro+rd*t;
        if(presence(p) <= 0.02){ gSkipped+=1.0; t += BIG_STEP; continue; }   // coarse empty-space skip
        gSegs+=1.0;                                                          // one fine DDA cell + exact GL
        vec3 nb=(floor(p/cs)+max(sign(rd),0.0))*cs; vec3 tn=(nb-p)*inv;
        float adv=clamp(min(tn.x,min(tn.y,tn.z)),0.02,cs*2.0);              // bounded, always advances
        float tnext=t+adv+1e-5; float tb=min(tnext,t1);
        vec3 c=ro+rd*(0.5*(t+tb));
        if(occ(c)){
            float dtau=glTau(ro,rd,t,tb,A3_CAMGL);
            if(dtau>1e-4){
                float jf=0.5+SUN_JITTER*(gJit-0.5);
                vec3 xc=ro+rd*mix(t,tb,jf);
                vec3 S=lightCloud(sunTau(xc), dot(rd,SUN), dtau, xc.y);
                L += Tc*S*(1.0-exp(-dtau)); Tc *= exp(-dtau);
            }
        }
        t=tnext;
    }
    alpha=1.0-Tc; return L;
}

void main(){
    vec2 uv=(gl_FragCoord.xy-0.5*uRes)/uRes.y;
    gJit = float(JITTER_ON) * hash13(vec3(gl_FragCoord.xy,0.5));
    vec3 rd=uCam*normalize(vec3(uv,1.25)); vec3 ro=uEye;
    float alpha; vec3 cloud;
#if METHOD==0
    cloud=renderA3(ro,rd,alpha);
#elif METHOD==2
    cloud=renderA3H(ro,rd,alpha);
#else
    cloud=renderMarch(ro,rd,alpha);
#endif
#if INSTRUMENT==1
    fragColor=vec4(gEvals,gSunRays,gSkipped,gSegs);
#else
    fragColor=vec4(tonemap(sky(rd)*(1.0-alpha)+cloud),1.0);
#endif
}
