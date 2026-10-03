#version 330 core
// ============================================================
// f8 DDA-segmented ReLU cloud renderer  (empirical baseline, NOT a formal A3 kernel)
//
// HONEST SCOPE (per collaborator audit -- these are the accurate claims):
//   * measured 1.60x equal-reference-RMSE speedup vs fixed-step raymarch on ONE
//     RTX 4070 scene/camera. It is an empirical engineering baseline.
//   * NOT exact polynomial integration: per segment it uses high-order GL
//     QUADRATURE of the shaped density. ReLU max(f-theta,0) is piecewise
//     polynomial (needs a root split at f=theta to be exact), and the envelope
//     (length + smoothstep) is not polynomial. So GL5 here is approximate,
//     transport-stable but not quadrature-exact.
//   * NOT a provable empty skip: occupied() tests the segment MIDPOINT only, so
//     the 68.7% skip is a MIDPOINT HEURISTIC, not a certified segment-empty bound.
//   * NOT the validated transactional DDA: cells are re-derived from floor(p/cs)
//     with a 1e-6 nudge per crossing (small gaps possible); no integer-plane
//     index / axis mask / atomic multi-axis commit.
//   * NO cached octave state / NO direct degree-9 builder: every density() call
//     recomputes 8 hashes/octave. (So the 1.60x is achieved WITHOUT those opts.)
//   * per-segment INDEPENDENT camera attenuation (no cross-segment scattering
//     batch), Tcam early termination. pure float32.
//
// The exact/certified/transactional/cached versions are the S1-S3 research line,
// not this shader. Injected: A3_CAMF A3_CAMGL A3_SUNF A3_SUNGL NOCC_SUN INSTRUMENT
// ============================================================
out vec4 fragColor;
uniform vec2 uRes;
uniform vec3 uEye;
uniform mat3 uCam;

#define PI 3.14159265358979
#define VOL 4.0
#define SIGMA_T 11.0
#define ALBEDO 0.9
#define HG_G 0.55
#define THRESH 0.60
#define DENS 3.2
#define NOCT 4
#define NOCC 2
#define DETAIL_AMP 0.1875

const vec3 SUN = vec3(0.55350, 0.75478, 0.35223);
const vec3 SUNCOL = vec3(1.7, 1.5, 1.25) * 3.0;
const vec3 SKYTOP = vec3(0.28, 0.45, 0.75);
const vec3 SKYHOR = vec3(0.72, 0.80, 0.92);

const float GL2X[2] = float[2](-0.5773502691896257, 0.5773502691896257);
const float GL3X[3] = float[3](-0.7745966692414834, 0.0, 0.7745966692414834);
const float GL3W[3] = float[3](0.5555555555555556, 0.8888888888888888, 0.5555555555555556);
const float GL5X[5] = float[5](-0.906179845938664,-0.538469310105683,0.0,0.538469310105683,0.906179845938664);
const float GL5W[5] = float[5](0.236926885056189,0.478628670499366,0.568888888888889,0.478628670499366,0.236926885056189);

// instrumentation counters (per-pixel, output when INSTRUMENT==1)
float gEvals = 0.0;    // density (extinction) evaluations
float gSunRays = 0.0;  // sun rays cast
float gSkipped = 0.0;  // segments skipped by the empty bound
float gSegs = 0.0;     // total segments visited

float hash13(vec3 p){ p = fract(p*vec3(0.1031,0.1030,0.0973)); p += dot(p,p.yzx+33.33); return fract((p.x+p.y)*p.z); }
float noiseCubic(vec3 p){
    vec3 i=floor(p), f=p-i; vec3 u=f*f*(3.0-2.0*f);
    float n000=hash13(i+vec3(0,0,0)),n100=hash13(i+vec3(1,0,0)),n010=hash13(i+vec3(0,1,0)),n110=hash13(i+vec3(1,1,0));
    float n001=hash13(i+vec3(0,0,1)),n101=hash13(i+vec3(1,0,1)),n011=hash13(i+vec3(0,1,1)),n111=hash13(i+vec3(1,1,1));
    float x00=mix(n000,n100,u.x),x10=mix(n010,n110,u.x),x01=mix(n001,n101,u.x),x11=mix(n011,n111,u.x);
    return mix(mix(x00,x10,u.y),mix(x01,x11,u.y),u.z);
}
float envelope(vec3 p){ vec3 q=(p-vec3(VOL*0.5))/(VOL*0.5);
    return smoothstep(1.05,0.55,length(q.xz))*smoothstep(1.0,0.35,abs(q.y)); }
float rawFbm(vec3 p,int n){ float d=0.0,a=0.5,f=1.0; for(int o=0;o<n;o++){d+=a*noiseCubic(p*f);a*=0.5;f*=2.0;} return d; }
float density(vec3 p){ gEvals+=1.0; return max(rawFbm(p,NOCT)-THRESH,0.0)*envelope(p)*DENS; }
float densitySun(vec3 p){ gEvals+=1.0; return max(rawFbm(p,NOCC_SUN)-THRESH,0.0)*envelope(p)*DENS; }
bool occupied(vec3 p){ if(envelope(p)<0.002) return false; gEvals+=1.0; return rawFbm(p,NOCC)>(THRESH-DETAIL_AMP); }
bool boxHit(vec3 ro,vec3 rd,out float t0,out float t1){
    vec3 inv=1.0/rd,a=(vec3(0.0)-ro)*inv,b=(vec3(VOL)-ro)*inv,tn=min(a,b),tf=max(a,b);
    t0=max(max(tn.x,tn.y),tn.z); t1=min(min(tf.x,tf.y),tf.z); t0=max(t0,0.0); return t1>t0; }
float phaseHG(float c){ float g2=HG_G*HG_G; return (1.0-g2)/(4.0*PI*pow(1.0+g2-2.0*HG_G*c,1.5)); }
vec3 sky(vec3 rd){ float h=clamp(rd.y*0.5+0.5,0.0,1.0); vec3 c=mix(SKYHOR,SKYTOP,h);
    float s=max(dot(rd,SUN),0.0); c+=SUNCOL*0.12*pow(s,220.0)+SUNCOL*0.03*pow(s,8.0); return c; }
vec3 tonemap(vec3 c){ c*=0.55; c=(c*(2.51*c+0.03))/(c*(2.43*c+0.59)+0.14); return pow(clamp(c,0.0,1.0),vec3(1.0/2.2)); }

float glTau(vec3 ro,vec3 rd,float ta,float tb,int order,bool sun){
    float h=0.5*(tb-ta),m=0.5*(tb+ta),s=0.0;
    if(order==2){ if(sun) s=densitySun(ro+rd*(m+h*GL2X[0]))+densitySun(ro+rd*(m+h*GL2X[1]));
                  else s=density(ro+rd*(m+h*GL2X[0]))+density(ro+rd*(m+h*GL2X[1])); }
    else if(order==3){ for(int k=0;k<3;k++) s+=GL3W[k]*(sun?densitySun(ro+rd*(m+h*GL3X[k])):density(ro+rd*(m+h*GL3X[k]))); }
    else { for(int k=0;k<5;k++) s+=GL5W[k]*(sun?densitySun(ro+rd*(m+h*GL5X[k])):density(ro+rd*(m+h*GL5X[k]))); }
    return SIGMA_T*h*s;
}
float sunTau(vec3 x){
    gSunRays+=1.0;
    vec3 ro=x+SUN*1e-3; float t0,t1; if(!boxHit(ro,SUN,t0,t1)) return 0.0; t1=min(t1,VOL*1.5);
    float cs=1.0/float(A3_SUNF); vec3 rd=SUN; vec3 inv=1.0/(rd+(1.0-step(vec3(1e-6),abs(rd)))*1e-6);
    float tau=0.0,t=t0;
    for(int i=0;i<128;i++){
        vec3 p=ro+rd*t; vec3 nb=(floor(p/cs)+max(sign(rd),0.0))*cs; vec3 tn=(nb-p)*inv;
        float tnext=t+min(tn.x,min(tn.y,tn.z))+1e-6; float tb=min(tnext,t1);
        if(tb>t){ vec3 c=ro+rd*(0.5*(t+tb)); if(occupied(c)){ tau+=glTau(ro,rd,t,tb,A3_SUNGL,true); if(tau>8.0) return tau; } }
        if(tnext>=t1) break; t=tnext;
    }
    return tau;
}
vec3 renderA3(vec3 ro,vec3 rd,out float alpha){
    alpha=0.0; float t0,t1; if(!boxHit(ro,rd,t0,t1)) return vec3(0.0);
    float cs=1.0/float(A3_CAMF); vec3 inv=1.0/(rd+(1.0-step(vec3(1e-6),abs(rd)))*1e-6);
    float Tc=1.0; vec3 L=vec3(0.0); float ph=phaseHG(dot(rd,SUN)); float t=t0;
    for(int i=0;i<256;i++){
        vec3 p=ro+rd*t; vec3 nb=(floor(p/cs)+max(sign(rd),0.0))*cs; vec3 tn=(nb-p)*inv;
        float tnext=t+min(tn.x,min(tn.y,tn.z))+1e-6; float tb=min(tnext,t1);
        if(tb>t){ gSegs+=1.0; vec3 c=ro+rd*(0.5*(t+tb));
            if(occupied(c)){
                float dtau=glTau(ro,rd,t,tb,A3_CAMGL,false);
                if(dtau>1e-5){
                    float Ts=exp(-sunTau(c)); float sc=1.0-exp(-dtau);
                    L += Tc*ALBEDO*ph*Ts*sc*SUNCOL + Tc*ALBEDO*0.12*SKYTOP*sc;   // per-segment attenuation
                    Tc *= exp(-dtau);
                }
            } else gSkipped+=1.0;
        }
        if(tnext>=t1 || Tc<0.004) break; t=tnext;
    }
    alpha=1.0-Tc; return L;
}
void main(){
    vec2 uv=(gl_FragCoord.xy-0.5*uRes)/uRes.y;
    vec3 rd=uCam*normalize(vec3(uv,1.3)); vec3 ro=uEye;
    float alpha; vec3 cloud=renderA3(ro,rd,alpha);
#if INSTRUMENT==1
    fragColor=vec4(gEvals,gSunRays,gSkipped,gSegs);
#else
    fragColor=vec4(tonemap(sky(rd)*(1.0-alpha)+cloud),1.0);
#endif
}
