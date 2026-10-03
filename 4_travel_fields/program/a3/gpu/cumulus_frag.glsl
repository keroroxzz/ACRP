#version 330 core
// ============================================================
// Fair-weather cumulus field (matches the reference photo): a single BROKEN low
// cumulus layer with FLAT bases at a constant condensation level, billowy puffy
// tops, blue gaps between clouds, appearing denser toward the horizon (perspective
// through one layer). Deep-blue zenith -> pale horizon; brilliant white sunlit tops.
// Shared density; METHOD 0=A3 uniform, 1=AAA raymarch, 2=A3 hierarchical (two-level).
// ============================================================
out vec4 fragColor;
uniform vec2 uRes;
uniform vec3 uEye;
uniform mat3 uCam;

#define PI 3.14159265358979
#define SIGMA_T 8.0
#define ALBEDO 0.95
#define HG_G 0.5
#define CBASE 1.30          // flat cloud base (condensation level)
#define CTOP  3.20          // max cumulus top
#define DENS 3.0
#ifndef SMALL_STEP
#define SMALL_STEP 0.05
#endif
#ifndef BIG_STEP
#define BIG_STEP 0.28
#endif
#ifndef MAXI
#define MAXI 1400
#endif
#ifndef JITTER_ON
#define JITTER_ON 1
#endif
#ifndef SUN_JITTER
#define SUN_JITTER 1.0
#endif

const vec3 SUN = vec3(0.28, 0.66, 0.70);            // high afternoon sun, slightly behind
const vec3 SUNCOL = vec3(1.75, 1.6, 1.35) * 4.0;
const vec3 SKYTOP = vec3(0.16, 0.36, 0.72);         // deep blue zenith
const vec3 SKYHOR = vec3(0.72, 0.80, 0.90);         // pale horizon haze
const vec3 GRND   = vec3(0.33, 0.42, 0.20);         // green field

const float GL2X[2] = float[2](-0.5773502691896257, 0.5773502691896257);
const float GL3X[3] = float[3](-0.7745966692414834, 0.0, 0.7745966692414834);
const float GL3W[3] = float[3](0.5555555555555556, 0.8888888888888888, 0.5555555555555556);
const float GL5X[5] = float[5](-0.906179845938664,-0.538469310105683,0.0,0.538469310105683,0.906179845938664);
const float GL5W[5] = float[5](0.236926885056189,0.478628670499366,0.568888888888889,0.478628670499366,0.236926885056189);

float gJit = 0.0;
float gEvals=0.0, gSunRays=0.0, gSkipped=0.0, gSegs=0.0;

float hash13(vec3 p){ p=fract(p*vec3(0.1031,0.1030,0.0973)); p+=dot(p,p.yzx+33.33); return fract((p.x+p.y)*p.z); }
float noiseCubic(vec3 p){
    vec3 i=floor(p), f=p-i; vec3 u=f*f*(3.0-2.0*f);
    float n000=hash13(i+vec3(0,0,0)),n100=hash13(i+vec3(1,0,0)),n010=hash13(i+vec3(0,1,0)),n110=hash13(i+vec3(1,1,0));
    float n001=hash13(i+vec3(0,0,1)),n101=hash13(i+vec3(1,0,1)),n011=hash13(i+vec3(0,1,1)),n111=hash13(i+vec3(1,1,1));
    float x00=mix(n000,n100,u.x),x10=mix(n010,n110,u.x),x01=mix(n001,n101,u.x),x11=mix(n011,n111,u.x);
    return mix(mix(x00,x10,u.y),mix(x01,x11,u.y),u.z);
}
float fbm(vec3 p,int n){ float d=0.0,a=0.5,f=1.0; for(int o=0;o<n;o++){d+=a*noiseCubic(p*f);a*=0.5;f*=2.0;} return d; }

// coverage: low-freq 2D field that decides where cumulus sit (broken -> blue gaps)
float coverage(vec3 p,int oct){ return fbm(vec3(p.xz*0.46, 7.0), oct); }
// per-cloud top height (taller where coverage stronger)
float cloudTop(float cov){ return CBASE + 0.30 + (CTOP-CBASE-0.30)*smoothstep(0.50,0.95,cov); }

float cumulus(vec3 p){
    gEvals+=1.0;
    float y=p.y;
    if(y < CBASE-0.12 || y > CTOP+0.2) return 0.0;
    float cov = coverage(p,4);
    float top = cloudTop(cov);
    float mask = smoothstep(0.52, 0.66, cov);                       // sharper -> distinct, separated puffs
    float prof = smoothstep(CBASE-0.02, CBASE+0.10, y)              // FLAT base
               * smoothstep(top, top-0.45, y);                      // rounded top
    float billow = fbm(p*vec3(2.4,2.6,2.4) + vec3(0.0,4.0,0.0), 6); // crisper cauliflower erosion
    float d = mask*prof*(0.22+billow*1.4);
    return max(d-0.22, 0.0)*DENS;
}

// coarse conservative presence for empty-space skipping (>0 wherever cumulus>0)
float presence(vec3 p){
    gEvals+=1.0;
    float y=p.y;
    if(y < CBASE-0.15 || y > CTOP+0.25) return 0.0;
    float cov = coverage(p,3);                                      // cheaper, generous threshold
    float mask = smoothstep(0.42, 0.62, cov);
    float top = cloudTop(cov)+0.15;
    float prof = smoothstep(CBASE-0.15, CBASE+0.05, y)*smoothstep(top+0.2, top-0.5, y);
    return mask*prof;
}

float phaseHG(float c){ float g2=HG_G*HG_G; return (1.0-g2)/(4.0*PI*pow(1.0+g2-2.0*HG_G*c,1.5)); }
vec3 lightCloud(float sunT, float mu, float dtau, float y){
    float ph=phaseHG(mu); vec3 acc=vec3(0.0); float a=1.0,b=1.0;
    for(int n=0;n<3;n++){ acc += a*ph*exp(-b*sunT)*SUNCOL; a*=0.5; b*=0.4; }
    float powder=1.0-exp(-dtau*2.5); acc*=mix(0.6,1.0,powder);
    vec3 amb = mix(SKYHOR,SKYTOP, clamp((y-CBASE)/(CTOP-CBASE),0.0,1.0))*0.7 + vec3(0.14,0.16,0.19);
    return ALBEDO*(acc+amb);
}
vec3 background(vec3 rd){
    if(rd.y < 0.0){                                                // simple ground + horizon haze
        float h = clamp(-rd.y*6.0, 0.0, 1.0);
        return mix(SKYHOR*0.92, GRND, h);
    }
    float h=clamp(rd.y,0.0,1.0); vec3 c=mix(SKYHOR,SKYTOP,pow(h,0.55));
    float s=max(dot(rd,SUN),0.0); c+=SUNCOL*0.04*pow(s,900.0)+SUNCOL*0.015*pow(s,6.0);
    return c;
}
vec3 tonemap(vec3 c){ c*=0.62; c=(c*(2.51*c+0.03))/(c*(2.43*c+0.59)+0.14); return pow(clamp(c,0.0,1.0),vec3(1.0/2.2)); }

bool slabHit(vec3 ro,vec3 rd,out float t0,out float t1){
    if(abs(rd.y)<1e-4){ t0=0.0; t1=(ro.y>CBASE-0.2&&ro.y<CTOP+0.3)?200.0:-1.0; return t1>0.0; }
    float a=(CBASE-0.2-ro.y)/rd.y, b=(CTOP+0.3-ro.y)/rd.y; t0=min(a,b); t1=max(a,b);
    t0=max(t0,0.0); t1=min(t1,220.0); return t1>t0;
}
float sunTau(vec3 x){
    gSunRays+=1.0;
    float t0,t1; if(!slabHit(x,SUN,t0,t1)) return 0.0; t1=min(t1-t0,8.0);
    float tau=0.0, t=0.02, st=t1/float(SUN_STEPS);
    for(int i=0;i<SUN_STEPS;i++){
        vec3 p=x+SUN*t;
        if(presence(p)>0.02) tau += cumulus(p)*SIGMA_T*st;
        if(tau>7.0) break; t += st*(1.0+0.3*float(i));
    }
    return tau;
}

vec3 renderMarch(vec3 ro,vec3 rd,out float alpha){
    alpha=0.0; float t0,t1; if(!slabHit(ro,rd,t0,t1)) return vec3(0.0);
    float t=t0+gJit*SMALL_STEP; float T=1.0; vec3 L=vec3(0.0);
    for(int i=0;i<MAXI;i++){
        if(t>t1||T<0.01) break;
        vec3 p=ro+rd*t; gSegs+=1.0;
        if(presence(p)<=0.02){ gSkipped+=1.0; t+=BIG_STEP; continue; }
        float dens=cumulus(p);
        if(dens>1e-3){
            float dtau=dens*SIGMA_T*SMALL_STEP;
            vec3 S=lightCloud(sunTau(p),dot(rd,SUN),dtau,p.y);
            L+=T*S*(1.0-exp(-dtau)); T*=exp(-dtau);
        }
        t+=SMALL_STEP;
    }
    alpha=1.0-T; return L;
}
float glTau(vec3 ro,vec3 rd,float ta,float tb,int order){
    float h=0.5*(tb-ta),m=0.5*(tb+ta),s=0.0;
    if(order==2){ for(int k=0;k<2;k++) s+=cumulus(ro+rd*(m+h*GL2X[k])); }
    else if(order==3){ for(int k=0;k<3;k++) s+=GL3W[k]*cumulus(ro+rd*(m+h*GL3X[k])); }
    else { for(int k=0;k<5;k++) s+=GL5W[k]*cumulus(ro+rd*(m+h*GL5X[k])); }
    return SIGMA_T*h*s;
}
bool occ(vec3 p){ return presence(p)>0.02; }
vec3 renderA3(vec3 ro,vec3 rd,out float alpha){
    alpha=0.0; float t0,t1; if(!slabHit(ro,rd,t0,t1)) return vec3(0.0);
    float cs=1.0/float(A3_CAMF); vec3 inv=1.0/(rd+(1.0-step(vec3(1e-6),abs(rd)))*1e-6);
    float Tc=1.0; vec3 L=vec3(0.0); float t=t0;
    for(int i=0;i<MAXI;i++){
        vec3 p=ro+rd*t; vec3 nb=(floor(p/cs)+max(sign(rd),0.0))*cs; vec3 tn=(nb-p)*inv;
        float tnext=t+min(tn.x,min(tn.y,tn.z))+1e-5; float tb=min(tnext,t1);
        if(tb>t){ gSegs+=1.0; vec3 c=ro+rd*(0.5*(t+tb));
            if(occ(c)){ float dtau=glTau(ro,rd,t,tb,A3_CAMGL);
                if(dtau>1e-4){ float jf=0.5+SUN_JITTER*(gJit-0.5); vec3 xc=ro+rd*mix(t,tb,jf);
                    vec3 S=lightCloud(sunTau(xc),dot(rd,SUN),dtau,xc.y);
                    L+=Tc*S*(1.0-exp(-dtau)); Tc*=exp(-dtau); } }
        }
        if(tnext>=t1||Tc<0.01) break; t=tnext;
    }
    alpha=1.0-Tc; return L;
}
vec3 renderA3H(vec3 ro,vec3 rd,out float alpha){
    alpha=0.0; float t0,t1; if(!slabHit(ro,rd,t0,t1)) return vec3(0.0);
    float cs=1.0/float(A3_CAMF); vec3 inv=1.0/(rd+(1.0-step(vec3(1e-6),abs(rd)))*1e-6);
    float Tc=1.0; vec3 L=vec3(0.0); float t=t0+gJit*BIG_STEP;
    for(int i=0;i<MAXI;i++){
        if(t>t1||Tc<0.01) break;
        vec3 p=ro+rd*t;
        if(presence(p)<=0.02){ gSkipped+=1.0; t+=BIG_STEP; continue; }
        gSegs+=1.0;
        vec3 nb=(floor(p/cs)+max(sign(rd),0.0))*cs; vec3 tn=(nb-p)*inv;
        float adv=clamp(min(tn.x,min(tn.y,tn.z)),0.02,cs*2.0);
        float tnext=t+adv+1e-5; float tb=min(tnext,t1);
        vec3 c=ro+rd*(0.5*(t+tb));
        if(occ(c)){ float dtau=glTau(ro,rd,t,tb,A3_CAMGL);
            if(dtau>1e-4){ float jf=0.5+SUN_JITTER*(gJit-0.5); vec3 xc=ro+rd*mix(t,tb,jf);
                vec3 S=lightCloud(sunTau(xc),dot(rd,SUN),dtau,xc.y);
                L+=Tc*S*(1.0-exp(-dtau)); Tc*=exp(-dtau); } }
        t=tnext;
    }
    alpha=1.0-Tc; return L;
}
void main(){
    vec2 uv=(gl_FragCoord.xy-0.5*uRes)/uRes.y;
    gJit=float(JITTER_ON)*hash13(vec3(gl_FragCoord.xy,0.5));
    vec3 rd=uCam*normalize(vec3(uv,0.95)); vec3 ro=uEye;   // wider FOV to match the photo
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
    fragColor=vec4(tonemap(background(rd)*(1.0-alpha)+cloud),1.0);
#endif
}
