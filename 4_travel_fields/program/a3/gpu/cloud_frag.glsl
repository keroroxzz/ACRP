#version 330 core
out vec4 fragColor;
uniform vec2 uRes;
uniform vec3 uEye;
uniform mat3 uCam;   // columns r,u,f

// Injected #defines from the harness:
//   METHOD  0 = fixed-step raymarch, 1 = A3 (DDA + per-cell GL + analytic sun)
//   CAM_STEPS, SUN_STEPS        (raymarch)
//   A3_CAMF, A3_CAMGL, A3_SUNF, A3_SUNGL   (A3 cell freq + GL order)
//   NOCT

#define PI 3.14159265358979
#define VOL 4.0
#define SIGMA_T 11.0
#define ALBEDO 0.9
#define HG_G 0.55
#define THRESH 0.60
#define DENS 3.2
#define EDGE 0.16    // coverage smoothstep width (small = sharp, high-contrast edges)

const vec3 SUN = vec3(0.55350, 0.75478, 0.35223);
const vec3 SUNCOL = vec3(1.7, 1.5, 1.25) * 3.0;
const vec3 SKYTOP = vec3(0.28, 0.45, 0.75);
const vec3 SKYHOR = vec3(0.72, 0.80, 0.92);

const float GL2X[2] = float[2](-0.5773502691896257, 0.5773502691896257);
const float GL3X[3] = float[3](-0.7745966692414834, 0.0, 0.7745966692414834);
const float GL3W[3] = float[3](0.5555555555555556, 0.8888888888888888, 0.5555555555555556);
const float GL5X[5] = float[5](-0.906179845938664,-0.538469310105683,0.0,0.538469310105683,0.906179845938664);
const float GL5W[5] = float[5](0.236926885056189,0.478628670499366,0.568888888888889,0.478628670499366,0.236926885056189);
const float GL8X[8] = float[8](-0.9602898564975363,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956498,0.5255324099163290,0.7966664774136267,0.9602898564975363);
const float GL8W[8] = float[8](0.1012285362903763,0.2223810344533745,0.3137066458778873,0.3626837833783620,0.3626837833783620,0.3137066458778873,0.2223810344533745,0.1012285362903763);

float hash13(vec3 p){
    p = fract(p * vec3(0.1031,0.1030,0.0973));
    p += dot(p, p.yzx + 33.33);
    return fract((p.x+p.y)*p.z);
}
float noiseCubic(vec3 p){
    vec3 i = floor(p); vec3 f = p - i;
    vec3 u = f*f*(3.0-2.0*f);
    float n000=hash13(i+vec3(0,0,0)), n100=hash13(i+vec3(1,0,0));
    float n010=hash13(i+vec3(0,1,0)), n110=hash13(i+vec3(1,1,0));
    float n001=hash13(i+vec3(0,0,1)), n101=hash13(i+vec3(1,0,1));
    float n011=hash13(i+vec3(0,1,1)), n111=hash13(i+vec3(1,1,1));
    float x00=mix(n000,n100,u.x), x10=mix(n010,n110,u.x);
    float x01=mix(n001,n101,u.x), x11=mix(n011,n111,u.x);
    return mix(mix(x00,x10,u.y), mix(x01,x11,u.y), u.z);
}
float envelope(vec3 p){
    vec3 q = (p - vec3(VOL*0.5))/(VOL*0.5);
    float r = smoothstep(1.05,0.55,length(q.xz));
    float v = smoothstep(1.0,0.35,abs(q.y));
    return r*v;
}
float rawFbm(vec3 p, int n){
    float d=0.0, a=0.5, f=1.0;
    for(int o=0;o<n;o++){ d += a*noiseCubic(p*f); a*=0.5; f*=2.0; }
    return d;
}
float density(vec3 p){
#if SHAPED==1
    float d = smoothstep(THRESH, THRESH+EDGE, rawFbm(p, NOCT));
#else
    float d = max(rawFbm(p, NOCT) - THRESH, 0.0);   // polynomial (ReLU) -- A3 GL exact per piece
#endif
    return d*envelope(p)*DENS;
}
float densitySun(vec3 p){
#if SHAPED==1
    float d = smoothstep(THRESH, THRESH+EDGE, rawFbm(p, NOCC_SUN) + SUN_BIAS);
#else
    float d = max(rawFbm(p, NOCC_SUN) + SUN_BIAS - THRESH, 0.0);
#endif
    return d*envelope(p)*DENS;
}
// Conservative cheap empty-space test: uses only NOCC backbone octaves. Adding the
// remaining detail octaves can raise the fbm by at most DETAIL_AMP, so if the backbone
// fbm is below THRESH-DETAIL_AMP the full density is provably 0 -> safe to skip.
bool occupied(vec3 p){
    if(envelope(p) < 0.002) return false;
    return rawFbm(p, NOCC) > (THRESH - DETAIL_AMP);
}
bool boxHit(vec3 ro, vec3 rd, out float t0, out float t1){
    vec3 inv=1.0/rd; vec3 a=(vec3(0.0)-ro)*inv; vec3 b=(vec3(VOL)-ro)*inv;
    vec3 tn=min(a,b), tf=max(a,b);
    t0=max(max(tn.x,tn.y),tn.z); t1=min(min(tf.x,tf.y),tf.z); t0=max(t0,0.0);
    return t1>t0;
}
float phaseHG(float c){ float g2=HG_G*HG_G; return (1.0-g2)/(4.0*PI*pow(1.0+g2-2.0*HG_G*c,1.5)); }
vec3 sky(vec3 rd){
    float h=clamp(rd.y*0.5+0.5,0.0,1.0);
    vec3 c=mix(SKYHOR,SKYTOP,h);
    float s=max(dot(rd,SUN),0.0);
    c += SUNCOL*0.12*pow(s,220.0); c += SUNCOL*0.03*pow(s,8.0);
    return c;
}
vec3 tonemap(vec3 c){
    c*=0.55; c=(c*(2.51*c+0.03))/(c*(2.43*c+0.59)+0.14);
    return pow(clamp(c,0.0,1.0), vec3(1.0/2.2));
}

// ---- fixed-step sun (raymarch) ----
float sunTauMarch(vec3 x){
    float t0,t1; if(!boxHit(x+SUN*1e-3,SUN,t0,t1)) return 0.0;
    float span=min(t1,VOL*1.5); float dt=span/float(SUN_STEPS);
    float tau=0.0, t=0.5*dt;
    for(int i=0;i<SUN_STEPS;i++){ if(t>span)break; tau+=SIGMA_T*density(x+SUN*t)*dt; if(tau>8.0)break; t+=dt; }
    return tau;
}
vec3 renderMarch(vec3 ro, vec3 rd, out float alpha){
    alpha=0.0; float t0,t1; if(!boxHit(ro,rd,t0,t1)) return vec3(0.0);
    float dt=(t1-t0)/float(CAM_STEPS);
    float Tc=1.0; vec3 L=vec3(0.0); float ph=phaseHG(dot(rd,SUN));
    float t=t0+0.5*dt;
    for(int i=0;i<CAM_STEPS;i++){
        vec3 p=ro+rd*t; float dtau=SIGMA_T*density(p)*dt;
        if(dtau>1e-5){
            float Ts=exp(-sunTauMarch(p)); float sc=1.0-exp(-dtau);
            L += Tc*ALBEDO*ph*Ts*sc*SUNCOL + Tc*ALBEDO*0.12*SKYTOP*sc;
            Tc *= exp(-dtau);
        }
        if(Tc<0.004) break;
        t+=dt;
    }
    alpha=1.0-Tc; return L;
}

// ---- A3: DDA + empty-skip + per-cell GL ----
float glTauCam(vec3 ro, vec3 rd, float ta, float tb){
    float h=0.5*(tb-ta), m=0.5*(tb+ta); float s=0.0;
#if SHAPED==1 && A3_ADAPT==1
    // Tier 1 segment classification (cheap): midpoint raw fbm +/- a conservative in-cell
    // variation margin. Cloud-shaping smoothstep is 0 below THRESH and 1 above THRESH+EDGE,
    // so a cell whose whole raw-fbm range is on one side needs no expensive integration.
    float fm = rawFbm(ro + rd * m, NOCT);
    if (fm + CLASS_MARG <= THRESH) return 0.0;                      // empty (density 0)
    if (fm - CLASS_MARG >= THRESH + EDGE)                           // saturated (smoothstep=1)
        return SIGMA_T * envelope(ro + rd * m) * DENS * (tb - ta);
    // Tier 2: crossing the transition band -> high-order GL on the shaped (degree-27) density
#if A3_TRGL==8
    for (int k = 0; k < 8; k++) s += GL8W[k] * density(ro + rd * (m + h * GL8X[k]));
#else
    for (int k = 0; k < 5; k++) s += GL5W[k] * density(ro + rd * (m + h * GL5X[k]));
#endif
    return SIGMA_T * h * s;
#else
    #if A3_CAMGL==2
    s = density(ro+rd*(m+h*GL2X[0])) + density(ro+rd*(m+h*GL2X[1]));
    #elif A3_CAMGL==3
    for(int k=0;k<3;k++) s += GL3W[k]*density(ro+rd*(m+h*GL3X[k]));
    #else
    for(int k=0;k<5;k++) s += GL5W[k]*density(ro+rd*(m+h*GL5X[k]));
    #endif
    return SIGMA_T*h*s;
#endif
}
float glTauSun(vec3 ro, vec3 rd, float ta, float tb){
    float h=0.5*(tb-ta), m=0.5*(tb+ta); float s=0.0;
#if A3_SUNGL==2
    s = densitySun(ro+rd*(m+h*GL2X[0])) + densitySun(ro+rd*(m+h*GL2X[1]));
#elif A3_SUNGL==3
    for(int k=0;k<3;k++) s += GL3W[k]*densitySun(ro+rd*(m+h*GL3X[k]));
#else
    for(int k=0;k<5;k++) s += GL5W[k]*densitySun(ro+rd*(m+h*GL5X[k]));
#endif
    return SIGMA_T*h*s;
}
float sunTauA3(vec3 x){
    vec3 ro=x+SUN*1e-3; float t0,t1; if(!boxHit(ro,SUN,t0,t1)) return 0.0;
    t1=min(t1,VOL*1.5);
    float cs=1.0/float(A3_SUNF); vec3 rd=SUN;
    vec3 inv=1.0/(rd+ (1.0-step(vec3(1e-6),abs(rd)))*1e-6);
    float tau=0.0, t=t0;
    for(int i=0;i<128;i++){
        vec3 p=ro+rd*t;
        vec3 nb=(floor(p/cs)+max(sign(rd),0.0))*cs;
        vec3 tn=(nb-p)*inv;
        float tnext=t+min(tn.x,min(tn.y,tn.z))+1e-6;
        float tb=min(tnext,t1);
        if(tb>t){
            vec3 c=ro+rd*(0.5*(t+tb));
            if(occupied(c)){ tau+=glTauSun(ro,rd,t,tb); if(tau>8.0) return tau; }
        }
        if(tnext>=t1) break;
        t=tnext;
    }
    return tau;
}
vec3 renderA3(vec3 ro, vec3 rd, out float alpha){
    alpha=0.0; float t0,t1; if(!boxHit(ro,rd,t0,t1)) return vec3(0.0);
    float cs=1.0/float(A3_CAMF);
    vec3 inv=1.0/(rd+ (1.0-step(vec3(1e-6),abs(rd)))*1e-6);
    float Tc=1.0; vec3 L=vec3(0.0); float ph=phaseHG(dot(rd,SUN));
    float t=t0;
    for(int i=0;i<256;i++){
        vec3 p=ro+rd*t;
        vec3 nb=(floor(p/cs)+max(sign(rd),0.0))*cs;
        vec3 tn=(nb-p)*inv;
        float tnext=t+min(tn.x,min(tn.y,tn.z))+1e-6;
        float tb=min(tnext,t1);
        if(tb>t){
            vec3 c=ro+rd*(0.5*(t+tb));
            if(occupied(c)){
                float dtau=glTauCam(ro,rd,t,tb);
                if(dtau>1e-5){
                    float Ts=exp(-sunTauA3(c)); float sc=1.0-exp(-dtau);
                    L += Tc*ALBEDO*ph*Ts*sc*SUNCOL + Tc*ALBEDO*0.12*SKYTOP*sc;
                    Tc *= exp(-dtau);
                }
            }
        }
        if(tnext>=t1 || Tc<0.004) break;
        t=tnext;
    }
    alpha=1.0-Tc; return L;
}

void main(){
    vec2 uv = (gl_FragCoord.xy - 0.5*uRes)/uRes.y;
    vec3 rd = uCam * normalize(vec3(uv, 1.3));
    vec3 ro = uEye;
    float alpha; vec3 cloud;
#if METHOD==1
    cloud = renderA3(ro, rd, alpha);
#else
    cloud = renderMarch(ro, rd, alpha);
#endif
    vec3 col = tonemap(sky(rd)*(1.0-alpha) + cloud);
    fragColor = vec4(col, 1.0);
}
