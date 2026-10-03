#version 330 core
// Uniform-driven cumulus renderer for inverse fitting to a photo. All scene parameters are
// uniforms so the optimizer changes them with zero recompile. Deterministic (no jitter) so the
// loss is smooth. Simple empty-skip march (fitting only needs a faithful, fast forward render).
out vec4 fragColor;
uniform vec2 uRes;
uniform vec3 uEye;
uniform mat3 uCam;
uniform float uFocal;
uniform vec3 uSkyTop, uSkyHor, uSunColV, uGround;
uniform vec3 uSunDir;
uniform float uCBase, uCTop, uCovFreq, uCovLo, uCovHi, uDens, uAmb, uExposure;
uniform vec2 uCovOff;
uniform float uBillowAmp, uBillowFreq, uSigma, uErode;
uniform float uSunIntensity, uAtmDens, uDetailAmp, uDetailFreq;
uniform vec3 uColGain;                     // final white-balance / exposure match
uniform float uCloudLo, uCloudHi;          // cloud radiance gain vs screen cloudiness (photo-guided shading)
uniform float uCovGamma;                   // opacity curve on the screen cloudiness
uniform float uHazeAmp, uHazeH;            // aerosol horizon haze (Mie-like, Paper 3 is Rayleigh only)
uniform sampler2D uCovTex;                 // photo-derived coverage map (world xz), if uUseTex>0.5
uniform vec2 uCovMin, uCovSize;
uniform float uUseTex;
uniform sampler2D uScrCov;                 // screen-space photo cloudiness (placement), if uUseScr>0.5
uniform float uUseScr;
float gScr = 1.0;                          // per-pixel screen cloudiness (set in main)

#define PI 3.14159265358979
#define HG_G 0.5
#define ALBEDO 0.95

// ---- Paper 3: single + multiple scattering + ozone daytime sky (hybrid solver, GPU port) ----
const float planetRadius = 6.371;
const float atmRadius    = 6.471;
const float Hatm         = 0.008;
const float Csw          = 1.12;
const vec3  betaR        = vec3(5.802, 13.558, 33.10);      // Rayleigh scattering /Mm
const vec3  betaO        = vec3(3.426, 8.298, 0.356);       // ozone Chappuis /Mm
const float ozoneCenter  = 0.025, ozoneWidth = 0.030, ozoneAmount = 0.30;
const float fDay = 0.55, msWeight = 1.0;
#ifndef VIEW_STEPS
#define VIEW_STEPS 16
#endif
float erfcx(float y){
    if (y < 1.5){ float y2=y*y,t=y,s=y; for(int n=1;n<10;n++){t*=-y2/float(n); s+=t/float(2*n+1);}
        float erf=1.12837916709551*s; return exp(y2)*(1.0-erf); }
    float f=y; for(int k=12;k>=1;k--) f=y+0.5*float(k)/f; return 0.564189583547756/f;
}
float chapmanPos(float r0,float mu){
    float s=sqrt(max(0.0,1.0-mu*mu)); float X=r0/Hatm; float h0=r0-planetRadius; float eps=1.0/X;
    if(s<1e-4) return Hatm*exp(-h0/Hatm);
    if(1.0/s>Csw){ float m2=mu*mu; float c1=1.0-1.0/m2; float c2=3.0*(1.0-m2)/(m2*m2);
        float c3=3.0*(m2*m2+4.0*m2-5.0)/(m2*m2*m2); float S=(1.0+eps*(c1+eps*(c2+eps*c3)))/mu;
        return Hatm*exp(-h0/Hatm)*S; }
    float z=X*s; float p=r0*s; float eta=sqrt(2.0*X*mu*mu/(1.0+s)); float iz=1.0/z;
    float P=1.0+iz*(0.375+iz*(-0.1171875+iz*(0.1025390625+iz*(-0.144195556640625))));
    float e2=eta*eta;
    float Q=iz*(0.375*eta+iz*(-0.0390625*(e2*eta+3.0*eta)+iz*(0.0068359375*(e2*e2*eta+5.0*e2*eta+15.0*eta)
        +iz*(-0.001373291015625*(e2*e2*e2*eta+7.0*e2*e2*eta+35.0*e2*eta+105.0*eta)))));
    float y=eta*0.707106781186548;
    return sqrt(p*Hatm)*exp(-h0/Hatm)*(P*1.25331413731550*erfcx(y)+Q);
}
float chapmanInf(float r0,float mu){
    if(mu<0.0){ float p=r0*sqrt(max(0.0,1.0-mu*mu)); if(p<planetRadius) return 1e20;
        return 2.0*chapmanPos(p,0.0)-chapmanPos(r0,-mu); }
    return chapmanPos(r0,mu);
}
vec2 raySphere(vec3 ro,vec3 rd,float ra){ float b=dot(ro,rd); float c=dot(ro,ro)-ra*ra; float d=b*b-c;
    if(d<0.0) return vec2(1e9,-1e9); d=sqrt(d); return vec2(-b-d,-b+d); }
float ozoneDensity(float h){ return max(0.0,1.0-abs(h-ozoneCenter)/ozoneWidth)*ozoneAmount; }
vec3 sunTransmittance(vec3 pos,float r,float dens,vec3 L){
    float muSun=dot(L,pos/r); float sunOD=chapmanInf(r,muSun);
    float airmass=sunOD/max(Hatm*dens,1e-4); float sunOz=(ozoneAmount*ozoneWidth)*airmass;
    return exp(-(betaR*sunOD+betaO*sunOz)*uAtmDens);
}
vec3 gatherDiffuse(vec3 P,vec3 up,vec3 L){                   // 2x2 coarse directional gather (MS)
    vec3 sp=L-dot(L,up)*up; vec3 t1=(length(sp)>1e-4)?normalize(sp):normalize(cross(up,vec3(0,0,1)));
    vec3 t2=cross(up,t1); vec3 J=vec3(0.0);
    for(int ie=0;ie<2;ie++){ float e=0.35+float(ie)*0.65; float ce=cos(e),se=sin(e);
        for(int ia=0;ia<2;ia++){ float a=float(ia)*3.14159265; vec3 d=ce*(cos(a)*t1+sin(a)*t2)+se*up;
            vec2 hit=raySphere(P,d,atmRadius); float Ld=max(0.0,hit.y)*0.4; vec3 Q=P+d*Ld;
            float rq=length(Q),hq=rq-planetRadius; if(hq<0.0) continue;
            float densq=exp(-hq/Hatm); vec3 Tsq=sunTransmittance(Q,rq,densq,L);
            float cph=dot(d,L); float ph=0.75*(1.0+cph*cph);
            J+=betaR*densq*Tsq*ph*(Ld+1e-4); } }
    return J*(1.0/4.0);
}
vec3 atmosphere(vec3 cam,vec3 ray,vec3 L){
    vec2 atm=raySphere(cam,ray,atmRadius); if(atm.y<0.0||atm.x>atm.y) return vec3(0.0);
    float tStart=max(atm.x,0.0), tEnd=atm.y; vec2 pl=raySphere(cam,ray,planetRadius);
    if(pl.x>tStart&&pl.x<tEnd) tEnd=pl.x;
    float msGain=msWeight/(1.0-fDay);
    float c=dot(ray,L); float phaseR=0.75*(1.0+c*c);
    float ds=(tEnd-tStart)/float(VIEW_STEPS); float viewOD=0.0,viewOz=0.0;
    vec3 single=vec3(0.0),multi=vec3(0.0);
    for(int i=0;i<VIEW_STEPS;i++){ vec3 pos=cam+ray*(tStart+(float(i)+0.5)*ds);
        float r=length(pos),h=r-planetRadius; float dens=exp(-h/Hatm); float ozo=ozoneDensity(h);
        viewOD+=dens*ds; viewOz+=ozo*ds; vec3 Tview=exp(-(betaR*viewOD+betaO*viewOz)*uAtmDens); vec3 up=pos/r;
        vec3 Tsun=sunTransmittance(pos,r,dens,L); single+=dens*Tview*Tsun*ds;
        vec3 Jdiff=gatherDiffuse(pos,up,L); multi+=dens*Tview*Jdiff*msGain*ds; }
    vec3 col=uSunIntensity*betaR*(single*phaseR+multi);
    if(c>0.9997) col+=vec3(1.0,0.95,0.9)*45.0;
    return col;
}

float hash13(vec3 p){ p=fract(p*vec3(0.1031,0.1030,0.0973)); p+=dot(p,p.yzx+33.33); return fract((p.x+p.y)*p.z); }
float noiseCubic(vec3 p){
    vec3 i=floor(p), f=p-i; vec3 u=f*f*(3.0-2.0*f);
    float n000=hash13(i+vec3(0,0,0)),n100=hash13(i+vec3(1,0,0)),n010=hash13(i+vec3(0,1,0)),n110=hash13(i+vec3(1,1,0));
    float n001=hash13(i+vec3(0,0,1)),n101=hash13(i+vec3(1,0,1)),n011=hash13(i+vec3(0,1,1)),n111=hash13(i+vec3(1,1,1));
    float x00=mix(n000,n100,u.x),x10=mix(n010,n110,u.x),x01=mix(n001,n101,u.x),x11=mix(n011,n111,u.x);
    return mix(mix(x00,x10,u.y),mix(x01,x11,u.y),u.z);
}
float fbm(vec3 p,int n){ float d=0.0,a=0.5,f=1.0; for(int o=0;o<n;o++){d+=a*noiseCubic(p*f);a*=0.5;f*=2.0;} return d; }
float coverage(vec3 p,int oct){
    if(uUseTex>0.5){                          // photo-guided placement (+ a little procedural break-up)
        vec2 uv=(p.xz-uCovMin)/uCovSize;
        if(uv.x<0.0||uv.x>1.0||uv.y<0.0||uv.y>1.0) return 0.0;
        float t=texture(uCovTex,uv).r;
        return t + 0.10*(fbm(vec3(p.xz*uCovFreq+uCovOff,7.0),oct)-0.5);
    }
    return fbm(vec3(p.xz*uCovFreq + uCovOff, 7.0), oct);
}
float cloudTop(float cov){ return uCBase + 0.30 + (uCTop-uCBase-0.30)*smoothstep(0.50,0.95,cov); }

float cumulus(vec3 p){
    float y=p.y;
    if(y<uCBase-0.12 || y>uCTop+0.2) return 0.0;
    float cov=coverage(p,4);
    float top=cloudTop(cov);
    float mask=(uUseScr>0.5) ? 1.0 : smoothstep(uCovLo,uCovHi,cov);  // screen mask fills the silhouette
    float prof=smoothstep(uCBase-0.02,uCBase+0.10,y)*smoothstep(top,top-0.45,y);
    float billow=fbm(p*vec3(uBillowFreq,uBillowFreq*1.1,uBillowFreq)+vec3(0.0,4.0,0.0),6);
    float d=mask*prof*(0.22+billow*uBillowAmp) - uErode;
    float detail=fbm(p*uDetailFreq + vec3(11.0,0.0,7.0), 4);   // high-freq erosion -> crisp wispy edges
    d -= uDetailAmp*detail*smoothstep(0.0,0.35,d);
    return max(d,0.0)*uDens;
}
float presence(vec3 p){
    float y=p.y; if(y<uCBase-0.15||y>uCTop+0.25) return 0.0;
    float cov=coverage(p,3);
    float mask=smoothstep(uCovLo-0.10,uCovHi-0.04,cov);
    float top=cloudTop(cov)+0.15;
    return mask*smoothstep(uCBase-0.15,uCBase+0.05,y)*smoothstep(top+0.2,top-0.5,y);
}
float phaseHG(float c){ float g2=HG_G*HG_G; return (1.0-g2)/(4.0*PI*pow(1.0+g2-2.0*HG_G*c,1.5)); }
vec3 lightCloud(float sunT,float mu,float dtau,float y){
    float ph=phaseHG(mu); vec3 acc=vec3(0.0); float a=1.0,b=1.0;
    for(int n=0;n<3;n++){ acc+=a*ph*exp(-b*sunT)*uSunColV; a*=0.5; b*=0.4; }
    float powder=1.0-exp(-dtau*2.5); acc*=mix(0.6,1.0,powder);
    vec3 amb=mix(uSkyHor,uSkyTop,clamp((y-uCBase)/max(uCTop-uCBase,0.1),0.0,1.0))*uAmb+vec3(0.14,0.16,0.19);
    return ALBEDO*(acc+amb);
}
vec3 background(vec3 rd){
    vec3 cam=vec3(0.0, planetRadius+0.0015, 0.0);          // ground observer, local up = +y
    if(rd.y < -0.004) return uGround;                       // ground (masked out of the loss)
    vec3 c = atmosphere(cam, rd, normalize(uSunDir));       // physical Paper-3 sky
    c += uHazeAmp*vec3(1.0,0.97,0.92)*smoothstep(uHazeH, -0.02, rd.y);   // aerosol horizon haze
    return c;
}
vec3 tonemap(vec3 c){ c*=uExposure; c=(c*(2.51*c+0.03))/(c*(2.43*c+0.59)+0.14); return pow(clamp(c,0.0,1.0),vec3(1.0/2.2)); }
bool slabHit(vec3 ro,vec3 rd,out float t0,out float t1){
    if(abs(rd.y)<1e-4){ t0=0.0; t1=(ro.y>uCBase-0.2&&ro.y<uCTop+0.3)?200.0:-1.0; return t1>0.0; }
    float a=(uCBase-0.2-ro.y)/rd.y,b=(uCTop+0.3-ro.y)/rd.y; t0=min(a,b); t1=max(a,b);
    t0=max(t0,0.0); t1=min(t1,220.0); return t1>t0;
}
float sunTau(vec3 x){
    float t0,t1; if(!slabHit(x,uSunDir,t0,t1)) return 0.0; t1=min(t1-t0,8.0);
    float tau=0.0,t=0.02,st=t1/6.0;
    for(int i=0;i<6;i++){ vec3 p=x+uSunDir*t; if(presence(p)>0.02) tau+=cumulus(p)*uSigma*st; if(tau>7.0) break; t+=st*(1.0+0.3*float(i)); }
    return tau;
}
vec3 renderMarch(vec3 ro,vec3 rd,out float alpha){
    alpha=0.0; float t0,t1; if(!slabHit(ro,rd,t0,t1)) return vec3(0.0);
    float t=t0,T=1.0; vec3 L=vec3(0.0);
    for(int i=0;i<900;i++){
        if(t>t1||T<0.01) break;
        vec3 p=ro+rd*t;
        float gs=(uUseScr>0.5)?pow(clamp(gScr,0.0,1.0),uCovGamma):1.0;
        if(presence(p)*gs<=0.02){ t+=0.28; continue; }
        float dens=cumulus(p)*gs;
        if(dens>1e-3){ float dtau=dens*uSigma*0.06; vec3 S=lightCloud(sunTau(p),dot(rd,uSunDir),dtau,p.y);
            float cg=(uUseScr>0.5)?mix(uCloudLo,uCloudHi,gScr):1.0;   // brighter where the photo cloud is brighter
            L+=T*S*cg*(1.0-exp(-dtau)); T*=exp(-dtau); }
        t+=0.06;
    }
    alpha=1.0-T; return L;
}
void main(){
    vec2 uv=(gl_FragCoord.xy-0.5*uRes)/uRes.y;
    gScr = (uUseScr>0.5) ? texture(uScrCov, gl_FragCoord.xy/uRes).r : 1.0;
    vec3 rd=uCam*normalize(vec3(uv,uFocal)); vec3 ro=uEye;
    float alpha; vec3 cloud=renderMarch(ro,rd,alpha);
    vec3 outc=clamp(tonemap(background(rd)*(1.0-alpha)+cloud)*uColGain, 0.0, 1.0);
    fragColor=vec4(outc,1.0);
}
