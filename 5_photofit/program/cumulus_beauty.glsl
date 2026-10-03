#version 330 core
// Aesthetic cumulus sky -- physically COHERENT (camera and sun march the same real 3-D density,
// no photo, no gating hacks). Paper-3 atmosphere + volumetric cumulus with multi-scatter octaves,
// silver-lining phase, self-shadowing, sky ambient fill, and aerial perspective.
out vec4 fragColor;
uniform vec2 uRes;
uniform vec3 uEye;
uniform mat3 uCam;
uniform float uFocal;
uniform vec3 uSunDir;
uniform float uExposure;
uniform float uCbase,uCtop,uCovLo,uCovHi,uCovFreq,uDetAmp,uDens,uSigma;
uniform vec3 uZen,uHor;

#define PI 3.14159265358979
// ---- cloud layer + look knobs ----
#define DET_FREQ 1.6
#define THRESH 0.10
#define CAM_STEPS 176
#define SUN_STEPS 8
#define SUNCOL (vec3(1.9,1.75,1.50)*3.4)
#define AMB_STR 1.0

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
// nonlinear value-space remaps -> realistic cloud shape (A3-compatible: abs is C1, needs root partition)
float billowFbm(vec3 p,int oct){ float s=0.0,a=0.5,f=1.0,n=0.0; for(int o=0;o<oct;o++){ s+=a*abs(vnoise(p*f)*2.0-1.0); n+=a; a*=0.5; f*=2.02; } return s/n; }      // lumpy cauliflower
float ridgedFbm(vec3 p,int oct){ float s=0.0,a=0.5,f=1.0,n=0.0; for(int o=0;o<oct;o++){ float r=1.0-abs(vnoise(p*f)*2.0-1.0); s+=a*r*r; n+=a; a*=0.5; f*=2.02; } return s/n; } // sharp ridges
float remap(float x,float a,float b,float c,float d){ return c+(clamp(x,a,b)-a)*(d-c)/max(b-a,1e-4); }

// ================= Paper-3 atmosphere =================
const float planetRadius=6.371, atmRadius=6.471, Hatm=0.008, Csw=1.12;
const vec3 betaR=vec3(5.802,13.558,33.10), betaO=vec3(3.426,8.298,0.356);
const float ozC=0.025, ozW=0.030, ozA=0.30, fDay=0.40;
const float ATM_DENS=1.7, SUN_I=5.5, MS_STR=0.45;
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
vec3 sunTrans(vec3 pos,float r,vec3 L){ float sunOD=chapInf(r,dot(L,pos/r)); float am=sunOD/max(Hatm*exp(-(r-planetRadius)/Hatm),1e-4); return exp(-(betaR*sunOD+betaO*(ozA*ozW)*am)*ATM_DENS); }
vec3 gather(vec3 P,vec3 up,vec3 L){ vec3 sp=L-dot(L,up)*up; vec3 t1=(length(sp)>1e-4)?normalize(sp):normalize(cross(up,vec3(0,0,1))); vec3 t2=cross(up,t1); vec3 J=vec3(0.0);
    for(int ie=0;ie<2;ie++){ float e=0.35+float(ie)*0.65,ce=cos(e),se=sin(e); for(int ia=0;ia<2;ia++){ float a=float(ia)*PI; vec3 d=ce*(cos(a)*t1+sin(a)*t2)+se*up; vec2 hit=raySphere(P,d,atmRadius); float Ld=max(0.0,hit.y)*0.4; vec3 Q=P+d*Ld; float rq=length(Q),hq=rq-planetRadius; if(hq<0.0) continue; float dq=exp(-hq/Hatm); float cph=dot(d,L); J+=betaR*dq*sunTrans(Q,rq,L)*0.75*(1.0+cph*cph)*(Ld+1e-4);} } return J*0.25; }
vec3 atmosphere(vec3 cam,vec3 ray,vec3 L){ vec2 atm=raySphere(cam,ray,atmRadius); if(atm.y<0.0||atm.x>atm.y) return vec3(0.0);
    float tS=max(atm.x,0.0),tE=atm.y; vec2 pl=raySphere(cam,ray,planetRadius); if(pl.x>tS&&pl.x<tE) tE=pl.x;
    float msG=1.0/(1.0-fDay); float c=dot(ray,L),phR=0.75*(1.0+c*c); float ds=(tE-tS)/16.0,vOD=0.0,vOz=0.0; vec3 single=vec3(0.0),multi=vec3(0.0);
    for(int i=0;i<16;i++){ vec3 pos=cam+ray*(tS+(float(i)+0.5)*ds); float r=length(pos),h=r-planetRadius,dens=exp(-h/Hatm),ozo=ozone(h); vOD+=dens*ds; vOz+=ozo*ds; vec3 Tv=exp(-(betaR*vOD+betaO*vOz)*ATM_DENS); vec3 up=pos/r;
        single+=dens*Tv*sunTrans(pos,r,L)*ds; multi+=dens*Tv*gather(pos,up,L)*msG*ds; }
    vec3 col=SUN_I*betaR*(single*phR+multi*MS_STR); if(c>0.9997) col+=vec3(1.0,0.95,0.9)*45.0; return col; }

// ================= clouds =================
float coverage(vec2 xz){ return fbm(vec3(xz*uCovFreq, 3.7), 4); }
float cloudDensity(vec3 p){
    float hf=(p.y-uCbase)/(uCtop-uCbase); if(hf<0.0||hf>1.0) return 0.0;
    float cov=smoothstep(uCovLo,uCovHi, coverage(p.xz));             // broken coverage (where clouds are)
    float prof=smoothstep(0.0,0.10,hf)*smoothstep(1.0,0.42,hf);     // flat base, rounded top
    float base=cov*prof;
    if(base<=0.001) return 0.0;
    // billow = lumpy cumulus body; erode its surface with ridged detail (Nubis remap) -> cauliflower + crisp edges
    float body=billowFbm(p*DET_FREQ + vec3(0.0,2.0,0.0), 5);         // [0,1] lumps
    float shape=base*(0.60 + 0.6*body);
    float er=uDetAmp*mix(0.45,1.0,hf)*ridgedFbm(p*DET_FREQ*2.3 + vec3(5.0,0.0,9.0), 4);  // sharp edge erosion, more at top
    float d=remap(shape, er, 1.0, 0.0, 1.0) - THRESH;
    return clamp(d,0.0,1.0)*uDens;
}
float sunShadow(vec3 p){ float t=0.03,tau=0.0; for(int i=0;i<SUN_STEPS;i++){ float st=0.12*(1.0+0.7*float(i)); tau+=cloudDensity(p+uSunDir*(t))*st; t+=st; if(tau>12.0) break; } return tau; }
float hg(float c,float g){ float g2=g*g; return (1.0-g2)/(4.0*PI*pow(1.0+g2-2.0*g*c,1.5)); }
vec3 tonemap(vec3 c){ c=1.0-exp(-c*uExposure); return pow(clamp(c,0.0,1.0),vec3(1.0/2.2)); }

// hand-tuned analytic clear-sky: deep blue zenith -> pale horizon + Mie sun glow/disk
vec3 skyCol(vec3 rd,vec3 L){
    float h=max(rd.y,0.0);
    vec3 col=mix(uHor,uZen,pow(h,0.52));
    col=mix(col,vec3(0.74,0.82,0.90),exp(-h*9.0)*0.5);           // thin horizon haze
    float mu=max(dot(rd,L),0.0);
    col+=vec3(1.0,0.80,0.52)*0.10*pow(mu,40.0);                   // tight forward glow
    col+=vec3(1.0,0.97,0.90)*8.0*pow(mu,5000.0);                  // sun disk
    return col;
}

void main(){
    vec2 uv=(gl_FragCoord.xy-0.5*uRes)/uRes.y;
    vec3 rd=uCam*normalize(vec3(uv,uFocal)); vec3 ro=uEye; vec3 L=normalize(uSunDir);
    vec3 zenAmb=uZen*1.1+vec3(0.05), horAmb=uHor*1.0;                // sky ambient fill for clouds
    vec3 sunCol=SUNCOL*clamp(L.y*1.4+0.15,0.0,1.6);

    vec3 col;
    if(rd.y < -0.012){                                            // simple lit ground
        float g=clamp(-rd.y*8.0,0.0,1.0);
        col=mix(uHor*1.05, vec3(0.30,0.40,0.20)*(0.55+0.45*max(L.y,0.0)), g);
    } else {
        vec3 sky=skyCol(rd,L); col=sky;
        float t0=(uCbase-ro.y)/rd.y, t1=(uCtop-ro.y)/rd.y; if(t0>t1){ float tmp=t0; t0=t1; t1=tmp; }
        t0=max(t0,0.0); t1=min(t1,120.0);
        if(t1>t0){
            float mu=dot(rd,L);
            float T=1.0; vec3 acc=vec3(0.0); float dt=(t1-t0)/float(CAM_STEPS);
            float jit=hash13(vec3(gl_FragCoord.xy,0.5));
            float t=t0+dt*jit, meanT=0.0, wsum=0.0;
            for(int i=0;i<CAM_STEPS;i++){
                if(t>t1||T<0.02) break;
                vec3 p=ro+rd*t; float d=cloudDensity(p);
                if(d>0.001){
                    float ss=sunShadow(p);
                    float sun=0.0,a=1.0,b=1.0,cc=1.0;                    // 3 octaves, less shadow leak -> real form
                    for(int n=0;n<3;n++){ sun+=a*exp(-b*ss)*mix(hg(mu,0.76*cc),hg(mu,-0.15),0.25); a*=0.42; b*=0.55; cc*=0.8; }
                    float dtau=d*uSigma*dt;
                    float powder=1.0-exp(-d*3.0);
                    float hf=clamp((p.y-uCbase)/(uCtop-uCbase),0.0,1.0);
                    vec3 amb=mix(horAmb*0.7,zenAmb,hf)*0.42*(0.35+0.65*hf);   // darker, bluer undersides
                    vec3 Lc=sunCol*sun*mix(0.4,1.0,powder) + amb*AMB_STR;
                    float ai=1.0-exp(-dtau);
                    acc+=T*ai*Lc; T*=exp(-dtau); meanT+=t*ai; wsum+=ai;
                }
                t+=dt;
            }
            float md=(wsum>0.0)?meanT/wsum:t1; float fog=exp(-max(md-6.0,0.0)*0.010);  // gentle, near clouds stay crisp
            vec3 cloudCol=mix(sky, acc, fog);
            col=sky*T + cloudCol*(1.0-T);
        }
    }
    fragColor=vec4(tonemap(col),1.0);
}
