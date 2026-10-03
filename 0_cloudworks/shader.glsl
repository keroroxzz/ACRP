/*===================================================================================
GTA SA - RenderHook 
RTU's Preset - Beta
by Brian Tu

See my Github page for more information:
https://github.com/keroroxzz

2021/5/30

This is the AtmScattering shader from vanilla RenderHook, with the new sky added by
me, most of the sky calculations are replaced by the new one.

Caution : Beta version is not a public version, this shouldn't appear else where.
===================================================================================*/


#define static /**/
#define inline /**/
#define float3 vec3
#define float2 vec2
#define float4 vec4
#define lerp mix
#define frac fract
#define atan2 atan
const float3 vGradingColor0=float3(0.0,0.0,0.0);
const float3 vGradingColor1=float3(0.7,0.5,0.7);
const float3 vHorizonCol=float3(1.0,1.0,1.0);
const float fFogStart=0.0;
const float CloudStartHeight=0.0;
const float CloudEndHeight=0.0;

float3 vSunLightDir=float3(0.0,0.0,1.0);

float saturate(float v)
{
    return min(max(v,0.0),1.0);
}

float3 saturate(float3 v)
{
    return float3(saturate(v.x),saturate(v.y),saturate(v.z));
}

/*===================================================================================
GTA SA - RenderHook 
RTU's Preset - Beta

2021/5/30

This is the NoiseGenerator shader, the noise functions is not my work, you can find it
on Shadertoy by Inigo Quilez.

Caution : Beta version is not a public version, this shouldn't appear else where.
===================================================================================*/

#ifndef NOISE_GEN
#define NOISE_GEN

static float randomSeed = 1618.03398875;

float hash(float n) {
    return frac(sin(n/1873.1873) * randomSeed);
}

float noise2d(float3 p) {

    float3 fr = floor(p),
    ft = frac(p);

    float n = 1153.0 * fr.x + 2381.0 * fr.y + p.z,
    nr = n + 1153.0,
    nd = n + 2381.0,
    no = nr + 2381.0,

    v = hash(n),
    vr = hash(nr),
    vd = hash(nd),
    vo = hash(no);

    return ((v * (1.0 - ft.x) + vr * (ft.x)) * (1.0 - ft.y) + (vd * (1.0 - ft.x) + vo * ft.x) * ft.y);
}

float noise3d(float3 p) {

    float3 fr = floor(p),
    ft = frac(p);

    float n = 1153.0 * fr.x + 2381.0 * fr.y + fr.z,
    nr = n + 1153.0,
    nd = n + 2381.0,
    no = nr + 2381.0,

    v = lerp(hash(n), hash(n + 1.0), ft.z),
    vr = lerp(hash(nr), hash(nr + 1.0), ft.z),
    vd = lerp(hash(nd), hash(nd + 1.0), ft.z),
    vo = lerp(hash(no), hash(no + 1.0), ft.z);

    return lerp(lerp(v,vr,ft.x), lerp(vd,vo,ft.x),ft.y);
}

float noise3d_timedep(float3 p, float time) {

    float3 fr = floor(p),
    ft = frac(p);

    float n = 1153.0 * fr.x + 2381.0 * fr.y + fr.z + time,
    nr = n + 1153.0,
    nd = n + 2381.0,
    no = nr + 2381.0,

    v = lerp(hash(n), hash(n + 1.0), ft.z),
    vr = lerp(hash(nr), hash(nr + 1.0), ft.z),
    vd = lerp(hash(nd), hash(nd + 1.0), ft.z),
    vo = lerp(hash(no), hash(no + 1.0), ft.z);

    return lerp(lerp(v,vr,ft.x), lerp(vd,vo,ft.x),ft.y);
}

inline float map(float x, float a, float b, float c, float d) {
    return (x - b) / (a - b) * (c - d) + d;
}

inline float clampMap(float x, float a, float b, float c, float d) {
    return saturate((x - b) / (a - b)) * (c - d) + d;
}

float PuddleNoise(float3 p, float3 Normals, float rainny, float puddle_term, float coverage)
{
    p.z/=8.0;
    p/=5.0f;
    float n = noise3d(p)*0.5f;
    n += noise3d(2.0f*p)*0.25f;
    n += noise3d(6.0f*p)*0.125f;
    
    float fac = saturate(2.0f*n-0.5f)*2.0f;
    n += noise3d(32.0f*p)*0.0625f*fac;
    n += noise3d(128.0f*p)*0.0625f*fac;
    
    n = (1.0-n) * clampMap(Normals.z, 1.0-rainny, -rainny, 1.0, 0.0);
    return smoothstep(0.58f, 1.0f - 0.35f*puddle_term, n + coverage - 0.5) * puddle_term;
}

#endif


float3 pow3(float3 v, float n)
{
    return float3(pow(v.x, n), pow(v.y, n), pow(v.z, n));
}


#define ATM_DENSE vGradingColor0.r*7.0f
#define MIST vGradingColor0.g

static float atmosphereStep = 15.0;
static float lightStep = 3.0;
static float fix=0.00001;

static float3 sunLightStrength = 685.0*float3(1.0, 0.96, 0.949);
static float moonReflecticity = 0.16;
static float starStrength = 0.75;
static float LightingDecay = 300.0;

static float rayleighStrength = 1.85;
static const float rayleighDecay = 900.0;
static float atmDensity = 3.0 + ATM_DENSE;
static float atmCondensitivity = 1.0;

//static const float3 waveLengthFactor = pow3(float3(6.5, 5.4, 4.5), 4.0);
static const float3 waveLengthFactor = float3(1785.0625, 850.3056, 410.0625);
static const float3 scatteringFactor = waveLengthFactor/rayleighDecay;   // for tuning, should be 1.0 physically, but you know.

static float mieStrength = 0.125 + MIST*0.1;
static float mieDecay = 100.0;
static float fogDensity = 0.25 + MIST;

static const float earthRadius = 6.421;
static const float groundHeight = 6.371;
static const float3 AtmOrigin = float3(0.0, 0.0, groundHeight);

static float4 earth = float4(0.0, 0.0, 0.0, earthRadius);

static float game2atm = 1000000.0;

float3 Game2Atm(float3 gamepos)
{
    return gamepos/game2atm + AtmOrigin;
}

float3 Game2Atm_Alt(float3 gamepos)
{
    return float3(0.0, 0.0, gamepos.z/game2atm) + AtmOrigin;
}

float4 sphereCast(float3 origin, float3 ray, float4 sphere, float step, out float3 begin)
{
    float3 p = origin - sphere.xyz;
    
    float 
        r = length(p),
        d = length(cross(p, ray));    //nearest distance from view-ray to the center
        
    if (d > sphere.w+fix) return float4(0.0);    //early exit
        
    float
        sr = sqrt(sphere.w*sphere.w - d*d),    //radius of slicing circle
        dr = -dot(p, ray);    //distance from camera to the nearest point
        
    float3
        pc = origin + ray * dr,
        pf = pc + ray * sr,
        pb = pc - ray * sr;
    
    float sl; //stpe length
    
    if(r > sphere.w){
        
        begin = pb;
        sl = sr*2.0/step;
    }
    else{
    
        begin = origin;
        sl = length(pf - origin)/step;
    }
    return float4(ray * sl, sl);
}


float2 Density(float3 pos, float4 sphere, float strength, float condense)
{
    float r = groundHeight;
    float h = length(pos-sphere.xyz)-r;
    float ep = exp(-(sphere.w-r)*condense);
    
    float fog = fogDensity*(1.0/(1200.0*h+0.5)-0.04)/1.96;
    
    if(h<0.0)
        return float2(strength, fogDensity);

    return float2((exp(-h*condense)-ep)/(1.0-ep)*strength, fog);
}

float3 rayleighScattering(float c)
{
    return (1.0+c*c)*rayleighStrength/waveLengthFactor;
}

float MiePhase(float c)
{
    return 1.0f+1.6f*exp(20.0f*(c-1.0f));
}

float MieScattering(float c)
{
    return mieStrength*MiePhase(c);
}

float3 LightDecay(float densityR, float densityM)
{
    return float3(exp(-densityR/scatteringFactor - densityM*mieDecay));
}

float3 SunLight(float3 light, float3 position, float3 lightDirection, float4 sphere)
{
    float3 smp;
    float4 sms = sphereCast(position, lightDirection, sphere, lightStep, smp);
        
    float2 dl;
    
    for(float j=0.0; j<lightStep; j++)
    {
        smp+=sms.xyz/2.0;
        dl += Density(smp, sphere, atmDensity, atmCondensitivity)*sms.w;
        smp+=sms.xyz/2.0;
    }
    
    return light*LightDecay(dl.x, dl.y)/LightingDecay;
}

float3 LightSource(float3 SunDirection, out float3 SourceDir)
{
    if(SunDirection.z<-0.075)
    {
        SourceDir = -SunDirection;
        return sunLightStrength*moonReflecticity*smoothstep(-0.075, -0.105, SunDirection.z);
    }
    
    SourceDir = SunDirection;
    return sunLightStrength*smoothstep(-0.075, -0.025, SunDirection.z);
}
  
float3 AtmosphereScattering(float3 background ,float3 marchPos, float4 marchStep, float3 ray, float3 lightStrength, float3 lightDirection, float strength, float4 sphere)
{
    float3 intensity=float3(0.0);
    
    float ang_cos = dot(ray, lightDirection);
    float mie = MieScattering(ang_cos);
    float3 raylei = rayleighScattering(ang_cos);
    
    if(marchStep.w>0.015)
        marchStep /= marchStep.w/0.015;
    
    float2 dv=float2(0.0);
    for(float i=0.0; i<atmosphereStep; i++)
    {
        float3 smp;
        float4 sms = sphereCast(marchPos, lightDirection, sphere, lightStep, smp);
        float2 sampling = Density(marchPos, sphere, atmDensity, atmCondensitivity)*marchStep.w;
            
        dv += sampling/2.0;
        
        float2 dl=dv;
        for(float j=0.0; j<lightStep; j++)
        {
            smp+=sms.xyz;
            dl += Density(smp, sphere, atmDensity, atmCondensitivity)*sms.w;
        }
        
        intensity += LightDecay(dl.x, dl.y)*(raylei*sampling.x + mie*sampling.y);
        
        dv += sampling/2.0;

        //marching
        marchPos+=marchStep.xyz;
    }
    
    return lightStrength*intensity*strength + background*LightDecay(dv.x, dv.y);
}

float3 ExponentialFog(float3 camera, float3 viewDir, float distance, out float transparency)
{
    float h0 = 0.0, 
        h1 = 160.0,
        a = 38.0,
        density = fFogStart/2000.0 + 0.00000001f;   //plus a small value to fix a weird problem

    float t, c1, e1, e2, e3, res;
    
    //fix horizontal error
    if(abs(viewDir.z)<=0.0001)
        viewDir.z=viewDir.z>0.0?0.0001:-0.0001;
    
    //Fix for altitude higher than h1
    if (camera.z>h1)
    {
        distance = max(0.0, distance + (camera.z - h1) / viewDir.z);
        camera.z = h1;
    }
    
    //ray length (inverse)
    t = max(1.0/distance, (viewDir.z>0.0) ? viewDir.z/(h1 - camera.z) : 0.0);
        
    //constants and values
    c1 = viewDir.z/a;
    e1 = exp((h0 - camera.z)/a);
    e2 = exp((h0 - camera.z - viewDir.z/t)/a);
    e3 = exp((h0 - h1)/a);
    
    //calculate the exp fog
    res = (e1 - e2)/c1 - e3/t;
    
    transparency = exp(-res*density);
    return (1.0f-transparency)*vHorizonCol;
}

float3 atmosphere_scattering(float strength, float3 color, float3 camera, float3 ray, float distance, float3 SunDirection, float4 sphere)
{
    //fade-in and cut off
    if(distance<200.0)
        return color;
    float fade=smoothstep(200.0, 300.0, distance);
    
    //additionally multiply the distance since the map scale is too small to be scattered.
    float4 marchStep=float4(0.0);
    marchStep.w = 15.0*distance/atmosphereStep/game2atm;
    marchStep.xyz = ray*marchStep.w;
    
    float3 lightDirection = float3(0.0);
    float3 lightStrength = LightSource(SunDirection, lightDirection);
    
    float3 scattered = AtmosphereScattering(color,  Game2Atm_Alt(camera), marchStep, ray, lightStrength, lightDirection, 1.0, sphere);
    return lerp(color, scattered, fade);
}

float3 WorldToSpace(float3 ray, float3 SunDirection)
{
    float3 
        axis_x = normalize(float3(0.85, -2.0, 0.45)),   //The rotation axis of the Earth
        axis_z = normalize(cross(axis_x, SunDirection)),
        axis_y = normalize(cross(axis_z, axis_x));
        
    return normalize(float3(dot(axis_x, ray), dot(axis_y, ray), dot(axis_z, ray)));
}

float3 StarSky(float strength, float density, float curve, float3 ray)
{
    float3 star_ray = ray*density;
    
    float
        x = noise3d(star_ray + float3(250.1654, 161.15, 200.5556)),
        y = noise3d(star_ray + float3(12.04645, 20.012631, 300.4580)),
        z = noise3d(star_ray - float3(100.234, -20.0156, 3000.0912));
        
    star_ray = 2.0*float3(x, y, z)-1.0;
            
    return float3(pow(saturate(dot(star_ray, ray)), curve)*strength*2.0f);
}

float3 OutterSpace(float3 ray, float3 SunDirection)
{
    float3 res=float3(0.0);

    float ang_cos = dot(ray, SunDirection);
    
    //The sun
    if(ang_cos>0.9999){
        res += 0.1*sunLightStrength*smoothstep(0.9999,0.99995,ang_cos);
    }
        
    //The moon
    else if(ang_cos<-0.99978){
        res += 0.03*moonReflecticity*sunLightStrength*smoothstep(-0.99978,-0.99985,ang_cos);
    }
    
    //Stars
    else{
        res += StarSky(starStrength, 300.0, 6.0, WorldToSpace(ray, SunDirection))*smoothstep(0.05, -0.1, SunDirection.z);
    }
    
    return res;
}

float3 RadientLight(float3 sunDirection, float3 ray)
{
    float strength = smoothstep(0.1, 0.0,  sunDirection.z) * smoothstep(-0.25, 0.0,  sunDirection.z);
    strength *= (pow((ray.z+1.0)/2.0, 3.0)*0.5 + 0.5*dot(ray, sunDirection)+0.5);
    
    return float3(0.13, 0.15, 0.15)*saturate(strength);
}

float3 Sky(float3 background, float3 camera, float3 ray, float3 SunDirection, float4 sphere)
{
    float3 marchPos;
    float4 marchStep = sphereCast(camera, ray, sphere, atmosphereStep, marchPos);
    
    if (marchStep.w == 0.0) return background;
    
    float strength=1.0;
    float3 ground;
    if(sphereCast(camera, ray, float4(0.0, 0.0, 0.0, groundHeight-0.00075), atmosphereStep, ground).w>0.0)
    {
        if(length(ground-marchPos)>0.0 && dot(ground-marchPos, ray)>0.0)
        {
            marchStep.xyz = (ground-marchPos)/atmosphereStep;
            marchStep.w = length(marchStep.xyz);
            background=float3(0.0);
        }
    }
    
    float3 lightDirection = float3(0.0);
    float3 lightStrength = LightSource(SunDirection, lightDirection);
    
    float3 radientFix = RadientLight(SunDirection, ray);
    
    return AtmosphereScattering(background, marchPos, marchStep, ray, lightStrength, lightDirection, strength, sphere) + radientFix;
}

/*===================================================================================
GTA SA - CloudWorks Beta
by Brian Tu

See my Github page for more information:
https://github.com/keroroxzz

2021/5/30

This is the CloudWorks shader by RTU, still in beta version. It's currently bulky, 
I'd like to make the codes looks neat.

Caution : Beta version is not a public version, this shouldn't appear else where.
===================================================================================*/

//=================Perameters========================
#define CLOUD_COVERAGE vGradingColor1.rgb

struct CloudProfile
{
    float4 march;   //min length, max length, multiplicand, max step
    
    float2 cutoff;  // density cut, transparency cut

    float2 volumeBox;   //top, bottom
    float4 shape;       //top, mid, bot, thickness

    float brightness;
    float3 range;    //total, top, bottom
    float2 solidness;   //top, bottom
    
    float2 densityChunk;    //dens A,B
    
    float4 shadow;  //step length, detail strength, expanding, strength
    float4 distortion;  //max angle, strength, bump strength, small bump strength
    
    float fade;
    
    float3 densityDetail;   //dens C,D,E
    
    // 
    float3 scaleChunk;    //scale A,B, vertical stretch
    float3 scaleDetail;   //scale C,D,E
    float3 cloudShift;

    float3 offsetA;
    float3 offsetB;
    float3 offsetC;
    float3 offsetD;
};

struct CloudBaseColor
{
    float3 CBC;
    float3 CBC_D;
    float3 CBC_S;
};

float gameTime(float it)
{
    float3 axis_x = normalize(float3(0.85, -2.0, 0.45)),
    axis_z = cross(axis_x, float3(1.0, 0.0, 0.0)),
    axis_y = cross(axis_z, axis_x),
    
    yz = normalize(float3(0.0, dot(vSunLightDir, axis_y), dot(vSunLightDir, axis_z)));
    
    float daytime = atan2(yz.y, yz.z)*10000.0;

    return CloudStartHeight*100.0f + CloudEndHeight + it + daytime*0.0;
}
static const float time = 1.0;

//Cloud settings
    CloudBaseColor baseColor=CloudBaseColor(
        float3(0.01, 0.015, 0.025),
        float3(0.55, 0.6, 0.65),
        float3(0.2, 0.25608, 0.33));
        
static CloudProfile cloudProfiles[3]= CloudProfile[3](CloudProfile(
        float4(6.0, 80.0, 8.7, 200.0),   //min length, max length, multiplicand, max step
        float2(0.0, 0.2),  // density cut, transparency cut
        float2(700.0, 300.0),   //top, bottom
        float4(700.0, 450.0, 0.0, 0.0),       //top, mid, bot, thickness
        0.25,
        float3(0.9 + CLOUD_COVERAGE[0]*0.2, 0.1, 0.2+CLOUD_COVERAGE[0]*0.4),    //total, top, bottom
        float2(3.0, 0.0) * CLOUD_COVERAGE[0],   //top, bottom
        float2(0.3, 0.5),    //dens A,B
        float4(50.0, 1.5, 0.05, 0.1),  //step length, detail strength, expanding, strength
        float4(3.0, 16.0, 25.0, 16.0),  //max angle, strength, bump strength, small bump strength
        6500.0,
        float3(0.3, 0.2, 0.6),   //dens C,D,E
        float3(0.0008, 0.005, 1.0),    //scale A,B, vertical stretch
        float3(0.02, 0.04, 0.1),   //scale C,D,E
        float3(-0.5, 0.0, 0.0),
        float3(1.8, -1.0, 0.0)*-time,
        float3(2.0, 0.2, 0.0)*-time,
        float3(3.0, 0.0, 0.5)*-time,
        float3(3.5, 0.0, -0.1)*-time),
        
CloudProfile(
    float4(12.0, 70.0, 8.0, 150.0),   //min length, max length, multiplicand, max step
    float2(0.0, 0.2),  // density cut, transparency cut
    float2(1900.0, 1500.0),   //top, bottom
    float4(2100.0, 1650.0, 0.0, 0.0),       //top, mid, bot, thickness
    0.25,
    float3(0.85+CLOUD_COVERAGE[1]*0.78, 0.0, 0.3+CLOUD_COVERAGE[1]*0.16),    //total, top, bottom
    float2(0.3, 0.1) * CLOUD_COVERAGE[1],   //top, bottom
    float2(0.25, 0.6),    //dens A,B
    float4(30.0, 1.0, 0.025, 0.1),  //step length, detail strength, expanding, strength
    float4(6.0, 50.0, 666.0, 50.0),  //max angle, strength, bump strength, small bump strength
    20000.0,
    float3(0.6, 0.2, 0.3),   //dens C,D,E
    float3(0.0008, 0.004, 1.5),    //scale A,B, vertical stretch
    float3(0.0142857, 0.0285714, 0.08),   //scale C,D,E
    float3(0.0, 0.0, 0.0),
    float3(1.5, -1.2, 0.0)*-time,
    float3(1.9, 0.5, 0.0)*-time,
    float3(2.5, 0.0, 0.5)*-time,
    float3(3.0, 0.1, -0.1)*-time),
    
CloudProfile(
    float4(5.0, 75.0, 500.0, 50.0),   //min length, max length, multiplicand, max step
    float2(0.0, 0.2),  // density cut, transparency cut
    float2(3600.0, 3500.0),   //top, bottom
    float4(3800.0, 3520.0, 3450.0, 0.0),       //top, mid, bot, thickness
    0.25,
    float3(1.0+CLOUD_COVERAGE[2]*0.9, 0.2, 0.35),    //total, top, bottom
    float2(0.25, 0.0),   //top, bottom
    float2(0.4, 0.3),    //dens A,B
    float4(50.0, 1.5, 0.02, 0.1),  //step length, detail strength, expanding, strength
    float4(2.5, 15000.0, 0.0, 0.0),  //max angle, strength, bump strength, small bump strength
    2000000.0,
    float3(0.2, 0.1, 0.6),   //dens C,D,E
    float3(0.00016, 0.0008, 1.5),    //scale A,B, vertical stretch
    float3(0.004, 0.006667, 0.02),   //scale C,D,E
    float3(0.0, 0.0, 0.0),
    float3(1.3, -1.8, 0.0)*-time,
    float3(1.6, 0.8, 0.0)*-time,
    float3(2.5, 0.2, 0.5)*-time,
    float3(3.0, 0.1, -0.1)*-time));


inline float3 PosOnPlane(const in float3 origin, const in float3 direction, const in float h, inout float dis) {
    dis = (h - origin.z) / direction.z;
    return origin + direction * dis;
}

//generate parameters
float4 CloudShape(float z, float4 shape, float3 range) {
    float soft = map(z, shape.y, shape.x, range.z, range.y); //range of the soft part

    return float4(
        smoothstep(shape.z, lerp(shape.y, shape.z, shape.w), z) * smoothstep(shape.x, lerp(shape.y, shape.x, shape.w), z),    //shape factor
        range.x + soft,
        range.x - soft,
        soft);
}

inline float3 Distortion(float lump, float4 distortion)
{
    return float3(cos(lump*distortion.x)*distortion.y, 0.0, -lump*distortion.z);
}

//large chunk for the distribution of clouds.
float Chunk(in float3 pos, const in float2 density, const in float3 scale, const in float3 cloud_shift, const in float3 offsetA, const in float3 offsetB, in float cs) {

    pos.z/=scale.z;
    pos += cloud_shift * pos.z;
    
    float3
        pA = (pos + offsetA)*scale.x,
        pB = (pos + offsetB)*scale.y;
    
    float dens_a = noise3d(pA) * (noise3d(pB)*density.y+density.x)*cs;
    
    return saturate(dens_a);
}

//Detial noise for clouds
inline float DetailA(const in float3 pos, const in float3 density, const in float3 scale, const in float3 offsetC, const in float3 distortion) {

    return density.x * noise3d((pos + offsetC + distortion) * scale.x);
}

float DetailB(const in float lump_density, const in float3 pos, const in float3 density, const in float3 scale, const in float4 distortionParm, const in float3 offsetC, const in float3 offsetD, float cs) {

    float3 d = Distortion(lump_density, distortionParm);
    
    float3
        pD = pos + offsetD;

    float nc = DetailA(pos, density, scale, offsetC, d);
    
    d.z -= nc * distortionParm.w;

    float dens = nc + density.y * (noise3d((pD + d/3.0) * scale.y));
    dens += dens * density.z * noise3d((pD + d*8.0) * scale.z) ;

    return dens;
}

inline float DensityField(float lump, float detail)
{
    return lump*detail+lump;
}

inline float DensityMix(float density_field, float height, float low, float high, float2 volumeBox, float2 solidness)
{
    return clampMap(density_field, low, high, 0.0, clampMap(height, volumeBox.y, volumeBox.x, solidness.y, solidness.x));
}

inline float ShadowMarching(float dens, float3 p, float2 densityChunk, float3 densityDetail, float3 scaleChunk, float3 scaleDetail, float3 shift, float4 profile, float3 dens_thres, const in float3 offsetA, const in float3 offsetB, const in float3 offsetC, float3 SunDirection, float4 shadow, float2 volumeBox, float2 solidness, float4 distortionParm) {
    
    //exit for low density
    if(dens<=0.025) return dens*shadow.x;  //return a approx. value
    
    float threshold = 2.0f/shadow.w/shadow.x;
    
    float d = 0.0;
    float4 step = float4(SunDirection.xyz * shadow.x, shadow.x);
    
    //sampling
    for(int i=0; d<threshold  && p.z<volumeBox.x && p.z>volumeBox.y && i<10; i++)
    {
        p += step.xyz;
        
        float4 cs = CloudShape(p.z, profile, dens_thres);
        float d1 = Chunk(p, densityChunk, scaleChunk, shift, offsetA, offsetB, cs.x);
        
        float3 displace = Distortion(d1, distortionParm);
        float d2 = DetailA(p, densityDetail, scaleDetail, offsetC, displace) * shadow.y;
        d += DensityMix(DensityField(d1, d2), p.z, cs.z-shadow.z, cs.y, volumeBox, solidness);
    }
    return d*shadow.w*step.w;
}

float3 CastStep(float3 origin, float3 ray)
{
    float height = 500.0;
    float2 radius = float2(500.0,600.0);
    float r = length(origin-float3(0.0,0.0,height));
    
    float3 begIn, begOut;
    origin.xy = float2(0.0,0.0);
    float4 inner = sphereCast(origin, ray, float4(0.0,0.0,0.0,radius.x), 1.0, begIn);
    float4 outer = sphereCast(origin, ray, float4(0.0,0.0,0.0,radius.y), 1.0, begOut);
    
    return origin;
}

float4 CloudAtRay(CloudProfile a, CloudBaseColor b, const in float3 cam_dir, in float3 cam_pos, float3 light, float3 lightDirection, float time, inout float depth) {   // inout: read before written

    float4 d = float4(0.0, 0.0, 0.0, a.march.y); //distance, approx. distance, gap, the last march step
    
    float3 p = PosOnPlane(cam_pos, cam_dir, clamp(cam_pos.z, a.volumeBox.y+0.001, a.volumeBox.x-0.001), d.x);
    d.y = d.x;
	
    if (d.x >= 0.0) {
    
        a.range.x = 1.0f/a.range.x;
        
        float3 fx = float3(0.0, 0.0, 1.0); //brightness, cumulative density, transparency
        
        float last_sample = 0.0, pdf=0.0;
        
        for (int i = 0; fx.z > 0.0 && p.z <= a.volumeBox.x && p.z >= a.volumeBox.y && i < int(a.march.w) && d.x - d.w < depth && d.x < a.fade; i++) {
            
            float3 
                cs = CloudShape(p.z, a.shape, a.range).xyz;

            float
                d1 = Chunk(p, a.densityChunk, a.scaleChunk, a.cloudShift, a.offsetA, a.offsetB, cs.x),
                d2 = DetailB(d1, p, a.densityDetail, a.scaleDetail, a.distortion, a.offsetC, a.offsetD, cs.x),
                df = DensityField(d1, d2);
                
            if(df>cs.z)
            {
                float
                    dens = DensityMix(df, p.z, cs.z, cs.y, a.volumeBox, a.solidness),
                    c_dens = (dens + last_sample) * a.march.x / 2.0f;
                    
                last_sample = dens;
                
                //fade into near objects
                if (d.x >= depth)
                    c_dens *= d.z / d.w;
                
                
                if(c_dens > 0.0f)
                    d.y = d.y * (1.0 - fx.z) + fx.z * d.x;
                    
                fx.y += c_dens;
                fx.z = (exp(-fx.y) - a.cutoff.y) / (1.0f - a.cutoff.y);
                d.z = depth - d.x;
                
                
                //lighting strength
                float dens_light=0.0;
                if(fx.y<2.3) 
                {
                    dens_light  = ShadowMarching(c_dens ,p, a.densityChunk, a.densityDetail, a.scaleChunk, a.scaleDetail, a.cloudShift, a.shape, a.range, a.offsetA, a.offsetB, a.offsetC, lightDirection, a.shadow, a.volumeBox, a.solidness, a.distortion); 
                    fx.x += c_dens * exp((-dens_light - fx.y));
                }  
            }
            
            //dynamic step length
            d.w = clampMap(2.0*df-pdf, cs.z*0.85, a.cutoff.x, a.march.x, a.march.y);
            d.w *= clampMap(d.x, 0.0, a.fade, 1.0, a.march.z);
            d.w += 2.0*noise2d(p)*a.march.x;
            pdf=df;
            
            //marching
            p += cam_dir * d.w;
            d.x += d.w;
        }
        
        if (fx.z < 1.0) {
            fx = saturate(fx);
            
            float suncross = dot(lightDirection.xyz, cam_dir);

            float3 z_pos = float3(0.0, 0.0, cam_pos.z);
            float3 sun_samp_pos = Game2Atm(z_pos + cam_dir*d.y);
            float3 my_cam_pos = Game2Atm(z_pos);

            float3
                c_bright = SunLight(light, sun_samp_pos, lightDirection, earth)*a.brightness;
            
            //Mie scattering

            float3 C = c_bright * fx.x*MiePhase(suncross) + b.CBC*(1.0-fx.z);
            C = atmosphere_scattering((1.0-fx.z), C, my_cam_pos, cam_dir, d.y/game2atm, lightDirection, earth);
            
            //calculate the depth for merging
            depth = depth*fx.z + d.y*(1.0-fx.z);
            
            return float4(C, fx.z);
        }
    }
    return float4(0.0, 0.0, 0.0, 1.0);
}

//=========================settings==============================

void CloudRandomness(float time)
{
    for(int i=0;i<3;i++)
    {
        float rand_grow = noise3d(float3(cloudProfiles[i].shape.y, cloudProfiles[i].volumeBox.y, time/2000.0))*0.45+0.65;
        cloudProfiles[i].range.x *= rand_grow*(1.0-CLOUD_COVERAGE[i])+CLOUD_COVERAGE[i];
    }
}

void CloudColorVarying(float3 SunDir, inout CloudBaseColor b)   // inout: read before written
{
    float 
        isNight = smoothstep(0.3, 0.1, SunDir.z),
        isDay = smoothstep(-0.05, 0.05, SunDir.z), 
        isSunSet = isNight*isDay;
        
    b.CBC += lerp(b.CBC_D, b.CBC_S, isSunSet)*isDay;
}

float ShadowOnGround(CloudProfile a, float3 position, float3 SunDirection) {
    
    //Remember this you mtf
    a.range.x = 1.0f/a.range.x;
        
	float d;
    float3 samppos = PosOnPlane(position, SunDirection, a.shape.y, d);
	
    float d1 = 0.0, d2 = 0.0;
    float4 cs;
    
	cs = CloudShape(samppos.z, a.shape, a.range);
	d1 = Chunk(samppos, a.densityChunk, a.scaleChunk, a.cloudShift, a.offsetA, a.offsetB, cs.x);
	
	float3 displace = Distortion(d1, a.distortion);
	d2 = DetailA(samppos, a.densityDetail, a.scaleDetail, a.offsetC, displace) * a.shadow.y;
    
	float adjustTerm =  max(a.shape.y - max(a.volumeBox.y, position.z), 0.0) * clampMap(d, 0.0, 3000.0, 1.0, 0.0);
    
	return DensityMix(DensityField(d1, d2), samppos.z, cs.z-a.shadow.z*2.0f, cs.y*2.0f, a.volumeBox, a.solidness)*adjustTerm;
}

float4 RenderClouds(float3 ViewDir, float3 ViewPos, float3 sunDir, float ViewZ)
{
    
    CloudColorVarying(sunDir, baseColor);
    
    float3 lightDir = float3(0.0);
    float3 light = LightSource(sunDir, lightDir);

    float4 cloud_temp = float4(0.0, 0.0, 0.0, 1.0);
        
    int i = (ViewPos.z>3500.0? 2:(ViewPos.z>1500.0? 1:0));
    float4 clouds = CloudAtRay(cloudProfiles[i], baseColor, ViewDir, ViewPos, light, lightDir, time, ViewZ);
        
    i = (ViewPos.z>3500.0? 1:(ViewPos.z>1500.0? 0:1));
    cloud_temp = CloudAtRay(cloudProfiles[i], baseColor, ViewDir, ViewPos, light, lightDir, time, ViewZ);
    clouds.rgb += cloud_temp.rgb*clouds.w;
    clouds.w = clouds.w*cloud_temp.w;
    
    i = (ViewPos.z>3500.0? 0:(ViewPos.z>1500.0? 2:2));
    cloud_temp = CloudAtRay(cloudProfiles[i], baseColor, ViewDir, ViewPos, light, lightDir, time, ViewZ);
    clouds.rgb += cloud_temp.rgb*clouds.w;
    clouds.w = clouds.w*cloud_temp.w;
    
    return clouds;
}

float fov=1.6;

void timeEffect(float t, inout CloudProfile c)   // inout: read before written
{
    c.offsetA*=t;
    c.offsetB*=t;
    c.offsetC*=t;
    c.offsetD*=t;
}

vec3 getViewRay(vec2 fragCoord, float fov, out float ang_v)
{
    vec2 uv = (fragCoord*2.0 - iResolution.xy)/iResolution.x;
    vec3 d = normalize(vec3(1.0/fov, -uv.x, uv.y));
  
    #ifdef NO_MOUSE
    return d;
    #endif
  
    //vec2 mouse = (iMouse.xy/iResolution.xy*2.0 - 1.0)*vec2(1.0, -1.0)*3.5;
    vec2 mouse = (float2(240.0, 180.0)/iResolution.xy*2.0 - 1.0)*vec2(1.0, -1.0)*3.5;
    vec3 x = vec3(cos(mouse.x)*cos(mouse.y), sin(mouse.x)*cos(mouse.y), sin(mouse.y)),
        y = vec3(-sin(mouse.x), cos(mouse.x), 0.0),
        z = cross(x, y);
        
    ang_v = mouse.y + atan(d.z, d.x);
    return d.x*x + d.y*y + d.z*z;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 mouse = -(iMouse.xy/iResolution.xy*2.0 - 1.0)*vec2(1.0, -1.0)*3.5;
    vSunLightDir = vec3(cos(mouse.x)*cos(mouse.y), sin(mouse.x)*cos(mouse.y), sin(mouse.y));
    
    float time = gameTime(iTime*100.0)-2000.0;
    timeEffect(time, cloudProfiles[0]);
    timeEffect(time, cloudProfiles[1]);
    timeEffect(time, cloudProfiles[2]);

    CloudRandomness(time);
    
    float a = 2.0*3.14159/iResolution.x,
        b = 2.0*3.14159/iResolution.y;
        
   
    float ang_v = (fragCoord.y + 40.0)*fov*b;
    
    vec3 CameraPosition = vec3(0.0,0.0,0.0);
    vec3 direction = getViewRay(fragCoord, fov, ang_v);
    float dist=0.0;
    vec2 dest = PosOnPlane(CameraPosition,direction,20.0,dist).xy;
    
    vec3 c = OutterSpace(direction, vSunLightDir);
    c = Sky(c, Game2Atm_Alt(float3(0.0)), direction, vSunLightDir, earth);
    
    vec4 testt = RenderClouds(direction, float3(0.0), vSunLightDir, 9999999999.0);
    
    c=testt.xyz+c*testt.w;
    
    fragColor = vec4(c,1.0);
}