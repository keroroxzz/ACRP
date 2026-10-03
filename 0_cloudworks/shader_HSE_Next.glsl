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

/*===================================================================================
CloudWorks "Next" - adaptive band-limited ray marching
2026/07/28

Changes against shader_HSE.glsl (all switchable, see NEXT_* below):

  (1) Two-sided conservative band rejection.  DensityMix() clamps the field to
      0 below the erosion band and to its ceiling above it, so wherever the
      cheap chunk field alone already proves that the sample lands outside the
      band, the expensive detail octaves cannot change the result and are
      skipped.  Exact, not an approximation.

  (2) Surface step budget.  Long steps are taken between clouds; the step that
      lands inside a cloud is followed backwards in fixed Nyquist-length
      substeps until the boundary is recovered.  The substep densities are
      kept in a small array and replayed forward, so the boundary is resolved
      at Nyquist without re-sampling and without any root finding or ordering.
      Shadow marching over the replayed samples is strided.

  (3) Regional band-limit ("Nyquist") step planner.  The step length is driven
      by the distance of the field from the erosion band, measured in band
      widths, divided by an empirical Lipschitz bound taken from the free
      backward difference along the ray.  This is sphere tracing on the
      *rendered* density rather than on the raw noise, so both empty space AND
      saturated cloud cores get long steps - the old rule ("denser -> shorter")
      only handled empty space and burned steps inside solid bodies.

  (4) HSE roof caching.  HSELowCloudTop() is up to 9 hashed events plus a noise
      tap, but the field is 5-10 km wide and C2.  It is now sampled on a fixed
      XY spacing along the ray and interpolated linearly, which removes it from
      the inner loop of both the view march and the shadow march.

  (5) Cone-step empty-space skip above the animated roof (Nubis' cone step
      mapping applied to the analytic HSE height field).

  (6) Optical depth is integrated over the REAL step length.  The old marcher
      weighted every trapezoid by the constant march.x, which silently coupled
      opacity to the step count - any step-length change would have altered
      cloud density.  NEXT_OPTICAL_SCALE re-calibrates the look if needed.
===================================================================================*/

//=================Perameters========================
#define CLOUD_COVERAGE vGradingColor1.rgb
#define SHADOW_FINE_MARCH_STEP 6
#define SHADOW_COARSE_MARCH_STEP 20

//================ NEXT: adaptive marching configuration ================

#define NEXT_DETAIL_EARLY_OUT    1      // (1) lossless two-sided band rejection
#define NEXT_SURFACE_BUDGET      1      // (2) boundary back-off + surface array
#define NEXT_BAND_PLANNER        1      // (3) regional band-limit step planner
#define NEXT_ROOF_CACHE          1      // (4) piecewise-linear HSE roof along a ray
#define NEXT_ROOF_SKIP           1      // (5) cone-step skip above the roof
#define NEXT_TRUE_SEGMENT        1      // (6) integrate over the real step length
#define NEXT_ENERGY_CONSERVING   1      // (7) step-invariant in-scatter integral
#define NEXT_WEIGHTED_DEPTH      1      // (8) step-invariant cloud depth for the
                                        //     atmospheric blend (Nubis Evolved p.187)

// --- step planner ---------------------------------------------------------
#define NEXT_OPTICAL_SCALE       1.0    // re-calibration knob for (6)
#define NEXT_REACH_SAFETY        1.5    // fraction of the predicted safe reach
#define NEXT_GRAD_MEMORY         0.30   // decay of the sticky Lipschitz estimate
#define NEXT_GRAD_FLOOR          0.0025 // 1/m, caps the reach of a flat region
#define NEXT_BAND_EPS            0.02   // guards a degenerate (zero width) band
#define NEXT_TAU_BUDGET          0.35   // max optical depth deposited per step
#define NEXT_DENS_EPS            0.0005
#define NEXT_NYQUIST_SAFETY      0.5    // 0.5 = Nyquist of the finest octave
#define NEXT_DETAIL_MEAN         0.45   // expected fraction of detailGain; used to
                                        //   report an unbiased df from a skipped
                                        //   sample so the planner is not perturbed
// Step jitter.  The original shader needed a large one to dither the banding
// from its fixed 5 m steps.  The band planner already varies the step length
// continuously with the field, so the sample lattice is decorrelated by the
// content itself and the jitter is pure cost: over the six gate configurations
// dropping it from 0.25 to 0 is 14% faster AND lower error, because it was
// adding per-pixel noise the converged reference does not have.
// Checked for the banding it used to prevent, and found none: zoomed stills of
// the three views most likely to band, plus a 25 m camera dolly whose
// frame-to-frame difference is identical in magnitude and in low-frequency
// content at 0.25 / 0.10 / 0 (a position-locked lattice would show as coherent
// arcs sweeping the image).  Raise it if a future scene ever bands.
#define NEXT_JITTER              0.0   // step is only ever SHORTENED by jitter

// --- surface step budget --------------------------------------------------
#define NEXT_SURF_MAX            4      // max substeps spent recovering a surface
#define NEXT_SURF_TRIGGER        1.75   // only back off if the step was this long
#define NEXT_SURF_MIN_T          0.15   // skip once the ray is already opaque
#define NEXT_SURF_JITTER         0.85   // decorrelates the substep lattice
#define NEXT_SURF_SHADOW_STRIDE  2      // shadow march every Nth surface substep
#define NEXT_SHADOW_STRIDE       2      // shadow march every Nth ordinary sample

// ---- light-march LOD -----------------------------------------------------
// Two independent reasons a sample tolerates a cheaper shadow:
//   distance - its shadow detail projects to far below a pixel, so the fine
//              octaves are aliasing rather than resolving
//   opacity  - what the sample contributes is already attenuated by everything
//              in front of it, so its shadow only has to be roughly right
// Both collapse into one quality scalar in [0,1]; 1 is the full march.  Dropped
// octaves are replaced by their EXPECTATION, not by zero, and fewer steps are
// stretched to cover the SAME distance (see kFine/kCoarse) rather than stopping
// the light ray short.
//
// MEASURED AND SHIPPED OFF.  At any setting that holds the quality gate this is
// worth 2.7% at most - barely above the benchmark's ~1% noise floor - and it
// eats nearly all the remaining gate margin to get it.  The reason is that the
// light march is ALREADY adaptive: it breaks as soon as d passes
// 2/shadow.w/shadow.x, which in dense cloud is about two steps.  The full 6+20
// steps therefore only run where the path is thin, and that is exactly where
// the early-out has already decided the steps are needed.  LOD is trying to cut
// what is not being spent.
//
// The octave axis is worse than the step axis (1.95 vs 1.20 gate ratio at equal
// aggressiveness) for a specific reason: replacing an octave by its mean is
// unbiased in DENSITY, but the density then goes through DensityMix(), a clamped
// nonlinear remap, so the mean does not survive it.
//
// Turn back on if the light march's early-out threshold is ever raised, or if
// the quality target is relaxed.  Gate-neutral settings measured:
//   distance only  MAXLOD 12, floors 4/12         gate 0.984, -1.4%
//   opacity only   T 0.45->0.25, floors 4/14      gate 0.989, -1.6%
//   opacity only   T 0.40->0.24, floors 3/10      gate 0.997, -2.7%
#define NEXT_LIGHT_LOD           0
#define NEXT_LIGHT_LOD_MAXLOD    6.0    // view LOD at which the light march is cheapest
#define NEXT_LIGHT_LOD_THI       0.75   // transmittance above which quality is full
#define NEXT_LIGHT_LOD_TLO       0.28   // ... and below which it is cheapest
#define NEXT_LIGHT_FINE_MIN      2      // floor on the fine light-march steps
#define NEXT_LIGHT_COARSE_MIN    6      // floor on the coarse light-march steps
#define NEXT_LIGHT_OCTAVE_LOD    1      // also fade the fbm octaves, not just steps
// 0 disables.  Measured neutral-to-slower on the gate scenes (the branch costs
// more than the light marches it skips), but it bounds the error of a much more
// aggressive NEXT_SHADOW_STRIDE if that is ever wanted.
#define NEXT_LIGHT_CULL          0.0    // skip REFRESHING the light march when this
                                        //   sample contributes less than this much
                                        //   radiance; the stale value is still used,
                                        //   so the error is bounded by the weight

// --- HSE roof cache / skip ------------------------------------------------
#define NEXT_ROOF_SPAN_XY        240.0  // anchor spacing (metres of XY travel)
#define NEXT_ROOF_SPAN_XY_SHADOW 900.0  // only used by NEXT_SHADOW_ROOF 2
// How the light march models the animated roof.  Rebuilding a rolling anchor
// model per shaded sample costs ~8 field evaluations, and the light march runs
// once per shaded sample - profiling at 2560x1440 put that alone at 48% of the
// whole shader.  The light ray only uses the roof to decide where the layer
// ends, so a coarser model is affordable here than on the view ray.
//   0 = constant, reuse the roof already known at the sample (0 evaluations)
//   1 = linear across the whole light ray                    (1 evaluation)
//   2 = rolling anchors every NEXT_ROOF_SPAN_XY_SHADOW       (~8 evaluations)
//   0 = constant, reuse the roof already known at the sample (0 evaluations)
//   1 = one linear segment over [0, SPAN1], held beyond      (1 evaluation)
//   2 = two segments, [0,SPAN1] and [SPAN1,SPAN2]            (2 evaluations)
//   3 = rolling anchors every NEXT_ROOF_SPAN_XY_SHADOW       (~8 evaluations)
// The roof at t=0 is always free: the view march already knows it.
#define NEXT_SHADOW_ROOF         2
#define NEXT_SHADOW_ROOF_SPAN1   1000.0  // covers the fine section of the light march
#define NEXT_SHADOW_ROOF_SPAN2   4000.0
#define NEXT_ROOF_SLOPE          2.5    // Lipschitz bound of the HSE height field
                                        //   max|dh/dr| = 1.72*MAX_RISE/MIN_RADIUS
#define NEXT_BIG_SKIP            1.0e6

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
        float4(5.0, 60.0, 8.0, 350.0),   //min length, max length, multiplicand, max step
        float2(0.0, 0.2),  // density cut, transparency cut
        float2(700.0, 300.0),   //top, bottom
        float4(700.0, 500.0, 0.0, 0.0),       //top, mid, bot, thickness
        0.25,
        float3(0.9 + CLOUD_COVERAGE[0]*0.2, 0.1, 0.2+CLOUD_COVERAGE[0]*0.4),    //total, top, bottom
        float2(3.0, 0.0) * CLOUD_COVERAGE[0],   //top, bottom
        float2(0.3, 0.5),    //dens A,B
        float4(50.0, 1.5, 0.05, 0.1),  //step length, detail strength, expanding, strength
        float4(3.0, 16.0, 25.0, 16.0),  //max angle, strength, bump strength, small bump strength
        40000.0,
        float3(0.3, 0.2, 0.4),   //dens C,D,E
        float3(0.0008, 0.005, 1.6),    //scale A,B, vertical stretch
        float3(0.02, 0.04, 0.08),   //scale C,D,E
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



//================ Large-Scale Hashed Spatiotemporal Low-Cloud Top ================
//
// This field deforms only the TOP of cloud profile 0.  It is intentionally cheap:
// one broad compact event per coarse cell, fixed 3x3 neighborhood, no texture,
// no temporal history and no trigonometric orientation.
//
// Coordinates are GTA world XY in metres.  The returned value is a local cloud top.
//
#define HSE_CELL_SIZE      8000.0
#define HSE_MAX_RISE       7000.0
#define HSE_MIN_RADIUS     5000.0
#define HSE_MAX_RADIUS     10000.0
#define HSE_TIME_SCALE     0.00008

// High-frequency world-space hash for march jitter.  noise2d()/noise3d() are
// smooth value noises, so using them to jitter a step correlates neighbouring
// pixels and turns dither into visible blocks.
float NextHash(float3 p)
{
    return frac(sin(dot(p, float3(12.9898, 78.233, 37.719))) * 43758.5453);
}

float HSEHash12(float2 p)
{
    float3 p3 = frac(float3(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

float2 HSEHash22(float2 p)
{
    float3 p3 = frac(float3(p.x, p.y, p.x) *
                     float3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return frac(float2((p3.x + p3.y) * p3.z,
                       (p3.x + p3.z) * p3.y));
}

float HSESmoother(float x)
{
    x = saturate(x);
    return x*x*x*(x*(x*6.0 - 15.0) + 10.0);
}

float HSELife(float u)
{
    // Fast vertical development, long mature phase, smooth dissipation.
    float grow  = HSESmoother(u / 0.20);
    float decay = 1.0 - HSESmoother((u - 0.70) / 0.30);
    return grow * decay;
}

float HSERadiusLife(float u)
{
    // Radius expands later than height.
    return lerp(0.52, 1.0, HSESmoother((u - 0.08) / 0.55));
}

float HSEEventRise(float2 p, float cloudTime, float2 cell)
{
    float existence = HSEHash12(cell + 19.71);
    if (existence < 0.38)
        return 0.0;

    float2 r0 = HSEHash22(cell + 7.13);
    float2 r1 = HSEHash22(cell + 23.91);
    float2 r2 = HSEHash22(cell + 51.37);

    float2 center =
        (cell + lerp(float2(0.25), float2(0.75), r0)) * HSE_CELL_SIZE;

    float u = frac(cloudTime * HSE_TIME_SCALE + HSEHash12(cell + 37.11));
    float life = HSELife(u);
    if (life <= 0.0001)
        return 0.0;

    // Small bounded drift. No sin/cos orientation is needed.
    float driftPhase = (u * 2.0 - 1.0);
    center += (r1 * 2.0 - 1.0) * (650.0 * driftPhase);

    float radius = lerp(HSE_MIN_RADIUS, HSE_MAX_RADIUS, r1.y) *
                   HSERadiusLife(u);
    float aspect = lerp(0.68, 1.0, r2.y);

    // Swap the ellipse axes pseudo-randomly without trigonometry.
    float2 radii = (r0.x < 0.5)
        ? float2(radius, radius * aspect)
        : float2(radius * aspect, radius);

    float2 q = (p - center) / max(radii, float2(1.0));
    float rSq = dot(q, q);
    if (rSq >= 1.0)
        return 0.0;

    float s = 1.0 - rSq;
    float bump = s*s*s;

    float maxHeight = lerp(700.0, HSE_MAX_RISE, r2.x);
    return maxHeight * life * bump;
}

//================ NEXT: per-cell event cache ================
//
// Everything about an event except the final ellipse falloff - existence, the
// four hashes, the drift, the life curve, the radii, the peak height - depends
// only on its CELL, and cells are 8 km across.  A ray therefore re-derives the
// same nine events over and over: ~5 hashes each, against ~10 ALU for the
// falloff itself.
//
// MEASURED AND REJECTED, kept so it is not tried again.  Caching the derived
// form keyed on the base cell is exact (identical RMSE to five decimals in all
// six gate configurations) and removes most of the arithmetic, but it needs 45
// extra VGPRs and the occupancy loss makes the shader 19% SLOWER at 2560x1440
// (49.0 -> 58.5 ms over the gate set).  Same lesson as A3's per-cell adaptive
// branching: on this architecture, register pressure beats instruction count.
#define NEXT_HSE_CELL_CACHE 0

#if NEXT_HSE_CELL_CACHE
float4 g_hseEvent[9];                            // xy = centre, zw = 1/radii
float  g_hseAmp[9];                              // maxHeight*life, 0 = no event
float2 g_hseCell = float2(1.0e9, 1.0e9);         // cell the cache was built for

void HSEBuildCell(float2 baseCell, float cloudTime)
{
    for (int j = -1; j <= 1; ++j)
    {
        for (int i = -1; i <= 1; ++i)
        {
            int idx = (j+1)*3 + (i+1);
            float2 cell = baseCell + float2(float(i), float(j));

            float2 r1 = HSEHash22(cell + 23.91);
            float2 r0 = HSEHash22(cell + 7.13);
            float2 r2 = HSEHash22(cell + 51.37);

            float u = frac(cloudTime * HSE_TIME_SCALE + HSEHash12(cell + 37.11));
            float life = HSELife(u);
            if (HSEHash12(cell + 19.71) < 0.38)
                life = 0.0;

            float2 center =
                (cell + lerp(float2(0.25), float2(0.75), r0)) * HSE_CELL_SIZE;
            center += (r1 * 2.0 - 1.0) * (650.0 * (u * 2.0 - 1.0));

            float radius = lerp(HSE_MIN_RADIUS, HSE_MAX_RADIUS, r1.y) *
                           HSERadiusLife(u);
            float aspect = lerp(0.68, 1.0, r2.y);
            float2 radii = (r0.x < 0.5)
                ? float2(radius, radius * aspect)
                : float2(radius * aspect, radius);

            g_hseEvent[idx] = float4(center, 1.0 / max(radii, float2(1.0)));
            g_hseAmp[idx]   = (life > 0.0001)
                ? lerp(700.0, HSE_MAX_RISE, r2.x) * life : 0.0;
        }
    }
    g_hseCell = baseCell;
}
#endif

float HSELowCloudTop(float2 p, float cloudTime, float baseTop)
{
    float2 baseCell = floor(p / HSE_CELL_SIZE);
    float rise = 0.0;

    // Fixed 3x3 neighborhood. Compact support prevents cell seams.
#if NEXT_HSE_CELL_CACHE
    if (any(notEqual(baseCell, g_hseCell)))
        HSEBuildCell(baseCell, cloudTime);

    for (int k = 0; k < 9; ++k)
    {
        float2 q = (p - g_hseEvent[k].xy) * g_hseEvent[k].zw;
        float s = max(1.0 - dot(q, q), 0.0);
        rise = max(rise, g_hseAmp[k] * s*s*s);
    }
#else
    for (int j=-1; j<=1; ++j)
    {
        for (int i=-1; i<=1; ++i)
        {
            rise = max(rise,
                       HSEEventRise(p, cloudTime,
                                    baseCell + float2(float(i), float(j))));
        }
    }
#endif

    // A tiny, very-low-frequency baseline undulation prevents a perfectly flat roof.
    float broad = noise2d(float3(p * 0.000055, cloudTime * 0.00002));
    broad = (broad - 0.5) * 90.0;

    return baseTop + max(0.0, rise + broad);
}

// Apply an already-evaluated local roof height to a profile.  Split out of the
// sampling code so the roof can come from the cache below instead of a fresh
// 9-event evaluation.
void HSEApplyLocalTop(
    float2 volumeBox,
    float4 shape,
    float localTop,
    float active,
    out float2 localBox,
    out float4 localShape)
{
    localBox = volumeBox;
    localShape = shape;

    if (active > 0.5)
    {
        float rise = localTop - volumeBox.x;

        localBox.x = localTop;
        localShape.x = localTop;

        // Lift the soft/mid transition more slowly than the top.
        localShape.y = min(localTop - 40.0, shape.y + rise * 0.52);
    }
}

//================ NEXT (4): HSE roof cached along a ray ================
//
// HSELowCloudTop() costs up to 9 hashed events plus a value-noise tap, and the
// old shader called it once per view step AND once per shadow step (31 of
// those per shaded sample).  The field it evaluates is however extremely low
// frequency - the events are 5-10 km across and C2 continuous - so sampling it
// at the march rate is pure waste.
//
// Two anchors are kept on the ray and interpolated linearly.  They are spaced
// a fixed distance apart *in XY*, so a steep ray - which barely moves in XY -
// re-evaluates almost never.  Linear interpolation error over a span L is
// |h''|*L^2/8; with L = 200 m and the HSE curvature scale that is under 10 m.

struct HSEAnchor
{
    float t0;       // ray parameter of the first anchor
    float span;     // anchor spacing along the ray
    float h0;       // roof at t0
    float h1;       // roof at t0 + span
    float baseTop;  // un-deformed layer top
    float active;   // 1.0 for the HSE layer, 0.0 otherwise
};

float HSERoofEval(float active, float baseTop, float3 org, float3 dir, float t, float cloudTime)
{
    if (active < 0.5) return baseTop;
    return HSELowCloudTop((org + dir*t).xy, cloudTime, baseTop);
}

HSEAnchor HSEAnchorInit(float3 org, float3 dir, float t, float cloudTime,
                        float baseTop, int layerIndex, float spanXY)
{
    HSEAnchor an;
    an.baseTop = baseTop;
    an.active  = (layerIndex == 0) ? 1.0 : 0.0;
    an.span    = spanXY / max(length(dir.xy), 0.05);
    an.t0      = t;
#if NEXT_ROOF_CACHE
    an.h0 = HSERoofEval(an.active, baseTop, org, dir, t,           cloudTime);
    an.h1 = HSERoofEval(an.active, baseTop, org, dir, t + an.span, cloudTime);
#else
    an.h0 = baseTop;
    an.h1 = baseTop;
#endif
    return an;
}

void HSEAnchorSync(inout HSEAnchor an, float3 org, float3 dir, float t, float cloudTime)
{
#if NEXT_ROOF_CACHE
    if (t <= an.t0 + an.span) return;

    if (t <= an.t0 + 2.0*an.span)
    {
        // Ordinary marching: slide the window forward by one span.
        an.t0 += an.span;
        an.h0  = an.h1;
        an.h1  = HSERoofEval(an.active, an.baseTop, org, dir, an.t0 + an.span, cloudTime);
    }
    else
    {
        // We just took an empty-space skip: re-anchor from scratch.
        an.t0 = t;
        an.h0 = HSERoofEval(an.active, an.baseTop, org, dir, t,           cloudTime);
        an.h1 = HSERoofEval(an.active, an.baseTop, org, dir, t + an.span, cloudTime);
    }
#endif
}

float HSEAnchorAt(HSEAnchor an, float3 org, float3 dir, float t, float cloudTime)
{
#if NEXT_ROOF_CACHE
    float u = saturate((t - an.t0) / max(an.span, 0.001));
    return lerp(an.h0, an.h1, u);
#else
    return HSERoofEval(an.active, an.baseTop, org, dir, t, cloudTime);
#endif
}

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

// Chunk() with the second octave faded toward its mean.  Used only by the light
// march.  At w = 1 this is identical to Chunk(); at w = 0 the noise3d(pB) tap is
// skipped entirely and replaced by its expectation, which keeps the mean density
// right while removing detail that is below a pixel anyway.
float ChunkLod(in float3 pos, const in float2 density, const in float3 scale, const in float3 cloud_shift, const in float3 offsetA, const in float3 offsetB, in float cs, in float w) {

    pos.z/=scale.z;
    pos += cloud_shift * pos.z;

    float nB = 0.5;
    if (w > 0.0)
        nB = lerp(0.5, noise3d((pos + offsetB)*scale.y), w);

    return saturate(noise3d((pos + offsetA)*scale.x) * (nB*density.y + density.x) * cs);
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

//================ NEXT (1): two-sided conservative band rejection ================
//
// DensityMix() is a clamped remap: the rendered density is exactly 0 for
// df <= bandLo and exactly its ceiling for df >= bandHi.  Only inside the band
// does the noise carry any visible signal.  Because every noise term is
// non-negative and bounded, the cheap chunk field alone brackets the result:
//
//        d1  <=  df = d1*(1 + d2)  <=  d1*(1 + detailGain)
//
// so whenever the upper bound is still below bandLo, or the lower bound is
// already above bandHi, the detail octaves cannot change the answer and are
// skipped.  This is exact - it changes cost, not pixels.
//
// gains:  chunkGain  = densityChunk.x  + densityChunk.y
//         detailGain = (densityDetail.x + densityDetail.y)*(1 + densityDetail.z)
//
void SampleCloudDensity(float3 p, CloudProfile a, float shapeF,
                        float bandLo, float bandHi,
                        float chunkGain, float detailGain,
                        float2 localBox,
                        out float df, out float dens)
{
    df = 0.0;
    dens = 0.0;

    // Above the local animated roof there is nothing at all.
    if (p.z > localBox.x) return;

#if NEXT_DETAIL_EARLY_OUT
    // Cheapest test of all: the vertical shape factor alone cannot lift the
    // field into the band, so neither noise field needs evaluating.
    float ceilShape = min(1.0, chunkGain*shapeF) * (1.0 + detailGain);
    if (ceilShape <= bandLo) { df = ceilShape*NEXT_DETAIL_MEAN; return; }
#endif

    float d1 = Chunk(p, a.densityChunk, a.scaleChunk, a.cloudShift, a.offsetA, a.offsetB, shapeF);

#if NEXT_DETAIL_EARLY_OUT
    float dfHi = d1*(1.0 + detailGain);
    // NOTE both branches below are exact in DENSITY.  They report an estimated
    // df (not the bound) purely so the step planner's backward difference sees
    // the same signal it would have seen from a full evaluation.
    if (dfHi <= bandLo)                 // provably empty
    {
        df = d1*(1.0 + detailGain*NEXT_DETAIL_MEAN);
        return;
    }
    if (d1 >= bandHi)                   // provably saturated
    {
        df = d1*(1.0 + detailGain*NEXT_DETAIL_MEAN);
        dens = DensityMix(d1, p.z, bandLo, bandHi, localBox, a.solidness);
        return;
    }
#endif

    float d2 = DetailB(d1, p, a.densityDetail, a.scaleDetail, a.distortion, a.offsetC, a.offsetD, shapeF);
    df = DensityField(d1, d2);

    if (df > bandLo)
        dens = DensityMix(df, p.z, bandLo, bandHi, localBox, a.solidness);
}

// Piecewise-linear roof along the light ray from three anchors (t = 0, SPAN1,
// SPAN2), held constant beyond SPAN2.  Two segments rather than one because the
// fine section of the light march needs the roof far more accurately than the
// coarse tail does.
float ShadowRoofAt(float3 anchors, float t)
{
    if (t <= NEXT_SHADOW_ROOF_SPAN1)
        return lerp(anchors.x, anchors.y, t / NEXT_SHADOW_ROOF_SPAN1);
    return lerp(anchors.y, anchors.z,
                saturate((t - NEXT_SHADOW_ROOF_SPAN1) /
                         (NEXT_SHADOW_ROOF_SPAN2 - NEXT_SHADOW_ROOF_SPAN1)));
}

inline float ShadowMarching(float dens, float3 p, float2 densityChunk, float3 densityDetail, float3 scaleChunk, float3 scaleDetail, float3 shift, float4 profile, float3 dens_thres, const in float3 offsetA, const in float3 offsetB, const in float3 offsetC, float3 SunDirection, float4 shadow, float2 volumeBox, float2 solidness, float4 distortionParm, float roofAtSample, float lightLod, int layerIndex, float cloudTime) {
    
    //exit for low density
    if(dens<=0.025) return dens*shadow.x;  //return a approx. value
    
    float threshold = 2.0f/shadow.w/shadow.x;
    
    float d = 0.0;
    float4 step = float4(SunDirection.xyz * shadow.x, shadow.x);
    
    //sampling
    float maxTop = volumeBox.x + (layerIndex == 0 ? HSE_MAX_RISE + 100.0 : 0.0);

    // NEXT (4): one cached roof model for the whole light ray instead of a
    // fresh 9-event evaluation at each of the 31 shadow steps.  The roof at the
    // sample itself is already known from the view march, so it is passed in
    // rather than re-evaluated - see NEXT_SHADOW_ROOF.
    float3 org = p;
    float t = 0.0;
    float active = (layerIndex == 0) ? 1.0 : 0.0;
#if NEXT_SHADOW_ROOF == 3
    HSEAnchor roof = HSEAnchorInit(org, SunDirection, 0.0, cloudTime,
                                   volumeBox.x, layerIndex, NEXT_ROOF_SPAN_XY_SHADOW);
#else
    // Anchors at t = 0 / SPAN1 / SPAN2; the first is free.
    float3 shRoof = float3(roofAtSample, roofAtSample, roofAtSample);
  #if NEXT_SHADOW_ROOF >= 1
    shRoof.y = HSERoofEval(active, volumeBox.x, org, SunDirection, NEXT_SHADOW_ROOF_SPAN1, cloudTime);
    shRoof.z = shRoof.y;
  #endif
  #if NEXT_SHADOW_ROOF >= 2
    shRoof.z = HSERoofEval(active, volumeBox.x, org, SunDirection, NEXT_SHADOW_ROOF_SPAN2, cloudTime);
  #endif
#endif

    // NEXT (1): conservative gains for the shadow taps.  The fine loop uses
    // DetailA only (scaled by shadow.y); the coarse loop uses a fixed 0.5.
    float chunkGain = densityChunk.x + densityChunk.y;
    float fineGain  = densityDetail.x * shadow.y;

#if NEXT_LIGHT_LOD
    int fineSteps   = int(lerp(float(NEXT_LIGHT_FINE_MIN),
                               float(SHADOW_FINE_MARCH_STEP), lightLod) + 0.5);
    int coarseSteps = int(lerp(float(NEXT_LIGHT_COARSE_MIN),
                               float(SHADOW_COARSE_MARCH_STEP), lightLod) + 0.5);

    // Fewer steps must cover the SAME distance, or the light ray simply stops
    // short and the cloud lights up.  d is a plain sum that gets multiplied by
    // one step length at the end, so lengthening a step by k means it has to
    // count k times.  Measured: dropping the step counts without this is a
    // +0.23% shift in mean luminance, which is already half the gate budget.
    float kFine   = float(SHADOW_FINE_MARCH_STEP)   / float(fineSteps);
    float kCoarse = float(SHADOW_COARSE_MARCH_STEP) / float(coarseSteps);
  #if NEXT_LIGHT_OCTAVE_LOD
    // The chunk's second octave is the expensive one - it is sampled in BOTH
    // loops - so it survives longest; the fine detail octave goes first.
    float octChunk  = saturate(lightLod * 1.6 - 0.3);
    float octDetail = saturate(lightLod * 1.6 - 0.6);
  #else
    float octChunk = 1.0;
    float octDetail = 1.0;
  #endif
#else
    int fineSteps = SHADOW_FINE_MARCH_STEP;
    int coarseSteps = SHADOW_COARSE_MARCH_STEP;
    float kFine = 1.0;
    float kCoarse = 1.0;
    float octChunk = 1.0;
    float octDetail = 1.0;
#endif

    // FINE Shadow Marching
    for(int i=0;
        d<threshold && p.z<maxTop && p.z>volumeBox.y && i<fineSteps; i++)
    {
        p += step.xyz * kFine;
        t += step.w * kFine;

#if NEXT_SHADOW_ROOF == 3
        HSEAnchorSync(roof, org, SunDirection, t, cloudTime);
        float localTop = HSEAnchorAt(roof, org, SunDirection, t, cloudTime);
#else
        float localTop = ShadowRoofAt(shRoof, t);
#endif

        float2 localBox;
        float4 localShape;
        HSEApplyLocalTop(volumeBox, profile, localTop, active, localBox, localShape);

        if (p.z > localBox.x)
            break;

        float4 cs = CloudShape(p.z, localShape, dens_thres);
        float bandLo = cs.z - shadow.z;

#if NEXT_DETAIL_EARLY_OUT
        if (min(1.0, chunkGain*cs.x)*(1.0 + fineGain) <= bandLo)
            continue;
#endif

        float d1 = ChunkLod(p, densityChunk, scaleChunk, shift, offsetA, offsetB, cs.x, octChunk);

#if NEXT_DETAIL_EARLY_OUT
        if (d1*(1.0 + fineGain) <= bandLo)
            continue;
        if (d1 >= cs.y)                 // saturated: the detail tap is clamped away
        {
            d += DensityMix(d1, p.z, bandLo, cs.y, localBox, solidness) * kFine;
            continue;
        }
#endif

        float d2 = densityDetail.x * 0.5 * shadow.y;         // expectation of the octave
        if (octDetail > 0.0)
        {
            float3 displace = Distortion(d1, distortionParm);
            d2 = lerp(d2, DetailA(p, densityDetail, scaleDetail, offsetC, displace) * shadow.y,
                      octDetail);
        }
        d += DensityMix(DensityField(d1, d2), p.z, bandLo, cs.y, localBox, solidness) * kFine;
    }

    // COARSE Shadow Marching
    for(int i=0;
        d<threshold && p.z<maxTop && p.z>volumeBox.y && i<coarseSteps; i++)
    {
        p += step.xyz*5.0f*kCoarse;
        t += step.w*5.0f*kCoarse;

#if NEXT_SHADOW_ROOF == 3
        HSEAnchorSync(roof, org, SunDirection, t, cloudTime);
        float localTop = HSEAnchorAt(roof, org, SunDirection, t, cloudTime);
#else
        float localTop = ShadowRoofAt(shRoof, t);
#endif

        float2 localBox;
        float4 localShape;
        HSEApplyLocalTop(volumeBox, profile, localTop, active, localBox, localShape);

        if (p.z <= localBox.x)
        {
            float4 cs = CloudShape(p.z, localShape, dens_thres);
            float bandLo = cs.z - shadow.z;

#if NEXT_DETAIL_EARLY_OUT
            if (min(1.0, chunkGain*cs.x)*1.5 <= bandLo)
                continue;
#endif

            float d1 = ChunkLod(p, densityChunk, scaleChunk, shift, offsetA, offsetB, cs.x, octChunk);
            d += DensityMix(DensityField(d1, 0.5), p.z, bandLo, cs.y, localBox, solidness) * kCoarse;
        }
    }
    return d*shadow.w*step.w;
}

//================ NEXT: shared accumulation of one view-ray segment ================
//
// Split out of the march so that the ordinary path and the surface-substep
// replay path deposit energy through exactly the same code.
//
// "seg" is the TRUE length of the segment that ends at p.  The old marcher
// weighted every trapezoid by the constant march.x, which made the accumulated
// opacity a function of the step COUNT rather than of the distance travelled -
// any change to the step planner would have silently changed cloud density.
//
// shadowStride > 1 re-uses the last light-march result for the next few
// samples.  ShadowMarching() integrates the optical depth of everything ABOVE
// the sample, which varies far more slowly along the ray than the local
// density does, so holding it over a few Nyquist substeps is cheap and benign.
//
void CloudAccumulate(inout float3 fx, inout float4 d, inout float last_sample,
                     inout float2 wacc, inout float shadowCache, inout int shadowCount,
                     float dens, float seg, float3 p, float depth, int shadowStride,
                     CloudProfile a, float3 lightDirection, float roofAtSample, float lod,
                     int layerIndex, float cloudTime)
{
    // Nothing to deposit, and no open trapezoid to close.
    if (dens <= 0.0 && last_sample <= 0.0)
        return;

#if NEXT_TRUE_SEGMENT
    float c_dens = (dens + last_sample) * 0.5f * seg * NEXT_OPTICAL_SCALE;
#else
    float c_dens = (dens + last_sample) * a.march.x / 2.0f;
#endif

    last_sample = dens;

    //fade into near objects
    if (d.x >= depth)
        c_dens *= d.z / max(d.w, 0.001);

    // Transmittance at the START of this segment - see the in-scatter below.
    float transBefore = exp(-fx.y);

#if NEXT_WEIGHTED_DEPTH
    // Opacity-weighted mean of the sample distances (Nubis Evolved p.187):
    //     distance_sum += d[n]*sample_opacity;  weighted = distance_sum/opacity_sum
    // The original was  d.y = d.y*(1-T) + T*d.x , an exponential moving average
    // whose result depends on HOW MANY times it runs.  Since d.y drives the
    // atmospheric blend, that made the cloud colour step-count dependent - the
    // single largest source of non-convergence in the top-down views.
    float wseg = transBefore * (1.0 - exp(-c_dens));
    wacc += float2(d.x*wseg, wseg);
#else
    if (c_dens > 0.0f)
        d.y = d.y * (1.0 - fx.z) + fx.z * d.x;
#endif

    fx.y += c_dens;
    fx.z = (exp(-fx.y) - a.cutoff.y) / (1.0f - a.cutoff.y);
    d.z = depth - d.x;

    //lighting strength
    if (fx.y < 2.3 && dens > 0.0)
    {
        // Weight this sample contributes to the in-scattered radiance.  It is
        // also the bound on the error of re-using a stale light march.
        float wseg2 = transBefore * (1.0 - exp(-c_dens));

#if NEXT_LIGHT_LOD
        // The opacity axis keys on the transmittance in FRONT of this sample,
        // not on the sample's own weight: individually thin samples contribute
        // little but collectively carry the look, so keying on per-sample weight
        // degrades all of them at once and the errors add instead of cancelling.
        float lightLod = min(clampMap(lod,         1.0, NEXT_LIGHT_LOD_MAXLOD, 1.0, 0.0),
                             clampMap(transBefore, NEXT_LIGHT_LOD_THI, NEXT_LIGHT_LOD_TLO, 1.0, 0.0));
#else
        float lightLod = 1.0;
#endif

        if (shadowCount <= 0 && (wseg2 > NEXT_LIGHT_CULL || shadowCache < 0.0))
        {
            // ShadowMarching()'s low-density early-out was tuned against the
            // old fixed-width c_dens, so probe it with the same scale.
            shadowCache = ShadowMarching(dens*a.march.x, p, a.densityChunk, a.densityDetail,
                                         a.scaleChunk, a.scaleDetail, a.cloudShift, a.shape,
                                         a.range, a.offsetA, a.offsetB, a.offsetC, lightDirection,
                                         a.shadow, a.volumeBox, a.solidness, a.distortion,
                                         roofAtSample, lightLod, layerIndex, cloudTime);
            shadowCount = shadowStride;
        }
        shadowCount--;

#if NEXT_ENERGY_CONSERVING
        // Analytic in-scatter of a segment of constant density:
        //
        //     integral_0^L  T_start * exp(-sigma*s) * J  ds
        //         =  T_start * (1 - exp(-tau)) * J / sigma
        //
        // The original form was  tau * exp(-(tau_total_AFTER_this_segment)),
        // i.e. it attenuated the segment by the transmittance at its far end
        // and used tau instead of (1-exp(-tau)).  Both make the accumulated
        // brightness a function of the STEP COUNT: halve the step and the
        // clouds get brighter.  That coupling has to go before any adaptive
        // step scheme can be validated against a converged reference.
        // Nubis uses the same pre-update transmittance (Nubis Evolved p.61).
        fx.x += exp(-max(shadowCache, 0.0)) * wseg2;
#else
        fx.x += c_dens * exp((-max(shadowCache, 0.0) - fx.y));
#endif
    }
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

float4 CloudAtRay(CloudProfile a, CloudBaseColor b, const in float3 cam_dir, in float3 cam_pos, float3 light, float3 lightDirection, float time, int layerIndex, inout float depth) {   // NOTE: inout - depth is READ by the loop condition below

    float4 d = float4(0.0, 0.0, 0.0, a.march.y); //distance, approx. distance, gap, the last march step
    
    float maxVolumeTop = a.volumeBox.x + (layerIndex == 0 ? HSE_MAX_RISE + 100.0 : 0.0);
    float3 p = PosOnPlane(cam_pos, cam_dir, clamp(cam_pos.z, a.volumeBox.y+0.001, maxVolumeTop-0.001), d.x);
    d.y = d.x;

    if (d.x >= 0.0) {

        a.range.x = 1.0f/a.range.x;

        float3 fx = float3(0.0, 0.0, 1.0); //brightness, cumulative density, transparency

        float last_sample = 0.0, pdf = 0.0;

        // ---- NEXT: band-limit constants of this profile ---------------------
        // Conservative gains of the two noise stages (see SampleCloudDensity).
        float chunkGain  = a.densityChunk.x + a.densityChunk.y;
        float detailGain = (a.densityDetail.x + a.densityDetail.y)*(1.0 + a.densityDetail.z);

        // The Nyquist spacing of the finest octave actually in use.  Note the
        // domain warp in DetailB() locally amplifies frequency, which is why
        // the author's march.x is kept as a hard ceiling rather than trusted
        // to be the true band limit.
        float finestScale = max(max(a.scaleDetail.x, a.scaleDetail.y), a.scaleDetail.z);
        float hNyq = min(a.march.x, NEXT_NYQUIST_SAFETY / max(finestScale, 0.000001));

        float roofSlope = (layerIndex == 0) ? NEXT_ROOF_SLOPE : 0.0;
        float dirXY = length(cam_dir.xy);

        HSEAnchor roof = HSEAnchorInit(cam_pos, cam_dir, d.x, time,
                                       a.volumeBox.x, layerIndex, NEXT_ROOF_SPAN_XY);

        float seg     = 0.0;    // true length of the segment that ends at p
        float qPrev   = 0.0;    // band coordinate of the previous sample
        float gRun    = 0.0;    // sticky Lipschitz estimate of dq/ds
        bool  hasPrev = false;
        bool  inCloud = false;
        float2 wacc = float2(0.0, 0.0);   // (sum of dist*opacity, sum of opacity)
        float shadowCache = -1.0;   // < 0 = never marched yet
        int   shadowCount = 0;

        for (int i = 0; fx.z > 0.0 && p.z <= maxVolumeTop && p.z >= a.volumeBox.y && i < int(a.march.w) && d.x - d.w < depth && d.x < a.fade; i++) {

            // Screen-footprint LOD: at range a cloud feature is smaller than a
            // pixel, so the band limit relaxes with distance.  Every length in
            // the planner AND in the surface back-off scales with it.
            float lod  = clampMap(d.x, 0.0, a.fade, 1.0, a.march.z);
            float hMin = hNyq      * lod;
            float hMax = a.march.y * lod;

            HSEAnchorSync(roof, cam_pos, cam_dir, d.x, time);
            float localTop = HSEAnchorAt(roof, cam_pos, cam_dir, d.x, time);

            float2 localBox;
            float4 localShape;
            HSEApplyLocalTop(a.volumeBox, a.shape, localTop, roof.active, localBox, localShape);

            float3 cs = CloudShape(p.z, localShape, a.range).xyz;
            float bandLo = cs.z;
            float bandHi = cs.y;

            float df, dens;
            SampleCloudDensity(p, a, cs.x, bandLo, bandHi, chunkGain, detailGain, localBox, df, dens);

            float segCur = seg;

#if NEXT_SURFACE_BUDGET
            //---------------------------------------------------------------
            // (2) Surface step budget.
            //
            // A long inter-cloud step has just landed inside a cloud, so the
            // boundary is somewhere behind us.  Walk back in fixed Nyquist
            // substeps until the field falls out of the band again - no root
            // finding, no ordering, bounded worst case - and keep every
            // substep density in a small array.  The array is then replayed
            // forward, so the entry is integrated at Nyquist for the price of
            // the back-off alone, and the shadow march over it is strided.
            //---------------------------------------------------------------
            if (!inCloud && dens > 0.0 && seg > hMin*NEXT_SURF_TRIGGER && fx.z > NEXT_SURF_MIN_T)
            {
                float surfDens[NEXT_SURF_MAX];

                float tHit = d.x;
                float tOut = tHit - seg;                        // last known-empty sample
                float jit  = NEXT_SURF_JITTER * NextHash(p);
                int   n    = 0;

                for (int k = 1; k <= NEXT_SURF_MAX; k++)
                {
                    float tb = tHit - (float(k) - jit)*hMin;
                    if (tb <= tOut) break;                      // never step past the last sample

                    float3 pb = cam_pos + cam_dir*tb;

                    float topB = HSEAnchorAt(roof, cam_pos, cam_dir, tb, time);
                    float2 boxB;
                    float4 shapeB;
                    HSEApplyLocalTop(a.volumeBox, a.shape, topB, roof.active, boxB, shapeB);

                    float3 csB = CloudShape(pb.z, shapeB, a.range).xyz;

                    float dfB, densB;
                    SampleCloudDensity(pb, a, csB.x, csB.z, csB.y, chunkGain, detailGain, boxB, dfB, densB);

                    surfDens[k-1] = densB;
                    n = k;

                    if (densB <= 0.0) break;                    // recovered the surface
                }

                // Replay forward.  The samples sit on a uniform hMin lattice,
                // so walking the array backwards is already the correct order.
                last_sample = 0.0;
                for (int k = NEXT_SURF_MAX; k >= 1; k--)
                {
                    if (k > n) continue;

                    float tb = tHit - (float(k) - jit)*hMin;
                    d.x = tb;
                    d.w = hMin;

                    CloudAccumulate(fx, d, last_sample, wacc, shadowCache, shadowCount,
                                    surfDens[k-1], hMin, cam_pos + cam_dir*tb, depth,
                                    NEXT_SURF_SHADOW_STRIDE, a, lightDirection,
                                    HSEAnchorAt(roof, cam_pos, cam_dir, tb, time), lod, layerIndex, time);
                }

                d.x = tHit;
                if (n > 0)
                {
                    segCur = (1.0 - jit)*hMin;                  // gap left to the coarse sample
                    d.w = segCur;
                    hasPrev = false;                            // the finite difference is stale
                }
            }
#endif

            CloudAccumulate(fx, d, last_sample, wacc, shadowCache, shadowCount,
                            dens, segCur, p, depth,
                            NEXT_SHADOW_STRIDE, a, lightDirection, localTop, lod, layerIndex, time);

            inCloud = (dens > 0.0);

            //---------------------------------------------------------------
            // (3) Regional band-limit step planner.
            //
            // What a region needs is not "short steps where it is dense" but
            // short steps where the RENDERED density still has bandwidth.
            // DensityMix() clamps outside [bandLo, bandHi], so measure how far
            // the field is from that band in band widths:
            //
            //     q    = (df - bandCentre)/bandWidth     ( +-0.5 at the edges )
            //     outQ = |q| - 0.5                       ( >0 => clamped )
            //
            // and divide by an empirical Lipschitz bound of dq/ds taken from
            // the backward difference along the ray - free, no extra taps.
            // That is sphere tracing on the rendered field, so empty space AND
            // saturated cores both get long steps, while the thin transition
            // shell that actually carries the silhouette gets Nyquist steps.
            //
            // Using q rather than df also absorbs the motion of the band
            // itself (the vertical profile moves bandLo/bandHi as z changes).
            //---------------------------------------------------------------
            float bandW = max(bandHi - bandLo, NEXT_BAND_EPS);
            float q     = (df - 0.5*(bandLo + bandHi)) / bandW;
            float outQ  = abs(q) - 0.5;

            float gq = hasPrev ? abs(q - qPrev)/max(segCur, 1.0) : 0.0;
            gRun  = max(gq, gRun*NEXT_GRAD_MEMORY);
            qPrev = q;
            hasPrev = true;

            float h;
#if NEXT_BAND_PLANNER
            h = NEXT_REACH_SAFETY * max(outQ, 0.0) / max(gRun, NEXT_GRAD_FLOOR);

            // Transport still has to be resolved even where the field does
            // not: cap the optical depth deposited per step.  The cap relaxes
            // as the ray goes opaque, because what is behind is attenuated.
            h = min(h, NEXT_TAU_BUDGET / max(dens*fx.z, NEXT_DENS_EPS));

            h = clamp(h, hMin, hMax);
#else
            //legacy dynamic step length
            h = clampMap(2.0*df - pdf, bandLo*0.85, a.cutoff.x, a.march.x, a.march.y) * lod;
            h += 2.0*noise2d(p)*a.march.x;
#endif
            pdf = df;

#if NEXT_BAND_PLANNER
            // Jitter only ever SHORTENS the step, so the plan stays band
            // limited while the sample lattice stays decorrelated.
            h *= lerp(1.0 - NEXT_JITTER, 1.0, NextHash(p + 7.31));
#endif

#if NEXT_ROOF_SKIP
            //---------------------------------------------------------------
            // (5) Cone-step empty-space skip above the animated roof.
            // Solve for the first t at which the ray can meet a roof that is
            // allowed to rise at roofSlope:
            //     p.z + t*dir.z <= roofTop + roofSlope*t*|dir.xy|
            // If the ray outruns the cone it can never re-enter the layer.
            //---------------------------------------------------------------
            if (p.z > localTop)
            {
                float denom = roofSlope*dirXY - cam_dir.z;
                float hSkip = (denom > 0.0001) ? (p.z - localTop)/denom : NEXT_BIG_SKIP;
                h = max(h, hSkip);
            }
#endif
            h = max(h, hMin*0.25);      // never stall

            //marching
            seg  = h;
            d.w  = h;
            p   += cam_dir * h;
            d.x += h;
        }
        
        if (fx.z < 1.0) {
            fx = saturate(fx);

#if NEXT_WEIGHTED_DEPTH
            if (wacc.y > 0.0)
                d.y = wacc.x / wacc.y;
#endif
            
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

float4 RenderClouds(float3 ViewDir, float3 ViewPos, float3 sunDir, float ViewZ, float cloudTime)
{
    
    CloudColorVarying(sunDir, baseColor);
    
    float3 lightDir = float3(0.0);
    float3 light = LightSource(sunDir, lightDir);

    float4 cloud_temp = float4(0.0, 0.0, 0.0, 1.0);
        
    int i = 0;//(ViewPos.z>3500.0? 2:(ViewPos.z>1500.0? 1:0));
    float4 clouds = CloudAtRay(cloudProfiles[i], baseColor, ViewDir, ViewPos, light, lightDir, cloudTime, i, ViewZ);
        
    /*i = (ViewPos.z>3500.0? 1:(ViewPos.z>1500.0? 0:1));
    cloud_temp = CloudAtRay(cloudProfiles[i], baseColor, ViewDir, ViewPos, light, lightDir, cloudTime, i, ViewZ);
    clouds.rgb += cloud_temp.rgb*clouds.w;
    clouds.w = clouds.w*cloud_temp.w;
    
    i = (ViewPos.z>3500.0? 0:(ViewPos.z>1500.0? 2:2));
    cloud_temp = CloudAtRay(cloudProfiles[i], baseColor, ViewDir, ViewPos, light, lightDir, cloudTime, i, ViewZ);
    clouds.rgb += cloud_temp.rgb*clouds.w;*/
    clouds.w = clouds.w*cloud_temp.w;
    
    return clouds;
}

float fov=1.6;

void timeEffect(float t, inout CloudProfile c)   // NOTE: inout - c is read here
{
    c.offsetA*=t;
    c.offsetB*=t;
    c.offsetC*=t;
    c.offsetD*=t;
}

vec3 getViewRay(vec2 fragCoord, float fov, out float ang_v)
{
    vec2 uv = (fragCoord*2.0 - iResolution.xy)/iResolution.x;
    vec3 d = normalize(vec3(-uv.x, 1.0/fov, uv.y+0.4));
  
    return d;
    #ifdef NO_MOUSE
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
    
    float time = gameTime(iTime*100.0)+5000.0;
    timeEffect(time, cloudProfiles[0]);
    timeEffect(time, cloudProfiles[1]);
    timeEffect(time, cloudProfiles[2]);

    CloudRandomness(time);
    
    float a = 2.0*3.14159/iResolution.x,
        b = 2.0*3.14159/iResolution.y;
        
   
    float ang_v = (fragCoord.y + 40.0)*fov*b;
    
    vec3 CameraPosition = vec3(25000.0,0.0,0.0);
    vec3 direction = getViewRay(fragCoord, fov, ang_v);
    float dist=0.0;
    vec2 dest = PosOnPlane(CameraPosition,direction,20.0,dist).xy;
    
    vec3 c = OutterSpace(direction, vSunLightDir);
    c = Sky(c, Game2Atm_Alt(float3(0.0)), direction, vSunLightDir, earth);
    
    vec4 clouds = RenderClouds(direction, CameraPosition, vSunLightDir, 9999999999.0, time);
    c = clouds.xyz+c*clouds.w;
    
    fragColor = vec4(c, 1.0);
}