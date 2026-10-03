/*===================================================================================
Paper 3 - Pseudo-Spherical Multiple Scattering
  "A Hybrid Two-Stream and Successive-Orders Solver"

This replaces CloudWorks' hand-rolled two-term Rayleigh/Mie sky wholesale.  What
the reference solver does, and what is ported here (following
articles/3_pseudo_spherical_ms/program/3. shader_complete.glsl):

  * Exact single scattering   - Rayleigh phase, analytic two-branch Chapman beam
                                (Papers 1-2, erfcx + uniform asymptotics), ozone
                                Chappuis absorption on the slant column.
  * Multiple scattering       - a coarse DIRECTIONAL GATHER of the single-
                                scattered sky, i.e. second-order in-scattering
                                computed from real directions so the diffuse
                                airlight is correctly anisotropic and arrives
                                from the sunlit part of the sky, plus a
                                geometric series for orders >= 3.
  * Day / twilight branch     - at theta_s ~ 88 deg the MS weight and the number
                                of effective orders rise, matching the finding
                                that twilight is MS-dominated (memory
                                exp4-switch-criterion / twilight-branch).

A single fragment pass cannot host the per-column two-stream BVP or the DOM
iteration, so the STRUCTURE and the physics are ported, not the solver.

WHY THE CLOUDS CARE.  The one function below that CloudWorks did not have is
PSMSIntegrate() over a FINITE segment.  Run it from the eye to the cloud and it
returns both the transmittance of that path and the airlight scattered into it.
As the cloud gets further away the transmittance falls to zero and the airlight
converges on the sky radiance for that direction, so the cloud and the sky
become the same colour on their own - no distance-fog hack.  The same function
with tMax = infinity is the sky itself, so the two can never disagree.

Units are megametres (Mm).  CloudWorks world space is GTA metres with +Z up, so
planet space is (0, 0, groundHeight + z_game/game2atm) - see Game2Atm_Alt.
===================================================================================*/

static const float planetRadius = 6.371;
static const float atmRadius    = 6.471;
static const float Hatm         = 0.008;      // Rayleigh scale height, 8 km
static const float Csw          = 1.12;       // Chapman branch switch

static const float3 betaR = float3(5.802, 13.558, 33.10);   // Rayleigh scattering /Mm
static const float3 betaO = float3(3.426, 8.298, 0.356);    // ozone Chappuis absorption /Mm

static const float ozC = 0.025;               // ozone layer centre, Mm
static const float ozW = 0.030;               // ozone layer half-width, Mm
// Ozone amount.  Paper 3's shader uses 0.30; Paper 6 independently found that
// value turns clouds purple and settled on 0.060.  Confirmed here: at 0.30 the
// cumulonimbus in gate scene B renders frankly magenta, because betaO absorbs
// green ~2.4x harder than red and 24x harder than blue, so an inflated ozone
// column eats the green out of the sunlight before it reaches the cloud.
// 0.30 is simply too much ozone: the model's vertical column is ozA*ozW, so
// 0.30 gives a green Chappuis optical depth of 0.30*0.030*8.298 = 0.075 against
// a real atmosphere's ~0.025.  0.10 reproduces the real column.
static const float ozA = 0.10;

// Day/twilight branch.  fDay/fTwi are the geometric-series fractions for
// scattering orders >= 3, so the gain is 1/(1-f).  Twilight is MS-dominated.
static const float switchDeg = 88.0;
static const float fDay      = 0.55;
static const float fTwi      = 0.72;

#define PSMS_MS_WEIGHT      1.0     // overall multiple-scattering strength
#define PSMS_SKY_STEPS      40      // view-ray steps for the full sky
#define PSMS_AERIAL_STEPS   24      // view-ray steps for the eye->cloud segment
#define PSMS_PROBE_STEPS    12      // view-ray steps for the ambient probes
#define PSMS_GATHER_ELEV    3       // gather elevations (Paper 3 uses 3)
#define PSMS_GATHER_AZIM    2       // gather azimuths  (toward / away from the sun)
// Evaluate the diffuse gather every Nth march step and interpolate (1 = every
// step, i.e. off).  J is by construction a SMOOTH function of position - that
// is what makes it the diffuse term - so it does not need resolving at the
// march rate, while it costs GATHER_ELEV*GATHER_AZIM Chapman beams against one
// for the whole rest of a step.  Profiling put the gather at 40% of the shader.
#define PSMS_GATHER_LERP    1
// Same argument for the direct solar transmittance: one Chapman beam per step,
// and along a view ray both the altitude and the solar zenith vary smoothly.
// Kept separate from the gather because this one feeds the SINGLE-scattering
// term, which is the sharp part of the sky - it tolerates far less.
#define PSMS_SUNTRANS_LERP  1

// Grey aerosol.  NOT part of Paper 3, which is a Rayleigh + ozone model; set
// PSMS_AEROSOL to 0 to get the paper exactly.  It is on by default because this
// shader is used from ground level, where without a boundary-layer haze the
// horizon stays implausibly clear and distant clouds refuse to wash out.
#define PSMS_AEROSOL        1
static const float Haer   = 0.0016;           // aerosol scale height, 1.6 km
static const float betaMs = 21.0;             // aerosol extinction /Mm
static const float aerG   = 0.76;             // aerosol anisotropy

// CloudWorks carries the sun as 685 * colour; Paper 3's radiometric scale has
// sunIntensity 22 against 1-exp(-L) tonemapping.  Rescale once, here, so the
// sky and the cloud lighting are on one scale and cannot drift apart.
#define PSMS_SUN_INTENSITY  22.0
// Paper 3's scale is written against its own 1-exp(-L) tonemap with a low sun.
// At the daylight elevations this shader is used at, that saturates: one
// exposure constant brings the whole model onto CloudWorks' range.  It is
// applied to the LIGHT, so the sky, the ambient probes, the aerial perspective
// and the sunlight on the clouds all move together and cannot drift apart.
#define PSMS_EXPOSURE       0.15
#define PSMS_LIGHT_SCALE    (PSMS_SUN_INTENSITY/685.0*PSMS_EXPOSURE)

// Sky radiance probes, filled once per pixel in mainImage.  The cloud's ambient
// term is the actual sky it can see, MS included - not a painted constant.
float3 gAmbZenith  = float3(0.0);
float3 gAmbHorizon = float3(0.0);

// How much of the ambient dome is horizon rather than zenith.  The dome average
// is horizon-heavy: near the horizon the sky is several times brighter than at
// the zenith, and a cloud in a thin deck sees mostly that band.
#define PSMS_AMB_ZENITH_MIX 0.35

//--------------------------------------------------------------------
// Two-branch uniform Chapman optical depth (Papers 1-2), occlusion-aware.
float erfcx(float y)
{
    if (y < 1.5)
    {
        float y2 = y*y, t = y, s = y;
        for (int n = 1; n < 10; n++) { t *= -y2/float(n); s += t/float(2*n+1); }
        return exp(y2)*(1.0 - 1.12837916709551*s);
    }
    float f = y;
    for (int k = 12; k >= 1; k--) f = y + 0.5*float(k)/f;
    return 0.564189583547756/f;
}

float chapmanPos(float r0, float mu)
{
    float s = sqrt(max(0.0, 1.0 - mu*mu));
    float X = r0/Hatm, h0 = r0 - planetRadius, eps = 1.0/X;
    if (s < 1e-4) return Hatm*exp(-h0/Hatm);

    if (1.0/s > Csw)                                   // steep branch
    {
        float m2 = mu*mu;
        float c1 = 1.0 - 1.0/m2;
        float c2 = 3.0*(1.0 - m2)/(m2*m2);
        float c3 = 3.0*(m2*m2 + 4.0*m2 - 5.0)/(m2*m2*m2);
        return Hatm*exp(-h0/Hatm)*(1.0 + eps*(c1 + eps*(c2 + eps*c3)))/mu;
    }
    // Grazing branch.  eta uses X*mu^2/(1+s) rather than sqrt(2*(X - X*s)):
    // the latter can go slightly negative in float32 near the horizon and the
    // resulting NaN shows up as GPU-dependent black stripes 90 deg from the sun.
    float z = X*s, p = r0*s, eta = sqrt(2.0*X*mu*mu/(1.0 + s)), iz = 1.0/z;
    float P = 1.0 + iz*(0.375 + iz*(-0.1171875 + iz*(0.1025390625 + iz*(-0.144195556640625))));
    float e2 = eta*eta;
    float Q = iz*( 0.375*eta
            + iz*( -0.0390625*(e2*eta + 3.0*eta)
            + iz*(  0.0068359375*(e2*e2*eta + 5.0*e2*eta + 15.0*eta)
            + iz*( -0.001373291015625*(e2*e2*e2*eta + 7.0*e2*e2*eta + 35.0*e2*eta + 105.0*eta) ))));
    return sqrt(p*Hatm)*exp(-h0/Hatm)*(P*1.25331413731550*erfcx(eta*0.707106781186548) + Q);
}

float chapmanInf(float r0, float mu)
{
    if (mu < 0.0)
    {
        float p = r0*sqrt(max(0.0, 1.0 - mu*mu));
        if (p < planetRadius) return 1e20;             // shadowed by the planet
        return 2.0*chapmanPos(p, 0.0) - chapmanPos(r0, -mu);
    }
    return chapmanPos(r0, mu);
}

float2 raySphere(float3 ro, float3 rd, float ra)
{
    float b = dot(ro, rd), c = dot(ro, ro) - ra*ra, d = b*b - c;
    if (d < 0.0) return float2(1e9, -1e9);
    d = sqrt(d);
    return float2(-b - d, -b + d);
}

float ozoneDensity(float h) { return max(0.0, 1.0 - abs(h - ozC)/ozW)*ozA; }

float MiePhase(float c)   { return 1.0f + 1.6f*exp(20.0f*(c - 1.0f)); }
float HenyeyGreenstein(float c, float g)
{
    float g2 = g*g;
    return (1.0 - g2)/(12.566370614*pow(max(1.0 + g2 - 2.0*g*c, 1e-4), 1.5));
}

// Solar transmittance (Rayleigh + ozone) reaching a point, RGB.
float3 PSMSSunTrans(float3 pos, float r, float3 L)
{
    float sunOD   = chapmanInf(max(r, planetRadius), dot(L, pos/r));
    float airmass = sunOD/max(Hatm*exp(-(r - planetRadius)/Hatm), 1e-4);
    return exp(-(betaR*sunOD + betaO*(ozA*ozW)*airmass));
}

//--------------------------------------------------------------------
// Second-order diffuse source at P: a coarse directional gather of the
// single-scattered sky - the GPU stand-in for the twilight branch's coarse-DOM
// orders.  Returns the mean incident single-scattered radiance J(P), which
// already carries one betaR; the caller's in-scattering supplies the second, so
// this really is a beta^2 second-order term rather than a fudge.
float3 PSMSGather(float3 P, float3 up, float3 L)
{
    float3 sp = L - dot(L, up)*up;
    float3 t1 = (length(sp) > 1e-4) ? normalize(sp) : normalize(cross(up, float3(0.0, 0.0, 1.0)));
    float3 t2 = cross(up, t1);

    float3 J = float3(0.0);
    for (int ie = 0; ie < PSMS_GATHER_ELEV; ie++)
    {
        float e = 0.25 + float(ie)*0.5;                 // elevations ~14, 43, 72 deg
        float ce = cos(e), se = sin(e);
        for (int ia = 0; ia < PSMS_GATHER_AZIM; ia++)
        {
            float a = float(ia)*3.14159265;             // toward / away from the sun
            float3 d = ce*(cos(a)*t1 + sin(a)*t2) + se*up;

            float2 hit = raySphere(P, d, atmRadius);
            float Ld = max(0.0, hit.y)*0.4;             // representative depth along d
            float3 Q = P + d*Ld;
            float rq = length(Q), hq = rq - planetRadius;
            if (hq < 0.0) continue;                     // gather ray hit the ground

            float cph = dot(d, L);
            J += betaR * exp(-hq/Hatm) * PSMSSunTrans(Q, rq, L)
                       * (0.75*(1.0 + cph*cph)) * (Ld + 1e-4);
        }
    }
    return J / float(PSMS_GATHER_ELEV*PSMS_GATHER_AZIM);
}

//--------------------------------------------------------------------
// THE workhorse.  In-scattered radiance along [0, tMax] of a ray, plus the
// transmittance of that segment.  tMax >= 1e8 means "to the top of the
// atmosphere", i.e. the sky.  Anything shorter is aerial perspective.
//
// camera and the returned radiance are in planet space / Paper 3 units.
float3 PSMSIntegrate(float3 camera, float3 ray, float tMax, int steps,
                     float3 L, float3 lightIntensity, out float3 trans)
{
    trans = float3(1.0);

    float2 atm = raySphere(camera, ray, atmRadius);
    if (atm.y < 0.0 || atm.x > atm.y) return float3(0.0);

    float tS = max(atm.x, 0.0);
    float tE = min(atm.y, tS + tMax);

    // Planet occlusion.  Paper 3's shader tests `pl.x > tS`, which is fine for
    // its observer 2 km up, but CloudWorks puts the eye exactly ON the planet
    // radius.  There raySphere's c = dot(ro,ro) - R*R is zero to within float
    // noise, and BOTH strict and loose forms of the test misbehave: the strict
    // one lets a downward tangent ray march underground until exp(-h/Hatm)
    // overflows to inf, and the loose one lets float noise turn a RISING ray
    // into a spurious grazing hit, which blacks the pixel out.  Both were
    // observed - full-frame NaN, then isolated dark specks.
    //
    // Decide it geometrically instead.  From the surface only a downward ray is
    // occluded, and it is occluded immediately; from altitude the usual entry
    // test is well conditioned.
    float rc = length(camera);
    float3 up = camera/rc;
    float muView = dot(up, ray);
    if (rc <= planetRadius*(1.0 + 1e-6))
    {
        if (muView < 0.0) { trans = float3(0.0); return float3(0.0); }
    }
    else
    {
        float2 pl = raySphere(camera, ray, planetRadius);
        if (pl.y > 0.0 && pl.x > 0.0) tE = min(tE, pl.x);
    }
    if (tE <= tS) return float3(0.0);

    // Day / twilight branch, decided at the OBSERVER (theta_s = angle between
    // local up and the sun), exactly as the reference solver switches columns.
    float thetaS = degrees(acos(clamp(dot(L, up), -1.0, 1.0)));
    float msGain = PSMS_MS_WEIGHT / (1.0 - (thetaS > switchDeg ? fTwi : fDay));

    float c = dot(ray, L);
    float phaseR = 0.75*(1.0 + c*c);
#if PSMS_AEROSOL
    float phaseM = HenyeyGreenstein(c, aerG);
    float3 betaM = float3(betaMs, betaMs, betaMs);
#endif

    float ds = (tE - tS)/float(steps);
    float viewOD = 0.0, viewOz = 0.0, viewAe = 0.0;
    float3 single = float3(0.0), multi = float3(0.0), singleM = float3(0.0);

#if PSMS_GATHER_LERP > 1
    float3 Jprev = float3(0.0), Jnext = float3(0.0);
#endif
#if PSMS_SUNTRANS_LERP > 1
    float3 Sprev = float3(0.0), Snext = float3(0.0);
#endif

    for (int i = 0; i < steps; i++)
    {
        float3 pos = camera + ray*(tS + (float(i) + 0.5)*ds);
        float r = length(pos);
        float h = max(r - planetRadius, 0.0);   // clamped: h < 0 overflows exp()
        float dens = exp(-h/Hatm);

        viewOD += dens*ds;
        viewOz += ozoneDensity(h)*ds;
#if PSMS_AEROSOL
        float dae = exp(-h/Haer);
        viewAe += dae*ds;
        float3 Tview = exp(-(betaR*viewOD + betaO*viewOz + betaM*viewAe));
#else
        float3 Tview = exp(-(betaR*viewOD + betaO*viewOz));
#endif
#if PSMS_SUNTRANS_LERP > 1
        int ms_ = i - (i/PSMS_SUNTRANS_LERP)*PSMS_SUNTRANS_LERP;
        if (ms_ == 0)
        {
            Sprev = (i == 0) ? PSMSSunTrans(pos, r, L) : Snext;
            int js = min(i + PSMS_SUNTRANS_LERP, steps - 1);
            float3 ps = camera + ray*(tS + (float(js) + 0.5)*ds);
            float rs = max(length(ps), planetRadius);
            Snext = (js > i) ? PSMSSunTrans(ps, rs, L) : Sprev;
        }
        float3 Tsun = lerp(Sprev, Snext, float(ms_)/float(PSMS_SUNTRANS_LERP));
#else
        float3 Tsun = PSMSSunTrans(pos, r, L);
#endif

#if PSMS_GATHER_LERP > 1
        int m = i - (i/PSMS_GATHER_LERP)*PSMS_GATHER_LERP;
        if (m == 0)
        {
            Jprev = (i == 0) ? PSMSGather(pos, pos/r, L) : Jnext;
            int jn = min(i + PSMS_GATHER_LERP, steps - 1);
            float3 pn = camera + ray*(tS + (float(jn) + 0.5)*ds);
            float rn = max(length(pn), planetRadius);
            Jnext = (jn > i) ? PSMSGather(pn, pn/rn, L) : Jprev;
        }
        float3 Jdiff = lerp(Jprev, Jnext, float(m)/float(PSMS_GATHER_LERP));
#else
        float3 Jdiff = PSMSGather(pos, pos/r, L);
#endif

        single += dens * Tview * Tsun * ds;                       // (1) exact single
        multi  += dens * Tview * Jdiff * msGain * ds;             // (2) MS orders >= 2
#if PSMS_AEROSOL
        singleM += dae * Tview * Tsun * ds;
#endif
    }

    // single carries one betaR (applied here) x phaseR; multi's inner betaR plus
    // the outer one make it beta^2, and it re-scatters isotropically (no phase).
    float3 col = betaR*(single*phaseR + multi);
#if PSMS_AEROSOL
    col += betaM*singleM*phaseM;
    trans = exp(-(betaR*viewOD + betaO*viewOz + betaM*viewAe));
#else
    trans = exp(-(betaR*viewOD + betaO*viewOz));
#endif
    return lightIntensity * col;
}

//--------------------------------------------------------------------
// Sun / moon selection.  The atmosphere is driven by whichever body is up, so a
// moonlit night gets a dim blue sky out of the same solver rather than a
// special case.
float3 LightSource(float3 SunDirection, out float3 SourceDir)
{
    if (SunDirection.z < -0.075)
    {
        SourceDir = -SunDirection;
        return sunLightStrength*moonReflecticity*smoothstep(-0.075, -0.105, SunDirection.z);
    }
    SourceDir = SunDirection;
    return sunLightStrength*smoothstep(-0.075, -0.025, SunDirection.z);
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

// Stars and the moon disk.  The SUN disk is no longer drawn here - it comes out
// of the atmosphere integral, correctly reddened by its own slant path.
float3 OutterSpace(float3 ray, float3 SunDirection)
{
    float ang_cos = dot(ray, SunDirection);

    if (ang_cos < -0.99978)
        return 0.03*moonReflecticity*sunLightStrength*PSMS_LIGHT_SCALE
               * smoothstep(-0.99978, -0.99985, ang_cos);

    return StarSky(starStrength, 300.0, 6.0, WorldToSpace(ray, SunDirection))
           * smoothstep(0.05, -0.1, SunDirection.z) * PSMS_LIGHT_SCALE;
}

//--------------------------------------------------------------------
// Full sky along a view ray: the same integral, run to the top of the
// atmosphere, plus the sun disk attenuated by its own path.
float3 Sky(float3 background, float3 camera, float3 ray, float3 SunDirection)
{
    float3 lightDirection;
    float3 lightStrength = LightSource(SunDirection, lightDirection)*PSMS_LIGHT_SCALE;

    float3 trans;
    float3 col = PSMSIntegrate(camera, ray, 1e8, PSMS_SKY_STEPS,
                               lightDirection, lightStrength, trans);

    // Sun disk: transmittance along the view ray already accounts for the
    // reddening, so a low sun goes orange without any extra term.
    float c = dot(ray, lightDirection);
    if (c > 0.9997)
        col += lightStrength * 45.0 * trans * smoothstep(0.9997, 0.99985, c);

    return col + background*trans;
}
