// Complete hybrid sky -- a faithful GPU adaptation of the Paper 3 hybrid solver.
//
// The validated hybrid (see the paper) is a reference solver: a per-column two-stream BVP
// with a single-scattering correction by day, and exact single scatter plus k coarse-DOM
// multiple-scattering orders in twilight, switched near a solar zenith angle of 88 deg.
// A single Shadertoy Image pass cannot host the per-column BVP / DOM iteration, so this
// shader ports the STRUCTURE and physics faithfully:
//
//   * Exact single scattering    -- Rayleigh phase, analytic Chapman beam (Papers 1-2), ozone.
//   * Multiple scattering        -- a coarse DIRECTIONAL GATHER of the single-scattered sky
//                                   (the GPU analogue of the twilight branch's coarse-DOM MS:
//                                   second-order in-scattering computed from real directions,
//                                   so the diffuse airlight is correctly anisotropic and comes
//                                   from the sunlit part of the sky), plus a geometric series
//                                   for orders >= 3.
//   * Day / twilight switch      -- at theta_s ~ 88 deg the multiple-scattering weight and the
//                                   number of effective orders (the geometric-series gain) are
//                                   raised, matching the finding that twilight is MS-dominated.
//
// Approximations vs the reference: the diffuse gather uses one representative tap per
// direction (K = 6 directions) rather than a converged DOM, and orders >= 3 are summed as a
// geometric series rather than iterated. Tune GATHER count / VIEW_STEPS / gains to taste.
// VALID for thin atmospheres X = Rp/H >> 1 (here Rp/H ~ 796).

// ---- Earth, in megameters (Mm) ----
const float planetRadius = 6.371;
const float atmRadius    = 6.471;
const float H            = 0.008;      // Rayleigh scale height (8 km)
const float Csw          = 1.12;
const int   VIEW_STEPS   = 40;

const vec3  betaR        = vec3(5.802, 13.558, 33.10);     // Rayleigh scattering /Mm
const vec3  betaO        = vec3(3.426, 8.298, 0.356);      // ozone Chappuis absorption /Mm
const float ozoneCenter  = 0.025;
const float ozoneWidth   = 0.030;
const float ozoneAmount  = 0.30;

const float switchDeg    = 88.0;       // day/twilight branch switch (theta_s)
const float fDay         = 0.55;       // MS geometric-series fraction by day  -> gain 1/(1-f)
const float fTwi         = 0.72;       // MS geometric-series fraction in twilight (MS-dominated)
const float msWeight     = 1.0;        // overall multiple-scattering strength knob

const float sunIntensity = 22.0;
vec3 lightDirection = normalize(vec3(1.0, 0.0, 0.0));

//--------------------------------------------------------------------
// Two-branch uniform Chapman optical depth (Papers 1-2), occlusion-aware.
float erfcx(float y){
    if (y < 1.5){
        float y2 = y*y, t = y, s = y;
        for (int n = 1; n < 10; n++){ t *= -y2/float(n); s += t/float(2*n+1); }
        float erf = 1.12837916709551*s;
        return exp(y2)*(1.0 - erf);
    }
    float f = y;
    for (int k = 12; k >= 1; k--) f = y + 0.5*float(k)/f;
    return 0.564189583547756/f;
}
float chapmanPos(float r0, float mu){
    float s = sqrt(max(0.0, 1.0 - mu*mu));
    float X = r0/H, h0 = r0 - planetRadius, eps = 1.0/X;
    if (s < 1e-4) return H*exp(-h0/H);
    float coshT0 = 1.0/s;
    if (coshT0 > Csw){
        float m2 = mu*mu;
        float c1 = 1.0 - 1.0/m2;
        float c2 = 3.0*(1.0 - m2)/(m2*m2);
        float c3 = 3.0*(m2*m2 + 4.0*m2 - 5.0)/(m2*m2*m2);
        return H*exp(-h0/H)*(1.0 + eps*(c1 + eps*(c2 + eps*c3)))/mu;
    }
    // eta = sqrt(2X(1-s)); use X*mu^2/(1+s) (cancellation-free, always >= 0) instead of
    // sqrt(2*(X - X*s)), whose argument can go slightly negative in float32 near the
    // horizon (mu->0) and yield a NaN -> GPU-dependent black stripes near 90deg from the sun.
    float z = X*s, p = r0*s, eta = sqrt(2.0*X*mu*mu/(1.0 + s)), iz = 1.0/z;
    float P = 1.0 + iz*(0.375 + iz*(-0.1171875 + iz*(0.1025390625 + iz*(-0.144195556640625))));
    float e2 = eta*eta;
    float Q = iz*( 0.375*eta
            + iz*( -0.0390625*(e2*eta + 3.0*eta)
            + iz*(  0.0068359375*(e2*e2*eta + 5.0*e2*eta + 15.0*eta)
            + iz*( -0.001373291015625*(e2*e2*e2*eta + 7.0*e2*e2*eta + 35.0*e2*eta + 105.0*eta) ))));
    float y = eta*0.707106781186548;
    return sqrt(p*H)*exp(-h0/H)*(P*1.25331413731550*erfcx(y) + Q);
}
float chapmanInf(float r0, float mu){
    if (mu < 0.0){
        float p = r0*sqrt(max(0.0, 1.0 - mu*mu));
        if (p < planetRadius) return 1e20;         // shadowed by the planet
        return 2.0*chapmanPos(p, 0.0) - chapmanPos(r0, -mu);
    }
    return chapmanPos(r0, mu);
}
vec2 raySphere(vec3 ro, vec3 rd, float ra){
    float b = dot(ro, rd), c = dot(ro, ro) - ra*ra, d = b*b - c;
    if (d < 0.0) return vec2(1e9, -1e9);
    d = sqrt(d); return vec2(-b - d, -b + d);
}
float ozoneDensity(float h){ return max(0.0, 1.0 - abs(h - ozoneCenter)/ozoneWidth)*ozoneAmount; }

// Solar transmittance (Rayleigh + ozone) reaching a point, RGB.
vec3 sunTransmittance(vec3 pos, float r, float dens){
    float muSun = dot(lightDirection, pos/r);
    float sunOD = chapmanInf(r, muSun);
    float airmass = sunOD/max(H*dens, 1e-4);
    float sunOz = (ozoneAmount*ozoneWidth)*airmass;            // ozone slant column
    return exp(-(betaR*sunOD + betaO*sunOz));
}

//--------------------------------------------------------------------
// Second-order diffuse source at P: a coarse directional gather of the single-scattered
// sky (the GPU stand-in for the coarse-DOM multiple-scattering orders). Returns the
// mean incident single-scattered radiance J(P) (already carrying one betaR); the outer
// in-scattering supplies the second betaR, so this is a genuine second-order (beta^2) term.
vec3 gatherDiffuse(vec3 P, vec3 up){
    vec3 sp = lightDirection - dot(lightDirection, up)*up;
    vec3 t1 = (length(sp) > 1e-4) ? normalize(sp) : normalize(cross(up, vec3(0.0,0.0,1.0)));
    vec3 t2 = cross(up, t1);
    vec3 J = vec3(0.0);
    for (int ie = 0; ie < 3; ie++){
        float e = 0.25 + float(ie)*0.5;                        // elevations ~14,43,72 deg
        float ce = cos(e), se = sin(e);
        for (int ia = 0; ia < 2; ia++){
            float a = float(ia)*3.14159265;                    // toward / away from the sun
            vec3 d = ce*(cos(a)*t1 + sin(a)*t2) + se*up;
            vec2 hit = raySphere(P, d, atmRadius);
            float L = max(0.0, hit.y)*0.4;                     // representative depth along d
            vec3 Q = P + d*L;
            float rq = length(Q), hq = rq - planetRadius;
            if (hq < 0.0) continue;
            float densq = exp(-hq/H);
            vec3 Tsq = sunTransmittance(Q, rq, densq);
            float cph = dot(d, lightDirection);
            float ph = 0.75*(1.0 + cph*cph);                   // Rayleigh phase sun->d
            J += betaR * densq * Tsq * ph * (L + 1e-4);        // single-scattered radiance from d
        }
    }
    return J * (1.0/6.0);
}

vec3 atmosphere(vec3 camera, vec3 ray){
    vec2 atm = raySphere(camera, ray, atmRadius);
    if (atm.y < 0.0 || atm.x > atm.y) return vec3(0.0);
    float tStart = max(atm.x, 0.0), tEnd = atm.y;
    bool hitGround = false;
    vec2 pl = raySphere(camera, ray, planetRadius);
    if (pl.x > tStart && pl.x < tEnd){ tEnd = pl.x; hitGround = true; }

    // Day / twilight branch at the observer (theta_s = angle between local up and the sun).
    float rc = length(camera);
    float muSunObs = dot(lightDirection, camera/rc);
    float thetaS = degrees(acos(clamp(muSunObs, -1.0, 1.0)));
    bool  twilight = thetaS > switchDeg;
    float msGain = msWeight / (1.0 - (twilight ? fTwi : fDay));   // geometric series, orders>=3

    float c = dot(ray, lightDirection);
    float phaseR = 0.75*(1.0 + c*c);
    float ds = (tEnd - tStart)/float(VIEW_STEPS);
    float viewOD = 0.0, viewOz = 0.0;
    vec3 single = vec3(0.0), multi = vec3(0.0);

    for (int i = 0; i < VIEW_STEPS; i++){
        vec3 pos = camera + ray*(tStart + (float(i)+0.5)*ds);
        float r = length(pos), h = r - planetRadius;
        float dens = exp(-h/H);
        float ozo  = ozoneDensity(h);
        viewOD += dens*ds; viewOz += ozo*ds;
        vec3 Tview = exp(-(betaR*viewOD + betaO*viewOz));       // camera -> sample
        vec3 up = pos/r;

        // (1) exact single scattering
        vec3 Tsun = sunTransmittance(pos, r, dens);
        single += dens * Tview * Tsun * ds;

        // (2) multiple scattering: second-order directional gather + higher orders (gain)
        vec3 Jdiff = gatherDiffuse(pos, up);                   // incident diffuse (has one betaR)
        multi += dens * Tview * Jdiff * msGain * ds;           // isotropic re-scatter (no phase)
    }

    // single term: one betaR (applied here) x phaseR ; multi term: outer betaR x (Jdiff's betaR) = beta^2
    vec3 col = sunIntensity * betaR * (single*phaseR + multi);

    if (!hitGround && c > 0.9997) col += vec3(1.0, 0.95, 0.9)*45.0;   // sun disk
    return col;
}

//--------------------------------------------------------------------
// Horizon-centred ground view: local up = +x (radial), forward = +z.
vec3 getViewRay(vec2 fragCoord){
    vec2 uv = (fragCoord*2.0 - iResolution.xy)/iResolution.x;
    return normalize(vec3(uv.y, -uv.x, 1.0));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord){
    // Sun: default 4 deg below the horizon (blue hour). Drag mouse: y=elevation, x=azimuth.
    float el = radians(-4.0), az = 0.0;
    if (iMouse.z > 0.0 || iMouse.x > 1.0){
        az = (iMouse.x/iResolution.x - 0.5)*6.28318;
        el = (iMouse.y/iResolution.y - 0.32)*1.2;
    }
    lightDirection = normalize(vec3(sin(el), cos(el)*sin(az), cos(el)*cos(az)));

    vec3 camera = vec3(planetRadius + 0.002, 0.0, 0.0);        // ground observer, up = +x
    vec3 ray    = getViewRay(fragCoord);
    vec3 col    = atmosphere(camera, ray);

    col = 1.0 - exp(-col);                                     // tonemap
    col = pow(col, vec3(1.0/2.2));                            // gamma
    fragColor = vec4(col, 1.0);
}
