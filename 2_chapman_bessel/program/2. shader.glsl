// Analytic single-scattering atmosphere.
// Inner light-march replaced by the two-branch uniform Chapman optical depth.
// VALID ONLY for thin atmospheres X = Rp/H >> 1 (here Rp/H = 320).

#define SUN_CONTROL

// ---- Earth, in megameters (Mm). Rp/H = 796  -> method is float32-accurate ----
const float planetRadius = 6.371;    // 6371 km
const float atmRadius    = 6.471;    // top of atmosphere, +100 km (~12.5 scale heights)
const float H            = 0.008;    // Rayleigh scale height = 8 km
const float Csw          = 1.12;     // optimal branch switch at X~800

const int   viewSteps    = 48;       // thin limb -> a few more steps helps banding

// Rayleigh sea-level extinction = scattering, per Mm, at 680/550/440 nm.
// betaR * OD(Mm) is the PHYSICAL optical depth, so densityScale stays 1.0.
const vec3  betaR        = vec3(5.802, 13.558, 33.10);
const float densityScale = 1.0;      // physical; use as an exposure/thickness knob only

const float sunIntensity = 20.0;     // exposure knob (try 12-40 to taste)

vec3 lightDirection = normalize(vec3(1.0, 0.0, 0.0));

//--------------------------------------------------------------------
// erfcx(y) = e^{y^2} erfc(y), y>=0 : Maclaurin for y<1.5, Laplace CF for y>=1.5
float erfcx(float y){
    if (y < 1.5){
        float y2 = y*y, t = y, s = y;
        for (int n = 1; n < 10; n++){ t *= -y2/float(n); s += t/float(2*n+1); }
        float erf = 1.12837916709551*s;            // 2/sqrt(pi)
        return exp(y2)*(1.0 - erf);
    }
    float f = y;
    for (int k = 12; k >= 1; k--) f = y + 0.5*float(k)/f;
    return 0.564189583547756/f;                    // 1/sqrt(pi)
}

// Core optical depth, ASSUMES mu >= 0 (non-recursive).
// INT exp(-(r-Rp)/H) ds from radius r0, mu=cos(zenith), to infinity.
float chapmanPos(float r0, float mu){
    float s   = sqrt(max(0.0, 1.0 - mu*mu));
    float X   = r0/H;
    float h0  = r0 - planetRadius;
    float eps = 1.0/X;
    if (s < 1e-4) return H*exp(-h0/H);             // straight up
    float coshT0 = 1.0/s;

    if (coshT0 > Csw){                             // ---- zenith branch (tau-form) ----
        float m2 = mu*mu;
        float c1 = 1.0 - 1.0/m2;
        float c2 = 3.0*(1.0 - m2)/(m2*m2);
        float c3 = 3.0*(m2*m2 + 4.0*m2 - 5.0)/(m2*m2*m2);
        float S  = (1.0 + eps*(c1 + eps*(c2 + eps*c3)))/mu;
        return H*exp(-h0/H)*S;
    }
    // ---- horizon branch (moment / erfcx) ----
    float z   = X*s;
    float p   = r0*s;
    // eta = sqrt(2X(1-s)); computed as X*mu^2/(1+s) to avoid float32 cancellation in
    // X - X*s near the horizon (mu->0), which can go slightly negative and make
    // sqrt() return a NaN (undefined for negative args) on some GPUs -> black stripes.
    float eta = sqrt(2.0*X*mu*mu/(1.0 + s));
    float iz  = 1.0/z;
    float P = 1.0 + iz*(0.375 + iz*(-0.1171875 + iz*(0.1025390625
                     + iz*(-0.144195556640625))));
    float e2 = eta*eta;
    float Q = iz*( 0.375*eta
            + iz*( -0.0390625*(e2*eta + 3.0*eta)
            + iz*(  0.0068359375*(e2*e2*eta + 5.0*e2*eta + 15.0*eta)
            + iz*( -0.001373291015625*(e2*e2*e2*eta + 7.0*e2*e2*eta
                                       + 35.0*e2*eta + 105.0*eta) ))));
    float y = eta*0.707106781186548;               // eta/sqrt(2)
    return sqrt(p*H)*exp(-h0/H)*(P*1.25331413731550*erfcx(y) + Q);   // sqrt(pi/2)
}

// Full optical depth, any mu. Downward rays handled by reflection (non-recursive).
float chapmanInf(float r0, float mu){
    if (mu < 0.0){
        float p = r0*sqrt(max(0.0, 1.0 - mu*mu));
        if (p < planetRadius) return 1e20;         // blocked by the planet -> shadow
        return 2.0*chapmanPos(p, 0.0) - chapmanPos(r0, -mu);
    }
    return chapmanPos(r0, mu);
}

//--------------------------------------------------------------------
vec2 raySphere(vec3 ro, vec3 rd, float ra){
    float b = dot(ro, rd);
    float c = dot(ro, ro) - ra*ra;
    float d = b*b - c;
    if (d < 0.0) return vec2(1e9, -1e9);
    d = sqrt(d);
    return vec2(-b - d, -b + d);
}

vec3 atmosphere(vec3 camera, vec3 ray){
    vec2 atm = raySphere(camera, ray, atmRadius);
    if (atm.y < 0.0 || atm.x > atm.y) return vec3(0.0);   // miss

    float tStart = max(atm.x, 0.0);
    float tEnd   = atm.y;
    bool  hitGround = false;
    vec2  pl = raySphere(camera, ray, planetRadius);
    if (pl.x > tStart && pl.x < tEnd){ tEnd = pl.x; hitGround = true; }

    float c     = dot(ray, lightDirection);
    float phase = 0.75*(1.0 + c*c);                        // Rayleigh phase

    float ds = (tEnd - tStart)/float(viewSteps);
    float viewOD = 0.0;                                    // camera -> sample OD
    vec3  color  = vec3(0.0);

    for (int i = 0; i < viewSteps; i++){
        vec3  pos  = camera + ray*(tStart + (float(i) + 0.5)*ds);
        float r    = length(pos);
        float dens = exp(-(r - planetRadius)/H);
        viewOD += dens*ds;

        float muSun = dot(lightDirection, pos/r);
        float sunOD = chapmanInf(r, muSun);                // ANALYTIC (was inner march)

        vec3 tr = exp(-betaR*densityScale*(viewOD + sunOD));
        color  += betaR*dens*tr*ds;
    }
    color *= sunIntensity*phase*densityScale;

    if (!hitGround && c > 0.9998) color += vec3(1.0, 0.95, 0.9)*40.0;   // sun disk

    return color;
}

//--------------------------------------------------------------------
vec3 getViewRay(vec2 fragCoord){
    vec2 uv = (fragCoord*2.0 - iResolution.xy)/iResolution.x;
    return normalize(vec3(1.0, -uv.x, uv.y));              // looking +x
}

void mainImage(out vec4 fragColor, in vec2 fragCoord){
#ifdef SUN_CONTROL
    vec2 m = -(iMouse.xy/iResolution.xy*2.0 - 1.0)*vec2(1.0, -1.0)*3.5;
    lightDirection = normalize(vec3(cos(m.x)*cos(m.y), sin(m.x)*cos(m.y), sin(m.y)));
#else
    lightDirection = normalize(vec3(cos(iTime), 0.0, sin(iTime)));
#endif

    vec3 camera = vec3(0.0, 0.0, 6.3712);                    // far view of the globe
    vec3 ray    = getViewRay(fragCoord);
    vec3 col    = atmosphere(camera, ray);

    col = 1.0 - exp(-col);                                 // tonemap
    col = pow(col, vec3(1.0/2.2));                         // gamma
    fragColor = vec4(col, 1.0);
}