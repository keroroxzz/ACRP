// Analytic sky & atmosphere -- Paper 3 upgrade of "2. shader.glsl".
//
// Paper 2 gave a LUT-free single-scattering renderer whose beam optical depth is the
// two-branch uniform Chapman (chapmanPos / chapmanInf below, unchanged). Paper 3 showed
// that single scattering alone is 30-40% short in twilight -- the twilight sky is carried
// by MULTIPLY-scattered airlight from the still-sunlit upper atmosphere. This shader adds:
//
//   (A) a cheap, LUT-free MULTIPLE-SCATTERING term (Paper 3). The multiply-scattered field
//       is treated as isotropic diffuse airlight whose brightness follows a "lit reference"
//       transmittance evaluated at the TOP of the local column: because chapmanInf is
//       occlusion-aware, the top stays sunlit after the ground is in shadow, so the ambient
//       glows through deep twilight instead of collapsing. A geometric-series gain
//       (msStrength) sets its magnitude (Paper 3: MS ~= single by day, dominant at twilight).
//
//   (B) an OZONE (Chappuis-band) absorber. Ozone barely scatters but absorbs green/red on
//       the long twilight slant path, so the surviving multiply-scattered light turns deep
//       blue -- the blue hour. (Not a Paper-3 result; the physical enabler of the color.)
//
// VALID for thin atmospheres X = Rp/H >> 1 (here Rp/H ~ 796), where the Chapman branch is
// float32-accurate.

#define SUN_CONTROL

// ---- Earth, in megameters (Mm). Rp/H = 796 ----
const float planetRadius = 6.371;    // 6371 km
const float atmRadius    = 6.471;    // top of atmosphere, +100 km
const float H            = 0.008;    // Rayleigh scale height = 8 km
const float Csw          = 1.12;     // optimal Chapman branch switch at X~800

const int   viewSteps    = 128;       // thin twilight limb -> more steps helps banding

// Rayleigh sea-level scattering, per Mm, at 680/550/440 nm.
const vec3  betaR        = vec3(5.802, 13.558, 33.10);

// (B) Ozone Chappuis absorption, per Mm (green/red >> blue). Concentrated in a ~25 km layer.
const vec3  betaO        = vec3(3.426, 8.298, 0.356);   // relative Chappuis shape
const float ozoneCenter  = 0.025;    // 25 km peak
const float ozoneWidth   = 0.030;    // +/- 30 km tent half-width
const float ozoneAmount  = 0.03;     // vertical-column strength (tune for blue-hour depth)

// (A) Multiple-scattering amplification (geometric series of isotropic bounces).
const float msStrength   = 0.0001;      // ~1 => MS comparable to single; higher = richer twilight, but I found 0.0001 is the best

const float sunIntensity = 12.0;     // exposure knob (try 12-40)

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
    // X - X*s near the horizon (mu->0), which can go slightly negative and make sqrt()
    // return a NaN (undefined for negative args) on some GPUs -> black stripes near 90deg.
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

// Full optical depth, any mu. Downward rays reflected (non-recursive); shadow -> huge OD.
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

// Ozone number density (tent layer, peak 1 at ozoneCenter), dimensionless.
float ozoneDensity(float h){
    return max(0.0, 1.0 - abs(h - ozoneCenter)/ozoneWidth) * ozoneAmount;
}

vec3 atmosphere(vec3 camera, vec3 ray){
    vec2 atm = raySphere(camera, ray, atmRadius);
    if (atm.y < 0.0 || atm.x > atm.y) return vec3(0.0);   // miss

    float tStart = max(atm.x, 0.0);
    float tEnd   = atm.y;
    bool  hitGround = false;
    vec2  pl = raySphere(camera, ray, planetRadius);
    if (pl.x > tStart && pl.x < tEnd){ tEnd = pl.x; hitGround = true; }

    float c      = dot(ray, lightDirection);
    float phaseR = 0.75*(1.0 + c*c);                       // Rayleigh phase (mean 1)
    // MS phase is isotropic (mean 1) -> factor 1.0.

    float ds     = (tEnd - tStart)/float(viewSteps);
    float viewOD = 0.0;                                    // Rayleigh view column
    float viewOz = 0.0;                                    // ozone view column
    vec3  single = vec3(0.0);
    vec3  multi  = vec3(0.0);

    // top-of-atmosphere ozone slant reference (thin, ~0) reused for the lit ambient
    for (int i = 0; i < viewSteps; i++){
        vec3  pos  = camera + ray*(tStart + (float(i) + 0.5)*ds);
        float r    = length(pos);
        float h    = r - planetRadius;
        float dens = exp(-h/H);
        float ozo  = ozoneDensity(h);

        viewOD += dens*ds;
        viewOz += ozo*ds;

        vec3  up    = pos/r;
        float muSun = dot(lightDirection, up);

        // ---- Rayleigh + ozone solar slant to this sample (analytic Chapman) ----
        float sunOD    = chapmanInf(r, muSun);
        float airmass  = sunOD / max(H*dens, 1e-4);        // slant/vertical scale ~ airmass
        // ozone solar column ~ (vertical ozone column) x airmass; long twilight slant path
        // absorbs green/red -> the surviving airlight turns blue.
        float sunOz    = (ozoneAmount * ozoneWidth) * airmass;

        vec3 Tview = exp(-(betaR*viewOD + betaO*viewOz));  // camera -> sample
        vec3 Tsun  = exp(-(betaR*sunOD  + betaO*sunOz));   // sample -> sun

        // ---- (1) single scattering (Rayleigh phase applied at the end) ----
        single += dens * Tview * Tsun * ds;

        // ---- (2) multiple scattering (Paper 3): isotropic airlight from the sunlit
        //          upper atmosphere. "Lit reference" = solar transmittance at the top of
        //          THIS column; chapmanInf keeps it bright after the ground is shadowed. ----
        float sunODtop = chapmanInf(atmRadius, muSun);
        vec3  Tlit     = exp(-betaR*sunODtop);             // ozone ~0 at top
        vec3  ambient  = Tlit * msStrength;                // geometric-series amplified
        multi  += dens * Tview * ambient * ds;
    }

    vec3 col = sunIntensity * betaR * (single*phaseR + multi);

    if (!hitGround && c > 0.9998) col += vec3(1.0, 0.95, 0.9)*40.0;   // sun disk
    return col;
}

//--------------------------------------------------------------------
// Horizon-centred view: local up = +x (radial), forward = +z. Screen centre is the
// horizon; the top of the frame looks toward the zenith.
vec3 getViewRay(vec2 fragCoord){
    vec2 uv = (fragCoord*2.0 - iResolution.xy)/iResolution.x;
    return normalize(vec3(uv.y+0.2, -uv.x, 0.6));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord){
    // Solar elevation/azimuth. Default: 4 deg BELOW the horizon (deep blue hour).
    // Drag the mouse: vertical = elevation (raise/lower the sun), horizontal = azimuth.
    float el = radians(-4.0), az = 0.0;
#ifdef SUN_CONTROL
    if (iMouse.z > 0.0 || iMouse.x > 1.0){
        az = (iMouse.x/iResolution.x - 0.5) * 6.28318;
        el = (iMouse.y/iResolution.y - 0.32) * 1.2;        // ~ -22 deg .. +40 deg
    }
#else
    el = 0.12*sin(0.2*iTime) - 0.03;                       // gently sweep through twilight
#endif
    lightDirection = normalize(vec3(sin(el), cos(el)*sin(az), cos(el)*cos(az)));

    // Ground-level observer on the +x axis (local up = +x).
    //vec3 camera = vec3(planetRadius*0.0 + 0.001, 0.0, -13.0);
    vec3 camera = vec3(planetRadius + 0.001, 0.0, 0.0);
    vec3 ray    = getViewRay(fragCoord);
    vec3 col    = atmosphere(camera, ray)*0.8;

    col = 1.0 - exp(-col);                                 // tonemap
    col = pow(col, vec3(1.0/2.2));                         // gamma
    fragColor = vec4(col, 1.0);
}
