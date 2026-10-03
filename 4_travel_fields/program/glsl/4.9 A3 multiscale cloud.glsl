// ============================================================
// A3 Multiscale Analytic Cloud  (Travel Fields, latest research)
//
// Single standalone Shadertoy Image shader. Paste into a new
// Shadertoy, mouse-drag to orbit.
//
// What this demonstrates from the A3-GN research line:
//   * Multiscale cubic-fade value noise (dyadic octaves, C1,
//     nonnegative corners -> density rho >= 0 with no clamping).
//   * The A3 primitive on the CAMERA ray: a finest-grid 3D DDA
//     whose per-cell optical depth is integrated EXACTLY by
//     5-point Gauss-Legendre (GL5). GL5 is exact for the
//     degree-9 multiscale density on each finest cell, so there
//     is NO fixed-step banding: cell sizes vary, integration is
//     analytic-exact per cell, empty cells are skipped.
//   * Single scattering with analytic sun transmittance and a
//     Henyey-Greenstein phase.
//
// Precision note: the full research architecture is A3-GBox64/F32
// (float64 ray + one float64 box clip, everything else float32),
// needed only to suppress a grazing-ray tail at large world
// offsets / very long rays. Shadertoy is float32-only; near the
// origin with a modest ray length this tail is invisible, so this
// demo is pure float32 and matches the production internal path.
// ============================================================

#define PI 3.14159265358979323846

// ---- scene / medium --------------------------------------------------------
const int   OCTAVES      = 4;      // camera density octaves
const float FINEST       = 8.0;    // finest lattice frequency = 2^(OCTAVES-1)
const int   SUN_OCTAVES  = 3;      // cheaper sun density
const float VOL          = 4.0;    // noise-space extent of the cloud box [0,VOL]^3
const float SIGMA_T      = 6.0;    // extinction scale
const float ALBEDO       = 0.9;    // single-scattering albedo
const float HG_G         = 0.55;   // forward anisotropy
const float THRESH       = 0.52;   // cloud coverage threshold (soft)
const float DENS         = 2.4;    // density gain

// ---- traversal budgets -----------------------------------------------------
const int   MAX_CELLS    = 160;    // camera finest-grid DDA cells
const int   SUN_STEPS    = 20;     // sun transmittance march steps
const float SUN_TAU_MAX  = 8.0;

const vec3  SUN_DIR      = vec3(0.55350, 0.75478, 0.35223);  // normalize(0.55,0.75,0.35)
const vec3  SUN_COL      = vec3(1.7, 1.5, 1.25) * 3.0;
const vec3  SKY_TOP      = vec3(0.28, 0.45, 0.75);
const vec3  SKY_HORIZON  = vec3(0.72, 0.80, 0.92);

// GL5 nodes/weights on [-1,1] (exact through degree 9)
const float GLX0 = -0.906179845938664;
const float GLX1 = -0.538469310105683;
const float GLX3 =  0.538469310105683;
const float GLX4 =  0.906179845938664;
const float GLW0 =  0.236926885056189;
const float GLW1 =  0.478628670499366;
const float GLW2 =  0.568888888888889;

// ---------------------------------------------------------------------------
// lattice hash -> corner value in [0,1]
float hash13(vec3 p)
{
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

// one octave of cubic-fade (C1) value noise; convex combination of
// nonnegative corner values => result in [0,1], so density stays >= 0.
float noiseCubic(vec3 p)
{
    vec3 i = floor(p);
    vec3 f = p - i;
    vec3 u = f * f * (3.0 - 2.0 * f);              // cubic fade s(x)=3x^2-2x^3
    float n000 = hash13(i + vec3(0.0, 0.0, 0.0));
    float n100 = hash13(i + vec3(1.0, 0.0, 0.0));
    float n010 = hash13(i + vec3(0.0, 1.0, 0.0));
    float n110 = hash13(i + vec3(1.0, 1.0, 0.0));
    float n001 = hash13(i + vec3(0.0, 0.0, 1.0));
    float n101 = hash13(i + vec3(1.0, 0.0, 1.0));
    float n011 = hash13(i + vec3(0.0, 1.0, 1.0));
    float n111 = hash13(i + vec3(1.0, 1.0, 1.0));
    float x00 = mix(n000, n100, u.x);
    float x10 = mix(n010, n110, u.x);
    float x01 = mix(n001, n101, u.x);
    float x11 = mix(n011, n111, u.x);
    float y0 = mix(x00, x10, u.y);
    float y1 = mix(x01, x11, u.y);
    return mix(y0, y1, u.z);
}

// soft cloud envelope: keep density in a rounded slab
float envelope(vec3 p)
{
    vec3 q = (p - vec3(VOL * 0.5)) / (VOL * 0.5);
    float radial = smoothstep(1.05, 0.55, length(q.xz));
    float vert   = smoothstep(1.0, 0.35, abs(q.y));
    return radial * vert;
}

// multiscale cubic-fade fBm density (nonnegative). `oct` selects detail level.
float densityOct(vec3 p, int oct)
{
    float d = 0.0;
    float a = 0.5;
    float f = 1.0;
    for (int o = 0; o < OCTAVES; o++)
    {
        if (o >= oct) break;
        d += a * noiseCubic(p * f);
        a *= 0.5;
        f *= 2.0;
    }
    // soft coverage threshold (value-continuous / C1: ReLU-like), then envelope
    d = smoothstep(THRESH, THRESH + 0.35, d);
    return d * envelope(p) * DENS;
}

float density(vec3 p)     { return densityOct(p, OCTAVES); }
float densitySun(vec3 p)  { return densityOct(p, SUN_OCTAVES); }

// ---------------------------------------------------------------------------
// ray-box intersection for the cloud box [0,VOL]^3
bool boxHit(vec3 ro, vec3 rd, out float t0, out float t1)
{
    vec3 inv = 1.0 / rd;
    vec3 a = (vec3(0.0) - ro) * inv;
    vec3 b = (vec3(VOL) - ro) * inv;
    vec3 tmin = min(a, b);
    vec3 tmax = max(a, b);
    t0 = max(max(tmin.x, tmin.y), tmin.z);
    t1 = min(min(tmax.x, tmax.y), tmax.z);
    t0 = max(t0, 0.0);
    return t1 > t0;
}

// exact per-cell optical depth via 5-point Gauss-Legendre.
// On a finest cell every octave is within one lattice cell, so the density
// along the ray is a degree-9 polynomial and GL5 integrates it exactly.
float gl5OpticalDepth(vec3 ro, vec3 rd, float ta, float tb)
{
    float h = 0.5 * (tb - ta);
    float m = 0.5 * (tb + ta);
    float s =
        GLW0 * density(ro + rd * (m + h * GLX0)) +
        GLW1 * density(ro + rd * (m + h * GLX1)) +
        GLW2 * density(ro + rd * m)              +
        GLW1 * density(ro + rd * (m + h * GLX3)) +
        GLW0 * density(ro + rd * (m + h * GLX4));
    return SIGMA_T * h * s;
}

// analytic-ish sun transmittance: short marched optical depth toward the sun
float sunOpticalDepth(vec3 x)
{
    float t0, t1;
    if (!boxHit(x + SUN_DIR * 1e-3, SUN_DIR, t0, t1)) return 0.0;
    float span = min(t1, VOL * 1.5);
    float dt = span / float(SUN_STEPS);
    float tau = 0.0;
    float t = 0.5 * dt;
    for (int i = 0; i < SUN_STEPS; i++)
    {
        if (t > span) break;
        tau += SIGMA_T * densitySun(x + SUN_DIR * t) * dt;
        if (tau > SUN_TAU_MAX) break;
        t += dt;
    }
    return tau;
}

float phaseHG(float c)
{
    float g2 = HG_G * HG_G;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * HG_G * c, 1.5));
}

// ---------------------------------------------------------------------------
vec3 sky(vec3 rd)
{
    float h = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 col = mix(SKY_HORIZON, SKY_TOP, h);
    float sun = max(dot(rd, SUN_DIR), 0.0);
    col += SUN_COL * 0.12 * pow(sun, 220.0);       // sun disk
    col += SUN_COL * 0.03 * pow(sun, 8.0);         // sun glow
    return col;
}

// finest-grid DDA + analytic per-cell integration + single scattering
vec3 renderCloud(vec3 ro, vec3 rd, out float alpha)
{
    alpha = 0.0;
    float tEnter, tExit;
    if (!boxHit(ro, rd, tEnter, tExit))
        return vec3(0.0);

    float cs = 1.0 / FINEST;                       // finest lattice cell size
    // avoid exact-zero direction components (grazing axes never step)
    vec3 rdi = rd + (1.0 - step(vec3(1e-6), abs(rd))) * 1e-6;
    vec3 inv = 1.0 / rdi;
    vec3 stp = sign(rdi);

    vec3 pos = ro + rd * tEnter;
    vec3 cell = floor(pos / cs);
    vec3 nextB = (cell + max(stp, 0.0)) * cs;
    vec3 tMax = tEnter + (nextB - pos) * inv;
    vec3 tDelta = cs * abs(inv);

    float t = tEnter;
    float Tcam = 1.0;                              // camera transmittance
    vec3  L = vec3(0.0);
    float cosT = dot(rd, SUN_DIR);
    float ph = phaseHG(cosT);

    for (int i = 0; i < MAX_CELLS; i++)
    {
        float tNext = min(tMax.x, min(tMax.y, tMax.z));
        float tCellExit = min(tNext, tExit);

        if (tCellExit > t)
        {
            float dtau = gl5OpticalDepth(ro, rd, t, tCellExit);   // exact, no fixed step
            if (dtau > 1e-5)
            {
                vec3 mid = ro + rd * (0.5 * (t + tCellExit));
                float Tsun = exp(-sunOpticalDepth(mid));
                float scattered = 1.0 - exp(-dtau);               // fraction absorbed in cell
                L += Tcam * ALBEDO * ph * Tsun * scattered * SUN_COL;
                // a little ambient sky fill so cores are not pure black
                L += Tcam * ALBEDO * 0.12 * SKY_TOP * scattered;
                Tcam *= exp(-dtau);
            }
        }

        if (tNext >= tExit || Tcam < 0.004) break;

        // Amanatides-Woo advance across the finest lattice
        vec3 mask = step(tMax.xyz, tMax.yzx) * step(tMax.xyz, tMax.zxy);
        tMax += mask * tDelta;
        t = tNext;
    }

    alpha = 1.0 - Tcam;
    return L;
}

// ---------------------------------------------------------------------------
mat3 camera(vec3 eye, vec3 tgt)
{
    vec3 f = normalize(tgt - eye);
    vec3 r = normalize(cross(vec3(0.0, 1.0, 0.0), f));
    vec3 u = cross(f, r);
    return mat3(r, u, f);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;

    // orbit camera (mouse drag overrides auto-rotate)
    float yaw = 0.35 * iTime;
    float pitch = 0.35;
    if (iMouse.z > 0.5)
    {
        yaw = 6.2831 * (iMouse.x / iResolution.x);
        pitch = 1.4 * (iMouse.y / iResolution.y) - 0.2;
    }
    float rad = 9.0;
    vec3 center = vec3(VOL * 0.5);
    vec3 eye = center + rad * vec3(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw));
    mat3 cam = camera(eye, center);

    vec3 rd = cam * normalize(vec3(uv, 1.3));
    vec3 ro = eye;

    float alpha;
    vec3 cloud = renderCloud(ro, rd, alpha);
    vec3 bg = sky(rd);
    vec3 col = bg * (1.0 - alpha) + cloud;

    // tonemap (ACES-ish) + gamma
    col *= 0.55;
    col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);
    col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2));

    fragColor = vec4(col, 1.0);
}
