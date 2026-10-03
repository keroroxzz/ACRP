// ============================================================
// A3 Analytic Cloud  vs  Classic Fixed-Step Raymarch  (split)
//
// LEFT  : our work -- finest-grid 3D DDA + EXACT per-cell GL5
//         optical depth (no fixed step, no banding).
// RIGHT : classic fixed-step raymarch (MARCH_STEPS Riemann steps).
//
// Everything else is identical (density field, camera, sun
// transmittance, single scattering, phase, tonemap), so the only
// difference you see is the camera-ray optical-depth integration.
//
// Mouse drag = wipe the split line left/right to compare the SAME
// region with both methods. Camera auto-rotates.
//
// Try lowering/raising MARCH_STEPS to watch the classic side band
// and slowly converge to the (always-exact) A3 side.
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
const int   MAX_CELLS    = 160;    // A3: max finest-grid DDA cells
const int   MARCH_STEPS  = 32;     // classic: fixed camera steps (raise to converge)
const int   SUN_STEPS    = 20;     // sun transmittance march (shared by both sides)
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
float hash13(vec3 p)
{
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

// one octave of cubic-fade (C1) value noise; nonnegative corners => rho >= 0.
float noiseCubic(vec3 p)
{
    vec3 i = floor(p);
    vec3 f = p - i;
    vec3 u = f * f * (3.0 - 2.0 * f);
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

float envelope(vec3 p)
{
    vec3 q = (p - vec3(VOL * 0.5)) / (VOL * 0.5);
    float radial = smoothstep(1.05, 0.55, length(q.xz));
    float vert   = smoothstep(1.0, 0.35, abs(q.y));
    return radial * vert;
}

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
    d = smoothstep(THRESH, THRESH + 0.35, d);
    return d * envelope(p) * DENS;
}

float density(vec3 p)     { return densityOct(p, OCTAVES); }
float densitySun(vec3 p)  { return densityOct(p, SUN_OCTAVES); }

// ---------------------------------------------------------------------------
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

// exact per-cell optical depth via GL5 (exact for the degree-9 density on a cell)
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

// shared sun transmittance (same for both sides, so the split isolates the
// camera-ray integration method)
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

// one single-scattering cell/step contribution (identical on both sides)
void scatterStep(inout vec3 L, inout float Tcam, vec3 mid, float dtau, float ph)
{
    if (dtau <= 1e-5) return;
    float Tsun = exp(-sunOpticalDepth(mid));
    float scattered = 1.0 - exp(-dtau);
    L += Tcam * ALBEDO * ph * Tsun * scattered * SUN_COL;
    L += Tcam * ALBEDO * 0.12 * SKY_TOP * scattered;   // ambient sky fill
    Tcam *= exp(-dtau);
}

// ---- LEFT: our work -- DDA + exact per-cell GL5 ----------------------------
vec3 renderA3(vec3 ro, vec3 rd, out float alpha)
{
    alpha = 0.0;
    float tEnter, tExit;
    if (!boxHit(ro, rd, tEnter, tExit)) return vec3(0.0);

    float cs = 1.0 / FINEST;
    vec3 rdi = rd + (1.0 - step(vec3(1e-6), abs(rd))) * 1e-6;
    vec3 inv = 1.0 / rdi;
    vec3 stp = sign(rdi);
    vec3 pos = ro + rd * tEnter;
    vec3 cell = floor(pos / cs);
    vec3 nextB = (cell + max(stp, 0.0)) * cs;
    vec3 tMax = tEnter + (nextB - pos) * inv;
    vec3 tDelta = cs * abs(inv);

    float t = tEnter;
    float Tcam = 1.0;
    vec3  L = vec3(0.0);
    float ph = phaseHG(dot(rd, SUN_DIR));

    for (int i = 0; i < MAX_CELLS; i++)
    {
        float tNext = min(tMax.x, min(tMax.y, tMax.z));
        float tCellExit = min(tNext, tExit);
        if (tCellExit > t)
        {
            float dtau = gl5OpticalDepth(ro, rd, t, tCellExit);       // exact
            scatterStep(L, Tcam, ro + rd * (0.5 * (t + tCellExit)), dtau, ph);
        }
        if (tNext >= tExit || Tcam < 0.004) break;
        vec3 mask = step(tMax.xyz, tMax.yzx) * step(tMax.xyz, tMax.zxy);
        tMax += mask * tDelta;
        t = tNext;
    }
    alpha = 1.0 - Tcam;
    return L;
}

// ---- RIGHT: classic fixed-step raymarch ------------------------------------
vec3 renderMarch(vec3 ro, vec3 rd, out float alpha)
{
    alpha = 0.0;
    float tEnter, tExit;
    if (!boxHit(ro, rd, tEnter, tExit)) return vec3(0.0);

    float dt = (tExit - tEnter) / float(MARCH_STEPS);
    float Tcam = 1.0;
    vec3  L = vec3(0.0);
    float ph = phaseHG(dot(rd, SUN_DIR));
    float t = tEnter + 0.5 * dt;                                       // midpoint rule

    for (int i = 0; i < MARCH_STEPS; i++)
    {
        vec3 p = ro + rd * t;
        float dtau = SIGMA_T * density(p) * dt;                        // fixed-step Riemann
        scatterStep(L, Tcam, p, dtau, ph);
        if (Tcam < 0.004) break;
        t += dt;
    }
    alpha = 1.0 - Tcam;
    return L;
}

// ---------------------------------------------------------------------------
vec3 sky(vec3 rd)
{
    float h = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 col = mix(SKY_HORIZON, SKY_TOP, h);
    float sun = max(dot(rd, SUN_DIR), 0.0);
    col += SUN_COL * 0.12 * pow(sun, 220.0);
    col += SUN_COL * 0.03 * pow(sun, 8.0);
    return col;
}

mat3 camera(vec3 eye, vec3 tgt)
{
    vec3 f = normalize(tgt - eye);
    vec3 r = normalize(cross(vec3(0.0, 1.0, 0.0), f));
    vec3 u = cross(f, r);
    return mat3(r, u, f);
}

vec3 tonemap(vec3 col)
{
    col *= 0.55;
    col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);
    return pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;

    // auto-rotate camera
    float yaw = 0.35 * iTime;
    float pitch = 0.32;
    vec3 center = vec3(VOL * 0.5);
    vec3 eye = center + 9.0 * vec3(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw));
    mat3 cam = camera(eye, center);
    vec3 rd = cam * normalize(vec3(uv, 1.3));
    vec3 ro = eye;

    // split position: mouse.x wipes it, default center
    float divider = (iMouse.z > 0.5) ? (iMouse.x / iResolution.x) : 0.5;
    bool leftSide = (fragCoord.x / iResolution.x) < divider;

    float alpha;
    vec3 cloud = leftSide ? renderA3(ro, rd, alpha) : renderMarch(ro, rd, alpha);
    vec3 bg = sky(rd);
    vec3 col = tonemap(bg * (1.0 - alpha) + cloud);

    // split line
    float px = 1.5 / iResolution.x;
    float dline = abs(fragCoord.x / iResolution.x - divider);
    col = mix(vec3(1.0), col, smoothstep(px, px * 2.0, dline));

    fragColor = vec4(col, 1.0);
}
