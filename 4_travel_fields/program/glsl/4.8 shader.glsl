// ============================================================
// Cubic-Fade Degree-10 Optical-Depth Single Scattering
// versus Trilinear Degree-4 Optical-Depth Single Scattering
//
// LEFT:
//   Cubic-fade C1 density
//   degree-9 density polynomial
//   degree-10 optical-depth polynomial
//   safeguarded Newton inversion
//
// RIGHT:
//   Trilinear C0 density
//   degree-3 density polynomial
//   degree-4 optical-depth polynomial
//   bisection inversion
//
// Both methods use:
//   - same lattice densities
//   - same camera ray
//   - same exponential optical-depth sample
//   - analytic camera free-path traversal
//   - analytic sun transmittance
//   - Henyey-Greenstein single scattering
//
// Paste into a Shadertoy Image shader.
// Mouse drag rotates the camera.
// ============================================================

#define PI 3.14159265358979323846

const float INF = 1e30;
const float EPS = 1e-6;

// ------------------------------------------------------------
// Debug modes
//
// 0 : final split render
// 1 : camera visited-cell count
// 2 : integrated / coefficient-build count
// 3 : inversion iterations
// 4 : safeguarded-Newton fallback count on left;
//     trilinear bisection count on right
// 5 : sun visited-cell count
// 6 : sun optical depth
// 7 : absolute RGB difference between both methods
// 8 : collision-distance comparison
//
// For mode 7 and 8 the same diagnostic image is shown
// across the full screen.
// ------------------------------------------------------------

const int DEBUG_MODE = 0;

// ------------------------------------------------------------
// Scene
// ------------------------------------------------------------

const vec3 VOLUME_MIN = vec3(-9.0, -1.0, -9.0);
const vec3 VOLUME_MAX = vec3( 9.0,  8.0,  9.0);

const float EXTINCTION_SCALE = 1.20;
const float SCATTERING_ALBEDO = 0.96;

const float HG_G = 0.68;

const vec3 SUN_DIRECTION =
    normalize(vec3(0.55, 0.72, 0.32));

const vec3 SUN_RADIANCE =
    vec3(7.0, 6.4, 5.3);

const int SAMPLES_PER_PIXEL = 2;

const int MAX_DDA_CELLS = 160;

// Smooth inversion.
const int SMOOTH_INVERSION_STEPS = 10;

// Trilinear reference inversion.
const int TRILINEAR_BISECTION_STEPS = 18;

const float SUN_TAU_CUTOFF = 12.0;


// ============================================================
// Structures
// ============================================================

struct TraceStats
{
    int visited;
    int integrated;
    int inversionSteps;
    int fallbackSteps;
    int sunVisited;
    int sunIntegrated;

    float sunTau;
    float collisionDistance;
};


// ============================================================
// Random/hash
// ============================================================

float hash31(vec3 p)
{
    return fract(
        sin(dot(p, vec3(127.1, 311.7, 74.7))) *
        43758.5453123
    );
}


float randomDimension(
    vec2 pixel,
    int sampleIndex,
    float dimension)
{
    vec3 p = vec3(
        pixel +
        vec2(
            37.0 * float(sampleIndex),
            91.0 * float(sampleIndex)
        ),
        dimension +
        17.0 * float(sampleIndex)
    );

    return clamp(
        hash31(p),
        1e-6,
        1.0 - 1e-6
    );
}


// ============================================================
// Lattice density
// ============================================================

float latticeDensity(ivec3 ip)
{
    vec3 p = vec3(ip);

    float normalizedHeight =
        (p.y - VOLUME_MIN.y) /
        (VOLUME_MAX.y - VOLUME_MIN.y);

    float verticalEnvelope =
        smoothstep(0.04, 0.20, normalizedHeight) *
        (
            1.0 -
            smoothstep(0.58, 0.98, normalizedHeight)
        );

    float radialDistance =
        length(
            (p.xz - vec2(0.5, -0.8)) /
            vec2(10.0, 8.0)
        );

    float radialEnvelope =
        1.0 -
        smoothstep(0.42, 1.10, radialDistance);

    float wave =
        0.53 +
        0.47 *
        sin(
            0.37 * p.x +
            0.25 * p.z +
            0.13 * p.y
        );

    float n0 = hash31(p);

    float n1 =
        hash31(
            p * 0.47 +
            vec3(19.0, 7.0, 31.0)
        );

    float n2 =
        hash31(
            p * 0.23 +
            vec3(41.0, 13.0, 5.0)
        );

    float noise =
        0.56 * n0 +
        0.29 * n1 +
        0.15 * n2;

    float structure =
        noise * wave +
        0.20 *
        sin(0.21 * p.x - 0.31 * p.z) +
        0.19;

    float density =
        smoothstep(
            0.43,
            0.72,
            structure
        );

    return max(
        density *
        verticalEnvelope *
        radialEnvelope,
        0.0
    );
}


void loadCellDensities(
    ivec3 cell,
    out float d000,
    out float d100,
    out float d010,
    out float d110,
    out float d001,
    out float d101,
    out float d011,
    out float d111)
{
    d000 = latticeDensity(cell + ivec3(0, 0, 0));
    d100 = latticeDensity(cell + ivec3(1, 0, 0));
    d010 = latticeDensity(cell + ivec3(0, 1, 0));
    d110 = latticeDensity(cell + ivec3(1, 1, 0));

    d001 = latticeDensity(cell + ivec3(0, 0, 1));
    d101 = latticeDensity(cell + ivec3(1, 0, 1));
    d011 = latticeDensity(cell + ivec3(0, 1, 1));
    d111 = latticeDensity(cell + ivec3(1, 1, 1));
}


float maximumCornerDensity(
    float d000,
    float d100,
    float d010,
    float d110,
    float d001,
    float d101,
    float d011,
    float d111)
{
    return max(
        max(
            max(d000, d100),
            max(d010, d110)
        ),
        max(
            max(d001, d101),
            max(d011, d111)
        )
    );
}


// ============================================================
// Density evaluation for final event shading
// ============================================================

float fadeCubic(float x)
{
    return x * x * (3.0 - 2.0 * x);
}


float evaluateCellDensity(
    vec3 p,
    bool smoothDensity)
{
    ivec3 cell = ivec3(floor(p));
    vec3 f = p - vec3(cell);

    float d000;
    float d100;
    float d010;
    float d110;
    float d001;
    float d101;
    float d011;
    float d111;

    loadCellDensities(
        cell,
        d000,
        d100,
        d010,
        d110,
        d001,
        d101,
        d011,
        d111
    );

    if (smoothDensity)
    {
        f = vec3(
            fadeCubic(f.x),
            fadeCubic(f.y),
            fadeCubic(f.z)
        );
    }

    float x00 = mix(d000, d100, f.x);
    float x10 = mix(d010, d110, f.x);
    float x01 = mix(d001, d101, f.x);
    float x11 = mix(d011, d111, f.x);

    float y0 = mix(x00, x10, f.y);
    float y1 = mix(x01, x11, f.y);

    return mix(y0, y1, f.z);
}


// ============================================================
// Ray-box
// ============================================================

bool intersectBox(
    vec3 ro,
    vec3 rd,
    out float tEnter,
    out float tExit)
{
    vec3 safeRd = rd;

    safeRd.x =
        abs(safeRd.x) < EPS
        ? (safeRd.x >= 0.0 ? EPS : -EPS)
        : safeRd.x;

    safeRd.y =
        abs(safeRd.y) < EPS
        ? (safeRd.y >= 0.0 ? EPS : -EPS)
        : safeRd.y;

    safeRd.z =
        abs(safeRd.z) < EPS
        ? (safeRd.z >= 0.0 ? EPS : -EPS)
        : safeRd.z;

    vec3 inverseDirection = 1.0 / safeRd;

    vec3 t0 =
        (VOLUME_MIN - ro) *
        inverseDirection;

    vec3 t1 =
        (VOLUME_MAX - ro) *
        inverseDirection;

    vec3 tSmall = min(t0, t1);
    vec3 tLarge = max(t0, t1);

    tEnter =
        max(
            max(tSmall.x, tSmall.y),
            tSmall.z
        );

    tExit =
        min(
            min(tLarge.x, tLarge.y),
            tLarge.z
        );

    tEnter = max(tEnter, 0.0);

    return tExit >= tEnter;
}


// ============================================================
// DDA
// ============================================================

void initializeDDA(
    vec3 ro,
    vec3 rd,
    float tStart,
    out ivec3 cell,
    out ivec3 stepDirection,
    out vec3 tDelta,
    out vec3 tMax)
{
    vec3 startPoint =
        ro + rd * (tStart + 1e-5);

    cell =
        ivec3(floor(startPoint));

    stepDirection =
        ivec3(
            rd.x > EPS
            ? 1
            : (rd.x < -EPS ? -1 : 0),

            rd.y > EPS
            ? 1
            : (rd.y < -EPS ? -1 : 0),

            rd.z > EPS
            ? 1
            : (rd.z < -EPS ? -1 : 0)
        );

    tDelta =
        vec3(
            abs(rd.x) > EPS
            ? abs(1.0 / rd.x)
            : INF,

            abs(rd.y) > EPS
            ? abs(1.0 / rd.y)
            : INF,

            abs(rd.z) > EPS
            ? abs(1.0 / rd.z)
            : INF
        );

    vec3 nextBoundary =
        vec3(
            stepDirection.x > 0
            ? float(cell.x + 1)
            : float(cell.x),

            stepDirection.y > 0
            ? float(cell.y + 1)
            : float(cell.y),

            stepDirection.z > 0
            ? float(cell.z + 1)
            : float(cell.z)
        );

    tMax =
        vec3(
            stepDirection.x != 0
            ? (
                nextBoundary.x -
                ro.x
              ) / rd.x
            : INF,

            stepDirection.y != 0
            ? (
                nextBoundary.y -
                ro.y
              ) / rd.y
            : INF,

            stepDirection.z != 0
            ? (
                nextBoundary.z -
                ro.z
              ) / rd.z
            : INF
        );

    tMax =
        max(
            tMax,
            vec3(tStart)
        );
}


void advanceDDA(
    inout ivec3 cell,
    ivec3 stepDirection,
    vec3 tDelta,
    inout vec3 tMax)
{
    float nextT =
        min(
            tMax.x,
            min(tMax.y, tMax.z)
        );

    bool crossX =
        tMax.x <= nextT + 1e-5;

    bool crossY =
        tMax.y <= nextT + 1e-5;

    bool crossZ =
        tMax.z <= nextT + 1e-5;

    if (crossX)
    {
        cell.x += stepDirection.x;
        tMax.x += tDelta.x;
    }

    if (crossY)
    {
        cell.y += stepDirection.y;
        tMax.y += tDelta.y;
    }

    if (crossZ)
    {
        cell.z += stepDirection.z;
        tMax.z += tDelta.z;
    }
}


// ============================================================
// Fade polynomial
//
// fade(a + b*s) = p0 + p1*s + p2*s² + p3*s³
// ============================================================

void buildFadePolynomial(
    float a,
    float b,
    out float p[4])
{
    p[0] =
        3.0 * a * a -
        2.0 * a * a * a;

    p[1] =
        6.0 * a * b *
        (1.0 - a);

    p[2] =
        3.0 * b * b *
        (1.0 - 2.0 * a);

    p[3] =
        -2.0 * b * b * b;
}


// ============================================================
// Fixed-size polynomial convolution
// ============================================================

void clearDegree6(out float p[7])
{
    for (int i = 0; i < 7; ++i)
    {
        p[i] = 0.0;
    }
}


void clearDegree9(out float p[10])
{
    for (int i = 0; i < 10; ++i)
    {
        p[i] = 0.0;
    }
}


void multiplyDegree3(
    float a[4],
    float b[4],
    out float result[7])
{
    clearDegree6(result);

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            result[i + j] +=
                a[i] * b[j];
        }
    }
}


void multiplyDegree6ByDegree3(
    float a[7],
    float b[4],
    out float result[10])
{
    clearDegree9(result);

    for (int i = 0; i < 7; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            result[i + j] +=
                a[i] * b[j];
        }
    }
}


// ============================================================
// Smooth degree-9 density polynomial
// ============================================================

void buildSmoothDensityPolynomial(
    vec3 entryPoint,
    vec3 rd,
    ivec3 cell,
    float d000,
    float d100,
    float d010,
    float d110,
    float d001,
    float d101,
    float d011,
    float d111,
    out float rho[10])
{
    vec3 f =
        entryPoint -
        vec3(cell);

    float fx[4];
    float fy[4];
    float fz[4];

    buildFadePolynomial(
        f.x,
        rd.x,
        fx
    );

    buildFadePolynomial(
        f.y,
        rd.y,
        fy
    );

    buildFadePolynomial(
        f.z,
        rd.z,
        fz
    );

    float fxy[7];
    float fxz[7];
    float fyz[7];

    multiplyDegree3(
        fx,
        fy,
        fxy
    );

    multiplyDegree3(
        fx,
        fz,
        fxz
    );

    multiplyDegree3(
        fy,
        fz,
        fyz
    );

    float fxyz[10];

    multiplyDegree6ByDegree3(
        fxy,
        fz,
        fxyz
    );

    float a =
        d000;

    float ax =
        d100 - d000;

    float ay =
        d010 - d000;

    float az =
        d001 - d000;

    float axy =
        d110 -
        d100 -
        d010 +
        d000;

    float axz =
        d101 -
        d100 -
        d001 +
        d000;

    float ayz =
        d011 -
        d010 -
        d001 +
        d000;

    float axyz =
        d111 -
        d110 -
        d101 -
        d011 +
        d100 +
        d010 +
        d001 -
        d000;

    clearDegree9(rho);

    rho[0] = a;

    for (int k = 0; k < 4; ++k)
    {
        rho[k] +=
            ax * fx[k] +
            ay * fy[k] +
            az * fz[k];
    }

    for (int k = 0; k < 7; ++k)
    {
        rho[k] +=
            axy * fxy[k] +
            axz * fxz[k] +
            ayz * fyz[k];
    }

    for (int k = 0; k < 10; ++k)
    {
        rho[k] +=
            axyz * fxyz[k];
    }
}


// ============================================================
// Degree-10 optical-depth polynomial
//
// tau(s) = sum_{k=1}^{10} q[k] s^k
// ============================================================

void buildOpticalDepthPolynomial(
    float rho[10],
    out float tau[11])
{
    tau[0] = 0.0;

    for (int k = 0; k < 10; ++k)
    {
        tau[k + 1] =
            EXTINCTION_SCALE *
            rho[k] /
            float(k + 1);
    }
}


float evaluateDegree9(
    float p[10],
    float x)
{
    float value = p[9];

    for (int k = 8; k >= 0; --k)
    {
        value =
            value * x +
            p[k];
    }

    return value;
}


float evaluateDegree10(
    float p[11],
    float x)
{
    float value = p[10];

    for (int k = 9; k >= 0; --k)
    {
        value =
            value * x +
            p[k];
    }

    return value;
}


// ============================================================
// Smooth safeguarded Newton inversion
// ============================================================

float invertSmoothOpticalDepth(
    float rho[10],
    float tau[11],
    float segmentLength,
    float targetTau,
    out int inversionSteps,
    out int fallbackSteps)
{
    inversionSteps = 0;
    fallbackSteps = 0;

    float totalTau =
        max(
            evaluateDegree10(
                tau,
                segmentLength
            ),
            0.0
        );

    float lo = 0.0;
    float hi = segmentLength;

    float x =
        segmentLength *
        clamp(
            targetTau /
            max(totalTau, 1e-7),
            0.0,
            1.0
        );

    for (
        int iteration = 0;
        iteration < SMOOTH_INVERSION_STEPS;
        ++iteration
    )
    {
        float currentTau =
            max(
                evaluateDegree10(
                    tau,
                    x
                ),
                0.0
            );

        float residual =
            currentTau -
            targetTau;

        inversionSteps =
            iteration + 1;

        if (residual < 0.0)
        {
            lo = x;
        }
        else
        {
            hi = x;
        }

        float density =
            max(
                evaluateDegree9(
                    rho,
                    x
                ),
                0.0
            );

        float derivative =
            EXTINCTION_SCALE *
            density;

        float newtonX =
            x;

        bool validNewton =
            derivative > 1e-6;

        if (validNewton)
        {
            newtonX =
                x -
                residual /
                derivative;

            validNewton =
                newtonX > lo &&
                newtonX < hi;
        }

        if (!validNewton)
        {
            newtonX =
                0.5 * (lo + hi);

            fallbackSteps += 1;
        }

        x = newtonX;
    }

    return clamp(
        0.5 * (lo + hi),
        0.0,
        segmentLength
    );
}


// ============================================================
// Trilinear polynomial
// ============================================================

void multiplyLinearLinear(
    float a0,
    float a1,
    float b0,
    float b1,
    out float r0,
    out float r1,
    out float r2)
{
    r0 = a0 * b0;

    r1 =
        a0 * b1 +
        a1 * b0;

    r2 =
        a1 * b1;
}


void multiplyQuadraticLinear(
    float q0,
    float q1,
    float q2,
    float b0,
    float b1,
    out float r0,
    out float r1,
    out float r2,
    out float r3)
{
    r0 =
        q0 * b0;

    r1 =
        q0 * b1 +
        q1 * b0;

    r2 =
        q1 * b1 +
        q2 * b0;

    r3 =
        q2 * b1;
}


void buildTrilinearPolynomial(
    vec3 entryPoint,
    vec3 rd,
    ivec3 cell,
    float d000,
    float d100,
    float d010,
    float d110,
    float d001,
    float d101,
    float d011,
    float d111,
    out float c0,
    out float c1,
    out float c2,
    out float c3)
{
    vec3 f =
        entryPoint -
        vec3(cell);

    float a =
        d000;

    float ax =
        d100 - d000;

    float ay =
        d010 - d000;

    float az =
        d001 - d000;

    float axy =
        d110 -
        d100 -
        d010 +
        d000;

    float axz =
        d101 -
        d100 -
        d001 +
        d000;

    float ayz =
        d011 -
        d010 -
        d001 +
        d000;

    float axyz =
        d111 -
        d110 -
        d101 -
        d011 +
        d100 +
        d010 +
        d001 -
        d000;

    c0 = a;
    c1 = 0.0;
    c2 = 0.0;
    c3 = 0.0;

    c0 += ax * f.x;
    c1 += ax * rd.x;

    c0 += ay * f.y;
    c1 += ay * rd.y;

    c0 += az * f.z;
    c1 += az * rd.z;

    float xy0;
    float xy1;
    float xy2;

    multiplyLinearLinear(
        f.x,
        rd.x,
        f.y,
        rd.y,
        xy0,
        xy1,
        xy2
    );

    c0 += axy * xy0;
    c1 += axy * xy1;
    c2 += axy * xy2;

    float xz0;
    float xz1;
    float xz2;

    multiplyLinearLinear(
        f.x,
        rd.x,
        f.z,
        rd.z,
        xz0,
        xz1,
        xz2
    );

    c0 += axz * xz0;
    c1 += axz * xz1;
    c2 += axz * xz2;

    float yz0;
    float yz1;
    float yz2;

    multiplyLinearLinear(
        f.y,
        rd.y,
        f.z,
        rd.z,
        yz0,
        yz1,
        yz2
    );

    c0 += ayz * yz0;
    c1 += ayz * yz1;
    c2 += ayz * yz2;

    float xyz0;
    float xyz1;
    float xyz2;
    float xyz3;

    multiplyQuadraticLinear(
        xy0,
        xy1,
        xy2,
        f.z,
        rd.z,
        xyz0,
        xyz1,
        xyz2,
        xyz3
    );

    c0 += axyz * xyz0;
    c1 += axyz * xyz1;
    c2 += axyz * xyz2;
    c3 += axyz * xyz3;
}


float trilinearOpticalDepth(
    float c0,
    float c1,
    float c2,
    float c3,
    float s)
{
    float s2 = s * s;
    float s3 = s2 * s;
    float s4 = s3 * s;

    float integral =
        c0 * s +
        0.5 * c1 * s2 +
        (1.0 / 3.0) * c2 * s3 +
        0.25 * c3 * s4;

    return
        EXTINCTION_SCALE *
        max(integral, 0.0);
}


float invertTrilinearOpticalDepth(
    float c0,
    float c1,
    float c2,
    float c3,
    float segmentLength,
    float targetTau,
    out int inversionSteps)
{
    float lo = 0.0;
    float hi = segmentLength;

    inversionSteps = 0;

    for (
        int iteration = 0;
        iteration < TRILINEAR_BISECTION_STEPS;
        ++iteration
    )
    {
        float mid =
            0.5 * (lo + hi);

        float tau =
            trilinearOpticalDepth(
                c0,
                c1,
                c2,
                c3,
                mid
            );

        inversionSteps =
            iteration + 1;

        if (tau < targetTau)
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }

    return
        0.5 * (lo + hi);
}


// ============================================================
// Camera free path
// ============================================================

bool sampleFreePath(
    vec3 ro,
    vec3 rd,
    float sampledTau,
    bool smoothDensity,
    out vec3 collisionPosition,
    inout TraceStats stats)
{
    stats.visited = 0;
    stats.integrated = 0;
    stats.inversionSteps = 0;
    stats.fallbackSteps = 0;
    stats.collisionDistance = 0.0;

    float volumeEnter;
    float volumeExit;

    if (
        !intersectBox(
            ro,
            rd,
            volumeEnter,
            volumeExit
        )
    )
    {
        return false;
    }

    ivec3 cell;
    ivec3 stepDirection;

    vec3 tDelta;
    vec3 tMax;

    initializeDDA(
        ro,
        rd,
        volumeEnter,
        cell,
        stepDirection,
        tDelta,
        tMax
    );

    float tEnter =
        volumeEnter;

    float remainingTau =
        sampledTau;

    for (
        int cellIndex = 0;
        cellIndex < MAX_DDA_CELLS;
        ++cellIndex
    )
    {
        stats.visited += 1;

        float tExit =
            min(
                min(
                    tMax.x,
                    min(tMax.y, tMax.z)
                ),
                volumeExit
            );

        float segmentLength =
            max(
                tExit - tEnter,
                0.0
            );

        float d000;
        float d100;
        float d010;
        float d110;
        float d001;
        float d101;
        float d011;
        float d111;

        loadCellDensities(
            cell,
            d000,
            d100,
            d010,
            d110,
            d001,
            d101,
            d011,
            d111
        );

        float densityMaximum =
            maximumCornerDensity(
                d000,
                d100,
                d010,
                d110,
                d001,
                d101,
                d011,
                d111
            );

        if (
            densityMaximum > 1e-7 &&
            segmentLength > 0.0
        )
        {
            stats.integrated += 1;

            vec3 entryPoint =
                ro +
                rd * tEnter;

            float cellTau = 0.0;
            float localDistance = 0.0;

            if (smoothDensity)
            {
                float rho[10];
                float tau[11];

                buildSmoothDensityPolynomial(
                    entryPoint,
                    rd,
                    cell,
                    d000,
                    d100,
                    d010,
                    d110,
                    d001,
                    d101,
                    d011,
                    d111,
                    rho
                );

                buildOpticalDepthPolynomial(
                    rho,
                    tau
                );

                cellTau =
                    max(
                        evaluateDegree10(
                            tau,
                            segmentLength
                        ),
                        0.0
                    );

                if (
                    cellTau >= remainingTau &&
                    cellTau > 0.0
                )
                {
                    localDistance =
                        invertSmoothOpticalDepth(
                            rho,
                            tau,
                            segmentLength,
                            remainingTau,
                            stats.inversionSteps,
                            stats.fallbackSteps
                        );

                    stats.collisionDistance =
                        tEnter +
                        localDistance;

                    collisionPosition =
                        ro +
                        rd *
                        stats.collisionDistance;

                    return true;
                }
            }
            else
            {
                float c0;
                float c1;
                float c2;
                float c3;

                buildTrilinearPolynomial(
                    entryPoint,
                    rd,
                    cell,
                    d000,
                    d100,
                    d010,
                    d110,
                    d001,
                    d101,
                    d011,
                    d111,
                    c0,
                    c1,
                    c2,
                    c3
                );

                cellTau =
                    trilinearOpticalDepth(
                        c0,
                        c1,
                        c2,
                        c3,
                        segmentLength
                    );

                if (
                    cellTau >= remainingTau &&
                    cellTau > 0.0
                )
                {
                    localDistance =
                        invertTrilinearOpticalDepth(
                            c0,
                            c1,
                            c2,
                            c3,
                            segmentLength,
                            remainingTau,
                            stats.inversionSteps
                        );

                    stats.collisionDistance =
                        tEnter +
                        localDistance;

                    collisionPosition =
                        ro +
                        rd *
                        stats.collisionDistance;

                    return true;
                }
            }

            remainingTau -=
                cellTau;
        }

        if (
            tExit >=
            volumeExit - 1e-5
        )
        {
            break;
        }

        tEnter =
            tExit;

        advanceDDA(
            cell,
            stepDirection,
            tDelta,
            tMax
        );
    }

    return false;
}


// ============================================================
// Analytic sun transmittance
// ============================================================

float sunTransmittance(
    vec3 startPosition,
    bool smoothDensity,
    inout TraceStats stats)
{
    stats.sunVisited = 0;
    stats.sunIntegrated = 0;
    stats.sunTau = 0.0;

    vec3 ro =
        startPosition +
        SUN_DIRECTION * 2e-4;

    float volumeEnter;
    float volumeExit;

    if (
        !intersectBox(
            ro,
            SUN_DIRECTION,
            volumeEnter,
            volumeExit
        )
    )
    {
        return 1.0;
    }

    volumeEnter =
        max(volumeEnter, 0.0);

    ivec3 cell;
    ivec3 stepDirection;

    vec3 tDelta;
    vec3 tMax;

    initializeDDA(
        ro,
        SUN_DIRECTION,
        volumeEnter,
        cell,
        stepDirection,
        tDelta,
        tMax
    );

    float tEnter =
        volumeEnter;

    for (
        int cellIndex = 0;
        cellIndex < MAX_DDA_CELLS;
        ++cellIndex
    )
    {
        stats.sunVisited += 1;

        float tExit =
            min(
                min(
                    tMax.x,
                    min(tMax.y, tMax.z)
                ),
                volumeExit
            );

        float segmentLength =
            max(
                tExit - tEnter,
                0.0
            );

        float d000;
        float d100;
        float d010;
        float d110;
        float d001;
        float d101;
        float d011;
        float d111;

        loadCellDensities(
            cell,
            d000,
            d100,
            d010,
            d110,
            d001,
            d101,
            d011,
            d111
        );

        float densityMaximum =
            maximumCornerDensity(
                d000,
                d100,
                d010,
                d110,
                d001,
                d101,
                d011,
                d111
            );

        if (
            densityMaximum > 1e-7 &&
            segmentLength > 0.0
        )
        {
            stats.sunIntegrated += 1;

            vec3 entryPoint =
                ro +
                SUN_DIRECTION *
                tEnter;

            if (smoothDensity)
            {
                float rho[10];
                float tau[11];

                buildSmoothDensityPolynomial(
                    entryPoint,
                    SUN_DIRECTION,
                    cell,
                    d000,
                    d100,
                    d010,
                    d110,
                    d001,
                    d101,
                    d011,
                    d111,
                    rho
                );

                buildOpticalDepthPolynomial(
                    rho,
                    tau
                );

                stats.sunTau +=
                    max(
                        evaluateDegree10(
                            tau,
                            segmentLength
                        ),
                        0.0
                    );
            }
            else
            {
                float c0;
                float c1;
                float c2;
                float c3;

                buildTrilinearPolynomial(
                    entryPoint,
                    SUN_DIRECTION,
                    cell,
                    d000,
                    d100,
                    d010,
                    d110,
                    d001,
                    d101,
                    d011,
                    d111,
                    c0,
                    c1,
                    c2,
                    c3
                );

                stats.sunTau +=
                    trilinearOpticalDepth(
                        c0,
                        c1,
                        c2,
                        c3,
                        segmentLength
                    );
            }

            if (
                stats.sunTau >=
                SUN_TAU_CUTOFF
            )
            {
                return 0.0;
            }
        }

        if (
            tExit >=
            volumeExit - 1e-5
        )
        {
            break;
        }

        tEnter =
            tExit;

        advanceDDA(
            cell,
            stepDirection,
            tDelta,
            tMax
        );
    }

    return exp(
        -stats.sunTau
    );
}


// ============================================================
// Lighting
// ============================================================

float henyeyGreenstein(
    float cosTheta,
    float g)
{
    float g2 =
        g * g;

    float denominator =
        1.0 +
        g2 -
        2.0 * g * cosTheta;

    return
        (1.0 - g2) /
        (
            4.0 *
            PI *
            pow(
                max(denominator, 1e-4),
                1.5
            )
        );
}


vec3 skyRadiance(vec3 rd)
{
    float elevation =
        clamp(
            0.5 +
            0.5 * rd.y,
            0.0,
            1.0
        );

    vec3 horizon =
        vec3(0.61, 0.75, 0.95);

    vec3 zenith =
        vec3(0.08, 0.20, 0.47);

    vec3 sky =
        mix(
            horizon,
            zenith,
            elevation
        );

    float sunCosine =
        max(
            dot(rd, SUN_DIRECTION),
            0.0
        );

    float sunDisk =
        smoothstep(
            0.99955,
            0.99982,
            sunCosine
        );

    float sunGlow =
        pow(
            sunCosine,
            256.0
        );

    sky +=
        SUN_RADIANCE *
        (
            0.12 * sunGlow +
            0.90 * sunDisk
        );

    return sky;
}


// ============================================================
// Camera
// ============================================================

mat3 cameraBasis(
    vec3 eye,
    vec3 target)
{
    vec3 forward =
        normalize(
            target - eye
        );

    vec3 right =
        normalize(
            cross(
                forward,
                vec3(0.0, 1.0, 0.0)
            )
        );

    vec3 up =
        cross(
            right,
            forward
        );

    return mat3(
        right,
        up,
        forward
    );
}


void makeCameraRay(
    vec2 uv,
    float panelAspect,
    out vec3 ro,
    out vec3 rd)
{
    float yaw;
    float pitch;

    if (iMouse.z > 0.0)
    {
        vec2 mouse =
            iMouse.xy /
            max(
                iResolution.xy,
                vec2(1.0)
            );

        yaw =
            (mouse.x - 0.5) *
            2.0 * PI;

        pitch =
            mix(
                0.03,
                0.62,
                clamp(
                    mouse.y,
                    0.0,
                    1.0
                )
            );
    }
    else
    {
        yaw =
            0.10 * iTime;

        pitch =
            0.22;
    }

    vec3 target =
        vec3(0.0, 2.8, 0.0);

    float radius =
        20.0;

    ro =
        target +
        vec3(
            radius *
            cos(pitch) *
            sin(yaw),

            radius *
            sin(pitch) +
            2.0,

            radius *
            cos(pitch) *
            cos(yaw)
        );

    mat3 camera =
        cameraBasis(
            ro,
            target
        );

    vec2 screen =
        vec2(
            (
                uv.x - 0.5
            ) *
            panelAspect,

            uv.y - 0.5
        );

    rd =
        normalize(
            camera *
            vec3(
                screen,
                1.30
            )
        );
}


// ============================================================
// Single-scattering sample
// ============================================================

vec3 traceSample(
    vec3 ro,
    vec3 rd,
    vec2 canonicalPixel,
    int sampleIndex,
    bool smoothDensity,
    out TraceStats stats)
{
    stats.visited = 0;
    stats.integrated = 0;
    stats.inversionSteps = 0;
    stats.fallbackSteps = 0;
    stats.sunVisited = 0;
    stats.sunIntegrated = 0;
    stats.sunTau = 0.0;
    stats.collisionDistance = 0.0;

    float u =
        randomDimension(
            canonicalPixel,
            sampleIndex,
            0.0
        );

    float sampledTau =
        -log(1.0 - u);

    vec3 collisionPosition;

    bool collided =
        sampleFreePath(
            ro,
            rd,
            sampledTau,
            smoothDensity,
            collisionPosition,
            stats
        );

    if (!collided)
    {
        return skyRadiance(rd);
    }

    float transmittanceToSun =
        sunTransmittance(
            collisionPosition,
            smoothDensity,
            stats
        );

    float cosTheta =
        clamp(
            dot(
                rd,
                SUN_DIRECTION
            ),
            -1.0,
            1.0
        );

    float phase =
        henyeyGreenstein(
            cosTheta,
            HG_G
        );

    return
        SCATTERING_ALBEDO *
        phase *
        transmittanceToSun *
        SUN_RADIANCE;
}


// ============================================================
// Heat maps
// ============================================================

vec3 heatmap(float x)
{
    x = clamp(x, 0.0, 1.0);

    vec3 blue =
        vec3(0.03, 0.10, 0.55);

    vec3 cyan =
        vec3(0.00, 0.75, 1.00);

    vec3 green =
        vec3(0.04, 0.88, 0.18);

    vec3 yellow =
        vec3(1.00, 0.88, 0.02);

    vec3 red =
        vec3(0.96, 0.03, 0.01);

    if (x < 0.25)
    {
        return mix(
            blue,
            cyan,
            x / 0.25
        );
    }

    if (x < 0.50)
    {
        return mix(
            cyan,
            green,
            (x - 0.25) / 0.25
        );
    }

    if (x < 0.75)
    {
        return mix(
            green,
            yellow,
            (x - 0.50) / 0.25
        );
    }

    return mix(
        yellow,
        red,
        (x - 0.75) / 0.25
    );
}


float logarithmicCost(
    float value,
    float maximumValue)
{
    return clamp(
        log2(1.0 + value) /
        log2(1.0 + maximumValue),
        0.0,
        1.0
    );
}


// ============================================================
// Render one method
// ============================================================

vec3 renderMethod(
    vec3 ro,
    vec3 rd,
    vec2 canonicalPixel,
    bool smoothDensity,
    out TraceStats averageStats)
{
    vec3 accumulated =
        vec3(0.0);

    float visited = 0.0;
    float integrated = 0.0;
    float inversion = 0.0;
    float fallback = 0.0;
    float sunVisited = 0.0;
    float sunIntegrated = 0.0;
    float sunTau = 0.0;
    float collisionDistance = 0.0;
    float collisionCount = 0.0;

    for (
        int sampleIndex = 0;
        sampleIndex < SAMPLES_PER_PIXEL;
        ++sampleIndex
    )
    {
        TraceStats sampleStats;

        accumulated +=
            traceSample(
                ro,
                rd,
                canonicalPixel,
                sampleIndex,
                smoothDensity,
                sampleStats
            );

        visited +=
            float(sampleStats.visited);

        integrated +=
            float(sampleStats.integrated);

        inversion +=
            float(sampleStats.inversionSteps);

        fallback +=
            float(sampleStats.fallbackSteps);

        sunVisited +=
            float(sampleStats.sunVisited);

        sunIntegrated +=
            float(sampleStats.sunIntegrated);

        sunTau +=
            sampleStats.sunTau;

        if (
            sampleStats.collisionDistance >
            0.0
        )
        {
            collisionDistance +=
                sampleStats.collisionDistance;

            collisionCount += 1.0;
        }
    }

    float inverseSamples =
        1.0 /
        float(SAMPLES_PER_PIXEL);

    averageStats.visited =
        int(visited * inverseSamples);

    averageStats.integrated =
        int(integrated * inverseSamples);

    averageStats.inversionSteps =
        int(inversion * inverseSamples);

    averageStats.fallbackSteps =
        int(fallback * inverseSamples);

    averageStats.sunVisited =
        int(sunVisited * inverseSamples);

    averageStats.sunIntegrated =
        int(sunIntegrated * inverseSamples);

    averageStats.sunTau =
        sunTau * inverseSamples;

    averageStats.collisionDistance =
        collisionCount > 0.0
        ? collisionDistance /
          collisionCount
        : 0.0;

    return accumulated *
        inverseSamples;
}


// ============================================================
// Main
// ============================================================

void mainImage(
    out vec4 fragColor,
    in vec2 fragCoord)
{
    vec2 resolution =
        max(
            iResolution.xy,
            vec2(1.0)
        );

    bool leftPanel =
        fragCoord.x <
        0.5 * resolution.x;

    float panelX =
        leftPanel
        ? fragCoord.x /
          (0.5 * resolution.x)
        : (
            fragCoord.x -
            0.5 * resolution.x
          ) /
          (0.5 * resolution.x);

    vec2 panelUV =
        vec2(
            panelX,
            fragCoord.y /
            resolution.y
        );

    float panelAspect =
        (0.5 * resolution.x) /
        resolution.y;

    vec3 ro;
    vec3 rd;

    makeCameraRay(
        panelUV,
        panelAspect,
        ro,
        rd
    );

    // Identical random coordinates for corresponding
    // left/right pixels.
    vec2 canonicalPixel =
        floor(
            panelUV *
            vec2(
                0.5 * resolution.x,
                resolution.y
            )
        );

    TraceStats smoothStats;
    TraceStats trilinearStats;

    // Both methods are always evaluated so difference modes
    // use exactly corresponding rays and random samples.
    vec3 smoothColor =
        renderMethod(
            ro,
            rd,
            canonicalPixel,
            true,
            smoothStats
        );

    vec3 trilinearColor =
        renderMethod(
            ro,
            rd,
            canonicalPixel,
            false,
            trilinearStats
        );

    vec3 color;

    if (DEBUG_MODE == 0)
    {
        color =
            leftPanel
            ? smoothColor
            : trilinearColor;
    }
    else if (DEBUG_MODE == 1)
    {
        float value =
            leftPanel
            ? float(smoothStats.visited)
            : float(trilinearStats.visited);

        color =
            heatmap(
                logarithmicCost(
                    value,
                    160.0
                )
            );
    }
    else if (DEBUG_MODE == 2)
    {
        float value =
            leftPanel
            ? float(smoothStats.integrated)
            : float(trilinearStats.integrated);

        color =
            heatmap(
                logarithmicCost(
                    value,
                    160.0
                )
            );
    }
    else if (DEBUG_MODE == 3)
    {
        float value =
            leftPanel
            ? float(smoothStats.inversionSteps)
            : float(trilinearStats.inversionSteps);

        color =
            heatmap(
                value / 18.0
            );
    }
    else if (DEBUG_MODE == 4)
    {
        float value =
            leftPanel
            ? float(smoothStats.fallbackSteps)
            : float(trilinearStats.inversionSteps);

        color =
            heatmap(
                value / 18.0
            );
    }
    else if (DEBUG_MODE == 5)
    {
        float value =
            leftPanel
            ? float(smoothStats.sunVisited)
            : float(trilinearStats.sunVisited);

        color =
            heatmap(
                logarithmicCost(
                    value,
                    160.0
                )
            );
    }
    else if (DEBUG_MODE == 6)
    {
        float value =
            leftPanel
            ? smoothStats.sunTau
            : trilinearStats.sunTau;

        color =
            heatmap(
                value /
                SUN_TAU_CUTOFF
            );
    }
    else if (DEBUG_MODE == 7)
    {
        vec3 difference =
            abs(
                smoothColor -
                trilinearColor
            );

        float error =
            max(
                difference.r,
                max(
                    difference.g,
                    difference.b
                )
            );

        color =
            heatmap(
                clamp(
                    error / 0.20,
                    0.0,
                    1.0
                )
            );
    }
    else
    {
        float smoothDistance =
            smoothStats.collisionDistance;

        float trilinearDistance =
            trilinearStats.collisionDistance;

        bool smoothHit =
            smoothDistance > 0.0;

        bool trilinearHit =
            trilinearDistance > 0.0;

        if (
            smoothHit &&
            !trilinearHit
        )
        {
            color =
                vec3(1.0, 0.0, 1.0);
        }
        else if (
            !smoothHit &&
            trilinearHit
        )
        {
            color =
                vec3(0.0, 1.0, 1.0);
        }
        else if (
            !smoothHit &&
            !trilinearHit
        )
        {
            color =
                vec3(0.02, 0.03, 0.09);
        }
        else
        {
            float error =
                abs(
                    smoothDistance -
                    trilinearDistance
                );

            color =
                heatmap(
                    clamp(
                        error / 0.50,
                        0.0,
                        1.0
                    )
                );
        }
    }

    // Center divider.
    if (
        DEBUG_MODE < 7 &&
        abs(
            fragCoord.x -
            0.5 * resolution.x
        ) < 1.5
    )
    {
        color =
            vec3(0.96);
    }

    // Tone mapping.
    if (DEBUG_MODE == 0)
    {
        color =
            color /
            (
                vec3(1.0) +
                color
            );
    }

    color =
        pow(
            max(color, vec3(0.0)),
            vec3(1.0 / 2.2)
        );

    fragColor =
        vec4(color, 1.0);
}