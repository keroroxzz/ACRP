// ============================================================
// Analytic Trilinear Single-Scattering Volume Cloud
//
// Camera path:
//   Exponential optical-depth sampling
//   + 3D DDA
//   + exact cell optical-depth integration
//   + one terminal quartic inversion
//
// Sun path:
//   3D DDA
//   + exact cell optical-depth integration
//   + analytic transmittance exp(-tauSun)
//
// Medium:
//   Nonnegative trilinearly interpolated procedural density
//
// Lighting:
//   Single scattering
//   + Henyey-Greenstein phase function
//
// Paste directly into a Shadertoy Image shader.
// Hold and drag mouse to rotate the camera.
// ============================================================

#define PI 3.14159265358979323846

const float INF = 1e30;
const float EPS = 1e-6;

// ------------------------------------------------------------
// Volume domain
// ------------------------------------------------------------

const vec3 VOLUME_MIN = vec3(-9.0, -1.0, -9.0);
const vec3 VOLUME_MAX = vec3( 9.0,  8.0,  9.0);

// Extinction and scattering.
const float EXTINCTION_SCALE = 1.25;
const float SINGLE_SCATTERING_ALBEDO = 0.96;

// Henyey-Greenstein anisotropy.
const float HG_G = 0.68;

// Direct sunlight.
const vec3 SUN_DIRECTION =
    normalize(vec3(0.55, 0.72, 0.32));

const vec3 SUN_RADIANCE =
    vec3(7.0, 6.4, 5.3);

// Per-pixel Monte Carlo samples.
// Increase to 4 for lower noise at greater cost.
const int SAMPLES_PER_PIXEL = 2;

// Maximum DDA cells crossed by one segment.
const int MAX_DDA_CELLS = 160;

// Terminal free-path inversion iterations.
const int FREE_PATH_BISECTION_STEPS = 18;

// Optical-depth early termination for sun rays.
// exp(-12) is approximately 6.1e-6.
const float SUN_TAU_CUTOFF = 12.0;

// ------------------------------------------------------------
// Debug modes
//
// 0: final single-scattering render
// 1: camera visited-cell heatmap
// 2: camera integrated-cell heatmap
// 3: sun visited-cell heatmap
// 4: sun optical-depth heatmap
// 5: sampled collision distance
// ------------------------------------------------------------

const int DEBUG_MODE = 0;


// ============================================================
// Hash utilities
// ============================================================

float hash31(vec3 p)
{
    return fract(
        sin(dot(p, vec3(127.1, 311.7, 74.7))) *
        43758.5453123
    );
}


float hash22(vec2 p)
{
    vec2 q = vec2(
        dot(p, vec2(127.1, 311.7)),
        dot(p, vec2(269.5, 183.3))
    );

    return fract(
        sin(q.x + sin(q.y) * 0.73) *
        43758.5453123
    );
}


float sampleRandom(
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
// Procedural nonnegative lattice density
// ============================================================

float latticeDensity(ivec3 ip)
{
    vec3 p = vec3(ip);

    float normalizedHeight =
        (p.y - VOLUME_MIN.y) /
        (VOLUME_MAX.y - VOLUME_MIN.y);

    // Flat-ish cloud base and softer cloud top.
    float verticalEnvelope =
        smoothstep(0.04, 0.20, normalizedHeight) *
        (
            1.0 -
            smoothstep(0.58, 0.98, normalizedHeight)
        );

    // Large horizontal cloud clusters.
    float radialDistance =
        length(
            (p.xz - vec2(0.5, -0.8)) /
            vec2(10.0, 8.0)
        );

    float radialEnvelope =
        1.0 -
        smoothstep(
            0.42,
            1.10,
            radialDistance
        );

    float wave =
        0.53 +
        0.47 *
        sin(
            0.37 * p.x +
            0.25 * p.z +
            0.13 * p.y
        );

    float n0 =
        hash31(p);

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

    // Exact zeros are useful for empty-cell skipping.
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
    d000 =
        latticeDensity(cell + ivec3(0, 0, 0));

    d100 =
        latticeDensity(cell + ivec3(1, 0, 0));

    d010 =
        latticeDensity(cell + ivec3(0, 1, 0));

    d110 =
        latticeDensity(cell + ivec3(1, 1, 0));

    d001 =
        latticeDensity(cell + ivec3(0, 0, 1));

    d101 =
        latticeDensity(cell + ivec3(1, 0, 1));

    d011 =
        latticeDensity(cell + ivec3(0, 1, 1));

    d111 =
        latticeDensity(cell + ivec3(1, 1, 1));
}


float densityField(vec3 p)
{
    ivec3 cell =
        ivec3(floor(p));

    vec3 f =
        p - vec3(cell);

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

    float x00 =
        mix(d000, d100, f.x);

    float x10 =
        mix(d010, d110, f.x);

    float x01 =
        mix(d001, d101, f.x);

    float x11 =
        mix(d011, d111, f.x);

    float y0 =
        mix(x00, x10, f.y);

    float y1 =
        mix(x01, x11, f.y);

    return mix(y0, y1, f.z);
}


// ============================================================
// Ray-box intersection
// ============================================================

bool intersectBox(
    vec3 ro,
    vec3 rd,
    vec3 boxMin,
    vec3 boxMax,
    out float tEnter,
    out float tExit)
{
    vec3 safeDirection = rd;

    safeDirection.x =
        abs(safeDirection.x) < EPS
        ? (
            safeDirection.x >= 0.0
            ? EPS
            : -EPS
        )
        : safeDirection.x;

    safeDirection.y =
        abs(safeDirection.y) < EPS
        ? (
            safeDirection.y >= 0.0
            ? EPS
            : -EPS
        )
        : safeDirection.y;

    safeDirection.z =
        abs(safeDirection.z) < EPS
        ? (
            safeDirection.z >= 0.0
            ? EPS
            : -EPS
        )
        : safeDirection.z;

    vec3 inverseDirection =
        1.0 / safeDirection;

    vec3 t0 =
        (boxMin - ro) *
        inverseDirection;

    vec3 t1 =
        (boxMax - ro) *
        inverseDirection;

    vec3 tSmall =
        min(t0, t1);

    vec3 tLarge =
        max(t0, t1);

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

    tEnter =
        max(tEnter, 0.0);

    return tExit >= tEnter;
}


// ============================================================
// Trilinear density restricted to one ray segment
//
// rho(s) = c0 + c1*s + c2*s² + c3*s³
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
    r0 =
        a0 * b0;

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


void buildTrilinearRayPolynomial(
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

    // Linear components.
    c0 += ax * f.x;
    c1 += ax * rd.x;

    c0 += ay * f.y;
    c1 += ay * rd.y;

    c0 += az * f.z;
    c1 += az * rd.z;

    // xy.
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

    // xz.
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

    // yz.
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

    // xyz.
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


// ============================================================
// Exact optical-depth integration
// ============================================================

float integrateDensityCubic(
    float c0,
    float c1,
    float c2,
    float c3,
    float distance)
{
    float d2 =
        distance * distance;

    float d3 =
        d2 * distance;

    float d4 =
        d3 * distance;

    return
        c0 * distance +
        0.5 * c1 * d2 +
        (1.0 / 3.0) * c2 * d3 +
        0.25 * c3 * d4;
}


float opticalDepthInCell(
    float c0,
    float c1,
    float c2,
    float c3,
    float distance)
{
    float densityIntegral =
        integrateDensityCubic(
            c0,
            c1,
            c2,
            c3,
            distance
        );

    return
        EXTINCTION_SCALE *
        max(densityIntegral, 0.0);
}


// Monotone inversion of:
//
// tau(s) = targetTau
//
// The underlying trilinear density is nonnegative, so tau(s)
// is monotone nondecreasing.
float invertOpticalDepthInCell(
    float c0,
    float c1,
    float c2,
    float c3,
    float segmentLength,
    float targetTau)
{
    float lo = 0.0;
    float hi = segmentLength;

    for (
        int i = 0;
        i < FREE_PATH_BISECTION_STEPS;
        ++i
    )
    {
        float mid =
            0.5 * (lo + hi);

        float tau =
            opticalDepthInCell(
                c0,
                c1,
                c2,
                c3,
                mid
            );

        if (tau < targetTau)
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }

    return 0.5 * (lo + hi);
}


// ============================================================
// Shared DDA initialization
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
        ro +
        rd * (tStart + 1e-5);

    cell =
        ivec3(floor(startPoint));

    stepDirection =
        ivec3(
            rd.x > EPS
            ? 1
            : (
                rd.x < -EPS
                ? -1
                : 0
            ),

            rd.y > EPS
            ? 1
            : (
                rd.y < -EPS
                ? -1
                : 0
            ),

            rd.z > EPS
            ? 1
            : (
                rd.z < -EPS
                ? -1
                : 0
            )
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
        tMax.x <=
        nextT + 1e-5;

    bool crossY =
        tMax.y <=
        nextT + 1e-5;

    bool crossZ =
        tMax.z <=
        nextT + 1e-5;

    if (crossX)
    {
        cell.x +=
            stepDirection.x;

        tMax.x +=
            tDelta.x;
    }

    if (crossY)
    {
        cell.y +=
            stepDirection.y;

        tMax.y +=
            tDelta.y;
    }

    if (crossZ)
    {
        cell.z +=
            stepDirection.z;

        tMax.z +=
            tDelta.z;
    }
}


// ============================================================
// Analytic camera free-path sampling
// ============================================================

bool sampleCameraFreePath(
    vec3 ro,
    vec3 rd,
    float sampledOpticalDepth,
    out float collisionDistance,
    out vec3 collisionPosition,
    out int visitedCells,
    out int integratedCells)
{
    collisionDistance = INF;
    collisionPosition = vec3(0.0);

    visitedCells = 0;
    integratedCells = 0;

    float volumeEnter;
    float volumeExit;

    if (
        !intersectBox(
            ro,
            rd,
            VOLUME_MIN,
            VOLUME_MAX,
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
        sampledOpticalDepth;

    for (
        int cellIndex = 0;
        cellIndex < MAX_DDA_CELLS;
        ++cellIndex
    )
    {
        visitedCells += 1;

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
            max(
                max(
                    max(d000, d100),
                    max(d010, d110)
                ),
                max(
                    max(d001, d101),
                    max(d011, d111)
                )
            );

        if (
            densityMaximum > 1e-7 &&
            segmentLength > 0.0
        )
        {
            integratedCells += 1;

            float c0;
            float c1;
            float c2;
            float c3;

            vec3 entryPoint =
                ro +
                rd * tEnter;

            buildTrilinearRayPolynomial(
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

            float cellTau =
                opticalDepthInCell(
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
                float localDistance =
                    invertOpticalDepthInCell(
                        c0,
                        c1,
                        c2,
                        c3,
                        segmentLength,
                        remainingTau
                    );

                collisionDistance =
                    tEnter +
                    localDistance;

                collisionPosition =
                    ro +
                    rd * collisionDistance;

                return true;
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

float analyticSunTransmittance(
    vec3 startPosition,
    vec3 sunDirection,
    out float sunOpticalDepth,
    out int visitedCells,
    out int integratedCells)
{
    sunOpticalDepth = 0.0;
    visitedCells = 0;
    integratedCells = 0;

    vec3 ro =
        startPosition +
        sunDirection * 2e-4;

    float volumeEnter;
    float volumeExit;

    if (
        !intersectBox(
            ro,
            sunDirection,
            VOLUME_MIN,
            VOLUME_MAX,
            volumeEnter,
            volumeExit
        )
    )
    {
        return 1.0;
    }

    // Start position is normally inside the box.
    volumeEnter =
        max(volumeEnter, 0.0);

    ivec3 cell;
    ivec3 stepDirection;

    vec3 tDelta;
    vec3 tMax;

    initializeDDA(
        ro,
        sunDirection,
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
        visitedCells += 1;

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
            max(
                max(
                    max(d000, d100),
                    max(d010, d110)
                ),
                max(
                    max(d001, d101),
                    max(d011, d111)
                )
            );

        if (
            densityMaximum > 1e-7 &&
            segmentLength > 0.0
        )
        {
            integratedCells += 1;

            float c0;
            float c1;
            float c2;
            float c3;

            vec3 entryPoint =
                ro +
                sunDirection * tEnter;

            buildTrilinearRayPolynomial(
                entryPoint,
                sunDirection,
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

            sunOpticalDepth +=
                opticalDepthInCell(
                    c0,
                    c1,
                    c2,
                    c3,
                    segmentLength
                );

            if (
                sunOpticalDepth >=
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
        -sunOpticalDepth
    );
}


// ============================================================
// Phase function
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


// ============================================================
// Sky
// ============================================================

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
            iResolution.x /
            iResolution.y,

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
// Heatmap
// ============================================================

vec3 heatmap(float x)
{
    x =
        clamp(x, 0.0, 1.0);

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


// ============================================================
// One Monte Carlo single-scattering sample
// ============================================================

vec3 traceSingleScatteringSample(
    vec3 ro,
    vec3 rd,
    vec2 pixel,
    int sampleIndex,
    out float debugCameraVisited,
    out float debugCameraIntegrated,
    out float debugSunVisited,
    out float debugSunTau,
    out float debugCollisionDistance)
{
    debugCameraVisited = 0.0;
    debugCameraIntegrated = 0.0;
    debugSunVisited = 0.0;
    debugSunTau = 0.0;
    debugCollisionDistance = 0.0;

    float u =
        sampleRandom(
            pixel,
            sampleIndex,
            0.0
        );

    float sampledOpticalDepth =
        -log(1.0 - u);

    float collisionDistance;
    vec3 collisionPosition;

    int cameraVisited;
    int cameraIntegrated;

    bool collided =
        sampleCameraFreePath(
            ro,
            rd,
            sampledOpticalDepth,
            collisionDistance,
            collisionPosition,
            cameraVisited,
            cameraIntegrated
        );

    debugCameraVisited =
        float(cameraVisited);

    debugCameraIntegrated =
        float(cameraIntegrated);

    if (!collided)
    {
        // Analog free-path estimator:
        // the probability of this no-collision branch already equals
        // camera transmittance, so no additional attenuation factor is
        // multiplied here.
        return skyRadiance(rd);
    }

    debugCollisionDistance =
        collisionDistance;

    float sunOpticalDepth;

    int sunVisited;
    int sunIntegrated;

    float sunTransmittance =
        analyticSunTransmittance(
            collisionPosition,
            SUN_DIRECTION,
            sunOpticalDepth,
            sunVisited,
            sunIntegrated
        );

    debugSunVisited =
        float(sunVisited);

    debugSunTau =
        sunOpticalDepth;

    // For directions:
    //
    // rd points from camera toward the scattering point.
    // SUN_DIRECTION points from the point toward the sun.
    //
    // dot(rd, SUN_DIRECTION) gives strong forward scattering when
    // looking toward the sun.
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

    // The sampled free-path PDF already contains:
    //
    // T_camera * sigma_t
    //
    // Therefore the remaining event weight is:
    //
    // sigma_s / sigma_t = scattering albedo
    //
    // followed by phase, sun transmittance, and sun radiance.
    vec3 singleScattering =
        SINGLE_SCATTERING_ALBEDO *
        phase *
        sunTransmittance *
        SUN_RADIANCE;

    return singleScattering;
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

    vec2 uv =
        fragCoord /
        resolution;

    vec3 ro;
    vec3 rd;

    makeCameraRay(
        uv,
        ro,
        rd
    );

    vec3 accumulatedColor =
        vec3(0.0);

    float cameraVisitedSum =
        0.0;

    float cameraIntegratedSum =
        0.0;

    float sunVisitedSum =
        0.0;

    float sunTauSum =
        0.0;

    float collisionDistanceSum =
        0.0;

    float collisionCount =
        0.0;

    vec2 pixel =
        floor(fragCoord);

    for (
        int sampleIndex = 0;
        sampleIndex < SAMPLES_PER_PIXEL;
        ++sampleIndex
    )
    {
        float cameraVisited;
        float cameraIntegrated;
        float sunVisited;
        float sunTau;
        float collisionDistance;

        vec3 sampleColor =
            traceSingleScatteringSample(
                ro,
                rd,
                pixel,
                sampleIndex,
                cameraVisited,
                cameraIntegrated,
                sunVisited,
                sunTau,
                collisionDistance
            );

        accumulatedColor +=
            sampleColor;

        cameraVisitedSum +=
            cameraVisited;

        cameraIntegratedSum +=
            cameraIntegrated;

        sunVisitedSum +=
            sunVisited;

        sunTauSum +=
            sunTau;

        if (collisionDistance > 0.0)
        {
            collisionDistanceSum +=
                collisionDistance;

            collisionCount +=
                1.0;
        }
    }

    float inverseSampleCount =
        1.0 /
        float(SAMPLES_PER_PIXEL);

    vec3 color =
        accumulatedColor *
        inverseSampleCount;

    float averageCameraVisited =
        cameraVisitedSum *
        inverseSampleCount;

    float averageCameraIntegrated =
        cameraIntegratedSum *
        inverseSampleCount;

    float averageSunVisited =
        sunVisitedSum *
        inverseSampleCount;

    float averageSunTau =
        sunTauSum *
        inverseSampleCount;

    float averageCollisionDistance =
        collisionCount > 0.0
        ? collisionDistanceSum /
          collisionCount
        : 0.0;

    // --------------------------------------------------------
    // Diagnostic views
    // --------------------------------------------------------

    if (DEBUG_MODE == 1)
    {
        float x =
            log2(
                1.0 +
                averageCameraVisited
            ) /
            log2(161.0);

        color =
            heatmap(x);
    }
    else if (DEBUG_MODE == 2)
    {
        float x =
            log2(
                1.0 +
                averageCameraIntegrated
            ) /
            log2(161.0);

        color =
            heatmap(x);
    }
    else if (DEBUG_MODE == 3)
    {
        float x =
            log2(
                1.0 +
                averageSunVisited
            ) /
            log2(161.0);

        color =
            heatmap(x);
    }
    else if (DEBUG_MODE == 4)
    {
        float x =
            averageSunTau /
            SUN_TAU_CUTOFF;

        color =
            heatmap(x);
    }
    else if (DEBUG_MODE == 5)
    {
        float diagonal =
            length(
                VOLUME_MAX -
                VOLUME_MIN
            );

        float x =
            averageCollisionDistance /
            diagonal;

        color =
            heatmap(x);
    }

    // Simple filmic-style compression.
    color =
        color /
        (
            vec3(1.0) +
            color
        );

    // Gamma correction.
    color =
        pow(
            max(color, vec3(0.0)),
            vec3(1.0 / 2.2)
        );

    fragColor =
        vec4(color, 1.0);
}