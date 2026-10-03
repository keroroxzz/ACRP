// ============================================================
// Fair Same-Ray Trilinear Volume Benchmark
//
// TOP LEFT:
//   Analytic 3D DDA traversal
//   + exact cell optical-depth integration
//   + one final quartic inversion
//
// TOP RIGHT:
//   Uniform midpoint integration
//
// BOTTOM LEFT, four full-view strips:
//   1. DDA visited cells
//   2. Analytically integrated cells
//   3. Uniform samples
//   4. Quartic inversion iterations
//
// BOTTOM RIGHT, three full-view strips:
//   1. Uniform / analytic-integrated query ratio
//   2. Collision-distance comparison
//   3. Transmittance error
//
// Every strip remaps its own horizontal coordinate to [0,1].
// Therefore every displayed metric covers the SAME full camera view.
//
// Analytic and uniform methods use the SAME random optical-depth
// sample for the same canonical ray.
//
// Paste into Shadertoy Image shader.
// Hold mouse and drag to rotate camera.
// ============================================================

#define PI 3.14159265359

const float INF = 1e30;
const float EPS = 1e-6;

const vec3 VOLUME_MIN = vec3(-8.0, -1.0, -8.0);
const vec3 VOLUME_MAX = vec3( 8.0,  7.0,  8.0);

const float SIGMA_SCALE = 1.35;

// Analytic traversal.
const int MAX_DDA_CELLS = 128;
const int QUARTIC_BISECTION_STEPS = 18;

// Uniform baseline.
const float UNIFORM_STEP = 0.10;
const int MAX_UNIFORM_STEPS = 320;

// Visualization.
const float COST_DISPLAY_MAX = 320.0;
const float RATIO_DISPLAY_MAX = 16.0;
const float TRAVEL_ERROR_MAX = 0.35;
const float TRANSMITTANCE_ERROR_MAX = 0.08;


// ============================================================
// Random / hash utilities
// ============================================================

float hash31(vec3 p)
{
    return fract(
        sin(dot(p, vec3(127.1, 311.7, 74.7))) *
        43758.5453123
    );
}


float pixelRandom(vec2 pixel)
{
    return fract(
        sin(dot(pixel, vec2(12.9898, 78.233))) *
        43758.5453
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

    float verticalEnvelope =
        smoothstep(0.02, 0.22, normalizedHeight) *
        (
            1.0 -
            smoothstep(0.60, 0.98, normalizedHeight)
        );

    float radial =
        length(p.xz) / 10.0;

    float radialEnvelope =
        1.0 -
        smoothstep(0.45, 1.05, radial);

    float largeScale =
        0.55 +
        0.45 *
        sin(
            0.43 * p.x +
            0.31 * p.z +
            0.17 * p.y
        );

    float n0 =
        hash31(p);

    float n1 =
        hash31(
            p * 0.47 +
            vec3(19.0, 7.0, 31.0)
        );

    float density =
        mix(n0, n1, 0.35);

    density =
        smoothstep(
            0.40,
            0.78,
            density * largeScale + 0.26
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
// Ray / box intersection
// ============================================================

bool intersectBox(
    vec3 ro,
    vec3 rd,
    vec3 boxMin,
    vec3 boxMax,
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

    vec3 inverseDirection =
        1.0 / safeRd;

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
// Trilinear density restricted to a ray
//
// rho(s) = c0 + c1*s + c2*s^2 + c3*s^3
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
    r1 = a0 * b1 + a1 * b0;
    r2 = a1 * b1;
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
    r0 = q0 * b0;
    r1 = q0 * b1 + q1 * b0;
    r2 = q1 * b1 + q2 * b0;
    r3 = q2 * b1;
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

    // x
    c0 += ax * f.x;
    c1 += ax * rd.x;

    // y
    c0 += ay * f.y;
    c1 += ay * rd.y;

    // z
    c0 += az * f.z;
    c1 += az * rd.z;

    // xy
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

    // xz
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

    // yz
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

    // xyz
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
// Analytic optical-depth integration
// ============================================================

float integrateDensityCubic(
    float c0,
    float c1,
    float c2,
    float c3,
    float s)
{
    float s2 = s * s;
    float s3 = s2 * s;
    float s4 = s3 * s;

    return
        c0 * s +
        0.5 * c1 * s2 +
        (1.0 / 3.0) * c2 * s3 +
        0.25 * c3 * s4;
}


float opticalDepthInCell(
    float c0,
    float c1,
    float c2,
    float c3,
    float s)
{
    return
        SIGMA_SCALE *
        max(
            integrateDensityCubic(
                c0,
                c1,
                c2,
                c3,
                s
            ),
            0.0
        );
}


// The cumulative optical-depth polynomial is monotone because
// the underlying trilinear density is nonnegative.
float invertOpticalDepthQuartic(
    float c0,
    float c1,
    float c2,
    float c3,
    float intervalLength,
    float targetTau,
    out int refinementIterations)
{
    float lo = 0.0;
    float hi = intervalLength;

    refinementIterations = 0;

    for (
        int i = 0;
        i < QUARTIC_BISECTION_STEPS;
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

        refinementIterations =
            i + 1;

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
// Analytic 3D DDA traversal
// ============================================================

bool traceAnalyticMedium(
    vec3 ro,
    vec3 rd,
    float sampledTau,
    out float tCollision,
    out float totalTau,
    out int visitedBeforeEvent,
    out int integratedBeforeEvent,
    out int inversionIterations)
{
    tCollision = INF;
    totalTau = 0.0;

    visitedBeforeEvent = 0;
    integratedBeforeEvent = 0;
    inversionIterations = 0;

    float boxEnter;
    float boxExit;

    if (
        !intersectBox(
            ro,
            rd,
            VOLUME_MIN,
            VOLUME_MAX,
            boxEnter,
            boxExit
        )
    )
    {
        return false;
    }

    vec3 startPoint =
        ro +
        rd * (boxEnter + 1e-5);

    ivec3 cell =
        ivec3(floor(startPoint));

    ivec3 stepDirection =
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

    vec3 tDelta =
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

    vec3 tMax =
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
            vec3(boxEnter)
        );

    float tEnter =
        boxEnter;

    float remainingTau =
        sampledTau;

    bool collisionFound =
        false;

    for (
        int cellIndex = 0;
        cellIndex < MAX_DDA_CELLS;
        ++cellIndex
    )
    {
        float tExit =
            min(
                min(
                    tMax.x,
                    min(tMax.y, tMax.z)
                ),
                boxExit
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

        if (!collisionFound)
        {
            visitedBeforeEvent += 1;
        }

        float cellTau = 0.0;

        if (
            densityMaximum > 1e-7 &&
            segmentLength > 0.0
        )
        {
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

            cellTau =
                opticalDepthInCell(
                    c0,
                    c1,
                    c2,
                    c3,
                    segmentLength
                );

            if (!collisionFound)
            {
                integratedBeforeEvent += 1;

                if (
                    cellTau >= remainingTau &&
                    cellTau > 0.0
                )
                {
                    int refinementCount;

                    float localDistance =
                        invertOpticalDepthQuartic(
                            c0,
                            c1,
                            c2,
                            c3,
                            segmentLength,
                            remainingTau,
                            refinementCount
                        );

                    inversionIterations =
                        refinementCount;

                    tCollision =
                        tEnter +
                        localDistance;

                    collisionFound =
                        true;
                }
                else
                {
                    remainingTau -=
                        cellTau;
                }
            }
        }

        totalTau +=
            cellTau;

        if (
            tExit >=
            boxExit - 1e-5
        )
        {
            break;
        }

        bool crossX =
            tMax.x <=
            min(tMax.y, tMax.z) + 1e-5;

        bool crossY =
            tMax.y <=
            min(tMax.x, tMax.z) + 1e-5;

        bool crossZ =
            tMax.z <=
            min(tMax.x, tMax.y) + 1e-5;

        tEnter =
            tExit;

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

    return collisionFound;
}


// ============================================================
// Uniform midpoint integration
// ============================================================

bool traceUniformMedium(
    vec3 ro,
    vec3 rd,
    float sampledTau,
    out float tCollision,
    out float totalTau,
    out int samplesBeforeEvent)
{
    tCollision = INF;
    totalTau = 0.0;
    samplesBeforeEvent = 0;

    float boxEnter;
    float boxExit;

    if (
        !intersectBox(
            ro,
            rd,
            VOLUME_MIN,
            VOLUME_MAX,
            boxEnter,
            boxExit
        )
    )
    {
        return false;
    }

    float remainingTau =
        sampledTau;

    bool collisionFound =
        false;

    float t =
        boxEnter;

    for (
        int i = 0;
        i < MAX_UNIFORM_STEPS;
        ++i
    )
    {
        if (t >= boxExit)
            break;

        float nextT =
            min(
                t + UNIFORM_STEP,
                boxExit
            );

        float segmentLength =
            nextT - t;

        float midpoint =
            0.5 * (t + nextT);

        float density =
            densityField(
                ro +
                rd * midpoint
            );

        float deltaTau =
            SIGMA_SCALE *
            density *
            segmentLength;

        if (!collisionFound)
        {
            samplesBeforeEvent += 1;

            if (
                deltaTau >= remainingTau &&
                deltaTau > 0.0
            )
            {
                float localFraction =
                    remainingTau /
                    deltaTau;

                tCollision =
                    mix(
                        t,
                        nextT,
                        clamp(
                            localFraction,
                            0.0,
                            1.0
                        )
                    );

                collisionFound =
                    true;
            }
            else
            {
                remainingTau -=
                    deltaTau;
            }
        }

        totalTau +=
            deltaTau;

        t =
            nextT;
    }

    return collisionFound;
}


// ============================================================
// Camera
// ============================================================

mat3 cameraBasis(
    vec3 eye,
    vec3 target)
{
    vec3 forward =
        normalize(target - eye);

    vec3 right =
        normalize(
            cross(
                forward,
                vec3(0.0, 1.0, 0.0)
            )
        );

    vec3 up =
        cross(right, forward);

    return mat3(
        right,
        up,
        forward
    );
}


void makeCameraRay(
    vec2 uv,
    float aspect,
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
                0.02,
                0.65,
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
            0.12 * iTime;

        pitch =
            0.23;
    }

    vec3 target =
        vec3(0.0, 2.4, 0.0);

    float radius =
        18.0;

    ro =
        target +
        vec3(
            radius *
            cos(pitch) *
            sin(yaw),

            radius *
            sin(pitch) +
            2.5,

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
            (uv.x - 0.5) *
            aspect,

            uv.y - 0.5
        );

    rd =
        normalize(
            camera *
            vec3(
                screen,
                1.35
            )
        );
}


// ============================================================
// Rendering
// ============================================================

vec3 skyColor(vec3 rd)
{
    float h =
        clamp(
            0.5 +
            0.5 * rd.y,
            0.0,
            1.0
        );

    vec3 horizon =
        vec3(0.72, 0.83, 0.98);

    vec3 zenith =
        vec3(0.12, 0.27, 0.55);

    vec3 sky =
        mix(
            horizon,
            zenith,
            h
        );

    vec3 sunDirection =
        normalize(
            vec3(0.55, 0.70, 0.30)
        );

    float sun =
        pow(
            max(
                dot(rd, sunDirection),
                0.0
            ),
            500.0
        );

    sky +=
        vec3(1.0, 0.88, 0.62) *
        sun *
        3.0;

    return sky;
}


vec3 shadeVolumeEvent(
    vec3 ro,
    vec3 rd,
    float tCollision,
    float totalTau)
{
    vec3 p =
        ro +
        rd * tCollision;

    float density =
        densityField(p);

    vec3 lightDirection =
        normalize(
            vec3(0.55, 0.70, 0.30)
        );

    float phase =
        0.45 +
        0.55 *
        pow(
            max(
                dot(rd, lightDirection),
                0.0
            ),
            2.0
        );

    float vertical =
        clamp(
            (
                p.y -
                VOLUME_MIN.y
            ) /
            (
                VOLUME_MAX.y -
                VOLUME_MIN.y
            ),
            0.0,
            1.0
        );

    vec3 cloudColor =
        mix(
            vec3(0.55, 0.62, 0.72),
            vec3(1.00, 0.97, 0.90),
            0.45 +
            0.55 * vertical
        );

    cloudColor *=
        0.55 +
        0.75 * phase;

    cloudColor *=
        0.65 +
        0.70 *
        sqrt(max(density, 0.0));

    float backgroundT =
        exp(-totalTau);

    return mix(
        cloudColor,
        skyColor(rd),
        0.12 * backgroundT
    );
}


vec3 renderMediumResult(
    vec3 ro,
    vec3 rd,
    bool hit,
    float tCollision,
    float totalTau)
{
    float transmittance =
        exp(-totalTau);

    if (hit)
    {
        return shadeVolumeEvent(
            ro,
            rd,
            tCollision,
            totalTau
        );
    }

    return
        skyColor(rd) *
        (
            0.25 +
            0.75 *
            transmittance
        );
}


// ============================================================
// Visualization
// ============================================================

float normalizeCost(int count)
{
    return clamp(
        log2(
            1.0 +
            float(count)
        ) /
        log2(
            1.0 +
            COST_DISPLAY_MAX
        ),
        0.0,
        1.0
    );
}


vec3 heatmap(float x)
{
    x =
        clamp(x, 0.0, 1.0);

    vec3 blue =
        vec3(0.03, 0.10, 0.55);

    vec3 cyan =
        vec3(0.00, 0.76, 1.00);

    vec3 green =
        vec3(0.05, 0.90, 0.18);

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


vec3 ratioHeatmap(float ratio)
{
    float normalizedRatio =
        clamp(
            log2(max(ratio, 1.0)) /
            log2(RATIO_DISPLAY_MAX),
            0.0,
            1.0
        );

    return heatmap(
        normalizedRatio
    );
}


vec3 collisionComparisonColor(
    bool hitAnalytic,
    bool hitUniform,
    float tAnalytic,
    float tUniform)
{
    if (
        hitAnalytic &&
        !hitUniform
    )
    {
        // Uniform false negative.
        return vec3(1.0, 0.0, 1.0);
    }

    if (
        !hitAnalytic &&
        hitUniform
    )
    {
        // Analytic miss / possible implementation error.
        return vec3(0.0, 1.0, 1.0);
    }

    if (
        !hitAnalytic &&
        !hitUniform
    )
    {
        return vec3(0.02, 0.03, 0.09);
    }

    float error =
        abs(
            tUniform -
            tAnalytic
        );

    float x =
        clamp(
            log2(
                1.0 +
                20.0 * error
            ) /
            log2(
                1.0 +
                20.0 *
                TRAVEL_ERROR_MAX
            ),
            0.0,
            1.0
        );

    vec3 accurate =
        vec3(0.04, 0.76, 0.12);

    vec3 moderate =
        vec3(1.00, 0.85, 0.02);

    vec3 bad =
        vec3(1.00, 0.02, 0.01);

    if (x < 0.5)
    {
        return mix(
            accurate,
            moderate,
            x / 0.5
        );
    }

    return mix(
        moderate,
        bad,
        (x - 0.5) / 0.5
    );
}


vec3 transmittanceErrorColor(
    float analyticTau,
    float uniformTau)
{
    float analyticT =
        exp(-analyticTau);

    float uniformT =
        exp(-uniformTau);

    float error =
        abs(
            analyticT -
            uniformT
        );

    return heatmap(
        clamp(
            error /
            TRANSMITTANCE_ERROR_MAX,
            0.0,
            1.0
        )
    );
}


// ============================================================
// Canonical panel mappings
// ============================================================

// Maps one strip within a panel back to the full [0,1] view.
float remapStripX(
    float localX,
    float stripCount)
{
    return fract(
        localX * stripCount
    );
}


// Stable canonical pixel coordinate used for random sampling.
// Same canonical UV => same U => same sampled optical depth.
vec2 canonicalRandomPixel(
    vec2 canonicalUV,
    vec2 resolution)
{
    vec2 canonicalResolution =
        vec2(
            0.5 * resolution.x,
            0.5 * resolution.y
        );

    return floor(
        canonicalUV *
        canonicalResolution
    );
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

    bool leftHalf =
        fragCoord.x <
        0.5 * resolution.x;

    bool topHalf =
        fragCoord.y >=
        0.5 * resolution.y;

    float localX =
        leftHalf
        ? fragCoord.x /
          (0.5 * resolution.x)
        : (
            fragCoord.x -
            0.5 * resolution.x
          ) /
          (0.5 * resolution.x);

    float localY =
        topHalf
        ? (
            fragCoord.y -
            0.5 * resolution.y
          ) /
          (0.5 * resolution.y)
        : fragCoord.y /
          (0.5 * resolution.y);

    // --------------------------------------------------------
    // Canonical full-view UV
    // --------------------------------------------------------

    float canonicalX =
        localX;

    int metricIndex =
        0;

    if (!topHalf && leftHalf)
    {
        // Four metric strips.
        canonicalX =
            remapStripX(
                localX,
                4.0
            );

        metricIndex =
            int(
                floor(
                    min(localX, 0.999999) *
                    4.0
                )
            );
    }
    else if (!topHalf && !leftHalf)
    {
        // Three comparison strips.
        canonicalX =
            remapStripX(
                localX,
                3.0
            );

        metricIndex =
            int(
                floor(
                    min(localX, 0.999999) *
                    3.0
                )
            );
    }

    vec2 canonicalUV =
        vec2(
            canonicalX,
            localY
        );

    float panelAspect =
        (0.5 * resolution.x) /
        (0.5 * resolution.y);

    vec3 ro;
    vec3 rd;

    makeCameraRay(
        canonicalUV,
        panelAspect,
        ro,
        rd
    );

    // --------------------------------------------------------
    // Same random optical-depth sample for the same ray.
    // --------------------------------------------------------

    vec2 randomPixel =
        canonicalRandomPixel(
            canonicalUV,
            resolution
        );

    float u =
        clamp(
            pixelRandom(
                randomPixel +
                vec2(17.0, 43.0)
            ),
            1e-5,
            1.0 - 1e-5
        );

    float sampledTau =
        -log(1.0 - u);

    // --------------------------------------------------------
    // Run both methods.
    // --------------------------------------------------------

    float tAnalytic;
    float analyticTotalTau;

    int visitedCells;
    int integratedCells;
    int inversionIterations;

    bool hitAnalytic =
        traceAnalyticMedium(
            ro,
            rd,
            sampledTau,
            tAnalytic,
            analyticTotalTau,
            visitedCells,
            integratedCells,
            inversionIterations
        );

    float tUniform;
    float uniformTotalTau;
    int uniformSamples;

    bool hitUniform =
        traceUniformMedium(
            ro,
            rd,
            sampledTau,
            tUniform,
            uniformTotalTau,
            uniformSamples
        );

    vec3 color;

    // --------------------------------------------------------
    // Top-left: analytic result.
    // --------------------------------------------------------
    if (topHalf && leftHalf)
    {
        color =
            renderMediumResult(
                ro,
                rd,
                hitAnalytic,
                tAnalytic,
                analyticTotalTau
            );
    }

    // --------------------------------------------------------
    // Top-right: uniform result.
    // --------------------------------------------------------
    else if (topHalf && !leftHalf)
    {
        color =
            renderMediumResult(
                ro,
                rd,
                hitUniform,
                tUniform,
                uniformTotalTau
            );
    }

    // --------------------------------------------------------
    // Bottom-left: four same-view cost strips.
    //
    // 0: visited cells
    // 1: integrated cells
    // 2: uniform samples
    // 3: inversion iterations
    // --------------------------------------------------------
    else if (!topHalf && leftHalf)
    {
        int shownCount = 0;

        if (metricIndex == 0)
        {
            shownCount =
                visitedCells;
        }
        else if (metricIndex == 1)
        {
            shownCount =
                integratedCells;
        }
        else if (metricIndex == 2)
        {
            shownCount =
                uniformSamples;
        }
        else
        {
            shownCount =
                inversionIterations;
        }

        color =
            heatmap(
                normalizeCost(
                    shownCount
                )
            );

        float stripWidthPixels =
            0.5 * resolution.x / 4.0;

        float positionInStrip =
            mod(
                fragCoord.x,
                stripWidthPixels
            );

        if (
            positionInStrip < 1.5 ||
            stripWidthPixels -
            positionInStrip < 1.5
        )
        {
            color =
                vec3(0.96);
        }
    }

    // --------------------------------------------------------
    // Bottom-right: three same-view comparison strips.
    //
    // 0: uniform / integrated query ratio
    // 1: collision comparison
    // 2: transmittance error
    // --------------------------------------------------------
    else
    {
        if (metricIndex == 0)
        {
            float ratio =
                float(uniformSamples) /
                max(
                    float(integratedCells),
                    1.0
                );

            color =
                ratioHeatmap(ratio);
        }
        else if (metricIndex == 1)
        {
            color =
                collisionComparisonColor(
                    hitAnalytic,
                    hitUniform,
                    tAnalytic,
                    tUniform
                );
        }
        else
        {
            color =
                transmittanceErrorColor(
                    analyticTotalTau,
                    uniformTotalTau
                );
        }

        float stripWidthPixels =
            0.5 * resolution.x / 3.0;

        float positionInStrip =
            mod(
                fragCoord.x -
                0.5 * resolution.x,
                stripWidthPixels
            );

        if (
            positionInStrip < 1.5 ||
            stripWidthPixels -
            positionInStrip < 1.5
        )
        {
            color =
                vec3(0.96);
        }
    }

    // Main dividers.
    if (
        abs(
            fragCoord.x -
            0.5 * resolution.x
        ) < 1.5
    )
    {
        color =
            vec3(0.96);
    }

    if (
        abs(
            fragCoord.y -
            0.5 * resolution.y
        ) < 1.5
    )
    {
        color =
            vec3(0.96);
    }

    color =
        pow(
            max(color, vec3(0.0)),
            vec3(1.0 / 2.2)
        );

    fragColor =
        vec4(color, 1.0);
}