// ============================================================
// Bilinear Procedural Heightfield Benchmark
//
// Top-left:
//   Exact 2D DDA + cell height-range rejection
//   + exact quadratic intersection
//
// Top-right:
//   Uniform fixed-step ray marching
//
// Bottom-left, left half:
//   DDA visited-cell count
//
// Bottom-left, right half:
//   Actual quadratic-test count after range rejection
//
// Bottom-right:
//   Hit/miss classification and Travel error
//
// Shadertoy single Image shader
// Mouse drag: rotate camera
// ============================================================

#define PI 3.14159265359

const float INF = 1e30;
const float EPS = 1e-6;
const float HIT_EPS = 1e-5;

const float MAX_DISTANCE = 80.0;

const int MAX_DDA_CELLS = 180;
const int MAX_UNIFORM_STEPS = 640;
const int BISECTION_STEPS = 10;

const float UNIFORM_STEP = 0.125;

// Common log-scale maximum for cost displays.
const float COST_DISPLAY_MAX = 640.0;

// Error visualization reaches red around this value.
const float ERROR_DISPLAY_MAX = 0.25;


// ============================================================
// Procedural height field
// ============================================================

float hash21(ivec2 p)
{
    vec2 q = vec2(p);

    return fract(
        sin(dot(q, vec2(127.1, 311.7))) *
        43758.5453123
    );
}


float latticeHeight(ivec2 p)
{
    float n = hash21(p);

    return -0.5 + 2.5 * n;
}


void loadCellHeights(
    ivec2 cell,
    out float h00,
    out float h10,
    out float h01,
    out float h11)
{
    h00 = latticeHeight(cell + ivec2(0, 0));
    h10 = latticeHeight(cell + ivec2(1, 0));
    h01 = latticeHeight(cell + ivec2(0, 1));
    h11 = latticeHeight(cell + ivec2(1, 1));
}


float terrainHeight(vec2 xz)
{
    ivec2 cell = ivec2(floor(xz));
    vec2 f = xz - vec2(cell);

    float h00;
    float h10;
    float h01;
    float h11;

    loadCellHeights(
        cell,
        h00,
        h10,
        h01,
        h11
    );

    float hx0 = mix(h00, h10, f.x);
    float hx1 = mix(h01, h11, f.x);

    return mix(hx0, hx1, f.y);
}


vec3 terrainNormal(vec2 xz)
{
    ivec2 cell = ivec2(floor(xz));
    vec2 f = xz - vec2(cell);

    float h00;
    float h10;
    float h01;
    float h11;

    loadCellHeights(
        cell,
        h00,
        h10,
        h01,
        h11
    );

    float b = h10 - h00;
    float c = h01 - h00;
    float d = h11 - h10 - h01 + h00;

    float dhdx = b + d * f.y;
    float dhdz = c + d * f.x;

    return normalize(
        vec3(-dhdx, 1.0, -dhdz)
    );
}


float terrainClearance(vec3 p)
{
    return p.y - terrainHeight(p.xz);
}


// ============================================================
// Cell range rejection
// ============================================================

bool cellHeightRangeCanIntersect(
    vec3 ro,
    vec3 rd,
    float tEnter,
    float tExit,
    float h00,
    float h10,
    float h01,
    float h11)
{
    // A bilinear function on a rectangle remains within
    // the min/max of its four corner values.
    float terrainMin =
        min(
            min(h00, h10),
            min(h01, h11)
        );

    float terrainMax =
        max(
            max(h00, h10),
            max(h01, h11)
        );

    float y0 = ro.y + rd.y * tEnter;
    float y1 = ro.y + rd.y * tExit;

    float rayMin = min(y0, y1);
    float rayMax = max(y0, y1);

    if (rayMax < terrainMin - HIT_EPS)
        return false;

    if (rayMin > terrainMax + HIT_EPS)
        return false;

    return true;
}


// ============================================================
// Stable polynomial roots
// ============================================================

float firstRootInInterval(
    float A,
    float B,
    float C,
    float tMin,
    float tMax)
{
    if (abs(A) < EPS)
    {
        if (abs(B) < EPS)
        {
            if (abs(C) < HIT_EPS)
                return tMin;

            return INF;
        }

        float t = -C / B;

        if (
            t >= tMin - HIT_EPS &&
            t <= tMax + HIT_EPS
        )
        {
            return clamp(t, tMin, tMax);
        }

        return INF;
    }

    float discriminant =
        B * B - 4.0 * A * C;

    if (discriminant < -EPS)
        return INF;

    discriminant = max(discriminant, 0.0);

    float sqrtD = sqrt(discriminant);

    float q =
        -0.5 *
        (
            B +
            (
                B >= 0.0
                ? sqrtD
                : -sqrtD
            )
        );

    float t0;
    float t1;

    if (abs(q) < EPS)
    {
        t0 = -B / (2.0 * A);
        t1 = t0;
    }
    else
    {
        t0 = q / A;
        t1 = C / q;
    }

    if (t0 > t1)
    {
        float tmp = t0;
        t0 = t1;
        t1 = tmp;
    }

    if (
        t0 >= tMin - HIT_EPS &&
        t0 <= tMax + HIT_EPS
    )
    {
        return clamp(t0, tMin, tMax);
    }

    if (
        t1 >= tMin - HIT_EPS &&
        t1 <= tMax + HIT_EPS
    )
    {
        return clamp(t1, tMin, tMax);
    }

    return INF;
}


// ============================================================
// Exact bilinear-cell intersection
// ============================================================

float intersectBilinearCellLoaded(
    vec3 ro,
    vec3 rd,
    ivec2 cell,
    float tEnter,
    float tExit,
    float h00,
    float h10,
    float h01,
    float h11)
{
    float intervalLength = tExit - tEnter;

    if (intervalLength < -HIT_EPS)
        return INF;

    vec3 p = ro + rd * tEnter;

    float xi =
        p.x - float(cell.x);

    float eta =
        p.z - float(cell.y);

    float a = h00;
    float b = h10 - h00;
    float c = h01 - h00;

    float d =
        h11 - h10 - h01 + h00;

    float u = rd.x;
    float v = rd.z;
    float w = rd.y;

    float heightAtEntry =
        a +
        b * xi +
        c * eta +
        d * xi * eta;

    // Clearance in local parameter s:
    //
    // C(s) = A s² + B s + C
    float A =
        -d * u * v;

    float B =
        w -
        (
            b * u +
            c * v +
            d * (xi * v + eta * u)
        );

    float C =
        p.y - heightAtEntry;

    float localHit =
        firstRootInInterval(
            A,
            B,
            C,
            0.0,
            max(intervalLength, 0.0)
        );

    if (localHit >= INF * 0.5)
        return INF;

    return tEnter + localHit;
}


// ============================================================
// Exact DDA with rejection counters
// ============================================================

bool traceExactDDA(
    vec3 ro,
    vec3 rd,
    out float tHit,
    out int visitedCells,
    out int quadraticTests,
    out int rejectedCells)
{
    tHit = INF;
    visitedCells = 0;
    quadraticTests = 0;
    rejectedCells = 0;

    vec2 hd = rd.xz;

    vec2 startXZ =
        ro.xz + hd * 1e-5;

    ivec2 cell =
        ivec2(floor(startXZ));

    int stepX = 0;
    int stepZ = 0;

    if (hd.x > EPS)
        stepX = 1;
    else if (hd.x < -EPS)
        stepX = -1;

    if (hd.y > EPS)
        stepZ = 1;
    else if (hd.y < -EPS)
        stepZ = -1;

    float tDeltaX =
        abs(hd.x) > EPS
        ? abs(1.0 / hd.x)
        : INF;

    float tDeltaZ =
        abs(hd.y) > EPS
        ? abs(1.0 / hd.y)
        : INF;

    float nextBoundaryX =
        stepX > 0
        ? float(cell.x + 1)
        : float(cell.x);

    float nextBoundaryZ =
        stepZ > 0
        ? float(cell.y + 1)
        : float(cell.y);

    float tMaxX =
        stepX != 0
        ? (
            nextBoundaryX - ro.x
        ) / hd.x
        : INF;

    float tMaxZ =
        stepZ != 0
        ? (
            nextBoundaryZ - ro.z
        ) / hd.y
        : INF;

    tMaxX = max(tMaxX, 0.0);
    tMaxZ = max(tMaxZ, 0.0);

    float tEnter = 0.0;

    for (
        int i = 0;
        i < MAX_DDA_CELLS;
        ++i
    )
    {
        visitedCells = i + 1;

        float tExit =
            min(
                min(tMaxX, tMaxZ),
                MAX_DISTANCE
            );

        float h00;
        float h10;
        float h01;
        float h11;

        loadCellHeights(
            cell,
            h00,
            h10,
            h01,
            h11
        );

        bool possible =
            cellHeightRangeCanIntersect(
                ro,
                rd,
                tEnter,
                tExit,
                h00,
                h10,
                h01,
                h11
            );

        if (possible)
        {
            quadraticTests += 1;

            float candidate =
                intersectBilinearCellLoaded(
                    ro,
                    rd,
                    cell,
                    tEnter,
                    tExit,
                    h00,
                    h10,
                    h01,
                    h11
                );

            if (
                candidate < INF * 0.5 &&
                candidate >= 0.0 &&
                candidate <= MAX_DISTANCE
            )
            {
                tHit = candidate;
                return true;
            }
        }
        else
        {
            rejectedCells += 1;
        }

        if (tExit >= MAX_DISTANCE)
            break;

        bool crossX =
            tMaxX <= tMaxZ + HIT_EPS;

        bool crossZ =
            tMaxZ <= tMaxX + HIT_EPS;

        tEnter = tExit;

        if (crossX)
        {
            cell.x += stepX;
            tMaxX += tDeltaX;
        }

        if (crossZ)
        {
            cell.y += stepZ;
            tMaxZ += tDeltaZ;
        }

        if (stepX == 0 && stepZ == 0)
            break;
    }

    return false;
}


// ============================================================
// Uniform fixed-step baseline
// ============================================================

float refineCrossingBisection(
    vec3 ro,
    vec3 rd,
    float ta,
    float tb)
{
    float ca =
        terrainClearance(ro + rd * ta);

    for (
        int i = 0;
        i < BISECTION_STEPS;
        ++i
    )
    {
        float tm =
            0.5 * (ta + tb);

        float cm =
            terrainClearance(ro + rd * tm);

        bool sameSign =
            (
                ca >= 0.0 &&
                cm >= 0.0
            ) ||
            (
                ca <= 0.0 &&
                cm <= 0.0
            );

        if (sameSign)
        {
            ta = tm;
            ca = cm;
        }
        else
        {
            tb = tm;
        }
    }

    return 0.5 * (ta + tb);
}


bool traceUniformMarch(
    vec3 ro,
    vec3 rd,
    out float tHit,
    out int samples)
{
    tHit = INF;
    samples = 0;

    float previousT = 0.0;

    float previousClearance =
        terrainClearance(ro);

    if (abs(previousClearance) < HIT_EPS)
    {
        tHit = 0.0;
        return true;
    }

    for (
        int i = 0;
        i < MAX_UNIFORM_STEPS;
        ++i
    )
    {
        float currentT =
            float(i + 1) *
            UNIFORM_STEP;

        samples = i + 1;

        if (currentT > MAX_DISTANCE)
            break;

        float currentClearance =
            terrainClearance(
                ro + rd * currentT
            );

        if (
            abs(currentClearance) <
            HIT_EPS
        )
        {
            tHit = currentT;
            return true;
        }

        if (
            previousClearance *
            currentClearance < 0.0
        )
        {
            tHit =
                refineCrossingBisection(
                    ro,
                    rd,
                    previousT,
                    currentT
                );

            return true;
        }

        previousT = currentT;
        previousClearance =
            currentClearance;
    }

    return false;
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
            max(iResolution.xy, vec2(1.0));

        yaw =
            (mouse.x - 0.5) *
            2.0 * PI;

        pitch =
            mix(
                0.10,
                0.75,
                clamp(mouse.y, 0.0, 1.0)
            );
    }
    else
    {
        yaw = 0.18 * iTime;
        pitch = 0.32;
    }

    vec3 target =
        vec3(0.0, 0.5, 0.0);

    float radius = 10.0;

    ro =
        target +
        vec3(
            radius *
            cos(pitch) *
            sin(yaw),

            radius *
            sin(pitch) +
            3.0,

            radius *
            cos(pitch) *
            cos(yaw)
        );

    mat3 cam =
        cameraBasis(ro, target);

    vec2 p =
        vec2(
            (uv.x - 0.5) * aspect,
            uv.y - 0.5
        );

    rd =
        normalize(
            cam *
            vec3(p, 1.35)
        );
}


// ============================================================
// Shading
// ============================================================

vec3 skyColor(vec3 rd)
{
    float h =
        clamp(
            0.5 + 0.5 * rd.y,
            0.0,
            1.0
        );

    vec3 horizon =
        vec3(0.72, 0.82, 0.96);

    vec3 zenith =
        vec3(0.12, 0.28, 0.55);

    vec3 sky =
        mix(horizon, zenith, h);

    vec3 sunDirection =
        normalize(
            vec3(0.55, 0.75, 0.30)
        );

    float sun =
        pow(
            max(dot(rd, sunDirection), 0.0),
            500.0
        );

    sky +=
        vec3(1.0, 0.85, 0.55) *
        sun * 3.0;

    return sky;
}


vec3 terrainMaterial(
    vec3 p,
    vec3 n)
{
    float slope =
        1.0 -
        clamp(n.y, 0.0, 1.0);

    vec3 grass =
        vec3(0.16, 0.34, 0.11);

    vec3 rock =
        vec3(0.32, 0.30, 0.27);

    vec3 high =
        vec3(0.46, 0.42, 0.31);

    vec3 base =
        mix(
            grass,
            rock,
            smoothstep(0.15, 0.70, slope)
        );

    base =
        mix(
            base,
            high,
            smoothstep(1.0, 2.0, p.y)
        );

    vec2 f =
        abs(fract(p.xz) - 0.5);

    float edge =
        smoothstep(
            0.47,
            0.495,
            max(f.x, f.y)
        );

    base *=
        mix(1.0, 0.68, edge);

    return base;
}


vec3 shadeTerrain(
    vec3 ro,
    vec3 rd,
    float tHit)
{
    vec3 p =
        ro + rd * tHit;

    vec3 n =
        terrainNormal(p.xz);

    vec3 lightDir =
        normalize(
            vec3(0.55, 0.75, 0.30)
        );

    float diffuse =
        max(dot(n, lightDir), 0.0);

    float hemisphere =
        0.35 +
        0.65 *
        max(n.y, 0.0);

    vec3 albedo =
        terrainMaterial(p, n);

    vec3 color =
        albedo *
        (
            vec3(0.20, 0.25, 0.32) *
            hemisphere +

            vec3(1.00, 0.92, 0.72) *
            diffuse
        );

    float fog =
        1.0 -
        exp(-0.018 * tHit * tHit);

    return mix(
        color,
        skyColor(rd),
        clamp(fog, 0.0, 1.0)
    );
}


// ============================================================
// Visualization
// ============================================================

float normalizeCost(int count)
{
    return clamp(
        log2(1.0 + float(count)) /
        log2(1.0 + COST_DISPLAY_MAX),
        0.0,
        1.0
    );
}


vec3 costHeatmap(float x)
{
    x = clamp(x, 0.0, 1.0);

    vec3 blue =
        vec3(0.03, 0.10, 0.55);

    vec3 cyan =
        vec3(0.00, 0.75, 1.00);

    vec3 green =
        vec3(0.05, 0.90, 0.18);

    vec3 yellow =
        vec3(1.00, 0.88, 0.02);

    vec3 red =
        vec3(0.95, 0.03, 0.01);

    if (x < 0.25)
        return mix(
            blue,
            cyan,
            x / 0.25
        );

    if (x < 0.50)
        return mix(
            cyan,
            green,
            (x - 0.25) / 0.25
        );

    if (x < 0.75)
        return mix(
            green,
            yellow,
            (x - 0.50) / 0.25
        );

    return mix(
        yellow,
        red,
        (x - 0.75) / 0.25
    );
}


vec3 comparisonColor(
    bool hitExact,
    bool hitUniform,
    float tExact,
    float tUniform)
{
    if (hitExact && !hitUniform)
    {
        // False negative: uniform missed a real hit.
        return vec3(1.0, 0.0, 1.0);
    }

    if (!hitExact && hitUniform)
    {
        // False positive or implementation error.
        return vec3(0.0, 1.0, 1.0);
    }

    if (!hitExact && !hitUniform)
    {
        return vec3(0.02, 0.03, 0.09);
    }

    float error =
        abs(tUniform - tExact);

    float x =
        clamp(
            log2(1.0 + 40.0 * error) /
            log2(
                1.0 +
                40.0 *
                ERROR_DISPLAY_MAX
            ),
            0.0,
            1.0
        );

    vec3 accurate =
        vec3(0.04, 0.75, 0.12);

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


// ============================================================
// Main
// ============================================================

void mainImage(
    out vec4 fragColor,
    in vec2 fragCoord)
{
    vec2 res =
        max(iResolution.xy, vec2(1.0));

    bool leftHalf =
        fragCoord.x < 0.5 * res.x;

    bool topHalf =
        fragCoord.y >= 0.5 * res.y;

    float localX =
        leftHalf
        ? fragCoord.x / (0.5 * res.x)
        : (
            fragCoord.x -
            0.5 * res.x
        ) / (0.5 * res.x);

    float localY =
        topHalf
        ? (
            fragCoord.y -
            0.5 * res.y
        ) / (0.5 * res.y)
        : fragCoord.y / (0.5 * res.y);

    vec2 panelUV =
        vec2(localX, localY);

    float panelAspect =
        (0.5 * res.x) /
        (0.5 * res.y);

    vec3 ro;
    vec3 rd;

    makeCameraRay(
        panelUV,
        panelAspect,
        ro,
        rd
    );

    float tExact;
    float tUniform;

    int visitedCells;
    int quadraticTests;
    int rejectedCells;
    int uniformSamples;

    bool hitExact =
        traceExactDDA(
            ro,
            rd,
            tExact,
            visitedCells,
            quadraticTests,
            rejectedCells
        );

    bool hitUniform =
        traceUniformMarch(
            ro,
            rd,
            tUniform,
            uniformSamples
        );

    vec3 color;

    // --------------------------------------------------------
    // Top-left: Exact render
    // --------------------------------------------------------
    if (topHalf && leftHalf)
    {
        if (hitExact)
        {
            color =
                shadeTerrain(
                    ro,
                    rd,
                    tExact
                );
        }
        else
        {
            color = skyColor(rd);
        }
    }

    // --------------------------------------------------------
    // Top-right: Uniform render
    // --------------------------------------------------------
    else if (topHalf && !leftHalf)
    {
        if (hitUniform)
        {
            color =
                shadeTerrain(
                    ro,
                    rd,
                    tUniform
                );
        }
        else
        {
            color = skyColor(rd);
        }
    }

    // --------------------------------------------------------
    // Bottom-left:
    //
    // Left half of this quadrant:
    //   visited DDA cells
    //
    // Right half:
    //   actual quadratic tests
    // --------------------------------------------------------
    else if (!topHalf && leftHalf)
    {
        bool showVisited =
            localX < 0.5;

        int shownCount =
            showVisited
            ? visitedCells
            : quadraticTests;

        color =
            costHeatmap(
                normalizeCost(shownCount)
            );

        if (!hitExact)
        {
            color =
                mix(
                    color,
                    vec3(0.35, 0.02, 0.50),
                    0.50
                );
        }

        // Thin divider between visited and quadratic views.
        if (
            abs(localX - 0.5) <
            2.0 / (0.5 * res.x)
        )
        {
            color = vec3(0.95);
        }
    }

    // --------------------------------------------------------
    // Bottom-right: comparison/error classification
    // --------------------------------------------------------
    else
    {
        color =
            comparisonColor(
                hitExact,
                hitUniform,
                tExact,
                tUniform
            );
    }

    // Main quadrant dividers.
    if (
        abs(
            fragCoord.x -
            0.5 * res.x
        ) < 1.5
    )
    {
        color = vec3(0.96);
    }

    if (
        abs(
            fragCoord.y -
            0.5 * res.y
        ) < 1.5
    )
    {
        color = vec3(0.96);
    }

    color =
        pow(
            max(color, vec3(0.0)),
            vec3(1.0 / 2.2)
        );

    fragColor =
        vec4(color, 1.0);
}