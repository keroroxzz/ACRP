// ============================================================
// Exact Bilinear Heightfield Travel vs Uniform Ray Marching
// Shadertoy single-image shader
//
// LEFT:
//   2D DDA traversal + exact quadratic intersection per cell
//
// RIGHT:
//   Uniform fixed-step ray marching + bisection refinement
//
// TOP:
//   Rendered result
//
// BOTTOM:
//   Per-pixel iteration heatmap
//
// Mouse:
//   Hold and drag to rotate camera
// ============================================================

#define PI 3.14159265359

const float INF              = 1e30;
const float EPS              = 1e-6;
const float HIT_EPS          = 1e-5;

const float MAX_DISTANCE     = 80.0;

// Exact method:
// Number of horizontally traversed bilinear cells.
const int MAX_DDA_CELLS      = 180;

// Uniform method:
// Number of fixed samples.
const int MAX_UNIFORM_STEPS  = 640;
const int BISECTION_STEPS    = 8;

const float UNIFORM_STEP     = 0.10;

// Bottom portion used for the step-count heatmap.
const float HEATMAP_HEIGHT   = 0.22;


// ============================================================
// Hash and procedural height field
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

    // Heights approximately in [-0.5, 2.0].
    return -0.5 + 2.5 * n;
}


// Raw bilinear interpolation.
//
// This deliberately does NOT use smoothstep interpolation.
// Each cell is therefore exactly bilinear and each ray-cell
// intersection is a polynomial of degree <= 2.
float terrainHeight(vec2 xz)
{
    ivec2 cell = ivec2(floor(xz));
    vec2 f = xz - vec2(cell);

    float h00 = latticeHeight(cell + ivec2(0, 0));
    float h10 = latticeHeight(cell + ivec2(1, 0));
    float h01 = latticeHeight(cell + ivec2(0, 1));
    float h11 = latticeHeight(cell + ivec2(1, 1));

    float hx0 = mix(h00, h10, f.x);
    float hx1 = mix(h01, h11, f.x);

    return mix(hx0, hx1, f.y);
}


// Returns the exact analytic normal of the bilinear cell at xz.
vec3 terrainNormal(vec2 xz)
{
    ivec2 cell = ivec2(floor(xz));
    vec2 f = xz - vec2(cell);

    float h00 = latticeHeight(cell + ivec2(0, 0));
    float h10 = latticeHeight(cell + ivec2(1, 0));
    float h01 = latticeHeight(cell + ivec2(0, 1));
    float h11 = latticeHeight(cell + ivec2(1, 1));

    // h(x,z) = a + b*x + c*z + d*x*z
    float b = h10 - h00;
    float c = h01 - h00;
    float d = h11 - h10 - h01 + h00;

    float dhdx = b + d * f.y;
    float dhdz = c + d * f.x;

    // Surface F(x,y,z) = y - h(x,z) = 0.
    return normalize(vec3(-dhdx, 1.0, -dhdz));
}


// Signed vertical clearance.
// Positive means the point is above the height field.
float terrainClearance(vec3 p)
{
    return p.y - terrainHeight(p.xz);
}


// ============================================================
// Numerically stable root selection
// ============================================================

// Returns the first root in [tMin, tMax].
// Returns INF if no admissible root exists.
float firstRootInInterval(
    float A,
    float B,
    float C,
    float tMin,
    float tMax)
{
    // Linear or constant polynomial.
    if (abs(A) < EPS)
    {
        if (abs(B) < EPS)
        {
            // Entire interval lies on the surface.
            if (abs(C) < HIT_EPS)
                return tMin;

            return INF;
        }

        float t = -C / B;

        if (t >= tMin - HIT_EPS &&
            t <= tMax + HIT_EPS)
        {
            return clamp(t, tMin, tMax);
        }

        return INF;
    }

    float discriminant = B * B - 4.0 * A * C;

    if (discriminant < -EPS)
        return INF;

    discriminant = max(discriminant, 0.0);

    float sqrtD = sqrt(discriminant);

    // Stable quadratic formula:
    //
    // q = -1/2 (B + sign(B) sqrt(D))
    //
    float q = -0.5 *
        (B + (B >= 0.0 ? sqrtD : -sqrtD));

    float t0;
    float t1;

    if (abs(q) < EPS)
    {
        // Repeated root.
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
        float temp = t0;
        t0 = t1;
        t1 = temp;
    }

    if (t0 >= tMin - HIT_EPS &&
        t0 <= tMax + HIT_EPS)
    {
        return clamp(t0, tMin, tMax);
    }

    if (t1 >= tMin - HIT_EPS &&
        t1 <= tMax + HIT_EPS)
    {
        return clamp(t1, tMin, tMax);
    }

    return INF;
}


// ============================================================
// Exact intersection inside one bilinear cell
// ============================================================

float intersectBilinearCell(
    vec3 rayOrigin,
    vec3 rayDirection,
    ivec2 cell,
    float tEnter,
    float tExit)
{
    float intervalLength = tExit - tEnter;

    if (intervalLength < -HIT_EPS)
        return INF;

    vec3 p = rayOrigin + rayDirection * tEnter;

    // Cell-local coordinates.
    float xi  = p.x - float(cell.x);
    float eta = p.z - float(cell.y);

    float h00 = latticeHeight(cell + ivec2(0, 0));
    float h10 = latticeHeight(cell + ivec2(1, 0));
    float h01 = latticeHeight(cell + ivec2(0, 1));
    float h11 = latticeHeight(cell + ivec2(1, 1));

    // Bilinear coefficients:
    //
    // h(xi,eta) = a + b*xi + c*eta + d*xi*eta
    //
    float a = h00;
    float b = h10 - h00;
    float c = h01 - h00;
    float d = h11 - h10 - h01 + h00;

    float u = rayDirection.x;
    float v = rayDirection.z;
    float w = rayDirection.y;

    float heightAtEntry =
        a +
        b * xi +
        c * eta +
        d * xi * eta;

    // Let local ray parameter be s = t - tEnter.
    //
    // rayY(s) - terrainHeight(s)
    //     = A*s^2 + B*s + C
    //
    float A = -d * u * v;

    float B =
        w -
        (
            b * u +
            c * v +
            d * (xi * v + eta * u)
        );

    float C = p.y - heightAtEntry;

    float localHit = firstRootInInterval(
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
// Exact DDA traversal
// ============================================================

bool traceExactDDA(
    vec3 rayOrigin,
    vec3 rayDirection,
    out float tHit,
    out int visitedCells)
{
    tHit = INF;
    visitedCells = 0;

    vec2 horizontalDirection = rayDirection.xz;

    // Slightly bias the cell classification in the ray direction.
    // This avoids ambiguity when the origin lies exactly on a grid line.
    vec2 startXZ =
        rayOrigin.xz +
        horizontalDirection * 1e-5;

    ivec2 cell = ivec2(floor(startXZ));

    int stepX = 0;
    int stepZ = 0;

    if (horizontalDirection.x > EPS) stepX = 1;
    if (horizontalDirection.x < -EPS) stepX = -1;

    if (horizontalDirection.y > EPS) stepZ = 1;
    if (horizontalDirection.y < -EPS) stepZ = -1;

    float tDeltaX =
        abs(horizontalDirection.x) > EPS
        ? abs(1.0 / horizontalDirection.x)
        : INF;

    float tDeltaZ =
        abs(horizontalDirection.y) > EPS
        ? abs(1.0 / horizontalDirection.y)
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
        ? (nextBoundaryX - rayOrigin.x) /
          horizontalDirection.x
        : INF;

    float tMaxZ =
        stepZ != 0
        ? (nextBoundaryZ - rayOrigin.z) /
          horizontalDirection.y
        : INF;

    // Numerical protection.
    tMaxX = max(tMaxX, 0.0);
    tMaxZ = max(tMaxZ, 0.0);

    float tEnter = 0.0;

    for (int i = 0; i < MAX_DDA_CELLS; ++i)
    {
        visitedCells = i + 1;

        float tExit = min(
            min(tMaxX, tMaxZ),
            MAX_DISTANCE
        );

        float candidate = intersectBilinearCell(
            rayOrigin,
            rayDirection,
            cell,
            tEnter,
            tExit
        );

        if (candidate < INF * 0.5)
        {
            // Reject a tiny negative or behind-origin result.
            if (candidate >= 0.0 &&
                candidate <= MAX_DISTANCE)
            {
                tHit = candidate;
                return true;
            }
        }

        if (tExit >= MAX_DISTANCE)
            break;

        // Advance through x, z, or a corner crossing.
        bool crossX = tMaxX <= tMaxZ + HIT_EPS;
        bool crossZ = tMaxZ <= tMaxX + HIT_EPS;

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

        // Purely vertical ray:
        // it never exits the current horizontal cell.
        if (stepX == 0 && stepZ == 0)
            break;
    }

    return false;
}


// ============================================================
// Uniform ray marching baseline
// ============================================================

float refineCrossingBisection(
    vec3 rayOrigin,
    vec3 rayDirection,
    float ta,
    float tb)
{
    float ca = terrainClearance(
        rayOrigin + rayDirection * ta
    );

    for (int i = 0; i < BISECTION_STEPS; ++i)
    {
        float tm = 0.5 * (ta + tb);

        float cm = terrainClearance(
            rayOrigin + rayDirection * tm
        );

        if (
            (ca <= 0.0 && cm <= 0.0) ||
            (ca >= 0.0 && cm >= 0.0)
        )
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
    vec3 rayOrigin,
    vec3 rayDirection,
    out float tHit,
    out int sampleCount)
{
    tHit = INF;
    sampleCount = 0;

    float previousT = 0.0;
    float previousClearance =
        terrainClearance(rayOrigin);

    if (abs(previousClearance) < HIT_EPS)
    {
        tHit = 0.0;
        return true;
    }

    for (int i = 0; i < MAX_UNIFORM_STEPS; ++i)
    {
        float currentT =
            float(i + 1) * UNIFORM_STEP;

        sampleCount = i + 1;

        if (currentT > MAX_DISTANCE)
            break;

        vec3 p =
            rayOrigin +
            rayDirection * currentT;

        float currentClearance =
            terrainClearance(p);

        // Direct near-zero sample.
        if (abs(currentClearance) < HIT_EPS)
        {
            tHit = currentT;
            return true;
        }

        // A sign change brackets a crossing.
        if (
            previousClearance *
            currentClearance < 0.0
        )
        {
            tHit = refineCrossingBisection(
                rayOrigin,
                rayDirection,
                previousT,
                currentT
            );

            return true;
        }

        previousT = currentT;
        previousClearance = currentClearance;
    }

    return false;
}


// ============================================================
// Camera
// ============================================================

mat3 cameraBasis(vec3 eye, vec3 target)
{
    vec3 forward = normalize(target - eye);

    vec3 right = normalize(
        cross(forward, vec3(0.0, 1.0, 0.0))
    );

    vec3 up = cross(right, forward);

    return mat3(right, up, forward);
}


void makeCameraRay(
    vec2 panelUV,
    float panelAspect,
    out vec3 rayOrigin,
    out vec3 rayDirection)
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

    vec3 target = vec3(0.0, 0.5, 0.0);

    float radius = 10.0;

    rayOrigin =
        target +
        vec3(
            radius * cos(pitch) * sin(yaw),
            radius * sin(pitch) + 3.0,
            radius * cos(pitch) * cos(yaw)
        );

    mat3 camera =
        cameraBasis(rayOrigin, target);

    vec2 screen =
        vec2(
            (panelUV.x - 0.5) * panelAspect,
            panelUV.y - 0.5
        );

    rayDirection = normalize(
        camera *
        vec3(screen, 1.35)
    );
}


// ============================================================
// Lighting and shading
// ============================================================

vec3 skyColor(vec3 rd)
{
    float h =
        clamp(0.5 + 0.5 * rd.y, 0.0, 1.0);

    vec3 horizon =
        vec3(0.72, 0.82, 0.96);

    vec3 zenith =
        vec3(0.12, 0.28, 0.55);

    vec3 sky = mix(horizon, zenith, h);

    vec3 sunDirection =
        normalize(vec3(0.55, 0.75, 0.30));

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


vec3 terrainMaterial(vec3 p, vec3 normal)
{
    float slope =
        1.0 -
        clamp(normal.y, 0.0, 1.0);

    vec3 grass =
        vec3(0.16, 0.34, 0.11);

    vec3 rock =
        vec3(0.32, 0.30, 0.27);

    vec3 high =
        vec3(0.46, 0.42, 0.31);

    vec3 base =
        mix(grass, rock, smoothstep(0.15, 0.70, slope));

    base =
        mix(
            base,
            high,
            smoothstep(1.0, 2.0, p.y)
        );

    // Subtle lattice visualization.
    vec2 f = abs(fract(p.xz) - 0.5);

    float cellEdge =
        1.0 -
        smoothstep(
            0.46,
            0.495,
            max(f.x, f.y)
        );

    base *= mix(0.72, 1.0, cellEdge);

    return base;
}


vec3 shadeTerrain(
    vec3 rayOrigin,
    vec3 rayDirection,
    float tHit)
{
    vec3 p =
        rayOrigin +
        rayDirection * tHit;

    vec3 n = terrainNormal(p.xz);

    vec3 sunDirection =
        normalize(vec3(0.55, 0.75, 0.30));

    float diffuse =
        max(dot(n, sunDirection), 0.0);

    float hemisphere =
        0.35 + 0.65 *
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

    // Distance fog.
    float fog =
        1.0 -
        exp(-0.018 * tHit * tHit);

    color =
        mix(
            color,
            skyColor(rayDirection),
            clamp(fog, 0.0, 1.0)
        );

    return color;
}


// ============================================================
// Heatmap
// ============================================================

vec3 iterationHeatmap(float x)
{
    x = clamp(x, 0.0, 1.0);

    // Blue -> cyan -> green -> yellow -> red.
    vec3 c0 = vec3(0.05, 0.12, 0.55);
    vec3 c1 = vec3(0.00, 0.80, 0.95);
    vec3 c2 = vec3(0.10, 0.90, 0.20);
    vec3 c3 = vec3(1.00, 0.90, 0.05);
    vec3 c4 = vec3(0.90, 0.05, 0.02);

    if (x < 0.25)
        return mix(c0, c1, x / 0.25);

    if (x < 0.50)
        return mix(c1, c2, (x - 0.25) / 0.25);

    if (x < 0.75)
        return mix(c2, c3, (x - 0.50) / 0.25);

    return mix(c3, c4, (x - 0.75) / 0.25);
}


// ============================================================
// Main
// ============================================================

void mainImage(
    out vec4 fragColor,
    in vec2 fragCoord)
{
    vec2 resolution =
        max(iResolution.xy, vec2(1.0));

    bool exactPanel =
        fragCoord.x < 0.5 * resolution.x;

    bool heatmapRegion =
        fragCoord.y <
        HEATMAP_HEIGHT * resolution.y;

    // Horizontal coordinate local to each half-screen panel.
    float panelX;

    if (exactPanel)
    {
        panelX =
            fragCoord.x /
            (0.5 * resolution.x);
    }
    else
    {
        panelX =
            (
                fragCoord.x -
                0.5 * resolution.x
            ) /
            (0.5 * resolution.x);
    }

    // Both the top render and bottom heatmap show the same
    // camera field, but the heatmap is vertically compressed.
    float panelY;

    if (heatmapRegion)
    {
        panelY =
            fragCoord.y /
            (HEATMAP_HEIGHT * resolution.y);
    }
    else
    {
        panelY =
            (
                fragCoord.y -
                HEATMAP_HEIGHT * resolution.y
            ) /
            (
                (1.0 - HEATMAP_HEIGHT) *
                resolution.y
            );
    }

    vec2 panelUV = vec2(panelX, panelY);

    float topRegionHeight =
        (1.0 - HEATMAP_HEIGHT) *
        resolution.y;

    float panelAspect =
        (0.5 * resolution.x) /
        topRegionHeight;

    vec3 rayOrigin;
    vec3 rayDirection;

    makeCameraRay(
        panelUV,
        panelAspect,
        rayOrigin,
        rayDirection
    );

    float tHit;
    int iterations;
    bool hit;

    if (exactPanel)
    {
        hit = traceExactDDA(
            rayOrigin,
            rayDirection,
            tHit,
            iterations
        );
    }
    else
    {
        hit = traceUniformMarch(
            rayOrigin,
            rayDirection,
            tHit,
            iterations
        );
    }

    vec3 color;

    if (heatmapRegion)
    {
        float normalizedIterations;

        if (exactPanel)
        {
            normalizedIterations =
                float(iterations) /
                float(MAX_DDA_CELLS);
        }
        else
        {
            normalizedIterations =
                float(iterations) /
                float(MAX_UNIFORM_STEPS);
        }

        color = iterationHeatmap(
            normalizedIterations
        );

        // Purple marks rays that were not resolved
        // before the distance/iteration limit.
        if (!hit)
        {
            color =
                mix(
                    color,
                    vec3(0.35, 0.02, 0.50),
                    0.72
                );
        }

        // Slight shading to preserve scene silhouette.
        if (hit)
        {
            vec3 p =
                rayOrigin +
                rayDirection * tHit;

            vec3 n =
                terrainNormal(p.xz);

            float shape =
                0.45 +
                0.55 *
                max(
                    dot(
                        n,
                        normalize(
                            vec3(0.55, 0.75, 0.30)
                        )
                    ),
                    0.0
                );

            color *= shape;
        }
    }
    else
    {
        if (hit)
        {
            color =
                shadeTerrain(
                    rayOrigin,
                    rayDirection,
                    tHit
                );
        }
        else
        {
            color =
                skyColor(rayDirection);
        }
    }

    // Center divider.
    float centerDistance =
        abs(
            fragCoord.x -
            0.5 * resolution.x
        );

    if (centerDistance < 1.5)
    {
        color = vec3(0.95);
    }

    // Divider between render and heatmap.
    float heatBoundary =
        abs(
            fragCoord.y -
            HEATMAP_HEIGHT *
            resolution.y
        );

    if (heatBoundary < 1.5)
    {
        color = vec3(0.95);
    }

    // Mild gamma correction.
    color =
        pow(
            max(color, vec3(0.0)),
            vec3(1.0 / 2.2)
        );

    fragColor = vec4(color, 1.0);
}