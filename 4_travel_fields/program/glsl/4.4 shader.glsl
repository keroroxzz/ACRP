// ============================================================
// Smooth Cubic-Fade Value Noise Heightfield
//
// LEFT TOP:
//   DDA + height-range rejection
//   + sextic Bernstein first-root subdivision
//
// RIGHT TOP:
//   Uniform fixed-step ray marching
//
// LEFT BOTTOM, three vertical strips:
//   1. DDA visited cells
//   2. Bernstein subdivision nodes
//   3. Root refinement iterations
//
// RIGHT BOTTOM:
//   Hit classification and Travel error
//
// Shadertoy: paste into Image shader.
//
// Mouse:
//   Hold and drag to rotate camera.
// ============================================================

#define PI 3.14159265359

const float INF = 1e30;
const float EPS = 1e-7;
const float HIT_EPS = 2e-5;

// Scene/traversal parameters.
const float MAX_DISTANCE = 60.0;
const int MAX_DDA_CELLS = 140;

// Uniform baseline.
const float UNIFORM_STEP = 0.125;
const int MAX_UNIFORM_STEPS = 480;
const int UNIFORM_BISECTION_STEPS = 10;

// Bernstein solver.
const int POLY_DEGREE = 6;
const int POLY_SIZE = 7;

const int MAX_BERNSTEIN_DEPTH = 10;
const int MAX_BERNSTEIN_NODES = 96;
const int MAX_BERNSTEIN_STACK = 16;

const int ROOT_BISECTION_STEPS = 14;
const int ROOT_NEWTON_STEPS = 10;

// A leaf interval has normalized width <= 1 / 2^10.
const float ROOT_VALUE_EPS = 2e-4;
const float ROOT_TANGENT_EPS = 8e-4;

// Shared heatmap scale.
const float COST_DISPLAY_MAX = 480.0;
const float ERROR_DISPLAY_MAX = 0.25;


// ============================================================
// Hash / lattice
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
    return -0.55 + 2.7 * hash21(p);
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


// ============================================================
// Cubic fade value noise
// ============================================================

float fadeCubic(float x)
{
    return x * x * (3.0 - 2.0 * x);
}


float fadeCubicDerivative(float x)
{
    return 6.0 * x * (1.0 - x);
}


float terrainHeight(vec2 xz)
{
    ivec2 cell = ivec2(floor(xz));
    vec2 local = xz - vec2(cell);

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

    float fx = fadeCubic(local.x);
    float fz = fadeCubic(local.y);

    float hx0 = mix(h00, h10, fx);
    float hx1 = mix(h01, h11, fx);

    return mix(hx0, hx1, fz);
}


vec3 terrainNormal(vec2 xz)
{
    ivec2 cell = ivec2(floor(xz));
    vec2 local = xz - vec2(cell);

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

    float a = h00;
    float b = h10 - h00;
    float c = h01 - h00;
    float d = h11 - h10 - h01 + h00;

    float fx = fadeCubic(local.x);
    float fz = fadeCubic(local.y);

    float dfx = fadeCubicDerivative(local.x);
    float dfz = fadeCubicDerivative(local.y);

    float dhdx =
        (b + d * fz) * dfx;

    float dhdz =
        (c + d * fx) * dfz;

    return normalize(
        vec3(-dhdx, 1.0, -dhdz)
    );
}


float terrainClearance(vec3 p)
{
    return p.y - terrainHeight(p.xz);
}


// ============================================================
// Cell-level conservative rejection
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
    // Cubic-fade value noise remains a convex combination
    // of the four corner heights.
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
// Polynomial utilities
// ============================================================

// fade(a + b*s) = c0 + c1*s + c2*s² + c3*s³
void buildFadePowerCoefficients(
    float a,
    float b,
    out float c[4])
{
    c[0] =
        3.0 * a * a -
        2.0 * a * a * a;

    c[1] =
        6.0 * a * b *
        (1.0 - a);

    c[2] =
        3.0 * b * b *
        (1.0 - 2.0 * a);

    c[3] =
        -2.0 * b * b * b;
}


// Build normalized power polynomial:
//
// P(r) = sum power[k] r^k, r in [0,1]
//
// where physical local distance s = intervalLength * r.
void buildClearancePowerPolynomial(
    vec3 entryPoint,
    vec3 rd,
    float intervalLength,
    ivec2 cell,
    float h00,
    float h10,
    float h01,
    float h11,
    out float power[7])
{
    float xi =
        entryPoint.x - float(cell.x);

    float eta =
        entryPoint.z - float(cell.y);

    float fx[4];
    float fz[4];

    buildFadePowerCoefficients(
        xi,
        rd.x,
        fx
    );

    buildFadePowerCoefficients(
        eta,
        rd.z,
        fz
    );

    float product[7];

    for (int k = 0; k < 7; ++k)
        product[k] = 0.0;

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            product[i + j] +=
                fx[i] * fz[j];
        }
    }

    float a = h00;
    float b = h10 - h00;
    float c = h01 - h00;

    float d =
        h11 - h10 - h01 + h00;

    float terrainPower[7];

    for (int k = 0; k < 7; ++k)
    {
        terrainPower[k] =
            d * product[k];

        if (k < 4)
        {
            terrainPower[k] +=
                b * fx[k] +
                c * fz[k];
        }
    }

    terrainPower[0] += a;

    float localPower[7];

    localPower[0] =
        entryPoint.y -
        terrainPower[0];

    localPower[1] =
        rd.y -
        terrainPower[1];

    for (int k = 2; k < 7; ++k)
    {
        localPower[k] =
            -terrainPower[k];
    }

    // Convert s-polynomial to normalized r-polynomial:
    //
    // s = intervalLength * r
    //
    float scale = 1.0;

    for (int k = 0; k < 7; ++k)
    {
        power[k] =
            localPower[k] * scale;

        scale *= intervalLength;
    }
}


// Convert degree-6 power basis to degree-6 Bernstein basis.
//
// beta_k = sum_{j=0}^k
//          power_j * C(k,j) / C(6,j)
void powerToBernstein6(
    float p[7],
    out float b[7])
{
    b[0] =
        p[0];

    b[1] =
        p[0] +
        p[1] / 6.0;

    b[2] =
        p[0] +
        p[1] / 3.0 +
        p[2] / 15.0;

    b[3] =
        p[0] +
        p[1] / 2.0 +
        p[2] / 5.0 +
        p[3] / 20.0;

    b[4] =
        p[0] +
        2.0 * p[1] / 3.0 +
        2.0 * p[2] / 5.0 +
        p[3] / 5.0 +
        p[4] / 15.0;

    b[5] =
        p[0] +
        5.0 * p[1] / 6.0 +
        2.0 * p[2] / 3.0 +
        p[3] / 2.0 +
        p[4] / 3.0 +
        p[5] / 6.0;

    b[6] =
        p[0] +
        p[1] +
        p[2] +
        p[3] +
        p[4] +
        p[5] +
        p[6];
}


float evalPower6(
    float p[7],
    float x)
{
    float value = p[6];

    for (int k = 5; k >= 0; --k)
    {
        value =
            value * x + p[k];
    }

    return value;
}


float evalPowerDerivative6(
    float p[7],
    float x)
{
    float value =
        6.0 * p[6];

    for (int k = 5; k >= 1; --k)
    {
        value =
            value * x +
            float(k) * p[k];
    }

    return value;
}


// ============================================================
// Bernstein subdivision
// ============================================================

void subdivideBernsteinHalf6(
    float inputCoeff[7],
    out float leftCoeff[7],
    out float rightCoeff[7])
{
    float temp[7];

    for (int i = 0; i < 7; ++i)
        temp[i] = inputCoeff[i];

    leftCoeff[0] =
        temp[0];

    rightCoeff[6] =
        temp[6];

    for (int level = 1; level <= 6; ++level)
    {
        int remaining =
            7 - level;

        for (int i = 0; i < 6; ++i)
        {
            if (i < remaining)
            {
                temp[i] =
                    0.5 *
                    (
                        temp[i] +
                        temp[i + 1]
                    );
            }
        }

        leftCoeff[level] =
            temp[0];

        rightCoeff[6 - level] =
            temp[6 - level];
    }
}


bool bernsteinStrictlyExcludesZero(
    float coeff[7])
{
    float minCoeff = coeff[0];
    float maxCoeff = coeff[0];

    for (int i = 1; i < 7; ++i)
    {
        minCoeff =
            min(minCoeff, coeff[i]);

        maxCoeff =
            max(maxCoeff, coeff[i]);
    }

    if (minCoeff > ROOT_VALUE_EPS)
        return true;

    if (maxCoeff < -ROOT_VALUE_EPS)
        return true;

    return false;
}


// ============================================================
// Root refinement
// ============================================================

bool refineRootCandidate(
    float power[7],
    float a,
    float b,
    out float root,
    out int iterations)
{
    iterations = 0;

    float fa = evalPower6(power, a);
    float fb = evalPower6(power, b);

    if (abs(fa) < ROOT_VALUE_EPS)
    {
        root = a;
        return true;
    }

    if (abs(fb) < ROOT_VALUE_EPS)
    {
        root = b;
        return true;
    }

    // Regular root: robust sign-change bisection.
    if (fa * fb < 0.0)
    {
        float lo = a;
        float hi = b;
        float flo = fa;

        for (
            int i = 0;
            i < ROOT_BISECTION_STEPS;
            ++i
        )
        {
            float mid =
                0.5 * (lo + hi);

            float fm =
                evalPower6(power, mid);

            iterations = i + 1;

            if (abs(fm) < ROOT_VALUE_EPS)
            {
                root = mid;
                return true;
            }

            if (flo * fm <= 0.0)
            {
                hi = mid;
            }
            else
            {
                lo = mid;
                flo = fm;
            }
        }

        root = 0.5 * (lo + hi);
        return true;
    }

    // Tangent / even-multiplicity fallback.
    //
    // Sample the narrow candidate interval and start Newton
    // from the lowest-|P| sample.
    float bestX = a;
    float bestValue = abs(fa);

    for (int i = 1; i <= 8; ++i)
    {
        float x =
            mix(
                a,
                b,
                float(i) / 8.0
            );

        float value =
            abs(evalPower6(power, x));

        if (value < bestValue)
        {
            bestValue = value;
            bestX = x;
        }
    }

    float x = bestX;

    for (
        int i = 0;
        i < ROOT_NEWTON_STEPS;
        ++i
    )
    {
        float f =
            evalPower6(power, x);

        float df =
            evalPowerDerivative6(power, x);

        iterations = i + 1;

        if (abs(f) < ROOT_TANGENT_EPS)
        {
            root = x;
            return true;
        }

        if (abs(df) < 1e-7)
            break;

        float nextX =
            x - f / df;

        // Safeguard Newton to the candidate interval.
        nextX =
            clamp(nextX, a, b);

        if (abs(nextX - x) < 1e-7)
            break;

        x = nextX;
    }

    float finalValue =
        abs(evalPower6(power, x));

    if (
        min(bestValue, finalValue) <
        ROOT_TANGENT_EPS
    )
    {
        root =
            finalValue < bestValue
            ? x
            : bestX;

        return true;
    }

    return false;
}


// ============================================================
// Bernstein first-root search
// ============================================================

bool findFirstBernsteinRoot(
    float power[7],
    float rootCoeff[7],
    out float rootR,
    out int nodeCount,
    out int candidateCount,
    out int refinementIterations)
{
    nodeCount = 0;
    candidateCount = 0;
    refinementIterations = 0;

    float currentCoeff[7];

    for (int i = 0; i < 7; ++i)
        currentCoeff[i] = rootCoeff[i];

    float currentA = 0.0;
    float currentB = 1.0;
    int currentDepth = 0;

    // DFS stack stores right children.
    float stackCoeff[
        MAX_BERNSTEIN_STACK * 7
    ];

    float stackA[MAX_BERNSTEIN_STACK];
    float stackB[MAX_BERNSTEIN_STACK];
    int stackDepth[MAX_BERNSTEIN_STACK];

    int stackSize = 0;

    for (
        int node = 0;
        node < MAX_BERNSTEIN_NODES;
        ++node
    )
    {
        nodeCount = node + 1;

        bool excluded =
            bernsteinStrictlyExcludesZero(
                currentCoeff
            );

        if (!excluded)
        {
            bool leaf =
                currentDepth >=
                MAX_BERNSTEIN_DEPTH;

            if (leaf)
            {
                candidateCount += 1;

                float candidateRoot;
                int refineCount;

                bool valid =
                    refineRootCandidate(
                        power,
                        currentA,
                        currentB,
                        candidateRoot,
                        refineCount
                    );

                refinementIterations +=
                    refineCount;

                if (valid)
                {
                    rootR = candidateRoot;
                    return true;
                }
            }
            else
            {
                float leftCoeff[7];
                float rightCoeff[7];

                subdivideBernsteinHalf6(
                    currentCoeff,
                    leftCoeff,
                    rightCoeff
                );

                float midpoint =
                    0.5 *
                    (
                        currentA +
                        currentB
                    );

                // Push right child.
                if (
                    stackSize <
                    MAX_BERNSTEIN_STACK
                )
                {
                    for (int k = 0; k < 7; ++k)
                    {
                        stackCoeff[
                            stackSize * 7 + k
                        ] = rightCoeff[k];
                    }

                    stackA[stackSize] =
                        midpoint;

                    stackB[stackSize] =
                        currentB;

                    stackDepth[stackSize] =
                        currentDepth + 1;

                    stackSize += 1;
                }

                // Continue immediately with left child.
                for (int k = 0; k < 7; ++k)
                    currentCoeff[k] = leftCoeff[k];

                currentB = midpoint;
                currentDepth += 1;

                continue;
            }
        }

        // Current node rejected or false candidate:
        // pop nearest right sibling.
        if (stackSize <= 0)
            break;

        stackSize -= 1;

        for (int k = 0; k < 7; ++k)
        {
            currentCoeff[k] =
                stackCoeff[
                    stackSize * 7 + k
                ];
        }

        currentA =
            stackA[stackSize];

        currentB =
            stackB[stackSize];

        currentDepth =
            stackDepth[stackSize];
    }

    return false;
}


// ============================================================
// One smooth cell intersection
// ============================================================

bool intersectSmoothCell(
    vec3 ro,
    vec3 rd,
    ivec2 cell,
    float tEnter,
    float tExit,
    float h00,
    float h10,
    float h01,
    float h11,
    out float tHit,
    out int nodeCount,
    out int candidateCount,
    out int refinementIterations)
{
    nodeCount = 0;
    candidateCount = 0;
    refinementIterations = 0;

    float intervalLength =
        tExit - tEnter;

    if (intervalLength < HIT_EPS)
        return false;

    vec3 entryPoint =
        ro + rd * tEnter;

    float power[7];

    buildClearancePowerPolynomial(
        entryPoint,
        rd,
        intervalLength,
        cell,
        h00,
        h10,
        h01,
        h11,
        power
    );

    float bernstein[7];

    powerToBernstein6(
        power,
        bernstein
    );

    float rootR;

    bool found =
        findFirstBernsteinRoot(
            power,
            bernstein,
            rootR,
            nodeCount,
            candidateCount,
            refinementIterations
        );

    if (!found)
        return false;

    tHit =
        tEnter +
        intervalLength *
        clamp(rootR, 0.0, 1.0);

    return true;
}


// ============================================================
// DDA + smooth-cell Bernstein traversal
// ============================================================

bool traceBernsteinDDA(
    vec3 ro,
    vec3 rd,
    out float tHit,
    out int visitedCells,
    out int rangeRejectedCells,
    out int totalBernsteinNodes,
    out int totalCandidates,
    out int totalRefinementIterations)
{
    tHit = INF;

    visitedCells = 0;
    rangeRejectedCells = 0;
    totalBernsteinNodes = 0;
    totalCandidates = 0;
    totalRefinementIterations = 0;

    vec2 horizontalDirection =
        rd.xz;

    vec2 startXZ =
        ro.xz +
        horizontalDirection * 1e-5;

    ivec2 cell =
        ivec2(floor(startXZ));

    int stepX = 0;
    int stepZ = 0;

    if (horizontalDirection.x > EPS)
        stepX = 1;
    else if (horizontalDirection.x < -EPS)
        stepX = -1;

    if (horizontalDirection.y > EPS)
        stepZ = 1;
    else if (horizontalDirection.y < -EPS)
        stepZ = -1;

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
        ? (
            nextBoundaryX - ro.x
        ) / horizontalDirection.x
        : INF;

    float tMaxZ =
        stepZ != 0
        ? (
            nextBoundaryZ - ro.z
        ) / horizontalDirection.y
        : INF;

    tMaxX = max(tMaxX, 0.0);
    tMaxZ = max(tMaxZ, 0.0);

    float tEnter = 0.0;

    for (
        int cellIndex = 0;
        cellIndex < MAX_DDA_CELLS;
        ++cellIndex
    )
    {
        visitedCells =
            cellIndex + 1;

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
            float candidateHit;
            int nodeCount;
            int candidateCount;
            int refineCount;

            bool hitCell =
                intersectSmoothCell(
                    ro,
                    rd,
                    cell,
                    tEnter,
                    tExit,
                    h00,
                    h10,
                    h01,
                    h11,
                    candidateHit,
                    nodeCount,
                    candidateCount,
                    refineCount
                );

            totalBernsteinNodes +=
                nodeCount;

            totalCandidates +=
                candidateCount;

            totalRefinementIterations +=
                refineCount;

            if (
                hitCell &&
                candidateHit >= 0.0 &&
                candidateHit <= MAX_DISTANCE
            )
            {
                tHit = candidateHit;
                return true;
            }
        }
        else
        {
            rangeRejectedCells += 1;
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
// Uniform baseline
// ============================================================

float refineUniformCrossing(
    vec3 ro,
    vec3 rd,
    float ta,
    float tb)
{
    float ca =
        terrainClearance(
            ro + rd * ta
        );

    for (
        int i = 0;
        i < UNIFORM_BISECTION_STEPS;
        ++i
    )
    {
        float tm =
            0.5 * (ta + tb);

        float cm =
            terrainClearance(
                ro + rd * tm
            );

        bool sameSide =
            (
                ca >= 0.0 &&
                cm >= 0.0
            ) ||
            (
                ca <= 0.0 &&
                cm <= 0.0
            );

        if (sameSide)
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


bool traceUniform(
    vec3 ro,
    vec3 rd,
    out float tHit,
    out int sampleCount)
{
    tHit = INF;
    sampleCount = 0;

    float previousT = 0.0;

    float previousClearance =
        terrainClearance(ro);

    if (
        abs(previousClearance) <
        HIT_EPS
    )
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

        sampleCount = i + 1;

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
                refineUniformCrossing(
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
            max(
                iResolution.xy,
                vec2(1.0)
            );

        yaw =
            (mouse.x - 0.5) *
            2.0 * PI;

        pitch =
            mix(
                0.10,
                0.72,
                clamp(mouse.y, 0.0, 1.0)
            );
    }
    else
    {
        yaw =
            0.15 * iTime;

        pitch =
            0.30;
    }

    vec3 target =
        vec3(0.0, 0.55, 0.0);

    float radius = 10.0;

    ro =
        target +
        vec3(
            radius *
            cos(pitch) *
            sin(yaw),

            radius *
            sin(pitch) +
            3.1,

            radius *
            cos(pitch) *
            cos(yaw)
        );

    mat3 camera =
        cameraBasis(ro, target);

    vec2 screen =
        vec2(
            (uv.x - 0.5) * aspect,
            uv.y - 0.5
        );

    rd =
        normalize(
            camera *
            vec3(screen, 1.35)
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
        vec3(0.73, 0.83, 0.97);

    vec3 zenith =
        vec3(0.13, 0.29, 0.56);

    vec3 result =
        mix(horizon, zenith, h);

    vec3 sunDirection =
        normalize(
            vec3(0.55, 0.75, 0.30)
        );

    float sun =
        pow(
            max(
                dot(rd, sunDirection),
                0.0
            ),
            500.0
        );

    result +=
        vec3(1.0, 0.86, 0.56) *
        sun * 3.0;

    return result;
}


vec3 terrainMaterial(
    vec3 p,
    vec3 normal)
{
    float slope =
        1.0 -
        clamp(
            normal.y,
            0.0,
            1.0
        );

    vec3 grass =
        vec3(0.15, 0.35, 0.11);

    vec3 rock =
        vec3(0.34, 0.31, 0.27);

    vec3 high =
        vec3(0.48, 0.44, 0.33);

    vec3 base =
        mix(
            grass,
            rock,
            smoothstep(
                0.16,
                0.68,
                slope
            )
        );

    base =
        mix(
            base,
            high,
            smoothstep(
                1.0,
                2.1,
                p.y
            )
        );

    return base;
}


vec3 shadeTerrain(
    vec3 ro,
    vec3 rd,
    float tHit)
{
    vec3 p =
        ro + rd * tHit;

    vec3 normal =
        terrainNormal(p.xz);

    vec3 lightDirection =
        normalize(
            vec3(0.55, 0.75, 0.30)
        );

    float diffuse =
        max(
            dot(normal, lightDirection),
            0.0
        );

    float hemisphere =
        0.35 +
        0.65 *
        max(normal.y, 0.0);

    vec3 albedo =
        terrainMaterial(p, normal);

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
        exp(
            -0.018 *
            tHit *
            tHit
        );

    return mix(
        color,
        skyColor(rd),
        clamp(fog, 0.0, 1.0)
    );
}


// ============================================================
// Heatmaps
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


vec3 costHeatmap(float x)
{
    x = clamp(x, 0.0, 1.0);

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


vec3 comparisonColor(
    bool hitBernstein,
    bool hitUniform,
    float tBernstein,
    float tUniform)
{
    if (
        hitBernstein &&
        !hitUniform
    )
    {
        // Uniform false negative.
        return vec3(1.0, 0.0, 1.0);
    }

    if (
        !hitBernstein &&
        hitUniform
    )
    {
        // Potential Bernstein miss or numerical bug.
        return vec3(0.0, 1.0, 1.0);
    }

    if (
        !hitBernstein &&
        !hitUniform
    )
    {
        return vec3(0.02, 0.03, 0.09);
    }

    float error =
        abs(
            tUniform -
            tBernstein
        );

    float x =
        clamp(
            log2(
                1.0 +
                40.0 * error
            ) /
            log2(
                1.0 +
                40.0 *
                ERROR_DISPLAY_MAX
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

    vec2 panelUV =
        vec2(localX, localY);

    float panelAspect =
        (0.5 * resolution.x) /
        (0.5 * resolution.y);

    vec3 ro;
    vec3 rd;

    makeCameraRay(
        panelUV,
        panelAspect,
        ro,
        rd
    );

    float tBernstein;
    int visitedCells;
    int rangeRejectedCells;
    int bernsteinNodes;
    int candidateCount;
    int refinementIterations;

    bool hitBernstein =
        traceBernsteinDDA(
            ro,
            rd,
            tBernstein,
            visitedCells,
            rangeRejectedCells,
            bernsteinNodes,
            candidateCount,
            refinementIterations
        );

    float tUniform;
    int uniformSamples;

    bool hitUniform =
        traceUniform(
            ro,
            rd,
            tUniform,
            uniformSamples
        );

    vec3 color;

    // Top-left: Bernstein result.
    if (topHalf && leftHalf)
    {
        color =
            hitBernstein
            ? shadeTerrain(
                ro,
                rd,
                tBernstein
              )
            : skyColor(rd);
    }

    // Top-right: uniform result.
    else if (topHalf && !leftHalf)
    {
        color =
            hitUniform
            ? shadeTerrain(
                ro,
                rd,
                tUniform
              )
            : skyColor(rd);
    }

    // Bottom-left: three cost strips.
    else if (!topHalf && leftHalf)
    {
        int shownCost;

        if (localX < 1.0 / 3.0)
        {
            shownCost =
                visitedCells;
        }
        else if (localX < 2.0 / 3.0)
        {
            shownCost =
                bernsteinNodes;
        }
        else
        {
            shownCost =
                refinementIterations;
        }

        color =
            costHeatmap(
                normalizeCost(shownCost)
            );

        if (!hitBernstein)
        {
            color =
                mix(
                    color,
                    vec3(0.35, 0.02, 0.50),
                    0.42
                );
        }

        float dividerWidth =
            2.0 /
            (0.5 * resolution.x);

        if (
            abs(
                localX -
                1.0 / 3.0
            ) < dividerWidth ||
            abs(
                localX -
                2.0 / 3.0
            ) < dividerWidth
        )
        {
            color = vec3(0.96);
        }
    }

    // Bottom-right: comparison.
    else
    {
        color =
            comparisonColor(
                hitBernstein,
                hitUniform,
                tBernstein,
                tUniform
            );
    }

    // Main dividers.
    if (
        abs(
            fragCoord.x -
            0.5 * resolution.x
        ) < 1.5
    )
    {
        color = vec3(0.96);
    }

    if (
        abs(
            fragCoord.y -
            0.5 * resolution.y
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