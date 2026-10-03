"""Regenerate shader_HSE_PSMS_Next.glsl from shader_HSE_Next.glsl.

    python build_psms.py

shader_HSE_PSMS_Next.glsl = the Next marcher + psms_atmosphere.glsl (Paper 3),
with the cloud lighting rewired onto it.  It is generated rather than
hand-forked so that marcher work keeps flowing into it: run this again after
any change to shader_HSE_Next.glsl.

Every substitution asserts, so if the base shader changes shape this fails
loudly instead of producing a half-patched file.  If an assert fires, fix the
anchor here - do not edit the generated shader, it will be overwritten.
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
src = io.open(os.path.join(ART, "shader_HSE_Next.glsl"), encoding="utf-8").read()
psms = io.open(os.path.join(HERE, "psms_atmosphere.glsl"), encoding="utf-8").read()


def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (src.count(old), old[:80])
    src = src.replace(old, new)


def rep_re(start_anchor, end_anchor, new):
    """Replace [start_anchor .. end_anchor] inclusive. The source has trailing
    whitespace on several lines, so exact-string matching is too brittle."""
    global src
    pat = re.compile(re.escape(start_anchor) + r".*?" + re.escape(end_anchor), re.S)
    assert len(pat.findall(src)) == 1, len(pat.findall(src))
    src = pat.sub(lambda m: new, src, count=1)


# ---------------------------------------------------------------- header
rep("""/*===================================================================================
CloudWorks "Next" - adaptive band-limited ray marching""",
    """/*===================================================================================
CloudWorks "Next" + Paper 3 atmosphere (PSMS)
2026/07/29

shader_HSE_Next.glsl with its hand-rolled two-term Rayleigh/Mie sky replaced by
the Paper 3 pseudo-spherical multiple-scattering model, and the CLOUDS wired
into it rather than lit by hand-keyed constants:

  * sun colour on a cloud   = Paper 3 solar transmittance at the cloud's own
                              altitude and slant path, so cloud light reddens
                              with the sun exactly as the sky does;
  * ambient on a cloud      = the real sky radiance the cloud can see (zenith
                              and horizon probes from the same integral, MS
                              included), mixed by height in the deck;
  * aerial perspective      = PSMSIntegrate() over the eye->cloud segment, which
                              returns both the path transmittance and the
                              airlight scattered into it.  Distant clouds fuse
                              into the sky because the same integral, run
                              further, IS the sky - no distance-fog lerp.

This is a BASELINE: correctness first, performance not considered.  The marcher
itself is untouched, so the NEXT_* ledger still applies.
===================================================================================*/

/*===================================================================================
CloudWorks "Next" - adaptive band-limited ray marching""")

# ------------------------------------------------- swap the atmosphere block
start = src.index("#define ATM_DENSE")
end = src.index("/*===================================================================================\nGTA SA - CloudWorks Beta")
old_block = src[start:end]

keep = """#define ATM_DENSE vGradingColor0.r*7.0f
#define MIST vGradingColor0.g

static float3 sunLightStrength = 685.0*float3(1.0, 0.96, 0.949);
static float moonReflecticity = 0.16;
static float starStrength = 0.75;

static const float earthRadius = 6.421;
static const float groundHeight = 6.371;
static const float3 AtmOrigin = float3(0.0, 0.0, groundHeight);

static float game2atm = 1000000.0;      // GTA metres -> megametres

float3 Game2Atm(float3 gamepos)
{
    return gamepos/game2atm + AtmOrigin;
}

float3 Game2Atm_Alt(float3 gamepos)
{
    return float3(0.0, 0.0, gamepos.z/game2atm) + AtmOrigin;
}

"""
src = src[:start] + keep + psms + "\n" + src[end:]

# --------------------------------------------------- cloud lighting rewrite
CLOUD_LIGHTING = """            float suncross = dot(lightDirection.xyz, cam_dir);

            // Planet-space positions: the atmosphere only cares about ALTITUDE,
            // so the camera's world XY is dropped (Game2Atm_Alt), while the
            // cloud sample keeps its own height along the ray.
            float3 z_pos        = float3(0.0, 0.0, cam_pos.z);
            float3 my_cam_pos   = Game2Atm_Alt(z_pos);
            float3 cloud_pos    = Game2Atm_Alt(z_pos + cam_dir*d.y);
            float  cloud_r      = length(cloud_pos);

            // --- direct sunlight on the cloud -----------------------------
            // Paper 3's slant-path transmittance at the cloud's own altitude.
            // A cloud at 6 km sees a shorter, less reddened path than one at
            // 400 m, and near sunset both go orange on their own.
            float3 sunCol = light*PSMS_LIGHT_SCALE
                          * PSMSSunTrans(cloud_pos, cloud_r, lightDirection);

            // --- ambient: the sky the cloud can actually see ---------------
            // Both probes come out of the same integral as the sky, so they
            // already carry the atmosphere's multiple scattering.
            //
            // The dome average is HORIZON-heavy, and mixing by height-in-deck
            // instead (deck top -> zenith) is wrong twice over: it confuses
            // "which way is the sky brighter" with "how much sky can this
            // sample see", and it leaves thin distant wisps scattering dim
            // zenith light while blocking bright horizon sky, which reads as
            // dark speckle.  Height only controls VISIBILITY of the dome.
            float hf  = saturate((d.y*cam_dir.z + cam_pos.z - a.volumeBox.y)
                                 / max(a.volumeBox.x - a.volumeBox.y, 1.0));
            float3 ambCol = lerp(gAmbHorizon, gAmbZenith, PSMS_AMB_ZENITH_MIX)
                          * lerp(0.55, 1.0, hf);

            float3 C = sunCol * fx.x * MiePhase(suncross) * a.brightness * PSMS_CLOUD_GAIN
                     + ambCol * (1.0 - fx.z) * PSMS_CLOUD_AMBIENT;

            // --- aerial perspective ----------------------------------------
            // The eye->cloud segment of the SAME integral that produces the
            // sky: it returns the path transmittance and the airlight scattered
            // into it.  Far enough away, trans -> 0 and airlight -> the sky
            // radiance for this direction, so the cloud dissolves into the sky
            // by construction instead of by a fog lerp.
            float3 apTrans;
            float3 apAir = PSMSIntegrate(my_cam_pos, cam_dir, d.y/game2atm,
                                         PSMS_AERIAL_STEPS, lightDirection,
                                         light*PSMS_LIGHT_SCALE, apTrans);
            C = C*apTrans + apAir*(1.0 - fx.z);

            //calculate the depth for merging
            depth = depth*fx.z + d.y*(1.0-fx.z);

            return float4(C, fx.z);"""

rep_re("            float suncross = dot(lightDirection.xyz, cam_dir);",
       "            return float4(C, fx.z);",
       CLOUD_LIGHTING)

# CloudColorVarying's hand-keyed palette is superseded by the sky probes
rep_re("void CloudColorVarying(float3 SunDir, inout CloudBaseColor b)",
       "b.CBC += lerp(b.CBC_D, b.CBC_S, isSunSet)*isDay;\n}",
       "// The hand-keyed day/sunset cloud palette is gone: the ambient colour is now\n"
       "// the sky probes, which get their day/sunset behaviour from the physics.\n"
       "// Kept as a no-op hook in case a stylised tint is ever wanted back.")
src = re.sub(r"[ \t]*CloudColorVarying\(sunDir, baseColor\);[ \t]*\n", "", src)

# CastStep() was the only caller of the dropped sphereCast(), and nothing calls
# CastStep - it is leftover scaffolding from the 2021 shader.
rep_re("float3 CastStep(float3 origin, float3 ray)",
       "    return origin;\n}",
       "// (CastStep removed: dead scaffolding, and its sphereCast() went with the\n"
       "//  old two-term atmosphere.)")

# --------------------------------------------------------------- mainImage
rep_re("    vec3 c = OutterSpace(direction, vSunLightDir);",
       "    c = clouds.xyz+c*clouds.w;",
       """    // Ambient probes for the clouds: view-INDEPENDENT, so in a real engine they
    // belong in a 1x1 pre-pass (Paper 6 does exactly that).  Recomputed per
    // pixel here because this is a single-pass baseline.
    float3 probeCam = Game2Atm_Alt(CameraPosition);
    float3 probeL;
    float3 probeI = LightSource(vSunLightDir, probeL)*PSMS_LIGHT_SCALE;
    float3 probeT;
    gAmbZenith = PSMSIntegrate(probeCam, float3(0.0, 0.0, 1.0), 1e8,
                               PSMS_PROBE_STEPS, probeL, probeI, probeT);
    // The shadow side of a cloud faces AWAY from the sun, so weight the
    // anti-solar horizon heavily rather than the bright patch beside the sun.
    float3 shz  = float3(probeL.x, probeL.y, 0.0);
    float3 hdir = (length(shz) > 1e-4) ? normalize(shz) : float3(1.0, 0.0, 0.0);
    gAmbHorizon = 0.30*PSMSIntegrate(probeCam, normalize( hdir + float3(0.0,0.0,0.10)), 1e8,
                                     PSMS_PROBE_STEPS, probeL, probeI, probeT)
                + 0.70*PSMSIntegrate(probeCam, normalize(-hdir + float3(0.0,0.0,0.10)), 1e8,
                                     PSMS_PROBE_STEPS, probeL, probeI, probeT);

    vec3 c = Sky(OutterSpace(direction, vSunLightDir), probeCam, direction, vSunLightDir);

    vec4 clouds = RenderClouds(direction, CameraPosition, vSunLightDir, 9999999999.0, time);
    c = clouds.xyz+c*clouds.w;""")

# ------------------------------------------------------------- calibration
rep("""#define NEXT_BIG_SKIP            1.0e6""",
    """#define NEXT_BIG_SKIP            1.0e6

// ---- cloud <-> Paper 3 radiometric calibration ---------------------------
// The marcher's fx.x is a transmittance-weighted scatter integral, not an
// albedo, so it needs one gain to sit correctly against sky radiances on Paper
// 3's scale.  Calibrated against shader_HSE_Next at the gate's sun elevation.
#define PSMS_CLOUD_GAIN     1.0
#define PSMS_CLOUD_AMBIENT  1.0""")

# ---------------------------------------------------------------------- emit
# TWO files from one source.  The baseline keeps every optimisation switch off,
# so it stays the frozen quality target psms_gate.py measures against; _2 is the
# one being optimised.  Both come from here, so psms_atmosphere.glsl never
# describes only one of them and they cannot silently diverge.
OPTIMISED = {
    # Round 1, validated by psms_gate.py over all twelve configurations.
    # The diffuse gather is 6 Chapman beams per march step and was 40% of the
    # shader; J is smooth by construction, so sampling it every 12 steps and
    # interpolating costs nothing measurable (worst metric ratio 1.00x).
    "PSMS_GATHER_LERP": 12,
    # With the gather amortised, the eye->cloud segment no longer needs 24 steps
    # for its own sake - the atmosphere is near-homogeneous over <40 km.
    "PSMS_AERIAL_STEPS": 12,
    # NOT taken: PSMS_SUNTRANS_LERP 4 (2.93x, and it fails specifically on the
    # SIDE-LIT sun configs - a midday-only test set would have passed it),
    # PSMS_PROBE_STEPS 6 (1.62x), PSMS_SKY_STEPS 28 (2.04x).
}


def emit(name, overrides):
    out = src
    for k, v in overrides.items():
        pat = r"(#define %s\s+)\S+" % k
        assert re.search(pat, out), "no such define: %s" % k
        out = re.sub(pat, lambda m, v=v: m.group(1) + str(v), out, count=1)
    io.open(os.path.join(ART, name), "w", encoding="utf-8", newline="\n").write(out)
    bad = [i+1 for i, l in enumerate(out.split("\n")) if any(ord(ch) > 127 for ch in l)]
    print("wrote %-30s %5d lines  non-ascii %s  %s"
          % (name, out.count("\n")+1, bad[:3], overrides or "(baseline: all switches off)"))


emit("shader_HSE_PSMS_Next.glsl", {})
emit("shader_HSE_PSMS_Next_2.glsl", OPTIMISED)
