"""
Scientist #9: Isaac Newton (1666-1687) — The Falling Apple, Rainbow Prism & Universal Gravity
Full 30-Second (720 Frames @ 24fps, 1080x1920 Lanczos) Blender 4.2 LTS Cycles V6 Dual-GPU Cinema Short

100% Unique V6 Visual Identity (Woolsthorpe & Trinity College Cambridge Golden-Twilight Stone Conservatory):
  - Procedural Honey-Limestone Gothic Ashlar Walls, Sculpted Fluted Stone Columns, Pointed Lancet Gothic
    Leaded Observatory Window opening onto a Luminous Golden-Twilight Starry Indigo Sky with 3D Crescent Moon,
    Carved English Oak & Brass-Inlaid Observatory Desk, Volumetric Golden God-Ray Sunbeams, 42 Floating Golden
    Dust Motes & Twin Ornate Brass Gothic Candelabra.
  - Blender 4.2 Cycles V6 Optical Compositor (`FOG_GLOW` Glare + Cinema Color Balance) + 16 spp Adaptive OIDN +
    AgX Punchy Color Management + `texture_limit_render = "2048"` (OOM-proof on Kaggle GPU T4 x2!).
  - Beat 1 (0.0-4.5s): Gnarled Woolsthorpe Apple Bough (22 sculpted emerald leaves + 3 glossy ruby-gold apples)
    -> Hero Apple detaches at t=0.65s, accelerates under gravity (1/2 g t^2), bounces onto crimson velvet pad at
    t=1.45s, spawning a glowing golden 3D downward gravity vector ('g') & expanding golden ripple ring!
  - Beat 2 (4.5-9.5s): Balanced Face-to-Face Debate (Isaac Newton vs Edmond Halley) flanking an open
    *Principia Mathematica* folio, brass dividers & crystal inkwell with razor-sharp 3D 'F = m a' morphing
    at t=6.95s into radiant cyan-gold 'F = G m1 m2 / r²'!
  - Beat 3 (9.5-15.5s): Ornate Brass Pedestal holding a Beveled Equilateral Flint-Glass Triangular Prism
    (IOR=1.62) -> crisp diagonal white sunbeam erupts into 7 distinct feathered translucent glowing Rainbow
    Dispersion Beams (Red..Violet) projecting 7 vivid spectral bands onto an angled dark obsidian optical screen!
  - Beat 4 (15.5-20.5s): True 3.8° Dutch-Tilt Stepped Mahogany & Brass Newton's Cradle with 5 mirror-chrome
    steel spheres (r=0.036m) -> Sphere 0 pulls back (-48°), CLACKS at t=17.45s launching Sphere 4 sky-high (+62°)
    with 14 kinetic gold sparks & shock ring, then rebounds at 18.25s & 19.10s as Halley recoils in shock!
  - Beat 5 (20.5-25.5s): Newton's Brass Reflecting Telescope cradling a luminous Sapphire-Gold 3D Curved
    Space-Time Gravity Well (7 concentric funnel rings + 12 meridian ribs), Procedural 3D Earth with Azure
    Fresnel Atmosphere Halo & Cratered 3D Moon tethered by a pulsing golden gravity force beam!
  - Beat 6 (25.5-30.0s): Grand Cast Finale — Earth-Moon Gravity Well glides center, sleek Metallic 3D Modern
    Orbital Rocket with glowing amber-cyan plasma thruster flies along Newton's tilted golden orbit loop,
    crowned by self-luminous 3D Cinema Gold title 'ISAAC NEWTON / F = G m1 m2 / r²'!
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import bmesh
import bpy
from mathutils import Euler, Quaternion, Vector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from mpfb_characters import apply_facial_acting, create_realistic_scientist, set_bone_char_rot


# ============================================================================
# 1. CLI ARGUMENTS & CONFIGURATION
# ============================================================================
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Isaac Newton V6 Cycles Dual-GPU 3D Short Renderer (MPFB2 Cast)")
    p.add_argument("--start-frame", type=int, default=1)
    p.add_argument("--end-frame", type=int, default=720)
    p.add_argument("--fps", type=int, default=int(os.environ.get("NEWTON_FPS", "24")))
    p.add_argument("--samples", type=int, default=int(os.environ.get("NEWTON_SAMPLES", "16")))
    p.add_argument("--scale", type=int, default=int(os.environ.get("NEWTON_SCALE", "75")))
    p.add_argument("--out-dir", type=str, default=os.environ.get("NEWTON_FRAMES_DIR", "render_frames"))
    p.add_argument("--qa-only", action="store_true", help="Render 6 hero Beat keyframes to qa_frames/")
    return p.parse_args(argv)


ARGS = parse_args()
FPS = ARGS.fps
DURATION_SEC = 30
TOTAL_FRAMES = FPS * DURATION_SEC
RES_X = 1080
RES_Y = 1920
OUT_DIR = os.path.abspath("qa_frames" if ARGS.qa_only else ARGS.out_dir)
os.makedirs(OUT_DIR, exist_ok=True)

# Load pre-mastered Kokoro voice_manifest.json for exact jaw lip-sync!
MANIFEST_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "audio", "output", "voice_manifest.json"))
VOICE_MANIFEST = json.load(open(MANIFEST_PATH, encoding="utf-8")) if os.path.isfile(MANIFEST_PATH) else {}


def sample_lip_sync(speaker: str, t: float) -> float:
    curve = VOICE_MANIFEST.get("lipSync", {}).get(speaker, [])
    if not curve:
        return 0.0
    idx = max(0, min(len(curve) - 1, int(round(t * 30.0))))
    return float(curve[idx])


# ============================================================================
# 2. MATH, EASING & EXACT QUATERNION LOOK-AT WITH DUTCH TILT
# ============================================================================
def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def smoothstep(e0: float, e1: float, x: float) -> float:
    t = clamp01((x - e0) / max(1e-6, e1 - e0))
    return t * t * (3.0 - 2.0 * t)


def ease_out_quint(x: float) -> float:
    t = clamp01(x)
    return 1.0 - (1.0 - t) ** 5


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def vlerp(a: Vector, b: Vector, t: float) -> Vector:
    return a.lerp(b, t)


def bezier_quad(p0: Vector, p1: Vector, p2: Vector, t: float) -> Vector:
    u = 1.0 - t
    return p0 * (u * u) + p1 * (2.0 * u * t) + p2 * (t * t)


def look_at_quat(cam_pos: Vector, target: Vector, roll_deg: float = 0.0) -> Quaternion:
    fwd = (target - cam_pos).normalized()
    q = fwd.to_track_quat("-Z", "Y")
    if abs(roll_deg) > 1e-4:
        q = q @ Quaternion(Vector((0.0, 0.0, 1.0)), math.radians(roll_deg))
    return q


# ============================================================================
# 3. CLEAN SCENE (EXTENSION-SAFE FOR MPFB2!) & CONFIGURE CYCLES V6 + BLOOM
# ============================================================================
# IMPORTANT: Do NOT call read_factory_settings(use_empty=True) — it unloads Blender 4.2 extensions like MPFB2!
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
for mesh in list(bpy.data.meshes):
    bpy.data.meshes.remove(mesh, do_unlink=True)
for crv in list(bpy.data.curves):
    bpy.data.curves.remove(crv, do_unlink=True)
for mat in list(bpy.data.materials):
    bpy.data.materials.remove(mat, do_unlink=True)
for cam_d in list(bpy.data.cameras):
    bpy.data.cameras.remove(cam_d, do_unlink=True)
for l_d in list(bpy.data.lights):
    bpy.data.lights.remove(l_d, do_unlink=True)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.render.resolution_x = RES_X
scene.render.resolution_y = RES_Y
scene.render.resolution_percentage = ARGS.scale
scene.render.fps = FPS
scene.render.use_persistent_data = True
scene.render.use_overwrite = False
scene.render.use_placeholder = False
scene.frame_start = max(1, ARGS.start_frame)
scene.frame_end = min(TOTAL_FRAMES, ARGS.end_frame)

# V6 Next-Level Cycles Quality + OOM-Proof Dual-T4 GPU Budget
scene.cycles.samples = ARGS.samples
scene.cycles.use_adaptive_sampling = True
scene.cycles.adaptive_threshold = 0.045
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPENIMAGEDENOISE"
scene.cycles.denoising_input_passes = "RGB_ALBEDO"
scene.cycles.denoising_prefilter = "NONE"
try:
    scene.cycles.denoising_use_gpu = True
except Exception:
    pass
scene.cycles.caustics_reflective = False
scene.cycles.caustics_refractive = False
scene.cycles.sample_clamp_indirect = 4.5
scene.cycles.blur_glossy = 0.85
scene.cycles.max_bounces = 6
scene.cycles.transmission_bounces = 6
scene.cycles.glossy_bounces = 3
scene.cycles.diffuse_bounces = 2
scene.cycles.transparent_max_bounces = 14
try:
    scene.cycles.texture_limit_render = "2048"
except Exception:
    pass

try:
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Punchy"
except Exception:
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - High Contrast"
    except Exception:
        scene.view_settings.view_transform = "Filmic"
        scene.view_settings.look = "High Contrast"
scene.view_settings.exposure = 0.14

# V6 Hollywood Optical Compositor Bloom (`FOG_GLOW` Glare + Subtle Filmic Color Balance!)
scene.use_nodes = True
comp_nt = scene.node_tree
comp_nt.nodes.clear()
comp_rl = comp_nt.nodes.new("CompositorNodeRLayers")
comp_glare = comp_nt.nodes.new("CompositorNodeGlare")
comp_glare.glare_type = "FOG_GLOW"
comp_glare.quality = "HIGH"
comp_glare.threshold = 1.20
comp_glare.mix = -0.62
comp_glare.size = 7
comp_out = comp_nt.nodes.new("CompositorNodeComposite")
comp_nt.links.new(comp_rl.outputs["Image"], comp_glare.inputs["Image"])
try:
    comp_cb = comp_nt.nodes.new("CompositorNodeColorBalance")
    comp_cb.correction_method = "LIFT_GAMMA_GAIN"
    comp_cb.lift = (0.988, 0.994, 1.012)
    comp_cb.gamma = (1.000, 1.000, 1.000)
    comp_cb.gain = (1.022, 1.008, 0.988)
    comp_nt.links.new(comp_glare.outputs["Image"], comp_cb.inputs["Image"])
    comp_nt.links.new(comp_cb.outputs["Image"], comp_out.inputs["Image"])
except Exception:
    comp_nt.links.new(comp_glare.outputs["Image"], comp_out.inputs["Image"])

gpu_enabled = False
try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            gpu_devices = [d for d in prefs.devices if d.type == backend]
            if gpu_devices:
                for d in prefs.devices:
                    d.use = d.type == backend
                scene.cycles.device = "GPU"
                gpu_enabled = True
                print(
                    f"[Cycles V6 Cinema] {backend} GPU ({len(gpu_devices)} device(s)) | "
                    f"Frames {scene.frame_start}..{scene.frame_end} ({FPS}fps, {ARGS.scale}% res, {ARGS.samples} spp + FOG_GLOW Bloom)"
                )
                break
        except Exception:
            continue
except Exception as e:
    print(f"[Cycles] GPU warning: {e}")

if not gpu_enabled:
    scene.cycles.device = "CPU"
    print("[Cycles] Falling back to CPU")

scene.render.image_settings.file_format = "JPEG"
scene.render.image_settings.color_mode = "RGB"
scene.render.image_settings.quality = 98
scene.render.filepath = os.path.join(OUT_DIR, "frame_")


# ============================================================================
# 4. V6 PROCEDURAL CYCLES PBR MATERIALS (WOOLSTHORPE & TRINITY COLLEGE CAMBRIDGE!)
# ============================================================================
def set_bsdf(bsdf, name: str, val):
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = val


def get_rgba_mix_sockets(mix_node):
    """Blender 4.2 safe RGBA socket lookup for ShaderNodeMix (never use duplicate string keys!)."""
    rgba_in = [s for s in mix_node.inputs if s.type == "RGBA"]
    rgba_out = [s for s in mix_node.outputs if s.type == "RGBA"]
    return mix_node.inputs[0], rgba_in[0], rgba_in[1], rgba_out[0]


def make_cambridge_limestone_mat(name="CambridgeGothicLimestone", block_scale=2.8):
    """Honey-golden Trinity College Cambridge Gothic ashlar limestone with mortar relief & mineral weathering."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Rotation"].default_value = (math.pi * 0.5, 0.0, 0.0)
    mp.inputs["Scale"].default_value = (1.15, 1.15, 1.15)

    br = nt.nodes.new("ShaderNodeTexBrick")
    br.offset = 0.5
    br.inputs["Color1"].default_value = (0.31, 0.24, 0.155, 1.0)
    br.inputs["Color2"].default_value = (0.22, 0.17, 0.112, 1.0)
    br.inputs["Mortar"].default_value = (0.078, 0.064, 0.048, 1.0)
    br.inputs["Scale"].default_value = block_scale
    br.inputs["Mortar Size"].default_value = 0.018
    br.inputs["Mortar Smooth"].default_value = 0.16
    br.inputs["Brick Width"].default_value = 0.62
    br.inputs["Row Height"].default_value = 0.28

    # Mineral mottling & antique limestone weathering overlay
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 7.5
    ns.inputs["Detail"].default_value = 9.0
    ns.inputs["Roughness"].default_value = 0.62

    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.25
    rmp.color_ramp.elements[0].color = (0.14, 0.105, 0.072, 1.0)
    rmp.color_ramp.elements[1].position = 0.78
    rmp.color_ramp.elements[1].color = (0.42, 0.33, 0.21, 1.0)

    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    fac_in, col_a, col_b, col_out = get_rgba_mix_sockets(mix)
    fac_in.default_value = 0.34

    # Dual-scale bump: fine grain feeding into ashlar mortar joints
    bmp_grain = nt.nodes.new("ShaderNodeBump")
    bmp_grain.inputs["Strength"].default_value = 0.14
    bmp_mortar = nt.nodes.new("ShaderNodeBump")
    bmp_mortar.inputs["Strength"].default_value = 0.34

    # Roughness variation across weathered honey stone
    r_map = nt.nodes.new("ShaderNodeMapRange")
    r_map.inputs["From Min"].default_value = 0.2
    r_map.inputs["From Max"].default_value = 0.8
    r_map.inputs["To Min"].default_value = 0.58
    r_map.inputs["To Max"].default_value = 0.84

    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], br.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(br.outputs["Color"], col_a)
    nt.links.new(rmp.outputs["Color"], col_b)
    nt.links.new(col_out, bsdf.inputs["Base Color"])
    nt.links.new(ns.outputs["Fac"], r_map.inputs["Value"])
    nt.links.new(r_map.outputs["Result"], bsdf.inputs["Roughness"])
    nt.links.new(ns.outputs["Fac"], bmp_grain.inputs["Height"])
    nt.links.new(br.outputs["Fac"], bmp_mortar.inputs["Height"])
    nt.links.new(bmp_grain.outputs["Normal"], bmp_mortar.inputs["Normal"])
    nt.links.new(bmp_mortar.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_carved_stone_mat(name="CarvedKettonStone"):
    """Sculpted golden-cream Ketton limestone for fluted columns, capitals & Gothic arch voussoirs."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 11.0
    ns.inputs["Detail"].default_value = 8.0
    ns.inputs["Roughness"].default_value = 0.58
    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.22
    rmp.color_ramp.elements[0].color = (0.21, 0.165, 0.110, 1.0)
    rmp.color_ramp.elements[1].position = 0.80
    rmp.color_ramp.elements[1].color = (0.39, 0.315, 0.210, 1.0)
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.16
    set_bsdf(bsdf, "Roughness", 0.58)
    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_english_oak_mat(name="CarvedEnglishOak", scale_vec=(1.2, 4.8, 1.2), dark=False):
    """Rich honey-amber aged English oak with organic wood-grain roughness variation & satin lacquer."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = scale_vec
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 6.2
    ns.inputs["Detail"].default_value = 9.0
    ns.inputs["Distortion"].default_value = 1.65
    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.22
    rmp.color_ramp.elements[0].color = (0.048, 0.024, 0.011, 1.0) if dark else (0.088, 0.046, 0.018, 1.0)
    rmp.color_ramp.elements[1].position = 0.80
    rmp.color_ramp.elements[1].color = (0.125, 0.068, 0.030, 1.0) if dark else (0.235, 0.128, 0.048, 1.0)
    r_map = nt.nodes.new("ShaderNodeMapRange")
    r_map.inputs["To Min"].default_value = 0.34 if dark else 0.25
    r_map.inputs["To Max"].default_value = 0.46 if dark else 0.37
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.11
    set_bsdf(bsdf, "Coat Weight", 0.38 if not dark else 0.22)
    set_bsdf(bsdf, "Coat Roughness", 0.14)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ns.outputs["Fac"], r_map.inputs["Value"])
    nt.links.new(r_map.outputs["Result"], bsdf.inputs["Roughness"])
    nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_metal_mat(name, rgb, rough=0.18, metallic=0.96, bump_str=0.022, coat=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    set_bsdf(bsdf, "Base Color", (*rgb, 1.0))
    set_bsdf(bsdf, "Metallic", metallic)
    set_bsdf(bsdf, "Roughness", rough)
    if coat > 0.0:
        set_bsdf(bsdf, "Coat Weight", coat)
    if bump_str > 0.0:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        ns = nt.nodes.new("ShaderNodeTexNoise")
        ns.inputs["Scale"].default_value = 46.0
        ns.inputs["Detail"].default_value = 6.0
        bmp = nt.nodes.new("ShaderNodeBump")
        bmp.inputs["Strength"].default_value = bump_str
        nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
        nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
        nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_glass_mat(name, ior=1.62, tint=(0.96, 0.99, 1.0), rough=0.006, coat=0.65):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    set_bsdf(bsdf, "Base Color", (*tint, 1.0))
    set_bsdf(bsdf, "Roughness", rough)
    set_bsdf(bsdf, "IOR", ior)
    set_bsdf(bsdf, "Transmission Weight", 0.98)
    set_bsdf(bsdf, "Transmission", 0.98)
    set_bsdf(bsdf, "Coat Weight", coat)
    set_bsdf(bsdf, "Coat Roughness", 0.02)
    return mat


def make_emissive_mat(name, rgb, strength=4.5):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb, 1.0)
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def make_feathered_beam_mat(name, rgb, strength=5.0, power=1.55, opacity=0.78):
    """Feathered soft-edge optical light-beam shader (Transparent <-> Emission via LayerWeight.Facing)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    mix = nt.nodes.new("ShaderNodeMixShader")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb, 1.0)
    em.inputs["Strength"].default_value = strength
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.5
    inv = nt.nodes.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = power
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = opacity
    nt.links.new(lw.outputs["Facing"], inv.inputs[1])
    nt.links.new(inv.outputs["Value"], pw.inputs[0])
    nt.links.new(pw.outputs["Value"], mul.inputs[0])
    nt.links.new(mul.outputs["Value"], mix.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
    nt.links.new(em.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


def make_godray_mat():
    """Silky feathered volumetric golden sunbeam shaft (LayerWeight Facing^2.5 + vertical fade!)."""
    mat = bpy.data.materials.new("VolumetricGoldenGodRay")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    mix = nt.nodes.new("ShaderNodeMixShader")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (1.0, 0.78, 0.36, 1.0)
    em.inputs["Strength"].default_value = 0.72

    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.5
    inv = nt.nodes.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = 2.5

    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    fade_rmp = nt.nodes.new("ShaderNodeValToRGB")
    fade_rmp.color_ramp.elements[0].position = 0.02
    fade_rmp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    e_mid = fade_rmp.color_ramp.elements.new(0.45)
    e_mid.color = (0.20, 0.20, 0.20, 1.0)
    fade_rmp.color_ramp.elements[-1].position = 0.98
    fade_rmp.color_ramp.elements[-1].color = (0.04, 0.04, 0.04, 1.0)

    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"

    nt.links.new(lw.outputs["Facing"], inv.inputs[1])
    nt.links.new(inv.outputs["Value"], pw.inputs[0])
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], fade_rmp.inputs["Fac"])
    nt.links.new(pw.outputs["Value"], mul.inputs[0])
    nt.links.new(fade_rmp.outputs["Color"], mul.inputs[1])
    nt.links.new(mul.outputs["Value"], mix.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
    nt.links.new(em.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


def make_cambridge_twilight_sky_mat():
    """Luminous Golden-Twilight Horizon fading up into Deep Starry Sapphire-Indigo Sky with procedural stars!"""
    mat = bpy.data.materials.new("CambridgeGoldenTwilightSky")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")

    sky_rmp = nt.nodes.new("ShaderNodeValToRGB")
    cr = sky_rmp.color_ramp
    cr.elements[0].position = 0.08
    cr.elements[0].color = (0.58, 0.30, 0.11, 1.0)  # Luminous warm Woolsthorpe golden-amber sunset horizon
    e1 = cr.elements.new(0.32)
    e1.color = (0.24, 0.16, 0.30, 1.0)  # Rich twilight amethyst transition
    e2 = cr.elements.new(0.58)
    e2.color = (0.045, 0.10, 0.28, 1.0)  # Cambridge royal twilight sapphire
    cr.elements[-1].position = 0.95
    cr.elements[-1].color = (0.014, 0.032, 0.11, 1.0)  # Deep velvety starry night zenith

    # Crisp procedural starfield pinpoints in upper sky
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 92.0
    star_rmp = nt.nodes.new("ShaderNodeValToRGB")
    star_rmp.color_ramp.elements[0].position = 0.0
    star_rmp.color_ramp.elements[0].color = (1.0, 0.96, 0.86, 1.0)
    star_rmp.color_ramp.elements[1].position = 0.065
    star_rmp.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)

    # Mask stars so they shine cleanly in the upper indigo sky
    mask_rmp = nt.nodes.new("ShaderNodeValToRGB")
    mask_rmp.color_ramp.elements[0].position = 0.36
    mask_rmp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    mask_rmp.color_ramp.elements[1].position = 0.62
    mask_rmp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    star_mul = nt.nodes.new("ShaderNodeMix")
    star_mul.data_type = "RGBA"
    star_mul.blend_type = "MULTIPLY"
    sm_fac, sm_a, sm_b, sm_out = get_rgba_mix_sockets(star_mul)
    sm_fac.default_value = 1.0

    sky_add = nt.nodes.new("ShaderNodeMix")
    sky_add.data_type = "RGBA"
    sky_add.blend_type = "ADD"
    sa_fac, sa_a, sa_b, sa_out = get_rgba_mix_sockets(sky_add)
    sa_fac.default_value = 1.0

    em.inputs["Strength"].default_value = 1.65
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], sky_rmp.inputs["Fac"])
    nt.links.new(tc.outputs["Generated"], vor.inputs["Vector"])
    nt.links.new(vor.outputs["Distance"], star_rmp.inputs["Fac"])
    nt.links.new(sep.outputs["Z"], mask_rmp.inputs["Fac"])
    nt.links.new(star_rmp.outputs["Color"], sm_a)
    nt.links.new(mask_rmp.outputs["Color"], sm_b)
    nt.links.new(sky_rmp.outputs["Color"], sa_a)
    nt.links.new(sm_out, sa_b)
    nt.links.new(sa_out, em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def make_ruby_apple_mat():
    """Glossy ruby-gold mottled Woolsthorpe apple skin with high clearcoat shine (Coat=0.90)."""
    mat = bpy.data.materials.new("WoolsthorpeRubyApple")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 6.5
    ns.inputs["Detail"].default_value = 7.0
    ns.inputs["Distortion"].default_value = 0.8

    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.28
    rmp.color_ramp.elements[0].color = (0.48, 0.018, 0.014, 1.0)
    e1 = rmp.color_ramp.elements.new(0.62)
    e1.color = (0.76, 0.055, 0.025, 1.0)
    rmp.color_ramp.elements[-1].position = 0.85
    rmp.color_ramp.elements[-1].color = (0.86, 0.52, 0.08, 1.0)

    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.035
    set_bsdf(bsdf, "Roughness", 0.16)
    set_bsdf(bsdf, "Coat Weight", 0.90)
    set_bsdf(bsdf, "Coat Roughness", 0.05)
    set_bsdf(bsdf, "Subsurface Weight", 0.08)

    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_apple_bark_mat():
    """Gnarled weathered orchard tree bark."""
    mat = bpy.data.materials.new("WoolsthorpeAppleBark")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (2.5, 6.0, 2.5)
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 14.0
    ns.inputs["Detail"].default_value = 10.0
    ns.inputs["Roughness"].default_value = 0.72
    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.25
    rmp.color_ramp.elements[0].color = (0.042, 0.026, 0.015, 1.0)
    rmp.color_ramp.elements[1].position = 0.78
    rmp.color_ramp.elements[1].color = (0.145, 0.095, 0.055, 1.0)
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.34
    set_bsdf(bsdf, "Roughness", 0.84)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_earth_planet_mat():
    """High-detail procedural 3D Earth: deep sapphire oceans, emerald/ochre continents & swirling white clouds!"""
    mat = bpy.data.materials.new("ProceduralEarth3D")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")

    # Continents noise
    ns_land = nt.nodes.new("ShaderNodeTexNoise")
    ns_land.inputs["Scale"].default_value = 3.2
    ns_land.inputs["Detail"].default_value = 10.0
    ns_land.inputs["Roughness"].default_value = 0.58

    land_rmp = nt.nodes.new("ShaderNodeValToRGB")
    cr = land_rmp.color_ramp
    cr.elements[0].position = 0.46
    cr.elements[0].color = (0.012, 0.095, 0.42, 1.0)  # Deep sapphire ocean
    e_sh = cr.elements.new(0.51)
    e_sh.color = (0.030, 0.24, 0.56, 1.0)  # Coastal turquoise
    e_gr = cr.elements.new(0.54)
    e_gr.color = (0.065, 0.36, 0.11, 1.0)  # Emerald continent
    cr.elements[-1].position = 0.72
    cr.elements[-1].color = (0.48, 0.36, 0.16, 1.0)  # Warm golden highlands

    # Swirling white atmospheric cloud wisps
    ns_cl = nt.nodes.new("ShaderNodeTexNoise")
    ns_cl.inputs["Scale"].default_value = 5.4
    ns_cl.inputs["Detail"].default_value = 8.0
    ns_cl.inputs["Distortion"].default_value = 1.45

    cl_rmp = nt.nodes.new("ShaderNodeValToRGB")
    cl_rmp.color_ramp.elements[0].position = 0.56
    cl_rmp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    cl_rmp.color_ramp.elements[1].position = 0.76
    cl_rmp.color_ramp.elements[1].color = (0.95, 0.97, 1.0, 1.0)

    mix_cl = nt.nodes.new("ShaderNodeMix")
    mix_cl.data_type = "RGBA"
    mix_cl.blend_type = "MIX"
    mc_fac, mc_a, mc_b, mc_out = get_rgba_mix_sockets(mix_cl)
    mc_b.default_value = (0.96, 0.98, 1.0, 1.0)

    # Ocean glossy vs land matte roughness
    rough_rmp = nt.nodes.new("ShaderNodeValToRGB")
    rough_rmp.color_ramp.elements[0].position = 0.49
    rough_rmp.color_ramp.elements[0].color = (0.16, 0.16, 0.16, 1.0)
    rough_rmp.color_ramp.elements[1].position = 0.54
    rough_rmp.color_ramp.elements[1].color = (0.68, 0.68, 0.68, 1.0)

    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.08

    set_bsdf(bsdf, "Coat Weight", 0.25)
    set_bsdf(bsdf, "Emission Strength", 0.28)

    nt.links.new(tc.outputs["Object"], ns_land.inputs["Vector"])
    nt.links.new(ns_land.outputs["Fac"], land_rmp.inputs["Fac"])
    nt.links.new(tc.outputs["Object"], ns_cl.inputs["Vector"])
    nt.links.new(ns_cl.outputs["Fac"], cl_rmp.inputs["Fac"])
    nt.links.new(cl_rmp.outputs["Color"], mc_fac)
    nt.links.new(land_rmp.outputs["Color"], mc_a)
    nt.links.new(mc_out, bsdf.inputs["Base Color"])
    if "Emission Color" in bsdf.inputs:
        nt.links.new(mc_out, bsdf.inputs["Emission Color"])
    nt.links.new(ns_land.outputs["Fac"], rough_rmp.inputs["Fac"])
    nt.links.new(rough_rmp.outputs["Color"], bsdf.inputs["Roughness"])
    nt.links.new(ns_land.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_earth_atmo_halo_mat():
    """Glowing blue Fresnel atmospheric halo around Earth (LayerWeight Fresnel^2.6)."""
    mat = bpy.data.materials.new("EarthAtmosphereHalo")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    mix = nt.nodes.new("ShaderNodeMixShader")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.16, 0.64, 1.0, 1.0)
    em.inputs["Strength"].default_value = 4.2
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.35
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = 2.6
    nt.links.new(lw.outputs["Facing"], pw.inputs[0])
    nt.links.new(pw.outputs["Value"], mix.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
    nt.links.new(em.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


def make_cratered_moon_mat():
    """Procedural Voronoi-cratered dusty lunar regolith."""
    mat = bpy.data.materials.new("CrateredMoon3D")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 18.0
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 5.5
    ns.inputs["Detail"].default_value = 8.0
    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.25
    rmp.color_ramp.elements[0].color = (0.30, 0.31, 0.34, 1.0)
    rmp.color_ramp.elements[1].position = 0.78
    rmp.color_ramp.elements[1].color = (0.76, 0.75, 0.72, 1.0)
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.28
    set_bsdf(bsdf, "Roughness", 0.82)
    set_bsdf(bsdf, "Emission Color", (0.78, 0.80, 0.86, 1.0))
    set_bsdf(bsdf, "Emission Strength", 0.22)
    nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(vor.outputs["Distance"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


# Instantiate core V6 material palette
limestone_mat = make_cambridge_limestone_mat()
carved_stone_mat = make_carved_stone_mat()
english_oak_mat = make_english_oak_mat("CarvedEnglishOak", (1.2, 4.8, 1.2), dark=False)
wainscot_oak_mat = make_english_oak_mat("WainscotEnglishOak", (2.2, 2.2, 2.2), dark=True)
brass_mat = make_metal_mat("CambridgeBrass", (0.90, 0.65, 0.22), rough=0.17, metallic=0.96, bump_str=0.020)
bronze_mat = make_metal_mat("AgedBronze", (0.28, 0.19, 0.095), rough=0.32, metallic=0.88, bump_str=0.026)
mirror_chrome_mat = make_metal_mat("NewtonCradleChrome", (0.93, 0.95, 0.98), rough=0.035, metallic=1.0, bump_str=0.0)
iron_mat = make_metal_mat("GothicWroughtIron", (0.052, 0.058, 0.070), rough=0.38, metallic=0.86, bump_str=0.030)
obsidian_mat = make_metal_mat("ObsidianOpticalSlate", (0.014, 0.016, 0.024), rough=0.22, metallic=0.08, bump_str=0.010, coat=0.35)
crimson_velvet_mat = make_metal_mat("CrimsonVelvetPad", (0.28, 0.024, 0.036), rough=0.88, metallic=0.0, bump_str=0.0)
leather_navy_mat = make_metal_mat("PrincipiaLeather", (0.06, 0.085, 0.18), rough=0.52, metallic=0.0, bump_str=0.035)
parchment_mat = make_metal_mat("PrincipiaParchment", (0.93, 0.86, 0.70), rough=0.66, metallic=0.0, bump_str=0.022)
ink_dark_mat = make_metal_mat("PrincipiaInk", (0.035, 0.028, 0.022), rough=0.30, metallic=0.0, bump_str=0.0)
quill_white_mat = make_metal_mat("GooseQuillWhite", (0.94, 0.92, 0.88), rough=0.45, metallic=0.0, bump_str=0.0)
wax_ivory_mat = make_metal_mat("CandleWaxIvory", (0.90, 0.84, 0.70), rough=0.42, metallic=0.0, bump_str=0.0)
rocket_white_mat = make_metal_mat("RocketCeramicWhite", (0.90, 0.92, 0.95), rough=0.18, metallic=0.35, bump_str=0.0, coat=0.55)
rocket_dark_mat = make_metal_mat("RocketCarbonObsidian", (0.06, 0.07, 0.095), rough=0.20, metallic=0.85, bump_str=0.0)
speculum_mirror_mat = make_metal_mat("SpeculumMirror", (0.88, 0.92, 0.96), rough=0.04, metallic=1.0, bump_str=0.0)

flint_glass_mat = make_glass_mat("FlintGlassPrism", ior=1.62, tint=(0.96, 0.99, 1.0), rough=0.006, coat=0.65)
inkwell_glass_mat = make_glass_mat("CrystalInkwellGlass", ior=1.52, tint=(0.88, 0.94, 1.0), rough=0.02, coat=0.45)

ruby_apple_mat = make_ruby_apple_mat()
apple_bark_mat = make_apple_bark_mat()
emerald_leaf_mat = make_metal_mat("OrchardEmeraldLeaf", (0.055, 0.29, 0.048), rough=0.32, metallic=0.0, bump_str=0.03, coat=0.22)

twilight_sky_mat = make_cambridge_twilight_sky_mat()
godray_mat = make_godray_mat()
earth_planet_mat = make_earth_planet_mat()
earth_atmo_mat = make_earth_atmo_halo_mat()
cratered_moon_mat = make_cratered_moon_mat()

book_mats = [
    leather_navy_mat,
    make_metal_mat("CambridgeTomeCrimson", (0.30, 0.045, 0.055), rough=0.58, metallic=0.0, bump_str=0.0),
    make_metal_mat("CambridgeTomeSage", (0.045, 0.16, 0.11), rough=0.58, metallic=0.0, bump_str=0.0),
    make_metal_mat("CambridgeTomeOchre", (0.34, 0.20, 0.06), rough=0.58, metallic=0.0, bump_str=0.0),
    make_metal_mat("CambridgeTomeVellum", (0.68, 0.58, 0.42), rough=0.64, metallic=0.0, bump_str=0.0),
]

# AgX-calibrated vivid V6 glowing optical materials (bloom beautifully via FOG_GLOW Compositor!)
gold_glow_mat = make_emissive_mat("NewtonGoldGlow", (1.0, 0.66, 0.10), strength=4.8)
cyan_glow_mat = make_emissive_mat("GravitySapphireGlow", (0.08, 0.68, 1.0), strength=4.4)
moon_crescent_mat = make_emissive_mat("CrescentMoonGlow", (1.0, 0.92, 0.72), strength=3.6)
star_pin_mat = make_emissive_mat("TwilightStarPin", (1.0, 0.95, 0.84), strength=4.2)
dust_mote_mat = make_emissive_mat("GoldenDustMote", (1.0, 0.80, 0.36), strength=2.6)
candle_flame_mat = make_emissive_mat("CandelabraFlame", (1.0, 0.52, 0.08), strength=7.5)
sunbeam_white_mat = make_feathered_beam_mat("PrismWhiteSunbeam", (1.0, 0.96, 0.86), strength=5.8, power=1.35, opacity=0.88)
rocket_plume_mat = make_feathered_beam_mat("RocketPlasmaPlume", (0.35, 0.78, 1.0), strength=7.5, power=1.4, opacity=0.90)
rocket_core_mat = make_emissive_mat("RocketCoreFlame", (1.0, 0.64, 0.14), strength=8.0)

# 7 Distinct Spectral Dispersion Beam & Screen Band Materials (Red, Orange, Yellow, Green, Cyan, Blue, Violet!)
SPECTRAL_RGBS = [
    ("Red", (1.00, 0.05, 0.04)),
    ("Orange", (1.00, 0.36, 0.03)),
    ("Yellow", (1.00, 0.84, 0.04)),
    ("Green", (0.05, 0.92, 0.18)),
    ("Cyan", (0.04, 0.80, 1.00)),
    ("Blue", (0.08, 0.28, 1.00)),
    ("Violet", (0.62, 0.10, 1.00)),
]
rainbow_beam_mats = [
    make_feathered_beam_mat(f"PrismBeam_{cname}", rgb, strength=5.6, power=1.25, opacity=0.84)
    for cname, rgb in SPECTRAL_RGBS
]
rainbow_band_mats = [
    make_emissive_mat(f"SpectralBand_{cname}", rgb, strength=4.4)
    for cname, rgb in SPECTRAL_RGBS
]

# Razor-sharp anti-washout 3D Formula & Title Text materials + bundled DejaVuSans-Bold font
text_gold_mat = make_emissive_mat("TextGoldCrisp", (1.0, 0.74, 0.16), strength=2.45)
text_cyangold_mat = make_emissive_mat("TextCyanGoldCrisp", (0.20, 0.84, 1.0), strength=2.65)
title_gold_mat = bpy.data.materials.new("TitleCinemaGold")
title_gold_mat.use_nodes = True
_tbsdf = title_gold_mat.node_tree.nodes["Principled BSDF"]
set_bsdf(_tbsdf, "Base Color", (1.0, 0.78, 0.24, 1.0))
set_bsdf(_tbsdf, "Metallic", 0.76)
set_bsdf(_tbsdf, "Roughness", 0.22)
set_bsdf(_tbsdf, "Emission Color", (1.0, 0.72, 0.18, 1.0))
set_bsdf(_tbsdf, "Emission Strength", 1.85)

FONT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "subtitles", "fonts", "DejaVuSans-Bold.ttf"))
HERO_FONT = bpy.data.fonts.load(FONT_PATH) if os.path.isfile(FONT_PATH) else None


# ============================================================================
# 5. HIGH-CRAFT PROCEDURAL GEOMETRY HELPERS (APPLE, PRISM, LEAVES, CURVES)
# ============================================================================
def shade_smooth(obj, angle_deg: float = 38.0):
    for p in obj.data.polygons:
        p.use_smooth = True
    es = obj.modifiers.new("SmoothSplit", "EDGE_SPLIT")
    es.split_angle = math.radians(angle_deg)


def add_box(name, loc, scale, mat, rot=(0.0, 0.0, 0.0), bevel_w=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    o.scale = scale
    # MANDATORY Blender 4.2 rule: location=False, rotation=False, scale=True!
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel_w > 0:
        bv = o.modifiers.new("Bevel", "BEVEL")
        bv.width = bevel_w
        bv.segments = 2
    o.data.materials.append(mat)
    return o


def add_cyl(name, loc, radius, depth, mat, rot=(0.0, 0.0, 0.0), verts=48, smooth=True, bevel_w=0.0):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    if smooth:
        shade_smooth(o)
    if bevel_w > 0:
        bv = o.modifiers.new("Bevel", "BEVEL")
        bv.width = bevel_w
        bv.segments = 2
    o.data.materials.append(mat)
    return o


def add_cone(name, loc, r1, r2, depth, mat, rot=(0.0, 0.0, 0.0), verts=40):
    bpy.ops.mesh.primitive_cone_add(vertices=verts, radius1=r1, radius2=r2, depth=depth, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    shade_smooth(o)
    o.data.materials.append(mat)
    return o


def add_sphere(name, loc, radius, mat, segs=32, rings=20):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segs, ring_count=rings, radius=radius, location=loc)
    o = bpy.context.active_object
    o.name = name
    for p in o.data.polygons:
        p.use_smooth = True
    o.data.materials.append(mat)
    return o


def add_torus(name, loc, major_r, minor_r, mat, rot=(0.0, 0.0, 0.0), maj_seg=64, min_seg=14):
    bpy.ops.mesh.primitive_torus_add(
        major_radius=major_r,
        minor_radius=minor_r,
        major_segments=maj_seg,
        minor_segments=min_seg,
        location=loc,
        rotation=rot,
    )
    o = bpy.context.active_object
    o.name = name
    for p in o.data.polygons:
        p.use_smooth = True
    o.data.materials.append(mat)
    return o


def build_curve_tube(name, points_with_radius, mat, bevel_res=5, loc=(0.0, 0.0, 0.0)):
    """Build a tapered smooth 3D tubular curve from [(x, y, z, r_scale), ...]."""
    cd = bpy.data.curves.new(name + "_Curve", type="CURVE")
    cd.dimensions = "3D"
    cd.bevel_depth = 1.0
    cd.bevel_resolution = bevel_res
    cd.use_fill_caps = True
    sp = cd.splines.new("POLY")
    sp.points.add(len(points_with_radius) - 1)
    for i, (px, py, pz, pr) in enumerate(points_with_radius):
        sp.points[i].co = (px, py, pz, 1.0)
        sp.points[i].radius = pr
    obj = bpy.data.objects.new(name, cd)
    obj.location = Vector(loc)
    scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def build_organic_apple(name, loc, radius=0.044):
    """Build a true sculpted organic Woolsthorpe apple (top stem dimple, bottom calyx dimple,
    subtle 5-lobe organic fullness, curved wooden stem & tiny green leaf!) parented to an Empty."""
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=loc)
    root = bpy.context.active_object
    root.name = name

    mesh = bpy.data.meshes.new(name + "_FruitMesh")
    bm = bmesh.new()
    n_seg = 28
    n_ring = 18
    # Revolved profile (r_frac, z_frac) from bottom calyx dimple to top stem dimple
    profile = []
    for i in range(n_ring + 1):
        u = i / n_ring
        th = u * math.pi  # 0=bottom, pi=top
        # Base sphere with heart-apple taper (wider upper shoulders, slight bottom taper) + deep polar dimples
        s = math.sin(th)
        c = -math.cos(th)
        shoulder = 1.0 + 0.14 * math.sin(u * math.pi) * (0.4 + 0.6 * u) - 0.10 * (1.0 - u)
        dimple = 0.0
        if u < 0.16:
            dimple = (0.16 - u) * 1.15
        elif u > 0.82:
            dimple = -(u - 0.82) * 1.65
        rf = max(0.002, s * shoulder)
        zf = c * 0.94 + dimple
        profile.append((rf * radius, zf * radius))

    ring_verts = []
    for r_i, (rr, zz) in enumerate(profile):
        ring = []
        for s_i in range(n_seg):
            ang = (s_i / n_seg) * math.pi * 2.0
            # Subtle 5-lobe natural heirloom apple fullness
            lobe = 1.0 + 0.032 * math.cos(5.0 * ang) * math.sin((r_i / n_ring) * math.pi)
            v = bm.verts.new((math.cos(ang) * rr * lobe, math.sin(ang) * rr * lobe, zz))
            ring.append(v)
        ring_verts.append(ring)

    for r_i in range(n_ring):
        for s_i in range(n_seg):
            sn = (s_i + 1) % n_seg
            bm.faces.new((ring_verts[r_i][s_i], ring_verts[r_i][sn], ring_verts[r_i + 1][sn], ring_verts[r_i + 1][s_i]))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    fruit = bpy.data.objects.new(name + "_Fruit", mesh)
    scene.collection.objects.link(fruit)
    for p in fruit.data.polygons:
        p.use_smooth = True
    fruit.data.materials.append(ruby_apple_mat)
    fruit.parent = root

    # Curved woody stem emerging cleanly from top dimple
    stem = build_curve_tube(
        f"{name}_Stem",
        [
            (0.0, 0.0, radius * 0.58, 0.0028),
            (0.004, -0.002, radius * 0.96, 0.0024),
            (0.011, -0.004, radius * 1.28, 0.0032),
        ],
        apple_bark_mat,
        bevel_res=4,
    )
    stem.parent = root

    # Tiny emerald apple leaf beside stem
    leaf = build_sculpted_leaf(f"{name}_Leaf", (0.008, -0.004, radius * 1.08), length=0.036, width=0.018, rot=(0.25, 0.45, 0.55))
    leaf.parent = root
    return root


def build_sculpted_leaf(name, loc, length=0.075, width=0.034, rot=(0.0, 0.0, 0.0)):
    """Build a gracefully curved almond-shaped botanical orchard leaf with midrib crease."""
    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()
    # 4 longitudinal stations along +X with raised edges & slight graceful downward arch
    stations = [
        (0.00, 0.000, 0.000, 0.000),
        (0.28, width * 0.46, 0.004, 0.004),
        (0.62, width * 0.50, 0.002, 0.005),
        (0.86, width * 0.24, -0.005, 0.003),
        (1.00, 0.000, -0.010, 0.000),
    ]
    v_tip0 = bm.verts.new((0.0, 0.0, 0.0))
    rows = []
    for u, w, z_arc, z_cup in stations[1:-1]:
        x = u * length
        v_l = bm.verts.new((x, w, z_arc + z_cup))
        v_c = bm.verts.new((x, 0.0, z_arc))
        v_r = bm.verts.new((x, -w, z_arc + z_cup))
        rows.append((v_l, v_c, v_r))
    v_tip1 = bm.verts.new((length, 0.0, stations[-1][2]))

    bm.faces.new((v_tip0, rows[0][1], rows[0][0]))
    bm.faces.new((v_tip0, rows[0][2], rows[0][1]))
    for i in range(len(rows) - 1):
        bm.faces.new((rows[i][0], rows[i][1], rows[i + 1][1], rows[i + 1][0]))
        bm.faces.new((rows[i][1], rows[i][2], rows[i + 1][2], rows[i + 1][1]))
    bm.faces.new((rows[-1][0], rows[-1][1], v_tip1))
    bm.faces.new((rows[-1][1], rows[-1][2], v_tip1))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(loc)
    obj.rotation_euler = Euler(rot, "XYZ")
    scene.collection.objects.link(obj)
    for p in obj.data.polygons:
        p.use_smooth = True
    obj.data.materials.append(emerald_leaf_mat)
    return obj


def build_equilateral_prism(name, loc, side=0.140, depth_y=0.112, mat=None):
    """Build a crisp beveled equilateral triangular optical flint-glass prism facing the camera (XZ triangle)."""
    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()
    h = side * math.sqrt(3.0) * 0.5
    # Centered equilateral triangle in XZ plane: apex at top (+Z), flat base at bottom (-Z)
    pts_xz = [
        (-side * 0.5, -h / 3.0),
        (side * 0.5, -h / 3.0),
        (0.0, 2.0 * h / 3.0),
    ]
    hy = depth_y * 0.5
    v_front = [bm.verts.new((px, -hy, pz)) for px, pz in pts_xz]
    v_back = [bm.verts.new((px, hy, pz)) for px, pz in pts_xz]
    bm.faces.new((v_front[0], v_front[1], v_front[2]))
    bm.faces.new((v_back[2], v_back[1], v_back[0]))
    for i in range(3):
        inext = (i + 1) % 3
        bm.faces.new((v_front[i], v_back[i], v_back[inext], v_front[inext]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(loc)
    scene.collection.objects.link(obj)
    bv = obj.modifiers.new("OpticalBevel", "BEVEL")
    bv.width = 0.0032
    bv.segments = 3
    if mat:
        obj.data.materials.append(mat)
    return obj


def orient_beam(obj, p1: Vector, p2: Vector, radius_scale: float = 1.0):
    diff = p2 - p1
    length = diff.length
    obj.location = (p1 + p2) * 0.5
    obj.scale = (radius_scale, radius_scale, max(0.0005, length))
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = diff.normalized().to_track_quat("Z", "Y") if length > 1e-5 else Quaternion((1, 0, 0, 0))


def build_fluted_stone_column(name, x, y, z_bot, z_top, radius=0.115):
    """Build a sculpted fluted Trinity College Gothic stone column with carved plinth, 10 flutes, astragal & capital."""
    h = z_top - z_bot
    zc = (z_bot + z_top) * 0.5
    add_cyl(f"{name}_Core", (x, y, zc), radius * 0.90, h, carved_stone_mat, verts=32)
    for fi in range(10):
        ang = (fi / 10.0) * math.pi * 2.0
        fx = x + math.cos(ang) * radius * 0.88
        fy = y + math.sin(ang) * radius * 0.88
        add_cyl(f"{name}_Flute_{fi}", (fx, fy, zc), radius * 0.16, h * 0.94, carved_stone_mat, verts=12)
    # Carved sill/plinth rings & brass astragal collar
    add_box(f"{name}_Plinth", (x, y, z_bot + 0.12), (radius * 2.5, radius * 2.5, 0.24), carved_stone_mat, bevel_w=0.008)
    add_cyl(f"{name}_SillCollar", (x, y, 0.74), radius * 1.18, 0.055, carved_stone_mat, verts=32, bevel_w=0.006)
    add_torus(f"{name}_BrassAstragal", (x, y, 0.79), radius * 1.08, 0.012, brass_mat, maj_seg=36, min_seg=10)
    add_cone(f"{name}_Capital", (x, y, z_top - 0.10), radius * 1.02, radius * 1.48, 0.20, carved_stone_mat, verts=24)
    add_box(f"{name}_Abacus", (x, y, z_top + 0.02), (radius * 2.8, radius * 2.8, 0.065), carved_stone_mat, bevel_w=0.006)


# ============================================================================
# 6. BUILD WOOLSTHORPE & TRINITY COLLEGE CAMBRIDGE GOLDEN-TWILIGHT CONSERVATORY!
# ============================================================================
FLOOR_Z = -0.99

world = bpy.data.worlds.new("CambridgeTwilightWorld")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.018, 0.028, 0.055, 1.0)
bg.inputs["Strength"].default_value = 0.65

# Carved English Oak & Brass-Inlaid Observatory Desk (top Z=0.0, Y in [-0.54, +0.34], solid front down to FLOOR_Z=-0.99!)
add_box("CambridgeOakDesk", (0.0, -0.10, -0.06), (6.2, 0.88, 0.12), english_oak_mat, bevel_w=0.009)
add_box("DeskBrassInlayStrip", (0.0, -0.544, -0.020), (6.22, 0.022, 0.036), brass_mat, bevel_w=0.004)
add_box("DeskLowerBrassMould", (0.0, -0.542, -0.104), (6.20, 0.018, 0.016), brass_mat, bevel_w=0.003)
add_box("DeskCabinetBody", (0.0, -0.12, -0.555), (6.0, 0.80, 0.87), wainscot_oak_mat)
for dr_x in (-1.72, -0.86, 0.0, 0.86, 1.72):
    add_box(f"DeskPanelOuter_{dr_x:.2f}", (dr_x, -0.524, -0.35), (0.72, 0.024, 0.36), english_oak_mat, bevel_w=0.007)
    add_box(f"DeskPanelInlay_{dr_x:.2f}", (dr_x, -0.538, -0.35), (0.58, 0.008, 0.24), brass_mat, bevel_w=0.002)
    add_box(f"DeskPanelInset_{dr_x:.2f}", (dr_x, -0.540, -0.35), (0.54, 0.010, 0.20), wainscot_oak_mat, bevel_w=0.004)
    add_cyl(f"DeskLionRosette_{dr_x:.2f}", (dr_x, -0.546, -0.35), 0.020, 0.014, brass_mat, rot=(math.pi * 0.5, 0, 0), verts=20)
    add_torus(f"DeskLionPull_{dr_x:.2f}", (dr_x, -0.552, -0.375), 0.030, 0.006, brass_mat, rot=(math.pi * 0.5, 0, 0))
add_box("ConservatoryStoneFloor", (0.0, 0.50, FLOOR_Z - 0.06), (8.2, 5.2, 0.12), carved_stone_mat)

# SEALED Honey-Limestone Ashlar Back Wall & Carved Oak Wainscot extending down to FLOOR_Z - 0.30 (-1.29m) -> ZERO blue void!
add_box("StoneWallLeft", (-2.60, 1.96, 1.15), (3.10, 0.26, 4.90), limestone_mat)
add_box("StoneWallRight", (2.60, 1.96, 1.15), (3.10, 0.26, 4.90), limestone_mat)
add_box("StoneWallLower", (0.0, 1.96, -0.28), (2.30, 0.26, 2.04), limestone_mat)  # Z in [-1.30, +0.74]
add_box("StoneWallHeader", (0.0, 1.96, 2.84), (2.50, 0.26, 1.24), limestone_mat)  # Z in [+2.22, +3.46]
add_box("CarvedOakWainscot", (0.0, 1.81, -0.28), (7.60, 0.06, 2.04), wainscot_oak_mat, bevel_w=0.006)
add_box("OakWainscotRail", (0.0, 1.78, 0.72), (7.60, 0.12, 0.055), english_oak_mat, bevel_w=0.006)
add_box("GothicStoneSill", (0.0, 1.80, 0.745), (2.30, 0.20, 0.075), carved_stone_mat, bevel_w=0.008)

# Twin Sculpted Fluted Stone Columns framing the Tall Arched Gothic Conservatory Window (X = -1.08, +1.08)
build_fluted_stone_column("GothicColLeft", -1.08, 1.80, FLOOR_Z - 0.30, 1.82, radius=0.115)
build_fluted_stone_column("GothicColRight", 1.08, 1.80, FLOOR_Z - 0.30, 1.82, radius=0.115)

# Pointed Gothic Lancet Stone Arch (14 sculpted voussoir arch stones meeting at Z = 2.26 apex!)
for side_sgn in (-1.0, 1.0):
    for seg_i in range(7):
        u_a = seg_i / 6.0
        # Circular lancet arc from (side_sgn * 1.05, 1.78) sweeping up to pointed apex (0.0, 2.26)
        ang = lerp(0.0, math.radians(66.0), u_a)
        ax = side_sgn * (1.05 * math.cos(ang))
        az = 1.76 + 0.56 * math.sin(ang)
        tilt = side_sgn * ang
        add_box(
            f"GothicVoussoir_{int(side_sgn)}_{seg_i}",
            (ax, 1.84, az),
            (0.22, 0.24, 0.13),
            carved_stone_mat,
            rot=(0.0, tilt, 0.0),
            bevel_w=0.007,
        )
add_box("GothicKeystone", (0.0, 1.82, 2.28), (0.16, 0.26, 0.20), carved_stone_mat, bevel_w=0.008)

# Wrought-Iron Gothic Window Tracery Mullions & Rose Ring (X in [-1.05, +1.05])
for mx in (-0.42, 0.42):
    add_box(f"GothicMullionV_{mx:+.2f}", (mx, 1.92, 1.48), (0.032, 0.036, 1.54), iron_mat)
add_box("GothicMullionH1", (0.0, 1.92, 1.18), (2.06, 0.036, 0.030), iron_mat)
add_box("GothicMullionH2", (0.0, 1.92, 1.66), (2.06, 0.036, 0.030), iron_mat)
add_torus("GothicRoseTraceryOuter", (0.0, 1.92, 1.94), 0.22, 0.016, iron_mat, rot=(math.pi * 0.5, 0, 0), maj_seg=40, min_seg=10)
add_torus("GothicRoseTraceryInner", (0.0, 1.92, 1.94), 0.09, 0.012, iron_mat, rot=(math.pi * 0.5, 0, 0), maj_seg=28, min_seg=10)

# Luminous Golden-Twilight / Deep Starry Indigo Sky Backdrop + 3D Glowing Crescent Moon + 32 Sparkling Stars!
add_box("CambridgeTwilightSky", (0.0, 2.42, 1.56), (3.20, 0.04, 2.80), twilight_sky_mat)
# Glowing 3D Crescent Moon visible cleanly through right upper Gothic window pane!
add_sphere("TwilightMoonGlow", (0.62, 2.28, 1.72), 0.092, moon_crescent_mat, segs=32, rings=20)
add_sphere("TwilightMoonShadow", (0.585, 2.23, 1.735), 0.086, obsidian_mat, segs=24, rings=16)
# 32 Sparkling 3D Twilight Star Pins
for st_i in range(32):
    phi = (st_i * 0.61803398875) % 1.0
    st_x = lerp(-0.95, 0.95, (st_i * 0.381966) % 1.0)
    st_z = lerp(1.16, 2.12, phi)
    if (st_x - 0.62) ** 2 + (st_z - 1.72) ** 2 < 0.025:
        continue
    st_r = 0.0045 + (st_i % 3) * 0.0022
    add_sphere(f"TwilightStar_{st_i}", (st_x, 2.30, st_z), st_r, star_pin_mat, segs=8, rings=6)

# Trinity College Oak Bookcases flanking the Gothic Window (rich scholarly depth & bokeh in all 6 beats!)
for side_idx, bx in enumerate((-2.44, -1.62, 1.62, 2.44)):
    add_box(f"TrinityCaseBack_{side_idx}", (bx, 1.72, 1.24), (0.78, 0.04, 1.48), wainscot_oak_mat)
    for sz in (0.76, 1.14, 1.52, 1.92):
        add_box(f"TrinityShelf_{side_idx}_{sz:.2f}", (bx, 1.60, sz), (0.82, 0.24, 0.036), english_oak_mat, bevel_w=0.004)
    for row_idx, rz in enumerate((0.88, 1.26, 1.64)):
        if side_idx == 2 and row_idx == 2:
            # Decorative brass armillary sphere on upper right bookcase shelf!
            arm_c = Vector((bx, 1.56, rz + 0.06))
            add_cyl("ShelfArmillaryBase", (arm_c.x, arm_c.y, rz - 0.10), 0.065, 0.024, brass_mat, verts=28)
            add_cyl("ShelfArmillaryStem", (arm_c.x, arm_c.y, rz - 0.03), 0.012, 0.12, brass_mat, verts=16)
            add_torus("ShelfArmillaryR1", arm_c, 0.105, 0.006, brass_mat, rot=(math.radians(65), 0, 0), maj_seg=32, min_seg=8)
            add_torus("ShelfArmillaryR2", arm_c, 0.095, 0.006, brass_mat, rot=(0, math.radians(72), 0.4), maj_seg=32, min_seg=8)
            continue
        for k in range(6):
            bh = 0.20 + ((k * 3 + row_idx) % 4) * 0.022
            bt = 0.044 + (k % 2) * 0.011
            add_box(
                f"ScholarTome_{side_idx}_{row_idx}_{k}",
                (bx - 0.27 + k * 0.108, 1.60, rz + bh * 0.5 - 0.10),
                (bt, 0.16, bh),
                book_mats[(k + row_idx * 2 + side_idx) % len(book_mats)],
                bevel_w=0.002,
            )

# 3 Volumetric Golden God-Ray Sunbeam Cones streaming diagonally through the Gothic Conservatory Window!
for gr_i, (gx, gy, gz, grot) in enumerate([
    (-0.68, 0.86, 1.22, (math.radians(-38.0), math.radians(18.0), 0.0)),
    (0.06, 0.90, 1.28, (math.radians(-36.0), math.radians(15.0), 0.0)),
    (0.74, 0.84, 1.20, (math.radians(-38.0), math.radians(12.0), 0.0)),
]):
    gr = add_cone(f"GoldenGodRay_{gr_i}", (gx, gy, gz), 0.42, 0.14, 2.35, godray_mat, rot=grot, verts=32)
    gr.visible_shadow = False
    gr.visible_diffuse = False
    gr.visible_glossy = False

# 42 Sparkling Floating Golden Dust Motes gently drifting in 3D space across the conservatory!
dust_motes = []
for dm_i in range(42):
    u1 = ((dm_i * 0.61803398875) + 0.13) % 1.0
    u2 = ((dm_i * 0.38196601125) + 0.27) % 1.0
    u3 = ((dm_i * 0.75487766625) + 0.07) % 1.0
    dx0 = lerp(-2.35, 1.95, u1)
    dy0 = lerp(-0.32, 1.42, u2)
    dz0 = lerp(0.18, 1.56, u3)
    dr = 0.0028 + (dm_i % 4) * 0.0009
    dm = add_sphere(f"GoldenDustMote_{dm_i:02d}", (dx0, dy0, dz0), dr, dust_mote_mat, segs=8, rings=6)
    dm.visible_shadow = False
    dust_motes.append((dm, Vector((dx0, dy0, dz0)), dm_i * 0.73))

# Twin Ornate Brass Gothic Desk Candelabra on far outer bench edges (X = -2.42, +1.96 — clear of ALL 6 shots!)
candle_flames = []
for c_idx, (cx, cy) in enumerate([(-2.42, 0.20), (1.96, 0.20)]):
    add_cyl(f"CandBase_{c_idx}", (cx, cy, 0.015), 0.078, 0.030, brass_mat, verts=32, bevel_w=0.004)
    add_cyl(f"CandStem_{c_idx}", (cx, cy, 0.125), 0.016, 0.20, brass_mat, verts=24)
    add_torus(f"CandKnop_{c_idx}", (cx, cy, 0.12), 0.026, 0.010, brass_mat, maj_seg=24, min_seg=8)
    for arm_off in (-0.078, 0.0, 0.078):
        if abs(arm_off) > 0.01:
            build_curve_tube(
                f"CandArm_{c_idx}_{arm_off:+.2f}",
                [(cx, cy, 0.17, 0.006), (cx + arm_off * 0.55, cy, 0.155, 0.005), (cx + arm_off, cy, 0.215, 0.006)],
                brass_mat,
                bevel_res=4,
            )
        cup_z = 0.225 if abs(arm_off) > 0.01 else 0.250
        add_cyl(f"CandCup_{c_idx}_{arm_off:+.2f}", (cx + arm_off, cy, cup_z), 0.022, 0.014, brass_mat, verts=20)
        add_cyl(f"CandWax_{c_idx}_{arm_off:+.2f}", (cx + arm_off, cy, cup_z + 0.060), 0.011, 0.11, wax_ivory_mat, verts=16)
        fl = add_cone(f"CandFlame_{c_idx}_{arm_off:+.2f}", (cx + arm_off, cy, cup_z + 0.132), 0.009, 0.001, 0.036, candle_flame_mat, verts=12)
        fl.visible_shadow = False
        candle_flames.append(fl)


# ============================================================================
# 7. STATION A (APPLE_POS = -0.74, -0.10, 0.0) — BEAT 1 WOOLSTHORPE FALLING APPLE!
# ============================================================================
APPLE_POS = Vector((-0.74, -0.10, 0.0))

# Gold-Braided Crimson Velvet Desk Pad resting cleanly at APPLE_POS (Z = 0.0..0.022)
add_cyl("AppleVelvetPad", (APPLE_POS.x, APPLE_POS.y, 0.011), 0.145, 0.022, crimson_velvet_mat, verts=48, bevel_w=0.006)
add_torus("AppleVelvetGoldBraid", (APPLE_POS.x, APPLE_POS.y, 0.014), 0.147, 0.008, brass_mat, maj_seg=56, min_seg=12)

# Overhead Woolsthorpe Apple Bough Root (smoothly retracts during Beat 1->2 whip-pan so Beat 2/3/6 stay 100% clean!)
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(APPLE_POS.x, APPLE_POS.y, 0.0))
apple_bough_root = bpy.context.active_object
apple_bough_root.name = "WoolsthorpeAppleBoughRoot"

# Main gnarled overarching apple branch + two forked twigs (local around APPLE_POS, Z=0.72..0.98)
main_bough = build_curve_tube(
    "WoolsthorpeMainBough",
    [
        (0.32, 0.14, 0.98, 0.026),
        (0.18, 0.06, 0.90, 0.021),
        (0.04, 0.00, 0.80, 0.016),
        (-0.06, -0.02, 0.76, 0.011),
        (-0.14, 0.01, 0.78, 0.006),
    ],
    apple_bark_mat,
    bevel_res=6,
)
main_bough.parent = apple_bough_root

twig_1 = build_curve_tube(
    "WoolsthorpeTwigA",
    [
        (0.16, 0.05, 0.89, 0.012),
        (0.14, -0.04, 0.82, 0.008),
        (0.17, -0.08, 0.79, 0.005),
    ],
    apple_bark_mat,
    bevel_res=4,
)
twig_1.parent = apple_bough_root

twig_2 = build_curve_tube(
    "WoolsthorpeTwigB",
    [
        (0.02, 0.00, 0.79, 0.010),
        (0.00, 0.00, 0.74, 0.005),
    ],
    apple_bark_mat,
    bevel_res=4,
)
twig_2.parent = apple_bough_root

# 22 Sculpted Emerald Foliage Leaves arranged naturally along the Woolsthorpe bough!
leaf_specs = [
    (0.24, 0.10, 0.95, 0.082, (0.3, -0.2, 2.5)),
    (0.21, 0.04, 0.92, 0.076, (-0.2, 0.3, -2.2)),
    (0.18, 0.08, 0.93, 0.080, (0.4, 0.2, 1.8)),
    (0.15, 0.02, 0.89, 0.074, (-0.3, -0.1, -1.6)),
    (0.12, 0.07, 0.87, 0.078, (0.2, 0.4, 2.2)),
    (0.10, -0.02, 0.85, 0.072, (-0.2, -0.3, -2.6)),
    (0.08, 0.05, 0.83, 0.080, (0.3, 0.1, 1.5)),
    (0.05, -0.03, 0.81, 0.070, (-0.4, 0.2, -1.8)),
    (0.02, 0.04, 0.81, 0.076, (0.2, -0.2, 2.8)),
    (-0.02, -0.04, 0.78, 0.074, (-0.3, 0.3, -2.1)),
    (-0.05, 0.03, 0.79, 0.078, (0.4, -0.1, 2.0)),
    (-0.09, -0.02, 0.77, 0.072, (-0.2, 0.2, -2.9)),
    (-0.12, 0.02, 0.79, 0.068, (0.2, 0.3, 2.6)),
    (-0.14, 0.00, 0.78, 0.066, (-0.3, -0.2, 3.0)),
    (0.17, -0.04, 0.85, 0.075, (0.1, 0.4, -0.7)),
    (0.15, -0.07, 0.82, 0.070, (-0.3, 0.2, -1.2)),
    (0.19, -0.08, 0.80, 0.068, (0.2, -0.3, -0.4)),
    (0.22, -0.03, 0.87, 0.072, (-0.2, 0.1, 0.4)),
    (0.00, 0.03, 0.78, 0.068, (0.3, 0.2, 1.1)),
    (-0.04, 0.05, 0.82, 0.074, (0.1, -0.2, 1.7)),
    (0.07, -0.06, 0.83, 0.072, (-0.3, 0.3, -1.0)),
    (0.13, 0.11, 0.90, 0.076, (0.4, -0.2, 1.4)),
]
for li, (lx, ly, lz, llen, lrot) in enumerate(leaf_specs):
    lf = build_sculpted_leaf(f"OrchardLeaf_{li:02d}", (lx, ly, lz), length=llen, width=llen * 0.46, rot=lrot)
    lf.parent = apple_bough_root

# 2 Companion Glossy Ruby Apples hanging gracefully from the bough
hang_apple_1 = build_organic_apple("HangingApple1", (0.16, -0.06, 0.75), radius=0.038)
hang_apple_1.parent = apple_bough_root
hang_apple_2 = build_organic_apple("HangingApple2", (-0.11, 0.02, 0.73), radius=0.035)
hang_apple_2.parent = apple_bough_root

# 1 HERO FALLING WOOLSTHORPE APPLE (detaches at t=0.65s, lands on crimson velvet pad at t=1.45s!)
hero_apple = build_organic_apple("HeroFallingApple", (APPLE_POS.x, APPLE_POS.y, 0.69), radius=0.046)

# Glowing Golden Downward Gravity Force Vector Arrow ('g') + Expanding Golden Impact Ripple Ring!
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(APPLE_POS.x + 0.11, APPLE_POS.y, 0.34))
gravity_arrow_root = bpy.context.active_object
gravity_arrow_root.name = "GravityForceArrowRoot"
g_shaft = add_cyl("GravityArrowShaft", (0.0, 0.0, 0.08), 0.009, 0.24, gold_glow_mat, verts=20)
g_shaft.parent = gravity_arrow_root
g_head = add_cone("GravityArrowHead", (0.0, 0.0, -0.075), 0.028, 0.001, 0.072, gold_glow_mat, rot=(math.pi, 0.0, 0.0), verts=24)
g_head.parent = gravity_arrow_root
bpy.ops.object.text_add(location=(0.055, -0.015, 0.06), rotation=(math.pi * 0.5, 0.0, 0.0))
g_label = bpy.context.active_object
g_label.name = "GravityLabelG"
if HERO_FONT is not None:
    g_label.data.font = HERO_FONT
g_label.data.body = "g"
g_label.data.align_x = "CENTER"
g_label.data.size = 0.088
g_label.data.extrude = 0.005
g_label.data.bevel_depth = 0.0005
g_label.data.materials.append(text_gold_mat)
g_label.parent = gravity_arrow_root

impact_ripple = add_torus("AppleImpactRipple", (APPLE_POS.x, APPLE_POS.y, 0.026), 0.060, 0.0045, gold_glow_mat, maj_seg=56, min_seg=10)
impact_ripple.visible_shadow = False


# ============================================================================
# 8. STATION CENTER (PRINCIPIA_POS = 0.00, -0.10, 0.0) — BEAT 2 DEBATE FOLIO & FORMULAS!
# ============================================================================
PRINCIPIA_POS = Vector((0.00, -0.10, 0.0))

# Open Leather-Bound *Principia Mathematica* Folio (angled V-spread boards + cream parchment pages + dark ink orbital diagram!)
add_box("PrincipiaSpine", (PRINCIPIA_POS.x, PRINCIPIA_POS.y, 0.014), (0.042, 0.265, 0.026), leather_navy_mat, bevel_w=0.004)
add_box("PrincipiaCoverL", (PRINCIPIA_POS.x - 0.112, PRINCIPIA_POS.y, 0.022), (0.215, 0.265, 0.016), leather_navy_mat, rot=(0.0, math.radians(8.0), 0.0), bevel_w=0.004)
add_box("PrincipiaCoverR", (PRINCIPIA_POS.x + 0.112, PRINCIPIA_POS.y, 0.022), (0.215, 0.265, 0.016), leather_navy_mat, rot=(0.0, math.radians(-8.0), 0.0), bevel_w=0.004)
add_box("PrincipiaPagesL", (PRINCIPIA_POS.x - 0.106, PRINCIPIA_POS.y, 0.036), (0.198, 0.246, 0.022), parchment_mat, rot=(0.0, math.radians(8.5), 0.0), bevel_w=0.003)
add_box("PrincipiaPagesR", (PRINCIPIA_POS.x + 0.106, PRINCIPIA_POS.y, 0.036), (0.198, 0.246, 0.022), parchment_mat, rot=(0.0, math.radians(-8.5), 0.0), bevel_w=0.003)
# Subtle dark ink text lines & geometric orbital diagram on open Principia pages!
for ln_i in range(4):
    add_box(
        f"PrincipiaInkL_{ln_i}",
        (PRINCIPIA_POS.x - 0.106, PRINCIPIA_POS.y - 0.065 + ln_i * 0.042, 0.049),
        (0.135, 0.008, 0.002),
        ink_dark_mat,
        rot=(0.0, math.radians(8.5), 0.0),
    )
add_torus(
    "PrincipiaOrbitalDiagram",
    (PRINCIPIA_POS.x + 0.106, PRINCIPIA_POS.y, 0.050),
    0.058,
    0.0028,
    ink_dark_mat,
    rot=(0.0, math.radians(-8.5), 0.0),
    maj_seg=40,
    min_seg=8,
)
# Brass Corner Clasps, Brass Geometry Dividers, Crystal Inkwell & White Goose Quill!
for sx_c in (-0.212, 0.212):
    add_box(f"PrincipiaClasp_{sx_c:+.2f}", (PRINCIPIA_POS.x + sx_c, PRINCIPIA_POS.y - 0.08, 0.036), (0.018, 0.026, 0.022), brass_mat, bevel_w=0.002)
# Brass Geometry Dividers resting neatly front-left of Principia
add_cyl("DividersJoint", (PRINCIPIA_POS.x - 0.24, PRINCIPIA_POS.y - 0.11, 0.012), 0.011, 0.012, brass_mat, verts=20)
add_cone("DividersLeg1", (PRINCIPIA_POS.x - 0.18, PRINCIPIA_POS.y - 0.14, 0.009), 0.005, 0.001, 0.14, brass_mat, rot=(0.0, math.pi * 0.5, -0.45), verts=16)
add_cone("DividersLeg2", (PRINCIPIA_POS.x - 0.18, PRINCIPIA_POS.y - 0.08, 0.009), 0.005, 0.001, 0.14, brass_mat, rot=(0.0, math.pi * 0.5, 0.35), verts=16)
# Crystal Inkwell & White Quill on right side of Principia (within +/-0.24 of center -> clean of Beat 4!)
add_box("CrystalInkwell", (PRINCIPIA_POS.x + 0.25, PRINCIPIA_POS.y + 0.06, 0.024), (0.048, 0.048, 0.048), inkwell_glass_mat, bevel_w=0.006)
add_cyl("InkwellCollar", (PRINCIPIA_POS.x + 0.25, PRINCIPIA_POS.y + 0.06, 0.052), 0.016, 0.014, brass_mat, verts=24)
add_cyl("InkDarkPool", (PRINCIPIA_POS.x + 0.25, PRINCIPIA_POS.y + 0.06, 0.020), 0.018, 0.028, ink_dark_mat, verts=20)
add_cone("QuillShaft", (PRINCIPIA_POS.x + 0.28, PRINCIPIA_POS.y + 0.06, 0.125), 0.002, 0.003, 0.18, quill_white_mat, rot=(0.10, math.radians(22.0), 0.0), verts=12)
add_box("QuillPlume", (PRINCIPIA_POS.x + 0.31, PRINCIPIA_POS.y + 0.065, 0.165), (0.026, 0.004, 0.11), quill_white_mat, rot=(0.10, math.radians(22.0), 0.0), bevel_w=0.002)

# Razor-Sharp Glowing 3D Formulas centered cleanly at Y=-0.14 between Newton (-0.46) & Halley (+0.46)!
bpy.ops.object.text_add(location=(PRINCIPIA_POS.x, PRINCIPIA_POS.y - 0.04, 0.26), rotation=(math.pi * 0.5, 0.0, 0.0))
formula_fma_txt = bpy.context.active_object
formula_fma_txt.name = "NewtonFormula_Fma"
if HERO_FONT is not None:
    formula_fma_txt.data.font = HERO_FONT
formula_fma_txt.data.body = "F = m a"
formula_fma_txt.data.align_x = "CENTER"
formula_fma_txt.data.size = 0.072
formula_fma_txt.data.space_character = 1.15
formula_fma_txt.data.extrude = 0.0055
formula_fma_txt.data.bevel_depth = 0.0006
formula_fma_txt.data.materials.append(text_gold_mat)

bpy.ops.object.text_add(location=(PRINCIPIA_POS.x, PRINCIPIA_POS.y - 0.04, 0.30), rotation=(math.pi * 0.5, 0.0, 0.0))
formula_grav_txt = bpy.context.active_object
formula_grav_txt.name = "NewtonFormula_Gravity"
if HERO_FONT is not None:
    formula_grav_txt.data.font = HERO_FONT
formula_grav_txt.data.body = "F = G m1 m2 / r²"
formula_grav_txt.data.align_x = "CENTER"
formula_grav_txt.data.size = 0.056
formula_grav_txt.data.space_character = 1.14
formula_grav_txt.data.extrude = 0.0055
formula_grav_txt.data.bevel_depth = 0.0006
formula_grav_txt.data.materials.append(text_cyangold_mat)


# ============================================================================
# 9. STATION B (PRISM_POS = -1.96, -0.06, 0.0) — BEAT 3 CRYSTAL GLASS PRISM & 7-SPECTRAL RAINBOW!
# ============================================================================
PRISM_POS = Vector((-1.96, -0.06, 0.0))

# Ornate Turned Brass Optical Pedestal holding the Flint-Glass Prism cleanly at Z = 0.28!
add_cyl("PrismBaseFoot", (PRISM_POS.x, PRISM_POS.y, 0.016), 0.095, 0.032, brass_mat, verts=48, bevel_w=0.005)
add_cyl("PrismBaseStep", (PRISM_POS.x, PRISM_POS.y, 0.040), 0.068, 0.020, bronze_mat, verts=40, bevel_w=0.004)
add_cyl("PrismStem", (PRISM_POS.x, PRISM_POS.y, 0.125), 0.015, 0.155, brass_mat, verts=28)
add_torus("PrismStemKnop", (PRISM_POS.x, PRISM_POS.y, 0.125), 0.024, 0.009, brass_mat, maj_seg=28, min_seg=10)
add_box("PrismCradlePlate", (PRISM_POS.x, PRISM_POS.y, 0.215), (0.145, 0.115, 0.014), brass_mat, bevel_w=0.003)

# Hero Beveled Equilateral Flint-Glass Triangular Prism (IOR=1.62, pure crystal glass, centered at Z=0.28!)
hero_prism = build_equilateral_prism("FlintGlassTriangularPrism", (PRISM_POS.x, PRISM_POS.y, 0.280), side=0.142, depth_y=0.116, mat=flint_glass_mat)

# Angled Dark Ebony / Obsidian Optical Projection Screen at X = -1.72 (tilted toward camera & prism!)
SCREEN_POS = Vector((PRISM_POS.x + 0.24, PRISM_POS.y + 0.01, 0.29))  # X = -1.72!
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(SCREEN_POS.x, SCREEN_POS.y, 0.0), rotation=(0.0, 0.0, math.radians(-26.0)))
screen_root = bpy.context.active_object
screen_root.name = "OpticalScreenRoot"

sc_base = add_cyl("OptScreenBase", (0.0, 0.0, 0.012), 0.052, 0.024, brass_mat, verts=32, bevel_w=0.003)
sc_base.parent = screen_root
sc_post = add_cyl("OptScreenPost", (0.0, 0.0, 0.085), 0.010, 0.125, brass_mat, verts=20)
sc_post.parent = screen_root
sc_frame = add_box("OptScreenBrassFrame", (0.0, 0.0, 0.295), (0.022, 0.126, 0.345), brass_mat, bevel_w=0.003)
sc_frame.parent = screen_root
sc_slate = add_box("OptScreenObsidian", (-0.006, 0.0, 0.295), (0.014, 0.114, 0.326), obsidian_mat, bevel_w=0.002)
sc_slate.parent = screen_root

# 7 Vivid Glowing Spectral Bands painted directly onto the front-left obsidian screen face!
spectral_bands = []
for b_i in range(7):
    # Physically accurate dispersion order: Red (0) highest (Z=0.415) -> Violet (6) lowest (Z=0.175)
    bz = lerp(0.415, 0.175, b_i / 6.0)
    band = add_box(f"SpectralBand_{b_i}", (-0.015, 0.0, bz), (0.005, 0.104, 0.032), rainbow_band_mats[b_i], bevel_w=0.001)
    band.parent = screen_root
    band.visible_shadow = False
    spectral_bands.append(band)

# Crisp Luminous White Sunbeam Ray entering prism's left face + internal dispersion cone + 7 Feathered Rainbow Beams!
SUN_START = Vector((PRISM_POS.x - 0.30, PRISM_POS.y, 0.50))
PRISM_ENTRY = Vector((PRISM_POS.x - 0.044, PRISM_POS.y, 0.300))
PRISM_EXIT = Vector((PRISM_POS.x + 0.042, PRISM_POS.y, 0.278))

white_sunbeam = add_cyl("WhiteSunbeamRay", (0, 0, 0), 0.015, 1.0, sunbeam_white_mat, verts=24)
white_sunbeam.visible_shadow = False
white_sunbeam.visible_diffuse = False
white_sunbeam.visible_glossy = False

internal_beam = add_cyl("PrismInternalRay", (0, 0, 0), 0.013, 1.0, sunbeam_white_mat, verts=20)
internal_beam.visible_shadow = False
internal_beam.visible_diffuse = False
internal_beam.visible_glossy = False

rainbow_beams = []
rainbow_targets = []
for b_i in range(7):
    bz = lerp(0.415, 0.175, b_i / 6.0)
    # Screen face hit position in world space (slightly inset toward prism face of screen)
    hit_pt = Vector((SCREEN_POS.x - 0.012, SCREEN_POS.y, bz))
    rainbow_targets.append(hit_pt)
    rb = add_cyl(f"RainbowBeam_{b_i}", (0, 0, 0), 0.0135, 1.0, rainbow_beam_mats[b_i], verts=20)
    rb.visible_shadow = False
    rb.visible_diffuse = False
    rb.visible_glossy = False
    rainbow_beams.append(rb)


# ============================================================================
# 10. STATION C (CRADLE_POS = +1.44, -0.08, 0.0) — BEAT 4 BRASS NEWTON'S CRADLE!
# ============================================================================
CRADLE_POS = Vector((1.44, -0.08, 0.0))
CRADLE_RAIL_Z = 0.56
CRADLE_BALL_R = 0.036
CRADLE_WIRE_L = 0.260  # Rest ball center at Z = 0.56 - 0.26 = 0.30!
CRADLE_HALF_Y = 0.085

# Stepped Dark-Oak & Brass Dual-Arch Cradle Frame (Z = 0.0..0.58)
add_box("CradleBaseLower", (CRADLE_POS.x, CRADLE_POS.y, 0.016), (0.44, 0.24, 0.032), wainscot_oak_mat, bevel_w=0.006)
add_box("CradleBaseUpper", (CRADLE_POS.x, CRADLE_POS.y, 0.040), (0.39, 0.20, 0.018), english_oak_mat, bevel_w=0.004)
add_box("CradleBrassTrim", (CRADLE_POS.x, CRADLE_POS.y - 0.121, 0.016), (0.442, 0.008, 0.012), brass_mat, bevel_w=0.002)

# Twin Front & Rear Curved Brass Dual-Arch Rails + Top Crossbars!
for sgn_y in (-1.0, 1.0):
    ry = CRADLE_POS.y + sgn_y * CRADLE_HALF_Y
    build_curve_tube(
        f"CradleArchRail_{int(sgn_y)}",
        [
            (CRADLE_POS.x - 0.195, ry, 0.044, 0.011),
            (CRADLE_POS.x - 0.195, ry, 0.500, 0.010),
            (CRADLE_POS.x - 0.165, ry, CRADLE_RAIL_Z, 0.010),
            (CRADLE_POS.x + 0.165, ry, CRADLE_RAIL_Z, 0.010),
            (CRADLE_POS.x + 0.195, ry, 0.500, 0.010),
            (CRADLE_POS.x + 0.195, ry, 0.044, 0.011),
        ],
        brass_mat,
        bevel_res=6,
    )
    for sgn_x in (-1.0, 1.0):
        add_cyl(f"CradleFootCollar_{int(sgn_y)}_{int(sgn_x)}", (CRADLE_POS.x + sgn_x * 0.195, ry, 0.056), 0.018, 0.018, brass_mat, verts=20)
for sgn_x in (-1.0, 1.0):
    add_cyl(
        f"CradleCrossBar_{int(sgn_x)}",
        (CRADLE_POS.x + sgn_x * 0.165, CRADLE_POS.y, CRADLE_RAIL_Z),
        0.008,
        CRADLE_HALF_Y * 2.0,
        brass_mat,
        rot=(math.pi * 0.5, 0.0, 0.0),
        verts=20,
    )

# 5 Mirror-Polished Chrome Steel Spheres (r=0.036m) suspended on Twin V-Angled Fine Brass Wires!
cradle_pivots = []
for b_i in range(5):
    bx_off = (b_i - 2.0) * (CRADLE_BALL_R * 2.005)
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(CRADLE_POS.x + bx_off, CRADLE_POS.y, CRADLE_RAIL_Z))
    piv = bpy.context.active_object
    piv.name = f"CradlePivot_{b_i}"
    piv.rotation_mode = "XYZ"

    # Mirror-chrome steel sphere
    ball = add_sphere(f"CradleChromeSphere_{b_i}", (0.0, 0.0, -CRADLE_WIRE_L), CRADLE_BALL_R, mirror_chrome_mat, segs=36, rings=24)
    ball.parent = piv
    eyelet = add_torus(f"CradleEyelet_{b_i}", (0.0, 0.0, -CRADLE_WIRE_L + CRADLE_BALL_R + 0.003), 0.006, 0.002, brass_mat, maj_seg=16, min_seg=6)
    eyelet.parent = piv

    # Twin fine V-wires connecting front & rear top brass rails cleanly down to the sphere eyelet!
    for w_sgn in (-1.0, 1.0):
        p_top = Vector((0.0, w_sgn * CRADLE_HALF_Y, 0.0))
        p_bot = Vector((0.0, 0.0, -CRADLE_WIRE_L + CRADLE_BALL_R + 0.003))
        wire = add_cyl(f"CradleWire_{b_i}_{int(w_sgn)}", (0, 0, 0), 0.0018, 1.0, brass_mat, verts=10)
        orient_beam(wire, p_top, p_bot, 1.0)
        wire.parent = piv

    cradle_pivots.append(piv)

# 14 Glowing Metallic-Gold Kinetic Sparks + Shock Ring erupting on Beat 4 Clack (t = 17.45s)!
clack_sparks = []
for s_i in range(14):
    sp = add_sphere(f"CradleClackSpark_{s_i:02d}", (CRADLE_POS.x + 0.144, CRADLE_POS.y, 0.30), 0.011, gold_glow_mat if s_i % 2 == 0 else cyan_glow_mat, segs=10, rings=8)
    sp.visible_shadow = False
    clack_sparks.append(sp)

clack_shock_ring = add_torus(
    "CradleShockRing",
    (CRADLE_POS.x + 0.11, CRADLE_POS.y, 0.30),
    0.048,
    0.0045,
    gold_glow_mat,
    rot=(0.0, math.pi * 0.5, 0.0),
    maj_seg=48,
    min_seg=10,
)
clack_shock_ring.visible_shadow = False


# ============================================================================
# 11. STATION D (ORBIT_POS = +0.28, +0.02, +0.34) — BEATS 5 & 6 EARTH-MOON GRAVITY WELL,
#     BRASS REFLECTING TELESCOPE, MODERN ORBITAL ROCKET & CINEMA GOLD TITLE!
# ============================================================================
ORBIT_POS = Vector((0.28, 0.02, 0.34))

# Entire station parented to orbit_station_root (hidden in Beats 1-4, revealed Beat 5 at X=+0.28, centered Beat 6 at X=+0.04!)
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(ORBIT_POS.x, ORBIT_POS.y, 0.0))
orbit_station_root = bpy.context.active_object
orbit_station_root.name = "OrbitStationRoot"

# Newton's Iconic Brass Newtonian Reflecting Telescope resting cleanly on Mahogany Desk Cradle (Z = 0.0..0.22)!
tel_base = add_box("TelescopeCradleBase", (0.0, 0.0, 0.015), (0.32, 0.16, 0.030), wainscot_oak_mat, bevel_w=0.005)
tel_base.parent = orbit_station_root
tel_yoke_l = add_box("TelescopeYokeL", (-0.09, 0.0, 0.068), (0.022, 0.042, 0.092), english_oak_mat, bevel_w=0.003)
tel_yoke_l.parent = orbit_station_root
tel_yoke_r = add_box("TelescopeYokeR", (0.09, 0.0, 0.068), (0.022, 0.042, 0.092), english_oak_mat, bevel_w=0.003)
tel_yoke_r.parent = orbit_station_root

bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 0.135), rotation=(0.0, math.radians(-16.0), math.radians(10.0)))
tel_tube_piv = bpy.context.active_object
tel_tube_piv.name = "NewtonReflectingTelescopePivot"
tel_tube_piv.parent = orbit_station_root

t_barrel = add_cyl("TelescopeBrassTube", (0.0, 0.0, 0.0), 0.046, 0.34, brass_mat, rot=(0.0, math.pi * 0.5, 0.0), verts=36, bevel_w=0.003)
t_barrel.parent = tel_tube_piv
for tr_x in (-0.155, -0.06, 0.06, 0.158):
    tr_ring = add_torus(f"TelescopeRing_{tr_x:+.2f}", (tr_x, 0.0, 0.0), 0.048, 0.0055, bronze_mat, rot=(0.0, math.pi * 0.5, 0.0), maj_seg=36, min_seg=10)
    tr_ring.parent = tel_tube_piv
t_mirror = add_cyl("TelescopeSpeculumMirror", (-0.162, 0.0, 0.0), 0.039, 0.012, speculum_mirror_mat, rot=(0.0, math.pi * 0.5, 0.0), verts=32)
t_mirror.parent = tel_tube_piv
t_open = add_cyl("TelescopeFrontBaffle", (0.164, 0.0, 0.0), 0.040, 0.014, obsidian_mat, rot=(0.0, math.pi * 0.5, 0.0), verts=32)
t_open.parent = tel_tube_piv
# Iconic Newtonian side-mounted eyepiece barrel!
t_eye = add_cyl("TelescopeSideEyepiece", (0.105, -0.056, 0.012), 0.014, 0.055, brass_mat, rot=(math.pi * 0.5, 0.0, 0.0), verts=24, bevel_w=0.002)
t_eye.parent = tel_tube_piv
t_lens = add_sphere("TelescopeEyeGlass", (0.105, -0.084, 0.012), 0.011, flint_glass_mat, segs=18, rings=12)
t_lens.scale = (1.0, 0.45, 1.0)
t_lens.parent = tel_tube_piv

# Mesmerizing Luminous Sapphire-Gold 3D Curved Space-Time Gravity Well hovering above Telescope (Z = 0.28..0.64)!
WELL_CENTER_Z = 0.46
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, WELL_CENTER_Z), rotation=(math.radians(15.0), math.radians(-6.0), 0.0))
gravity_well_group = bpy.context.active_object
gravity_well_group.name = "CurvedSpacetimeGravityWell"
gravity_well_group.parent = orbit_station_root


def gravity_funnel_z(r: float) -> float:
    """Authentic gravitational potential funnel curve: flat outer rim plunging smoothly beneath Earth."""
    return 0.032 - 0.022 / (r + 0.068)


# 7 Concentric Glowing Gravity Well Funnel Rings (outer electric sapphire -> inner molten gold!)
well_radii = [0.065, 0.095, 0.125, 0.155, 0.185, 0.215, 0.245]
for wi, wr in enumerate(well_radii):
    wz = gravity_funnel_z(wr)
    wmat = gold_glow_mat if wi in (0, 1, 6) else cyan_glow_mat
    w_ring = add_torus(f"GravityWellRing_{wi}", (0.0, 0.0, wz), wr, 0.0028 if wi < 6 else 0.0042, wmat, maj_seg=64, min_seg=10)
    w_ring.visible_shadow = False
    w_ring.parent = gravity_well_group

# 12 Curved Space-Time Meridian Grid Ribs plunging smoothly down the gravitational potential funnel!
for mi in range(12):
    ang = (mi / 12.0) * math.pi * 2.0
    ca, sa = math.cos(ang), math.sin(ang)
    pts = []
    for si in range(11):
        rr = lerp(0.055, 0.245, si / 10.0)
        pts.append((ca * rr, sa * rr, gravity_funnel_z(rr), 0.0020))
    rib = build_curve_tube(f"GravityWellMeridian_{mi:02d}", pts, cyan_glow_mat if mi % 2 == 0 else gold_glow_mat, bevel_res=3)
    rib.visible_shadow = False
    rib.parent = gravity_well_group

# High-Detail Procedural 3D Earth + Glowing Azure Atmospheric Fresnel Halo Shell!
earth_obj = add_sphere("ProceduralEarth", (0.0, 0.0, 0.012), 0.084, earth_planet_mat, segs=48, rings=32)
earth_obj.parent = gravity_well_group
earth_halo = add_sphere("EarthAtmoHalo", (0.0, 0.0, 0.012), 0.091, earth_atmo_mat, segs=36, rings=24)
earth_halo.visible_shadow = False
earth_halo.parent = gravity_well_group

# Cratered 3D Moon Orbiting Rapidly Along Outer Golden Ring + Pulsing Golden Gravity Force Tether!
MOON_ORBIT_R = 0.238
MOON_ORBIT_Z = gravity_funnel_z(MOON_ORBIT_R) + 0.018
moon_obj = add_sphere("OrbitingMoon3D", (MOON_ORBIT_R, 0.0, MOON_ORBIT_Z), 0.027, cratered_moon_mat, segs=32, rings=20)
moon_obj.parent = gravity_well_group

gravity_tether = add_cyl("EarthMoonGravityBeam", (0, 0, 0), 0.0042, 1.0, gold_glow_mat, verts=16)
gravity_tether.visible_shadow = False
gravity_tether.parent = gravity_well_group

# BEAT 6 MODERN LEGACY: Sleek Metallic 3D Orbital Rocket Flying Along Newton's Tilted Golden Orbit (Z = 0.96)!
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.04, 0.06, 0.96), rotation=(math.radians(16.0), math.radians(-10.0), 0.0))
rocket_orbit_group = bpy.context.active_object
rocket_orbit_group.name = "ModernRocketOrbitGroup"

rocket_orbit_ring = add_torus("RocketGoldenOrbitRing", (0.0, 0.0, 0.0), 0.255, 0.0040, gold_glow_mat, maj_seg=72, min_seg=10)
rocket_orbit_ring.visible_shadow = False
rocket_orbit_ring.parent = rocket_orbit_group

bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.255, 0.0, 0.0))
rocket_ship = bpy.context.active_object
rocket_ship.name = "ModernOrbitalRocket"
rocket_ship.parent = rocket_orbit_group

# Rocket oriented cleanly along local +Y (tangent to circular orbit at angle 0!)
r_body = add_cyl("RocketFuselage", (0.0, 0.0, 0.0), 0.022, 0.125, rocket_white_mat, rot=(math.pi * 0.5, 0.0, 0.0), verts=28)
r_body.parent = rocket_ship
r_band = add_cyl("RocketGoldStageBand", (0.0, 0.022, 0.0), 0.023, 0.014, brass_mat, rot=(math.pi * 0.5, 0.0, 0.0), verts=28)
r_band.parent = rocket_ship
r_nose = add_cone("RocketNosecone", (0.0, 0.088, 0.0), 0.022, 0.001, 0.056, rocket_dark_mat, rot=(-math.pi * 0.5, 0.0, 0.0), verts=28)
r_nose.parent = rocket_ship
r_nozzle = add_cone("RocketEngineNozzle", (0.0, -0.072, 0.0), 0.016, 0.010, 0.022, rocket_dark_mat, rot=(-math.pi * 0.5, 0.0, 0.0), verts=20)
r_nozzle.parent = rocket_ship
for fin_i in range(3):
    fa = math.radians(fin_i * 120.0)
    fin = add_box(
        f"RocketFin_{fin_i}",
        (math.cos(fa) * 0.026, -0.044, math.sin(fa) * 0.026),
        (0.026, 0.042, 0.004),
        rocket_dark_mat,
        rot=(0.0, -fa, 0.0),
        bevel_w=0.001,
    )
    fin.parent = rocket_ship
r_plume = add_cone("RocketPlasmaThruster", (0.0, -0.128, 0.0), 0.001, 0.018, 0.095, rocket_plume_mat, rot=(-math.pi * 0.5, 0.0, 0.0), verts=20)
r_plume.visible_shadow = False
r_plume.parent = rocket_ship
r_core = add_cone("RocketCoreFlame", (0.0, -0.100, 0.0), 0.001, 0.010, 0.045, rocket_core_mat, rot=(-math.pi * 0.5, 0.0, 0.0), verts=16)
r_core.visible_shadow = False
r_core.parent = rocket_ship

# Hero 2-Line Self-Luminous Cinema Gold 3D Title centered cleanly inside the tall upper Gothic arch (Beats 5-6!)
bpy.ops.object.text_add(location=(0.0, 0.56, 1.54), rotation=(math.pi * 0.5, 0.0, 0.0))
title_txt = bpy.context.active_object
title_txt.name = "NewtonTitle3D"
if HERO_FONT is not None:
    title_txt.data.font = HERO_FONT
title_txt.data.body = "ISAAC NEWTON\nF = G m1 m2 / r²"
title_txt.data.align_x = "CENTER"
title_txt.data.align_y = "CENTER"
title_txt.data.size = 0.072
title_txt.data.space_character = 1.13
title_txt.data.space_line = 1.28
title_txt.data.extrude = 0.0060
title_txt.data.bevel_depth = 0.0006
title_txt.data.materials.append(title_gold_mat)


# ============================================================================
# 11B. REALISTIC MPFB2 V6 4K PHOTO-SCANNED HUMAN CAST (NEWTON & HALLEY!)
# ============================================================================
NEWTON_ACTOR = None
HALLEY_ACTOR = None
try:
    NEWTON_ACTOR = create_realistic_scientist(
        name="IsaacNewton",
        floor_z=FLOOR_Z,
        start_xy=(-1.22, 0.44),
        yaw_deg=24.0,
        cfg={
            "age": 0.34,
            "muscle": 0.52,
            "weight": 0.45,
            "height": 0.56,
            "skin_assets": ["young_caucasian_male.mhmat"],
            "iris_major_rgb": (0.14, 0.36, 0.56),
            "iris_minor_rgb": (0.06, 0.14, 0.26),
            "eyebrow_assets": ["eyebrow001.mhclo"],
            "hair_assets": ["punkduck_alpha7_curly2.mhclo", "punkduck_alpha7_long.mhclo"],
            "hair_value_boost": 1.80,
            "hair_tint_rgb": (0.58, 0.34, 0.16),
            "hair_tint_fac": 0.65,
            "brow_tint_fac": 0.48,
            "silver_hair": False,
            "suit_assets": [
                "toigo_male_suit_3.mhclo",
                "toigo_male_suit_tie_and_jacket.mhclo",
            ],
            "suit_tint_rgb": (0.30, 0.42, 0.72),  # Deep Royal Cambridge midnight-navy suit with 2K normal map!
            "suit_tint_fac": 0.35,
            "suit_value_boost": 1.15,
            "shoe_assets": ["shoes01.mhclo"],
        },
    )
    HALLEY_ACTOR = create_realistic_scientist(
        name="EdmondHalley",
        floor_z=FLOOR_Z,
        start_xy=(0.46, 0.46),
        yaw_deg=-28.0,
        cfg={
            "age": 0.52,
            "muscle": 0.55,
            "weight": 0.52,
            "height": 0.58,
            "skin_assets": ["middleage_caucasian_male.mhmat"],
            "iris_major_rgb": (0.30, 0.20, 0.09),
            "iris_minor_rgb": (0.12, 0.08, 0.04),
            "eyebrow_assets": ["eyebrow003.mhclo", "eyebrow002.mhclo"],
            "hair_assets": ["short02.mhclo", "bob02.mhclo"],
            "hair_tint_rgb": (0.55, 0.40, 0.28),
            "hair_tint_fac": 0.35,
            "brow_tint_fac": 0.45,
            "silver_hair": False,
            "suit_assets": [
                "toigo_suit_with_jacket_and_bowtie.mhclo",
                "male_elegantsuit01.mhclo",
            ],
            "suit_tint_rgb": (0.54, 0.16, 0.16),  # Rich Royal Burgundy academic jacket + crisp white pleated shirt & bowtie!
            "suit_tint_fac": 0.72,
            "suit_value_boost": 1.15,
            "shoe_assets": ["shoes04.mhclo", "shoes02.mhclo"],
        },
    )
except Exception as mpfb_err:
    print(f"[MPFB2 Warning] Skipping human cast ({mpfb_err}) — install MPFB2 & packs in Cell 2!")


# ============================================================================
# 12. V6 GOLDEN-HOUR REMBRANDT + LUNAR SAPPHIRE RIM LIGHTING & 9:16 CAMERA
# ============================================================================
# Warm Golden-Hour Rembrandt Key Area Light (amber-gold sun sculpting skin pores, cheekbones & stone relief!)
bpy.ops.object.light_add(type="AREA", location=(-2.05, -1.65, 1.78))
key_light = bpy.context.active_object
key_light.data.energy = 320.0
key_light.data.size = 1.30
key_light.data.color = (1.0, 0.79, 0.48)
key_light.rotation_euler = Euler((math.radians(50.0), 0.0, math.radians(-42.0)))
key_light.visible_camera = False

# Crisp Lunar Sapphire Rim Light (cool celestial cyan-blue backlit silhouette on hair strands & Gothic columns!)
bpy.ops.object.light_add(type="AREA", location=(1.48, 2.38, 2.10))
rim_light = bpy.context.active_object
rim_light.data.energy = 385.0
rim_light.data.size = 1.65
rim_light.data.color = (0.38, 0.78, 1.0)
rim_light.rotation_euler = Euler((math.radians(56.0), 0.0, math.radians(146.0)))
rim_light.visible_camera = False

# Soft Warm Conservatory Fill (controlled cinema contrast ratio so faces keep rich sculptural shadow depth!)
bpy.ops.object.light_add(type="AREA", location=(0.90, -1.68, 1.50))
fill_light = bpy.context.active_object
fill_light.data.energy = 72.0
fill_light.data.size = 2.25
fill_light.data.color = (0.95, 0.87, 0.78)
fill_light.rotation_euler = Euler((math.radians(66.0), 0.0, math.radians(24.0)))
fill_light.visible_camera = False

# Crisp Wet-Cornea Eye Catchlight & Glass Prism / Chrome Sphere Sparkle Accent
bpy.ops.object.light_add(type="POINT", location=(0.0, -0.84, 0.68))
catch_light = bpy.context.active_object
catch_light.data.energy = 24.0
catch_light.data.color = (0.90, 0.96, 1.0)
catch_light.visible_camera = False

bpy.ops.object.empty_add(location=(-0.96, 0.14, 0.45))
cam_target = bpy.context.active_object
cam_target.name = "CinemaCamTarget"

bpy.ops.object.camera_add(location=(-0.98, -1.48, 0.56))
cam = bpy.context.active_object
cam.name = "CinemaCamera"
scene.camera = cam
cam.data.sensor_fit = "HORIZONTAL"
cam.data.sensor_width = 36.0
cam.data.lens = 41.0
cam.data.dof.use_dof = True
cam.data.dof.focus_object = cam_target
cam.data.dof.aperture_fstop = 2.8
cam.rotation_mode = "QUATERNION"


# ============================================================================
# 13. 6-BEAT ZERO-OVERLAP CINEMA CAMERA GRAMMAR (1:1 SYNC WITH verify_framing.py!)
# ============================================================================
def compute_beat_raw(beat_idx: int, t: float):
    """Return (cam_pos, tgt_pos, lens_mm, fstop, dutch_deg) — synchronized 1:1 with src/verify_framing.py!"""
    if beat_idx == 0:
        # BEAT 1 (0..4.5s): Newton (Left, X=-1.22, facing RIGHT +24°) + Falling Woolsthorpe Apple & g Arrow (Right, X=-0.74)!
        u = clamp01(t / 4.5)
        pos = bezier_quad(Vector((-0.98, -1.48, 0.56)), Vector((-0.93, -1.38, 0.54)), Vector((-0.89, -1.32, 0.52)), u)
        tgt = vlerp(Vector((-0.96, 0.14, 0.45)), Vector((-0.94, 0.14, 0.42)), u)
        # Subtle spring thud shake when apple hits velvet pad at t=1.45s
        thud = math.sin((t - 1.45) * 44.0) * math.exp(-(t - 1.45) * 8.0) * 0.010 if t >= 1.45 else 0.0
        pos.z += thud
        return pos, tgt, lerp(41.0, 43.0, u), 2.8, 0.0
    if beat_idx == 1:
        # BEAT 2 (4.5..9.5s): Balanced Face-to-Face 2-Shot (BOTH Newton -0.46 & Halley +0.46 100% INSIDE frame!)
        pan_u = ease_out_quint(clamp01((t - 6.95) / 1.1))
        pos = vlerp(Vector((0.06, -1.72, 0.56)), Vector((-0.04, -1.64, 0.56)), pan_u)
        tgt = vlerp(Vector((0.03, 0.18, 0.46)), Vector((-0.01, 0.20, 0.46)), pan_u)
        return pos, tgt, lerp(36.0, 35.5, pan_u), 3.0, 0.0
    if beat_idx == 2:
        # BEAT 3 (9.5..15.5s): Crystal Glass Prism & 7-Spectral Rainbow (Left, X=-1.96..-1.72) + Newton (Right, X=-1.42)!
        u = clamp01((t - 9.5) / 6.0)
        pos = bezier_quad(Vector((-1.74, -1.56, 0.54)), Vector((-1.69, -1.46, 0.56)), Vector((-1.65, -1.40, 0.58)), u)
        tgt = vlerp(Vector((-1.71, 0.14, 0.44)), Vector((-1.69, 0.14, 0.48)), smoothstep(10.6, 13.0, t))
        return pos, tgt, lerp(38.5, 40.5, u), 2.8, 0.0
    if beat_idx == 3:
        # BEAT 4 (15.5..20.5s): 3.8° Dutch-Tilt Halley (Left, X=+0.94) + Brass Newton's Cradle Clack (Right, X=+1.44)!
        u = clamp01((t - 15.5) / 5.0)
        pos = bezier_quad(Vector((1.22, -1.62, 0.54)), Vector((1.26, -1.54, 0.55)), Vector((1.28, -1.60, 0.56)), u)
        tgt = Vector((1.24, 0.14, 0.46))
        shake = math.sin((t - 17.45) * 54.0) * math.exp(-(t - 17.45) * 6.5) * 0.024 if t >= 17.45 else 0.0
        pos.x += shake
        pos.z += shake * 0.45
        return pos, tgt, lerp(35.5, 37.0, u), 2.8, 3.8 * math.sin(u * math.pi)
    if beat_idx == 4:
        # BEAT 5 (20.5..25.5s): Newton (Left, X=-0.30) + Glowing Earth-Moon Gravity Well & Telescope (Right, X=+0.28)!
        u = clamp01((t - 20.5) / 5.0)
        pos = bezier_quad(Vector((-0.04, -1.50, 0.56)), Vector((0.02, -1.44, 0.60)), Vector((0.06, -1.48, 0.64)), u)
        tgt = vlerp(Vector((0.01, 0.18, 0.46)), Vector((0.02, 0.20, 0.48)), u)
        return pos, tgt, lerp(37.0, 38.5, u), 3.0, 0.0
    # BEAT 6 (25.5..30.0s): Grand Cast Finale — Newton (-0.54) & Halley (+0.56) flanking Centered Well, Rocket & Title!
    u = clamp01((t - 25.5) / 4.5)
    pos = bezier_quad(Vector((0.04, -1.96, 0.72)), Vector((0.03, -2.12, 0.70)), Vector((0.02, -2.24, 0.68)), u)
    tgt = vlerp(Vector((0.02, 0.26, 0.56)), Vector((0.02, 0.24, 0.54)), u)
    return pos, tgt, lerp(34.0, 32.0, u), 3.6, 0.0


BEAT_BOUNDS = [0.0, 4.5, 9.5, 15.5, 20.5, 25.5, 30.0]


def compute_camera_state(t: float):
    """Smooth single-take cinema camera with 0.42s easeOutQuint whip-pans synced to exact whip_whoosh SFX!"""
    sway_x = math.sin(t * 1.15) * 0.009 + math.sin(t * 2.40) * 0.003
    sway_z = math.cos(t * 0.92) * 0.007

    b_idx = 5
    for i in range(6):
        if BEAT_BOUNDS[i] <= t < BEAT_BOUNDS[i + 1]:
            b_idx = i
            break

    pos, tgt, lens, fstop, dutch = compute_beat_raw(b_idx, t)

    whip_dur = 0.42
    dt_in = t - BEAT_BOUNDS[b_idx]
    if b_idx > 0 and dt_in < whip_dur:
        w = ease_out_quint(dt_in / whip_dur)
        p0, t0, l0, f0, d0 = compute_beat_raw(b_idx - 1, BEAT_BOUNDS[b_idx])
        pos = vlerp(p0, pos, w)
        tgt = vlerp(t0, tgt, w)
        lens = lerp(l0, lens, w)
        fstop = lerp(f0, fstop, w)
        dutch = lerp(d0, dutch, w)

    pos.x += sway_x
    pos.z += sway_z
    return pos, tgt, lens, fstop, dutch


# ============================================================================
# 14. KEYFRAME ALL 720 FRAMES DETERMINISTICALLY (EXACT 30.000s AUDIO SYNC!)
# ============================================================================
for f in range(1, TOTAL_FRAMES + 1):
    t = (f - 1) / FPS

    # 1. Exact Camera Position, Look-At Quaternion + True Dutch Roll, Lens & DOF Focus
    c_pos, c_tgt, c_lens, c_fstop, c_dutch = compute_camera_state(t)
    cam.location = c_pos
    cam_target.location = c_tgt
    cam.rotation_quaternion = look_at_quat(c_pos, c_tgt, c_dutch)
    cam.data.lens = c_lens
    cam.data.dof.aperture_fstop = c_fstop
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_quaternion", frame=f)
    cam_target.keyframe_insert("location", frame=f)
    cam.data.keyframe_insert("lens", frame=f)
    cam.data.keyframe_insert(data_path="dof.aperture_fstop", frame=f)

    # 2. Atmospheric Candelabra Flicker + 42 Floating Golden Dust Motes Drift!
    for idx, fl in enumerate(candle_flames):
        flick = 1.0 + 0.13 * math.sin(t * 21.0 + idx * 1.7) + 0.06 * math.sin(t * 35.0 + idx)
        fl.scale = (1.0, 1.0, flick)
        fl.keyframe_insert("scale", frame=f)

    for dm, p0, ph in dust_motes:
        dm.location = Vector((
            p0.x + math.sin(t * 0.55 + ph) * 0.055,
            p0.y + math.cos(t * 0.42 + ph * 1.3) * 0.040,
            p0.z + math.sin(t * 0.68 + ph * 0.8) * 0.048,
        ))
        dm.keyframe_insert("location", frame=f)

    # 3. BEAT 1: Woolsthorpe Apple Gravity Fall (t=0.65s -> 1.45s exact 1/2 g t^2!) + Velvet Bounce + Golden 'g' Vector!
    bough_vis = max(0.0001, 1.0 - smoothstep(4.30, 4.65, t))
    bough_sway = math.sin(t * 2.2) * 0.015 if t < 0.65 else math.sin((t - 0.65) * 9.0) * math.exp(-(t - 0.65) * 3.0) * 0.045
    apple_bough_root.scale = Vector((bough_vis, bough_vis, bough_vis))
    apple_bough_root.rotation_euler = Euler((0.0, bough_sway, 0.0))
    apple_bough_root.keyframe_insert("scale", frame=f)
    apple_bough_root.keyframe_insert("rotation_euler", frame=f)

    z_hang = 0.690
    z_pad = 0.064
    if t < 0.65:
        # Gentle orchard breeze sway before twig snap at t=0.65s
        az = z_hang + math.sin(t * 3.2) * 0.004
        ax = APPLE_POS.x + math.sin(t * 2.2) * 0.006
        arot = Euler((0.0, math.sin(t * 2.2) * 0.08, t * 0.15))
    elif t < 1.45:
        # Exact constant-acceleration gravitational free fall (1/2 g t^2) from t=0.65s to t=1.45s!
        u_f = (t - 0.65) / 0.80
        az = z_hang - (z_hang - z_pad) * (u_f * u_f)
        ax = APPLE_POS.x
        arot = Euler((u_f * 0.45, -u_f * 0.30, u_f * 0.70))
    else:
        dt_b = t - 1.45
        if dt_b < 0.34:
            hb = math.sin((dt_b / 0.34) * math.pi) * 0.062
        elif dt_b < 0.54:
            hb = math.sin(((dt_b - 0.34) / 0.20) * math.pi) * 0.018
        else:
            hb = 0.0
        az = z_pad + hb
        ax = APPLE_POS.x + min(0.018, dt_b * 0.032)
        arot = Euler((0.45 + min(0.20, dt_b * 0.35), -0.30, 0.70 + min(0.35, dt_b * 0.50)))
    hero_apple.location = Vector((ax, APPLE_POS.y, az))
    hero_apple.rotation_euler = arot
    hero_apple.keyframe_insert("location", frame=f)
    hero_apple.keyframe_insert("rotation_euler", frame=f)

    # Golden Downward Gravity Vector ('g') & Expanding Impact Ripple Ring!
    g_vis = smoothstep(1.50, 1.85, t) * (1.0 - smoothstep(4.25, 4.55, t))
    g_sc = max(0.0001, g_vis)
    gravity_arrow_root.scale = Vector((g_sc, g_sc, g_sc))
    gravity_arrow_root.location = Vector((APPLE_POS.x + 0.11, APPLE_POS.y, 0.34 - math.sin(t * 4.2) * 0.014))
    gravity_arrow_root.keyframe_insert("scale", frame=f)
    gravity_arrow_root.keyframe_insert("location", frame=f)

    if 1.45 <= t <= 3.10:
        rp_u = (t - 1.45) / 1.65
        rp_rad = 0.55 + rp_u * 2.15
        rp_z = max(0.0001, 1.0 - rp_u)
        impact_ripple.scale = Vector((rp_rad, rp_rad, rp_z))
    else:
        impact_ripple.scale = Vector((0.0001, 0.0001, 0.0001))
    impact_ripple.keyframe_insert("scale", frame=f)

    # 4. BEAT 2: Floating 3D 'F = m a' (4.5-6.95s) -> Morphs into Radiant 'F = G m1 m2 / r²' (6.95-10.2s)!
    fma_vis = smoothstep(4.55, 5.05, t) * (1.0 - smoothstep(6.80, 7.05, t))
    fma_sc = max(0.0001, fma_vis)
    formula_fma_txt.scale = Vector((fma_sc, fma_sc, fma_sc))
    formula_fma_txt.location = Vector((PRINCIPIA_POS.x, PRINCIPIA_POS.y - 0.04, 0.26 + math.sin(t * 2.5) * 0.010))
    formula_fma_txt.keyframe_insert("scale", frame=f)
    formula_fma_txt.keyframe_insert("location", frame=f)

    fgrav_vis = smoothstep(6.95, 7.35, t) * (1.0 - smoothstep(9.85, 10.25, t))
    fgrav_sc = max(0.0001, fgrav_vis)
    formula_grav_txt.scale = Vector((fgrav_sc, fgrav_sc, fgrav_sc))
    formula_grav_txt.location = Vector((PRINCIPIA_POS.x, PRINCIPIA_POS.y - 0.04, 0.30 + math.sin(t * 2.3) * 0.010))
    formula_grav_txt.keyframe_insert("scale", frame=f)
    formula_grav_txt.keyframe_insert("location", frame=f)

    # 5. BEAT 3: Crystal Glass Prism White Sunbeam -> 7 Distinct Glowing Spectral Rainbow Dispersion Beams!
    sun_u = smoothstep(9.85, 10.55, t) * (1.0 - smoothstep(15.35, 15.80, t))
    if sun_u > 0.002:
        p_sun_end = vlerp(SUN_START, PRISM_ENTRY, smoothstep(9.85, 10.45, t))
        orient_beam(white_sunbeam, SUN_START, p_sun_end, radius_scale=max(0.001, sun_u))
        int_u = smoothstep(10.35, 10.65, t) * (1.0 - smoothstep(15.35, 15.80, t))
        orient_beam(internal_beam, PRISM_ENTRY, PRISM_EXIT, radius_scale=max(0.001, int_u))
    else:
        white_sunbeam.scale = Vector((0.0001, 0.0001, 0.0001))
        internal_beam.scale = Vector((0.0001, 0.0001, 0.0001))
    for obj_ray in (white_sunbeam, internal_beam):
        obj_ray.keyframe_insert("location", frame=f)
        obj_ray.keyframe_insert("scale", frame=f)
        obj_ray.keyframe_insert("rotation_quaternion", frame=f)

    for b_i, (rb, b_hit, sband) in enumerate(zip(rainbow_beams, rainbow_targets, spectral_bands)):
        # Bloom smoothly & sequentially from Red (10.60s) -> Violet (11.80s) right on prism_rainbow_bloom SFX!
        t_start = 10.60 + b_i * 0.12
        t_end = t_start + 0.48
        ext_u = smoothstep(t_start, t_end, t)
        fade_out = 1.0 - smoothstep(15.35, 15.80, t)
        vis_b = ext_u * fade_out
        shimmer = 1.0 + 0.08 * math.sin(t * 7.5 + b_i * 0.9)
        if vis_b > 0.002:
            cur_end = vlerp(PRISM_EXIT, b_hit, ext_u)
            orient_beam(rb, PRISM_EXIT, cur_end, radius_scale=max(0.001, vis_b * shimmer))
        else:
            rb.scale = Vector((0.0001, 0.0001, 0.0001))
        rb.keyframe_insert("location", frame=f)
        rb.keyframe_insert("scale", frame=f)
        rb.keyframe_insert("rotation_quaternion", frame=f)

        band_sc = max(0.0001, smoothstep(t_end - 0.10, t_end + 0.15, t) * fade_out)
        sband.scale = Vector((1.0, band_sc, band_sc))
        sband.keyframe_insert("scale", frame=f)

    # 6. BEAT 4: Brass Newton's Cradle Exact Pendulum Action-Reaction Clack Physics (t = 17.45s, 18.25s, 19.10s)!
    # Note: local ball is at (0, 0, -L) -> +ry swings LEFT (-X, toward Halley), -ry swings RIGHT (+X)!
    ang_b0 = 0.0
    ang_b4 = 0.0
    if 15.70 <= t < 17.10:
        # Pull Sphere 0 smoothly back to -48° (ry = +48° toward -X)
        ang_b0 = math.radians(48.0) * smoothstep(15.70, 17.05, t)
    elif 17.10 <= t < 17.45:
        # Gravity-accelerated drop into Sphere 1, CLACKING at exact t=17.45s!
        u_d = (t - 17.10) / 0.35
        ang_b0 = math.radians(48.0) * math.cos(u_d * math.pi * 0.5)
    elif 17.45 <= t < 18.25:
        # Instant momentum transfer launches Sphere 4 sky-high (+62° toward +X -> ry = -62°)!
        u_s = (t - 17.45) / 0.80
        ang_b4 = -math.radians(62.0) * math.sin(u_s * math.pi)
    elif 18.25 <= t < 19.10:
        # Rebound CLACK at 18.25s sends Sphere 0 flying back toward Halley (-54° -> ry = +54°)!
        u_s = (t - 18.25) / 0.85
        ang_b0 = math.radians(54.0) * math.sin(u_s * math.pi)
    elif 19.10 <= t < 19.95:
        # Second rebound CLACK at 19.10s sends Sphere 4 swinging (+46° -> ry = -46°)!
        u_s = (t - 19.10) / 0.85
        ang_b4 = -math.radians(46.0) * math.sin(u_s * math.pi)
    elif 19.95 <= t < 20.60:
        u_s = (t - 19.95) / 0.65
        ang_b0 = math.radians(34.0) * math.sin(u_s * math.pi)

    # Subtle micro-shudder on center spheres 1, 2, 3 right on clack impacts!
    clack_jolt = 0.0
    for tc_imp in (17.45, 18.25, 19.10):
        if 0.0 <= (t - tc_imp) < 0.14:
            clack_jolt = math.sin((t - tc_imp) * 68.0) * math.exp(-(t - tc_imp) * 20.0) * math.radians(2.2)

    for b_i, piv in enumerate(cradle_pivots):
        if b_i == 0:
            ry = ang_b0
        elif b_i == 4:
            ry = ang_b4
        else:
            ry = clack_jolt * (1.0 if b_i % 2 == 0 else -1.0)
        piv.rotation_euler = Euler((0.0, ry, 0.0))
        piv.keyframe_insert("rotation_euler", frame=f)

    # 14 Metallic-Gold Kinetic Sparks + Glowing Shock Ring Burst at exact t=17.45s CLACK!
    dt_ck = t - 17.45
    for s_i, sp in enumerate(clack_sparks):
        if 0.0 <= dt_ck < 1.35:
            u_sp = dt_ck / 1.35
            ang = (s_i / 14.0) * math.pi * 2.0
            rad = 0.03 + u_sp * (0.30 + (s_i % 4) * 0.07)
            sx = CRADLE_POS.x + 0.12 + math.cos(ang) * rad * 0.95
            sy = CRADLE_POS.y + math.sin(ang * 1.4) * rad * 0.55
            sz = 0.30 + u_sp * (0.32 - u_sp * 0.56)
            sc = max(0.0001, (1.0 - u_sp) * 0.95)
            sp.location = Vector((sx, sy, max(0.04, sz)))
            sp.scale = Vector((sc, sc, sc))
        else:
            sp.scale = Vector((0.0001, 0.0001, 0.0001))
        sp.keyframe_insert("location", frame=f)
        sp.keyframe_insert("scale", frame=f)

    if 0.0 <= dt_ck < 0.85:
        u_sr = dt_ck / 0.85
        sr_rad = 0.65 + u_sr * 2.60
        sr_th = max(0.0001, 1.0 - u_sr)
        clack_shock_ring.scale = Vector((sr_rad, sr_rad, sr_th))
    else:
        clack_shock_ring.scale = Vector((0.0001, 0.0001, 0.0001))
    clack_shock_ring.keyframe_insert("scale", frame=f)

    # 7. BEAT 5 & 6: Glowing Earth-Moon Curved Gravity Well + Brass Telescope (X=+0.28 -> X=+0.04)!
    orb_vis = max(0.0001, smoothstep(20.10, 20.65, t))
    orb_center_u = smoothstep(25.30, 26.05, t)
    orbit_station_root.scale = Vector((orb_vis, orb_vis, orb_vis))
    orbit_station_root.location = Vector((lerp(ORBIT_POS.x, 0.04, orb_center_u), lerp(ORBIT_POS.y, 0.04, orb_center_u), 0.0))
    orbit_station_root.keyframe_insert("scale", frame=f)
    orbit_station_root.keyframe_insert("location", frame=f)

    # Rotate procedural 3D Earth smoothly & orbit Cratered Moon rapidly with live golden gravity tether!
    earth_obj.rotation_euler = Euler((0.0, math.radians(23.5), t * 0.68))
    earth_obj.keyframe_insert("rotation_euler", frame=f)

    moon_ang = (t - 20.5) * 1.85 + 0.45
    moon_local = Vector((math.cos(moon_ang) * MOON_ORBIT_R, math.sin(moon_ang) * MOON_ORBIT_R, MOON_ORBIT_Z))
    moon_obj.location = moon_local
    moon_obj.rotation_euler = Euler((0.0, 0.0, moon_ang))
    moon_obj.keyframe_insert("location", frame=f)
    moon_obj.keyframe_insert("rotation_euler", frame=f)

    tether_pulse = 0.85 + 0.35 * math.sin(t * 7.5)
    orient_beam(gravity_tether, Vector((0.0, 0.0, 0.012)), moon_local, radius_scale=tether_pulse)
    gravity_tether.keyframe_insert("location", frame=f)
    gravity_tether.keyframe_insert("scale", frame=f)
    gravity_tether.keyframe_insert("rotation_quaternion", frame=f)

    # 8. BEAT 6 MODERN LEGACY: Sleek Metallic 3D Orbital Rocket + Self-Luminous Cinema Gold 3D Title!
    rkt_u = smoothstep(25.55, 26.35, t)
    rkt_sc = max(0.0001, rkt_u)
    rocket_orbit_group.scale = Vector((rkt_sc, rkt_sc, rkt_sc))
    rocket_orbit_group.keyframe_insert("scale", frame=f)

    rkt_ang = (t - 25.5) * 1.55 - 0.65
    rkt_r = 0.255
    rocket_ship.location = Vector((math.cos(rkt_ang) * rkt_r, math.sin(rkt_ang) * rkt_r, 0.0))
    rocket_ship.rotation_euler = Euler((0.0, math.radians(-14.0), rkt_ang))
    rocket_ship.keyframe_insert("location", frame=f)
    rocket_ship.keyframe_insert("rotation_euler", frame=f)

    plume_flick = 0.90 + 0.24 * math.sin(t * 42.0)
    r_plume.scale = Vector((1.0, 1.0, plume_flick))
    r_plume.keyframe_insert("scale", frame=f)

    tit_u = smoothstep(21.20, 22.35, t)
    tit_sc = max(0.0001, tit_u)
    title_txt.scale = Vector((tit_sc, tit_sc, tit_sc))
    title_txt.keyframe_insert("scale", frame=f)

    # 9. REALISTIC MPFB2 CAST ACTING — TRUE +Z FACING SIGNS (+yaw/+rz = LOOK RIGHT +X, -yaw/-rz = LOOK LEFT -X)!
    # All forearms elevated cleanly (la = -34..-38°, wrists +0.21..+0.34m ABOVE desk Z=0.0, ZERO clipping!)
    breath = math.sin(t * 2.25) * 1.2
    if NEWTON_ACTOR is not None:
        nrig = NEWTON_ACTOR["rig"]
        n_lip = sample_lip_sync("newton", t)
        n_base_z = NEWTON_ACTOR["base_z"]

        if t < 4.5:
            # Beat 1: Newton LEFT (-1.22, 0.44), yaw=+24°, head_rz=+14° (faces RIGHT at Woolsthorpe Apple -0.74!)
            nx, ny, nyaw = -1.22, 0.44, 24.0
            fall_track = smoothstep(0.65, 1.45, t)
            wonder_up = smoothstep(2.40, 3.30, t)
            lean_x = 4.5 + fall_track * 2.0 + breath * 0.4
            head_rx = lerp(-7.0, 10.0, fall_track) - wonder_up * 11.0
            head_rz = 14.0 + math.sin(t * 1.7) * 3.0
            ua_l = (-22.0 - fall_track * 7.0, 22.0, -30.0 - fall_track * 5.0)
            la_l = (-34.0 - fall_track * 4.0 + math.sin(t * 3.0) * 3.5, 0.0, -20.0)
            ua_r = (-16.0, -30.0, 18.0)
            la_r = (-38.0, 0.0, 28.0)
            n_brow = lerp(0.25, 0.85, smoothstep(1.45, 2.20, t))
            n_smile = lerp(0.10, 0.45, wonder_up)
        elif t < 9.5:
            # Beat 2: Balanced Face-to-Face Debate — Newton LEFT (-0.46, 0.46), yaw=+28°, head_rz=+14° (faces Halley +0.46!)
            nx, ny, nyaw = -0.46, 0.46, 28.0
            sp_u = smoothstep(6.90, 7.35, t)
            lean_x = 3.5 + sp_u * 3.5 + breath * 0.4
            head_rx = 4.0 - sp_u * 3.5
            head_rz = 14.0 + math.sin(t * 3.4) * 3.2 * sp_u
            ua_l = (lerp(-18.0, -28.0, sp_u), lerp(26.0, 20.0, sp_u), lerp(-24.0, -34.0, sp_u))
            la_l = (lerp(-35.0, -38.0, sp_u) + math.sin(t * 4.6) * 4.0 * sp_u, 0.0, lerp(-26.0, -18.0, sp_u))
            ua_r = (lerp(-16.0, -21.0, sp_u), lerp(-30.0, -26.0, sp_u), lerp(18.0, 24.0, sp_u))
            la_r = (lerp(-36.0, -38.0, sp_u), 0.0, lerp(28.0, 24.0, sp_u))
            n_brow = lerp(-0.15, 0.55, sp_u)
            n_smile = lerp(0.05, 0.55, sp_u)
        elif t < 15.5:
            # Beat 3: Newton RIGHT (-1.42, 0.44), yaw=-26°, head_rz=-16° (faces LEFT at Glass Prism & Rainbow -1.96!)
            nx, ny, nyaw = -1.42, 0.44, -26.0
            bloom_u = smoothstep(10.60, 12.20, t)
            lean_x = 5.5 - bloom_u * 2.5 + breath * 0.4
            head_rx = 7.0 - bloom_u * 8.0
            head_rz = -16.0 + math.sin(t * 1.6) * 3.2
            ua_r = (lerp(-22.0, -30.0, bloom_u), lerp(-24.0, -20.0, bloom_u), lerp(28.0, 34.0, bloom_u))
            la_r = (lerp(-34.0, -38.0, bloom_u) + math.sin(t * 2.6) * 3.0, 0.0, lerp(22.0, 18.0, bloom_u))
            ua_l = (-16.0, 30.0, -18.0)
            la_l = (-38.0, 0.0, -28.0)
            n_brow = lerp(0.15, 0.85, bloom_u)
            n_smile = lerp(0.15, 0.55, bloom_u)
        elif t < 20.5:
            # Beat 4: Newton off-camera left watching Halley's Newton's Cradle shock!
            nx, ny, nyaw = -0.65, 0.56, 22.0
            lean_x, head_rx, head_rz = 3.0, 4.0, 14.0
            ua_r, la_r = (-16.0, -30.0, 18.0), (-36.0, 0.0, 28.0)
            ua_l, la_l = (-16.0, 30.0, -18.0), (-36.0, 0.0, -28.0)
            n_brow, n_smile = 0.20, 0.40
        elif t < 25.5:
            # Beat 5: Newton LEFT (-0.30, 0.44), yaw=+24°, head_rz=+14° (faces RIGHT at Earth-Moon Gravity Well +0.28!)
            nx, ny, nyaw = -0.30, 0.44, 24.0
            lean_x = 4.0 + breath * 0.4
            head_rx, head_rz = 4.5, 14.0 + math.sin(t * 1.8) * 3.5
            ua_l = (-28.0 + math.sin(t * 2.2) * 3.0, 20.0, -34.0)
            la_l = (-36.0, 0.0, -18.0)
            ua_r = (-16.0, -30.0, 18.0)
            la_r = (-38.0, 0.0, 28.0)
            n_brow, n_smile = 0.65, 0.65
        else:
            # Beat 6: Grand Finale — Newton LEFT (-0.54, 0.52), yaw=+18° angled cleanly inward toward Earth-Moon Well & Rocket!
            nx, ny, nyaw = -0.54, 0.52, 18.0
            lean_x = 2.0 + breath * 0.4
            head_rx, head_rz = -3.0, 10.0 + math.sin(t * 1.5) * 3.0
            ua_l = (-26.0, 22.0, -30.0)
            la_l = (-36.0, 0.0, -20.0)
            ua_r = (-16.0, -30.0, 18.0)
            la_r = (-38.0, 0.0, 28.0)
            n_brow, n_smile = 0.55, 0.72

        nrig.location = Vector((nx, ny, n_base_z))
        nrig.rotation_euler = Euler((0.0, 0.0, math.radians(nyaw)))
        nrig.keyframe_insert("location", frame=f)
        nrig.keyframe_insert("rotation_euler", frame=f)

        set_bone_char_rot(nrig, "spine02", lean_x * 0.45, 0.0, 0.0, frame=f)
        set_bone_char_rot(nrig, "spine01", lean_x * 0.55, 0.0, 0.0, frame=f)
        set_bone_char_rot(nrig, "neck01", head_rx * 0.35, 0.0, head_rz * 0.35, frame=f)
        set_bone_char_rot(nrig, "head", head_rx * 0.65, 0.0, head_rz * 0.65, frame=f)
        apply_facial_acting(
            nrig, t, f, jaw_open_deg=1.0 + n_lip * 16.5, brow_lift=n_brow, smile=n_smile, blink_period=3.1, blink_phase=0.5
        )
        set_bone_char_rot(nrig, "upperarm01.R", *ua_r, frame=f)
        set_bone_char_rot(nrig, "lowerarm01.R", *la_r, frame=f)
        set_bone_char_rot(nrig, "upperarm01.L", *ua_l, frame=f)
        set_bone_char_rot(nrig, "lowerarm01.L", *la_l, frame=f)

    if HALLEY_ACTOR is not None:
        hrig = HALLEY_ACTOR["rig"]
        h_lip = sample_lip_sync("halley", t)
        h_base_z = HALLEY_ACTOR["base_z"]

        if t < 4.5:
            # Beat 1: Off-camera far right waiting cleanly for Beat 2
            hx_p, hy_p, hyaw = 1.80, 0.54, -22.0
            lean_x, head_rx, head_rz = 2.0, 4.0, -12.0
            ua_l, la_l = (-16.0, 30.0, -18.0), (-36.0, 0.0, -28.0)
            ua_r, la_r = (-16.0, -30.0, 18.0), (-36.0, 0.0, 28.0)
            jaw_bias, h_brow, h_smile = 0.0, -0.1, 0.0
        elif t < 9.5:
            # Beat 2: Halley RIGHT (+0.46, 0.46), yaw=-28°, head_rz=-14° (faces LEFT directly at Newton -0.46!)
            hx_p, hy_p, hyaw = 0.46, 0.46, -28.0
            ask = 1.0 - smoothstep(6.75, 7.20, t)
            lean_x = 5.0 * ask + breath * 0.4
            head_rx = 4.5 * ask - 2.5 * (1.0 - ask)
            head_rz = -14.0 + math.sin(t * 3.5) * 3.5 * ask
            ua_r = (lerp(-20.0, -28.0, ask), lerp(-26.0, -20.0, ask), lerp(24.0, 34.0, ask))
            la_r = (lerp(-35.0, -38.0, ask) + math.sin(t * 4.2) * 4.0 * ask, 0.0, lerp(24.0, 18.0, ask))
            ua_l = (-18.0, 28.0, -22.0)
            la_l = (-36.0, 0.0, -26.0)
            jaw_bias = 0.0
            h_brow = lerp(0.75, -0.45, ask)
            h_smile = lerp(0.15, -0.20, ask)
        elif t < 15.5:
            # Beat 3: Off-camera right during Newton's Glass Prism showcase
            hx_p, hy_p, hyaw = 1.05, 0.52, -18.0
            lean_x, head_rx, head_rz = 3.0, 5.0, -10.0
            ua_l, la_l = (-16.0, 30.0, -18.0), (-36.0, 0.0, -28.0)
            ua_r, la_r = (-16.0, -30.0, 18.0), (-36.0, 0.0, 28.0)
            jaw_bias, h_brow, h_smile = 0.0, 0.0, 0.0
        elif t < 20.5:
            # Beat 4: Halley LEFT (+0.94, 0.44), yaw=+24°, head_rz=+16°->+14° (faces RIGHT at Newton's Cradle +1.44 -> SHOCK RECOIL!)
            shock = smoothstep(17.42, 17.68, t)
            recoil_osc = math.sin((t - 17.45) * 19.0) * math.exp(-(t - 17.45) * 3.4) * 4.2 if t >= 17.45 else 0.0
            hx_p = 0.94 - shock * 0.06
            hy_p = 0.44 + shock * 0.09
            hyaw = 24.0
            lean_x = lerp(6.5, -11.0, shock) + recoil_osc
            head_rx = lerp(7.5, -9.0, shock)
            head_rz = lerp(16.0, 14.0, shock) + math.sin(t * 6.2) * 3.5 * shock
            ua_l = (lerp(-26.0, -30.0, shock), lerp(22.0, 24.0, shock), lerp(-32.0, -34.0, shock))
            la_l = (lerp(-34.0, -50.0, shock), 0.0, lerp(-18.0, -32.0, shock))
            ua_r = (lerp(-16.0, -30.0, shock), lerp(-30.0, -24.0, shock), lerp(18.0, 34.0, shock))
            la_r = (lerp(-36.0, -50.0, shock), 0.0, lerp(28.0, 32.0, shock))
            jaw_bias = 12.5 * shock * (1.0 - smoothstep(18.10, 18.60, t))
            h_brow = lerp(-0.20, 1.0, shock)
            h_smile = lerp(0.30, -0.45, shock)
        elif t < 25.5:
            # Beat 5: Off-camera far right during Earth-Moon Gravity Well climax
            hx_p, hy_p, hyaw = 1.55, 0.52, -20.0
            lean_x, head_rx, head_rz = 2.0, 4.0, -14.0
            ua_l, la_l = (-16.0, 30.0, -18.0), (-36.0, 0.0, -28.0)
            ua_r, la_r = (-16.0, -30.0, 18.0), (-36.0, 0.0, 28.0)
            jaw_bias, h_brow, h_smile = 0.0, 0.25, 0.15
        else:
            # Beat 6: Grand Finale — Halley RIGHT (+0.56, 0.52), yaw=-18° angled cleanly inward toward Earth-Moon Well & Rocket!
            hx_p, hy_p, hyaw = 0.56, 0.52, -18.0
            lean_x = 1.5 + breath * 0.4
            head_rx, head_rz = -3.0, -10.0 + math.sin(t * 1.6) * 3.0
            ua_r = (-26.0, -22.0, 30.0)
            la_r = (-36.0, 0.0, 20.0)
            ua_l = (-16.0, 30.0, -18.0)
            la_l = (-38.0, 0.0, -28.0)
            jaw_bias, h_brow, h_smile = 0.0, 0.55, 0.55

        hrig.location = Vector((hx_p, hy_p, h_base_z))
        hrig.rotation_euler = Euler((0.0, 0.0, math.radians(hyaw)))
        hrig.keyframe_insert("location", frame=f)
        hrig.keyframe_insert("rotation_euler", frame=f)

        set_bone_char_rot(hrig, "spine02", lean_x * 0.45, 0.0, 0.0, frame=f)
        set_bone_char_rot(hrig, "spine01", lean_x * 0.55, 0.0, 0.0, frame=f)
        set_bone_char_rot(hrig, "neck01", head_rx * 0.35, 0.0, head_rz * 0.35, frame=f)
        set_bone_char_rot(hrig, "head", head_rx * 0.65, 0.0, head_rz * 0.65, frame=f)
        apply_facial_acting(
            hrig,
            t,
            f,
            jaw_open_deg=max(jaw_bias, 1.0 + h_lip * 17.0),
            brow_lift=h_brow,
            smile=h_smile,
            blink_period=3.5,
            blink_phase=1.8,
        )
        set_bone_char_rot(hrig, "upperarm01.L", *ua_l, frame=f)
        set_bone_char_rot(hrig, "lowerarm01.L", *la_l, frame=f)
        set_bone_char_rot(hrig, "upperarm01.R", *ua_r, frame=f)
        set_bone_char_rot(hrig, "lowerarm01.R", *la_r, frame=f)


# ============================================================================
# 15. EXECUTE RENDER (QA 6-BEAT PREVIEW OR PRODUCTION FRAME RANGE!)
# ============================================================================
if ARGS.qa_only:
    # Preview all 6 hero Beats (Apple+g, Debate F=Gm1m2/r², 7-Spectral Prism, Cradle Clack, Earth-Moon Well, Rocket Finale!)
    qa_times = [2.6, 8.5, 12.8, 17.8, 23.2, 28.2]
    qa_frames = [max(1, min(TOTAL_FRAMES, int(round(ts * FPS)) + 1)) for ts in qa_times]
    print(f"[Newton V6 Cinema QA] Rendering 6 Beat keyframes {qa_frames} ({FPS}fps, {ARGS.samples}spp, {ARGS.scale}% res) -> {OUT_DIR} ...")
    t_render_start = time.time()
    for idx, qf in enumerate(qa_frames, 1):
        scene.frame_set(qf)
        scene.render.filepath = os.path.join(OUT_DIR, f"qa_beat{idx}_f{qf:04d}.jpg")
        bpy.ops.render.render(write_still=True)
    dt_pure = time.time() - t_render_start
    print(
        f"[QA Pure Render Speed] 6 frames in {dt_pure:.1f}s (~{dt_pure/6:.2f}s/frame -> "
        f"True Est. Full 720-frame Dual-T4 Render Time: ~{(dt_pure/6)*360/60:.1f} min!)"
    )
else:
    print(
        f"[Newton V6 Cinema] Rendering frames {scene.frame_start}..{scene.frame_end} "
        f"({RES_X}x{RES_Y} @ {ARGS.scale}%, {FPS}fps, {ARGS.samples} spp + FOG_GLOW Bloom) -> {OUT_DIR} ..."
    )
    t_render_start = time.time()
    bpy.ops.render.render(animation=True)
    print(f"[Newton V6 Cinema] Batch {scene.frame_start}..{scene.frame_end} complete in {(time.time()-t_render_start)/60:.1f} min!")
