"""
Scientist #9: Isaac Newton (1666-1687) — The Falling Apple, Rainbow Prism & Universal Gravity
Full 30-Second (720 Frames @ 24fps, 1080x1920 Lanczos) Blender 4.2 LTS Cycles V7 Dual-GPU Cinema Short

100% Brand-New V7 Visual Identity (Outdoor Woolsthorpe Apple Orchard & Cotswold Stone Terrace -> Starry Twilight Cosmos):
  - True Ground-Level Outdoor World (`FLOOR_Z = 0.0`) built by `src/newton_world_v7.py`: mossy Cotswold
    flagstones, lush orchard meadow, towering Woolsthorpe Apple Tree with real overhanging branch,
    carved stone garden plinths, dramatic shaded stone arch optical alcove with pinhole shutter, and
    seamless Golden-Hour -> Starry Twilight Cosmos sky transformation with Crescent Moon & 160 stars!
  - Full 3/4-Body Adult 4K MPFB2 Cast (`1.78m` Sir Isaac Newton with 17th-century white ruffled linen
    cravat & dark-indigo scholar cloak + `1.79m` Edmond Halley with white cravat — ZERO shopkeeper desk!).
  - Blender 4.2 Cycles V7 Optical Compositor (`FOG_GLOW` Glare + Filmic Color Balance) + 16 spp Adaptive
    GPU OIDN + `AgX - Punchy` + `texture_limit_render = "2048"` (OOM-proof on Kaggle `GPU T4 x2`!).
  - Beat 1 (0.0-4.5s): Golden-Hour Woolsthorpe Orchard — Hero ruby-gold lacquered apple detaches from real
    overhanging branch (`HERO_APPLE_ATTACH`), accelerates under `1/2 g t²` onto gold-braided crimson velvet
    pad atop `APPLE_PLINTH`, spawning glowing golden downward `g` force arrow & expanding ripple ring!
  - Beat 2 (4.5-9.5s): Sunlit Stone Courtyard Face-to-Face Debate — Full-stature eye-to-eye Newton & Halley
    flanking open *Principia Mathematica*, brass dividers, crystal inkwell & quill on `PRINCIPIA_PLINTH` as
    crisp 3D `F = m a` snaps at `6.95s` into radiant cyan-gold `F = G m1 m2 / r²`!
  - Beat 3 (9.5-15.5s): Shaded Stone Arch Optical Alcove — Tall Brass Optical Field Tripod holding a crystal
    beveled Flint-Glass Equilateral Prism (`IOR = 1.62`) splits a golden-white sunbeam from `PRISM_PINHOLE`
    into 7 wide glowing translucent diverging rainbow dispersion ribbons projecting onto an obsidian easel!
  - Beat 4 (15.5-20.5s): `3.8°` Dutch-Tilt Garden Balustrade Comedy — Stepped mahogany & brass Newton's Cradle
    on `CRADLE_PLINTH` with 5 mirror-chrome spheres (`Roughness = 0.03`): Sphere 0 clacks at `17.45s` launching
    Sphere 4 to `+62°` with 16 golden sparks & shock ring as Halley performs a full-body knee-bend recoil!
  - Beat 5 (20.5-25.5s): Starry Twilight Cosmos Transformation — Full-height Brass Newtonian Reflecting
    Telescope on a tall mahogany outdoor tripod beneath a GIGANTIC luminous sapphire-gold 3D Curved Space-Time
    Gravity Well cradling procedural 3D Earth (with Fresnel atmosphere) & orbiting cratered 3D Moon!
  - Beat 6 (25.5-30.0s): Epic Moonlit Orchard-to-Cosmos Rocket Slingshot Finale — Centered Gravity Well,
    sleek metallic 3D Orbital Rocket with blazing amber-cyan plasma thruster along tilted golden orbit halo,
    crowned by self-luminous gold 3D monument `ISAAC NEWTON / F = G m1 m2 / r²` & full flanking cast!
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
from newton_world_v7 import (
    APPLE_PLINTH,
    CRADLE_PLINTH,
    FLOOR_Z,
    HERO_APPLE_ATTACH,
    ORBIT_ANCHOR,
    PRINCIPIA_PLINTH,
    PRISM_ANCHOR,
    PRISM_PINHOLE,
    animate_world_frame,
    build_woolsthorpe_world,
)


# ============================================================================
# 1. CLI ARGUMENTS & CONFIGURATION
# ============================================================================
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Isaac Newton V7 Cycles Dual-GPU Outdoor-to-Cosmos 3D Short Renderer")
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
# 3. CLEAN SCENE (EXTENSION-SAFE FOR MPFB2!) & CONFIGURE CYCLES V7 + BLOOM
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

# V7 Zero-Quality-Drop Cycles OPTIX Turbo (Full 75%->1080p, 2048px 4K Cast, ~6.5-8m on Dual-T4!)
scene.cycles.samples = ARGS.samples
scene.cycles.use_adaptive_sampling = True
scene.cycles.adaptive_threshold = 0.055
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPENIMAGEDENOISE"  # Auto-upgraded to hardware OPTIX below when RTX/T4 GPU active!
scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
scene.cycles.denoising_prefilter = "NONE"
try:
    scene.cycles.denoising_use_gpu = True
except Exception:
    pass
try:
    scene.cycles.use_light_tree = False  # Faster GPU traversal for compact 4-5 light cinema rigs
except Exception:
    pass
scene.render.threads_mode = "FIXED"
scene.render.threads = 2
scene.cycles.caustics_reflective = False
scene.cycles.caustics_refractive = False
scene.cycles.sample_clamp_indirect = 4.5
scene.cycles.blur_glossy = 0.90
scene.cycles.max_bounces = 4
scene.cycles.transmission_bounces = 4
scene.cycles.glossy_bounces = 2
scene.cycles.diffuse_bounces = 2
scene.cycles.transparent_max_bounces = 12
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

# Keep Blender CPU Compositor OFF (saves ~0.9s/frame = ~10 min of idle GPU stalls on Kaggle 2-vCPU!);
# True cinema highlight optical bloom + warm/cool color polish runs in Cell 5 FFmpeg in ~3s total!
scene.use_nodes = False

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
                if backend == "OPTIX":
                    try:
                        scene.cycles.denoiser = "OPTIX"
                    except Exception:
                        pass
                print(
                    f"[Cycles V7 Zero-Loss Turbo] {backend} GPU ({len(gpu_devices)} device(s)) | "
                    f"Denoiser={scene.cycles.denoiser} (RGB_ALBEDO_NORMAL) | "
                    f"Frames {scene.frame_start}..{scene.frame_end} ({FPS}fps, {ARGS.scale}% res, {ARGS.samples} spp)"
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
scene.render.image_settings.quality = 95
scene.render.filepath = os.path.join(OUT_DIR, "frame_")


# ============================================================================
# 4. BUILD V7 OUTDOOR WOOLSTHORPE APPLE ORCHARD & COTSWOLD STONE TERRACE WORLD!
# ============================================================================
world_handles = build_woolsthorpe_world(scene)


# ============================================================================
# 5. V7 HERO PHYSICS & SCIENTIFIC APPARATUS PBR MATERIALS
# ============================================================================
def set_bsdf(bsdf, name: str, val):
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = val


def get_rgba_mix_sockets(mix_node):
    """Blender 4.2 safe RGBA socket lookup for ShaderNodeMix (never use duplicate string keys!)."""
    rgba_in = [s for s in mix_node.inputs if s.type == "RGBA"]
    rgba_out = [s for s in mix_node.outputs if s.type == "RGBA"]
    return mix_node.inputs[0], rgba_in[0], rgba_in[1], rgba_out[0]


def make_mahogany_mat(name="InstrumentMahogany", scale_vec=(1.5, 5.2, 1.5), dark=False):
    """Hand-rubbed polished English mahogany for field tripods, Newton's Cradle base & telescope yoke."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = scale_vec
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 7.0
    ns.inputs["Detail"].default_value = 9.0
    ns.inputs["Distortion"].default_value = 1.6
    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.22
    rmp.color_ramp.elements[0].color = (0.042, 0.014, 0.008, 1.0) if dark else (0.095, 0.032, 0.014, 1.0)
    rmp.color_ramp.elements[1].position = 0.80
    rmp.color_ramp.elements[1].color = (0.125, 0.045, 0.020, 1.0) if dark else (0.240, 0.088, 0.034, 1.0)
    r_map = nt.nodes.new("ShaderNodeMapRange")
    r_map.inputs["To Min"].default_value = 0.26 if dark else 0.20
    r_map.inputs["To Max"].default_value = 0.38 if dark else 0.32
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.09
    set_bsdf(bsdf, "Coat Weight", 0.45 if not dark else 0.28)
    set_bsdf(bsdf, "Coat Roughness", 0.12)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ns.outputs["Fac"], r_map.inputs["Value"])
    nt.links.new(r_map.outputs["Result"], bsdf.inputs["Roughness"])
    nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_metal_mat(name, rgb, rough=0.18, metallic=0.96, bump_str=0.020, coat=0.0):
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


def make_glass_mat(name, ior=1.62, tint=(0.96, 0.99, 1.0), rough=0.005, coat=0.72):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    set_bsdf(bsdf, "Base Color", (*tint, 1.0))
    set_bsdf(bsdf, "Roughness", rough)
    set_bsdf(bsdf, "IOR", ior)
    set_bsdf(bsdf, "Transmission Weight", 0.98)
    set_bsdf(bsdf, "Transmission", 0.98)
    set_bsdf(bsdf, "Coat Weight", coat)
    set_bsdf(bsdf, "Coat Roughness", 0.015)
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


def make_feathered_beam_mat(name, rgb, strength=5.4, power=1.40, opacity=0.84):
    """Feathered soft-edge optical light-beam & rainbow ribbon shader (Transparent <-> Emission via LayerWeight)."""
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


def make_ruby_apple_mat():
    """Glossy ruby-gold mottled Woolsthorpe heirloom apple skin with rich clearcoat lacquer (Coat=0.92)."""
    mat = bpy.data.materials.new("WoolsthorpeRubyAppleV7")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 6.5
    ns.inputs["Detail"].default_value = 7.0
    ns.inputs["Distortion"].default_value = 0.85

    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.28
    rmp.color_ramp.elements[0].color = (0.48, 0.016, 0.012, 1.0)
    e1 = rmp.color_ramp.elements.new(0.60)
    e1.color = (0.78, 0.052, 0.022, 1.0)
    rmp.color_ramp.elements[-1].position = 0.84
    rmp.color_ramp.elements[-1].color = (0.90, 0.56, 0.08, 1.0)

    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.032
    set_bsdf(bsdf, "Roughness", 0.14)
    set_bsdf(bsdf, "Coat Weight", 0.92)
    set_bsdf(bsdf, "Coat Roughness", 0.04)
    set_bsdf(bsdf, "Subsurface Weight", 0.08)

    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_earth_planet_mat():
    """High-detail procedural 3D Earth: deep sapphire oceans, emerald/ochre continents & swirling white clouds!"""
    mat = bpy.data.materials.new("ProceduralEarth3DV7")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")

    ns_land = nt.nodes.new("ShaderNodeTexNoise")
    ns_land.inputs["Scale"].default_value = 3.2
    ns_land.inputs["Detail"].default_value = 10.0
    ns_land.inputs["Roughness"].default_value = 0.58

    land_rmp = nt.nodes.new("ShaderNodeValToRGB")
    cr = land_rmp.color_ramp
    cr.elements[0].position = 0.46
    cr.elements[0].color = (0.010, 0.095, 0.44, 1.0)  # Deep sapphire ocean
    e_sh = cr.elements.new(0.51)
    e_sh.color = (0.028, 0.25, 0.58, 1.0)  # Coastal turquoise
    e_gr = cr.elements.new(0.54)
    e_gr.color = (0.062, 0.38, 0.11, 1.0)  # Emerald continent
    cr.elements[-1].position = 0.72
    cr.elements[-1].color = (0.50, 0.37, 0.16, 1.0)  # Warm ochre highlands

    ns_cl = nt.nodes.new("ShaderNodeTexNoise")
    ns_cl.inputs["Scale"].default_value = 5.4
    ns_cl.inputs["Detail"].default_value = 8.0
    ns_cl.inputs["Distortion"].default_value = 1.48

    cl_rmp = nt.nodes.new("ShaderNodeValToRGB")
    cl_rmp.color_ramp.elements[0].position = 0.56
    cl_rmp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    cl_rmp.color_ramp.elements[1].position = 0.76
    cl_rmp.color_ramp.elements[1].color = (0.96, 0.98, 1.0, 1.0)

    mix_cl = nt.nodes.new("ShaderNodeMix")
    mix_cl.data_type = "RGBA"
    mix_cl.blend_type = "MIX"
    mc_fac, mc_a, mc_b, mc_out = get_rgba_mix_sockets(mix_cl)
    mc_b.default_value = (0.96, 0.98, 1.0, 1.0)

    rough_rmp = nt.nodes.new("ShaderNodeValToRGB")
    rough_rmp.color_ramp.elements[0].position = 0.49
    rough_rmp.color_ramp.elements[0].color = (0.15, 0.15, 0.15, 1.0)
    rough_rmp.color_ramp.elements[1].position = 0.54
    rough_rmp.color_ramp.elements[1].color = (0.68, 0.68, 0.68, 1.0)

    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.08

    set_bsdf(bsdf, "Coat Weight", 0.28)
    set_bsdf(bsdf, "Emission Strength", 0.34)

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
    """Glowing blue Fresnel atmospheric halo around Earth (LayerWeight Facing^2.5)."""
    mat = bpy.data.materials.new("EarthAtmosphereHaloV7")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    mix = nt.nodes.new("ShaderNodeMixShader")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.16, 0.66, 1.0, 1.0)
    em.inputs["Strength"].default_value = 4.6
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.36
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = 2.5
    nt.links.new(lw.outputs["Facing"], pw.inputs[0])
    nt.links.new(pw.outputs["Value"], mix.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
    nt.links.new(em.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


def make_cratered_moon_mat():
    """Procedural Voronoi-cratered dusty lunar regolith."""
    mat = bpy.data.materials.new("CrateredMoon3DV7")
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
    rmp.color_ramp.elements[0].color = (0.30, 0.31, 0.35, 1.0)
    rmp.color_ramp.elements[1].position = 0.78
    rmp.color_ramp.elements[1].color = (0.78, 0.77, 0.74, 1.0)
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.28
    set_bsdf(bsdf, "Roughness", 0.80)
    set_bsdf(bsdf, "Emission Color", (0.80, 0.84, 0.90, 1.0))
    set_bsdf(bsdf, "Emission Strength", 0.28)
    nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(vor.outputs["Distance"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


# Instantiate V7 hero scientific & physics apparatus material palette
mahogany_mat = make_mahogany_mat("InstrumentMahogany", (1.5, 5.2, 1.5), dark=False)
mahogany_dark_mat = make_mahogany_mat("InstrumentMahoganyDark", (2.0, 2.0, 2.0), dark=True)
brass_mat = make_metal_mat("OpticalBrassV7", (0.91, 0.66, 0.22), rough=0.16, metallic=0.96, bump_str=0.018)
bronze_mat = make_metal_mat("AgedBronzeV7", (0.28, 0.19, 0.095), rough=0.30, metallic=0.88, bump_str=0.024)
mirror_chrome_mat = make_metal_mat("NewtonCradleChromeV7", (0.94, 0.96, 0.99), rough=0.030, metallic=1.0, bump_str=0.0)
ebony_mat = make_metal_mat("ObsidianOpticalEaselV7", (0.012, 0.014, 0.022), rough=0.20, metallic=0.06, bump_str=0.008, coat=0.32)
crimson_velvet_mat = make_metal_mat("CrimsonVelvetPadV7", (0.30, 0.022, 0.036), rough=0.88, metallic=0.0, bump_str=0.0)
leather_navy_mat = make_metal_mat("PrincipiaLeatherV7", (0.055, 0.082, 0.185), rough=0.50, metallic=0.0, bump_str=0.032)
parchment_mat = make_metal_mat("PrincipiaParchmentV7", (0.94, 0.87, 0.71), rough=0.64, metallic=0.0, bump_str=0.020)
ink_dark_mat = make_metal_mat("PrincipiaInkV7", (0.032, 0.026, 0.020), rough=0.28, metallic=0.0, bump_str=0.0)
quill_white_mat = make_metal_mat("GooseQuillWhiteV7", (0.95, 0.93, 0.89), rough=0.44, metallic=0.0, bump_str=0.0)
rocket_white_mat = make_metal_mat("RocketCeramicWhiteV7", (0.91, 0.93, 0.96), rough=0.16, metallic=0.35, bump_str=0.0, coat=0.58)
rocket_dark_mat = make_metal_mat("RocketCarbonObsidianV7", (0.055, 0.065, 0.092), rough=0.18, metallic=0.86, bump_str=0.0)
speculum_mirror_mat = make_metal_mat("SpeculumMirrorV7", (0.90, 0.93, 0.97), rough=0.035, metallic=1.0, bump_str=0.0)

flint_glass_mat = make_glass_mat("FlintGlassPrismV7", ior=1.62, tint=(0.96, 0.99, 1.0), rough=0.005, coat=0.72)
inkwell_glass_mat = make_glass_mat("CrystalInkwellGlassV7", ior=1.52, tint=(0.88, 0.95, 1.0), rough=0.02, coat=0.48)

ruby_apple_mat = make_ruby_apple_mat()
apple_stem_mat = make_metal_mat("AppleWoodStemV7", (0.11, 0.068, 0.032), rough=0.80, metallic=0.0, bump_str=0.04)
emerald_leaf_mat = make_metal_mat("AppleEmeraldLeafV7", (0.052, 0.30, 0.045), rough=0.30, metallic=0.0, bump_str=0.025, coat=0.24)
earth_planet_mat = make_earth_planet_mat()
earth_atmo_mat = make_earth_atmo_halo_mat()
cratered_moon_mat = make_cratered_moon_mat()

# AgX-calibrated vivid glowing optical materials (bloom beautifully via FOG_GLOW Compositor!)
gold_glow_mat = make_emissive_mat("NewtonGoldGlowV7", (1.0, 0.68, 0.10), strength=5.0)
cyan_glow_mat = make_emissive_mat("GravitySapphireGlowV7", (0.08, 0.70, 1.0), strength=4.6)
pinhole_flare_mat = make_emissive_mat("PrismPinholeFlareV7", (1.0, 0.92, 0.68), strength=8.5)
sunbeam_goldwhite_mat = make_feathered_beam_mat("PrismGoldenSunbeamV7", (1.0, 0.94, 0.78), strength=6.2, power=1.30, opacity=0.90)
rocket_plume_mat = make_feathered_beam_mat("RocketPlasmaPlumeV7", (0.30, 0.78, 1.0), strength=8.0, power=1.35, opacity=0.92)
rocket_core_mat = make_emissive_mat("RocketCoreFlameV7", (1.0, 0.64, 0.14), strength=8.5)

# 7 Distinct Spectral Dispersion Ribbon & Easel Band Materials (Red..Violet!)
SPECTRAL_RGBS = [
    ("Red", (1.00, 0.05, 0.04)),
    ("Orange", (1.00, 0.36, 0.03)),
    ("Yellow", (1.00, 0.84, 0.04)),
    ("Green", (0.05, 0.94, 0.18)),
    ("Cyan", (0.04, 0.82, 1.00)),
    ("Blue", (0.08, 0.28, 1.00)),
    ("Violet", (0.64, 0.10, 1.00)),
]
rainbow_ribbon_mats = [
    make_feathered_beam_mat(f"PrismRibbonV7_{cname}", rgb, strength=6.0, power=1.18, opacity=0.86)
    for cname, rgb in SPECTRAL_RGBS
]
rainbow_band_mats = [
    make_emissive_mat(f"SpectralBandV7_{cname}", rgb, strength=4.8)
    for cname, rgb in SPECTRAL_RGBS
]

# Razor-sharp anti-washout 3D Formula & Monument Title materials + bundled DejaVuSans-Bold font
text_gold_mat = make_emissive_mat("TextGoldCrispV7", (1.0, 0.74, 0.14), strength=2.60)
text_cyangold_mat = make_emissive_mat("TextCyanGoldCrispV7", (0.18, 0.86, 1.0), strength=2.80)
title_gold_mat = bpy.data.materials.new("TitleCinemaGoldV7")
title_gold_mat.use_nodes = True
_tbsdf = title_gold_mat.node_tree.nodes["Principled BSDF"]
set_bsdf(_tbsdf, "Base Color", (1.0, 0.80, 0.26, 1.0))
set_bsdf(_tbsdf, "Metallic", 0.78)
set_bsdf(_tbsdf, "Roughness", 0.20)
set_bsdf(_tbsdf, "Emission Color", (1.0, 0.74, 0.18, 1.0))
set_bsdf(_tbsdf, "Emission Strength", 2.15)

FONT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "subtitles", "fonts", "DejaVuSans-Bold.ttf"))
HERO_FONT = bpy.data.fonts.load(FONT_PATH) if os.path.isfile(FONT_PATH) else None


# ============================================================================
# 6. HIGH-CRAFT PROCEDURAL GEOMETRY HELPERS (APPLE, PRISM, TRIPODS, CURVES)
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


def orient_beam(obj, p1: Vector, p2: Vector, radius_scale: float = 1.0, ribbon_y_mult: float = 1.0):
    """Orient a unit-Z cylinder or tapered cone cleanly from p1 (-Z/2) to p2 (+Z/2)."""
    diff = p2 - p1
    length = diff.length
    obj.location = (p1 + p2) * 0.5
    obj.scale = (radius_scale, radius_scale * ribbon_y_mult, max(0.0005, length))
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = diff.normalized().to_track_quat("Z", "Y") if length > 1e-5 else Quaternion((1, 0, 0, 0))


def build_sculpted_leaf(name, loc, length=0.040, width=0.019, rot=(0.0, 0.0, 0.0)):
    """Gracefully arched almond-shaped apple leaf with central midrib crease."""
    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()
    stations = [
        (0.00, 0.000, 0.000, 0.000),
        (0.28, width * 0.46, 0.003, 0.003),
        (0.62, width * 0.50, 0.001, 0.004),
        (0.86, width * 0.24, -0.004, 0.002),
        (1.00, 0.000, -0.008, 0.000),
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


def build_organic_apple(name, loc, radius=0.044):
    """Sculpted heirloom Woolsthorpe ruby-gold apple (polar dimples, subtle 5-lobe fullness, stem & emerald leaf!)."""
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=loc)
    root = bpy.context.active_object
    root.name = name

    mesh = bpy.data.meshes.new(name + "_FruitMesh")
    bm = bmesh.new()
    n_seg = 28
    n_ring = 18
    profile = []
    for i in range(n_ring + 1):
        u = i / n_ring
        th = u * math.pi
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

    stem = build_curve_tube(
        f"{name}_Stem",
        [
            (0.0, 0.0, radius * 0.58, 0.0028),
            (0.004, -0.002, radius * 0.96, 0.0024),
            (0.010, -0.004, radius * 1.30, 0.0030),
        ],
        apple_stem_mat,
        bevel_res=4,
    )
    stem.parent = root

    leaf = build_sculpted_leaf(f"{name}_Leaf", (0.008, -0.004, radius * 1.08), length=0.042, width=0.020, rot=(0.25, 0.45, 0.55))
    leaf.parent = root
    return root


def build_equilateral_prism(name, loc, side=0.148, depth_y=0.120, mat=None):
    """Crisp beveled equilateral triangular optical flint-glass prism facing the camera (XZ triangle)."""
    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()
    h = side * math.sqrt(3.0) * 0.5
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


def build_tripod_legs(prefix: str, parent_obj, hub_z: float, foot_r: float, leg_r: float, leg_mat, collar_mat, az_offset_deg: float = 30.0):
    """Build 3 splayed mahogany/brass outdoor scientific tripod legs cleanly grounded at Z = 0.0!"""
    for i in range(3):
        ang = math.radians(az_offset_deg + i * 120.0)
        p_top = Vector((math.cos(ang) * 0.036, math.sin(ang) * 0.036, hub_z))
        p_bot = Vector((math.cos(ang) * foot_r, math.sin(ang) * foot_r, 0.012))
        leg = add_cyl(f"{prefix}_Leg_{i}", (0, 0, 0), leg_r, 1.0, leg_mat, verts=20)
        orient_beam(leg, p_bot, p_top, 1.0)
        leg.parent = parent_obj

        ferrule = add_cone(f"{prefix}_Foot_{i}", (p_bot.x, p_bot.y, 0.022), leg_r * 0.65, leg_r * 1.15, 0.044, collar_mat, verts=18)
        ferrule.parent = parent_obj

        # Mid-height brass spreader strut connecting each leg to central spreader collar
        p_mid = vlerp(p_bot, p_top, 0.48)
        p_hub_mid = Vector((0.0, 0.0, p_mid.z))
        strut = add_cyl(f"{prefix}_Strut_{i}", (0, 0, 0), 0.0055, 1.0, collar_mat, verts=12)
        orient_beam(strut, p_hub_mid, p_mid, 1.0)
        strut.parent = parent_obj

    mid_z = hub_z * 0.48 + 0.006
    spreader = add_cyl(f"{prefix}_SpreaderHub", (0.0, 0.0, mid_z), 0.024, 0.024, collar_mat, verts=24, bevel_w=0.003)
    spreader.parent = parent_obj


# ============================================================================
# 7. BEAT 1 — HERO WOOLSTHORPE FALLING APPLE, VELVET PAD & GOLDEN 'g' VECTOR
# ============================================================================
# Gold-Braided Crimson Velvet Pad resting cleanly ON TOP of carved stone garden plinth APPLE_PLINTH (Z=0.88)
add_cyl("AppleVelvetPad", (APPLE_PLINTH.x, APPLE_PLINTH.y, APPLE_PLINTH.z + 0.011), 0.145, 0.022, crimson_velvet_mat, verts=48, bevel_w=0.006)
add_torus("AppleVelvetGoldBraid", (APPLE_PLINTH.x, APPLE_PLINTH.y, APPLE_PLINTH.z + 0.013), 0.147, 0.008, brass_mat, maj_seg=56, min_seg=12)

# HERO FALLING APPLE attached to REAL overhanging Woolsthorpe tree branch at HERO_APPLE_ATTACH = (-0.52, 0.12, 1.94)!
hero_apple = build_organic_apple("HeroFallingApple", HERO_APPLE_ATTACH, radius=0.044)

# Glowing Golden Downward Gravity Force Arrow ('g', Z = 1.15..1.52) + Expanding Energy Ripple Rings!
G_ARROW_POS = Vector((APPLE_PLINTH.x + 0.08, APPLE_PLINTH.y, 1.34))
bpy.ops.object.empty_add(type="PLAIN_AXES", location=G_ARROW_POS)
gravity_arrow_root = bpy.context.active_object
gravity_arrow_root.name = "GravityForceArrowRoot"

g_shaft = add_cyl("GravityArrowShaft", (0.0, 0.0, 0.055), 0.010, 0.250, gold_glow_mat, verts=20)
g_shaft.parent = gravity_arrow_root
g_head = add_cone("GravityArrowHead", (0.0, 0.0, -0.115), 0.032, 0.001, 0.084, gold_glow_mat, rot=(math.pi, 0.0, 0.0), verts=24)
g_head.parent = gravity_arrow_root

bpy.ops.object.text_add(location=(0.068, -0.012, 0.035), rotation=(math.pi * 0.5, 0.0, 0.0))
g_label = bpy.context.active_object
g_label.name = "GravityLabelG"
if HERO_FONT is not None:
    g_label.data.font = HERO_FONT
g_label.data.body = "g"
g_label.data.align_x = "CENTER"
g_label.data.size = 0.094
g_label.data.extrude = 0.0055
g_label.data.bevel_depth = 0.0006
g_label.data.materials.append(text_gold_mat)
g_label.parent = gravity_arrow_root

impact_ripple_1 = add_torus(
    "AppleImpactRipple1", (APPLE_PLINTH.x, APPLE_PLINTH.y, APPLE_PLINTH.z + 0.026), 0.062, 0.005, gold_glow_mat, maj_seg=56, min_seg=10
)
impact_ripple_1.visible_shadow = False
impact_ripple_2 = add_torus(
    "AppleImpactRipple2", (APPLE_PLINTH.x, APPLE_PLINTH.y, APPLE_PLINTH.z + 0.028), 0.052, 0.0038, gold_glow_mat, maj_seg=56, min_seg=10
)
impact_ripple_2.visible_shadow = False


# ============================================================================
# 8. BEAT 2 — PRINCIPIA MATHEMATICA FOLIO, DIVIDERS, INKWELL & 3D FORMULAS
# ============================================================================
# Parented to principia_props_root at PRINCIPIA_PLINTH = (0.55, 0.08, 0.90) so it clears cleanly before Beat 5!
bpy.ops.object.empty_add(type="PLAIN_AXES", location=PRINCIPIA_PLINTH)
principia_props_root = bpy.context.active_object
principia_props_root.name = "PrincipiaPropsRoot"

p_spine = add_box("PrincipiaSpine", (0.0, -0.015, 0.014), (0.040, 0.230, 0.026), leather_navy_mat, bevel_w=0.004)
p_spine.parent = principia_props_root
p_cov_l = add_box("PrincipiaCoverL", (-0.098, -0.015, 0.022), (0.186, 0.230, 0.016), leather_navy_mat, rot=(0.0, math.radians(8.0), 0.0), bevel_w=0.004)
p_cov_l.parent = principia_props_root
p_cov_r = add_box("PrincipiaCoverR", (0.098, -0.015, 0.022), (0.186, 0.230, 0.016), leather_navy_mat, rot=(0.0, math.radians(-8.0), 0.0), bevel_w=0.004)
p_cov_r.parent = principia_props_root
p_pg_l = add_box("PrincipiaPagesL", (-0.092, -0.015, 0.036), (0.170, 0.212, 0.022), parchment_mat, rot=(0.0, math.radians(8.5), 0.0), bevel_w=0.003)
p_pg_l.parent = principia_props_root
p_pg_r = add_box("PrincipiaPagesR", (0.092, -0.015, 0.036), (0.170, 0.212, 0.022), parchment_mat, rot=(0.0, math.radians(-8.5), 0.0), bevel_w=0.003)
p_pg_r.parent = principia_props_root

# Crisp Latin ink lines & elliptical orbital diagram on open Principia pages!
for ln_i in range(4):
    ink_ln = add_box(
        f"PrincipiaInkL_{ln_i}",
        (-0.092, -0.072 + ln_i * 0.038, 0.049),
        (0.118, 0.007, 0.002),
        ink_dark_mat,
        rot=(0.0, math.radians(8.5), 0.0),
    )
    ink_ln.parent = principia_props_root

orb_diag = add_torus(
    "PrincipiaOrbitalDiagram", (0.092, -0.015, 0.050), 0.052, 0.0028, ink_dark_mat, rot=(0.0, math.radians(-8.5), 0.0), maj_seg=40, min_seg=8
)
orb_diag.parent = principia_props_root
orb_sun = add_sphere("PrincipiaOrbitFocus", (0.074, -0.015, 0.051), 0.008, brass_mat, segs=16, rings=10)
orb_sun.parent = principia_props_root

# Gold corner clasps, brass dividers, faceted crystal inkwell & white goose quill (all cleanly inside r=0.238m!)
for sx_c in (-0.184, 0.184):
    for sy_c in (-0.098, 0.068):
        cl = add_box(f"PrincipiaClasp_{sx_c:+.2f}_{sy_c:+.2f}", (sx_c, sy_c, 0.035), (0.016, 0.022, 0.020), brass_mat, bevel_w=0.002)
        cl.parent = principia_props_root

div_j = add_cyl("DividersJoint", (-0.135, 0.145, 0.012), 0.010, 0.012, brass_mat, verts=20)
div_j.parent = principia_props_root
div_l1 = add_cone("DividersLeg1", (-0.075, 0.128, 0.009), 0.0045, 0.001, 0.125, brass_mat, rot=(0.0, math.pi * 0.5, -0.28), verts=16)
div_l1.parent = principia_props_root
div_l2 = add_cone("DividersLeg2", (-0.075, 0.168, 0.009), 0.0045, 0.001, 0.125, brass_mat, rot=(0.0, math.pi * 0.5, 0.34), verts=16)
div_l2.parent = principia_props_root

ink_box = add_box("CrystalInkwell", (0.145, 0.138, 0.024), (0.044, 0.044, 0.046), inkwell_glass_mat, bevel_w=0.005)
ink_box.parent = principia_props_root
ink_col = add_cyl("InkwellCollar", (0.145, 0.138, 0.050), 0.015, 0.013, brass_mat, verts=24)
ink_col.parent = principia_props_root
ink_pool = add_cyl("InkDarkPool", (0.145, 0.138, 0.020), 0.016, 0.026, ink_dark_mat, verts=20)
ink_pool.parent = principia_props_root
q_shaft = add_cone("QuillShaft", (0.172, 0.138, 0.122), 0.002, 0.003, 0.175, quill_white_mat, rot=(0.10, math.radians(20.0), 0.0), verts=12)
q_shaft.parent = principia_props_root
q_plume = add_box("QuillPlume", (0.198, 0.142, 0.160), (0.024, 0.004, 0.105), quill_white_mat, rot=(0.10, math.radians(20.0), 0.0), bevel_w=0.002)
q_plume.parent = principia_props_root

# Floating 3D Formulas (`DejaVuSans-Bold.ttf`, Y = -0.02, centered in the gap between Newton & Halley!)
bpy.ops.object.text_add(location=(PRINCIPIA_PLINTH.x, -0.02, 1.22), rotation=(math.pi * 0.5, 0.0, 0.0))
formula_fma_txt = bpy.context.active_object
formula_fma_txt.name = "NewtonFormula_Fma"
if HERO_FONT is not None:
    formula_fma_txt.data.font = HERO_FONT
formula_fma_txt.data.body = "F = m a"
formula_fma_txt.data.align_x = "CENTER"
formula_fma_txt.data.size = 0.078
formula_fma_txt.data.space_character = 1.15
formula_fma_txt.data.extrude = 0.0060
formula_fma_txt.data.bevel_depth = 0.0006
formula_fma_txt.data.materials.append(text_gold_mat)

bpy.ops.object.text_add(location=(PRINCIPIA_PLINTH.x, -0.02, 1.26), rotation=(math.pi * 0.5, 0.0, 0.0))
formula_grav_txt = bpy.context.active_object
formula_grav_txt.name = "NewtonFormula_Gravity"
if HERO_FONT is not None:
    formula_grav_txt.data.font = HERO_FONT
formula_grav_txt.data.body = "F = G m1 m2 / r²"
formula_grav_txt.data.align_x = "CENTER"
formula_grav_txt.data.size = 0.062
formula_grav_txt.data.space_character = 1.13
formula_grav_txt.data.extrude = 0.0060
formula_grav_txt.data.bevel_depth = 0.0006
formula_grav_txt.data.materials.append(text_cyangold_mat)


# ============================================================================
# 9. BEAT 3 — BRASS OPTICAL FIELD TRIPOD, FLINT-GLASS PRISM & 7-SPECTRAL EASEL
# ============================================================================
# Sleek Tall Brass Optical Field Tripod grounded cleanly on FLOOR_Z=0.0 at PRISM_ANCHOR = (-2.85, 0.06, 0.0)
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(PRISM_ANCHOR.x, PRISM_ANCHOR.y, FLOOR_Z))
prism_tripod_root = bpy.context.active_object
prism_tripod_root.name = "PrismFieldTripodRoot"

build_tripod_legs("PrismTripod", prism_tripod_root, hub_z=1.02, foot_r=0.25, leg_r=0.012, leg_mat=mahogany_mat, collar_mat=brass_mat, az_offset_deg=30.0)
pt_hub = add_cyl("PrismTripodHub", (0.0, 0.0, 1.04), 0.044, 0.056, brass_mat, verts=32, bevel_w=0.004)
pt_hub.parent = prism_tripod_root
pt_col = add_cyl("PrismTripodColumn", (0.0, 0.0, 1.145), 0.015, 0.165, brass_mat, verts=28)
pt_col.parent = prism_tripod_root
pt_knop = add_torus("PrismTripodKnop", (0.0, 0.0, 1.135), 0.026, 0.009, bronze_mat, maj_seg=28, min_seg=10)
pt_knop.parent = prism_tripod_root
# Equilateral prism side=0.148 centered at Z=1.280 -> base sits cleanly at Z=1.2373 -> cradle plate top at Z=1.237!
pt_plate = add_box("PrismCradlePlate", (0.0, 0.0, 1.230), (0.154, 0.124, 0.014), brass_mat, bevel_w=0.003)
pt_plate.parent = prism_tripod_root

# Hero Beveled Equilateral Flint-Glass Prism (IOR=1.62, centered at (-2.85, 0.06, 1.28))
PRISM_CENTER = Vector((PRISM_ANCHOR.x, PRISM_ANCHOR.y, 1.28))
hero_prism = build_equilateral_prism("FlintGlassTriangularPrism", PRISM_CENTER, side=0.148, depth_y=0.120, mat=flint_glass_mat)

# Dark Ebony / Obsidian Projection Easel at X = -2.54, Y = 0.06, Z = 1.24 (mounted on tall brass stand from Z = 0.0!)
EASEL_POS = Vector((-2.54, 0.06, 1.24))
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(EASEL_POS.x, EASEL_POS.y, FLOOR_Z), rotation=(0.0, 0.0, math.radians(-24.0)))
easel_root = bpy.context.active_object
easel_root.name = "OpticalEaselRoot"

es_foot = add_cyl("OptEaselBaseFoot", (0.0, 0.0, 0.022), 0.135, 0.044, brass_mat, verts=40, bevel_w=0.006)
es_foot.parent = easel_root
es_step = add_cyl("OptEaselBaseStep", (0.0, 0.0, 0.056), 0.092, 0.026, bronze_mat, verts=32, bevel_w=0.004)
es_step.parent = easel_root
es_pole = add_cyl("OptEaselBrassPole", (0.0, 0.0, 0.56), 0.014, 0.98, brass_mat, verts=28)
es_pole.parent = easel_root
es_knop = add_torus("OptEaselPoleKnop", (0.0, 0.0, 0.88), 0.024, 0.009, brass_mat, maj_seg=28, min_seg=10)
es_knop.parent = easel_root
es_frame = add_box("OptEaselBrassFrame", (0.0, 0.0, EASEL_POS.z), (0.024, 0.148, 0.390), brass_mat, bevel_w=0.004)
es_frame.parent = easel_root
es_slate = add_box("OptEaselObsidian", (-0.007, 0.0, EASEL_POS.z), (0.015, 0.132, 0.368), ebony_mat, bevel_w=0.002)
es_slate.parent = easel_root

# 7 Vivid Glowing Spectral Bands projected cleanly onto the dark obsidian easel (Red high -> Violet low!)
spectral_bands = []
for b_i in range(7):
    bz = lerp(1.385, 1.095, b_i / 6.0)
    band = add_box(f"SpectralBand_{b_i}", (-0.016, 0.0, bz), (0.005, 0.120, 0.036), rainbow_band_mats[b_i], bevel_w=0.001)
    band.parent = easel_root
    band.visible_shadow = False
    spectral_bands.append(band)

# Golden-White Sunbeam streaming diagonally straight from PRISM_PINHOLE = (-3.38, 0.06, 1.54) into prism left face!
PRISM_ENTRY = Vector((PRISM_CENTER.x - 0.045, PRISM_CENTER.y, PRISM_CENTER.z + 0.022))
PRISM_EXIT = Vector((PRISM_CENTER.x + 0.044, PRISM_CENTER.y, PRISM_CENTER.z - 0.004))

pinhole_flare = add_sphere("PrismPinholeFlare", PRISM_PINHOLE, 0.022, pinhole_flare_mat, segs=18, rings=12)
pinhole_flare.visible_shadow = False

white_sunbeam = add_cyl("GoldenWhiteSunbeamRay", (0, 0, 0), 0.017, 1.0, sunbeam_goldwhite_mat, verts=24)
white_sunbeam.visible_shadow = False
white_sunbeam.visible_diffuse = False
white_sunbeam.visible_glossy = False

internal_beam = add_cyl("PrismInternalRay", (0, 0, 0), 0.014, 1.0, sunbeam_goldwhite_mat, verts=20)
internal_beam.visible_shadow = False
internal_beam.visible_diffuse = False
internal_beam.visible_glossy = False

# 7 Wide Diverging Translucent Rainbow Dispersion Ribbons fanning from PRISM_EXIT onto the obsidian easel!
rainbow_ribbons = []
rainbow_targets = []
for b_i in range(7):
    bz = lerp(1.385, 1.095, b_i / 6.0)
    hit_pt = Vector((EASEL_POS.x - 0.015, EASEL_POS.y, bz))
    rainbow_targets.append(hit_pt)
    # Tapered optical fan frustum: narrow at prism exit (r1=0.0085) flaring wide (r2=0.022) at obsidian easel!
    rb = add_cone(f"RainbowRibbon_{b_i}", (0, 0, 0), 0.0085, 0.022, 1.0, rainbow_ribbon_mats[b_i], verts=22)
    rb.visible_shadow = False
    rb.visible_diffuse = False
    rb.visible_glossy = False
    rainbow_ribbons.append(rb)


# ============================================================================
# 10. BEAT 4 — BRASS NEWTON'S CRADLE ON RIGHT GARDEN BALUSTRADE (CRADLE_PLINTH)
# ============================================================================
# CRADLE_PLINTH = (2.88, 0.08, 0.90) -> Cradle Z spans 0.90..1.48!
CRADLE_BASE_Z = CRADLE_PLINTH.z
CRADLE_RAIL_Z = CRADLE_BASE_Z + 0.56  # 1.46m (top rail + arch reaches 1.48m!)
CRADLE_BALL_R = 0.036
CRADLE_WIRE_L = 0.260  # Rest ball center at Z = 1.46 - 0.26 = 1.20m!
CRADLE_HALF_Y = 0.085

add_box("CradleBaseLower", (CRADLE_PLINTH.x, CRADLE_PLINTH.y, CRADLE_BASE_Z + 0.016), (0.44, 0.24, 0.032), mahogany_dark_mat, bevel_w=0.006)
add_box("CradleBaseUpper", (CRADLE_PLINTH.x, CRADLE_PLINTH.y, CRADLE_BASE_Z + 0.040), (0.39, 0.20, 0.018), mahogany_mat, bevel_w=0.004)
add_box("CradleBrassTrim", (CRADLE_PLINTH.x, CRADLE_PLINTH.y - 0.121, CRADLE_BASE_Z + 0.016), (0.442, 0.008, 0.012), brass_mat, bevel_w=0.002)

for sgn_y in (-1.0, 1.0):
    ry = CRADLE_PLINTH.y + sgn_y * CRADLE_HALF_Y
    build_curve_tube(
        f"CradleArchRail_{int(sgn_y)}",
        [
            (CRADLE_PLINTH.x - 0.195, ry, CRADLE_BASE_Z + 0.044, 0.011),
            (CRADLE_PLINTH.x - 0.195, ry, CRADLE_BASE_Z + 0.500, 0.010),
            (CRADLE_PLINTH.x - 0.165, ry, CRADLE_RAIL_Z, 0.010),
            (CRADLE_PLINTH.x + 0.165, ry, CRADLE_RAIL_Z, 0.010),
            (CRADLE_PLINTH.x + 0.195, ry, CRADLE_BASE_Z + 0.500, 0.010),
            (CRADLE_PLINTH.x + 0.195, ry, CRADLE_BASE_Z + 0.044, 0.011),
        ],
        brass_mat,
        bevel_res=6,
    )
    for sgn_x in (-1.0, 1.0):
        add_cyl(
            f"CradleFootCollar_{int(sgn_y)}_{int(sgn_x)}",
            (CRADLE_PLINTH.x + sgn_x * 0.195, ry, CRADLE_BASE_Z + 0.056),
            0.018,
            0.018,
            brass_mat,
            verts=20,
        )
for sgn_x in (-1.0, 1.0):
    add_cyl(
        f"CradleCrossBar_{int(sgn_x)}",
        (CRADLE_PLINTH.x + sgn_x * 0.165, CRADLE_PLINTH.y, CRADLE_RAIL_Z),
        0.008,
        CRADLE_HALF_Y * 2.0,
        brass_mat,
        rot=(math.pi * 0.5, 0.0, 0.0),
        verts=20,
    )

# 5 Mirror-Chrome Steel Spheres (r=0.036m, Roughness=0.03) suspended on Twin V-Wires
cradle_pivots = []
for b_i in range(5):
    bx_off = (b_i - 2.0) * (CRADLE_BALL_R * 2.005)
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(CRADLE_PLINTH.x + bx_off, CRADLE_PLINTH.y, CRADLE_RAIL_Z))
    piv = bpy.context.active_object
    piv.name = f"CradlePivot_{b_i}"
    piv.rotation_mode = "XYZ"

    ball = add_sphere(f"CradleChromeSphere_{b_i}", (0.0, 0.0, -CRADLE_WIRE_L), CRADLE_BALL_R, mirror_chrome_mat, segs=36, rings=24)
    ball.parent = piv
    eyelet = add_torus(f"CradleEyelet_{b_i}", (0.0, 0.0, -CRADLE_WIRE_L + CRADLE_BALL_R + 0.003), 0.006, 0.002, brass_mat, maj_seg=16, min_seg=6)
    eyelet.parent = piv

    for w_sgn in (-1.0, 1.0):
        p_top = Vector((0.0, w_sgn * CRADLE_HALF_Y, 0.0))
        p_bot = Vector((0.0, 0.0, -CRADLE_WIRE_L + CRADLE_BALL_R + 0.003))
        wire = add_cyl(f"CradleWire_{b_i}_{int(w_sgn)}", (0, 0, 0), 0.0018, 1.0, brass_mat, verts=10)
        orient_beam(wire, p_top, p_bot, 1.0)
        wire.parent = piv

    cradle_pivots.append(piv)

# 16 Golden Kinetic Sparks + Shock Ring erupting on exact Beat 4 Clack (t = 17.45s)!
CRADLE_BALL_Z = CRADLE_RAIL_Z - CRADLE_WIRE_L  # 1.20m
clack_sparks = []
for s_i in range(16):
    sp = add_sphere(
        f"CradleClackSpark_{s_i:02d}",
        (CRADLE_PLINTH.x + 0.144, CRADLE_PLINTH.y, CRADLE_BALL_Z),
        0.0115,
        gold_glow_mat if s_i % 3 != 2 else cyan_glow_mat,
        segs=10,
        rings=8,
    )
    sp.visible_shadow = False
    clack_sparks.append(sp)

clack_shock_ring = add_torus(
    "CradleShockRing",
    (CRADLE_PLINTH.x + 0.11, CRADLE_PLINTH.y, CRADLE_BALL_Z),
    0.050,
    0.0048,
    gold_glow_mat,
    rot=(0.0, math.pi * 0.5, 0.0),
    maj_seg=48,
    min_seg=10,
)
clack_shock_ring.visible_shadow = False


# ============================================================================
# 11. BEATS 5 & 6 — OUTDOOR REFLECTING TELESCOPE, GIGANTIC EARTH-MOON GRAVITY WELL,
#     ORBITAL ROCKET SLINGSHOT & SELF-LUMINOUS GOLD 3D SKY MONUMENT!
# ============================================================================
# Unveiled cleanly at t = 20.1s at ORBIT_ANCHOR = (0.64, 0.14, 0.0), gliding to X = +0.55 in Beat 6!
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(ORBIT_ANCHOR.x, ORBIT_ANCHOR.y, FLOOR_Z))
orbit_station_root = bpy.context.active_object
orbit_station_root.name = "OrbitStationRoot"

# Full-Height Brass Newtonian Reflecting Telescope on Tall Mahogany Outdoor Tripod (Z = 0.0..1.18!)
build_tripod_legs("TelescopeTripod", orbit_station_root, hub_z=0.86, foot_r=0.30, leg_r=0.015, leg_mat=mahogany_mat, collar_mat=brass_mat, az_offset_deg=90.0)
th_hub = add_cyl("TelescopeTripodHub", (0.0, 0.0, 0.88), 0.054, 0.068, brass_mat, verts=32, bevel_w=0.005)
th_hub.parent = orbit_station_root
th_yoke_base = add_box("TelescopeYokeBase", (0.0, 0.0, 0.935), (0.145, 0.064, 0.028), bronze_mat, bevel_w=0.004)
th_yoke_base.parent = orbit_station_root
th_yoke_l = add_box("TelescopeYokeArmL", (-0.062, 0.0, 1.005), (0.022, 0.044, 0.125), brass_mat, bevel_w=0.003)
th_yoke_l.parent = orbit_station_root
th_yoke_r = add_box("TelescopeYokeArmR", (0.062, 0.0, 1.005), (0.022, 0.044, 0.125), brass_mat, bevel_w=0.003)
th_yoke_r.parent = orbit_station_root

# Brass Reflecting Barrel aimed skyward (+28° elevation into the Starry Twilight Cosmos -> top Z = 1.18!)
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 1.055), rotation=(0.0, math.radians(-28.0), math.radians(12.0)))
tel_tube_piv = bpy.context.active_object
tel_tube_piv.name = "NewtonReflectingTelescopePivot"
tel_tube_piv.parent = orbit_station_root

t_barrel = add_cyl("TelescopeBrassTube", (0.0, 0.0, 0.0), 0.050, 0.44, brass_mat, rot=(0.0, math.pi * 0.5, 0.0), verts=36, bevel_w=0.003)
t_barrel.parent = tel_tube_piv
for tr_x in (-0.20, -0.07, 0.07, 0.205):
    tr_ring = add_torus(f"TelescopeRing_{tr_x:+.2f}", (tr_x, 0.0, 0.0), 0.053, 0.0060, bronze_mat, rot=(0.0, math.pi * 0.5, 0.0), maj_seg=36, min_seg=10)
    tr_ring.parent = tel_tube_piv
t_mirror = add_cyl("TelescopeSpeculumMirror", (-0.210, 0.0, 0.0), 0.042, 0.014, speculum_mirror_mat, rot=(0.0, math.pi * 0.5, 0.0), verts=32)
t_mirror.parent = tel_tube_piv
t_open = add_cyl("TelescopeFrontBaffle", (0.212, 0.0, 0.0), 0.044, 0.014, ebony_mat, rot=(0.0, math.pi * 0.5, 0.0), verts=32)
t_open.parent = tel_tube_piv
# Iconic Newtonian side-mounted eyepiece + brass finder scope
t_eye = add_cyl("TelescopeSideEyepiece", (0.145, -0.062, 0.012), 0.015, 0.062, brass_mat, rot=(math.pi * 0.5, 0.0, 0.0), verts=24, bevel_w=0.002)
t_eye.parent = tel_tube_piv
t_lens = add_sphere("TelescopeEyeGlass", (0.145, -0.094, 0.012), 0.012, flint_glass_mat, segs=18, rings=12)
t_lens.scale = (1.0, 0.45, 1.0)
t_lens.parent = tel_tube_piv
t_finder = add_cyl("TelescopeFinderScope", (0.02, 0.0, 0.072), 0.011, 0.20, bronze_mat, rot=(0.0, math.pi * 0.5, 0.0), verts=20)
t_finder.parent = tel_tube_piv

# GIGANTIC Luminous Sapphire-Gold 3D Curved Space-Time Gravity Well floating overhead (Z = 1.56 -> 1.50 in Beat 6)!
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 1.56), rotation=(math.radians(16.0), math.radians(-6.0), 0.0))
gravity_well_group = bpy.context.active_object
gravity_well_group.name = "CurvedSpacetimeGravityWell"
gravity_well_group.parent = orbit_station_root


def gravity_funnel_z(r: float) -> float:
    """Deep gravitational potential funnel plunging smoothly beneath procedural 3D Earth."""
    return 0.042 - 0.028 / (r + 0.072)


# 7 Concentric Warped Glowing Funnel Rings (electric sapphire & molten gold -> 0.59m diameter!)
well_radii = [0.080, 0.116, 0.152, 0.188, 0.224, 0.260, 0.296]
for wi, wr in enumerate(well_radii):
    wz = gravity_funnel_z(wr)
    wmat = gold_glow_mat if wi in (0, 1, 6) else cyan_glow_mat
    w_ring = add_torus(f"GravityWellRing_{wi}", (0.0, 0.0, wz), wr, 0.0034 if wi < 6 else 0.0050, wmat, maj_seg=64, min_seg=10)
    w_ring.visible_shadow = False
    w_ring.parent = gravity_well_group

# 12 Curved Space-Time Meridian Grid Ribs plunging down the gravitational potential throat!
for mi in range(12):
    ang = (mi / 12.0) * math.pi * 2.0
    ca, sa = math.cos(ang), math.sin(ang)
    pts = []
    for si in range(11):
        rr = lerp(0.068, 0.296, si / 10.0)
        pts.append((ca * rr, sa * rr, gravity_funnel_z(rr), 0.0024))
    rib = build_curve_tube(f"GravityWellMeridian_{mi:02d}", pts, cyan_glow_mat if mi % 2 == 0 else gold_glow_mat, bevel_res=3)
    rib.visible_shadow = False
    rib.parent = gravity_well_group

# High-Detail Procedural 3D Earth + Glowing Blue Fresnel Atmosphere Halo!
earth_obj = add_sphere("ProceduralEarth", (0.0, 0.0, 0.016), 0.102, earth_planet_mat, segs=48, rings=32)
earth_obj.parent = gravity_well_group
earth_halo = add_sphere("EarthAtmoHalo", (0.0, 0.0, 0.016), 0.111, earth_atmo_mat, segs=36, rings=24)
earth_halo.visible_shadow = False
earth_halo.parent = gravity_well_group

# Cratered 3D Moon Orbiting Rapidly Along Outer Golden Rim + Pulsing Golden Gravity Force Vector!
MOON_ORBIT_R = 0.288
MOON_ORBIT_Z = gravity_funnel_z(MOON_ORBIT_R) + 0.022
moon_obj = add_sphere("OrbitingMoon3D", (MOON_ORBIT_R, 0.0, MOON_ORBIT_Z), 0.032, cratered_moon_mat, segs=32, rings=20)
moon_obj.parent = gravity_well_group

gravity_tether = add_cyl("EarthMoonGravityBeam", (0, 0, 0), 0.0052, 1.0, gold_glow_mat, verts=16)
gravity_tether.visible_shadow = False
gravity_tether.parent = gravity_well_group

# BEAT 6 MODERN LEGACY: Sleek Metallic 3D Orbital Rocket Flying Along Tilted Golden Orbit Halo (Z = 2.04)!
ROCKET_ORBIT_R = 0.290
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.55, 0.14, 2.04), rotation=(math.radians(16.0), math.radians(-10.0), 0.0))
rocket_orbit_group = bpy.context.active_object
rocket_orbit_group.name = "ModernRocketOrbitGroup"

rocket_orbit_ring = add_torus("RocketGoldenOrbitRing", (0.0, 0.0, 0.0), ROCKET_ORBIT_R, 0.0046, gold_glow_mat, maj_seg=72, min_seg=10)
rocket_orbit_ring.visible_shadow = False
rocket_orbit_ring.parent = rocket_orbit_group

bpy.ops.object.empty_add(type="PLAIN_AXES", location=(ROCKET_ORBIT_R, 0.0, 0.0))
rocket_ship = bpy.context.active_object
rocket_ship.name = "ModernOrbitalRocket"
rocket_ship.parent = rocket_orbit_group

# Rocket oriented cleanly along local +Y (tangent to circular orbit at angle 0!)
r_body = add_cyl("RocketFuselage", (0.0, 0.0, 0.0), 0.025, 0.145, rocket_white_mat, rot=(math.pi * 0.5, 0.0, 0.0), verts=28)
r_body.parent = rocket_ship
r_band = add_cyl("RocketGoldStageBand", (0.0, 0.026, 0.0), 0.0262, 0.016, brass_mat, rot=(math.pi * 0.5, 0.0, 0.0), verts=28)
r_band.parent = rocket_ship
r_nose = add_cone("RocketNosecone", (0.0, 0.102, 0.0), 0.025, 0.001, 0.064, rocket_dark_mat, rot=(-math.pi * 0.5, 0.0, 0.0), verts=28)
r_nose.parent = rocket_ship
r_nozzle = add_cone("RocketEngineNozzle", (0.0, -0.084, 0.0), 0.018, 0.011, 0.025, rocket_dark_mat, rot=(-math.pi * 0.5, 0.0, 0.0), verts=20)
r_nozzle.parent = rocket_ship
for fin_i in range(3):
    fa = math.radians(fin_i * 120.0)
    fin = add_box(
        f"RocketFin_{fin_i}",
        (math.cos(fa) * 0.030, -0.050, math.sin(fa) * 0.030),
        (0.030, 0.048, 0.0045),
        rocket_dark_mat,
        rot=(0.0, -fa, 0.0),
        bevel_w=0.001,
    )
    fin.parent = rocket_ship
r_plume = add_cone("RocketPlasmaThruster", (0.0, -0.148, 0.0), 0.001, 0.021, 0.112, rocket_plume_mat, rot=(-math.pi * 0.5, 0.0, 0.0), verts=20)
r_plume.visible_shadow = False
r_plume.parent = rocket_ship
r_core = add_cone("RocketCoreFlame", (0.0, -0.116, 0.0), 0.001, 0.012, 0.054, rocket_core_mat, rot=(-math.pi * 0.5, 0.0, 0.0), verts=16)
r_core.visible_shadow = False
r_core.parent = rocket_ship

# Hero 2-Line Self-Luminous Cinema Gold 3D Sky Monument floating cleanly at (0.55, 0.60, 2.52)!
bpy.ops.object.text_add(location=(0.55, 0.60, 2.52), rotation=(math.pi * 0.5, 0.0, 0.0))
title_txt = bpy.context.active_object
title_txt.name = "NewtonTitle3D"
if HERO_FONT is not None:
    title_txt.data.font = HERO_FONT
title_txt.data.body = "ISAAC NEWTON\nF = G m1 m2 / r²"
title_txt.data.align_x = "CENTER"
title_txt.data.align_y = "CENTER"
title_txt.data.size = 0.084
title_txt.data.space_character = 1.12
title_txt.data.space_line = 1.28
title_txt.data.extrude = 0.0070
title_txt.data.bevel_depth = 0.0007
title_txt.data.materials.append(title_gold_mat)


# ============================================================================
# 12. REALISTIC ADULT MPFB2 V7 4K HUMAN CAST (1.78m NEWTON & 1.79m HALLEY!)
# ============================================================================
# CRITICAL V7 CAST FIX: BOTH Newton & Halley are FULL ADULT HEIGHT (age >= 0.56, 1.78m & 1.79m!)
# Standing on mossy Cotswold flagstones at FLOOR_Z = 0.0 with period-authentic 17th-c cravat & scholar cloak!
NEWTON_ACTOR = None
HALLEY_ACTOR = None
try:
    NEWTON_ACTOR = create_realistic_scientist(
        name="IsaacNewton",
        floor_z=FLOOR_Z,
        start_xy=(-1.06, 0.36),
        yaw_deg=26.0,
        cfg={
            "age": 0.58,
            "muscle": 0.52,
            "weight": 0.46,
            "height": 0.58,
            "target_height_m": 1.78,
            "skin_assets": ["middleage_caucasian_male.mhmat"],
            "iris_major_rgb": (0.14, 0.36, 0.56),
            "iris_minor_rgb": (0.06, 0.14, 0.26),
            "eyebrow_assets": ["eyebrow002.mhclo", "eyebrow001.mhclo"],
            "hair_assets": ["culturalibre_hair_01.mhclo", "bob02.mhclo"],
            # Lifts diffuse_black (meanRGB=0.07) into lustrous 1689 Kneller silver-pewter / warm ash-chestnut!
            "hair_gamma_lift": 0.34,
            "hair_value_boost": 1.28,
            "hair_tint_rgb": (0.82, 0.76, 0.68),
            "hair_tint_fac": 0.58,
            "brow_tint_rgb": (0.45, 0.38, 0.32),
            "brow_tint_fac": 0.48,
            "silver_hair": False,
            "silver_hair_main": False,
            # Authentic 17th-century scholar doublet + wool trousers + leather boots (ZERO modern tie!)
            "shirt_assets": ["elvs_male_boho_top1.mhclo"],
            "shirt_gamma_lift": 0.72,
            "shirt_tint_rgb": (0.30, 0.42, 0.78),
            "shirt_tint_fac": 0.55,
            "pants_assets": ["mindfront_male_trousers_2.mhclo", "toigo_wool_pants.mhclo"],
            "pants_tint_rgb": (0.16, 0.19, 0.28),
            "pants_tint_fac": 0.68,
            "shoe_assets": ["toigo_ankle_boots_male.mhclo", "punkduck_medieval_boots.mhclo"],
            # Rigged 1689 Kneller white silk-linen ruffled jabot cravat + draped Cambridge velvet scholar cloak!
            "add_cravat": True,
            "cravat_style": "jabot",
            "cravat_rgb": (0.93, 0.91, 0.86),
            "add_scholar_cloak": True,
            "cloak_style": "cambridge_scholar",
            "cloak_rgb": (0.042, 0.068, 0.175),
            "cloak_trim_rgb": (0.72, 0.54, 0.22),
            "cloak_clasp": False,
        },
        brass_mat=brass_mat,
    )
    HALLEY_ACTOR = create_realistic_scientist(
        name="EdmondHalley",
        floor_z=FLOOR_Z,
        start_xy=(1.05, 0.45),
        yaw_deg=-28.0,
        cfg={
            "age": 0.60,
            "muscle": 0.58,
            "weight": 0.52,
            "height": 0.60,
            "target_height_m": 1.79,
            "skin_assets": ["middleage_caucasian_male.mhmat"],
            "iris_major_rgb": (0.32, 0.21, 0.10),
            "iris_minor_rgb": (0.12, 0.08, 0.04),
            "eyebrow_assets": ["eyebrow003.mhclo", "eyebrow002.mhclo"],
            "hair_assets": ["short04.mhclo", "short02.mhclo"],
            # Distinguished dark espresso-chestnut gentleman hair
            "hair_gamma_lift": 0.68,
            "hair_value_boost": 1.05,
            "hair_tint_rgb": (0.42, 0.26, 0.16),
            "hair_tint_fac": 0.62,
            "brow_tint_rgb": (0.30, 0.20, 0.12),
            "brow_tint_fac": 0.50,
            "silver_hair": False,
            # Authentic 17th-century Royal Society & HMS Paramour Sea Captain attire (ZERO modern bowtie!)
            "shirt_assets": ["elvs_male_shirt_untucked_bd1.mhclo"],
            "shirt_gamma_lift": 0.58,
            "shirt_value_boost": 1.15,
            "shirt_tint_rgb": (0.68, 0.16, 0.18),
            "shirt_tint_fac": 0.78,
            "pants_assets": ["mindfront_male_trousers_2.mhclo", "toigo_wool_pants.mhclo"],
            "pants_tint_rgb": (0.20, 0.17, 0.15),
            "pants_tint_fac": 0.68,
            "shoe_assets": ["punkduck_medieval_boots.mhclo", "culturalibre_male_boots.mhclo"],
            # Rigged 17th-century folded Steinkirk white linen cravat + Royal Claret velvet shoulder-mantle w/ brass clasps!
            "add_cravat": True,
            "cravat_style": "tied",
            "cravat_rgb": (0.91, 0.89, 0.84),
            "add_scholar_cloak": True,
            "cloak_style": "captain_mantle",
            "cloak_rgb": (0.22, 0.036, 0.046),
            "cloak_trim_rgb": (0.76, 0.58, 0.24),
            "cloak_clasp": True,
        },
        brass_mat=brass_mat,
    )
except Exception as mpfb_err:
    print(f"[MPFB2 Warning] Skipping human cast ({mpfb_err}) — install MPFB2 & packs in Cell 2!")


# ============================================================================
# 13. WORLD LIGHTING HANDLES & 9:16 PORTRAIT CINEMA CAMERA
# ============================================================================
# Reuse newton_world_v7.py's calibrated Golden-Hour -> Starry Twilight 5-light rig (ZERO duplicate over-exposure!)
# and guide key/rim/fill/catch positions smoothly along the 6.6m outdoor terrace to track the active Beat!
w_key_light = world_handles.get("key_light")
w_rim_light = world_handles.get("rim_light")
w_fill_light = world_handles.get("fill_light")
w_catch_light = world_handles.get("catch_light")
w_principia_plinth = world_handles.get("principia_plinth")
for _wl in (w_key_light, w_rim_light, w_fill_light):
    if _wl is not None:
        _wl.rotation_mode = "QUATERNION"

bpy.ops.object.empty_add(location=(-0.79, 0.24, 1.32))
cam_target = bpy.context.active_object
cam_target.name = "CinemaCamTarget"

bpy.ops.object.camera_add(location=(-0.82, -1.78, 1.24))
cam = bpy.context.active_object
cam.name = "CinemaCamera"
scene.camera = cam
cam.data.sensor_fit = "HORIZONTAL"
cam.data.sensor_width = 36.0
cam.data.lens = 38.0
cam.data.dof.use_dof = True
cam.data.dof.focus_object = cam_target
cam.data.dof.aperture_fstop = 2.8
cam.rotation_mode = "QUATERNION"


# ============================================================================
# 14. 6-BEAT FULL-3/4-BODY OUTDOOR-TO-COSMOS CAMERA (1:1 SYNC WITH verify_framing.py!)
# ============================================================================
def compute_beat_raw(beat_idx: int, t: float):
    """Return (cam_pos, tgt_pos, lens_mm, fstop, dutch_deg) — synchronized 1:1 with src/verify_framing.py!"""
    if beat_idx == 0:
        # BEAT 1 (0.0..4.5s): Golden-Hour Woolsthorpe Apple Orchard Low-Angle Hero Portrait!
        u = clamp01(t / 4.5)
        pos = bezier_quad(Vector((-0.82, -1.78, 1.24)), Vector((-0.79, -1.70, 1.27)), Vector((-0.76, -1.62, 1.30)), u)
        tgt = vlerp(Vector((-0.79, 0.24, 1.32)), Vector((-0.78, 0.22, 1.24)), u)
        thud = math.sin((t - 1.45) * 44.0) * math.exp(-(t - 1.45) * 8.0) * 0.011 if t >= 1.45 else 0.0
        pos.z += thud
        return pos, tgt, lerp(38.0, 40.0, u), 2.8, 0.0
    if beat_idx == 1:
        # BEAT 2 (4.5..9.5s): Sunlit Stone Courtyard Face-to-Face Debate (BOTH Newton 0.05 & Halley 1.05 eye-to-eye!)
        pan_u = ease_out_quint(clamp01((t - 6.95) / 1.1))
        pos = vlerp(Vector((0.62, -2.02, 1.42)), Vector((0.52, -1.94, 1.42)), pan_u)
        tgt = vlerp(Vector((0.58, 0.28, 1.32)), Vector((0.54, 0.30, 1.32)), pan_u)
        return pos, tgt, lerp(36.0, 35.5, pan_u), 2.8, 0.0
    if beat_idx == 2:
        # BEAT 3 (9.5..15.5s): Shaded Stone Arch Optical Alcove — Crystal Prism & 7-Spectral Rainbow!
        u = clamp01((t - 9.5) / 6.0)
        pos = bezier_quad(Vector((-2.54, -1.68, 1.40)), Vector((-2.51, -1.58, 1.41)), Vector((-2.48, -1.52, 1.42)), u)
        tgt = vlerp(Vector((-2.52, 0.18, 1.32)), Vector((-2.50, 0.20, 1.36)), smoothstep(10.6, 13.0, t))
        return pos, tgt, lerp(38.5, 40.5, u), 2.8, 0.0
    if beat_idx == 3:
        # BEAT 4 (15.5..20.5s): 3.8° Dutch-Tilt Comedy — Brass Newton's Cradle on Right Garden Balustrade!
        u = clamp01((t - 15.5) / 5.0)
        pos = bezier_quad(Vector((2.62, -1.78, 1.38)), Vector((2.66, -1.72, 1.39)), Vector((2.70, -1.70, 1.40)), u)
        tgt = Vector((2.62, 0.22, 1.32))
        shake = math.sin((t - 17.45) * 54.0) * math.exp(-(t - 17.45) * 6.5) * 0.025 if t >= 17.45 else 0.0
        pos.x += shake
        pos.z += shake * 0.45
        return pos, tgt, lerp(36.0, 38.0, u), 2.8, 3.8 * math.sin(u * math.pi)
    if beat_idx == 4:
        # BEAT 5 (20.5..25.5s): Starry Twilight Transformation — Rising Crane on Telescope & Giant Gravity Well!
        u = clamp01((t - 20.5) / 5.0)
        pos = bezier_quad(Vector((0.30, -1.74, 1.34)), Vector((0.34, -1.68, 1.42)), Vector((0.38, -1.66, 1.50)), u)
        tgt = vlerp(Vector((0.34, 0.24, 1.44)), Vector((0.36, 0.26, 1.52)), u)
        return pos, tgt, lerp(37.0, 38.5, u), 3.0, 0.0
    # BEAT 6 (25.5..30.0s): Epic Moonlit Orchard-to-Cosmos Rocket Slingshot Grand Finale Wide Pull-Back!
    u = clamp01((t - 25.5) / 4.5)
    pos = bezier_quad(Vector((0.55, -2.35, 1.52)), Vector((0.55, -2.52, 1.58)), Vector((0.55, -2.68, 1.62)), u)
    tgt = vlerp(Vector((0.55, 0.30, 1.54)), Vector((0.55, 0.28, 1.52)), u)
    return pos, tgt, lerp(34.0, 32.0, u), 3.6, 0.0


BEAT_BOUNDS = [0.0, 4.5, 9.5, 15.5, 20.5, 25.5, 30.0]


def compute_camera_state(t: float):
    """Smooth single-take cinema camera with 0.42s easeOutQuint whip-pans synced to exact whip_whoosh SFX!"""
    sway_x = math.sin(t * 1.15) * 0.008 + math.sin(t * 2.40) * 0.003
    sway_z = math.cos(t * 0.92) * 0.006

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
# 15. KEYFRAME ALL 720 FRAMES DETERMINISTICALLY (EXACT 30.000s AUDIO SYNC!)
# ============================================================================
for f in range(1, TOTAL_FRAMES + 1):
    t = (f - 1) / FPS

    # 0. Animate Woolsthorpe Orchard -> Starry Twilight Cosmos world (sky, sun, moon, stars, breeze, plinths!)
    animate_world_frame(world_handles, f, t)

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

    # 2. Smoothly Guide World Rembrandt Key, Rim, Fill & Wet-Cornea Catchlight Along the Active Terrace Station!
    # Preserves 100% of animate_world_frame()'s calibrated Golden-Hour -> Beat 3 Shade -> Starry Twilight energies/colors!
    alcove_u = smoothstep(9.35, 9.85, t) * (1.0 - smoothstep(15.20, 15.70, t))
    if w_key_light is not None:
        key_off_x = lerp(-1.85, 1.35, alcove_u)
        k_pos = Vector((c_tgt.x + key_off_x, c_pos.y - 0.55, 2.48))
        w_key_light.location = k_pos
        w_key_light.rotation_quaternion = look_at_quat(k_pos, Vector((c_tgt.x, c_tgt.y + 0.15, 1.48)))
        w_key_light.keyframe_insert("location", frame=f)
        w_key_light.keyframe_insert("rotation_quaternion", frame=f)
    if w_rim_light is not None:
        r_pos = Vector((c_tgt.x + 1.45, c_tgt.y + 2.45, 2.75))
        w_rim_light.location = r_pos
        w_rim_light.rotation_quaternion = look_at_quat(r_pos, Vector((c_tgt.x, c_tgt.y, 1.46)))
        w_rim_light.keyframe_insert("location", frame=f)
        w_rim_light.keyframe_insert("rotation_quaternion", frame=f)
    if w_fill_light is not None:
        f_pos = Vector((c_tgt.x + 0.85, c_pos.y - 0.45, 1.65))
        w_fill_light.location = f_pos
        w_fill_light.rotation_quaternion = look_at_quat(f_pos, Vector((c_tgt.x, c_tgt.y, 1.38)))
        w_fill_light.keyframe_insert("location", frame=f)
        w_fill_light.keyframe_insert("rotation_quaternion", frame=f)
    if w_catch_light is not None:
        w_catch_light.location = Vector((c_pos.x + 0.16, c_pos.y + 0.52, c_pos.z + 0.22))
        w_catch_light.keyframe_insert("location", frame=f)

    # 3. BEAT 1: Hero Woolsthorpe Apple Gravity Fall (`HERO_APPLE_ATTACH` Z=1.94 -> `APPLE_PLINTH + 0.058` Z=0.938)!
    z_hang = HERO_APPLE_ATTACH.z  # 1.94m
    z_pad = APPLE_PLINTH.z + 0.058  # 0.938m (resting cleanly on velvet pad!)
    if t < 0.65:
        # Gentle golden-hour orchard breeze sway attached to overhanging branch before twig snap at t=0.65s
        az = z_hang + math.sin(t * 3.1) * 0.004
        ax = HERO_APPLE_ATTACH.x + math.sin(t * 2.2) * 0.007
        arot = Euler((0.0, math.sin(t * 2.2) * 0.09, t * 0.16))
    elif t < 1.45:
        # Exact constant-acceleration gravitational free fall (`1/2 g t²`) from t=0.65s to t=1.45s!
        u_f = (t - 0.65) / 0.80
        az = z_hang - (z_hang - z_pad) * (u_f * u_f)
        ax = HERO_APPLE_ATTACH.x
        arot = Euler((u_f * 0.48, -u_f * 0.32, u_f * 0.72))
    else:
        dt_b = t - 1.45
        if dt_b < 0.34:
            hb = math.sin((dt_b / 0.34) * math.pi) * 0.068
        elif dt_b < 0.54:
            hb = math.sin(((dt_b - 0.34) / 0.20) * math.pi) * 0.020
        else:
            hb = 0.0
        az = z_pad + hb
        ax = HERO_APPLE_ATTACH.x + min(0.016, dt_b * 0.028)
        arot = Euler((0.48 + min(0.20, dt_b * 0.32), -0.32, 0.72 + min(0.32, dt_b * 0.48)))
    hero_apple.location = Vector((ax, HERO_APPLE_ATTACH.y, az))
    hero_apple.rotation_euler = arot
    hero_apple.keyframe_insert("location", frame=f)
    hero_apple.keyframe_insert("rotation_euler", frame=f)

    # Golden Downward Gravity Force Arrow ('g', Z = 1.15..1.52) & Expanding Energy Ripple Rings!
    g_vis = smoothstep(1.50, 1.85, t) * (1.0 - smoothstep(4.25, 4.55, t))
    g_sc = max(0.0001, g_vis)
    gravity_arrow_root.scale = Vector((g_sc, g_sc, g_sc))
    gravity_arrow_root.location = Vector((G_ARROW_POS.x, G_ARROW_POS.y, G_ARROW_POS.z - math.sin(t * 4.2) * 0.015))
    gravity_arrow_root.keyframe_insert("scale", frame=f)
    gravity_arrow_root.keyframe_insert("location", frame=f)

    if 1.45 <= t <= 3.15:
        rp_u = (t - 1.45) / 1.70
        impact_ripple_1.scale = Vector((0.55 + rp_u * 2.25, 0.55 + rp_u * 2.25, max(0.0001, 1.0 - rp_u)))
    else:
        impact_ripple_1.scale = Vector((0.0001, 0.0001, 0.0001))
    impact_ripple_1.keyframe_insert("scale", frame=f)

    if 1.72 <= t <= 3.30:
        rp2_u = (t - 1.72) / 1.58
        impact_ripple_2.scale = Vector((0.45 + rp2_u * 1.95, 0.45 + rp2_u * 1.95, max(0.0001, 1.0 - rp2_u)))
    else:
        impact_ripple_2.scale = Vector((0.0001, 0.0001, 0.0001))
    impact_ripple_2.keyframe_insert("scale", frame=f)

    # 4. BEAT 2: Principia Pedestal & Props (retract cleanly off-screen before Beat 5 so Beats 5-6 are 100% unobstructed!)
    p_props_vis = max(0.0001, 1.0 - smoothstep(19.65, 20.05, t))
    principia_props_root.scale = Vector((p_props_vis, p_props_vis, p_props_vis))
    principia_props_root.keyframe_insert("scale", frame=f)
    if w_principia_plinth is not None:
        w_principia_plinth.scale = Vector((p_props_vis, p_props_vis, p_props_vis))
        w_principia_plinth.keyframe_insert("scale", frame=f)

    fma_vis = smoothstep(4.55, 5.02, t) * (1.0 - smoothstep(6.82, 7.02, t))
    fma_sc = max(0.0001, fma_vis)
    formula_fma_txt.scale = Vector((fma_sc, fma_sc, fma_sc))
    formula_fma_txt.location = Vector((PRINCIPIA_PLINTH.x, -0.02, 1.22 + math.sin(t * 2.5) * 0.010))
    formula_fma_txt.keyframe_insert("scale", frame=f)
    formula_fma_txt.keyframe_insert("location", frame=f)

    fgrav_vis = smoothstep(6.95, 7.32, t) * (1.0 - smoothstep(9.80, 10.20, t))
    fgrav_sc = max(0.0001, fgrav_vis)
    formula_grav_txt.scale = Vector((fgrav_sc, fgrav_sc, fgrav_sc))
    formula_grav_txt.location = Vector((PRINCIPIA_PLINTH.x, -0.02, 1.26 + math.sin(t * 2.3) * 0.010))
    formula_grav_txt.keyframe_insert("scale", frame=f)
    formula_grav_txt.keyframe_insert("location", frame=f)

    # 5. BEAT 3: Shaded Optical Alcove — Sunbeam from `PRISM_PINHOLE` -> Crystal Prism -> 7 Wide Rainbow Ribbons!
    sun_u = smoothstep(9.75, 10.45, t) * (1.0 - smoothstep(15.35, 15.80, t))
    flare_sc = max(0.0001, sun_u * (1.0 + 0.08 * math.sin(t * 8.0)))
    pinhole_flare.scale = Vector((flare_sc, flare_sc, flare_sc))
    pinhole_flare.keyframe_insert("scale", frame=f)

    if sun_u > 0.002:
        p_sun_end = vlerp(PRISM_PINHOLE, PRISM_ENTRY, smoothstep(9.75, 10.40, t))
        orient_beam(white_sunbeam, PRISM_PINHOLE, p_sun_end, radius_scale=max(0.001, sun_u))
        int_u = smoothstep(10.30, 10.60, t) * (1.0 - smoothstep(15.35, 15.80, t))
        orient_beam(internal_beam, PRISM_ENTRY, PRISM_EXIT, radius_scale=max(0.001, int_u))
    else:
        white_sunbeam.scale = Vector((0.0001, 0.0001, 0.0001))
        internal_beam.scale = Vector((0.0001, 0.0001, 0.0001))
    for obj_ray in (white_sunbeam, internal_beam):
        obj_ray.keyframe_insert("location", frame=f)
        obj_ray.keyframe_insert("scale", frame=f)
        obj_ray.keyframe_insert("rotation_quaternion", frame=f)

    for b_i, (rb, b_hit, sband) in enumerate(zip(rainbow_ribbons, rainbow_targets, spectral_bands)):
        # Bloom smoothly & sequentially Red (10.60s) -> Violet (11.80s) synced to prism_rainbow_bloom SFX!
        t_start = 10.60 + b_i * 0.12
        t_end = t_start + 0.48
        ext_u = smoothstep(t_start, t_end, t)
        fade_out = 1.0 - smoothstep(15.35, 15.80, t)
        vis_b = ext_u * fade_out
        shimmer = 1.0 + 0.08 * math.sin(t * 7.5 + b_i * 0.9)
        if vis_b > 0.002:
            cur_end = vlerp(PRISM_EXIT, b_hit, ext_u)
            # Wide optical light ribbon (ribbon_y_mult = 2.1 along depth Y so it looks like a glowing dispersion sheet!)
            orient_beam(rb, PRISM_EXIT, cur_end, radius_scale=max(0.001, vis_b * shimmer), ribbon_y_mult=2.1)
        else:
            rb.scale = Vector((0.0001, 0.0001, 0.0001))
        rb.keyframe_insert("location", frame=f)
        rb.keyframe_insert("scale", frame=f)
        rb.keyframe_insert("rotation_quaternion", frame=f)

        band_sc = max(0.0001, smoothstep(t_end - 0.10, t_end + 0.15, t) * fade_out)
        sband.scale = Vector((1.0, band_sc, band_sc))
        sband.keyframe_insert("scale", frame=f)

    # 6. BEAT 4: Brass Newton's Cradle Exact Pendulum Clack Physics (t = 17.45s, 18.25s, 19.10s)!
    ang_b0 = 0.0
    ang_b4 = 0.0
    if 15.70 <= t < 17.10:
        ang_b0 = math.radians(48.0) * smoothstep(15.70, 17.05, t)
    elif 17.10 <= t < 17.45:
        u_d = (t - 17.10) / 0.35
        ang_b0 = math.radians(48.0) * math.cos(u_d * math.pi * 0.5)
    elif 17.45 <= t < 18.25:
        u_s = (t - 17.45) / 0.80
        ang_b4 = -math.radians(62.0) * math.sin(u_s * math.pi)
    elif 18.25 <= t < 19.10:
        u_s = (t - 18.25) / 0.85
        ang_b0 = math.radians(54.0) * math.sin(u_s * math.pi)
    elif 19.10 <= t < 19.95:
        u_s = (t - 19.10) / 0.85
        ang_b4 = -math.radians(46.0) * math.sin(u_s * math.pi)
    elif 19.95 <= t < 20.60:
        u_s = (t - 19.95) / 0.65
        ang_b0 = math.radians(34.0) * math.sin(u_s * math.pi)

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

    # 16 Golden Kinetic Sparks + Glowing Shock Ring Burst at exact t = 17.45s CLACK!
    dt_ck = t - 17.45
    for s_i, sp in enumerate(clack_sparks):
        if 0.0 <= dt_ck < 1.35:
            u_sp = dt_ck / 1.35
            ang = (s_i / 16.0) * math.pi * 2.0
            rad = 0.03 + u_sp * (0.32 + (s_i % 4) * 0.07)
            sx = CRADLE_PLINTH.x + 0.12 + math.cos(ang) * rad * 0.95
            sy = CRADLE_PLINTH.y + math.sin(ang * 1.4) * rad * 0.55
            sz = CRADLE_BALL_Z + u_sp * (0.34 - u_sp * 0.58)
            sc = max(0.0001, (1.0 - u_sp) * 0.98)
            sp.location = Vector((sx, sy, max(CRADLE_BASE_Z + 0.04, sz)))
            sp.scale = Vector((sc, sc, sc))
        else:
            sp.scale = Vector((0.0001, 0.0001, 0.0001))
        sp.keyframe_insert("location", frame=f)
        sp.keyframe_insert("scale", frame=f)

    if 0.0 <= dt_ck < 0.85:
        u_sr = dt_ck / 0.85
        sr_rad = 0.65 + u_sr * 2.75
        sr_th = max(0.0001, 1.0 - u_sr)
        clack_shock_ring.scale = Vector((sr_rad, sr_rad, sr_th))
    else:
        clack_shock_ring.scale = Vector((0.0001, 0.0001, 0.0001))
    clack_shock_ring.keyframe_insert("scale", frame=f)

    # 7. BEAT 5 & 6: Outdoor Reflecting Telescope + GIGANTIC Earth-Moon Gravity Well (`X=0.64 -> 0.55`, `Z=1.56 -> 1.50`)!
    orb_vis = max(0.0001, smoothstep(20.10, 20.65, t))
    orb_center_u = smoothstep(25.30, 26.05, t)
    orbit_station_root.scale = Vector((orb_vis, orb_vis, orb_vis))
    orbit_station_root.location = Vector((lerp(ORBIT_ANCHOR.x, 0.55, orb_center_u), ORBIT_ANCHOR.y, FLOOR_Z))
    orbit_station_root.keyframe_insert("scale", frame=f)
    orbit_station_root.keyframe_insert("location", frame=f)

    gravity_well_group.location = Vector((0.0, 0.0, lerp(1.56, 1.50, orb_center_u)))
    gravity_well_group.keyframe_insert("location", frame=f)

    earth_obj.rotation_euler = Euler((0.0, math.radians(23.5), t * 0.72))
    earth_obj.keyframe_insert("rotation_euler", frame=f)

    moon_ang = (t - 20.5) * 1.90 + 0.45
    moon_local = Vector((math.cos(moon_ang) * MOON_ORBIT_R, math.sin(moon_ang) * MOON_ORBIT_R, MOON_ORBIT_Z))
    moon_obj.location = moon_local
    moon_obj.rotation_euler = Euler((0.0, 0.0, moon_ang))
    moon_obj.keyframe_insert("location", frame=f)
    moon_obj.keyframe_insert("rotation_euler", frame=f)

    tether_pulse = 0.85 + 0.35 * math.sin(t * 7.5)
    orient_beam(gravity_tether, Vector((0.0, 0.0, 0.016)), moon_local, radius_scale=tether_pulse)
    gravity_tether.keyframe_insert("location", frame=f)
    gravity_tether.keyframe_insert("scale", frame=f)
    gravity_tether.keyframe_insert("rotation_quaternion", frame=f)

    # 8. BEAT 6 MODERN LEGACY: Sleek Metallic 3D Orbital Rocket + Self-Luminous Cinema Gold 3D Sky Monument!
    rkt_u = smoothstep(25.55, 26.35, t)
    rkt_sc = max(0.0001, rkt_u)
    rocket_orbit_group.scale = Vector((rkt_sc, rkt_sc, rkt_sc))
    rocket_orbit_group.keyframe_insert("scale", frame=f)

    rkt_ang = (t - 25.5) * 1.55 - 0.65
    rocket_ship.location = Vector((math.cos(rkt_ang) * ROCKET_ORBIT_R, math.sin(rkt_ang) * ROCKET_ORBIT_R, 0.0))
    rocket_ship.rotation_euler = Euler((0.0, math.radians(-14.0), rkt_ang))
    rocket_ship.keyframe_insert("location", frame=f)
    rocket_ship.keyframe_insert("rotation_euler", frame=f)

    plume_flick = 0.90 + 0.25 * math.sin(t * 44.0)
    r_plume.scale = Vector((1.0, 1.0, plume_flick))
    r_plume.keyframe_insert("scale", frame=f)

    tit_u = smoothstep(25.60, 26.45, t)
    tit_sc = max(0.0001, tit_u)
    title_txt.scale = Vector((tit_sc, tit_sc, tit_sc))
    title_txt.keyframe_insert("scale", frame=f)

    # 9. FULL 3/4-BODY ADULT MPFB2 CAST ACTING ON MOSSY FLAGSTONES (`FLOOR_Z = 0.0`)!
    # TRUE MPFB2 +Z FACING: +yaw/+head_rz = LOOK RIGHT (+X), -yaw/-head_rz = LOOK LEFT (-X);
    # head_rx < 0 tilts chin UP (sky/branch), head_rx > 0 tilts chin DOWN (plinth/falling apple)!
    breath = math.sin(t * 2.25) * 1.1
    if NEWTON_ACTOR is not None:
        nrig = NEWTON_ACTOR["rig"]
        n_lip = sample_lip_sync("newton", t)
        n_base_z = NEWTON_ACTOR.get("base_z", FLOOR_Z)

        if t < 4.5:
            # Beat 1: Newton stands full 3/4 body (-1.06, 0.36, 0.0), yaw=+26°, looking UP (-18°) then tracking DOWN (+12°)!
            nx, ny, nyaw = -1.06, 0.36, 26.0
            fall_track = smoothstep(0.65, 1.45, t)
            wonder_up = smoothstep(2.55, 3.55, t)
            lean_x = lerp(-2.5, 4.0, fall_track) - wonder_up * 4.5 + breath * 0.4
            head_rx = lerp(-18.0, 12.0, fall_track) - wonder_up * 22.0
            head_rz = 14.0 + math.sin(t * 1.6) * 2.8
            ua_l = (lerp(-28.0, -20.0, fall_track), 20.0, lerp(-32.0, -24.0, fall_track))
            la_l = (lerp(-42.0, -34.0, fall_track) + math.sin(t * 2.8) * 3.0, 0.0, -18.0)
            ua_r = (-10.0, -28.0, 14.0)
            la_r = (-24.0, 0.0, 18.0)
            n_brow = lerp(0.25, 0.85, smoothstep(1.45, 2.20, t))
            n_smile = lerp(0.08, 0.48, wonder_up)
        elif t < 9.5:
            # Beat 2: Sunlit Stone Courtyard Debate — Newton (0.05, 0.45, 0.0), yaw=+28°, head_rz=+14° facing Halley eye-to-eye!
            nx, ny, nyaw = 0.05, 0.45, 28.0
            sp_u = smoothstep(6.90, 7.35, t)
            lean_x = 2.5 + sp_u * 3.2 + breath * 0.4
            head_rx = 3.0 - sp_u * 3.5
            head_rz = 14.0 + math.sin(t * 3.4) * 3.2 * sp_u
            ua_l = (lerp(-14.0, -30.0, sp_u), lerp(26.0, 18.0, sp_u), lerp(-20.0, -34.0, sp_u))
            la_l = (lerp(-28.0, -42.0, sp_u) + math.sin(t * 4.6) * 4.2 * sp_u, 0.0, lerp(-22.0, -16.0, sp_u))
            ua_r = (lerp(-10.0, -18.0, sp_u), lerp(-28.0, -24.0, sp_u), lerp(14.0, 22.0, sp_u))
            la_r = (lerp(-24.0, -32.0, sp_u), 0.0, lerp(18.0, 20.0, sp_u))
            n_brow = lerp(-0.15, 0.58, sp_u)
            n_smile = lerp(0.05, 0.55, sp_u)
        elif t < 15.5:
            # Beat 3: Shaded Stone Arch Alcove — Newton (-2.12, 0.42, 0.0), yaw=-28°, head_rz=-16° gazing LEFT in awe at rainbow!
            nx, ny, nyaw = -2.12, 0.42, -28.0
            bloom_u = smoothstep(10.60, 12.20, t)
            lean_x = 4.5 - bloom_u * 2.5 + breath * 0.4
            head_rx = 5.5 - bloom_u * 8.5
            head_rz = -16.0 + math.sin(t * 1.6) * 3.0
            ua_r = (lerp(-20.0, -32.0, bloom_u), lerp(-24.0, -18.0, bloom_u), lerp(24.0, 34.0, bloom_u))
            la_r = (lerp(-32.0, -40.0, bloom_u) + math.sin(t * 2.6) * 3.0, 0.0, lerp(20.0, 16.0, bloom_u))
            ua_l = (-10.0, 28.0, -14.0)
            la_l = (-24.0, 0.0, -18.0)
            n_brow = lerp(0.18, 0.88, bloom_u)
            n_smile = lerp(0.15, 0.58, bloom_u)
        elif t < 20.5:
            # Beat 4: Newton off-camera far left while Halley reacts to Newton's Cradle on right balustrade!
            nx, ny, nyaw = -0.60, 0.58, 22.0
            lean_x, head_rx, head_rz = 2.0, 3.0, 14.0
            ua_r, la_r = (-10.0, -28.0, 14.0), (-24.0, 0.0, 18.0)
            ua_l, la_l = (-10.0, 28.0, -14.0), (-24.0, 0.0, -18.0)
            n_brow, n_smile = 0.20, 0.40
        elif t < 25.5:
            # Beat 5: Starry Twilight — Newton (0.02, 0.45, 0.0), yaw=+24°, head_rx=-8°, head_rz=+14° gazing up-right in triumph!
            nx, ny, nyaw = 0.02, 0.45, 24.0
            lean_x = 2.0 + breath * 0.4
            head_rx = -8.0 + math.sin(t * 1.4) * 2.0
            head_rz = 14.0 + math.sin(t * 1.8) * 3.0
            ua_l = (-32.0 + math.sin(t * 2.2) * 3.0, 18.0, -34.0)
            la_l = (-38.0, 0.0, -16.0)
            ua_r = (-12.0, -28.0, 14.0)
            la_r = (-26.0, 0.0, 18.0)
            n_brow, n_smile = 0.68, 0.68
        else:
            # Beat 6: Moonlit Grand Finale — Newton (-0.05, 0.55, 0.0), yaw=+18° flanking cosmic centerpiece!
            nx, ny, nyaw = -0.05, 0.55, 18.0
            lean_x = 1.0 + breath * 0.4
            head_rx = -7.0 + math.sin(t * 1.3) * 2.0
            head_rz = 10.0 + math.sin(t * 1.5) * 2.8
            ua_l = (-28.0, 20.0, -30.0)
            la_l = (-36.0, 0.0, -18.0)
            ua_r = (-12.0, -28.0, 14.0)
            la_r = (-26.0, 0.0, 18.0)
            n_brow, n_smile = 0.58, 0.75

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
        h_base_z = HALLEY_ACTOR.get("base_z", FLOOR_Z)
        knee_bend_u = 0.0

        if t < 4.5:
            # Beat 1: Off-camera far right waiting cleanly for Beat 2
            hx_p, hy_p, hyaw = 2.40, 0.54, -22.0
            lean_x, head_rx, head_rz = 1.5, 3.0, -12.0
            ua_l, la_l = (-10.0, 28.0, -14.0), (-24.0, 0.0, -18.0)
            ua_r, la_r = (-10.0, -28.0, 14.0), (-24.0, 0.0, 18.0)
            jaw_bias, h_brow, h_smile = 0.0, -0.1, 0.0
        elif t < 9.5:
            # Beat 2: Sunlit Stone Courtyard — Halley (1.05, 0.45, 0.0), yaw=-28°, head_rz=-14° eye-to-eye with Newton!
            hx_p, hy_p, hyaw = 1.05, 0.45, -28.0
            ask = 1.0 - smoothstep(6.75, 7.20, t)
            lean_x = 4.5 * ask + breath * 0.4
            head_rx = 3.5 * ask - 2.0 * (1.0 - ask)
            head_rz = -14.0 + math.sin(t * 3.5) * 3.5 * ask
            ua_r = (lerp(-16.0, -30.0, ask), lerp(-26.0, -18.0, ask), lerp(20.0, 34.0, ask))
            la_r = (lerp(-28.0, -42.0, ask) + math.sin(t * 4.2) * 4.2 * ask, 0.0, lerp(22.0, 16.0, ask))
            ua_l = (-12.0, 28.0, -16.0)
            la_l = (-26.0, 0.0, -20.0)
            jaw_bias = 0.0
            h_brow = lerp(0.75, -0.45, ask)
            h_smile = lerp(0.15, -0.20, ask)
        elif t < 15.5:
            # Beat 3: Off-camera far right during Newton's Shaded Arch Optical Prism showcase
            hx_p, hy_p, hyaw = 1.85, 0.52, -18.0
            lean_x, head_rx, head_rz = 2.0, 4.0, -10.0
            ua_l, la_l = (-10.0, 28.0, -14.0), (-24.0, 0.0, -18.0)
            ua_r, la_r = (-10.0, -28.0, 14.0), (-24.0, 0.0, 18.0)
            jaw_bias, h_brow, h_smile = 0.0, 0.0, 0.0
        elif t < 20.5:
            # Beat 4: Halley (2.34, 0.42, 0.0), yaw=+24° looking RIGHT at Cradle -> FULL-BODY KNEE-BEND & TORSO RECOIL at 17.45s!
            shock = smoothstep(17.42, 17.68, t)
            recoil_osc = math.sin((t - 17.45) * 19.0) * math.exp(-(t - 17.45) * 3.4) * 4.5 if t >= 17.45 else 0.0
            knee_bend_u = shock * (1.0 - 0.35 * smoothstep(18.60, 20.20, t))
            hx_p = 2.34 - shock * 0.07
            hy_p = 0.42 + shock * 0.09
            hyaw = 24.0
            lean_x = lerp(6.0, -12.5, shock) + recoil_osc
            head_rx = lerp(7.0, -10.0, shock)
            head_rz = lerp(16.0, 14.0, shock) + math.sin(t * 6.2) * 3.5 * shock
            ua_l = (lerp(-24.0, -36.0, shock), lerp(22.0, 24.0, shock), lerp(-28.0, -38.0, shock))
            la_l = (lerp(-32.0, -56.0, shock), 0.0, lerp(-18.0, -34.0, shock))
            ua_r = (lerp(-14.0, -34.0, shock), lerp(-28.0, -24.0, shock), lerp(16.0, 36.0, shock))
            la_r = (lerp(-28.0, -56.0, shock), 0.0, lerp(20.0, 34.0, shock))
            jaw_bias = 13.0 * shock * (1.0 - smoothstep(18.10, 18.60, t))
            h_brow = lerp(-0.20, 1.0, shock)
            h_smile = lerp(0.30, -0.48, shock)
        elif t < 25.5:
            # Beat 5: Off-camera far right during Starry Twilight Gravity Well Climax
            hx_p, hy_p, hyaw = 2.40, 0.54, -20.0
            lean_x, head_rx, head_rz = 2.0, 3.0, -14.0
            ua_l, la_l = (-10.0, 28.0, -14.0), (-24.0, 0.0, -18.0)
            ua_r, la_r = (-10.0, -28.0, 14.0), (-24.0, 0.0, 18.0)
            jaw_bias, h_brow, h_smile = 0.0, 0.25, 0.15
        else:
            # Beat 6: Moonlit Grand Finale — Halley (1.15, 0.55, 0.0), yaw=-18° flanking cosmic centerpiece!
            hx_p, hy_p, hyaw = 1.15, 0.55, -18.0
            lean_x = 1.0 + breath * 0.4
            head_rx = -7.0 + math.sin(t * 1.4) * 2.0
            head_rz = -10.0 + math.sin(t * 1.6) * 2.8
            ua_r = (-28.0, -20.0, 30.0)
            la_r = (-36.0, 0.0, 18.0)
            ua_l = (-12.0, 28.0, -14.0)
            la_l = (-26.0, 0.0, -18.0)
            jaw_bias, h_brow, h_smile = 0.0, 0.58, 0.58

        # Exact trigonometric hip-drop keeps Halley's feet 100% grounded on FLOOR_Z=0.0 during Beat 4 knee-bend recoil!
        hip_drop = -0.056 * knee_bend_u
        hrig.location = Vector((hx_p, hy_p, h_base_z + hip_drop))
        hrig.rotation_euler = Euler((0.0, 0.0, math.radians(hyaw)))
        hrig.keyframe_insert("location", frame=f)
        hrig.keyframe_insert("rotation_euler", frame=f)

        # Full-body knee-bend leg articulation (zero rotation outside Beat 4 recoil!)
        thigh_rx = -20.0 * knee_bend_u
        shin_rx = 40.0 * knee_bend_u
        foot_rx = -20.0 * knee_bend_u
        for side in ("L", "R"):
            set_bone_char_rot(hrig, f"upperleg01.{side}", thigh_rx, 0.0, 0.0, frame=f)
            set_bone_char_rot(hrig, f"lowerleg01.{side}", shin_rx, 0.0, 0.0, frame=f)
            set_bone_char_rot(hrig, f"foot.{side}", foot_rx, 0.0, 0.0, frame=f)

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
# 16. EXECUTE RENDER (QA 6-BEAT PREVIEW OR PRODUCTION FRAME RANGE!)
# ============================================================================
if ARGS.qa_only:
    # Preview all 6 V7 Outdoor-to-Cosmos Beats (Orchard Apple+g, Courtyard Debate, Shaded Prism, Cradle Recoil, Twilight Well, Moonlit Rocket Finale!)
    qa_times = [2.6, 8.5, 12.8, 17.8, 23.2, 28.2]
    qa_frames = [max(1, min(TOTAL_FRAMES, int(round(ts * FPS)) + 1)) for ts in qa_times]
    print(f"[Newton V7 Outdoor-to-Cosmos QA] Rendering 6 Beat keyframes {qa_frames} ({FPS}fps, {ARGS.samples}spp, {ARGS.scale}% res) -> {OUT_DIR} ...")
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
        f"[Newton V7 Outdoor-to-Cosmos] Rendering frames {scene.frame_start}..{scene.frame_end} "
        f"({RES_X}x{RES_Y} @ {ARGS.scale}%, {FPS}fps, {ARGS.samples} spp + FOG_GLOW Bloom) -> {OUT_DIR} ..."
    )
    t_render_start = time.time()
    bpy.ops.render.render(animation=True)
    print(f"[Newton V7 Outdoor-to-Cosmos] Batch {scene.frame_start}..{scene.frame_end} complete in {(time.time()-t_render_start)/60:.1f} min!")
