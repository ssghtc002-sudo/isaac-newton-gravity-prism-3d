"""
How Stitches Work — 4K Medical Macro Cinema (Blender 4.2.3 LTS Cycles OPTIX + GPU OIDN)
Photoreal Recreation & Upgrade of the 253M-View Viral Short (AGENTS.md V10 Standard):
  - Beat 1 (0.0-6.0s): Curved 3D Stainless-Steel Surgical Needle weaving 6 continuous royal-blue nylon sutures across a living wound
  - Beat 2 (6.0-10.2s): Ultra-satisfying Zipper Pull closing the deforming 3D subsurface skin + Articulated Surgical Scissors SNIP!
  - Beat 3 (10.2-15.2s): Crimson inflammation blush wave + 140 glowing 3D repair cells swarming into the wound seam
  - Beat 4 (15.2-20.4s): 36 interwoven 3D golden collagen fiber strands bridging the gap and morphing into living pink new skin
  - Beat 5 (20.4-25.3s): 18 branching 3D ruby capillary blood vessels growing across the seam, pulsing, then sinking beneath the epidermis
  - Beat 6 (25.3-30.0s): Scissors snip + blue sutures smoothly sliding out -> seamless healed skin camera pullback!
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FONT_PATH = str(PROJECT_ROOT / "subtitles" / "fonts" / "DejaVuSans-Bold.ttf")

RES_X, RES_Y = 1080, 1920
FPS = 24
DURATION = 30.0
TOTAL_FRAMES = int(round(DURATION * FPS))  # 720 frames

# Anatomical Limb Macro Dimensions (meters — scaled cleanly for crisp Cycles DOF & SSS!)
LIMB_R = 0.22       # Macro hero scale (equivalent to hyper-macro 100mm lens!)
LIMB_LEN = 0.92
WOUND_HALF_LEN = 0.145
WOUND_HALF_W = 0.036  # Open wound half-width at center (Y=0)
NUM_LOOPS = 6         # 6 continuous criss-cross loops = 12 entry/exit skin passes!


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--frames-dir", default=str(PROJECT_ROOT / "render_frames"))
    p.add_argument("--start-frame", type=int, default=1)
    p.add_argument("--end-frame", type=int, default=TOTAL_FRAMES)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--scale", type=int, default=100)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--device", default="OPTIX")
    return p.parse_args(argv)


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def smoothstep(e0: float, e1: float, x: float) -> float:
    u = clamp01((x - e0) / max(1e-6, e1 - e0))
    return u * u * (3.0 - 2.0 * u)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def set_bsdf(bsdf, name: str, val) -> None:
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = val


def get_rgba_mix_sockets(mix_node):
    rgba_in = [s for s in mix_node.inputs if s.type == "RGBA"]
    rgba_out = [s for s in mix_node.outputs if s.type == "RGBA"]
    return mix_node.inputs[0], rgba_in[0], rgba_in[1], rgba_out[0]


def link_obj(scene, obj):
    if obj.name not in scene.collection.objects:
        scene.collection.objects.link(obj)
    return obj


def add_mesh(scene, name: str, bm: bmesh.types.BMesh, mat, loc=(0, 0, 0), rot=(0, 0, 0), smooth=True, shadow=True):
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(loc)
    obj.rotation_euler = Euler(rot, "XYZ")
    if smooth:
        for p in mesh.polygons:
            p.use_smooth = True
    if mat is not None:
        obj.data.materials.append(mat)
    obj.visible_shadow = shadow
    return link_obj(scene, obj)


def limb_surface_z(x: float) -> float:
    """Top surface Z of anatomical forearm cylinder centered at Z = -LIMB_R so crown sits at Z = 0.0!"""
    r2 = max(1e-4, LIMB_R * LIMB_R - x * x)
    return math.sqrt(r2) - LIMB_R


def wound_open_half_width(y: float) -> float:
    """Smooth elliptical tapered wound opening profile along Y in [-WOUND_HALF_LEN, +WOUND_HALF_LEN]."""
    u = abs(y) / WOUND_HALF_LEN
    if u >= 1.0:
        return 0.0
    return WOUND_HALF_W * (math.cos(u * math.pi * 0.5) ** 1.35)


# ============================================================================
# 1. PHOTOREAL MEDICAL MACRO MATERIALS (CYCLES SSS SKIN, STEEL, NYLON, BIO)
# ============================================================================
def make_photoreal_skin_mat():
    """Cycles Burley Subsurface Skin with multi-scale micro-pores + animated crimson inflammation mask around the wound!"""
    mat = bpy.data.materials.new("PhotorealHumanSkin_SSS")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")

    # Multi-scale organic skin tone mottling
    ns_macro = nt.nodes.new("ShaderNodeTexNoise")
    ns_macro.inputs["Scale"].default_value = 18.0
    ns_macro.inputs["Detail"].default_value = 6.0
    ns_macro.inputs["Roughness"].default_value = 0.55

    rmp_skin = nt.nodes.new("ShaderNodeValToRGB")
    rmp_skin.color_ramp.elements[0].position = 0.25
    rmp_skin.color_ramp.elements[0].color = (0.52, 0.31, 0.22, 1.0)
    e_mid = rmp_skin.color_ramp.elements.new(0.55)
    e_mid.color = (0.64, 0.40, 0.29, 1.0)
    rmp_skin.color_ramp.elements[-1].position = 0.85
    rmp_skin.color_ramp.elements[-1].color = (0.72, 0.47, 0.35, 1.0)

    # Radial elliptical wound inflammation mask (active ONLY near |X| < 0.065, |Y| < 0.165)
    # dist^2 = (x/0.062)^2 + (y/0.168)^2
    div_x = nt.nodes.new("ShaderNodeMath")
    div_x.operation = "DIVIDE"
    div_x.inputs[1].default_value = 0.062
    sq_x = nt.nodes.new("ShaderNodeMath")
    sq_x.operation = "POWER"
    sq_x.inputs[1].default_value = 2.0

    div_y = nt.nodes.new("ShaderNodeMath")
    div_y.operation = "DIVIDE"
    div_y.inputs[1].default_value = 0.168
    sq_y = nt.nodes.new("ShaderNodeMath")
    sq_y.operation = "POWER"
    sq_y.inputs[1].default_value = 2.0

    add_d = nt.nodes.new("ShaderNodeMath")
    add_d.operation = "ADD"

    mr_inf = nt.nodes.new("ShaderNodeMapRange")
    mr_inf.clamp = True
    mr_inf.interpolation_type = "SMOOTHSTEP"
    mr_inf.inputs["From Min"].default_value = 0.05
    mr_inf.inputs["From Max"].default_value = 1.0
    mr_inf.inputs["To Min"].default_value = 1.0
    mr_inf.inputs["To Max"].default_value = 0.0

    # Animated inflammation intensity multiplier node (keyframed 0.0 -> 0.85 -> 0.08!)
    inf_drive = nt.nodes.new("ShaderNodeValue")
    inf_drive.outputs[0].default_value = 0.12

    inf_mul = nt.nodes.new("ShaderNodeMath")
    inf_mul.operation = "MULTIPLY"
    inf_mul.use_clamp = True

    mix_inf = nt.nodes.new("ShaderNodeMix")
    mix_inf.data_type = "RGBA"
    mi_fac, mi_a, mi_b, mi_out = get_rgba_mix_sockets(mix_inf)
    mi_b.default_value = (0.48, 0.065, 0.055, 1.0)  # Realistic warm inflamed crimson-rose skin!

    # Faint healed scar line tint (along |X| < 0.008)
    abs_x = nt.nodes.new("ShaderNodeMath")
    abs_x.operation = "ABSOLUTE"
    mr_scar = nt.nodes.new("ShaderNodeMapRange")
    mr_scar.clamp = True
    mr_scar.inputs["From Min"].default_value = 0.0
    mr_scar.inputs["From Max"].default_value = 0.009
    mr_scar.inputs["To Min"].default_value = 1.0
    mr_scar.inputs["To Max"].default_value = 0.0

    scar_drive = nt.nodes.new("ShaderNodeValue")
    scar_drive.outputs[0].default_value = 0.0
    scar_mul = nt.nodes.new("ShaderNodeMath")
    scar_mul.operation = "MULTIPLY"
    mix_scar = nt.nodes.new("ShaderNodeMix")
    mix_scar.data_type = "RGBA"
    ms_fac, ms_a, ms_b, ms_out = get_rgba_mix_sockets(mix_scar)
    ms_b.default_value = (0.58, 0.34, 0.28, 1.0)

    # Photoreal Micro-Pore Voronoi + Fine Epidermal Wrinkle Bump!
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 260.0
    ns_fine = nt.nodes.new("ShaderNodeTexNoise")
    ns_fine.inputs["Scale"].default_value = 95.0
    ns_fine.inputs["Detail"].default_value = 8.0

    bmp_pores = nt.nodes.new("ShaderNodeBump")
    bmp_pores.inputs["Strength"].default_value = 0.10
    bmp_pores.inputs["Distance"].default_value = 0.0012
    bmp_fine = nt.nodes.new("ShaderNodeBump")
    bmp_fine.inputs["Strength"].default_value = 0.08
    bmp_fine.inputs["Distance"].default_value = 0.0022

    # Cycles Burley Subsurface Scattering + Natural Skin Sheen!
    set_bsdf(bsdf, "Subsurface Weight", 0.45)
    set_bsdf(bsdf, "Subsurface Scale", 0.012)
    set_bsdf(bsdf, "Subsurface Radius", (1.0, 0.36, 0.18))
    if hasattr(bsdf, "subsurface_method"):
        bsdf.subsurface_method = "BURLEY"
    set_bsdf(bsdf, "Roughness", 0.38)
    set_bsdf(bsdf, "Specular IOR Level", 0.52)
    set_bsdf(bsdf, "Coat Weight", 0.18)
    set_bsdf(bsdf, "Coat Roughness", 0.28)

    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], ns_macro.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], ns_fine.inputs["Vector"])
    nt.links.new(ns_macro.outputs["Fac"], rmp_skin.inputs["Fac"])

    nt.links.new(sep.outputs["X"], div_x.inputs[0])
    nt.links.new(div_x.outputs["Value"], sq_x.inputs[0])
    nt.links.new(sep.outputs["Y"], div_y.inputs[0])
    nt.links.new(div_y.outputs["Value"], sq_y.inputs[0])
    nt.links.new(sq_x.outputs["Value"], add_d.inputs[0])
    nt.links.new(sq_y.outputs["Value"], add_d.inputs[1])
    nt.links.new(add_d.outputs["Value"], mr_inf.inputs["Value"])
    nt.links.new(mr_inf.outputs["Result"], inf_mul.inputs[0])
    nt.links.new(inf_drive.outputs[0], inf_mul.inputs[1])

    nt.links.new(inf_mul.outputs["Value"], mi_fac)
    nt.links.new(rmp_skin.outputs["Color"], mi_a)

    nt.links.new(sep.outputs["X"], abs_x.inputs[0])
    nt.links.new(abs_x.outputs["Value"], mr_scar.inputs["Value"])
    nt.links.new(mr_scar.outputs["Result"], scar_mul.inputs[0])
    nt.links.new(scar_drive.outputs[0], scar_mul.inputs[1])
    nt.links.new(scar_mul.outputs["Value"], ms_fac)
    nt.links.new(mi_out, ms_a)
    nt.links.new(ms_out, bsdf.inputs["Base Color"])

    nt.links.new(vor.outputs["Distance"], bmp_pores.inputs["Height"])
    nt.links.new(ns_fine.outputs["Fac"], bmp_fine.inputs["Height"])
    nt.links.new(bmp_pores.outputs["Normal"], bmp_fine.inputs["Normal"])
    nt.links.new(bmp_fine.outputs["Normal"], bsdf.inputs["Normal"])

    return mat, inf_drive.outputs[0], scar_drive.outputs[0]


def make_dermis_wound_bed_mat():
    """Clean clinical deep crimson-rose moist subcutaneous dermis bed inside the open incision."""
    mat = bpy.data.materials.new("ClinicalDermisBed")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 32.0
    ns.inputs["Detail"].default_value = 7.0
    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].color = (0.20, 0.022, 0.024, 1.0)
    rmp.color_ramp.elements[-1].color = (0.44, 0.068, 0.062, 1.0)
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.25
    bmp.inputs["Distance"].default_value = 0.003
    set_bsdf(bsdf, "Roughness", 0.22)
    set_bsdf(bsdf, "Coat Weight", 0.75)
    set_bsdf(bsdf, "Coat Roughness", 0.08)
    set_bsdf(bsdf, "Subsurface Weight", 0.35)
    set_bsdf(bsdf, "Subsurface Scale", 0.008)
    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_surgical_steel_mat(name: str = "SurgicalStainlessSteel"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    set_bsdf(bsdf, "Base Color", (0.86, 0.89, 0.94, 1.0))
    set_bsdf(bsdf, "Metallic", 1.0)
    set_bsdf(bsdf, "Roughness", 0.12)
    set_bsdf(bsdf, "Coat Weight", 0.40)
    set_bsdf(bsdf, "Coat Roughness", 0.04)
    return mat


def make_suture_nylon_mat():
    """Vibrant Royal-Blue Braided Nylon Surgical Suture Thread (pops vividly against skin & steel!)."""
    mat = bpy.data.materials.new("BraidedBlueNylonSuture")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    wv = nt.nodes.new("ShaderNodeTexWave")
    wv.wave_type = "BANDS"
    wv.inputs["Scale"].default_value = 140.0
    wv.inputs["Distortion"].default_value = 1.6
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.16
    bmp.inputs["Distance"].default_value = 0.0005
    set_bsdf(bsdf, "Base Color", (0.025, 0.16, 0.78, 1.0))
    set_bsdf(bsdf, "Roughness", 0.28)
    set_bsdf(bsdf, "Coat Weight", 0.55)
    set_bsdf(bsdf, "Coat Roughness", 0.16)
    nt.links.new(tc.outputs["Object"], wv.inputs["Vector"])
    nt.links.new(wv.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_sterile_drape_mat():
    mat = bpy.data.materials.new("SterileORDrapeBlue")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 140.0
    ns.inputs["Detail"].default_value = 6.0
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.14
    bmp.inputs["Distance"].default_value = 0.002
    set_bsdf(bsdf, "Base Color", (0.028, 0.105, 0.165, 1.0))
    set_bsdf(bsdf, "Roughness", 0.82)
    set_bsdf(bsdf, "Sheen Weight", 0.45)
    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_repair_cell_mat():
    """Luminous pearl-gold translucent biological repair cell (macrophage/fibroblast)."""
    mat = bpy.data.materials.new("BiologicalRepairCellMat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.35
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (1.0, 0.90, 0.58, 1.0)
    em.inputs["Strength"].default_value = 2.2
    mix = nt.nodes.new("ShaderNodeMixShader")
    set_bsdf(bsdf, "Base Color", (0.98, 0.88, 0.66, 1.0))
    set_bsdf(bsdf, "Roughness", 0.22)
    set_bsdf(bsdf, "Subsurface Weight", 0.65)
    nt.links.new(lw.outputs["Facing"], mix.inputs["Fac"])
    nt.links.new(bsdf.outputs["BSDF"], mix.inputs[1])
    nt.links.new(em.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], nt.nodes["Material Output"].inputs["Surface"])
    return mat


def make_collagen_morph_mat():
    """Animated Golden Collagen Fiber -> Living Pink New Tissue shader!"""
    mat = bpy.data.materials.new("CollagenToNewSkinMat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    mix_c = nt.nodes.new("ShaderNodeMix")
    mix_c.data_type = "RGBA"
    fac, c_gold, c_pink, c_out = get_rgba_mix_sockets(mix_c)
    fac.default_value = 0.0
    c_gold.default_value = (1.0, 0.78, 0.20, 1.0)  # Luminous 24K golden collagen scaffold
    c_pink.default_value = (0.76, 0.34, 0.30, 1.0)  # Living regenerated rose-pink skin!

    em_val = nt.nodes.new("ShaderNodeValue")
    em_val.outputs[0].default_value = 1.65

    set_bsdf(bsdf, "Roughness", 0.28)
    set_bsdf(bsdf, "Subsurface Weight", 0.55)
    set_bsdf(bsdf, "Subsurface Scale", 0.01)
    nt.links.new(c_out, bsdf.inputs["Base Color"])
    if "Emission Color" in bsdf.inputs:
        nt.links.new(c_out, bsdf.inputs["Emission Color"])
    if "Emission Strength" in bsdf.inputs:
        nt.links.new(em_val.outputs[0], bsdf.inputs["Emission Strength"])
    return mat, fac, em_val.outputs[0]


def make_capillary_vessel_mat():
    """Rich oxygenated ruby-crimson branching blood vessel with subtle rhythmic life glow."""
    mat = bpy.data.materials.new("RubyCapillaryVesselMat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    set_bsdf(bsdf, "Base Color", (0.72, 0.035, 0.045, 1.0))
    set_bsdf(bsdf, "Roughness", 0.18)
    set_bsdf(bsdf, "Coat Weight", 0.85)
    set_bsdf(bsdf, "Coat Roughness", 0.06)
    set_bsdf(bsdf, "Subsurface Weight", 0.40)
    if "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = (0.95, 0.10, 0.12, 1.0)
    em_in = bsdf.inputs.get("Emission Strength")
    if em_in is not None:
        em_in.default_value = 0.85
    return mat, em_in


def make_crisp_badge_mat(name: str, rgb=(1.0, 0.96, 0.88), strength: float = 1.65):
    """AGENTS.md Section 2: Object-space |N.z| crisp 2-zone brushed badge shader."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    vtr = nt.nodes.new("ShaderNodeVectorTransform")
    vtr.vector_type = "NORMAL"
    vtr.convert_from = "WORLD"
    vtr.convert_to = "OBJECT"
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    abz = nt.nodes.new("ShaderNodeMath")
    abz.operation = "ABSOLUTE"
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.clamp = True
    mr.inputs["From Min"].default_value = 0.82
    mr.inputs["From Max"].default_value = 0.94

    face = nt.nodes.new("ShaderNodeBsdfPrincipled")
    set_bsdf(face, "Base Color", (*rgb, 1.0))
    set_bsdf(face, "Emission Color", (*rgb, 1.0))
    set_bsdf(face, "Emission Strength", strength)
    side = nt.nodes.new("ShaderNodeBsdfPrincipled")
    set_bsdf(side, "Base Color", (0.03, 0.04, 0.06, 1.0))
    set_bsdf(side, "Metallic", 0.9)
    mix = nt.nodes.new("ShaderNodeMixShader")

    nt.links.new(geom.outputs["Normal"], vtr.inputs["Vector"])
    nt.links.new(vtr.outputs["Vector"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], abz.inputs[0])
    nt.links.new(abz.outputs["Value"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], mix.inputs["Fac"])
    nt.links.new(side.outputs["BSDF"], mix.inputs[1])
    nt.links.new(face.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


def make_pill_halo_mat():
    mat = bpy.data.materials.new("StageBadgePillHalo")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    sx = nt.nodes.new("ShaderNodeMath"); sx.operation = "SUBTRACT"; sx.inputs[1].default_value = 0.5
    mx = nt.nodes.new("ShaderNodeMath"); mx.operation = "MULTIPLY"; mx.inputs[1].default_value = 2.0
    px = nt.nodes.new("ShaderNodeMath"); px.operation = "POWER"; px.inputs[1].default_value = 4.0
    sy = nt.nodes.new("ShaderNodeMath"); sy.operation = "SUBTRACT"; sy.inputs[1].default_value = 0.5
    my = nt.nodes.new("ShaderNodeMath"); my.operation = "MULTIPLY"; my.inputs[1].default_value = 2.0
    py = nt.nodes.new("ShaderNodeMath"); py.operation = "POWER"; py.inputs[1].default_value = 2.0
    ad = nt.nodes.new("ShaderNodeMath"); ad.operation = "ADD"
    mr = nt.nodes.new("ShaderNodeMapRange"); mr.clamp = True; mr.interpolation_type = "SMOOTHSTEP"
    mr.inputs["From Min"].default_value = 0.28; mr.inputs["From Max"].default_value = 1.0
    mr.inputs["To Min"].default_value = 0.78; mr.inputs["To Max"].default_value = 0.0
    lp = nt.nodes.new("ShaderNodeLightPath")
    cm = nt.nodes.new("ShaderNodeMath"); cm.operation = "MULTIPLY"
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    dk = nt.nodes.new("ShaderNodeEmission"); dk.inputs["Color"].default_value = (0.010, 0.022, 0.036, 1.0); dk.inputs["Strength"].default_value = 0.16
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["X"], sx.inputs[0]); nt.links.new(sx.outputs["Value"], mx.inputs[0]); nt.links.new(mx.outputs["Value"], px.inputs[0])
    nt.links.new(sep.outputs["Y"], sy.inputs[0]); nt.links.new(sy.outputs["Value"], my.inputs[0]); nt.links.new(my.outputs["Value"], py.inputs[0])
    nt.links.new(px.outputs["Value"], ad.inputs[0]); nt.links.new(py.outputs["Value"], ad.inputs[1])
    nt.links.new(ad.outputs["Value"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], cm.inputs[0]); nt.links.new(lp.outputs["Is Camera Ray"], cm.inputs[1])
    nt.links.new(cm.outputs["Value"], mix.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[1]); nt.links.new(dk.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


# ============================================================================
# 2. DEFORMING 3D HUMAN LIMB WITH LIVE WOUND SHAPE KEYS (OPEN -> CLOSED -> HEALED)
# ============================================================================
def build_living_wound_limb(scene, skin_mat, dermis_mat):
    """Builds a high-resolution curved human limb skin shell split cleanly down the central incision,
    with 2 physical shape keys:
      - `Closed`: Left & Right wound edges smoothly glide shut like a zipper with surgical seam ridge!
      - `Healed`: Smooth seamless healthy skin contour!
    Plus a realistic glistening crimson subcutaneous dermis bed underneath!"""
    mesh = bpy.data.meshes.new("LivingSkinLimb_Mesh")
    obj = bpy.data.objects.new("LivingSkinLimb", mesh)
    link_obj(scene, obj)
    obj.data.materials.append(skin_mat)

    ny = 72
    nx_half = 24
    # Build Left half (side=-1) and Right half (side=+1) cleanly separated at the incision so the gap opens & zips shut!
    verts_open = []
    verts_closed = []
    verts_healed = []
    faces = []

    def build_half(side: float, base_idx: int):
        for iy in range(ny + 1):
            v_y = lerp(-LIMB_LEN * 0.5, LIMB_LEN * 0.5, iy / float(ny))
            w_gap = wound_open_half_width(v_y)
            for ix in range(nx_half + 1):
                u = ix / float(nx_half)  # 0 at center seam, 1 at outer limb flank (angle ~72 deg)
                ang_closed = side * (u * math.radians(74.0))
                x_closed = LIMB_R * math.sin(ang_closed)
                z_closed = LIMB_R * math.cos(ang_closed) - LIMB_R

                # Open wound pulls the inner skin edge outward by w_gap with organic tissue falloff
                pull = math.exp(-u * 5.2) * w_gap
                x_open = x_closed + side * pull
                z_open = limb_surface_z(min(LIMB_R * 0.96, abs(x_open)))
                # Subtle natural curling lip depth right at the cut margin
                lip_curl = -0.0055 * math.exp(-u * 16.0) * (w_gap / max(1e-5, WOUND_HALF_W))
                z_open += lip_curl

                # Closed sutured seam has a subtle realistic surgical eversion pucker ridge along the incision
                in_wound = 1.0 if abs(v_y) < WOUND_HALF_LEN else 0.0
                env_y = math.cos(min(1.0, abs(v_y) / WOUND_HALF_LEN) * math.pi * 0.5) if in_wound else 0.0
                pucker = 0.0038 * math.exp(-u * 14.0) * env_y
                verts_open.append((x_open, v_y, z_open))
                verts_closed.append((x_closed, v_y, z_closed + pucker))
                verts_healed.append((x_closed, v_y, z_closed))

        stride = nx_half + 1
        for iy in range(ny):
            for ix in range(nx_half):
                a = base_idx + iy * stride + ix
                b = a + 1
                c = a + stride + 1
                d = a + stride
                if side > 0:
                    faces.append((a, b, c, d))
                else:
                    faces.append((a, d, c, b))

    build_half(-1.0, 0)
    build_half(+1.0, (ny + 1) * (nx_half + 1))

    mesh.from_pydata(verts_open, [], faces)
    mesh.update()
    for p in mesh.polygons:
        p.use_smooth = True

    # Add Basis + Closed + Healed Shape Keys!
    obj.shape_key_add(name="Basis", from_mix=False)
    sk_closed = obj.shape_key_add(name="Closed", from_mix=False)
    for i, co in enumerate(verts_closed):
        sk_closed.data[i].co = Vector(co)
    sk_healed = obj.shape_key_add(name="Healed", from_mix=False)
    for i, co in enumerate(verts_healed):
        sk_healed.data[i].co = Vector(co)

    # Underlying glistening crimson subcutaneous dermis cylinder (sits cleanly 5mm below skin crown!)
    bm_d = bmesh.new()
    bmesh.ops.create_cone(
        bm_d, cap_ends=True, cap_tris=False, segments=48,
        radius1=LIMB_R - 0.006, radius2=LIMB_R - 0.006, depth=LIMB_LEN * 0.98,
        matrix=Matrix.Translation((0.0, 0.0, -LIMB_R)) @ Matrix.Rotation(math.radians(90.0), 4, "X"),
    )
    dermis_obj = add_mesh(scene, "SubcutaneousDermisBed", bm_d, dermis_mat, smooth=True, shadow=True)
    return obj, sk_closed, sk_healed, dermis_obj


# ============================================================================
# 3. CURVED SURGICAL NEEDLE, BLUE NYLON SUTURE LOOPS & ARTICULATED SCISSORS
# ============================================================================
def build_curved_surgical_needle(scene, steel_mat):
    """Gleaming 3D 3/8-circle curved stainless-steel surgical suture needle."""
    bm = bmesh.new()
    segs = 22
    r_arc = 0.042
    ring_prev = None
    for i in range(segs + 1):
        u = i / float(segs)
        ang = lerp(-math.pi * 0.42, math.pi * 0.42, u)
        cx = r_arc * math.sin(ang)
        cz = -r_arc * math.cos(ang) + r_arc
        # Taper sharply at leading tip (u=1)
        rad = 0.0022 * (1.0 - 0.88 * (u ** 2.6))
        rad = max(0.00025, rad)
        tx = math.cos(ang)
        tz = math.sin(ang)
        n_r = Vector((-tz, 0.0, tx)).normalized()
        n_y = Vector((0.0, 1.0, 0.0))
        ring = []
        for j in range(10):
            a = 2.0 * math.pi * j / 10
            pt = Vector((cx, 0.0, cz)) + n_r * (rad * math.cos(a)) + n_y * (rad * math.sin(a))
            ring.append(bm.verts.new(pt))
        if ring_prev is not None:
            for j in range(10):
                jn = (j + 1) % 10
                bm.faces.new((ring_prev[j], ring_prev[jn], ring[jn], ring[j]))
        ring_prev = ring
    return add_mesh(scene, "CurvedSurgicalNeedle", bm, steel_mat, smooth=True, shadow=True)


def build_dynamic_suture_loops(scene, nylon_mat):
    """6 independent royal-blue nylon suture loops (each with Open & Closed shape keys so they cinch smoothly with the skin!)
    plus a continuous threading curve + loose tail that gets snipped by surgical scissors!"""
    loop_ys = [lerp(-0.108, 0.108, i / float(NUM_LOOPS - 1)) for i in range(NUM_LOOPS)]
    loops = []

    for idx, ly in enumerate(loop_ys):
        w_open = wound_open_half_width(ly) + 0.016
        w_closed = 0.016
        n_pts = 24

        def make_arc_points(half_w: float, lift: float):
            pts = []
            for k in range(n_pts + 1):
                u = k / float(n_pts)
                # Loop arcs cleanly over the skin from Left entry (-half_w) -> arch top -> Right exit (+half_w)
                x = lerp(-half_w, half_w, u)
                z_skin = limb_surface_z(abs(x))
                arch = math.sin(math.pi * u) * lift
                pts.append(Vector((x, ly, z_skin + arch - 0.0015)))
            return pts

        pts_open = make_arc_points(w_open, 0.019)
        pts_closed = make_arc_points(w_closed, 0.0095)

        # Build smooth tubular mesh with Open (Basis) and Closed (Tight) shape keys!
        r_thread = 0.0024
        segs_r = 10

        def tube_verts(pts):
            out_v = []
            for i, p in enumerate(pts):
                t_dir = (pts[min(len(pts) - 1, i + 1)] - pts[max(0, i - 1)]).normalized()
                up = Vector((0.0, 1.0, 0.0))
                b1 = t_dir.cross(up).normalized()
                b2 = t_dir.cross(b1).normalized()
                for j in range(segs_r):
                    a = 2.0 * math.pi * j / segs_r
                    out_v.append(p + b1 * (r_thread * math.cos(a)) + b2 * (r_thread * math.sin(a)))
            return out_v

        v_open = tube_verts(pts_open)
        v_closed = tube_verts(pts_closed)
        faces = []
        for i in range(len(pts_open) - 1):
            for j in range(segs_r):
                jn = (j + 1) % segs_r
                a0 = i * segs_r + j
                a1 = i * segs_r + jn
                b0 = (i + 1) * segs_r + j
                b1 = (i + 1) * segs_r + jn
                faces.append((a0, a1, b1, b0))

        mesh = bpy.data.meshes.new(f"SutureLoop_{idx:02d}_Mesh")
        mesh.from_pydata([tuple(v) for v in v_open], [], faces)
        mesh.update()
        for p in mesh.polygons:
            p.use_smooth = True
        obj = bpy.data.objects.new(f"SutureLoop_{idx:02d}", mesh)
        obj.data.materials.append(nylon_mat)
        link_obj(scene, obj)

        obj.shape_key_add(name="Basis", from_mix=False)
        sk_tight = obj.shape_key_add(name="Tight", from_mix=False)
        for i, v in enumerate(v_closed):
            sk_tight.data[i].co = v

        # Small surgical knot bead sitting cleanly on the right side of each stitch
        bm_kn = bmesh.new()
        bmesh.ops.create_icosphere(bm_kn, subdivisions=2, radius=0.0042)
        knot = add_mesh(scene, f"SutureKnot_{idx:02d}", bm_kn, nylon_mat, loc=(w_closed + 0.002, ly, limb_surface_z(w_closed) + 0.004), smooth=True, shadow=True)
        loops.append((obj, sk_tight, knot, ly, w_open, w_closed))

    # Loose trailing blue thread pulled up at top stitch (Y = +0.108) for Beat 2 Zipper Pull & Scissors SNIP!
    bm_tail = bmesh.new()
    bmesh.ops.create_cone(bm_tail, cap_ends=True, cap_tris=False, segments=10, radius1=0.0023, radius2=0.0023, depth=0.14, matrix=Matrix.Translation((0, 0, 0.07)))
    tail_obj = add_mesh(scene, "SuturePullTail", bm_tail, nylon_mat, loc=(0.018, 0.108, limb_surface_z(0.018) + 0.002), smooth=True, shadow=True)
    return loops, tail_obj


def build_surgical_scissors(scene, steel_mat):
    """Articulated 3D Stainless-Steel Surgical Scissors (twin pivoting blades + rings!)."""
    root = bpy.data.objects.new("SurgicalScissors_Root", None)
    link_obj(scene, root)

    def make_blade(name: str, sign: float):
        bm = bmesh.new()
        # Tapered sharp blade extending along -X (tip at X=-0.095, pivot at X=0)
        bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Translation((-0.048, sign * 0.002, 0.0)) @ Matrix.Diagonal((0.096, 0.0032, 0.011, 1.0)))
        # Handle shaft extending along +X
        bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Translation((0.068, sign * 0.004, sign * 0.006)) @ Matrix.Diagonal((0.135, 0.0045, 0.0085, 1.0)))
        # Finger ring at end of handle
        segs = 20
        r_ring = 0.022
        for i in range(segs):
            a0 = 2.0 * math.pi * i / segs
            a1 = 2.0 * math.pi * (i + 1) / segs
            p0 = Vector((0.152 + r_ring * math.cos(a0), sign * 0.004, sign * 0.022 + r_ring * math.sin(a0)))
            p1 = Vector((0.152 + r_ring * math.cos(a1), sign * 0.004, sign * 0.022 + r_ring * math.sin(a1)))
            mid = (p0 + p1) * 0.5
            q = (p1 - p0).normalized().to_track_quat("Z", "Y")
            bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=8, radius1=0.0035, radius2=0.0035, depth=(p1 - p0).length * 1.1, matrix=Matrix.Translation(mid) @ q.to_matrix().to_4x4())
        b = add_mesh(scene, name, bm, steel_mat, smooth=True, shadow=True)
        b.parent = root
        return b

    blade_a = make_blade("ScissorsBlade_A", +1.0)
    blade_b = make_blade("ScissorsBlade_B", -1.0)
    bm_p = bmesh.new()
    bmesh.ops.create_cone(bm_p, cap_ends=True, cap_tris=False, segments=12, radius1=0.0055, radius2=0.0055, depth=0.012, matrix=Matrix.Rotation(math.radians(90.0), 4, "X"))
    piv = add_mesh(scene, "ScissorsPivot", bm_p, steel_mat, smooth=True, shadow=False)
    piv.parent = root
    return root, blade_a, blade_b


# ============================================================================
# 4. BIOLOGICAL HEALING VFX: REPAIR CELLS, GOLDEN COLLAGEN WEB & CAPILLARY TREES
# ============================================================================
def build_repair_cells(scene, cell_mat):
    """140 glowing biological repair cells (macrophages/fibroblasts) that swarm into the wound seam during Beat 3!"""
    rng = random.Random(2026)
    cells = []
    for i in range(140):
        side = -1.0 if i % 2 == 0 else 1.0
        y_t = rng.uniform(-0.135, 0.135)
        x_start = side * rng.uniform(0.048, 0.115)
        x_end = side * rng.uniform(0.001, 0.022)
        rad = rng.uniform(0.0024, 0.0048)
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=rad)
        obj = add_mesh(scene, f"RepairCell_{i:03d}", bm, cell_mat, loc=(x_start, y_t, limb_surface_z(abs(x_start)) + 0.003), smooth=True, shadow=False)
        cells.append((obj, x_start, x_end, y_t, rng.uniform(0.0, 1.0), rng.uniform(0.0, 6.28)))
    return cells


def build_collagen_scaffold(scene, collagen_mat):
    """36 criss-crossing 3D golden collagen fiber bridges weaving across the closed seam in Beat 4!"""
    rng = random.Random(4096)
    strands = []
    for i in range(36):
        y0 = lerp(-0.132, 0.132, i / 35.0)
        dy = (0.018 if i % 2 == 0 else -0.018) + rng.uniform(-0.005, 0.005)
        x_span = rng.uniform(0.020, 0.034)
        p0 = Vector((-x_span, y0 - dy, limb_surface_z(x_span) + 0.0032))
        p1 = Vector((+x_span, y0 + dy, limb_surface_z(x_span) + 0.0032))
        mid = (p0 + p1) * 0.5 + Vector((0.0, 0.0, 0.0042 + rng.uniform(-0.001, 0.0015)))

        bm = bmesh.new()
        n_seg = 10
        pts = [p0 * ((1 - u) ** 2) + mid * (2 * (1 - u) * u) + p1 * (u * u) for u in [k / float(n_seg) for k in range(n_seg + 1)]]
        for k in range(n_seg):
            dv = pts[k + 1] - pts[k]
            q = dv.normalized().to_track_quat("Z", "Y")
            m = Matrix.Translation((pts[k] + pts[k + 1]) * 0.5) @ q.to_matrix().to_4x4()
            bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=6, radius1=0.0017, radius2=0.0017, depth=dv.length * 1.12, matrix=m)
        obj = add_mesh(scene, f"CollagenFiber_{i:02d}", bm, collagen_mat, smooth=True, shadow=False)
        strands.append((obj, i / 36.0))
    return strands


def build_capillary_network(scene, vessel_mat):
    """18 organic multi-branched 3D ruby-red capillary blood vessels growing across the seam in Beat 5,
    then sinking cleanly beneath the skin surface!"""
    root = bpy.data.objects.new("CapillaryNetwork_Root", None)
    link_obj(scene, root)
    rng = random.Random(8192)
    trees = []

    def add_seg(bm, a: Vector, b: Vector, r1: float, r2: float):
        dv = b - a
        ln = dv.length
        if ln < 1e-5:
            return
        q = dv.normalized().to_track_quat("Z", "Y")
        m = Matrix.Translation((a + b) * 0.5) @ q.to_matrix().to_4x4()
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=8, radius1=r1, radius2=r2, depth=ln * 1.08, matrix=m)

    for i in range(18):
        side = -1.0 if i % 2 == 0 else 1.0
        y_base = lerp(-0.130, 0.130, i / 17.0)
        x0 = side * 0.068
        p0 = Vector((x0, y_base, limb_surface_z(abs(x0)) + 0.0028))
        p1 = Vector((side * 0.036, y_base + rng.uniform(-0.012, 0.012), limb_surface_z(0.036) + 0.0044))
        p2 = Vector((side * 0.004, y_base + rng.uniform(-0.015, 0.015), limb_surface_z(0.004) + 0.0048))

        bm = bmesh.new()
        add_seg(bm, p0, p1, 0.0026, 0.0019)
        add_seg(bm, p1, p2, 0.0019, 0.0011)
        # Forked secondary & tertiary fine capillaries reaching across the seam!
        for b_sgn in (-1.0, 1.0):
            pb1 = p1 + Vector((-side * 0.022, b_sgn * rng.uniform(0.014, 0.024), 0.0))
            pb1.z = limb_surface_z(abs(pb1.x)) + 0.0044
            add_seg(bm, p1, pb1, 0.0016, 0.0009)
            pb2 = pb1 + Vector((-side * 0.018, b_sgn * rng.uniform(0.008, 0.016), 0.0))
            pb2.z = limb_surface_z(abs(pb2.x)) + 0.0042
            add_seg(bm, pb1, pb2, 0.0009, 0.00045)

        obj = add_mesh(scene, f"CapillaryTree_{i:02d}", bm, vessel_mat, smooth=True, shadow=False)
        obj.parent = root
        trees.append((obj, i / 18.0))
    return root, trees


def build_stage_badge(scene, name: str, label: str, loc: Vector, face_mat, halo_mat):
    """Clean, subtle floating 3D Medical Stage Micro-Badge at top of frame (AGENTS.md Section 2)."""
    root = bpy.data.objects.new(f"{name}_Root", None)
    root.location = loc
    link_obj(scene, root)

    font = bpy.data.fonts.load(FONT_PATH, check_existing=True)
    cd = bpy.data.curves.new(f"{name}_Curve", type="FONT")
    cd.body = label
    cd.font = font
    cd.size = 0.0125
    cd.align_x = "CENTER"
    cd.align_y = "CENTER"
    cd.extrude = 0.0008
    cd.bevel_depth = 0.00018
    txt = bpy.data.objects.new(name, cd)
    txt.data.materials.append(face_mat)
    txt.visible_shadow = False
    txt.parent = root
    link_obj(scene, txt)

    bm_h = bmesh.new()
    bmesh.ops.create_grid(bm_h, x_segments=1, y_segments=1, size=0.5, matrix=Matrix.Diagonal((0.165, 0.032, 1.0, 1.0)))
    halo = add_mesh(scene, f"{name}_Halo", bm_h, halo_mat, loc=(0.0, 0.0, -0.003), smooth=False, shadow=False)
    halo.parent = root
    return root


# ============================================================================
# 5. OPERATING ROOM STUDIO LIGHTING & MACRO CINEMA CAMERA
# ============================================================================
def setup_or_studio(scene, drape_mat):
    # Clean deep sterile operating-room slate-blue world (never muddy grey void!)
    world = bpy.data.worlds.new("MedicalMacroStudioWorld")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes["Background"]
    bg.inputs["Color"].default_value = (0.032, 0.068, 0.105, 1.0)
    bg.inputs["Strength"].default_value = 0.65

    # Sterile blue OR surgical drape surface underneath the forearm
    bm_dr = bmesh.new()
    bmesh.ops.create_grid(bm_dr, x_segments=16, y_segments=16, size=1.4, matrix=Matrix.Translation((0.0, 0.0, -LIMB_R * 1.12)))
    add_mesh(scene, "SterileORDrape", bm_dr, drape_mat, smooth=True, shadow=True)

    # 1. Overhead Surgical Dome Key Light (crisp clinical top-left illumination + glossy steel reflections)
    k_d = bpy.data.lights.new("OR_SurgicalDomeKey", type="AREA")
    k_d.shape = "DISK"
    k_d.size = 0.55
    k_d.energy = 42.0
    k_d.color = (1.0, 0.96, 0.92)
    k_o = bpy.data.objects.new("OR_SurgicalDomeKey", k_d)
    k_o.location = Vector((-0.34, -0.26, 0.68))
    k_o.rotation_euler = (Vector((0, 0, 0)) - k_o.location).to_track_quat("-Z", "Y").to_euler()
    link_obj(scene, k_o)

    # 2. Cool Clinical Teal Rim Light (right-rear edge separation on steel scissors, blue nylon & skin pores!)
    r_d = bpy.data.lights.new("OR_ClinicalRim", type="AREA")
    r_d.shape = "RECTANGLE"
    r_d.size = 0.65
    r_d.size_y = 0.40
    r_d.energy = 30.0
    r_d.color = (0.62, 0.88, 1.0)
    r_o = bpy.data.objects.new("OR_ClinicalRim", r_d)
    r_o.location = Vector((0.46, 0.34, 0.48))
    r_o.rotation_euler = (Vector((0, 0, 0)) - r_o.location).to_track_quat("-Z", "Y").to_euler()
    link_obj(scene, r_o)

    # 3. Soft Warm Fill (keeps skin SSS rich & lifelike)
    f_d = bpy.data.lights.new("OR_WarmFill", type="AREA")
    f_d.shape = "RECTANGLE"
    f_d.size = 0.70
    f_d.size_y = 0.50
    f_d.energy = 14.0
    f_d.color = (1.0, 0.88, 0.80)
    f_o = bpy.data.objects.new("OR_WarmFill", f_d)
    f_o.location = Vector((0.36, -0.44, 0.38))
    f_o.rotation_euler = (Vector((0, 0, 0)) - f_o.location).to_track_quat("-Z", "Y").to_euler()
    link_obj(scene, f_o)


def compute_macro_camera(t: float) -> tuple[Vector, Vector, float]:
    """Smooth continuous viral macro camera choreography across all 6 beats (1:1 synchronized with verify_framing.py!)."""
    if t < 6.0:
        # Beat 1: Zoom smoothly from wide forearm hook down into surgical macro tracking the needle!
        u = smoothstep(0.0, 5.8, t)
        pos = Vector((lerp(0.14, 0.16, u), lerp(-0.34, -0.18, u), lerp(0.56, 0.34, u)))
        tgt = Vector((0.0, lerp(-0.04, 0.04, u), 0.0))
        return pos, tgt, lerp(46.0, 58.0, u)
    if t < 10.2:
        # Beat 2: Hero overhead-angled macro on the Zipper Pull & Surgical Scissors SNIP!
        u = smoothstep(6.0, 10.0, t)
        pos = Vector((lerp(0.16, 0.11, u), lerp(-0.18, -0.15, u), lerp(0.34, 0.36, u)))
        tgt = Vector((0.0, lerp(0.04, 0.05, u), 0.0))
        return pos, tgt, lerp(58.0, 54.0, u)
    if t < 15.2:
        # Beat 3: Intimate cellular macro gliding along the inflamed seam & swarming repair cells!
        u = smoothstep(10.2, 15.0, t)
        pos = Vector((lerp(0.11, -0.14, u), lerp(-0.15, -0.22, u), lerp(0.36, 0.35, u)))
        tgt = Vector((0.0, lerp(0.02, 0.0, u), 0.0))
        return pos, tgt, lerp(54.0, 54.0, u)
    if t < 20.4:
        # Beat 4: Dramatic macro glide across the glowing golden collagen scaffold weaving into living skin!
        u = smoothstep(15.2, 20.2, t)
        pos = Vector((lerp(-0.14, 0.14, u), lerp(-0.22, -0.19, u), lerp(0.35, 0.36, u)))
        tgt = Vector((0.0, 0.0, 0.0))
        return pos, tgt, lerp(54.0, 54.0, u)
    if t < 25.3:
        # Beat 5: Overhead macro revealing branching ruby blood vessels nourishing & sinking beneath the skin!
        u = smoothstep(20.4, 25.1, t)
        pos = Vector((lerp(0.14, 0.04, u), lerp(-0.19, -0.14, u), lerp(0.36, 0.38, u)))
        tgt = Vector((0.0, 0.0, 0.0))
        return pos, tgt, lerp(54.0, 52.0, u)
    # Beat 6: Smooth stitch slide-out & satisfying pullback reveal of seamless healed skin!
    u = smoothstep(25.3, 29.8, t)
    pos = Vector((lerp(0.04, 0.12, u), lerp(-0.14, -0.32, u), lerp(0.36, 0.56, u)))
    tgt = Vector((0.0, 0.0, 0.0))
    return pos, tgt, lerp(54.0, 46.0, u)


# ============================================================================
# 6. MAIN SCENE BUILD & CYCLES RENDER LOOP
# ============================================================================
def configure_cycles(scene, args):
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.030
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_input_passes = "RGB_ALBEDO"
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 3
    scene.cycles.transmission_bounces = 4
    scene.cycles.transparent_max_bounces = 10
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False

    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in (args.device, "OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type != "CPU"]
            if gpus:
                for d in prefs.devices:
                    d.use = (d.type != "CPU")
                print(f"[Cycles GPU] Active backend: {backend} ({len(gpus)} GPU(s))")
                break
        except Exception:
            continue

    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = args.scale
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = TOTAL_FRAMES
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 95
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Punchy"
    except Exception:
        pass
    scene.view_settings.exposure = 0.12


def build_and_render(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    configure_cycles(scene, args)

    # 1. Materials & OR Studio
    skin_mat, inf_val, scar_val = make_photoreal_skin_mat()
    dermis_mat = make_dermis_wound_bed_mat()
    steel_mat = make_surgical_steel_mat()
    nylon_mat = make_suture_nylon_mat()
    drape_mat = make_sterile_drape_mat()
    cell_mat = make_repair_cell_mat()
    collagen_mat, col_mix_fac, col_em_val = make_collagen_morph_mat()
    vessel_mat, vessel_em = make_capillary_vessel_mat()
    badge_gold = make_crisp_badge_mat("StageBadgeGold", (1.0, 0.86, 0.36), 1.65)
    badge_cyan = make_crisp_badge_mat("StageBadgeCyan", (0.52, 0.94, 1.0), 1.70)
    halo_mat = make_pill_halo_mat()

    setup_or_studio(scene, drape_mat)

    # 2. Hero Deforming Wound Limb, Surgical Instruments & Biological Healing Systems
    skin_obj, sk_closed, sk_healed, _ = build_living_wound_limb(scene, skin_mat, dermis_mat)
    needle_obj = build_curved_surgical_needle(scene, steel_mat)
    suture_loops, tail_obj = build_dynamic_suture_loops(scene, nylon_mat)
    sciss_root, blade_a, blade_b = build_surgical_scissors(scene, steel_mat)
    repair_cells = build_repair_cells(scene, cell_mat)
    collagen_strands = build_collagen_scaffold(scene, collagen_mat)
    cap_root, cap_trees = build_capillary_network(scene, vessel_mat)

    # Clean, floating camera-tracked top micro-stage badges (placed cleanly in top safe zone!)
    cam_data = bpy.data.cameras.new("MacroMedicalCam")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.clip_start = 0.01
    cam_data.clip_end = 25.0
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = 4.5
    cam_obj = bpy.data.objects.new("MacroMedicalCam", cam_data)
    link_obj(scene, cam_obj)
    scene.camera = cam_obj

    # Subtle HUD stage titles parented cleanly to camera upper safe zone (Z = -0.35m in cam local space, Y = +0.115m)
    stage_badges = [
        (0.3, 9.8, build_stage_badge(scene, "Badge_Suture", "STEP 1 · SUTURE CLOSURE", Vector((0, 0.118, -0.36)), badge_cyan, halo_mat)),
        (10.3, 15.1, build_stage_badge(scene, "Badge_Repair", "STEP 2 · CELL REPAIR", Vector((0, 0.118, -0.36)), badge_gold, halo_mat)),
        (15.3, 20.3, build_stage_badge(scene, "Badge_Collagen", "STEP 3 · COLLAGEN MATRIX", Vector((0, 0.118, -0.36)), badge_gold, halo_mat)),
        (20.5, 25.2, build_stage_badge(scene, "Badge_Vessels", "STEP 4 · NEW BLOOD VESSELS", Vector((0, 0.118, -0.36)), badge_cyan, halo_mat)),
        (25.5, 29.8, build_stage_badge(scene, "Badge_Healed", "STEP 5 · FULLY HEALED", Vector((0, 0.118, -0.36)), badge_cyan, halo_mat)),
    ]
    for _, _, b_root in stage_badges:
        b_root.parent = cam_obj

    # 3. Keyframe Smooth Animation Across All 720 Frames (step=2 for ultra-fast scene build!)
    key_frames = sorted(set(list(range(1, TOTAL_FRAMES + 1, 2)) + [1, TOTAL_FRAMES]))

    for f in key_frames:
        t = (f - 1) / float(FPS)

        # --- Camera ---
        cpos, ctgt, lens = compute_macro_camera(t)
        cam_obj.location = cpos
        cam_obj.rotation_mode = "QUATERNION"
        cam_obj.rotation_quaternion = (ctgt - cpos).normalized().to_track_quat("-Z", "Y")
        cam_data.lens = lens
        cam_data.dof.focus_distance = (ctgt - cpos).length
        cam_obj.keyframe_insert(data_path="location", frame=f)
        cam_obj.keyframe_insert(data_path="rotation_quaternion", frame=f)
        cam_data.keyframe_insert(data_path="lens", frame=f)
        cam_data.dof.keyframe_insert(data_path="focus_distance", frame=f)

        # --- Stage Badges ---
        for t0_b, t1_b, b_root in stage_badges:
            vis = smoothstep(t0_b, t0_b + 0.35, t) * (1.0 - smoothstep(t1_b - 0.35, t1_b, t))
            bs = max(0.0001, vis)
            b_root.scale = (bs, bs, bs)
            b_root.keyframe_insert(data_path="scale", frame=f)

        # --- Deforming Skin Wound Closure (Beat 2 Zipper Pull t=6.0..8.1s) -> Healed (Beat 6 t=25.5..28.5s) ---
        zip_u = smoothstep(6.0, 8.1, t)
        heal_u = smoothstep(25.5, 28.4, t)
        sk_closed.value = zip_u * (1.0 - heal_u)
        sk_healed.value = heal_u
        sk_closed.keyframe_insert(data_path="value", frame=f)
        sk_healed.keyframe_insert(data_path="value", frame=f)

        # --- Skin Inflammation Crimson Blush (Beat 3 peak -> fades to healthy skin in Beat 6) ---
        inf_pulse = 0.08 * math.sin(t * 7.8) * smoothstep(10.0, 11.2, t) * (1.0 - smoothstep(15.5, 17.5, t))
        inf_level = 0.14 + 0.72 * smoothstep(10.1, 12.2, t) * (1.0 - smoothstep(19.5, 27.2, t)) + inf_pulse
        inf_level *= (1.0 - 0.90 * heal_u)
        inf_val.default_value = clamp01(inf_level)
        inf_val.keyframe_insert(data_path="default_value", frame=f)
        scar_val.default_value = 0.36 * heal_u
        scar_val.keyframe_insert(data_path="default_value", frame=f)

        # --- Beat 1: Curved Surgical Needle Weaving Through 6 Suture Loops (t=0.4..5.6s) ---
        if t < 6.0:
            pass_f = clamp01((t - 0.35) / 5.15) * NUM_LOOPS
            idx_p = min(NUM_LOOPS - 1, int(pass_f))
            frac = pass_f - idx_p
            ly = lerp(-0.108, 0.108, idx_p / float(NUM_LOOPS - 1))
            w_o = wound_open_half_width(ly) + 0.018
            nx = lerp(-w_o, w_o, smoothstep(0.08, 0.92, frac))
            nz = limb_surface_z(abs(nx)) + 0.022 * math.sin(math.pi * frac) - 0.004
            needle_obj.location = Vector((nx, ly, nz))
            needle_obj.rotation_euler = Euler((0.0, math.radians(lerp(-55.0, 55.0, frac)), 0.0), "XYZ")
            ns = max(0.0001, smoothstep(0.15, 0.40, t) * (1.0 - smoothstep(5.55, 5.95, t)))
            needle_obj.scale = (ns, ns, ns)
        else:
            needle_obj.scale = (0.0001, 0.0001, 0.0001)
        needle_obj.keyframe_insert(data_path="location", frame=f)
        needle_obj.keyframe_insert(data_path="rotation_euler", frame=f)
        needle_obj.keyframe_insert(data_path="scale", frame=f)

        # --- 6 Blue Nylon Suture Loops: Appear sequentially in Beat 1, cinch shut in Beat 2, slide out in Beat 6! ---
        slide_out = smoothstep(26.0, 27.8, t)
        for idx, (l_obj, sk_tight, knot, ly, w_o, w_c) in enumerate(suture_loops):
            t_appear = 0.55 + idx * 0.82
            vis_in = smoothstep(t_appear, t_appear + 0.45, t)
            # Progressive zipper cinch from bottom loop (idx=0) to top loop (idx=5)!
            loop_zip = smoothstep(6.0 + idx * 0.24, 7.0 + idx * 0.24, t)
            sk_tight.value = loop_zip
            sk_tight.keyframe_insert(data_path="value", frame=f)

            # Slide smoothly up-and-out during Beat 6 stitch removal!
            l_out = smoothstep(26.0 + idx * 0.18, 27.0 + idx * 0.18, t)
            l_vis = vis_in * (1.0 - smoothstep(27.1 + idx * 0.15, 27.6 + idx * 0.15, t))
            ls = max(0.0001, l_vis)
            l_obj.location = Vector((0.025 * l_out, 0.0, 0.065 * l_out))
            l_obj.scale = (ls, ls, ls)
            l_obj.keyframe_insert(data_path="location", frame=f)
            l_obj.keyframe_insert(data_path="scale", frame=f)

            ks = max(0.0001, smoothstep(7.4 + idx * 0.12, 7.8 + idx * 0.12, t) * (1.0 - l_out))
            knot.location = Vector((w_c + 0.003 + 0.025 * l_out, ly, limb_surface_z(w_c) + 0.0045 + 0.065 * l_out))
            knot.scale = (ks, ks, ks)
            knot.keyframe_insert(data_path="location", frame=f)
            knot.keyframe_insert(data_path="scale", frame=f)

        # --- Trailing Pull Thread (Beat 2 Zipper Pull -> Snipped Away at t=8.65s!) ---
        tail_in = smoothstep(5.4, 6.0, t)
        tail_snip = smoothstep(8.65, 9.25, t)
        ts = max(0.0001, tail_in * (1.0 - tail_snip))
        tail_obj.location = Vector((0.018 + 0.05 * tail_snip, 0.108, limb_surface_z(0.018) + 0.002 + 0.045 * zip_u + 0.08 * tail_snip))
        tail_obj.scale = (ts, ts, ts)
        tail_obj.keyframe_insert(data_path="location", frame=f)
        tail_obj.keyframe_insert(data_path="scale", frame=f)

        # --- Articulated 3D Surgical Scissors (Beat 2 Snip @ t=8.65s AND Beat 6 Removal Snip @ t=25.65s!) ---
        sc2_vis = smoothstep(7.95, 8.35, t) * (1.0 - smoothstep(9.15, 9.65, t))
        sc6_vis = smoothstep(25.15, 25.45, t) * (1.0 - smoothstep(26.15, 26.55, t))
        if sc2_vis > 0.001:
            sc_s = sc2_vis
            sciss_root.location = Vector((lerp(0.16, 0.112, smoothstep(8.0, 8.55, t)), 0.108, 0.045))
            sciss_root.rotation_euler = Euler((math.radians(15.0), math.radians(-12.0), math.radians(-18.0)), "XYZ")
            open_ang = 22.0 * (1.0 - smoothstep(8.55, 8.68, t))
        elif sc6_vis > 0.001:
            sc_s = sc6_vis
            sciss_root.location = Vector((lerp(0.16, 0.112, smoothstep(25.2, 25.55, t)), -0.02, 0.032))
            sciss_root.rotation_euler = Euler((math.radians(12.0), math.radians(-14.0), math.radians(-12.0)), "XYZ")
            open_ang = 22.0 * (1.0 - smoothstep(25.58, 25.70, t))
        else:
            sc_s = 0.0001
            open_ang = 0.0
        sciss_root.scale = (sc_s, sc_s, sc_s)
        sciss_root.keyframe_insert(data_path="location", frame=f)
        sciss_root.keyframe_insert(data_path="rotation_euler", frame=f)
        sciss_root.keyframe_insert(data_path="scale", frame=f)
        blade_a.rotation_euler = Euler((0.0, math.radians(+open_ang), 0.0), "XYZ")
        blade_b.rotation_euler = Euler((0.0, math.radians(-open_ang), 0.0), "XYZ")
        blade_a.keyframe_insert(data_path="rotation_euler", frame=f)
        blade_b.keyframe_insert(data_path="rotation_euler", frame=f)

        # --- Beat 3: 140 Glowing Repair Cells Swarming Into Wound Seam (t=10.3..15.6s) ---
        cell_vis = smoothstep(10.3, 11.2, t) * (1.0 - smoothstep(14.8, 15.8, t))
        if cell_vis > 0.001 or abs(t - 10.2) < 0.25 or abs(t - 15.9) < 0.25 or f == 1:
            for obj, xs, xe, yt, stg, ph in repair_cells:
                cu = smoothstep(10.4 + stg * 1.4, 13.8 + stg * 1.0, t)
                cx = lerp(xs, xe, cu) + 0.004 * math.sin(t * 4.5 + ph)
                cy = yt + 0.005 * math.cos(t * 3.8 + ph)
                cz = limb_surface_z(abs(cx)) + 0.0038 + 0.002 * math.sin(t * 6.0 + ph)
                cs = max(0.0001, cell_vis * (0.75 + 0.25 * math.sin(t * 5.0 + ph)))
                obj.location = Vector((cx, cy, cz))
                obj.scale = (cs, cs, cs)
                obj.keyframe_insert(data_path="location", frame=f)
                obj.keyframe_insert(data_path="scale", frame=f)

        # --- Beat 4: 36 Golden Collagen Fibers Weaving Across Seam & Morphing Into Living Pink Skin! ---
        col_morph = smoothstep(18.0, 20.1, t)
        col_mix_fac.default_value = col_morph
        col_mix_fac.keyframe_insert(data_path="default_value", frame=f)
        col_em_val.default_value = lerp(1.75, 0.18, col_morph)
        col_em_val.keyframe_insert(data_path="default_value", frame=f)

        col_fade_out = 1.0 - smoothstep(23.5, 25.4, t)
        for obj, stg in collagen_strands:
            c_grow = smoothstep(15.3 + stg * 2.0, 16.4 + stg * 2.0, t) * col_fade_out
            cs = max(0.0001, c_grow)
            obj.scale = (cs, cs, cs)
            obj.keyframe_insert(data_path="scale", frame=f)

        # --- Beat 5: 18 Branching Ruby Capillary Blood Vessels Growing, Pulsing & Sinking Into Skin! ---
        sink_u = smoothstep(23.3, 25.4, t)
        cap_root.location = Vector((0.0, 0.0, lerp(0.0, -0.016, sink_u)))
        cap_root.keyframe_insert(data_path="location", frame=f)
        if vessel_em is not None:
            vessel_em.default_value = (0.75 + 0.45 * (0.5 + 0.5 * math.sin(t * 8.2))) * (1.0 - sink_u)
            vessel_em.keyframe_insert(data_path="default_value", frame=f)
        for obj, stg in cap_trees:
            v_grow = smoothstep(20.5 + stg * 1.8, 21.8 + stg * 1.8, t) * (1.0 - smoothstep(24.9, 25.5, t))
            vs = max(0.0001, v_grow)
            obj.scale = (vs, vs, vs)
            obj.keyframe_insert(data_path="scale", frame=f)

    # 4. Render Requested Frame Range
    out_dir = Path(args.frames_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_to_render = [48, 168, 210, 312, 432, 540, 672] if args.preview else list(range(max(1, args.start_frame), min(TOTAL_FRAMES, args.end_frame) + 1))

    for f in frames_to_render:
        scene.frame_set(f)
        scene.render.filepath = str(out_dir / f"frame_{f:04d}.jpg")
        bpy.ops.render.render(write_still=True)

    print(f"[How Stitches Work 4K] Rendered {len(frames_to_render)} frame(s) into {out_dir}")


if __name__ == "__main__":
    build_and_render(parse_args())
