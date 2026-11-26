"""
How Stitches Work — V2.1 Photoreal 4K Medical Macro Cinema (Blender 4.2.3 LTS Cycles OPTIX + OIDN)

Visually QA-verified across all 7 hero beats on real MakeHuman MPFB2 4K Photo-Scanned Human Anatomy:
  1. Complete Bare Human Right Leg (Thigh + Knee + Shin + Calf + Ankle Malleoli + Heel + 5 Toes) resting
     on an infinite sterile dark-teal surgical examination pad (ZERO cardboard knee tent, ZERO black void!).
  2. Natural Warm Sun-Kissed 4K Photo-Scanned Human Skin (`middleage_lightskinned_male_diffuse.png`) with
     true microscopic `0.5mm` (`Scale=1900`) skin grain, velvety `0.56..0.68` roughness & softbox dome lighting
     (ZERO plastic glare stripe, ZERO reptile scales!) + ~380 curved 3D human leg hairs on outer flanks.
  3. Surgically Bisected 3D Multi-Layer Shin Incision (`bisect_plane` + rich coral-crimson dermis walls +
     deep vascular bed + tucked 0.85mm golden adipose lobules) with `Basis` (Open), `Closed` (Taut V-crease +
     12 sunken needle-puncture dimples + quilted thread compression), and `Healed` (Seamless 4K skin) shape keys.
  4. Authentic Surgical Instruments & 12-Puncture Suturing:
       - Mayo-Hegar Needle-Holder Forceps + curved 3/8-circle steel needle + swaged Prolene thread weaving 12 passes
       - 6 Twisted 3-Wrap Surgeon's Knots with twin `4.5mm` cut whisker tails (ZERO lollipop balls!)
       - Sleek single-piece beveled Curved Iris Surgical Scissors straddling & snipping the tension tail at t=8.65s
       - Fine Toothed Adson Forceps extracting each stitch sequentially in Beat 6!
  5. Clean Sub-Epidermal Translucent X-Ray Dermal Window (Beats 3-5):
       - Beat 3: Crimson arterial blush + sculpted biconcave RBCs, golden-pearl Macrophages & Fibroblasts inside cleft
       - Beat 4: 60 sinuous curved golden collagen fibers strictly inside the dermal cleft knitting into pink dermis
       - Beat 5: Tortuous crimson arterioles & deep-ruby venules branching & anastomosing across the healed cleft!
"""
from __future__ import annotations

import argparse
import glob
import importlib
import math
import os
import random
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RES_X, RES_Y = 1080, 1920
FPS = 24
DURATION = 30.0
TOTAL_FRAMES = int(round(DURATION * FPS))  # 720 frames

WOUND_HALF_LEN = 0.054   # 10.8 cm clean anterior shin incision
WOUND_MAX_OPEN = 0.0125  # 12.5 mm half-gap (25 mm realistic gaping surgical incision)
WOUND_DEPTH = 0.0068     # 6.8 mm deep 3D multi-layer dermal/subcutaneous cavity
NUM_STITCHES = 6         # 6 horizontal rungs = 12 skin punctures!


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


def wound_half_gap(y: float) -> float:
    u = abs(y) / WOUND_HALF_LEN
    if u >= 1.0:
        return 0.0
    return WOUND_MAX_OPEN * (math.cos(u * math.pi * 0.5) ** 1.30)


# ============================================================================
# 1. MPFB2 LOADER & 4K PHOTO-SCANNED TEXTURE FINDER
# ============================================================================
def enable_mpfb2_services():
    if not any(m.endswith("mpfb.services.humanservice") for m in sys.modules):
        for mod_name in ("bl_ext.user_default.mpfb", "mpfb"):
            try:
                bpy.ops.preferences.addon_enable(module=mod_name)
                if any(m.endswith("mpfb.services.humanservice") for m in sys.modules):
                    break
            except Exception:
                continue

    def dyn(pkg: str, key: str):
        for amod in list(sys.modules.keys()):
            if amod.endswith(pkg):
                m = importlib.import_module(amod)
                if hasattr(m, key):
                    return getattr(m, key)
        raise ImportError(f"MPFB2 '{pkg}.{key}' not found")

    return (
        dyn("mpfb.services.humanservice", "HumanService"),
        dyn("mpfb.services.targetservice", "TargetService"),
        dyn("mpfb.services.locationservice", "LocationService"),
        dyn("mpfb.entities.objectproperties", "HumanObjectProperties"),
    )


def find_4k_skin_diffuse_path(LocationService) -> str | None:
    search_roots = [
        LocationService.get_user_data(),
        "/tmp/mpfb_extracted",
        os.path.expanduser("~/.config/blender/4.2/extensions/.user/user_default/mpfb/data"),
    ]
    for root in search_roots:
        if root and os.path.isdir(root):
            hits = glob.glob(os.path.join(root, "**", "*lightskinned_male_diffuse.png"), recursive=True)
            if hits:
                return hits[0]
    return None


# ============================================================================
# 2. PHOTOREAL V2.1 SHADERS (VELVETY 4K SKIN + SUBDERMAL X-RAY + CORAL DERMIS)
# ============================================================================
def make_v2_4k_human_skin_mat(diffuse_img_path: str | None):
    """Natural, warm, velvety 4K photo-scanned human skin shader:
    - Uses real MakeHuman 4K photo-scanned diffuse map (`middleage_lightskinned_male_diffuse.png`)
    - Microscopic 0.5mm (`Scale=1900`) Voronoi micro-grain & velvety `0.56..0.68` roughness (ZERO plastic glare, ZERO scales!)
    - Animated radial crimson inflammation blush + crisp dark-rose apposition seam
    - Animated Sub-Epidermal Translucent X-Ray Cutaway Window (`cutaway_val`) for Beats 3-5!
    """
    mat = bpy.data.materials.new("V21_Photoreal4KHumanSkin")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")

    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep_obj = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep_obj.inputs["Vector"])

    if diffuse_img_path and os.path.isfile(diffuse_img_path):
        tex_img = nt.nodes.new("ShaderNodeTexImage")
        tex_img.image = bpy.data.images.load(diffuse_img_path, check_existing=True)
        nt.links.new(tc.outputs["UV"], tex_img.inputs["Vector"])
        gamma = nt.nodes.new("ShaderNodeGamma")
        gamma.inputs["Gamma"].default_value = 1.14
        nt.links.new(tex_img.outputs["Color"], gamma.inputs["Color"])
        warm_mul = nt.nodes.new("ShaderNodeMix")
        warm_mul.data_type = "RGBA"
        warm_mul.blend_type = "MULTIPLY"
        wf, wa, wb, wout = get_rgba_mix_sockets(warm_mul)
        wf.default_value = 0.70
        wb.default_value = (0.94, 0.80, 0.70, 1.0)
        nt.links.new(gamma.outputs["Color"], wa)
        base_skin_socket = wout
    else:
        ns_fb = nt.nodes.new("ShaderNodeTexNoise")
        ns_fb.inputs["Scale"].default_value = 18.0
        rmp_fb = nt.nodes.new("ShaderNodeValToRGB")
        rmp_fb.color_ramp.elements[0].color = (0.30, 0.16, 0.105, 1.0)
        rmp_fb.color_ramp.elements[-1].color = (0.42, 0.24, 0.165, 1.0)
        nt.links.new(tc.outputs["Object"], ns_fb.inputs["Vector"])
        nt.links.new(ns_fb.outputs["Fac"], rmp_fb.inputs["Fac"])
        base_skin_socket = rmp_fb.outputs["Color"]

    # Radial elliptical wound-zone mask: r2 = (x / 0.021)^2 + (y / 0.060)^2
    dx = nt.nodes.new("ShaderNodeMath"); dx.operation = "DIVIDE"; dx.inputs[1].default_value = 0.021
    sx = nt.nodes.new("ShaderNodeMath"); sx.operation = "POWER"; sx.inputs[1].default_value = 2.0
    dy = nt.nodes.new("ShaderNodeMath"); dy.operation = "DIVIDE"; dy.inputs[1].default_value = 0.060
    sy = nt.nodes.new("ShaderNodeMath"); sy.operation = "POWER"; sy.inputs[1].default_value = 2.0
    r2 = nt.nodes.new("ShaderNodeMath"); r2.operation = "ADD"
    nt.links.new(sep_obj.outputs["X"], dx.inputs[0]); nt.links.new(dx.outputs["Value"], sx.inputs[0])
    nt.links.new(sep_obj.outputs["Y"], dy.inputs[0]); nt.links.new(dy.outputs["Value"], sy.inputs[0])
    nt.links.new(sx.outputs["Value"], r2.inputs[0]); nt.links.new(sy.outputs["Value"], r2.inputs[1])

    mr_inf = nt.nodes.new("ShaderNodeMapRange")
    mr_inf.clamp = True
    mr_inf.interpolation_type = "SMOOTHSTEP"
    mr_inf.inputs["From Min"].default_value = 0.04
    mr_inf.inputs["From Max"].default_value = 1.10
    mr_inf.inputs["To Min"].default_value = 1.0
    mr_inf.inputs["To Max"].default_value = 0.0
    nt.links.new(r2.outputs["Value"], mr_inf.inputs["Value"])

    inf_drive = nt.nodes.new("ShaderNodeValue")
    inf_drive.outputs[0].default_value = 0.18
    inf_mul = nt.nodes.new("ShaderNodeMath"); inf_mul.operation = "MULTIPLY"; inf_mul.use_clamp = True
    nt.links.new(mr_inf.outputs["Result"], inf_mul.inputs[0])
    nt.links.new(inf_drive.outputs[0], inf_mul.inputs[1])

    mix_inf = nt.nodes.new("ShaderNodeMix"); mix_inf.data_type = "RGBA"
    if_fac, if_a, if_b, if_out = get_rgba_mix_sockets(mix_inf)
    if_b.default_value = (0.36, 0.045, 0.042, 1.0)
    nt.links.new(inf_mul.outputs["Value"], if_fac)
    nt.links.new(base_skin_socket, if_a)

    # Crisp dark-rose apposition seam shading along |X| < 0.0024
    abs_x = nt.nodes.new("ShaderNodeMath"); abs_x.operation = "ABSOLUTE"
    abs_y = nt.nodes.new("ShaderNodeMath"); abs_y.operation = "ABSOLUTE"
    mr_seam_x = nt.nodes.new("ShaderNodeMapRange"); mr_seam_x.clamp = True
    mr_seam_x.inputs["From Min"].default_value = 0.0; mr_seam_x.inputs["From Max"].default_value = 0.0024
    mr_seam_x.inputs["To Min"].default_value = 1.0; mr_seam_x.inputs["To Max"].default_value = 0.0
    mr_seam_y = nt.nodes.new("ShaderNodeMapRange"); mr_seam_y.clamp = True
    mr_seam_y.inputs["From Min"].default_value = 0.046; mr_seam_y.inputs["From Max"].default_value = 0.055
    mr_seam_y.inputs["To Min"].default_value = 1.0; mr_seam_y.inputs["To Max"].default_value = 0.0
    seam_xy = nt.nodes.new("ShaderNodeMath"); seam_xy.operation = "MULTIPLY"
    seam_drive = nt.nodes.new("ShaderNodeValue"); seam_drive.outputs[0].default_value = 0.85
    seam_mul = nt.nodes.new("ShaderNodeMath"); seam_mul.operation = "MULTIPLY"; seam_mul.use_clamp = True
    mix_seam = nt.nodes.new("ShaderNodeMix"); mix_seam.data_type = "RGBA"
    sm_f, sm_a, sm_b, sm_out = get_rgba_mix_sockets(mix_seam)
    sm_b.default_value = (0.15, 0.020, 0.020, 1.0)

    nt.links.new(sep_obj.outputs["X"], abs_x.inputs[0]); nt.links.new(abs_x.outputs["Value"], mr_seam_x.inputs["Value"])
    nt.links.new(sep_obj.outputs["Y"], abs_y.inputs[0]); nt.links.new(abs_y.outputs["Value"], mr_seam_y.inputs["Value"])
    nt.links.new(mr_seam_x.outputs["Result"], seam_xy.inputs[0]); nt.links.new(mr_seam_y.outputs["Result"], seam_xy.inputs[1])
    nt.links.new(seam_xy.outputs["Value"], seam_mul.inputs[0]); nt.links.new(seam_drive.outputs[0], seam_mul.inputs[1])
    nt.links.new(seam_mul.outputs["Value"], sm_f); nt.links.new(if_out, sm_a)
    nt.links.new(sm_out, bsdf.inputs["Base Color"])

    # Microscopic 0.5mm Human Skin Grain (Scale=1900 -> fine velvety human pores, ZERO big scales or wet shine!)
    vor_sulci = nt.nodes.new("ShaderNodeTexVoronoi")
    vor_sulci.feature = "DISTANCE_TO_EDGE"
    vor_sulci.inputs["Scale"].default_value = 1900.0
    vor_sulci.inputs["Randomness"].default_value = 0.90

    ns_micro = nt.nodes.new("ShaderNodeTexNoise")
    ns_micro.inputs["Scale"].default_value = 650.0
    ns_micro.inputs["Detail"].default_value = 5.0

    mr_rough = nt.nodes.new("ShaderNodeMapRange"); mr_rough.clamp = True
    mr_rough.inputs["From Min"].default_value = 0.0; mr_rough.inputs["From Max"].default_value = 0.25
    mr_rough.inputs["To Min"].default_value = 0.68; mr_rough.inputs["To Max"].default_value = 0.54
    nt.links.new(tc.outputs["Object"], vor_sulci.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], ns_micro.inputs["Vector"])
    nt.links.new(vor_sulci.outputs["Distance"], mr_rough.inputs["Value"])
    nt.links.new(mr_rough.outputs["Result"], bsdf.inputs["Roughness"])

    bmp_sulci = nt.nodes.new("ShaderNodeBump")
    bmp_sulci.inputs["Strength"].default_value = 0.075
    bmp_sulci.inputs["Distance"].default_value = 0.00020
    bmp_micro = nt.nodes.new("ShaderNodeBump")
    bmp_micro.inputs["Strength"].default_value = 0.055
    bmp_micro.inputs["Distance"].default_value = 0.00025
    nt.links.new(vor_sulci.outputs["Distance"], bmp_sulci.inputs["Height"])
    nt.links.new(ns_micro.outputs["Fac"], bmp_micro.inputs["Height"])
    nt.links.new(bmp_sulci.outputs["Normal"], bmp_micro.inputs["Normal"])
    nt.links.new(bmp_micro.outputs["Normal"], bsdf.inputs["Normal"])

    set_bsdf(bsdf, "Subsurface Weight", 0.30)
    set_bsdf(bsdf, "Subsurface Scale", 0.0018)
    set_bsdf(bsdf, "Subsurface Radius", (1.0, 0.36, 0.18))
    if hasattr(bsdf, "subsurface_method"):
        bsdf.subsurface_method = "BURLEY"
    set_bsdf(bsdf, "Specular IOR Level", 0.36)
    set_bsdf(bsdf, "Sheen Weight", 0.18)
    set_bsdf(bsdf, "Sheen Roughness", 0.55)

    # Sub-Epidermal X-Ray Cutaway Window (strictly inside the stitched ellipse r2 < 0.72!)
    mr_xray = nt.nodes.new("ShaderNodeMapRange"); mr_xray.clamp = True; mr_xray.interpolation_type = "SMOOTHSTEP"
    mr_xray.inputs["From Min"].default_value = 0.22
    mr_xray.inputs["From Max"].default_value = 0.72
    mr_xray.inputs["To Min"].default_value = 1.0
    mr_xray.inputs["To Max"].default_value = 0.0
    nt.links.new(r2.outputs["Value"], mr_xray.inputs["Value"])

    cutaway_drive = nt.nodes.new("ShaderNodeValue")
    cutaway_drive.outputs[0].default_value = 0.0
    cut_mul = nt.nodes.new("ShaderNodeMath"); cut_mul.operation = "MULTIPLY"; cut_mul.use_clamp = True
    nt.links.new(mr_xray.outputs["Result"], cut_mul.inputs[0])
    nt.links.new(cutaway_drive.outputs[0], cut_mul.inputs[1])

    tr_bsdf = nt.nodes.new("ShaderNodeBsdfTransparent")
    tr_bsdf.inputs["Color"].default_value = (0.98, 0.90, 0.86, 1.0)
    mix_cut = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(cut_mul.outputs["Value"], mix_cut.inputs["Fac"])
    nt.links.new(bsdf.outputs["BSDF"], mix_cut.inputs[1])
    nt.links.new(tr_bsdf.outputs["BSDF"], mix_cut.inputs[2])
    nt.links.new(mix_cut.outputs["Shader"], out.inputs["Surface"])

    return mat, inf_drive.outputs[0], seam_drive.outputs[0], cutaway_drive.outputs[0]


def make_multilayer_wound_wall_mat():
    """Rich 3-Layer Anatomical Wound Cavity driven deterministically by Object Z depth below Z=0."""
    mat = bpy.data.materials.new("V21_MultiLayerDermalWall")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    mr_z = nt.nodes.new("ShaderNodeMapRange")
    mr_z.clamp = True
    mr_z.inputs["From Min"].default_value = -0.0003
    mr_z.inputs["From Max"].default_value = -0.0065
    mr_z.inputs["To Min"].default_value = 0.0
    mr_z.inputs["To Max"].default_value = 1.0

    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 160.0
    ns.inputs["Detail"].default_value = 8.0

    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.02
    rmp.color_ramp.elements[0].color = (0.38, 0.14, 0.10, 1.0)    # Epidermal cut rim
    e1 = rmp.color_ramp.elements.new(0.16)
    e1.color = (0.34, 0.030, 0.028, 1.0)                          # Rich coral-crimson Dermis wall!
    e2 = rmp.color_ramp.elements.new(0.60)
    e2.color = (0.22, 0.014, 0.015, 1.0)                          # Deep vascular reticular dermis
    rmp.color_ramp.elements[-1].position = 0.95
    rmp.color_ramp.elements[-1].color = (0.12, 0.006, 0.008, 1.0) # Deep moist wound bed

    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.28
    bmp.inputs["Distance"].default_value = 0.0005

    set_bsdf(bsdf, "Roughness", 0.20)
    set_bsdf(bsdf, "Coat Weight", 0.78)
    set_bsdf(bsdf, "Coat Roughness", 0.06)
    set_bsdf(bsdf, "Subsurface Weight", 0.30)
    set_bsdf(bsdf, "Subsurface Scale", 0.0016)

    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], mr_z.inputs["Value"])
    nt.links.new(mr_z.outputs["Result"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_adipose_fat_mat():
    mat = bpy.data.materials.new("SubcutaneousAdiposeMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    set_bsdf(bsdf, "Base Color", (0.48, 0.24, 0.042, 1.0))
    set_bsdf(bsdf, "Roughness", 0.24)
    set_bsdf(bsdf, "Coat Weight", 0.65)
    set_bsdf(bsdf, "Subsurface Weight", 0.45)
    set_bsdf(bsdf, "Subsurface Scale", 0.0018)
    return mat


def make_leg_hair_mat():
    mat = bpy.data.materials.new("FineHumanLegHairMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    set_bsdf(bsdf, "Base Color", (0.075, 0.045, 0.024, 1.0))
    set_bsdf(bsdf, "Roughness", 0.42)
    return mat


def make_prolene_suture_mat():
    mat = bpy.data.materials.new("Prolene3_0_SurgicalSuture")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    set_bsdf(bsdf, "Base Color", (0.005, 0.028, 0.18, 1.0))
    set_bsdf(bsdf, "Roughness", 0.18)
    set_bsdf(bsdf, "Specular IOR Level", 0.72)
    set_bsdf(bsdf, "Coat Weight", 0.85)
    set_bsdf(bsdf, "Coat Roughness", 0.06)
    return mat


def make_puncture_pore_mat():
    mat = bpy.data.materials.new("NeedlePuncturePoreMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    set_bsdf(bsdf, "Base Color", (0.11, 0.015, 0.016, 1.0))
    set_bsdf(bsdf, "Roughness", 0.30)
    return mat


def make_surgical_steel_mat(name: str = "SurgicalGradeSteel", gold_ring: bool = False):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    if gold_ring:
        set_bsdf(bsdf, "Base Color", (0.86, 0.65, 0.22, 1.0))
        set_bsdf(bsdf, "Metallic", 1.0)
        set_bsdf(bsdf, "Roughness", 0.16)
    else:
        set_bsdf(bsdf, "Base Color", (0.84, 0.87, 0.92, 1.0))
        set_bsdf(bsdf, "Metallic", 1.0)
        set_bsdf(bsdf, "Roughness", 0.12)
        set_bsdf(bsdf, "Anisotropic", 0.50)
    return mat


def make_sterile_drape_mat():
    mat = bpy.data.materials.new("SterileSurgicalTableMat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 220.0
    ns.inputs["Detail"].default_value = 5.0
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.12
    bmp.inputs["Distance"].default_value = 0.0008
    set_bsdf(bsdf, "Base Color", (0.014, 0.048, 0.068, 1.0))
    set_bsdf(bsdf, "Roughness", 0.85)
    set_bsdf(bsdf, "Sheen Weight", 0.35)
    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


# ============================================================================
# 3. COMPLETE BARE HUMAN RIGHT LEG (THIGH -> KNEE -> SHIN -> ANKLE -> 5 TOES)
# ============================================================================
def build_anatomical_human_leg_with_3d_wound(scene, skin_mat, wall_mat, fat_mat, hair_mat):
    HumanService, TargetService, LocationService, HumanObjectProperties = enable_mpfb2_services()

    h_obj = HumanService.create_human(feet_on_ground=True, scale=0.1)
    HumanObjectProperties.set_value("gender", 1.0, entity_reference=h_obj)
    HumanObjectProperties.set_value("age", 0.38, entity_reference=h_obj)
    HumanObjectProperties.set_value("muscle", 0.56, entity_reference=h_obj)
    HumanObjectProperties.set_value("weight", 0.48, entity_reference=h_obj)
    HumanObjectProperties.set_value("height", 0.58, entity_reference=h_obj)
    TargetService.reapply_macro_details(h_obj)

    rig = HumanService.add_builtin_rig(h_obj, "default")
    bpy.context.view_layer.update()
    if "foot.R" in rig.pose.bones:
        pb = rig.pose.bones["foot.R"]
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = Euler((math.radians(28.0), 0.0, math.radians(-4.0)), "XYZ")
    bpy.context.view_layer.update()

    deps = bpy.context.evaluated_depsgraph_get()
    baked_mesh = bpy.data.meshes.new_from_object(h_obj.evaluated_get(deps))
    body_vg_idx = h_obj.vertex_groups["body"].index if "body" in h_obj.vertex_groups else None

    bpy.data.objects.remove(h_obj, do_unlink=True)
    bpy.data.objects.remove(rig, do_unlink=True)

    # Keep FULL Right Leg all the way up through Knee & Upper Thigh (v.co.z <= 0.84) so it extends naturally off top frame!
    bm = bmesh.new()
    bm.from_mesh(baked_mesh)
    bpy.data.meshes.remove(baked_mesh)

    dvert = bm.verts.layers.deform.active
    to_del = []
    for v in bm.verts:
        in_body = True
        if body_vg_idx is not None and dvert is not None:
            in_body = (body_vg_idx in v[dvert]) and (v[dvert][body_vg_idx] > 0.5)
        if (not in_body) or (v.co.x > -0.010) or (v.co.z > 0.92):
            to_del.append(v)
    bmesh.ops.delete(bm, geom=to_del, context="VERTS")
    # Smoothly extend upper thigh above knee (z > 0.62) so it continues cleanly off the top-right edge of wide shots!
    for v in bm.verts:
        if v.co.z > 0.62:
            v.co.z += (v.co.z - 0.62) * 0.85

    bmesh.ops.transform(bm, matrix=Matrix.Rotation(-math.pi * 0.5, 4, "X"), verts=bm.verts)

    def ridge_pt(y_target: float) -> Vector:
        band = [v.co.copy() for v in bm.verts if abs(v.co.y - y_target) < 0.016]
        band.sort(key=lambda p: p.z, reverse=True)
        top_n = band[: max(1, min(4, len(band)))]
        return sum(top_n, Vector((0, 0, 0))) / len(top_n)

    p_lo = ridge_pt(0.22)
    p_hi = ridge_pt(0.38)
    p_mid = ridge_pt(0.30)

    d_vec = p_hi - p_lo
    yaw_fix = math.atan2(d_vec.x, d_vec.y)
    pitch_fix = -math.atan2(d_vec.z, math.hypot(d_vec.x, d_vec.y))
    M_align = Matrix.Rotation(pitch_fix, 4, "X") @ Matrix.Rotation(yaw_fix, 4, "Z") @ Matrix.Translation(-p_mid)
    bmesh.ops.transform(bm, matrix=M_align, verts=bm.verts)

    top_band = [v.co.copy() for v in bm.verts if abs(v.co.y) < 0.025 and v.co.z > -0.015]
    if top_band:
        pk = max(top_band, key=lambda p: p.z)
        bmesh.ops.translate(bm, vec=-Vector((pk.x, 0.0, pk.z)), verts=bm.verts)

    leg_mesh = bpy.data.meshes.new("AnatomicalHumanLeg_Mesh")
    bm.to_mesh(leg_mesh)
    bm.free()
    leg_obj = bpy.data.objects.new("AnatomicalHumanLeg", leg_mesh)
    link_obj(scene, leg_obj)
    sub = leg_obj.modifiers.new("SmoothAnatomicalSubd", "SUBSURF")
    sub.levels = 2
    sub.render_levels = 2
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    sub_mesh = bpy.data.meshes.new_from_object(leg_obj.evaluated_get(deps))
    leg_obj.modifiers.clear()
    old_m = leg_obj.data
    leg_obj.data = sub_mesh
    bpy.data.meshes.remove(old_m)

    bm = bmesh.new()
    bm.from_mesh(leg_obj.data)
    uv_lay = bm.loops.layers.uv.verify()

    zone_edges = [
        e for e in bm.edges
        if all(abs(v.co.x) < 0.032 and abs(v.co.y) < 0.078 and v.co.z > -0.022 for v in e.verts)
    ]
    if zone_edges:
        bmesh.ops.subdivide_edges(bm, edges=zone_edges, cuts=1, use_grid_fill=True)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])

    top_pts = [(v.co.x, v.co.y, v.co.z) for v in bm.verts if abs(v.co.x) < 0.045 and abs(v.co.y) < 0.095 and v.co.z > -0.03]

    def shin_z_at(x: float, y: float) -> float:
        best_d2 = 1e9
        best_z = 0.0
        for px, py, pz in top_pts:
            d2 = (px - x) ** 2 + (py - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_z = pz
        return best_z

    cut_geom = [
        g for g in (list(bm.verts) + list(bm.edges) + list(bm.faces))
        if (
            isinstance(g, bmesh.types.BMVert) and abs(g.co.x) < 0.025 and abs(g.co.y) < WOUND_HALF_LEN + 0.015 and g.co.z > -0.020
        ) or (
            not isinstance(g, bmesh.types.BMVert) and any(abs(v.co.x) < 0.025 and abs(v.co.y) < WOUND_HALF_LEN + 0.015 and v.co.z > -0.020 for v in g.verts)
        )
    ]
    bmesh.ops.bisect_plane(bm, geom=cut_geom, dist=1e-5, plane_co=Vector((0, 0, 0)), plane_no=Vector((1, 0, 0)))

    seam_verts = [v for v in bm.verts if abs(v.co.x) < 1.5e-4 and abs(v.co.y) <= WOUND_HALF_LEN and v.co.z > -0.012]
    seam_verts.sort(key=lambda v: v.co.y)

    left_chain = [seam_verts[0]]
    right_chain = [seam_verts[0]]

    for v_orig in seam_verts[1:-1]:
        v_right = bm.verts.new(v_orig.co.copy())
        right_faces = [f for f in list(v_orig.link_faces) if f.calc_center_median().x > 0.0]
        for f in right_faces:
            f_verts = [v_right if fv == v_orig else fv for fv in f.verts]
            f_uvs = [lp[uv_lay].uv.copy() for lp in f.loops]
            mat_idx = f.material_index
            bm.faces.remove(f)
            try:
                nf = bm.faces.new(f_verts)
                nf.material_index = mat_idx
                for lp, uv_c in zip(nf.loops, f_uvs):
                    lp[uv_lay].uv = uv_c
            except ValueError:
                pass
        left_chain.append(v_orig)
        right_chain.append(v_right)

    left_chain.append(seam_verts[-1])
    right_chain.append(seam_verts[-1])

    prof = [(-0.88, 0.16), (-0.68, 0.65), (0.00, 1.00), (+0.68, 0.65), (+0.88, 0.16)]
    uv_depths = [0.0, 0.16, 0.65, 1.00, 0.65, 0.16, 0.0]
    cavity_rings = []
    for k in range(1, len(left_chain) - 1):
        base_co = left_chain[k].co.copy()
        row = [left_chain[k]]
        for _, depth_u in prof:
            row.append(bm.verts.new((base_co.x, base_co.y, base_co.z - WOUND_DEPTH * depth_u)))
        row.append(right_chain[k])
        cavity_rings.append((k, base_co, row))

    for idx_r in range(len(cavity_rings) - 1):
        _, _, r_a = cavity_rings[idx_r]
        _, _, r_b = cavity_rings[idx_r + 1]
        for s in range(6):
            try:
                f = bm.faces.new((r_a[s], r_b[s], r_b[s + 1], r_a[s + 1]))
                f.material_index = 1
                for lp, d_u in zip(f.loops, (uv_depths[s], uv_depths[s], uv_depths[s + 1], uv_depths[s + 1])):
                    lp[uv_lay].uv = (0.5, d_u)
            except ValueError:
                pass

    if cavity_rings:
        v_bot = seam_verts[0]
        _, _, r_first = cavity_rings[0]
        for s in range(6):
            try:
                f = bm.faces.new((v_bot, r_first[s], r_first[s + 1]))
                f.material_index = 1
                for lp, d_u in zip(f.loops, (0.0, uv_depths[s], uv_depths[s + 1])):
                    lp[uv_lay].uv = (0.5, d_u)
            except ValueError:
                pass
        v_top = seam_verts[-1]
        _, _, r_last = cavity_rings[-1]
        for s in range(6):
            try:
                f = bm.faces.new((r_last[s], v_top, r_last[s + 1]))
                f.material_index = 1
                for lp, d_u in zip(f.loops, (uv_depths[s], 0.0, uv_depths[s + 1])):
                    lp[uv_lay].uv = (0.5, d_u)
            except ValueError:
                pass

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.verts.ensure_lookup_table()
    n_verts = len(bm.verts)
    co_healed = [v.co.copy() for v in bm.verts]
    co_open = [v.co.copy() for v in bm.verts]
    co_closed = [v.co.copy() for v in bm.verts]

    cavity_vert_info = {}
    for _, base_co, row in cavity_rings:
        ux_list = [-1.0, -0.88, -0.68, 0.0, +0.68, +0.88, +1.0]
        for s_i, vv in enumerate(row):
            if 1 <= s_i <= 5:
                cavity_vert_info[vv.index] = (ux_list[s_i], uv_depths[s_i], base_co)

    left_lip_set = {v.index for v in left_chain[1:-1]}
    right_lip_set = {v.index for v in right_chain[1:-1]}
    stitch_ys = [lerp(-0.040, 0.040, i / float(NUM_STITCHES - 1)) for i in range(NUM_STITCHES)]
    CLOSED_PUNCTURE_X = 0.0115

    for vi in range(n_verts):
        p = co_healed[vi]
        if vi in cavity_vert_info:
            ux, du, bco = cavity_vert_info[vi]
            w_g = wound_half_gap(bco.y)
            co_open[vi] = Vector((ux * w_g, bco.y, bco.z - WOUND_DEPTH * du))
            co_closed[vi] = Vector((ux * 0.0004, bco.y, bco.z - 0.0022 - WOUND_DEPTH * du * 0.85))
            continue

        if abs(p.y) < WOUND_HALF_LEN + 0.025 and p.z > -0.025 and abs(p.x) < 0.048:
            if vi in left_lip_set:
                side, dist_x = -1.0, 0.0
            elif vi in right_lip_set:
                side, dist_x = +1.0, 0.0
            else:
                side = -1.0 if p.x < 0.0 else (+1.0 if p.x > 0.0 else 0.0)
                dist_x = abs(p.x)

            w_g = wound_half_gap(p.y)
            falloff = math.exp(-dist_x / 0.016)
            dx_open = side * w_g * falloff
            lip_roll = -0.0010 * math.exp(-dist_x / 0.0024) * (w_g / WOUND_MAX_OPEN)
            z_follow = shin_z_at(p.x + dx_open, p.y) - shin_z_at(p.x, p.y)
            co_open[vi] = Vector((p.x + dx_open, p.y, p.z + z_follow * 0.65 + lip_roll))

            in_len = math.cos(min(1.0, abs(p.y) / WOUND_HALF_LEN) * math.pi * 0.5) if abs(p.y) < WOUND_HALF_LEN else 0.0
            v_groove = -0.0016 * math.exp(-dist_x / 0.0013) * in_len
            eversion = +0.00080 * math.exp(-((dist_x - 0.0028) / 0.0022) ** 2) * in_len
            dimple_dz = 0.0
            groove_dz = 0.0
            for sy in stitch_ys:
                for sx_p in (-CLOSED_PUNCTURE_X, +CLOSED_PUNCTURE_X):
                    r_p2 = (p.x - sx_p) ** 2 + (p.y - sy) ** 2
                    dimple_dz -= 0.0014 * math.exp(-r_p2 / (0.0021 ** 2))
                if abs(p.x) <= CLOSED_PUNCTURE_X + 0.002:
                    groove_dz -= 0.00070 * math.exp(-((p.y - sy) / 0.0020) ** 2)
            co_closed[vi] = Vector((p.x, p.y, p.z + v_groove + eversion + dimple_dz + groove_dz))

    final_mesh = bpy.data.meshes.new("V21_SurgicalHumanLeg_Mesh")
    final_mesh.materials.append(skin_mat)
    final_mesh.materials.append(wall_mat)
    bm.to_mesh(final_mesh)
    bm.free()
    final_mesh.update()
    for poly in final_mesh.polygons:
        poly.use_smooth = True

    old_d = leg_obj.data
    leg_obj.data = final_mesh
    bpy.data.meshes.remove(old_d)

    sk_basis = leg_obj.shape_key_add(name="Basis", from_mix=False)
    for i, co in enumerate(co_open):
        sk_basis.data[i].co = co
    sk_closed = leg_obj.shape_key_add(name="Closed", from_mix=False)
    for i, co in enumerate(co_closed):
        sk_closed.data[i].co = co
    sk_healed = leg_obj.shape_key_add(name="Healed", from_mix=False)
    for i, co in enumerate(co_healed):
        sk_healed.data[i].co = co

    # Subtle 0.85mm deep-tucked adipose lobules along lower dermal margin
    rng_f = random.Random(77)
    fat_lobules = []
    for i in range(32):
        side = -1.0 if i % 2 == 0 else 1.0
        fy = lerp(-WOUND_HALF_LEN * 0.78, WOUND_HALF_LEN * 0.78, i / 31.0)
        w_g = wound_half_gap(fy)
        fx_open = side * (w_g * 0.62 + rng_f.uniform(-0.0008, 0.0008))
        fz = shin_z_at(0.0, fy) - WOUND_DEPTH * rng_f.uniform(0.72, 0.90)
        bm_f = bmesh.new()
        bmesh.ops.create_icosphere(bm_f, subdivisions=1, radius=rng_f.uniform(0.00075, 0.00115))
        f_obj = add_mesh(scene, f"AdiposeLobule_{i:02d}", bm_f, fat_mat, loc=(fx_open, fy, fz), smooth=True, shadow=False)
        fat_lobules.append((f_obj, fx_open, fy, fz))

    # ~380 Fine Curved 3D Human Leg Hairs along outer shin & calf flanks (|X| > 0.022m)
    bm_hair = bmesh.new()
    rng_h = random.Random(101)
    hair_candidates = [
        (co_healed[i], final_mesh.vertices[i].normal.copy())
        for i in range(0, n_verts, 3)
        if abs(co_healed[i].x) > 0.022 and -0.18 < co_healed[i].y < 0.34 and co_healed[i].z > -0.045
    ]
    rng_h.shuffle(hair_candidates)
    for root_p, nrm in hair_candidates[:380]:
        if nrm.length < 1e-4:
            continue
        n_u = nrm.normalized()
        grow_dir = (n_u * 0.35 + Vector((0.0, -0.92, 0.08))).normalized()
        h_len = rng_h.uniform(0.0040, 0.0070)
        p_mid_h = root_p + (n_u * 0.50 + grow_dir * 0.50).normalized() * (h_len * 0.5)
        p_tip_h = root_p + grow_dir * h_len
        for a, b, r1, r2 in [(root_p, p_mid_h, 0.00014, 0.00009), (p_mid_h, p_tip_h, 0.00009, 0.00003)]:
            dv = b - a
            q = dv.normalized().to_track_quat("Z", "Y")
            m = Matrix.Translation((a + b) * 0.5) @ q.to_matrix().to_4x4()
            bmesh.ops.create_cone(bm_hair, cap_ends=False, cap_tris=False, segments=4, radius1=r1, radius2=r2, depth=dv.length * 1.05, matrix=m)
    add_mesh(scene, "HumanLegVellusHairs", bm_hair, hair_mat, smooth=True, shadow=False)

    return leg_obj, sk_closed, sk_healed, fat_lobules, shin_z_at, stitch_ys


# ============================================================================
# 4. SUTURES, TWISTED SURGEON'S KNOTS & SLEEK SURGICAL INSTRUMENTS
# ============================================================================
def build_tube_with_tight_shapekey(scene, name: str, pts_open: list[Vector], pts_closed: list[Vector], radius: float, mat, segs_r: int = 10):
    def make_verts(pts):
        out_v = []
        for i, p in enumerate(pts):
            t_dir = (pts[min(len(pts) - 1, i + 1)] - pts[max(0, i - 1)]).normalized()
            ref = Vector((0.0, 1.0, 0.0)) if abs(t_dir.y) < 0.92 else Vector((1.0, 0.0, 0.0))
            b1 = t_dir.cross(ref).normalized()
            b2 = t_dir.cross(b1).normalized()
            for j in range(segs_r):
                a = 2.0 * math.pi * j / segs_r
                out_v.append(p + b1 * (radius * math.cos(a)) + b2 * (radius * math.sin(a)))
        return out_v

    vo = make_verts(pts_open)
    vc = make_verts(pts_closed)
    faces = []
    for i in range(len(pts_open) - 1):
        for j in range(segs_r):
            jn = (j + 1) % segs_r
            faces.append((i * segs_r + j, i * segs_r + jn, (i + 1) * segs_r + jn, (i + 1) * segs_r + j))

    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata([tuple(v) for v in vo], [], faces)
    mesh.update()
    for p in mesh.polygons:
        p.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(mat)
    link_obj(scene, obj)

    obj.shape_key_add(name="Basis", from_mix=False)
    sk = obj.shape_key_add(name="Tight", from_mix=False)
    for i, v in enumerate(vc):
        sk.data[i].co = v
    return obj, sk


def build_surgeons_square_knot(scene, name: str, loc: Vector, suture_mat):
    root = bpy.data.objects.new(name, None)
    root.location = loc
    link_obj(scene, root)

    bm = bmesh.new()
    for w_idx in range(3):
        z0 = w_idx * 0.00062
        segs = 16
        r_maj = 0.00140 - w_idx * 0.00010
        r_min = 0.00046
        pts = [
            Vector((r_maj * math.cos(2 * math.pi * i / segs), r_maj * 0.75 * math.sin(2 * math.pi * i / segs), z0 + 0.0002 * math.sin(4 * math.pi * i / segs)))
            for i in range(segs + 1)
        ]
        for i in range(segs):
            dv = pts[i + 1] - pts[i]
            q = dv.normalized().to_track_quat("Z", "Y")
            m = Matrix.Translation((pts[i] + pts[i + 1]) * 0.5) @ q.to_matrix().to_4x4()
            bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=6, radius1=r_min, radius2=r_min, depth=dv.length * 1.15, matrix=m)

    for sgn in (-1.0, +1.0):
        p0 = Vector((0.0, 0.0, 0.0014))
        dir_t = Vector((0.52, sgn * 0.52, 0.68)).normalized()
        p1 = p0 + dir_t * 0.0046
        dv = p1 - p0
        q = dv.normalized().to_track_quat("Z", "Y")
        m = Matrix.Translation((p0 + p1) * 0.5) @ q.to_matrix().to_4x4()
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=8, radius1=0.00045, radius2=0.00040, depth=dv.length, matrix=m)

    k_mesh = add_mesh(scene, f"{name}_Mesh", bm, suture_mat, smooth=True, shadow=True)
    k_mesh.parent = root
    return root


def build_suture_system(scene, suture_mat, pore_mat, shin_z_at, stitch_ys):
    CLOSED_X = 0.0115
    THREAD_R = 0.00058
    rungs, pores, knots = [], [], []

    for idx, sy in enumerate(stitch_ys):
        open_x = wound_half_gap(sy) + CLOSED_X
        n_pts = 26
        pts_o, pts_c = [], []
        for k in range(n_pts + 1):
            u = k / float(n_pts)
            xo = lerp(-open_x, open_x, u)
            xc = lerp(-CLOSED_X, CLOSED_X, u)
            zo = shin_z_at(xo, sy) + math.sin(math.pi * u) * 0.0095 - 0.0011
            zc = shin_z_at(xc, sy) + math.sin(math.pi * u) * 0.0022 - 0.0009
            pts_o.append(Vector((xo, sy, zo)))
            pts_c.append(Vector((xc, sy, zc)))

        rung_obj, sk_tight = build_tube_with_tight_shapekey(scene, f"SutureRung_{idx:02d}", pts_o, pts_c, THREAD_R, suture_mat)
        rungs.append((rung_obj, sk_tight, sy, open_x, CLOSED_X))

        for s_lbl, sgn in [("L", -1.0), ("R", +1.0)]:
            bm_p = bmesh.new()
            bmesh.ops.create_cone(bm_p, cap_ends=True, cap_tris=False, segments=12, radius1=0.0012, radius2=0.0008, depth=0.0008)
            p_obj = add_mesh(scene, f"PuncturePore_{idx:02d}_{s_lbl}", bm_p, pore_mat, smooth=True, shadow=False)
            pores.append((p_obj, idx, sgn, sy, open_x, CLOSED_X))

        kn = build_surgeons_square_knot(scene, f"SurgeonKnot_{idx:02d}", Vector((CLOSED_X, sy, shin_z_at(CLOSED_X, sy) + 0.0004)), suture_mat)
        knots.append((kn, idx, sy, CLOSED_X))

    diag_laces = []
    for idx in range(NUM_STITCHES - 1):
        y0, y1 = stitch_ys[idx], stitch_ys[idx + 1]
        ox0 = wound_half_gap(y0) + CLOSED_X
        ox1 = wound_half_gap(y1) + CLOSED_X
        pts_o, pts_c = [], []
        for k in range(21):
            u = k / 20.0
            yo = lerp(y0, y1, u)
            xo = lerp(+ox0, -ox1, u)
            xc = lerp(+CLOSED_X, -CLOSED_X, u)
            zo = shin_z_at(0.0, yo) - 0.0020 + math.sin(math.pi * u) * 0.0014
            zc = shin_z_at(0.0, yo) - 0.0024
            pts_o.append(Vector((xo, yo, zo)))
            pts_c.append(Vector((xc, yo, zc)))
        d_obj, d_sk = build_tube_with_tight_shapekey(scene, f"SutureDiag_{idx:02d}", pts_o, pts_c, THREAD_R * 0.90, suture_mat)
        diag_laces.append((d_obj, d_sk, idx))

    bm_tail = bmesh.new()
    bmesh.ops.create_cone(bm_tail, cap_ends=True, cap_tris=False, segments=10, radius1=THREAD_R, radius2=THREAD_R, depth=0.075, matrix=Matrix.Translation((0, 0, 0.0375)))
    pull_tail = add_mesh(scene, "SutureTensionTail", bm_tail, suture_mat, smooth=True, shadow=True)

    return rungs, diag_laces, pores, knots, pull_tail


def build_needle_and_holder_rig(scene, steel_mat, gold_mat, suture_mat):
    rig_root = bpy.data.objects.new("NeedleAndHolder_Root", None)
    link_obj(scene, rig_root)

    bm_n = bmesh.new()
    r_arc = 0.0165
    segs = 24
    ring_prev = None
    swage_pt = Vector((0, 0, 0))
    for i in range(segs + 1):
        u = i / float(segs)
        ang = lerp(-math.pi * 0.44, math.pi * 0.44, u)
        cx = r_arc * math.sin(ang)
        cz = -r_arc * math.cos(ang) + r_arc * 0.85
        if i == 0:
            swage_pt = Vector((cx, 0.0, cz))
        rad = max(0.00010, 0.00068 * (1.0 - 0.86 * (u ** 2.8)))
        tx, tz = math.cos(ang), math.sin(ang)
        nr = Vector((-tz, 0.0, tx)).normalized()
        ny = Vector((0.0, 1.0, 0.0))
        b_ring = [bm_n.verts.new(Vector((cx, 0.0, cz)) + nr * (rad * math.cos(2 * math.pi * j / 10)) + ny * (rad * math.sin(2 * math.pi * j / 10))) for j in range(10)]
        if ring_prev is not None:
            for j in range(10):
                jn = (j + 1) % 10
                bm_n.faces.new((ring_prev[j], ring_prev[jn], b_ring[jn], b_ring[j]))
        ring_prev = b_ring
    needle = add_mesh(scene, "CurvedSurgicalNeedle_V21", bm_n, steel_mat, smooth=True, shadow=True)
    needle.parent = rig_root

    bm_tr = bmesh.new()
    tr_pts = [swage_pt + Vector((-0.028 * u, -0.012 * math.sin(u * 2.2), 0.032 * (u ** 1.25))) for u in [k / 16.0 for k in range(17)]]
    for k in range(16):
        dv = tr_pts[k + 1] - tr_pts[k]
        q = dv.normalized().to_track_quat("Z", "Y")
        m = Matrix.Translation((tr_pts[k] + tr_pts[k + 1]) * 0.5) @ q.to_matrix().to_4x4()
        bmesh.ops.create_cone(bm_tr, cap_ends=True, cap_tris=False, segments=8, radius1=0.00056, radius2=0.00056, depth=dv.length * 1.1, matrix=m)
    swaged_thread = add_mesh(scene, "NeedleSwagedThread", bm_tr, suture_mat, smooth=True, shadow=True)
    swaged_thread.parent = rig_root

    bm_h = bmesh.new()
    grip_dir = Vector((0.42, 0.45, 0.78)).normalized()
    q_h = grip_dir.to_track_quat("Z", "Y")
    for sgn in (-1.0, 1.0):
        m_jaw = Matrix.Translation((0.0, 0.0, 0.001)) @ q_h.to_matrix().to_4x4() @ Matrix.Translation((sgn * 0.0011, 0.0, 0.011))
        bmesh.ops.create_cone(bm_h, cap_ends=True, cap_tris=False, segments=10, radius1=0.0013, radius2=0.0024, depth=0.022, matrix=m_jaw)
        m_sh = Matrix.Translation((0.0, 0.0, 0.001)) @ q_h.to_matrix().to_4x4() @ Matrix.Translation((sgn * 0.0032, 0.0, 0.058))
        bmesh.ops.create_cone(bm_h, cap_ends=True, cap_tris=False, segments=10, radius1=0.0020, radius2=0.0022, depth=0.074, matrix=m_sh)
    m_pv = Matrix.Translation((0.0, 0.0, 0.001)) @ q_h.to_matrix().to_4x4() @ Matrix.Translation((0.0, 0.0, 0.022)) @ Matrix.Rotation(math.radians(90.0), 4, "X")
    bmesh.ops.create_cone(bm_h, cap_ends=True, cap_tris=False, segments=14, radius1=0.0034, radius2=0.0034, depth=0.0055, matrix=m_pv)
    holder = add_mesh(scene, "MayoHegarNeedleHolder", bm_h, steel_mat, smooth=True, shadow=True)
    holder.parent = rig_root

    bm_g = bmesh.new()
    for sgn in (-1.0, 1.0):
        m_gr = Matrix.Translation((0.0, 0.0, 0.001)) @ q_h.to_matrix().to_4x4() @ Matrix.Translation((sgn * 0.0036, 0.0, 0.092))
        bmesh.ops.create_cone(bm_g, cap_ends=True, cap_tris=False, segments=10, radius1=0.0025, radius2=0.0028, depth=0.012, matrix=m_gr)
    gold_trim = add_mesh(scene, "NeedleHolderGoldRings", bm_g, gold_mat, smooth=True, shadow=False)
    gold_trim.parent = rig_root
    return rig_root


def build_sleek_iris_scissors(scene, steel_mat, gold_mat):
    """Single-piece sleek beveled surgical steel Iris Scissors (ZERO bead segments!) whose cutting point sits at X = -0.022m."""
    root = bpy.data.objects.new("IrisSurgicalScissors_Root", None)
    link_obj(scene, root)

    def make_sleek_blade(name: str, sgn: float):
        bm = bmesh.new()
        # Continuous lofted beveled surgical steel blade from X = +0.006 (pivot) to X = -0.036 (sharp tip!)
        # Cutting contact zone is right around X = -0.022!
        n_s = 12
        rings = []
        for k in range(n_s + 1):
            u = k / float(n_s)
            x = lerp(0.006, -0.036, u)
            w_z = lerp(0.0028, 0.00025, u ** 0.85)
            th_y = lerp(0.0009, 0.00018, u)
            y_inner = sgn * 0.00015
            y_outer = sgn * (0.00015 + th_y)
            # 4-corner beveled surgical blade cross-section
            r_verts = [
                bm.verts.new((x, y_inner, -w_z * 0.5)),  # Inner cutting edge
                bm.verts.new((x, y_outer, -w_z * 0.3)),  # Outer bevel
                bm.verts.new((x, y_outer, +w_z * 0.5)),  # Spine outer
                bm.verts.new((x, y_inner, +w_z * 0.5)),  # Spine inner
            ]
            rings.append(r_verts)
        for k in range(n_s):
            ra, rb = rings[k], rings[k + 1]
            for j in range(4):
                jn = (j + 1) % 4
                bm.faces.new((ra[j], ra[jn], rb[jn], rb[j]))
        bm.faces.new(rings[0][::-1])
        bm.faces.new(rings[-1])

        # Smooth handle shank extending toward +X (+0.006 -> +0.082m)
        m_sh = Matrix.Translation((0.044, sgn * 0.0012, sgn * 0.0032)) @ Matrix.Rotation(math.radians(90.0), 4, "Y")
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=12, radius1=0.0020, radius2=0.0016, depth=0.076, matrix=m_sh)
        obj = add_mesh(scene, name, bm, steel_mat, smooth=False, shadow=True)
        es = obj.modifiers.new("CrispBevel", "EDGE_SPLIT")
        es.split_angle = math.radians(35.0)
        obj.parent = root
        return obj

    b_a = make_sleek_blade("IrisBlade_A", +1.0)
    b_b = make_sleek_blade("IrisBlade_B", -1.0)
    bm_p = bmesh.new()
    bmesh.ops.create_cone(bm_p, cap_ends=True, cap_tris=False, segments=16, radius1=0.0024, radius2=0.0024, depth=0.0036, matrix=Matrix.Rotation(math.radians(90.0), 4, "X"))
    pv = add_mesh(scene, "IrisPivotScrew", bm_p, gold_mat, smooth=True, shadow=False)
    pv.parent = root
    return root, b_a, b_b


def build_adson_tweezers(scene, steel_mat):
    root = bpy.data.objects.new("AdsonForceps_Root", None)
    link_obj(scene, root)
    bm = bmesh.new()
    grip_dir = Vector((-0.35, 0.30, 0.88)).normalized()
    q = grip_dir.to_track_quat("Z", "Y")
    for sgn in (-1.0, 1.0):
        m = q.to_matrix().to_4x4() @ Matrix.Translation((sgn * 0.0014, 0.0, 0.036)) @ Matrix.Diagonal((0.45, 1.5, 1.0, 1.0))
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=10, radius1=0.00055, radius2=0.0022, depth=0.072, matrix=m)
    tw = add_mesh(scene, "AdsonForceps_Mesh", bm, steel_mat, smooth=True, shadow=True)
    tw.parent = root
    return root


# ============================================================================
# 5. SUB-EPIDERMAL BIOLOGICAL HEALING VFX (STRICTLY INSIDE DERMAL CLEFT!)
# ============================================================================
def build_subdermal_healing_systems(scene, shin_z_at):
    # 1. Beat 3 Sculpted RBCs + Ruffled Golden-Pearl Macrophages + Spindle Fibroblasts
    rbc_mat = bpy.data.materials.new("SculptedRBCMat")
    rbc_mat.use_nodes = True
    b_rbc = rbc_mat.node_tree.nodes["Principled BSDF"]
    set_bsdf(b_rbc, "Base Color", (0.45, 0.014, 0.016, 1.0))
    set_bsdf(b_rbc, "Roughness", 0.22)
    set_bsdf(b_rbc, "Subsurface Weight", 0.45)
    set_bsdf(b_rbc, "Emission Color", (0.62, 0.022, 0.025, 1.0))
    set_bsdf(b_rbc, "Emission Strength", 0.50)

    wbc_mat = bpy.data.materials.new("MacrophageGoldPearlMat")
    wbc_mat.use_nodes = True
    b_wbc = wbc_mat.node_tree.nodes["Principled BSDF"]
    set_bsdf(b_wbc, "Base Color", (0.94, 0.76, 0.40, 1.0))
    set_bsdf(b_wbc, "Roughness", 0.25)
    set_bsdf(b_wbc, "Emission Color", (0.98, 0.78, 0.36, 1.0))
    set_bsdf(b_wbc, "Emission Strength", 0.95)

    fib_mat = bpy.data.materials.new("SpindleFibroblastMat")
    fib_mat.use_nodes = True
    b_fib = fib_mat.node_tree.nodes["Principled BSDF"]
    set_bsdf(b_fib, "Base Color", (0.88, 0.42, 0.38, 1.0))
    set_bsdf(b_fib, "Roughness", 0.25)
    set_bsdf(b_fib, "Emission Color", (0.95, 0.50, 0.40, 1.0))
    set_bsdf(b_fib, "Emission Strength", 0.85)

    bm_rbc = bmesh.new()
    bmesh.ops.create_uvsphere(bm_rbc, u_segments=14, v_segments=10, radius=0.00135)
    for v in bm_rbc.verts:
        r_xy = math.hypot(v.co.x, v.co.y) / 0.00135
        v.co.z *= (0.28 + 0.52 * (r_xy ** 2))
    bmesh.ops.recalc_face_normals(bm_rbc, faces=bm_rbc.faces)
    rbc_mesh = bpy.data.meshes.new("BiconcaveRBC_SharedMesh")
    bm_rbc.to_mesh(rbc_mesh)
    bm_rbc.free()
    for p in rbc_mesh.polygons:
        p.use_smooth = True
    rbc_mesh.materials.append(rbc_mat)

    bm_wbc = bmesh.new()
    bmesh.ops.create_icosphere(bm_wbc, subdivisions=2, radius=0.00165)
    rng_m = random.Random(303)
    for v in bm_wbc.verts:
        v.co *= rng_m.uniform(0.84, 1.16)
    bmesh.ops.recalc_face_normals(bm_wbc, faces=bm_wbc.faces)
    wbc_mesh = bpy.data.meshes.new("RuffledMacrophage_SharedMesh")
    bm_wbc.to_mesh(wbc_mesh)
    bm_wbc.free()
    for p in wbc_mesh.polygons:
        p.use_smooth = True
    wbc_mesh.materials.append(wbc_mat)

    bm_fib = bmesh.new()
    bmesh.ops.create_icosphere(bm_fib, subdivisions=1, radius=0.00115, matrix=Matrix.Diagonal((2.5, 0.65, 0.65, 1.0)))
    bmesh.ops.recalc_face_normals(bm_fib, faces=bm_fib.faces)
    fib_mesh = bpy.data.meshes.new("SpindleFibroblast_SharedMesh")
    bm_fib.to_mesh(fib_mesh)
    bm_fib.free()
    for p in fib_mesh.polygons:
        p.use_smooth = True
    fib_mesh.materials.append(fib_mat)

    rng = random.Random(2026)
    cells = []
    c_idx = 0
    for shared_m, count in [(rbc_mesh, 65), (wbc_mesh, 42), (fib_mesh, 35)]:
        for _ in range(count):
            side = -1.0 if c_idx % 2 == 0 else 1.0
            yt = rng.uniform(-0.042, 0.042)
            # Strictly inside the dermal cutaway chamber (|X| <= 0.011m)!
            xs = side * rng.uniform(0.0065, 0.0110)
            xe = side * rng.uniform(0.0004, 0.0048)
            z_sub = shin_z_at(0.0, yt) - rng.uniform(0.0024, 0.0048)
            obj = bpy.data.objects.new(f"BioCell_{c_idx:03d}", shared_m)
            obj.visible_shadow = False
            link_obj(scene, obj)
            cells.append((obj, xs, xe, yt, z_sub, rng.uniform(0.0, 1.0), rng.uniform(0.0, 6.28)))
            c_idx += 1

    # 2. Beat 4 Organic Sinuous Curved Golden 3D Collagen Fibers (STRICTLY inside |X| <= 0.0085m dermal cleft!)
    col_mat = bpy.data.materials.new("V21_OrganicCollagenWebMat")
    col_mat.use_nodes = True
    nt_c = col_mat.node_tree
    bsdf_c = nt_c.nodes["Principled BSDF"]
    mix_c = nt_c.nodes.new("ShaderNodeMix"); mix_c.data_type = "RGBA"
    col_fac, c_gold, c_pink, c_out = get_rgba_mix_sockets(mix_c)
    col_fac.default_value = 0.0
    c_gold.default_value = (0.96, 0.66, 0.14, 1.0)  # Rich warm 24K amber-gold collagen
    c_pink.default_value = (0.48, 0.10, 0.09, 1.0)  # Regenerated rose-crimson dermal tissue!
    col_em = nt_c.nodes.new("ShaderNodeValue"); col_em.outputs[0].default_value = 1.15
    set_bsdf(bsdf_c, "Roughness", 0.28)
    set_bsdf(bsdf_c, "Subsurface Weight", 0.45)
    nt_c.links.new(c_out, bsdf_c.inputs["Base Color"])
    nt_c.links.new(c_out, bsdf_c.inputs["Emission Color"])
    nt_c.links.new(col_em.outputs[0], bsdf_c.inputs["Emission Strength"])

    collagen_bundles = []
    for i in range(60):
        y0 = lerp(-0.043, 0.043, i / 59.0)
        # Strictly inside the dermal chamber (|X| <= 0.0085m, well inside the ±0.0115m suture punctures!)
        x_w = rng.uniform(0.0058, 0.0086) * (1.0 - 0.35 * (abs(y0) / 0.046) ** 2)
        slant = rng.uniform(-0.0055, 0.0055)
        z_b = shin_z_at(0.0, y0) - rng.uniform(0.0022, 0.0042)
        wave_ph = rng.uniform(0.0, 6.28)

        bm_c = bmesh.new()
        n_s = 16
        s_pts = []
        for k in range(n_s + 1):
            u = k / float(n_s)
            x_u = lerp(-x_w, +x_w, u)
            # Natural S-curve biological fibril wave (looks like real microscopy collagen, NEVER straight toothpicks!)
            y_u = y0 + lerp(-slant, +slant, u) + 0.0018 * math.sin(u * math.pi * 2.5 + wave_ph)
            z_u = z_b + 0.0006 * math.cos(u * math.pi * 2.0 + wave_ph)
            s_pts.append(Vector((x_u, y_u, z_u)))
        for k in range(n_s):
            dv = s_pts[k + 1] - s_pts[k]
            q = dv.normalized().to_track_quat("Z", "Y")
            m = Matrix.Translation((s_pts[k] + s_pts[k + 1]) * 0.5) @ q.to_matrix().to_4x4()
            bmesh.ops.create_cone(bm_c, cap_ends=True, cap_tris=False, segments=6, radius1=0.00024, radius2=0.00024, depth=dv.length * 1.12, matrix=m)
        c_obj = add_mesh(scene, f"CollagenFiber_{i:02d}", bm_c, col_mat, smooth=True, shadow=False)
        collagen_bundles.append((c_obj, i / 60.0))

    # 3. Beat 5 Deep Ruby-Crimson Tortuous Arterioles & Venules (strictly inside |X| <= 0.0092m!)
    ves_mat = bpy.data.materials.new("V21_RubyMicrovasculatureMat")
    ves_mat.use_nodes = True
    bsdf_v = ves_mat.node_tree.nodes["Principled BSDF"]
    set_bsdf(bsdf_v, "Base Color", (0.36, 0.006, 0.008, 1.0))
    set_bsdf(bsdf_v, "Roughness", 0.16)
    set_bsdf(bsdf_v, "Coat Weight", 0.85)
    set_bsdf(bsdf_v, "Emission Color", (0.68, 0.012, 0.016, 1.0))
    ves_em = bsdf_v.inputs["Emission Strength"]
    ves_em.default_value = 0.65

    vessel_root = bpy.data.objects.new("SubdermalVessels_Root", None)
    link_obj(scene, vessel_root)

    def add_vseg(bm, a: Vector, b: Vector, r1: float, r2: float):
        dv = b - a
        if dv.length < 1e-5:
            return
        q = dv.normalized().to_track_quat("Z", "Y")
        m = Matrix.Translation((a + b) * 0.5) @ q.to_matrix().to_4x4()
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=8, radius1=r1, radius2=r2, depth=dv.length * 1.1, matrix=m)

    bm_trunks = bmesh.new()
    for side in (-1.0, 1.0):
        t_pts = []
        for k in range(29):
            yv = lerp(-0.043, 0.043, k / 28.0)
            taper = 1.0 - 0.30 * (abs(yv) / 0.046) ** 2
            xv = side * (0.0088 * taper + 0.0009 * math.sin(k * 0.9))
            zv = shin_z_at(0.0, yv) - 0.0030
            t_pts.append(Vector((xv, yv, zv)))
        for k in range(28):
            add_vseg(bm_trunks, t_pts[k], t_pts[k + 1], 0.00062, 0.00062)
    trunks_obj = add_mesh(scene, "ParentDermalArterioles", bm_trunks, ves_mat, smooth=True, shadow=False)
    trunks_obj.parent = vessel_root

    cap_sprouts = []
    for i in range(26):
        side = -1.0 if i % 2 == 0 else 1.0
        yb = lerp(-0.040, 0.040, i / 25.0)
        taper = 1.0 - 0.30 * (abs(yb) / 0.046) ** 2
        zb = shin_z_at(0.0, yb) - 0.0026
        # Tortuous curved capillary path snaking across X=0!
        bm_v = bmesh.new()
        n_v = 10
        v_pts = []
        x_start = side * 0.0088 * taper
        x_end = -side * rng.uniform(0.0020, 0.0055)
        ph_v = rng.uniform(0.0, 6.28)
        for k in range(n_v + 1):
            u = k / float(n_v)
            vx = lerp(x_start, x_end, u)
            vy = yb + 0.0024 * math.sin(u * math.pi * 2.0 + ph_v)
            vz = zb + 0.0004 * math.sin(u * math.pi)
            v_pts.append(Vector((vx, vy, vz)))
        for k in range(n_v):
            r0 = lerp(0.00052, 0.00020, k / float(n_v))
            r1 = lerp(0.00052, 0.00020, (k + 1) / float(n_v))
            add_vseg(bm_v, v_pts[k], v_pts[k + 1], r0, r1)
        # Secondary fine fork
        fork_p0 = v_pts[4]
        fork_p1 = fork_p0 + Vector((-side * 0.0038, rng.choice([-1.0, 1.0]) * 0.0042, 0.0002))
        add_vseg(bm_v, fork_p0, fork_p1, 0.00032, 0.00014)

        v_obj = add_mesh(scene, f"CapillarySprout_{i:02d}", bm_v, ves_mat, smooth=True, shadow=False)
        v_obj.parent = vessel_root
        cap_sprouts.append((v_obj, i / 26.0))

    return cells, collagen_bundles, col_fac, col_em.outputs[0], vessel_root, trunks_obj, cap_sprouts, ves_em


# ============================================================================
# 6. SOFTBOX SURGICAL SUITE LIGHTING & 6-BEAT MACRO CAMERA
# ============================================================================
def setup_surgical_environment_and_lights(scene):
    world = bpy.data.worlds.new("SurgicalSuiteWorld_V21")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.014, 0.042, 0.062, 1.0)
    bg.inputs["Strength"].default_value = 0.55

    # Huge 4.5m x 4.5m sterile surgical examination pad (ZERO table edges or black void anywhere in frame!)
    bm_pad = bmesh.new()
    bmesh.ops.create_grid(bm_pad, x_segments=16, y_segments=16, size=2.4, matrix=Matrix.Translation((0.0, 0.10, -0.068)))
    add_mesh(scene, "InfiniteSurgicalTablePad", bm_pad, make_sterile_drape_mat(), smooth=True, shadow=True)

    # Large diffused 0.85m Softbox Surgical Dome Key (velvety smooth skin gradients, zero harsh plastic stripe!)
    k_d = bpy.data.lights.new("SurgicalSoftboxKey", type="AREA")
    k_d.shape = "DISK"
    k_d.size = 0.85
    k_d.energy = 8.2
    k_d.color = (1.0, 0.96, 0.92)
    k_o = bpy.data.objects.new("SurgicalSoftboxKey", k_d)
    k_o.location = Vector((-0.42, -0.22, 0.62))
    k_o.rotation_euler = (Vector((0, 0, 0)) - k_o.location).to_track_quat("-Z", "Y").to_euler()
    link_obj(scene, k_o)

    # Cool Clinical Rim Light
    r_d = bpy.data.lights.new("SurgicalCoolRim", type="AREA")
    r_d.shape = "RECTANGLE"
    r_d.size = 0.70
    r_d.size_y = 0.45
    r_d.energy = 5.6
    r_d.color = (0.72, 0.90, 1.0)
    r_o = bpy.data.objects.new("SurgicalCoolRim", r_d)
    r_o.location = Vector((0.46, 0.28, 0.46))
    r_o.rotation_euler = (Vector((0, 0, 0)) - r_o.location).to_track_quat("-Z", "Y").to_euler()
    link_obj(scene, r_o)

    # Wide Warm Bounce Fill
    f_d = bpy.data.lights.new("SurgicalWarmFill", type="AREA")
    f_d.shape = "RECTANGLE"
    f_d.size = 0.90
    f_d.size_y = 0.65
    f_d.energy = 3.6
    f_d.color = (1.0, 0.94, 0.88)
    f_o = bpy.data.objects.new("SurgicalWarmFill", f_d)
    f_o.location = Vector((0.34, -0.44, 0.44))
    f_o.rotation_euler = (Vector((0, 0, 0)) - f_o.location).to_track_quat("-Z", "Y").to_euler()
    link_obj(scene, f_o)

    bio_d = bpy.data.lights.new("SubdermalBioGlow", type="POINT")
    bio_d.energy = 0.0
    bio_d.color = (1.0, 0.40, 0.22)
    bio_d.shadow_soft_size = 0.022
    bio_o = bpy.data.objects.new("SubdermalBioGlow", bio_d)
    bio_o.location = Vector((0.0, 0.0, -0.003))
    link_obj(scene, bio_o)
    return bio_o


def compute_v2_camera(t: float) -> tuple[Vector, Vector, float, float]:
    if t < 6.0:
        u_zoom = smoothstep(0.15, 2.10, t)
        u_pan = smoothstep(1.20, 5.70, t)
        pos = Vector((
            lerp(0.16, 0.095, u_zoom),
            lerp(-0.44, -0.155, u_zoom) + lerp(-0.015, 0.025, u_pan),
            lerp(0.42, 0.215, u_zoom),
        ))
        tgt = Vector((
            lerp(-0.010, 0.0, u_zoom),
            lerp(-0.065, 0.0, u_zoom) + lerp(-0.018, 0.018, u_pan),
            lerp(0.005, 0.0, u_zoom),
        ))
        return pos, tgt, lerp(48.0, 62.0, u_zoom), lerp(7.0, 8.0, u_zoom)
    if t < 10.2:
        u = smoothstep(6.0, 10.0, t)
        pos = Vector((lerp(0.095, 0.082, u), lerp(-0.130, -0.115, u), lerp(0.215, 0.205, u)))
        tgt = Vector((0.0, lerp(0.018, 0.012, u), 0.0))
        return pos, tgt, lerp(62.0, 64.0, u), 8.5
    if t < 15.2:
        u = smoothstep(10.2, 15.0, t)
        pos = Vector((lerp(0.082, -0.075, u), lerp(-0.115, -0.125, u), lerp(0.205, 0.178, u)))
        tgt = Vector((0.0, lerp(0.012, -0.004, u), -0.002))
        return pos, tgt, lerp(64.0, 68.0, u), 9.0
    if t < 20.4:
        u = smoothstep(15.2, 20.2, t)
        pos = Vector((lerp(-0.075, 0.078, u), lerp(-0.125, -0.118, u), lerp(0.178, 0.185, u)))
        tgt = Vector((0.0, 0.0, -0.002))
        return pos, tgt, lerp(68.0, 66.0, u), 9.0
    if t < 25.3:
        u = smoothstep(20.4, 25.1, t)
        pos = Vector((lerp(0.078, 0.055, u), lerp(-0.118, -0.130, u), lerp(0.185, 0.215, u)))
        tgt = Vector((0.0, 0.0, 0.0))
        return pos, tgt, lerp(66.0, 60.0, u), 8.5
    u_pull = smoothstep(27.0, 29.8, t)
    pos = Vector((
        lerp(0.055, 0.155, u_pull),
        lerp(-0.130, -0.430, u_pull),
        lerp(0.215, 0.415, u_pull),
    ))
    tgt = Vector((
        lerp(0.0, -0.010, u_pull),
        lerp(0.0, -0.060, u_pull),
        lerp(0.0, 0.005, u_pull),
    ))
    return pos, tgt, lerp(60.0, 48.0, u_pull), lerp(8.5, 7.0, u_pull)


# ============================================================================
# 7. CYCLES RENDER & MASTER ANIMATION
# ============================================================================
def configure_cycles(scene, args):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.028
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

    has_gpu = False
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in (args.device, "OPTIX", "CUDA"):
        if backend == "CPU":
            break
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type != "CPU"]
            if gpus:
                for d in prefs.devices:
                    d.use = (d.type != "CPU")
                scene.cycles.device = "GPU"
                has_gpu = True
                print(f"[Cycles GPU] Active backend: {backend} ({len(gpus)} GPU(s))")
                break
        except Exception:
            continue
    if not has_gpu:
        scene.cycles.device = "CPU"
        print("[Cycles CPU] Rendering on local CPU for visual QA")

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
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass
    scene.view_settings.exposure = 0.05


def build_and_render(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    configure_cycles(scene, args)

    _, _, LocationService, _ = enable_mpfb2_services()
    diff_path = find_4k_skin_diffuse_path(LocationService)

    skin_mat, inf_val, seam_val, cutaway_val = make_v2_4k_human_skin_mat(diff_path)
    wall_mat = make_multilayer_wound_wall_mat()
    fat_mat = make_adipose_fat_mat()
    hair_mat = make_leg_hair_mat()
    suture_mat = make_prolene_suture_mat()
    pore_mat = make_puncture_pore_mat()
    steel_mat = make_surgical_steel_mat("SurgicalSteel_V21", gold_ring=False)
    gold_mat = make_surgical_steel_mat("SurgicalGoldTrim_V21", gold_ring=True)

    bio_light = setup_surgical_environment_and_lights(scene)

    leg_obj, sk_closed, sk_healed, fat_lobules, shin_z_at, stitch_ys = build_anatomical_human_leg_with_3d_wound(
        scene, skin_mat, wall_mat, fat_mat, hair_mat
    )
    rungs, diag_laces, pores, knots, pull_tail = build_suture_system(scene, suture_mat, pore_mat, shin_z_at, stitch_ys)
    needle_rig = build_needle_and_holder_rig(scene, steel_mat, gold_mat, suture_mat)
    sciss_root, blade_a, blade_b = build_sleek_iris_scissors(scene, steel_mat, gold_mat)
    tweezers_root = build_adson_tweezers(scene, steel_mat)
    cells, collagen_bundles, col_fac, col_em, vessel_root, trunks_obj, cap_sprouts, ves_em = build_subdermal_healing_systems(
        scene, shin_z_at
    )

    cam_data = bpy.data.cameras.new("MacroSurgicalCam_V21")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.clip_start = 0.008
    cam_data.clip_end = 15.0
    cam_data.dof.use_dof = True
    cam_obj = bpy.data.objects.new("MacroSurgicalCam_V21", cam_data)
    link_obj(scene, cam_obj)
    scene.camera = cam_obj

    key_frames = sorted(set(list(range(1, TOTAL_FRAMES + 1, 2)) + [1, TOTAL_FRAMES]))

    for f in key_frames:
        t = (f - 1) / float(FPS)

        cpos, ctgt, lens, fstop = compute_v2_camera(t)
        cam_obj.location = cpos
        cam_obj.rotation_mode = "QUATERNION"
        cam_obj.rotation_quaternion = (ctgt - cpos).normalized().to_track_quat("-Z", "Y")
        cam_data.lens = lens
        cam_data.dof.focus_distance = (ctgt - cpos).length
        cam_data.dof.aperture_fstop = fstop
        cam_obj.keyframe_insert(data_path="location", frame=f)
        cam_obj.keyframe_insert(data_path="rotation_quaternion", frame=f)
        cam_data.keyframe_insert(data_path="lens", frame=f)
        cam_data.dof.keyframe_insert(data_path="focus_distance", frame=f)
        cam_data.dof.keyframe_insert(data_path="aperture_fstop", frame=f)

        zip_u = smoothstep(6.0, 8.0, t)
        heal_u = smoothstep(26.6, 28.6, t)
        sk_closed.value = zip_u * (1.0 - heal_u)
        sk_healed.value = heal_u
        sk_closed.keyframe_insert(data_path="value", frame=f)
        sk_healed.keyframe_insert(data_path="value", frame=f)

        fat_vis = 1.0 - smoothstep(7.2, 7.9, t)
        fs = max(0.0001, fat_vis)
        for f_obj, fx_o, fy, fz in fat_lobules:
            f_obj.location = Vector((lerp(fx_o, fx_o * 0.08, zip_u), fy, fz))
            f_obj.scale = (fs, fs, fs)
            f_obj.keyframe_insert(data_path="location", frame=f)
            f_obj.keyframe_insert(data_path="scale", frame=f)

        pulse_inf = 0.08 * math.sin(t * 7.6) * smoothstep(10.2, 11.2, t) * (1.0 - smoothstep(15.2, 16.5, t))
        inf_lvl = (0.18 + 0.62 * smoothstep(10.2, 11.8, t) * (1.0 - smoothstep(20.5, 25.0, t)) + pulse_inf) * (1.0 - 0.92 * heal_u)
        inf_val.default_value = clamp01(inf_lvl)
        inf_val.keyframe_insert(data_path="default_value", frame=f)

        seam_val.default_value = 0.88 * (1.0 - heal_u)
        seam_val.keyframe_insert(data_path="default_value", frame=f)

        xray_open = smoothstep(10.25, 11.15, t) * (1.0 - smoothstep(24.20, 25.15, t))
        cutaway_val.default_value = xray_open * 0.82
        cutaway_val.keyframe_insert(data_path="default_value", frame=f)

        bio_light.data.energy = xray_open * (0.16 + 0.04 * math.sin(t * 7.6))
        bio_light.data.keyframe_insert(data_path="energy", frame=f)

        if 0.65 <= t <= 5.95:
            p_prog = clamp01((t - 0.85) / 4.80) * NUM_STITCHES
            idx_n = min(NUM_STITCHES - 1, int(p_prog))
            frac = p_prog - idx_n
            sy = stitch_ys[idx_n]
            ox = wound_half_gap(sy) + 0.0115
            nx = lerp(-ox, +ox, smoothstep(0.06, 0.88, frac))
            nz = shin_z_at(nx, sy) + 0.0075 * math.sin(math.pi * frac) + 0.002
            needle_rig.location = Vector((nx, sy, nz))
            needle_rig.rotation_euler = Euler((0.0, math.radians(lerp(-42.0, 42.0, frac)), 0.0), "XYZ")
            ns = max(0.0001, smoothstep(0.65, 0.95, t) * (1.0 - smoothstep(5.65, 5.92, t)))
            needle_rig.scale = (ns, ns, ns)
        else:
            needle_rig.scale = (0.0001, 0.0001, 0.0001)
        needle_rig.keyframe_insert(data_path="location", frame=f)
        needle_rig.keyframe_insert(data_path="rotation_euler", frame=f)
        needle_rig.keyframe_insert(data_path="scale", frame=f)

        for idx, (r_obj, r_sk, sy, ox, cx) in enumerate(rungs):
            t_in = 1.05 + idx * (4.80 / NUM_STITCHES)
            vis_in = smoothstep(t_in - 0.25, t_in + 0.15, t)
            r_zip = smoothstep(6.0 + idx * 0.22, 7.0 + idx * 0.22, t)
            r_sk.value = r_zip
            r_sk.keyframe_insert(data_path="value", frame=f)

            ext_u = smoothstep(25.55 + idx * 0.22, 26.25 + idx * 0.22, t)
            vis_out = 1.0 - smoothstep(26.15 + idx * 0.22, 26.40 + idx * 0.22, t)
            rs = max(0.0001, vis_in * vis_out)
            r_obj.location = Vector((0.008 * ext_u, 0.0, 0.022 * ext_u))
            r_obj.scale = (rs, rs, rs)
            r_obj.keyframe_insert(data_path="location", frame=f)
            r_obj.keyframe_insert(data_path="scale", frame=f)

        for d_obj, d_sk, idx in diag_laces:
            t_in = 1.55 + idx * (4.80 / NUM_STITCHES)
            d_vis = smoothstep(t_in - 0.15, t_in + 0.20, t) * (1.0 - smoothstep(7.4, 7.9, t))
            d_sk.value = smoothstep(6.0 + idx * 0.22, 7.0 + idx * 0.22, t)
            d_sk.keyframe_insert(data_path="value", frame=f)
            ds = max(0.0001, d_vis)
            d_obj.scale = (ds, ds, ds)
            d_obj.keyframe_insert(data_path="scale", frame=f)

        for p_obj, idx, sgn, sy, ox, cx in pores:
            t_in = (0.95 if sgn < 0 else 1.25) + idx * (4.80 / NUM_STITCHES)
            p_vis = smoothstep(t_in - 0.1, t_in + 0.1, t) * (1.0 - heal_u)
            px_cur = sgn * lerp(ox, cx, zip_u)
            p_obj.location = Vector((px_cur, sy, shin_z_at(px_cur, sy) - 0.0002))
            ps = max(0.0001, p_vis)
            p_obj.scale = (ps, ps, ps)
            p_obj.keyframe_insert(data_path="location", frame=f)
            p_obj.keyframe_insert(data_path="scale", frame=f)

        for kn, idx, sy, cx in knots:
            kn_in = smoothstep(7.35 + idx * 0.12, 7.75 + idx * 0.12, t)
            ext_u = smoothstep(25.55 + idx * 0.22, 26.25 + idx * 0.22, t)
            kn_out = 1.0 - smoothstep(26.15 + idx * 0.22, 26.40 + idx * 0.22, t)
            ks = max(0.0001, kn_in * kn_out)
            kn.location = Vector((cx + 0.008 * ext_u, sy, shin_z_at(cx, sy) + 0.0005 + 0.022 * ext_u))
            kn.scale = (ks, ks, ks)
            kn.keyframe_insert(data_path="location", frame=f)
            kn.keyframe_insert(data_path="scale", frame=f)

        # --- Tension Pull Tail & Sleek Iris Scissors Straddling & Snipping the Thread at (0.0115, top_sy, 0.014)! ---
        top_sy = stitch_ys[-1]
        tail_vis = smoothstep(5.6, 6.0, t) * (1.0 - smoothstep(8.66, 9.15, t))
        snip_fly = smoothstep(8.66, 9.15, t)
        ts = max(0.0001, tail_vis)
        pull_tail.location = Vector((0.0115 + 0.025 * snip_fly, top_sy, shin_z_at(0.0115, top_sy) + 0.003 + 0.006 * zip_u + 0.045 * snip_fly))
        pull_tail.scale = (ts, ts, ts)
        pull_tail.keyframe_insert(data_path="location", frame=f)
        pull_tail.keyframe_insert(data_path="scale", frame=f)

        sc2 = smoothstep(7.95, 8.35, t) * (1.0 - smoothstep(9.10, 9.55, t))
        sc6 = smoothstep(25.15, 25.40, t) * (1.0 - smoothstep(26.35, 26.70, t))
        if sc2 > 0.001:
            sc_s = sc2
            # Blade cutting contact zone is at local X = -0.022m -> placing pivot at X = 0.0335 puts blades RIGHT ON X = 0.0115!
            sciss_root.location = Vector((lerp(0.058, 0.0335, smoothstep(8.0, 8.52, t)), top_sy, 0.014))
            sciss_root.rotation_euler = Euler((math.radians(12.0), math.radians(-8.0), math.radians(-10.0)), "XYZ")
            open_deg = 18.0 * (1.0 - smoothstep(8.48, 8.66, t))
        elif sc6 > 0.001:
            sc_s = sc6
            y_sc6 = lerp(stitch_ys[0], stitch_ys[-1], clamp01((t - 25.35) / 1.10))
            sciss_root.location = Vector((0.031, y_sc6, 0.009))
            sciss_root.rotation_euler = Euler((math.radians(10.0), math.radians(-10.0), math.radians(-8.0)), "XYZ")
            open_deg = 15.0 * abs(math.sin((t - 25.35) * 14.0))
        else:
            sc_s = 0.0001
            open_deg = 0.0
        sciss_root.scale = (sc_s, sc_s, sc_s)
        sciss_root.keyframe_insert(data_path="location", frame=f)
        sciss_root.keyframe_insert(data_path="rotation_euler", frame=f)
        sciss_root.keyframe_insert(data_path="scale", frame=f)
        blade_a.rotation_euler = Euler((0.0, math.radians(+open_deg), 0.0), "XYZ")
        blade_b.rotation_euler = Euler((0.0, math.radians(-open_deg), 0.0), "XYZ")
        blade_a.keyframe_insert(data_path="rotation_euler", frame=f)
        blade_b.keyframe_insert(data_path="rotation_euler", frame=f)

        tw_vis = smoothstep(25.35, 25.60, t) * (1.0 - smoothstep(26.85, 27.20, t))
        if tw_vis > 0.001:
            u_tw = clamp01((t - 25.55) / 1.25)
            y_tw = lerp(stitch_ys[0], stitch_ys[-1], u_tw)
            lift_tw = 0.016 * abs(math.sin(u_tw * math.pi * NUM_STITCHES))
            tweezers_root.location = Vector((-0.006, y_tw, 0.006 + lift_tw))
            tweezers_root.scale = (tw_vis, tw_vis, tw_vis)
        else:
            tweezers_root.scale = (0.0001, 0.0001, 0.0001)
        tweezers_root.keyframe_insert(data_path="location", frame=f)
        tweezers_root.keyframe_insert(data_path="scale", frame=f)

        # --- Beat 3: Sculpted Cells Streaming Inside Dermal Cleft ---
        cell_vis = smoothstep(10.4, 11.2, t) * (1.0 - smoothstep(14.9, 15.6, t))
        if cell_vis > 0.001 or abs(t - 10.2) < 0.25 or abs(t - 15.8) < 0.25 or f == 1:
            for obj, xs, xe, yt, z_sub, stg, ph in cells:
                cu = smoothstep(10.5 + stg * 1.3, 13.6 + stg * 1.0, t)
                cx = lerp(xs, xe, cu) + 0.0009 * math.sin(t * 4.2 + ph)
                cy = yt + 0.0014 * math.cos(t * 3.6 + ph)
                cs = max(0.0001, cell_vis)
                obj.location = Vector((cx, cy, z_sub + 0.0004 * math.sin(t * 5.2 + ph)))
                obj.rotation_euler = Euler((t * 1.4 + ph, t * 1.1, ph), "XYZ")
                obj.scale = (cs, cs, cs)
                obj.keyframe_insert(data_path="location", frame=f)
                obj.keyframe_insert(data_path="rotation_euler", frame=f)
                obj.keyframe_insert(data_path="scale", frame=f)

        # --- Beat 4: 60 Sinuous Curved Golden Collagen Fibers (fades into solid dermis at t=19.8..20.5s so Beat 5 vessels shine solo!) ---
        c_morph = smoothstep(18.2, 20.1, t)
        col_fac.default_value = c_morph
        col_fac.keyframe_insert(data_path="default_value", frame=f)
        col_em.default_value = lerp(1.15, 0.15, c_morph)
        col_em.keyframe_insert(data_path="default_value", frame=f)
        col_out = 1.0 - smoothstep(19.9, 20.6, t)
        for c_obj, stg in collagen_bundles:
            cg = smoothstep(15.3 + stg * 1.8, 16.2 + stg * 1.8, t) * col_out
            cs = max(0.0001, cg)
            c_obj.scale = (cs, cs, cs)
            c_obj.keyframe_insert(data_path="scale", frame=f)

        # --- Beat 5: Tortuous Ruby Arterioles & Capillary Sprouts Growing Solo Across Dermal Cleft! ---
        tr_vis = smoothstep(20.3, 21.0, t) * (1.0 - smoothstep(24.3, 25.1, t))
        trs = max(0.0001, tr_vis)
        trunks_obj.scale = (trs, trs, trs)
        trunks_obj.keyframe_insert(data_path="scale", frame=f)

        sink_z = lerp(0.0, -0.0040, smoothstep(23.6, 25.1, t))
        vessel_root.location = Vector((0.0, 0.0, sink_z))
        vessel_root.keyframe_insert(data_path="location", frame=f)
        ves_em.default_value = (0.55 + 0.45 * (0.5 + 0.5 * math.sin(t * 8.4))) * tr_vis
        ves_em.keyframe_insert(data_path="default_value", frame=f)
        for v_obj, stg in cap_sprouts:
            vg = smoothstep(20.5 + stg * 1.6, 21.5 + stg * 1.6, t) * (1.0 - smoothstep(24.3, 25.1, t))
            vs = max(0.0001, vg)
            v_obj.scale = (vs, vs, vs)
            v_obj.keyframe_insert(data_path="scale", frame=f)

    out_dir = Path(args.frames_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_to_render = [8, 76, 206, 312, 428, 545, 705] if args.preview else list(range(max(1, args.start_frame), min(TOTAL_FRAMES, args.end_frame) + 1))

    for f in frames_to_render:
        scene.frame_set(f)
        scene.render.filepath = str(out_dir / f"frame_{f:04d}.jpg")
        bpy.ops.render.render(write_still=True)

    print(f"[How Stitches Work V2.1] Rendered {len(frames_to_render)} frame(s) into {out_dir}")


if __name__ == "__main__":
    build_and_render(parse_args())
