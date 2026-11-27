"""
Hollywood Golden-Hour Desert Tactical Outpost Battlefield + 4K MakeHuman MPFB2 Soldiers
Episode: How a Grenade Actually Works (Blender 4.2.3 LTS Cycles — V10 Cinema Standard)

Public API:
  - build_battlefield(col, total_frames=720) -> dict
  - build_soldiers(col, total_frames=720) -> dict
  - Key constants: HERO_POS, HERO_YAW_DEG, ENEMY_A_POS, ENEMY_B_POS,
    ENEMY_BUNKER_CENTER, GRENADE_LANDING_POS, EXPLOSION_ORIGIN,
    RELEASE_FRAME, LANDING_FRAME, FRACTURE_FRAME
"""
from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

# Ensure root `src/mpfb_characters.py` is importable regardless of CWD
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
_ROOT_SRC = str(_REPO_ROOT / "src")
if _ROOT_SRC not in sys.path:
    sys.path.insert(0, _ROOT_SRC)

import mpfb_characters as mc  # noqa: E402

# ============================================================================
# CANONICAL SCENE COORDINATES & BEAT FRAME CONSTANTS
# ============================================================================
TOTAL_FRAMES_DEFAULT = 720
FPS = 24

HERO_POS = (0.0, 0.0, 0.0)
HERO_YAW_DEG = 162.0  # Faces +Y angled +18 deg toward +X (heroic 3/4 close-up from front-right!)
HERO_HEIGHT_M = 1.80

ENEMY_A_POS = (-0.78, 8.55, 0.0)
ENEMY_A_YAW_DEG = 10.0  # Faces -Y slightly angled inward toward center
ENEMY_B_POS = (0.82, 8.65, 0.0)
ENEMY_B_YAW_DEG = -12.0  # Faces -Y slightly angled inward toward center

ENEMY_BUNKER_CENTER = (0.0, 7.80, 0.0)
GRENADE_LANDING_POS = (0.0, 7.25, 0.04)
EXPLOSION_ORIGIN = (0.0, 7.45, 0.0)

PIN_PULL_START_FRAME = 24
PIN_PULL_END_FRAME = 54
RELEASE_FRAME = 134
LANDING_FRAME = 496
FRACTURE_FRAME = 612


# ============================================================================
# SHARED SCENE, COLLECTION, ANIMATION & MATERIAL UTILITIES
# ============================================================================
def _link_obj(obj: bpy.types.Object, col: bpy.types.Collection | None = None) -> bpy.types.Object:
    target = col if col is not None else bpy.context.scene.collection
    if obj.name not in target.objects:
        target.objects.link(obj)
    # Remove from scene root if linked into a dedicated sub-collection
    if col is not None and col != bpy.context.scene.collection:
        if obj.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(obj)
    return obj


def _move_hierarchy_to_col(objs: list[bpy.types.Object], col: bpy.types.Collection | None) -> None:
    if col is None:
        return
    for obj in objs:
        if obj is None:
            continue
        for c in list(obj.users_collection):
            if c != col:
                c.objects.unlink(obj)
        if obj.name not in col.objects:
            col.objects.link(obj)


def _set_smooth(mesh: bpy.types.Mesh) -> None:
    if len(mesh.polygons) > 0:
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mesh.update()


def _set_bezier(obj_or_data) -> None:
    ad = getattr(obj_or_data, "animation_data", None)
    if not ad or not ad.action:
        return
    for fc in ad.action.fcurves:
        if "hide_" in fc.data_path:
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = "BEZIER"
            kp.handle_left_type = "AUTO_CLAMPED"
            kp.handle_right_type = "AUTO_CLAMPED"


def _key_visibility(obj: bpy.types.Object, visible_ranges: list[tuple[int, int]], total_frames: int = 720) -> None:
    """Keyframe hide_viewport & hide_render with CONSTANT interpolation."""
    frames = {1, total_frames}
    for s, e in visible_ranges:
        if s > 1:
            frames.add(s - 1)
        frames.add(s)
        frames.add(e)
        if e < total_frames:
            frames.add(e + 1)

    for f in sorted(frames):
        vis = any(s <= f <= e for s, e in visible_ranges)
        obj.hide_viewport = not vis
        obj.hide_render = not vis
        obj.keyframe_insert("hide_viewport", frame=f)
        obj.keyframe_insert("hide_render", frame=f)

    ad = obj.animation_data
    if ad and ad.action:
        for fc in ad.action.fcurves:
            if fc.data_path.startswith("hide_"):
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"


def _aim_euler(src, dst) -> Euler:
    d = Vector(dst) - Vector(src)
    if d.length < 1e-6:
        return Euler((0.0, 0.0, 0.0), "XYZ")
    return d.normalized().to_track_quat("-Z", "Y").to_euler("XYZ")


def _rgba_mix_sockets(mix_node):
    rgba_in = [s for s in mix_node.inputs if s.type == "RGBA"]
    rgba_out = [s for s in mix_node.outputs if s.type == "RGBA"]
    return mix_node.inputs[0], rgba_in[0], rgba_in[1], rgba_out[0]


def _set_bsdf(bsdf, name: str, val) -> None:
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = val


def _parent_to_bone_preserve_world(obj: bpy.types.Object, rig: bpy.types.Object, bone_name: str) -> None:
    """Parent `obj` cleanly to `rig`'s pose bone `bone_name` while preserving `obj`'s exact current world transform."""
    bpy.context.view_layer.update()
    pb = rig.pose.bones[bone_name]
    bone_world = rig.matrix_world @ pb.matrix @ Matrix.Translation((0.0, pb.length, 0.0))
    orig_world = obj.matrix_world.copy()
    obj.parent = rig
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    obj.matrix_parent_inverse = bone_world.inverted()
    obj.matrix_world = orig_world
    bpy.context.view_layer.update()


# ============================================================================
# PROCEDURAL DESERT BATTLEFIELD & TACTICAL PBR MATERIALS
# ============================================================================
def _make_desert_ground_mat() -> bpy.types.Material:
    """Photoreal arid desert dust, gravel & subtle warped dry-clay ground shader (zero cartoon Voronoi!)."""
    mat = bpy.data.materials.new("Mat_DesertGround")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Roughness", 0.92)
    _set_bsdf(bsdf, "Specular IOR Level", 0.20)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = nt.nodes.new("ShaderNodeTexCoord")

    # Macro sun-baked desert sand/dirt patches
    n_macro = nt.nodes.new("ShaderNodeTexNoise")
    n_macro.inputs["Scale"].default_value = 0.24
    n_macro.inputs["Detail"].default_value = 10.0
    n_macro.inputs["Roughness"].default_value = 0.65
    nt.links.new(tc.outputs["Object"], n_macro.inputs["Vector"])

    # Medium desert pebbles & scattered gravel patches
    n_med = nt.nodes.new("ShaderNodeTexNoise")
    n_med.inputs["Scale"].default_value = 2.6
    n_med.inputs["Detail"].default_value = 10.0
    n_med.inputs["Roughness"].default_value = 0.72
    nt.links.new(tc.outputs["Object"], n_med.inputs["Vector"])

    # Fine gritty sand & dust grains
    n_fine = nt.nodes.new("ShaderNodeTexNoise")
    n_fine.inputs["Scale"].default_value = 34.0
    n_fine.inputs["Detail"].default_value = 12.0
    n_fine.inputs["Roughness"].default_value = 0.78
    nt.links.new(tc.outputs["Object"], n_fine.inputs["Vector"])

    # Organically warped subtle micro-cracks (heavily distorted so NEVER uniform geometric tiles!)
    mix_warp = nt.nodes.new("ShaderNodeMix")
    mix_warp.data_type = "VECTOR"
    mix_warp.inputs[0].default_value = 0.28
    nt.links.new(tc.outputs["Object"], mix_warp.inputs["A"])
    nt.links.new(n_med.outputs["Color"], mix_warp.inputs["B"])

    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.feature = "DISTANCE_TO_EDGE"
    vor.inputs["Scale"].default_value = 5.5
    nt.links.new(mix_warp.outputs["Result"], vor.inputs["Vector"])

    cr_crack = nt.nodes.new("ShaderNodeValToRGB")
    cr_crack.color_ramp.elements[0].position = 0.0
    cr_crack.color_ramp.elements[0].color = (0.68, 0.68, 0.68, 1.0)
    cr_crack.color_ramp.elements[1].position = 0.032
    cr_crack.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(vor.outputs["Distance"], cr_crack.inputs["Fac"])

    # Subtle wind-rippled desert sand dunes
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "Y"
    wave.inputs["Scale"].default_value = 1.4
    wave.inputs["Distortion"].default_value = 8.5
    wave.inputs["Detail"].default_value = 5.0
    nt.links.new(tc.outputs["Object"], wave.inputs["Vector"])

    # Blend macro + medium variation for natural earthy ochre / sienna / dry gravel palette
    m_blend = nt.nodes.new("ShaderNodeMath")
    m_blend.operation = "MULTIPLY_ADD"
    m_blend.inputs[1].default_value = 0.65
    nt.links.new(n_macro.outputs["Fac"], m_blend.inputs[0])

    m_blend2 = nt.nodes.new("ShaderNodeMath")
    m_blend2.operation = "MULTIPLY_ADD"
    m_blend2.inputs[1].default_value = 0.35
    nt.links.new(n_med.outputs["Fac"], m_blend2.inputs[0])
    nt.links.new(m_blend.outputs["Value"], m_blend2.inputs[2])

    cr_col = nt.nodes.new("ShaderNodeValToRGB")
    cr_col.color_ramp.elements[0].position = 0.28
    cr_col.color_ramp.elements[0].color = (0.16, 0.11, 0.065, 1.0)
    e_mid = cr_col.color_ramp.elements.new(0.56)
    e_mid.color = (0.28, 0.20, 0.12, 1.0)
    cr_col.color_ramp.elements[2].position = 0.84
    cr_col.color_ramp.elements[2].color = (0.38, 0.28, 0.17, 1.0)
    nt.links.new(m_blend2.outputs["Value"], cr_col.inputs["Fac"])

    # Subtle crack darkening (multiplied softly so it never looks like cartoon paving!)
    mix_crack = nt.nodes.new("ShaderNodeMix")
    mix_crack.data_type = "RGBA"
    mix_crack.blend_type = "MULTIPLY"
    fac_c, a_c, b_c, out_c = _rgba_mix_sockets(mix_crack)
    fac_c.default_value = 0.35
    nt.links.new(cr_col.outputs["Color"], a_c)
    nt.links.new(cr_crack.outputs["Color"], b_c)

    # Ambient Occlusion contact darkening around sandbags, boots & crates
    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.inputs["Distance"].default_value = 0.65
    nt.links.new(out_c, ao.inputs["Color"])

    mix_ao = nt.nodes.new("ShaderNodeMix")
    mix_ao.data_type = "RGBA"
    mix_ao.blend_type = "MULTIPLY"
    fac_a, a_a, b_a, out_a = _rgba_mix_sockets(mix_ao)
    fac_a.default_value = 0.62
    nt.links.new(out_c, a_a)
    nt.links.new(ao.outputs["Color"], b_a)
    nt.links.new(out_a, bsdf.inputs["Base Color"])

    # Roughness variation (0.86 .. 0.98)
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value = 0.86
    mr.inputs["To Max"].default_value = 0.98
    nt.links.new(n_fine.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])

    # Layered natural desert bump: sand ripples + gravel lumps + crisp grit (subtle cracks)
    b_ripple = nt.nodes.new("ShaderNodeBump")
    b_ripple.inputs["Strength"].default_value = 0.18
    b_ripple.inputs["Distance"].default_value = 0.030
    nt.links.new(wave.outputs["Fac"], b_ripple.inputs["Height"])

    b_med = nt.nodes.new("ShaderNodeBump")
    b_med.inputs["Strength"].default_value = 0.32
    b_med.inputs["Distance"].default_value = 0.020
    nt.links.new(n_med.outputs["Fac"], b_med.inputs["Height"])
    nt.links.new(b_ripple.outputs["Normal"], b_med.inputs["Normal"])

    b_grit = nt.nodes.new("ShaderNodeBump")
    b_grit.inputs["Strength"].default_value = 0.36
    b_grit.inputs["Distance"].default_value = 0.008
    nt.links.new(n_fine.outputs["Fac"], b_grit.inputs["Height"])
    nt.links.new(b_med.outputs["Normal"], b_grit.inputs["Normal"])
    nt.links.new(b_grit.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def _make_burlap_sandbag_mat(name: str = "Mat_BurlapSandbag", variant: float = 0.0) -> bpy.types.Material:
    """Heavy dusty military woven jute/burlap sandbag shader with crisp canvas weave & deep stack AO."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Roughness", 0.94)
    _set_bsdf(bsdf, "Sheen Weight", 0.35)
    _set_bsdf(bsdf, "Sheen Roughness", 0.72)
    _set_bsdf(bsdf, "Specular IOR Level", 0.14)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = nt.nodes.new("ShaderNodeTexCoord")

    # Authentic crisscross coarse burlap weave
    w_u = nt.nodes.new("ShaderNodeTexWave")
    w_u.bands_direction = "X"
    w_u.inputs["Scale"].default_value = 64.0
    w_u.inputs["Distortion"].default_value = 1.4
    nt.links.new(tc.outputs["Object"], w_u.inputs["Vector"])

    w_v = nt.nodes.new("ShaderNodeTexWave")
    w_v.bands_direction = "Y"
    w_v.inputs["Scale"].default_value = 64.0
    w_v.inputs["Distortion"].default_value = 1.4
    nt.links.new(tc.outputs["Object"], w_v.inputs["Vector"])

    weave_add = nt.nodes.new("ShaderNodeMath")
    weave_add.operation = "MULTIPLY"
    nt.links.new(w_u.outputs["Fac"], weave_add.inputs[0])
    nt.links.new(w_v.outputs["Fac"], weave_add.inputs[1])

    # Grime, sweat & desert dirt blotches
    n_dust = nt.nodes.new("ShaderNodeTexNoise")
    n_dust.inputs["Scale"].default_value = 4.6
    n_dust.inputs["Detail"].default_value = 8.0
    n_dust.inputs["Roughness"].default_value = 0.68
    nt.links.new(tc.outputs["Object"], n_dust.inputs["Vector"])

    # Rich earthy military jute brown/olive-tan (NEVER pale white pillows!)
    base_dark = (0.14 + 0.025 * variant, 0.11 + 0.020 * variant, 0.065 + 0.012 * variant, 1.0)
    base_light = (0.28 + 0.035 * variant, 0.22 + 0.025 * variant, 0.135 + 0.018 * variant, 1.0)

    mix_col = nt.nodes.new("ShaderNodeMix")
    mix_col.data_type = "RGBA"
    fac_m, a_m, b_m, out_m = _rgba_mix_sockets(mix_col)
    a_m.default_value = base_dark
    b_m.default_value = base_light
    nt.links.new(n_dust.outputs["Fac"], fac_m)

    # Subtle woven thread brightness modulation
    mix_w = nt.nodes.new("ShaderNodeMix")
    mix_w.data_type = "RGBA"
    mix_w.blend_type = "MULTIPLY"
    fac_w, a_w, b_w, out_w = _rgba_mix_sockets(mix_w)
    fac_w.default_value = 0.22
    nt.links.new(out_m, a_w)
    nt.links.new(weave_add.outputs["Value"], b_w)

    # Deep AO crevice darkening where stacked sandbags press tightly together
    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.inputs["Distance"].default_value = 0.30
    nt.links.new(out_w, ao.inputs["Color"])

    mix_ao = nt.nodes.new("ShaderNodeMix")
    mix_ao.data_type = "RGBA"
    mix_ao.blend_type = "MULTIPLY"
    fac_a, a_a, b_a, out_a = _rgba_mix_sockets(mix_ao)
    fac_a.default_value = 0.72
    nt.links.new(out_w, a_a)
    nt.links.new(ao.outputs["Color"], b_a)
    nt.links.new(out_a, bsdf.inputs["Base Color"])

    # Crisp burlap weave + cloth crease bump
    b_weave = nt.nodes.new("ShaderNodeBump")
    b_weave.inputs["Strength"].default_value = 0.48
    b_weave.inputs["Distance"].default_value = 0.0045
    nt.links.new(weave_add.outputs["Value"], b_weave.inputs["Height"])

    b_wrinkle = nt.nodes.new("ShaderNodeBump")
    b_wrinkle.inputs["Strength"].default_value = 0.26
    b_wrinkle.inputs["Distance"].default_value = 0.015
    nt.links.new(n_dust.outputs["Fac"], b_wrinkle.inputs["Height"])
    nt.links.new(b_weave.outputs["Normal"], b_wrinkle.inputs["Normal"])
    nt.links.new(b_wrinkle.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def _make_rock_mat(name: str = "Mat_DesertRock", stratified: bool = False) -> bpy.types.Material:
    """Weathered Utah/Afghanistan sandstone & canyon mesa rock shader (natural crags, zero zebra stripes!)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Roughness", 0.92)
    _set_bsdf(bsdf, "Specular IOR Level", 0.20)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = nt.nodes.new("ShaderNodeTexCoord")

    # Organic craggy sandstone noise (stretched horizontally on mesas for natural sedimentary erosion!)
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (0.35, 0.35, 1.45) if stratified else (1.0, 1.0, 1.0)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])

    n1 = nt.nodes.new("ShaderNodeTexNoise")
    n1.inputs["Scale"].default_value = 0.65 if stratified else 3.8
    n1.inputs["Detail"].default_value = 12.0
    n1.inputs["Roughness"].default_value = 0.72
    n1.inputs["Distortion"].default_value = 0.65 if stratified else 0.35
    nt.links.new(mp.outputs["Vector"], n1.inputs["Vector"])

    n_detail = nt.nodes.new("ShaderNodeTexNoise")
    n_detail.inputs["Scale"].default_value = 2.8 if stratified else 11.0
    n_detail.inputs["Detail"].default_value = 8.0
    n_detail.inputs["Roughness"].default_value = 0.68
    nt.links.new(tc.outputs["Object"], n_detail.inputs["Vector"])

    cr = nt.nodes.new("ShaderNodeValToRGB")
    if stratified:
        cr.color_ramp.elements[0].position = 0.26
        cr.color_ramp.elements[0].color = (0.14, 0.085, 0.048, 1.0)
        e1 = cr.color_ramp.elements.new(0.55)
        e1.color = (0.26, 0.165, 0.095, 1.0)
        cr.color_ramp.elements[2].position = 0.84
        cr.color_ramp.elements[2].color = (0.36, 0.24, 0.145, 1.0)
    else:
        cr.color_ramp.elements[0].position = 0.24
        cr.color_ramp.elements[0].color = (0.15, 0.10, 0.062, 1.0)
        cr.color_ramp.elements[1].position = 0.80
        cr.color_ramp.elements[1].color = (0.32, 0.23, 0.145, 1.0)
    nt.links.new(n1.outputs["Fac"], cr.inputs["Fac"])
    nt.links.new(cr.outputs["Color"], bsdf.inputs["Base Color"])

    bump1 = nt.nodes.new("ShaderNodeBump")
    bump1.inputs["Strength"].default_value = 0.55
    bump1.inputs["Distance"].default_value = 0.08 if stratified else 0.025
    nt.links.new(n1.outputs["Fac"], bump1.inputs["Height"])

    bump2 = nt.nodes.new("ShaderNodeBump")
    bump2.inputs["Strength"].default_value = 0.32
    bump2.inputs["Distance"].default_value = 0.02
    nt.links.new(n_detail.outputs["Fac"], bump2.inputs["Height"])
    nt.links.new(bump1.outputs["Normal"], bump2.inputs["Normal"])
    nt.links.new(bump2.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def _make_wood_crate_mat(name: str = "Mat_OliveWoodCrate", olive: bool = True) -> bpy.types.Material:
    """Weathered olive-drab military wood ammo crate or raw desert timber material."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Roughness", 0.78)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.5, 9.0, 1.5)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])

    n_grain = nt.nodes.new("ShaderNodeTexNoise")
    n_grain.inputs["Scale"].default_value = 8.0
    n_grain.inputs["Detail"].default_value = 7.0
    n_grain.inputs["Distortion"].default_value = 1.4
    nt.links.new(mp.outputs["Vector"], n_grain.inputs["Vector"])

    cr = nt.nodes.new("ShaderNodeValToRGB")
    if olive:
        cr.color_ramp.elements[0].position = 0.25
        cr.color_ramp.elements[0].color = (0.12, 0.14, 0.08, 1.0)
        cr.color_ramp.elements[1].position = 0.78
        cr.color_ramp.elements[1].color = (0.24, 0.26, 0.15, 1.0)
    else:
        cr.color_ramp.elements[0].position = 0.22
        cr.color_ramp.elements[0].color = (0.16, 0.11, 0.06, 1.0)
        cr.color_ramp.elements[1].position = 0.80
        cr.color_ramp.elements[1].color = (0.34, 0.24, 0.14, 1.0)
    nt.links.new(n_grain.outputs["Fac"], cr.inputs["Fac"])
    nt.links.new(cr.outputs["Color"], bsdf.inputs["Base Color"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.24
    bump.inputs["Distance"].default_value = 0.005
    nt.links.new(n_grain.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def _make_camo_tarp_mat() -> bpy.types.Material:
    """Sun-faded desert camo canvas tarp canopy shader."""
    mat = bpy.data.materials.new("Mat_DesertCamoTarp")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Roughness", 0.88)
    _set_bsdf(bsdf, "Sheen Weight", 0.35)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = nt.nodes.new("ShaderNodeTexCoord")
    n_camo = nt.nodes.new("ShaderNodeTexNoise")
    n_camo.inputs["Scale"].default_value = 2.6
    n_camo.inputs["Detail"].default_value = 3.0
    n_camo.inputs["Roughness"].default_value = 0.45
    nt.links.new(tc.outputs["Object"], n_camo.inputs["Vector"])

    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.interpolation = "CONSTANT"
    cr.color_ramp.elements[0].position = 0.0
    cr.color_ramp.elements[0].color = (0.28, 0.22, 0.15, 1.0)
    e1 = cr.color_ramp.elements.new(0.42)
    e1.color = (0.48, 0.39, 0.26, 1.0)
    e2 = cr.color_ramp.elements.new(0.68)
    e2.color = (0.34, 0.32, 0.20, 1.0)
    nt.links.new(n_camo.outputs["Fac"], cr.inputs["Fac"])
    nt.links.new(cr.outputs["Color"], bsdf.inputs["Base Color"])

    n_weave = nt.nodes.new("ShaderNodeTexNoise")
    n_weave.inputs["Scale"].default_value = 95.0
    nt.links.new(tc.outputs["Object"], n_weave.inputs["Vector"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.14
    bump.inputs["Distance"].default_value = 0.004
    nt.links.new(n_weave.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def _make_tactical_metal_mat(
    name: str,
    base_rgb: tuple[float, float, float] = (0.09, 0.10, 0.11),
    metallic: float = 0.78,
    roughness: float = 0.42,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Base Color", (*base_rgb, 1.0))
    _set_bsdf(bsdf, "Metallic", metallic)
    _set_bsdf(bsdf, "Roughness", roughness)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = nt.nodes.new("ShaderNodeTexCoord")
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = 24.0
    n.inputs["Detail"].default_value = 6.0
    nt.links.new(tc.outputs["Object"], n.inputs["Vector"])

    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value = max(0.15, roughness - 0.10)
    mr.inputs["To Max"].default_value = min(0.88, roughness + 0.12)
    nt.links.new(n.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    bump.inputs["Distance"].default_value = 0.002
    nt.links.new(n.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def _make_tactical_gear_mat(
    name: str,
    base_rgb: tuple[float, float, float],
    roughness: float = 0.62,
    is_cordura: bool = False,
    metallic: float = 0.05,
) -> bpy.types.Material:
    """Matte ballistic composite (helmet) or Cordura nylon fabric (plate carrier vest) with micro-texture."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Base Color", (*base_rgb, 1.0))
    _set_bsdf(bsdf, "Metallic", metallic)
    _set_bsdf(bsdf, "Roughness", roughness)
    if is_cordura:
        _set_bsdf(bsdf, "Sheen Weight", 0.35)
        _set_bsdf(bsdf, "Sheen Roughness", 0.55)
    else:
        _set_bsdf(bsdf, "Coat Weight", 0.14)
        _set_bsdf(bsdf, "Coat Roughness", 0.38)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = nt.nodes.new("ShaderNodeTexCoord")
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = 180.0 if is_cordura else 95.0
    n.inputs["Detail"].default_value = 7.0
    nt.links.new(tc.outputs["Object"], n.inputs["Vector"])

    # Slight desert dust variation
    n_dust = nt.nodes.new("ShaderNodeTexNoise")
    n_dust.inputs["Scale"].default_value = 7.5
    n_dust.inputs["Detail"].default_value = 5.0
    nt.links.new(tc.outputs["Object"], n_dust.inputs["Vector"])

    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    fac, ca, cb, cout = _rgba_mix_sockets(mix)
    ca.default_value = (*base_rgb, 1.0)
    cb.default_value = (
        min(1.0, base_rgb[0] * 1.45 + 0.07),
        min(1.0, base_rgb[1] * 1.38 + 0.06),
        min(1.0, base_rgb[2] * 1.25 + 0.04),
        1.0,
    )
    mr_d = nt.nodes.new("ShaderNodeMapRange")
    mr_d.inputs["From Min"].default_value = 0.48
    mr_d.inputs["From Max"].default_value = 0.82
    mr_d.inputs["To Min"].default_value = 0.0
    mr_d.inputs["To Max"].default_value = 0.32
    nt.links.new(n_dust.outputs["Fac"], mr_d.inputs["Value"])
    nt.links.new(mr_d.outputs["Result"], fac)
    nt.links.new(cout, bsdf.inputs["Base Color"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.16 if is_cordura else 0.09
    bump.inputs["Distance"].default_value = 0.0025
    nt.links.new(n.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def _make_goggle_lens_mat() -> bpy.types.Material:
    """Dark tactical smoke-amber ballistic polycarbonate goggle visor (clean specular reflections)."""
    mat = bpy.data.materials.new("Mat_TacticalGoggleLens")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Base Color", (0.04, 0.035, 0.03, 1.0))
    _set_bsdf(bsdf, "Metallic", 0.35)
    _set_bsdf(bsdf, "Roughness", 0.06)
    _set_bsdf(bsdf, "Coat Weight", 0.90)
    _set_bsdf(bsdf, "Coat Roughness", 0.03)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _make_explosion_dust_mat(name: str = "Mat_ExplosionDust", warm_glow: float = 0.0) -> bpy.types.Material:
    """Billowing warm golden-hour sunlit desert dust cloud shader (soft Fresnel feathered edges)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Base Color", (0.58, 0.44, 0.28, 1.0))
    _set_bsdf(bsdf, "Roughness", 0.96)
    _set_bsdf(bsdf, "Subsurface Weight", 0.35)
    _set_bsdf(bsdf, "Subsurface Scale", 0.22)
    if warm_glow > 0.0:
        _set_bsdf(bsdf, "Emission Color", (1.0, 0.46, 0.14, 1.0))
        _set_bsdf(bsdf, "Emission Strength", warm_glow)

    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.42

    tc = nt.nodes.new("ShaderNodeTexCoord")
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = 3.2
    n.inputs["Detail"].default_value = 6.0
    nt.links.new(tc.outputs["Object"], n.inputs["Vector"])

    # Soft feathered alpha (opaque core, soft dusty perimeter)
    inv_f = nt.nodes.new("ShaderNodeMath")
    inv_f.operation = "SUBTRACT"
    inv_f.inputs[0].default_value = 1.0
    inv_f.use_clamp = True
    nt.links.new(lw.outputs["Facing"], inv_f.inputs[1])

    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.use_clamp = True
    nt.links.new(inv_f.outputs["Value"], mul.inputs[0])
    nt.links.new(n.outputs["Fac"], mul.inputs[1])

    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 0.05
    mr.inputs["From Max"].default_value = 0.55
    mr.inputs["To Min"].default_value = 0.0
    mr.inputs["To Max"].default_value = 0.82
    nt.links.new(mul.outputs["Value"], mr.inputs["Value"])

    mix_sh = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(mr.outputs["Result"], mix_sh.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix_sh.inputs[1])
    nt.links.new(bsdf.outputs["BSDF"], mix_sh.inputs[2])
    nt.links.new(mix_sh.outputs["Shader"], out.inputs["Surface"])
    return mat


def _make_fireball_core_mat() -> bpy.types.Material:
    """AgX-safe incandescent Hollywood fireball core (additive Camera-ray emission, zero opaque pastel wall)."""
    mat = bpy.data.materials.new("Mat_FireballCore")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    add_sh = nt.nodes.new("ShaderNodeAddShader")

    tc = nt.nodes.new("ShaderNodeTexCoord")
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = 3.8
    n.inputs["Detail"].default_value = 6.0
    nt.links.new(tc.outputs["Object"], n.inputs["Vector"])

    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].position = 0.20
    cr.color_ramp.elements[0].color = (0.85, 0.18, 0.02, 1.0)
    e1 = cr.color_ramp.elements.new(0.55)
    e1.color = (1.0, 0.48, 0.08, 1.0)
    cr.color_ramp.elements[2].position = 0.85
    cr.color_ramp.elements[2].color = (1.0, 0.82, 0.35, 1.0)
    nt.links.new(n.outputs["Fac"], cr.inputs["Fac"])
    nt.links.new(cr.outputs["Color"], em.inputs["Color"])

    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.45
    inv = nt.nodes.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    inv.use_clamp = True
    nt.links.new(lw.outputs["Facing"], inv.inputs[1])

    lp = nt.nodes.new("ShaderNodeLightPath")
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    front = nt.nodes.new("ShaderNodeMath")
    front.operation = "SUBTRACT"
    front.inputs[0].default_value = 1.0
    front.use_clamp = True
    nt.links.new(geom.outputs["Backfacing"], front.inputs[1])

    gate = nt.nodes.new("ShaderNodeMath")
    gate.operation = "MULTIPLY"
    nt.links.new(lp.outputs["Is Camera Ray"], gate.inputs[0])
    nt.links.new(front.outputs["Value"], gate.inputs[1])

    str_m = nt.nodes.new("ShaderNodeMath")
    str_m.operation = "MULTIPLY"
    nt.links.new(inv.outputs["Value"], str_m.inputs[0])
    nt.links.new(gate.outputs["Value"], str_m.inputs[1])

    scale_m = nt.nodes.new("ShaderNodeMath")
    scale_m.operation = "MULTIPLY"
    scale_m.inputs[1].default_value = 3.2  # AgX-safe fiery glow without white blowout!
    nt.links.new(str_m.outputs["Value"], scale_m.inputs[0])
    nt.links.new(scale_m.outputs["Value"], em.inputs["Strength"])

    nt.links.new(tr.outputs["BSDF"], add_sh.inputs[0])
    nt.links.new(em.outputs["Emission"], add_sh.inputs[1])
    nt.links.new(add_sh.outputs["Shader"], out.inputs["Surface"])
    return mat


# ============================================================================
# GEOMETRY BUILDERS: TERRAIN, SANDBAGS, CRATES, CANOPY, MESAS, EXPLOSION FX
# ============================================================================
def _ground_height(x: float, y: float) -> float:
    """Undulating arid desert terrain with level standing pads at Hero (0,0) & Enemy Bunker (0,7.8..9.0)."""
    # Smooth flattening weights around hero pad & enemy bunker pad
    d_hero = math.hypot(x - 0.0, y - 0.25)
    d_enemy = math.hypot(x - 0.0, y - 8.10)
    w_hero = min(1.0, max(0.0, (d_hero - 1.55) / 2.2))
    w_enemy = min(1.0, max(0.0, (d_enemy - 2.35) / 2.5))
    # Smooth corridor between hero & bunker for clean grenade bounce at (0, 7.25)
    w_corridor = min(1.0, max(0.0, (abs(x) - 1.1) / 2.4)) if (-0.5 <= y <= 9.2) else 1.0
    flat_w = w_hero * w_enemy * (0.25 + 0.75 * w_corridor)

    h = (
        0.26 * math.sin(x * 0.24 + 0.4) * math.cos(y * 0.19 - 0.3)
        + 0.14 * math.sin(x * 0.62 - y * 0.41)
        + 0.06 * math.cos(x * 1.35 + y * 1.15)
        + 0.03 * math.sin(x * 2.8 - y * 2.4)
    )
    # Gentle flanking berms at outer edges for dramatic bowl composition
    edge_rise = max(0.0, (abs(x) - 8.5) * 0.14) + max(0.0, (y - 14.5) * 0.12)
    return h * flat_w + edge_rise * min(1.0, w_enemy)


def _build_desert_terrain(mat_ground, mat_rock, col) -> dict:
    """Build undulating rocky desert ground (X in [-18, 18], Y in [-6, 26]) + scattered boulders & gravel."""
    mesh = bpy.data.meshes.new("Desert_Terrain_Mesh")
    bm = bmesh.new()

    nx, ny = 84, 76
    x0, x1 = -18.0, 18.0
    y0, y1 = -6.0, 26.0
    grid = []
    for j in range(ny + 1):
        row = []
        v_y = y0 + (y1 - y0) * (j / ny)
        for i in range(nx + 1):
            v_x = x0 + (x1 - x0) * (i / nx)
            v_z = _ground_height(v_x, v_y)
            row.append(bm.verts.new((v_x, v_y, v_z)))
        grid.append(row)

    for j in range(ny):
        for i in range(nx):
            bm.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))

    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    _set_smooth(mesh)

    terrain = bpy.data.objects.new("Desert_Terrain", mesh)
    terrain.data.materials.append(mat_ground)
    _link_obj(terrain, col)

    # Scattered desert boulders & gravel clusters (strictly outside hero/bunker standing zones!)
    rng = random.Random(4207)
    rocks = []
    rock_specs = [
        (-2.6, 1.4, 0.34), (2.4, 1.8, 0.28), (-3.8, 4.2, 0.52), (3.4, 3.6, 0.44),
        (-2.2, 5.4, 0.26), (2.8, 5.9, 0.36), (-4.5, 8.2, 0.68), (4.2, 8.8, 0.62),
        (-6.2, 12.0, 0.95), (5.8, 11.4, 0.85), (-2.8, 13.5, 0.55), (3.1, 14.2, 0.58),
        (-1.45, 2.8, 0.16), (1.65, 3.2, 0.14), (-1.9, 6.4, 0.18), (1.75, 6.5, 0.15),
        (-3.1, -1.2, 0.38), (2.9, -1.5, 0.32),
    ]
    for idx, (rx, ry, rad) in enumerate(rock_specs):
        rm = bpy.data.meshes.new(f"Desert_Boulder_{idx:02d}_Mesh")
        rbm = bmesh.new()
        bmesh.ops.create_icosphere(rbm, subdivisions=2, radius=rad)
        sx = rng.uniform(0.85, 1.45)
        sy = rng.uniform(0.80, 1.35)
        sz = rng.uniform(0.42, 0.72)
        for v in rbm.verts:
            p = v.co
            nval = (
                0.22 * math.sin(p.x * 5.3 + idx)
                + 0.18 * math.cos(p.y * 6.1 - idx * 1.7)
                + 0.14 * math.sin(p.z * 7.4 + idx * 0.8)
            )
            v.co = Vector((p.x * sx * (1.0 + nval), p.y * sy * (1.0 - nval * 0.6), p.z * sz * (1.0 + nval * 0.5)))
        rbm.normal_update()
        rbm.to_mesh(rm)
        rbm.free()
        _set_smooth(rm)
        ro = bpy.data.objects.new(f"Desert_Boulder_{idx:02d}", rm)
        ro.location = Vector((rx, ry, _ground_height(rx, ry) + rad * sz * 0.38))
        ro.rotation_euler = Euler((rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), rng.uniform(0, 6.28)), "XYZ")
        ro.data.materials.append(mat_rock)
        _link_obj(ro, col)
        rocks.append(ro)

    return {"terrain": terrain, "rocks": rocks}


def _create_sculpted_sandbag(
    name: str,
    loc: tuple[float, float, float],
    yaw_deg: float,
    mat: bpy.types.Material,
    col: bpy.types.Collection,
    length: float = 0.54,
    width: float = 0.28,
    height: float = 0.155,
    seed: int = 0,
    tilt_deg: float = 0.0,
) -> bpy.types.Object:
    """Create a plump, gravity-sagged, stitched-end burlap sandbag (zero boxy bricks!)."""
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=20, v_segments=12, radius=1.0)

    hl, hw, hh = length * 0.5, width * 0.5, height * 0.5
    for v in bm.verts:
        nx, ny, nz = v.co.x, v.co.y, v.co.z
        # Flatten bottom cleanly so stacked bags settle with realistic weight
        z_shaped = nz * (0.68 if nz < 0.0 else 1.05)
        # Plump belly in middle, pinched/tied stitched ears at +/-X ends
        end_pinch = 1.0 - 0.52 * (nx ** 4)
        ear_flare = 1.0 + 0.14 * (abs(nx) ** 3) * (abs(ny) ** 1.5)
        # Subtle organic cloth lumpiness
        lump = 0.045 * math.sin(nx * 5.2 + seed * 1.9) * math.cos(ny * 4.1 - seed)
        # Slight center sag under gravity
        sag = -0.08 * (1.0 - nx * nx) if nz > 0.0 else 0.0
        v.co = Vector((
            nx * hl * (1.0 + lump * 0.4),
            ny * hw * end_pinch * ear_flare * (1.0 + lump),
            (z_shaped * end_pinch + sag) * hh * (1.0 + lump * 0.7) + hh * 0.68,
        ))

    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    _set_smooth(mesh)

    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(loc)
    obj.rotation_euler = Euler((math.radians(tilt_deg), 0.0, math.radians(yaw_deg)), "XYZ")
    obj.data.materials.append(mat)
    _link_obj(obj, col)
    return obj


def _create_ammo_crate(
    name: str,
    loc: tuple[float, float, float],
    yaw_deg: float,
    dims: tuple[float, float, float],
    mat_wood: bpy.types.Material,
    mat_metal: bpy.types.Material,
    col: bpy.types.Collection,
    separate_lid: bool = False,
) -> dict:
    """Create a detailed weathered military olive wooden ammo crate with corner cleats & steel latches."""
    lx, ly, lz = dims
    body_h = lz * 0.84
    lid_h = lz * 0.16

    # Main crate box with beveled edges & recessed plank rim
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(lx, ly, body_h), verts=bm.verts)
    bmesh.ops.translate(bm, vec=(0.0, 0.0, body_h * 0.5), verts=bm.verts)
    bmesh.ops.bevel(bm, geom=list(bm.edges), offset=0.008, segments=2, affect="EDGES")
    # Reinforcing end vertical wood battens
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            res = bmesh.ops.create_cube(bm, size=1.0)
            bmesh.ops.scale(bm, vec=(0.055, 0.055, body_h * 0.96), verts=res["verts"])
            bmesh.ops.translate(
                bm,
                vec=(sx * (lx * 0.5 - 0.02), sy * (ly * 0.5 - 0.02), body_h * 0.5),
                verts=res["verts"],
            )
    mesh = bpy.data.meshes.new(f"{name}_Body_Mesh")
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()

    crate = bpy.data.objects.new(name, mesh)
    crate.location = Vector(loc)
    crate.rotation_euler = Euler((0.0, 0.0, math.radians(yaw_deg)), "XYZ")
    crate.data.materials.append(mat_wood)
    _link_obj(crate, col)

    # Steel latches on front face
    lbm = bmesh.new()
    for sx in (-0.28 * lx, 0.28 * lx):
        res = bmesh.ops.create_cube(lbm, size=1.0)
        bmesh.ops.scale(lbm, vec=(0.045, 0.022, 0.09), verts=res["verts"])
        bmesh.ops.translate(lbm, vec=(sx, -ly * 0.5 - 0.008, body_h * 0.86), verts=res["verts"])
    lmesh = bpy.data.meshes.new(f"{name}_Latches_Mesh")
    lbm.normal_update()
    lbm.to_mesh(lmesh)
    lbm.free()
    latches = bpy.data.objects.new(f"{name}_Latches", lmesh)
    latches.parent = crate
    latches.data.materials.append(mat_metal)
    _link_obj(latches, col)

    # Lid (optionally independent in world space so Beat 6 explosion can blow it sky-high!)
    lbm2 = bmesh.new()
    bmesh.ops.create_cube(lbm2, size=1.0)
    bmesh.ops.scale(lbm2, vec=(lx * 1.03, ly * 1.03, lid_h), verts=lbm2.verts)
    bmesh.ops.bevel(lbm2, geom=list(lbm2.edges), offset=0.006, segments=2, affect="EDGES")
    lid_mesh = bpy.data.meshes.new(f"{name}_Lid_Mesh")
    lbm2.normal_update()
    lbm2.to_mesh(lid_mesh)
    lbm2.free()
    lid = bpy.data.objects.new(f"{name}_Lid", lid_mesh)
    lid.data.materials.append(mat_wood)
    if separate_lid:
        lid.location = Vector((loc[0], loc[1], loc[2] + body_h + lid_h * 0.5))
        lid.rotation_euler = Euler((0.0, 0.0, math.radians(yaw_deg)), "XYZ")
    else:
        lid.parent = crate
        lid.location = Vector((0.0, 0.0, body_h + lid_h * 0.5))
    _link_obj(lid, col)

    return {"crate": crate, "lid": lid, "latches": latches}


def _create_oil_drum(name: str, loc: tuple[float, float, float], mat_metal: bpy.types.Material, col) -> bpy.types.Object:
    """Create a 55-gallon dark tactical ribbed steel oil drum."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=28, radius1=0.29, radius2=0.29, depth=0.88)
    bmesh.ops.translate(bm, vec=(0.0, 0.0, 0.44), verts=bm.verts)
    # Add 2 rolling chime ribs + top/bottom rims
    for rz in (0.04, 0.30, 0.58, 0.86):
        res = bmesh.ops.create_cone(bm, cap_ends=False, cap_tris=False, segments=28, radius1=0.305, radius2=0.305, depth=0.032)
        bmesh.ops.translate(bm, vec=(0.0, 0.0, rz), verts=res["verts"])
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    _set_smooth(mesh)

    drum = bpy.data.objects.new(name, mesh)
    drum.location = Vector(loc)
    drum.data.materials.append(mat_metal)
    _link_obj(drum, col)
    return drum


def _build_hero_cover(mats_bag, mat_wood, mat_metal, col) -> dict:
    """Build Hero Cover Position: tightly interlocked low sandbag parapet (Y=0.72, Z=0..0.76)
    + weathered olive wooden ammo crate at (-0.85, 0.55, 0.0)."""
    bags = []
    # 5 tightly settled running-bond courses (step=0.122m, bag height=0.175m -> solid interlocking stack!)
    courses = [
        # (z, count, x_span, y_base, curve_y)
        (0.000, 6, 1.38, 0.72, 0.20),
        (0.122, 5, 1.20, 0.73, 0.18),
        (0.244, 5, 1.16, 0.74, 0.17),
        (0.366, 4, 0.94, 0.75, 0.15),
        (0.488, 3, 0.70, 0.75, 0.12),
    ]
    idx = 0
    for c_i, (cz, cnt, x_span, y_base, cy) in enumerate(courses):
        for k in range(cnt):
            u = 0.0 if cnt == 1 else (k / (cnt - 1)) * 2.0 - 1.0
            bx = u * x_span
            by = y_base - cy * (u * u) + (0.015 if (k + c_i) % 2 == 0 else -0.015)
            dy_dx = -2.0 * cy * u / max(0.2, x_span)
            yaw = math.degrees(math.atan2(dy_dx, 1.0)) + ((idx % 3) - 1) * 2.0
            bag = _create_sculpted_sandbag(
                f"Hero_Sandbag_{idx:02d}",
                (bx, by, cz),
                yaw,
                mats_bag[idx % len(mats_bag)],
                col,
                length=0.60,
                width=0.31,
                height=0.175,
                seed=100 + idx,
            )
            bags.append(bag)
            idx += 1

    ammo_crate = _create_ammo_crate(
        "Hero_AmmoCrate",
        (-0.85, 0.55, 0.0),
        yaw_deg=18.0,
        dims=(0.62, 0.36, 0.34),
        mat_wood=mat_wood,
        mat_metal=mat_metal,
        col=col,
        separate_lid=False,
    )
    return {"sandbags": bags, "ammo_crate": ammo_crate}


def _build_enemy_bunker(mats_bag, mat_wood_olive, mat_timber, mat_tarp, mat_metal, mat_drum, col, total_frames: int) -> dict:
    """Build dramatic semicircular Enemy Tactical Sandbag Bunker at Y≈7.8 with 24 plump interlocking sandbags,
    timber canopy posts + desert camo tarp, supply crates, oil drum & Beat 6 blast animation!"""
    cx, cy = ENEMY_BUNKER_CENTER[0], ENEMY_BUNKER_CENTER[1]
    bags = []
    animated_bags = []

    # Semicircular arc: 3 tightly interlocked running-bond courses x 8 sandbags = 24 plump burlap sandbags!
    # Vertical step = 0.165m with bag height = 0.245m -> bags settle heavily into each other with ZERO floating gaps!
    radius_x = 1.78
    radius_y = 0.86
    center_y = cy + radius_y  # Front apex right at Y=7.80!

    bag_idx = 0
    for row in range(3):
        n_row = 8
        z_row = row * 0.165
        # Half-bag running-bond stagger on middle course
        angle_offset = math.radians(8.5) if row == 1 else 0.0
        for k in range(n_row):
            u = (k / (n_row - 1)) * 2.0 - 1.0
            ang = math.radians(u * 66.0) + angle_offset * (1.0 - 0.35 * abs(u))
            bx = cx + math.sin(ang) * (radius_x - row * 0.035)
            by = center_y - math.cos(ang) * (radius_y - row * 0.025)
            yaw = math.degrees(ang) + ((bag_idx % 3) - 1) * 1.8
            bag = _create_sculpted_sandbag(
                f"Enemy_Sandbag_{bag_idx:02d}",
                (bx, by, z_row),
                yaw,
                mats_bag[bag_idx % len(mats_bag)],
                col,
                length=0.72,
                width=0.35,
                height=0.245,
                seed=300 + bag_idx,
            )
            bags.append(bag)
            # Pick 10 front-center / upper-course sandbags to blast outward at FRACTURE_FRAME (612)!
            is_front_blast = (row == 2 and 1 <= k <= 6) or (row == 1 and 2 <= k <= 5)
            if is_front_blast and len(animated_bags) < 10:
                animated_bags.append((bag, (bx, by, z_row), yaw, row, k))
            bag_idx += 1

    # 4 Weathered Timber Roof Posts framing the bunker perimeter
    posts = []
    post_coords = [
        (-1.82, 8.12, 2.28),
        (1.82, 8.12, 2.28),
        (-1.95, 9.65, 2.08),
        (1.95, 9.65, 2.08),
    ]
    for p_i, (px, py, ph) in enumerate(post_coords):
        pbm = bmesh.new()
        bmesh.ops.create_cone(pbm, cap_ends=True, cap_tris=False, segments=10, radius1=0.068, radius2=0.058, depth=ph)
        bmesh.ops.translate(pbm, vec=(0.0, 0.0, ph * 0.5), verts=pbm.verts)
        pmesh = bpy.data.meshes.new(f"Bunker_Post_{p_i}_Mesh")
        pbm.normal_update()
        pbm.to_mesh(pmesh)
        pbm.free()
        _set_smooth(pmesh)
        post = bpy.data.objects.new(f"Bunker_Post_{p_i}", pmesh)
        post.location = Vector((px, py, 0.0))
        post.data.materials.append(mat_timber)
        _link_obj(post, col)
        posts.append(post)

    # Timber crossbeams
    for b_i, (y_beam, z_beam) in enumerate([(8.12, 2.24), (9.65, 2.05)]):
        bbm = bmesh.new()
        bmesh.ops.create_cube(bbm, size=1.0)
        bmesh.ops.scale(bbm, vec=(4.15, 0.09, 0.09), verts=bbm.verts)
        beam_mesh = bpy.data.meshes.new(f"Bunker_Beam_{b_i}_Mesh")
        bbm.normal_update()
        bbm.to_mesh(beam_mesh)
        bbm.free()
        beam = bpy.data.objects.new(f"Bunker_Beam_{b_i}", beam_mesh)
        beam.location = Vector((0.0, y_beam, z_beam))
        beam.data.materials.append(mat_timber)
        _link_obj(beam, col)
        posts.append(beam)

    # Draped Desert Camo Tarp Canopy with natural catenary sag & edge ruffle
    tbm = bmesh.new()
    tx_n, ty_n = 18, 12
    tgrid = []
    for j in range(ty_n + 1):
        v = j / ty_n  # 0 (front y=7.95) .. 1 (back y=9.85)
        ty = 7.95 + v * 1.90
        z_slope = 2.30 - v * 0.22
        row_v = []
        for i in range(tx_n + 1):
            u = (i / tx_n) * 2.0 - 1.0  # -1 .. 1
            tx = u * 2.18
            # Natural sag in center + slight side flap droop outside posts
            center_sag = -0.15 * (1.0 - u * u) * math.sin(v * math.pi)
            flap_droop = -0.22 * max(0.0, abs(u) - 0.84) / 0.16
            ripple = 0.024 * math.sin(u * 7.5 + v * 4.2)
            row_v.append(tbm.verts.new((tx, ty, z_slope + center_sag + flap_droop + ripple)))
        tgrid.append(row_v)
    for j in range(ty_n):
        for i in range(tx_n):
            tbm.faces.new((tgrid[j][i], tgrid[j][i + 1], tgrid[j + 1][i + 1], tgrid[j + 1][i]))
    tbm.normal_update()
    tmesh = bpy.data.meshes.new("Bunker_CamoTarp_Mesh")
    tbm.to_mesh(tmesh)
    tbm.free()
    _set_smooth(tmesh)
    tarp = bpy.data.objects.new("Bunker_CamoTarp", tmesh)
    tarp.data.materials.append(mat_tarp)
    sol = tarp.modifiers.new("Solidify", "SOLIDIFY")
    sol.thickness = 0.012
    _link_obj(tarp, col)

    # Olive Military Supply Crates (one with separate lid that blows sky-high at f=612!) + Oil Drum
    crate_front = _create_ammo_crate(
        "Enemy_SupplyCrate_Front",
        (-1.42, 7.62, 0.0),
        yaw_deg=-26.0,
        dims=(0.74, 0.44, 0.42),
        mat_wood=mat_wood_olive,
        mat_metal=mat_metal,
        col=col,
        separate_lid=True,
    )
    crate_right = _create_ammo_crate(
        "Enemy_SupplyCrate_Right",
        (1.62, 7.82, 0.0),
        yaw_deg=32.0,
        dims=(0.68, 0.42, 0.40),
        mat_wood=mat_wood_olive,
        mat_metal=mat_metal,
        col=col,
        separate_lid=False,
    )
    oil_drum = _create_oil_drum("Enemy_TacticalOilDrum", (1.92, 8.62, 0.0), mat_drum, col)

    # Animate 10 Enemy Front Sandbags + Front Crate Lid blasting outward in slow-mo at FRACTURE_FRAME (612)!
    orig = Vector(EXPLOSION_ORIGIN)
    rng = random.Random(6129)
    for b_idx, (bag, base_loc, base_yaw, row, k) in enumerate(animated_bags):
        p0 = Vector(base_loc)
        radial = p0 - orig
        radial.z = 0.0
        if radial.length < 0.08:
            radial = Vector((rng.uniform(-0.4, 0.4), 1.0, 0.0))
        radial.normalize()

        # Dramatic radial outward + upward slow-mo trajectory (f = 612 .. 720)
        speed_xy = rng.uniform(1.85, 3.15) * (1.0 + 0.14 * row)
        speed_z = rng.uniform(1.45, 2.65) + 0.25 * row
        spin = Vector((rng.uniform(-4.5, 4.5), rng.uniform(-4.5, 4.5), rng.uniform(-3.8, 3.8)))

        for f in (1, FRACTURE_FRAME):
            bag.location = p0
            bag.rotation_euler = Euler((0.0, 0.0, math.radians(base_yaw)), "XYZ")
            bag.keyframe_insert("location", frame=f)
            bag.keyframe_insert("rotation_euler", frame=f)

        # Sample slow-mo ballistic arc (bullet-time deceleration curve!)
        for f in (624, 644, 672, 700, total_frames):
            # Non-linear bullet-time parameter u in [0, 1] (fast initial blast -> dramatic slow-mo hang!)
            lin = (f - FRACTURE_FRAME) / max(1.0, float(total_frames - FRACTURE_FRAME))
            u = lin ** 0.62
            pos = Vector((
                p0.x + radial.x * speed_xy * u,
                p0.y + radial.y * speed_xy * u,
                max(0.08, p0.z + speed_z * u - 1.35 * (u * u)),
            ))
            rot = Euler((
                spin.x * u,
                spin.y * u,
                math.radians(base_yaw) + spin.z * u,
            ), "XYZ")
            bag.location = pos
            bag.rotation_euler = rot
            bag.keyframe_insert("location", frame=f)
            bag.keyframe_insert("rotation_euler", frame=f)
        _set_bezier(bag)

    # Animate front crate lid blasting up & spinning outward!
    lid = crate_front["lid"]
    lid_p0 = lid.location.copy()
    lid_r0 = lid.rotation_euler.copy()
    for f in (1, FRACTURE_FRAME):
        lid.location = lid_p0
        lid.rotation_euler = lid_r0
        lid.keyframe_insert("location", frame=f)
        lid.keyframe_insert("rotation_euler", frame=f)
    for f in (626, 650, 682, total_frames):
        lin = (f - FRACTURE_FRAME) / max(1.0, float(total_frames - FRACTURE_FRAME))
        u = lin ** 0.60
        lid.location = Vector((
            lid_p0.x - 1.55 * u,
            lid_p0.y + 0.85 * u,
            lid_p0.z + 2.45 * u - 1.10 * (u * u),
        ))
        lid.rotation_euler = Euler((lid_r0.x + 3.6 * u, lid_r0.y - 4.2 * u, lid_r0.z + 2.8 * u), "XYZ")
        lid.keyframe_insert("location", frame=f)
        lid.keyframe_insert("rotation_euler", frame=f)
    _set_bezier(lid)

    return {
        "sandbags": bags,
        "animated_sandbags": [x[0] for x in animated_bags],
        "posts": posts,
        "tarp": tarp,
        "crate_front": crate_front,
        "crate_right": crate_right,
        "oil_drum": oil_drum,
    }


def _build_canyon_mesas(mat_mesa, col) -> list[bpy.types.Object]:
    """Build distant sunlit desert canyon mesas (Y=20..36 + flank ridges) for dramatic silhouette depth."""
    mesas = []
    # (name, x, y, width_x, depth_y, height_z, yaw_deg, seed)
    specs = [
        ("Desert_Mesa_CenterLeft", -9.5, 24.5, 13.0, 6.2, 6.4, 12.0, 11),
        ("Desert_Mesa_CenterRight", 10.2, 26.0, 14.5, 6.8, 7.2, -14.0, 23),
        ("Desert_Mesa_FarCenter", 0.6, 33.5, 19.0, 7.5, 8.8, 4.0, 37),
        ("Desert_Mesa_FarLeft", -21.0, 29.0, 16.0, 7.0, 7.8, 28.0, 49),
        ("Desert_Mesa_FarRight", 21.5, 28.0, 16.0, 7.0, 8.1, -26.0, 61),
        # Subtle back ridge behind hero (visible when camera faces -Y toward hero!)
        ("Desert_Mesa_BackLeft", -11.5, -18.5, 14.0, 6.0, 5.6, -16.0, 73),
        ("Desert_Mesa_BackRight", 11.0, -20.5, 15.0, 6.4, 6.2, 18.0, 89),
    ]
    for name, mx, my, wx, dy, hz, yaw, seed in specs:
        bm = bmesh.new()
        nx, ny = 22, 14
        grid = []
        for j in range(ny + 1):
            v = (j / ny) * 2.0 - 1.0
            row = []
            for i in range(nx + 1):
                u = (i / nx) * 2.0 - 1.0
                # Flat-topped stepped canyon mesa profile
                r_box = max(abs(u), abs(v))
                # Terraced sandstone cliff profile
                base_h = max(0.0, min(1.0, (1.0 - r_box) / 0.42))
                terrace = base_h + 0.07 * math.sin(base_h * math.pi * 4.0)
                crag = (
                    0.14 * math.sin(u * 5.2 + seed) * math.cos(v * 4.7 - seed)
                    + 0.08 * math.cos(u * 11.0 - v * 9.5 + seed)
                )
                z = max(-0.4, (terrace + crag * (1.0 - 0.5 * r_box)) * hz)
                row.append(bm.verts.new((u * wx * 0.5, v * dy * 0.5, z)))
            grid.append(row)
        for j in range(ny):
            for i in range(nx):
                bm.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))
        bm.normal_update()
        mesh = bpy.data.meshes.new(f"{name}_Mesh")
        bm.to_mesh(mesh)
        bm.free()
        _set_smooth(mesh)
        obj = bpy.data.objects.new(name, mesh)
        obj.location = Vector((mx, my, -0.35))
        obj.rotation_euler = Euler((0.0, 0.0, math.radians(yaw)), "XYZ")
        obj.data.materials.append(mat_mesa)
        _link_obj(obj, col)
        mesas.append(obj)
    return mesas


def _build_explosion_plume(col, total_frames: int) -> dict:
    """Build Beat 6 (f=612..720) PG-13 Hollywood dust/dirt shockwave ring + billowing warm dust clouds
    + inner fireball core erupting from EXPLOSION_ORIGIN (0.0, 7.45, 0.0)."""
    mat_dust_outer = _make_explosion_dust_mat("Mat_ExplosionDust_Outer", warm_glow=0.04)
    mat_dust_inner = _make_explosion_dust_mat("Mat_ExplosionDust_Lit", warm_glow=0.55)
    mat_fire = _make_fireball_core_mat()

    ox, oy, oz = EXPLOSION_ORIGIN
    fx_objs = []

    # 1. Ground-Zero Dust & Dirt Shockwave Torus Ring
    bm_ring = bmesh.new()
    segs_maj, segs_min = 40, 14
    maj_r, min_r = 1.0, 0.24
    ring_verts = []
    for i in range(segs_maj):
        a = (i / segs_maj) * math.tau
        ca, sa = math.cos(a), math.sin(a)
        loop = []
        for j in range(segs_min):
            b = (j / segs_min) * math.tau
            bump = 1.0 + 0.18 * math.sin(a * 7.0 + b * 2.0) + 0.10 * math.cos(a * 13.0)
            rr = maj_r + (min_r * bump) * math.cos(b)
            zz = (min_r * 0.72 * bump) * math.sin(b) + min_r * 0.65
            loop.append(bm_ring.verts.new((rr * ca, rr * sa, zz)))
        ring_verts.append(loop)
    for i in range(segs_maj):
        ni = (i + 1) % segs_maj
        for j in range(segs_min):
            nj = (j + 1) % segs_min
            bm_ring.faces.new((ring_verts[i][j], ring_verts[ni][j], ring_verts[ni][nj], ring_verts[i][nj]))
    bm_ring.normal_update()
    ring_mesh = bpy.data.meshes.new("Explosion_ShockwaveDustRing_Mesh")
    bm_ring.to_mesh(ring_mesh)
    bm_ring.free()
    _set_smooth(ring_mesh)

    shock_ring = bpy.data.objects.new("Explosion_ShockwaveDustRing", ring_mesh)
    shock_ring.location = Vector((ox, oy, oz + 0.04))
    shock_ring.data.materials.append(mat_dust_outer)
    shock_ring.visible_shadow = False
    _link_obj(shock_ring, col)
    _key_visibility(shock_ring, [(FRACTURE_FRAME, total_frames)], total_frames)
    for f, sc_xy, sc_z in [(612, 0.28, 0.35), (624, 1.55, 0.95), (656, 2.75, 1.25), (720, 3.60, 1.45)]:
        shock_ring.scale = Vector((sc_xy, sc_xy, sc_z))
        shock_ring.keyframe_insert("scale", frame=f)
    _set_bezier(shock_ring)
    fx_objs.append(shock_ring)

    # 2. Multi-lobe Billowing Sunlit Dust Plume Clouds
    rng = random.Random(9012)
    cloud_specs = [
        # (dx, dy, dz_start, dz_end, r_start, r_end, is_lit)
        (0.0, 0.0, 0.35, 2.35, 0.30, 1.65, True),
        (-0.55, -0.15, 0.25, 1.75, 0.24, 1.28, False),
        (0.58, -0.12, 0.28, 1.85, 0.25, 1.32, False),
        (-0.35, 0.42, 0.30, 2.05, 0.26, 1.42, True),
        (0.42, 0.38, 0.32, 2.15, 0.26, 1.38, False),
        (0.05, -0.35, 0.20, 1.40, 0.22, 1.18, True),
        (-0.95, 0.15, 0.18, 1.15, 0.20, 1.05, False),
        (0.98, 0.20, 0.18, 1.22, 0.20, 1.08, False),
    ]
    for c_i, (dx, dy, zs, ze, rs, re, is_lit) in enumerate(cloud_specs):
        cbm = bmesh.new()
        bmesh.ops.create_icosphere(cbm, subdivisions=3, radius=1.0)
        for v in cbm.verts:
            p = v.co
            wobble = (
                1.0
                + 0.22 * math.sin(p.x * 3.8 + c_i) * math.cos(p.y * 3.5)
                + 0.14 * math.cos(p.z * 4.6 - c_i * 1.4)
            )
            v.co = p * wobble
        cbm.normal_update()
        cmesh = bpy.data.meshes.new(f"Explosion_DustCloud_{c_i:02d}_Mesh")
        cbm.to_mesh(cmesh)
        cbm.free()
        _set_smooth(cmesh)

        cloud = bpy.data.objects.new(f"Explosion_DustCloud_{c_i:02d}", cmesh)
        cloud.data.materials.append(mat_dust_inner if is_lit else mat_dust_outer)
        cloud.visible_shadow = False
        _link_obj(cloud, col)
        _key_visibility(cloud, [(FRACTURE_FRAME, total_frames)], total_frames)

        for f, frac in [(612, 0.0), (626, 0.42), (664, 0.78), (total_frames, 1.0)]:
            rad = rs + (re - rs) * frac
            cloud.location = Vector((ox + dx * (0.4 + 1.3 * frac), oy + dy * (0.4 + 1.3 * frac), oz + zs + (ze - zs) * frac))
            cloud.scale = Vector((rad, rad, rad * (0.85 + 0.22 * frac)))
            cloud.rotation_euler = Euler((0.0, 0.0, rng.uniform(0.0, 1.2) + frac * 0.65), "XYZ")
            cloud.keyframe_insert("location", frame=f)
            cloud.keyframe_insert("scale", frame=f)
            cloud.keyframe_insert("rotation_euler", frame=f)
        _set_bezier(cloud)
        fx_objs.append(cloud)

    # 3. Inner Incandescent Fireball Core (additive volumetric glow!)
    fbm = bmesh.new()
    bmesh.ops.create_icosphere(fbm, subdivisions=3, radius=1.0)
    for v in fbm.verts:
        p = v.co
        v.co = p * (1.0 + 0.20 * math.sin(p.x * 4.5) * math.cos(p.y * 4.2 + p.z * 3.8))
    fbm.normal_update()
    fmesh = bpy.data.meshes.new("Explosion_FireballCore_Mesh")
    fbm.to_mesh(fmesh)
    fbm.free()
    _set_smooth(fmesh)

    fireball = bpy.data.objects.new("Explosion_FireballCore", fmesh)
    fireball.data.materials.append(mat_fire)
    fireball.visible_shadow = False
    _link_obj(fireball, col)
    _key_visibility(fireball, [(FRACTURE_FRAME, total_frames)], total_frames)
    for f, r_fb, z_fb in [(612, 0.22, 0.25), (622, 1.15, 0.72), (650, 1.48, 1.22), (720, 1.25, 1.65)]:
        fireball.location = Vector((ox, oy, oz + z_fb))
        fireball.scale = Vector((r_fb, r_fb, r_fb * 1.12))
        fireball.keyframe_insert("location", frame=f)
        fireball.keyframe_insert("scale", frame=f)
    _set_bezier(fireball)
    fx_objs.append(fireball)

    return {"shock_ring": shock_ring, "fireball": fireball, "all_fx": fx_objs}


# ============================================================================
# GOLDEN-HOUR NISHITA SKY & HOLLYWOOD CINEMA LIGHTING RIG (AGENTS.MD SEC 1!)
# ============================================================================
def _add_light(
    name: str,
    ltype: str,
    loc,
    rot,
    energy: float,
    color: tuple[float, float, float],
    col: bpy.types.Collection,
    size: float = 1.0,
) -> bpy.types.Object:
    ld = bpy.data.lights.new(f"{name}_Data", type=ltype)
    ld.energy = energy
    ld.color = color
    if ltype == "AREA":
        ld.size = size
    elif ltype == "POINT":
        ld.shadow_soft_size = size
    obj = bpy.data.objects.new(name, ld)
    obj.location = Vector(loc)
    obj.rotation_euler = rot if isinstance(rot, Euler) else Euler(rot, "XYZ")
    obj.visible_camera = False
    _link_obj(obj, col)
    return obj


def _build_golden_hour_world_and_lights(col: bpy.types.Collection, total_frames: int) -> dict:
    """Physical Nishita golden-hour desert sky (sun_elevation=16 deg, raking side-back sunlight, crisp dust haze)
    + Hero Rembrandt bounce fill + AGENTS.md Section 1 0.42m AREA Softbox Eye Catchlight + Beat 6 blast flash."""
    scene = bpy.context.scene
    world = bpy.data.worlds.new("DesertOutpost_GoldenHour_World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_disc = True
    sky.sun_size = math.radians(0.85)
    sky.sun_intensity = 1.0
    sky.sun_elevation = math.radians(16.0)
    # Raking golden-hour side-back sunlight (carves dramatic rim light across Hero & sandbag bunker!)
    sky.sun_rotation = math.radians(122.0)
    sky.altitude = 450.0
    sky.air_density = 1.45
    sky.dust_density = 3.15  # Rich sunlit golden desert atmospheric haze!
    sky.ozone_density = 1.25
    bg.inputs["Strength"].default_value = 0.50

    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    # Ensure Cycles bounce budget satisfies AGENTS.md Section 3 (>=10 transparent bounces for hair/corneas/dust!)
    if scene.render.engine == "CYCLES":
        scene.cycles.transparent_max_bounces = max(12, scene.cycles.transparent_max_bounces)

    # Hero eye focus target (1.80m Hero standing at (0,0), eye height ~1.67m)
    hero_eye_focus = Vector((0.03, 0.12, 1.67))
    hero_chest_focus = Vector((0.0, 0.05, 1.42))

    # 1. Subtle Warm Desert Sand Bounce Fill Softbox (lifts facial shadows on Hero without killing golden-hour drama)
    fill_loc = (1.35, 1.75, 1.55)
    fill_light = _add_light(
        "Hero_DesertBounceFill",
        "AREA",
        fill_loc,
        _aim_euler(fill_loc, hero_chest_focus),
        energy=48.0,
        color=(1.0, 0.80, 0.56),
        col=col,
        size=1.85,
    )

    # 2. AGENTS.md Section 1 Mandatory Wet-Cornea Softbox Eye Catchlight (0.42m AREA, 52W, diff=0.28, spec=1.75!)
    # Positioned in front-right (+X, +Y) where Hero faces (yaw=162 deg -> faces +0.31X, +0.95Y)
    catch_loc = (0.58, 1.28, 1.78)
    catch_light = _add_light(
        "Hero_EyeCatchlight",
        "AREA",
        catch_loc,
        _aim_euler(catch_loc, hero_eye_focus),
        energy=52.0,
        color=(1.0, 0.96, 0.90),
        col=col,
        size=0.42,
    )
    if hasattr(catch_light.data, "diffuse_factor"):
        catch_light.data.diffuse_factor = 0.28
    if hasattr(catch_light.data, "specular_factor"):
        catch_light.data.specular_factor = 1.75

    # 3. Subtle Cool Desert Sky Rim on Hero opposite sun for sculptural helmet & shoulder separation
    rim_loc = (-1.75, -1.65, 2.35)
    rim_light = _add_light(
        "Hero_SkyBackRim",
        "AREA",
        rim_loc,
        _aim_euler(rim_loc, hero_chest_focus),
        energy=95.0,
        color=(0.68, 0.82, 1.0),
        col=col,
        size=1.60,
    )

    # 4. Enemy Bunker Eye/Fill Bounce (keeps Rival Sentries crisp & readable under the camo tarp canopy!)
    bunker_fill_loc = (0.0, 6.10, 1.65)
    bunker_fill = _add_light(
        "Bunker_CanopyFill",
        "AREA",
        bunker_fill_loc,
        _aim_euler(bunker_fill_loc, Vector((0.0, 8.45, 1.35))),
        energy=90.0,
        color=(1.0, 0.84, 0.62),
        col=col,
        size=2.20,
    )

    # 5. Beat 6 Explosion Dynamic Fireball Flash Light (lights sandbags, sentries & dust clouds at f=612..720!)
    blast_light = _add_light(
        "Bunker_ExplosionFlashLight",
        "POINT",
        (EXPLOSION_ORIGIN[0], EXPLOSION_ORIGIN[1], 0.95),
        (0.0, 0.0, 0.0),
        energy=0.0,
        color=(1.0, 0.48, 0.14),
        col=col,
        size=0.65,
    )
    for f, en in [(1, 0.0), (611, 0.0), (615, 6500.0), (632, 3200.0), (675, 1100.0), (total_frames, 380.0)]:
        blast_light.data.energy = en
        blast_light.data.keyframe_insert("energy", frame=f)
    _set_bezier(blast_light.data)

    return {
        "world": world,
        "sky_node": sky,
        "fill_light": fill_light,
        "catch_light": catch_light,
        "rim_light": rim_light,
        "bunker_fill": bunker_fill,
        "blast_light": blast_light,
    }


def build_battlefield(col: bpy.types.Collection | None = None, total_frames: int = TOTAL_FRAMES_DEFAULT) -> dict:
    """Build the complete Hollywood golden-hour desert tactical outpost battlefield:
    - Physical Nishita sky (16 deg elevation, warm raking sunlight & crisp dust haze) + cinema lights
    - Undulating cracked-ochre rocky desert ground + boulders
    - Hero cover sandbag parapet (Y=0.72) + olive ammo crate (-0.85, 0.55)
    - Enemy semicircular 24-sandbag bunker (Y=7.8) + timber camo tarp canopy + supply crates + oil drum
      (with 10 front sandbags + crate lid blasting outward in slow-mo at f=612!)
    - Distant sunlit desert canyon mesas (Y=20..36 + back flank ridges)
    - Beat 6 PG-13 explosion dust shockwave ring + billowing dust clouds + fireball core (Y=7.45)"""
    target_col = col if col is not None else bpy.context.scene.collection

    lights = _build_golden_hour_world_and_lights(target_col, total_frames)

    mat_ground = _make_desert_ground_mat()
    mat_rock = _make_rock_mat("Mat_DesertBoulder", stratified=False)
    mat_mesa = _make_rock_mat("Mat_CanyonMesa", stratified=True)
    mats_bag = [
        _make_burlap_sandbag_mat("Mat_BurlapSandbag_A", variant=-0.6),
        _make_burlap_sandbag_mat("Mat_BurlapSandbag_B", variant=0.0),
        _make_burlap_sandbag_mat("Mat_BurlapSandbag_C", variant=0.7),
    ]
    mat_wood_olive = _make_wood_crate_mat("Mat_OliveWoodCrate", olive=True)
    mat_timber = _make_wood_crate_mat("Mat_WeatheredTimber", olive=False)
    mat_tarp = _make_camo_tarp_mat()
    mat_metal = _make_tactical_metal_mat("Mat_CrateSteel", (0.12, 0.13, 0.12), metallic=0.80, roughness=0.45)
    mat_drum = _make_tactical_metal_mat("Mat_TacticalOilDrum", (0.06, 0.07, 0.075), metallic=0.72, roughness=0.38)

    terrain = _build_desert_terrain(mat_ground, mat_rock, target_col)
    hero_cover = _build_hero_cover(mats_bag, mat_wood_olive, mat_metal, target_col)
    enemy_bunker = _build_enemy_bunker(
        mats_bag, mat_wood_olive, mat_timber, mat_tarp, mat_metal, mat_drum, target_col, total_frames
    )
    mesas = _build_canyon_mesas(mat_mesa, target_col)
    explosion_fx = _build_explosion_plume(target_col, total_frames)

    return {
        "world": lights["world"],
        "lights": lights,
        "terrain": terrain,
        "hero_cover": hero_cover,
        "enemy_bunker": enemy_bunker,
        "animated_sandbags": enemy_bunker["animated_sandbags"],
        "crate_lid": enemy_bunker["crate_front"]["lid"],
        "mesas": mesas,
        "explosion_fx": explosion_fx,
        "grenade_landing_pos": GRENADE_LANDING_POS,
        "explosion_origin": EXPLOSION_ORIGIN,
    }


# ============================================================================
# CUSTOM SCULPTED TACTICAL HELMET (MICH/FAST) & PLATE CARRIER VEST
# ============================================================================
def _eval_mesh_world_bounds(obj: bpy.types.Object) -> tuple[Vector, Vector, Vector]:
    """Return (min_v, max_v, center_v) of an evaluated mesh in world space."""
    deps = bpy.context.evaluated_depsgraph_get()
    eo = obj.evaluated_get(deps)
    em = eo.to_mesh()
    pts = [eo.matrix_world @ v.co for v in em.vertices]
    eo.to_mesh_clear()
    if not pts:
        z = Vector((0.0, 0.0, 0.0))
        return z, z, z
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    ctr = sum(pts, Vector((0.0, 0.0, 0.0))) / len(pts)
    return mn, mx, ctr


def _normalize_actor_height_and_ground(actor: dict, target_height_m: float, floor_z: float) -> dict:
    """Re-normalize MPFB2 actor scale & floor Z using EVALUATED mesh bounds (Body + Shoes)
    so every soldier stands at exact `target_height_m` with boot soles planted at `floor_z`."""
    rig = actor["rig"]
    body = actor["body"]
    shoes = actor["equipped"].get("Shoes")

    bpy.context.view_layer.update()
    b_min, b_max, _ = _eval_mesh_world_bounds(body)
    s_min = _eval_mesh_world_bounds(shoes)[0] if shoes is not None else b_min
    cur_min_z = min(b_min.z, s_min.z)
    cur_max_z = b_max.z
    cur_h = max(0.5, cur_max_z - cur_min_z)

    ratio = target_height_m / cur_h
    rig.scale = rig.scale * ratio
    bpy.context.view_layer.update()

    b_min2, b_max2, _ = _eval_mesh_world_bounds(body)
    s_min2 = _eval_mesh_world_bounds(shoes)[0] if shoes is not None else b_min2
    cur_min_z2 = min(b_min2.z, s_min2.z)
    rig.location.z += (floor_z - cur_min_z2)
    bpy.context.view_layer.update()

    _, b_max3, _ = _eval_mesh_world_bounds(body)
    _, _, eye_ctr = _eval_mesh_world_bounds(actor["equipped"]["Eyes"]) if "Eyes" in actor["equipped"] else (None, None, Vector((0, 0, floor_z + target_height_m * 0.93)))
    _, _, teeth_ctr = _eval_mesh_world_bounds(actor["equipped"]["Teeth"]) if "Teeth" in actor["equipped"] else (None, None, Vector((0, 0, floor_z + target_height_m * 0.89)))

    actor["height"] = b_max3.z - floor_z
    actor["base_z"] = rig.location.z
    actor["eye_center_rest"] = eye_ctr
    actor["teeth_center_rest"] = teeth_ctr
    return actor


def _build_tactical_helmet(
    name: str,
    actor: dict,
    mat_shell: bpy.types.Material,
    mat_hardware: bpy.types.Material,
    mat_goggle_frame: bpy.types.Material,
    mat_goggle_lens: bpy.types.Material,
    mat_strap: bpy.types.Material,
    col: bpy.types.Collection,
) -> list[bpy.types.Object]:
    """Build a sculpted MICH/FAST tactical combat helmet + NVG bracket + side rails + brow goggles + chin strap,
    fitted cleanly to the actor's evaluated skull & parented to the `head` bone!"""
    rig = actor["rig"]
    body = actor["body"]
    bpy.context.view_layer.update()

    # Measure actor skull in rig-local space (-Y is forward, +X is left, +Z is up!)
    deps = bpy.context.evaluated_depsgraph_get()
    eb = body.evaluated_get(deps)
    em = eb.to_mesh()
    inv_rig = rig.matrix_world.inverted()
    head_top_w = max((eb.matrix_world @ v.co).z for v in em.vertices)
    all_head_local = [inv_rig @ (eb.matrix_world @ v.co) for v in em.vertices if (eb.matrix_world @ v.co).z > head_top_w - 0.28]
    skull_local = [p for p in all_head_local if p.z > (inv_rig @ Vector((0.0, 0.0, head_top_w))).z - 0.155]
    eye_ctr_local = inv_rig @ actor["eye_center_rest"]
    teeth_ctr_local = inv_rig @ actor["teeth_center_rest"]
    jaw_local = [p for p in all_head_local if abs(p.x) < 0.115 and p.z <= eye_ctr_local.z + 0.02]
    eb.to_mesh_clear()

    sx_min = min(v.x for v in skull_local)
    sx_max = max(v.x for v in skull_local)
    sy_min = min(v.y for v in skull_local)  # Front forehead (-Y)
    sy_max = max(v.y for v in skull_local)  # Back occiput (+Y)
    sz_max = max(v.z for v in skull_local)

    skull_cx = 0.5 * (sx_min + sx_max)
    skull_cy = 0.5 * (sy_min + sy_max)
    skull_rx = 0.5 * (sx_max - sx_min)
    skull_ry = 0.5 * (sy_max - sy_min)

    # Helmet hugs the cranium cleanly above brows (sleek tactical MICH/FAST profile, zero mushroom bulge!)
    brow_z = eye_ctr_local.z + 0.020
    crown_z = sz_max + 0.024
    helm_h = crown_z - brow_z
    helm_cz = brow_z + 0.012
    rx = skull_rx + 0.011
    ry = skull_ry + 0.012

    # 1. Ballistic MICH/FAST Shell (high-cut ear scallops + rear nape drop + thick edge rim)
    bm = bmesh.new()
    segs_u, segs_v = 36, 16
    rings = []
    for j in range(segs_v + 1):
        # phi from 0 (rim) to pi/2 (crown)
        v_frac = j / segs_v
        phi = v_frac * (math.pi * 0.5)
        cp, sp = math.cos(phi), math.sin(phi)
        row = []
        for i in range(segs_u):
            th = (i / segs_u) * math.tau
            # cos(th)=+1 -> +X (left), sin(th)=-1 -> -Y (front brow!), sin(th)=+1 -> +Y (rear nape!)
            ct, st = math.cos(th), math.sin(th)
            # Lower rim profile: front brow stays clear above eyes, ears high-cut, rear nape drops cleanly!
            ear_cut = 0.026 * (ct ** 4)  # High-cut FAST ear arch
            nape_drop = -0.056 * max(0.0, st) ** 1.4  # Rear occipital protection
            brow_lift = 0.008 * max(0.0, -st) ** 2.0  # Clean brow visibility
            rim_z_mod = (ear_cut + nape_drop + brow_lift) * ((1.0 - v_frac) ** 1.6)
            # Subtle rear flare at the very bottom rim
            flare = 1.0 + 0.045 * ((1.0 - v_frac) ** 3.0) * (0.4 + 0.6 * max(0.0, st))

            px = skull_cx + ct * cp * rx * flare
            py = skull_cy + st * cp * ry * flare
            pz = helm_cz + sp * (crown_z - helm_cz) + rim_z_mod
            row.append(bm.verts.new((px, py, pz)))
        rings.append(row)

    for j in range(segs_v):
        for i in range(segs_u):
            ni = (i + 1) % segs_u
            bm.faces.new((rings[j][i], rings[j][ni], rings[j + 1][ni], rings[j + 1][i]))
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.001)
    bm.normal_update()
    shell_mesh = bpy.data.meshes.new(f"{name}_HelmetShell_Mesh")
    bm.to_mesh(shell_mesh)
    bm.free()
    _set_smooth(shell_mesh)

    shell = bpy.data.objects.new(f"{name}_HelmetShell", shell_mesh)
    shell.matrix_world = rig.matrix_world.copy()
    shell.data.materials.append(mat_shell)
    sol = shell.modifiers.new("Solidify", "SOLIDIFY")
    sol.thickness = 0.011
    sol.offset = 1.0
    _link_obj(shell, col)

    # 2. Front NVG Shroud Mount + Side ARC Accessory Rails
    hbm = bmesh.new()
    # Forehead NVG shroud plate
    res_nvg = bmesh.ops.create_cube(hbm, size=1.0)
    bmesh.ops.scale(hbm, vec=(0.046, 0.016, 0.052), verts=res_nvg["verts"])
    bmesh.ops.rotate(hbm, cent=(0, 0, 0), matrix=Euler((math.radians(-16.0), 0.0, 0.0)).to_matrix(), verts=res_nvg["verts"])
    bmesh.ops.translate(hbm, vec=(skull_cx, skull_cy - ry * 0.99 - 0.006, helm_cz + helm_h * 0.34), verts=res_nvg["verts"])
    # Left & Right ARC side rails
    for sgn in (-1.0, 1.0):
        res_r = bmesh.ops.create_cube(hbm, size=1.0)
        bmesh.ops.scale(hbm, vec=(0.012, 0.105, 0.022), verts=res_r["verts"])
        bmesh.ops.translate(hbm, vec=(skull_cx + sgn * (rx * 1.02 + 0.005), skull_cy + 0.006, helm_cz + 0.032), verts=res_r["verts"])
    hbm.normal_update()
    hw_mesh = bpy.data.meshes.new(f"{name}_HelmetHardware_Mesh")
    hbm.to_mesh(hw_mesh)
    hbm.free()

    hw = bpy.data.objects.new(f"{name}_HelmetHardware", hw_mesh)
    hw.matrix_world = rig.matrix_world.copy()
    hw.data.materials.append(mat_hardware)
    _link_obj(hw, col)

    # 3. Tactical Ballistic Goggles resting on helmet brow (curved visor + frame + elastic band)
    gbm = bmesh.new()
    lbm = bmesh.new()
    g_z = helm_cz + helm_h * 0.24
    g_r_x = rx * 1.08
    g_r_y = ry * 1.08
    # Build curved goggle frame & tinted visor spanning front arc (-52 deg .. +52 deg around -Y)
    g_u, g_v = 18, 4
    f_grid = []
    l_grid = []
    for j in range(g_v + 1):
        vf = (j / g_v) * 2.0 - 1.0  # -1..1
        for i in range(g_u + 1):
            uf = (i / g_u) * 2.0 - 1.0  # -1..1
            ang = uf * math.radians(48.0)
            # Nose notch slightly dips on lower center
            notch = 0.009 * math.exp(-(uf * 4.5) ** 2) if vf < 0 else 0.0
            sx = skull_cx + math.sin(ang) * g_r_x
            sy = skull_cy - math.cos(ang) * g_r_y
            sz = g_z + vf * 0.028 + notch
            f_grid.append(gbm.verts.new((sx, sy - 0.006, sz)))
            if abs(uf) < 0.88 and abs(vf) < 0.78:
                pass
            l_grid.append(lbm.verts.new((sx, sy - 0.011, g_z + vf * 0.022 + notch * 0.7)))
    for j in range(g_v):
        for i in range(g_u):
            idx0 = j * (g_u + 1) + i
            gbm.faces.new((f_grid[idx0], f_grid[idx0 + 1], f_grid[idx0 + g_u + 2], f_grid[idx0 + g_u + 1]))
            lbm.faces.new((l_grid[idx0], l_grid[idx0 + 1], l_grid[idx0 + g_u + 2], l_grid[idx0 + g_u + 1]))
    gbm.normal_update()
    gmesh = bpy.data.meshes.new(f"{name}_GoggleFrame_Mesh")
    gbm.to_mesh(gmesh)
    gbm.free()
    _set_smooth(gmesh)
    gframe = bpy.data.objects.new(f"{name}_GoggleFrame", gmesh)
    gframe.matrix_world = rig.matrix_world.copy()
    gframe.data.materials.append(mat_goggle_frame)
    gsol = gframe.modifiers.new("Solidify", "SOLIDIFY")
    gsol.thickness = 0.014
    _link_obj(gframe, col)

    lbm.normal_update()
    lmesh = bpy.data.meshes.new(f"{name}_GoggleLens_Mesh")
    lbm.to_mesh(lmesh)
    lbm.free()
    _set_smooth(lmesh)
    glens = bpy.data.objects.new(f"{name}_GoggleLens", lmesh)
    glens.matrix_world = rig.matrix_world.copy()
    glens.data.materials.append(mat_goggle_lens)
    lsol = glens.modifiers.new("Solidify", "SOLIDIFY")
    lsol.thickness = 0.004
    _link_obj(glens, col)

    # 4. Elastic goggle strap wrapping helmet + matte chin strap cleanly under jawline
    sbm = bmesh.new()
    # Goggle elastic band wrapping around helmet perimeter
    st_segs = 32
    band_top, band_bot = [], []
    for i in range(st_segs):
        th = (i / st_segs) * math.tau
        px = skull_cx + math.cos(th) * (rx * 1.075)
        py = skull_cy + math.sin(th) * (ry * 1.075)
        band_bot.append(sbm.verts.new((px, py, g_z - 0.012)))
        band_top.append(sbm.verts.new((px, py, g_z + 0.012)))
    for i in range(st_segs):
        ni = (i + 1) % st_segs
        sbm.faces.new((band_bot[i], band_bot[ni], band_top[ni], band_top[i]))

    # Chin strap tucked snugly along the outer jawline (ear -> snug under-chin -> ear; strictly above neck!)
    chin_y = teeth_ctr_local.y + 0.026
    jaw_z_floor = teeth_ctr_local.z - 0.056
    jaw_only = [p for p in jaw_local if p.z >= jaw_z_floor - 0.006 and p.y < skull_cy + 0.025]
    under_center = [
        p for p in jaw_only
        if abs(p.x - skull_cx) < 0.018 and abs(p.y - chin_y) < 0.022 and p.z < teeth_ctr_local.z - 0.022
    ]
    chin_z = max(jaw_z_floor, (min(p.z for p in under_center) - 0.0018) if under_center else (teeth_ctr_local.z - 0.052))
    cs_pts: list[Vector] = []
    cs_n = 22
    for k in range(cs_n + 1):
        u = (k / cs_n) * 2.0 - 1.0  # -1 (right ear) .. 0 (under jaw) .. +1 (left ear)
        au = abs(u)
        px = skull_cx + u * (rx * 0.90)
        pz = chin_z + (helm_cz + 0.004 - chin_z) * (au ** 2.25)
        py = chin_y * (1.0 - au * au) + (skull_cy + 0.014) * (au * au)

        if 0.0 <= au <= 0.85 and jaw_only:
            if au < 0.45:
                near_u = [p for p in jaw_only if abs(p.x - px) < 0.014 and abs(p.y - py) < 0.018 and p.z < teeth_ctr_local.z - 0.018]
                if near_u:
                    pz = max(jaw_z_floor, min(p.z for p in near_u) - 0.0018)
            else:
                sgn = 1.0 if u > 0.0 else -1.0
                near_s = [p for p in jaw_only if (p.x - skull_cx) * sgn > 0.015 and abs(p.y - py) < 0.018 and abs(p.z - pz) < 0.016]
                if near_s:
                    outer_x = max((p.x - skull_cx) * sgn for p in near_s)
                    px = skull_cx + sgn * max(abs(px - skull_cx), outer_x + 0.0026)
        cs_pts.append(Vector((px, py, pz)))

    # 2 smooth passes (keeping endpoints anchored inside helmet rim) for crisp natural webbing drape
    for _ in range(2):
        smoothed = [cs_pts[0]]
        for k in range(1, len(cs_pts) - 1):
            smoothed.append(cs_pts[k - 1] * 0.22 + cs_pts[k] * 0.56 + cs_pts[k + 1] * 0.22)
        smoothed.append(cs_pts[-1])
        cs_pts = smoothed

    r_prev = None
    for k, pt in enumerate(cs_pts):
        v1 = sbm.verts.new((pt.x, pt.y - 0.0042, pt.z))
        v2 = sbm.verts.new((pt.x, pt.y + 0.0042, pt.z))
        if r_prev is not None:
            sbm.faces.new((r_prev[0], v1, v2, r_prev[1]))
        r_prev = (v1, v2)

    sbm.normal_update()
    smesh = bpy.data.meshes.new(f"{name}_HelmetStraps_Mesh")
    sbm.to_mesh(smesh)
    sbm.free()
    _set_smooth(smesh)
    straps = bpy.data.objects.new(f"{name}_HelmetStraps", smesh)
    straps.matrix_world = rig.matrix_world.copy()
    straps.data.materials.append(mat_strap)
    ssol = straps.modifiers.new("Solidify", "SOLIDIFY")
    ssol.thickness = 0.0026
    _link_obj(straps, col)

    parts = [shell, hw, gframe, glens, straps]
    for p in parts:
        _parent_to_bone_preserve_world(p, rig, "head")
    return parts


def _build_plate_carrier_vest(
    name: str,
    actor: dict,
    mat_cordura: bpy.types.Material,
    mat_webbing: bpy.types.Material,
    col: bpy.types.Collection,
) -> list[bpy.types.Object]:
    """Build a snug sculpted Tactical Plate Carrier Vest with slim front/rear SAPI plates, shoulder pads,
    side cummerbund, 3 front rifle magazine pouches & MOLLE straps, parented cleanly to `spine02`!"""
    rig = actor["rig"]
    body = actor["body"]
    bpy.context.view_layer.update()

    deps = bpy.context.evaluated_depsgraph_get()
    es = body.evaluated_get(deps)
    em = es.to_mesh()
    inv_rig = rig.matrix_world.inverted()
    pb_s3 = rig.pose.bones["spine03"]
    pb_s1 = rig.pose.bones["spine01"]
    z_lo = pb_s3.head.z - 0.03
    z_hi = pb_s1.head.z + 0.11
    torso_pts = [
        inv_rig @ (es.matrix_world @ v.co)
        for v in em.vertices
        if abs((inv_rig @ (es.matrix_world @ v.co)).x) < 0.175 and z_lo <= (inv_rig @ (es.matrix_world @ v.co)).z <= z_hi
    ]
    es.to_mesh_clear()

    if torso_pts:
        y_front = min(p.y for p in torso_pts) - 0.010
        raw_back = max(p.y for p in torso_pts) + 0.008
        # Clamp torso front-to-back depth <= 0.23m so back plate hugs spine snugly (zero boxy backpack!)
        y_back = min(raw_back, y_front + 0.235)
        half_w = min(0.172, max(0.140, 0.5 * (max(p.x for p in torso_pts) - min(p.x for p in torso_pts)) + 0.010))
    else:
        y_front, y_back, half_w = -0.140, 0.090, 0.160

    z_bot = z_lo + 0.04
    z_top = z_hi - 0.02
    plate_h = z_top - z_bot
    plate_zc = 0.5 * (z_bot + z_top)
    plate_w = half_w * 1.48

    bm = bmesh.new()

    def add_beveled_plate(cx, cy, cz, sx, sy, sz, rot_x=0.0):
        res = bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=res["verts"])
        if abs(rot_x) > 1e-4:
            bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Euler((math.radians(rot_x), 0, 0)).to_matrix(), verts=res["verts"])
        bmesh.ops.translate(bm, vec=(cx, cy, cz), verts=res["verts"])

    # 1. Slim Contoured Front & Rear Armored SAPI Plates
    add_beveled_plate(0.0, y_front - 0.011, plate_zc, plate_w, 0.024, plate_h, rot_x=-3.5)
    add_beveled_plate(0.0, y_back + 0.010, plate_zc + 0.008, plate_w * 0.94, 0.022, plate_h * 0.96, rot_x=2.5)

    # 2. Padded Tactical Shoulder Straps bridging front & back plates snugly
    y_mid = 0.5 * (y_front + y_back)
    shoulder_depth = (y_back - y_front) + 0.026
    for sgn in (-1.0, 1.0):
        add_beveled_plate(sgn * (half_w * 0.58), y_mid, z_top + 0.020, 0.060, shoulder_depth, 0.024, rot_x=0.0)
        add_beveled_plate(sgn * (half_w * 0.60), y_front - 0.008, z_top - 0.018, 0.056, 0.024, 0.075, rot_x=-6.0)

    # 3. Slim Side Cummerbund Panels wrapping ribs snugly
    for sgn in (-1.0, 1.0):
        add_beveled_plate(sgn * (half_w + 0.008), y_mid, z_bot + plate_h * 0.34, 0.020, shoulder_depth * 0.86, plate_h * 0.42)

    # 4. 3 Front Sculpted Rifle Magazine Pouches across lower chest!
    pouch_w = (plate_w * 0.86) / 3.0 - 0.008
    pouch_h = plate_h * 0.42
    pouch_z = z_bot + pouch_h * 0.56
    for p_i in (-1, 0, 1):
        px = p_i * (pouch_w + 0.009)
        add_beveled_plate(px, y_front - 0.032, pouch_z, pouch_w, 0.028, pouch_h, rot_x=-3.0)
        add_beveled_plate(px, y_front - 0.037, pouch_z + pouch_h * 0.32, pouch_w * 0.88, 0.012, pouch_h * 0.34, rot_x=-5.0)

    # 5. Upper-Left Tactical Comms / Admin Pouch
    add_beveled_plate(half_w * 0.38, y_front - 0.028, z_top - 0.072, 0.064, 0.022, 0.074, rot_x=-4.0)

    bmesh.ops.bevel(bm, geom=list(bm.edges), offset=0.006, segments=2, affect="EDGES")
    bm.normal_update()
    vmesh = bpy.data.meshes.new(f"{name}_PlateCarrier_Mesh")
    bm.to_mesh(vmesh)
    bm.free()
    _set_smooth(vmesh)

    vest = bpy.data.objects.new(f"{name}_PlateCarrier", vmesh)
    vest.matrix_world = rig.matrix_world.copy()
    vest.data.materials.append(mat_cordura)
    _link_obj(vest, col)

    # MOLLE horizontal webbing strips & morale patch panel
    wbm = bmesh.new()
    for strip_i in range(3):
        sz = z_top - 0.045 - strip_i * 0.034
        res = bmesh.ops.create_cube(wbm, size=1.0)
        bmesh.ops.scale(wbm, vec=(plate_w * 0.88, 0.010, 0.016), verts=res["verts"])
        bmesh.ops.translate(wbm, vec=(0.0, y_front - 0.036, sz), verts=res["verts"])
    wbm.normal_update()
    wmesh = bpy.data.meshes.new(f"{name}_VestMOLLE_Mesh")
    wbm.to_mesh(wmesh)
    wbm.free()

    molle = bpy.data.objects.new(f"{name}_VestMOLLE", wmesh)
    molle.matrix_world = rig.matrix_world.copy()
    molle.data.materials.append(mat_webbing)
    _link_obj(molle, col)

    parts = [vest, molle]
    for p in parts:
        _parent_to_bone_preserve_world(p, rig, "spine02")
    return parts


# ============================================================================
# TACTICAL HAND GRIP POSE + GRENADE SOCKET
# ============================================================================
def apply_grenade_grip(rig, side: str = "R", grip_amount: float = 1.0, frame: int | None = None) -> None:
    """Wrap fingers securely around an M67 fragmentation grenade (R_OUTER=42mm) with zero shell poke-through.
    grip_amount=1.0 -> firm tactical grip clamping safety spoon; 0.0 -> open athletic release."""
    side_u = "L" if str(side).upper().startswith("L") else "R"
    g = max(0.0, min(1.0, float(grip_amount)))

    has_finger5 = f"finger5-1.{side_u}" in rig.pose.bones
    digit_indices = (2, 3, 4, 5) if has_finger5 else (1, 2, 3, 4)
    # Calibrated around 84mm M67 shell so fingertips hug the cast-iron curvature without piercing it
    curls_closed = (46.0, 52.0, 56.0, 60.0)
    curls_open = (10.0, 12.0, 15.0, 18.0)

    for d_slot, f_idx in enumerate(digit_indices):
        c_deg = curls_open[d_slot] + (curls_closed[d_slot] - curls_open[d_slot]) * g
        for seg, mult in ((1, 0.78), (2, 1.02), (3, 0.72)):
            bname = f"finger{f_idx}-{seg}.{side_u}"
            if bname in rig.pose.bones:
                pb = rig.pose.bones[bname]
                pb.rotation_mode = "XYZ"
                pb.rotation_euler = Euler((math.radians(c_deg * mult), 0.0, 0.0), "XYZ")
                if frame is not None:
                    pb.keyframe_insert("rotation_euler", frame=frame)

    # Thumb wraps opposing the fingers firmly over the rolled-steel safety spoon
    thm_c = 8.0 + 30.0 * g
    if has_finger5:
        for seg, mult in ((1, 0.42), (2, 0.88), (3, 0.78)):
            bname = f"finger1-{seg}.{side_u}"
            if bname in rig.pose.bones:
                pb = rig.pose.bones[bname]
                pb.rotation_mode = "XYZ"
                pb.rotation_euler = Euler((math.radians(thm_c * mult), 0.0, 0.0), "XYZ")
                if frame is not None:
                    pb.keyframe_insert("rotation_euler", frame=frame)


def _create_hero_grenade_socket(actor: dict, col: bpy.types.Collection) -> bpy.types.Object:
    """Create `Hero_Grenade_Socket` Empty parented cleanly to `wrist.R`, positioned +0.025m forward
    along the distal gripping palm & cupped fingers so the M67 grenade sits snugly inside his closed
    fist and NEVER clips the shirt cuff during Beat 1 pin-pull or Beat 2 throw wind-up."""
    rig = actor["rig"]
    bpy.context.view_layer.update()

    pb_w = rig.pose.bones["wrist.R"]
    pb_f3 = rig.pose.bones.get("finger3-1.R")
    pb_f3_tip = rig.pose.bones.get("finger3-3.R")

    w_tail = rig.matrix_world @ pb_w.tail
    if pb_f3 is not None and pb_f3_tip is not None:
        knuckle_pt = rig.matrix_world @ pb_f3.head
        f3_tip = rig.matrix_world @ pb_f3_tip.tail
        hand_fwd = (knuckle_pt - w_tail).normalized()
        # Inward palm normal perpendicular to hand_fwd pointing toward curled fingertips/thumb
        raw_in = f3_tip - knuckle_pt
        palm_in = (raw_in - hand_fwd * raw_in.dot(hand_fwd)).normalized()
        # Distal palm-cup center (+0.018m past metacarpal knuckles -> ~0.025m further from wrist cuff)
        socket_world = knuckle_pt + hand_fwd * 0.018 + palm_in * 0.038
    else:
        socket_world = w_tail + Vector((0.0, 0.0, -0.05))

    sock = bpy.data.objects.new("Hero_Grenade_Socket", None)
    sock.empty_display_type = "SPHERE"
    sock.empty_display_size = 0.036
    sock.location = socket_world
    sock.rotation_euler = rig.rotation_euler.copy()
    _link_obj(sock, col)
    _parent_to_bone_preserve_world(sock, rig, "wrist.R")
    return sock


# ============================================================================
# SOLDIER ANIMATION CHOREOGRAPHY (BEAT 1 PIN-PULL, BEAT 2 THROW, BEAT 5-6 DIVE!)
# ============================================================================
def _animate_hero_soldier(actor: dict, total_frames: int) -> None:
    """Animate Hero Soldier:
    - Beat 1 (f=1..108): Authentic Action-Movie Teeth Pin-Pull!
      * f=1..24: Right hand brings grenade cleanly up to mouth/lips so the chrome pull-ring sits
        literally at his teeth (`socket_z ≈ teeth_z - 0.058m`, ~0.052m laterally in front-left of lips,
        gentle clamped jaw ~6.0 deg — never gaping!).
      * f=24..50: Unmistakable opposite-direction tug — head snaps decisively back-right while right
        arm pulls firmly forward-left (~0.14m separation!), visually ripping the pin out with his teeth!
      * f=50..92: Hero returns gaze straight ahead with intense determined eyes and lowers grenade
        slightly into hero-inspection stance (~0.14m out from chest/chin) with spoon visibly clamped!
      * Left arm stays cleanly in a low tactical ready/brace posture out of foreground/subtitles!
    - Beat 2 (f=109..148): Athletic wind-up (f=108..122), whip acceleration (f=128), explosive
      overhand throw releasing at EXACT f=134, natural follow-through (f=148)!
    - Beats 3-6 (f=172..720): Tactical crouch behind sandbag parapet watching the enemy bunker."""
    rig = actor["rig"]

    # Helper to apply a complete upper-body + facial pose cleanly at a keyframe
    def pose_hero(
        f: int,
        spine2: tuple[float, float, float],
        spine1: tuple[float, float, float],
        neck: tuple[float, float, float],
        head: tuple[float, float, float],
        ua_r: tuple[float, float, float],
        la_r: tuple[float, float, float],
        ua_l: tuple[float, float, float],
        la_l: tuple[float, float, float],
        grip_r: float = 1.0,
        hand_l: str = "relaxed",
        jaw_deg: float = 1.5,
        brow: float = -0.35,
        smile: float = 0.0,
        eye_yaw: float = 0.0,
        eye_pitch: float = 0.0,
        weight_shift: float = -0.18,
        knee_bend: float = 0.14,
        hip_drop_z: float = 0.0,
    ):
        t = (f - 1) / float(FPS)
        rig.location = Vector((HERO_POS[0], HERO_POS[1], actor["base_z"] - hip_drop_z))
        rig.keyframe_insert("location", frame=f)

        mc.apply_natural_stance(rig, frame=f, weight_shift=weight_shift, breath_phase=t * 1.8, knee_bend=knee_bend)
        mc.set_bone_char_rot(rig, "spine02", *spine2, frame=f)
        mc.set_bone_char_rot(rig, "spine01", *spine1, frame=f)
        mc.set_bone_char_rot(rig, "neck01", *neck, frame=f)
        mc.set_bone_char_rot(rig, "head", *head, frame=f)

        mc.apply_arm_with_clavicle(rig, "R", ua_r, la_r, frame=f)
        mc.apply_arm_with_clavicle(rig, "L", ua_l, la_l, frame=f)

        apply_grenade_grip(rig, "R", grip_amount=grip_r, frame=f)
        mc.apply_hand_gesture(rig, "L", hand_l, frame=f)

        # Clamp jaw <= 10.0 deg per AGENTS.md Section 1 (gentle ring bite ~6 deg, never gaping!)
        mc.apply_facial_acting(
            rig,
            t=t,
            frame=f,
            jaw_open_deg=min(10.0, jaw_deg),
            brow_lift=brow,
            smile=smile,
            blink_period=3.6,
            blink_phase=1.1,
            eye_yaw_deg=eye_yaw,
            eye_pitch_deg=eye_pitch,
            lip_narrow=0.28 if jaw_deg >= 4.5 else 0.0,
        )

    # Arm kinematics in MPFB2 character space:
    # Right arm (-X rest):
    #   +ry lifts -X toward +Z; +rz swings -X toward -Y (forward!); -rz swings -X toward +Y (back!)
    #   For lowerarm01.R: +rz bends elbow inward toward face/chest; -rx lifts forearm up toward mouth!
    # Left arm (+X rest):
    #   +rx/+rz keeps left arm cleanly down & braced by left hip/side (ZERO foreground belly/subtitle clutter!)

    # BEAT 1 (f=1..108): Action-Movie Teeth Pin-Pull Close-Up
    # f=1: Bringing live grenade smoothly up toward lips/mouth (low tactical left arm brace!)
    pose_hero(
        1,
        spine2=(3.0, -1.0, -1.5), spine1=(2.0, 0.0, -1.5),
        neck=(1.5, 0.0, -2.5), head=(2.5, 0.5, -3.5),
        ua_r=(-0.5, -9.0, 31.0), la_r=(-79.5, 12.0, 57.0),
        ua_l=(10.0, -4.0, 14.0), la_l=(-18.0, 0.0, -4.0),
        grip_r=1.0, hand_l="relaxed",
        jaw_deg=1.8, brow=-0.36, eye_yaw=-3.0, eye_pitch=-4.0,
    )
    # f=18: Chrome pull-ring arrives literally at teeth/lips! Head leans gently forward, jaw ready (5.2 deg)
    pose_hero(
        18,
        spine2=(4.2, -1.5, -2.2), spine1=(2.8, 0.0, -1.8),
        neck=(2.8, -0.5, -3.8), head=(4.2, 1.2, -5.2),
        ua_r=(-2.5, -4.0, 43.5), la_r=(-86.0, 12.0, 57.5),
        ua_l=(10.0, -4.0, 14.0), la_l=(-18.0, 0.0, -4.0),
        grip_r=1.0, hand_l="relaxed",
        jaw_deg=5.2, brow=-0.46, eye_yaw=-4.0, eye_pitch=-5.0,
    )
    # f=24: BITE LOCKED! Teeth clamped cleanly around chrome pull-ring (jaw=6.0 deg, socket_z=teeth_z-0.058m!)
    pose_hero(
        24,
        spine2=(4.5, -1.5, -2.5), spine1=(3.0, 0.0, -2.0),
        neck=(3.0, -0.5, -4.0), head=(4.5, 1.5, -5.5),
        ua_r=(-3.0, -3.0, 44.0), la_r=(-86.0, 12.0, 57.5),
        ua_l=(10.0, -4.0, 14.0), la_l=(-18.0, 0.0, -4.0),
        grip_r=1.0, hand_l="relaxed",
        jaw_deg=6.0, brow=-0.54, eye_yaw=-4.0, eye_pitch=-5.5,
    )
    # f=36: MID-PULL TENSION! Opposite-direction tug — head pulls back-right while right arm pulls forward-left!
    pose_hero(
        36,
        spine2=(3.8, -0.5, -0.5), spine1=(2.5, 0.2, -0.5),
        neck=(1.2, 0.4, -0.8), head=(1.5, 0.0, -0.8),
        ua_r=(-3.5, -7.5, 38.0), la_r=(-86.0, 11.0, 58.5),
        ua_l=(11.0, -4.0, 15.0), la_l=(-18.0, 0.0, -4.0),
        grip_r=1.0, hand_l="relaxed",
        jaw_deg=6.2, brow=-0.58, eye_yaw=-1.0, eye_pitch=-3.0,
    )
    # f=50: DECISIVE PIN POP! Head snaps sharply back-right while right arm pulls firmly forward-left (~0.14m gap!)
    pose_hero(
        50,
        spine2=(3.0, 0.5, 1.5), spine1=(2.0, 0.5, 1.0),
        neck=(-0.5, 1.0, 2.5), head=(-1.5, -1.5, 4.0),
        ua_r=(-4.0, -12.0, 32.0), la_r=(-86.0, 10.0, 60.0),
        ua_l=(12.0, -5.0, 16.0), la_l=(-20.0, 0.0, -4.0),
        grip_r=1.0, hand_l="relaxed",
        jaw_deg=3.8, brow=-0.46, eye_yaw=3.5, eye_pitch=-1.0,
    )
    # f=70: HERO INSPECTION STANCE — intense determined eyes straight ahead, grenade ~0.14m out, spoon clamped!
    pose_hero(
        70,
        spine2=(3.0, -0.5, 0.0), spine1=(2.0, 0.0, 0.0),
        neck=(0.5, 0.0, 0.5), head=(0.0, 0.0, 0.5),
        ua_r=(-10.0, -12.0, 34.5), la_r=(-72.5, 8.0, 48.5),
        ua_l=(10.0, -4.0, 14.0), la_l=(-18.0, 0.0, -4.0),
        grip_r=1.0, hand_l="relaxed",
        jaw_deg=1.5, brow=-0.38, eye_yaw=0.0, eye_pitch=0.0,
    )
    # f=92: Rock-steady tactical composure holding live safety spoon clamped down before throw
    pose_hero(
        92,
        spine2=(3.2, -0.6, -0.4), spine1=(2.2, 0.0, -0.2),
        neck=(0.6, 0.0, 0.6), head=(0.0, 0.0, 0.5),
        ua_r=(-10.5, -11.0, 34.0), la_r=(-70.0, 8.0, 46.0),
        ua_l=(10.0, -4.0, 14.0), la_l=(-18.0, 0.0, -4.0),
        grip_r=1.0, hand_l="relaxed",
        jaw_deg=1.4, brow=-0.38, eye_yaw=0.0, eye_pitch=0.0,
    )
    # f=108: Transition into athletic throw stance (left arm stays cleanly braced low-side, never blocking chest!)
    pose_hero(
        108,
        spine2=(3.5, -1.8, -6.0), spine1=(2.5, 0.0, -5.5),
        neck=(1.0, 0.0, 4.0), head=(0.0, 0.0, 6.0),
        ua_r=(-12.0, 14.0, 20.0), la_r=(-48.0, 8.0, 22.0),
        ua_l=(8.0, -6.0, 12.0), la_l=(-22.0, -2.0, -6.0),
        grip_r=1.0, hand_l="relaxed",
        jaw_deg=2.0, brow=-0.42, eye_yaw=2.0, eye_pitch=1.0,
        weight_shift=-0.25, knee_bend=0.18,
    )

    # BEAT 2 (f=109..148): Athletic Wind-Up & Explosive Overhand Throw (EXACT Release at f=134!)
    # f=122: Full athletic wind-up! Torso coiled right, right arm cocked high-back behind right shoulder!
    pose_hero(
        122,
        spine2=(-4.0, -5.0, -22.0), spine1=(-3.0, -3.0, -16.0),
        neck=(2.0, 2.0, 14.0), head=(0.0, 2.0, 18.0),
        ua_r=(-12.0, 56.0, -54.0), la_r=(-42.0, 10.0, 18.0),
        ua_l=(16.0, -4.0, 28.0), la_l=(-16.0, 0.0, -4.0),
        grip_r=1.0, hand_l="relaxed",
        jaw_deg=5.8, brow=-0.55, eye_yaw=6.0, eye_pitch=3.0,
        weight_shift=-0.45, knee_bend=0.26, hip_drop_z=0.04,
    )
    # f=128: Mid-whip acceleration — torso driving forward, elbow leading high!
    pose_hero(
        128,
        spine2=(4.0, 1.0, -4.0), spine1=(4.0, 0.0, -2.0),
        neck=(0.0, 0.0, 3.0), head=(-2.0, 0.0, 4.0),
        ua_r=(-50.0, 56.0, 14.0), la_r=(-28.0, 8.0, 10.0),
        ua_l=(14.0, -5.0, 22.0), la_l=(-18.0, -2.0, -4.0),
        grip_r=0.92, hand_l="relaxed",
        jaw_deg=7.5, brow=-0.60, eye_yaw=2.0, eye_pitch=5.0,
        weight_shift=0.12, knee_bend=0.22, hip_drop_z=0.03,
    )
    # f=134: EXACT GRENADE RELEASE FRAME! Right arm extended high-forward toward enemy bunker (+Y), fingers open!
    pose_hero(
        RELEASE_FRAME,
        spine2=(10.0, 3.0, 10.0), spine1=(6.0, 2.0, 8.0),
        neck=(-4.0, -2.0, -8.0), head=(-5.0, -2.0, -10.0),
        ua_r=(-38.0, 48.0, 52.0), la_r=(-8.0, 0.0, -4.0),
        ua_l=(10.0, -6.0, 18.0), la_l=(-24.0, -2.0, -6.0),
        grip_r=0.0, hand_l="relaxed",
        jaw_deg=7.2, brow=-0.52, eye_yaw=-2.0, eye_pitch=8.0,
        weight_shift=0.40, knee_bend=0.24, hip_drop_z=0.035,
    )
    # f=148: Athletic follow-through — right arm sweeps down across torso, eyes tracking arc!
    pose_hero(
        148,
        spine2=(15.0, 4.0, 18.0), spine1=(9.0, 3.0, 14.0),
        neck=(-5.0, -2.0, -12.0), head=(-6.0, -2.0, -16.0),
        ua_r=(14.0, -8.0, 54.0), la_r=(-26.0, 6.0, 28.0),
        ua_l=(12.0, -6.0, 20.0), la_l=(-26.0, -4.0, -14.0),
        grip_r=0.18, hand_l="relaxed",
        jaw_deg=3.0, brow=-0.38, eye_yaw=-4.0, eye_pitch=10.0,
        weight_shift=0.35, knee_bend=0.30, hip_drop_z=0.055,
    )
    # f=172: Drop smoothly into tactical cover crouch behind sandbag parapet!
    pose_hero(
        172,
        spine2=(18.0, 0.0, 4.0), spine1=(12.0, 0.0, 2.0),
        neck=(-10.0, 0.0, -2.0), head=(-10.0, 0.0, -2.0),
        ua_r=(-14.0, 16.0, 44.0), la_r=(-44.0, 8.0, 36.0),
        ua_l=(-14.0, -16.0, -44.0), la_l=(-44.0, -8.0, -36.0),
        grip_r=0.45, hand_l="relaxed",
        jaw_deg=1.5, brow=-0.35, eye_yaw=0.0, eye_pitch=2.0,
        weight_shift=-0.10, knee_bend=0.58, hip_drop_z=0.18,
    )
    # Hold cover crouch through Beats 3-6 with subtle alive breathing
    for f in (360, 540, 612, total_frames):
        pose_hero(
            f,
            spine2=(18.0, 0.0, 3.0), spine1=(12.0, 0.0, 2.0),
            neck=(-10.0, 0.0, -2.0), head=(-9.0, 0.0, -2.0),
            ua_r=(-14.0, 16.0, 44.0), la_r=(-44.0, 8.0, 36.0),
            ua_l=(-14.0, -16.0, -44.0), la_l=(-44.0, -8.0, -36.0),
            grip_r=0.45, hand_l="relaxed",
            jaw_deg=1.5, brow=-0.35, eye_yaw=0.0, eye_pitch=2.0,
            weight_shift=-0.08, knee_bend=0.58, hip_drop_z=0.18,
        )

    _set_bezier(rig)


def _animate_enemy_sentry(
    actor: dict,
    base_pos: tuple[float, float, float],
    base_yaw_deg: float,
    dive_dir_x: float,
    total_frames: int,
    phase_seed: float = 0.0,
) -> None:
    """Animate Rival Armored Sentry behind enemy sandbag bunker:
    - f=1..495: Tactical alert sentry guard scanning horizon
    - f=496..611 (Beat 5): Spots grenade at (0, 7.25, 0.04) -> shock recoil -> frantic dive behind sandbag wings!
    - f=612..720 (Beat 6): PG-13 Hollywood slow-mo shockwave knockback/dive outward away from blast!"""
    rig = actor["rig"]
    sgn = 1.0 if dive_dir_x >= 0.0 else -1.0  # Enemy_A dives -X, Enemy_B dives +X

    def key_sentry(
        f: int,
        loc_offset: tuple[float, float, float],
        rot_euler_deg: tuple[float, float, float],
        spine2: tuple[float, float, float],
        spine1: tuple[float, float, float],
        head: tuple[float, float, float],
        ua_r: tuple[float, float, float],
        la_r: tuple[float, float, float],
        ua_l: tuple[float, float, float],
        la_l: tuple[float, float, float],
        hand_pose: str = "relaxed",
        jaw_deg: float = 1.2,
        brow: float = -0.20,
        knee_bend: float = 0.16,
    ):
        t = (f - 1) / float(FPS)
        rig.location = Vector((
            base_pos[0] + loc_offset[0],
            base_pos[1] + loc_offset[1],
            actor["base_z"] + loc_offset[2],
        ))
        rig.rotation_euler = Euler((
            math.radians(rot_euler_deg[0]),
            math.radians(rot_euler_deg[1]),
            math.radians(base_yaw_deg + rot_euler_deg[2]),
        ), "XYZ")
        rig.keyframe_insert("location", frame=f)
        rig.keyframe_insert("rotation_euler", frame=f)

        mc.apply_natural_stance(rig, frame=f, weight_shift=0.14 * sgn, breath_phase=t * 1.6 + phase_seed, knee_bend=knee_bend)
        mc.set_bone_char_rot(rig, "spine02", *spine2, frame=f)
        mc.set_bone_char_rot(rig, "spine01", *spine1, frame=f)
        mc.set_bone_char_rot(rig, "head", *head, frame=f)
        mc.apply_arm_with_clavicle(rig, "R", ua_r, la_r, frame=f)
        mc.apply_arm_with_clavicle(rig, "L", ua_l, la_l, frame=f)
        mc.apply_hand_gesture(rig, "R", hand_pose, frame=f)
        mc.apply_hand_gesture(rig, "L", hand_pose, frame=f)

        mc.apply_facial_acting(
            rig,
            t=t,
            frame=f,
            jaw_open_deg=min(10.5, jaw_deg),
            brow_lift=brow,
            smile=0.0,
            blink_period=3.9,
            blink_phase=0.5 + phase_seed,
            eye_yaw_deg=0.0,
            eye_pitch_deg=-10.0 if f >= LANDING_FRAME else 0.0,
        )

    # f=1..495: Guarding sandbag bunker in alert stance (subtle scanning)
    for f, h_yaw in [(1, -8.0 * sgn), (220, 6.0 * sgn), (460, -5.0 * sgn), (495, -4.0 * sgn)]:
        key_sentry(
            f,
            loc_offset=(0.0, 0.0, 0.0),
            rot_euler_deg=(0.0, 0.0, 0.0),
            spine2=(6.0, 0.0, 0.0),
            spine1=(4.0, 0.0, 0.0),
            head=(2.0, 0.0, h_yaw),
            ua_r=(-14.0, 12.0, 42.0), la_r=(-36.0, 6.0, 34.0),
            ua_l=(-14.0, -12.0, -42.0), la_l=(-36.0, -6.0, -34.0),
            hand_pose="relaxed",
            jaw_deg=1.0,
            brow=-0.22,
            knee_bend=0.14,
        )

    # f=512: SHOCK! Notices live grenade landing right in front at (0, 7.25, 0.04)!
    key_sentry(
        512,
        loc_offset=(0.06 * sgn, 0.14, 0.0),
        rot_euler_deg=(0.0, 0.0, -8.0 * sgn),
        spine2=(-8.0, -4.0 * sgn, -6.0 * sgn),
        spine1=(-6.0, 0.0, -4.0 * sgn),
        head=(22.0, 0.0, -14.0 * sgn),  # Looks sharply down at grenade!
        ua_r=(-34.0, 26.0, 48.0), la_r=(-46.0, 8.0, 26.0),
        ua_l=(-34.0, -26.0, -48.0), la_l=(-46.0, -8.0, -26.0),
        hand_pose="shock",
        jaw_deg=9.8,
        brow=0.68,
        knee_bend=0.26,
    )

    # f=540 (Visual verification frame!): Recoiling & diving sideways-back for cover behind sandbags!
    key_sentry(
        540,
        loc_offset=(0.34 * sgn, 0.36, -0.14),
        rot_euler_deg=(8.0, -12.0 * sgn, -28.0 * sgn),
        spine2=(22.0, -8.0 * sgn, -16.0 * sgn),
        spine1=(14.0, -4.0 * sgn, -10.0 * sgn),
        head=(18.0, 4.0 * sgn, -18.0 * sgn),
        ua_r=(-46.0, 36.0, 56.0), la_r=(-58.0, 12.0, 48.0),
        ua_l=(-46.0, -36.0, -56.0), la_l=(-58.0, -12.0, -48.0),
        hand_pose="shock",
        jaw_deg=9.2,
        brow=0.62,
        knee_bend=0.56,
    )

    # f=611: Full tactical cover duck shielding helmet right before detonation!
    key_sentry(
        611,
        loc_offset=(0.62 * sgn, 0.56, -0.26),
        rot_euler_deg=(14.0, -18.0 * sgn, -42.0 * sgn),
        spine2=(30.0, -8.0 * sgn, -18.0 * sgn),
        spine1=(22.0, 0.0, -12.0 * sgn),
        head=(14.0, 0.0, -10.0 * sgn),
        ua_r=(-52.0, 42.0, 62.0), la_r=(-66.0, 14.0, 58.0),
        ua_l=(-52.0, -42.0, -62.0), la_l=(-66.0, -14.0, -58.0),
        hand_pose="shock",
        jaw_deg=7.5,
        brow=0.52,
        knee_bend=0.72,
    )

    # f=612..720 (Beat 6): PG-13 Slow-Mo Action-Movie Shockwave Knockback Dive Outward!
    for f, frac in [(624, 0.34), (656, 0.72), (total_frames, 1.0)]:
        arch_z = math.sin(frac * math.pi) * 0.42
        key_sentry(
            f,
            loc_offset=(
                (0.62 + 1.08 * frac) * sgn,
                0.56 + 1.38 * frac,
                -0.22 + arch_z - 0.14 * frac,
            ),
            rot_euler_deg=(
                14.0 + 38.0 * frac,
                (-18.0 - 28.0 * frac) * sgn,
                (-42.0 - 34.0 * frac) * sgn,
            ),
            spine2=(14.0 - 12.0 * frac, -10.0 * sgn, -16.0 * sgn),
            spine1=(8.0 - 10.0 * frac, 0.0, -12.0 * sgn),
            head=(-6.0, 6.0 * sgn, -12.0 * sgn),
            ua_r=(-38.0, 52.0, 20.0), la_r=(-24.0, 8.0, 18.0),
            ua_l=(-38.0, -52.0, -20.0), la_l=(-24.0, -8.0, -18.0),
            hand_pose="shock",
            jaw_deg=9.5,
            brow=0.65,
            knee_bend=0.45,
        )

    _set_bezier(rig)


def _cinch_shirt_cuffs(actor: dict) -> None:
    """Taper the loose flared boho cuff of `elvs_male_boho_top1` around both wrists into a crisp,
    fitted military BDU combat-shirt cuff so the grenade NEVER clips the sleeve during Beat 1 or Beat 2."""
    shirt_obj = actor["equipped"].get("Shirt")
    rig = actor["rig"]
    if shirt_obj is None or shirt_obj.type != "MESH":
        return
    bpy.context.view_layer.update()
    rig_from_shirt = rig.matrix_world.inverted() @ shirt_obj.matrix_world
    shirt_from_rig = rig_from_shirt.inverted()
    mesh = shirt_obj.data

    for side in ("R", "L"):
        pb_w = rig.pose.bones.get(f"wrist.{side}")
        pb_la2 = rig.pose.bones.get(f"lowerarm02.{side}")
        if pb_w is None or pb_la2 is None:
            continue
        w_head = pb_w.head.copy()
        arm_dir = (w_head - pb_la2.head).normalized()
        for v in mesh.vertices:
            p_rig = rig_from_shirt @ v.co
            rel = p_rig - w_head
            proj = rel.dot(arm_dir)
            if -0.115 < proj < 0.055:
                rad_vec = rel - arm_dir * proj
                rad = rad_vec.length
                if rad < 0.135 and rad > 1e-5:
                    u = max(0.0, min(1.0, (proj + 0.115) / 0.095))
                    w = u * u * (3.0 - 2.0 * u)
                    max_r = 0.041
                    new_rad = (rad * (1.0 - w) + max_r * w) if rad > max_r else rad
                    new_proj = (proj * (1.0 - 0.85 * w) + (-0.015) * (0.85 * w)) if proj > -0.015 else proj
                    p_new = w_head + arm_dir * new_proj + rad_vec * (new_rad / rad)
                    v.co = shirt_from_rig @ p_new
    mesh.update()


def _find_front_teeth_vertex_ids(actor: dict) -> list[int]:
    """Find vertex indices of the central front incisors on `teeth_base` in rest pose so
    evaluated mouth/teeth world coordinates remain 100% invariant across all head rotations."""
    teeth_obj = actor["equipped"].get("Teeth")
    rig = actor["rig"]
    if teeth_obj is None:
        return []
    bpy.context.view_layer.update()
    inv_rig = rig.matrix_world.inverted()
    deps = bpy.context.evaluated_depsgraph_get()
    et = teeth_obj.evaluated_get(deps)
    tm = et.to_mesh()
    t_local = [(i, inv_rig @ (et.matrix_world @ v.co)) for i, v in enumerate(tm.vertices)]
    et.to_mesh_clear()
    if not t_local:
        return []
    central = [it for it in t_local if abs(it[1].x) < 0.014]
    if len(central) < 8:
        central = t_local
    central.sort(key=lambda it: it[1].y)  # -Y is front in MPFB2 local space
    return [it[0] for it in central[:16]]


def _sample_socket_trajectory(sock: bpy.types.Object, start_f: int = 1, end_f: int = RELEASE_FRAME) -> tuple[dict[int, tuple[float, float, float]], dict[int, tuple[float, float, float, float]]]:
    """Evaluate exact world-space position & quaternion of `Hero_Grenade_Socket` across frames `start_f..end_f`."""
    scene = bpy.context.scene
    prev_f = scene.frame_current
    positions: dict[int, tuple[float, float, float]] = {}
    quats: dict[int, tuple[float, float, float, float]] = {}

    for f in range(start_f, end_f + 1):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        mw = sock.matrix_world
        loc = mw.translation
        q = mw.to_quaternion()
        positions[f] = (round(float(loc.x), 5), round(float(loc.y), 5), round(float(loc.z), 5))
        quats[f] = (round(float(q.w), 5), round(float(q.x), 5), round(float(q.y), 5), round(float(q.z), 5))

    scene.frame_set(prev_f)
    bpy.context.view_layer.update()
    return positions, quats


def _sample_hero_mouth_trajectory(hero: dict, start_f: int = 1, end_f: int = 108) -> dict[int, Vector]:
    """Evaluate exact world-space front-center of Hero's `teeth_base` across frames `start_f..end_f`
    so the master scene can keep the grenade's chrome pull-ring locked literally at his teeth!"""
    scene = bpy.context.scene
    prev_f = scene.frame_current
    teeth_obj = hero["equipped"].get("Teeth")
    front_vids: list[int] = hero.get("front_teeth_vids", [])
    mouth_positions: dict[int, Vector] = {}

    for f in range(start_f, end_f + 1):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        if teeth_obj is not None:
            deps = bpy.context.evaluated_depsgraph_get()
            et = teeth_obj.evaluated_get(deps)
            tm = et.to_mesh()
            if front_vids and max(front_vids) < len(tm.vertices):
                pts = [et.matrix_world @ tm.vertices[idx].co for idx in front_vids]
            else:
                pts = [et.matrix_world @ v.co for v in tm.vertices]
            et.to_mesh_clear()
            if pts:
                ctr = sum(pts, Vector((0.0, 0.0, 0.0))) / len(pts)
                mouth_positions[f] = Vector((round(float(ctr.x), 5), round(float(ctr.y), 5), round(float(ctr.z), 5)))
                continue
        mouth_positions[f] = Vector((0.09, 0.16, 1.59))

    scene.frame_set(prev_f)
    bpy.context.view_layer.update()
    return mouth_positions


def _build_single_soldier(
    name: str,
    pos: tuple[float, float, float],
    yaw_deg: float,
    height_m: float,
    is_hero: bool,
    col: bpy.types.Collection,
    gear_mats: dict,
) -> dict:
    """Create one photoreal 4K MakeHuman MPFB2 tactical soldier + helmet + plate carrier vest."""
    if is_hero:
        cfg = {
            "age": 0.54,  # Adult combat specialist (per MPFB2 guard >= 0.54!)
            "gender": 1.0,
            "muscle": 0.74,
            "weight": 0.50,
            "height": 0.62,
            "target_height_m": height_m,
            "skin_assets": ["middleage_caucasian_male.mhmat"],
            "iris_major_rgb": (0.28, 0.52, 0.66),
            "iris_minor_rgb": (0.12, 0.24, 0.34),
            "eyebrow_assets": ["eyebrow002.mhclo"],
            "hair_assets": ["short04.mhclo"],
            "hair_tint_rgb": (0.24, 0.17, 0.12),
            "hair_tint_fac": 0.65,
            "brow_tint_rgb": (0.20, 0.14, 0.10),
            "brow_tint_fac": 0.60,
            # Dusty olive-tan desert combat uniform
            "pants_assets": ["mindfront_male_trousers_2.mhclo"],
            "pants_sat_override": 0.32,
            "pants_gamma_lift": 0.78,
            "pants_value_boost": 1.12,
            "pants_tint_rgb": (0.64, 0.56, 0.40),
            "pants_tint_fac": 0.80,
            "pants_roughness": 0.84,
            "shirt_assets": ["elvs_male_boho_top1.mhclo"],
            "shirt_sat_override": 0.25,
            "shirt_gamma_lift": 0.76,
            "shirt_value_boost": 0.96,
            "shirt_tint_rgb": (0.44, 0.48, 0.33),
            "shirt_tint_fac": 0.85,
            "shirt_roughness": 0.82,
            "shoe_assets": ["toigo_ankle_boots_male.mhclo"],
            "shoe_sat_override": 0.30,
            "shoe_tint_rgb": (0.38, 0.30, 0.21),
            "shoe_tint_fac": 0.78,
            "shoe_roughness": 0.72,
        }
    else:
        cfg = {
            "age": 0.56,
            "gender": 1.0,
            "muscle": 0.68,
            "weight": 0.52,
            "height": 0.60,
            "target_height_m": height_m,
            "skin_assets": ["middleage_caucasian_male.mhmat"],
            "iris_major_rgb": (0.36, 0.26, 0.16),
            "iris_minor_rgb": (0.16, 0.11, 0.07),
            "eyebrow_assets": ["eyebrow003.mhclo", "eyebrow002.mhclo"],
            "hair_assets": ["short04.mhclo"],
            "hair_tint_rgb": (0.14, 0.12, 0.10),
            "hair_tint_fac": 0.75,
            "brow_tint_rgb": (0.12, 0.10, 0.09),
            "brow_tint_fac": 0.70,
            # Dark slate-charcoal rival tactical uniform
            "pants_assets": ["mindfront_male_trousers_2.mhclo"],
            "pants_sat_override": 0.15,
            "pants_value_boost": 0.52,
            "pants_tint_rgb": (0.24, 0.27, 0.32),
            "pants_tint_fac": 0.85,
            "pants_roughness": 0.82,
            "shirt_assets": ["elvs_male_boho_top1.mhclo"],
            "shirt_sat_override": 0.15,
            "shirt_value_boost": 0.46,
            "shirt_tint_rgb": (0.20, 0.23, 0.28),
            "shirt_tint_fac": 0.88,
            "shirt_roughness": 0.80,
            "shoe_assets": ["toigo_ankle_boots_male.mhclo"],
            "shoe_sat_override": 0.10,
            "shoe_value_boost": 0.38,
            "shoe_tint_rgb": (0.14, 0.15, 0.16),
            "shoe_tint_fac": 0.82,
            "shoe_roughness": 0.56,
        }

    actor = mc.create_realistic_scientist(
        name=name,
        floor_z=pos[2],
        start_xy=(pos[0], pos[1]),
        yaw_deg=yaw_deg,
        cfg=cfg,
    )
    # Enforce exact evaluated stature & ground contact
    _normalize_actor_height_and_ground(actor, target_height_m=height_m, floor_z=pos[2])

    # Double-check AGENTS.md Section 1 shadow flags on eyes & lashes
    for part_key in ("Eyes", "Eyelashes"):
        p_obj = actor["equipped"].get(part_key)
        if p_obj is not None:
            p_obj.visible_shadow = False

    # Replace boho floral shirt texture with authentic military ripstop combat-shirt fabric (preserving 3D folds!)
    shirt_obj = actor["equipped"].get("Shirt")
    if shirt_obj is not None:
        shirt_mat = gear_mats["hero_shirt"] if is_hero else gear_mats["enemy_shirt"]
        shirt_obj.data.materials.clear()
        shirt_obj.data.materials.append(shirt_mat)
        # Cinch loose boho bell-cuffs around wrists into snug BDU tactical cuffs (zero grenade clipping!)
        _cinch_shirt_cuffs(actor)

    # Record front-center incisor vertex IDs in rest pose for exact mouth/teeth world tracking
    actor["front_teeth_vids"] = _find_front_teeth_vertex_ids(actor)

    # Move MPFB2 objects cleanly into target collection
    mpfb_objs = [actor["rig"], actor["body"], *actor["equipped"].values()]
    _move_hierarchy_to_col(mpfb_objs, col)

    # Equip sculpted MICH/FAST Tactical Combat Helmet (parented to `head` bone!)
    helmet_parts = _build_tactical_helmet(
        name=name,
        actor=actor,
        mat_shell=gear_mats["hero_helmet"] if is_hero else gear_mats["enemy_helmet"],
        mat_hardware=gear_mats["hardware"],
        mat_goggle_frame=gear_mats["goggle_frame"],
        mat_goggle_lens=gear_mats["goggle_lens"],
        mat_strap=gear_mats["strap"],
        col=col,
    )
    # Equip sculpted Tactical Plate Carrier Vest (parented to `spine02` bone!)
    vest_parts = _build_plate_carrier_vest(
        name=name,
        actor=actor,
        mat_cordura=gear_mats["hero_vest"] if is_hero else gear_mats["enemy_vest"],
        mat_webbing=gear_mats["webbing"],
        col=col,
    )
    actor["helmet_parts"] = helmet_parts
    actor["vest_parts"] = vest_parts
    return actor


def build_soldiers(col: bpy.types.Collection | None = None, total_frames: int = TOTAL_FRAMES_DEFAULT) -> dict:
    """Build real 4K MakeHuman MPFB2 soldiers:
    - Hero_Soldier at (0,0,0), 1.80m, yaw=162 deg (+18 deg toward +X for 3/4 movie close-up!),
      with MICH/FAST helmet, plate carrier vest, Beat 1 teeth pin-pull + Beat 2 overhand throw (release f=134),
      exact `Hero_Grenade_Socket` Empty parented to `wrist.R` + pre-baked world coordinates f=1..134,
      and exact `hero_mouth_world_positions` (evaluated `teeth_base` front-center) for f=1..108!
    - Enemy_A at (-0.78, 8.55, 0) & Enemy_B at (+0.82, 8.65, 0), dark slate-charcoal armor,
      guarding bunker (f=1..495), recoiling/diving (f=496..611), PG-13 slow-mo shockwave knockback (f=612..720)."""
    target_col = col if col is not None else bpy.context.scene.collection

    gear_mats = {
        "hero_helmet": _make_tactical_gear_mat("Mat_HeroHelmet_Olive", (0.075, 0.095, 0.055), roughness=0.72, is_cordura=False),
        "enemy_helmet": _make_tactical_gear_mat("Mat_EnemyHelmet_Slate", (0.032, 0.036, 0.044), roughness=0.62, is_cordura=False),
        "hero_vest": _make_tactical_gear_mat("Mat_HeroVest_OliveTan", (0.12, 0.135, 0.08), roughness=0.86, is_cordura=True),
        "enemy_vest": _make_tactical_gear_mat("Mat_EnemyVest_Charcoal", (0.035, 0.040, 0.048), roughness=0.82, is_cordura=True),
        "hero_shirt": _make_tactical_gear_mat("Mat_HeroShirt_Ripstop", (0.17, 0.16, 0.105), roughness=0.88, is_cordura=True),
        "enemy_shirt": _make_tactical_gear_mat("Mat_EnemyShirt_Ripstop", (0.055, 0.060, 0.072), roughness=0.86, is_cordura=True),
        "hardware": _make_tactical_metal_mat("Mat_HelmetHardware", (0.04, 0.045, 0.05), metallic=0.65, roughness=0.42),
        "goggle_frame": _make_tactical_gear_mat("Mat_GoggleFrame", (0.035, 0.035, 0.035), roughness=0.66, is_cordura=False),
        "goggle_lens": _make_goggle_lens_mat(),
        "strap": _make_tactical_gear_mat("Mat_TacticalStrap", (0.055, 0.055, 0.045), roughness=0.88, is_cordura=True),
        "webbing": _make_tactical_gear_mat("Mat_VestWebbing", (0.065, 0.075, 0.048), roughness=0.90, is_cordura=True),
    }

    for mat in gear_mats.values():
        mat.use_fake_user = True

    hero = _build_single_soldier("Hero_Soldier", HERO_POS, HERO_YAW_DEG, HERO_HEIGHT_M, True, target_col, gear_mats)
    enemy_a = _build_single_soldier("Enemy_A", ENEMY_A_POS, ENEMY_A_YAW_DEG, 1.78, False, target_col, gear_mats)
    enemy_b = _build_single_soldier("Enemy_B", ENEMY_B_POS, ENEMY_B_YAW_DEG, 1.81, False, target_col, gear_mats)

    # Create exact Hero_Grenade_Socket Empty parented to wrist.R BEFORE animation so bone inverse is exact!
    grenade_socket = _create_hero_grenade_socket(hero, target_col)

    # Animate Hero & both Rival Sentries
    _animate_hero_soldier(hero, total_frames)
    _animate_enemy_sentry(enemy_a, ENEMY_A_POS, ENEMY_A_YAW_DEG, dive_dir_x=-1.0, total_frames=total_frames, phase_seed=0.4)
    _animate_enemy_sentry(enemy_b, ENEMY_B_POS, ENEMY_B_YAW_DEG, dive_dir_x=1.0, total_frames=total_frames, phase_seed=1.7)

    # Re-aim Hero_EyeCatchlight to Hero's exact evaluated eye center if present in scene!
    catch = bpy.data.objects.get("Hero_EyeCatchlight")
    if catch is not None:
        eye_pt = hero["eye_center_rest"]
        catch.rotation_euler = _aim_euler(catch.location, eye_pt)

    # Sample exact world positions & rotations of Hero_Grenade_Socket (f=1..134) & teeth_base front-center (f=1..108)!
    socket_positions, socket_quats = _sample_socket_trajectory(grenade_socket, 1, RELEASE_FRAME)
    mouth_positions = _sample_hero_mouth_trajectory(hero, 1, 108)

    return {
        "hero": hero,
        "enemy_a": enemy_a,
        "enemy_b": enemy_b,
        "hero_grenade_socket": grenade_socket,
        "hero_socket_world_positions": socket_positions,
        "hero_socket_world_quats": socket_quats,
        "hero_mouth_world_positions": mouth_positions,
        "release_frame": RELEASE_FRAME,
        "release_pos": socket_positions[RELEASE_FRAME],
        "grenade_landing_pos": GRENADE_LANDING_POS,
    }


# ============================================================================
# STANDALONE VISUAL VERIFICATION SUITE (renders hero_closeup f=30 & enemy_bunker f=540)
# ============================================================================
def _configure_preview_cycles(samples: int = 6, res_x: int = 1080, res_y: int = 1920, scale: int = 35) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 6
    scene.cycles.transparent_max_bounces = 12
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False

    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.resolution_percentage = scale
    scene.render.image_settings.file_format = "PNG"

    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Punchy"
    except Exception:
        pass
    scene.view_settings.exposure = -0.55


def _create_dummy_grenade_proxy(socket_obj: bpy.types.Object, col: bpy.types.Collection) -> bpy.types.Object:
    """Small olive-steel grenade proxy ONLY for standalone visual verification of hand-to-mouth framing."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=20, v_segments=14, radius=0.033)
    bmesh.ops.scale(bm, vec=(1.0, 1.0, 1.25), verts=bm.verts)
    mesh = bpy.data.meshes.new("Preview_Grenade_Proxy_Mesh")
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    _set_smooth(mesh)
    g = bpy.data.objects.new("Preview_Grenade_Proxy", mesh)
    g.parent = socket_obj
    g.location = Vector((0.0, 0.0, 0.0))
    g.data.materials.append(_make_tactical_gear_mat("Mat_PreviewGrenade", (0.09, 0.13, 0.06), roughness=0.42))
    _link_obj(g, col)
    return g


def _run_standalone_verification() -> None:
    out_dir = Path("/tmp/battlefield_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = TOTAL_FRAMES_DEFAULT
    scene.render.fps = FPS
    _configure_preview_cycles(samples=6, res_x=1080, res_y=1920, scale=35)

    env_col = bpy.data.collections.new("Battlefield_Env")
    scene.collection.children.link(env_col)
    sol_col = bpy.data.collections.new("Battlefield_Soldiers")
    scene.collection.children.link(sol_col)

    bf = build_battlefield(env_col, total_frames=TOTAL_FRAMES_DEFAULT)
    sols = build_soldiers(sol_col, total_frames=TOTAL_FRAMES_DEFAULT)

    # Preview-only grenade proxy in Hero's socket so we can visually verify hand/grenade near mouth!
    _create_dummy_grenade_proxy(sols["hero_grenade_socket"], sol_col)

    # Diagnostic readout of exact Hero socket & mouth distance at f=30 and release f=134!
    scene.frame_set(30)
    bpy.context.view_layer.update()
    sock_30 = sols["hero_grenade_socket"].matrix_world.translation.copy()
    _, _, teeth_30 = _eval_mesh_world_bounds(sols["hero"]["equipped"]["Teeth"])
    _, _, eye_30 = _eval_mesh_world_bounds(sols["hero"]["equipped"]["Eyes"])
    dist_mouth_30 = (sock_30 - teeth_30).length
    print(f"[VERIFY f=30] Hero Eye={tuple(round(c, 3) for c in eye_30)} Teeth={tuple(round(c, 3) for c in teeth_30)}")
    print(f"[VERIFY f=30] Hero_Grenade_Socket={tuple(round(c, 3) for c in sock_30)} | dist_to_teeth={dist_mouth_30:.3f}m")
    for kf in (1, 24, 30, 34, 46, 54, 108, 122, 128, 134):
        print(f"  [SOCKET f={kf:3d}] {sols['hero_socket_world_positions'][kf]}")
    print(f"[VERIFY] Grenade landing pos = {bf['grenade_landing_pos']} | Explosion origin = {bf['explosion_origin']}")

    # Create preview camera
    cam_data = bpy.data.cameras.new("PreviewCam_Data")
    cam_data.lens = 50.0
    cam_obj = bpy.data.objects.new("PreviewCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # 1) Hero Close-Up at f=30 (heroic 3/4 movie close-up from front-right!)
    scene.frame_set(30)
    bpy.context.view_layer.update()
    cam_data.lens = 58.0
    cam_loc_1 = Vector((0.44, 1.02, 1.62))
    cam_tgt_1 = Vector((0.04, 0.08, 1.59))
    cam_obj.location = cam_loc_1
    cam_obj.rotation_euler = _aim_euler(cam_loc_1, cam_tgt_1)
    scene.render.filepath = str(out_dir / "hero_closeup.png")
    bpy.ops.render.render(write_still=True)
    print(f"[VERIFY] Rendered {out_dir / 'hero_closeup.png'}")

    # 2) Enemy Sandbag Bunker at f=540 (Beat 5 diving sentries + dramatic bunker + canyon mesas!)
    scene.frame_set(540)
    bpy.context.view_layer.update()
    cam_data.lens = 35.0
    cam_loc_2 = Vector((1.45, 4.45, 1.68))
    cam_tgt_2 = Vector((0.0, 8.35, 1.05))
    cam_obj.location = cam_loc_2
    cam_obj.rotation_euler = _aim_euler(cam_loc_2, cam_tgt_2)
    scene.render.filepath = str(out_dir / "enemy_bunker.png")
    bpy.ops.render.render(write_still=True)
    print(f"[VERIFY] Rendered {out_dir / 'enemy_bunker.png'}")


if __name__ == "__main__":
    _run_standalone_verification()
