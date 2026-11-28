"""
tank_env.py — Crystal Ballistic Tank, Oceanic/Blueprint Dual-Mode Studio & Lighting
Episode #2: "Why Bullets Stop Dead Just 3 Feet Underwater" (60 fps, 1800 frames).

Follows AGENTS.md V12 & Blender 5.3.0 rules:
- Zero `action.fcurves` direct access (uses `C.iter_action_fcurves`).
- All `ShaderNodeMix` inputs indexed strictly from `node.inputs[2:]`.
- `mesh.materials` attached strictly BEFORE `bm.to_mesh(mesh)`.
- `Water_Surface` routes `LightPath.Is Shadow Ray -> Transparent BSDF` (zero volume fog).
- `cycles.transparent_max_bounces >= 16`.

Public API:
    build_tank_env(col: bpy.types.Collection, total_frames: int = C.TOTAL_FRAMES) -> dict[str, Any]
    Returns: {"tank_root": obj, "water_surface": obj, "ruler": obj, "bp_nodes": list[Node], "lights": list[Obj]}
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

_THIS_FILE = Path(__file__).resolve()
_EPISODE_SRC = str(_THIS_FILE.parent)
_EPISODE_DIR = _THIS_FILE.parents[1]
if _EPISODE_SRC not in sys.path:
    sys.path.insert(0, _EPISODE_SRC)

import bullet_contract as C  # noqa: E402

FONT_PATH = _EPISODE_DIR / "subtitles" / "fonts" / "DejaVuSans-Bold.ttf"


# ============================================================================
# BLENDER 5.3.0 SAFE COLLECTION, BMESH, NODE & ANIMATION HELPERS
# ============================================================================
def _link_obj(obj: bpy.types.Object, col: bpy.types.Collection | None = None) -> bpy.types.Object:
    target = col if col is not None else bpy.context.scene.collection
    for c in list(obj.users_collection):
        if c != target:
            c.objects.unlink(obj)
    if obj.name not in target.objects:
        target.objects.link(obj)
    return obj


def _set_bsdf(bsdf: bpy.types.Node, name: str, val: Any) -> None:
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = val


def _mix_inputs_from_2(mix_node: bpy.types.Node, socket_type: str) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """Blender 5.3.0 safe ShaderNodeMix A/B input lookup skipping inputs[0..1] Factor sockets."""
    norm_t = "VALUE" if socket_type == "FLOAT" else socket_type
    typed = [s for s in mix_node.inputs[2:] if s.type == norm_t]
    if len(typed) < 2:
        raise RuntimeError(f"ShaderNodeMix {mix_node.name} missing 2 '{socket_type}' sockets in inputs[2:]")
    return typed[0], typed[1]


def _mix_output(mix_node: bpy.types.Node, socket_type: str) -> bpy.types.NodeSocket:
    norm_t = "VALUE" if socket_type == "FLOAT" else socket_type
    typed = [s for s in mix_node.outputs if s.type == norm_t]
    if not typed:
        raise RuntimeError(f"ShaderNodeMix {mix_node.name} missing '{socket_type}' output socket")
    return typed[0]


def _aim_euler(src: tuple[float, float, float], dst: tuple[float, float, float]) -> Euler:
    d = Vector(dst) - Vector(src)
    if d.length < 1e-6:
        return Euler((0.0, 0.0, 0.0), "XYZ")
    return d.normalized().to_track_quat("-Z", "Y").to_euler("XYZ")


def _set_fcurve_interp(anim_target: Any, interp: str = "LINEAR", prefix: str | None = None) -> None:
    ad = getattr(anim_target, "animation_data", None)
    if not ad or not ad.action:
        return
    for fc in C.iter_action_fcurves(ad.action):
        if prefix is not None and not fc.data_path.startswith(prefix):
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = interp


def _finalize_bmesh(
    bm: bmesh.types.BMesh,
    name: str,
    col: bpy.types.Collection,
    materials: list[bpy.types.Material] | None = None,
    smooth: bool = True,
    recalc_normals: bool = True,
    parent: bpy.types.Object | None = None,
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    visible_shadow: bool = True,
) -> bpy.types.Object:
    """Attach materials BEFORE bm.to_mesh(mesh) per AGENTS.md V12 Sec 3."""
    if recalc_normals and len(bm.faces) > 0:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    if materials:
        for mat in materials:
            mesh.materials.append(mat)

    bm.to_mesh(mesh)
    bm.free()

    if len(mesh.polygons) > 0:
        mesh.polygons.foreach_set("use_smooth", [smooth] * len(mesh.polygons))
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(location)
    obj.rotation_euler = Euler(rotation, "XYZ")
    obj.visible_shadow = visible_shadow
    if parent is not None:
        obj.parent = parent
    _link_obj(obj, col)
    return obj


def _add_box(
    bm: bmesh.types.BMesh,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    mat_index: int = 0,
    rot_euler: tuple[float, float, float] | None = None,
    bevel_width: float = 0.0,
) -> None:
    tmp = bmesh.new()
    bmesh.ops.create_cube(tmp, size=1.0)
    bmesh.ops.scale(tmp, vec=Vector(size), verts=tmp.verts)
    if bevel_width > 0.0:
        bmesh.ops.bevel(
            tmp,
            geom=list(tmp.edges),
            offset=bevel_width,
            segments=2,
            affect="EDGES",
            clamp_overlap=True,
        )
    if rot_euler is not None:
        bmesh.ops.rotate(tmp, matrix=Euler(rot_euler, "XYZ").to_matrix(), verts=tmp.verts)
    bmesh.ops.translate(tmp, vec=Vector(center), verts=tmp.verts)

    vert_map = {v: bm.verts.new(v.co) for v in tmp.verts}
    for f in tmp.faces:
        nf = bm.faces.new([vert_map[v] for v in f.verts])
        nf.material_index = mat_index
    tmp.free()


def _add_cyl(
    bm: bmesh.types.BMesh,
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    r0: float,
    r1: float,
    segments: int = 16,
    cap_ends: bool = True,
    mat_index: int = 0,
) -> None:
    v0 = Vector(p0)
    v1 = Vector(p1)
    dv = v1 - v0
    length = dv.length
    if length < 1e-6:
        return
    mid = (v0 + v1) * 0.5
    rot = dv.normalized().to_track_quat("Z", "Y").to_matrix().to_4x4()
    mat = Matrix.Translation(mid) @ rot

    tmp = bmesh.new()
    bmesh.ops.create_cone(
        tmp,
        cap_ends=cap_ends,
        cap_tris=False,
        segments=segments,
        radius1=r0,
        radius2=r1,
        depth=length,
        matrix=mat,
    )
    vert_map = {v: bm.verts.new(v.co) for v in tmp.verts}
    for f in tmp.faces:
        nf = bm.faces.new([vert_map[v] for v in f.verts])
        nf.material_index = mat_index
    tmp.free()


# ============================================================================
# PROCEDURAL LOCAL-SPACE GRID HELPER (EXACT WORLD-UNIT METRIC TICKS)
# ============================================================================
def _build_axis_grid_mask(
    nt: bpy.types.NodeTree,
    coord_socket: bpy.types.NodeSocket,
    step_m: float,
    half_width_m: float,
) -> bpy.types.NodeSocket:
    """
    Build a symmetric 0..1 mask peaking at 1.0 whenever `coord_socket` is within
    `half_width_m` of any integer multiple `k * step_m` (including negative coords).
    """
    div = nt.nodes.new("ShaderNodeMath")
    div.operation = "DIVIDE"
    div.inputs[1].default_value = step_m
    nt.links.new(coord_socket, div.inputs[0])

    add_half = nt.nodes.new("ShaderNodeMath")
    add_half.operation = "ADD"
    add_half.inputs[1].default_value = 0.5
    nt.links.new(div.outputs["Value"], add_half.inputs[0])

    frac = nt.nodes.new("ShaderNodeMath")
    frac.operation = "FRACT"
    nt.links.new(add_half.outputs["Value"], frac.inputs[0])

    sub_half = nt.nodes.new("ShaderNodeMath")
    sub_half.operation = "SUBTRACT"
    sub_half.inputs[1].default_value = 0.5
    nt.links.new(frac.outputs["Value"], sub_half.inputs[0])

    abs_n = nt.nodes.new("ShaderNodeMath")
    abs_n.operation = "ABSOLUTE"
    nt.links.new(sub_half.outputs["Value"], abs_n.inputs[0])

    to_m = nt.nodes.new("ShaderNodeMath")
    to_m.operation = "MULTIPLY"
    to_m.inputs[1].default_value = step_m
    nt.links.new(abs_n.outputs["Value"], to_m.inputs[0])

    # Crisp anti-aliased edge via MapRange [0.5*hw .. 1.35*hw] -> [1 .. 0]
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.clamp = True
    mr.inputs["From Min"].default_value = half_width_m * 0.45
    mr.inputs["From Max"].default_value = half_width_m * 1.40
    mr.inputs["To Min"].default_value = 1.0
    mr.inputs["To Max"].default_value = 0.0
    nt.links.new(to_m.outputs["Value"], mr.inputs["Value"])
    return mr.outputs["Result"]


def _combine_max_masks(
    nt: bpy.types.NodeTree,
    a_socket: bpy.types.NodeSocket,
    b_socket: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    mx = nt.nodes.new("ShaderNodeMath")
    mx.operation = "MAXIMUM"
    mx.use_clamp = True
    nt.links.new(a_socket, mx.inputs[0])
    nt.links.new(b_socket, mx.inputs[1])
    return mx.outputs["Value"]


# ============================================================================
# STUDIO MATERIALS (LOCAL-SPACE, BLENDER 5.3.0 & AGX SAFE)
# ============================================================================
def _make_backdrop_mat(
    tank_root: bpy.types.Object,
    bp_nodes: list[bpy.types.Node],
) -> bpy.types.Material:
    """
    Dual-mode studio backdrop wall shader:
    - BlueprintMix = 0.0: Deep oceanic-navy depth gradient + subtle underwater caustic shimmer.
    - BlueprintMix = 1.0: Vibrant @zackdfilms Cyan-Blue CAD Studio Blueprint Grid (Beat 5).
    """
    mat = bpy.data.materials.new("Mat_Tank_Backdrop")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Roughness", 0.78)
    _set_bsdf(bsdf, "Specular IOR Level", 0.14)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = tank_root
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])

    # --- Mode A: Deep Oceanic-Navy Gradient + Subtle Underwater Caustic Shimmer ---
    z_norm = nt.nodes.new("ShaderNodeMapRange")
    z_norm.clamp = True
    z_norm.inputs["From Min"].default_value = -2.25
    z_norm.inputs["From Max"].default_value = 0.35
    z_norm.inputs["To Min"].default_value = 0.0
    z_norm.inputs["To Max"].default_value = 1.0
    nt.links.new(sep.outputs["Z"], z_norm.inputs["Value"])

    cr_ocean = nt.nodes.new("ShaderNodeValToRGB")
    cr_ocean.color_ramp.elements[0].position = 0.0
    cr_ocean.color_ramp.elements[0].color = (0.004, 0.013, 0.030, 1.0)
    e_mid = cr_ocean.color_ramp.elements.new(0.62)
    e_mid.color = (0.010, 0.036, 0.072, 1.0)
    cr_ocean.color_ramp.elements[2].position = 0.96
    cr_ocean.color_ramp.elements[2].color = (0.016, 0.056, 0.105, 1.0)
    nt.links.new(z_norm.outputs["Result"], cr_ocean.inputs["Fac"])

    # Subtle pelagic caustic shimmer below waterline (Z < 0.0)
    mapping_c = nt.nodes.new("ShaderNodeMapping")
    mapping_c.inputs["Scale"].default_value = (1.6, 1.0, 2.8)
    nt.links.new(tc.outputs["Object"], mapping_c.inputs["Vector"])

    noise_c = nt.nodes.new("ShaderNodeTexNoise")
    noise_c.inputs["Scale"].default_value = 1.4
    noise_c.inputs["Detail"].default_value = 4.0
    noise_c.inputs["Roughness"].default_value = 0.55
    noise_c.inputs["Distortion"].default_value = 0.85
    nt.links.new(mapping_c.outputs["Vector"], noise_c.inputs["Vector"])

    cr_caustic = nt.nodes.new("ShaderNodeValToRGB")
    cr_caustic.color_ramp.elements[0].position = 0.48
    cr_caustic.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    cr_caustic.color_ramp.elements[1].position = 0.86
    cr_caustic.color_ramp.elements[1].color = (0.009, 0.038, 0.058, 1.0)
    nt.links.new(noise_c.outputs["Fac"], cr_caustic.inputs["Fac"])

    # Fade caustic gently above water Z > -0.05
    subwater_mask = nt.nodes.new("ShaderNodeMapRange")
    subwater_mask.clamp = True
    subwater_mask.inputs["From Min"].default_value = -0.15
    subwater_mask.inputs["From Max"].default_value = 0.05
    subwater_mask.inputs["To Min"].default_value = 1.0
    subwater_mask.inputs["To Max"].default_value = 0.0
    nt.links.new(sep.outputs["Z"], subwater_mask.inputs["Value"])

    ocean_plus_caustic = nt.nodes.new("ShaderNodeMix")
    ocean_plus_caustic.data_type = "RGBA"
    ocean_plus_caustic.blend_type = "ADD"
    nt.links.new(subwater_mask.outputs["Result"], ocean_plus_caustic.inputs[0])
    oc_a, oc_b = _mix_inputs_from_2(ocean_plus_caustic, "RGBA")
    nt.links.new(cr_ocean.outputs["Color"], oc_a)
    nt.links.new(cr_caustic.outputs["Color"], oc_b)
    ocean_col_out = _mix_output(ocean_plus_caustic, "RGBA")

    # --- Mode B: @zackdfilms Cyan-Blue CAD Studio Blueprint Grid ---
    cr_bp_base = nt.nodes.new("ShaderNodeValToRGB")
    cr_bp_base.color_ramp.elements[0].position = 0.0
    cr_bp_base.color_ramp.elements[0].color = (0.010, 0.068, 0.195, 1.0)
    cr_bp_base.color_ramp.elements[1].position = 0.85
    cr_bp_base.color_ramp.elements[1].color = (0.024, 0.145, 0.345, 1.0)
    nt.links.new(z_norm.outputs["Result"], cr_bp_base.inputs["Fac"])

    # Major 1 FT (0.3048m) & Minor 0.25 FT (0.0762m) CAD grid lines on X and Z
    maj_x = _build_axis_grid_mask(nt, sep.outputs["X"], C.FT_TO_M, 0.0065)
    maj_z = _build_axis_grid_mask(nt, sep.outputs["Z"], C.FT_TO_M, 0.0065)
    maj_grid = _combine_max_masks(nt, maj_x, maj_z)

    min_x = _build_axis_grid_mask(nt, sep.outputs["X"], C.FT_TO_M * 0.25, 0.0028)
    min_z = _build_axis_grid_mask(nt, sep.outputs["Z"], C.FT_TO_M * 0.25, 0.0028)
    min_grid = _combine_max_masks(nt, min_x, min_z)

    mix_bp_min = nt.nodes.new("ShaderNodeMix")
    mix_bp_min.data_type = "RGBA"
    nt.links.new(min_grid, mix_bp_min.inputs[0])
    mmin_a, mmin_b = _mix_inputs_from_2(mix_bp_min, "RGBA")
    nt.links.new(cr_bp_base.outputs["Color"], mmin_a)
    mmin_b.default_value = (0.048, 0.34, 0.64, 1.0)

    mix_bp_maj = nt.nodes.new("ShaderNodeMix")
    mix_bp_maj.data_type = "RGBA"
    nt.links.new(maj_grid, mix_bp_maj.inputs[0])
    mmaj_a, mmaj_b = _mix_inputs_from_2(mix_bp_maj, "RGBA")
    nt.links.new(_mix_output(mix_bp_min, "RGBA"), mmaj_a)
    mmaj_b.default_value = (0.22, 0.84, 1.0, 1.0)
    bp_col_out = _mix_output(mix_bp_maj, "RGBA")

    # --- Driven BlueprintMix Value Node ---
    bp_val = nt.nodes.new("ShaderNodeValue")
    bp_val.name = "BlueprintMix"
    bp_val.outputs[0].default_value = 0.0
    bp_nodes.append(bp_val)

    final_base = nt.nodes.new("ShaderNodeMix")
    final_base.data_type = "RGBA"
    nt.links.new(bp_val.outputs[0], final_base.inputs[0])
    fb_a, fb_b = _mix_inputs_from_2(final_base, "RGBA")
    nt.links.new(ocean_col_out, fb_a)
    nt.links.new(bp_col_out, fb_b)
    nt.links.new(_mix_output(final_base, "RGBA"), bsdf.inputs["Base Color"])

    # Subtle AgX-safe background illumination (boosted gently in Beat 5 blueprint mode)
    emit_bp_str = nt.nodes.new("ShaderNodeMath")
    emit_bp_str.operation = "MULTIPLY_ADD"
    emit_bp_str.inputs[1].default_value = 0.88
    emit_bp_str.inputs[2].default_value = 0.28
    nt.links.new(maj_grid, emit_bp_str.inputs[0])

    emit_ocean_str = nt.nodes.new("ShaderNodeMath")
    emit_ocean_str.operation = "MULTIPLY_ADD"
    emit_ocean_str.inputs[1].default_value = 0.08
    emit_ocean_str.inputs[2].default_value = 0.14
    nt.links.new(maj_grid, emit_ocean_str.inputs[0])

    emit_str_mix = nt.nodes.new("ShaderNodeMix")
    emit_str_mix.data_type = "FLOAT"
    nt.links.new(bp_val.outputs[0], emit_str_mix.inputs[0])
    es_a, es_b = _mix_inputs_from_2(emit_str_mix, "FLOAT")
    nt.links.new(emit_ocean_str.outputs["Value"], es_a)
    nt.links.new(emit_bp_str.outputs["Value"], es_b)

    nt.links.new(_mix_output(final_base, "RGBA"), bsdf.inputs["Emission Color"])
    nt.links.new(_mix_output(emit_str_mix, "FLOAT"), bsdf.inputs["Emission Strength"])
    return mat


def _make_floor_mat(
    tank_root: bpy.types.Object,
    bp_nodes: list[bpy.types.Node],
) -> bpy.types.Material:
    """
    Crisp illuminated measurement floor (`Z = -2.15m`) with 1 FT (`0.3048m`) grid lines
    aligned to `X = 0, 1, 2, 3, 6 FT` and Beat 5 Blueprint transition.
    """
    mat = bpy.data.materials.new("Mat_Tank_Floor")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Roughness", 0.44)
    _set_bsdf(bsdf, "Specular IOR Level", 0.35)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = tank_root
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])

    # Subtle micro-textured composite slab bump
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 26.0
    noise.inputs["Detail"].default_value = 6.0
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.035
    bump.inputs["Distance"].default_value = 0.002
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    maj_x = _build_axis_grid_mask(nt, sep.outputs["X"], C.FT_TO_M, 0.007)
    maj_y = _build_axis_grid_mask(nt, sep.outputs["Y"], C.FT_TO_M, 0.007)
    maj_grid = _combine_max_masks(nt, maj_x, maj_y)

    min_x = _build_axis_grid_mask(nt, sep.outputs["X"], C.FT_TO_M * 0.25, 0.0028)
    min_y = _build_axis_grid_mask(nt, sep.outputs["Y"], C.FT_TO_M * 0.25, 0.0028)
    min_grid = _combine_max_masks(nt, min_x, min_y)

    bp_val = nt.nodes.new("ShaderNodeValue")
    bp_val.name = "BlueprintMix"
    bp_val.outputs[0].default_value = 0.0
    bp_nodes.append(bp_val)

    # Base slab color transitions dark navy -> cobalt CAD blueprint
    base_mix = nt.nodes.new("ShaderNodeMix")
    base_mix.data_type = "RGBA"
    nt.links.new(bp_val.outputs[0], base_mix.inputs[0])
    bm_a, bm_b = _mix_inputs_from_2(base_mix, "RGBA")
    bm_a.default_value = (0.011, 0.022, 0.038, 1.0)
    bm_b.default_value = (0.014, 0.088, 0.228, 1.0)

    # Mix sub-grid (modulated by 0.28 + 0.72*BlueprintMix)
    sub_scale = nt.nodes.new("ShaderNodeMath")
    sub_scale.operation = "MULTIPLY_ADD"
    sub_scale.inputs[1].default_value = 0.72
    sub_scale.inputs[2].default_value = 0.25
    nt.links.new(bp_val.outputs[0], sub_scale.inputs[0])

    sub_fac = nt.nodes.new("ShaderNodeMath")
    sub_fac.operation = "MULTIPLY"
    sub_fac.use_clamp = True
    nt.links.new(min_grid, sub_fac.inputs[0])
    nt.links.new(sub_scale.outputs["Value"], sub_fac.inputs[1])

    col_sub = nt.nodes.new("ShaderNodeMix")
    col_sub.data_type = "RGBA"
    nt.links.new(sub_fac.outputs["Value"], col_sub.inputs[0])
    cs_a, cs_b = _mix_inputs_from_2(col_sub, "RGBA")
    nt.links.new(_mix_output(base_mix, "RGBA"), cs_a)
    cs_b.default_value = (0.045, 0.30, 0.56, 1.0)

    # Major 1 FT illuminated lines
    col_maj = nt.nodes.new("ShaderNodeMix")
    col_maj.data_type = "RGBA"
    nt.links.new(maj_grid, col_maj.inputs[0])
    cm_a, cm_b = _mix_inputs_from_2(col_maj, "RGBA")
    nt.links.new(_mix_output(col_sub, "RGBA"), cm_a)
    cm_b.default_value = (0.14, 0.82, 1.0, 1.0)
    nt.links.new(_mix_output(col_maj, "RGBA"), bsdf.inputs["Base Color"])

    # Illuminated grid emission (AgX-safe <= 1.25)
    nt.links.new(_mix_output(col_maj, "RGBA"), bsdf.inputs["Emission Color"])

    grid_em = nt.nodes.new("ShaderNodeMath")
    grid_em.operation = "MULTIPLY"
    nt.links.new(maj_grid, grid_em.inputs[0])
    em_gain = nt.nodes.new("ShaderNodeMath")
    em_gain.operation = "MULTIPLY_ADD"
    em_gain.inputs[1].default_value = 0.40
    em_gain.inputs[2].default_value = 0.85
    nt.links.new(bp_val.outputs[0], em_gain.inputs[0])
    nt.links.new(em_gain.outputs["Value"], grid_em.inputs[1])

    base_em = nt.nodes.new("ShaderNodeMath")
    base_em.operation = "MULTIPLY"
    base_em.inputs[1].default_value = 0.22
    nt.links.new(bp_val.outputs[0], base_em.inputs[0])

    total_em = nt.nodes.new("ShaderNodeMath")
    total_em.operation = "MAXIMUM"
    nt.links.new(grid_em.outputs["Value"], total_em.inputs[0])
    nt.links.new(base_em.outputs["Value"], total_em.inputs[1])
    nt.links.new(total_em.outputs["Value"], bsdf.inputs["Emission Strength"])

    return mat


def _make_water_surface_mat(tank_root: bpy.types.Object) -> bpy.types.Material:
    """
    Crystal-clear glossy wave-rippled transmissive water surface (`Z = 0.0m`)
    with mandatory `LightPath.Is Shadow Ray -> Transparent BSDF` bypass (ZERO volume fog).
    """
    mat = bpy.data.materials.new("Mat_Tank_WaterSurface")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Base Color", (0.86, 0.97, 1.0, 1.0))
    _set_bsdf(bsdf, "Transmission Weight", 0.95)
    _set_bsdf(bsdf, "IOR", 1.333)
    _set_bsdf(bsdf, "Roughness", 0.04)
    _set_bsdf(bsdf, "Specular IOR Level", 0.58)

    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = tank_root

    noise_rip = nt.nodes.new("ShaderNodeTexNoise")
    noise_rip.inputs["Scale"].default_value = 4.2
    noise_rip.inputs["Detail"].default_value = 5.0
    noise_rip.inputs["Roughness"].default_value = 0.42
    nt.links.new(tc.outputs["Object"], noise_rip.inputs["Vector"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.065
    bump.inputs["Distance"].default_value = 0.008
    nt.links.new(noise_rip.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # Mandatory AGENTS.md Sec 3 Shadow Ray Bypass
    lp = nt.nodes.new("ShaderNodeLightPath")
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    transp.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)

    mix_sh = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Shadow Ray"], mix_sh.inputs["Fac"])
    nt.links.new(bsdf.outputs["BSDF"], mix_sh.inputs[1])
    nt.links.new(transp.outputs["BSDF"], mix_sh.inputs[2])
    nt.links.new(mix_sh.outputs["Shader"], out.inputs["Surface"])

    return mat


def _make_metal_mat(
    name: str,
    rgb: tuple[float, float, float],
    metallic: float,
    roughness: float,
    tank_root: bpy.types.Object,
    bump_strength: float = 0.025,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Base Color", (*rgb, 1.0))
    _set_bsdf(bsdf, "Metallic", metallic)
    _set_bsdf(bsdf, "Roughness", roughness)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    if bump_strength > 0.0:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        tc.object = tank_root
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 32.0
        noise.inputs["Detail"].default_value = 6.0
        nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = bump_strength
        bump.inputs["Distance"].default_value = 0.0015
        nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def _make_emissive_trim_mat(
    name: str,
    rgb: tuple[float, float, float],
    strength: float,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Base Color", (*rgb, 1.0))
    _set_bsdf(bsdf, "Roughness", 0.25)
    _set_bsdf(bsdf, "Emission Color", (*rgb, 1.0))
    _set_bsdf(bsdf, "Emission Strength", min(strength, 1.65))
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _make_laser_halo_mat(rgb: tuple[float, float, float], strength: float = 0.72) -> bpy.types.Material:
    """Open-ended additive soft optical halo for the 3 FT vertical laser datum line."""
    mat = bpy.data.materials.new("Mat_Tank_LaserHalo")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    transp.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)

    lp = nt.nodes.new("ShaderNodeLightPath")
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    one_minus_bf = nt.nodes.new("ShaderNodeMath")
    one_minus_bf.operation = "SUBTRACT"
    one_minus_bf.inputs[0].default_value = 1.0
    nt.links.new(geom.outputs["Backfacing"], one_minus_bf.inputs[1])

    cam_front = nt.nodes.new("ShaderNodeMath")
    cam_front.operation = "MULTIPLY"
    nt.links.new(lp.outputs["Is Camera Ray"], cam_front.inputs[0])
    nt.links.new(one_minus_bf.outputs["Value"], cam_front.inputs[1])

    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.35
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = 2.6
    nt.links.new(lw.outputs["Facing"], pw.inputs[0])

    em_gain = nt.nodes.new("ShaderNodeMath")
    em_gain.operation = "MULTIPLY"
    nt.links.new(cam_front.outputs["Value"], em_gain.inputs[0])
    nt.links.new(pw.outputs["Value"], em_gain.inputs[1])

    em_scaled = nt.nodes.new("ShaderNodeMath")
    em_scaled.operation = "MULTIPLY"
    em_scaled.inputs[1].default_value = strength
    nt.links.new(em_gain.outputs["Value"], em_scaled.inputs[0])

    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (*rgb, 1.0)
    nt.links.new(em_scaled.outputs["Value"], emit.inputs["Strength"])

    add_sh = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(transp.outputs["BSDF"], add_sh.inputs[0])
    nt.links.new(emit.outputs["Emission"], add_sh.inputs[1])
    nt.links.new(add_sh.outputs["Shader"], out.inputs["Surface"])
    return mat


# ============================================================================
# GEOMETRY BUILDERS
# ============================================================================
def _build_backdrop_dome(
    col: bpy.types.Collection,
    tank_root: bpy.types.Object,
    mat_backdrop: bpy.types.Material,
) -> bpy.types.Object:
    """Curved oceanic-navy / blueprint studio cyclorama wall centered at Y = +2.6m."""
    bm = bmesh.new()
    nx, nz = 48, 20
    x0, x1 = -4.8, 5.8
    z0, z1 = -2.45, 3.10
    y_back = 2.60

    grid: list[list[bmesh.types.BMVert]] = []
    for ix in range(nx + 1):
        u = ix / nx
        x = x0 + (x1 - x0) * u
        # Keep rear flat around hero action X in [-2.2, 3.2], gently wrap outer wings forward
        if x < -2.2:
            d = (-2.2 - x) / 2.6
            y = y_back - 1.45 * (d * d)
        elif x > 3.2:
            d = (x - 3.2) / 2.6
            y = y_back - 1.45 * (d * d)
        else:
            y = y_back
        row = []
        for iz in range(nz + 1):
            v = iz / nz
            z = z0 + (z1 - z0) * v
            row.append(bm.verts.new((x, y, z)))
        grid.append(row)

    for ix in range(nx):
        for iz in range(nz):
            bm.faces.new((grid[ix][iz], grid[ix + 1][iz], grid[ix + 1][iz + 1], grid[ix][iz + 1]))

    return _finalize_bmesh(
        bm,
        "Tank_Backdrop",
        col,
        materials=[mat_backdrop],
        smooth=True,
        parent=tank_root,
        visible_shadow=False,
    )


def _build_measurement_floor(
    col: bpy.types.Collection,
    tank_root: bpy.types.Object,
    mat_floor: bpy.types.Material,
    mat_metal_dark: bpy.types.Material,
) -> bpy.types.Object:
    """Illuminated measurement floor strictly at top Z = C.TANK['floor_z'] (-2.15m)."""
    fz = float(C.TANK["floor_z"])
    bm = bmesh.new()
    # Thick solid floor slab (top = fz, bottom = fz - 0.20m)
    _add_box(bm, (0.50, 0.15, fz - 0.10), (10.6, 5.2, 0.20), mat_index=0)
    # Subtle rear laboratory curb rail along Y = +2.42m
    _add_box(bm, (0.50, 2.42, fz + 0.045), (10.4, 0.16, 0.09), mat_index=1, bevel_width=0.008)

    return _finalize_bmesh(
        bm,
        "Tank_Floor",
        col,
        materials=[mat_floor, mat_metal_dark],
        smooth=False,
        parent=tank_root,
        visible_shadow=True,
    )


def _build_water_surface(
    col: bpy.types.Collection,
    tank_root: bpy.types.Object,
    mat_water: bpy.types.Material,
) -> bpy.types.Object:
    """Glossy wave-rippled transmissive water meniscus sheet at Z = C.TANK['water_z'] (0.0m)."""
    wz = float(C.TANK["water_z"])
    bm = bmesh.new()
    nx, ny = 120, 72
    x0, x1 = -2.4, 3.6
    y0, y1 = -1.8, 2.60

    verts: list[list[bmesh.types.BMVert]] = []
    for ix in range(nx + 1):
        x = x0 + (x1 - x0) * (ix / nx)
        row = []
        for iy in range(ny + 1):
            y = y0 + (y1 - y0) * (iy / ny)
            r = math.hypot(x - 0.0, y - 0.0)
            rip = (
                0.0042 * math.sin(3.4 * x + 1.8 * y)
                + 0.0026 * math.cos(5.8 * y - 2.4 * x)
                + 0.0038 * math.exp(-1.1 * r) * math.cos(14.0 * r)
            )
            row.append(bm.verts.new((x, y, wz + rip)))
        verts.append(row)

    for ix in range(nx):
        for iy in range(ny):
            bm.faces.new((verts[ix][iy], verts[ix + 1][iy], verts[ix + 1][iy + 1], verts[ix][iy + 1]))

    return _finalize_bmesh(
        bm,
        "Water_Surface",
        col,
        materials=[mat_water],
        smooth=True,
        parent=tank_root,
        visible_shadow=True,
    )


def _load_studio_font() -> bpy.types.VectorFont | None:
    if FONT_PATH.exists():
        for fnt in bpy.data.fonts:
            if Path(bpy.path.abspath(fnt.filepath)) == FONT_PATH:
                return fnt
        return bpy.data.fonts.load(str(FONT_PATH))
    return None


def _build_ruler_and_datum(
    col: bpy.types.Collection,
    tank_root: bpy.types.Object,
    mat_metal_dark: bpy.types.Material,
    mat_cyan_emiss: bpy.types.Material,
    mat_emerald_emiss: bpy.types.Material,
    mat_laser_halo: bpy.types.Material,
    mat_plaque: bpy.types.Material,
) -> bpy.types.Object:
    """
    Build glowing horizontal ruler beam at Z = C.TANK['ruler_z'] (-1.42m) with chamfered tick posts,
    embossed `0`, `1`, `2`, `3 FT`, `6 FT` indicators, and ultra-clean `3 FT` vertical laser datum line.
    """
    rz = float(C.TANK["ruler_z"])
    fz = float(C.TANK["floor_z"])
    wz = float(C.TANK["water_z"])
    ry = 0.42  # Behind bullet flight lane (Y=0.0) and diver so zero intersections

    bm = bmesh.new()
    # Main anodized rail spanning X = -0.32m .. +2.25m
    _add_box(bm, (0.965, ry, rz), (2.57, 0.044, 0.036), mat_index=0, bevel_width=0.004)
    # Recessed glowing cyan LED strip along front face
    _add_box(bm, (0.965, ry - 0.022, rz), (2.50, 0.006, 0.011), mat_index=1)

    # Support stanchions anchoring ruler beam down to floor Z=-2.15m
    for sx in (-0.24, 0.9144, 2.15):
        _add_cyl(bm, (sx, ry + 0.015, fz), (sx, ry + 0.015, rz - 0.018), 0.014, 0.012, segments=12, mat_index=0)
        _add_box(bm, (sx, ry + 0.015, fz + 0.012), (0.070, 0.070, 0.024), mat_index=0, bevel_width=0.004)

    # Major foot ticks (0..6 FT) + chamfered hero posts & plaques
    major_specs = [
        (0, "0", False),
        (1, "1", False),
        (2, "2", False),
        (3, "3 FT", True),
        (4, None, False),
        (5, None, False),
        (6, "6 FT", False),
    ]
    for ft_idx, label_txt, is_hero in major_specs:
        tx = ft_idx * C.FT_TO_M
        post_h = 0.095 if is_hero else 0.068
        post_w = 0.018 if is_hero else 0.013
        glow_idx = 2 if is_hero else 1
        # Chamfered tick post rising from ruler top
        _add_box(bm, (tx, ry, rz + 0.018 + 0.5 * post_h), (post_w, 0.026, post_h), mat_index=0, bevel_width=0.003)
        _add_box(bm, (tx, ry - 0.014, rz + 0.018 + 0.5 * post_h), (post_w * 0.55, 0.005, post_h * 0.86), mat_index=glow_idx)

        if label_txt is not None:
            pw = 0.115 if "FT" in label_txt else 0.056
            _add_box(bm, (tx, ry - 0.016, rz - 0.052), (pw, 0.014, 0.052), mat_index=3, bevel_width=0.004)

    # Minor half-foot ticks (0.5 .. 5.5 FT)
    for half_idx in range(6):
        hx = (half_idx + 0.5) * C.FT_TO_M
        _add_box(bm, (hx, ry, rz + 0.032), (0.008, 0.018, 0.030), mat_index=0, bevel_width=0.0015)
        _add_box(bm, (hx, ry - 0.010, rz + 0.032), (0.004, 0.004, 0.024), mat_index=1)

    # Ultra-clean 3 FT vertical laser datum emitter/receiver collars at X = 0.9144m, Y = +0.28m
    lx = float(C.THREE_FT_X)
    ly = 0.28
    _add_box(bm, (lx, ly, fz + 0.015), (0.076, 0.076, 0.030), mat_index=0, bevel_width=0.005)
    _add_cyl(bm, (lx, ly, fz + 0.030), (lx, ly, fz + 0.052), 0.020, 0.014, segments=16, mat_index=2)
    _add_cyl(bm, (lx, ly, wz - 0.030), (lx, ly, wz - 0.008), 0.012, 0.018, segments=16, mat_index=0)

    ruler_obj = _finalize_bmesh(
        bm,
        "Tank_Ruler_Beam",
        col,
        materials=[mat_metal_dark, mat_cyan_emiss, mat_emerald_emiss, mat_plaque],
        smooth=False,
        parent=tank_root,
        visible_shadow=True,
    )

    # Embossed crisp `0`, `1`, `2`, `3 FT`, `6 FT` text curves parented to ruler_obj
    font = _load_studio_font()
    for ft_idx, label_txt, is_hero in major_specs:
        if label_txt is None:
            continue
        tx = ft_idx * C.FT_TO_M
        curve = bpy.data.curves.new(f"Tank_Ruler_Text_{ft_idx}FT_Curve", type="FONT")
        curve.body = label_txt
        if font is not None:
            curve.font = font
        curve.align_x = "CENTER"
        curve.align_y = "CENTER"
        curve.size = 0.030 if is_hero else 0.026
        curve.space_character = 1.10
        curve.extrude = 0.0018
        curve.bevel_depth = 0.00018
        curve.bevel_resolution = 1
        curve.materials.append(mat_emerald_emiss if is_hero else mat_cyan_emiss)

        t_obj = bpy.data.objects.new(f"Tank_Ruler_Text_{ft_idx}FT", curve)
        t_obj.location = Vector((tx, ry - 0.025, rz - 0.052))
        t_obj.rotation_euler = Euler((math.radians(90.0), 0.0, 0.0), "XYZ")
        t_obj.visible_shadow = False
        t_obj.parent = ruler_obj
        _link_obj(t_obj, col)

    # 3 FT vertical laser datum core bar (thin crisp cyan-emerald rod, Emiss <= 1.65)
    bm_core = bmesh.new()
    _add_cyl(bm_core, (lx, ly, fz + 0.048), (lx, ly, wz - 0.022), 0.0032, 0.0032, segments=12, mat_index=0)
    _finalize_bmesh(
        bm_core,
        "Tank_Datum_3FT_Core",
        col,
        materials=[mat_emerald_emiss],
        smooth=True,
        parent=ruler_obj,
        visible_shadow=False,
    )

    # Open-ended additive laser halo sheath (end_fill_type='NOTHING' / cap_ends=False)
    bm_halo = bmesh.new()
    _add_cyl(bm_halo, (lx, ly, fz + 0.050), (lx, ly, wz - 0.024), 0.015, 0.015, segments=16, cap_ends=False, mat_index=0)
    _finalize_bmesh(
        bm_halo,
        "Tank_Datum_3FT_Halo",
        col,
        materials=[mat_laser_halo],
        smooth=True,
        recalc_normals=False,
        parent=ruler_obj,
        visible_shadow=False,
    )

    return ruler_obj


def _build_dock_and_muzzle_mount(
    col: bpy.types.Collection,
    tank_root: bpy.types.Object,
    mat_metal_dark: bpy.types.Material,
    mat_metal_steel: bpy.types.Material,
    mat_cyan_emiss: bpy.types.Material,
) -> bpy.types.Object:
    """
    Subtle dark architectural dock deck slab & machined muzzle mount clamp strictly at
    `Z > +0.02m`, `X in [-1.60, -0.02]` for Beat 2 above-water barrel framing context.
    """
    bm = bmesh.new()
    # Main dock deck slab: X in [-1.60, -0.04], Z in [+0.04, +0.16]
    _add_box(bm, (-0.82, 0.12, 0.10), (1.56, 1.60, 0.12), mat_index=0, bevel_width=0.008)
    # Brushed steel dock edge coping rail at X = -0.06m
    _add_box(bm, (-0.065, 0.12, 0.168), (0.05, 1.56, 0.016), mat_index=1, bevel_width=0.003)
    # Subtle illuminated waterline datum bar along dock lip
    _add_box(bm, (-0.038, 0.12, 0.075), (0.006, 1.48, 0.008), mat_index=2)

    # Ballistic test-bench stanchion & C-clamp mount offset behind Y=+0.26m so flight centerline stays clear
    _add_box(bm, (-0.54, 0.34, 0.175), (0.58, 0.24, 0.030), mat_index=1, bevel_width=0.004)
    _add_cyl(bm, (-0.50, 0.36, 0.19), (-0.50, 0.36, 0.40), 0.024, 0.020, segments=16, mat_index=0)
    # Cantilever clamp yoke reaching toward muzzle test axis without blocking X=0 sightline
    _add_box(bm, (-0.50, 0.24, 0.36), (0.14, 0.26, 0.044), mat_index=1, bevel_width=0.004)
    _add_cyl(bm, (-0.50, 0.14, 0.36), (-0.50, 0.08, 0.36), 0.034, 0.034, segments=18, mat_index=0)

    return _finalize_bmesh(
        bm,
        "Tank_Dock_MuzzleMount",
        col,
        materials=[mat_metal_dark, mat_metal_steel, mat_cyan_emiss],
        smooth=False,
        parent=tank_root,
        visible_shadow=True,
    )


# ============================================================================
# WORLD & STUDIO THREE-POINT + SOFTBOX EYE LIGHTING
# ============================================================================
def _build_world_and_lights(
    col: bpy.types.Collection,
    tank_root: bpy.types.Object,
    total_frames: int,
) -> list[bpy.types.Object]:
    scene = bpy.context.scene
    world = bpy.data.worlds.new("Tank_Env_World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])

    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].position = 0.20
    cr.color_ramp.elements[0].color = (0.006, 0.018, 0.040, 1.0)
    cr.color_ramp.elements[1].position = 0.85
    cr.color_ramp.elements[1].color = (0.018, 0.048, 0.092, 1.0)
    nt.links.new(sep.outputs["Z"], cr.inputs["Fac"])
    nt.links.new(cr.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.28
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    if scene.render.engine == "CYCLES":
        scene.cycles.transparent_max_bounces = max(16, scene.cycles.transparent_max_bounces)

    def _add_area(
        name: str,
        loc: tuple[float, float, float],
        target: tuple[float, float, float],
        energy: float,
        color: tuple[float, float, float],
        size: float,
        diffuse_fac: float = 1.0,
        specular_fac: float = 1.0,
    ) -> bpy.types.Object:
        ld = bpy.data.lights.new(f"{name}_Data", type="AREA")
        ld.energy = energy
        ld.color = color
        ld.size = size
        if hasattr(ld, "diffuse_factor"):
            ld.diffuse_factor = diffuse_fac
        if hasattr(ld, "specular_factor"):
            ld.specular_factor = specular_fac
        obj = bpy.data.objects.new(name, ld)
        obj.location = Vector(loc)
        obj.rotation_euler = _aim_euler(loc, target)
        obj.visible_camera = False
        obj.parent = tank_root
        _link_obj(obj, col)
        return obj

    # 1. Cool cyan rim AREA light behind/above ((0.5, +1.4, +0.9))
    rim_light = _add_area(
        "Tank_Rim_CoolCyan",
        loc=(0.50, 1.40, 0.90),
        target=(0.85, 0.0, -0.62),
        energy=72.0,
        color=(0.36, 0.82, 1.0),
        size=1.65,
        diffuse_fac=0.85,
        specular_fac=1.10,
    )

    # 2. Warm crisp top caustic AREA key ((0.4, -0.8, +1.8))
    key_light = _add_area(
        "Tank_Key_WarmCaustic",
        loc=(0.40, -0.80, 1.80),
        target=(0.65, 0.05, -0.58),
        energy=115.0,
        color=(1.0, 0.94, 0.84),
        size=1.45,
        diffuse_fac=0.95,
        specular_fac=0.90,
    )

    # 3. Soft oceanic floor bounce AREA fill
    floor_bounce = _add_area(
        "Tank_Bounce_Floor",
        loc=(0.65, -0.55, -2.02),
        target=(0.75, 0.05, -0.55),
        energy=28.0,
        color=(0.38, 0.84, 1.0),
        size=2.20,
        diffuse_fac=0.78,
        specular_fac=0.28,
    )

    # 4. Mandatory V12 0.42m Softbox Eye Catchlight aimed toward diver face in Beats 1 & 6
    b1_eye_loc = (0.32, -1.42, -0.44)
    b1_eye_tgt = (1.50, 0.06, -0.54)
    eye_catch = _add_area(
        "Tank_Swimmer_EyeCatchlight",
        loc=b1_eye_loc,
        target=b1_eye_tgt,
        energy=54.0,
        color=(1.0, 0.96, 0.92),
        size=0.42,
        diffuse_fac=0.14,
        specular_fac=1.78,
    )

    b6_eye_loc = (0.36, -1.42, -0.78)
    b6_eye_tgt = (1.36, 0.12, -0.88)
    for f, loc, tgt in (
        (1, b1_eye_loc, b1_eye_tgt),
        (210, b1_eye_loc, b1_eye_tgt),
        (1500, b6_eye_loc, b6_eye_tgt),
        (total_frames, b6_eye_loc, b6_eye_tgt),
    ):
        eye_catch.location = Vector(loc)
        eye_catch.rotation_euler = _aim_euler(loc, tgt)
        eye_catch.keyframe_insert("location", frame=f)
        eye_catch.keyframe_insert("rotation_euler", frame=f)
    _set_fcurve_interp(eye_catch, "LINEAR")

    # 5. Beat 5 Blueprint Studio Fill Light synced cleanly to C.blueprint_bg_weight(f)
    bp_fill = _add_area(
        "Tank_Blueprint_StudioFill",
        loc=(0.85, -1.65, -0.55),
        target=(0.85, 0.0, -0.56),
        energy=0.0,
        color=(0.86, 0.96, 1.0),
        size=2.10,
        diffuse_fac=0.80,
        specular_fac=0.65,
    )
    bp_frames = sorted(
        {
            1,
            1260,
            *range(1261, 1290, 2),
            1410,
            *range(1536, 1564, 2),
            total_frames,
        }
    )
    for f in bp_frames:
        if 1 <= f <= total_frames:
            bp_fill.data.energy = 44.0 * C.blueprint_bg_weight(float(f))
            bp_fill.data.keyframe_insert("energy", frame=f)
    _set_fcurve_interp(bp_fill.data, "LINEAR")

    return [rim_light, key_light, floor_bounce, eye_catch, bp_fill]


# ============================================================================
# BLUEPRINT MIX ANIMATION
# ============================================================================
def _animate_blueprint_nodes(bp_nodes: list[bpy.types.Node], total_frames: int) -> None:
    """Drive `BlueprintMix` ShaderNodeValue nodes smoothly via C.blueprint_bg_weight(f)."""
    frames = sorted(
        {
            1,
            1258,
            1260,
            *range(1261, 1292, 2),
            1410,
            *range(1536, 1566, 2),
            1568,
            total_frames,
        }
    )
    for node in bp_nodes:
        for f in frames:
            if 1 <= f <= total_frames:
                node.outputs[0].default_value = C.blueprint_bg_weight(float(f))
                node.outputs[0].keyframe_insert("default_value", frame=f)
        _set_fcurve_interp(node.id_data, "LINEAR")


def _animate_hide_in_beat5(obj: bpy.types.Object, total_frames: int) -> None:
    """Keep obj visible in underwater Beats 1-4 & Beat 6, hidden in Beat 5 Studio Blueprint."""
    for f, hidden in (
        (1, False),
        (1260, False),
        (1261, True),
        (1560, True),
        (1561, False),
        (total_frames, False),
    ):
        obj.hide_viewport = hidden
        obj.hide_render = hidden
        obj.keyframe_insert("hide_viewport", frame=f)
        obj.keyframe_insert("hide_render", frame=f)
    _set_fcurve_interp(obj, "CONSTANT", prefix="hide_")


# ============================================================================
# PUBLIC ENTRYPOINT
# ============================================================================
def build_tank_env(col: bpy.types.Collection, total_frames: int = C.TOTAL_FRAMES) -> dict[str, Any]:
    """
    Build the Crystal Ballistic Studio Tank, Dual-Mode Backdrop, Water Surface,
    Illuminated Ruler & 3 FT Laser Datum, Architectural Dock, and Studio Lighting.
    """
    if col is None:
        col = bpy.context.scene.collection

    tank_root = bpy.data.objects.new("Tank_Root", None)
    tank_root.empty_display_type = "PLAIN_AXES"
    tank_root.empty_display_size = 0.50
    tank_root.location = Vector((0.0, 0.0, 0.0))
    _link_obj(tank_root, col)

    bp_nodes: list[bpy.types.Node] = []

    mat_backdrop = _make_backdrop_mat(tank_root, bp_nodes)
    mat_floor = _make_floor_mat(tank_root, bp_nodes)
    mat_water = _make_water_surface_mat(tank_root)
    mat_metal_dark = _make_metal_mat("Mat_Tank_AnodizedDark", (0.024, 0.030, 0.040), 0.86, 0.36, tank_root)
    mat_metal_steel = _make_metal_mat("Mat_Tank_BrushedSteel", (0.24, 0.27, 0.31), 0.92, 0.28, tank_root)
    mat_plaque = _make_metal_mat("Mat_Tank_RulerPlaque", (0.008, 0.012, 0.020), 0.10, 0.45, tank_root, bump_strength=0.0)
    mat_cyan_emiss = _make_emissive_trim_mat("Mat_Tank_CyanTrim", (0.10, 0.84, 1.0), 1.42)
    mat_emerald_emiss = _make_emissive_trim_mat("Mat_Tank_Emerald3FT", (0.08, 0.96, 0.78), 1.60)
    mat_laser_halo = _make_laser_halo_mat((0.08, 0.96, 0.80), strength=0.72)

    _build_backdrop_dome(col, tank_root, mat_backdrop)
    _build_measurement_floor(col, tank_root, mat_floor, mat_metal_dark)
    water_surface = _build_water_surface(col, tank_root, mat_water)
    ruler = _build_ruler_and_datum(
        col,
        tank_root,
        mat_metal_dark,
        mat_cyan_emiss,
        mat_emerald_emiss,
        mat_laser_halo,
        mat_plaque,
    )
    dock_obj = _build_dock_and_muzzle_mount(col, tank_root, mat_metal_dark, mat_metal_steel, mat_cyan_emiss)

    lights = _build_world_and_lights(col, tank_root, total_frames)
    _animate_blueprint_nodes(bp_nodes, total_frames)
    _animate_hide_in_beat5(water_surface, total_frames)
    _animate_hide_in_beat5(dock_obj, total_frames)

    return {
        "tank_root": tank_root,
        "water_surface": water_surface,
        "ruler": ruler,
        "bp_nodes": bp_nodes,
        "lights": lights,
    }
