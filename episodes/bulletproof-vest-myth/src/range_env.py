"""
range_env.py — Ballistic Testing Range, Frameless Tempered Glass Backdrop Wall,
Dual-Mode Studio Blueprint (Beat 5) & Three-Point + Softbox Eye Lighting
for Episode #3: "Why Bulletproof Vests DON'T Send You Flying Backward" (60 fps, 1800 frames).

Strictly follows AGENTS.md V12 & Blender 5.3.0 rules:
- Zero `action.fcurves` direct access (uses `C.iter_action_fcurves`).
- All `ShaderNodeMix` inputs indexed strictly via `C.get_mix_sockets` (`inputs[2:]`).
- `mesh.materials` attached strictly BEFORE `bm.to_mesh(mesh)`.
- Frameless tempered glass wall routes `LightPath.Is Shadow Ray -> Transparent BSDF`
  with ZERO vertical center mullions blocking faces.
- Mandatory `0.42 m` softbox `AREA` eye catchlights aimed at hero & shooter eye height (`Z=1.62 m`).
- `cycles.transparent_max_bounces >= 16`.

Public API:
    build_range_env(col: bpy.types.Collection, total_frames: int = C.TOTAL_FRAMES) -> dict[str, Any]
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
_REPO_ROOT = _THIS_FILE.parents[3]
if _EPISODE_SRC not in sys.path:
    sys.path.insert(0, _EPISODE_SRC)

import vest_contract as C  # noqa: E402


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


def _mix_output(mix_node: bpy.types.Node, socket_type: str) -> bpy.types.NodeSocket:
    norm_t = "VALUE" if socket_type == "FLOAT" else socket_type
    typed = [s for s in mix_node.outputs if s.type == norm_t]
    if not typed:
        raise RuntimeError(f"ShaderNodeMix {mix_node.name} missing '{socket_type}' output")
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


def _key_visibility(
    obj: bpy.types.Object,
    visible_ranges: list[tuple[int, int]],
    total_frames: int,
) -> None:
    """Keyframe `hide_viewport` & `hide_render` strictly CONSTANT across inclusive windows."""
    t_max = max(1, int(total_frames))
    frames = {1, t_max}
    for s, e in visible_ranges:
        sc = max(1, min(t_max, int(s)))
        ec = max(1, min(t_max, int(e)))
        if sc > 1:
            frames.add(sc - 1)
        frames.add(sc)
        frames.add(ec)
        if ec < t_max:
            frames.add(ec + 1)

    for f in sorted(frames):
        vis = any(s <= f <= e for s, e in visible_ranges)
        obj.hide_viewport = not vis
        obj.hide_render = not vis
        obj.keyframe_insert("hide_viewport", frame=f)
        obj.keyframe_insert("hide_render", frame=f)
    _set_fcurve_interp(obj, "CONSTANT", prefix="hide_")


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
    """Attach materials strictly BEFORE `bm.to_mesh(mesh)` per AGENTS.md V12."""
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
    return _link_obj(obj, col)


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
# PROCEDURAL LOCAL-SPACE METRIC GRID MASK (0.50 m MAJOR / 0.125 m MINOR)
# ============================================================================
def _build_axis_grid_mask(
    nt: bpy.types.NodeTree,
    coord_socket: bpy.types.NodeSocket,
    step_m: float,
    half_width_m: float,
) -> bpy.types.NodeSocket:
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
# STUDIO & BALLISTIC RANGE MATERIALS (LOCAL-SPACE, BLENDER 5.3.0 SAFE)
# ============================================================================
def _make_backdrop_mat(
    env_root: bpy.types.Object,
    bp_nodes: list[bpy.types.Node],
) -> bpy.types.Material:
    """
    Dual-mode studio cyclorama shader:
    - BlueprintMix = 0.0: Moody architectural charcoal-slate ballistic range wall with subtle panel ribs.
    - BlueprintMix = 1.0: Clean @zackdfilms Cyan-Blue CAD Studio Blueprint Grid (Beat 5).
    """
    mat = bpy.data.materials.new("Mat_Range_Backdrop")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Roughness", 0.72)
    _set_bsdf(bsdf, "Specular IOR Level", 0.18)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = env_root
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])

    # Mode A: Moody dark architectural ballistic studio gradient
    z_norm = nt.nodes.new("ShaderNodeMapRange")
    z_norm.clamp = True
    z_norm.inputs["From Min"].default_value = 0.0
    z_norm.inputs["From Max"].default_value = 3.2
    z_norm.inputs["To Min"].default_value = 0.0
    z_norm.inputs["To Max"].default_value = 1.0
    nt.links.new(sep.outputs["Z"], z_norm.inputs["Value"])

    cr_dark = nt.nodes.new("ShaderNodeValToRGB")
    cr_dark.color_ramp.elements[0].position = 0.0
    cr_dark.color_ramp.elements[0].color = (0.008, 0.011, 0.017, 1.0)
    e_mid = cr_dark.color_ramp.elements.new(0.55)
    e_mid.color = (0.014, 0.022, 0.036, 1.0)
    cr_dark.color_ramp.elements[2].position = 1.0
    cr_dark.color_ramp.elements[2].color = (0.007, 0.013, 0.024, 1.0)
    nt.links.new(z_norm.outputs["Result"], cr_dark.inputs["Fac"])

    # Subtle concrete micro-grain bump
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 14.0
    noise.inputs["Detail"].default_value = 6.0
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.04
    bump.inputs["Distance"].default_value = 0.003
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # Mode B: @zackdfilms Vibrant Cyan-Blue CAD Blueprint Grid (0.50m major / 0.125m minor)
    cr_bp = nt.nodes.new("ShaderNodeValToRGB")
    cr_bp.color_ramp.elements[0].position = 0.0
    cr_bp.color_ramp.elements[0].color = (0.010, 0.068, 0.195, 1.0)
    cr_bp.color_ramp.elements[1].position = 0.90
    cr_bp.color_ramp.elements[1].color = (0.022, 0.140, 0.340, 1.0)
    nt.links.new(z_norm.outputs["Result"], cr_bp.inputs["Fac"])

    maj_x = _build_axis_grid_mask(nt, sep.outputs["X"], 0.50, 0.0075)
    maj_z = _build_axis_grid_mask(nt, sep.outputs["Z"], 0.50, 0.0075)
    maj_grid = _combine_max_masks(nt, maj_x, maj_z)

    min_x = _build_axis_grid_mask(nt, sep.outputs["X"], 0.125, 0.0030)
    min_z = _build_axis_grid_mask(nt, sep.outputs["Z"], 0.125, 0.0030)
    min_grid = _combine_max_masks(nt, min_x, min_z)

    mix_min = nt.nodes.new("ShaderNodeMix")
    mix_min.data_type = "RGBA"
    fac, a_in, b_in = C.get_mix_sockets(mix_min)
    nt.links.new(min_grid, fac)
    nt.links.new(cr_bp.outputs["Color"], a_in)
    b_in.default_value = (0.045, 0.34, 0.64, 1.0)

    mix_maj = nt.nodes.new("ShaderNodeMix")
    mix_maj.data_type = "RGBA"
    fac, a_in, b_in = C.get_mix_sockets(mix_maj)
    nt.links.new(maj_grid, fac)
    nt.links.new(_mix_output(mix_min, "RGBA"), a_in)
    b_in.default_value = (0.24, 0.86, 1.0, 1.0)

    bp_val = nt.nodes.new("ShaderNodeValue")
    bp_val.name = "BlueprintMix"
    bp_val.outputs[0].default_value = 0.0
    bp_nodes.append(bp_val)

    final_col = nt.nodes.new("ShaderNodeMix")
    final_col.data_type = "RGBA"
    fac, a_in, b_in = C.get_mix_sockets(final_col)
    nt.links.new(bp_val.outputs[0], fac)
    nt.links.new(cr_dark.outputs["Color"], a_in)
    nt.links.new(_mix_output(mix_maj, "RGBA"), b_in)
    nt.links.new(_mix_output(final_col, "RGBA"), bsdf.inputs["Base Color"])

    # AgX-safe backdrop illumination (dark moody range -> crisp glowing blueprint in Beat 5)
    emit_bp = nt.nodes.new("ShaderNodeMath")
    emit_bp.operation = "MULTIPLY_ADD"
    emit_bp.inputs[1].default_value = 0.90
    emit_bp.inputs[2].default_value = 0.26
    nt.links.new(maj_grid, emit_bp.inputs[0])

    emit_dark = nt.nodes.new("ShaderNodeMath")
    emit_dark.operation = "MULTIPLY_ADD"
    emit_dark.inputs[1].default_value = 0.06
    emit_dark.inputs[2].default_value = 0.08
    nt.links.new(maj_grid, emit_dark.inputs[0])

    emit_mix = nt.nodes.new("ShaderNodeMix")
    emit_mix.data_type = "FLOAT"
    fac, a_in, b_in = C.get_mix_sockets(emit_mix)
    nt.links.new(bp_val.outputs[0], fac)
    nt.links.new(emit_dark.outputs["Value"], a_in)
    nt.links.new(emit_bp.outputs["Value"], b_in)

    nt.links.new(_mix_output(final_col, "RGBA"), bsdf.inputs["Emission Color"])
    nt.links.new(_mix_output(emit_mix, "FLOAT"), bsdf.inputs["Emission Strength"])
    return mat


def _make_floor_mat(
    env_root: bpy.types.Object,
    bp_nodes: list[bpy.types.Node],
) -> bpy.types.Material:
    """Dark concrete ballistic studio floor (`Z = 0.00 m`) with glowing cyan metric grid & Beat 5 blueprint."""
    mat = bpy.data.materials.new("Mat_Range_Floor")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Roughness", 0.38)
    _set_bsdf(bsdf, "Specular IOR Level", 0.36)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = env_root
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])

    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 22.0
    noise.inputs["Detail"].default_value = 6.0
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.032
    bump.inputs["Distance"].default_value = 0.002
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    maj_x = _build_axis_grid_mask(nt, sep.outputs["X"], 0.50, 0.007)
    maj_y = _build_axis_grid_mask(nt, sep.outputs["Y"], 0.50, 0.007)
    maj_grid = _combine_max_masks(nt, maj_x, maj_y)

    min_x = _build_axis_grid_mask(nt, sep.outputs["X"], 0.125, 0.0028)
    min_y = _build_axis_grid_mask(nt, sep.outputs["Y"], 0.125, 0.0028)
    min_grid = _combine_max_masks(nt, min_x, min_y)

    bp_val = nt.nodes.new("ShaderNodeValue")
    bp_val.name = "BlueprintMix"
    bp_val.outputs[0].default_value = 0.0
    bp_nodes.append(bp_val)

    base_mix = nt.nodes.new("ShaderNodeMix")
    base_mix.data_type = "RGBA"
    fac, a_in, b_in = C.get_mix_sockets(base_mix)
    nt.links.new(bp_val.outputs[0], fac)
    a_in.default_value = (0.015, 0.018, 0.025, 1.0)
    b_in.default_value = (0.012, 0.085, 0.225, 1.0)

    sub_scale = nt.nodes.new("ShaderNodeMath")
    sub_scale.operation = "MULTIPLY_ADD"
    sub_scale.inputs[1].default_value = 0.72
    sub_scale.inputs[2].default_value = 0.18
    nt.links.new(bp_val.outputs[0], sub_scale.inputs[0])

    sub_fac = nt.nodes.new("ShaderNodeMath")
    sub_fac.operation = "MULTIPLY"
    sub_fac.use_clamp = True
    nt.links.new(min_grid, sub_fac.inputs[0])
    nt.links.new(sub_scale.outputs["Value"], sub_fac.inputs[1])

    col_sub = nt.nodes.new("ShaderNodeMix")
    col_sub.data_type = "RGBA"
    fac, a_in, b_in = C.get_mix_sockets(col_sub)
    nt.links.new(sub_fac.outputs["Value"], fac)
    nt.links.new(_mix_output(base_mix, "RGBA"), a_in)
    b_in.default_value = (0.045, 0.30, 0.56, 1.0)

    col_maj = nt.nodes.new("ShaderNodeMix")
    col_maj.data_type = "RGBA"
    fac, a_in, b_in = C.get_mix_sockets(col_maj)
    nt.links.new(maj_grid, fac)
    nt.links.new(_mix_output(col_sub, "RGBA"), a_in)
    b_in.default_value = (0.14, 0.84, 1.0, 1.0)
    nt.links.new(_mix_output(col_maj, "RGBA"), bsdf.inputs["Base Color"])
    nt.links.new(_mix_output(col_maj, "RGBA"), bsdf.inputs["Emission Color"])

    grid_em = nt.nodes.new("ShaderNodeMath")
    grid_em.operation = "MULTIPLY"
    nt.links.new(maj_grid, grid_em.inputs[0])
    em_gain = nt.nodes.new("ShaderNodeMath")
    em_gain.operation = "MULTIPLY_ADD"
    em_gain.inputs[1].default_value = 0.65
    em_gain.inputs[2].default_value = 0.45
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


def _make_tempered_glass_mat() -> bpy.types.Material:
    """
    Pristine frameless tempered architectural glass wall behind hero (`X = +1.85 m`).
    Mandatory V12 Shadow Ray Bypass (`LightPath.Is Shadow Ray -> Transparent BSDF`)
    so Cycles (`caustics_refractive = False`) never casts black shadows!
    """
    mat = bpy.data.materials.new("Mat_Range_TemperedGlass")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Base Color", (0.78, 0.96, 1.0, 1.0))
    _set_bsdf(bsdf, "Transmission Weight", 0.92)
    _set_bsdf(bsdf, "IOR", 1.52)
    _set_bsdf(bsdf, "Roughness", 0.03)
    _set_bsdf(bsdf, "Specular IOR Level", 0.62)

    # Subtle glossy cyan-white edge sheen so glass panel reads clearly against dark range
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.22
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = 3.2
    nt.links.new(lw.outputs["Facing"], pw.inputs[0])
    rim_s = nt.nodes.new("ShaderNodeMath")
    rim_s.operation = "MULTIPLY"
    rim_s.inputs[1].default_value = 0.14  # Capped <= 0.14 per AGENTS.md Sec 3
    nt.links.new(pw.outputs["Value"], rim_s.inputs[0])
    _set_bsdf(bsdf, "Emission Color", (0.30, 0.88, 1.0, 1.0))
    nt.links.new(rim_s.outputs["Value"], bsdf.inputs["Emission Strength"])

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
    env_root: bpy.types.Object,
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

    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = env_root
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 28.0
    noise.inputs["Detail"].default_value = 5.0
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.025
    bump.inputs["Distance"].default_value = 0.0015
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _make_emissive_mat(
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


# ============================================================================
# ARCHITECTURAL RANGE GEOMETRY BUILDERS
# ============================================================================
def _build_backdrop_cyclorama(
    col: bpy.types.Collection,
    env_root: bpy.types.Object,
    mat_backdrop: bpy.types.Material,
) -> bpy.types.Object:
    bm = bmesh.new()
    nx, nz = 48, 32
    x0, x1 = -7.0, 7.0
    z0, z1 = -0.15, 8.80
    y_back = 2.65

    grid: list[list[bmesh.types.BMVert]] = []
    for ix in range(nx + 1):
        x = x0 + (x1 - x0) * (ix / nx)
        if x < -2.8:
            d = (-2.8 - x) / 2.8
            y = y_back - 1.50 * (d * d)
        elif x > 2.8:
            d = (x - 2.8) / 2.8
            y = y_back - 1.50 * (d * d)
        else:
            y = y_back
        row = [bm.verts.new((x, y, z0 + (z1 - z0) * (iz / nz))) for iz in range(nz + 1)]
        grid.append(row)

    for ix in range(nx):
        for iz in range(nz):
            bm.faces.new((grid[ix][iz], grid[ix + 1][iz], grid[ix + 1][iz + 1], grid[ix][iz + 1]))

    return _finalize_bmesh(
        bm,
        "Range_Backdrop",
        col,
        materials=[mat_backdrop],
        smooth=True,
        parent=env_root,
        visible_shadow=False,
    )


def _build_range_floor(
    col: bpy.types.Collection,
    env_root: bpy.types.Object,
    mat_floor: bpy.types.Material,
    mat_metal_dark: bpy.types.Material,
    mat_cyan_trim: bpy.types.Material,
) -> bpy.types.Object:
    fz = float(C.RANGE["floor_z"])
    bm = bmesh.new()
    # Thick concrete slab top exactly at Z = 0.00 m
    _add_box(bm, (0.0, 0.0, fz - 0.12), (13.6, 11.6, 0.24), mat_index=0)
    # Rear dark architectural baseboard curb + subtle cyan floor accent strip
    _add_box(bm, (0.0, 2.48, fz + 0.05), (10.8, 0.18, 0.10), mat_index=1, bevel_width=0.008)
    _add_box(bm, (0.0, 2.38, fz + 0.012), (9.6, 0.010, 0.010), mat_index=2)

    return _finalize_bmesh(
        bm,
        "Range_Floor",
        col,
        materials=[mat_floor, mat_metal_dark, mat_cyan_trim],
        smooth=False,
        parent=env_root,
        visible_shadow=True,
    )


def _build_glass_backdrop_wall(
    col: bpy.types.Collection,
    env_root: bpy.types.Object,
    mat_glass: bpy.types.Material,
    mat_metal_dark: bpy.types.Material,
    mat_cyan_trim: bpy.types.Material,
    total_frames: int,
) -> list[bpy.types.Object]:
    """
    Pristine frameless tempered glass studio wall at `X = +1.85 m` behind hero.
    Visible ONLY in Beats 1, 2, 6 (`1..510`, `1561..1800`). Zero center mullions!
    """
    gx = float(C.RANGE["glass_wall_x"])
    fz = float(C.RANGE["floor_z"])

    # Frameless 14mm tempered glass pane spanning Y in [-1.55, +2.05], Z in [0.06, 2.55]
    bm_g = bmesh.new()
    _add_box(bm_g, (gx, 0.25, fz + 1.30), (0.014, 3.60, 2.48), mat_index=0, bevel_width=0.002)
    glass_obj = _finalize_bmesh(
        bm_g,
        "Range_TemperedGlass_Wall",
        col,
        materials=[mat_glass],
        smooth=False,
        parent=env_root,
        visible_shadow=True,
    )

    # Top & bottom anodized clamp rails + outer perimeter posts far outside hero sightline
    bm_f = bmesh.new()
    _add_box(bm_f, (gx, 0.25, fz + 0.032), (0.064, 3.66, 0.064), mat_index=0, bevel_width=0.005)
    _add_box(bm_f, (gx, 0.25, fz + 2.57), (0.064, 3.66, 0.064), mat_index=0, bevel_width=0.005)
    # Subtle glowing glass-edge highlight strip along front edge so pane silhouette reads crisply
    _add_box(bm_f, (gx - 0.008, -1.545, fz + 1.30), (0.006, 0.006, 2.44), mat_index=1)
    _add_box(bm_f, (gx, 0.25, fz + 0.066), (0.012, 3.58, 0.005), mat_index=1)
    # Outer vertical framing posts at far Y = -1.56 & +2.06 (never blocking hero/face!)
    for py in (-1.56, 2.06):
        _add_box(bm_f, (gx, py, fz + 1.30), (0.068, 0.068, 2.60), mat_index=0, bevel_width=0.006)

    frame_obj = _finalize_bmesh(
        bm_f,
        "Range_TemperedGlass_Frame",
        col,
        materials=[mat_metal_dark, mat_cyan_trim],
        smooth=False,
        parent=env_root,
        visible_shadow=True,
    )

    vis = [(1, 510), (1561, total_frames)]
    _key_visibility(glass_obj, vis, total_frames)
    _key_visibility(frame_obj, vis, total_frames)
    return [glass_obj, frame_obj]


def _build_range_architecture(
    col: bpy.types.Collection,
    env_root: bpy.types.Object,
    mat_metal_dark: bpy.types.Material,
    mat_steel: bpy.types.Material,
    mat_cyan_trim: bpy.types.Material,
    total_frames: int,
) -> bpy.types.Object:
    """Overhead ballistic range gantry truss, acoustic ribs & illuminated range stanchions (hidden in Beat 5)."""
    bm = bmesh.new()
    # Overhead truss beams at Z = 2.95 m
    for y_beam in (-0.95, 1.45):
        _add_box(bm, (0.0, y_beam, 2.95), (6.4, 0.09, 0.09), mat_index=0, bevel_width=0.006)
    for x_cross in (-2.4, -1.2, 0.0, 1.2, 2.4):
        _add_box(bm, (x_cross, 0.25, 2.95), (0.07, 2.40, 0.07), mat_index=1, bevel_width=0.005)
        # Overhead linear LED practical strips
        _add_box(bm, (x_cross, 0.25, 2.905), (0.024, 1.60, 0.012), mat_index=2)

    # Rear acoustic vertical columns along Y = +2.32 m
    for x_col in (-2.4, -0.8, 0.8, 2.4):
        _add_box(bm, (x_col, 2.32, 1.48), (0.14, 0.12, 2.96), mat_index=0, bevel_width=0.008)
        _add_box(bm, (x_col, 2.255, 1.48), (0.014, 0.008, 2.65), mat_index=2)

    # Subtle floor impact datum laser strip directly beneath hero vest (X = 1.00 m)
    vx = float(C.RANGE["vest_surface_x"])
    _add_box(bm, (vx, 0.0, 0.004), (0.016, 1.40, 0.006), mat_index=2)

    arch_obj = _finalize_bmesh(
        bm,
        "Range_Architecture_Gantry",
        col,
        materials=[mat_metal_dark, mat_steel, mat_cyan_trim],
        smooth=False,
        parent=env_root,
        visible_shadow=True,
    )
    _key_visibility(arch_obj, [(1, 1260), (1561, total_frames)], total_frames)
    return arch_obj


# ============================================================================
# WORLD, THREE-POINT STUDIO LIGHTING, SOFTBOX EYE CATCHLIGHTS & IMPACT FLASH
# ============================================================================
def _build_world_and_lights(
    col: bpy.types.Collection,
    env_root: bpy.types.Object,
    total_frames: int,
) -> list[bpy.types.Object]:
    scene = bpy.context.scene
    world = bpy.data.worlds.new("Range_Env_World")
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
    cr.color_ramp.elements[0].color = (0.008, 0.013, 0.024, 1.0)
    cr.color_ramp.elements[1].position = 0.85
    cr.color_ramp.elements[1].color = (0.018, 0.036, 0.068, 1.0)
    nt.links.new(sep.outputs["Z"], cr.inputs["Fac"])
    nt.links.new(cr.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.24
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
        obj.parent = env_root
        return _link_obj(obj, col)

    # 1. Cool Cyan Architectural Rim AREA Light (separates hero & armor silhouette cleanly)
    rim_light = _add_area(
        "Range_Rim_CoolCyan",
        loc=(0.65, 1.65, 2.38),
        target=(0.95, 0.0, 1.32),
        energy=125.0,
        color=(0.34, 0.82, 1.0),
        size=1.85,
        diffuse_fac=0.85,
        specular_fac=1.18,
    )

    # 2. Warm Overhead Studio Key AREA Light
    key_light = _add_area(
        "Range_Key_WarmStudio",
        loc=(0.15, -1.45, 2.65),
        target=(0.85, 0.0, 1.28),
        energy=145.0,
        color=(1.0, 0.94, 0.86),
        size=1.55,
        diffuse_fac=0.95,
        specular_fac=0.95,
    )

    # 3. Shooter Key Fill (illuminates shooter cleanly in Beat 2 & Beat 5)
    shooter_key = _add_area(
        "Range_Shooter_Key",
        loc=(-1.45, -1.65, 2.35),
        target=(-1.35, 0.0, 1.32),
        energy=95.0,
        color=(0.96, 0.95, 0.98),
        size=1.35,
        diffuse_fac=0.90,
        specular_fac=0.85,
    )

    # 4. Macro Kevlar/Ceramic Side-Rim Accent Light (boosted across Beats 3-4 so golden weave gleams!)
    macro_rim = _add_area(
        "Range_Macro_VestRim",
        loc=(0.62, -0.58, 1.72),
        target=(1.02, 0.0, 1.32),
        energy=0.0,
        color=(1.0, 0.92, 0.78),
        size=0.48,
        diffuse_fac=0.90,
        specular_fac=1.35,
    )
    for f, e_val in ((1, 0.0), (508, 0.0), (514, 24.0), (900, 26.0), (1256, 24.0), (1262, 0.0), (total_frames, 0.0)):
        if 1 <= f <= total_frames:
            macro_rim.data.energy = e_val
            macro_rim.data.keyframe_insert("energy", frame=f)
    _set_fcurve_interp(macro_rim.data, "LINEAR")

    # 5. Mandatory V12 0.42m Softbox AREA Eye Catchlight for Hero (aimed at eye height Z=1.62m)
    hero_eye_catch = _add_area(
        "Hero_Softbox_EyeCatchlight",
        loc=(0.25, -1.18, 1.66),
        target=(1.15, 0.0, float(C.HERO["eye_z"])),
        energy=58.0,
        color=(1.0, 0.96, 0.92),
        size=0.42,
        diffuse_fac=0.14,
        specular_fac=1.78,
    )

    # 6. Mandatory V12 0.42m Softbox AREA Eye Catchlight for Shooter (aimed at eye height Z=1.62m)
    shooter_eye_catch = _add_area(
        "Shooter_Softbox_EyeCatchlight",
        loc=(-0.75, -1.25, 1.66),
        target=(-1.60, 0.0, float(C.SHOOTER["eye_z"])),
        energy=54.0,
        color=(1.0, 0.96, 0.92),
        size=0.42,
        diffuse_fac=0.14,
        specular_fac=1.78,
    )
    for f, loc, tgt in (
        (1, (-0.75, -1.25, 1.66), (-1.60, 0.0, float(C.SHOOTER["eye_z"]))),
        (510, (-0.75, -1.25, 1.66), (-1.60, 0.0, float(C.SHOOTER["eye_z"]))),
        (1261, (-0.45, -1.25, 1.66), (-1.20, 0.0, float(C.SHOOTER["eye_z"]))),
        (total_frames, (-0.45, -1.25, 1.66), (-1.20, 0.0, float(C.SHOOTER["eye_z"]))),
    ):
        if 1 <= f <= total_frames:
            shooter_eye_catch.location = Vector(loc)
            shooter_eye_catch.rotation_euler = _aim_euler(loc, tgt)
            shooter_eye_catch.keyframe_insert("location", frame=f)
            shooter_eye_catch.keyframe_insert("rotation_euler", frame=f)
    _set_fcurve_interp(shooter_eye_catch, "CONSTANT")

    # 7. Beat 5 Studio Blueprint Front Fill Light synced cleanly to C.blueprint_bg_weight(f)
    bp_fill = _add_area(
        "Range_Blueprint_StudioFill",
        loc=(0.0, -2.80, 1.50),
        target=(0.0, 0.0, 1.25),
        energy=0.0,
        color=(0.86, 0.96, 1.0),
        size=2.60,
        diffuse_fac=0.82,
        specular_fac=0.65,
    )
    bp_frames = sorted({1, 1260, *range(1261, 1290, 2), 1410, *range(1536, 1564, 2), total_frames})
    for f in bp_frames:
        if 1 <= f <= total_frames:
            bp_fill.data.energy = 85.0 * C.blueprint_bg_weight(float(f))
            bp_fill.data.keyframe_insert("energy", frame=f)
    _set_fcurve_interp(bp_fill.data, "LINEAR")

    # 8. Dynamic Warm Spark / Impact Flash Light at vest strike face (1.00, -0.14, 1.32)
    ld_imp = bpy.data.lights.new("Range_ImpactSpark_LightData", type="POINT")
    ld_imp.energy = 0.0
    ld_imp.color = (1.0, 0.56, 0.14)
    ld_imp.shadow_soft_size = 0.12
    imp_light = bpy.data.objects.new("Range_ImpactSpark_Light", ld_imp)
    imp_light.location = Vector((float(C.RANGE["vest_surface_x"]) - 0.14, -0.16, float(C.RANGE["chest_z"])))
    imp_light.parent = env_root
    _link_obj(imp_light, col)

    imp_frames = sorted({
        1,
        *range(72, 146, 3),
        *range(332, 386, 3),
        *range(962, 1044, 3),
        total_frames,
    })
    for f in imp_frames:
        if 1 <= f <= total_frames:
            ld_imp.energy = 42.0 * C.spark_burst_t(float(f))
            ld_imp.keyframe_insert("energy", frame=f)
    _set_fcurve_interp(ld_imp, "LINEAR")

    return [
        rim_light,
        key_light,
        shooter_key,
        macro_rim,
        hero_eye_catch,
        shooter_eye_catch,
        bp_fill,
        imp_light,
    ]


def _animate_blueprint_nodes(bp_nodes: list[bpy.types.Node], total_frames: int) -> None:
    frames = sorted({1, 1258, 1260, *range(1261, 1292, 2), 1410, *range(1536, 1566, 2), total_frames})
    for node in bp_nodes:
        for f in frames:
            if 1 <= f <= total_frames:
                node.outputs[0].default_value = C.blueprint_bg_weight(float(f))
                node.outputs[0].keyframe_insert("default_value", frame=f)
        if node.id_data and node.id_data.animation_data:
            _set_fcurve_interp(node.id_data, "LINEAR")


# ============================================================================
# PUBLIC ENTRYPOINT
# ============================================================================
def build_range_env(
    col: bpy.types.Collection,
    total_frames: int = C.TOTAL_FRAMES,
) -> dict[str, Any]:
    """Assemble the complete dual-mode ballistic range studio, glass wall, and lights."""
    env_root = bpy.data.objects.new("Range_Env_Root", None)
    env_root.empty_display_type = "PLAIN_AXES"
    env_root.empty_display_size = 0.25
    _link_obj(env_root, col)

    bp_nodes: list[bpy.types.Node] = []
    mat_backdrop = _make_backdrop_mat(env_root, bp_nodes)
    mat_floor = _make_floor_mat(env_root, bp_nodes)
    mat_glass = _make_tempered_glass_mat()
    mat_metal_dark = _make_metal_mat("Mat_Range_DarkAnodized", (0.042, 0.048, 0.060), 0.82, 0.34, env_root)
    mat_steel = _make_metal_mat("Mat_Range_BrushedSteel", (0.28, 0.31, 0.36), 0.90, 0.28, env_root)
    mat_cyan_trim = _make_emissive_mat("Mat_Range_CyanTrim", (0.08, 0.82, 1.0), 1.45)

    backdrop = _build_backdrop_cyclorama(col, env_root, mat_backdrop)
    floor = _build_range_floor(col, env_root, mat_floor, mat_metal_dark, mat_cyan_trim)
    glass_objs = _build_glass_backdrop_wall(col, env_root, mat_glass, mat_metal_dark, mat_cyan_trim, total_frames)
    arch = _build_range_architecture(col, env_root, mat_metal_dark, mat_steel, mat_cyan_trim, total_frames)
    lights = _build_world_and_lights(col, env_root, total_frames)

    _animate_blueprint_nodes(bp_nodes, total_frames)

    return {
        "env_root": env_root,
        "backdrop": backdrop,
        "floor": floor,
        "glass_wall": glass_objs[0],
        "glass_frame": glass_objs[1],
        "architecture": arch,
        "bp_nodes": bp_nodes,
        "lights": lights,
    }
