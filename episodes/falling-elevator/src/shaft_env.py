"""
shaft_env.py — High-Rise Elevator Shaft, Guide Rails, Pit Buffers & Cinematic Lighting
Episode #16: "Why Jumping in a Falling Elevator WON'T Save You" (60 fps, 1800 frames)

Public API:
    build_environment(col: bpy.types.Collection, total_frames: int = 1800) -> dict

Returned dict keys:
    - "root": bpy.types.Object (Env_Shaft_Root Empty)
    - "shaft_root": bpy.types.Object (alias for Env_Shaft_Root)
    - "scroll_root": bpy.types.Object (Env_Shaft_Scroll animated via C.shaft_scroll_z)
    - "t_rails": bpy.types.Object (Env_Shaft_TRails at X = +-C.CAB["rail_x"])
    - "buffers": list[bpy.types.Object] (Pit hydraulic buffer cylinders, springs & caps at X = +-C.CAB["buffer_x"])
    - "world": bpy.types.World (Env_Shaft_World)
    - "lights": dict[str, bpy.types.Object] (includes mandatory 0.42m AREA eye catchlight)
    - "pit_floor": bpy.types.Object (Env_Pit_Floor)
    - "sheave": bpy.types.Object (Env_Shaft_Sheave at Z = C.SHAFT["ceiling_sheave_z"])
    - "scroll_bays": list[bpy.types.Object] (seamless wrapped story bays)
    - "speed_streaks": list[bpy.types.Object] (open-ended additive wind ribbons)
    - "materials": dict[str, bpy.types.Material]
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

_THIS_FILE = Path(__file__).resolve()
_EPISODE_SRC = str(_THIS_FILE.parent)
_REPO_ROOT_SRC = str(_THIS_FILE.parents[3] / "src")
for _p in (_EPISODE_SRC, _REPO_ROOT_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import elevator_contract as C  # noqa: E402


# ============================================================================
# SHARED SCENE, COLLECTION, BMESH & ANIMATION HELPERS
# ============================================================================
def _link_obj(obj: bpy.types.Object, col: bpy.types.Collection | None = None) -> bpy.types.Object:
    target = col if col is not None else bpy.context.scene.collection
    for c in list(obj.users_collection):
        if c != target:
            c.objects.unlink(obj)
    if obj.name not in target.objects:
        target.objects.link(obj)
    return obj


def _set_bsdf(bsdf: bpy.types.Node, name: str, val) -> None:
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = val


def _rgba_mix_sockets(mix_node: bpy.types.Node):
    rgba_in = [s for s in mix_node.inputs if s.type == "RGBA"]
    rgba_out = [s for s in mix_node.outputs if s.type == "RGBA"]
    return mix_node.inputs[0], rgba_in[0], rgba_in[1], rgba_out[0]


def _aim_euler(src, dst) -> Euler:
    d = Vector(dst) - Vector(src)
    if d.length < 1e-6:
        return Euler((0.0, 0.0, 0.0), "XYZ")
    return d.normalized().to_track_quat("-Z", "Y").to_euler("XYZ")


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
) -> bpy.types.Object:
    """Convert BMesh -> Mesh -> Object with AGENTS.md Sec 3 material-slot-before-to_mesh rule."""
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
    if parent is not None:
        obj.parent = parent
    _link_obj(obj, col)
    return obj


def _set_fcurve_interp(anim_target, interp: str = "LINEAR", prefix: str | None = None) -> None:
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
    """Keyframe hide_viewport & hide_render strictly with CONSTANT interpolation."""
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

    _set_fcurve_interp(obj, "CONSTANT", prefix="hide_")


def _add_box(
    bm: bmesh.types.BMesh,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    mat_index: int = 0,
    rot_euler: tuple[float, float, float] | None = None,
    bevel_width: float = 0.0,
) -> list[bmesh.types.BMFace]:
    """Create a box (optionally rotated & beveled) inside an existing BMesh."""
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
    faces = []
    for f in tmp.faces:
        nf = bm.faces.new([vert_map[v] for v in f.verts])
        nf.material_index = mat_index
        faces.append(nf)
    tmp.free()
    return faces


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
    """Create a cylinder/cone segment from p0 to p1 inside an existing BMesh."""
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


def _add_ibeam_z(
    bm: bmesh.types.BMesh,
    cx: float,
    cy: float,
    z0: float,
    z1: float,
    width_x: float,
    depth_y: float,
    flange_t: float,
    web_t: float,
    mat_index: int = 0,
) -> None:
    """Create a vertical I-beam along Z centered at (cx, cy)."""
    h = z1 - z0
    cz = 0.5 * (z0 + z1)
    # Front and back flanges along X, web along Y
    _add_box(bm, (cx, cy - 0.5 * (depth_y - flange_t), cz), (width_x, flange_t, h), mat_index=mat_index)
    _add_box(bm, (cx, cy + 0.5 * (depth_y - flange_t), cz), (width_x, flange_t, h), mat_index=mat_index)
    _add_box(bm, (cx, cy, cz), (web_t, max(0.01, depth_y - 2.0 * flange_t), h), mat_index=mat_index)


# ============================================================================
# LOCAL-SPACE ARCHITECTURAL PBR, X-RAY CUTAWAY & SPEED STREAK SHADERS
# ============================================================================
def _make_concrete_shaft_mat(
    name: str,
    coord_obj: bpy.types.Object,
    base_rgb: tuple[float, float, float] = (0.044, 0.048, 0.056),
) -> bpy.types.Material:
    """
    Rich dark charcoal architectural concrete PBR with procedural formwork panel grooves
    and vertical shaft grime streaks (local-space TexCoord.Object; NEVER chalky white).
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Metallic", 0.04)
    _set_bsdf(bsdf, "Specular IOR Level", 0.28)

    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = coord_obj

    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])

    # Formwork panel grooves across X+Y vs Z
    comb_panel = nt.nodes.new("ShaderNodeCombineXYZ")
    add_xy = nt.nodes.new("ShaderNodeMath")
    add_xy.operation = "ADD"
    nt.links.new(sep.outputs["X"], add_xy.inputs[0])
    nt.links.new(sep.outputs["Y"], add_xy.inputs[1])
    nt.links.new(add_xy.outputs["Value"], comb_panel.inputs["X"])
    nt.links.new(sep.outputs["Z"], comb_panel.inputs["Y"])

    brick = nt.nodes.new("ShaderNodeTexBrick")
    brick.offset = 0.0
    brick.inputs["Scale"].default_value = 0.72
    brick.inputs["Mortar Size"].default_value = 0.014
    brick.inputs["Mortar Smooth"].default_value = 0.30
    brick.inputs["Brick Width"].default_value = 1.15
    brick.inputs["Row Height"].default_value = 0.80
    brick.inputs["Color1"].default_value = (1.0, 1.0, 1.0, 1.0)
    brick.inputs["Color2"].default_value = (0.90, 0.91, 0.93, 1.0)
    brick.inputs["Mortar"].default_value = (0.38, 0.40, 0.44, 1.0)
    nt.links.new(comb_panel.outputs["Vector"], brick.inputs["Vector"])

    # Vertical grime/oil streak mapping (stretched in Z)
    mp_grime = nt.nodes.new("ShaderNodeMapping")
    mp_grime.inputs["Scale"].default_value = (2.4, 2.4, 0.42)
    nt.links.new(tc.outputs["Object"], mp_grime.inputs["Vector"])

    n_grime = nt.nodes.new("ShaderNodeTexNoise")
    n_grime.inputs["Scale"].default_value = 1.8
    n_grime.inputs["Detail"].default_value = 6.0
    n_grime.inputs["Roughness"].default_value = 0.65
    nt.links.new(mp_grime.outputs["Vector"], n_grime.inputs["Vector"])

    n_fine = nt.nodes.new("ShaderNodeTexNoise")
    n_fine.inputs["Scale"].default_value = 22.0
    n_fine.inputs["Detail"].default_value = 7.0
    n_fine.inputs["Roughness"].default_value = 0.72
    nt.links.new(tc.outputs["Object"], n_fine.inputs["Vector"])

    mix_tone = nt.nodes.new("ShaderNodeMix")
    mix_tone.data_type = "RGBA"
    fac_t, ca_t, cb_t, out_t = _rgba_mix_sockets(mix_tone)
    ca_t.default_value = (base_rgb[0] * 0.62, base_rgb[1] * 0.64, base_rgb[2] * 0.68, 1.0)
    cb_t.default_value = (base_rgb[0] * 1.32, base_rgb[1] * 1.32, base_rgb[2] * 1.35, 1.0)
    nt.links.new(n_grime.outputs["Fac"], fac_t)

    mix_panel = nt.nodes.new("ShaderNodeMix")
    mix_panel.data_type = "RGBA"
    mix_panel.blend_type = "MULTIPLY"
    fac_p, ca_p, cb_p, out_p = _rgba_mix_sockets(mix_panel)
    fac_p.default_value = 0.70
    nt.links.new(out_t, ca_p)
    nt.links.new(brick.outputs["Color"], cb_p)
    nt.links.new(out_p, bsdf.inputs["Base Color"])

    mr_rough = nt.nodes.new("ShaderNodeMapRange")
    mr_rough.inputs["To Min"].default_value = 0.62
    mr_rough.inputs["To Max"].default_value = 0.90
    nt.links.new(n_grime.outputs["Fac"], mr_rough.inputs["Value"])
    nt.links.new(mr_rough.outputs["Result"], bsdf.inputs["Roughness"])

    bump_groove = nt.nodes.new("ShaderNodeBump")
    bump_groove.inputs["Strength"].default_value = 0.22
    bump_groove.inputs["Distance"].default_value = 0.008
    nt.links.new(brick.outputs["Fac"], bump_groove.inputs["Height"])

    bump_fine = nt.nodes.new("ShaderNodeBump")
    bump_fine.inputs["Strength"].default_value = 0.10
    bump_fine.inputs["Distance"].default_value = 0.003
    nt.links.new(n_fine.outputs["Fac"], bump_fine.inputs["Height"])
    nt.links.new(bump_groove.outputs["Normal"], bump_fine.inputs["Normal"])
    nt.links.new(bump_fine.outputs["Normal"], bsdf.inputs["Normal"])

    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _make_front_cutaway_mat(
    name: str,
    xray_value_nodes: list[bpy.types.ShaderNodeValue],
) -> bpy.types.Material:
    """
    98%+ invisible crystal-clear cutaway shader for subtle horizontal front landing cues.
    Routes LightPath.Is Shadow Ray -> Transparent BSDF unconditionally so studio lights
    are never blocked, keeps front-facing surfaces ~99.9% transparent, and caps grazing
    architectural hints <= 0.018 opacity (98.2%+ invisible) so T-rail clamps, sparks,
    and X-ray bones read with zero distracting glass bars or glowing stripes.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    tr.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)

    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.08

    pow_rim = nt.nodes.new("ShaderNodeMath")
    pow_rim.operation = "POWER"
    pow_rim.inputs[1].default_value = 5.0
    nt.links.new(lw.outputs["Facing"], pow_rim.inputs[0])

    # X-ray activation scalar (keyframed via C.xray_cab_weight(f))
    xray_val = nt.nodes.new("ShaderNodeValue")
    xray_val.name = "XRayShaftCutaway"
    xray_val.outputs[0].default_value = 0.0
    xray_value_nodes.append(xray_val)

    # Max opacity <= 0.018 (0.006 baseline + 0.012 * xray_weight) -> 98.2%-99.9% transparent
    mul_x = nt.nodes.new("ShaderNodeMath")
    mul_x.operation = "MULTIPLY"
    mul_x.inputs[1].default_value = 0.012
    nt.links.new(xray_val.outputs[0], mul_x.inputs[0])

    add_op = nt.nodes.new("ShaderNodeMath")
    add_op.operation = "ADD"
    add_op.inputs[1].default_value = 0.006
    nt.links.new(mul_x.outputs["Value"], add_op.inputs[0])

    mul_rim = nt.nodes.new("ShaderNodeMath")
    mul_rim.operation = "MULTIPLY"
    mul_rim.use_clamp = True
    nt.links.new(pow_rim.outputs["Value"], mul_rim.inputs[0])
    nt.links.new(add_op.outputs["Value"], mul_rim.inputs[1])

    # Only visible to camera rays (LightPath.Is Camera Ray), 100% transparent to shadow/diffuse/glossy!
    lp = nt.nodes.new("ShaderNodeLightPath")
    cam_gate = nt.nodes.new("ShaderNodeMath")
    cam_gate.operation = "MULTIPLY"
    cam_gate.use_clamp = True
    nt.links.new(mul_rim.outputs["Value"], cam_gate.inputs[0])
    nt.links.new(lp.outputs["Is Camera Ray"], cam_gate.inputs[1])

    # Shadow ray bypass explicitly reinforced
    shad_inv = nt.nodes.new("ShaderNodeMath")
    shad_inv.operation = "SUBTRACT"
    shad_inv.inputs[0].default_value = 1.0
    shad_inv.use_clamp = True
    nt.links.new(lp.outputs["Is Shadow Ray"], shad_inv.inputs[1])

    final_fac = nt.nodes.new("ShaderNodeMath")
    final_fac.operation = "MULTIPLY"
    final_fac.use_clamp = True
    nt.links.new(cam_gate.outputs["Value"], final_fac.inputs[0])
    nt.links.new(shad_inv.outputs["Value"], final_fac.inputs[1])

    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.18, 0.52, 0.72, 1.0)
    em.inputs["Strength"].default_value = 0.55  # Ultra-subtle AgX-safe <= 2.05

    mix_sh = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(final_fac.outputs["Value"], mix_sh.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix_sh.inputs[1])
    nt.links.new(em.outputs["Emission"], mix_sh.inputs[2])

    nt.links.new(mix_sh.outputs["Shader"], out.inputs["Surface"])
    return mat


def _make_metal_mat(
    name: str,
    rgb: tuple[float, float, float],
    metallic: float,
    roughness: float,
    coord_obj: bpy.types.Object,
    bump_strength: float = 0.06,
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
    _set_bsdf(bsdf, "Coat Weight", 0.08)

    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = coord_obj
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = 28.0
    n.inputs["Detail"].default_value = 6.0
    nt.links.new(tc.outputs["Object"], n.inputs["Vector"])

    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value = max(0.10, roughness - 0.08)
    mr.inputs["To Max"].default_value = min(0.85, roughness + 0.08)
    nt.links.new(n.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = bump_strength
    bump.inputs["Distance"].default_value = 0.002
    nt.links.new(n.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _make_hazard_stripe_mat(name: str, coord_obj: bpy.types.Object) -> bpy.types.Material:
    """Industrial yellow/black diagonal hazard safety stripe material for pit perimeter."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Metallic", 0.08)
    _set_bsdf(bsdf, "Roughness", 0.42)

    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = coord_obj

    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "DIAGONAL"
    wave.wave_profile = "SAW"
    wave.inputs["Scale"].default_value = 3.2
    wave.inputs["Distortion"].default_value = 0.0
    nt.links.new(tc.outputs["Object"], wave.inputs["Vector"])

    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.interpolation = "CONSTANT"
    cr.color_ramp.elements[0].position = 0.0
    cr.color_ramp.elements[0].color = (0.016, 0.018, 0.020, 1.0)
    cr.color_ramp.elements[1].position = 0.48
    cr.color_ramp.elements[1].color = (0.88, 0.58, 0.02, 1.0)
    nt.links.new(wave.outputs["Fac"], cr.inputs["Fac"])
    nt.links.new(cr.outputs["Color"], bsdf.inputs["Base Color"])

    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _make_emissive_strip_mat(
    name: str,
    rgb: tuple[float, float, float],
    strength: float,
) -> bpy.types.Material:
    """AgX-safe architectural shaft LED strip material (strength <= 2.05)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Base Color", (*rgb, 1.0))
    _set_bsdf(bsdf, "Roughness", 0.18)
    _set_bsdf(bsdf, "Emission Color", (*rgb, 1.0))
    _set_bsdf(bsdf, "Emission Strength", min(2.05, strength))
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _make_speed_streak_mat(name: str = "Mat_Env_SpeedStreak") -> bpy.types.Material:
    """
    Additive wispy vertical speed ribbon shader:
    Camera Ray * (1 - Backfacing) * LayerWeight * Noise -> AgX-safe pale cyan-white emission.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.70, 0.92, 1.0, 1.0)

    lp = nt.nodes.new("ShaderNodeLightPath")
    geom = nt.nodes.new("ShaderNodeNewGeometry")

    not_back = nt.nodes.new("ShaderNodeMath")
    not_back.operation = "SUBTRACT"
    not_back.inputs[0].default_value = 1.0
    not_back.use_clamp = True
    nt.links.new(geom.outputs["Backfacing"], not_back.inputs[1])

    cam_front = nt.nodes.new("ShaderNodeMath")
    cam_front.operation = "MULTIPLY"
    cam_front.use_clamp = True
    nt.links.new(lp.outputs["Is Camera Ray"], cam_front.inputs[0])
    nt.links.new(not_back.outputs["Value"], cam_front.inputs[1])

    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.42

    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (4.0, 4.0, 0.25)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])

    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 2.6
    noise.inputs["Detail"].default_value = 3.0
    nt.links.new(mp.outputs["Vector"], noise.inputs["Vector"])

    m1 = nt.nodes.new("ShaderNodeMath")
    m1.operation = "MULTIPLY"
    m1.use_clamp = True
    nt.links.new(lw.outputs["Facing"], m1.inputs[0])
    nt.links.new(noise.outputs["Fac"], m1.inputs[1])

    m2 = nt.nodes.new("ShaderNodeMath")
    m2.operation = "MULTIPLY"
    m2.use_clamp = True
    nt.links.new(cam_front.outputs["Value"], m2.inputs[0])
    nt.links.new(m1.outputs["Value"], m2.inputs[1])

    gain = nt.nodes.new("ShaderNodeMath")
    gain.operation = "MULTIPLY"
    gain.inputs[1].default_value = 1.15  # AgX-safe subtle streak glow
    nt.links.new(m2.outputs["Value"], gain.inputs[0])
    nt.links.new(gain.outputs["Value"], em.inputs["Strength"])

    add_sh = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(tr.outputs["BSDF"], add_sh.inputs[0])
    nt.links.new(em.outputs["Emission"], add_sh.inputs[1])
    nt.links.new(add_sh.outputs["Shader"], out.inputs["Surface"])
    return mat


# ============================================================================
# STATIC SHAFT STEEL: T-RAILS, BRACKETS & STRUCTURAL CORNER I-BEAMS
# ============================================================================
def _build_t_rails_and_structure(
    col: bpy.types.Collection,
    shaft_root: bpy.types.Object,
    mat_steel_bright: bpy.types.Material,
    mat_steel_dark: bpy.types.Material,
) -> tuple[bpy.types.Object, bpy.types.Object]:
    """
    Build vertical T-profile elevator guide rails at X = +-C.CAB['rail_x'] (+-0.96m), Y=0,
    spanning Z = -2.0 .. +40.0m with bolted wall brackets every 2.5m, plus 2 rear shaft I-columns.
    """
    rail_x = float(C.CAB["rail_x"])  # 0.96
    wall_x = float(C.SHAFT["half_w"])  # 1.25
    wall_y = float(C.SHAFT["half_d"])  # 1.05
    z_min, z_max = -2.0, 40.0
    z_len = z_max - z_min
    z_mid = 0.5 * (z_min + z_max)

    bm_rails = bmesh.new()
    for sgn in (-1.0, 1.0):
        # Machined T-rail blade centered around X = +-rail_x (tip +-0.920 -> flange +-1.020)
        blade_cx = sgn * (rail_x + 0.010)  # +-0.970, width 0.10 -> [0.920..1.020]
        _add_box(bm_rails, (blade_cx, 0.0, z_mid), (0.100, 0.026, z_len), mat_index=0, bevel_width=0.002)

        # T-rail base flange at X = +-(1.020..1.044)
        flange_cx = sgn * (rail_x + 0.072)  # +-1.032
        _add_box(bm_rails, (flange_cx, 0.0, z_mid), (0.024, 0.145, z_len), mat_index=1, bevel_width=0.002)

        # Bolted wall rail brackets every 2.5m from Z = -1.5 .. 38.5m
        z_br = -1.5
        while z_br <= 38.6:
            # Horizontal C-channel arm spanning flange (+-1.044) to wall (+-wall_x)
            arm_x0 = rail_x + 0.084
            arm_x1 = wall_x
            arm_cx = sgn * 0.5 * (arm_x0 + arm_x1)
            arm_w = arm_x1 - arm_x0
            _add_box(bm_rails, (arm_cx, 0.0, z_br), (arm_w, 0.175, 0.078), mat_index=1, bevel_width=0.003)

            # Diagonal lower gusset plate
            _add_box(
                bm_rails,
                (arm_cx, 0.0, z_br - 0.065),
                (arm_w * 0.85, 0.022, 0.070),
                mat_index=1,
            )

            # Rail clip plates & bolts clamping T-flange
            for by in (-0.055, 0.055):
                _add_box(
                    bm_rails,
                    (sgn * (rail_x + 0.056), by, z_br),
                    (0.036, 0.028, 0.050),
                    mat_index=0,
                )
                # Anchor bolts on wall mounting plate
                _add_cyl(
                    bm_rails,
                    (sgn * (wall_x - 0.025), by * 1.25, z_br + 0.024),
                    (sgn * (wall_x + 0.005), by * 1.25, z_br + 0.024),
                    0.011,
                    0.011,
                    segments=6,
                    mat_index=0,
                )
                _add_cyl(
                    bm_rails,
                    (sgn * (wall_x - 0.025), by * 1.25, z_br - 0.024),
                    (sgn * (wall_x + 0.005), by * 1.25, z_br - 0.024),
                    0.011,
                    0.011,
                    segments=6,
                    mat_index=0,
                )
            z_br += 2.5

    t_rails_obj = _finalize_bmesh(
        bm_rails,
        "Env_Shaft_TRails",
        col,
        materials=[mat_steel_bright, mat_steel_dark],
        smooth=False,
        parent=shaft_root,
    )

    # 2 Rear Corner Vertical Steel I-Columns (front corners kept completely open so exterior
    # camera sightlines to T-rail clamps & cab never pass through double vertical I-beam mullions)
    bm_cols = bmesh.new()
    for sx in (-1.0, 1.0):
        # Rear corner I-beams (solid structural steel)
        _add_ibeam_z(
            bm_cols,
            cx=sx * (wall_x - 0.045),
            cy=wall_y - 0.045,
            z0=z_min,
            z1=z_max,
            width_x=0.11,
            depth_y=0.11,
            flange_t=0.014,
            web_t=0.012,
            mat_index=0,
        )

    corner_beams_obj = _finalize_bmesh(
        bm_cols,
        "Env_Shaft_CornerBeams",
        col,
        materials=[mat_steel_dark],
        smooth=False,
        parent=shaft_root,
    )

    return t_rails_obj, corner_beams_obj


# ============================================================================
# SCROLLING MULTI-STORY CONCRETE SHAFT BAYS (SEAMLESS WRAPPED CAROUSEL)
# ============================================================================
def _build_single_shaft_bay_mesh(
    name: str,
    col: bpy.types.Collection,
    scroll_root: bpy.types.Object,
    bay_h: float,
    materials: list[bpy.types.Material],
) -> bpy.types.Object:
    """
    Build one architectural multi-story concrete/steel shaft floor bay (height = bay_h = 3.20m):
    - Mat 0: Rich charcoal concrete (rear wall Y=+1.05, rear-side return wings Y=+0.35..+1.05)
    - Mat 1: Dark structural steel (horizontal divider I-beams, conduits, sconce housings)
    - Mat 2: Brushed stainless steel (sill plate, door landing edge)
    - Mat 3: Warm amber-white architectural LED wall sconce strips
    - Mat 4: Ultra-clear 98%+ invisible horizontal front landing datum cues (no vertical piers/bars)
    """
    wall_x = float(C.SHAFT["half_w"])  # 1.25
    wall_y = float(C.SHAFT["half_d"])  # 1.05
    bm = bmesh.new()

    # 1. Rear concrete shaft wall (Y = +1.05 .. +1.20)
    _add_box(bm, (0.0, wall_y + 0.075, 0.5 * bay_h), (2.0 * wall_x + 0.18, 0.15, bay_h), mat_index=0)

    # 2. Rear-side concrete side-wall wings (X = +-1.25, Y = +0.35 .. +1.05) -> keeps front 3/4 view open!
    side_rear_len = wall_y - 0.35  # 0.70m
    side_rear_cy = 0.35 + 0.5 * side_rear_len
    for sgn in (-1.0, 1.0):
        _add_box(
            bm,
            (sgn * (wall_x + 0.07), side_rear_cy, 0.5 * bay_h),
            (0.14, side_rear_len, bay_h),
            mat_index=0,
        )

    # 3. Horizontal steel floor-divider beam & concrete spandrel ledge on rear + side wings
    _add_box(bm, (0.0, wall_y - 0.035, 0.08), (2.0 * wall_x - 0.16, 0.09, 0.16), mat_index=1)
    for sgn in (-1.0, 1.0):
        _add_box(
            bm,
            (sgn * (wall_x - 0.035), side_rear_cy, 0.08),
            (0.08, side_rear_len, 0.16),
            mat_index=1,
        )

    # 4. Vertical electrical conduit pipes + junction boxes on rear concrete wall
    for cx in (-0.64, 0.64):
        _add_cyl(bm, (cx, wall_y - 0.02, 0.0), (cx, wall_y - 0.02, bay_h), 0.016, 0.016, segments=10, mat_index=1)
        _add_box(bm, (cx, wall_y - 0.032, 1.35), (0.11, 0.045, 0.16), mat_index=1, bevel_width=0.004)

    # 5. Twin Warm Amber-White Architectural Shaft LED Sconce Strips (rear corners)
    for sgn in (-1.0, 1.0):
        sx = sgn * (wall_x - 0.14)
        sy = wall_y - 0.045
        # Dark anodized aluminum sconce channel
        _add_box(bm, (sx, sy, 1.55), (0.048, 0.034, 1.72), mat_index=1, bevel_width=0.003)
        # Warm glowing LED diffuser lens
        _add_box(bm, (sx, sy - 0.014, 1.55), (0.024, 0.012, 1.64), mat_index=3)
        # Brushed steel horizontal floor datum bar at Z=2.50m
        _add_box(bm, (sgn * (wall_x - 0.22), wall_y - 0.03, 2.52), (0.28, 0.018, 0.08), mat_index=2, bevel_width=0.002)

    # 6. Front Cutaway Horizontal Floor-Datum Cues Only (Mat 4: 98%+ crystal-clear Transparent BSDF)
    # Front cutaway side walls (Y=-1.05..+0.35) and vertical side landing piers (X=+-0.62..1.17) are
    # intentionally omitted so Beat 5a (frame_1285 right T-rail clamp & SAFETY CLAMPS label) and
    # Beats 2/4 have zero distracting double vertical glass mullions/stripes across camera sightlines.
    # Keep only slim horizontal landing sill & upper datum bar as ultra-subtle floor-spacing hints.
    _add_box(bm, (0.0, -wall_y + 0.02, 0.02), (2.0 * wall_x - 0.24, 0.04, 0.035), mat_index=4)
    _add_box(bm, (0.0, -wall_y + 0.02, bay_h - 0.025), (2.0 * wall_x - 0.24, 0.04, 0.045), mat_index=4)

    obj = _finalize_bmesh(
        bm,
        name,
        col,
        materials=materials,
        smooth=False,
        parent=scroll_root,
    )
    return obj


def _build_scrolling_shaft_bays(
    col: bpy.types.Collection,
    shaft_root: bpy.types.Object,
    total_frames: int,
    mat_steel_dark: bpy.types.Material,
    mat_steel_bright: bpy.types.Material,
    mat_led_warm: bpy.types.Material,
    mat_front_cutaway: bpy.types.Material,
) -> tuple[bpy.types.Object, list[bpy.types.Object]]:
    """
    Create Env_Shaft_Scroll parented to shaft_root, animated smoothly upward by C.shaft_scroll_z(f),
    and 12 modular story bays (BAY_H = 3.20m, LOOP_H = 38.40m) parented to Env_Shaft_Scroll with
    seamless modulo wrapping inside world Z in [-1.10, +37.30] so nothing ever pokes below the pit!
    """
    scroll_root = bpy.data.objects.new("Env_Shaft_Scroll", None)
    scroll_root.empty_display_type = "PLAIN_AXES"
    scroll_root.empty_display_size = 0.50
    scroll_root.location = Vector((0.0, 0.0, 0.0))
    scroll_root.parent = shaft_root
    _link_obj(scroll_root, col)

    for f in range(1, total_frames + 1):
        scroll_root.location = Vector((0.0, 0.0, C.shaft_scroll_z(float(f))))
        scroll_root.keyframe_insert("location", frame=f)
    _set_fcurve_interp(scroll_root, "LINEAR", prefix="location")

    mat_scroll_concrete = _make_concrete_shaft_mat(
        "Mat_Env_Shaft_Concrete", scroll_root, base_rgb=(0.048, 0.052, 0.060)
    )
    bay_materials = [
        mat_scroll_concrete,
        mat_steel_dark,
        mat_steel_bright,
        mat_led_warm,
        mat_front_cutaway,
    ]

    bay_h = 3.20
    num_bays = 12
    loop_h = bay_h * num_bays  # 38.40 m
    z_min = -1.10

    bays: list[bpy.types.Object] = []
    for idx in range(num_bays):
        bay_obj = _build_single_shaft_bay_mesh(
            f"Env_Shaft_Bay_{idx + 1:02d}",
            col,
            scroll_root,
            bay_h,
            bay_materials,
        )
        prev_k: int | None = None
        wrap_frames: list[int] = []
        for f in range(1, total_frames + 1):
            s = C.shaft_scroll_z(float(f))
            total_shift = idx * bay_h + s
            k = int(math.floor(total_shift / loop_h))
            local_z = z_min + idx * bay_h - k * loop_h
            if prev_k is None:
                bay_obj.location = Vector((0.0, 0.0, local_z))
                bay_obj.keyframe_insert("location", frame=f)
                prev_k = k
            elif k != prev_k:
                old_local_z = z_min + idx * bay_h - prev_k * loop_h
                bay_obj.location = Vector((0.0, 0.0, old_local_z))
                bay_obj.keyframe_insert("location", frame=f - 1)
                bay_obj.location = Vector((0.0, 0.0, local_z))
                bay_obj.keyframe_insert("location", frame=f)
                wrap_frames.append(f - 1)
                prev_k = k

        bay_obj.location = Vector((0.0, 0.0, z_min + idx * bay_h - (prev_k or 0) * loop_h))
        bay_obj.keyframe_insert("location", frame=total_frames)

        ad = getattr(bay_obj, "animation_data", None)
        if ad and ad.action:
            wrap_set = set(wrap_frames)
            for fc in C.iter_action_fcurves(ad.action):
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT" if int(round(kp.co.x)) in wrap_set else "LINEAR"

        bays.append(bay_obj)

    return scroll_root, bays


# ============================================================================
# SHAFT PIT FLOOR & TWIN HYDRAULIC BUFFER SPRINGS (C.SHAFT["pit_floor_z"])
# ============================================================================
def _build_coil_spring_bmesh(
    bm: bmesh.types.BMesh,
    z0: float,
    z1: float,
    coil_r: float,
    wire_r: float,
    turns: float = 8.5,
    steps_per_turn: int = 18,
    ring_segs: int = 10,
    mat_index: int = 0,
) -> None:
    """Loft a clean closed-surface helical coil spring from z0 to z1 inside BMesh."""
    n_pts = int(turns * steps_per_turn) + 1
    rings: list[list[bmesh.types.BMVert]] = []

    for i in range(n_pts):
        u = i / float(n_pts - 1)
        theta = u * turns * 2.0 * math.pi
        z = z0 + u * (z1 - z0)
        cx = coil_r * math.cos(theta)
        cy = coil_r * math.sin(theta)
        center = Vector((cx, cy, z))

        # Tangent vector along helix
        dz_dtheta = (z1 - z0) / (turns * 2.0 * math.pi)
        tang = Vector((-coil_r * math.sin(theta), coil_r * math.cos(theta), dz_dtheta)).normalized()
        radial = Vector((math.cos(theta), math.sin(theta), 0.0)).normalized()
        binorm = tang.cross(radial).normalized()

        ring = []
        for j in range(ring_segs):
            phi = (j / float(ring_segs)) * 2.0 * math.pi
            pt = center + wire_r * (math.cos(phi) * radial + math.sin(phi) * binorm)
            ring.append(bm.verts.new(pt))
        rings.append(ring)

    for i in range(len(rings) - 1):
        r0 = rings[i]
        r1 = rings[i + 1]
        for j in range(ring_segs):
            jn = (j + 1) % ring_segs
            f = bm.faces.new([r0[j], r0[jn], r1[jn], r1[j]])
            f.material_index = mat_index

    # Cap start and end wire faces
    f_start = bm.faces.new(list(reversed(rings[0])))
    f_start.material_index = mat_index
    f_end = bm.faces.new(rings[-1])
    f_end.material_index = mat_index


def _build_pit_and_buffers(
    col: bpy.types.Collection,
    shaft_root: bpy.types.Object,
    total_frames: int,
    mat_concrete: bpy.types.Material,
    mat_hazard: bpy.types.Material,
    mat_steel_dark: bpy.types.Material,
    mat_steel_bright: bpy.types.Material,
    mat_yellow_paint: bpy.types.Material,
    mat_rubber: bpy.types.Material,
) -> tuple[bpy.types.Object, list[bpy.types.Object]]:
    """
    Build pit concrete slab at Z = C.SHAFT['pit_floor_z'] (-1.15m), hazard yellow/black
    perimeter stripe, steel tie-beams, and twin hydraulic pit buffer springs at
    X = +-C.CAB['buffer_x'] (+-0.42m), Y = 0.0 compressing -0.26m in sync with C.cab_world_z(f).
    """
    pit_z = float(C.SHAFT["pit_floor_z"])        # -1.15
    buf_top_rest = float(C.SHAFT["buffer_top_z"])  # -0.02
    buf_x = float(C.CAB["buffer_x"])              # 0.42
    wall_x = float(C.SHAFT["half_w"])             # 1.25
    wall_y = float(C.SHAFT["half_d"])             # 1.05

    # 1. Pit Concrete Floor Slab + Hazard Stripes + Tie Beams
    bm_pit = bmesh.new()
    # Main pit slab (top at pit_z = -1.15m, depth 0.42m)
    _add_box(bm_pit, (0.0, 0.0, pit_z - 0.21), (2.0 * wall_x + 0.36, 2.0 * wall_y + 0.36, 0.42), mat_index=0)

    # Low concrete pit curb apron along rear + sides (never blocks front camera)
    _add_box(bm_pit, (0.0, wall_y - 0.04, pit_z + 0.04), (2.0 * wall_x, 0.10, 0.08), mat_index=0)
    for sgn in (-1.0, 1.0):
        _add_box(bm_pit, (sgn * (wall_x - 0.04), 0.25, pit_z + 0.04), (0.10, 1.50, 0.08), mat_index=0)

    # Hazard yellow/black concrete pit perimeter safety strips at Z = pit_z + 0.004
    _add_box(bm_pit, (0.0, -wall_y + 0.14, pit_z + 0.004), (2.0 * wall_x - 0.18, 0.16, 0.008), mat_index=1)
    _add_box(bm_pit, (0.0, wall_y - 0.16, pit_z + 0.004), (2.0 * wall_x - 0.18, 0.16, 0.008), mat_index=1)
    for sgn in (-1.0, 1.0):
        _add_box(bm_pit, (sgn * (wall_x - 0.16), 0.0, pit_z + 0.004), (0.16, 2.0 * wall_y - 0.44, 0.008), mat_index=1)

    # Heavy structural steel pit channel tie-beam linking rail bases and buffer pedestals
    _add_box(bm_pit, (0.0, 0.0, pit_z + 0.032), (2.0 * wall_x - 0.18, 0.24, 0.064), mat_index=2, bevel_width=0.004)

    # Pit sump drain grate at (0.0, 0.58, pit_z + 0.005)
    _add_box(bm_pit, (0.0, 0.58, pit_z + 0.005), (0.34, 0.28, 0.012), mat_index=2)

    pit_floor_obj = _finalize_bmesh(
        bm_pit,
        "Env_Pit_Floor",
        col,
        materials=[mat_concrete, mat_hazard, mat_steel_dark],
        smooth=False,
        parent=shaft_root,
    )

    # 2. Twin Hydraulic Pit Buffers at X = +-0.42, Y = 0.0
    z_anchor = -0.62  # Fixed shoulder where cylinder meets active spring/piston stroke
    rest_stroke = buf_top_rest - z_anchor  # -0.02 - (-0.62) = 0.60 m

    buffers: list[bpy.types.Object] = []
    spring_objs: list[bpy.types.Object] = []
    cap_objs: list[bpy.types.Object] = []

    for side_tag, bx in (("L", -buf_x), ("R", buf_x)):
        # A) Fixed Hydraulic Cylinder Base & Bolted Pedestal (Z = pit_z .. -0.62)
        bm_cyl = bmesh.new()
        _add_box(bm_cyl, (bx, 0.0, pit_z + 0.082), (0.28, 0.28, 0.036), mat_index=1, bevel_width=0.003)
        for ox in (-0.105, 0.105):
            for oy in (-0.105, 0.105):
                _add_cyl(bm_cyl, (bx + ox, oy, pit_z + 0.10), (bx + ox, oy, pit_z + 0.122), 0.013, 0.013, segments=6, mat_index=2)
        # Main industrial yellow hydraulic cylinder barrel
        _add_cyl(bm_cyl, (bx, 0.0, pit_z + 0.10), (bx, 0.0, z_anchor - 0.03), 0.072, 0.072, segments=24, mat_index=0)
        # Dark steel collar flange at z_anchor
        _add_cyl(bm_cyl, (bx, 0.0, z_anchor - 0.03), (bx, 0.0, z_anchor), 0.112, 0.112, segments=24, mat_index=1)

        cyl_obj = _finalize_bmesh(
            bm_cyl,
            f"Env_Pit_Buffer_Cyl_{side_tag}",
            col,
            materials=[mat_yellow_paint, mat_steel_dark, mat_steel_bright],
            smooth=True,
            parent=shaft_root,
        )
        buffers.append(cyl_obj)

        # B) Compressing Coil Spring & Inner Chrome Plunger (origin at (bx, 0.0, z_anchor))
        bm_spr = bmesh.new()
        _add_cyl(bm_spr, (0.0, 0.0, 0.0), (0.0, 0.0, rest_stroke - 0.05), 0.036, 0.036, segments=20, mat_index=1)
        _build_coil_spring_bmesh(
            bm_spr,
            z0=0.015,
            z1=rest_stroke - 0.065,
            coil_r=0.086,
            wire_r=0.019,
            turns=8.5,
            steps_per_turn=16,
            ring_segs=10,
            mat_index=0,
        )
        spr_obj = _finalize_bmesh(
            bm_spr,
            f"Env_Pit_Buffer_Spring_{side_tag}",
            col,
            materials=[mat_steel_dark, mat_steel_bright],
            smooth=True,
            parent=shaft_root,
            location=(bx, 0.0, z_anchor),
        )
        spring_objs.append(spr_obj)
        buffers.append(spr_obj)

        # C) Floating Impact Strike Pad & Upper Spring Seat (origin at (bx, 0.0, buf_top_rest))
        bm_cap = bmesh.new()
        _add_cyl(bm_cap, (0.0, 0.0, -0.068), (0.0, 0.0, -0.042), 0.112, 0.112, segments=24, mat_index=0)
        _add_cyl(bm_cap, (0.0, 0.0, -0.042), (0.0, 0.0, 0.0), 0.104, 0.098, segments=24, mat_index=1)
        cap_obj = _finalize_bmesh(
            bm_cap,
            f"Env_Pit_Buffer_Cap_{side_tag}",
            col,
            materials=[mat_steel_dark, mat_rubber],
            smooth=True,
            parent=shaft_root,
            location=(bx, 0.0, buf_top_rest),
        )
        cap_objs.append(cap_obj)
        buffers.append(cap_obj)

    # Animate spring compression & strike cap translation in exact 0.000mm sync with C.cab_world_z(f)
    # Active compression window f = 935..1236; static uncompressed elsewhere.
    for obj in spring_objs:
        obj.scale = Vector((1.0, 1.0, 1.0))
        obj.keyframe_insert("scale", frame=1)
        obj.keyframe_insert("scale", frame=935)
    for obj in cap_objs:
        obj.location.z = buf_top_rest
        obj.keyframe_insert("location", frame=1)
        obj.keyframe_insert("location", frame=935)

    for f in range(936, 1237):
        cab_z = C.cab_world_z(float(f))
        comp_offset = min(0.0, cab_z)  # -0.26m peak at f=985
        top_z = C.clamp(buf_top_rest + comp_offset, buf_top_rest - 0.26, buf_top_rest)
        scale_z = (top_z - z_anchor) / rest_stroke

        for obj in spring_objs:
            obj.scale = Vector((1.0, 1.0, scale_z))
            obj.keyframe_insert("scale", frame=f)
        for obj in cap_objs:
            obj.location.z = top_z
            obj.keyframe_insert("location", frame=f)

    for obj in spring_objs:
        obj.scale = Vector((1.0, 1.0, 1.0))
        obj.keyframe_insert("scale", frame=1237)
        obj.keyframe_insert("scale", frame=total_frames)
        _set_fcurve_interp(obj, "LINEAR", prefix="scale")

    for obj in cap_objs:
        obj.location.z = buf_top_rest
        obj.keyframe_insert("location", frame=1237)
        obj.keyframe_insert("location", frame=total_frames)
        _set_fcurve_interp(obj, "LINEAR", prefix="location")

    return pit_floor_obj, buffers


# ============================================================================
# OVERHEAD TRACTION SHEAVE DRUM & SPEED WIND STREAKS
# ============================================================================
def _build_traction_sheave(
    col: bpy.types.Collection,
    shaft_root: bpy.types.Object,
    total_frames: int,
    mat_steel_dark: bpy.types.Material,
    mat_steel_bright: bpy.types.Material,
    mat_motor_blue: bpy.types.Material,
) -> bpy.types.Object:
    """
    Build overhead traction sheave wheel + structural steel machine beam at Z = C.SHAFT['ceiling_sheave_z'] (27.5m).
    """
    sheave_z = float(C.SHAFT["ceiling_sheave_z"])  # 27.5
    wall_x = float(C.SHAFT["half_w"])              # 1.25

    # Machine room steel I-beam support frame + traction motor (static, parented to shaft_root)
    bm_frame = bmesh.new()
    for by in (-0.34, 0.34):
        _add_box(bm_frame, (0.0, by, sheave_z - 0.56), (2.0 * wall_x, 0.14, 0.22), mat_index=0, bevel_width=0.004)
    # Cross bolster plates + pillow block bearings flanking sheave at X = +-0.24
    for sx in (-0.24, 0.24):
        _add_box(bm_frame, (sx, 0.0, sheave_z - 0.26), (0.12, 0.78, 0.42), mat_index=0, bevel_width=0.004)
        _add_cyl(bm_frame, (sx - 0.07, 0.0, sheave_z), (sx + 0.07, 0.0, sheave_z), 0.11, 0.11, segments=18, mat_index=1)
    # Traction drive motor housing on -X side
    _add_cyl(bm_frame, (-0.88, 0.0, sheave_z), (-0.34, 0.0, sheave_z), 0.26, 0.26, segments=24, mat_index=2)
    # Drive shaft spanning motor through sheave hub
    _add_cyl(bm_frame, (-0.38, 0.0, sheave_z), (0.30, 0.0, sheave_z), 0.065, 0.065, segments=18, mat_index=1)

    _finalize_bmesh(
        bm_frame,
        "Env_Shaft_SheaveFrame",
        col,
        materials=[mat_steel_dark, mat_steel_bright, mat_motor_blue],
        smooth=True,
        parent=shaft_root,
    )

    # Rotating multi-groove traction sheave drum centered at (0, 0, sheave_z)
    bm_wheel = bmesh.new()
    wheel_r = 0.48
    # Central hub + spoked flange plates
    _add_cyl(bm_wheel, (-0.17, 0.0, 0.0), (0.17, 0.0, 0.0), 0.14, 0.14, segments=24, mat_index=0)
    for ox in (-0.12, 0.12):
        _add_cyl(bm_wheel, (ox - 0.015, 0.0, 0.0), (ox + 0.015, 0.0, 0.0), wheel_r - 0.03, wheel_r - 0.03, segments=32, mat_index=0)
    # 5 outer cable-groove rim ribs (4 V-grooves for the 4 hoist cables)
    for gx in (-0.14, -0.07, 0.0, 0.07, 0.14):
        _add_cyl(bm_wheel, (gx - 0.022, 0.0, 0.0), (gx + 0.022, 0.0, 0.0), wheel_r, wheel_r, segments=36, mat_index=1)
    # Radial ribs so drum spin is visibly legible
    for k in range(6):
        ang = k * (math.pi / 3.0)
        p_out = (0.0, 0.40 * math.cos(ang), 0.40 * math.sin(ang))
        _add_cyl(bm_wheel, (0.0, 0.0, 0.0), p_out, 0.028, 0.024, segments=8, mat_index=0)

    sheave_obj = _finalize_bmesh(
        bm_wheel,
        "Env_Shaft_Sheave",
        col,
        materials=[mat_steel_dark, mat_steel_bright],
        smooth=True,
        parent=shaft_root,
        location=(0.0, 0.0, sheave_z),
    )

    # Rotate wheel smoothly during Beat 1 descent, then freewheel burst on cable snap f=325 -> coast down
    rot_x = 0.0
    for f in range(1, total_frames + 1, 5):
        if f <= 325:
            omega = 0.16 if f >= 100 else 0.04
        elif f <= 460:
            omega = 0.34 * (1.0 - C.smoothstep(325.0, 460.0, float(f)))
        else:
            omega = 0.0
        rot_x += omega * 5.0
        sheave_obj.rotation_euler = Euler((rot_x, 0.0, 0.0), "XYZ")
        sheave_obj.keyframe_insert("rotation_euler", frame=f)
    _set_fcurve_interp(sheave_obj, "LINEAR", prefix="rotation_euler")

    return sheave_obj


def _build_speed_streaks(
    col: bpy.types.Collection,
    shaft_root: bpy.types.Object,
    total_frames: int,
    mat_streak: bpy.types.Material,
) -> list[bpy.types.Object]:
    """
    Build 16 lightweight open-ended vertical speed wind streak ribbons (cap_ends=False,
    visible_shadow=False) scrolling rapidly +Z around the outer shaft perimeter during
    Beats 2-3 (f=323..954) and Beat 5a (f=1237..1345) to sell 50 mph freefall.
    """
    rng = random.Random(1616)
    streaks: list[bpy.types.Object] = []

    # Perimeter coordinates strictly outside cab (|X| > 0.88 or Y > 0.76) so center view stays clean
    perimeter_xy = [
        (-1.08, -0.45), (-1.12, 0.18), (-1.04, 0.68), (-0.88, 0.88),
        (-0.42, 0.92), (0.04, 0.94), (0.48, 0.91), (0.89, 0.87),
        (1.06, 0.64), (1.13, 0.15), (1.07, -0.42), (-0.98, -0.68),
        (0.98, -0.66), (-1.10, 0.45), (1.10, 0.42), (0.0, 0.96),
    ]

    for idx, (px, py) in enumerate(perimeter_xy):
        length_z = rng.uniform(2.4, 4.2)
        radius = rng.uniform(0.014, 0.026)

        # Open-ended tapered tube (cap_ends=False -> zero flat end-caps!)
        bm = bmesh.new()
        _add_cyl(
            bm,
            (0.0, 0.0, -0.5 * length_z),
            (0.0, 0.0, 0.5 * length_z),
            radius * 0.35,
            radius,
            segments=8,
            cap_ends=False,
            mat_index=0,
        )
        obj = _finalize_bmesh(
            bm,
            f"Env_SpeedStreak_{idx + 1:02d}",
            col,
            materials=[mat_streak],
            smooth=True,
            recalc_normals=False,
            parent=shaft_root,
        )
        obj.visible_shadow = False

        # Scroll +Z past cab_world_z(f) in Beats 2-3 (323..954) and Beat 5a (1237..1345)
        phase = rng.uniform(0.0, 1.0)
        speed_f = rng.uniform(0.42, 0.62)  # meters per frame upward rush
        span_z = 9.0
        vis_ranges = [(323, 954), (1237, 1345)]

        for s_f, e_f in vis_ranges:
            prev_u: float | None = None
            wrap_frames: list[int] = []
            for f in range(s_f, e_f + 1, 2):
                cab_z = C.cab_world_z(float(f))
                raw = phase + (f - s_f) * speed_f / span_z
                u = raw % 1.0
                z_pos = max(-0.4, cab_z - 2.2 + u * span_z)
                if prev_u is not None and u < prev_u:
                    # Seamless wrap: CONSTANT hold at previous frame
                    old_z = max(-0.4, C.cab_world_z(float(f - 2)) - 2.2 + prev_u * span_z)
                    obj.location = Vector((px, py, old_z))
                    obj.keyframe_insert("location", frame=f - 2)
                    wrap_frames.append(f - 2)

                obj.location = Vector((px, py, z_pos))
                obj.keyframe_insert("location", frame=f)

                # Smooth fade in/out scale at window edges
                if s_f == 323:
                    fade = C.smoothstep(323.0, 345.0, float(f)) * (1.0 - C.smoothstep(932.0, 954.0, float(f)))
                else:
                    fade = C.smoothstep(1237.0, 1255.0, float(f)) * (1.0 - C.smoothstep(1320.0, 1345.0, float(f)))
                obj.scale = Vector((1.0, 1.0, max(0.01, fade)))
                obj.keyframe_insert("scale", frame=f)
                prev_u = u

            ad = getattr(obj, "animation_data", None)
            if ad and ad.action:
                wset = set(wrap_frames)
                for fc in C.iter_action_fcurves(ad.action):
                    if fc.data_path.startswith("location"):
                        for kp in fc.keyframe_points:
                            kp.interpolation = "CONSTANT" if int(round(kp.co.x)) in wset else "LINEAR"
                    elif fc.data_path.startswith("scale"):
                        for kp in fc.keyframe_points:
                            kp.interpolation = "LINEAR"

        _key_visibility(obj, vis_ranges, total_frames)
        streaks.append(obj)

    return streaks


# ============================================================================
# WORLD AMBIANCE & STUDIO LIGHTING (AgX - PUNCHY + MANDATORY EYE CATCHLIGHT)
# ============================================================================
def _build_world_and_lights(
    col: bpy.types.Collection,
    shaft_root: bpy.types.Object,
    total_frames: int,
) -> tuple[bpy.types.World, dict[str, bpy.types.Object]]:
    """
    Build moody skyscraper night/dusk architectural shaft world and balanced Cycles studio
    lighting (AgX exposure=0.0 safe), including mandatory AGENTS.md Sec 1 0.42m AREA
    softbox eye catchlight tracking C.cab_world_z(f).
    """
    scene = bpy.context.scene
    world = bpy.data.worlds.new("Env_Shaft_World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")

    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])

    # Deep atmospheric slate-indigo dusk/night shaft backdrop (zero chalky/void blowout)
    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].position = 0.25
    cr.color_ramp.elements[0].color = (0.012, 0.018, 0.032, 1.0)
    cr.color_ramp.elements[1].position = 0.82
    cr.color_ramp.elements[1].color = (0.038, 0.062, 0.112, 1.0)
    nt.links.new(sep.outputs["Z"], cr.inputs["Fac"])
    nt.links.new(cr.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.42

    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    if scene.render.engine == "CYCLES":
        scene.cycles.transparent_max_bounces = max(14, scene.cycles.transparent_max_bounces)

    def _add_area_light(
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
        obj.parent = shaft_root
        _link_obj(obj, col)
        return obj

    z0 = C.cab_world_z(1.0)  # 18.0m

    # 1. MANDATORY AGENTS.md Sec 1 Softbox Eye Catchlight (0.42m AREA, 52W, diff=0.10, spec=1.78)
    eye_catch = _add_area_light(
        "Env_Passenger_EyeCatchlight",
        loc=(0.55, -1.85, z0 + 1.85),
        target=(0.0, 0.0, z0 + 1.65),
        energy=52.0,
        color=(1.0, 0.96, 0.92),
        size=0.42,
        diffuse_fac=0.10,
        specular_fac=1.78,
    )

    # 2. Balanced Cab Front-Left Warm Key Light (sculpts face & jacket cleanly through glass front)
    cab_warm_key = _add_area_light(
        "Env_Cab_WarmKey",
        loc=(-0.55, -1.55, z0 + 2.25),
        target=(-0.05, 0.05, z0 + 1.15),
        energy=36.0,
        color=(1.0, 0.89, 0.74),
        size=1.35,
        diffuse_fac=0.92,
        specular_fac=0.65,
    )

    # 3. Gentle Interior Cab Ceiling Warm Bounce (illuminates cab interior cleanly in Beat 1 solid cab)
    cab_ceiling_fill = _add_area_light(
        "Env_Cab_CeilingFill",
        loc=(0.0, 0.02, z0 + 2.24),
        target=(0.0, 0.04, z0 + 0.40),
        energy=14.0,
        color=(1.0, 0.92, 0.80),
        size=0.90,
        diffuse_fac=0.85,
        specular_fac=0.35,
    )

    # 4. Cool Cerulean Shaft Rim Light (separates cab frame, T-rails & passenger silhouette from dark shaft)
    shaft_cool_rim = _add_area_light(
        "Env_Shaft_CoolRim",
        loc=(1.60, 0.85, z0 + 2.55),
        target=(0.0, 0.0, z0 + 1.05),
        energy=48.0,
        color=(0.48, 0.76, 1.0),
        size=1.65,
        diffuse_fac=0.88,
        specular_fac=0.95,
    )

    # 5. Beat 4 X-Ray Skeleton Rim Light (active f=905..1236 near pit Z=0)
    xray_rim = _add_area_light(
        "Env_XRay_Beat4Rim",
        loc=(-1.65, -0.95, 1.65),
        target=(0.0, 0.0, 0.85),
        energy=0.0,
        color=(0.35, 0.86, 1.0),
        size=1.80,
        diffuse_fac=0.75,
        specular_fac=1.10,
    )

    # 6. Beat 5-6 Brake-Inspection & Floor-Grid Hero Fill Light (Env_Brake_Spot)
    brake_spot = _add_area_light(
        "Env_Brake_Spot",
        loc=(1.45, -1.50, 2.35),
        target=(0.92, 0.0, 2.26),
        energy=0.0,
        color=(1.0, 0.94, 0.86),
        size=0.85,
        diffuse_fac=0.95,
        specular_fac=0.85,
    )

    # 7. Pit Hydraulic Buffer Accent Fill (f=900..1800)
    pit_fill = _add_area_light(
        "Env_Pit_SafetyFill",
        loc=(0.0, -1.65, 0.30),
        target=(0.0, 0.0, -0.58),
        energy=0.0,
        color=(1.0, 0.84, 0.56),
        size=1.25,
        diffuse_fac=0.90,
        specular_fac=0.70,
    )

    # Animate tracking lights smoothly along C.cab_world_z(f)
    for f in range(1, total_frames + 1, 2):
        cz = C.cab_world_z(float(f))

        eye_catch.location = Vector((0.55, -1.85, cz + 1.85))
        eye_catch.rotation_euler = _aim_euler((0.55, -1.85, cz + 1.85), (0.0, 0.0, cz + 1.65))
        eye_catch.keyframe_insert("location", frame=f)
        eye_catch.keyframe_insert("rotation_euler", frame=f)

        cab_warm_key.location = Vector((-0.55, -1.55, cz + 2.25))
        cab_warm_key.rotation_euler = _aim_euler((-0.55, -1.55, cz + 2.25), (-0.05, 0.05, cz + 1.15))
        cab_warm_key.keyframe_insert("location", frame=f)
        cab_warm_key.keyframe_insert("rotation_euler", frame=f)

        cab_ceiling_fill.location = Vector((0.0, 0.02, cz + 2.24))
        cab_ceiling_fill.keyframe_insert("location", frame=f)

        shaft_cool_rim.location = Vector((1.60, 0.85, cz + 2.55))
        shaft_cool_rim.rotation_euler = _aim_euler((1.60, 0.85, cz + 2.55), (0.0, 0.0, cz + 1.05))
        shaft_cool_rim.keyframe_insert("location", frame=f)
        shaft_cool_rim.keyframe_insert("rotation_euler", frame=f)

        # Beat 5a (1237..1385) tracks safety clamps at cz - 0.14; Beat 5b-6 (1386..1800) illuminates flat floor grid
        if 1237 <= f <= 1385:
            b_loc = (1.45, -1.50, cz + 0.15)
            b_tgt = (0.92, 0.0, cz - 0.14)
        else:
            b_loc = (0.65, -1.60, 1.95)
            b_tgt = (0.0, 0.0, 0.14)
        brake_spot.location = Vector(b_loc)
        brake_spot.rotation_euler = _aim_euler(b_loc, b_tgt)
        brake_spot.keyframe_insert("location", frame=f)
        brake_spot.keyframe_insert("rotation_euler", frame=f)

    # Ensure exact end frame 1800 keyframes
    for obj in (eye_catch, cab_warm_key, cab_ceiling_fill, shaft_cool_rim, brake_spot):
        _set_fcurve_interp(obj, "LINEAR", prefix="location")
        _set_fcurve_interp(obj, "LINEAR", prefix="rotation_euler")

    # Energy gating for Beat 4 X-ray rim, Beat 5-6 brake spot, and pit fill
    for f, e_xray, e_brake, e_pit in [
        (1, 0.0, 0.0, 0.0),
        (900, 0.0, 0.0, 6.0),
        (930, 56.0, 0.0, 18.0),
        (1210, 56.0, 0.0, 18.0),
        (1236, 0.0, 0.0, 12.0),
        (1237, 0.0, 44.0, 8.0),
        (1385, 0.0, 44.0, 8.0),
        (1386, 0.0, 22.0, 14.0),
        (1800, 0.0, 22.0, 14.0),
    ]:
        xray_rim.data.energy = e_xray
        xray_rim.data.keyframe_insert("energy", frame=f)
        brake_spot.data.energy = e_brake
        brake_spot.data.keyframe_insert("energy", frame=f)
        pit_fill.data.energy = e_pit
        pit_fill.data.keyframe_insert("energy", frame=f)

    for ld in (xray_rim.data, brake_spot.data, pit_fill.data):
        _set_fcurve_interp(ld, "LINEAR")

    return world, {
        "eye_catchlight": eye_catch,
        "cab_warm_key": cab_warm_key,
        "cab_ceiling_fill": cab_ceiling_fill,
        "shaft_cool_rim": shaft_cool_rim,
        "xray_rim": xray_rim,
        "brake_spot": brake_spot,
        "pit_fill": pit_fill,
    }


def _animate_xray_cutaway_nodes(
    xray_value_nodes: list[bpy.types.ShaderNodeValue],
    total_frames: int,
) -> None:
    """Keyframe front-shaft cutaway rim intensity directly from C.xray_cab_weight(f)."""
    frames = sorted(
        {
            1, 339, *range(340, 366, 2), *range(575, 597, 2), 600,
            904, *range(905, 933, 2), *range(1212, 1238, 2), 1238,
            1534, *range(1535, 1567, 2), total_frames,
        }
    )
    for node in xray_value_nodes:
        for f in frames:
            if 1 <= f <= total_frames:
                node.outputs[0].default_value = C.xray_cab_weight(float(f))
                node.outputs[0].keyframe_insert("default_value", frame=f)
        _set_fcurve_interp(node.id_data, "LINEAR")


# ============================================================================
# PUBLIC ENTRYPOINT
# ============================================================================
def build_environment(col: bpy.types.Collection, total_frames: int = C.TOTAL_FRAMES) -> dict:
    """
    Build complete high-rise elevator shaft environment:
    - Vertical T-profile guide rails at X = +-0.96m with bolted brackets every 2.5m
    - Seamless scrolling multi-story charcoal concrete bays & warm LED wall sconces
    - Concrete pit slab at Z = -1.15m with hazard stripes & compressing twin hydraulic buffers
    - Overhead traction sheave at Z = 27.5m & additive wispy speed wind ribbons
    - Moody skyscraper night/dusk world + mandatory 0.42m AREA eye catchlight
    """
    if col is None:
        col = bpy.context.scene.collection

    shaft_root = bpy.data.objects.new("Env_Shaft_Root", None)
    shaft_root.empty_display_type = "PLAIN_AXES"
    shaft_root.empty_display_size = 0.75
    shaft_root.location = Vector((0.0, 0.0, 0.0))
    _link_obj(shaft_root, col)

    xray_value_nodes: list[bpy.types.ShaderNodeValue] = []
    mat_front_cutaway = _make_front_cutaway_mat("Mat_Env_Shaft_FrontCutaway", xray_value_nodes)
    mat_steel_dark = _make_metal_mat("Mat_Env_Shaft_SteelDark", (0.045, 0.050, 0.058), 0.86, 0.40, shaft_root)
    mat_steel_bright = _make_metal_mat("Mat_Env_Shaft_SteelBright", (0.34, 0.36, 0.40), 0.94, 0.22, shaft_root, bump_strength=0.03)
    mat_motor_blue = _make_metal_mat("Mat_Env_Shaft_MotorBlue", (0.04, 0.10, 0.18), 0.55, 0.42, shaft_root)
    mat_yellow_paint = _make_metal_mat("Mat_Env_Pit_SafetyYellow", (0.78, 0.50, 0.02), 0.14, 0.36, shaft_root)
    mat_rubber = _make_metal_mat("Mat_Env_Pit_Rubber", (0.020, 0.022, 0.024), 0.02, 0.78, shaft_root, bump_strength=0.10)
    mat_pit_concrete = _make_concrete_shaft_mat("Mat_Env_Pit_Concrete", shaft_root, base_rgb=(0.042, 0.046, 0.054))
    mat_hazard = _make_hazard_stripe_mat("Mat_Env_Pit_Hazard", shaft_root)
    mat_led_warm = _make_emissive_strip_mat("Mat_Env_Shaft_LEDWarm", (1.0, 0.72, 0.34), strength=1.65)
    mat_streak = _make_speed_streak_mat("Mat_Env_SpeedStreak")

    # 1. Static T-Rails, Brackets & Rear Corner I-Columns
    t_rails_obj, corner_beams_obj = _build_t_rails_and_structure(
        col, shaft_root, mat_steel_bright, mat_steel_dark
    )

    # 2. Scrolling Multi-Story Shaft Bays (bound to Env_Shaft_Scroll local coordinates)
    scroll_root, scroll_bays = _build_scrolling_shaft_bays(
        col,
        shaft_root,
        total_frames,
        mat_steel_dark=mat_steel_dark,
        mat_steel_bright=mat_steel_bright,
        mat_led_warm=mat_led_warm,
        mat_front_cutaway=mat_front_cutaway,
    )

    # 3. Pit Floor & Synchronized Compressing Twin Hydraulic Buffers
    pit_floor_obj, buffers = _build_pit_and_buffers(
        col,
        shaft_root,
        total_frames,
        mat_pit_concrete,
        mat_hazard,
        mat_steel_dark,
        mat_steel_bright,
        mat_yellow_paint,
        mat_rubber,
    )

    # 4. Overhead Traction Sheave Drum & Speed Wind Streaks
    sheave_obj = _build_traction_sheave(
        col, shaft_root, total_frames, mat_steel_dark, mat_steel_bright, mat_motor_blue
    )
    speed_streaks = _build_speed_streaks(col, shaft_root, total_frames, mat_streak)

    # 5. Animate Front-Shaft Cutaway & Build World + Studio Lights
    _animate_xray_cutaway_nodes(xray_value_nodes, total_frames)
    world, lights = _build_world_and_lights(col, shaft_root, total_frames)

    return {
        "root": shaft_root,
        "shaft_root": shaft_root,
        "scroll_root": scroll_root,
        "t_rails": t_rails_obj,
        "corner_beams": corner_beams_obj,
        "buffers": buffers,
        "pit_floor": pit_floor_obj,
        "sheave": sheave_obj,
        "scroll_bays": scroll_bays,
        "speed_streaks": speed_streaks,
        "world": world,
        "lights": lights,
        "xray_value_nodes": xray_value_nodes,
        "materials": {
            "front_cutaway": mat_front_cutaway,
            "steel_dark": mat_steel_dark,
            "steel_bright": mat_steel_bright,
            "pit_concrete": mat_pit_concrete,
            "hazard": mat_hazard,
            "led_warm": mat_led_warm,
            "speed_streak": mat_streak,
        },
    }
