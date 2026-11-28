"""
bullet_fx.py — Ballistic Hero Projectiles, Barrel/Muzzle Flash, X-Ray Jacket & Core,
Supercavitation Cone, Hydrodynamic Streamlines, Shatter Shards & Dual-Lane Race FX
for Episode #2 ("Why Bullets Stop Dead Just 3 Feet Underwater", 60 fps, 1800 frames).

Strictly follows AGENTS.md V12 & Blender 5.3.0 rules:
- Imports all dimensions, milestones, trajectories, and curves from `bullet_contract as C`.
- Uses `C.iter_action_fcurves(action)` for Blender 5.x Slotted Actions.
- Attaches `mesh.materials` BEFORE `bm.to_mesh(mesh)`.
- Indexes `ShaderNodeMix` inputs strictly from `node.inputs[2:]` (never `inputs[0]` Factor).
- Open-ended optical/cavitation cones & curved tubes (`cap_ends=False`, zero flat end-caps).
- Optical beams/bubbles use `ShaderNodeAddShader` (`Transparent` + `Emission * CameraRay * (1 - Backfacing)`).
- 100% Local-Space shaders (`ShaderNodeTexCoord.Object`).

Public API:
    build_bullet_fx(col: bpy.types.Collection, total_frames: int = C.TOTAL_FRAMES) -> dict[str, Any]
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any

import bmesh
import bpy
from mathutils import Euler, Vector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import bullet_contract as C  # noqa: E402

# 2.2x cinematic macro scale for hero supersonic projectile (~6.5 cm)
HERO_BULLET_LEN: float = 0.065
HERO_BULLET_R: float = 0.0088

# Heavy chunky subsonic round-nose projectile for Beat 5 Bottom Lane
SUBSONIC_LEN: float = 0.056
SUBSONIC_R: float = 0.0114


# ---------------------------------------------------------------------------
# BLENDER 5.3.0 COLLECTION, MESH, SHADER & KEYFRAME HELPERS
# ---------------------------------------------------------------------------
def _link_to_col(obj: bpy.types.Object, col: bpy.types.Collection) -> None:
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)


def _create_empty(
    name: str,
    col: bpy.types.Collection,
    parent: bpy.types.Object | None = None,
) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = "PLAIN_AXES"
    obj.empty_display_size = 0.03
    _link_to_col(obj, col)
    if parent is not None:
        obj.parent = parent
    obj.rotation_mode = "QUATERNION"
    return obj


def _finalize_bmesh(
    name: str,
    bm: bmesh.types.BMesh,
    mat: bpy.types.Material,
    col: bpy.types.Collection,
    parent: bpy.types.Object | None = None,
    smooth: bool = True,
    shadow: bool = False,
) -> bpy.types.Object:
    """Attach material BEFORE bm.to_mesh(mesh) per Blender 5.3.0 / AGENTS.md V12."""
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for f in bm.faces:
        f.smooth = smooth
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.materials.append(mat)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    _link_to_col(obj, col)
    if parent is not None:
        obj.parent = parent
    obj.rotation_mode = "QUATERNION"
    obj.visible_shadow = shadow
    return obj


def _get_mix_sockets(mix_node: bpy.types.Node, sock_type: str) -> tuple[Any, Any, Any, Any]:
    """Blender 5.3.0 safe ShaderNodeMix socket lookup: skips Factor inputs[0..1]."""
    typed_in = [s for s in mix_node.inputs[2:] if s.type == sock_type]
    typed_out = [s for s in mix_node.outputs if s.type == sock_type]
    return mix_node.inputs[0], typed_in[0], typed_in[1], typed_out[0]


def _set_visibility_windows(
    obj: bpy.types.Object,
    windows: list[tuple[int, int]],
    total_frames: int = C.TOTAL_FRAMES,
) -> None:
    """Keyframe hide_viewport & hide_render strictly CONSTANT across inclusive [start, end] windows."""
    if not windows:
        obj.hide_viewport = True
        obj.hide_render = True
        return

    vis_at_1 = any(s <= 1 <= e for s, e in windows)
    keys: dict[int, bool] = {1: not vis_at_1}
    for s, e in sorted(windows):
        if s > 1:
            keys[s] = False
        if e < total_frames:
            keys[e + 1] = True

    for fr in sorted(keys):
        hidden = keys[fr]
        for attr in ("hide_viewport", "hide_render"):
            setattr(obj, attr, hidden)
            obj.keyframe_insert(data_path=attr, frame=fr)

    if obj.animation_data and obj.animation_data.action:
        for fc in C.iter_action_fcurves(obj.animation_data.action):
            if fc.data_path in ("hide_viewport", "hide_render"):
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"


def _set_action_interp(anim_data: Any, data_paths: tuple[str, ...], interp: str = "LINEAR") -> None:
    if not (anim_data and anim_data.action):
        return
    for fc in C.iter_action_fcurves(anim_data.action):
        if not data_paths or any(fc.data_path == p or fc.data_path.endswith(f".{p}") for p in data_paths):
            for kp in fc.keyframe_points:
                kp.interpolation = interp


def _sample_frames(
    start: int,
    end: int,
    step: int = 4,
    dense_ranges: tuple[tuple[int, int, int], ...] = (),
) -> list[int]:
    """Generate sorted frame list with base step plus dense 1-2f windows on impacts/cuts."""
    frames: set[int] = {start, end}
    for f in range(start, end + 1, max(1, step)):
        frames.add(f)
    for ds, de, dstep in dense_ranges:
        lo = max(start, ds)
        hi = min(end, de)
        for f in range(lo, hi + 1, max(1, dstep)):
            frames.add(f)
    return sorted(frames)


# ---------------------------------------------------------------------------
# PROCEDURAL BMESH GEOMETRY BUILDERS (OPEN-ENDED CONES & SWEPT TUBES)
# ---------------------------------------------------------------------------
def _add_revolved_profile(
    bm: bmesh.types.BMesh,
    profile_xr: list[tuple[float, float]],
    segs: int = 32,
    cap_start: bool = False,
    cap_end: bool = False,
    radial_mod_fn: Any = None,
) -> list[list[bmesh.types.BMVert]]:
    """Revolve an (x, radius) profile around +X axis. Returns rings for optional shaping."""
    rings: list[list[bmesh.types.BMVert]] = []
    for ring_idx, (x, r) in enumerate(profile_xr):
        ring: list[bmesh.types.BMVert] = []
        for s in range(segs):
            ang = (2.0 * math.pi * s) / segs
            rr = r
            if radial_mod_fn is not None:
                rr = radial_mod_fn(x, r, ang, ring_idx, s)
            v = bm.verts.new((x, rr * math.cos(ang), rr * math.sin(ang)))
            ring.append(v)
        rings.append(ring)

    for i in range(len(rings) - 1):
        r0 = rings[i]
        r1 = rings[i + 1]
        for s in range(segs):
            sn = (s + 1) % segs
            bm.faces.new((r0[s], r0[sn], r1[sn], r1[s]))

    if cap_start and rings:
        bm.faces.new(list(reversed(rings[0])))
    if cap_end and rings:
        bm.faces.new(rings[-1])
    return rings


def _add_swept_tube(
    bm: bmesh.types.BMesh,
    points: list[Vector],
    radii: list[float],
    segs: int = 10,
    cap_ends: bool = False,
) -> None:
    """Sweep an open-ended (cap_ends=False) smooth 3D tube along points."""
    if len(points) < 2:
        return
    rings: list[list[bmesh.types.BMVert]] = []
    prev_n = Vector((0.0, 0.0, 1.0))
    for idx, pt in enumerate(points):
        if idx == 0:
            tangent = (points[1] - points[0]).normalized()
        elif idx == len(points) - 1:
            tangent = (points[-1] - points[-2]).normalized()
        else:
            tangent = (points[idx + 1] - points[idx - 1]).normalized()
        if tangent.length < 1e-6:
            tangent = Vector((1.0, 0.0, 0.0))

        # Parallel transport frame to avoid twist kinks
        n = prev_n - tangent * prev_n.dot(tangent)
        if n.length < 1e-5:
            alt = Vector((0.0, 1.0, 0.0)) if abs(tangent.y) < 0.8 else Vector((1.0, 0.0, 0.0))
            n = alt - tangent * alt.dot(tangent)
        n.normalize()
        b = tangent.cross(n).normalized()
        prev_n = n

        rad = radii[idx]
        ring: list[bmesh.types.BMVert] = []
        for s in range(segs):
            ang = (2.0 * math.pi * s) / segs
            offset = n * (rad * math.cos(ang)) + b * (rad * math.sin(ang))
            ring.append(bm.verts.new(pt + offset))
        rings.append(ring)

    for i in range(len(rings) - 1):
        r0 = rings[i]
        r1 = rings[i + 1]
        for s in range(segs):
            sn = (s + 1) % segs
            bm.faces.new((r0[s], r0[sn], r1[sn], r1[s]))

    if cap_ends and rings:
        bm.faces.new(list(reversed(rings[0])))
        bm.faces.new(rings[-1])


# ---------------------------------------------------------------------------
# LOCAL-SPACE SHADERS (BLENDER 5.3.0 SAFE)
# ---------------------------------------------------------------------------
def _mat_gunmetal_barrel(name: str, local_obj: bpy.types.Object) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (280, 0)
    bsdf.inputs["Base Color"].default_value = (0.068, 0.075, 0.086, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.94
    bsdf.inputs["Roughness"].default_value = 0.25

    tex = nt.nodes.new("ShaderNodeTexCoord")
    tex.location = (-520, -120)
    tex.object = local_obj

    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.location = (-320, -120)
    mapping.inputs["Scale"].default_value = (2.5, 48.0, 48.0)

    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-120, -120)
    noise.inputs["Scale"].default_value = 18.0
    noise.inputs["Detail"].default_value = 6.0

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (80, -180)
    ramp.color_ramp.elements[0].position = 0.25
    ramp.color_ramp.elements[0].color = (0.19, 0.19, 0.19, 1.0)
    ramp.color_ramp.elements[1].position = 0.80
    ramp.color_ramp.elements[1].color = (0.34, 0.34, 0.34, 1.0)

    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (80, -320)
    bump.inputs["Strength"].default_value = 0.06
    bump.inputs["Distance"].default_value = 0.001

    nt.links.new(tex.outputs["Object"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Roughness"])
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _mat_muzzle_flash(
    name: str,
    flash_obj: bpy.types.Object,
) -> tuple[bpy.types.Material, bpy.types.Node]:
    """Open-ended AddShader muzzle flash with local axial falloff + turbulent core."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (980, 0)

    tex = nt.nodes.new("ShaderNodeTexCoord")
    tex.location = (-960, 0)
    tex.object = flash_obj

    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (-740, 140)
    nt.links.new(tex.outputs["Object"], sep.inputs["Vector"])

    # Fade from bright root (x=0.0) smoothly to 0 before flame tip (x=0.10)
    axial = nt.nodes.new("ShaderNodeMapRange")
    axial.location = (-520, 140)
    axial.clamp = True
    axial.inputs["From Min"].default_value = 0.0
    axial.inputs["From Max"].default_value = 0.10
    axial.inputs["To Min"].default_value = 1.0
    axial.inputs["To Max"].default_value = 0.0
    nt.links.new(sep.outputs["X"], axial.inputs["Value"])

    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-740, -140)
    noise.inputs["Scale"].default_value = 24.0
    noise.inputs["Detail"].default_value = 5.0
    nt.links.new(tex.outputs["Object"], noise.inputs["Vector"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-440, -140)
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (1.0, 0.14, 0.004, 1.0)
    ramp.color_ramp.elements[1].position = 0.85
    ramp.color_ramp.elements[1].color = (1.0, 0.44, 0.04, 1.0)
    nt.links.new(axial.outputs["Result"], ramp.inputs["Fac"])

    # CameraRay * (1 - Backfacing)
    lp = nt.nodes.new("ShaderNodeLightPath")
    lp.location = (-440, 380)
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    geom.location = (-440, 220)

    inv_back = nt.nodes.new("ShaderNodeMath")
    inv_back.operation = "SUBTRACT"
    inv_back.location = (-220, 220)
    inv_back.inputs[0].default_value = 1.0
    nt.links.new(geom.outputs["Backfacing"], inv_back.inputs[1])

    cam_front = nt.nodes.new("ShaderNodeMath")
    cam_front.operation = "MULTIPLY"
    cam_front.location = (-20, 280)
    nt.links.new(lp.outputs["Is Camera Ray"], cam_front.inputs[0])
    nt.links.new(inv_back.outputs["Value"], cam_front.inputs[1])

    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.location = (-440, -340)
    lw.inputs["Blend"].default_value = 0.42

    inv_face = nt.nodes.new("ShaderNodeMath")
    inv_face.operation = "SUBTRACT"
    inv_face.location = (-220, -340)
    inv_face.inputs[0].default_value = 1.0
    nt.links.new(lw.outputs["Facing"], inv_face.inputs[1])

    soft_edge = nt.nodes.new("ShaderNodeMath")
    soft_edge.operation = "POWER"
    soft_edge.location = (-100, -340)
    soft_edge.inputs[1].default_value = 1.8
    nt.links.new(inv_face.outputs["Value"], soft_edge.inputs[0])

    turb = nt.nodes.new("ShaderNodeMath")
    turb.operation = "MULTIPLY"
    turb.location = (-220, -40)
    nt.links.new(axial.outputs["Result"], turb.inputs[0])
    nt.links.new(noise.outputs["Fac"], turb.inputs[1])

    soft = nt.nodes.new("ShaderNodeMath")
    soft.operation = "MULTIPLY"
    soft.location = (-20, -140)
    nt.links.new(turb.outputs["Value"], soft.inputs[0])
    nt.links.new(soft_edge.outputs["Value"], soft.inputs[1])

    mask = nt.nodes.new("ShaderNodeMath")
    mask.operation = "MULTIPLY"
    mask.location = (180, 120)
    nt.links.new(cam_front.outputs["Value"], mask.inputs[0])
    nt.links.new(soft.outputs["Value"], mask.inputs[1])

    flash_val = nt.nodes.new("ShaderNodeValue")
    flash_val.name = "FlashIntensity"
    flash_val.location = (-20, -360)
    flash_val.outputs[0].default_value = 0.0

    strength = nt.nodes.new("ShaderNodeMath")
    strength.operation = "MULTIPLY"
    strength.location = (380, -40)
    nt.links.new(mask.outputs["Value"], strength.inputs[0])
    nt.links.new(flash_val.outputs[0], strength.inputs[1])

    emis = nt.nodes.new("ShaderNodeEmission")
    emis.location = (580, 80)
    nt.links.new(ramp.outputs["Color"], emis.inputs["Color"])
    nt.links.new(strength.outputs["Value"], emis.inputs["Strength"])

    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    transp.location = (580, -120)

    add_sh = nt.nodes.new("ShaderNodeAddShader")
    add_sh.location = (780, 0)
    nt.links.new(transp.outputs["BSDF"], add_sh.inputs[0])
    nt.links.new(emis.outputs["Emission"], add_sh.inputs[1])
    nt.links.new(add_sh.outputs["Shader"], out.inputs["Surface"])
    return mat, flash_val


def _mat_hero_copper_jacket(
    name: str,
    local_obj: bpy.types.Object,
) -> tuple[bpy.types.Material, bpy.types.Node]:
    """
    Burnished copper-gold jacket (Metallic=0.92) + Beat 4 XRayBullet cutaway mode:
    - Translucent Fresnel rim <= 0.35 so internal spinning lead core pops!
    - LightPath.Is Shadow Ray -> Transparent BSDF gated by XRayBullet.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (1120, 0)

    tex = nt.nodes.new("ShaderNodeTexCoord")
    tex.location = (-980, -160)
    tex.object = local_obj

    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.location = (-780, -160)
    mapping.inputs["Scale"].default_value = (12.0, 70.0, 70.0)
    nt.links.new(tex.outputs["Object"], mapping.inputs["Vector"])

    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-560, -160)
    noise.inputs["Scale"].default_value = 26.0
    noise.inputs["Detail"].default_value = 4.0
    nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

    rough_ramp = nt.nodes.new("ShaderNodeValToRGB")
    rough_ramp.location = (-340, -160)
    rough_ramp.color_ramp.elements[0].position = 0.2
    rough_ramp.color_ramp.elements[0].color = (0.16, 0.16, 0.16, 1.0)
    rough_ramp.color_ramp.elements[1].position = 0.85
    rough_ramp.color_ramp.elements[1].color = (0.30, 0.30, 0.30, 1.0)
    nt.links.new(noise.outputs["Fac"], rough_ramp.inputs["Fac"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (-340, -340)
    bump.inputs["Strength"].default_value = 0.04
    bump.inputs["Distance"].default_value = 0.0005
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (-60, -160)
    bsdf.inputs["Base Color"].default_value = (0.85, 0.44, 0.18, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.92
    nt.links.new(rough_ramp.outputs["Color"], bsdf.inputs["Roughness"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # Hollow X-Ray Fresnel shell (rim <= 0.35)
    lp = nt.nodes.new("ShaderNodeLightPath")
    lp.location = (-560, 420)
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    geom.location = (-560, 260)

    inv_back = nt.nodes.new("ShaderNodeMath")
    inv_back.operation = "SUBTRACT"
    inv_back.location = (-340, 260)
    inv_back.inputs[0].default_value = 1.0
    nt.links.new(geom.outputs["Backfacing"], inv_back.inputs[1])

    cam_front = nt.nodes.new("ShaderNodeMath")
    cam_front.operation = "MULTIPLY"
    cam_front.location = (-140, 320)
    nt.links.new(lp.outputs["Is Camera Ray"], cam_front.inputs[0])
    nt.links.new(inv_back.outputs["Value"], cam_front.inputs[1])

    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.location = (-560, 80)
    lw.inputs["Blend"].default_value = 0.42

    rim_pow = nt.nodes.new("ShaderNodeMath")
    rim_pow.operation = "POWER"
    rim_pow.location = (-340, 80)
    rim_pow.inputs[1].default_value = 3.2
    nt.links.new(lw.outputs["Facing"], rim_pow.inputs[0])

    rim_cap = nt.nodes.new("ShaderNodeMath")
    rim_cap.operation = "MULTIPLY"
    rim_cap.location = (-140, 80)
    rim_cap.inputs[1].default_value = 0.34  # Strictly <= 0.35 translucent rim
    nt.links.new(rim_pow.outputs["Value"], rim_cap.inputs[0])

    rim_mask = nt.nodes.new("ShaderNodeMath")
    rim_mask.operation = "MULTIPLY"
    rim_mask.location = (60, 180)
    nt.links.new(cam_front.outputs["Value"], rim_mask.inputs[0])
    nt.links.new(rim_cap.outputs["Value"], rim_mask.inputs[1])

    rim_str = nt.nodes.new("ShaderNodeMath")
    rim_str.operation = "MULTIPLY"
    rim_str.location = (240, 180)
    rim_str.inputs[1].default_value = 1.55
    nt.links.new(rim_mask.outputs["Value"], rim_str.inputs[0])

    xray_emis = nt.nodes.new("ShaderNodeEmission")
    xray_emis.location = (420, 240)
    xray_emis.inputs["Color"].default_value = (0.34, 0.90, 1.0, 1.0)
    nt.links.new(rim_str.outputs["Value"], xray_emis.inputs["Strength"])

    xray_transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    xray_transp.location = (420, 80)

    xray_add = nt.nodes.new("ShaderNodeAddShader")
    xray_add.location = (620, 160)
    nt.links.new(xray_transp.outputs["BSDF"], xray_add.inputs[0])
    nt.links.new(xray_emis.outputs["Emission"], xray_add.inputs[1])

    xray_val = nt.nodes.new("ShaderNodeValue")
    xray_val.name = "XRayBullet"
    xray_val.location = (240, -60)
    xray_val.outputs[0].default_value = 0.0

    mix_xray = nt.nodes.new("ShaderNodeMixShader")
    mix_xray.location = (780, 0)
    nt.links.new(xray_val.outputs[0], mix_xray.inputs["Fac"])
    nt.links.new(bsdf.outputs["BSDF"], mix_xray.inputs[1])
    nt.links.new(xray_add.outputs["Shader"], mix_xray.inputs[2])

    # Shadow ray bypass when XRayBullet > 0
    shad_gate = nt.nodes.new("ShaderNodeMath")
    shad_gate.operation = "MULTIPLY"
    shad_gate.location = (740, 320)
    nt.links.new(lp.outputs["Is Shadow Ray"], shad_gate.inputs[0])
    nt.links.new(xray_val.outputs[0], shad_gate.inputs[1])

    shad_transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    shad_transp.location = (740, 440)

    mix_shad = nt.nodes.new("ShaderNodeMixShader")
    mix_shad.location = (940, 120)
    nt.links.new(shad_gate.outputs["Value"], mix_shad.inputs["Fac"])
    nt.links.new(mix_xray.outputs["Shader"], mix_shad.inputs[1])
    nt.links.new(shad_transp.outputs["BSDF"], mix_shad.inputs[2])

    nt.links.new(mix_shad.outputs["Shader"], out.inputs["Surface"])
    return mat, xray_val


def _mat_hero_lead_core(
    name: str,
    core_obj: bpy.types.Object,
) -> tuple[bpy.types.Material, bpy.types.Node, bpy.types.Node]:
    """
    Internal spinning Lead Core: metallic lead normally, shifting
    cyan -> molten amber -> crimson across Beat 4 radial torque stress (f=930..1055).
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (1020, 0)

    tex = nt.nodes.new("ShaderNodeTexCoord")
    tex.location = (-980, 0)
    tex.object = core_obj

    # Spiral helical stress variation in local space so rapid core spin pops visually
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.location = (-780, 140)
    mapping.inputs["Scale"].default_value = (18.0, 52.0, 52.0)
    nt.links.new(tex.outputs["Object"], mapping.inputs["Vector"])

    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-560, 140)
    noise.inputs["Scale"].default_value = 9.0
    noise.inputs["Detail"].default_value = 4.0
    nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

    stress_val = nt.nodes.new("ShaderNodeValue")
    stress_val.name = "TorqueStress"
    stress_val.location = (-780, -120)
    stress_val.outputs[0].default_value = 0.0

    # Modulate stress slightly by local-space noise for swirling torque bands
    noise_centered = nt.nodes.new("ShaderNodeMath")
    noise_centered.operation = "MULTIPLY_ADD"
    noise_centered.location = (-360, 140)
    noise_centered.inputs[1].default_value = 0.24
    noise_centered.inputs[2].default_value = -0.12
    nt.links.new(noise.outputs["Fac"], noise_centered.inputs[0])

    stress_mod = nt.nodes.new("ShaderNodeMath")
    stress_mod.operation = "ADD"
    stress_mod.use_clamp = True
    stress_mod.location = (-160, 40)
    nt.links.new(stress_val.outputs[0], stress_mod.inputs[0])
    nt.links.new(noise_centered.outputs["Value"], stress_mod.inputs[1])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (40, 100)
    e0 = ramp.color_ramp.elements[0]
    e0.position = 0.0
    e0.color = (0.14, 0.90, 1.0, 1.0)  # Healthy/initial cyan
    e1 = ramp.color_ramp.elements[1]
    e1.position = 0.96
    e1.color = (1.0, 0.11, 0.07, 1.0)  # Peak fracture crimson
    emid = ramp.color_ramp.elements.new(0.48)
    emid.color = (1.0, 0.56, 0.08, 1.0)  # Molten amber torque
    nt.links.new(stress_mod.outputs["Value"], ramp.inputs["Fac"])

    # Strength ramps gently 1.55 -> 2.02 (AgX-safe)
    str_mix = nt.nodes.new("ShaderNodeMix")
    str_mix.data_type = "FLOAT"
    str_mix.location = (80, -140)
    f_in, a_in, b_in, str_out = _get_mix_sockets(str_mix, "VALUE")
    a_in.default_value = 1.55
    b_in.default_value = 2.02
    nt.links.new(stress_val.outputs[0], f_in)

    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.location = (-160, -240)
    lw.inputs["Blend"].default_value = 0.35

    rim_boost = nt.nodes.new("ShaderNodeMath")
    rim_boost.operation = "MULTIPLY_ADD"
    rim_boost.location = (280, -140)
    rim_boost.inputs[1].default_value = 0.35
    nt.links.new(lw.outputs["Facing"], rim_boost.inputs[0])
    nt.links.new(str_out, rim_boost.inputs[2])

    xray_emis = nt.nodes.new("ShaderNodeEmission")
    xray_emis.location = (500, 80)
    nt.links.new(ramp.outputs["Color"], xray_emis.inputs["Color"])
    nt.links.new(rim_boost.outputs["Value"], xray_emis.inputs["Strength"])

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (460, -220)
    bsdf.inputs["Base Color"].default_value = (0.30, 0.32, 0.36, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.88
    bsdf.inputs["Roughness"].default_value = 0.34

    xray_core = nt.nodes.new("ShaderNodeValue")
    xray_core.name = "XRayCore"
    xray_core.location = (480, 260)
    xray_core.outputs[0].default_value = 0.0

    mix_sh = nt.nodes.new("ShaderNodeMixShader")
    mix_sh.location = (760, 0)
    nt.links.new(xray_core.outputs[0], mix_sh.inputs["Fac"])
    nt.links.new(bsdf.outputs["BSDF"], mix_sh.inputs[1])
    nt.links.new(xray_emis.outputs["Emission"], mix_sh.inputs[2])

    nt.links.new(mix_sh.outputs["Shader"], out.inputs["Surface"])
    return mat, xray_core, stress_val


def _mat_shard_metallic(
    name: str,
    base_color: tuple[float, float, float],
    glow_color: tuple[float, float, float],
    metallic: float = 0.92,
    roughness: float = 0.24,
    with_heat: bool = True,
) -> tuple[bpy.types.Material, bpy.types.Node | None]:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (720, 0)

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (340, 0)
    bsdf.inputs["Base Color"].default_value = (*base_color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness

    heat_node: bpy.types.Node | None = None
    if with_heat:
        lw = nt.nodes.new("ShaderNodeLayerWeight")
        lw.location = (-240, -180)
        lw.inputs["Blend"].default_value = 0.38

        heat_node = nt.nodes.new("ShaderNodeValue")
        heat_node.name = "ShardHeat"
        heat_node.location = (-240, -340)
        heat_node.outputs[0].default_value = 1.0

        mul_h = nt.nodes.new("ShaderNodeMath")
        mul_h.operation = "MULTIPLY"
        mul_h.location = (-20, -220)
        nt.links.new(lw.outputs["Facing"], mul_h.inputs[0])
        nt.links.new(heat_node.outputs[0], mul_h.inputs[1])

        mul_s = nt.nodes.new("ShaderNodeMath")
        mul_s.operation = "MULTIPLY"
        mul_s.location = (160, -220)
        mul_s.inputs[1].default_value = 1.75
        nt.links.new(mul_h.outputs["Value"], mul_s.inputs[0])

        bsdf.inputs["Emission Color"].default_value = (*glow_color, 1.0)
        nt.links.new(mul_s.outputs["Value"], bsdf.inputs["Emission Strength"])

    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, heat_node


def _mat_cavitation_vapor(
    name: str,
    local_obj: bpy.types.Object,
    color: tuple[float, float, float] = (0.50, 0.93, 1.0),
    rim_strength: float = 1.55,
    x_front: float = 0.036,
    x_back: float = -0.26,
) -> bpy.types.Material:
    """Open-ended AddShader cavitation envelope: CameraRay * (1 - Backfacing) * Fresnel * local streaks."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (960, 0)

    tex = nt.nodes.new("ShaderNodeTexCoord")
    tex.location = (-980, -40)
    tex.object = local_obj

    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (-780, 120)
    nt.links.new(tex.outputs["Object"], sep.inputs["Vector"])

    # Soft axial fade to 0 at open back rim (x_back) so zero hard cut line appears
    axial = nt.nodes.new("ShaderNodeMapRange")
    axial.location = (-560, 120)
    axial.clamp = True
    axial.inputs["From Min"].default_value = x_back
    axial.inputs["From Max"].default_value = x_front * 0.4
    axial.inputs["To Min"].default_value = 0.0
    axial.inputs["To Max"].default_value = 1.0
    nt.links.new(sep.outputs["X"], axial.inputs["Value"])

    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.location = (-780, -140)
    mapping.inputs["Scale"].default_value = (4.5, 36.0, 36.0)
    nt.links.new(tex.outputs["Object"], mapping.inputs["Vector"])

    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-560, -140)
    noise.inputs["Scale"].default_value = 11.0
    noise.inputs["Detail"].default_value = 4.0
    nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

    streak = nt.nodes.new("ShaderNodeMix")
    streak.data_type = "FLOAT"
    streak.location = (-340, -140)
    f_in, a_in, b_in, streak_out = _get_mix_sockets(streak, "VALUE")
    a_in.default_value = 0.35
    b_in.default_value = 1.0
    nt.links.new(noise.outputs["Fac"], f_in)

    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.location = (-560, -340)
    lw.inputs["Blend"].default_value = 0.36

    fres_pow = nt.nodes.new("ShaderNodeMath")
    fres_pow.operation = "POWER"
    fres_pow.location = (-340, -340)
    fres_pow.inputs[1].default_value = 2.0
    nt.links.new(lw.outputs["Facing"], fres_pow.inputs[0])

    lp = nt.nodes.new("ShaderNodeLightPath")
    lp.location = (-560, 380)
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    geom.location = (-560, 240)

    inv_back = nt.nodes.new("ShaderNodeMath")
    inv_back.operation = "SUBTRACT"
    inv_back.location = (-340, 240)
    inv_back.inputs[0].default_value = 1.0
    nt.links.new(geom.outputs["Backfacing"], inv_back.inputs[1])

    cam_front = nt.nodes.new("ShaderNodeMath")
    cam_front.operation = "MULTIPLY"
    cam_front.location = (-140, 300)
    nt.links.new(lp.outputs["Is Camera Ray"], cam_front.inputs[0])
    nt.links.new(inv_back.outputs["Value"], cam_front.inputs[1])

    m1 = nt.nodes.new("ShaderNodeMath")
    m1.operation = "MULTIPLY"
    m1.location = (-120, -40)
    nt.links.new(axial.outputs["Result"], m1.inputs[0])
    nt.links.new(streak_out, m1.inputs[1])

    m2 = nt.nodes.new("ShaderNodeMath")
    m2.operation = "MULTIPLY"
    m2.location = (80, -120)
    nt.links.new(m1.outputs["Value"], m2.inputs[0])
    nt.links.new(fres_pow.outputs["Value"], m2.inputs[1])

    m3 = nt.nodes.new("ShaderNodeMath")
    m3.operation = "MULTIPLY"
    m3.location = (280, 60)
    nt.links.new(cam_front.outputs["Value"], m3.inputs[0])
    nt.links.new(m2.outputs["Value"], m3.inputs[1])

    m_str = nt.nodes.new("ShaderNodeMath")
    m_str.operation = "MULTIPLY"
    m_str.location = (460, 60)
    m_str.inputs[1].default_value = rim_strength
    nt.links.new(m3.outputs["Value"], m_str.inputs[0])

    emis = nt.nodes.new("ShaderNodeEmission")
    emis.location = (620, 100)
    emis.inputs["Color"].default_value = (*color, 1.0)
    nt.links.new(m_str.outputs["Value"], emis.inputs["Strength"])

    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    transp.location = (620, -100)

    add_sh = nt.nodes.new("ShaderNodeAddShader")
    add_sh.location = (800, 0)
    nt.links.new(transp.outputs["BSDF"], add_sh.inputs[0])
    nt.links.new(emis.outputs["Emission"], add_sh.inputs[1])
    nt.links.new(add_sh.outputs["Shader"], out.inputs["Surface"])
    return mat


def _mat_streamline_flow(
    name: str,
    local_obj: bpy.types.Object,
    color: tuple[float, float, float] = (0.36, 0.94, 1.0),
    strength: float = 1.65,
) -> tuple[bpy.types.Material, bpy.types.Node]:
    """Local-space traveling energy pulse along curved hydrodynamic streamlines (AddShader)."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (960, 0)

    tex = nt.nodes.new("ShaderNodeTexCoord")
    tex.location = (-980, 0)
    tex.object = local_obj

    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (-780, 60)
    nt.links.new(tex.outputs["Object"], sep.inputs["Vector"])

    phase_val = nt.nodes.new("ShaderNodeValue")
    phase_val.name = "FlowPhase"
    phase_val.location = (-780, -140)
    phase_val.outputs[0].default_value = 0.0

    # Travelling wave: sin(x * 36.0 + phase) mapped to [0.28, 1.0]
    scale_x = nt.nodes.new("ShaderNodeMath")
    scale_x.operation = "MULTIPLY"
    scale_x.location = (-580, 60)
    scale_x.inputs[1].default_value = 38.0
    nt.links.new(sep.outputs["X"], scale_x.inputs[0])

    add_ph = nt.nodes.new("ShaderNodeMath")
    add_ph.operation = "ADD"
    add_ph.location = (-400, 0)
    nt.links.new(scale_x.outputs["Value"], add_ph.inputs[0])
    nt.links.new(phase_val.outputs[0], add_ph.inputs[1])

    sine = nt.nodes.new("ShaderNodeMath")
    sine.operation = "SINE"
    sine.location = (-220, 0)
    nt.links.new(add_ph.outputs["Value"], sine.inputs[0])

    wave_map = nt.nodes.new("ShaderNodeMapRange")
    wave_map.location = (-40, 0)
    wave_map.clamp = True
    wave_map.inputs["From Min"].default_value = -1.0
    wave_map.inputs["From Max"].default_value = 1.0
    wave_map.inputs["To Min"].default_value = 0.30
    wave_map.inputs["To Max"].default_value = 1.0
    nt.links.new(sine.outputs["Value"], wave_map.inputs["Value"])

    # Soft fade at tail x = -0.17 .. -0.08
    tail_fade = nt.nodes.new("ShaderNodeMapRange")
    tail_fade.location = (-220, -200)
    tail_fade.clamp = True
    tail_fade.inputs["From Min"].default_value = -0.17
    tail_fade.inputs["From Max"].default_value = -0.06
    tail_fade.inputs["To Min"].default_value = 0.0
    tail_fade.inputs["To Max"].default_value = 1.0
    nt.links.new(sep.outputs["X"], tail_fade.inputs["Value"])

    lp = nt.nodes.new("ShaderNodeLightPath")
    lp.location = (-400, 320)
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    geom.location = (-400, 180)

    inv_back = nt.nodes.new("ShaderNodeMath")
    inv_back.operation = "SUBTRACT"
    inv_back.location = (-200, 180)
    inv_back.inputs[0].default_value = 1.0
    nt.links.new(geom.outputs["Backfacing"], inv_back.inputs[1])

    cam_front = nt.nodes.new("ShaderNodeMath")
    cam_front.operation = "MULTIPLY"
    cam_front.location = (0, 240)
    nt.links.new(lp.outputs["Is Camera Ray"], cam_front.inputs[0])
    nt.links.new(inv_back.outputs["Value"], cam_front.inputs[1])

    m1 = nt.nodes.new("ShaderNodeMath")
    m1.operation = "MULTIPLY"
    m1.location = (160, -40)
    nt.links.new(wave_map.outputs["Result"], m1.inputs[0])
    nt.links.new(tail_fade.outputs["Result"], m1.inputs[1])

    m2 = nt.nodes.new("ShaderNodeMath")
    m2.operation = "MULTIPLY"
    m2.location = (340, 60)
    nt.links.new(cam_front.outputs["Value"], m2.inputs[0])
    nt.links.new(m1.outputs["Value"], m2.inputs[1])

    m_str = nt.nodes.new("ShaderNodeMath")
    m_str.operation = "MULTIPLY"
    m_str.location = (500, 60)
    m_str.inputs[1].default_value = strength
    nt.links.new(m2.outputs["Value"], m_str.inputs[0])

    emis = nt.nodes.new("ShaderNodeEmission")
    emis.location = (640, 100)
    emis.inputs["Color"].default_value = (*color, 1.0)
    nt.links.new(m_str.outputs["Value"], emis.inputs["Strength"])

    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    transp.location = (640, -100)

    add_sh = nt.nodes.new("ShaderNodeAddShader")
    add_sh.location = (800, 0)
    nt.links.new(transp.outputs["BSDF"], add_sh.inputs[0])
    nt.links.new(emis.outputs["Emission"], add_sh.inputs[1])
    nt.links.new(add_sh.outputs["Shader"], out.inputs["Surface"])
    return mat, phase_val


# ---------------------------------------------------------------------------
# 1. SUBMERGED / SURFACE RIFLE BARREL + MUZZLE FLASH (f=211..540)
# ---------------------------------------------------------------------------
def _build_barrel_and_flash(
    root: bpy.types.Object,
    col: bpy.types.Collection,
    total_frames: int,
) -> dict[str, Any]:
    muzzle_x = float(C.TANK["muzzle_x"])
    flight_z = float(C.TANK["flight_z"])

    barrel_anchor = _create_empty("FX_Barrel_Anchor", col, parent=root)
    barrel_anchor.location = (muzzle_x, 0.0, flight_z)

    mat_barrel = _mat_gunmetal_barrel("Mat_FX_BarrelGunmetal", barrel_anchor)

    # Fluted barrel along -X from muzzle_x (0.0 local) back to -0.72m
    bm = bmesh.new()
    # Profile from rear (-0.72) to muzzle crown (0.0) with recessed inner bore at x=-0.03
    profile: list[tuple[float, float]] = [
        (-0.72, 0.024),
        (-0.56, 0.024),
        (-0.55, 0.021),
        (-0.50, 0.020),
        (-0.36, 0.019),
        (-0.22, 0.0185),
        (-0.08, 0.018),
        # Muzzle brake collar with radial port grooves
        (-0.075, 0.024),
        (-0.055, 0.024),
        (-0.052, 0.019),
        (-0.042, 0.019),
        (-0.039, 0.024),
        (-0.022, 0.024),
        (-0.019, 0.019),
        (-0.010, 0.019),
        (-0.007, 0.024),
        (0.0, 0.023),
        (0.0, 0.0115),     # Bore crown rim
        (-0.06, 0.0112),   # Inner bore depth
    ]

    def _flute_mod(x: float, r: float, ang: float, ring_idx: int, _seg: int) -> float:
        if -0.52 <= x <= -0.09 and r > 0.014:
            # 12 crisp longitudinal machined flutes
            flute = max(0.0, math.cos(12.0 * ang))
            return r - 0.0022 * (flute ** 1.6)
        return r

    _add_revolved_profile(
        bm,
        profile,
        segs=48,
        cap_start=True,
        cap_end=True,
        radial_mod_fn=_flute_mod,
    )
    barrel_obj = _finalize_bmesh(
        "FX_Barrel",
        bm,
        mat_barrel,
        col,
        parent=barrel_anchor,
        smooth=True,
        shadow=True,
    )
    _set_visibility_windows(barrel_obj, [(211, 540)], total_frames)

    # Muzzle flash plume (f=248..264) at muzzle_x
    flash_anchor = _create_empty("FX_MuzzleFlash_Anchor", col, parent=barrel_anchor)
    mat_flash, flash_val = _mat_muzzle_flash("Mat_FX_MuzzleFlash", flash_anchor)

    bm_f = bmesh.new()
    # Outer open-ended fluted star-cone flame plume (x: 0.0 -> 0.105m so projectile bursts cleanly ahead)
    outer_prof = [
        (0.002, 0.009),
        (0.018, 0.024),
        (0.044, 0.030),
        (0.078, 0.020),
        (0.105, 0.006),
    ]

    def _star_mod(_x: float, r: float, ang: float, _ri: int, _s: int) -> float:
        return r * (0.76 + 0.34 * (0.5 + 0.5 * math.cos(6.0 * ang)))

    _add_revolved_profile(
        bm_f,
        outer_prof,
        segs=36,
        cap_start=False,
        cap_end=False,
        radial_mod_fn=_star_mod,
    )
    # Inner hot plasma spear (open-ended)
    inner_prof = [
        (0.002, 0.006),
        (0.022, 0.014),
        (0.052, 0.011),
        (0.082, 0.003),
    ]
    _add_revolved_profile(bm_f, inner_prof, segs=24, cap_start=False, cap_end=False)

    # 4 subtle tight compensator vents angled forward along barrel (+18 deg)
    for k in range(4):
        ang = k * (math.pi * 0.5)
        dir_vec = Vector((0.92, 0.38 * math.cos(ang), 0.38 * math.sin(ang))).normalized()
        pts = [
            Vector((-0.020, 0.014 * math.cos(ang), 0.014 * math.sin(ang))) + dir_vec * (0.010 * j)
            for j in range(5)
        ]
        rads = [0.003, 0.005, 0.005, 0.003, 0.001]
        _add_swept_tube(bm_f, pts, rads, segs=10, cap_ends=False)

    flash_obj = _finalize_bmesh(
        "FX_MuzzleFlash",
        bm_f,
        mat_flash,
        col,
        parent=flash_anchor,
        smooth=True,
        shadow=False,
    )
    _set_visibility_windows(flash_obj, [(248, 264)], total_frames)

    # Animate AgX-safe fiery amber intensity & slight forward stretch across f=248..264
    flash_curve = [
        (247, 0.0, 0.50),
        (248, 3.6, 0.85),
        (250, 4.8, 1.12),
        (254, 4.2, 1.02),
        (258, 3.0, 0.86),
        (264, 0.0, 0.55),
    ]
    for fr, inten, sc in flash_curve:
        flash_val.outputs[0].default_value = inten
        flash_val.outputs[0].keyframe_insert(data_path="default_value", frame=fr)
        flash_obj.scale = (sc, sc, sc)
        flash_obj.keyframe_insert(data_path="scale", frame=fr)

    _set_action_interp(mat_flash.node_tree.animation_data, ("default_value",), "LINEAR")
    _set_action_interp(flash_obj.animation_data, ("scale",), "LINEAR")

    return {
        "barrel_anchor": barrel_anchor,
        "barrel": barrel_obj,
        "muzzle_flash": flash_obj,
    }


# ---------------------------------------------------------------------------
# 2. PRIMARY HERO SUPERSONIC ROUND, X-RAY CORE & 8 SHARDS (Beats 1-4)
# ---------------------------------------------------------------------------
def _build_spitzer_jacket_bmesh() -> bmesh.types.BMesh:
    bm = bmesh.new()
    # Profile from tail (-0.0325) to sharp Spitzer ogive tip (+0.0325), total = 6.5 cm
    prof = [
        (-0.0325, 0.0068),
        (-0.0255, 0.0087),
        (-0.0150, 0.0088),
        (-0.0138, 0.0083),  # Cannelure crimp ring
        (-0.0124, 0.0088),
        (-0.0030, 0.0088),
        (0.0060, 0.0079),
        (0.0145, 0.0062),
        (0.0225, 0.0039),
        (0.0285, 0.0017),
        (0.0325, 0.0004),
    ]
    _add_revolved_profile(bm, prof, segs=36, cap_start=True, cap_end=True)
    return bm


def _build_lead_core_bmesh() -> bmesh.types.BMesh:
    bm = bmesh.new()
    # Internal core inside jacket (~78% radius, fluted subtly to show rapid spin)
    prof = [
        (-0.0290, 0.0052),
        (-0.0220, 0.0067),
        (-0.0040, 0.0067),
        (0.0060, 0.0058),
        (0.0155, 0.0041),
        (0.0235, 0.0020),
        (0.0268, 0.0005),
    ]

    def _core_flute(x: float, r: float, ang: float, _ri: int, _s: int) -> float:
        if -0.024 <= x <= 0.016:
            twist = ang + x * 85.0
            return r * (0.93 + 0.07 * math.cos(5.0 * twist))
        return r

    _add_revolved_profile(bm, prof, segs=30, cap_start=True, cap_end=True, radial_mod_fn=_core_flute)
    return bm


def _build_copper_strip_bmesh(idx: int) -> bmesh.types.BMesh:
    """Create jagged curled thick copper jacket petal strip."""
    bm = bmesh.new()
    ang0 = (idx * 2.0 * math.pi) / 5.0 + 0.25
    arc = 0.72
    segs_a = 4
    segs_x = 5
    thick = 0.0015
    r_base = 0.0082
    x0 = -0.022 + (idx % 2) * 0.008
    x1 = 0.018 + (idx % 3) * 0.004

    outer_grid: list[list[bmesh.types.BMVert]] = []
    inner_grid: list[list[bmesh.types.BMVert]] = []
    for ix in range(segs_x):
        tx = ix / (segs_x - 1)
        x = C.lerp(x0, x1, tx)
        flare = 0.0025 * (tx ** 1.5) if idx % 2 == 0 else -0.001 * tx
        jag = 0.08 * math.sin(ix * 2.3 + idx)
        row_out: list[bmesh.types.BMVert] = []
        row_in: list[bmesh.types.BMVert] = []
        for ia in range(segs_a):
            ta = ia / (segs_a - 1)
            a = ang0 + (ta - 0.5) * arc + jag
            ro = r_base * (1.0 - 0.35 * max(0.0, tx - 0.4)) + flare
            ri = max(0.001, ro - thick)
            row_out.append(bm.verts.new((x, ro * math.cos(a), ro * math.sin(a))))
            row_in.append(bm.verts.new((x, ri * math.cos(a), ri * math.sin(a))))
        outer_grid.append(row_out)
        inner_grid.append(row_in)

    for ix in range(segs_x - 1):
        for ia in range(segs_a - 1):
            bm.faces.new((
                outer_grid[ix][ia],
                outer_grid[ix + 1][ia],
                outer_grid[ix + 1][ia + 1],
                outer_grid[ix][ia + 1],
            ))
            bm.faces.new((
                inner_grid[ix][ia],
                inner_grid[ix][ia + 1],
                inner_grid[ix + 1][ia + 1],
                inner_grid[ix + 1][ia],
            ))
    # Side walls for solid 3D thickness
    for ix in range(segs_x - 1):
        bm.faces.new((outer_grid[ix][0], inner_grid[ix][0], inner_grid[ix + 1][0], outer_grid[ix + 1][0]))
        bm.faces.new((outer_grid[ix][-1], outer_grid[ix + 1][-1], inner_grid[ix + 1][-1], inner_grid[ix][-1]))
    for ia in range(segs_a - 1):
        bm.faces.new((outer_grid[0][ia], outer_grid[0][ia + 1], inner_grid[0][ia + 1], inner_grid[0][ia]))
        bm.faces.new((outer_grid[-1][ia], inner_grid[-1][ia], inner_grid[-1][ia + 1], outer_grid[-1][ia + 1]))
    return bm


def _build_deformed_core_chunk_bmesh(idx: int) -> bmesh.types.BMesh:
    """Create jagged deformed lead core chunk (closed icosphere displaced asymmetrically)."""
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=0.0055)
    for v in bm.verts:
        p = v.co
        n = 1.0 + 0.38 * math.sin(p.x * 420.0 + idx * 1.9) * math.cos(p.y * 390.0 - idx)
        p.x *= 1.35 * n
        p.y *= 0.85 * n
        p.z *= 0.92 * (1.0 + 0.25 * math.sin(p.z * 350.0 + idx))
    return bm


def _build_hero_round_and_shards(
    root: bpy.types.Object,
    col: bpy.types.Collection,
    total_frames: int,
) -> dict[str, Any]:
    hero_root = _create_empty("FX_HeroBullet_Root", col, parent=root)

    mat_jacket, xray_j_val = _mat_hero_copper_jacket("Mat_FX_HeroCopperJacket", hero_root)
    mat_core, xray_c_val, stress_val = _mat_hero_lead_core("Mat_FX_HeroLeadCore", hero_root)

    jacket_obj = _finalize_bmesh(
        "FX_Hero_Jacket",
        _build_spitzer_jacket_bmesh(),
        mat_jacket,
        col,
        parent=hero_root,
        smooth=True,
        shadow=True,
    )
    core_obj = _finalize_bmesh(
        "FX_Hero_Core",
        _build_lead_core_bmesh(),
        mat_core,
        col,
        parent=hero_root,
        smooth=True,
        shadow=False,
    )

    # Visible Beats 1, 2 (post-muzzle f=248), 3, and Beat 4 until shatter burst at 1055
    _set_visibility_windows(jacket_obj, [(1, 210), (248, 1055)], total_frames)
    _set_visibility_windows(core_obj, [(1, 210), (248, 1055)], total_frames)

    # Animate hero trajectory, rifling spin & Beat 4 X-ray/torque shaders
    frames_hero = _sample_frames(
        1,
        1060,
        step=4,
        dense_ranges=((10, 145, 2), (248, 320, 2), (915, 1060, 2)),
    )
    prev_q_root: Any = None
    prev_q_core: Any = None
    spin_acc = 0.0
    prev_f = frames_hero[0]

    for fr in frames_hero:
        dt = float(fr - prev_f)
        prev_f = fr
        # High-speed macro rifling spin (~14 deg/frame) decelerating under water torque in Beat 4
        if fr <= 900:
            spin_rate = math.radians(12.0)
        elif fr <= 1055:
            spin_rate = math.radians(12.0) * (1.0 - C.smootherstep(930.0, 1055.0, float(fr)))
        else:
            spin_rate = 0.0
        spin_acc += spin_rate * dt

        hx = C.supersonic_x(float(fr))
        hz = C.supersonic_z(float(fr))
        pitch_deg, yaw_deg = C.supersonic_tumble_deg(float(fr))

        hero_root.location = (hx, 0.0, hz)
        hero_root.keyframe_insert(data_path="location", frame=fr)

        q_root = Euler(
            (0.35 * spin_acc, math.radians(pitch_deg), math.radians(yaw_deg)),
            "XYZ",
        ).to_quaternion()
        if prev_q_root is not None and q_root.dot(prev_q_root) < 0.0:
            q_root.negate()
        prev_q_root = q_root.copy()
        hero_root.rotation_quaternion = q_root
        hero_root.keyframe_insert(data_path="rotation_quaternion", frame=fr)

        # Core spins visibly faster inside jacket during X-ray torque stress
        q_core = Euler((spin_acc * 1.1, 0.0, 0.0), "XYZ").to_quaternion()
        if prev_q_core is not None and q_core.dot(prev_q_core) < 0.0:
            q_core.negate()
        prev_q_core = q_core.copy()
        core_obj.rotation_quaternion = q_core
        core_obj.keyframe_insert(data_path="rotation_quaternion", frame=fr)

    _set_action_interp(hero_root.animation_data, ("location", "rotation_quaternion"), "LINEAR")
    _set_action_interp(core_obj.animation_data, ("rotation_quaternion",), "LINEAR")

    # Keyframe XRayBullet (f=915..1245) & TorqueStress (f=930..1055)
    xray_frames = _sample_frames(905, 1060, step=4, dense_ranges=((915, 1058, 2),))
    for fr in xray_frames:
        xw = C.xray_bullet_weight(float(fr))
        st = C.smootherstep(930.0, 1052.0, float(fr))
        xray_j_val.outputs[0].default_value = xw
        xray_j_val.outputs[0].keyframe_insert(data_path="default_value", frame=fr)
        xray_c_val.outputs[0].default_value = xw
        xray_c_val.outputs[0].keyframe_insert(data_path="default_value", frame=fr)
        stress_val.outputs[0].default_value = st
        stress_val.outputs[0].keyframe_insert(data_path="default_value", frame=fr)

    _set_action_interp(mat_jacket.node_tree.animation_data, ("default_value",), "LINEAR")
    _set_action_interp(mat_core.node_tree.animation_data, ("default_value",), "LINEAR")

    # 8 Fractured Copper & Lead Shards (f=1045..1260)
    shard_root = _create_empty("FX_HeroShards_Root", col, parent=root)
    mat_sc, heat_sc = _mat_shard_metallic(
        "Mat_FX_ShardCopper",
        base_color=(0.84, 0.43, 0.19),
        glow_color=(1.0, 0.48, 0.10),
        metallic=0.92,
        roughness=0.22,
        with_heat=True,
    )
    mat_sl, heat_sl = _mat_shard_metallic(
        "Mat_FX_ShardLead",
        base_color=(0.34, 0.36, 0.40),
        glow_color=(1.0, 0.18, 0.08),
        metallic=0.88,
        roughness=0.34,
        with_heat=True,
    )

    # Burst vectors (dx, dy, dz, spin_vx, spin_vy, spin_vz) around X = THREE_FT_X (0.9144m)
    shard_dirs: list[tuple[float, float, float, float, float, float]] = [
        (-0.045,  0.078,  0.092,  6.2,  3.5, -4.1),
        ( 0.055, -0.095,  0.068, -5.0,  7.4,  3.2),
        (-0.065, -0.070, -0.105,  4.8, -6.1,  5.5),
        ( 0.038,  0.110, -0.075, -7.2,  2.9,  6.0),
        ( 0.072,  0.032,  0.115,  5.5,  5.8, -6.8),
        # 3 deformed lead core chunks
        ( 0.025, -0.055,  0.042, -4.4, -3.9,  4.8),
        (-0.035,  0.048, -0.068,  3.8,  5.2, -3.5),
        ( 0.062,  0.012, -0.045, -6.0,  4.2,  2.9),
    ]

    shards: list[bpy.types.Object] = []
    frames_shard = _sample_frames(1045, 1260, step=5, dense_ranges=((1045, 1085, 2),))
    bx_1055 = C.supersonic_x(1055.0)
    bz_1055 = C.supersonic_z(1055.0)
    p0, y0 = C.supersonic_tumble_deg(1055.0)
    base_q = Euler((0.0, math.radians(p0), math.radians(y0)), "XYZ").to_quaternion()

    for idx in range(8):
        if idx < 5:
            bm_s = _build_copper_strip_bmesh(idx)
            s_mat = mat_sc
            s_name = f"FX_Hero_Shard_Copper_{idx:02d}"
        else:
            bm_s = _build_deformed_core_chunk_bmesh(idx - 5)
            s_mat = mat_sl
            s_name = f"FX_Hero_Shard_Lead_{idx:02d}"

        sh_obj = _finalize_bmesh(s_name, bm_s, s_mat, col, parent=shard_root, smooth=True, shadow=True)
        _set_visibility_windows(sh_obj, [(1045, 1260)], total_frames)
        shards.append(sh_obj)

        dx, dy, dz, rx, ry, rz = shard_dirs[idx]
        prev_qs: Any = None
        for fr in frames_shard:
            if fr < 1055:
                # Pre-fracture seam swell inside X-ray jacket (1045..1054)
                hx = C.supersonic_x(float(fr))
                hz = C.supersonic_z(float(fr))
                pd, yd = C.supersonic_tumble_deg(float(fr))
                pre = C.smoothstep(1045.0, 1055.0, float(fr)) * 0.0022
                pos = Vector((hx + dx * pre * 12.0, dy * pre * 12.0, hz + dz * pre * 12.0))
                q = Euler((0.0, math.radians(pd), math.radians(yd)), "XYZ").to_quaternion()
            else:
                st = C.supersonic_shatter_t(float(fr))  # 0..1 across 1045..1185
                # Smoothly anchor relative to shatter burst at f=1055
                burst_u = C.clamp((st - 0.015) / 0.985, 0.0, 1.0)
                drift_u = C.smootherstep(1055.0, 1260.0, float(fr))
                # Slight gravity drift downward around X=0.9144m after water drag halts radial burst
                x_pos = C.lerp(bx_1055, C.THREE_FT_X + dx * 0.72, burst_u)
                y_pos = dy * burst_u + 0.015 * math.sin(drift_u * 4.5 + idx)
                z_pos = bz_1055 + dz * burst_u - 0.16 * (drift_u ** 1.3)
                pos = Vector((x_pos, y_pos, z_pos))

                t_spin = (fr - 1055.0) / 60.0
                flutter = 1.0 - 0.55 * drift_u
                dq = Euler(
                    (rx * t_spin * flutter, ry * t_spin * flutter, rz * t_spin * flutter),
                    "XYZ",
                ).to_quaternion()
                q = dq @ base_q

            if prev_qs is not None and q.dot(prev_qs) < 0.0:
                q.negate()
            prev_qs = q.copy()
            sh_obj.location = pos
            sh_obj.rotation_quaternion = q
            sh_obj.keyframe_insert(data_path="location", frame=fr)
            sh_obj.keyframe_insert(data_path="rotation_quaternion", frame=fr)

        _set_action_interp(sh_obj.animation_data, ("location", "rotation_quaternion"), "LINEAR")

    # Cool down shard thermal glow across f=1055..1245
    for fr in (1045, 1055, 1110, 1180, 1245):
        heat = 1.0 - C.smootherstep(1065.0, 1240.0, float(fr))
        for hn in (heat_sc, heat_sl):
            if hn is not None:
                hn.outputs[0].default_value = heat
                hn.outputs[0].keyframe_insert(data_path="default_value", frame=fr)
    _set_action_interp(mat_sc.node_tree.animation_data, ("default_value",), "LINEAR")
    _set_action_interp(mat_sl.node_tree.animation_data, ("default_value",), "LINEAR")

    return {
        "hero_root": hero_root,
        "hero_jacket": jacket_obj,
        "hero_core": core_obj,
        "hero_shards": shards,
        "xray_jacket_node": xray_j_val,
        "xray_core_node": xray_c_val,
    }


# ---------------------------------------------------------------------------
# 3. SUPERCAVITATION CONE, SPIRAL BUBBLES & 800x STREAMLINES (Beats 1-4)
# ---------------------------------------------------------------------------
def _build_cavitation_and_streamlines(
    root: bpy.types.Object,
    col: bpy.types.Collection,
    total_frames: int,
) -> dict[str, Any]:
    cav_root = _create_empty("FX_Cavitation_Root", col, parent=root)

    # Open-ended aerodynamic parabolic cavitation cone (cap_start=False, cap_end=False)
    mat_cav = _mat_cavitation_vapor(
        "Mat_FX_CavitationCone",
        cav_root,
        color=(0.50, 0.93, 1.0),
        rim_strength=1.55,
        x_front=0.036,
        x_back=-0.28,
    )
    bm_c = bmesh.new()
    # Parabolic nose-enveloping profile along X: +0.036 -> -0.28
    n_rings = 22
    cav_prof: list[tuple[float, float]] = []
    for i in range(n_rings):
        u = i / (n_rings - 1)
        x = C.lerp(0.036, -0.28, u)
        r = 0.0022 + 0.068 * (u ** 0.56)
        cav_prof.append((x, r))

    _add_revolved_profile(bm_c, cav_prof, segs=36, cap_start=False, cap_end=False)
    cav_obj = _finalize_bmesh(
        "FX_Cavitation_Cone",
        bm_c,
        mat_cav,
        col,
        parent=cav_root,
        smooth=True,
        shadow=False,
    )

    # Add ShapeKey for dramatic sideways collapse at f=970..1055
    sk_basis = cav_obj.shape_key_add(name="Basis", from_mix=False)
    sk_basis.interpolation = "KEY_LINEAR"
    sk_coll = cav_obj.shape_key_add(name="CollapsePinch", from_mix=False)
    sk_coll.interpolation = "KEY_LINEAR"
    for v_idx, v in enumerate(cav_obj.data.vertices):
        x, y, z = v.co.x, v.co.y, v.co.z
        u = C.clamp((0.036 - x) / 0.316, 0.0, 1.0)
        # Asymmetric lateral pinch + violent helical buckle wave
        pinch_y = y * (0.22 if y > 0 else 0.55) - 0.028 * math.sin(u * 5.2) * u
        pinch_z = z * (0.45 + 0.35 * math.cos(u * 7.5)) + 0.022 * math.cos(u * 4.0) * u
        sk_coll.data[v_idx].co = (x + 0.04 * u, pinch_y, pinch_z)

    _set_visibility_windows(cav_obj, [(10, 185), (300, 1057)], total_frames)

    # Keyframe cavitation root tracking + cone collapse/scale
    frames_cav = _sample_frames(
        10,
        1057,
        step=4,
        dense_ranges=((10, 185, 3), (300, 340, 2), (965, 1057, 2)),
    )
    prev_qc: Any = None
    for fr in frames_cav:
        if 186 <= fr <= 299:
            continue
        hx = C.supersonic_x(float(fr))
        hz = C.supersonic_z(float(fr))
        pd, yd = C.supersonic_tumble_deg(float(fr))

        cav_root.location = (hx, 0.0, hz)
        qc = Euler((0.0, math.radians(pd), math.radians(yd)), "XYZ").to_quaternion()
        if prev_qc is not None and qc.dot(prev_qc) < 0.0:
            qc.negate()
        prev_qc = qc.copy()
        cav_root.rotation_quaternion = qc
        cav_root.keyframe_insert(data_path="location", frame=fr)
        cav_root.keyframe_insert(data_path="rotation_quaternion", frame=fr)

        # Dynamic scale envelope + collapse
        if fr <= 135:
            sc = (1.05, 1.0, 1.0)
            coll_val = 0.0
        elif fr <= 185:
            d = C.smootherstep(135.0, 185.0, float(fr))
            sc = (1.0 - 0.65 * d, 1.0 - 0.88 * d, 1.0 - 0.88 * d)
            coll_val = 0.65 * d
        elif fr < 300:
            sc = (0.1, 0.1, 0.1)
            coll_val = 0.0
        elif fr <= 335:
            b = C.smootherstep(300.0, 335.0, float(fr))
            sc = (0.25 + 0.75 * b, 0.30 + 0.70 * b, 0.30 + 0.70 * b)
            coll_val = 0.0
        elif fr <= 970:
            peak = 1.0 + 0.18 * math.sin(C.smootherstep(540.0, 900.0, float(fr)) * math.pi)
            sc = (peak, peak, peak)
            coll_val = 0.0
        else:
            c = C.smootherstep(970.0, 1055.0, float(fr))
            sc = (1.0 - 0.70 * c, 1.0 - 0.85 * c, 1.0 - 0.60 * c)
            coll_val = c

        cav_obj.scale = sc
        cav_obj.keyframe_insert(data_path="scale", frame=fr)
        sk_coll.value = coll_val
        sk_coll.keyframe_insert(data_path="value", frame=fr)

    _set_action_interp(cav_root.animation_data, ("location", "rotation_quaternion"), "LINEAR")
    _set_action_interp(cav_obj.animation_data, ("scale",), "LINEAR")
    if cav_obj.data.shape_keys and cav_obj.data.shape_keys.animation_data:
        _set_action_interp(cav_obj.data.shape_keys.animation_data, ("value",), "LINEAR")

    # Trailing spiraling curved 3D bubble tubes + 3D spherical micro-bubbles (ZERO flat polygons)
    mat_wake = _mat_cavitation_vapor(
        "Mat_FX_WakeBubbles",
        cav_root,
        color=(0.62, 0.96, 1.0),
        rim_strength=1.40,
        x_front=-0.04,
        x_back=-0.52,
    )
    bm_wake = bmesh.new()
    # 12 spiraling open-ended 3D vapor tubes trailing behind the cone
    for t_idx in range(12):
        ang0 = (t_idx * 2.0 * math.pi) / 12.0
        pts: list[Vector] = []
        rads: list[float] = []
        for s_idx in range(10):
            u = s_idx / 9.0
            x = C.lerp(-0.12, -0.48 - 0.06 * (t_idx % 3), u)
            r_helix = C.lerp(0.046, 0.022 + 0.032 * (t_idx % 2), u)
            ang = ang0 + u * 2.6
            pts.append(Vector((x, r_helix * math.cos(ang), r_helix * math.sin(ang))))
            rads.append(0.0042 * math.sin(max(0.05, min(0.95, u)) * math.pi))
        _add_swept_tube(bm_wake, pts, rads, segs=8, cap_ends=False)

    # 20 closed 3D icosphere bubble beads swirling in the wake
    for b_idx in range(20):
        ang = b_idx * 2.39996
        u = (b_idx + 0.5) / 20.0
        bx = C.lerp(-0.14, -0.46, u)
        br = 0.032 + 0.036 * ((b_idx * 7) % 5) / 4.0
        rad_b = 0.0032 + 0.0035 * ((b_idx % 3) / 2.0)
        mat_pos = (
            Vector((bx, br * math.cos(ang), br * math.sin(ang)))
        )
        res = bmesh.ops.create_icosphere(bm_wake, subdivisions=1, radius=rad_b)
        for v in res["verts"]:
            v.co += mat_pos

    wake_obj = _finalize_bmesh(
        "FX_Cavitation_WakeTubes",
        bm_wake,
        mat_wake,
        col,
        parent=cav_root,
        smooth=True,
        shadow=False,
    )
    _set_visibility_windows(wake_obj, [(14, 190), (304, 1048)], total_frames)

    # Gentle spiral roll on wake tubes + inward collapse prior to shatter burst
    prev_qw: Any = None
    for fr in _sample_frames(14, 1048, step=6, dense_ranges=((970, 1048, 3),)):
        if 191 <= fr <= 303:
            continue
        roll = (fr * 0.14) % (2.0 * math.pi)
        qw = Euler((roll, 0.0, 0.0), "XYZ").to_quaternion()
        if prev_qw is not None and qw.dot(prev_qw) < 0.0:
            qw.negate()
        prev_qw = qw.copy()
        wake_obj.rotation_quaternion = qw
        wake_obj.keyframe_insert(data_path="rotation_quaternion", frame=fr)
        if fr >= 975:
            c = C.smootherstep(975.0, 1048.0, float(fr))
            wake_obj.scale = (1.0 - 0.55 * c, 1.0 - 0.72 * c, 1.0 - 0.72 * c)
            wake_obj.keyframe_insert(data_path="scale", frame=fr)
        elif fr >= 135 and fr <= 190:
            d = C.smootherstep(135.0, 190.0, float(fr))
            wake_obj.scale = (1.0 - 0.6 * d, 1.0 - 0.7 * d, 1.0 - 0.7 * d)
            wake_obj.keyframe_insert(data_path="scale", frame=fr)
        else:
            wake_obj.scale = (1.0, 1.0, 1.0)
            wake_obj.keyframe_insert(data_path="scale", frame=fr)

    _set_action_interp(wake_obj.animation_data, ("rotation_quaternion", "scale"), "LINEAR")

    # Beat 2 (f=300..535): Smooth Bow-Wrapping Hydrodynamic Flow Streamlines
    stream_root = _create_empty("FX_Streamlines_Root", col, parent=root)
    mat_stream, stream_phase = _mat_streamline_flow(
        "Mat_FX_Streamlines",
        stream_root,
        color=(0.36, 0.94, 1.0),
        strength=1.50,
    )
    bm_st = bmesh.new()
    # 12 smooth body-hugging aerodynamic bow streamlines wrapping cleanly around cavitation boundary
    for s_idx in range(12):
        ang = (s_idx * 2.0 * math.pi) / 12.0 + 0.15
        pts = []
        rads = []
        n_pts = 13
        for k in range(n_pts):
            u = k / (n_pts - 1)
            x = C.lerp(0.044, -0.22, u)
            r = 0.003 + 0.054 * math.sin(min(1.0, u * 1.45) * math.pi * 0.5)
            rad_t = 0.0014 * math.sin(max(0.08, min(0.92, u)) * math.pi)
            pts.append(Vector((x, r * math.cos(ang), r * math.sin(ang))))
            rads.append(rad_t)
        _add_swept_tube(bm_st, pts, rads, segs=8, cap_ends=False)

    stream_obj = _finalize_bmesh(
        "FX_Hydro_Streamlines",
        bm_st,
        mat_stream,
        col,
        parent=stream_root,
        smooth=True,
        shadow=False,
    )
    _set_visibility_windows(stream_obj, [(300, 535)], total_frames)

    for fr in _sample_frames(300, 535, step=4, dense_ranges=((300, 322, 2), (510, 535, 2))):
        hx = C.supersonic_x(float(fr))
        hz = C.supersonic_z(float(fr))
        stream_root.location = (hx, 0.0, hz)
        stream_root.keyframe_insert(data_path="location", frame=fr)

        bloom = C.smootherstep(300.0, 318.0, float(fr)) * (1.0 - C.smootherstep(512.0, 535.0, float(fr)))
        sc = 0.15 + 0.85 * bloom
        stream_obj.scale = (sc, sc, sc)
        stream_obj.keyframe_insert(data_path="scale", frame=fr)

        stream_phase.outputs[0].default_value = (fr - 300.0) * 0.42
        stream_phase.outputs[0].keyframe_insert(data_path="default_value", frame=fr)

    _set_action_interp(stream_root.animation_data, ("location",), "LINEAR")
    _set_action_interp(stream_obj.animation_data, ("scale",), "LINEAR")
    _set_action_interp(mat_stream.node_tree.animation_data, ("default_value",), "LINEAR")

    return {
        "cavitation_root": cav_root,
        "cavitation_cone": cav_obj,
        "cavitation_wake": wake_obj,
        "streamlines_root": stream_root,
        "streamlines": stream_obj,
    }


# ---------------------------------------------------------------------------
# 4. BEAT 5 DUAL-LANE COMPARISON RACE (f=1261..1560)
# ---------------------------------------------------------------------------
def _build_subsonic_slug_bmesh() -> bmesh.types.BMesh:
    bm = bmesh.new()
    # Chunky heavy subsonic round-nose slug (length 5.6 cm, radius 1.14 cm)
    prof = [
        (-0.028, 0.0108),
        (-0.022, 0.0114),
        (-0.008, 0.0114),
        (-0.006, 0.0108),  # Heavy grease/crimp groove
        (-0.004, 0.0114),
        (0.008, 0.0114),
        (0.016, 0.0104),
        (0.022, 0.0082),
        (0.026, 0.0048),
        (0.028, 0.0014),
    ]
    _add_revolved_profile(bm, prof, segs=32, cap_start=True, cap_end=True)
    return bm


def _build_beat5_race(
    root: bpy.types.Object,
    col: bpy.types.Collection,
    total_frames: int,
) -> dict[str, Any]:
    fast_z = float(C.TANK["race_fast_z"])   # -0.28
    slow_z = float(C.TANK["race_slow_z"])   # -0.84

    # --- Top Lane: Supersonic + Violent Orange Cavitation + 3 FT Shatter ---
    fast_root = _create_empty("FX_RaceFast_Root", col, parent=root)
    mat_fast_jacket = _mat_shard_metallic(
        "Mat_FX_RaceFastCopper",
        base_color=(0.86, 0.44, 0.18),
        glow_color=(1.0, 0.42, 0.08),
        metallic=0.92,
        roughness=0.20,
        with_heat=False,
    )[0]
    fast_bullet = _finalize_bmesh(
        "FX_RaceFast_Bullet",
        _build_spitzer_jacket_bmesh(),
        mat_fast_jacket,
        col,
        parent=fast_root,
        smooth=True,
        shadow=True,
    )
    _set_visibility_windows(fast_bullet, [(1261, 1418)], total_frames)

    mat_fast_cav = _mat_cavitation_vapor(
        "Mat_FX_RaceFastCavitation",
        fast_root,
        color=(1.0, 0.50, 0.10),
        rim_strength=2.25,
        x_front=0.036,
        x_back=-0.26,
    )
    bm_fc = bmesh.new()
    fc_prof = [
        (C.lerp(0.036, -0.26, i / 17.0), 0.0026 + 0.080 * ((i / 17.0) ** 0.58))
        for i in range(18)
    ]
    _add_revolved_profile(bm_fc, fc_prof, segs=30, cap_start=False, cap_end=False)
    fast_cav = _finalize_bmesh(
        "FX_RaceFast_Cavitation",
        bm_fc,
        mat_fast_cav,
        col,
        parent=fast_root,
        smooth=True,
        shadow=False,
    )
    _set_visibility_windows(fast_cav, [(1285, 1420)], total_frames)

    frames_fast = _sample_frames(1261, 1420, step=4, dense_ranges=((1285, 1300, 2), (1395, 1420, 2)))
    prev_qf: Any = None
    for fr in frames_fast:
        fx = C.race_fast_x(float(fr))
        fast_root.location = (fx, 0.0, fast_z)
        wob = C.smootherstep(1392.0, 1410.0, float(fr))
        qf = Euler(
            ((fr - 1261.0) * 0.22, math.radians(15.0 * wob), math.radians(-26.0 * wob)),
            "XYZ",
        ).to_quaternion()
        if prev_qf is not None and qf.dot(prev_qf) < 0.0:
            qf.negate()
        prev_qf = qf.copy()
        fast_root.rotation_quaternion = qf
        fast_root.keyframe_insert(data_path="location", frame=fr)
        fast_root.keyframe_insert(data_path="rotation_quaternion", frame=fr)

        if fr >= 1285:
            bloom = C.smootherstep(1285.0, 1300.0, float(fr))
            coll = C.smootherstep(1398.0, 1420.0, float(fr))
            sc_x = (0.3 + 0.7 * bloom) * (1.0 - 0.72 * coll)
            sc_yz = (0.3 + 0.7 * bloom) * (1.0 - 0.85 * coll)
            fast_cav.scale = (sc_x, sc_yz, sc_yz)
            fast_cav.keyframe_insert(data_path="scale", frame=fr)

    _set_action_interp(fast_root.animation_data, ("location", "rotation_quaternion"), "LINEAR")
    _set_action_interp(fast_cav.animation_data, ("scale",), "LINEAR")

    # Top Lane 6 Shards at X = 0.9144m (f=1398..1560)
    mat_r_sc, heat_r_sc = _mat_shard_metallic(
        "Mat_FX_RaceFastShardCopper",
        base_color=(0.84, 0.43, 0.19),
        glow_color=(1.0, 0.42, 0.08),
        with_heat=True,
    )
    mat_r_sl, heat_r_sl = _mat_shard_metallic(
        "Mat_FX_RaceFastShardLead",
        base_color=(0.34, 0.36, 0.40),
        glow_color=(1.0, 0.18, 0.08),
        metallic=0.88,
        roughness=0.34,
        with_heat=True,
    )
    race_shard_dirs = [
        (-0.04,  0.070,  0.080,  5.5,  3.2, -3.8),
        ( 0.05, -0.082,  0.062, -4.8,  6.5,  2.9),
        (-0.05, -0.062, -0.088,  4.2, -5.8,  5.0),
        ( 0.03,  0.090, -0.065, -6.5,  2.8,  5.5),
        ( 0.05, -0.045,  0.038, -4.0, -3.5,  4.2),
        (-0.03,  0.040, -0.058,  3.5,  4.8, -3.2),
    ]
    race_shards: list[bpy.types.Object] = []
    frames_r_sh = _sample_frames(1398, 1560, step=5, dense_ranges=((1398, 1430, 2),))
    for idx in range(6):
        if idx < 4:
            bm_rs = _build_copper_strip_bmesh(idx)
            r_mat = mat_r_sc
        else:
            bm_rs = _build_deformed_core_chunk_bmesh(idx - 4)
            r_mat = mat_r_sl
        for v in bm_rs.verts:
            v.co *= 1.55
        rs_obj = _finalize_bmesh(
            f"FX_RaceFast_Shard_{idx:02d}",
            bm_rs,
            r_mat,
            col,
            parent=root,
            smooth=True,
            shadow=True,
        )
        _set_visibility_windows(rs_obj, [(1398, 1560)], total_frames)
        race_shards.append(rs_obj)

        dx, dy, dz, rx, ry, rz = race_shard_dirs[idx]
        prev_qr: Any = None
        for fr in frames_r_sh:
            if fr < 1410:
                rx_p = C.race_fast_x(float(fr))
                pos = Vector((rx_p, 0.0, fast_z))
                q = Euler((0.0, 0.0, 0.0), "XYZ").to_quaternion()
            else:
                st = C.supersonic_shatter_t(float(fr))
                bu = C.clamp((st - 0.025) / 0.975, 0.0, 1.0)
                du = C.smootherstep(1410.0, 1560.0, float(fr))
                pos = Vector((
                    C.THREE_FT_X + dx * bu,
                    dy * bu,
                    fast_z + dz * bu - 0.14 * (du ** 1.3),
                ))
                t_sp = (fr - 1410.0) / 60.0
                fl = 1.0 - 0.5 * du
                q = Euler((rx * t_sp * fl, ry * t_sp * fl, rz * t_sp * fl), "XYZ").to_quaternion()
            if prev_qr is not None and q.dot(prev_qr) < 0.0:
                q.negate()
            prev_qr = q.copy()
            rs_obj.location = pos
            rs_obj.rotation_quaternion = q
            rs_obj.keyframe_insert(data_path="location", frame=fr)
            rs_obj.keyframe_insert(data_path="rotation_quaternion", frame=fr)

        _set_action_interp(rs_obj.animation_data, ("location", "rotation_quaternion"), "LINEAR")

    for fr in (1398, 1410, 1470, 1545):
        h = max(0.45, 1.0 - 0.55 * C.smootherstep(1415.0, 1545.0, float(fr)))
        for hn in (heat_r_sc, heat_r_sl):
            if hn is not None:
                hn.outputs[0].default_value = h
                hn.outputs[0].keyframe_insert(data_path="default_value", frame=fr)
    _set_action_interp(mat_r_sc.node_tree.animation_data, ("default_value",), "LINEAR")
    _set_action_interp(mat_r_sl.node_tree.animation_data, ("default_value",), "LINEAR")

    # --- Bottom Lane: Heavy Subsonic Round + Calm Laminar Cyan Slipstream ---
    slow_root = _create_empty("FX_RaceSlow_Root", col, parent=root)
    mat_sub = _mat_shard_metallic(
        "Mat_FX_SubsonicSlug",
        base_color=(0.76, 0.45, 0.22),
        glow_color=(0.0, 0.0, 0.0),
        metallic=0.90,
        roughness=0.25,
        with_heat=False,
    )[0]
    slow_bullet = _finalize_bmesh(
        "FX_RaceSlow_Bullet",
        _build_subsonic_slug_bmesh(),
        mat_sub,
        col,
        parent=slow_root,
        smooth=True,
        shadow=True,
    )
    _set_visibility_windows(slow_bullet, [(1261, 1560)], total_frames)

    # Calm thin laminar cyan slipstream (open-ended sheath + 6 smooth flow tubes)
    mat_slip, slip_phase = _mat_streamline_flow(
        "Mat_FX_LaminarSlipstream",
        slow_root,
        color=(0.24, 0.88, 1.0),
        strength=1.20,
    )
    bm_sl = bmesh.new()
    # Thin hugging nose-to-tail laminar cone (open-ended)
    slip_prof = [
        (0.032, 0.002),
        (0.018, 0.013),
        (0.000, 0.016),
        (-0.045, 0.018),
        (-0.120, 0.017),
        (-0.180, 0.012),
    ]
    _add_revolved_profile(bm_sl, slip_prof, segs=24, cap_start=False, cap_end=False)
    for k in range(6):
        ang = (k * 2.0 * math.pi) / 6.0
        pts = [
            Vector((
                C.lerp(0.036, -0.175, u),
                (0.003 + 0.020 * math.sin(min(1.0, u * 1.4) * math.pi * 0.5)) * math.cos(ang),
                (0.003 + 0.020 * math.sin(min(1.0, u * 1.4) * math.pi * 0.5)) * math.sin(ang),
            ))
            for u in [j / 9.0 for j in range(10)]
        ]
        rads = [0.0014 * math.sin(max(0.08, min(0.92, j / 9.0)) * math.pi) for j in range(10)]
        _add_swept_tube(bm_sl, pts, rads, segs=6, cap_ends=False)

    slow_slip = _finalize_bmesh(
        "FX_RaceSlow_Slipstream",
        bm_sl,
        mat_slip,
        col,
        parent=slow_root,
        smooth=True,
        shadow=False,
    )
    _set_visibility_windows(slow_slip, [(1285, 1560)], total_frames)

    prev_qs_l: Any = None
    for fr in _sample_frames(1261, 1560, step=4, dense_ranges=((1285, 1305, 2),)):
        sx = C.race_slow_x(float(fr))
        slow_root.location = (sx, 0.0, slow_z)
        ql = Euler(((fr - 1261.0) * 0.07, 0.0, 0.0), "XYZ").to_quaternion()
        if prev_qs_l is not None and ql.dot(prev_qs_l) < 0.0:
            ql.negate()
        prev_qs_l = ql.copy()
        slow_bullet.rotation_quaternion = ql
        slow_root.keyframe_insert(data_path="location", frame=fr)
        slow_bullet.keyframe_insert(data_path="rotation_quaternion", frame=fr)

        if fr >= 1285:
            b = C.smootherstep(1285.0, 1305.0, float(fr))
            slow_slip.scale = (0.3 + 0.7 * b, 0.3 + 0.7 * b, 0.3 + 0.7 * b)
            slow_slip.keyframe_insert(data_path="scale", frame=fr)
            slip_phase.outputs[0].default_value = (fr - 1285.0) * 0.25
            slip_phase.outputs[0].keyframe_insert(data_path="default_value", frame=fr)

    _set_action_interp(slow_root.animation_data, ("location",), "LINEAR")
    _set_action_interp(slow_bullet.animation_data, ("rotation_quaternion",), "LINEAR")
    _set_action_interp(slow_slip.animation_data, ("scale",), "LINEAR")
    _set_action_interp(mat_slip.node_tree.animation_data, ("default_value",), "LINEAR")

    return {
        "race_fast_root": fast_root,
        "race_fast_bullet": fast_bullet,
        "race_fast_cavitation": fast_cav,
        "race_fast_shards": race_shards,
        "race_slow_root": slow_root,
        "race_slow_bullet": slow_bullet,
        "race_slow_slipstream": slow_slip,
    }


# ---------------------------------------------------------------------------
# 5. BEAT 6 HARMLESS SCRAP DRIFT (f=1561..1800)
# ---------------------------------------------------------------------------
def _build_beat6_scrap(
    root: bpy.types.Object,
    col: bpy.types.Collection,
    total_frames: int,
) -> list[bpy.types.Object]:
    """6 crushed copper flakes drifting gently downward around X=0.88..0.95m, Z=-0.75..-1.25m."""
    scrap_root = _create_empty("FX_Scrap_Root", col, parent=root)
    mat_scrap = _mat_shard_metallic(
        "Mat_FX_ScrapCopper",
        base_color=(0.78, 0.42, 0.20),
        glow_color=(0.0, 0.0, 0.0),
        metallic=0.92,
        roughness=0.28,
        with_heat=False,
    )[0]

    # Deterministic flake anchor coords strictly inside X=[0.88, 0.95], Z=[-0.75, -1.25]
    # well clear of diver at X=1.42m
    flake_specs = [
        (0.882, -0.045, -0.75, -0.32),
        (0.914,  0.038, -0.79, -0.34),
        (0.946, -0.022, -0.83, -0.31),
        (0.895,  0.052, -0.88, -0.30),
        (0.928, -0.058, -0.92, -0.29),
        (0.905,  0.012, -0.96, -0.27),
    ]

    flakes: list[bpy.types.Object] = []
    frames_b6 = _sample_frames(1561, 1800, step=8)
    for idx, (fx, fy, z0, dz) in enumerate(flake_specs):
        bm_fl = _build_copper_strip_bmesh(idx % 5)
        # Flatten slightly into crushed copper scrap flake
        for v in bm_fl.verts:
            v.co.z *= 0.65
            v.co *= 1.12
        fl_obj = _finalize_bmesh(
            f"FX_Scrap_Flake_{idx:02d}",
            bm_fl,
            mat_scrap,
            col,
            parent=scrap_root,
            smooth=True,
            shadow=True,
        )
        _set_visibility_windows(fl_obj, [(1561, 1800)], total_frames)
        flakes.append(fl_obj)

        prev_q: Any = None
        for fr in frames_b6:
            u = C.smootherstep(1561.0, 1800.0, float(fr))
            sway_x = 0.008 * math.sin(u * 4.2 + idx * 1.1)
            sway_y = 0.012 * math.cos(u * 3.6 + idx * 0.9)
            z = C.clamp(z0 + dz * u, -1.25, -0.75)
            fl_obj.location = (C.clamp(fx + sway_x, 0.88, 0.95), fy + sway_y, z)

            q = Euler(
                (
                    0.5 + 0.9 * u + idx * 0.6,
                    0.3 * math.sin(u * 3.5 + idx),
                    idx * 1.1 + 0.8 * u,
                ),
                "XYZ",
            ).to_quaternion()
            if prev_q is not None and q.dot(prev_q) < 0.0:
                q.negate()
            prev_q = q.copy()
            fl_obj.rotation_quaternion = q
            fl_obj.keyframe_insert(data_path="location", frame=fr)
            fl_obj.keyframe_insert(data_path="rotation_quaternion", frame=fr)

        _set_action_interp(fl_obj.animation_data, ("location", "rotation_quaternion"), "LINEAR")

    return flakes


# ---------------------------------------------------------------------------
# PUBLIC ENTRYPOINT
# ---------------------------------------------------------------------------
def build_bullet_fx(
    col: bpy.types.Collection,
    total_frames: int = C.TOTAL_FRAMES,
) -> dict[str, Any]:
    """Build and keyframe all Episode #2 ballistic, cavitation, X-ray & race FX."""
    root = _create_empty("BulletFX_Root", col, parent=None)

    barrel_info = _build_barrel_and_flash(root, col, total_frames)
    hero_info = _build_hero_round_and_shards(root, col, total_frames)
    cav_info = _build_cavitation_and_streamlines(root, col, total_frames)
    race_info = _build_beat5_race(root, col, total_frames)
    scrap_flakes = _build_beat6_scrap(root, col, total_frames)

    all_objs: list[bpy.types.Object] = [
        root,
        barrel_info["barrel_anchor"],
        barrel_info["barrel"],
        barrel_info["muzzle_flash"],
        hero_info["hero_root"],
        hero_info["hero_jacket"],
        hero_info["hero_core"],
        *hero_info["hero_shards"],
        cav_info["cavitation_root"],
        cav_info["cavitation_cone"],
        cav_info["cavitation_wake"],
        cav_info["streamlines_root"],
        cav_info["streamlines"],
        race_info["race_fast_root"],
        race_info["race_fast_bullet"],
        race_info["race_fast_cavitation"],
        *race_info["race_fast_shards"],
        race_info["race_slow_root"],
        race_info["race_slow_bullet"],
        race_info["race_slow_slipstream"],
        *scrap_flakes,
    ]

    return {
        "root": root,
        **barrel_info,
        **hero_info,
        **cav_info,
        **race_info,
        "scrap_flakes": scrap_flakes,
        "all_objs": all_objs,
    }
