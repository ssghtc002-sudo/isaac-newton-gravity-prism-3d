"""
vest_fx.py — Macro Multi-Layer Ballistic Armor Cutaway (X-Ray Nylon Carrier,
Woven Golden Kevlar Trampoline Mesh, Silicon-Carbide Ceramic Strike Plate + Fracture Shards,
UHMWPE Dyneema Trauma Backer), Hero Projectiles, Muzzle Flashes, 3D Tapered Spark Bursts,
Beat 5 Newton's Third Law Opposing Force Vectors (F1 = -F2) & Beat 6 Settling Dust
for Episode #3: "Why Bulletproof Vests DON'T Send You Flying Backward" (60 fps, 1800 frames).

Strictly follows AGENTS.md V12 & Blender 5.3.0 rules:
- Imports all dimensions, milestones, trajectories, and curves from `vest_contract as C`.
- Uses `C.iter_action_fcurves(action)` for Blender 5.x Slotted Actions.
- Uses `C.get_mix_sockets(node)` for all `ShaderNodeMix` sockets (`inputs[2:]`).
- Attaches `mesh.materials` BEFORE `bm.to_mesh(mesh)`.
- Open-ended optical/muzzle/spark cones (`cap_ends=False`, zero flat end-caps).
- Optical beams/flashes/sparks use `AddShader(Transparent, Emission * CameraRay * (1 - Backfacing))`.
- 100% Local-Space shaders driven by `ShaderNodeTexCoord.Object` bound to local Empty.

Public API:
    build_vest_fx(col: bpy.types.Collection, total_frames: int = C.TOTAL_FRAMES) -> dict[str, Any]
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any

import bmesh
import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import vest_contract as C  # noqa: E402


# ---------------------------------------------------------------------------
# BLENDER 5.3.0 HELPERS
# ---------------------------------------------------------------------------
def _link_to_col(obj: bpy.types.Object, col: bpy.types.Collection) -> bpy.types.Object:
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)
    return obj


def _create_empty(
    name: str,
    col: bpy.types.Collection,
    parent: bpy.types.Object | None = None,
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = "PLAIN_AXES"
    obj.empty_display_size = 0.04
    obj.location = Vector(location)
    obj.rotation_mode = "QUATERNION"
    if parent is not None:
        obj.parent = parent
    return _link_to_col(obj, col)


def _finalize_bmesh(
    name: str,
    bm: bmesh.types.BMesh,
    materials: bpy.types.Material | list[bpy.types.Material],
    col: bpy.types.Collection,
    parent: bpy.types.Object | None = None,
    smooth: bool = True,
    shadow: bool = False,
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    """Attach material slots BEFORE `bm.to_mesh(mesh)` per AGENTS.md V12."""
    if len(bm.faces) > 0:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for f in bm.faces:
        f.smooth = smooth
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mat_list = materials if isinstance(materials, list) else [materials]
    for m in mat_list:
        mesh.materials.append(m)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(location)
    obj.rotation_mode = "QUATERNION"
    obj.visible_shadow = shadow
    if parent is not None:
        obj.parent = parent
    return _link_to_col(obj, col)


def _set_visibility_windows(
    obj: bpy.types.Object,
    windows: list[tuple[int, int]],
    total_frames: int = C.TOTAL_FRAMES,
) -> None:
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


def _set_action_interp(anim_data: Any, interp: str = "LINEAR", skip_prefix: str = "hide_") -> None:
    if not (anim_data and anim_data.action):
        return
    for fc in C.iter_action_fcurves(anim_data.action):
        if skip_prefix and fc.data_path.startswith(skip_prefix):
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = interp


def _sample_frames(
    start: int,
    end: int,
    step: int = 4,
    dense_ranges: tuple[tuple[int, int, int], ...] = (),
) -> list[int]:
    frames: set[int] = {start, end}
    for f in range(start, end + 1, max(1, step)):
        frames.add(f)
    for ds, de, dstep in dense_ranges:
        for f in range(max(start, ds), min(end, de) + 1, max(1, dstep)):
            frames.add(f)
    return sorted(frames)


# ---------------------------------------------------------------------------
# PROCEDURAL BMESH GEOMETRY PRIMITIVES (OPEN-ENDED CONES & SWEPT TUBES)
# ---------------------------------------------------------------------------
def _add_revolved_profile(
    bm: bmesh.types.BMesh,
    profile_xr: list[tuple[float, float]],
    segs: int = 32,
    cap_start: bool = False,
    cap_end: bool = False,
    radial_mod_fn: Any = None,
) -> list[list[bmesh.types.BMVert]]:
    rings: list[list[bmesh.types.BMVert]] = []
    for ring_idx, (x, r) in enumerate(profile_xr):
        ring: list[bmesh.types.BMVert] = []
        for s in range(segs):
            ang = (math.tau * s) / segs
            rr = radial_mod_fn(x, r, ang, ring_idx, s) if radial_mod_fn else r
            ring.append(bm.verts.new((x, rr * math.cos(ang), rr * math.sin(ang))))
        rings.append(ring)

    for i in range(len(rings) - 1):
        r0, r1 = rings[i], rings[i + 1]
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
    segs: int = 8,
    cap_ends: bool = False,
) -> None:
    """Sweep an open-ended (`cap_ends=False`) smooth 3D tube along points."""
    if len(points) < 2:
        return
    rings: list[list[bmesh.types.BMVert]] = []
    prev_n = Vector((0.0, 0.0, 1.0))
    for idx, pt in enumerate(points):
        if idx == 0:
            tan = (points[1] - points[0]).normalized()
        elif idx == len(points) - 1:
            tan = (points[-1] - points[-2]).normalized()
        else:
            tan = (points[idx + 1] - points[idx - 1]).normalized()
        if tan.length < 1e-6:
            tan = Vector((1.0, 0.0, 0.0))

        n = prev_n - tan * prev_n.dot(tan)
        if n.length < 1e-5:
            alt = Vector((0.0, 1.0, 0.0)) if abs(tan.y) < 0.8 else Vector((1.0, 0.0, 0.0))
            n = alt - tan * alt.dot(tan)
        n.normalize()
        b = tan.cross(n).normalized()
        prev_n = n

        rad = radii[idx]
        ring = [
            bm.verts.new(pt + n * (rad * math.cos(math.tau * s / segs)) + b * (rad * math.sin(math.tau * s / segs)))
            for s in range(segs)
        ]
        rings.append(ring)

    for i in range(len(rings) - 1):
        r0, r1 = rings[i], rings[i + 1]
        for s in range(segs):
            sn = (s + 1) % segs
            bm.faces.new((r0[s], r0[sn], r1[sn], r1[s]))
    if cap_ends and rings:
        bm.faces.new(list(reversed(rings[0])))
        bm.faces.new(rings[-1])


# ---------------------------------------------------------------------------
# LOCAL-SPACE SHADERS (100% TexCoord.Object, Blender 5.3.0 & AgX Safe)
# ---------------------------------------------------------------------------
def _mat_xray_carrier(local_root: bpy.types.Object) -> tuple[bpy.types.Material, bpy.types.Node]:
    """Outer black nylon carrier fading into hollow Fresnel cyan X-ray cutaway with shadow bypass."""
    mat = bpy.data.materials.new("Mat_Vest_CarrierXRay")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    tex = nt.nodes.new("ShaderNodeTexCoord")
    tex.object = local_root

    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 90.0
    noise.inputs["Detail"].default_value = 4.0
    nt.links.new(tex.outputs["Object"], noise.inputs["Vector"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    bump.inputs["Distance"].default_value = 0.001
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.032, 0.038, 0.048, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.62
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # Hollow Fresnel cyan shell (rim capped <= 0.24)
    lp = nt.nodes.new("ShaderNodeLightPath")
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    inv_bf = nt.nodes.new("ShaderNodeMath")
    inv_bf.operation = "SUBTRACT"
    inv_bf.inputs[0].default_value = 1.0
    nt.links.new(geom.outputs["Backfacing"], inv_bf.inputs[1])

    cam_front = nt.nodes.new("ShaderNodeMath")
    cam_front.operation = "MULTIPLY"
    nt.links.new(lp.outputs["Is Camera Ray"], cam_front.inputs[0])
    nt.links.new(inv_bf.outputs["Value"], cam_front.inputs[1])

    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.35
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = 3.4
    nt.links.new(lw.outputs["Facing"], pw.inputs[0])

    rim_str = nt.nodes.new("ShaderNodeMath")
    rim_str.operation = "MULTIPLY"
    nt.links.new(cam_front.outputs["Value"], rim_str.inputs[0])
    nt.links.new(pw.outputs["Value"], rim_str.inputs[1])

    rim_gain = nt.nodes.new("ShaderNodeMath")
    rim_gain.operation = "MULTIPLY"
    rim_gain.inputs[1].default_value = 0.22
    nt.links.new(rim_str.outputs["Value"], rim_gain.inputs[0])

    emis = nt.nodes.new("ShaderNodeEmission")
    emis.inputs["Color"].default_value = (0.20, 0.88, 1.0, 1.0)
    nt.links.new(rim_gain.outputs["Value"], emis.inputs["Strength"])

    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    xray_add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(transp.outputs["BSDF"], xray_add.inputs[0])
    nt.links.new(emis.outputs["Emission"], xray_add.inputs[1])

    xray_val = nt.nodes.new("ShaderNodeValue")
    xray_val.name = "VestXRay"
    xray_val.outputs[0].default_value = 0.0

    mix_x = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(xray_val.outputs[0], mix_x.inputs["Fac"])
    nt.links.new(bsdf.outputs["BSDF"], mix_x.inputs[1])
    nt.links.new(xray_add.outputs["Shader"], mix_x.inputs[2])

    shad_gate = nt.nodes.new("ShaderNodeMath")
    shad_gate.operation = "MULTIPLY"
    nt.links.new(lp.outputs["Is Shadow Ray"], shad_gate.inputs[0])
    nt.links.new(xray_val.outputs[0], shad_gate.inputs[1])

    mix_shad = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(shad_gate.outputs["Value"], mix_shad.inputs["Fac"])
    nt.links.new(mix_x.outputs["Shader"], mix_shad.inputs[1])
    nt.links.new(transp.outputs["BSDF"], mix_shad.inputs[2])
    nt.links.new(mix_shad.outputs["Shader"], out.inputs["Surface"])
    return mat, xray_val


def _mat_golden_kevlar(local_root: bpy.types.Object) -> tuple[bpy.types.Material, bpy.types.Node]:
    """
    Luminous woven golden aramid Kevlar shader driven 100% by `TexCoord.Object`:
    Features fine woven fabric bump + concentric glowing golden energy ripple ring expanding
    radially across the trampoline net as `KevlarRipple` ramps 0 -> 1 in Beat 3.
    """
    mat = bpy.data.materials.new("Mat_Vest_GoldenKevlar")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    tex = nt.nodes.new("ShaderNodeTexCoord")
    tex.object = local_root
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tex.outputs["Object"], sep.inputs["Vector"])

    # Compute local radial distance r = sqrt(y^2 + z^2) from center impact (0, 0)
    y2 = nt.nodes.new("ShaderNodeMath")
    y2.operation = "MULTIPLY"
    nt.links.new(sep.outputs["Y"], y2.inputs[0])
    nt.links.new(sep.outputs["Y"], y2.inputs[1])

    z2 = nt.nodes.new("ShaderNodeMath")
    z2.operation = "MULTIPLY"
    nt.links.new(sep.outputs["Z"], z2.inputs[0])
    nt.links.new(sep.outputs["Z"], z2.inputs[1])

    r2 = nt.nodes.new("ShaderNodeMath")
    r2.operation = "ADD"
    nt.links.new(y2.outputs["Value"], r2.inputs[0])
    nt.links.new(z2.outputs["Value"], r2.inputs[1])

    r_len = nt.nodes.new("ShaderNodeMath")
    r_len.operation = "SQRT"
    nt.links.new(r2.outputs["Value"], r_len.inputs[0])

    # Fine micro-fiber sheen & woven anisotropic bump
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 180.0
    noise.inputs["Detail"].default_value = 5.0
    nt.links.new(tex.outputs["Object"], noise.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.14
    bump.inputs["Distance"].default_value = 0.0008
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])

    ripple_val = nt.nodes.new("ShaderNodeValue")
    ripple_val.name = "KevlarRipple"
    ripple_val.outputs[0].default_value = 0.0

    # Concentric energy wave ring expanding outward: sin(r * 145 - ripple * 18) * radial falloff
    r_freq = nt.nodes.new("ShaderNodeMath")
    r_freq.operation = "MULTIPLY"
    r_freq.inputs[1].default_value = 135.0
    nt.links.new(r_len.outputs["Value"], r_freq.inputs[0])

    ph = nt.nodes.new("ShaderNodeMath")
    ph.operation = "MULTIPLY"
    ph.inputs[1].default_value = 18.0
    nt.links.new(ripple_val.outputs[0], ph.inputs[0])

    wave_arg = nt.nodes.new("ShaderNodeMath")
    wave_arg.operation = "SUBTRACT"
    nt.links.new(r_freq.outputs["Value"], wave_arg.inputs[0])
    nt.links.new(ph.outputs["Value"], wave_arg.inputs[1])

    wave_sin = nt.nodes.new("ShaderNodeMath")
    wave_sin.operation = "SINE"
    nt.links.new(wave_arg.outputs["Value"], wave_sin.inputs[0])

    wave_pos = nt.nodes.new("ShaderNodeMapRange")
    wave_pos.clamp = True
    wave_pos.inputs["From Min"].default_value = 0.10
    wave_pos.inputs["From Max"].default_value = 1.0
    wave_pos.inputs["To Min"].default_value = 0.0
    wave_pos.inputs["To Max"].default_value = 1.0
    nt.links.new(wave_sin.outputs["Value"], wave_pos.inputs["Value"])

    # Radial envelope expanding to r = 0.16 m
    r_env = nt.nodes.new("ShaderNodeMapRange")
    r_env.clamp = True
    r_env.inputs["From Min"].default_value = 0.0
    r_env.inputs["From Max"].default_value = 0.165
    r_env.inputs["To Min"].default_value = 1.0
    r_env.inputs["To Max"].default_value = 0.0
    nt.links.new(r_len.outputs["Value"], r_env.inputs["Value"])

    em_mask = nt.nodes.new("ShaderNodeMath")
    em_mask.operation = "MULTIPLY"
    nt.links.new(wave_pos.outputs["Result"], em_mask.inputs[0])
    nt.links.new(r_env.outputs["Result"], em_mask.inputs[1])

    em_active = nt.nodes.new("ShaderNodeMath")
    em_active.operation = "MULTIPLY"
    nt.links.new(em_mask.outputs["Value"], em_active.inputs[0])
    nt.links.new(ripple_val.outputs[0], em_active.inputs[1])

    # AgX-safe golden tension glow (0.14 ambient golden luster -> 1.80 ripple crests)
    em_str = nt.nodes.new("ShaderNodeMath")
    em_str.operation = "MULTIPLY_ADD"
    em_str.inputs[1].default_value = 1.65
    em_str.inputs[2].default_value = 0.16
    nt.links.new(em_active.outputs["Value"], em_str.inputs[0])

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.86, 0.62, 0.10, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.18
    bsdf.inputs["Roughness"].default_value = 0.34
    if "Sheen Weight" in bsdf.inputs:
        bsdf.inputs["Sheen Weight"].default_value = 0.45
    bsdf.inputs["Emission Color"].default_value = (1.0, 0.72, 0.12, 1.0)
    nt.links.new(em_str.outputs["Value"], bsdf.inputs["Emission Strength"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, ripple_val


def _mat_sic_ceramic(local_root: bpy.types.Object) -> tuple[bpy.types.Material, bpy.types.Node]:
    """
    Silicon-Carbide (`SiC`) dark matte graphite-ceramic strike plate with local-space Voronoi
    brittle fracture lines glowing molten amber-crimson as `CeramicCrack` ramps 0 -> 1 in Beat 4.
    """
    mat = bpy.data.materials.new("Mat_Vest_SiCCeramic")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    tex = nt.nodes.new("ShaderNodeTexCoord")
    tex.object = local_root
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tex.outputs["Object"], sep.inputs["Vector"])

    # Voronoi edge-distance crack network centered in local space
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.feature = "DISTANCE_TO_EDGE"
    vor.inputs["Scale"].default_value = 34.0
    nt.links.new(tex.outputs["Object"], vor.inputs["Vector"])

    crack_edge = nt.nodes.new("ShaderNodeMapRange")
    crack_edge.clamp = True
    crack_edge.inputs["From Min"].default_value = 0.0
    crack_edge.inputs["From Max"].default_value = 0.065
    crack_edge.inputs["To Min"].default_value = 1.0
    crack_edge.inputs["To Max"].default_value = 0.0
    nt.links.new(vor.outputs["Distance"], crack_edge.inputs["Value"])

    # Radial fracture zone expanding around center (0, 0)
    y2 = nt.nodes.new("ShaderNodeMath")
    y2.operation = "MULTIPLY"
    nt.links.new(sep.outputs["Y"], y2.inputs[0])
    nt.links.new(sep.outputs["Y"], y2.inputs[1])
    z2 = nt.nodes.new("ShaderNodeMath")
    z2.operation = "MULTIPLY"
    nt.links.new(sep.outputs["Z"], z2.inputs[0])
    nt.links.new(sep.outputs["Z"], z2.inputs[1])
    r2 = nt.nodes.new("ShaderNodeMath")
    r2.operation = "ADD"
    nt.links.new(y2.outputs["Value"], r2.inputs[0])
    nt.links.new(z2.outputs["Value"], r2.inputs[1])
    r_len = nt.nodes.new("ShaderNodeMath")
    r_len.operation = "SQRT"
    nt.links.new(r2.outputs["Value"], r_len.inputs[0])

    crack_val = nt.nodes.new("ShaderNodeValue")
    crack_val.name = "CeramicCrack"
    crack_val.outputs[0].default_value = 0.0

    r_lim = nt.nodes.new("ShaderNodeMath")
    r_lim.operation = "MULTIPLY"
    r_lim.inputs[1].default_value = 0.145
    nt.links.new(crack_val.outputs[0], r_lim.inputs[0])

    r_mask = nt.nodes.new("ShaderNodeMapRange")
    r_mask.clamp = True
    r_mask.inputs["From Min"].default_value = 0.0
    r_mask.inputs["From Max"].default_value = 0.12
    r_mask.inputs["To Min"].default_value = 1.0
    r_mask.inputs["To Max"].default_value = 0.0
    nt.links.new(r_len.outputs["Value"], r_mask.inputs["Value"])

    c_active = nt.nodes.new("ShaderNodeMath")
    c_active.operation = "MULTIPLY"
    nt.links.new(crack_edge.outputs["Result"], c_active.inputs[0])
    nt.links.new(r_mask.outputs["Result"], c_active.inputs[1])

    c_gain = nt.nodes.new("ShaderNodeMath")
    c_gain.operation = "MULTIPLY"
    nt.links.new(c_active.outputs["Value"], c_gain.inputs[0])
    nt.links.new(crack_val.outputs[0], c_gain.inputs[1])

    em_str = nt.nodes.new("ShaderNodeMath")
    em_str.operation = "MULTIPLY"
    em_str.inputs[1].default_value = 1.95
    nt.links.new(c_gain.outputs["Value"], em_str.inputs[0])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.002
    nt.links.new(c_gain.outputs["Value"], bump.inputs["Height"])

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.075, 0.082, 0.095, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.22
    bsdf.inputs["Roughness"].default_value = 0.28
    bsdf.inputs["Emission Color"].default_value = (1.0, 0.34, 0.05, 1.0)
    nt.links.new(em_str.outputs["Value"], bsdf.inputs["Emission Strength"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, crack_val


def _mat_dyneema_backer(local_root: bpy.types.Object) -> tuple[bpy.types.Material, bpy.types.Node]:
    """Luminous pearlescent cyan-white UHMWPE Dyneema laminate catching pulverized fragments."""
    mat = bpy.data.materials.new("Mat_Vest_Dyneema")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    tex = nt.nodes.new("ShaderNodeTexCoord")
    tex.object = local_root
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (6.0, 95.0, 95.0)
    nt.links.new(tex.outputs["Object"], mapping.inputs["Vector"])

    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 18.0
    noise.inputs["Detail"].default_value = 5.0
    nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.09
    bump.inputs["Distance"].default_value = 0.001
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])

    flex_val = nt.nodes.new("ShaderNodeValue")
    flex_val.name = "DyneemaFlex"
    flex_val.outputs[0].default_value = 0.0

    em_str = nt.nodes.new("ShaderNodeMath")
    em_str.operation = "MULTIPLY_ADD"
    em_str.inputs[1].default_value = 1.25
    em_str.inputs[2].default_value = 0.18
    nt.links.new(flex_val.outputs[0], em_str.inputs[0])

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.76, 0.90, 0.96, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.06
    bsdf.inputs["Roughness"].default_value = 0.30
    bsdf.inputs["Emission Color"].default_value = (0.28, 0.88, 1.0, 1.0)
    nt.links.new(em_str.outputs["Value"], bsdf.inputs["Emission Strength"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, flex_val


def _mat_copper_bullet(local_obj: bpy.types.Object) -> bpy.types.Material:
    mat = bpy.data.materials.new("Mat_FX_CopperJacket")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.90, 0.48, 0.18, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.94
    bsdf.inputs["Roughness"].default_value = 0.20
    bsdf.inputs["Emission Color"].default_value = (1.0, 0.50, 0.14, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 0.35

    tex = nt.nodes.new("ShaderNodeTexCoord")
    tex.object = local_obj
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 42.0
    nt.links.new(tex.outputs["Object"], noise.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.035
    bump.inputs["Distance"].default_value = 0.0004
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _mat_additive_glow(
    name: str,
    rgb: tuple[float, float, float],
    base_strength: float,
    local_obj: bpy.types.Object | None = None,
) -> tuple[bpy.types.Material, bpy.types.Node]:
    """Open-ended AddShader optical glow (`CameraRay * (1 - Backfacing)`), AgX-safe."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    lp = nt.nodes.new("ShaderNodeLightPath")
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    inv_bf = nt.nodes.new("ShaderNodeMath")
    inv_bf.operation = "SUBTRACT"
    inv_bf.inputs[0].default_value = 1.0
    nt.links.new(geom.outputs["Backfacing"], inv_bf.inputs[1])

    cam_front = nt.nodes.new("ShaderNodeMath")
    cam_front.operation = "MULTIPLY"
    nt.links.new(lp.outputs["Is Camera Ray"], cam_front.inputs[0])
    nt.links.new(inv_bf.outputs["Value"], cam_front.inputs[1])

    gain_val = nt.nodes.new("ShaderNodeValue")
    gain_val.name = "GlowIntensity"
    gain_val.outputs[0].default_value = base_strength

    eff = nt.nodes.new("ShaderNodeMath")
    eff.operation = "MULTIPLY"
    nt.links.new(cam_front.outputs["Value"], eff.inputs[0])
    nt.links.new(gain_val.outputs[0], eff.inputs[1])

    emis = nt.nodes.new("ShaderNodeEmission")
    emis.inputs["Color"].default_value = (*rgb, 1.0)
    nt.links.new(eff.outputs["Value"], emis.inputs["Strength"])

    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    add_sh = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(transp.outputs["BSDF"], add_sh.inputs[0])
    nt.links.new(emis.outputs["Emission"], add_sh.inputs[1])
    nt.links.new(add_sh.outputs["Shader"], out.inputs["Surface"])
    _ = local_obj
    return mat, gain_val


def _mat_solid_emissive(
    name: str,
    rgb: tuple[float, float, float],
    strength: float,
    metallic: float = 0.20,
    roughness: float = 0.25,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Emission Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Emission Strength"].default_value = min(1.85, strength)
    return mat


# ---------------------------------------------------------------------------
# 1. MACRO MULTI-LAYER ARMOR CUTAWAY (Beats 3-4, 100% Local Space)
# ---------------------------------------------------------------------------
def _build_macro_vest_cutaway(
    root: bpy.types.Object,
    col: bpy.types.Collection,
    total_frames: int,
) -> dict[str, Any]:
    vest_root = _create_empty("Vest_Macro_Root", col, parent=root, location=tuple(C.VEST["root_pos"]))
    w_y = float(C.VEST["width_y"])
    h_z = float(C.VEST["height_z"])

    mat_carrier, xray_val = _mat_xray_carrier(vest_root)
    mat_kevlar, ripple_val = _mat_golden_kevlar(vest_root)
    mat_ceramic, crack_val = _mat_sic_ceramic(vest_root)
    mat_dyneema, flex_val = _mat_dyneema_backer(vest_root)

    # A. Outer X-Ray Carrier Frame (surrounds layers cleanly, fades translucent via VestXRay)
    bm_car = bmesh.new()
    cx0, cx1 = C.VEST["carrier_x"]
    bmesh.ops.create_cube(bm_car, size=1.0)
    bmesh.ops.scale(bm_car, vec=Vector((cx1 - cx0, w_y + 0.025, h_z + 0.025)), verts=bm_car.verts)
    bmesh.ops.translate(bm_car, vec=Vector((0.5 * (cx0 + cx1), 0.0, 0.0)), verts=bm_car.verts)
    carrier_obj = _finalize_bmesh("Vest_Macro_Carrier", bm_car, mat_carrier, col, parent=vest_root, smooth=False)
    _set_visibility_windows(carrier_obj, [(511, 1260)], total_frames)

    # B. Woven Golden Kevlar Trampoline Mesh (16x16 subdivided 3D interlaced weave + TrampolineBulge shape key!)
    kx0, kx1 = C.VEST["kevlar_x"]
    k_mid_x = 0.5 * (kx0 + kx1)
    bm_k = bmesh.new()
    n_strands = 15
    n_pts = 29
    # Horizontal weft & vertical warp undulating golden aramid ribbons
    for is_vert in (False, True):
        for s_idx in range(n_strands):
            v_pos = C.lerp(-0.46, 0.46, s_idx / float(n_strands - 1))
            pts: list[Vector] = []
            rads: list[float] = []
            for p_idx in range(n_pts):
                u_pos = C.lerp(-0.48, 0.48, p_idx / float(n_pts - 1))
                weave_x = 0.0022 * math.sin((p_idx + s_idx) * math.pi * 0.5) * (1.0 if is_vert else -1.0)
                if is_vert:
                    pts.append(Vector((k_mid_x + weave_x, v_pos * w_y, u_pos * h_z)))
                else:
                    pts.append(Vector((k_mid_x + weave_x, u_pos * w_y, v_pos * h_z)))
                rads.append(0.0034)
            _add_swept_tube(bm_k, pts, rads, segs=6, cap_ends=True)

    kevlar_obj = _finalize_bmesh("Vest_Macro_KevlarWeave", bm_k, mat_kevlar, col, parent=vest_root, smooth=True, shadow=True)
    _set_visibility_windows(kevlar_obj, [(511, 1260)], total_frames)

    # Add physical TrampolineBulge Shape Key displacing center +X by 26 mm with concentric ripple!
    kevlar_obj.shape_key_add(name="Basis", from_mix=False)
    sk_bulge = kevlar_obj.shape_key_add(name="TrampolineBulge", from_mix=False)
    max_bulge = float(C.VEST["max_kevlar_bulge_m"])
    for v_idx, v in enumerate(kevlar_obj.data.vertices):
        r = math.hypot(v.co.y, v.co.z)
        gauss = math.exp(-((r / 0.072) ** 2))
        ring_wave = 0.0045 * math.exp(-r * 14.0) * math.cos(r * 92.0)
        sk_bulge.data[v_idx].co = Vector((v.co.x + max_bulge * gauss + ring_wave, v.co.y, v.co.z))

    # C. Silicon-Carbide (SiC) Ceramic Strike Plate (visible Beat 4: 901..1260)
    sx0, sx1 = C.VEST["ceramic_x"]
    bm_cer = bmesh.new()
    bmesh.ops.create_cube(bm_cer, size=1.0)
    bmesh.ops.scale(bm_cer, vec=Vector((sx1 - sx0, w_y * 0.94, h_z * 0.94)), verts=bm_cer.verts)
    bmesh.ops.bevel(bm_cer, geom=list(bm_cer.edges), offset=0.002, segments=2, affect="EDGES", clamp_overlap=True)
    bmesh.ops.translate(bm_cer, vec=Vector((0.5 * (sx0 + sx1), 0.0, 0.0)), verts=bm_cer.verts)
    ceramic_obj = _finalize_bmesh("Vest_Macro_SiC_Plate", bm_cer, mat_ceramic, col, parent=vest_root, smooth=False, shadow=True)
    _set_visibility_windows(ceramic_obj, [(901, 1260)], total_frames)

    # D. UHMWPE Dyneema Trauma Backer Plate (visible Beats 3-4, flexes backward in Beat 4)
    dx0, dx1 = C.VEST["dyneema_x"]
    dx0, dx1 = dx0 + 0.006, dx1 + 0.006
    bm_dyn = bmesh.new()
    ny_d, nz_d = 18, 22
    # Build subdivided double-sided slab so DyneemaFlex shape key bulges smoothly
    for side_x in (dx0, dx1):
        verts_d = [
            [
                bm_dyn.verts.new((side_x, C.lerp(-0.48 * w_y, 0.48 * w_y, iy / ny_d), C.lerp(-0.48 * h_z, 0.48 * h_z, iz / nz_d)))
                for iz in range(nz_d + 1)
            ]
            for iy in range(ny_d + 1)
        ]
        for iy in range(ny_d):
            for iz in range(nz_d):
                if side_x == dx0:
                    bm_dyn.faces.new((verts_d[iy][iz], verts_d[iy][iz + 1], verts_d[iy + 1][iz + 1], verts_d[iy + 1][iz]))
                else:
                    bm_dyn.faces.new((verts_d[iy][iz], verts_d[iy + 1][iz], verts_d[iy + 1][iz + 1], verts_d[iy][iz + 1]))
    dyneema_obj = _finalize_bmesh("Vest_Macro_DyneemaBacker", bm_dyn, mat_dyneema, col, parent=vest_root, smooth=True, shadow=True)
    _set_visibility_windows(dyneema_obj, [(511, 1260)], total_frames)

    dyneema_obj.shape_key_add(name="Basis", from_mix=False)
    sk_dyn = dyneema_obj.shape_key_add(name="DyneemaFlex", from_mix=False)
    max_db = float(C.VEST["max_dyneema_bulge_m"])
    for v_idx, v in enumerate(dyneema_obj.data.vertices):
        r = math.hypot(v.co.y, v.co.z)
        sk_dyn.data[v_idx].co = Vector((v.co.x + max_db * math.exp(-((r / 0.085) ** 2)), v.co.y, v.co.z))

    # Animate all macro shader values & shape keys across Beats 3-4 (511..1260)
    kfs_macro = _sample_frames(511, 1260, step=6, dense_ranges=((520, 565, 3), (610, 760, 3), (960, 1150, 3)))
    for f in kfs_macro:
        ff = float(f)
        xray_val.outputs[0].default_value = C.xray_vest_weight(ff)
        xray_val.outputs[0].keyframe_insert("default_value", frame=f)

        kt = C.kevlar_ripple_t(ff)
        ripple_val.outputs[0].default_value = kt
        ripple_val.outputs[0].keyframe_insert("default_value", frame=f)
        sk_bulge.value = kt
        sk_bulge.keyframe_insert("value", frame=f)

        ct = C.ceramic_shatter_t(ff)
        crack_val.outputs[0].default_value = ct
        crack_val.outputs[0].keyframe_insert("default_value", frame=f)

        dt = max(C.dyneema_bulge_t(ff), 0.78 * C.kevlar_ripple_t(ff))
        flex_val.outputs[0].default_value = dt
        flex_val.outputs[0].keyframe_insert("default_value", frame=f)
        sk_dyn.value = dt
        sk_dyn.keyframe_insert("value", frame=f)

    for id_b in (
        mat_carrier.node_tree,
        mat_kevlar.node_tree,
        mat_ceramic.node_tree,
        mat_dyneema.node_tree,
        kevlar_obj.data.shape_keys,
        dyneema_obj.data.shape_keys,
    ):
        if id_b and id_b.animation_data:
            _set_action_interp(id_b.animation_data, "LINEAR")

    # E. Beat 4 Exploding Ceramic & Pulverized Copper Tip Shards (16 shards, f=1015..1260)
    mat_shard_cer = _mat_solid_emissive("Mat_FX_CeramicShard", (0.20, 0.22, 0.26), 0.25, metallic=0.18, roughness=0.35)
    mat_shard_cop = _mat_solid_emissive("Mat_FX_PulverizedCopper", (1.0, 0.56, 0.14), 1.65, metallic=0.88, roughness=0.22)
    shards: list[bpy.types.Object] = []
    for idx in range(16):
        bm_s = bmesh.new()
        bmesh.ops.create_icosphere(bm_s, subdivisions=1, radius=0.0045 + 0.0025 * (idx % 3))
        bmesh.ops.scale(bm_s, vec=Vector((1.4, 0.85, 0.65)), verts=bm_s.verts)
        s_mat = mat_shard_cop if idx % 2 == 0 else mat_shard_cer
        sh_obj = _finalize_bmesh(f"FX_CeramicShard_{idx:02d}", bm_s, s_mat, col, parent=vest_root, smooth=False)
        _set_visibility_windows(sh_obj, [(1005, 1260)], total_frames)

        ang = idx * (math.tau / 16.0) + 0.25
        # Half spray backward/sideways off ceramic strike face; half caught inside Dyneema pocket
        is_caught = idx % 3 != 0
        dx_end = 0.030 if is_caught else -0.055 - 0.03 * (idx % 4)
        rad_end = 0.055 + 0.065 * ((idx * 7) % 5) / 4.0
        for f in _sample_frames(1005, 1260, step=8, dense_ranges=((1005, 1130, 4),)):
            u = C.smootherstep(1005.0, 1135.0, float(f))
            sh_obj.location = Vector((
                0.004 + dx_end * u,
                math.cos(ang) * rad_end * u,
                math.sin(ang) * rad_end * u - 0.020 * (u * u),
            ))
            sh_obj.rotation_quaternion = Euler((u * 4.5 + idx, u * 6.0, u * 3.5), "XYZ").to_quaternion()
            sh_obj.keyframe_insert("location", frame=f)
            sh_obj.keyframe_insert("rotation_quaternion", frame=f)
        _set_action_interp(sh_obj.animation_data, "LINEAR")
        shards.append(sh_obj)

    return {
        "vest_root": vest_root,
        "carrier": carrier_obj,
        "kevlar": kevlar_obj,
        "ceramic": ceramic_obj,
        "dyneema": dyneema_obj,
        "shards": shards,
    }


# ---------------------------------------------------------------------------
# 2. HERO PROJECTILES (9mm Pistol Beats 1-3 + Pointed Rifle Beats 4-5) & MUZZLE FLASH
# ---------------------------------------------------------------------------
def _build_projectiles_and_flashes(
    root: bpy.types.Object,
    col: bpy.types.Collection,
    total_frames: int,
) -> dict[str, Any]:
    # A. 9mm Round-Nose Pistol Projectile (Beats 1, 2, 3)
    pistol_anchor = _create_empty("FX_Pistol9mm_Anchor", col, parent=root)
    mat_copper = _mat_copper_bullet(pistol_anchor)
    bm_p = bmesh.new()
    prof_9mm = [
        (-0.022, 0.0082),
        (-0.006, 0.0086),
        (0.004, 0.0084),
        (0.012, 0.0070),
        (0.018, 0.0042),
        (0.021, 0.0014),
    ]
    _add_revolved_profile(bm_p, prof_9mm, segs=32, cap_start=True, cap_end=True)
    pistol_obj = _finalize_bmesh("FX_Pistol9mm_Round", bm_p, mat_copper, col, parent=pistol_anchor, smooth=True, shadow=True)
    _set_visibility_windows(pistol_obj, [(1, 900)], total_frames)

    # Open-ended glowing slipstream trail behind 9mm (cap_ends=False)
    mat_trail, trail_gain = _mat_additive_glow("Mat_FX_Slipstream", (1.0, 0.62, 0.16), 0.36, pistol_anchor)
    bm_tr = bmesh.new()
    _add_revolved_profile(
        bm_tr,
        [(-0.24, 0.0015), (-0.11, 0.0065), (-0.018, 0.0092)],
        segs=20,
        cap_start=False,
        cap_end=False,
    )
    trail_obj = _finalize_bmesh("FX_Pistol9mm_Trail", bm_tr, mat_trail, col, parent=pistol_anchor, smooth=True)
    _set_visibility_windows(trail_obj, [(22, 78), (245, 340), (535, 622)], total_frames)

    cz = float(C.RANGE["chest_z"])
    for f in _sample_frames(1, 900, step=4, dense_ranges=((20, 105, 1), (244, 365, 2), (534, 745, 2))):
        ff = float(f)
        pistol_anchor.location = Vector((C.pistol_bullet_x(ff), 0.0, cz))
        # High-speed rifled spin around +X axis
        spin = ff * 0.55 if (22 <= f <= 78 or 245 <= f <= 340 or 535 <= f <= 660) else 0.0
        pistol_anchor.rotation_quaternion = Euler((spin, 0.0, 0.0), "XYZ").to_quaternion()
        pistol_anchor.keyframe_insert("location", frame=f)
        pistol_anchor.keyframe_insert("rotation_quaternion", frame=f)
    _set_action_interp(pistol_anchor.animation_data, "LINEAR")

    # B. High-Velocity Pointed Spitzer Rifle Round (Beat 4 ceramic shatter & Beat 5 Newton)
    rifle_anchor = _create_empty("FX_RifleSpitzer_Anchor", col, parent=root)
    bm_r = bmesh.new()
    prof_rifle = [
        (-0.028, 0.0062),
        (-0.018, 0.0076),
        (0.000, 0.0076),
        (0.012, 0.0054),
        (0.022, 0.0024),
        (0.028, 0.0004),
    ]
    _add_revolved_profile(bm_r, prof_rifle, segs=32, cap_start=True, cap_end=True)
    rifle_obj = _finalize_bmesh("FX_RifleSpitzer_Round", bm_r, mat_copper, col, parent=rifle_anchor, smooth=True, shadow=True)
    # Visible until shattered into fragments at f=1025 in Beat 4, then intact again in Beat 5 (1261..1560)
    _set_visibility_windows(rifle_obj, [(901, 1030), (1261, 1560)], total_frames)

    bm_rtr = bmesh.new()
    _add_revolved_profile(
        bm_rtr,
        [(-0.28, 0.0014), (-0.12, 0.0060), (-0.024, 0.0080)],
        segs=20,
        cap_start=False,
        cap_end=False,
    )
    rtrail_obj = _finalize_bmesh("FX_RifleSpitzer_Trail", bm_rtr, mat_trail, col, parent=rifle_anchor, smooth=True)
    _set_visibility_windows(rtrail_obj, [(925, 975), (1295, 1370)], total_frames)

    for f in _sample_frames(901, 1560, step=4, dense_ranges=((924, 1035, 2), (1294, 1395, 2))):
        ff = float(f)
        rifle_anchor.location = Vector((C.rifle_bullet_x(ff), 0.0, cz))
        spin = ff * 0.72
        rifle_anchor.rotation_quaternion = Euler((spin, 0.0, 0.0), "XYZ").to_quaternion()
        # Slightly flatten nose as it hits SiC ceramic (f=965..1030)
        sq = 1.0 - 0.55 * C.smoothstep(965.0, 1028.0, ff) if f <= 1035 else 1.0
        rifle_anchor.scale = Vector((sq, 1.0 + 0.35 * (1.0 - sq), 1.0 + 0.35 * (1.0 - sq)))
        rifle_anchor.keyframe_insert("location", frame=f)
        rifle_anchor.keyframe_insert("rotation_quaternion", frame=f)
        rifle_anchor.keyframe_insert("scale", frame=f)
    _set_action_interp(rifle_anchor.animation_data, "LINEAR")

    # C. Open-Ended Muzzle Flash Plumes (Beat 2 f=245..261 & Beat 5 f=1295..1311)
    mat_flash, _ = _mat_additive_glow("Mat_FX_MuzzleFlash", (1.0, 0.46, 0.06), 3.2)
    flashes: list[bpy.types.Object] = []
    for tag, mx, f_start in (
        ("B2", float(C.RANGE["muzzle_x"]), int(C.EV["b2_muzzle_fire"])),
        ("B5", float(C.RANGE["b5_muzzle_x"]), int(C.EV["b5_dual_fire"])),
    ):
        bm_f = bmesh.new()
        _add_revolved_profile(
            bm_f,
            [(0.002, 0.008), (0.025, 0.032), (0.068, 0.024), (0.115, 0.004)],
            segs=24,
            cap_start=False,
            cap_end=False,
            radial_mod_fn=lambda _x, r, ang, _ri, _s: r * (0.72 + 0.38 * (0.5 + 0.5 * math.cos(6.0 * ang))),
        )
        fl = _finalize_bmesh(f"FX_MuzzleFlash_{tag}", bm_f, mat_flash, col, parent=root, location=(mx, 0.0, cz))
        _set_visibility_windows(fl, [(f_start, f_start + 14)], total_frames)
        for df, sc in ((0, 0.65), (2, 1.15), (7, 0.90), (14, 0.25)):
            fl.scale = Vector((sc, sc, sc))
            fl.keyframe_insert("scale", frame=f_start + df)
        _set_action_interp(fl.animation_data, "LINEAR")
        flashes.append(fl)

    return {
        "pistol_anchor": pistol_anchor,
        "pistol": pistol_obj,
        "rifle_anchor": rifle_anchor,
        "rifle": rifle_obj,
        "flashes": flashes,
        "trail_gain": trail_gain,
    }


# ---------------------------------------------------------------------------
# 3. CURVED 3D TAPERED SPARK BURSTS + EMBER SPHERES (Beats 1, 2, 4)
# ---------------------------------------------------------------------------
def _build_impact_sparks(
    root: bpy.types.Object,
    col: bpy.types.Collection,
    total_frames: int,
) -> dict[str, Any]:
    """Curved 3D tapered spark tubes (cap_ends=False) + glowing ember spheres (AGENTS.md Sec 3)."""
    vx = float(C.RANGE["vest_surface_x"])
    cz = float(C.RANGE["chest_z"])
    spark_root = _create_empty("FX_SparkBurst_Root", col, parent=root, location=(vx - 0.004, 0.0, cz))

    mat_spark, spark_gain = _mat_additive_glow("Mat_FX_SparkGold", (1.0, 0.64, 0.12), 2.4, spark_root)
    mat_ember = _mat_solid_emissive("Mat_FX_EmberSphere", (1.0, 0.72, 0.18), 1.85)

    # 18 curved 3D tapered spark arcs bursting radially backward/outward off chest plate
    bm_sp = bmesh.new()
    for i in range(18):
        ang = (math.tau * i) / 18.0 + 0.14 * (i % 3)
        spread_l = 0.14 + 0.11 * ((i * 5) % 4) / 3.0
        back_x = -0.08 - 0.06 * ((i * 3) % 3) / 2.0
        pts: list[Vector] = []
        rads: list[float] = []
        for j in range(6):
            t = j / 5.0
            px = back_x * t
            py = math.cos(ang) * spread_l * t
            pz = math.sin(ang) * spread_l * t - 0.055 * (t * t)
            pts.append(Vector((px, py, pz)))
            rads.append(0.0030 * (1.0 - 0.85 * t))
        _add_swept_tube(bm_sp, pts, rads, segs=6, cap_ends=False)

    sparks_obj = _finalize_bmesh("FX_SparkBurst_Tubes", bm_sp, mat_spark, col, parent=spark_root, smooth=True)

    # 14 glowing 3D ember spheres
    bm_em = bmesh.new()
    for i in range(14):
        ang = (math.tau * i) / 14.0 + 0.31
        r_dist = 0.09 + 0.13 * ((i * 7) % 5) / 4.0
        loc = Vector((-0.04 - 0.05 * (i % 3), math.cos(ang) * r_dist, math.sin(ang) * r_dist - 0.03))
        bmesh.ops.create_icosphere(bm_em, subdivisions=1, radius=0.0038 + 0.0018 * (i % 2), matrix=Matrix.Translation(loc))
    embers_obj = _finalize_bmesh("FX_SparkBurst_Embers", bm_em, mat_ember, col, parent=spark_root, smooth=True)

    spark_wins = [(75, 145), (335, 385), (965, 1045)]
    _set_visibility_windows(sparks_obj, spark_wins, total_frames)
    _set_visibility_windows(embers_obj, spark_wins, total_frames)

    for f in _sample_frames(1, total_frames, step=15, dense_ranges=((74, 146, 2), (334, 386, 2), (964, 1046, 2))):
        t_sp = C.spark_burst_t(float(f))
        sc = 0.18 + 1.05 * (t_sp ** 0.55)
        spark_root.scale = Vector((sc, sc, sc))
        spark_root.keyframe_insert("scale", frame=f)
        spark_gain.outputs[0].default_value = 2.6 * t_sp
        spark_gain.outputs[0].keyframe_insert("default_value", frame=f)

    _set_action_interp(spark_root.animation_data, "LINEAR")
    _set_action_interp(mat_spark.node_tree.animation_data, "LINEAR")

    return {"spark_root": spark_root, "sparks": sparks_obj, "embers": embers_obj}


# ---------------------------------------------------------------------------
# 4. BEAT 5 SYMMETRICAL NEWTON'S THIRD LAW 3D VECTORS (F1 vs -F2)
# ---------------------------------------------------------------------------
def _build_newton_vectors(
    root: bpy.types.Object,
    col: bpy.types.Collection,
    total_frames: int,
) -> dict[str, Any]:
    """Equal-length glowing 3D opposing force arrows at Shooter (-X) and Target (+X) in Beat 5."""
    vlen = float(C.RANGE["b5_vector_len"])
    vz = float(C.RANGE["b5_vector_z"])

    mat_f1 = _mat_solid_emissive("Mat_FX_VectorF1_Crimson", (1.0, 0.28, 0.08), 1.80)
    mat_f2 = _mat_solid_emissive("Mat_FX_VectorF2_Cyan", (0.08, 0.88, 1.0), 1.80)

    def _make_arrow_mesh(name: str, sign_x: float, mat: bpy.types.Material, origin_x: float) -> bpy.types.Object:
        bm = bmesh.new()
        shaft_end = sign_x * (vlen * 0.72)
        tip_end = sign_x * vlen
        prof = [
            (0.0, 0.016),
            (shaft_end, 0.016),
            (shaft_end, 0.044),
            (tip_end, 0.002),
        ]
        if sign_x < 0.0:
            prof = list(reversed(prof))
        _add_revolved_profile(bm, prof, segs=24, cap_start=True, cap_end=True)
        obj = _finalize_bmesh(name, bm, mat, col, parent=root, smooth=True, location=(origin_x, -0.04, vz))
        _set_visibility_windows(obj, [(1295, 1558)], total_frames)
        return obj

    vec_f1 = _make_arrow_mesh("FX_Vector_F1_Shooter", -1.0, mat_f1, float(C.RANGE["b5_shooter_x"]) - 0.14)
    vec_f2 = _make_arrow_mesh("FX_Vector_F2_Target", +1.0, mat_f2, float(C.RANGE["b5_target_x"]) + 0.14)

    for f in _sample_frames(1295, 1558, step=6, dense_ranges=((1295, 1415, 2),)):
        u = C.newton_vector_t(float(f))
        sx = max(0.02, u)
        for v_obj in (vec_f1, vec_f2):
            v_obj.scale = Vector((sx, 0.75 + 0.25 * sx, 0.75 + 0.25 * sx))
            v_obj.keyframe_insert("scale", frame=f)
    _set_action_interp(vec_f1.animation_data, "LINEAR")
    _set_action_interp(vec_f2.animation_data, "LINEAR")
    return {"vec_f1": vec_f1, "vec_f2": vec_f2}


# ---------------------------------------------------------------------------
# 5. BEAT 6 SETTLING SPENT COPPER & CERAMIC DUST (1561..1800)
# ---------------------------------------------------------------------------
def _build_beat6_dust(
    root: bpy.types.Object,
    col: bpy.types.Collection,
    total_frames: int,
) -> list[bpy.types.Object]:
    vx = float(C.RANGE["vest_surface_x"])
    cz = float(C.RANGE["chest_z"])
    mat_cop = _mat_solid_emissive("Mat_FX_SpentCopperFlake", (0.55, 0.30, 0.14), 0.0, metallic=0.88, roughness=0.35)
    flakes: list[bpy.types.Object] = []

    for i in range(10):
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=0.0055 + 0.002 * (i % 3))
        bmesh.ops.scale(bm, vec=Vector((1.3, 0.9, 0.35)), verts=bm.verts)
        fl = _finalize_bmesh(f"FX_B6_SpentFlake_{i:02d}", bm, mat_cop, col, parent=root, smooth=False)
        _set_visibility_windows(fl, [(1561, total_frames)], total_frames)

        x0 = vx - 0.04 - 0.04 * (i % 4)
        y0 = -0.14 + 0.032 * i
        z0 = 0.22 - 0.03 * (i % 5)
        for f in range(1561, total_frames + 1, 12):
            u = C.smootherstep(1561.0, float(total_frames), float(f))
            fl.location = Vector((
                x0 - 0.03 * u + 0.015 * math.sin(u * 5.0 + i),
                y0 + 0.02 * math.cos(u * 4.0 + i),
                max(0.010, z0 - 0.20 * u),
            ))
            fl.rotation_quaternion = Euler((u * 4.0 + i, u * 5.5, u * 3.0), "XYZ").to_quaternion()
            fl.keyframe_insert("location", frame=f)
            fl.keyframe_insert("rotation_quaternion", frame=f)
        _set_action_interp(fl.animation_data, "LINEAR")
        flakes.append(fl)
    return flakes


# ---------------------------------------------------------------------------
# PUBLIC ENTRYPOINT
# ---------------------------------------------------------------------------
def build_vest_fx(
    col: bpy.types.Collection,
    total_frames: int = C.TOTAL_FRAMES,
) -> dict[str, Any]:
    """Assemble macro multi-layer armor cutaway, projectiles, sparks, Newton vectors & spent dust."""
    fx_root = _create_empty("VestFX_Root", col)
    macro_res = _build_macro_vest_cutaway(fx_root, col, total_frames)
    proj_res = _build_projectiles_and_flashes(fx_root, col, total_frames)
    spark_res = _build_impact_sparks(fx_root, col, total_frames)
    vec_res = _build_newton_vectors(fx_root, col, total_frames)
    dust_objs = _build_beat6_dust(fx_root, col, total_frames)

    return {
        "root": fx_root,
        "macro": macro_res,
        "projectiles": proj_res,
        "sparks": spark_res,
        "vectors": vec_res,
        "dust": dust_objs,
    }
