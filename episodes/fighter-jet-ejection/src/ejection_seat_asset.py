"""
ejection_seat_asset.py — Hero Ejection Seat, Guide Rails, Catapult & Rocket FX
Episode #1: "Why Ejecting From a Fighter Jet Shrinks You by 1 Inch" (4K 60fps, 1800 frames).

Owned by the Ejection Seat & Rocket Asset Artist.
ALL timings, coordinates, trajectories, and anchors are imported from `ejection_contract.py`.

Public API:
    build_ejection_seat(col: bpy.types.Collection, total_frames: int = 1800) -> dict

Returned dict structure (all values are real `bpy.types.Object` or `list[bpy.types.Object]`):
    {
        "root": bpy.types.Object,                 # `Seat_Root` Empty at SRP (0,0,0)
        "rails_root": bpy.types.Object,           # `Seat_Rails_Root` Empty (for parenting to jet_root)
        "handle": bpy.types.Object,               # `Seat_Handle` mesh (yellow-black double-loop handle)
        "garters": list[bpy.types.Object],        # `[Seat_Garter_L, Seat_Garter_R, Seat_Ankle_Cuff_L, Seat_Ankle_Cuff_R]`
        "catapult_fx": list[bpy.types.Object],    # `[Seat_Catapult_Exhaust, Seat_Catapult_Core, ..., Seat_Catapult_Light]`
        "rocket_fx": list[bpy.types.Object],      # `[Seat_Rocket_Plume_L/R, ..., Seat_Rocket_Light]`
        "all_objs": list[bpy.types.Object],       # Every `Seat_*` object created and linked into `col`
    }
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
for _p in (str(SCRIPT_DIR), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ejection_contract as C  # noqa: E402

# ---------------------------------------------------------------------------
# GEOMETRY / KINEMATICS CONSTANTS (Derived directly from ejection_contract)
# ---------------------------------------------------------------------------
_TILT_RAD = math.radians(C.RAIL_TILT_DEG)
_TAN_TILT = math.tan(_TILT_RAD)
_RAIL_VEC = Vector(C.RAIL_AXIS).normalized()
_RAIL_QUAT = _RAIL_VEC.to_track_quat("Z", "Y")
_RAIL_EULER = _RAIL_QUAT.to_euler("XYZ")

_NOZZLE_PITCH_RAD = math.radians(10.0)  # angled slightly forward (+Y) & downward (-Z)
_NOZZLE_DIR = Vector((0.0, math.sin(_NOZZLE_PITCH_RAD), -math.cos(_NOZZLE_PITCH_RAD))).normalized()
_NOZZLE_QUAT = _NOZZLE_DIR.to_track_quat("Z", "Y")

_HANDLE_ORIFICE_LOCAL = Vector((0.0, 0.315, -0.005))
_HANDLE_CLEVIS_OFFSET = Vector((0.0, 0.005, -0.052))

_RAIL_LOWER_LEN = 1.20  # Fits cleanly inside closed canopy during Beat 1-2
_RAIL_UPPER_LEN = 0.95  # Telescopes +0.80m during catapult firing to full C.RAIL_LENGTH_M (2.0m)


# ---------------------------------------------------------------------------
# MATH & BLENDER DATA-BLOCK HELPERS (Zero bpy.ops scene pollution)
# ---------------------------------------------------------------------------
def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _catmull_rom(pts: list[Vector], samples_per_seg: int = 4) -> list[Vector]:
    if len(pts) < 2:
        return list(pts)
    ext = [pts[0] + (pts[0] - pts[1])] + list(pts) + [pts[-1] + (pts[-1] - pts[-2])]
    out: list[Vector] = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for s in range(samples_per_seg):
            t = s / samples_per_seg
            t2 = t * t
            t3 = t2 * t
            out.append(
                0.5
                * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
            )
    out.append(pts[-1])
    return out


def _link_obj(obj: bpy.types.Object, col: bpy.types.Collection) -> bpy.types.Object:
    if obj.name not in col.objects:
        col.objects.link(obj)
    return obj


def _new_empty(
    name: str,
    col: bpy.types.Collection,
    parent: bpy.types.Object | None = None,
    location: tuple[float, float, float] | Vector = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float] | Euler = (0.0, 0.0, 0.0),
    display_size: float = 0.12,
) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = "PLAIN_AXES"
    obj.empty_display_size = display_size
    obj.location = Vector(location)
    obj.rotation_euler = Euler(rotation, "XYZ") if not isinstance(rotation, Euler) else rotation.copy()
    if parent is not None:
        obj.parent = parent
    return _link_obj(obj, col)


def _create_mesh_obj(
    name: str,
    bm: bmesh.types.BMesh,
    materials: list[bpy.types.Material] | bpy.types.Material,
    col: bpy.types.Collection,
    parent: bpy.types.Object | None = None,
    location: tuple[float, float, float] | Vector = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float] | Euler = (0.0, 0.0, 0.0),
    smooth_all: bool | None = True,
) -> bpy.types.Object:
    """Attach material slots BEFORE bm.to_mesh(mesh) per AGENTS.md Section 3."""
    mesh = bpy.data.meshes.new(name)
    mat_list = materials if isinstance(materials, list) else [materials]
    for mat in mat_list:
        mesh.materials.append(mat)

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()

    if smooth_all is not None and len(mesh.polygons) > 0:
        mesh.polygons.foreach_set("use_smooth", [smooth_all] * len(mesh.polygons))
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(location)
    obj.rotation_euler = Euler(rotation, "XYZ") if not isinstance(rotation, Euler) else rotation.copy()
    if parent is not None:
        obj.parent = parent
    return _link_obj(obj, col)


def _merge_bm(dst: bmesh.types.BMesh, src: bmesh.types.BMesh, mat_index: int | None = None) -> None:
    if mat_index is not None:
        for f in src.faces:
            f.material_index = mat_index
    tmp = bpy.data.meshes.new("_seat_tmp_merge")
    src.to_mesh(tmp)
    src.free()
    dst.from_mesh(tmp)
    bpy.data.meshes.remove(tmp)


def _set_interp(id_block: Any, interp: str, data_path_prefix: str = "") -> None:
    ad = getattr(id_block, "animation_data", None)
    if not ad or not ad.action:
        return
    for fc in ad.action.fcurves:
        if data_path_prefix and not fc.data_path.startswith(data_path_prefix):
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = interp


def _key_visibility(
    obj: bpy.types.Object,
    visible_ranges: list[tuple[int, int]],
    total_frames: int,
) -> None:
    """Keyframe hide_viewport & hide_render with CONSTANT interpolation."""
    if not visible_ranges:
        for f in (1, total_frames):
            obj.hide_viewport = True
            obj.hide_render = True
            obj.keyframe_insert("hide_viewport", frame=f)
            obj.keyframe_insert("hide_render", frame=f)
        _set_interp(obj, "CONSTANT", "hide_")
        return

    frames = {1, total_frames}
    for s, e in visible_ranges:
        sc = max(1, min(total_frames, int(s)))
        ec = max(1, min(total_frames, int(e)))
        if sc > 1:
            frames.add(sc - 1)
        frames.add(sc)
        frames.add(ec)
        if ec < total_frames:
            frames.add(ec + 1)

    for f in sorted(frames):
        vis = any(s <= f <= e for s, e in visible_ranges)
        obj.hide_viewport = not vis
        obj.hide_render = not vis
        obj.keyframe_insert("hide_viewport", frame=f)
        obj.keyframe_insert("hide_render", frame=f)
    _set_interp(obj, "CONSTANT", "hide_")


def _sample_frames(
    total_frames: int,
    dense_windows: list[tuple[int, int]] | None = None,
    step: int = 2,
) -> list[int]:
    frames = set(range(1, total_frames + 1, step))
    frames.add(1)
    frames.add(total_frames)
    for ev_f in C.EV.values():
        if 1 <= ev_f <= total_frames:
            frames.add(ev_f)
    if dense_windows:
        for s, e in dense_windows:
            for f in range(max(1, s), min(total_frames, e) + 1):
                frames.add(f)
    return sorted(frames)


# ---------------------------------------------------------------------------
# PROCEDURAL BMESH PRIMITIVE BUILDERS
# ---------------------------------------------------------------------------
def _make_beveled_box_bm(
    dims: tuple[float, float, float],
    bevel_w: float,
    segments: int = 2,
) -> bmesh.types.BMesh:
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Diagonal((dims[0], dims[1], dims[2], 1.0)))
    if bevel_w > 0.0:
        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bevel_w, segments=segments, affect="EDGES")
    return bm


def _make_profile_prism_bm(
    profile_yz: list[tuple[float, float]],
    x_min: float,
    x_max: float,
    mat_index: int = 0,
    bevel_w: float = 0.0,
) -> bmesh.types.BMesh:
    """Extrude a 2D (y, z) polygon across X ∈ [x_min, x_max]."""
    bm = bmesh.new()
    v_l = [bm.verts.new((x_min, y, z)) for y, z in profile_yz]
    v_r = [bm.verts.new((x_max, y, z)) for y, z in profile_yz]
    n = len(profile_yz)
    for i in range(n):
        inext = (i + 1) % n
        f = bm.faces.new((v_l[i], v_l[inext], v_r[inext], v_r[i]))
        f.material_index = mat_index
        f.smooth = False
    fl = bm.faces.new(v_l[::-1])
    fl.material_index = mat_index
    fl.smooth = False
    fr = bm.faces.new(v_r)
    fr.material_index = mat_index
    fr.smooth = False
    if bevel_w > 0.0:
        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bevel_w, segments=2, affect="EDGES")
    return bm


def _make_tube_along_path_bm(
    points: list[Vector],
    radius: float,
    n_seg: int = 14,
    cap_ends: bool = True,
    mat_index: int = 0,
) -> bmesh.types.BMesh:
    bm = bmesh.new()
    rings: list[list[bmesh.types.BMVert]] = []
    up_ref = Vector((0.0, 1.0, 0.0))

    for idx, pt in enumerate(points):
        if idx == 0:
            tang = (points[1] - points[0]).normalized()
        elif idx == len(points) - 1:
            tang = (points[-1] - points[-2]).normalized()
        else:
            t1 = (points[idx] - points[idx - 1]).normalized()
            t2 = (points[idx + 1] - points[idx]).normalized()
            tang = (t1 + t2)
            tang = tang.normalized() if tang.length > 1e-6 else t1

        ref = up_ref if abs(tang.dot(up_ref)) < 0.90 else Vector((0.0, 0.0, 1.0))
        b1 = tang.cross(ref).normalized()
        b2 = tang.cross(b1).normalized()
        ring = []
        for k in range(n_seg):
            a = k * 2.0 * math.pi / n_seg
            ring.append(bm.verts.new(pt + (b1 * math.cos(a) + b2 * math.sin(a)) * radius))
        rings.append(ring)

    for i in range(len(rings) - 1):
        for k in range(n_seg):
            kn = (k + 1) % n_seg
            f = bm.faces.new((rings[i][k], rings[i][kn], rings[i + 1][kn], rings[i + 1][k]))
            f.material_index = mat_index
            f.smooth = True

    if cap_ends and len(rings) >= 2:
        f0 = bm.faces.new(rings[0][::-1])
        f0.material_index = mat_index
        f1 = bm.faces.new(rings[-1])
        f1.material_index = mat_index
    return bm


def _make_c_channel_bm(
    length: float,
    width: float,
    depth: float,
    thick: float,
    open_side: str = "+X",
) -> bmesh.types.BMesh:
    """Extrude a crisp structural aerospace C-channel beam along local +Z from z=0 to z=length."""
    hw = 0.5 * width
    hd = 0.5 * depth
    if open_side == "+X":
        poly = [
            (-hw, -hd), (+hw, -hd), (+hw, -hd + thick),
            (-hw + thick, -hd + thick), (-hw + thick, +hd - thick),
            (+hw, +hd - thick), (+hw, +hd), (-hw, +hd),
        ]
    else:
        poly = [
            (-hw, -hd), (+hw, -hd), (+hw, +hd), (-hw, +hd),
            (-hw, +hd - thick), (+hw - thick, +hd - thick),
            (+hw - thick, -hd + thick), (-hw, -hd + thick),
        ]
    bm = bmesh.new()
    bot = [bm.verts.new((x, y, 0.0)) for x, y in poly]
    top = [bm.verts.new((x, y, length)) for x, y in poly]
    n = len(poly)
    for i in range(n):
        inext = (i + 1) % n
        bm.faces.new((bot[i], bot[inext], top[inext], top[i])).smooth = False
    bm.faces.new(bot[::-1]).smooth = False
    bm.faces.new(top).smooth = False
    bmesh.ops.bevel(bm, geom=list(bm.edges), offset=0.0014, segments=1, affect="EDGES")
    return bm


def _make_delaval_nozzle_bm(mat_throat: int = 0, mat_bell: int = 0) -> bmesh.types.BMesh:
    """Lathe a crisp De Laval rocket bell nozzle pointing along +Z (origin at nozzle exit)."""
    # Profile (z, radius): mounting collar at z=-0.055 -> throat z=-0.036 -> bell exit z=0.0
    prof = [
        (-0.058, 0.034),
        (-0.044, 0.036),
        (-0.036, 0.022),
        (-0.024, 0.030),
        (-0.010, 0.042),
        (0.000, 0.048),
        (0.000, 0.042),
        (-0.012, 0.036),
        (-0.034, 0.018),
    ]
    n_seg = 28
    bm = bmesh.new()
    rings: list[list[bmesh.types.BMVert]] = []
    for z, r in prof:
        ring = []
        for k in range(n_seg):
            a = k * 2.0 * math.pi / n_seg
            ring.append(bm.verts.new((r * math.cos(a), r * math.sin(a), z)))
        rings.append(ring)

    for i in range(len(rings) - 1):
        for k in range(n_seg):
            kn = (k + 1) % n_seg
            f = bm.faces.new((rings[i][k], rings[i][kn], rings[i + 1][kn], rings[i + 1][k]))
            f.material_index = mat_bell if i >= 2 else mat_throat
            f.smooth = True
    bm.faces.new(rings[0][::-1]).smooth = False
    bm.faces.new(rings[-1]).smooth = False
    return bm


# ---------------------------------------------------------------------------
# LOCAL-SPACE MATERIALS LIBRARY (Blender 4.2 Principled BSDF & AddShader FX)
# ---------------------------------------------------------------------------
def _wire_camera_front_emission(
    nt: bpy.types.NodeTree,
    em_strength_node: bpy.types.NodeSocket,
    em_node: bpy.types.ShaderNode,
) -> None:
    """Gate additive emission by (CameraRay OR TransmissionRay) * (1 - Backfacing)."""
    lp = nt.nodes.new("ShaderNodeLightPath")
    geom = nt.nodes.new("ShaderNodeNewGeometry")

    ray_ok = nt.nodes.new("ShaderNodeMath")
    ray_ok.operation = "MAXIMUM"
    nt.links.new(lp.outputs["Is Camera Ray"], ray_ok.inputs[0])
    nt.links.new(lp.outputs["Is Transmission Ray"], ray_ok.inputs[1])

    inv_bf = nt.nodes.new("ShaderNodeMath")
    inv_bf.operation = "SUBTRACT"
    inv_bf.inputs[0].default_value = 1.0
    nt.links.new(geom.outputs["Backfacing"], inv_bf.inputs[1])

    gate = nt.nodes.new("ShaderNodeMath")
    gate.operation = "MULTIPLY"
    nt.links.new(ray_ok.outputs["Value"], gate.inputs[0])
    nt.links.new(inv_bf.outputs["Value"], gate.inputs[1])

    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    nt.links.new(em_strength_node, mul.inputs[0])
    nt.links.new(gate.outputs["Value"], mul.inputs[1])
    nt.links.new(mul.outputs["Value"], em_node.inputs["Strength"])


def _build_materials(
    seat_root: bpy.types.Object,
    rails_root: bpy.types.Object,
) -> tuple[dict[str, bpy.types.Material], dict[str, Any]]:
    mats: dict[str, bpy.types.Material] = {}
    ctrl: dict[str, Any] = {}

    # 1. Tactical Anodized Aluminum / Titanium Seat Frame (local to Seat_Root)
    m_frame = bpy.data.materials.new("Seat_Mat_Frame_Gunmetal")
    m_frame.use_nodes = True
    nt = m_frame.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Metallic"].default_value = 0.86
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = seat_root
    n_macro = nt.nodes.new("ShaderNodeTexNoise")
    n_macro.inputs["Scale"].default_value = 18.0
    n_macro.inputs["Detail"].default_value = 6.0
    n_fine = nt.nodes.new("ShaderNodeTexNoise")
    n_fine.inputs["Scale"].default_value = 180.0
    n_fine.inputs["Detail"].default_value = 8.0
    nt.links.new(tc.outputs["Object"], n_macro.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], n_fine.inputs["Vector"])

    col_ramp = nt.nodes.new("ShaderNodeValToRGB")
    col_ramp.color_ramp.elements[0].position = 0.25
    col_ramp.color_ramp.elements[0].color = (0.055, 0.062, 0.074, 1.0)
    col_ramp.color_ramp.elements[1].position = 0.78
    col_ramp.color_ramp.elements[1].color = (0.095, 0.106, 0.124, 1.0)
    nt.links.new(n_macro.outputs["Fac"], col_ramp.inputs["Fac"])
    nt.links.new(col_ramp.outputs["Color"], bsdf.inputs["Base Color"])

    rough_map = nt.nodes.new("ShaderNodeMapRange")
    rough_map.clamp = True
    rough_map.inputs["To Min"].default_value = 0.30
    rough_map.inputs["To Max"].default_value = 0.46
    nt.links.new(n_macro.outputs["Fac"], rough_map.inputs["Value"])
    nt.links.new(rough_map.outputs["Result"], bsdf.inputs["Roughness"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.0005
    nt.links.new(n_fine.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    mats["frame"] = m_frame

    # 2. Olive-Charcoal Tactical Ballistic-Nylon Cushion
    m_cush = bpy.data.materials.new("Seat_Mat_Cushion_Olive")
    m_cush.use_nodes = True
    nt = m_cush.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.80
    bsdf.inputs["Sheen Weight"].default_value = 0.24
    bsdf.inputs["Sheen Roughness"].default_value = 0.55
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = seat_root
    n_w = nt.nodes.new("ShaderNodeTexNoise")
    n_w.inputs["Scale"].default_value = 340.0
    n_w.inputs["Detail"].default_value = 10.0
    n_p = nt.nodes.new("ShaderNodeTexNoise")
    n_p.inputs["Scale"].default_value = 14.0
    n_p.inputs["Detail"].default_value = 4.0
    nt.links.new(tc.outputs["Object"], n_w.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], n_p.inputs["Vector"])

    c_ramp = nt.nodes.new("ShaderNodeValToRGB")
    c_ramp.color_ramp.elements[0].position = 0.24
    c_ramp.color_ramp.elements[0].color = (0.042, 0.052, 0.038, 1.0)
    c_ramp.color_ramp.elements[1].position = 0.80
    c_ramp.color_ramp.elements[1].color = (0.070, 0.084, 0.062, 1.0)
    nt.links.new(n_p.outputs["Fac"], c_ramp.inputs["Fac"])
    nt.links.new(c_ramp.outputs["Color"], bsdf.inputs["Base Color"])

    b_c = nt.nodes.new("ShaderNodeBump")
    b_c.inputs["Strength"].default_value = 0.24
    b_c.inputs["Distance"].default_value = 0.0008
    nt.links.new(n_w.outputs["Fac"], b_c.inputs["Height"])
    nt.links.new(b_c.outputs["Normal"], bsdf.inputs["Normal"])
    mats["cushion"] = m_cush

    # 3. Chrome Steel (Catapult piston, rail sliding shoes, pitot tips, cables)
    m_chrome = bpy.data.materials.new("Seat_Mat_Chrome_Steel")
    m_chrome.use_nodes = True
    bsdf = m_chrome.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.76, 0.80, 0.86, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.98
    bsdf.inputs["Roughness"].default_value = 0.16
    mats["chrome"] = m_chrome

    # 4. Stationary Cockpit Rails & Catapult Breech (local to Seat_Rails_Root)
    m_rail = bpy.data.materials.new("Seat_Mat_Rail_Dark")
    m_rail.use_nodes = True
    nt = m_rail.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.11, 0.125, 0.148, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.91
    bsdf.inputs["Roughness"].default_value = 0.30
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = rails_root
    nf = nt.nodes.new("ShaderNodeTexNoise")
    nf.inputs["Scale"].default_value = 140.0
    nf.inputs["Detail"].default_value = 6.0
    bm_n = nt.nodes.new("ShaderNodeBump")
    bm_n.inputs["Strength"].default_value = 0.10
    bm_n.inputs["Distance"].default_value = 0.0004
    nt.links.new(tc.outputs["Object"], nf.inputs["Vector"])
    nt.links.new(nf.outputs["Fac"], bm_n.inputs["Height"])
    nt.links.new(bm_n.outputs["Normal"], bsdf.inputs["Normal"])
    mats["rail"] = m_rail

    # 5. Yellow-Black Hazard Braided Ejection Handle (self-local stripe pattern!)
    m_handle = bpy.data.materials.new("Seat_Mat_Hazard_Handle")
    m_handle.use_nodes = True
    nt = m_handle.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Metallic"].default_value = 0.04
    bsdf.inputs["Roughness"].default_value = 0.44
    tc = nt.nodes.new("ShaderNodeTexCoord")
    # Use Generated/Object on handle itself so stripes stay glued to handle during yank!
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])

    # Diagonal striping along |X| + Z so twin loops show bold alternating yellow/black rings
    abs_x = nt.nodes.new("ShaderNodeMath")
    abs_x.operation = "ABSOLUTE"
    nt.links.new(sep.outputs["X"], abs_x.inputs[0])

    diag = nt.nodes.new("ShaderNodeMath")
    diag.operation = "ADD"
    nt.links.new(abs_x.outputs["Value"], diag.inputs[0])
    nt.links.new(sep.outputs["Z"], diag.inputs[1])

    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "X"
    wave.wave_profile = "SAW"
    wave.inputs["Scale"].default_value = 14.0
    wave.inputs["Distortion"].default_value = 0.0
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(diag.outputs["Value"], comb.inputs["X"])
    nt.links.new(comb.outputs["Vector"], wave.inputs["Vector"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.015, 0.016, 0.020, 1.0)
    ramp.color_ramp.elements[1].position = 0.46
    ramp.color_ramp.elements[1].color = (0.96, 0.68, 0.015, 1.0)
    nt.links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 0.10

    n_braid = nt.nodes.new("ShaderNodeTexNoise")
    n_braid.inputs["Scale"].default_value = 320.0
    n_braid.inputs["Detail"].default_value = 8.0
    nt.links.new(tc.outputs["Object"], n_braid.inputs["Vector"])
    b_h = nt.nodes.new("ShaderNodeBump")
    b_h.inputs["Strength"].default_value = 0.24
    b_h.inputs["Distance"].default_value = 0.0005
    nt.links.new(n_braid.outputs["Fac"], b_h.inputs["Height"])
    nt.links.new(b_h.outputs["Normal"], bsdf.inputs["Normal"])
    mats["hazard_handle"] = m_handle

    # 6. Headbox Danger Triangle Badges (Yellow border + Red/Black danger center)
    m_tri_yellow = bpy.data.materials.new("Seat_Mat_Danger_Yellow")
    m_tri_yellow.use_nodes = True
    b_y = m_tri_yellow.node_tree.nodes["Principled BSDF"]
    b_y.inputs["Base Color"].default_value = (0.95, 0.68, 0.02, 1.0)
    b_y.inputs["Roughness"].default_value = 0.36
    b_y.inputs["Emission Color"].default_value = (0.95, 0.68, 0.02, 1.0)
    b_y.inputs["Emission Strength"].default_value = 0.08
    mats["danger_yellow"] = m_tri_yellow

    m_tri_red = bpy.data.materials.new("Seat_Mat_Danger_Red")
    m_tri_red.use_nodes = True
    b_r = m_tri_red.node_tree.nodes["Principled BSDF"]
    b_r.inputs["Base Color"].default_value = (0.78, 0.055, 0.028, 1.0)
    b_r.inputs["Roughness"].default_value = 0.36
    b_r.inputs["Emission Color"].default_value = (0.78, 0.055, 0.028, 1.0)
    b_r.inputs["Emission Strength"].default_value = 0.08
    mats["danger_red"] = m_tri_red

    # 7. Leg-Restraint Garters + Ankle Cuffs (with animated Beat 3 amber/cyan highlight!)
    m_garter = bpy.data.materials.new("Seat_Mat_Garter_Strap")
    m_garter.use_nodes = True
    bsdf = m_garter.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.14, 0.16, 0.18, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.58
    bsdf.inputs["Emission Color"].default_value = (1.0, 0.58, 0.06, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 0.0
    ctrl["garter_em_socket"] = bsdf.inputs["Emission Strength"]
    mats["garter_strap"] = m_garter

    m_cuff = bpy.data.materials.new("Seat_Mat_Ankle_Cuff")
    m_cuff.use_nodes = True
    bsdf = m_cuff.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.09, 0.11, 0.14, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.25
    bsdf.inputs["Roughness"].default_value = 0.48
    bsdf.inputs["Emission Color"].default_value = (0.12, 0.84, 1.0, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 0.0
    ctrl["cuff_em_socket"] = bsdf.inputs["Emission Strength"]
    mats["ankle_cuff"] = m_cuff

    # 8. Heat-tinted Titanium Rocket Nozzles
    m_noz = bpy.data.materials.new("Seat_Mat_Nozzle_Titanium")
    m_noz.use_nodes = True
    bsdf = m_noz.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.18, 0.16, 0.15, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.96
    bsdf.inputs["Roughness"].default_value = 0.24
    mats["nozzle"] = m_noz

    # 9. LOCAL-SPACE Catapult Exhaust Plume (Cyan-White Core -> Hot Gold Edge, bound to Seat_Root!)
    def make_axial_plume_mat(
        name: str,
        z_hot: float,
        z_tip: float,
        mach_freq: float,
        noise_scale: float,
        falloff_power: float,
        base_strength: float,
        c_edge: tuple[float, float, float],
        c_mid: tuple[float, float, float],
        c_hot: tuple[float, float, float],
        gain_key: str,
    ) -> bpy.types.Material:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        em = nt.nodes.new("ShaderNodeEmission")

        # MANDATORY: TexCoord.Object bound to Seat_Root (100% local space!)
        tc = nt.nodes.new("ShaderNodeTexCoord")
        tc.object = seat_root
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])

        # Axial fade from nozzle exit (1.0) down to plume tip (0.0)
        axial = nt.nodes.new("ShaderNodeMapRange")
        axial.clamp = True
        axial.inputs["From Min"].default_value = z_tip
        axial.inputs["From Max"].default_value = z_hot
        axial.inputs["To Min"].default_value = 0.0
        axial.inputs["To Max"].default_value = 1.0
        nt.links.new(sep.outputs["Z"], axial.inputs["Value"])

        # Radial Fresnel soft falloff: (1 - Facing)^p -> zero hard geometry outline
        lw = nt.nodes.new("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = 0.50
        inv_f = nt.nodes.new("ShaderNodeMath")
        inv_f.operation = "SUBTRACT"
        inv_f.inputs[0].default_value = 1.0
        nt.links.new(lw.outputs["Facing"], inv_f.inputs[1])
        soft_edge = nt.nodes.new("ShaderNodeMath")
        soft_edge.operation = "POWER"
        soft_edge.inputs[1].default_value = falloff_power
        nt.links.new(inv_f.outputs["Value"], soft_edge.inputs[0])

        # Animated flicker phase offset in local-space turbulence
        phase = nt.nodes.new("ShaderNodeValue")
        phase.name = "FlickerPhase"
        phase.outputs[0].default_value = 0.0
        ctrl.setdefault(f"{gain_key}_phase_nodes", []).append(phase)

        vec_off = nt.nodes.new("ShaderNodeCombineXYZ")
        nt.links.new(phase.outputs[0], vec_off.inputs["Z"])
        v_add = nt.nodes.new("ShaderNodeVectorMath")
        v_add.operation = "ADD"
        nt.links.new(tc.outputs["Object"], v_add.inputs[0])
        nt.links.new(vec_off.outputs["Vector"], v_add.inputs[1])

        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = noise_scale
        noise.inputs["Detail"].default_value = 8.0
        noise.inputs["Roughness"].default_value = 0.64
        noise.inputs["Distortion"].default_value = 1.45
        nt.links.new(v_add.outputs["Vector"], noise.inputs["Vector"])

        n_map = nt.nodes.new("ShaderNodeMapRange")
        n_map.clamp = True
        n_map.inputs["From Min"].default_value = 0.25
        n_map.inputs["From Max"].default_value = 0.78
        n_map.inputs["To Min"].default_value = 0.28
        n_map.inputs["To Max"].default_value = 1.0
        nt.links.new(noise.outputs["Fac"], n_map.inputs["Value"])

        # Mach-diamond standing shock pattern along local Z
        z_mul = nt.nodes.new("ShaderNodeMath")
        z_mul.operation = "MULTIPLY"
        z_mul.inputs[1].default_value = mach_freq
        nt.links.new(sep.outputs["Z"], z_mul.inputs[0])
        z_sin = nt.nodes.new("ShaderNodeMath")
        z_sin.operation = "SINE"
        nt.links.new(z_mul.outputs["Value"], z_sin.inputs[0])
        mach_map = nt.nodes.new("ShaderNodeMapRange")
        mach_map.clamp = True
        mach_map.inputs["From Min"].default_value = -1.0
        mach_map.inputs["From Max"].default_value = 1.0
        mach_map.inputs["To Min"].default_value = 0.42
        mach_map.inputs["To Max"].default_value = 1.0
        nt.links.new(z_sin.outputs["Value"], mach_map.inputs["Value"])

        d1 = nt.nodes.new("ShaderNodeMath")
        d1.operation = "MULTIPLY"
        nt.links.new(axial.outputs["Result"], d1.inputs[0])
        nt.links.new(soft_edge.outputs["Value"], d1.inputs[1])

        d2 = nt.nodes.new("ShaderNodeMath")
        d2.operation = "MULTIPLY"
        nt.links.new(n_map.outputs["Result"], d2.inputs[0])
        nt.links.new(mach_map.outputs["Result"], d2.inputs[1])

        density = nt.nodes.new("ShaderNodeMath")
        density.operation = "MULTIPLY"
        nt.links.new(d1.outputs["Value"], density.inputs[0])
        nt.links.new(d2.outputs["Value"], density.inputs[1])

        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.08
        ramp.color_ramp.elements[0].color = (*c_edge, 1.0)
        e_m = ramp.color_ramp.elements.new(0.45)
        e_m.color = (*c_mid, 1.0)
        ramp.color_ramp.elements[2].position = 0.85
        ramp.color_ramp.elements[2].color = (*c_hot, 1.0)
        nt.links.new(density.outputs["Value"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], em.inputs["Color"])

        gain = nt.nodes.new("ShaderNodeValue")
        gain.name = "PlumeGain"
        gain.outputs[0].default_value = 1.0
        ctrl.setdefault(f"{gain_key}_gain_nodes", []).append(gain)

        s1 = nt.nodes.new("ShaderNodeMath")
        s1.operation = "MULTIPLY"
        s1.inputs[1].default_value = base_strength
        nt.links.new(density.outputs["Value"], s1.inputs[0])

        s2 = nt.nodes.new("ShaderNodeMath")
        s2.operation = "MULTIPLY"
        nt.links.new(s1.outputs["Value"], s2.inputs[0])
        nt.links.new(gain.outputs[0], s2.inputs[1])

        _wire_camera_front_emission(nt, s2.outputs["Value"], em)

        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(tr.outputs["BSDF"], add.inputs[0])
        nt.links.new(em.outputs["Emission"], add.inputs[1])
        nt.links.new(add.outputs["Shader"], out.inputs["Surface"])
        return mat

    mats["catapult_core"] = make_axial_plume_mat(
        "Seat_Mat_Catapult_Core",
        z_hot=-0.25, z_tip=-0.86, mach_freq=18.0, noise_scale=11.0,
        falloff_power=1.9, base_strength=4.8,
        c_edge=(1.0, 0.56, 0.08), c_mid=(0.35, 0.90, 1.0), c_hot=(0.94, 0.99, 1.0),
        gain_key="catapult",
    )
    mats["catapult_plume"] = make_axial_plume_mat(
        "Seat_Mat_Catapult_Plume",
        z_hot=-0.25, z_tip=-1.10, mach_freq=14.0, noise_scale=8.5,
        falloff_power=2.6, base_strength=3.4,
        c_edge=(0.95, 0.34, 0.03), c_mid=(1.0, 0.72, 0.22), c_hot=(0.55, 0.94, 1.0),
        gain_key="catapult",
    )

    mats["rocket_core"] = make_axial_plume_mat(
        "Seat_Mat_Rocket_Core",
        z_hot=-0.24, z_tip=-1.28, mach_freq=26.0, noise_scale=10.5,
        falloff_power=2.0, base_strength=5.2,
        c_edge=(1.0, 0.42, 0.03), c_mid=(1.0, 0.78, 0.24), c_hot=(1.0, 0.97, 0.85),
        gain_key="rocket",
    )
    mats["rocket_mach"] = make_axial_plume_mat(
        "Seat_Mat_Rocket_Mach",
        z_hot=-0.24, z_tip=-1.72, mach_freq=32.0, noise_scale=14.0,
        falloff_power=2.2, base_strength=4.2,
        c_edge=(0.92, 0.20, 0.01), c_mid=(1.0, 0.62, 0.14), c_hot=(1.0, 0.96, 0.78),
        gain_key="rocket",
    )
    mats["rocket_plume"] = make_axial_plume_mat(
        "Seat_Mat_Rocket_Plume",
        z_hot=-0.24, z_tip=-2.25, mach_freq=20.0, noise_scale=7.0,
        falloff_power=2.9, base_strength=2.9,
        c_edge=(0.78, 0.12, 0.005), c_mid=(1.0, 0.44, 0.04), c_hot=(1.0, 0.84, 0.35),
        gain_key="rocket",
    )

    # 10. Additive Spark Needles (Catapult cyan-gold / Rocket solar-gold)
    def make_spark_mat(name: str, col_rgb: tuple[float, float, float], strength: float) -> bpy.types.Material:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*col_rgb, 1.0)
        s_val = nt.nodes.new("ShaderNodeValue")
        s_val.outputs[0].default_value = strength
        _wire_camera_front_emission(nt, s_val.outputs[0], em)
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(tr.outputs["BSDF"], add.inputs[0])
        nt.links.new(em.outputs["Emission"], add.inputs[1])
        nt.links.new(add.outputs["Shader"], out.inputs["Surface"])
        return mat

    mats["catapult_spark"] = make_spark_mat("Seat_Mat_Catapult_Spark", (0.55, 0.92, 1.0), 5.6)
    mats["rocket_spark"] = make_spark_mat("Seat_Mat_Rocket_Spark", (1.0, 0.68, 0.18), 6.2)

    return mats, ctrl


# ---------------------------------------------------------------------------
# GEOMETRY BUILDERS: STATIONARY GUIDE RAILS & CATAPULT BREECH
# ---------------------------------------------------------------------------
def _build_guide_rails(
    rails_root: bpy.types.Object,
    mats: dict[str, bpy.types.Material],
    col: bpy.types.Collection,
    total_frames: int,
) -> list[bpy.types.Object]:
    """
    Build twin angled guide rails at C.RAIL_X = (-0.20, +0.20) starting at C.RAIL_BASE
    extending C.RAIL_LENGTH_M = 2.0 along C.RAIL_AXIS (1.20m fixed cockpit rails +
    nested 0.95m telescoping upper rails extending cleanly during catapult stroke so
    Beat 1-2 closed canopy has zero clipping), plus cross-braces & catapult breech.
    """
    objs: list[bpy.types.Object] = []
    base = Vector(C.RAIL_BASE)

    # 1. Fixed lower guide rails (0.0 .. 1.20m) — inside closed canopy!
    for side, rx in (("L", C.RAIL_X[0]), ("R", C.RAIL_X[1])):
        open_dir = "+X" if rx < 0.0 else "-X"
        bm = _make_c_channel_bm(length=_RAIL_LOWER_LEN, width=0.052, depth=0.078, thick=0.010, open_side=open_dir)
        # Base floor mounting flange
        foot = _make_beveled_box_bm((0.085, 0.115, 0.022), 0.003, segments=1)
        bmesh.ops.translate(foot, verts=foot.verts, vec=(0.0, 0.0, 0.011))
        _merge_bm(bm, foot)
        rail_obj = _create_mesh_obj(
            f"Seat_Rail_{side}", bm, mats["rail"], col,
            parent=rails_root,
            location=base + Vector((rx, 0.0, 0.0)),
            rotation=_RAIL_EULER,
            smooth_all=None,
        )
        objs.append(rail_obj)

    # 2. Telescoping upper guide rails (nested cleanly inside lower C-channels at rest,
    #    extending +0.80m along RAIL_AXIS as catapult fires -> full C.RAIL_LENGTH_M = 2.0m!)
    upper_rig = _new_empty(
        "Seat_Rail_Upper_Rig", col, parent=rails_root,
        location=base, rotation=_RAIL_EULER, display_size=0.08,
    )
    objs.append(upper_rig)

    z_upper_base = _RAIL_LOWER_LEN - _RAIL_UPPER_LEN  # 0.25m up from base at rest -> top = 1.20m
    for side, rx in (("L", C.RAIL_X[0]), ("R", C.RAIL_X[1])):
        open_dir = "+X" if rx < 0.0 else "-X"
        bm = _make_c_channel_bm(length=_RAIL_UPPER_LEN, width=0.030, depth=0.052, thick=0.007, open_side=open_dir)
        u_obj = _create_mesh_obj(
            f"Seat_Rail_Upper_{side}", bm, mats["chrome"], col,
            parent=upper_rig,
            location=(rx, 0.0, z_upper_base),
            smooth_all=None,
        )
        objs.append(u_obj)

    # Upper rail top tie-bar
    bm_top_brace = _make_beveled_box_bm((abs(C.RAIL_X[1] - C.RAIL_X[0]) + 0.04, 0.044, 0.032), 0.003, segments=1)
    top_brace = _create_mesh_obj(
        "Seat_Rail_Upper_Brace", bm_top_brace, mats["rail"], col,
        parent=upper_rig,
        location=(0.0, -0.028, _RAIL_LOWER_LEN - 0.016),
        smooth_all=None,
    )
    objs.append(top_brace)

    # Animate telescoping rail extension: stowed at 0.0 until catapult_ignite-5 (695),
    # deploys smoothly ahead of seat up to exact C.RAIL_LENGTH_M - _RAIL_LOWER_LEN = 0.80m!
    ext_max = C.RAIL_LENGTH_M - _RAIL_LOWER_LEN
    for f in _sample_frames(total_frames, [(C.EV["catapult_ignite"] - 8, 920)]):
        if f <= C.EV["catapult_ignite"] - 5:
            ext = 0.0
        else:
            u = C.smootherstep(C.EV["catapult_ignite"] - 5, 900, f)
            ext = ext_max * u
        upper_rig.location = base + _RAIL_VEC * ext
        upper_rig.keyframe_insert("location", frame=f)
    _set_interp(upper_rig, "LINEAR", "location")

    # 3. Fixed cross-braces connecting the twin guide rails
    bm_braces = bmesh.new()
    span_x = abs(C.RAIL_X[1] - C.RAIL_X[0]) + 0.036
    for s_brace in (0.10, 0.55, 1.02):
        b_bm = _make_beveled_box_bm((span_x, 0.046, 0.036), 0.003, segments=1)
        bmesh.ops.translate(b_bm, verts=b_bm.verts, vec=(0.0, -0.025, s_brace))
        _merge_bm(bm_braces, b_bm)
    braces_obj = _create_mesh_obj(
        "Seat_Rail_CrossBraces", bm_braces, mats["rail"], col,
        parent=rails_root, location=base, rotation=_RAIL_EULER, smooth_all=None,
    )
    objs.append(braces_obj)

    # 4. Stationary Outer Catapult Breech Cylinder (between rails, with exhaust relief vents)
    bm_breech = bmesh.new()
    # Main barrel from s=0.26 to s=1.05 (leaving lower s=0.0..0.26 open manifold for catapult flame!)
    bmesh.ops.create_cone(
        bm_breech, cap_ends=True, cap_tris=False, segments=28,
        radius1=0.048, radius2=0.046, depth=0.79,
        matrix=Matrix.Translation((0.0, 0.0, 0.26 + 0.395)),
    )
    # Lower breech base cap + 4 heavy stanchion struts framing the open exhaust chamber (s=0.0..0.26)
    bmesh.ops.create_cone(
        bm_breech, cap_ends=True, cap_tris=False, segments=28,
        radius1=0.056, radius2=0.052, depth=0.045,
        matrix=Matrix.Translation((0.0, 0.0, 0.0225)),
    )
    for ang_deg in (35.0, 145.0, 215.0, 325.0):
        ar = math.radians(ang_deg)
        strut = _make_beveled_box_bm((0.016, 0.016, 0.25), 0.002, segments=1)
        bmesh.ops.translate(strut, verts=strut.verts, vec=(0.046 * math.cos(ar), 0.046 * math.sin(ar), 0.135))
        _merge_bm(bm_breech, strut)
    # Strengthening clamp collars at s=0.26, 0.58, 1.02
    for cz in (0.265, 0.58, 1.02):
        col_bm = bmesh.new()
        bmesh.ops.create_cone(
            col_bm, cap_ends=True, cap_tris=False, segments=28,
            radius1=0.054, radius2=0.054, depth=0.036,
            matrix=Matrix.Translation((0.0, 0.0, cz)),
        )
        _merge_bm(bm_breech, col_bm)

    breech_obj = _create_mesh_obj(
        "Seat_Catapult_Breech", bm_breech, mats["rail"], col,
        parent=rails_root, location=base, rotation=_RAIL_EULER, smooth_all=True,
    )
    objs.append(breech_obj)

    return objs


# ---------------------------------------------------------------------------
# GEOMETRY BUILDERS: HERO MARTIN-BAKER / ACES-II SEAT (LOCAL TO SEAT_ROOT)
# ---------------------------------------------------------------------------
def _build_seat_chassis(
    seat_root: bpy.types.Object,
    mats: dict[str, bpy.types.Material],
    col: bpy.types.Collection,
) -> list[bpy.types.Object]:
    """
    Build seat pan, olive-charcoal cushions, 15 deg angled backrest (slim in z=0.12..0.68
    for Beat 4 Spine_XRay clearance), headbox, danger triangles, drogue mortar, slippers,
    and under-seat rocket nozzles.
    """
    objs: list[bpy.types.Object] = []

    # ------------------------------------------------------------------
    # 1. Seat Pan Bucket & Survival Kit Housing (|x| >= 0.205 side bolsters!)
    # ------------------------------------------------------------------
    bm_pan = bmesh.new()
    # Main survival kit pan under SRP: x ∈ [-0.198, +0.198], y ∈ [-0.20, +0.32], z ∈ [-0.17, -0.018]
    pan_prof = [
        (-0.20, -0.165),
        (+0.28, -0.165),
        (+0.32, -0.095),
        (+0.32, -0.018),
        (+0.26, -0.014),
        (-0.19, -0.018),
    ]
    _merge_bm(bm_pan, _make_profile_prism_bm(pan_prof, -0.198, +0.198, mat_index=0, bevel_w=0.005))

    # Left & Right Side Thigh Bolsters strictly OUTSIDE |x| >= 0.205 (clear for thighs!)
    bolster_prof = [
        (-0.21, -0.165),
        (+0.27, -0.165),
        (+0.32, -0.085),
        (+0.29, +0.065),
        (+0.08, +0.105),
        (-0.20, +0.105),
    ]
    _merge_bm(bm_pan, _make_profile_prism_bm(bolster_prof, -0.244, -0.205, mat_index=0, bevel_w=0.005))
    _merge_bm(bm_pan, _make_profile_prism_bm(bolster_prof, +0.205, +0.244, mat_index=0, bevel_w=0.005))

    # Front garter anchor brackets at SEAT["garter_anchor_L/R"] = (±0.10, 0.30, -0.04)
    for gx, gy, gz in (C.SEAT["garter_anchor_L"], C.SEAT["garter_anchor_R"]):
        g_blk = _make_beveled_box_bm((0.034, 0.026, 0.028), 0.003, segments=1)
        bmesh.ops.translate(g_blk, verts=g_blk.verts, vec=(gx, gy, gz))
        _merge_bm(bm_pan, g_blk, mat_index=1)

    # Under-seat sustainer rocket housing block (z = -0.195 .. -0.155)
    r_box = _make_beveled_box_bm((0.32, 0.16, 0.048), 0.004, segments=1)
    bmesh.ops.translate(r_box, verts=r_box.verts, vec=(0.0, 0.05, -0.172))
    _merge_bm(bm_pan, r_box, mat_index=0)

    pan_obj = _create_mesh_obj(
        "Seat_Pan_Structure", bm_pan, [mats["frame"], mats["chrome"]], col,
        parent=seat_root, smooth_all=None,
    )
    objs.append(pan_obj)

    # ------------------------------------------------------------------
    # 2. Olive-Charcoal Tactical Seat Cushion (Top surface centered at SRP (0,0,0))
    # ------------------------------------------------------------------
    bm_cush = bmesh.new()
    # Left & Right thigh pads leaving subtle central channel for Seat_Handle cable
    for sx in (-1.0, 1.0):
        c_prof = [
            (-0.145, -0.022),
            (+0.295, -0.022),
            (+0.305, +0.014),
            (+0.220, +0.016),
            (+0.020, +0.002),
            (-0.130, +0.008),
        ]
        _merge_bm(bm_cush, _make_profile_prism_bm(c_prof, min(sx * 0.028, sx * 0.194), max(sx * 0.028, sx * 0.194), bevel_w=0.006))
    # Center saddle bridge at SRP (0,0,0) stopping behind y = 0.25 so handle cable slides freely
    mid_prof = [
        (-0.145, -0.022),
        (+0.250, -0.022),
        (+0.250, -0.004),
        (+0.000, +0.000),
        (-0.130, +0.004),
    ]
    _merge_bm(bm_cush, _make_profile_prism_bm(mid_prof, -0.030, +0.030, bevel_w=0.004))

    cush_obj = _create_mesh_obj(
        "Seat_Cushion_Pan", bm_cush, mats["cushion"], col,
        parent=seat_root, smooth_all=True,
    )
    objs.append(cush_obj)

    # ------------------------------------------------------------------
    # 3. 15 deg Angled Backrest Frame & Cushion (CRITICAL BEAT 4 CLEARANCE!)
    #    Keep ALL geometry strictly behind y <= -0.14 in z = 0.12..0.68!
    # ------------------------------------------------------------------
    def _back_y_at(z: float, base_y: float) -> float:
        return base_y - z * _TAN_TILT

    bm_bframe = bmesh.new()
    # Main tilted backrest structural plate: z = -0.14 .. +0.76, y_front = -0.15 - z*tan(15), thickness = 0.075m
    z0, z1 = -0.14, 0.76
    plate_prof = [
        (_back_y_at(z0, -0.225), z0),
        (_back_y_at(z0, -0.150), z0),
        (_back_y_at(z1, -0.150), z1),
        (_back_y_at(z1, -0.225), z1),
    ]
    _merge_bm(bm_bframe, _make_profile_prism_bm(plate_prof, -0.185, +0.185, mat_index=0, bevel_w=0.004))

    # Twin rear rail slipper blocks gripping C.RAIL_X = (-0.20, +0.20) rails
    # Rail centerline y_rail(z) = -0.30 - (z + 0.25)*tan(15) = -0.367 - z*tan(15)
    for rx in C.RAIL_X:
        x_lo, x_hi = rx - 0.032, rx + 0.032
        for sz0, sz1 in ((-0.12, 0.04), (0.48, 0.64)):
            sl_prof = [
                (_back_y_at(sz0, -0.405), sz0),
                (_back_y_at(sz0, -0.215), sz0),
                (_back_y_at(sz1, -0.215), sz1),
                (_back_y_at(sz1, -0.405), sz1),
            ]
            _merge_bm(bm_bframe, _make_profile_prism_bm(sl_prof, x_lo, x_hi, mat_index=1, bevel_w=0.003))

    bframe_obj = _create_mesh_obj(
        "Seat_Backrest_Frame", bm_bframe, [mats["frame"], mats["chrome"]], col,
        parent=seat_root, smooth_all=None,
    )
    objs.append(bframe_obj)

    # Backrest Cushion: front face at y = -0.115 - z*tan(15) (at z=0.12 -> y=-0.147 < -0.10!)
    bm_bcush = bmesh.new()
    for x_lo, x_hi in ((-0.165, -0.024), (+0.024, +0.165)):
        for cz0, cz1 in ((0.01, 0.34), (0.36, 0.71)):
            c_prof = [
                (_back_y_at(cz0, -0.150), cz0),
                (_back_y_at(cz0, -0.115), cz0),
                (_back_y_at(cz1, -0.115), cz1),
                (_back_y_at(cz1, -0.150), cz1),
            ]
            _merge_bm(bm_bcush, _make_profile_prism_bm(c_prof, x_lo, x_hi, bevel_w=0.006))

    bcush_obj = _create_mesh_obj(
        "Seat_Cushion_Back", bm_bcush, mats["cushion"], col,
        parent=seat_root, smooth_all=True,
    )
    objs.append(bcush_obj)

    # ------------------------------------------------------------------
    # 4. Headbox (`SEAT["headbox_top"] = (0, -0.30, 1.00)`), Pad & Danger Triangles
    # ------------------------------------------------------------------
    bm_hbox = bmesh.new()
    hz0, hz1 = 0.72, 1.00
    # Armored headbox body: top face centered around y = -0.34 (-0.45 .. -0.24) at z=1.00
    hb_prof = [
        (_back_y_at(hz0, -0.245), hz0),
        (_back_y_at(hz0, -0.075), hz0),
        (_back_y_at(hz1, -0.015), hz1),
        (_back_y_at(hz1, -0.190), hz1),
    ]
    # Wait: check exact y at z=1.00: _back_y_at(1.00, -0.015) = -0.015 - 0.268 = -0.283; -0.190 - 0.268 = -0.458!
    # So SEAT["headbox_top"] = (0, -0.30, 1.00) sits cleanly ON the top deck of the headbox!
    _merge_bm(bm_hbox, _make_profile_prism_bm(hb_prof, -0.145, +0.145, mat_index=0, bevel_w=0.005))

    # Side head-containment cheek guards (strictly z = 0.76..0.99, well above Beat 4 spine window!)
    cheek_prof = [
        (-0.42, 0.76),
        (-0.18, 0.76),
        (-0.22, 0.99),
        (-0.44, 0.99),
    ]
    _merge_bm(bm_hbox, _make_profile_prism_bm(cheek_prof, -0.168, -0.142, mat_index=0, bevel_w=0.004))
    _merge_bm(bm_hbox, _make_profile_prism_bm(cheek_prof, +0.142, +0.168, mat_index=0, bevel_w=0.004))

    # Padded headrest cushion against front of headbox (material_index=1)
    hpad_prof = [
        (_back_y_at(0.75, -0.080), 0.75),
        (_back_y_at(0.75, -0.032), 0.75),
        (_back_y_at(0.95, -0.025), 0.95),
        (_back_y_at(0.95, -0.072), 0.95),
    ]
    _merge_bm(bm_hbox, _make_profile_prism_bm(hpad_prof, -0.115, +0.115, mat_index=1, bevel_w=0.007))

    # Upper shoulder brackets extending to SEAT["pitot_booms"] = (±0.24, -0.20, 0.98)
    for sx in (-1.0, 1.0):
        px = sx * 0.24
        br_bm = _make_beveled_box_bm((0.095, 0.055, 0.034), 0.003, segments=1)
        bmesh.ops.translate(br_bm, verts=br_bm.verts, vec=(sx * 0.195, -0.215, 0.975))
        _merge_bm(bm_hbox, br_bm, mat_index=0)
        boss = bmesh.new()
        bmesh.ops.create_cone(
            boss, cap_ends=True, cap_tris=False, segments=18,
            radius1=0.016, radius2=0.016, depth=0.028,
            matrix=Matrix.Translation((px, -0.20, 0.98)) @ Matrix.Rotation(math.pi / 2.0, 4, "Y"),
        )
        _merge_bm(bm_hbox, boss, mat_index=2)

    # Drogue mortar mounting saddle to (0.18, -0.34, 0.92)
    m_sad = _make_beveled_box_bm((0.055, 0.085, 0.075), 0.003, segments=1)
    bmesh.ops.translate(m_sad, verts=m_sad.verts, vec=(0.155, -0.34, 0.90))
    _merge_bm(bm_hbox, m_sad, mat_index=0)

    hbox_obj = _create_mesh_obj(
        "Seat_Headbox", bm_hbox, [mats["frame"], mats["cushion"], mats["chrome"]], col,
        parent=seat_root, smooth_all=None,
    )
    objs.append(hbox_obj)

    # Black/Yellow/Red Ejection Danger Triangles on headbox left/right cheeks
    for side, sx in (("L", -1.0), ("R", 1.0)):
        bm_tri = bmesh.new()
        x_base = sx * 0.168
        # Outer yellow hazard triangle + inner red center triangle
        for scale_t, dx, m_idx in ((1.0, 0.0025, 0), (0.64, 0.0045, 1)):
            cy, cz = -0.31, 0.875
            pts_t = [
                (cy - 0.055 * scale_t, cz - 0.036 * scale_t),
                (cy + 0.055 * scale_t, cz + 0.002),
                (cy - 0.042 * scale_t, cz + 0.042 * scale_t),
            ]
            v_back = [bm_tri.verts.new((x_base, py, pz)) for py, pz in pts_t]
            v_front = [bm_tri.verts.new((x_base + sx * dx, py, pz)) for py, pz in pts_t]
            for i in range(3):
                inext = (i + 1) % 3
                f = bm_tri.faces.new((v_back[i], v_back[inext], v_front[inext], v_front[i]))
                f.material_index = m_idx
            face_c = bm_tri.faces.new(v_front if sx > 0 else v_front[::-1])
            face_c.material_index = m_idx
        tri_obj = _create_mesh_obj(
            f"Seat_Danger_Tri_{side}", bm_tri, [mats["danger_yellow"], mats["danger_red"]], col,
            parent=seat_root, smooth_all=False,
        )
        objs.append(tri_obj)

    # ------------------------------------------------------------------
    # 5. Drogue Mortar Barrel (`SEAT["drogue_mortar"] = (0.18, -0.34, 0.92)`)
    # ------------------------------------------------------------------
    bm_mort = bmesh.new()
    bmesh.ops.create_cone(
        bm_mort, cap_ends=True, cap_tris=False, segments=24,
        radius1=0.034, radius2=0.032, depth=0.22,
        matrix=Matrix.Translation((0.0, 0.0, -0.02)),
    )
    bmesh.ops.create_cone(
        bm_mort, cap_ends=True, cap_tris=False, segments=24,
        radius1=0.038, radius2=0.038, depth=0.022,
        matrix=Matrix.Translation((0.0, 0.0, +0.085)),
    )
    mort_obj = _create_mesh_obj(
        "Seat_Drogue_Mortar", bm_mort, mats["frame"], col,
        parent=seat_root,
        location=C.SEAT["drogue_mortar"],
        rotation=_RAIL_EULER,
        smooth_all=True,
    )
    objs.append(mort_obj)

    # ------------------------------------------------------------------
    # 6. Twin Deployable Pitot Booms (`SEAT["pitot_booms"]`)
    # ------------------------------------------------------------------
    for idx, (side, p_pos) in enumerate(zip(("L", "R"), C.SEAT["pitot_booms"])):
        bm_pitot = bmesh.new()
        # Sleek boom arm along local +Z from pivot (0,0,0) to +0.26m + tapered pitot cone
        bmesh.ops.create_cone(
            bm_pitot, cap_ends=True, cap_tris=False, segments=14,
            radius1=0.011, radius2=0.007, depth=0.22,
            matrix=Matrix.Translation((0.0, 0.0, 0.11)),
        )
        bmesh.ops.create_cone(
            bm_pitot, cap_ends=True, cap_tris=False, segments=14,
            radius1=0.007, radius2=0.002, depth=0.055,
            matrix=Matrix.Translation((0.0, 0.0, 0.245)),
        )
        pitot_obj = _create_mesh_obj(
            f"Seat_Pitot_Boom_{side}", bm_pitot, mats["chrome"], col,
            parent=seat_root,
            location=p_pos,
            rotation=Euler((math.radians(-78.0), 0.0, 0.0), "XYZ"),
            smooth_all=True,
        )
        objs.append(pitot_obj)

    # ------------------------------------------------------------------
    # 7. Telescoping Inner Catapult Tube (attached to Seat_Root at RAIL_BASE)
    # ------------------------------------------------------------------
    bm_piston = bmesh.new()
    bmesh.ops.create_cone(
        bm_piston, cap_ends=True, cap_tris=False, segments=28,
        radius1=0.036, radius2=0.035, depth=1.15,
        matrix=Matrix.Translation((0.0, 0.0, 0.575)),
    )
    # Lower catapult piston pressure collar & brass gas-seal ring at s=0.02
    seal = bmesh.new()
    bmesh.ops.create_cone(
        seal, cap_ends=True, cap_tris=False, segments=28,
        radius1=0.041, radius2=0.038, depth=0.055,
        matrix=Matrix.Translation((0.0, 0.0, 0.028)),
    )
    _merge_bm(bm_piston, seal, mat_index=1)
    piston_obj = _create_mesh_obj(
        "Seat_Catapult_Piston", bm_piston, [mats["chrome"], mats["nozzle"]], col,
        parent=seat_root, location=C.RAIL_BASE, rotation=_RAIL_EULER, smooth_all=True,
    )
    objs.append(piston_obj)

    # ------------------------------------------------------------------
    # 8. Twin Under-Seat Rocket Nozzles (`SEAT["rocket_nozzles"]`)
    # ------------------------------------------------------------------
    for side, n_pos in zip(("L", "R"), C.SEAT["rocket_nozzles"]):
        bm_noz = _make_delaval_nozzle_bm()
        # Rotate nozzle so bell exit points along _NOZZLE_DIR (+10 deg forward, -Z down)
        # Note: _make_delaval_nozzle_bm has bell exit at z=0 and collar at z=-0.058, so +Z points OUT of nozzle!
        noz_obj = _create_mesh_obj(
            f"Seat_Rocket_Nozzle_{side}", bm_noz, mats["nozzle"], col,
            parent=seat_root,
            location=n_pos,
            rotation=_NOZZLE_QUAT.to_euler("XYZ"),
            smooth_all=True,
        )
        objs.append(noz_obj)

    return objs


# ---------------------------------------------------------------------------
# INTERACTIVE HARDWARE: YELLOW-BLACK EJECTION HANDLE & LEG-RESTRAINT GARTERS
# ---------------------------------------------------------------------------
def _build_handle_and_cable(
    seat_root: bpy.types.Object,
    mats: dict[str, bpy.types.Material],
    col: bpy.types.Collection,
    total_frames: int,
) -> tuple[bpy.types.Object, list[bpy.types.Object]]:
    objs: list[bpy.types.Object] = []

    # 1. Fixed front pan firing-valve housing collar
    bm_hs = _make_beveled_box_bm((0.056, 0.038, 0.026), 0.004, segments=1)
    housing = _create_mesh_obj(
        "Seat_Handle_Housing", bm_hs, mats["frame"], col,
        parent=seat_root, location=_HANDLE_ORIFICE_LOCAL, smooth_all=None,
    )
    objs.append(housing)

    # 2. Hero Yellow-Black Double-Loop Ejection Handle (Origin == C.handle_local grip center!)
    bm_handle = bmesh.new()
    left_ctrl = [
        Vector((-0.008, +0.012, -0.044)),
        Vector((-0.036, +0.008, -0.032)),
        Vector((-0.052, +0.002, -0.008)),
        Vector((-0.045,  0.000, +0.005)),  # exact left grip bar
        Vector((-0.022, -0.006, +0.014)),
        Vector((-0.006, +0.006, -0.022)),
        Vector((-0.006, +0.012, -0.044)),
    ]
    base_pts = _catmull_rom(left_ctrl, 4)
    for sign_x in (-1.0, 1.0):
        pts = [Vector((sign_x * abs(p.x), p.y, p.z)) for p in base_pts]
        loop_bm = _make_tube_along_path_bm(pts, radius=0.0086, n_seg=14, cap_ends=True, mat_index=0)
        _merge_bm(bm_handle, loop_bm)

    # Central anodized clevis block joining the twin loops
    clevis = _make_beveled_box_bm((0.034, 0.024, 0.024), 0.003, segments=1)
    bmesh.ops.translate(clevis, verts=clevis.verts, vec=(0.0, +0.010, -0.044))
    _merge_bm(bm_handle, clevis, mat_index=1)

    handle_obj = _create_mesh_obj(
        "Seat_Handle", bm_handle, [mats["hazard_handle"], mats["frame"]], col,
        parent=seat_root, location=C.handle_local(1), smooth_all=True,
    )
    objs.append(handle_obj)

    # 3. Steel Firing Pull Cable (unit-Z cylinder from 0..1 anchored at _HANDLE_ORIFICE_LOCAL)
    bm_cab = bmesh.new()
    bmesh.ops.create_cone(
        bm_cab, cap_ends=True, cap_tris=False, segments=12,
        radius1=0.0044, radius2=0.0044, depth=1.0,
        matrix=Matrix.Translation((0.0, 0.0, 0.5)),
    )
    cable_obj = _create_mesh_obj(
        "Seat_Handle_Cable", bm_cab, mats["chrome"], col,
        parent=seat_root, location=_HANDLE_ORIFICE_LOCAL, smooth_all=True,
    )
    objs.append(cable_obj)

    # Animate handle & pull cable cleanly across 1..EV["separation"]-1, hidden for f >= separation!
    vis_end = min(total_frames, C.EV["separation"] - 1, C.EV["seat_hidden"] - 1)
    yank_win = [(C.EV["yank_start"] - 2, C.EV["yank_end"] + 4)]
    for f in _sample_frames(vis_end, yank_win):
        h_pos = Vector(C.handle_local(f))
        handle_obj.location = h_pos
        handle_obj.keyframe_insert("location", frame=f)

        top_pt = h_pos + _HANDLE_CLEVIS_OFFSET
        delta = top_pt - _HANDLE_ORIFICE_LOCAL
        length = max(0.015, delta.length)
        cable_obj.rotation_euler = delta.normalized().to_track_quat("Z", "Y").to_euler("XYZ")
        cable_obj.scale = Vector((1.0, 1.0, length))
        cable_obj.keyframe_insert("rotation_euler", frame=f)
        cable_obj.keyframe_insert("scale", frame=f)

    _set_interp(handle_obj, "LINEAR", "location")
    _set_interp(cable_obj, "LINEAR", "rotation_euler")
    _set_interp(cable_obj, "LINEAR", "scale")
    _key_visibility(handle_obj, [(1, vis_end)], total_frames)
    _key_visibility(cable_obj, [(1, vis_end)], total_frames)

    return handle_obj, objs


def _build_garters(
    seat_root: bpy.types.Object,
    mats: dict[str, bpy.types.Material],
    ctrl: dict[str, Any],
    col: bpy.types.Collection,
    total_frames: int,
) -> list[bpy.types.Object]:
    """Build Seat_Garter_L/R straps + Seat_Ankle_Cuff_L/R collars with Beat 3 emissive highlight."""
    garters: list[bpy.types.Object] = []
    vis_end = min(total_frames, C.EV["separation"] - 1, C.EV["seat_hidden"] - 1)
    snap_win = [(C.EV["garter_start"] - 4, C.EV["garter_end"] + 6)]

    for side in ("L", "R"):
        anchor = Vector(C.SEAT[f"garter_anchor_{side}"])

        # 1. Flat tactical webbing strap extending along local +Z (0..1)
        bm_strap = _make_beveled_box_bm((0.028, 0.0055, 1.0), 0.0012, segments=1)
        bmesh.ops.translate(bm_strap, verts=bm_strap.verts, vec=(0.0, 0.0, 0.5))
        strap_obj = _create_mesh_obj(
            f"Seat_Garter_{side}", bm_strap, mats["garter_strap"], col,
            parent=seat_root, location=anchor, smooth_all=None,
        )

        # 2. Padded Ankle Restraint Cuff + Quick-Release D-Ring around C.ankle_local(side, f)
        bm_cuff = bmesh.new()
        n_seg = 24
        r_in, r_out, hz = 0.055, 0.064, 0.022
        ri_b, ri_t, ro_b, ro_t = [], [], [], []
        for k in range(n_seg):
            a = k * 2.0 * math.pi / n_seg
            ca, sa = math.cos(a), math.sin(a)
            ri_b.append(bm_cuff.verts.new((r_in * ca, r_in * sa, -hz)))
            ri_t.append(bm_cuff.verts.new((r_in * ca, r_in * sa, +hz)))
            ro_b.append(bm_cuff.verts.new((r_out * ca, r_out * sa, -hz)))
            ro_t.append(bm_cuff.verts.new((r_out * ca, r_out * sa, +hz)))
        for k in range(n_seg):
            kn = (k + 1) % n_seg
            bm_cuff.faces.new((ro_b[k], ro_b[kn], ro_t[kn], ro_t[k])).smooth = True
            bm_cuff.faces.new((ri_b[k], ri_t[k], ri_t[kn], ri_b[kn])).smooth = True
            bm_cuff.faces.new((ro_t[k], ro_t[kn], ri_t[kn], ri_t[k])).smooth = False
            bm_cuff.faces.new((ro_b[k], ri_b[k], ri_b[kn], ro_b[kn])).smooth = False
        # Front D-ring buckle facing -Y (toward garter strap anchor)
        d_blk = _make_beveled_box_bm((0.030, 0.014, 0.024), 0.002, segments=1)
        bmesh.ops.translate(d_blk, verts=d_blk.verts, vec=(0.0, -r_out - 0.003, 0.0))
        _merge_bm(bm_cuff, d_blk, mat_index=1)

        cuff_obj = _create_mesh_obj(
            f"Seat_Ankle_Cuff_{side}", bm_cuff, [mats["ankle_cuff"], mats["chrome"]], col,
            parent=seat_root, location=C.ankle_local(side, 1), smooth_all=None,
        )

        # Animate strap & cuff across 1 .. vis_end
        for f in _sample_frames(vis_end, snap_win):
            ankle = Vector(C.ankle_local(side, f))
            cuff_target = ankle + Vector((0.0, -r_out, 0.0))
            delta = cuff_target - anchor
            length = max(0.03, delta.length)
            strap_obj.rotation_euler = delta.normalized().to_track_quat("Z", "Y").to_euler("XYZ")
            strap_obj.scale = Vector((1.0, 1.0, length))
            strap_obj.keyframe_insert("rotation_euler", frame=f)
            strap_obj.keyframe_insert("scale", frame=f)

            cuff_obj.location = ankle
            cuff_obj.keyframe_insert("location", frame=f)

        _set_interp(strap_obj, "LINEAR", "rotation_euler")
        _set_interp(strap_obj, "LINEAR", "scale")
        _set_interp(cuff_obj, "LINEAR", "location")
        _key_visibility(strap_obj, [(1, vis_end)], total_frames)
        _key_visibility(cuff_obj, [(1, vis_end)], total_frames)

        garters.extend([strap_obj, cuff_obj])

    # Animate Beat 3 subtle emissive highlight across f=600..760 (AgX-safe <= 1.85)
    for sock_key, peak in (("garter_em_socket", 1.55), ("cuff_em_socket", 1.85)):
        sock = ctrl[sock_key]
        for f, val in (
            (1, 0.0),
            (592, 0.0),
            (C.EV["garter_start"] + 4, peak),
            (725, peak * 0.85),
            (768, 0.0),
            (total_frames, 0.0),
        ):
            fc = max(1, min(total_frames, f))
            sock.default_value = val
            sock.keyframe_insert("default_value", frame=fc)

    return garters


# ---------------------------------------------------------------------------
# EXHAUST & ROCKET PLUME FX (Local-Space TexCoord.Object -> Seat_Root!)
# ---------------------------------------------------------------------------
def _make_displaced_plume_core_bm(
    radius: float,
    length: float,
    seed: int = 17,
) -> bmesh.types.BMesh:
    """Create a 3D turbulent displaced elongated plasma core extending along +Z (0..length)."""
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0)
    for v in bm.verts:
        p = v.co.normalized()
        # Stretch into tapered teardrop along +Z in [0, length]
        z_01 = 0.5 * (p.z + 1.0)  # 0 at nozzle, 1 at tip
        r_prof = math.sin(math.pi * max(0.02, min(0.98, z_01 ** 0.72)))
        turb = (
            math.sin(p.x * 5.3 + seed) * math.cos(p.y * 4.9 - seed)
            + 0.45 * math.sin(p.z * 8.1 + p.x * 3.7)
        )
        r_here = radius * r_prof * (1.0 + 0.22 * turb)
        v.co = Vector((p.x * r_here, p.y * r_here, z_01 * length))
    return bm


def _build_catapult_fx(
    seat_root: bpy.types.Object,
    mats: dict[str, bpy.types.Material],
    ctrl: dict[str, Any],
    col: bpy.types.Collection,
    total_frames: int,
) -> list[bpy.types.Object]:
    """Catapult blast flame cones (cap_ends=False), plasma core, sparks, & bounded light (700..920)."""
    f_start = C.EV["catapult_ignite"]
    f_end = min(920, C.EV["seat_hidden"] - 1)
    vis_ranges = [(f_start, f_end)]
    fx_objs: list[bpy.types.Object] = []

    # Anchor moving with Seat_Root's catapult piston base along -RAIL_AXIS
    cat_rig = _new_empty(
        "Seat_Catapult_Exhaust", col, parent=seat_root,
        location=C.RAIL_BASE, rotation=(- _RAIL_VEC).to_track_quat("Z", "Y").to_euler("XYZ"),
        display_size=0.08,
    )
    _key_visibility(cat_rig, vis_ranges, total_frames)
    fx_objs.append(cat_rig)

    # 1. Open-ended outer catapult exhaust cone (cap_ends=False)
    bm_cone = bmesh.new()
    bmesh.ops.create_cone(
        bm_cone, cap_ends=False, cap_tris=False, segments=24,
        radius1=0.052, radius2=0.19, depth=0.95,
        matrix=Matrix.Translation((0.0, 0.0, 0.475)),
    )
    cone_obj = _create_mesh_obj(
        "Seat_Catapult_Plume_Cone", bm_cone, mats["catapult_plume"], col,
        parent=cat_rig, smooth_all=True,
    )
    _key_visibility(cone_obj, vis_ranges, total_frames)
    fx_objs.append(cone_obj)

    # 2. Turbulent displaced inner plasma core + upper breech vent collar flame
    bm_core = _make_displaced_plume_core_bm(radius=0.055, length=0.68, seed=29)
    core_obj = _create_mesh_obj(
        "Seat_Catapult_Core", bm_core, mats["catapult_core"], col,
        parent=cat_rig, smooth_all=True,
    )
    _key_visibility(core_obj, vis_ranges, total_frames)
    fx_objs.append(core_obj)

    # 3. 14 Open-ended spark streaks shooting downward from catapult breech
    rng = random.Random(700)
    for k in range(14):
        ang = rng.uniform(0.0, 2.0 * math.pi)
        spread = rng.uniform(0.06, 0.36)
        d = Vector((math.sin(spread) * math.cos(ang), math.sin(spread) * math.sin(ang), math.cos(spread))).normalized()
        s_len = rng.uniform(0.18, 0.42)
        bm_sp = bmesh.new()
        bmesh.ops.create_cone(
            bm_sp, cap_ends=False, cap_tris=False, segments=6,
            radius1=0.0055, radius2=0.0005, depth=s_len,
        )
        sp_obj = _create_mesh_obj(
            f"Seat_Catapult_Spark_{k:02d}", bm_sp, mats["catapult_spark"], col,
            parent=cat_rig,
            location=d * (0.10 + 0.5 * s_len),
            rotation=d.to_track_quat("Z", "Y").to_euler("XYZ"),
            smooth_all=True,
        )
        _key_visibility(sp_obj, vis_ranges, total_frames)
        fx_objs.append(sp_obj)

    # 4. Bounded dynamic POINT light (115W, strictly hidden outside 700..920)
    l_data = bpy.data.lights.new("Seat_Catapult_Light", type="POINT")
    l_data.color = (0.65, 0.88, 1.0)
    l_data.energy = 0.0
    l_data.shadow_soft_size = 0.16
    l_obj = bpy.data.objects.new("Seat_Catapult_Light", l_data)
    l_obj.parent = cat_rig
    l_obj.location = Vector((0.0, 0.0, 0.14))
    _link_obj(l_obj, col)
    _key_visibility(l_obj, vis_ranges, total_frames)
    fx_objs.append(l_obj)

    for f, e, sc in (
        (f_start - 1, 0.0, 0.10),
        (f_start, 45.0, 0.45),
        (f_start + 8, 115.0, 1.00),
        (780, 110.0, 1.12),
        (860, 75.0, 0.92),
        (f_end, 0.0, 0.12),
    ):
        fc = max(1, min(total_frames, f))
        l_data.energy = e
        l_data.keyframe_insert("energy", frame=fc)
        cat_rig.scale = Vector((sc, sc, sc))
        cat_rig.keyframe_insert("scale", frame=fc)

    # Animate shader gain & flicker phase across 700..920
    for gn in ctrl.get("catapult_gain_nodes", []):
        sock = gn.outputs[0]
        for f, val in ((f_start - 1, 0.0), (f_start + 6, 1.15), (840, 1.0), (f_end, 0.0)):
            sock.default_value = val
            sock.keyframe_insert("default_value", frame=max(1, min(total_frames, f)))
    for pn in ctrl.get("catapult_phase_nodes", []):
        sock = pn.outputs[0]
        for f, val in ((f_start, 0.0), (f_end, 6.0)):
            sock.default_value = val
            sock.keyframe_insert("default_value", frame=max(1, min(total_frames, f)))

    return fx_objs


def _build_rocket_fx(
    seat_root: bpy.types.Object,
    mats: dict[str, bpy.types.Material],
    ctrl: dict[str, Any],
    col: bpy.types.Collection,
    total_frames: int,
) -> list[bpy.types.Object]:
    """Twin under-seat rocket plumes (`cap_ends=False` Mach-diamond cones + plasma cores + <=140W light)."""
    f_start = C.EV["rocket_ignite"]
    f_end = min(C.EV["rocket_burnout"], C.EV["seat_hidden"] - 1)
    vis_ranges = [(f_start, f_end)]
    fx_objs: list[bpy.types.Object] = []
    noz_euler = _NOZZLE_QUAT.to_euler("XYZ")

    for side, n_pos in zip(("L", "R"), C.SEAT["rocket_nozzles"]):
        rig = _new_empty(
            f"Seat_Rocket_PlumeRig_{side}", col, parent=seat_root,
            location=n_pos, rotation=noz_euler, display_size=0.06,
        )
        _key_visibility(rig, vis_ranges, total_frames)
        fx_objs.append(rig)

        # 1. Displaced turbulent inner plasma core (0 .. 1.15m)
        bm_core = _make_displaced_plume_core_bm(radius=0.064, length=1.15, seed=41 if side == "L" else 73)
        core_obj = _create_mesh_obj(
            f"Seat_Rocket_Core_{side}", bm_core, mats["rocket_core"], col,
            parent=rig, smooth_all=True,
        )
        _key_visibility(core_obj, vis_ranges, total_frames)
        fx_objs.append(core_obj)

        # 2. Open-ended Mach-diamond shock cone (cap_ends=False, 0 .. 1.60m)
        bm_mach = bmesh.new()
        bmesh.ops.create_cone(
            bm_mach, cap_ends=False, cap_tris=False, segments=24,
            radius1=0.044, radius2=0.145, depth=1.60,
            matrix=Matrix.Translation((0.0, 0.0, 0.80)),
        )
        mach_obj = _create_mesh_obj(
            f"Seat_Rocket_Mach_{side}", bm_mach, mats["rocket_mach"], col,
            parent=rig, smooth_all=True,
        )
        _key_visibility(mach_obj, vis_ranges, total_frames)
        fx_objs.append(mach_obj)

        # 3. Open-ended wide turbulent exhaust envelope (cap_ends=False, 0 .. 2.20m)
        bm_plume = bmesh.new()
        bmesh.ops.create_cone(
            bm_plume, cap_ends=False, cap_tris=False, segments=28,
            radius1=0.048, radius2=0.255, depth=2.20,
            matrix=Matrix.Translation((0.0, 0.0, 1.10)),
        )
        plume_obj = _create_mesh_obj(
            f"Seat_Rocket_Plume_{side}", bm_plume, mats["rocket_plume"], col,
            parent=rig, smooth_all=True,
        )
        _key_visibility(plume_obj, vis_ranges, total_frames)
        fx_objs.append(plume_obj)

        # 4. 8 Open-ended radial spark streaks per nozzle
        rng = random.Random(1250 + (0 if side == "L" else 99))
        for k in range(8):
            ang = rng.uniform(0.0, 2.0 * math.pi)
            spr = rng.uniform(0.04, 0.22)
            d = Vector((math.sin(spr) * math.cos(ang), math.sin(spr) * math.sin(ang), math.cos(spr))).normalized()
            s_len = rng.uniform(0.28, 0.62)
            bm_sp = bmesh.new()
            bmesh.ops.create_cone(
                bm_sp, cap_ends=False, cap_tris=False, segments=6,
                radius1=0.0050, radius2=0.0005, depth=s_len,
            )
            sp_obj = _create_mesh_obj(
                f"Seat_Rocket_Spark_{side}_{k:02d}", bm_sp, mats["rocket_spark"], col,
                parent=rig,
                location=d * (0.20 + 0.5 * s_len),
                rotation=d.to_track_quat("Z", "Y").to_euler("XYZ"),
                smooth_all=True,
            )
            _key_visibility(sp_obj, vis_ranges, total_frames)
            fx_objs.append(sp_obj)

        # Animate crisp ignition punch -> pulsing sustain -> crisp burnout taper (single-axis Z spin on core_obj!)
        for f, sc_xy, sc_z, rz in (
            (f_start - 1, 0.10, 0.05, 0.0),
            (f_start, 0.55, 0.45, 0.1),
            (f_start + 8, 1.05, 1.00, 0.6),
            (1320, 1.00, 1.12, 2.2),
            (1390, 1.06, 1.04, 4.0),
            (f_end - 18, 0.90, 0.85, 5.4),
            (f_end, 0.12, 0.08, 6.2),
        ):
            fc = max(1, min(total_frames, f))
            rig.scale = Vector((sc_xy, sc_xy, sc_z))
            rig.keyframe_insert("scale", frame=fc)
            core_obj.rotation_euler = Euler((0.0, 0.0, rz), "XYZ")
            core_obj.keyframe_insert("rotation_euler", frame=fc)

    # Bounded dynamic POINT light (<= 140W -> 135W peak, hidden outside 1250..1460)
    rl_data = bpy.data.lights.new("Seat_Rocket_Light", type="POINT")
    rl_data.color = (1.0, 0.56, 0.16)
    rl_data.energy = 0.0
    rl_data.shadow_soft_size = 0.24
    rl_obj = bpy.data.objects.new("Seat_Rocket_Light", rl_data)
    rl_obj.parent = seat_root
    rl_obj.location = Vector((0.0, 0.14, -0.52))
    _link_obj(rl_obj, col)
    _key_visibility(rl_obj, vis_ranges, total_frames)
    fx_objs.append(rl_obj)

    for f, e in (
        (f_start - 1, 0.0),
        (f_start, 60.0),
        (f_start + 6, 135.0),
        (1340, 130.0),
        (f_end - 18, 95.0),
        (f_end, 0.0),
    ):
        rl_data.energy = e
        rl_data.keyframe_insert("energy", frame=max(1, min(total_frames, f)))

    # Animate rocket shader gain & turbulence flicker
    for gn in ctrl.get("rocket_gain_nodes", []):
        sock = gn.outputs[0]
        for f, val in ((f_start - 1, 0.0), (f_start + 5, 1.15), (f_end - 18, 1.0), (f_end, 0.0)):
            sock.default_value = val
            sock.keyframe_insert("default_value", frame=max(1, min(total_frames, f)))
    for pn in ctrl.get("rocket_phase_nodes", []):
        sock = pn.outputs[0]
        for f, val in ((f_start, 0.0), (f_end, 14.0)):
            sock.default_value = val
            sock.keyframe_insert("default_value", frame=max(1, min(total_frames, f)))

    return fx_objs


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------
def build_ejection_seat(col: bpy.types.Collection, total_frames: int = 1800) -> dict[str, Any]:
    """
    Build the complete Martin-Baker / ACES-II ejection seat, telescoping guide rails,
    catapult breech & exhaust FX, yellow-black ejection handle, leg-restraint garters,
    and under-seat rocket nozzles & Mach-diamond plumes.
    """
    if col is None:
        col = bpy.context.scene.collection

    total_frames = int(total_frames)
    all_objs: list[bpy.types.Object] = []

    # 1. Two Top-Level Roots
    seat_root = _new_empty("Seat_Root", col, location=C.SRP, display_size=0.16)
    rails_root = _new_empty("Seat_Rails_Root", col, location=C.SRP, display_size=0.18)
    all_objs.extend([seat_root, rails_root])

    # 2. Local-Space Materials
    mats, ctrl = _build_materials(seat_root, rails_root)

    # 3. Stationary Guide Rails & Catapult Breech (parented to Seat_Rails_Root)
    rail_objs = _build_guide_rails(rails_root, mats, col, total_frames)
    all_objs.extend(rail_objs)

    # 4. Hero Seat Chassis, Cushions, Headbox, Mortar, Pitot Booms, Piston, Nozzles
    chassis_objs = _build_seat_chassis(seat_root, mats, col)
    all_objs.extend(chassis_objs)

    # Animate pitot booms deployment cleanly upon rail_exit (1236 -> 1272)
    for obj in chassis_objs:
        if obj.name.startswith("Seat_Pitot_Boom_"):
            obj.rotation_euler = Euler((math.radians(-78.0), 0.0, 0.0), "XYZ")
            obj.keyframe_insert("rotation_euler", frame=1)
            obj.keyframe_insert("rotation_euler", frame=min(total_frames, C.EV["rail_exit"]))
            obj.rotation_euler = Euler((math.radians(-12.0), 0.0, 0.0), "XYZ")
            obj.keyframe_insert("rotation_euler", frame=min(total_frames, C.EV["rail_exit"] + 36))
            _set_interp(obj, "LINEAR", "rotation_euler")

    # 5. Yellow-Black Ejection Handle + Firing Cable
    handle_obj, handle_objs = _build_handle_and_cable(seat_root, mats, col, total_frames)
    all_objs.extend(handle_objs)

    # 6. Leg-Restraint Garters + Ankle Cuffs
    garters = _build_garters(seat_root, mats, ctrl, col, total_frames)
    all_objs.extend(garters)

    # 7. Catapult & Rocket Optical FX
    catapult_fx = _build_catapult_fx(seat_root, mats, ctrl, col, total_frames)
    rocket_fx = _build_rocket_fx(seat_root, mats, ctrl, col, total_frames)
    all_objs.extend(catapult_fx)
    all_objs.extend(rocket_fx)

    # 8. Animate Seat_Root across 1..total_frames (every 2 frames + all EV events, LINEAR)
    for f in _sample_frames(total_frames):
        seat_root.location = Vector(C.seat_world_location(f))
        seat_root.rotation_euler = Euler(C.seat_world_rotation_euler(f), "XYZ")
        seat_root.keyframe_insert("location", frame=f)
        seat_root.keyframe_insert("rotation_euler", frame=f)
    _set_interp(seat_root, "LINEAR", "location")
    _set_interp(seat_root, "LINEAR", "rotation_euler")

    # 9. Mandatory Cleanup: Seat_Root and ALL Seat_* children (excluding Seat_Rails_Root)
    #    MUST be CONSTANT-hidden from EV["seat_hidden"] = 1740 through total_frames!
    seat_vis_end = min(total_frames, C.EV["seat_hidden"] - 1)
    _key_visibility(seat_root, [(1, seat_vis_end)], total_frames)
    rails_set = {rails_root} | set(rail_objs)
    custom_vis_objs = {handle_obj} | {o for o in handle_objs if o.name == "Seat_Handle_Cable"} | set(garters) | set(catapult_fx) | set(rocket_fx)
    for obj in all_objs:
        if obj in rails_set or obj == seat_root or obj in custom_vis_objs:
            continue
        _key_visibility(obj, [(1, seat_vis_end)], total_frames)

    return {
        "root": seat_root,
        "rails_root": rails_root,
        "handle": handle_obj,
        "garters": garters,
        "catapult_fx": catapult_fx,
        "rocket_fx": rocket_fx,
        "all_objs": all_objs,
    }


# ---------------------------------------------------------------------------
# STANDALONE SELF-TEST (Builds seat into empty scene & renders f=160 and f=1320)
# ---------------------------------------------------------------------------
def _run_self_test() -> None:
    out_dir = Path("/tmp/fje_seat")
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1920
    scene.render.resolution_percentage = 35
    scene.render.fps = C.FPS
    scene.frame_start = 1
    scene.frame_end = C.TOTAL_FRAMES
    scene.render.image_settings.file_format = "PNG"

    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Punchy"
    except Exception:
        scene.view_settings.look = "None"

    cycles = scene.cycles
    cycles.device = "CPU"
    cycles.samples = 6
    cycles.use_denoising = True
    cycles.denoiser = "OPENIMAGEDENOISE"
    cycles.transparent_max_bounces = 12
    cycles.caustics_reflective = False
    cycles.caustics_refractive = False

    world = bpy.data.worlds.new("Seat_QA_World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.14, 0.22, 0.34, 1.0)
        bg.inputs["Strength"].default_value = 0.85

    sun_data = bpy.data.lights.new("Seat_QA_Sun", type="SUN")
    sun_data.energy = 3.2
    sun_obj = bpy.data.objects.new("Seat_QA_Sun", sun_data)
    sun_obj.rotation_euler = Vector(C.SUN_TOWARD).to_track_quat("Z", "Y").to_euler("XYZ")
    scene.collection.objects.link(sun_obj)

    col = bpy.data.collections.new("Seat_QA")
    scene.collection.children.link(col)
    res = build_ejection_seat(col, total_frames=C.TOTAL_FRAMES)

    cam_data = bpy.data.cameras.new("Seat_QA_Cam")
    cam_data.lens = 35.0
    cam_obj = bpy.data.objects.new("Seat_QA_Cam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    for frame, cam_off, target_off in (
        (160, Vector((0.85, 1.55, 0.35)), Vector((0.0, 0.15, 0.25))),
        (1320, Vector((2.6, 3.2, -0.35)), Vector((0.0, -0.1, -0.35))),
    ):
        scene.frame_set(frame)
        seat_loc = Vector(C.seat_world_location(frame))
        cam_obj.location = seat_loc + cam_off
        cam_obj.rotation_euler = (seat_loc + target_off - cam_obj.location).to_track_quat("-Z", "Y").to_euler("XYZ")
        scene.render.filepath = str(out_dir / f"seat_f{frame:04d}.png")
        bpy.ops.render.render(write_still=True)

    print(
        f"[Seat Self-Test] Built {len(res['all_objs'])} Seat_* objects; "
        f"rendered f=160 and f=1320 to {out_dir}"
    )


if __name__ == "__main__":
    _run_self_test()
