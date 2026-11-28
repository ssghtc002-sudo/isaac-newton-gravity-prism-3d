"""
elevator_cab_asset.py — Luxury Brushed-Steel Elevator Cab, Hoist Cables, Safety Clamps & Floor Grid
Episode #16: "Why Jumping in a Falling Elevator WON'T Save You" (4K 60fps, 1800 frames).

Owned by the Elevator Cab & Safety Brakes Artist (`AGENTS.md` Rule 7).
ALL coordinates, dimensions, timings, and trajectories are imported from `elevator_contract.py`.

Public API:
    build_elevator_cab(col: bpy.types.Collection, total_frames: int = 1800) -> dict

Returned dict schema:
    {
        "root": bpy.types.Object,                  # `Cab_Root` animated via C.cab_world_location(f)
        "xray_nodes": list[Any],                   # `XRayCab` ShaderNodeValue drivers
        "alarm_light": bpy.types.Object,           # `Cab_Alarm_Light` flashing crimson strobe
        "cables": list[bpy.types.Object],          # Intact, upper recoiling & lower whipping cables
        "snap_sparks": list[bpy.types.Object],     # f=325..365 additive cable-snap spark burst
        "clamps": list[bpy.types.Object],          # Under-cab wedge safety clamp assemblies
        "safety_clamps": list[bpy.types.Object],   # Alias to `clamps`
        "brake_sparks": list[bpy.types.Object],    # f=1270..1365 upward rail-friction sparks
        "floor_led": bpy.types.Object,             # `Cab_Floor_LED_Housing` header display bezel
        "floor_led_readouts": list[bpy.types.Object],  # Discrete animated floor readout texts
        "force_grid": bpy.types.Object,            # Beat 6 flush 12x floor force-distribution grid
        "doors": list[bpy.types.Object],           # [Cab_Door_L, Cab_Door_R]
        "handrail": bpy.types.Object,              # `Cab_Rear_Handrail`
        "cop_panel": bpy.types.Object,             # `Cab_COP_Panel`
        "ceiling_lights": list[bpy.types.Object],  # Ceiling LED diffuser & area light
        "all_objs": list[bpy.types.Object],        # Every `Cab_*` object created & linked to `col`
    }
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EPISODE_DIR = SCRIPT_DIR.parent
REPO_ROOT = EPISODE_DIR.parent.parent
for _p in (str(SCRIPT_DIR), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import elevator_contract as C  # noqa: E402

try:
    import bmesh  # type: ignore[import-not-found]
    import bpy  # type: ignore[import-not-found]
    from mathutils import Euler, Matrix, Vector  # type: ignore[import-not-found]
except ImportError:  # Pure-Python static QA outside Blender
    bmesh = None  # type: ignore[assignment]
    bpy = None  # type: ignore[assignment]
    Euler = None  # type: ignore[assignment]
    Matrix = None  # type: ignore[assignment]
    Vector = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# CONTRACT-DERIVED CONSTANTS
# ---------------------------------------------------------------------------
PREFIX = "Cab_"
DEFAULT_FONT_PATH = EPISODE_DIR / "subtitles" / "fonts" / "DejaVuSans-Bold.ttf"

HALF_W: float = float(C.CAB["half_w"])          # 0.85
HALF_D: float = float(C.CAB["half_d"])          # 0.70
CEILING_Z: float = float(C.CAB["ceiling_z"])    # 2.35
ROOF_Z: float = float(C.CAB["roof_z"])          # 2.45
HITCH_Z: float = float(C.CAB["hitch_z"])        # 2.55
HANDRAIL_Y: float = float(C.CAB["handrail_y"])  # 0.61
HANDRAIL_Z: float = float(C.CAB["handrail_z"])  # 0.92
FLOOR_LED_POS: tuple[float, float, float] = tuple(float(v) for v in C.CAB["floor_led"])
COP_POS: tuple[float, float, float] = tuple(float(v) for v in C.CAB["cop_panel"])
RAIL_X: float = float(C.CAB["rail_x"])          # 0.96
CLAMP_Z: float = float(C.CAB["clamp_z"])        # -0.14
BUFFER_X: float = float(C.CAB["buffer_x"])      # 0.42
SHEAVE_Z: float = float(C.SHAFT["ceiling_sheave_z"])  # 27.5

ALARM_START: int = int(C.EV["alarm_start"])     # 110
CABLE_SNAP: int = int(C.EV["cable_snap"])       # 325
BRAKE_START: int = int(C.EV["brake_cut_start"]) # 1237
BRAKE_CLAMP: int = int(C.EV["brake_clamp"])     # 1275
BRAKE_END: int = int(C.EV["brake_cut_end"])     # 1385
GRID_REVEAL: int = int(C.EV["grid_reveal"])     # 1545

CABLE_ANCHORS: tuple[tuple[float, float], ...] = (
    (-0.08, -0.03),
    (0.08, -0.03),
    (-0.08, 0.03),
    (0.08, 0.03),
)
SEVER_OFFSET_Z: float = 0.25
SNAP_ORIGIN_Z: float = HITCH_Z + SEVER_OFFSET_Z  # 2.80

CLAMP_OPEN_CLEARANCE: float = 0.008  # 8 mm exact clearance from rail blade through f=1260
RAIL_BLADE_HALF_Y: float = 0.010     # 20 mm thick vertical T-rail blade centered at Y=0


# ---------------------------------------------------------------------------
# PURE-MATH CURVES & TIMING EVALUATORS
# ---------------------------------------------------------------------------
def clamp_engage_factor(f: float) -> float:
    """Under-cab wedge safety clamp engagement in [0.0 (open 8mm), 1.0 (locked on rail)]."""
    if f <= 1260.0 or f > float(BRAKE_END):
        return 0.0
    if f < 1265.0:
        return 0.05 * C.smoothstep(1260.0, 1265.0, f)
    if f <= float(BRAKE_CLAMP):
        u = C.smootherstep(1265.0, float(BRAKE_CLAMP), f)
        return 0.05 + 0.95 * (u ** 0.72)
    if f <= 1285.0:
        t = (f - float(BRAKE_CLAMP)) / 10.0
        return 1.0 - 0.025 * math.sin(t * math.pi) * (1.0 - t)
    return 1.0


def alarm_pulse_factor(f: float) -> float:
    """Crimson alarm strobe pulse [0.0..1.0], active f=110..1236 (~2.5 Hz at 60 fps)."""
    if f < float(ALARM_START) or f > 1236.0:
        return 0.0
    phase = ((f - float(ALARM_START)) % 24.0) / 24.0
    if phase < 0.08:
        return C.lerp(0.10, 1.0, phase / 0.08)
    if phase < 0.36:
        return max(0.08, math.exp(-7.5 * (phase - 0.08)))
    return 0.08


def ceiling_glow_factor(f: float) -> float:
    """Ceiling LED intensity factor across Beats 1-6 (brief flicker at f=325 cable snap)."""
    if f < float(CABLE_SNAP):
        return 1.0
    if f <= 355.0:
        flicker = 0.62 + 0.36 * math.sin(f * 2.15) * math.cos(f * 0.95)
        return C.clamp(flicker, 0.25, 1.0)
    if f <= 1236.0:
        return 0.62
    return 0.95


def snap_spark_gain(f: float) -> float:
    """Additive golden-orange cable-snap spark burst gain across f=325..365."""
    if f < float(CABLE_SNAP) or f > 365.0:
        return 0.0
    rise = C.smoothstep(324.5, 326.5, f)
    decay = 1.0 - C.smootherstep(327.0, 365.0, f)
    return C.clamp(rise * (decay ** 1.35), 0.0, 1.0)


def brake_spark_gain(f: float) -> float:
    """Upward additive rail-friction spark gain across f=1270..1365."""
    if f < 1270.0 or f > 1365.0:
        return 0.0
    rise = C.smoothstep(1270.0, 1278.0, f)
    tail = 1.0 - C.smootherstep(1318.0, 1365.0, f)
    flicker = 0.82 + 0.18 * math.sin(f * 1.85)
    return C.clamp(rise * tail * flicker, 0.0, 1.0)


def force_grid_fade(f: float) -> float:
    """Beat 6 floor 12x force-distribution grid fade-in across f=1545..1580."""
    if f < float(GRID_REVEAL):
        return 0.0
    if f <= 1580.0:
        return C.smootherstep(float(GRID_REVEAL), 1580.0, f)
    return 1.0


def floor_led_schedule(total_frames: int) -> list[tuple[str, str, int, int]]:
    """
    Deterministic frame-partitioned header LED readout schedule (18 -> 09 -> 01 + ERR).
    Returns (text, theme, f_start, f_end) covering 1..total_frames with zero gaps.
    """
    t_max = max(1, int(total_frames))
    raw = [
        ("18 ↓", "amber", 1, 45),
        ("17 ↓", "amber", 46, 90),
        ("16 ↓", "amber", 91, 135),
        ("15 ↓", "amber", 136, 185),
        ("14 ↓", "amber", 186, 235),
        ("13 ↓", "amber", 236, 280),
        ("12 ↓", "amber", 281, 324),
        ("ERR", "crimson", 325, 355),
        ("11 ↓", "amber", 356, 385),
        ("10 ↓", "amber", 386, 415),
        ("09 ↓", "amber", 416, 445),
        ("ERR", "crimson", 446, 470),
        ("08 ↓", "amber", 471, 505),
        ("07 ↓", "amber", 506, 540),
        ("06 ↓", "amber", 541, 575),
        ("ERR", "crimson", 576, 600),
        ("05 ↓", "amber", 601, 665),
        ("04 ↓", "amber", 666, 745),
        ("03 ↓", "amber", 746, 820),
        ("02 ↓", "amber", 821, 890),
        ("01 ↓", "crimson", 891, 939),
        ("ERR", "crimson", 940, 1236),
        ("06 ↓", "amber", 1237, 1275),
        ("05 ↓", "amber", 1276, 1320),
        ("STOP", "emerald", 1321, 1385),
        ("01", "emerald", 1386, t_max),
    ]
    out: list[tuple[str, str, int, int]] = []
    for txt, theme, s, e in raw:
        if s > t_max:
            continue
        out.append((txt, theme, s, min(e, t_max)))
    return out


def force_grid_anchor() -> tuple[float, float, float]:
    """Center (x, y, yaw_rad) aligned along PASSENGER['flat_head'] -> PASSENGER['flat_feet']."""
    hx, hy, _ = (float(v) for v in C.PASSENGER["flat_head"])
    fx, fy, _ = (float(v) for v in C.PASSENGER["flat_feet"])
    cx = 0.5 * (hx + fx)
    cy = 0.5 * (hy + fy)
    yaw = math.atan2(fy - hy, fx - hx)
    return cx, cy, yaw


# ---------------------------------------------------------------------------
# BLENDER & BMESH HELPERS (Zero bpy.ops usage)
# ---------------------------------------------------------------------------
def _link_obj(obj: Any, col: Any) -> Any:
    if obj.name not in col.objects:
        col.objects.link(obj)
    return obj


def _new_empty(
    name: str,
    col: Any,
    parent: Any | None = None,
    location: tuple[float, float, float] | Any = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float] | Any = (0.0, 0.0, 0.0),
    display_size: float = 0.12,
) -> Any:
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
    bm: Any,
    materials: list[Any] | Any,
    col: Any,
    parent: Any | None = None,
    location: tuple[float, float, float] | Any = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float] | Any = (0.0, 0.0, 0.0),
    smooth_all: bool | None = True,
    visible_shadow: bool = True,
    recalc_normals: bool = True,
) -> Any:
    """Attach material slots BEFORE bm.to_mesh(mesh) per AGENTS.md Section 3."""
    mesh = bpy.data.meshes.new(name)
    mat_list = materials if isinstance(materials, list) else [materials]
    for mat in mat_list:
        mesh.materials.append(mat)

    if recalc_normals and len(bm.faces) > 0:
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
    obj.visible_shadow = visible_shadow
    return _link_obj(obj, col)


def _merge_bm(dst: Any, src: Any, mat_index: int | None = None) -> None:
    if mat_index is not None:
        for f in src.faces:
            f.material_index = mat_index
    tmp = bpy.data.meshes.new("_cab_tmp_merge")
    src.to_mesh(tmp)
    src.free()
    dst.from_mesh(tmp)
    bpy.data.meshes.remove(tmp)


def _add_box_to_bm(
    bm: Any,
    center: tuple[float, float, float],
    dims: tuple[float, float, float],
    bevel_w: float = 0.0,
    segments: int = 2,
    mat_index: int = 0,
    smooth: bool = False,
) -> None:
    sub = bmesh.new()
    mat = Matrix.Translation(center) @ Matrix.Diagonal((dims[0], dims[1], dims[2], 1.0))
    bmesh.ops.create_cube(sub, size=1.0, matrix=mat)
    if bevel_w > 0.0:
        bmesh.ops.bevel(sub, geom=list(sub.edges), offset=bevel_w, segments=segments, affect="EDGES")
    for f in sub.faces:
        f.material_index = mat_index
        f.smooth = smooth
    _merge_bm(bm, sub)


def _add_cyl_to_bm(
    bm: Any,
    center: tuple[float, float, float],
    radius: float,
    depth: float,
    segs: int = 16,
    axis: str = "Z",
    mat_index: int = 0,
    cap_ends: bool = True,
) -> None:
    sub = bmesh.new()
    rot = Matrix.Identity(4)
    if axis == "X":
        rot = Matrix.Rotation(math.pi * 0.5, 4, "Y")
    elif axis == "Y":
        rot = Matrix.Rotation(math.pi * 0.5, 4, "X")
    xf = Matrix.Translation(center) @ rot
    bmesh.ops.create_cone(
        sub,
        cap_ends=cap_ends,
        cap_tris=False,
        segments=segs,
        radius1=radius,
        radius2=radius,
        depth=depth,
        matrix=xf,
    )
    for f in sub.faces:
        f.material_index = mat_index
        f.smooth = True
    _merge_bm(bm, sub)


def _add_icosphere_bm(
    bm: Any,
    center: tuple[float, float, float] | Any,
    radius: float,
    subdivisions: int = 1,
    mat_index: int = 0,
) -> None:
    sub = bmesh.new()
    bmesh.ops.create_icosphere(
        sub,
        subdivisions=subdivisions,
        radius=radius,
        matrix=Matrix.Translation(center),
    )
    for f in sub.faces:
        f.material_index = mat_index
        f.smooth = True
    _merge_bm(bm, sub)


def _add_tube_along_path_bm(
    bm: Any,
    points: list[Any],
    r_start: float,
    r_end: float | None = None,
    n_seg: int = 12,
    cap_ends: bool = True,
    mat_index: int = 0,
) -> None:
    if len(points) < 2:
        return
    r1 = r_start if r_end is None else r_end
    rings: list[list[Any]] = []
    up_ref = Vector((0.0, 1.0, 0.0))

    for idx, pt in enumerate(points):
        u = idx / max(1, len(points) - 1)
        rad = C.lerp(r_start, r1, u)
        if idx == 0:
            tang = (points[1] - points[0]).normalized()
        elif idx == len(points) - 1:
            tang = (points[-1] - points[-2]).normalized()
        else:
            t1 = (points[idx] - points[idx - 1]).normalized()
            t2 = (points[idx + 1] - points[idx]).normalized()
            tang = (t1 + t2).normalized() if (t1 + t2).length > 1e-6 else t1
        ref = up_ref if abs(tang.dot(up_ref)) < 0.90 else Vector((1.0, 0.0, 0.0))
        b1 = tang.cross(ref).normalized()
        b2 = tang.cross(b1).normalized()
        ring = []
        for k in range(n_seg):
            a = k * 2.0 * math.pi / n_seg
            ring.append(bm.verts.new(pt + (b1 * math.cos(a) + b2 * math.sin(a)) * rad))
        rings.append(ring)

    for i in range(len(rings) - 1):
        for k in range(n_seg):
            kn = (k + 1) % n_seg
            f = bm.faces.new((rings[i][k], rings[i][kn], rings[i + 1][kn], rings[i + 1][k]))
            f.material_index = mat_index
            f.smooth = True

    if cap_ends:
        f0 = bm.faces.new(rings[0][::-1])
        f0.material_index = mat_index
        f1 = bm.faces.new(rings[-1])
        f1.material_index = mat_index


def _add_open_ribbon_bm(
    bm: Any,
    points: list[Any],
    w_start: float,
    w_end: float,
    normal_hint: Any,
    mat_index: int = 0,
) -> None:
    """Open-ended tapered quad ribbon (end_fill_type=NOTHING) per AGENTS.md Section 3."""
    if len(points) < 2:
        return
    rows: list[tuple[Any, Any]] = []
    for idx, pt in enumerate(points):
        u = idx / max(1, len(points) - 1)
        hw = 0.5 * C.lerp(w_start, w_end, u)
        if idx == 0:
            tang = (points[1] - points[0]).normalized()
        elif idx == len(points) - 1:
            tang = (points[-1] - points[-2]).normalized()
        else:
            tang = (points[idx + 1] - points[idx - 1]).normalized()
        side = tang.cross(normal_hint)
        if side.length < 1e-5:
            side = tang.cross(Vector((0.0, 1.0, 0.0)))
        if side.length < 1e-5:
            side = tang.cross(Vector((1.0, 0.0, 0.0)))
        side.normalize()
        rows.append((bm.verts.new(pt - side * hw), bm.verts.new(pt + side * hw)))

    for i in range(len(rows) - 1):
        f = bm.faces.new((rows[i][0], rows[i][1], rows[i + 1][1], rows[i + 1][0]))
        f.material_index = mat_index
        f.smooth = True


def _set_interp(id_block: Any, interp: str, data_path_prefix: str = "") -> None:
    ad = getattr(id_block, "animation_data", None)
    if not ad or not ad.action:
        return
    for fc in C.iter_action_fcurves(ad.action):
        if data_path_prefix and not fc.data_path.startswith(data_path_prefix):
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = interp


def _key_visibility(
    obj: Any,
    visible_ranges: list[tuple[int, int]],
    total_frames: int,
) -> None:
    """Keyframe CONSTANT hide_viewport & hide_render so hidden/detached parts never reappear."""
    t_max = max(1, int(total_frames))
    if not visible_ranges:
        for f in (1, t_max):
            obj.hide_viewport = True
            obj.hide_render = True
            obj.keyframe_insert("hide_viewport", frame=f)
            obj.keyframe_insert("hide_render", frame=f)
        _set_interp(obj, "CONSTANT", "hide_")
        return

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
    _set_interp(obj, "CONSTANT", "hide_")


def _key_value_node(node: Any, frames: list[int], eval_fn: Any, interp: str = "LINEAR") -> None:
    sock = node.outputs[0]
    for f in frames:
        sock.default_value = float(eval_fn(float(f)))
        sock.keyframe_insert("default_value", frame=int(f))
    mat = node.id_data
    ad = getattr(mat, "animation_data", None)
    if ad and ad.action:
        for fc in C.iter_action_fcurves(ad.action):
            if node.name in fc.data_path:
                for kp in fc.keyframe_points:
                    kp.interpolation = interp


def _sample_frames(
    total_frames: int,
    dense_windows: list[tuple[int, int]] | None = None,
    step: int = 3,
) -> list[int]:
    t_max = max(1, int(total_frames))
    frames = set(range(1, t_max + 1, step))
    frames.update((1, t_max))
    for ev_f in C.EV.values():
        if 1 <= ev_f <= t_max:
            frames.add(ev_f)
    for b0, b1 in C.BEATS.values():
        for b in (b0, b1):
            if 1 <= b <= t_max:
                frames.add(b)
    if dense_windows:
        for s, e in dense_windows:
            for f in range(max(1, s), min(t_max, e) + 1):
                frames.add(f)
    return sorted(frames)


# ---------------------------------------------------------------------------
# LOCAL-SPACE SHADERS & MATERIALS (AGENTS.md V11)
# ---------------------------------------------------------------------------
def _wire_camera_front_mask(nt: Any) -> Any:
    """Return Camera Ray * (1 - Backfacing) scalar socket."""
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
    return cam_front.outputs["Value"]


def _wire_shadow_bypass(
    nt: Any,
    surface_shader_sock: Any,
    transp_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> Any:
    """Route LightPath.Is Shadow Ray -> Transparent BSDF per AGENTS.md Section 3."""
    lp = nt.nodes.new("ShaderNodeLightPath")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    tr.inputs["Color"].default_value = transp_color

    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Shadow Ray"], mix.inputs["Fac"])
    nt.links.new(surface_shader_sock, mix.inputs[1])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[2])
    return mix.outputs["Shader"]


def _build_materials(cab_root: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    mats: dict[str, Any] = {}
    ctrl: dict[str, Any] = {"xray_nodes": []}

    # 1. Dark Slate-Terrazzo Floor Slab (TexCoord.Object -> Cab_Root)
    m_floor = bpy.data.materials.new(f"{PREFIX}Mat_Floor_Terrazzo")
    m_floor.use_nodes = True
    nt = m_floor.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Metallic"].default_value = 0.06
    bsdf.inputs["Roughness"].default_value = 0.28

    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = cab_root
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 55.0
    nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])

    nf = nt.nodes.new("ShaderNodeTexNoise")
    nf.inputs["Scale"].default_value = 18.0
    nf.inputs["Detail"].default_value = 6.0
    nt.links.new(tc.outputs["Object"], nf.inputs["Vector"])

    speck = nt.nodes.new("ShaderNodeMapRange")
    speck.clamp = True
    speck.inputs["From Min"].default_value = 0.20
    speck.inputs["From Max"].default_value = 0.30
    speck.inputs["To Min"].default_value = 1.0
    speck.inputs["To Max"].default_value = 0.0
    nt.links.new(vor.outputs["Distance"], speck.inputs["Value"])

    col_mix = nt.nodes.new("ShaderNodeMix")
    col_mix.data_type = "RGBA"
    col_mix.inputs[6].default_value = (0.028, 0.032, 0.040, 1.0)
    col_mix.inputs[7].default_value = (0.085, 0.095, 0.112, 1.0)
    nt.links.new(speck.outputs["Result"], col_mix.inputs["Factor"])
    nt.links.new(col_mix.outputs[2], bsdf.inputs["Base Color"])

    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.05
    bmp.inputs["Distance"].default_value = 0.0004
    nt.links.new(nf.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    mats["floor_terrazzo"] = m_floor

    # 2. Solid Structural Metals & Walnut Flutes (TexCoord.Object -> Cab_Root)
    def make_metal_mat(
        name: str,
        base_col: tuple[float, float, float],
        metallic: float,
        roughness: float,
        noise_scale: tuple[float, float, float] = (14.0, 90.0, 90.0),
        bump_str: float = 0.07,
    ) -> Any:
        mat = bpy.data.materials.new(f"{PREFIX}Mat_{name}")
        mat.use_nodes = True
        nt_m = mat.node_tree
        bs = nt_m.nodes["Principled BSDF"]
        bs.inputs["Base Color"].default_value = (*base_col, 1.0)
        bs.inputs["Metallic"].default_value = metallic
        bs.inputs["Roughness"].default_value = roughness
        tc_m = nt_m.nodes.new("ShaderNodeTexCoord")
        tc_m.object = cab_root
        mp = nt_m.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = noise_scale
        nt_m.links.new(tc_m.outputs["Object"], mp.inputs["Vector"])
        nf_m = nt_m.nodes.new("ShaderNodeTexNoise")
        nf_m.inputs["Scale"].default_value = 6.0
        nf_m.inputs["Detail"].default_value = 6.0
        nt_m.links.new(mp.outputs["Vector"], nf_m.inputs["Vector"])
        bm_m = nt_m.nodes.new("ShaderNodeBump")
        bm_m.inputs["Strength"].default_value = bump_str
        bm_m.inputs["Distance"].default_value = 0.0004
        nt_m.links.new(nf_m.outputs["Fac"], bm_m.inputs["Height"])
        nt_m.links.new(bm_m.outputs["Normal"], bs.inputs["Normal"])
        return mat

    mats["dark_steel"] = make_metal_mat("Dark_Steel", (0.048, 0.054, 0.065), 0.90, 0.34)
    mats["brushed_steel"] = make_metal_mat("Brushed_Steel", (0.68, 0.72, 0.76), 0.96, 0.22, (8.0, 140.0, 140.0), 0.05)
    mats["clamp_steel"] = make_metal_mat("Clamp_Steel", (0.24, 0.26, 0.30), 0.95, 0.24, (24.0, 60.0, 24.0), 0.09)
    mats["obsidian_trim"] = make_metal_mat("Obsidian_Trim", (0.012, 0.015, 0.020), 0.35, 0.40, (20.0, 20.0, 20.0), 0.04)

    # Rich Walnut / Charcoal Fluted Rear Panel
    m_walnut = bpy.data.materials.new(f"{PREFIX}Mat_Walnut_Fluted")
    m_walnut.use_nodes = True
    nt = m_walnut.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Metallic"].default_value = 0.02
    bsdf.inputs["Roughness"].default_value = 0.36
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = cab_root
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (18.0, 18.0, 1.8)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nf = nt.nodes.new("ShaderNodeTexNoise")
    nf.inputs["Scale"].default_value = 12.0
    nf.inputs["Detail"].default_value = 8.0
    nt.links.new(mp.outputs["Vector"], nf.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.25
    ramp.color_ramp.elements[0].color = (0.042, 0.024, 0.015, 1.0)
    ramp.color_ramp.elements[1].position = 0.78
    ramp.color_ramp.elements[1].color = (0.095, 0.054, 0.032, 1.0)
    nt.links.new(nf.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.09
    bmp.inputs["Distance"].default_value = 0.0005
    nt.links.new(nf.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    mats["walnut_fluted"] = m_walnut

    # Safety Amber-Gold Coil Spring & Linkage
    m_spring = bpy.data.materials.new(f"{PREFIX}Mat_Clamp_Spring")
    m_spring.use_nodes = True
    bsdf = m_spring.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.95, 0.58, 0.04, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.45
    bsdf.inputs["Roughness"].default_value = 0.28
    bsdf.inputs["Emission Color"].default_value = (0.95, 0.58, 0.04, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 0.12
    mats["clamp_spring"] = m_spring

    # 3. Low-Glare Crystal-Clear Front Tempered Glass + Local-Space XRayCab (<= 0.08 Fresnel cap!)
    m_glass = bpy.data.materials.new(f"{PREFIX}Mat_Front_Glass")
    m_glass.use_nodes = True
    nt = m_glass.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    xray_g = nt.nodes.new("ShaderNodeValue")
    xray_g.name = "XRayCab"
    xray_g.outputs[0].default_value = 0.0
    ctrl["xray_nodes"].append(xray_g)

    tr_g = nt.nodes.new("ShaderNodeBsdfTransparent")
    tr_g.inputs["Color"].default_value = (0.992, 0.996, 1.0, 1.0)

    gl_g = nt.nodes.new("ShaderNodeBsdfGlossy")
    gl_g.inputs["Color"].default_value = (0.94, 0.98, 1.0, 1.0)
    gl_g.inputs["Roughness"].default_value = 0.020

    lw_g = nt.nodes.new("ShaderNodeLayerWeight")
    lw_g.inputs["Blend"].default_value = 0.20

    cam_front_g = _wire_camera_front_mask(nt)

    # Cap reflection strictly <= 0.08 and fade down during X-ray mode
    inv_xr = nt.nodes.new("ShaderNodeMath")
    inv_xr.operation = "SUBTRACT"
    inv_xr.inputs[0].default_value = 1.0
    nt.links.new(xray_g.outputs[0], inv_xr.inputs[1])

    fres_cap = nt.nodes.new("ShaderNodeMath")
    fres_cap.operation = "MULTIPLY"
    fres_cap.inputs[1].default_value = 0.08
    nt.links.new(lw_g.outputs["Fresnel"], fres_cap.inputs[0])

    refl_m1 = nt.nodes.new("ShaderNodeMath")
    refl_m1.operation = "MULTIPLY"
    nt.links.new(fres_cap.outputs["Value"], refl_m1.inputs[0])
    nt.links.new(cam_front_g, refl_m1.inputs[1])

    refl_fac = nt.nodes.new("ShaderNodeMath")
    refl_fac.operation = "MULTIPLY"
    nt.links.new(refl_m1.outputs["Value"], refl_fac.inputs[0])
    nt.links.new(inv_xr.outputs["Value"], refl_fac.inputs[1])

    mix_g = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(refl_fac.outputs["Value"], mix_g.inputs["Fac"])
    nt.links.new(tr_g.outputs["BSDF"], mix_g.inputs[1])
    nt.links.new(gl_g.outputs["BSDF"], mix_g.inputs[2])

    # Subtle cyan X-ray glass grazing rim ONLY (<= 0.14 AgX-safe, crystal clear face)
    inv_f_g = nt.nodes.new("ShaderNodeMath")
    inv_f_g.operation = "SUBTRACT"
    inv_f_g.inputs[0].default_value = 1.0
    nt.links.new(lw_g.outputs["Facing"], inv_f_g.inputs[1])

    pow_g = nt.nodes.new("ShaderNodeMath")
    pow_g.operation = "POWER"
    pow_g.inputs[1].default_value = 3.6
    nt.links.new(inv_f_g.outputs["Value"], pow_g.inputs[0])

    xr_em_m1 = nt.nodes.new("ShaderNodeMath")
    xr_em_m1.operation = "MULTIPLY"
    nt.links.new(pow_g.outputs["Value"], xr_em_m1.inputs[0])
    nt.links.new(xray_g.outputs[0], xr_em_m1.inputs[1])

    xr_em_m2 = nt.nodes.new("ShaderNodeMath")
    xr_em_m2.operation = "MULTIPLY"
    nt.links.new(xr_em_m1.outputs["Value"], xr_em_m2.inputs[0])
    nt.links.new(cam_front_g, xr_em_m2.inputs[1])

    xr_em_str = nt.nodes.new("ShaderNodeMath")
    xr_em_str.operation = "MULTIPLY"
    xr_em_str.inputs[1].default_value = 0.14
    nt.links.new(xr_em_m2.outputs["Value"], xr_em_str.inputs[0])

    em_g = nt.nodes.new("ShaderNodeEmission")
    em_g.inputs["Color"].default_value = (0.14, 0.88, 1.0, 1.0)
    nt.links.new(xr_em_str.outputs["Value"], em_g.inputs["Strength"])

    add_g = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(mix_g.outputs["Shader"], add_g.inputs[0])
    nt.links.new(em_g.outputs["Emission"], add_g.inputs[1])

    surf_g = _wire_shadow_bypass(nt, add_g.outputs["Shader"], (1.0, 1.0, 1.0, 1.0))
    nt.links.new(surf_g, out.inputs["Surface"])
    mats["front_glass"] = m_glass

    # 4. Local-Space X-Ray Structural Walls & Door Mullions (`TexCoord.Object` -> Cab_Root)
    def make_xray_structural_mat(
        name: str,
        base_col: tuple[float, float, float],
        metallic: float,
        roughness: float,
        edge_emit: float,
    ) -> Any:
        mat = bpy.data.materials.new(f"{PREFIX}Mat_{name}")
        mat.use_nodes = True
        nt_x = mat.node_tree
        nt_x.nodes.clear()
        out_x = nt_x.nodes.new("ShaderNodeOutputMaterial")

        xray_v = nt_x.nodes.new("ShaderNodeValue")
        xray_v.name = "XRayCab"
        xray_v.outputs[0].default_value = 0.0
        ctrl["xray_nodes"].append(xray_v)

        tc_x = nt_x.nodes.new("ShaderNodeTexCoord")
        tc_x.object = cab_root

        nf_x = nt_x.nodes.new("ShaderNodeTexNoise")
        nf_x.inputs["Scale"].default_value = 32.0
        nf_x.inputs["Detail"].default_value = 6.0
        nt_x.links.new(tc_x.outputs["Object"], nf_x.inputs["Vector"])

        bmp_x = nt_x.nodes.new("ShaderNodeBump")
        bmp_x.inputs["Strength"].default_value = 0.06
        bmp_x.inputs["Distance"].default_value = 0.0004
        nt_x.links.new(nf_x.outputs["Fac"], bmp_x.inputs["Height"])

        bs_x = nt_x.nodes.new("ShaderNodeBsdfPrincipled")
        bs_x.inputs["Base Color"].default_value = (*base_col, 1.0)
        bs_x.inputs["Metallic"].default_value = metallic
        bs_x.inputs["Roughness"].default_value = roughness
        nt_x.links.new(bmp_x.outputs["Normal"], bs_x.inputs["Normal"])

        # Holographic cyan edge + local-space Z scanline grid
        lw_x = nt_x.nodes.new("ShaderNodeLayerWeight")
        lw_x.inputs["Blend"].default_value = 0.45
        inv_f_x = nt_x.nodes.new("ShaderNodeMath")
        inv_f_x.operation = "SUBTRACT"
        inv_f_x.inputs[0].default_value = 1.0
        nt_x.links.new(lw_x.outputs["Facing"], inv_f_x.inputs[1])

        pow_x = nt_x.nodes.new("ShaderNodeMath")
        pow_x.operation = "POWER"
        pow_x.inputs[1].default_value = 3.2
        nt_x.links.new(inv_f_x.outputs["Value"], pow_x.inputs[0])

        wave_x = nt_x.nodes.new("ShaderNodeTexWave")
        wave_x.wave_type = "BANDS"
        wave_x.bands_direction = "Z"
        wave_x.wave_profile = "SIN"
        wave_x.inputs["Scale"].default_value = 18.0
        nt_x.links.new(tc_x.outputs["Object"], wave_x.inputs["Vector"])

        scan_line = nt_x.nodes.new("ShaderNodeMapRange")
        scan_line.clamp = True
        scan_line.inputs["From Min"].default_value = 0.88
        scan_line.inputs["From Max"].default_value = 0.99
        scan_line.inputs["To Min"].default_value = 0.0
        scan_line.inputs["To Max"].default_value = 0.08
        nt_x.links.new(wave_x.outputs["Fac"], scan_line.inputs["Value"])

        holo_add = nt_x.nodes.new("ShaderNodeMath")
        holo_add.operation = "ADD"
        holo_add.use_clamp = True
        nt_x.links.new(pow_x.outputs["Value"], holo_add.inputs[0])
        nt_x.links.new(scan_line.outputs["Result"], holo_add.inputs[1])

        cam_front_x = _wire_camera_front_mask(nt_x)
        holo_cam = nt_x.nodes.new("ShaderNodeMath")
        holo_cam.operation = "MULTIPLY"
        nt_x.links.new(holo_add.outputs["Value"], holo_cam.inputs[0])
        nt_x.links.new(cam_front_x, holo_cam.inputs[1])

        holo_gain = nt_x.nodes.new("ShaderNodeMath")
        holo_gain.operation = "MULTIPLY"
        holo_gain.inputs[1].default_value = min(0.55, float(edge_emit))
        nt_x.links.new(holo_cam.outputs["Value"], holo_gain.inputs[0])

        tr_x = nt_x.nodes.new("ShaderNodeBsdfTransparent")
        tr_x.inputs["Color"].default_value = (0.992, 0.996, 1.0, 1.0)

        em_x = nt_x.nodes.new("ShaderNodeEmission")
        em_x.inputs["Color"].default_value = (0.14, 0.88, 1.0, 1.0)
        nt_x.links.new(holo_gain.outputs["Value"], em_x.inputs["Strength"])

        xray_shell = nt_x.nodes.new("ShaderNodeAddShader")
        nt_x.links.new(tr_x.outputs["BSDF"], xray_shell.inputs[0])
        nt_x.links.new(em_x.outputs["Emission"], xray_shell.inputs[1])

        mix_x = nt_x.nodes.new("ShaderNodeMixShader")
        nt_x.links.new(xray_v.outputs[0], mix_x.inputs["Fac"])
        nt_x.links.new(bs_x.outputs["BSDF"], mix_x.inputs[1])
        nt_x.links.new(xray_shell.outputs["Shader"], mix_x.inputs[2])

        surf_x = _wire_shadow_bypass(nt_x, mix_x.outputs["Shader"], (0.992, 0.996, 1.0, 1.0))
        nt_x.links.new(surf_x, out_x.inputs["Surface"])
        return mat

    mats["xray_frame"] = make_xray_structural_mat("XRay_Frame", (0.42, 0.45, 0.50), 0.92, 0.25, 0.55)
    mats["xray_panel"] = make_xray_structural_mat("XRay_Panel", (0.068, 0.075, 0.088), 0.72, 0.34, 0.24)

    # 5. Recessed Warm Ceiling LED Slats & Flashing Crimson Alarm Lens (AgX-safe <= 2.05)
    m_ceil_led = bpy.data.materials.new(f"{PREFIX}Mat_Ceiling_LED")
    m_ceil_led.use_nodes = True
    nt = m_ceil_led.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (1.0, 0.94, 0.82, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.20
    bsdf.inputs["Emission Color"].default_value = (1.0, 0.90, 0.74, 1.0)
    ceil_val = nt.nodes.new("ShaderNodeValue")
    ceil_val.name = "CeilingGlow"
    ceil_val.outputs[0].default_value = 1.55
    ctrl["ceiling_glow"] = ceil_val
    nt.links.new(ceil_val.outputs[0], bsdf.inputs["Emission Strength"])
    mats["ceiling_led"] = m_ceil_led

    m_alarm = bpy.data.materials.new(f"{PREFIX}Mat_Alarm_Lens")
    m_alarm.use_nodes = True
    nt = m_alarm.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.45, 0.03, 0.02, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.14
    bsdf.inputs["Emission Color"].default_value = (1.0, 0.08, 0.04, 1.0)
    alarm_val = nt.nodes.new("ShaderNodeValue")
    alarm_val.name = "AlarmGain"
    alarm_val.outputs[0].default_value = 0.0
    ctrl["alarm_gain"] = alarm_val
    nt.links.new(alarm_val.outputs[0], bsdf.inputs["Emission Strength"])
    mats["alarm_lens"] = m_alarm

    # 6. Digital Header LED & COP Button Materials (AgX-safe <= 1.88)
    def make_emissive_mat(name: str, col_rgb: tuple[float, float, float], strength: float) -> Any:
        mat = bpy.data.materials.new(f"{PREFIX}Mat_{name}")
        mat.use_nodes = True
        bs = mat.node_tree.nodes["Principled BSDF"]
        bs.inputs["Base Color"].default_value = (*col_rgb, 1.0)
        bs.inputs["Roughness"].default_value = 0.26
        bs.inputs["Emission Color"].default_value = (*col_rgb, 1.0)
        bs.inputs["Emission Strength"].default_value = min(2.05, float(strength))
        return mat

    mats["led_amber"] = make_emissive_mat("LED_Amber", (1.0, 0.72, 0.14), 1.78)
    mats["led_crimson"] = make_emissive_mat("LED_Crimson", (1.0, 0.12, 0.08), 1.88)
    mats["led_emerald"] = make_emissive_mat("LED_Emerald", (0.14, 0.94, 0.56), 1.72)
    mats["cop_btn_warm"] = make_emissive_mat("COP_Btn_Warm", (1.0, 0.84, 0.55), 1.25)

    # 7. Braided Galvanized Steel Hoist Cables (self-local helical strand bump)
    m_cable = bpy.data.materials.new(f"{PREFIX}Mat_Hoist_Cable")
    m_cable.use_nodes = True
    nt = m_cable.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.30, 0.32, 0.36, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.94
    bsdf.inputs["Roughness"].default_value = 0.30
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Rotation"].default_value = (0.0, 0.42, 0.0)
    mp.inputs["Scale"].default_value = (85.0, 85.0, 12.0)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "DIAGONAL"
    wave.inputs["Scale"].default_value = 6.0
    nt.links.new(mp.outputs["Vector"], wave.inputs["Vector"])
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.35
    bmp.inputs["Distance"].default_value = 0.0012
    nt.links.new(wave.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    mats["hoist_cable"] = m_cable

    # 8. Open-Ended Additive Spark Materials (Snap Sparks & Brake Friction Sparks, <= 1.95)
    def make_spark_mat(name: str, gain_key: str, c_edge: tuple[float, float, float], c_core: tuple[float, float, float]) -> Any:
        mat = bpy.data.materials.new(f"{PREFIX}Mat_{name}")
        mat.use_nodes = True
        nt_s = mat.node_tree
        nt_s.nodes.clear()
        out_s = nt_s.nodes.new("ShaderNodeOutputMaterial")

        tr_s = nt_s.nodes.new("ShaderNodeBsdfTransparent")
        em_s = nt_s.nodes.new("ShaderNodeEmission")

        tc_s = nt_s.nodes.new("ShaderNodeTexCoord")
        tc_s.object = cab_root
        nf_s = nt_s.nodes.new("ShaderNodeTexNoise")
        nf_s.inputs["Scale"].default_value = 26.0
        nf_s.inputs["Detail"].default_value = 6.0
        nt_s.links.new(tc_s.outputs["Object"], nf_s.inputs["Vector"])

        n_map = nt_s.nodes.new("ShaderNodeMapRange")
        n_map.clamp = True
        n_map.inputs["From Min"].default_value = 0.32
        n_map.inputs["From Max"].default_value = 0.78
        n_map.inputs["To Min"].default_value = 0.30
        n_map.inputs["To Max"].default_value = 1.0
        nt_s.links.new(nf_s.outputs["Fac"], n_map.inputs["Value"])

        ramp_s = nt_s.nodes.new("ShaderNodeValToRGB")
        ramp_s.color_ramp.elements[0].position = 0.25
        ramp_s.color_ramp.elements[0].color = (*c_edge, 1.0)
        ramp_s.color_ramp.elements[1].position = 0.85
        ramp_s.color_ramp.elements[1].color = (*c_core, 1.0)
        nt_s.links.new(n_map.outputs["Result"], ramp_s.inputs["Fac"])
        nt_s.links.new(ramp_s.outputs["Color"], em_s.inputs["Color"])

        gain_v = nt_s.nodes.new("ShaderNodeValue")
        gain_v.name = gain_key
        gain_v.outputs[0].default_value = 0.0
        ctrl[gain_key] = gain_v

        cam_front_s = _wire_camera_front_mask(nt_s)
        m1 = nt_s.nodes.new("ShaderNodeMath")
        m1.operation = "MULTIPLY"
        nt_s.links.new(n_map.outputs["Result"], m1.inputs[0])
        nt_s.links.new(gain_v.outputs[0], m1.inputs[1])

        m2 = nt_s.nodes.new("ShaderNodeMath")
        m2.operation = "MULTIPLY"
        nt_s.links.new(m1.outputs["Value"], m2.inputs[0])
        nt_s.links.new(cam_front_s, m2.inputs[1])

        m3 = nt_s.nodes.new("ShaderNodeMath")
        m3.operation = "MULTIPLY"
        m3.inputs[1].default_value = 1.90  # Strictly <= 1.90 AgX-safe
        nt_s.links.new(m2.outputs["Value"], m3.inputs[0])
        nt_s.links.new(m3.outputs["Value"], em_s.inputs["Strength"])

        add_s = nt_s.nodes.new("ShaderNodeAddShader")
        nt_s.links.new(tr_s.outputs["BSDF"], add_s.inputs[0])
        nt_s.links.new(em_s.outputs["Emission"], add_s.inputs[1])
        nt_s.links.new(add_s.outputs["Shader"], out_s.inputs["Surface"])
        return mat

    mats["snap_sparks"] = make_spark_mat("Snap_Sparks", "snap_spark_gain", (1.0, 0.36, 0.04), (1.0, 0.84, 0.28))
    mats["brake_sparks"] = make_spark_mat("Brake_Sparks", "brake_spark_gain", (1.0, 0.32, 0.03), (1.0, 0.80, 0.22))

    # 9. Beat 6 Flush 12x Force-Distribution Floor Grid (`TexCoord.Object` self-local, <= 1.80)
    m_grid = bpy.data.materials.new(f"{PREFIX}Mat_Force_Grid")
    m_grid.use_nodes = True
    nt = m_grid.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    grid_fade = nt.nodes.new("ShaderNodeValue")
    grid_fade.name = "GridFade"
    grid_fade.outputs[0].default_value = 0.0
    ctrl["grid_fade"] = grid_fade

    # Self-local object coords: X in [-0.78, +0.78], Y in [-0.31, +0.31]
    tc_f = nt.nodes.new("ShaderNodeTexCoord")
    sep_f = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc_f.outputs["Object"], sep_f.inputs["Vector"])

    # 12 longitudinal cells x 5 transverse cells
    map_u = nt.nodes.new("ShaderNodeMapRange")
    map_u.clamp = False
    map_u.inputs["From Min"].default_value = -0.78
    map_u.inputs["From Max"].default_value = 0.78
    map_u.inputs["To Min"].default_value = 0.0
    map_u.inputs["To Max"].default_value = 12.0
    nt.links.new(sep_f.outputs["X"], map_u.inputs["Value"])

    map_v = nt.nodes.new("ShaderNodeMapRange")
    map_v.clamp = False
    map_v.inputs["From Min"].default_value = -0.31
    map_v.inputs["From Max"].default_value = 0.31
    map_v.inputs["To Min"].default_value = 0.0
    map_v.inputs["To Max"].default_value = 5.0
    nt.links.new(sep_f.outputs["Y"], map_v.inputs["Value"])

    def _grid_line_mask(coord_sock: Any, line_w: float) -> Any:
        frac = nt.nodes.new("ShaderNodeMath")
        frac.operation = "FRACT"
        nt.links.new(coord_sock, frac.inputs[0])
        one_minus = nt.nodes.new("ShaderNodeMath")
        one_minus.operation = "SUBTRACT"
        one_minus.inputs[0].default_value = 1.0
        nt.links.new(frac.outputs["Value"], one_minus.inputs[1])
        dist = nt.nodes.new("ShaderNodeMath")
        dist.operation = "MINIMUM"
        nt.links.new(frac.outputs["Value"], dist.inputs[0])
        nt.links.new(one_minus.outputs["Value"], dist.inputs[1])
        ln = nt.nodes.new("ShaderNodeMapRange")
        ln.clamp = True
        ln.inputs["From Min"].default_value = line_w
        ln.inputs["From Max"].default_value = 0.0
        nt.links.new(dist.outputs["Value"], ln.inputs["Value"])
        return ln.outputs["Result"]

    line_u = _grid_line_mask(map_u.outputs["Result"], 0.065)
    line_v = _grid_line_mask(map_v.outputs["Result"], 0.055)
    lines = nt.nodes.new("ShaderNodeMath")
    lines.operation = "MAXIMUM"
    nt.links.new(line_u, lines.inputs[0])
    nt.links.new(line_v, lines.inputs[1])

    # Soft elliptical support-load fill across entire lying body footprint
    nx = nt.nodes.new("ShaderNodeMath")
    nx.operation = "DIVIDE"
    nx.inputs[1].default_value = 0.72
    nt.links.new(sep_f.outputs["X"], nx.inputs[0])
    nx2 = nt.nodes.new("ShaderNodeMath")
    nx2.operation = "POWER"
    nx2.inputs[1].default_value = 2.0
    nt.links.new(nx.outputs["Value"], nx2.inputs[0])

    ny = nt.nodes.new("ShaderNodeMath")
    ny.operation = "DIVIDE"
    ny.inputs[1].default_value = 0.27
    nt.links.new(sep_f.outputs["Y"], ny.inputs[0])
    ny2 = nt.nodes.new("ShaderNodeMath")
    ny2.operation = "POWER"
    ny2.inputs[1].default_value = 2.0
    nt.links.new(ny.outputs["Value"], ny2.inputs[0])

    rad2 = nt.nodes.new("ShaderNodeMath")
    rad2.operation = "ADD"
    nt.links.new(nx2.outputs["Value"], rad2.inputs[0])
    nt.links.new(ny2.outputs["Value"], rad2.inputs[1])

    body_fill = nt.nodes.new("ShaderNodeMapRange")
    body_fill.clamp = True
    body_fill.inputs["From Min"].default_value = 1.08
    body_fill.inputs["From Max"].default_value = 0.15
    body_fill.inputs["To Min"].default_value = 0.18
    body_fill.inputs["To Max"].default_value = 0.85
    nt.links.new(rad2.outputs["Value"], body_fill.inputs["Value"])

    # Outer border glow
    abs_x = nt.nodes.new("ShaderNodeMath")
    abs_x.operation = "ABSOLUTE"
    nt.links.new(sep_f.outputs["X"], abs_x.inputs[0])
    bx = nt.nodes.new("ShaderNodeMapRange")
    bx.clamp = True
    bx.inputs["From Min"].default_value = 0.74
    bx.inputs["From Max"].default_value = 0.78
    nt.links.new(abs_x.outputs["Value"], bx.inputs["Value"])

    abs_y = nt.nodes.new("ShaderNodeMath")
    abs_y.operation = "ABSOLUTE"
    nt.links.new(sep_f.outputs["Y"], abs_y.inputs[0])
    by = nt.nodes.new("ShaderNodeMapRange")
    by.clamp = True
    by.inputs["From Min"].default_value = 0.285
    by.inputs["From Max"].default_value = 0.31
    nt.links.new(abs_y.outputs["Value"], by.inputs[0])

    border = nt.nodes.new("ShaderNodeMath")
    border.operation = "MAXIMUM"
    nt.links.new(bx.outputs["Result"], border.inputs[0])
    nt.links.new(by.outputs["Result"], border.inputs[1])

    hi_mask = nt.nodes.new("ShaderNodeMath")
    hi_mask.operation = "MAXIMUM"
    nt.links.new(lines.outputs["Value"], hi_mask.inputs[0])
    nt.links.new(border.outputs["Value"], hi_mask.inputs[1])

    # Color: emerald body cells -> luminous cyan lines/border
    col_grid = nt.nodes.new("ShaderNodeMix")
    col_grid.data_type = "RGBA"
    col_grid.inputs[6].default_value = (0.06, 0.94, 0.50, 1.0)
    col_grid.inputs[7].default_value = (0.18, 0.92, 1.00, 1.0)
    nt.links.new(hi_mask.outputs["Value"], col_grid.inputs["Factor"])

    fill_s = nt.nodes.new("ShaderNodeMath")
    fill_s.operation = "MULTIPLY"
    fill_s.inputs[1].default_value = 0.55
    nt.links.new(body_fill.outputs["Result"], fill_s.inputs[0])

    line_s = nt.nodes.new("ShaderNodeMath")
    line_s.operation = "MULTIPLY"
    line_s.inputs[1].default_value = 1.80
    nt.links.new(hi_mask.outputs["Value"], line_s.inputs[0])

    raw_s = nt.nodes.new("ShaderNodeMath")
    raw_s.operation = "MAXIMUM"
    nt.links.new(fill_s.outputs["Value"], raw_s.inputs[0])
    nt.links.new(line_s.outputs["Value"], raw_s.inputs[1])

    final_grid_s = nt.nodes.new("ShaderNodeMath")
    final_grid_s.operation = "MULTIPLY"
    nt.links.new(raw_s.outputs["Value"], final_grid_s.inputs[0])
    nt.links.new(grid_fade.outputs[0], final_grid_s.inputs[1])

    tr_fg = nt.nodes.new("ShaderNodeBsdfTransparent")
    em_fg = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(col_grid.outputs[2], em_fg.inputs["Color"])
    nt.links.new(final_grid_s.outputs["Value"], em_fg.inputs["Strength"])

    add_fg = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(tr_fg.outputs["BSDF"], add_fg.inputs[0])
    nt.links.new(em_fg.outputs["Emission"], add_fg.inputs[1])

    surf_fg = _wire_shadow_bypass(nt, add_fg.outputs["Shader"], (1.0, 1.0, 1.0, 1.0))
    nt.links.new(surf_fg, out.inputs["Surface"])
    mats["force_grid"] = m_grid

    return mats, ctrl


# ---------------------------------------------------------------------------
# GEOMETRY SUB-BUILDERS
# ---------------------------------------------------------------------------
def _build_floor_and_chassis(col: Any, cab_root: Any, mats: dict[str, Any]) -> list[Any]:
    objs: list[Any] = []

    # Floor slab Z = -0.06..0.0 (mat 0: terrazzo, mat 1: brushed trim, mat 2: dark steel)
    bm_fl = bmesh.new()
    _add_box_to_bm(bm_fl, (0.0, 0.0, -0.03), (2.0 * HALF_W + 0.08, 2.0 * HALF_D + 0.08, 0.06), 0.004, 2, mat_index=0)
    # Perimeter stainless floor inlay strips flush at Z = +0.0005
    for sx in (-1.0, 1.0):
        _add_box_to_bm(bm_fl, (sx * (HALF_W - 0.025), 0.0, -0.004), (0.028, 2.0 * HALF_D - 0.02, 0.010), 0.001, 1, mat_index=1)
    for sy in (-1.0, 1.0):
        _add_box_to_bm(bm_fl, (0.0, sy * (HALF_D - 0.025), -0.004), (2.0 * HALF_W - 0.02, 0.028, 0.010), 0.001, 1, mat_index=1)
    # Door threshold sill plate with machined track grooves at Y = -HALF_D
    _add_box_to_bm(bm_fl, (0.0, -HALF_D - 0.01, -0.006), (2.0 * HALF_W + 0.04, 0.07, 0.016), 0.0015, 1, mat_index=1)
    obj_fl = _create_mesh_obj(
        f"{PREFIX}Floor_Slab", bm_fl,
        [mats["floor_terrazzo"], mats["brushed_steel"], mats["dark_steel"]],
        col, parent=cab_root, smooth_all=False, visible_shadow=True,
    )
    objs.append(obj_fl)

    # Structural under-cab chassis frame (Z = -0.20..-0.06) + buffer strike pads (X = +-BUFFER_X)
    bm_ch = bmesh.new()
    for sx in (-1.0, 1.0):
        _add_box_to_bm(bm_ch, (sx * (HALF_W - 0.02), 0.0, -0.13), (0.075, 2.0 * HALF_D + 0.04, 0.14), 0.003, 1, mat_index=0)
    for sy in (-1.0, 1.0):
        _add_box_to_bm(bm_ch, (0.0, sy * (HALF_D - 0.03), -0.13), (2.0 * HALF_W + 0.04, 0.075, 0.14), 0.003, 1, mat_index=0)
    # Central safety plank channel bridging X = -0.90..+0.90 at Z = CLAMP_Z (-0.14)
    _add_box_to_bm(bm_ch, (0.0, 0.0, CLAMP_Z), (2.0 * HALF_W + 0.10, 0.16, 0.095), 0.004, 1, mat_index=0)
    # Twin buffer strike plates at X = +-BUFFER_X (aligned to shaft pit buffers)
    for sx in (-1.0, 1.0):
        _add_box_to_bm(bm_ch, (sx * BUFFER_X, 0.0, -0.145), (0.20, 0.24, 0.13), 0.004, 2, mat_index=0)
        _add_cyl_to_bm(bm_ch, (sx * BUFFER_X, 0.0, -0.205), 0.075, 0.025, segs=18, axis="Z", mat_index=1)
    obj_ch = _create_mesh_obj(
        f"{PREFIX}Chassis_Underframe", bm_ch,
        [mats["dark_steel"], mats["brushed_steel"]],
        col, parent=cab_root, smooth_all=False, visible_shadow=True,
    )
    objs.append(obj_ch)
    return objs


def _build_rear_wall_and_handrail(col: Any, cab_root: Any, mats: dict[str, Any]) -> tuple[list[Any], Any]:
    objs: list[Any] = []

    # Rear acoustic fluted walnut wall (Y = +0.70..+0.75, X in [-0.90, +0.90], Z in [0.0, 2.45])
    bm_rw = bmesh.new()
    # Structural outer backing plate (mat 0: dark steel, mat 1: obsidian reveal, mat 2: walnut, mat 3: brushed steel)
    _add_box_to_bm(bm_rw, (0.0, HALF_D + 0.035, 0.5 * ROOF_Z), (2.0 * HALF_W + 0.10, 0.05, ROOF_Z), 0.003, 1, mat_index=0)
    _add_box_to_bm(bm_rw, (0.0, HALF_D + 0.008, 0.5 * CEILING_Z), (2.0 * HALF_W, 0.014, CEILING_Z - 0.12), 0.001, 1, mat_index=1)
    # Brushed steel kickplate & top frieze
    _add_box_to_bm(bm_rw, (0.0, HALF_D + 0.004, 0.045), (2.0 * HALF_W, 0.018, 0.09), 0.002, 1, mat_index=3)
    _add_box_to_bm(bm_rw, (0.0, HALF_D + 0.004, CEILING_Z - 0.025), (2.0 * HALF_W, 0.018, 0.05), 0.002, 1, mat_index=3)
    # Vertical fluted walnut slats (31 crisp slats across X in [-0.78, +0.78])
    n_slats = 31
    for i in range(n_slats):
        sx = C.lerp(-0.78, 0.78, i / (n_slats - 1))
        _add_box_to_bm(bm_rw, (sx, HALF_D - 0.003, 1.20), (0.032, 0.015, 2.18), 0.002, 1, mat_index=2)
    obj_rw = _create_mesh_obj(
        f"{PREFIX}Rear_Wall", bm_rw,
        [mats["dark_steel"], mats["obsidian_trim"], mats["walnut_fluted"], mats["brushed_steel"]],
        col, parent=cab_root, smooth_all=False, visible_shadow=True,
    )
    objs.append(obj_rw)

    # Rear brushed stainless safety handrail at (y=HANDRAIL_Y=0.61, z=HANDRAIL_Z=0.92)
    bm_hr = bmesh.new()
    x_span = 0.72
    pts = [Vector((-x_span, HALF_D, HANDRAIL_Z))]
    # Smooth 90-degree return elbows at left & right ends
    r_elb = 0.055
    for k in range(1, 6):
        a = k * (math.pi * 0.5) / 5.0
        pts.append(Vector((-x_span + r_elb * (1.0 - math.cos(a)) * 0.65, HALF_D - (HALF_D - HANDRAIL_Y) * math.sin(a), HANDRAIL_Z)))
    pts.append(Vector((0.0, HANDRAIL_Y, HANDRAIL_Z)))
    for k in range(5):
        a = (5 - k) * (math.pi * 0.5) / 5.0
        pts.append(Vector((x_span - r_elb * (1.0 - math.cos(a)) * 0.65, HALF_D - (HALF_D - HANDRAIL_Y) * math.sin(a), HANDRAIL_Z)))
    pts.append(Vector((x_span, HALF_D, HANDRAIL_Z)))
    _add_tube_along_path_bm(bm_hr, pts, r_start=0.019, r_end=0.019, n_seg=18, cap_ends=True, mat_index=0)

    # Wall mounting rosettes & intermediate stanchions
    for rx in (-x_span, -0.36, 0.36, x_span):
        _add_cyl_to_bm(bm_hr, (rx, HALF_D - 0.004, HANDRAIL_Z), 0.034, 0.012, segs=18, axis="Y", mat_index=0)
        if abs(rx) < x_span - 0.05:
            _add_cyl_to_bm(bm_hr, (rx, 0.5 * (HALF_D + HANDRAIL_Y), HANDRAIL_Z), 0.010, HALF_D - HANDRAIL_Y, segs=12, axis="Y", mat_index=0)

    obj_hr = _create_mesh_obj(
        f"{PREFIX}Rear_Handrail", bm_hr, [mats["brushed_steel"]],
        col, parent=cab_root, smooth_all=True, visible_shadow=True,
    )
    objs.append(obj_hr)
    return objs, obj_hr


def _build_side_walls_and_cop(col: Any, cab_root: Any, mats: dict[str, Any]) -> tuple[list[Any], Any]:
    objs: list[Any] = []

    # Side walls (X = +-0.85..+-0.90) with Local-Space XRayCab shader + shadow bypass
    for side, sx in (("L", -1.0), ("R", 1.0)):
        bm_sw = bmesh.new()
        # Structural corner/center vertical mullions (mat 0: xray_frame, mat 1: xray_panel)
        for my in (-HALF_D + 0.025, 0.0, HALF_D - 0.025):
            _add_box_to_bm(bm_sw, (sx * (HALF_W + 0.022), my, 0.5 * ROOF_Z), (0.044, 0.052, ROOF_Z), 0.003, 1, mat_index=0)
        # Top & bottom horizontal rails
        _add_box_to_bm(bm_sw, (sx * (HALF_W + 0.022), 0.0, 0.04), (0.044, 2.0 * HALF_D, 0.08), 0.002, 1, mat_index=0)
        _add_box_to_bm(bm_sw, (sx * (HALF_W + 0.022), 0.0, CEILING_Z - 0.025), (0.044, 2.0 * HALF_D, 0.05), 0.002, 1, mat_index=0)
        # Fluted acoustic titanium side panels
        for py in (-0.34, 0.34):
            _add_box_to_bm(bm_sw, (sx * (HALF_W + 0.020), py, 1.20), (0.028, 0.60, 2.18), 0.002, 1, mat_index=1)
        obj_sw = _create_mesh_obj(
            f"{PREFIX}Side_Wall_{side}", bm_sw,
            [mats["xray_frame"], mats["xray_panel"]],
            col, parent=cab_root, smooth_all=False, visible_shadow=False,
        )
        objs.append(obj_sw)

    # COP (Car Operating Panel) anchored around C.CAB["cop_panel"] = (0.76, -0.52, 1.24)
    cx, cy, cz = COP_POS
    bm_cop = bmesh.new()
    # Chamfered brushed-steel console bridging from right wall (X=0.85) inward to X=0.76
    _add_box_to_bm(bm_cop, (0.5 * (cx + HALF_W) + 0.005, cy, cz), (HALF_W - cx + 0.04, 0.14, 0.56), 0.006, 2, mat_index=0)
    # Recessed dark faceplate
    _add_box_to_bm(bm_cop, (cx + 0.005, cy, cz), (0.014, 0.11, 0.48), 0.002, 1, mat_index=1)
    # 12 illuminated floor pushbuttons (2 cols x 6 rows)
    for row in range(6):
        bz = cz - 0.15 + row * 0.052
        for col_i in (-1, 1):
            by = cy + col_i * 0.026
            _add_cyl_to_bm(bm_cop, (cx - 0.003, by, bz), 0.011, 0.008, segs=12, axis="X", mat_index=2)
    # Crimson emergency ALARM button at bottom of COP
    _add_cyl_to_bm(bm_cop, (cx - 0.004, cy, cz - 0.21), 0.014, 0.010, segs=14, axis="X", mat_index=3)

    obj_cop = _create_mesh_obj(
        f"{PREFIX}COP_Panel", bm_cop,
        [mats["xray_frame"], mats["xray_panel"], mats["cop_btn_warm"], mats["led_crimson"]],
        col, parent=cab_root, smooth_all=False, visible_shadow=False,
    )
    objs.append(obj_cop)
    return objs, obj_cop


def _build_doors_and_ceiling(col: Any, cab_root: Any, mats: dict[str, Any]) -> tuple[list[Any], list[Any], list[Any]]:
    objs: list[Any] = []
    doors: list[Any] = []
    ceiling_lights: list[Any] = []

    # Bi-parting frameless-center tempered low-glare glass doors at Y = -HALF_D (-0.70)
    gz_bot = 0.09
    gz_top = CEILING_Z - 0.07
    gz_center = 0.5 * (gz_bot + gz_top)
    gz_height = gz_top - gz_bot
    for side, x_min, x_max in (
        ("L", -HALF_W, -0.002),
        ("R", 0.002, HALF_W),
    ):
        bm_d = bmesh.new()
        w = x_max - x_min
        xc = 0.5 * (x_min + x_max)
        # Sleek outer perimeter stile, bottom kick rail, top hanger rail (mat 0: xray_frame) — NO center mullion
        outer_x = x_min + 0.032 if side == "L" else x_max - 0.032
        _add_box_to_bm(bm_d, (outer_x, -HALF_D - 0.008, 0.5 * CEILING_Z), (0.064, 0.032, CEILING_Z), 0.002, 1, mat_index=0)
        _add_box_to_bm(bm_d, (xc, -HALF_D - 0.008, 0.045), (w, 0.032, 0.09), 0.002, 1, mat_index=0)
        _add_box_to_bm(bm_d, (xc, -HALF_D - 0.008, CEILING_Z - 0.035), (w, 0.032, 0.07), 0.002, 1, mat_index=0)
        # Crystal-clear low-glare tempered glass viewing pane widened cleanly to x = +-0.002 (mat 1: front_glass)
        gx_min = outer_x + 0.030 if side == "L" else x_min
        gx_max = x_max if side == "L" else outer_x - 0.030
        _add_box_to_bm(
            bm_d,
            (0.5 * (gx_min + gx_max), -HALF_D - 0.008, gz_center),
            (gx_max - gx_min, 0.012, gz_height),
            0.0, 1, mat_index=1,
        )
        obj_d = _create_mesh_obj(
            f"{PREFIX}Door_{side}", bm_d,
            [mats["xray_frame"], mats["front_glass"]],
            col, parent=cab_root, smooth_all=False, visible_shadow=False,
        )
        objs.append(obj_d)
        doors.append(obj_d)

    # Outer roof slab + dark ceiling soffit (Z = 2.35..2.45) + recessed warm LED coves (z = 2.34)
    bm_rf = bmesh.new()
    _add_box_to_bm(bm_rf, (0.0, 0.0, 0.5 * (CEILING_Z + ROOF_Z)), (2.0 * HALF_W + 0.10, 2.0 * HALF_D + 0.10, ROOF_Z - CEILING_Z), 0.004, 1, mat_index=0)
    # Emergency roof escape hatch frame
    _add_box_to_bm(bm_rf, (0.0, 0.12, CEILING_Z - 0.004), (0.52, 0.42, 0.012), 0.002, 1, mat_index=1)
    # 4 recessed warm ceiling LED slats at z = 2.34 (mat 2: ceiling_led)
    for ly in (-0.48, -0.18, 0.18, 0.48):
        _add_box_to_bm(bm_rf, (0.0, ly, CEILING_Z - 0.006), (1.36, 0.042, 0.012), 0.001, 1, mat_index=2)
    obj_rf = _create_mesh_obj(
        f"{PREFIX}Roof_And_Ceiling", bm_rf,
        [mats["dark_steel"], mats["brushed_steel"], mats["ceiling_led"]],
        col, parent=cab_root, smooth_all=False, visible_shadow=True,
    )
    objs.append(obj_rf)
    ceiling_lights.append(obj_rf)

    # Soft interior ceiling AREA downlight for grounded realism inside cab
    lt_data = bpy.data.lights.new(f"{PREFIX}Ceiling_AreaLight", type="AREA")
    lt_data.shape = "RECTANGLE"
    lt_data.size = 1.10
    lt_data.size_y = 0.80
    lt_data.color = (1.0, 0.92, 0.80)
    lt_data.energy = 26.0
    lt_data.diffuse_factor = 0.85
    lt_data.specular_factor = 0.65
    lt_obj = bpy.data.objects.new(f"{PREFIX}Ceiling_AreaLight", lt_data)
    lt_obj.location = Vector((0.0, 0.0, CEILING_Z - 0.04))
    lt_obj.rotation_euler = Euler((0.0, 0.0, 0.0), "XYZ")  # default AREA points along -Z
    lt_obj.parent = cab_root
    _link_obj(lt_obj, col)
    objs.append(lt_obj)
    ceiling_lights.append(lt_obj)

    return objs, doors, ceiling_lights


def _load_font(font_path: Path) -> Any:
    if font_path.is_file():
        return bpy.data.fonts.load(str(font_path), check_existing=True)
    return None


def _build_header_display_and_alarm(
    col: Any,
    cab_root: Any,
    mats: dict[str, Any],
    total_frames: int,
) -> tuple[list[Any], Any, list[Any], Any]:
    objs: list[Any] = []
    lx, ly, lz = FLOOR_LED_POS  # (0.0, -0.66, 2.18)

    # Dark obsidian digital header display bezel
    bm_bez = bmesh.new()
    _add_box_to_bm(bm_bez, (lx, ly, lz), (0.38, 0.032, 0.125), 0.004, 2, mat_index=0)
    _add_box_to_bm(bm_bez, (lx, ly - 0.008, lz), (0.35, 0.018, 0.10), 0.002, 1, mat_index=1)
    # Brushed steel mounting struts up to ceiling soffit
    for sx in (-0.14, 0.14):
        _add_box_to_bm(bm_bez, (lx + sx, ly, 0.5 * (lz + 0.06 + CEILING_Z)), (0.018, 0.018, CEILING_Z - lz - 0.05), 0.001, 1, mat_index=0)
    obj_bez = _create_mesh_obj(
        f"{PREFIX}Floor_LED_Housing", bm_bez,
        [mats["dark_steel"], mats["obsidian_trim"]],
        col, parent=cab_root, smooth_all=False, visible_shadow=False,
    )
    objs.append(obj_bez)

    # Discrete live digital readout FONT objects (`18 ↓` .. `01` / `ERR` / `STOP`)
    font_data = _load_font(DEFAULT_FONT_PATH)
    schedule = floor_led_schedule(total_frames)
    theme_mat = {
        "amber": mats["led_amber"],
        "crimson": mats["led_crimson"],
        "emerald": mats["led_emerald"],
    }
    variant_ranges: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for txt, theme, s, e in schedule:
        variant_ranges.setdefault((txt, theme), []).append((s, e))

    readouts: list[Any] = []
    for idx, ((txt, theme), ranges) in enumerate(variant_ranges.items()):
        cdata = bpy.data.curves.new(f"{PREFIX}Floor_LED_Text_{idx:02d}", type="FONT")
        cdata.body = txt
        if font_data is not None:
            cdata.font = font_data
        cdata.size = 0.072
        cdata.align_x = "CENTER"
        cdata.align_y = "CENTER"
        cdata.extrude = 0.0025
        cdata.bevel_depth = 0.0004
        cdata.space_character = 1.05

        tobj = bpy.data.objects.new(f"{PREFIX}Floor_LED_Text_{idx:02d}", cdata)
        tobj.data.materials.append(theme_mat[theme])
        tobj.location = Vector((lx, ly - 0.019, lz))
        tobj.rotation_euler = Euler((math.pi * 0.5, 0.0, 0.0), "XYZ")  # faces -Y viewer front
        tobj.parent = cab_root
        tobj.visible_shadow = False
        _link_obj(tobj, col)
        _key_visibility(tobj, ranges, total_frames)
        objs.append(tobj)
        readouts.append(tobj)

    # Flashing crimson ALARM strobe housing + POINT light beside header display
    ax, ay, az = (0.30, ly, lz)
    bm_alm = bmesh.new()
    _add_box_to_bm(bm_alm, (ax, ay, az), (0.085, 0.036, 0.095), 0.003, 1, mat_index=0)
    _add_cyl_to_bm(bm_alm, (ax, ay - 0.022, az), 0.024, 0.024, segs=16, axis="Y", mat_index=1)
    obj_alm = _create_mesh_obj(
        f"{PREFIX}Alarm_Strobe_Housing", bm_alm,
        [mats["obsidian_trim"], mats["alarm_lens"]],
        col, parent=cab_root, smooth_all=True, visible_shadow=False,
    )
    objs.append(obj_alm)

    alm_data = bpy.data.lights.new(f"{PREFIX}Alarm_Light", type="POINT")
    alm_data.color = (1.0, 0.08, 0.04)
    alm_data.shadow_soft_size = 0.06
    alm_data.energy = 0.0
    alm_light = bpy.data.objects.new(f"{PREFIX}Alarm_Light", alm_data)
    alm_light.location = Vector((ax, ay - 0.06, az - 0.02))
    alm_light.parent = cab_root
    _link_obj(alm_light, col)
    objs.append(alm_light)

    return objs, obj_bez, readouts, alm_light


def _build_crosshead_and_cables(
    col: Any,
    cab_root: Any,
    mats: dict[str, Any],
    total_frames: int,
) -> tuple[list[Any], list[Any], list[Any], Any]:
    objs: list[Any] = []
    cables: list[Any] = []
    snap_sparks: list[Any] = []

    # Structural overhead crosshead beam & central hitch block at (0, 0, HITCH_Z=2.55)
    bm_xh = bmesh.new()
    for sy in (-0.09, 0.09):
        _add_box_to_bm(bm_xh, (0.0, sy, ROOF_Z + 0.065), (2.0 * HALF_W + 0.10, 0.045, 0.13), 0.003, 1, mat_index=0)
    # Central machined hitch block & 4 forged shackle clevises at (+-0.08, +-0.03, HITCH_Z)
    _add_box_to_bm(bm_xh, (0.0, 0.0, HITCH_Z - 0.02), (0.30, 0.22, 0.12), 0.005, 2, mat_index=0)
    for ax, ay in CABLE_ANCHORS:
        _add_cyl_to_bm(bm_xh, (ax, ay, HITCH_Z), 0.020, 0.048, segs=12, axis="Z", mat_index=1)
    obj_xh = _create_mesh_obj(
        f"{PREFIX}Crosshead_Hitch", bm_xh,
        [mats["dark_steel"], mats["brushed_steel"]],
        col, parent=cab_root, smooth_all=False, visible_shadow=True,
    )
    objs.append(obj_xh)

    # Helper: unit-length cable segment [0..1] along +Z for intact & upper recoiling cables
    def make_cable_unit(name: str, radius: float, parent_obj: Any | None = None) -> Any:
        bm_c = bmesh.new()
        pts = [Vector((0.0, 0.0, k / 16.0)) for k in range(17)]
        _add_tube_along_path_bm(bm_c, pts, radius, radius, n_seg=10, cap_ends=True, mat_index=0)
        return _create_mesh_obj(name, bm_c, [mats["hoist_cable"]], col, parent=parent_obj, smooth_all=True, visible_shadow=True)

    # 1) Intact cables (World space, taut from cab hitch z+2.55 to sheave 27.5 for f=1..324)
    intact_frames = _sample_frames(min(total_frames, CABLE_SNAP - 1), dense_windows=[(225, 324)], step=4)
    for idx, (ax, ay) in enumerate(CABLE_ANCHORS):
        c_int = make_cable_unit(f"{PREFIX}Cable_Intact_{idx}", 0.011, parent_obj=None)
        _key_visibility(c_int, [(1, CABLE_SNAP - 1)], total_frames)
        for f in intact_frames:
            z_bot = C.cab_world_z(float(f)) + HITCH_Z
            span = max(0.10, SHEAVE_Z - z_bot)
            c_int.location = Vector((ax, ay, z_bot))
            c_int.scale = Vector((1.0, 1.0, span))
            c_int.keyframe_insert("location", frame=f)
            c_int.keyframe_insert("scale", frame=f)
        _set_interp(c_int, "LINEAR", "location")
        _set_interp(c_int, "LINEAR", "scale")
        objs.append(c_int)
        cables.append(c_int)

    # 2) Lower short whipping stubs (parented to Cab_Root at (+-0.08, +-0.03, HITCH_Z), length=0.25m)
    stub_frames = _sample_frames(
        total_frames,
        dense_windows=[(CABLE_SNAP, 440), (935, 1020)],
        step=6,
    )
    stub_frames = [f for f in stub_frames if f >= CABLE_SNAP]
    for idx, (ax, ay) in enumerate(CABLE_ANCHORS):
        bm_s = bmesh.new()
        pts = [Vector((0.0, 0.0, k * SEVER_OFFSET_Z / 8.0)) for k in range(9)]
        _add_tube_along_path_bm(bm_s, pts, 0.011, 0.0095, n_seg=10, cap_ends=True, mat_index=0)
        # Frayed splayed wire tips at z = SEVER_OFFSET_Z (0.25 m above hitch)
        for k in range(5):
            ang = (idx * 1.3) + k * (2.0 * math.pi / 5.0)
            p_tip = Vector((0.022 * math.cos(ang), 0.022 * math.sin(ang), SEVER_OFFSET_Z + 0.038))
            _add_tube_along_path_bm(bm_s, [Vector((0.0, 0.0, SEVER_OFFSET_Z - 0.01)), p_tip], 0.0032, 0.0012, n_seg=6, cap_ends=True, mat_index=0)
        stub = _create_mesh_obj(
            f"{PREFIX}Cable_Stub_{idx}", bm_s, [mats["hoist_cable"]],
            col, parent=cab_root, location=(ax, ay, HITCH_Z), smooth_all=True, visible_shadow=True,
        )
        _key_visibility(stub, [(CABLE_SNAP, total_frames)], total_frames)
        phase = idx * 1.45
        for f in stub_frames:
            if f <= 450:
                u = (f - CABLE_SNAP) / 125.0
                amp = 0.68 * math.exp(-1.6 * u)
                rx = amp * math.sin(f * 0.62 + phase)
                ry = amp * math.cos(f * 0.54 + phase * 0.7)
            elif f <= 940:
                rx = 0.14 * math.sin(f * 0.38 + phase)
                ry = 0.14 * math.cos(f * 0.33 + phase)
            elif f <= 1236:
                u_s = C.smootherstep(940.0, 1020.0, float(f))
                rx = C.lerp(0.14, 0.38 * (1 if idx % 2 == 0 else -1), u_s)
                ry = C.lerp(0.14, 0.28 * (1 if idx < 2 else -1), u_s)
            else:
                rx = 0.22 * (1 if idx % 2 == 0 else -1)
                ry = 0.16 * (1 if idx < 2 else -1)
            stub.rotation_euler = Euler((rx, ry, 0.0), "XYZ")
            stub.keyframe_insert("rotation_euler", frame=f)
        _set_interp(stub, "LINEAR", "rotation_euler")
        objs.append(stub)
        cables.append(stub)

    # 3) Upper long snapped recoiling cables (World space, visible f=325..395, CONSTANT hidden >= 396!)
    recoil_end = min(total_frames, 395)
    z_sever_world = C.cab_world_z(float(CABLE_SNAP)) + SNAP_ORIGIN_Z  # 20.80 m
    for idx, (ax, ay) in enumerate(CABLE_ANCHORS):
        c_up = make_cable_unit(f"{PREFIX}Cable_Upper_{idx}", 0.011, parent_obj=None)
        _key_visibility(c_up, [(CABLE_SNAP, recoil_end)], total_frames)
        phase = idx * 1.2
        for f in range(CABLE_SNAP, recoil_end + 1, 2):
            u = C.smootherstep(float(CABLE_SNAP), float(recoil_end), float(f))
            z_b = C.lerp(z_sever_world, SHEAVE_Z - 0.12, u ** 1.25)
            span = max(0.12, SHEAVE_Z - z_b)
            whip_x = 0.18 * math.sin(u * math.pi * 2.5 + phase) * (1.0 - u)
            whip_y = 0.16 * math.cos(u * math.pi * 2.1 + phase) * (1.0 - u)
            c_up.location = Vector((ax + whip_x * 0.4, ay + whip_y * 0.4, z_b))
            c_up.rotation_euler = Euler((whip_x, whip_y, 0.0), "XYZ")
            c_up.scale = Vector((1.0, 1.0, span))
            c_up.keyframe_insert("location", frame=f)
            c_up.keyframe_insert("rotation_euler", frame=f)
            c_up.keyframe_insert("scale", frame=f)
        _set_interp(c_up, "LINEAR", "location")
        _set_interp(c_up, "LINEAR", "rotation_euler")
        _set_interp(c_up, "LINEAR", "scale")
        objs.append(c_up)
        cables.append(c_up)

    # 4) Tapered 3D incandescent parabolic tube filaments + ember droplets at (0, 0, 2.80) across f=325..365
    rng = random.Random(16325)
    bm_sp = bmesh.new()
    for i in range(24):
        az = i * (2.0 * math.pi / 24.0) + rng.uniform(-0.10, 0.10)
        elev = rng.uniform(-0.08, 0.65)
        length = rng.uniform(0.32, 0.66)
        dir_v = Vector((math.cos(az) * math.cos(elev), math.sin(az) * math.cos(elev), math.sin(elev))).normalized()
        droop = rng.uniform(0.06, 0.16)
        pts = [
            dir_v * (0.02 + length * u) + Vector((0.0, 0.0, -droop * u * u))
            for u in (0.0, 0.20, 0.40, 0.60, 0.80, 1.0)
        ]
        _add_tube_along_path_bm(bm_sp, pts, r_start=0.0048, r_end=0.0008, n_seg=6, cap_ends=True, mat_index=0)
        u_drop = rng.uniform(0.78, 1.08)
        drop_pt = dir_v * (0.02 + length * u_drop) + Vector((0.0, 0.0, -droop * u_drop * u_drop))
        _add_icosphere_bm(bm_sp, drop_pt, radius=0.0045, subdivisions=1, mat_index=0)
    # Inner core parabolic incandescent filaments + trailing embers
    for r_i in range(8):
        a0 = r_i * (2.0 * math.pi / 8.0) + rng.uniform(-0.12, 0.12)
        v_dir = Vector((math.cos(a0), math.sin(a0), 0.22)).normalized()
        pts_in = [
            v_dir * (0.015 + 0.24 * u) + Vector((0.0, 0.0, -0.045 * u * u))
            for u in (0.0, 0.25, 0.50, 0.75, 1.0)
        ]
        _add_tube_along_path_bm(bm_sp, pts_in, r_start=0.0048, r_end=0.0008, n_seg=6, cap_ends=True, mat_index=0)
        _add_icosphere_bm(bm_sp, pts_in[-1] + v_dir * 0.02, radius=0.0045, subdivisions=1, mat_index=0)
    obj_sp = _create_mesh_obj(
        f"{PREFIX}Snap_Sparks", bm_sp, [mats["snap_sparks"]],
        col, parent=cab_root, location=(0.0, 0.0, SNAP_ORIGIN_Z),
        smooth_all=True, visible_shadow=False, recalc_normals=True,
    )
    _key_visibility(obj_sp, [(CABLE_SNAP, 365)], total_frames)
    for f in range(CABLE_SNAP, 366, 2):
        u = (f - CABLE_SNAP) / 40.0
        sc = C.lerp(0.25, 1.35, u ** 0.55)
        obj_sp.scale = Vector((sc, sc, sc))
        obj_sp.keyframe_insert("scale", frame=f)
    _set_interp(obj_sp, "LINEAR", "scale")
    objs.append(obj_sp)
    snap_sparks.append(obj_sp)

    # Dynamic snap spark POINT light
    sp_lt_data = bpy.data.lights.new(f"{PREFIX}Snap_SparkLight", type="POINT")
    sp_lt_data.color = (1.0, 0.56, 0.12)
    sp_lt_data.shadow_soft_size = 0.08
    sp_lt_data.energy = 0.0
    sp_lt = bpy.data.objects.new(f"{PREFIX}Snap_SparkLight", sp_lt_data)
    sp_lt.location = Vector((0.0, -0.05, SNAP_ORIGIN_Z + 0.04))
    sp_lt.parent = cab_root
    _link_obj(sp_lt, col)
    for f in [1, CABLE_SNAP - 1] + list(range(CABLE_SNAP, 366, 2)) + [total_frames]:
        if 1 <= f <= total_frames:
            sp_lt_data.energy = 160.0 * snap_spark_gain(float(f))
            sp_lt_data.keyframe_insert("energy", frame=f)
    _set_interp(sp_lt_data, "LINEAR", "energy")
    objs.append(sp_lt)
    snap_sparks.append(sp_lt)

    return objs, cables, snap_sparks, sp_lt


def _build_safety_clamps(
    col: Any,
    cab_root: Any,
    mats: dict[str, Any],
    total_frames: int,
) -> tuple[list[Any], list[Any], list[Any]]:
    objs: list[Any] = []
    clamps: list[Any] = []
    brake_sparks: list[Any] = []

    clamp_frames = _sample_frames(total_frames, dense_windows=[(1258, 1290)], step=12)
    # Include exact Beat 5a start/end and clamp keyframes
    for extra in (1, BRAKE_START - 1, BRAKE_START, 1260, 1265, BRAKE_CLAMP, 1285, BRAKE_END, BRAKE_END + 1, total_frames):
        if 1 <= extra <= total_frames and extra not in clamp_frames:
            clamp_frames.append(extra)
    clamp_frames.sort()

    for side, sx in (("L", -1.0), ("R", 1.0)):
        rail_cx = sx * RAIL_X  # +-0.96

        # 1) Fixed heavy machined steel mounting bracket bridging X = +-0.85 .. +-1.02 at Z = CLAMP_Z
        bm_brk = bmesh.new()
        # Bridge arm from cab side (0.85) out to rail housing (0.94)
        _add_box_to_bm(bm_brk, (sx * 0.895, 0.0, CLAMP_Z), (0.09, 0.22, 0.13), 0.004, 1, mat_index=0)
        # Twin Fore (-Y) & Aft (+Y) cheek plates straddling the T-rail blade (leaving |Y| < 0.032 clear!)
        for sy in (-1.0, 1.0):
            _add_box_to_bm(bm_brk, (rail_cx, sy * 0.072, CLAMP_Z), (0.11, 0.075, 0.145), 0.004, 2, mat_index=0)
        # Outer C-tie plate beyond rail flange (X = sx * 1.025)
        _add_box_to_bm(bm_brk, (sx * 1.022, 0.0, CLAMP_Z), (0.024, 0.22, 0.145), 0.003, 1, mat_index=0)
        obj_brk = _create_mesh_obj(
            f"{PREFIX}Clamp_Bracket_{side}", bm_brk, [mats["dark_steel"]],
            col, parent=cab_root, smooth_all=False, visible_shadow=True,
        )
        objs.append(obj_brk)
        clamps.append(obj_brk)

        # 2) Twin serrated wedge safety jaws (Fore -Y, Aft +Y) at X = +-0.96
        # Open clearance = 8 mm from rail blade (|Y| = 0.010 -> inner face at |Y| = 0.018)
        for jname, sy in (("Fore", -1.0), ("Aft", 1.0)):
            bm_jaw = bmesh.new()
            # Wedge shoe body: inner bite face at local Y = -sy * 0.016 (when placed at Y = sy * (RAIL_BLADE_HALF_Y + clearance + 0.016))
            _add_box_to_bm(bm_jaw, (0.0, 0.0, 0.0), (0.072, 0.032, 0.115), 0.0025, 1, mat_index=0)
            # Machined grip teeth strips along inner clamping face
            for tz in (-0.035, 0.0, 0.035):
                _add_box_to_bm(bm_jaw, (0.0, -sy * 0.016, tz), (0.058, 0.004, 0.012), 0.001, 1, mat_index=1)
            jaw = _create_mesh_obj(
                f"{PREFIX}Clamp_Jaw_{side}_{jname}", bm_jaw,
                [mats["clamp_steel"], mats["brushed_steel"]],
                col, parent=cab_root, smooth_all=False, visible_shadow=True,
            )
            for f in clamp_frames:
                eng = clamp_engage_factor(float(f))
                y_inner = RAIL_BLADE_HALF_Y + CLAMP_OPEN_CLEARANCE * (1.0 - eng)
                y_center = sy * (y_inner + 0.016)
                z_wedge = CLAMP_Z + C.lerp(-0.018, 0.012, eng)
                jaw.location = Vector((rail_cx, y_center, z_wedge))
                jaw.keyframe_insert("location", frame=f)
            _set_interp(jaw, "LINEAR", "location")
            objs.append(jaw)
            clamps.append(jaw)

        # 3) Amber-Gold Spring Linkage & Helical Actuator Coil (bridges X = +-0.85..+-0.94)
        bm_spg = bmesh.new()
        n_turns = 5.5
        n_pts = 48
        sp_pts = []
        for k in range(n_pts):
            u = k / (n_pts - 1)
            ang = u * n_turns * 2.0 * math.pi
            sp_pts.append(Vector((u * 0.075, 0.022 * math.cos(ang), 0.022 * math.sin(ang))))
        _add_tube_along_path_bm(bm_spg, sp_pts, 0.0045, 0.0045, n_seg=8, cap_ends=True, mat_index=0)
        # Central pushrod & clevis collar
        _add_cyl_to_bm(bm_spg, (0.038, 0.0, 0.0), 0.007, 0.085, segs=10, axis="X", mat_index=1)
        spring = _create_mesh_obj(
            f"{PREFIX}Clamp_Spring_{side}", bm_spg,
            [mats["clamp_spring"], mats["brushed_steel"]],
            col, parent=cab_root,
            location=(sx * 0.855, 0.0, CLAMP_Z),
            rotation=(0.0, 0.0, 0.0 if sx > 0 else math.pi),
            smooth_all=True, visible_shadow=True,
        )
        for f in clamp_frames:
            eng = clamp_engage_factor(float(f))
            spring.scale = Vector((C.lerp(1.0, 0.72, eng), 1.0, 1.0))
            spring.keyframe_insert("scale", frame=f)
        _set_interp(spring, "LINEAR", "scale")
        objs.append(spring)
        clamps.append(spring)

        # 4) Upward Tapered 3D Incandescent Rail-Friction Tube Filaments + Ember Droplets (f=1270..1365)
        rng_b = random.Random(161270 + (0 if side == "L" else 99))
        bm_bs = bmesh.new()
        for i in range(20):
            az = rng_b.uniform(-1.05, 1.05)
            spread_x = sx * rng_b.uniform(0.025, 0.18)
            spread_y = math.sin(az) * rng_b.uniform(0.03, 0.18)
            y0 = rng_b.uniform(-0.015, 0.015)
            h_top = rng_b.uniform(0.28, 0.72)
            pts = [
                Vector((
                    spread_x * (u ** 0.75),
                    y0 + spread_y * (u ** 0.80),
                    h_top * u - 0.10 * (u * u),
                ))
                for u in (0.0, 0.20, 0.40, 0.60, 0.80, 1.0)
            ]
            _add_tube_along_path_bm(bm_bs, pts, r_start=0.0048, r_end=0.0008, n_seg=6, cap_ends=True, mat_index=0)
            u_drop = rng_b.uniform(0.75, 1.08)
            drop_pt = Vector((
                spread_x * (u_drop ** 0.75),
                y0 + spread_y * (u_drop ** 0.80),
                h_top * u_drop - 0.10 * (u_drop * u_drop),
            ))
            _add_icosphere_bm(bm_bs, drop_pt, radius=0.0045, subdivisions=1, mat_index=0)
        obj_bs = _create_mesh_obj(
            f"{PREFIX}Brake_Sparks_{side}", bm_bs, [mats["brake_sparks"]],
            col, parent=cab_root, location=(rail_cx, 0.0, CLAMP_Z + 0.04),
            smooth_all=True, visible_shadow=False, recalc_normals=True,
        )
        _key_visibility(obj_bs, [(1270, 1365)], total_frames)
        for f in range(1270, 1366, 2):
            g = brake_spark_gain(float(f))
            jitter = 0.90 + 0.22 * math.sin(f * 2.3 + (0.0 if side == "L" else 1.7))
            obj_bs.scale = Vector((1.0, 1.0, C.clamp(g * jitter, 0.20, 1.15)))
            obj_bs.keyframe_insert("scale", frame=f)
        _set_interp(obj_bs, "LINEAR", "scale")
        objs.append(obj_bs)
        brake_sparks.append(obj_bs)

        # Warm friction spark POINT light at clamp bite zone
        blt_data = bpy.data.lights.new(f"{PREFIX}Brake_SparkLight_{side}", type="POINT")
        blt_data.color = (1.0, 0.52, 0.10)
        blt_data.shadow_soft_size = 0.06
        blt_data.energy = 0.0
        blt = bpy.data.objects.new(f"{PREFIX}Brake_SparkLight_{side}", blt_data)
        blt.location = Vector((rail_cx - sx * 0.04, -0.08, CLAMP_Z + 0.06))
        blt.parent = cab_root
        _link_obj(blt, col)
        for f in [1, 1269] + list(range(1270, 1366, 2)) + [1366, total_frames]:
            if 1 <= f <= total_frames:
                blt_data.energy = 95.0 * brake_spark_gain(float(f))
                blt_data.keyframe_insert("energy", frame=f)
        _set_interp(blt_data, "LINEAR", "energy")
        objs.append(blt)
        brake_sparks.append(blt)

    return objs, clamps, brake_sparks


def _build_force_grid(
    col: Any,
    cab_root: Any,
    mats: dict[str, Any],
    total_frames: int,
) -> tuple[list[Any], Any]:
    """Beat 6 flush emerald-cyan 12x floor force-distribution holographic grid (z=+0.004)."""
    cx, cy, yaw = force_grid_anchor()
    bm = bmesh.new()
    hw, hd = 0.78, 0.31
    v0 = bm.verts.new((-hw, -hd, 0.0))
    v1 = bm.verts.new((hw, -hd, 0.0))
    v2 = bm.verts.new((hw, hd, 0.0))
    v3 = bm.verts.new((-hw, hd, 0.0))
    f = bm.faces.new((v0, v1, v2, v3))
    f.material_index = 0
    f.smooth = False

    obj_fg = _create_mesh_obj(
        f"{PREFIX}Force_Grid", bm, [mats["force_grid"]],
        col, parent=cab_root,
        location=(cx, cy, 0.004),
        rotation=(0.0, 0.0, yaw),
        smooth_all=False, visible_shadow=False, recalc_normals=False,
    )
    _key_visibility(obj_fg, [(GRID_REVEAL, total_frames)], total_frames)
    return [obj_fg], obj_fg


# ---------------------------------------------------------------------------
# PUBLIC ENTRYPOINT
# ---------------------------------------------------------------------------
def build_elevator_cab(col: Any, total_frames: int = C.TOTAL_FRAMES) -> dict[str, Any]:
    """
    Build and animate the full Episode #16 elevator cab, hoist cables, safety clamps,
    flashing alarm strobe, digital floor header display, and Beat 6 12x force grid.
    """
    if bpy is None:
        raise RuntimeError("build_elevator_cab() must be executed inside Blender's Python runtime.")

    t_max = max(1, int(total_frames))

    # Defensive transparent bounce floor >= 14 per AGENTS.md V11
    scene = getattr(bpy.context, "scene", None)
    cycles = getattr(scene, "cycles", None) if scene is not None else None
    if cycles is not None and hasattr(cycles, "transparent_max_bounces"):
        cycles.transparent_max_bounces = max(14, int(cycles.transparent_max_bounces))

    # 1. Cab_Root trajectory (LINEAR keyframes 1..total_frames exact to C.cab_world_location)
    cab_root = _new_empty(f"{PREFIX}Root", col, location=C.cab_world_location(1.0), display_size=0.25)
    for f in range(1, t_max + 1):
        cab_root.location = Vector(C.cab_world_location(float(f)))
        cab_root.keyframe_insert("location", frame=f)
    _set_interp(cab_root, "LINEAR", "location")

    # 2. Local-space materials
    mats, ctrl = _build_materials(cab_root)
    all_objs: list[Any] = [cab_root]

    # 3. Structural & interior geometry
    all_objs.extend(_build_floor_and_chassis(col, cab_root, mats))
    rw_objs, handrail = _build_rear_wall_and_handrail(col, cab_root, mats)
    all_objs.extend(rw_objs)
    sw_objs, cop_panel = _build_side_walls_and_cop(col, cab_root, mats)
    all_objs.extend(sw_objs)
    dc_objs, doors, ceiling_lights = _build_doors_and_ceiling(col, cab_root, mats)
    all_objs.extend(dc_objs)
    hd_objs, floor_led, floor_led_readouts, alarm_light = _build_header_display_and_alarm(
        col, cab_root, mats, t_max
    )
    all_objs.extend(hd_objs)
    cb_objs, cables, snap_sparks, _ = _build_crosshead_and_cables(col, cab_root, mats, t_max)
    all_objs.extend(cb_objs)
    cl_objs, clamps, brake_sparks = _build_safety_clamps(col, cab_root, mats, t_max)
    all_objs.extend(cl_objs)
    fg_objs, force_grid = _build_force_grid(col, cab_root, mats, t_max)
    all_objs.extend(fg_objs)

    # 4. Animate shader value nodes & lights across 1..total_frames
    xray_frames = _sample_frames(
        t_max,
        dense_windows=[(338, 368), (572, 598), (902, 938), (1205, 1238), (1532, 1568)],
        step=4,
    )
    for xnode in ctrl["xray_nodes"]:
        _key_value_node(xnode, xray_frames, C.xray_cab_weight, interp="LINEAR")

    alarm_frames = _sample_frames(t_max, dense_windows=[(ALARM_START, 1237)], step=12)
    _key_value_node(
        ctrl["alarm_gain"],
        alarm_frames,
        lambda f: 1.95 * alarm_pulse_factor(f),
        interp="LINEAR",
    )
    alm_lt_data = alarm_light.data
    for f in alarm_frames:
        alm_lt_data.energy = 72.0 * alarm_pulse_factor(float(f))
        alm_lt_data.keyframe_insert("energy", frame=f)
    _set_interp(alm_lt_data, "LINEAR", "energy")

    ceil_frames = _sample_frames(t_max, dense_windows=[(322, 358)], step=8)
    _key_value_node(
        ctrl["ceiling_glow"],
        ceil_frames,
        lambda f: 1.55 * ceiling_glow_factor(f),
        interp="LINEAR",
    )
    ceil_lt_data = ceiling_lights[-1].data
    for f in ceil_frames:
        ceil_lt_data.energy = 26.0 * ceiling_glow_factor(float(f))
        ceil_lt_data.keyframe_insert("energy", frame=f)
    _set_interp(ceil_lt_data, "LINEAR", "energy")

    snap_frames = _sample_frames(t_max, dense_windows=[(CABLE_SNAP - 1, 366)], step=12)
    _key_value_node(ctrl["snap_spark_gain"], snap_frames, snap_spark_gain, interp="LINEAR")

    brake_frames = _sample_frames(t_max, dense_windows=[(1269, 1366)], step=12)
    _key_value_node(ctrl["brake_spark_gain"], brake_frames, brake_spark_gain, interp="LINEAR")

    grid_frames = _sample_frames(t_max, dense_windows=[(GRID_REVEAL - 1, 1582)], step=12)
    _key_value_node(ctrl["grid_fade"], grid_frames, force_grid_fade, interp="LINEAR")

    return {
        "root": cab_root,
        "xray_nodes": ctrl["xray_nodes"],
        "alarm_light": alarm_light,
        "cables": cables,
        "snap_sparks": snap_sparks,
        "clamps": clamps,
        "safety_clamps": clamps,
        "brake_sparks": brake_sparks,
        "floor_led": floor_led,
        "floor_led_readouts": floor_led_readouts,
        "force_grid": force_grid,
        "doors": doors,
        "handrail": handrail,
        "cop_panel": cop_panel,
        "ceiling_lights": ceiling_lights,
        "materials": mats,
        "all_objs": all_objs,
    }


# ---------------------------------------------------------------------------
# PURE-PYTHON SELF-CHECK (when invoked outside Blender)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    assert clamp_engage_factor(1260.0) == 0.0, "Clamps must stay open (8 mm clearance) through f=1260"
    assert clamp_engage_factor(1275.0) >= 0.99, "Clamps must engage by f=1275"
    assert alarm_pulse_factor(109.0) == 0.0 and alarm_pulse_factor(112.0) > 0.5
    assert snap_spark_gain(324.0) == 0.0 and snap_spark_gain(327.0) > 0.8 and snap_spark_gain(366.0) == 0.0
    assert brake_spark_gain(1269.0) == 0.0 and brake_spark_gain(1285.0) > 0.5 and brake_spark_gain(1366.0) == 0.0
    assert force_grid_fade(1544.0) == 0.0 and abs(force_grid_fade(1580.0) - 1.0) < 1e-6

    sched = floor_led_schedule(C.TOTAL_FRAMES)
    covered = []
    for _, _, s, e in sched:
        covered.extend(range(s, e + 1))
    assert covered == list(range(1, C.TOTAL_FRAMES + 1)), "Floor LED schedule must cover 1..1800 seamlessly"
    cx, cy, yaw = force_grid_anchor()
    print(f"[OK] elevator_cab_asset self-check passed (force_grid=({cx:.3f}, {cy:.3f}), yaw={math.degrees(yaw):.2f} deg)")
