"""
canopy_fx.py — Teardrop Bubble Canopy, Zigzag MDC Detonation Wave & Shatter Shards
Episode #1: "Why Ejecting From a Fighter Jet Shrinks You by 1 Inch" (60 fps, 1800 frames).

Owned by the Canopy & MDC Shatter FX specialist.
All timings, dimensions, and event frames are imported directly from `ejection_contract`.

Public API:
    build_canopy(col: bpy.types.Collection, jet_root: bpy.types.Object, total_frames: int = 1800) -> dict

Returned dict schema (every value is a bpy.types.Object or list[bpy.types.Object]):
    {
        "root": bpy.types.Object,                   # Canopy_Root (parented to jet_root)
        "intact_objs": list[bpy.types.Object],      # [Canopy_Glass_Intact, Canopy_Frame_Intact]
        "shards": list[bpy.types.Object],           # 48 beveled shards: Canopy_Shard_00..47
        "mdc_cord": bpy.types.Object,               # Canopy_MDC_Cord
        "flash_objs": list[bpy.types.Object],       # [Canopy_Detonation_Flash, Canopy_MDC_Flash_Ribbon, Canopy_MDC_SparkOrb]
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
from mathutils import Euler, Vector

SCRIPT_DIR = Path(__file__).resolve().parent
EPISODE_DIR = SCRIPT_DIR.parent
REPO_ROOT = EPISODE_DIR.parent.parent
for _p in (str(SCRIPT_DIR), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ejection_contract as C  # noqa: E402

# ---------------------------------------------------------------------------
# CONTRACT-DERIVED GEOMETRY & TIMING CONSTANTS
# ---------------------------------------------------------------------------
Y_MIN, Y_MAX = float(C.CANOPY["y_range"][0]), float(C.CANOPY["y_range"][1])  # (-1.40, 1.60)
HALF_WIDTH = float(C.CANOPY["half_width"])                                  # 0.48
APEX_X, APEX_Y, APEX_Z = (float(v) for v in C.CANOPY["apex"])               # (0.0, 0.15, 1.25)
SILL_Z = float(C.CANOPY["sill_z"])                                          # 0.55
DOME_HEIGHT = APEX_Z - SILL_Z                                               # 0.70

WINDSHIELD_BOW_Y = 1.42
REAR_ARCH_Y = -1.26

MDC_Y_FORE = 1.25
MDC_Y_AFT = -1.15
MDC_AMPLITUDE_X = 0.15
MDC_TUBE_RADIUS = 0.0046
MDC_INWARD_OFFSET = 0.0068

SHARD_COUNT_U = 8
SHARD_COUNT_V = 6   # 8 * 6 = 48 beveled canopy shards (within 42..52)
SHARD_THICKNESS = 0.0115
SHARD_CHAMFER_INSET = 0.0016

PREFIX = "Canopy_"


# ---------------------------------------------------------------------------
# PARAMETRIC TEARDROP CANOPY SURFACE MATH
# ---------------------------------------------------------------------------
def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _canopy_envelope(y: float) -> tuple[float, float]:
    """Return (half_width, dome_height) at jet-local longitudinal coordinate y."""
    if y >= APEX_Y:
        span = max(1e-6, Y_MAX - APEX_Y)
        s = min(1.0, max(0.0, (y - APEX_Y) / span))
        s2 = s * s
        w = HALF_WIDTH * ((1.0 - s2) ** 0.36)
        h = DOME_HEIGHT * ((1.0 - s2) ** 0.58)
    else:
        span = max(1e-6, APEX_Y - Y_MIN)
        s = min(1.0, max(0.0, (APEX_Y - y) / span))
        s2 = s * s
        w = HALF_WIDTH * ((1.0 - s2) ** 0.46)
        h = DOME_HEIGHT * ((1.0 - s2) ** 0.56)
    return w, h


def _canopy_point_uv(u: float, v: float) -> Vector:
    """
    Evaluate 3D point on the sleek teardrop canopy surface in Canopy_Root local space.
    u in [0, 1] -> y in [Y_MIN, Y_MAX] (-1.40 .. +1.60)
    v in [0, 1] -> theta in [0, pi] (0 = +X right sill, 0.5 = crown, 1 = -X left sill).
    """
    u_c = min(1.0, max(0.0, u))
    v_c = min(1.0, max(0.0, v))
    y = _lerp(Y_MIN, Y_MAX, u_c)
    w, h = _canopy_envelope(y)
    theta = v_c * math.pi
    ct = math.cos(theta)
    st = max(0.0, math.sin(theta))
    px = APEX_X + w * math.copysign(abs(ct) ** 0.92, ct)
    pz = SILL_Z + h * (st ** 0.92)
    return Vector((px, y, pz))


def _canopy_normal_uv(u: float, v: float) -> Vector:
    """Outward unit normal at parametric canopy coordinates (u, v)."""
    eps = 1e-3
    u0, u1 = max(0.0, u - eps), min(1.0, u + eps)
    v0, v1 = max(0.0, v - eps), min(1.0, v + eps)
    dp_du = _canopy_point_uv(u1, v) - _canopy_point_uv(u0, v)
    dp_dv = _canopy_point_uv(u, v1) - _canopy_point_uv(u, v0)
    n = dp_du.cross(dp_dv)
    if n.length < 1e-8:
        return Vector((0.0, 0.0, 1.0))
    return n.normalized()


def _canopy_point_xy(x: float, y: float) -> tuple[Vector, Vector]:
    """Map crown-region (x, y) onto the canopy dome, returning (surface_pt, outward_normal)."""
    u = (y - Y_MIN) / max(1e-6, Y_MAX - Y_MIN)
    w, _ = _canopy_envelope(y)
    if w < 1e-5:
        v = 0.5
    else:
        ratio = max(-0.98, min(0.98, x / w))
        ct = math.copysign(abs(ratio) ** (1.0 / 0.92), ratio)
        ct = max(-1.0, min(1.0, ct))
        v = math.acos(ct) / math.pi
    return _canopy_point_uv(u, v), _canopy_normal_uv(u, v)


# ---------------------------------------------------------------------------
# BLENDER & BMESH HELPERS
# ---------------------------------------------------------------------------
def _link_obj(obj: bpy.types.Object, col: bpy.types.Collection) -> bpy.types.Object:
    if obj.name not in col.objects:
        col.objects.link(obj)
    return obj


def _create_mesh_obj(
    name: str,
    bm: bmesh.types.BMesh,
    materials: list[bpy.types.Material] | bpy.types.Material,
    col: bpy.types.Collection,
    parent: bpy.types.Object,
    location: tuple[float, float, float] | Vector = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float] | Euler = (0.0, 0.0, 0.0),
    smooth_all: bool | None = True,
    visible_shadow: bool = False,
    recalc_normals: bool = True,
) -> bpy.types.Object:
    """Attach material slots BEFORE bm.to_mesh(mesh) per AGENTS.md Section 3."""
    mesh = bpy.data.meshes.new(name)
    mat_list = materials if isinstance(materials, list) else [materials]
    for m in mat_list:
        mesh.materials.append(m)

    if recalc_normals:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()

    if smooth_all is not None and len(mesh.polygons) > 0:
        mesh.polygons.foreach_set("use_smooth", [smooth_all] * len(mesh.polygons))
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(location)
    obj.rotation_euler = Euler(rotation) if not isinstance(rotation, Euler) else rotation.copy()
    obj.parent = parent
    obj.visible_shadow = visible_shadow
    _link_obj(obj, col)
    return obj


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
    """Keyframe CONSTANT hide_viewport / hide_render so detached/hidden objects never reappear."""
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
        is_vis = any(s <= f <= e for s, e in visible_ranges)
        obj.hide_viewport = not is_vis
        obj.hide_render = not is_vis
        obj.keyframe_insert("hide_viewport", frame=f)
        obj.keyframe_insert("hide_render", frame=f)
    _set_interp(obj, "CONSTANT", "hide_")


def _sweep_chamfer_profile(
    bm: bmesh.types.BMesh,
    pts: list[Vector],
    up_dirs: list[Vector],
    half_w: float,
    half_h: float,
    chamfer: float,
    closed: bool = False,
) -> None:
    """Sweep an 8-point chamfered rectangular rail cross-section into an existing BMesh."""
    ch = min(chamfer, half_w * 0.42, half_h * 0.42)
    profile = [
        (half_w - ch, -half_h),
        (half_w, -half_h + ch),
        (half_w, half_h - ch),
        (half_w - ch, half_h),
        (-half_w + ch, half_h),
        (-half_w, half_h - ch),
        (-half_w, -half_h + ch),
        (-half_w + ch, -half_h),
    ]
    n_pts = len(pts)
    rings: list[list[bmesh.types.BMVert]] = []
    for i, p in enumerate(pts):
        if closed:
            p_prev = pts[(i - 1) % n_pts]
            p_next = pts[(i + 1) % n_pts]
            tang = p_next - p_prev
        elif i == 0:
            tang = pts[1] - pts[0]
        elif i == n_pts - 1:
            tang = pts[-1] - pts[-2]
        else:
            tang = pts[i + 1] - pts[i - 1]
        tang = tang.normalized() if tang.length > 1e-8 else Vector((0.0, 1.0, 0.0))

        up = up_dirs[i]
        up_ortho = up - tang * up.dot(tang)
        if up_ortho.length < 1e-6:
            ref = Vector((0.0, 0.0, 1.0)) if abs(tang.z) < 0.9 else Vector((1.0, 0.0, 0.0))
            up_ortho = (ref - tang * ref.dot(tang)).normalized()
        else:
            up_ortho.normalize()
        side = tang.cross(up_ortho).normalized()
        rings.append([bm.verts.new(p + side * sx + up_ortho * sy) for sx, sy in profile])

    n_prof = len(profile)
    seg_count = n_pts if closed else n_pts - 1
    for i in range(seg_count):
        inext = (i + 1) % n_pts if closed else i + 1
        for k in range(n_prof):
            kn = (k + 1) % n_prof
            f = bm.faces.new((rings[i][k], rings[i][kn], rings[inext][kn], rings[inext][k]))
            f.smooth = True

    if not closed and len(rings) >= 2:
        bm.faces.new(rings[0][::-1])
        bm.faces.new(rings[-1])


def _make_open_tube_bm(
    points: list[Vector],
    radius: float,
    n_seg: int = 12,
    cap_ends: bool = False,
) -> bmesh.types.BMesh:
    """Sweep a smooth circular tube along a 3D polyline (default open-ended)."""
    bm = bmesh.new()
    rings: list[list[bmesh.types.BMVert]] = []
    up_ref = Vector((0.0, 0.0, 1.0))

    for idx, pt in enumerate(points):
        if idx == 0:
            tang = (points[1] - points[0]).normalized()
        elif idx == len(points) - 1:
            tang = (points[-1] - points[-2]).normalized()
        else:
            t1 = (points[idx] - points[idx - 1]).normalized()
            t2 = (points[idx + 1] - points[idx]).normalized()
            tang = t1 + t2
            tang = tang.normalized() if tang.length > 1e-6 else t1

        ref = up_ref if abs(tang.dot(up_ref)) < 0.92 else Vector((1.0, 0.0, 0.0))
        b1 = tang.cross(ref).normalized()
        b2 = tang.cross(b1).normalized()
        ring = [
            bm.verts.new(pt + (b1 * math.cos(k * 2.0 * math.pi / n_seg) + b2 * math.sin(k * 2.0 * math.pi / n_seg)) * radius)
            for k in range(n_seg)
        ]
        rings.append(ring)

    for i in range(len(rings) - 1):
        for k in range(n_seg):
            kn = (k + 1) % n_seg
            f = bm.faces.new((rings[i][k], rings[i][kn], rings[i + 1][kn], rings[i + 1][k]))
            f.smooth = True

    if cap_ends and len(rings) >= 2:
        bm.faces.new(rings[0][::-1])
        bm.faces.new(rings[-1])

    return bm


def _build_mdc_polyline() -> list[Vector]:
    """
    Build the smooth zigzag MDC cord centerline hugging the inside crown of the canopy
    from +1.25 (fore) to -1.15 (aft) with amplitude +-0.15 m.
    """
    n_peaks = 9
    anchors_2d: list[tuple[float, float]] = [(0.0, MDC_Y_FORE)]
    for i in range(n_peaks):
        frac = (i + 0.5) / n_peaks
        y = _lerp(MDC_Y_FORE - 0.04, MDC_Y_AFT + 0.04, frac)
        sx = MDC_AMPLITUDE_X if (i % 2 == 0) else -MDC_AMPLITUDE_X
        anchors_2d.append((sx, y))
    anchors_2d.append((0.0, MDC_Y_AFT))

    # Slightly bevel/fillet each zigzag apex in 2D so the cord bends crisply without kinks
    dense_2d: list[tuple[float, float]] = [anchors_2d[0]]
    sub_per_leg = 8
    for i in range(len(anchors_2d) - 1):
        x0, y0 = anchors_2d[i]
        x1, y1 = anchors_2d[i + 1]
        for s in range(1, sub_per_leg + 1):
            t = s / sub_per_leg
            # Smooth small rounding near apexes while keeping clean straight diagonal runs
            tx = 0.5 - 0.5 * math.cos(t * math.pi) if (i == 0 and t < 0.25) else t
            dense_2d.append((_lerp(x0, x1, tx), _lerp(y0, y1, t)))

    # Project onto inner dome surface
    pts_3d: list[Vector] = []
    for x, y in dense_2d:
        surf_p, surf_n = _canopy_point_xy(x, y)
        pts_3d.append(surf_p - surf_n * MDC_INWARD_OFFSET)
    return pts_3d


def _sample_polyline_by_y(pts: list[Vector], y_target: float) -> Vector:
    """Evaluate exact 3D point on the monotonically decreasing-Y zigzag polyline."""
    if y_target >= pts[0].y:
        return pts[0].copy()
    if y_target <= pts[-1].y:
        return pts[-1].copy()
    for i in range(len(pts) - 1):
        y0, y1 = pts[i].y, pts[i + 1].y
        if y0 >= y_target >= y1:
            span = max(1e-8, y0 - y1)
            t = (y0 - y_target) / span
            return pts[i].lerp(pts[i + 1], t)
    return pts[-1].copy()


# ---------------------------------------------------------------------------
# LOCAL-SPACE SHADERS & MATERIALS
# ---------------------------------------------------------------------------
def _wire_camera_front_mask(nt: bpy.types.NodeTree) -> bpy.types.NodeSocket:
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
    nt: bpy.types.NodeTree,
    surface_shader_sock: bpy.types.NodeSocket,
    transp_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> bpy.types.NodeSocket:
    """Route LightPath.Is Shadow Ray -> Transparent BSDF per AGENTS.md Section 3."""
    lp = nt.nodes.new("ShaderNodeLightPath")
    tr_shad = nt.nodes.new("ShaderNodeBsdfTransparent")
    tr_shad.inputs["Color"].default_value = transp_color

    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Shadow Ray"], mix.inputs["Fac"])
    nt.links.new(surface_shader_sock, mix.inputs[1])
    nt.links.new(tr_shad.outputs["BSDF"], mix.inputs[2])
    return mix.outputs["Shader"]


def _build_materials(canopy_root: bpy.types.Object) -> tuple[dict[str, bpy.types.Material], dict[str, Any]]:
    mats: dict[str, bpy.types.Material] = {}
    ctrl: dict[str, Any] = {}

    # 1. Beat 1 Crystal-Clear Glass Shader:
    # LayerWeight(Fresnel, Blend=0.22) mixing Transparent BSDF with crisp Glossy BSDF(0.018)
    # ONLY on Camera Ray * (1 - Backfacing), with Is Shadow Ray -> Transparent BSDF.
    m_glass = bpy.data.materials.new(f"{PREFIX}Glass_Intact_Mat")
    m_glass.use_nodes = True
    nt = m_glass.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    tr.inputs["Color"].default_value = (0.99, 0.995, 1.0, 1.0)

    glossy = nt.nodes.new("ShaderNodeBsdfGlossy")
    glossy.inputs["Color"].default_value = (0.94, 0.97, 1.0, 1.0)
    glossy.inputs["Roughness"].default_value = 0.018

    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.22

    cam_front = _wire_camera_front_mask(nt)

    fresnel_cam = nt.nodes.new("ShaderNodeMath")
    fresnel_cam.operation = "MULTIPLY"
    nt.links.new(lw.outputs["Fresnel"], fresnel_cam.inputs[0])
    nt.links.new(cam_front, fresnel_cam.inputs[1])

    mix_glass = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(fresnel_cam.outputs["Value"], mix_glass.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix_glass.inputs[1])
    nt.links.new(glossy.outputs["BSDF"], mix_glass.inputs[2])

    surf_out = _wire_shadow_bypass(nt, mix_glass.outputs["Shader"], (1.0, 1.0, 1.0, 1.0))
    nt.links.new(surf_out, out.inputs["Surface"])
    mats["glass_intact"] = m_glass

    # 2. Dark Gunmetal Canopy Frame Rails & Bow
    m_frame = bpy.data.materials.new(f"{PREFIX}Frame_Gunmetal_Mat")
    m_frame.use_nodes = True
    nt = m_frame.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.045, 0.052, 0.064, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.88
    bsdf.inputs["Roughness"].default_value = 0.30
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = canopy_root
    nf = nt.nodes.new("ShaderNodeTexNoise")
    nf.inputs["Scale"].default_value = 38.0
    nf.inputs["Detail"].default_value = 6.0
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.08
    bmp.inputs["Distance"].default_value = 0.0004
    nt.links.new(tc.outputs["Object"], nf.inputs["Vector"])
    nt.links.new(nf.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    mats["frame_gunmetal"] = m_frame

    # 3. MDC Cord Local-Space Detonation Wave Shader (`TexCoord.Object` -> Canopy_Root)
    m_mdc = bpy.data.materials.new(f"{PREFIX}MDC_Cord_Mat")
    m_mdc.use_nodes = True
    nt = m_mdc.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = canopy_root
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])

    wave_y = nt.nodes.new("ShaderNodeValue")
    wave_y.name = "WaveY"
    wave_y.outputs[0].default_value = MDC_Y_FORE + 0.08
    ctrl["mdc_wave_y"] = wave_y

    wave_on = nt.nodes.new("ShaderNodeValue")
    wave_on.name = "WaveOn"
    wave_on.outputs[0].default_value = 0.0
    ctrl["mdc_wave_on"] = wave_on

    # diff = Y - WaveY (positive behind wave, 0 right at wave front)
    diff = nt.nodes.new("ShaderNodeMath")
    diff.operation = "SUBTRACT"
    nt.links.new(sep.outputs["Y"], diff.inputs[0])
    nt.links.new(wave_y.outputs[0], diff.inputs[1])

    abs_d = nt.nodes.new("ShaderNodeMath")
    abs_d.operation = "ABSOLUTE"
    nt.links.new(diff.outputs["Value"], abs_d.inputs[0])

    # Crisp cyan-white wave front pulse (within +-0.12 m of WaveY)
    front_mask = nt.nodes.new("ShaderNodeMapRange")
    front_mask.clamp = True
    front_mask.inputs["From Min"].default_value = 0.12
    front_mask.inputs["From Max"].default_value = 0.0
    nt.links.new(abs_d.outputs["Value"], front_mask.inputs["Value"])

    # Hot trailing amber detonation wake behind wave front (diff >= 0)
    wake_gate = nt.nodes.new("ShaderNodeMapRange")
    wake_gate.clamp = True
    wake_gate.inputs["From Min"].default_value = -0.04
    wake_gate.inputs["From Max"].default_value = 0.03
    nt.links.new(diff.outputs["Value"], wake_gate.inputs["Value"])

    wake_decay = nt.nodes.new("ShaderNodeMapRange")
    wake_decay.clamp = True
    wake_decay.inputs["From Min"].default_value = 0.0
    wake_decay.inputs["From Max"].default_value = 0.85
    wake_decay.inputs["To Min"].default_value = 1.0
    wake_decay.inputs["To Max"].default_value = 0.22
    nt.links.new(diff.outputs["Value"], wake_decay.inputs["Value"])

    wake_mask = nt.nodes.new("ShaderNodeMath")
    wake_mask.operation = "MULTIPLY"
    nt.links.new(wake_gate.outputs["Result"], wake_mask.inputs[0])
    nt.links.new(wake_decay.outputs["Result"], wake_mask.inputs[1])

    # Color blends hot amber wake -> blazing cyan-white front
    col_mix = nt.nodes.new("ShaderNodeMix")
    col_mix.data_type = "RGBA"
    col_mix.inputs[6].default_value = (1.0, 0.44, 0.04, 1.0)
    col_mix.inputs[7].default_value = (0.55, 0.94, 1.0, 1.0)
    nt.links.new(front_mask.outputs["Result"], col_mix.inputs["Factor"])

    # Strength = WaveOn * (2.6 * wake_mask + 5.2 * front_mask)
    w_str = nt.nodes.new("ShaderNodeMath")
    w_str.operation = "MULTIPLY"
    w_str.inputs[1].default_value = 2.6
    nt.links.new(wake_mask.outputs["Value"], w_str.inputs[0])

    f_str = nt.nodes.new("ShaderNodeMath")
    f_str.operation = "MULTIPLY"
    f_str.inputs[1].default_value = 5.2
    nt.links.new(front_mask.outputs["Result"], f_str.inputs[0])

    sum_str = nt.nodes.new("ShaderNodeMath")
    sum_str.operation = "ADD"
    nt.links.new(w_str.outputs["Value"], sum_str.inputs[0])
    nt.links.new(f_str.outputs["Value"], sum_str.inputs[1])

    final_str = nt.nodes.new("ShaderNodeMath")
    final_str.operation = "MULTIPLY"
    nt.links.new(sum_str.outputs["Value"], final_str.inputs[0])
    nt.links.new(wave_on.outputs[0], final_str.inputs[1])

    bsdf_mdc = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf_mdc.inputs["Base Color"].default_value = (0.34, 0.37, 0.42, 1.0)
    bsdf_mdc.inputs["Metallic"].default_value = 0.88
    bsdf_mdc.inputs["Roughness"].default_value = 0.28
    nt.links.new(col_mix.outputs[2], bsdf_mdc.inputs["Emission Color"])
    nt.links.new(final_str.outputs["Value"], bsdf_mdc.inputs["Emission Strength"])

    nt.links.new(bsdf_mdc.outputs["BSDF"], out.inputs["Surface"])
    mats["mdc_cord"] = m_mdc

    # 4. Shard Glass Face Material (Local-space shard TexCoord.Object + craze cracks + shadow bypass)
    m_shard_glass = bpy.data.materials.new(f"{PREFIX}Shard_Glass_Mat")
    m_shard_glass.use_nodes = True
    nt = m_shard_glass.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    tc_s = nt.nodes.new("ShaderNodeTexCoord")
    # TexCoord.Object with object=None evaluates each tumbling shard's local coordinates
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.feature = "DISTANCE_TO_EDGE"
    vor.inputs["Scale"].default_value = 26.0
    nt.links.new(tc_s.outputs["Object"], vor.inputs["Vector"])

    craze_mask = nt.nodes.new("ShaderNodeMapRange")
    craze_mask.clamp = True
    craze_mask.inputs["From Min"].default_value = 0.040
    craze_mask.inputs["From Max"].default_value = 0.0
    nt.links.new(vor.outputs["Distance"], craze_mask.inputs["Value"])

    tr_s = nt.nodes.new("ShaderNodeBsdfTransparent")
    tr_s.inputs["Color"].default_value = (0.90, 0.96, 1.0, 1.0)

    gl_s = nt.nodes.new("ShaderNodeBsdfGlossy")
    gl_s.inputs["Color"].default_value = (0.94, 0.98, 1.0, 1.0)
    gl_s.inputs["Roughness"].default_value = 0.025

    lw_s = nt.nodes.new("ShaderNodeLayerWeight")
    lw_s.inputs["Blend"].default_value = 0.34

    mix_sg = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lw_s.outputs["Fresnel"], mix_sg.inputs["Fac"])
    nt.links.new(tr_s.outputs["BSDF"], mix_sg.inputs[1])
    nt.links.new(gl_s.outputs["BSDF"], mix_sg.inputs[2])

    # Delicate luminous internal fracture craze hairlines (AgX-safe)
    craze_em = nt.nodes.new("ShaderNodeEmission")
    craze_em.inputs["Color"].default_value = (0.48, 0.92, 1.0, 1.0)
    craze_gain = nt.nodes.new("ShaderNodeValue")
    craze_gain.name = "ShardCrazeGain"
    craze_gain.outputs[0].default_value = 0.85
    ctrl["shard_craze_gain"] = craze_gain

    craze_str = nt.nodes.new("ShaderNodeMath")
    craze_str.operation = "MULTIPLY"
    nt.links.new(craze_mask.outputs["Result"], craze_str.inputs[0])
    nt.links.new(craze_gain.outputs[0], craze_str.inputs[1])
    nt.links.new(craze_str.outputs["Value"], craze_em.inputs["Strength"])

    add_c = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(mix_sg.outputs["Shader"], add_c.inputs[0])
    nt.links.new(craze_em.outputs["Emission"], add_c.inputs[1])

    surf_sg = _wire_shadow_bypass(nt, add_c.outputs["Shader"], (0.96, 0.99, 1.0, 1.0))
    nt.links.new(surf_sg, out.inputs["Surface"])
    mats["shard_glass"] = m_shard_glass

    # 5. Shard Beveled Fracture Rim Veins (`TexCoord.Object` glowing cyan/amber veins!)
    m_rim = bpy.data.materials.new(f"{PREFIX}Shard_FractureRim_Mat")
    m_rim.use_nodes = True
    nt = m_rim.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    tc_r = nt.nodes.new("ShaderNodeTexCoord")
    nf_r = nt.nodes.new("ShaderNodeTexNoise")
    nf_r.inputs["Scale"].default_value = 42.0
    nf_r.inputs["Detail"].default_value = 7.0
    nf_r.inputs["Roughness"].default_value = 0.68
    nt.links.new(tc_r.outputs["Object"], nf_r.inputs["Vector"])

    vein_mask = nt.nodes.new("ShaderNodeMapRange")
    vein_mask.clamp = True
    vein_mask.inputs["From Min"].default_value = 0.44
    vein_mask.inputs["From Max"].default_value = 0.68
    nt.links.new(nf_r.outputs["Fac"], vein_mask.inputs["Value"])

    vein_ramp = nt.nodes.new("ShaderNodeValToRGB")
    vein_ramp.color_ramp.elements[0].position = 0.15
    vein_ramp.color_ramp.elements[0].color = (0.14, 0.88, 1.0, 1.0)
    vein_ramp.color_ramp.elements[1].position = 0.82
    vein_ramp.color_ramp.elements[1].color = (1.0, 0.56, 0.08, 1.0)
    nt.links.new(nf_r.outputs["Fac"], vein_ramp.inputs["Fac"])

    vein_gain = nt.nodes.new("ShaderNodeValue")
    vein_gain.name = "FractureVeinGain"
    vein_gain.outputs[0].default_value = 3.2
    ctrl["shard_vein_gain"] = vein_gain

    vein_str = nt.nodes.new("ShaderNodeMath")
    vein_str.operation = "MULTIPLY"
    nt.links.new(vein_mask.outputs["Result"], vein_str.inputs[0])
    nt.links.new(vein_gain.outputs[0], vein_str.inputs[1])

    tr_r = nt.nodes.new("ShaderNodeBsdfTransparent")
    tr_r.inputs["Color"].default_value = (0.78, 0.94, 1.0, 1.0)

    gl_r = nt.nodes.new("ShaderNodeBsdfGlossy")
    gl_r.inputs["Color"].default_value = (0.85, 0.97, 1.0, 1.0)
    gl_r.inputs["Roughness"].default_value = 0.06

    mix_r = nt.nodes.new("ShaderNodeMixShader")
    mix_r.inputs["Fac"].default_value = 0.58
    nt.links.new(tr_r.outputs["BSDF"], mix_r.inputs[1])
    nt.links.new(gl_r.outputs["BSDF"], mix_r.inputs[2])

    em_r = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(vein_ramp.outputs["Color"], em_r.inputs["Color"])
    nt.links.new(vein_str.outputs["Value"], em_r.inputs["Strength"])

    add_r = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(mix_r.outputs["Shader"], add_r.inputs[0])
    nt.links.new(em_r.outputs["Emission"], add_r.inputs[1])

    surf_r = _wire_shadow_bypass(nt, add_r.outputs["Shader"], (0.88, 0.97, 1.0, 1.0))
    nt.links.new(surf_r, out.inputs["Surface"])
    mats["shard_rim"] = m_rim

    # 6. Open-Ended Additive FX Shaders (Camera Ray * (1 - Backfacing), (1 - Facing)^p * Noise)
    def make_additive_plasma_mat(
        mat_name: str,
        gain_key: str,
        noise_scale: float,
        falloff_power: float,
        c_edge: tuple[float, float, float],
        c_core: tuple[float, float, float],
        init_gain: float,
    ) -> bpy.types.Material:
        mat = bpy.data.materials.new(f"{PREFIX}{mat_name}")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")

        tr_fx = nt.nodes.new("ShaderNodeBsdfTransparent")
        em_fx = nt.nodes.new("ShaderNodeEmission")

        lw_fx = nt.nodes.new("ShaderNodeLayerWeight")
        lw_fx.inputs["Blend"].default_value = 0.50
        inv_f = nt.nodes.new("ShaderNodeMath")
        inv_f.operation = "SUBTRACT"
        inv_f.inputs[0].default_value = 1.0
        nt.links.new(lw_fx.outputs["Facing"], inv_f.inputs[1])

        pow_f = nt.nodes.new("ShaderNodeMath")
        pow_f.operation = "POWER"
        pow_f.inputs[1].default_value = falloff_power
        nt.links.new(inv_f.outputs["Value"], pow_f.inputs[0])

        tc_fx = nt.nodes.new("ShaderNodeTexCoord")
        tc_fx.object = canopy_root
        nf_fx = nt.nodes.new("ShaderNodeTexNoise")
        nf_fx.inputs["Scale"].default_value = noise_scale
        nf_fx.inputs["Detail"].default_value = 6.0
        nf_fx.inputs["Roughness"].default_value = 0.62
        nt.links.new(tc_fx.outputs["Object"], nf_fx.inputs["Vector"])

        n_map = nt.nodes.new("ShaderNodeMapRange")
        n_map.clamp = True
        n_map.inputs["From Min"].default_value = 0.28
        n_map.inputs["From Max"].default_value = 0.78
        n_map.inputs["To Min"].default_value = 0.18
        n_map.inputs["To Max"].default_value = 1.0
        nt.links.new(nf_fx.outputs["Fac"], n_map.inputs["Value"])

        dens = nt.nodes.new("ShaderNodeMath")
        dens.operation = "MULTIPLY"
        nt.links.new(pow_f.outputs["Value"], dens.inputs[0])
        nt.links.new(n_map.outputs["Result"], dens.inputs[1])

        ramp_fx = nt.nodes.new("ShaderNodeValToRGB")
        ramp_fx.color_ramp.elements[0].position = 0.10
        ramp_fx.color_ramp.elements[0].color = (*c_edge, 1.0)
        ramp_fx.color_ramp.elements[1].position = 0.85
        ramp_fx.color_ramp.elements[1].color = (*c_core, 1.0)
        nt.links.new(dens.outputs["Value"], ramp_fx.inputs["Fac"])
        nt.links.new(ramp_fx.outputs["Color"], em_fx.inputs["Color"])

        gain_val = nt.nodes.new("ShaderNodeValue")
        gain_val.name = gain_key
        gain_val.outputs[0].default_value = init_gain
        ctrl[gain_key] = gain_val

        cam_front = _wire_camera_front_mask(nt)
        g_mul = nt.nodes.new("ShaderNodeMath")
        g_mul.operation = "MULTIPLY"
        nt.links.new(dens.outputs["Value"], g_mul.inputs[0])
        nt.links.new(gain_val.outputs[0], g_mul.inputs[1])

        c_mul = nt.nodes.new("ShaderNodeMath")
        c_mul.operation = "MULTIPLY"
        nt.links.new(g_mul.outputs["Value"], c_mul.inputs[0])
        nt.links.new(cam_front, c_mul.inputs[1])
        nt.links.new(c_mul.outputs["Value"], em_fx.inputs["Strength"])

        add_fx = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(tr_fx.outputs["BSDF"], add_fx.inputs[0])
        nt.links.new(em_fx.outputs["Emission"], add_fx.inputs[1])
        nt.links.new(add_fx.outputs["Shader"], out.inputs["Surface"])
        return mat

    mats["detonation_flash"] = make_additive_plasma_mat(
        "Detonation_Flash_Mat", "flash_gain",
        noise_scale=9.5, falloff_power=2.2,
        c_edge=(1.0, 0.62, 0.14), c_core=(0.62, 0.96, 1.0),
        init_gain=3.6,
    )
    mats["flash_ribbon"] = make_additive_plasma_mat(
        "MDC_Flash_Ribbon_Mat", "ribbon_gain",
        noise_scale=14.0, falloff_power=1.8,
        c_edge=(1.0, 0.48, 0.08), c_core=(0.56, 0.95, 1.0),
        init_gain=4.0,
    )
    mats["spark_orb"] = make_additive_plasma_mat(
        "MDC_SparkOrb_Mat", "spark_gain",
        noise_scale=18.0, falloff_power=2.0,
        c_edge=(1.0, 0.56, 0.12), c_core=(0.72, 0.98, 1.0),
        init_gain=4.8,
    )

    return mats, ctrl


# ---------------------------------------------------------------------------
# GEOMETRY BUILDERS
# ---------------------------------------------------------------------------
def _build_intact_glass(
    mat: bpy.types.Material,
    col: bpy.types.Collection,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    """Build smooth single-surface teardrop bubble canopy dome matching C.CANOPY."""
    bm = bmesh.new()
    n_u = 52
    n_v = 38

    tip_aft = bm.verts.new((APEX_X, Y_MIN, SILL_Z))
    tip_fore = bm.verts.new((APEX_X, Y_MAX, SILL_Z))

    rings: list[list[bmesh.types.BMVert]] = []
    for i in range(1, n_u):
        u = i / n_u
        row = [bm.verts.new(_canopy_point_uv(u, j / n_v)) for j in range(n_v + 1)]
        rings.append(row)

    # Aft nose cap fan (outward normal matching main dome)
    for j in range(n_v):
        f = bm.faces.new((tip_aft, rings[0][j], rings[0][j + 1]))
        f.smooth = True

    # Main dome grid
    for i in range(len(rings) - 1):
        for j in range(n_v):
            f = bm.faces.new((
                rings[i][j],
                rings[i + 1][j],
                rings[i + 1][j + 1],
                rings[i][j + 1],
            ))
            f.smooth = True

    # Fore nose cap fan
    for j in range(n_v):
        f = bm.faces.new((rings[-1][j], tip_fore, rings[-1][j + 1]))
        f.smooth = True

    return _create_mesh_obj(
        f"{PREFIX}Glass_Intact", bm, mat, col, parent=parent,
        smooth_all=True, visible_shadow=False, recalc_normals=False,
    )


def _build_intact_frame(
    mat: bpy.types.Material,
    col: bpy.types.Collection,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    """
    Build slim dark-gunmetal canopy sill rails at z = SILL_Z (0.55),
    curved front windshield bow near y = 1.42, and aft deck arch near y = -1.26.
    Keeps Beat 1 center-right sightline from (0.7, 1.1, 0.9) unobstructed.
    """
    bm = bmesh.new()

    # 1. Closed perimeter sill rail at z = SILL_Z
    n_rail = 44
    sill_pts: list[Vector] = []
    sill_ups: list[Vector] = []
    # Right sill (-1.40 -> +1.60)
    for i in range(n_rail + 1):
        u = i / n_rail
        p = _canopy_point_uv(u, 0.0)
        sill_pts.append(p)
        sill_ups.append(Vector((0.0, 0.0, 1.0)))
    # Left sill (+1.60 -> -1.40), skipping duplicate tips
    for i in range(n_rail - 1, 0, -1):
        u = i / n_rail
        p = _canopy_point_uv(u, 1.0)
        sill_pts.append(p)
        sill_ups.append(Vector((0.0, 0.0, 1.0)))
    _sweep_chamfer_profile(bm, sill_pts, sill_ups, half_w=0.0080, half_h=0.0060, chamfer=0.0020, closed=True)

    # 2. Curved front windshield bow near y = 1.42 + subtle rear fairing arch at y = -1.26
    for y_arch, hw, hh in ((WINDSHIELD_BOW_Y, 0.0075, 0.0070), (REAR_ARCH_Y, 0.0080, 0.0075)):
        u_arch = (y_arch - Y_MIN) / (Y_MAX - Y_MIN)
        n_arch = 24
        arch_pts: list[Vector] = []
        arch_ups: list[Vector] = []
        for j in range(n_arch + 1):
            v = j / n_arch
            p = _canopy_point_uv(u_arch, v)
            n_out = _canopy_normal_uv(u_arch, v)
            arch_pts.append(p + n_out * 0.0015)
            arch_ups.append(n_out)
        _sweep_chamfer_profile(bm, arch_pts, arch_ups, half_w=hw, half_h=hh, chamfer=0.0022, closed=False)

    return _create_mesh_obj(
        f"{PREFIX}Frame_Intact", bm, mat, col, parent=parent, smooth_all=True, visible_shadow=False,
    )


def _build_shatter_shards(
    mats: dict[str, bpy.types.Material],
    col: bpy.types.Collection,
    parent: bpy.types.Object,
) -> list[dict[str, Any]]:
    """
    Build 48 irregular-polygon, curved, beveled-thickness canopy shards covering the bubble dome.
    Returns metadata list [{'obj', 'centroid', 'n_out', 'side_x'}] for bullet-time animation.
    """
    rng = random.Random(300)
    n_u, n_v = SHARD_COUNT_U, SHARD_COUNT_V
    u_min, u_max = 0.018, 0.982
    v_min, v_max = 0.008, 0.992
    du = (u_max - u_min) / n_u
    dv = (v_max - v_min) / n_v

    # Irregular Voronoi-like perturbed (u, v) vertex grid with jagged zigzag crown crack seam
    uv_grid: list[list[tuple[float, float]]] = []
    for i in range(n_u + 1):
        row: list[tuple[float, float]] = []
        for j in range(n_v + 1):
            u = _lerp(u_min, u_max, i / n_u)
            v = _lerp(v_min, v_max, j / n_v)
            if 0 < i < n_u:
                u += rng.uniform(-0.23, 0.23) * du
            if 0 < j < n_v:
                if j == n_v // 2:
                    # Jagged zigzag split matching MDC detonation line along crown
                    v += (0.14 if i % 2 == 0 else -0.14) * dv + rng.uniform(-0.08, 0.08) * dv
                else:
                    v += rng.uniform(-0.22, 0.22) * dv
            row.append((max(u_min, min(u_max, u)), max(v_min, min(v_max, v))))
        uv_grid.append(row)

    sub = 3
    perim_coords: list[tuple[int, int]] = (
        [(si, 0) for si in range(sub)]
        + [(sub, sj) for sj in range(sub)]
        + [(si, sub) for si in range(sub, 0, -1)]
        + [(0, sj) for sj in range(sub, 0, -1)]
    )

    shards: list[dict[str, Any]] = []
    idx = 0
    for i in range(n_u):
        for j in range(n_v):
            uv00 = uv_grid[i][j]
            uv10 = uv_grid[i + 1][j]
            uv11 = uv_grid[i + 1][j + 1]
            uv01 = uv_grid[i][j + 1]

            # Sample 4x4 curved sub-patch on dome
            p_out: list[list[Vector]] = []
            p_in: list[list[Vector]] = []
            n_patch = Vector((0.0, 0.0, 0.0))
            centroid = Vector((0.0, 0.0, 0.0))
            cnt = 0
            for si in range(sub + 1):
                fu = si / sub
                row_o, row_i = [], []
                for sj in range(sub + 1):
                    fv = sj / sub
                    u = _lerp(_lerp(uv00[0], uv10[0], fu), _lerp(uv01[0], uv11[0], fu), fv)
                    v = _lerp(_lerp(uv00[1], uv10[1], fu), _lerp(uv01[1], uv11[1], fu), fv)
                    po = _canopy_point_uv(u, v)
                    n_uv = _canopy_normal_uv(u, v)
                    pi = po - n_uv * SHARD_THICKNESS
                    row_o.append(po)
                    row_i.append(pi)
                    n_patch += n_uv
                    centroid += 0.5 * (po + pi)
                    cnt += 1
                p_out.append(row_o)
                p_in.append(row_i)

            centroid /= float(cnt)
            n_patch = n_patch.normalized() if n_patch.length > 1e-6 else Vector((0.0, 0.0, 1.0))

            # Construct local shard BMesh with 2-facet beveled fracture rim
            bm = bmesh.new()
            perim_set = set(perim_coords)
            og: list[list[bmesh.types.BMVert]] = []
            ig: list[list[bmesh.types.BMVert]] = []
            for si in range(sub + 1):
                orow, irow = [], []
                for sj in range(sub + 1):
                    vo_local = p_out[si][sj] - centroid
                    vi_local = p_in[si][sj] - centroid
                    if (si, sj) in perim_set:
                        tang_o = vo_local - n_patch * vo_local.dot(n_patch)
                        if tang_o.length > 1e-5:
                            inset = min(SHARD_CHAMFER_INSET, 0.12 * tang_o.length)
                            vo_local = vo_local - tang_o.normalized() * inset
                            vi_local = vi_local - tang_o.normalized() * inset
                    orow.append(bm.verts.new(vo_local))
                    irow.append(bm.verts.new(vi_local))
                og.append(orow)
                ig.append(irow)

            for si in range(sub):
                for sj in range(sub):
                    fo = bm.faces.new((og[si][sj], og[si + 1][sj], og[si + 1][sj + 1], og[si][sj + 1]))
                    fo.material_index = 0
                    fo.smooth = True
                    fi = bm.faces.new((ig[si][sj], ig[si][sj + 1], ig[si + 1][sj + 1], ig[si + 1][sj]))
                    fi.material_index = 0
                    fi.smooth = True

            # Unshrunk mid-wall bevel ring at 0.5 * thickness
            ring_mid = [
                bm.verts.new(0.5 * (p_out[si][sj] + p_in[si][sj]) - centroid)
                for si, sj in perim_coords
            ]
            n_p = len(perim_coords)
            for k in range(n_p):
                kn = (k + 1) % n_p
                si0, sj0 = perim_coords[k]
                si1, sj1 = perim_coords[kn]

                f1 = bm.faces.new((og[si0][sj0], ring_mid[k], ring_mid[kn], og[si1][sj1]))
                f1.material_index = 1
                f1.smooth = False
                f2 = bm.faces.new((ring_mid[k], ig[si0][sj0], ig[si1][sj1], ring_mid[kn]))
                f2.material_index = 1
                f2.smooth = False

            obj = _create_mesh_obj(
                f"{PREFIX}Shard_{idx:02d}",
                bm,
                [mats["shard_glass"], mats["shard_rim"]],
                col,
                parent=parent,
                location=centroid,
                smooth_all=None,
                visible_shadow=False,
            )
            side_x = 1.0 if (centroid.x > 0.01 or (abs(centroid.x) <= 0.01 and j < n_v // 2)) else -1.0
            shards.append({
                "obj": obj,
                "centroid": centroid.copy(),
                "n_out": n_patch.copy(),
                "side_x": side_x,
            })
            idx += 1

    return shards


def _build_flash_fx(
    mats: dict[str, bpy.types.Material],
    mdc_pts: list[Vector],
    col: bpy.types.Collection,
    parent: bpy.types.Object,
) -> dict[str, bpy.types.Object]:
    """Build open-ended additive shock flash cone, zigzag crack flash ribbon, and spark orb."""
    # 1. Open-ended radial canopy shock flash cone (cap_ends=False per AGENTS.md Section 3;
    # wider base radius1=0.50 -> top radius2=0.22 keeps normals outward+upward from all side/top views)
    bm_flash = bmesh.new()
    bmesh.ops.create_cone(
        bm_flash, cap_ends=False, cap_tris=False, segments=48,
        radius1=0.50, radius2=0.22, depth=0.38,
    )
    for v in bm_flash.verts:
        p = v.co.normalized()
        v.co += p * (0.025 * math.sin(p.x * 7.0 + p.y * 5.0))
    flash_cone = _create_mesh_obj(
        f"{PREFIX}Detonation_Flash",
        bm_flash,
        mats["detonation_flash"],
        col,
        parent=parent,
        location=(0.0, 0.08, APEX_Z - 0.04),
        rotation=(math.radians(-10.0), 0.0, 0.0),
        smooth_all=True,
        visible_shadow=False,
        recalc_normals=False,
    )

    # 2. Open-ended zigzag flash ribbon tracing the shattered MDC cord (cap_ends=False)
    bm_rib = _make_open_tube_bm(mdc_pts, radius=0.020, n_seg=10, cap_ends=False)
    flash_ribbon = _create_mesh_obj(
        f"{PREFIX}MDC_Flash_Ribbon",
        bm_rib,
        mats["flash_ribbon"],
        col,
        parent=parent,
        smooth_all=True,
        visible_shadow=False,
        recalc_normals=False,
    )

    # 3. Displaced soft plasma spark orb racing along the MDC cord
    bm_orb = bmesh.new()
    bmesh.ops.create_icosphere(bm_orb, subdivisions=2, radius=0.042)
    for v in bm_orb.verts:
        p = v.co.normalized()
        v.co += p * (0.008 * math.sin(p.x * 6.0) * math.cos(p.z * 5.0))
    spark_orb = _create_mesh_obj(
        f"{PREFIX}MDC_SparkOrb",
        bm_orb,
        mats["spark_orb"],
        col,
        parent=parent,
        location=mdc_pts[0],
        smooth_all=True,
        visible_shadow=False,
    )

    return {
        "flash_cone": flash_cone,
        "flash_ribbon": flash_ribbon,
        "spark_orb": spark_orb,
    }


# ---------------------------------------------------------------------------
# ANIMATION CHOREOGRAPHY
# ---------------------------------------------------------------------------
def _animate_canopy(
    intact_objs: list[bpy.types.Object],
    mdc_obj: bpy.types.Object,
    shards_meta: list[dict[str, Any]],
    flash_dict: dict[str, bpy.types.Object],
    mdc_pts: list[Vector],
    ctrl: dict[str, Any],
    total_frames: int,
) -> None:
    T = max(1, int(total_frames))
    f_spark = int(C.EV["cord_spark"])        # 282
    f_det = int(C.EV["mdc_detonate"])        # 300
    f_fly_end = int(C.EV["shards_fly_end"])  # 560
    f_hidden = int(C.EV["shards_hidden"])    # 589

    # 1. Intact canopy glass, frame, and MDC cord visible 1..299, CONSTANT-hidden >= 300
    intact_end = max(1, min(T, f_det - 1))
    for obj in intact_objs + [mdc_obj]:
        _key_visibility(obj, [(1, intact_end)], T)

    # 2. MDC detonation wave shader & racing spark orb (f = 282..300)
    wy_sock = ctrl["mdc_wave_y"].outputs[0]
    won_sock = ctrl["mdc_wave_on"].outputs[0]

    for f, won in ((1, 0.0), (max(1, f_spark - 2), 0.0), (f_spark, 1.0), (f_det, 1.0)):
        if f <= T:
            won_sock.default_value = won
            won_sock.keyframe_insert("default_value", frame=f)
    _set_interp(ctrl["mdc_wave_on"].id_data, "LINEAR")

    spark_orb = flash_dict["spark_orb"]
    spark_vis = [(f_spark, min(T, f_det + 2))] if f_spark <= T else []
    _key_visibility(spark_orb, spark_vis, T)

    span_spark = max(1, f_det - f_spark)
    if f_spark <= T:
        for f in range(f_spark, min(T, f_det) + 1):
            u = (f - f_spark) / float(span_spark)
            y_curr = _lerp(MDC_Y_FORE, MDC_Y_AFT, u)
            pt = _sample_polyline_by_y(mdc_pts, y_curr)
            wy_sock.default_value = y_curr
            wy_sock.keyframe_insert("default_value", frame=f)
            spark_orb.location = pt
            sc = 1.0 + 0.35 * math.sin(u * math.pi)
            spark_orb.scale = (sc, sc, sc)
            spark_orb.keyframe_insert("location", frame=f)
            spark_orb.keyframe_insert("scale", frame=f)
    if f_spark > 1:
        wy_sock.default_value = MDC_Y_FORE + 0.08
        wy_sock.keyframe_insert("default_value", frame=1)
        if f_spark - 1 <= T:
            wy_sock.keyframe_insert("default_value", frame=f_spark - 1)
    _set_interp(ctrl["mdc_wave_y"].id_data, "LINEAR")
    _set_interp(spark_orb, "LINEAR", "location")
    _set_interp(spark_orb, "LINEAR", "scale")

    # 3. Open-ended detonation flash & ribbon (f = 300..318, CONSTANT-hidden >= 319)
    flash_cone = flash_dict["flash_cone"]
    flash_ribbon = flash_dict["flash_ribbon"]
    f_flash_end = min(T, f_det + 18)
    _key_visibility(flash_cone, [(f_det, f_flash_end)] if f_det <= T else [], T)
    _key_visibility(flash_ribbon, [(f_det, min(T, f_det + 16))] if f_det <= T else [], T)

    if f_det <= T:
        for f, sc, g_cone, g_rib in (
            (f_det, (0.55, 0.90, 0.45), 3.8, 4.2),
            (min(T, f_det + 4), (1.05, 1.35, 0.90), 2.9, 3.0),
            (min(T, f_det + 10), (1.35, 1.65, 1.25), 1.4, 1.1),
            (f_flash_end, (1.55, 1.90, 1.45), 0.05, 0.0),
        ):
            flash_cone.scale = Vector(sc)
            flash_cone.keyframe_insert("scale", frame=f)
            flash_ribbon.scale = Vector((1.0 + 0.35 * (sc[0] - 0.55), 1.0, 1.0 + 0.28 * (sc[2] - 0.45)))
            flash_ribbon.keyframe_insert("scale", frame=f)

            ctrl["flash_gain"].outputs[0].default_value = g_cone
            ctrl["flash_gain"].outputs[0].keyframe_insert("default_value", frame=f)
            ctrl["ribbon_gain"].outputs[0].default_value = g_rib
            ctrl["ribbon_gain"].outputs[0].keyframe_insert("default_value", frame=f)

        _set_interp(flash_cone, "LINEAR", "scale")
        _set_interp(flash_ribbon, "LINEAR", "scale")

    # 4. Shard fracture vein cool-down (f = 300..560)
    vein_sock = ctrl["shard_vein_gain"].outputs[0]
    craze_sock = ctrl["shard_craze_gain"].outputs[0]
    for f, v_g, c_g in (
        (1, 0.0, 0.0),
        (max(1, f_det - 1), 0.0, 0.0),
        (f_det, 3.4, 0.95),
        (min(T, f_det + 24), 2.5, 0.85),
        (min(T, f_det + 90), 1.3, 0.65),
        (min(T, f_fly_end), 0.55, 0.40),
    ):
        if f <= T:
            vein_sock.default_value = v_g
            vein_sock.keyframe_insert("default_value", frame=f)
            craze_sock.default_value = c_g
            craze_sock.keyframe_insert("default_value", frame=f)

    # 5. 48 Beveled Shards bullet-time outward + aft (-Y) tumble (f = 300..560),
    #    then CONSTANT-hidden from f >= EV["shards_hidden"] = 589 through total_frames!
    rng = random.Random(560)
    shard_vis = [(f_det, min(T, f_hidden - 1))] if f_det <= T else []
    span_fly = max(1, f_fly_end - f_det)
    sample_frames = list(range(f_det, min(T, f_fly_end) + 1, 2)) if f_det <= T else []
    if sample_frames and sample_frames[-1] != min(T, f_fly_end):
        sample_frames.append(min(T, f_fly_end))

    for sm in shards_meta:
        obj = sm["obj"]
        c0: Vector = sm["centroid"]
        n_out: Vector = sm["n_out"]
        sx: float = sm["side_x"]

        _key_visibility(obj, shard_vis, T)

        # Crack burst impulse (strictly outward +-X and upward +Z, never inside cockpit!)
        crack_vec = Vector((
            sx * rng.uniform(0.28, 0.68) + n_out.x * rng.uniform(0.18, 0.42),
            rng.uniform(-0.32, -0.12),
            max(0.28, n_out.z) * rng.uniform(0.55, 1.15) + rng.uniform(0.22, 0.52),
        ))
        # Supersonic slipstream aft (-Y) + upward/outward fan sweep
        aero_vec = Vector((
            sx * rng.uniform(0.55, 1.45),
            -rng.uniform(4.20, 7.60),
            rng.uniform(0.60, 1.75),
        ))
        rot_total = Vector((
            rng.uniform(-4.6, -1.6),
            rng.uniform(-3.4, 3.4) * sx,
            rng.uniform(-3.0, 3.0) * sx,
        ))

        for f in sample_frames:
            u = (f - f_det) / float(span_fly)
            u_crack = min(1.0, u * 3.4)
            crack_w = 1.0 - (1.0 - u_crack) ** 2.3
            aero_w = u ** 1.35
            rot_w = u ** 1.12

            obj.location = c0 + crack_vec * crack_w + aero_vec * aero_w
            obj.rotation_euler = Euler((rot_total.x * rot_w, rot_total.y * rot_w, rot_total.z * rot_w), "XYZ")
            obj.keyframe_insert("location", frame=f)
            obj.keyframe_insert("rotation_euler", frame=f)

        _set_interp(obj, "LINEAR", "location")
        _set_interp(obj, "LINEAR", "rotation_euler")


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------
def build_canopy(
    col: bpy.types.Collection,
    jet_root: bpy.types.Object,
    total_frames: int = 1800,
) -> dict[str, Any]:
    """
    Build the fighter-jet bubble canopy, gunmetal frame, zigzag MDC cord,
    detonation plasma wave & spark orb, open-ended additive shock flash,
    and 48 beveled shattering glass shards parented to `Canopy_Root`.
    """
    canopy_root = bpy.data.objects.new(f"{PREFIX}Root", None)
    canopy_root.empty_display_type = "PLAIN_AXES"
    canopy_root.empty_display_size = 0.25
    canopy_root.location = Vector((0.0, 0.0, 0.0))
    canopy_root.rotation_euler = Euler((0.0, 0.0, 0.0), "XYZ")
    if jet_root is not None:
        canopy_root.parent = jet_root
    _link_obj(canopy_root, col)

    mats, ctrl = _build_materials(canopy_root)
    mdc_pts = _build_mdc_polyline()

    glass_obj = _build_intact_glass(mats["glass_intact"], col, canopy_root)
    frame_obj = _build_intact_frame(mats["frame_gunmetal"], col, canopy_root)
    intact_objs = [glass_obj, frame_obj]

    bm_mdc = _make_open_tube_bm(mdc_pts, radius=MDC_TUBE_RADIUS, n_seg=10, cap_ends=True)
    mdc_obj = _create_mesh_obj(
        f"{PREFIX}MDC_Cord", bm_mdc, mats["mdc_cord"], col,
        parent=canopy_root, smooth_all=True, visible_shadow=False,
    )

    shards_meta = _build_shatter_shards(mats, col, canopy_root)
    flash_dict = _build_flash_fx(mats, mdc_pts, col, canopy_root)
    flash_objs = [
        flash_dict["flash_cone"],
        flash_dict["flash_ribbon"],
        flash_dict["spark_orb"],
    ]

    _animate_canopy(
        intact_objs=intact_objs,
        mdc_obj=mdc_obj,
        shards_meta=shards_meta,
        flash_dict=flash_dict,
        mdc_pts=mdc_pts,
        ctrl=ctrl,
        total_frames=total_frames,
    )

    return {
        "root": canopy_root,
        "intact_objs": intact_objs,
        "shards": [s["obj"] for s in shards_meta],
        "mdc_cord": mdc_obj,
        "flash_objs": flash_objs,
    }


# ---------------------------------------------------------------------------
# STANDALONE SELF-TEST
# ---------------------------------------------------------------------------
def _run_standalone_test() -> None:
    out_dir = Path("/tmp/fje_canopy")
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1920
    scene.render.resolution_percentage = 25
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
    cycles.transparent_max_bounces = 16
    cycles.caustics_reflective = False
    cycles.caustics_refractive = False

    world = bpy.data.worlds.new("Canopy_Test_World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.18, 0.36, 0.65, 1.0)
        bg.inputs["Strength"].default_value = 1.2

    col = bpy.data.collections.new("Canopy_Test_Col")
    scene.collection.children.link(col)

    sun_data = bpy.data.lights.new("Test_Sun", type="SUN")
    sun_data.energy = 3.5
    sun_obj = bpy.data.objects.new("Test_Sun", sun_data)
    sun_obj.rotation_euler = Vector(C.SUN_TOWARD).to_track_quat("Z", "Y").to_euler()
    col.objects.link(sun_obj)

    jet_root = bpy.data.objects.new("Jet_Root", None)
    col.objects.link(jet_root)

    # Simple pilot head / handle reference spheres so glass clarity is immediately verifiable
    bm_ref = bmesh.new()
    bmesh.ops.create_icosphere(bm_ref, subdivisions=2, radius=0.11)
    ref_head = _create_mesh_obj(
        "Test_Head_Marker", bm_ref,
        bpy.data.materials.new("Test_Head_Mat"), col, parent=jet_root,
        location=C.PILOT["head_center"], visible_shadow=True,
    )
    ref_head.data.materials[0].use_nodes = True
    ref_head.data.materials[0].node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.85, 0.55, 0.15, 1.0)

    res = build_canopy(col, jet_root, total_frames=C.TOTAL_FRAMES)
    print(
        f"[canopy_fx test] built root={res['root'].name}, "
        f"intact={len(res['intact_objs'])}, shards={len(res['shards'])}, "
        f"mdc={res['mdc_cord'].name}, flash={len(res['flash_objs'])}",
        flush=True,
    )

    cam_d = bpy.data.cameras.new("Test_Cam")
    cam_d.lens = 35.0
    cam_o = bpy.data.objects.new("Test_Cam", cam_d)
    col.objects.link(cam_o)
    scene.camera = cam_o

    tgt = bpy.data.objects.new("Test_Cam_Target", None)
    col.objects.link(tgt)
    tr = cam_o.constraints.new(type="TRACK_TO")
    tr.target = tgt
    tr.track_axis = "TRACK_NEGATIVE_Z"
    tr.up_axis = "UP_Y"

    for frame, cam_loc, tgt_loc in (
        (100, (0.7, 1.1, 0.9), (0.0, 0.0, 0.72)),
        (380, (2.2, 0.5, 1.45), (0.0, -1.0, 1.20)),
    ):
        scene.frame_set(frame)
        cam_o.location = Vector(cam_loc)
        tgt.location = Vector(tgt_loc)
        out_path = str(out_dir / f"frame_{frame:04d}.png")
        scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        print(f"[canopy_fx test] rendered {out_path}", flush=True)


if __name__ == "__main__":
    _run_standalone_test()
