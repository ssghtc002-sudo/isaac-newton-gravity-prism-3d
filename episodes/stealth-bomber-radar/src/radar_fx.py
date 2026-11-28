"""
Volumetric Radar Optics, Delicate Sunlit Wingtip Trails, Enemy Command Console Blank Scope (Beat 1),
Red Target-Lock Alert Ring (Beat 2), S-Duct Trapped RF Wave (Beat 3), Macro RAM Honeycomb +
Live 3D Shrinking RCS Counter (`100 m² -> 0.001 m²`) + Sculpted Bumblebee (Beat 4),
Main-Scene Daylight Real-Jet 3D Deflection Arrows (Beat 5) & Hooded Anti-Glare Command Room
PPI Radar Scope (Beat 6) for Episode #4 (60 fps, 1800 frames).

Strictly obeys AGENTS.md V12 & Blender 5.3.0 rules:
- All open-ended volumetric cones, radar beams & trails use `cap_ends=False` / `end_fill_type="NOTHING"`
  + AgX-safe `AddShader(Transparent, Emission * CameraRay * (1 - Backfacing))`.
- All materials appended to `mesh.materials` BEFORE `bm.to_mesh(mesh)`.
- Blender 5.x Slotted Actions via `C.iter_action_fcurves`, `ShaderNodeMix` via `C.get_mix_sockets`.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EPISODE_ROOT = SCRIPT_DIR.parent
FONT_PATH = EPISODE_ROOT / "subtitles" / "fonts" / "DejaVuSans-Bold.ttf"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stealth_contract as C  # noqa: E402

import bmesh  # type: ignore[import-not-found]
import bpy  # type: ignore[import-not-found]
from mathutils import Euler, Matrix, Vector  # type: ignore[import-not-found]


def _set_const_hide(obj: bpy.types.Object) -> None:
    ad = getattr(obj, "animation_data", None)
    if not ad or not ad.action:
        return
    for fc in C.iter_action_fcurves(ad.action):
        if fc.data_path.startswith("hide_"):
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"


def _key_visibility(obj: bpy.types.Object, windows: list[tuple[int, int]], total_frames: int) -> None:
    frames = {1, total_frames}
    for s, e in windows:
        sc = max(1, min(total_frames, s))
        ec = max(1, min(total_frames, e))
        if sc > 1:
            frames.add(sc - 1)
        frames.add(sc)
        frames.add(ec)
        if ec < total_frames:
            frames.add(ec + 1)
    for f in sorted(frames):
        vis = any(s <= f <= e for s, e in windows)
        obj.hide_viewport = not vis
        obj.hide_render = not vis
        obj.keyframe_insert("hide_viewport", frame=f)
        obj.keyframe_insert("hide_render", frame=f)
    _set_const_hide(obj)


def _mesh_from_bm(
    name: str,
    bm: bmesh.types.BMesh,
    mats: list[bpy.types.Material],
    col: bpy.types.Collection,
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    parent: bpy.types.Object | None = None,
    smooth: bool = True,
    no_shadow: bool = True,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    for m in mats:
        mesh.materials.append(m)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    if len(mesh.polygons) > 0:
        mesh.polygons.foreach_set("use_smooth", [smooth] * len(mesh.polygons))
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(location)
    obj.rotation_euler = Euler(rotation, "XYZ")
    if no_shadow:
        obj.visible_shadow = False
    if parent is not None:
        obj.parent = parent
    col.objects.link(obj)
    return obj


# ==============================================================================
# AGX-SAFE ADDITIVE OPTICAL & PROCEDURAL SHADERS
# ==============================================================================
def _mat_additive_beam(
    name: str,
    color: tuple[float, float, float],
    strength: float = 0.36,
    fresnel_pow: float = 2.4,
) -> tuple[bpy.types.Material, Any]:
    """Sleek feathered volumetric radar beam shader: AddShader(Transparent, Emission*Cam*(1-Back))."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    lp = nt.nodes.new("ShaderNodeLightPath")
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.42

    front = nt.nodes.new("ShaderNodeMath")
    front.operation = "SUBTRACT"
    front.inputs[0].default_value = 1.0
    nt.links.new(geom.outputs["Backfacing"], front.inputs[1])

    cam_f = nt.nodes.new("ShaderNodeMath")
    cam_f.operation = "MULTIPLY"
    nt.links.new(lp.outputs["Is Camera Ray"], cam_f.inputs[0])
    nt.links.new(front.outputs[0], cam_f.inputs[1])

    inv_f = nt.nodes.new("ShaderNodeMath")
    inv_f.operation = "SUBTRACT"
    inv_f.inputs[0].default_value = 1.0
    nt.links.new(lw.outputs["Facing"], inv_f.inputs[1])

    soft = nt.nodes.new("ShaderNodeMath")
    soft.operation = "POWER"
    soft.inputs[1].default_value = fresnel_pow
    nt.links.new(inv_f.outputs[0], soft.inputs[0])

    gain_val = nt.nodes.new("ShaderNodeValue")
    gain_val.name = f"{name}_Gain"
    gain_val.outputs[0].default_value = strength

    m1 = nt.nodes.new("ShaderNodeMath")
    m1.operation = "MULTIPLY"
    nt.links.new(soft.outputs[0], m1.inputs[0])
    nt.links.new(gain_val.outputs[0], m1.inputs[1])

    m2 = nt.nodes.new("ShaderNodeMath")
    m2.operation = "MULTIPLY"
    nt.links.new(m1.outputs[0], m2.inputs[0])
    nt.links.new(cam_f.outputs[0], m2.inputs[1])

    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (*color, 1.0)
    nt.links.new(m2.outputs[0], emit.inputs["Strength"])

    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(transp.outputs["BSDF"], add.inputs[0])
    nt.links.new(emit.outputs["Emission"], add.inputs[1])
    nt.links.new(add.outputs["Shader"], out.inputs["Surface"])
    return mat, gain_val


def _mat_solid_glow(
    name: str,
    base_col: tuple[float, float, float],
    emit_col: tuple[float, float, float],
    emit_str: float,
    metallic: float = 0.20,
    roughness: float = 0.35,
    spec_ior: float = 0.25,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*base_col, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = spec_ior
    bsdf.inputs["Emission Color"].default_value = (*emit_col, 1.0)
    bsdf.inputs["Emission Strength"].default_value = min(2.0, emit_str)
    return mat


def _open_cone_bm(r1: float, r2: float, length: float, segs: int = 28) -> bmesh.types.BMesh:
    """Open-ended cone along +Z from 0..length (`cap_ends=False` per AGENTS.md V12!)."""
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm,
        cap_ends=False,
        cap_tris=False,
        segments=segs,
        radius1=r1,
        radius2=r2,
        depth=length,
        matrix=Matrix.Translation((0.0, 0.0, 0.5 * length)),
    )
    for f in bm.faces:
        f.material_index = 0
    return bm


def _make_torus_bm(major_r: float, minor_r: float, major_seg: int = 36, minor_seg: int = 10) -> bmesh.types.BMesh:
    bm = bmesh.new()
    rings: list[list[bmesh.types.BMVert]] = []
    for i in range(major_seg):
        a = 2.0 * math.pi * i / major_seg
        ca, sa = math.cos(a), math.sin(a)
        row = []
        for j in range(minor_seg):
            b = 2.0 * math.pi * j / minor_seg
            cb, sb = math.cos(b), math.sin(b)
            r = major_r + minor_r * cb
            row.append(bm.verts.new((r * ca, r * sa, minor_r * sb)))
        rings.append(row)
    for i in range(major_seg):
        inext = (i + 1) % major_seg
        for j in range(minor_seg):
            jnext = (j + 1) % minor_seg
            bm.faces.new((rings[i][j], rings[inext][j], rings[inext][jnext], rings[i][jnext])).material_index = 0
    return bm


def _make_beam_obj(
    name: str,
    p0: Vector,
    p1: Vector,
    r0: float,
    r1: float,
    mat: bpy.types.Material,
    col: bpy.types.Collection,
) -> bpy.types.Object:
    seg = p1 - p0
    length = max(0.05, seg.length)
    bm = _open_cone_bm(r0, r1, length, segs=24)
    obj = _mesh_from_bm(name, bm, [mat], col, location=(p0.x, p0.y, p0.z), smooth=True, no_shadow=True)
    obj.rotation_euler = seg.normalized().to_track_quat("Z", "Y").to_euler("XYZ")
    return obj


def _make_3d_arrow_obj(
    name: str,
    p0: Vector,
    p1: Vector,
    shaft_r: float,
    head_r: float,
    head_len: float,
    mat: bpy.types.Material,
    col: bpy.types.Collection,
) -> bpy.types.Object:
    """Crisp unmistakable 3D glowing directional arrow (shaft + conical arrowhead) from p0 -> p1."""
    seg = p1 - p0
    total_len = max(0.08, seg.length)
    shaft_len = max(0.02, total_len - head_len)

    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=16,
        radius1=shaft_r, radius2=shaft_r, depth=shaft_len,
        matrix=Matrix.Translation((0.0, 0.0, 0.5 * shaft_len)),
    )
    bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=18,
        radius1=head_r, radius2=0.002, depth=head_len,
        matrix=Matrix.Translation((0.0, 0.0, shaft_len + 0.5 * head_len)),
    )
    for f in bm.faces:
        f.material_index = 0

    obj = _mesh_from_bm(name, bm, [mat], col, location=(p0.x, p0.y, p0.z), smooth=True, no_shadow=True)
    obj.rotation_euler = seg.normalized().to_track_quat("Z", "Y").to_euler("XYZ")
    return obj


# ==============================================================================
# 1. DELICATE SUNLIT WINGTIP CONDENSATION TRAILS (BEATS 1 & 6)
# ==============================================================================
def _build_wingtip_contrails(
    col: bpy.types.Collection,
    bomber_root: bpy.types.Object,
    total_frames: int,
) -> list[bpy.types.Object]:
    """Delicate open-ended sunlit condensation wisps off the 172-ft wingtips (zero chunky sticks!)."""
    mat_trail, _ = _mat_additive_beam("Mat_FX_WingtipContrail", (0.94, 0.97, 1.00), strength=0.20, fresnel_pow=2.8)
    trails: list[bpy.types.Object] = []
    for side in (-1.0, +1.0):
        bm = _open_cone_bm(0.004, 0.026, 1.35, segs=14)
        obj = _mesh_from_bm(
            f"FX_WingtipTrail_{'L' if side < 0 else 'R'}",
            bm,
            [mat_trail],
            col,
            location=(side * 1.58, 0.44, 0.005),
            rotation=(math.radians(-90.0), 0.0, 0.0),
            parent=bomber_root,
            smooth=True,
            no_shadow=True,
        )
        _key_visibility(obj, [(1, 220), (1561, total_frames)], total_frames)
        trails.append(obj)
    return trails


# ==============================================================================
# 2. BEAT 1 ENEMY COMMAND ROOM CONSOLE BLANK RADAR SCREEN (EMBEDDED IN DESK!)
# ==============================================================================
def _build_beat1_blank_monitor(col: bpy.types.Collection, total_frames: int) -> list[bpy.types.Object]:
    """
    Beat 1 Instant Visual Hook: Recessed anti-glare green-phosphor PPI radar screen embedded cleanly
    right inside the Enemy Radar Command Room console desk (`b1_console_scope_pos`), spinning over
    an EMPTY grid (`0 TARGETS`) while the 172-ft B-2 glides outside the armored bunker window!
    """
    objs: list[bpy.types.Object] = []
    mon_root = bpy.data.objects.new("FX_B1_BlankMonitor_Root", None)
    mon_root.location = Vector(C.STAGE["b1_console_scope_pos"])
    # Matches the Enemy Command Room console desk tilt cleanly facing the camera
    mon_root.rotation_euler = Euler((math.radians(66.0), 0.0, math.radians(-10.0)), "XYZ")
    col.objects.link(mon_root)

    r_sc = 0.25
    mat_bezel = _mat_solid_glow("Mat_B1Mon_Bezel", (0.04, 0.05, 0.07), (0.02, 0.20, 0.28), 0.08, metallic=0.85, roughness=0.34, spec_ior=0.18)
    mat_dial = _mat_solid_glow("Mat_B1Mon_Dial", (0.002, 0.012, 0.010), (0.005, 0.06, 0.045), 0.14, metallic=0.05, roughness=0.68, spec_ior=0.04)
    mat_grid = _mat_solid_glow("Mat_B1Mon_Grid", (0.06, 0.96, 0.64), (0.06, 0.96, 0.64), 1.65)
    mat_wedge, _ = _mat_additive_beam("Mat_B1Mon_Sweep", (0.08, 0.98, 0.64), strength=0.85, fresnel_pow=0.5)

    # Hooded console bezel + mounting pedestal + anti-glare recessed matte phosphor dial
    bm_m = bmesh.new()
    bmesh.ops.create_cone(
        bm_m, cap_ends=True, cap_tris=False, segments=40,
        radius1=r_sc * 1.14, radius2=r_sc * 1.08, depth=0.048,
        matrix=Matrix.Translation((0.0, 0.0, -0.024)),
    )
    # Recessed protective glare hood ring around perimeter
    bmesh.ops.create_cone(
        bm_m, cap_ends=False, cap_tris=False, segments=40,
        radius1=r_sc * 1.07, radius2=r_sc * 1.04, depth=0.032,
        matrix=Matrix.Translation((0.0, 0.0, 0.014)),
    )
    # Angled military console pedestal base anchoring monitor cleanly to console deck
    bmesh.ops.create_cube(
        bm_m, size=1.0,
        matrix=Matrix.Translation((0.0, -0.14, -0.11)) @ Matrix.Diagonal((0.22, 0.24, 0.16, 1.0)),
    )
    for fm in bm_m.faces:
        fm.material_index = 0
    res_d = bmesh.ops.create_cone(
        bm_m, cap_ends=True, cap_tris=False, segments=40,
        radius1=r_sc, radius2=r_sc, depth=0.008,
        matrix=Matrix.Translation((0.0, 0.0, -0.002)),
    )
    dv = set(res_d["verts"])
    for fd in bm_m.faces:
        if all(v in dv for v in fd.verts):
            fd.material_index = 1
    body = _mesh_from_bm("FX_B1_MonitorBody", bm_m, [mat_bezel, mat_dial], col, parent=mon_root, no_shadow=False)
    _key_visibility(body, [(1, 220)], total_frames)
    objs.append(body)

    # Concentric range rings + crosshairs + 12 azimuth ticks (100% BLANK — ZERO BLIPS!)
    bm_g = bmesh.new()
    for frac in (0.33, 0.66, 0.96):
        bm_t = _make_torus_bm(r_sc * frac, 0.0024, major_seg=40, minor_seg=8)
        for v in bm_t.verts:
            v.co.z += 0.006
        tmp = bpy.data.meshes.new("_tmp_b1r")
        bm_t.to_mesh(tmp)
        bm_t.free()
        bm_g.from_mesh(tmp)
        bpy.data.meshes.remove(tmp)
    bmesh.ops.create_cube(bm_g, size=1.0, matrix=Matrix.Translation((0.0, 0.0, 0.006)) @ Matrix.Diagonal((r_sc * 1.92, 0.003, 0.002, 1.0)))
    bmesh.ops.create_cube(bm_g, size=1.0, matrix=Matrix.Translation((0.0, 0.0, 0.006)) @ Matrix.Diagonal((0.003, r_sc * 1.92, 0.002, 1.0)))
    for k in range(12):
        a = 2.0 * math.pi * k / 12.0
        bmesh.ops.create_cube(
            bm_g, size=1.0,
            matrix=Matrix.Translation((r_sc * 0.91 * math.cos(a), r_sc * 0.91 * math.sin(a), 0.006))
            @ Euler((0.0, 0.0, a), "XYZ").to_matrix().to_4x4()
            @ Matrix.Diagonal((0.022, 0.003, 0.002, 1.0)),
        )
    for fg in bm_g.faces:
        fg.material_index = 0
    grid = _mesh_from_bm("FX_B1_MonitorGrid", bm_g, [mat_grid], col, parent=mon_root)
    _key_visibility(grid, [(1, 220)], total_frames)
    objs.append(grid)

    # Rotating phosphor sweep needle
    bm_sw = bmesh.new()
    bmesh.ops.create_cube(bm_sw, size=1.0, matrix=Matrix.Translation((0.0, 0.48 * r_sc, 0.009)) @ Matrix.Diagonal((0.005, 0.96 * r_sc, 0.003, 1.0)))
    for fsw in bm_sw.faces:
        fsw.material_index = 0
    vc = bm_sw.verts.new((0.0, 0.0, 0.008))
    pts = [
        bm_sw.verts.new((r_sc * 0.96 * math.cos(math.radians(90.0 + k * 3.8)), r_sc * 0.96 * math.sin(math.radians(90.0 + k * 3.8)), 0.008))
        for k in range(10)
    ]
    for k in range(len(pts) - 1):
        bm_sw.faces.new((vc, pts[k], pts[k + 1])).material_index = 1
    sweep = _mesh_from_bm("FX_B1_MonitorSweep", bm_sw, [mat_grid, mat_wedge], col, parent=mon_root)
    _key_visibility(sweep, [(1, 220)], total_frames)
    for f in range(1, 221, 3):
        sweep.rotation_euler = Euler((0.0, 0.0, -(f - 1) * 0.052), "XYZ")
        sweep.keyframe_insert("rotation_euler", frame=f)
    objs.append(sweep)

    return objs


# ==============================================================================
# 3. BEAT 1 & 2: SLEEK VOLUMETRIC RADAR WAVES & DEFLECTION BURST (ZERO CLIPPING!)
# ==============================================================================
def _build_beats12_radar_optics(col: bpy.types.Collection, total_frames: int) -> list[bpy.types.Object]:
    objs: list[bpy.types.Object] = []
    mat_cyan, gain_cyan = _mat_additive_beam("Mat_FX_RadarCyan", (0.04, 0.88, 1.00), strength=0.40, fresnel_pow=2.2)
    mat_defl, gain_defl = _mat_additive_beam("Mat_FX_RadarDeflect", (0.10, 0.96, 1.00), strength=0.44, fresnel_pow=2.0)
    mat_red, gain_red = _mat_additive_beam("Mat_FX_RadarCrimsonEcho", (1.00, 0.14, 0.06), strength=0.68, fresnel_pow=1.8)
    mat_ring_c = _mat_solid_glow("Mat_FX_WaveRingCyan", (0.06, 0.92, 1.00), (0.06, 0.92, 1.00), 1.92)
    mat_ring_r = _mat_solid_glow("Mat_FX_WaveRingRed", (1.00, 0.16, 0.06), (1.00, 0.16, 0.06), 1.98)

    # --- BEAT 1: Outdoor radar dish -> hits B-2 chined leading edge -> ricochets dynamically into upper sky! ---
    dish1 = Vector(C.STAGE["radar_dish_b1"]) + Vector((0.06, 0.0, 0.28))
    hit1 = Vector((-0.20, 0.56, 2.28))
    sky_away = Vector((1.65, 0.85, 3.32))

    b1_in = _make_beam_obj("FX_B1_RadarInCone", dish1, hit1, 0.025, 0.13, mat_cyan, col)
    b1_out = _make_beam_obj("FX_B1_RadarDeflectCone", hit1, sky_away, 0.048, 0.25, mat_defl, col)
    _key_visibility(b1_in, [(12, 220)], total_frames)
    _key_visibility(b1_out, [(34, 220)], total_frames)
    objs.extend([b1_in, b1_out])

    # Pulsing cyan chine deflection shock ring at the wing bounce point!
    burst1 = _mesh_from_bm(
        "FX_B1_ChineDeflectBurst", _make_torus_bm(0.14, 0.006, major_seg=32, minor_seg=8),
        [mat_ring_c], col, location=(hit1.x, hit1.y, hit1.z),
        rotation=(math.radians(52.0), math.radians(-24.0), 0.0),
    )
    _key_visibility(burst1, [(34, 220)], total_frames)
    for f in range(34, 221, 3):
        p = 0.75 + 0.40 * abs(math.sin((f - 34) * 0.24))
        burst1.scale = (p, p, p)
        burst1.keyframe_insert("scale", frame=f)
    objs.append(burst1)

    for ri in range(3):
        ring_in = _mesh_from_bm(f"FX_B1_InRing_{ri}", _make_torus_bm(0.12, 0.0052), [mat_ring_c], col)
        ring_out = _mesh_from_bm(f"FX_B1_OutRing_{ri}", _make_torus_bm(0.15, 0.0056), [mat_ring_c], col)
        _key_visibility(ring_in, [(12, 220)], total_frames)
        _key_visibility(ring_out, [(36, 220)], total_frames)
        ring_in.rotation_euler = (hit1 - dish1).normalized().to_track_quat("Z", "Y").to_euler("XYZ")
        ring_out.rotation_euler = (sky_away - hit1).normalized().to_track_quat("Z", "Y").to_euler("XYZ")
        for f in range(12, 221, 3):
            ph = ((f - 12 + ri * 22) % 66) / 66.0
            ring_in.location = dish1.lerp(hit1, ph)
            sc = C.lerp(0.30, 1.18, ph)
            ring_in.scale = (sc, sc, 1.0)
            ring_in.keyframe_insert("location", frame=f)
            ring_in.keyframe_insert("scale", frame=f)

            if f >= 36:
                ph2 = ((f - 36 + ri * 22) % 66) / 66.0
                ring_out.location = hit1.lerp(sky_away, ph2)
                sc2 = C.lerp(0.45, 1.65, ph2)
                ring_out.scale = (sc2, sc2, 1.0)
                ring_out.keyframe_insert("location", frame=f)
                ring_out.keyframe_insert("scale", frame=f)
        objs.extend([ring_in, ring_out])

    # --- BEAT 2: Split Daylight Aerial Contrast (Conv Jet Crimson Return vs Stealth Deflect) ---
    dish2 = Vector(C.STAGE["radar_dish_b2"]) + Vector((0.20, 0.0, 0.10))
    conv_hit = Vector(C.STAGE["b2_conv_pos"]) + Vector((-0.24, 0.0, -0.02))
    st_hit = Vector(C.STAGE["b2_stealth_pos"]) + Vector((-0.22, -0.06, -0.02))
    st_away = Vector((1.65, 0.25, 0.56))

    b2_in_top = _make_beam_obj("FX_B2_ConvInBeam", dish2, conv_hit, 0.022, 0.12, mat_cyan, col)
    b2_echo_top = _make_beam_obj("FX_B2_ConvCrimsonEcho", conv_hit, dish2, 0.085, 0.22, mat_red, col)
    b2_in_bot = _make_beam_obj("FX_B2_StealthInBeam", dish2, st_hit, 0.022, 0.11, mat_cyan, col)
    b2_def_bot = _make_beam_obj("FX_B2_StealthDeflectBeam", st_hit, st_away, 0.045, 0.21, mat_defl, col)

    _key_visibility(b2_in_top, [(245, 540)], total_frames)
    _key_visibility(b2_echo_top, [(305, 540)], total_frames)
    _key_visibility(b2_in_bot, [(355, 540)], total_frames)
    _key_visibility(b2_def_bot, [(395, 540)], total_frames)
    objs.extend([b2_in_top, b2_echo_top, b2_in_bot, b2_def_bot])

    # Pulsing Red Target-Lock Alert Ring on the Radar Dish
    lock_ring = _mesh_from_bm(
        "FX_B2_DishRedLockRing", _make_torus_bm(0.32, 0.011, major_seg=36, minor_seg=10), [mat_ring_r],
        col, location=(dish2.x, dish2.y, dish2.z),
        rotation=(0.0, math.radians(90.0), 0.0),
    )
    _key_visibility(lock_ring, [(315, 540)], total_frames)
    for f in range(315, 541, 3):
        p = 0.88 + 0.30 * abs(math.sin((f - 315) * 0.22))
        lock_ring.scale = (p, p, p)
        lock_ring.keyframe_insert("scale", frame=f)
    objs.append(lock_ring)

    # Crisp thin crimson echo rings shooting straight back to dish
    for ri in range(3):
        rr = _mesh_from_bm(f"FX_B2_EchoRing_{ri}", _make_torus_bm(0.14, 0.0068), [mat_ring_r], col)
        _key_visibility(rr, [(308, 540)], total_frames)
        rr.rotation_euler = (dish2 - conv_hit).normalized().to_track_quat("Z", "Y").to_euler("XYZ")
        for f in range(308, 541, 3):
            ph = ((f - 308 + ri * 22) % 66) / 66.0
            rr.location = conv_hit.lerp(dish2, ph)
            sc = C.lerp(0.60, 1.45, ph)
            rr.scale = (sc, sc, 1.0)
            rr.keyframe_insert("location", frame=f)
            rr.keyframe_insert("scale", frame=f)
        objs.append(rr)

    # Crisp thin deflected cyan rings shooting sideways away from B-2
    for ri in range(3):
        rc = _mesh_from_bm(f"FX_B2_DeflectRing_{ri}", _make_torus_bm(0.12, 0.0052), [mat_ring_c], col)
        _key_visibility(rc, [(400, 540)], total_frames)
        rc.rotation_euler = (st_away - st_hit).normalized().to_track_quat("Z", "Y").to_euler("XYZ")
        for f in range(400, 541, 3):
            ph = ((f - 400 + ri * 20) % 60) / 60.0
            rc.location = st_hit.lerp(st_away, ph)
            sc = C.lerp(0.48, 1.40, ph)
            rc.scale = (sc, sc, 1.0)
            rc.keyframe_insert("location", frame=f)
            rc.keyframe_insert("scale", frame=f)
        objs.append(rc)

    for f in range(1, 541, 15):
        pulse = 0.85 + 0.22 * math.sin(f * 0.28)
        gain_cyan.outputs[0].default_value = 0.34 * pulse
        gain_defl.outputs[0].default_value = 0.38 * pulse
        gain_red.outputs[0].default_value = 0.68 * pulse
        gain_cyan.outputs[0].keyframe_insert("default_value", frame=f)
        gain_defl.outputs[0].keyframe_insert("default_value", frame=f)
        gain_red.outputs[0].keyframe_insert("default_value", frame=f)

    return objs


# ==============================================================================
# 4. BEAT 3: TRAPPED RF WAVE PACKET BOUNCING & DYING INSIDE SERPENTINE S-DUCT
# ==============================================================================
def _build_beat3_sduct_wave(
    col: bpy.types.Collection,
    bomber_root: bpy.types.Object,
    total_frames: int,
) -> list[bpy.types.Object]:
    """Glowing cyan RF pulse & warm absorption trail bouncing inside the curved S-duct and dying before the blades."""
    mat_pulse = _mat_solid_glow("Mat_FX_SDuctPulse", (0.10, 0.94, 1.00), (0.10, 0.94, 1.00), 1.95)
    mat_spark = _mat_solid_glow("Mat_FX_SDuctAbsorb", (1.00, 0.62, 0.10), (1.00, 0.62, 0.10), 1.85)

    bm_p = bmesh.new()
    bmesh.ops.create_icosphere(bm_p, subdivisions=2, radius=0.024)
    for f_face in bm_p.faces:
        f_face.material_index = 0
    pulse_obj = _mesh_from_bm("FX_B3_TrappedRFPulse", bm_p, [mat_pulse], col, parent=bomber_root)
    _key_visibility(pulse_obj, [(640, 955)], total_frames)

    trail_objs: list[bpy.types.Object] = []
    for ti in range(4):
        bm_t = bmesh.new()
        bmesh.ops.create_icosphere(bm_t, subdivisions=1, radius=0.015 * (1.0 - 0.16 * ti))
        for f_face in bm_t.faces:
            f_face.material_index = 0
        to = _mesh_from_bm(f"FX_B3_RFTrail_{ti}", bm_t, [mat_pulse if ti < 2 else mat_spark], col, parent=bomber_root)
        _key_visibility(to, [(640, 955)], total_frames)
        trail_objs.append(to)

    def duct_bounce_local(u: float) -> Vector:
        t = C.clamp(u, 0.0, 0.80)
        y = C.lerp(-0.26, +0.26, t)
        x_center = -(0.24 + 0.088 * math.sin(math.pi * t) - 0.048 * math.sin(2.0 * math.pi * t))
        z_center = 0.068 - 0.088 * C.smootherstep(0.08, 0.72, t) + 0.024 * math.sin(2.0 * math.pi * t)
        bounce = 0.026 * math.sin(t * 5.0 * math.pi) * (1.0 - t)
        return Vector((x_center + bounce, y, z_center + 0.015 * math.cos(t * 5.0 * math.pi)))

    for f in range(640, 956, 2):
        u = C.sduct_wave_progress(float(f))
        p_loc = duct_bounce_local(u)
        atten = max(0.05, 1.0 - C.smootherstep(0.55, 0.96, u))
        pulse_obj.location = p_loc
        pulse_obj.scale = (atten, atten, atten)
        pulse_obj.keyframe_insert("location", frame=f)
        pulse_obj.keyframe_insert("scale", frame=f)

        for ti, to in enumerate(trail_objs):
            u_t = max(0.0, u - 0.045 * (ti + 1))
            to.location = duct_bounce_local(u_t)
            sc = atten * max(0.1, 1.0 - 0.18 * ti)
            to.scale = (sc, sc, sc)
            to.keyframe_insert("location", frame=f)
            to.keyframe_insert("scale", frame=f)

    return [pulse_obj] + trail_objs


# ==============================================================================
# 5. BEAT 4: MACRO HEXAGONAL RAM HONEYCOMB + LIVE RCS COUNTER + 3D BUMBLEBEE
# ==============================================================================
def _build_beat4_live_rcs_counter(
    col: bpy.types.Collection,
    bee_root: bpy.types.Object,
    total_frames: int,
) -> list[bpy.types.Object]:
    """
    Live 3D holographic RCS numeric countdown (`100 m² -> 10 m² -> 1.0 m² -> 0.01 m² -> 0.001 m²`)
    backed by a sleek dark-obsidian pill-free rectangular tag so numbers stay >15:1 crisp on daylight sky!
    """
    font = bpy.data.fonts.load(str(FONT_PATH), check_existing=True) if FONT_PATH.is_file() else None
    mat_bg = _mat_solid_glow("Mat_FX_RCSCounterBg", (0.008, 0.012, 0.020), (0.0, 0.0, 0.0), 0.0, metallic=0.1, roughness=0.5, spec_ior=0.08)

    # Dark obsidian backing plate behind the countdown numbers
    bm_bg = bmesh.new()
    bmesh.ops.create_cube(bm_bg, size=1.0, matrix=Matrix.Diagonal((0.40, 0.012, 0.125, 1.0)))
    for f in bm_bg.faces:
        f.material_index = 0
    bg_obj = _mesh_from_bm(
        "FX_B4_RCSNum_Backplate", bm_bg, [mat_bg], col,
        location=(0.0, -0.048, -0.22), parent=bee_root, smooth=False, no_shadow=True,
    )
    _key_visibility(bg_obj, [(1125, 1318)], total_frames)

    steps = [
        ("100 m²",   (1125, 1159), (1.00, 0.22, 0.10)),
        ("10 m²",    (1160, 1185), (1.00, 0.45, 0.08)),
        ("1.0 m²",   (1186, 1212), (1.00, 0.72, 0.10)),
        ("0.01 m²",  (1213, 1238), (0.12, 0.92, 1.00)),
        ("0.001 m²", (1239, 1318), (1.00, 0.84, 0.18)),
    ]
    out_objs: list[bpy.types.Object] = [bg_obj]
    for idx, (txt, (f0, f1), rgb) in enumerate(steps):
        mat = _mat_solid_glow(f"Mat_FX_RCSCounter_{idx}", rgb, rgb, 1.85, metallic=0.1, roughness=0.25)
        cdata = bpy.data.curves.new(f"FX_B4_RCSNum_{idx}", type="FONT")
        cdata.body = txt
        if font is not None:
            cdata.font = font
        cdata.size = 0.076
        cdata.align_x = "CENTER"
        cdata.align_y = "CENTER"
        cdata.extrude = 0.003
        cdata.bevel_depth = 0.0006
        cdata.space_character = 1.08
        cdata.materials.append(mat)
        tobj = bpy.data.objects.new(f"FX_B4_RCSNum_{idx}", cdata)
        tobj.location = Vector((0.0, -0.060, -0.22))
        tobj.rotation_euler = Euler((math.radians(90.0), 0.0, 0.0), "XYZ")
        tobj.visible_shadow = False
        tobj.parent = bee_root
        col.objects.link(tobj)
        _key_visibility(tobj, [(f0, f1)], total_frames)
        out_objs.append(tobj)
    return out_objs


def _build_beat4_ram_and_bumblebee(col: bpy.types.Collection, total_frames: int) -> list[bpy.types.Object]:
    objs: list[bpy.types.Object] = []

    ram_root = bpy.data.objects.new("FX_B4_RAM_Root", None)
    ram_root.location = Vector(C.STAGE["ram_tile_pos"])
    ram_root.rotation_euler = Euler((math.radians(24.0), math.radians(-10.0), math.radians(16.0)), "XYZ")
    col.objects.link(ram_root)

    mat_hex = bpy.data.materials.new("Mat_FX_RAM_HexFerrite")
    mat_hex.use_nodes = True
    nt = mat_hex.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.048, 0.056, 0.070, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.65
    bsdf.inputs["Roughness"].default_value = 0.28
    bsdf.inputs["Emission Color"].default_value = (1.00, 0.36, 0.06, 1.0)
    heat_val = nt.nodes.new("ShaderNodeValue")
    heat_val.name = "RAM_Heat_Glow"
    heat_val.outputs[0].default_value = 0.0
    nt.links.new(heat_val.outputs[0], bsdf.inputs["Emission Strength"])

    mat_back = _mat_solid_glow("Mat_FX_RAM_Backplate", (0.20, 0.25, 0.32), (0.04, 0.45, 0.72), 0.22, metallic=0.88, roughness=0.26)

    bm_ram = bmesh.new()
    bmesh.ops.create_cube(bm_ram, size=1.0, matrix=Matrix.Translation((0.0, 0.0, -0.04)) @ Matrix.Diagonal((0.52, 0.40, 0.020, 1.0)))
    for f_face in bm_ram.faces:
        f_face.material_index = 1

    hex_r = 0.042
    dx = hex_r * 1.62
    dy = hex_r * math.sqrt(3.0) * 0.94
    for row in range(-2, 3):
        for col_i in range(-2, 3):
            hx = col_i * dx
            hy = row * dy + (0.5 * dy if (col_i % 2) else 0.0)
            if abs(hx) > 0.21 or abs(hy) > 0.16:
                continue
            res_h = bmesh.ops.create_cone(
                bm_ram, cap_ends=False, cap_tris=False, segments=6,
                radius1=hex_r * 0.88, radius2=hex_r * 0.88, depth=0.062,
                matrix=Matrix.Translation((hx, hy, 0.0)),
            )
            hv = set(res_h["verts"])
            h_faces = [f for f in bm_ram.faces if all(v in hv for v in f.verts)]
            for hf in h_faces:
                hf.material_index = 0
            bmesh.ops.solidify(bm_ram, geom=h_faces, thickness=0.008)

    ram_obj = _mesh_from_bm("FX_B4_RAM_Honeycomb", bm_ram, [mat_hex, mat_back], col, parent=ram_root, smooth=False)
    _key_visibility(ram_obj, [(961, 1210)], total_frames)
    objs.append(ram_obj)

    mat_rf = _mat_solid_glow("Mat_FX_RAM_InWave", (0.06, 0.90, 1.00), (0.06, 0.90, 1.00), 1.90)
    mat_heat = _mat_solid_glow("Mat_FX_RAM_HeatRing", (1.00, 0.42, 0.08), (1.00, 0.42, 0.08), 1.80)

    for i in range(3):
        rf_r = _mesh_from_bm(f"FX_B4_RFWave_{i}", _make_torus_bm(0.11 + 0.03 * i, 0.0055), [mat_rf], col, parent=ram_root)
        ht_r = _mesh_from_bm(f"FX_B4_HeatRipple_{i}", _make_torus_bm(0.08 + 0.04 * i, 0.0048), [mat_heat], col, parent=ram_root)
        _key_visibility(rf_r, [(975, 1155)], total_frames)
        _key_visibility(ht_r, [(1020, 1195)], total_frames)
        for f in range(975, 1196, 3):
            ph = ((f - 975 + i * 20) % 60) / 60.0
            rf_r.location = Vector((0.0, 0.0, C.lerp(0.36, 0.01, ph)))
            sc_rf = C.lerp(1.15, 0.15, ph)
            rf_r.scale = (sc_rf, sc_rf, 1.0)
            rf_r.keyframe_insert("location", frame=f)
            rf_r.keyframe_insert("scale", frame=f)

            ph_h = ((f - 1020 + i * 24) % 72) / 72.0
            ht_r.location = Vector((0.0, 0.0, C.lerp(0.03, 0.24, ph_h)))
            sc_h = C.lerp(0.45, 1.35, ph_h) * C.ram_heat_t(float(f))
            ht_r.scale = (max(0.02, sc_h), max(0.02, sc_h), 1.0)
            ht_r.keyframe_insert("location", frame=f)
            ht_r.keyframe_insert("scale", frame=f)
        objs.extend([rf_r, ht_r])

    for f in range(961, 1221, 5):
        heat_val.outputs[0].default_value = 1.25 * C.ram_heat_t(float(f))
        heat_val.outputs[0].keyframe_insert("default_value", frame=f)

    # Holographic 100 m^2 -> 0.001 m^2 RCS Collapse Orb, Live Counter & Sculpted Bumblebee
    bee_root = bpy.data.objects.new("FX_B4_Bumblebee_Root", None)
    bee_root.location = Vector(C.STAGE["rcs_bee_pos"])
    col.objects.link(bee_root)

    mat_orb, _ = _mat_additive_beam("Mat_FX_RCS_Orb", (1.00, 0.28, 0.06), strength=0.72, fresnel_pow=2.0)
    bm_orb = bmesh.new()
    bmesh.ops.create_icosphere(bm_orb, subdivisions=3, radius=0.40)
    for f_face in bm_orb.faces:
        f_face.material_index = 0
    orb_obj = _mesh_from_bm("FX_B4_RCS_ShrinkOrb", bm_orb, [mat_orb], col, parent=bee_root)
    _key_visibility(orb_obj, [(1120, 1320)], total_frames)

    counter_objs = _build_beat4_live_rcs_counter(col, bee_root, total_frames)

    mat_gold = _mat_solid_glow("Mat_Bee_GoldFur", (0.98, 0.72, 0.06), (1.00, 0.74, 0.08), 0.35, metallic=0.10, roughness=0.48)
    mat_blk = _mat_solid_glow("Mat_Bee_Obsidian", (0.025, 0.025, 0.032), (0.02, 0.02, 0.03), 0.04, metallic=0.22, roughness=0.34)
    mat_eye = _mat_solid_glow("Mat_Bee_CompoundEye", (0.04, 0.06, 0.08), (0.10, 0.22, 0.30), 0.20, metallic=0.78, roughness=0.06)
    mat_wing, _ = _mat_additive_beam("Mat_Bee_Wing", (0.84, 0.96, 1.00), strength=0.58, fresnel_pow=1.4)

    bm_bee = bmesh.new()
    segments_spec = [
        (-0.052, 0.022, 1),
        (-0.026, 0.032, 0),
        (-0.004, 0.034, 1),
        (+0.018, 0.035, 0),
        (+0.038, 0.030, 1),
        (+0.056, 0.022, 0),
        (+0.070, 0.012, 1),
    ]
    for sx, rad, midx in segments_spec:
        res_seg = bmesh.ops.create_icosphere(
            bm_bee, subdivisions=2, radius=rad,
            matrix=Matrix.Translation((sx, 0.0, 0.0)) @ Matrix.Diagonal((0.88, 1.0, 0.96, 1.0)),
        )
        sv = set(res_seg["verts"])
        for fs in bm_bee.faces:
            if all(v in sv for v in fs.verts):
                fs.material_index = midx

    for side in (-1.0, +1.0):
        res_e = bmesh.ops.create_icosphere(
            bm_bee, subdivisions=2, radius=0.010,
            matrix=Matrix.Translation((-0.062, side * 0.014, 0.007)) @ Matrix.Diagonal((0.85, 0.75, 1.25, 1.0)),
        )
        ev = set(res_e["verts"])
        for fe in bm_bee.faces:
            if all(v in ev for v in fe.verts):
                fe.material_index = 2
        res_a = bmesh.ops.create_cone(
            bm_bee, cap_ends=True, cap_tris=False, segments=8,
            radius1=0.0022, radius2=0.0014, depth=0.028,
            matrix=Matrix.Translation((-0.075, side * 0.012, 0.018))
            @ Euler((side * math.radians(24.0), math.radians(-52.0), 0.0), "XYZ").to_matrix().to_4x4(),
        )
        av = set(res_a["verts"])
        for fa in bm_bee.faces:
            if all(v in av for v in fa.verts):
                fa.material_index = 1

    bee_body = _mesh_from_bm("FX_B4_BumblebeeBody", bm_bee, [mat_gold, mat_blk, mat_eye], col, parent=bee_root, no_shadow=False)
    _key_visibility(bee_body, [(1145, 1320)], total_frames)

    bee_wings: list[tuple[bpy.types.Object, float]] = []
    for side in (-1.0, +1.0):
        bm_w = bmesh.new()
        bmesh.ops.create_icosphere(
            bm_w, subdivisions=2, radius=1.0,
            matrix=Matrix.Translation((0.0, side * 0.048, 0.0)) @ Matrix.Diagonal((0.028, 0.052, 0.002, 1.0)),
        )
        for fw in bm_w.faces:
            fw.material_index = 0
        w_obj = _mesh_from_bm(
            f"FX_B4_BeeWing_{'L' if side < 0 else 'R'}", bm_w, [mat_wing],
            col, parent=bee_root, location=(-0.010, side * 0.020, 0.028),
        )
        _key_visibility(w_obj, [(1145, 1320)], total_frames)
        bee_wings.append((w_obj, side))

    for f in range(1120, 1321, 2):
        sh = C.rcs_shrink_t(float(f))
        orb_s = C.lerp(1.0, 0.24, sh)
        orb_obj.scale = (orb_s, orb_s, orb_s)
        orb_obj.rotation_euler = Euler((0.0, 0.0, (f - 1120) * 0.04), "XYZ")
        orb_obj.keyframe_insert("scale", frame=f)
        orb_obj.keyframe_insert("rotation_euler", frame=f)

        bee_s = C.smootherstep(1150.0, 1195.0, float(f)) * 1.48
        bee_root.rotation_euler = Euler((math.radians(10.0), math.radians(-8.0), math.radians(-24.0 + 18.0 * sh)), "XYZ")
        bee_root.location = Vector(C.STAGE["rcs_bee_pos"]) + Vector((0.0, 0.0, 0.014 * math.sin((f - 1120) * 0.16)))
        bee_root.keyframe_insert("location", frame=f)
        bee_root.keyframe_insert("rotation_euler", frame=f)

        bee_body.scale = (bee_s, bee_s, bee_s)
        bee_body.keyframe_insert("scale", frame=f)
        for w_obj, side in bee_wings:
            flap = math.radians(32.0 * math.sin(f * 1.45) + 18.0)
            w_obj.rotation_euler = Euler((side * flap, 0.0, side * math.radians(-14.0)), "XYZ")
            w_obj.scale = (bee_s, bee_s, bee_s)
            w_obj.keyframe_insert("rotation_euler", frame=f)
            w_obj.keyframe_insert("scale", frame=f)

    objs.extend([orb_obj, bee_body] + counter_objs + [w for w, _ in bee_wings])
    return objs


# ==============================================================================
# 6. BEAT 5: MAIN-SCENE DAYLIGHT REAL-JET 3D REFLECTION ARROWS (ZERO BLUE WALL!)
# ==============================================================================
def _build_beat5_mainscene_arrows(col: bpy.types.Collection, total_frames: int) -> list[bpy.types.Object]:
    """
    Explains radar scattering DIRECTLY ON THE REAL JETS in bright daylight sky:
      - TOP (Real Conventional Jet): Incoming cyan 3D arrow hits its 90-degree vertical tail/fuselage
        (highlighted with a pulsing crimson echo burst ring!) -> BOLD RED 3D RETURN ARROW
        bounces 100% STRAIGHT BACK into the ground radar dish!
      - BOTTOM (Real B-2 Stealth Bomber): Incoming cyan 3D arrow hits its 33-degree tilted stealth
        wing facet (highlighted with a glowing tilted mirror plane & gold angle arc!) -> BOLD
        GOLDEN-CYAN 3D DEFLECTED ARROW ricochets SHARPLY AWAY into open sky!
    All 4 arrows are visible right from frame 1322 so the comparison reads instantly!
    """
    objs: list[bpy.types.Object] = []
    mat_cyan = _mat_solid_glow("Mat_B5_ArrowCyan", (0.04, 0.90, 1.00), (0.04, 0.90, 1.00), 1.92)
    mat_red = _mat_solid_glow("Mat_B5_ArrowRed", (1.00, 0.14, 0.06), (1.00, 0.14, 0.06), 1.98)
    mat_gold = _mat_solid_glow("Mat_B5_ArrowGold", (1.00, 0.82, 0.12), (1.00, 0.82, 0.12), 1.95)
    mat_facet, _ = _mat_additive_beam("Mat_B5_FacetMirror", (0.12, 0.96, 1.00), strength=0.65, fresnel_pow=1.2)

    dish5 = Vector(C.STAGE["radar_dish_b5"]) + Vector((0.20, 0.0, 0.12))
    top_hit = Vector(C.STAGE["b5_top_mirror_pos"]) + Vector((-0.14, -0.08, 0.06))
    bot_hit = Vector(C.STAGE["b5_bot_facet_pos"]) + Vector((-0.24, -0.10, 0.08))
    # Ricochets sharply down-right / forward-right into open sky away from the dish AND clear of top jet!
    bot_sky = bot_hit + Vector((0.92, -0.10, -0.58))

    # --- TOP (Real Conv Jet): Cyan incoming arrow -> Crimson echo ring -> Bold RED return arrow ---
    arr_top_in = _make_3d_arrow_obj(
        "FX_B5_TopArrowIn", dish5 + Vector((0.0, 0.0, 0.14)), top_hit + Vector((0.0, 0.0, 0.05)),
        shaft_r=0.016, head_r=0.045, head_len=0.12, mat=mat_cyan, col=col,
    )
    arr_top_ret = _make_3d_arrow_obj(
        "FX_B5_TopArrowEchoBack", top_hit + Vector((0.0, 0.0, -0.07)), dish5 + Vector((0.04, 0.0, -0.02)),
        shaft_r=0.026, head_r=0.066, head_len=0.16, mat=mat_red, col=col,
    )
    _key_visibility(arr_top_in, [(1322, 1560)], total_frames)
    _key_visibility(arr_top_ret, [(1326, 1560)], total_frames)
    objs.extend([arr_top_in, arr_top_ret])

    # Pulsing crimson radar-return shock ring at the Conventional Jet bounce point (replaces awkward stick chair!)
    echo_ring = _mesh_from_bm(
        "FX_B5_JetEchoRing", _make_torus_bm(0.16, 0.007, major_seg=32, minor_seg=8),
        [mat_red], col, location=(top_hit.x, top_hit.y - 0.05, top_hit.z),
        rotation=(math.radians(78.0), math.radians(-18.0), 0.0),
    )
    _key_visibility(echo_ring, [(1326, 1560)], total_frames)
    for f in range(1326, 1561, 3):
        p = 0.82 + 0.35 * abs(math.sin((f - 1326) * 0.22))
        echo_ring.scale = (p, p, p)
        echo_ring.keyframe_insert("scale", frame=f)
    objs.append(echo_ring)

    # --- BOTTOM (Real B-2 Stealth Bomber): Cyan incoming arrow -> Tilted Mirror Plate -> Bold GOLD-CYAN arrow AWAY! ---
    arr_bot_in = _make_3d_arrow_obj(
        "FX_B5_BotArrowIn", dish5 + Vector((0.0, 0.0, -0.10)), bot_hit,
        shaft_r=0.016, head_r=0.045, head_len=0.12, mat=mat_cyan, col=col,
    )
    arr_bot_out = _make_3d_arrow_obj(
        "FX_B5_BotArrowDeflectUp", bot_hit, bot_sky,
        shaft_r=0.026, head_r=0.066, head_len=0.16, mat=mat_gold, col=col,
    )
    _key_visibility(arr_bot_in, [(1322, 1560)], total_frames)
    _key_visibility(arr_bot_out, [(1328, 1560)], total_frames)
    objs.extend([arr_bot_in, arr_bot_out])

    # Glowing 33-deg tilted mirror facet plate + gold rim sitting flush along B-2 leading edge chine!
    bm_fp = bmesh.new()
    bmesh.ops.create_cube(bm_fp, size=1.0, matrix=Matrix.Diagonal((0.36, 0.16, 0.008, 1.0)))
    for f_fp in bm_fp.faces:
        f_fp.material_index = 0
    facet_plate = _mesh_from_bm(
        "FX_B5_TiltedFacetPlate", bm_fp, [mat_facet], col,
        location=(bot_hit.x + 0.04, bot_hit.y - 0.04, bot_hit.z + 0.01),
        rotation=(math.radians(24.0), math.radians(-28.0), math.radians(-38.0)),
        smooth=False,
    )
    _key_visibility(facet_plate, [(1325, 1560)], total_frames)
    objs.append(facet_plate)

    # Crisp glowing gold reflection-angle arc right at the B-2 wing bounce vertex!
    bm_arc = bmesh.new()
    arc_r = 0.20
    n_arc = 14
    for k in range(n_arc):
        a0 = math.radians(175.0 + k * (135.0 / n_arc))
        a1 = math.radians(175.0 + (k + 1) * (135.0 / n_arc))
        p0 = Vector((arc_r * math.cos(a0), 0.0, arc_r * math.sin(a0)))
        p1 = Vector((arc_r * math.cos(a1), 0.0, arc_r * math.sin(a1)))
        seg = p1 - p0
        mid = 0.5 * (p0 + p1)
        rot_m = seg.normalized().to_track_quat("Z", "Y").to_matrix().to_4x4()
        bmesh.ops.create_cone(
            bm_arc, cap_ends=True, cap_tris=False, segments=8,
            radius1=0.009, radius2=0.009, depth=seg.length + 0.006,
            matrix=Matrix.Translation(mid) @ rot_m,
        )
    for fa in bm_arc.faces:
        fa.material_index = 0
    arc_obj = _mesh_from_bm(
        "FX_B5_StealthAngleArc", bm_arc, [mat_gold], col,
        location=(bot_hit.x, bot_hit.y - 0.08, bot_hit.z), smooth=True,
    )
    _key_visibility(arc_obj, [(1328, 1560)], total_frames)
    objs.append(arc_obj)

    # Animated traveling glowing RF energy pulse orbs sliding cleanly along all 4 arrow paths!
    for tag, p0, p1, f_start, mat_dot in (
        ("TopIn", dish5 + Vector((0.0, 0.0, 0.14)), top_hit + Vector((0.0, 0.0, 0.05)), 1322, mat_cyan),
        ("TopEcho", top_hit + Vector((0.0, 0.0, -0.07)), dish5 + Vector((0.04, 0.0, -0.02)), 1328, mat_red),
        ("BotIn", dish5 + Vector((0.0, 0.0, -0.10)), bot_hit, 1324, mat_cyan),
        ("BotOut", bot_hit, bot_sky, 1330, mat_gold),
    ):
        bm_d = bmesh.new()
        bmesh.ops.create_icosphere(bm_d, subdivisions=2, radius=0.034)
        for fd in bm_d.faces:
            fd.material_index = 0
        dot = _mesh_from_bm(f"FX_B5_Pulse_{tag}", bm_d, [mat_dot], col)
        _key_visibility(dot, [(f_start, 1560)], total_frames)
        for f in range(f_start, 1561, 3):
            ph = ((f - f_start) % 40) / 40.0
            dot.location = p0.lerp(p1, ph)
            dot.keyframe_insert("location", frame=f)
        objs.append(dot)

    return objs


# ==============================================================================
# 7. BEAT 6: HOODED ANTI-GLARE ENEMY COMMAND ROOM PPI RADAR SCOPE (ZERO GLARE!)
# ==============================================================================
def _build_beat6_radar_scope(col: bpy.types.Collection, total_frames: int) -> list[bpy.types.Object]:
    """
    Hero recessed hooded military PPI Radar Scope embedded directly in the Enemy Command Room
    console desk (`b6_scope_pos`):
    - Anti-glare matte dark green phosphor CRT surface (ZERO ugly white square light reflections!)
    - Machined steel bezel + 8 hex mounting bolts + protective top sun-glare visor hood
    - Razor-sharp concentric range rings, 12 compass ticks, clockwise phosphor sweep wedge,
      and ONE tiny pulsing golden-amber `0.001 m²` bumblebee speck with target acquisition halo!
    """
    objs: list[bpy.types.Object] = []
    scope_root = bpy.data.objects.new("FX_B6_RadarScope_Root", None)
    scope_root.location = Vector(C.STAGE["b6_scope_pos"])
    scope_root.rotation_euler = Euler((math.radians(66.0), 0.0, 0.0), "XYZ")
    col.objects.link(scope_root)

    r_scope = float(C.STAGE["b6_scope_radius"])
    mat_bezel = _mat_solid_glow("Mat_Scope_Bezel", (0.045, 0.055, 0.072), (0.02, 0.22, 0.30), 0.08, metallic=0.88, roughness=0.32, spec_ior=0.18)
    # Matte anti-glare CRT phosphor dial so ZERO rectangular AREA lights ever reflect on the radar screen!
    mat_dial = _mat_solid_glow("Mat_Scope_AntiGlareDial", (0.003, 0.014, 0.012), (0.006, 0.075, 0.058), 0.16, metallic=0.04, roughness=0.72, spec_ior=0.02)
    mat_grid = _mat_solid_glow("Mat_Scope_PhosphorGrid", (0.06, 0.96, 0.66), (0.06, 0.96, 0.66), 1.65)
    mat_blip = _mat_solid_glow("Mat_Scope_BeeBlip", (1.00, 0.80, 0.10), (1.00, 0.80, 0.10), 1.98)
    mat_wedge, _ = _mat_additive_beam("Mat_Scope_SweepWedge", (0.08, 0.98, 0.66), strength=0.92, fresnel_pow=0.5)

    bm_sc = bmesh.new()
    bmesh.ops.create_cone(
        bm_sc, cap_ends=True, cap_tris=False, segments=48,
        radius1=r_scope * 1.12, radius2=r_scope * 1.07, depth=0.064,
        matrix=Matrix.Translation((0.0, 0.0, -0.032)),
    )
    # Anti-glare raised military visor collar around screen perimeter
    bmesh.ops.create_cone(
        bm_sc, cap_ends=False, cap_tris=False, segments=48,
        radius1=r_scope * 1.06, radius2=r_scope * 1.03, depth=0.042,
        matrix=Matrix.Translation((0.0, 0.0, 0.018)),
    )
    for fs in bm_sc.faces:
        fs.material_index = 0
    for k in range(8):
        a = 2.0 * math.pi * k / 8.0 + math.pi / 8.0
        bmesh.ops.create_cone(
            bm_sc, cap_ends=True, cap_tris=False, segments=6,
            radius1=0.013, radius2=0.013, depth=0.014,
            matrix=Matrix.Translation((r_scope * 1.08 * math.cos(a), r_scope * 1.08 * math.sin(a), 0.005)),
        )
    res_d = bmesh.ops.create_cone(
        bm_sc, cap_ends=True, cap_tris=False, segments=48,
        radius1=r_scope, radius2=r_scope, depth=0.010,
        matrix=Matrix.Translation((0.0, 0.0, -0.002)),
    )
    dv = set(res_d["verts"])
    for fd in bm_sc.faces:
        if all(v in dv for v in fd.verts):
            fd.material_index = 1
    scope_body = _mesh_from_bm("FX_B6_ScopeBody", bm_sc, [mat_bezel, mat_dial], col, parent=scope_root, no_shadow=False)
    _key_visibility(scope_body, [(1561, total_frames)], total_frames)
    objs.append(scope_body)

    # Concentric range rings + crosshairs + 12 compass ticks
    bm_g = bmesh.new()
    for frac in (0.33, 0.66, 0.96):
        bm_t = _make_torus_bm(r_scope * frac, 0.0030, major_seg=48, minor_seg=8)
        for v in bm_t.verts:
            v.co.z += 0.008
        tmp_m = bpy.data.meshes.new("_tmp_ring6")
        bm_t.to_mesh(tmp_m)
        bm_t.free()
        bm_g.from_mesh(tmp_m)
        bpy.data.meshes.remove(tmp_m)
    bmesh.ops.create_cube(bm_g, size=1.0, matrix=Matrix.Translation((0.0, 0.0, 0.008)) @ Matrix.Diagonal((r_scope * 1.92, 0.004, 0.003, 1.0)))
    bmesh.ops.create_cube(bm_g, size=1.0, matrix=Matrix.Translation((0.0, 0.0, 0.008)) @ Matrix.Diagonal((0.004, r_scope * 1.92, 0.003, 1.0)))
    for k in range(12):
        a = 2.0 * math.pi * k / 12.0
        bmesh.ops.create_cube(
            bm_g, size=1.0,
            matrix=Matrix.Translation((r_scope * 0.90 * math.cos(a), r_scope * 0.90 * math.sin(a), 0.008))
            @ Euler((0.0, 0.0, a), "XYZ").to_matrix().to_4x4()
            @ Matrix.Diagonal((0.028, 0.0036, 0.003, 1.0)),
        )
    for fg in bm_g.faces:
        fg.material_index = 0
    scope_grid = _mesh_from_bm("FX_B6_ScopeGrid", bm_g, [mat_grid], col, parent=scope_root)
    _key_visibility(scope_grid, [(1561, total_frames)], total_frames)
    objs.append(scope_grid)

    # Rotating Phosphor Sweep Needle + Trailing Wedge
    bm_sw = bmesh.new()
    bmesh.ops.create_cube(bm_sw, size=1.0, matrix=Matrix.Translation((0.0, 0.48 * r_scope, 0.012)) @ Matrix.Diagonal((0.006, 0.96 * r_scope, 0.003, 1.0)))
    for fsw in bm_sw.faces:
        fsw.material_index = 0
    vc = bm_sw.verts.new((0.0, 0.0, 0.011))
    wedge_pts = []
    for k in range(11):
        ang = math.radians(90.0 + k * 3.6)
        wedge_pts.append(bm_sw.verts.new((r_scope * 0.95 * math.cos(ang), r_scope * 0.95 * math.sin(ang), 0.011)))
    for k in range(len(wedge_pts) - 1):
        bm_sw.faces.new((vc, wedge_pts[k], wedge_pts[k + 1])).material_index = 1
    sweep_obj = _mesh_from_bm("FX_B6_ScopeSweep", bm_sw, [mat_grid, mat_wedge], col, parent=scope_root)
    _key_visibility(sweep_obj, [(1561, total_frames)], total_frames)
    for f in range(1561, total_frames + 1, 2):
        sweep_obj.rotation_euler = Euler((0.0, 0.0, C.radar_scope_angle_rad(float(f))), "XYZ")
        sweep_obj.keyframe_insert("rotation_euler", frame=f)
    objs.append(sweep_obj)

    # Solitary Tiny `0.001 m²` Bumblebee Speck + Pulsing Target-Acquisition Halo
    bm_bl = bmesh.new()
    bmesh.ops.create_icosphere(bm_bl, subdivisions=2, radius=0.013)
    for fbl in bm_bl.faces:
        fbl.material_index = 0
    blip_obj = _mesh_from_bm(
        "FX_B6_SolitaryBeeBlip", bm_bl, [mat_blip], col,
        location=(0.075, 0.068, 0.014), parent=scope_root,
    )
    _key_visibility(blip_obj, [(1561, total_frames)], total_frames)

    ret_obj = _mesh_from_bm(
        "FX_B6_BlipReticle", _make_torus_bm(0.036, 0.0028, major_seg=28, minor_seg=8), [mat_blip], col,
        location=(0.075, 0.068, 0.014), parent=scope_root,
    )
    _key_visibility(ret_obj, [(1561, total_frames)], total_frames)

    for f in range(1561, total_frames + 1, 3):
        pulse = 0.80 + 0.45 * abs(math.sin((f - 1561) * 0.14))
        blip_obj.scale = (pulse, pulse, pulse)
        blip_obj.keyframe_insert("scale", frame=f)
        rp = 0.90 + 0.35 * ((f - 1561) % 36) / 36.0
        ret_obj.scale = (rp, rp, 1.0)
        ret_obj.keyframe_insert("scale", frame=f)
    objs.extend([blip_obj, ret_obj])

    return objs


def build_radar_fx(
    col: bpy.types.Collection,
    bomber_root: bpy.types.Object,
    total_frames: int = C.TOTAL_FRAMES,
) -> dict[str, Any]:
    """Assemble all 6 beats of volumetric radar FX, console scopes, RAM honeycomb, bumblebee & real-jet arrows."""
    contrails = _build_wingtip_contrails(col, bomber_root, total_frames)
    b1_monitor = _build_beat1_blank_monitor(col, total_frames)
    b12 = _build_beats12_radar_optics(col, total_frames)
    b3 = _build_beat3_sduct_wave(col, bomber_root, total_frames)
    b4 = _build_beat4_ram_and_bumblebee(col, total_frames)
    b5 = _build_beat5_mainscene_arrows(col, total_frames)
    b6 = _build_beat6_radar_scope(col, total_frames)
    return {
        "contrails": contrails,
        "b1_monitor": b1_monitor,
        "beats12": b12,
        "beat3": b3,
        "beat4": b4,
        "beat5": b5,
        "beat6": b6,
    }
