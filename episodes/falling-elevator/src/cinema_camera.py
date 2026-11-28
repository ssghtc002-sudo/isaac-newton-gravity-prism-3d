"""
Cinema Camera Engine for Episode #16:
"Why Jumping in a Falling Elevator WON'T Save You" (60 fps, 1800 frames, 9:16 Portrait).

Pure-Python safe (`bpy` / `mathutils` imported conditionally) so `verify_framing.py` and
`labels_3d.py` can import `camera_state_at` and `project_ndc` directly outside Blender
with 0.000 mm trajectory, focus-plane, or lens drift.

Public API:
  - project_ndc(pt_w, cam_pos, tgt_pos, lens_mm: float) -> tuple[float, float, float]
  - camera_state_at(f: float, include_motion_fx: bool = True) -> dict
      returns {"cam_pos": (x, y, z), "tgt_pos": (x, y, z),
               "lens_mm": float, "fstop": float, "focus_dist": float}
  - build_cinema_camera(col: bpy.types.Collection, total_frames: int = 1800) -> bpy.types.Object
      creates `Cam_Main` + `Cam_Target` (`TRACK_TO` -Z / UP_Y), keys transform, lens,
      DOF `focus_distance` and `aperture_fstop` across `1..total_frames`, sets `CONSTANT`
      handles on hard-cut boundary frames, and sets `scene.camera`.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Sequence

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import elevator_contract as C  # noqa: E402

try:
    import bpy  # type: ignore[import-not-found]
    from mathutils import Vector  # type: ignore[import-not-found]
except ImportError:  # Pure-Python mode (verify_framing.py, labels_3d static checks)
    bpy = None  # type: ignore[assignment]
    Vector = None  # type: ignore[assignment]

Vec3 = tuple[float, float, float]

# ==============================================================================
# 9:16 PORTRAIT OPTICAL CONSTANTS (Blender sensor_fit='AUTO', 1080x1920)
# ==============================================================================
RES_X = 1080
RES_Y = 1920
ASPECT_W_OVER_H = RES_X / RES_Y        # 0.5625
SENSOR_WIDTH_MM = 36.0
SENSOR_H_EFF_MM = SENSOR_WIDTH_MM      # 36.0 mm (portrait height is major axis)
SENSOR_W_EFF_MM = SENSOR_WIDTH_MM * ASPECT_W_OVER_H  # 20.25 mm

HARD_CUT_FRAMES: tuple[int, ...] = (313, 601, 901, 1237, 1386)
_HARD_CUT_PREV_FRAMES: set[int] = {cut - 1 for cut in HARD_CUT_FRAMES}


# ==============================================================================
# PURE-PYTHON VECTOR & INTERPOLATION HELPERS
# ==============================================================================
def _add3(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2]))


def _sub3(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2]))


def _dot3(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def _cross3(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    )


def _len3(a: Sequence[float]) -> float:
    return math.sqrt(_dot3(a, a))


def _norm3(a: Sequence[float]) -> Vec3:
    length = _len3(a) or 1e-9
    return (float(a[0]) / length, float(a[1]) / length, float(a[2]) / length)


def _eval_piecewise_scalar(kfs: Sequence[tuple[float, float]], f: float) -> float:
    if f <= kfs[0][0]:
        return kfs[0][1]
    if f >= kfs[-1][0]:
        return kfs[-1][1]
    for (f0, v0), (f1, v1) in zip(kfs, kfs[1:]):
        if f <= f1:
            u = C.smootherstep(f0, f1, f)
            return C.lerp(v0, v1, u)
    return kfs[-1][1]


def _eval_piecewise_vec3(kfs: Sequence[tuple[float, Vec3]], f: float) -> Vec3:
    if f <= kfs[0][0]:
        return kfs[0][1]
    if f >= kfs[-1][0]:
        return kfs[-1][1]
    for (f0, v0), (f1, v1) in zip(kfs, kfs[1:]):
        if f <= f1:
            u = C.smootherstep(f0, f1, f)
            return C.lerp3(v0, v1, u)
    return kfs[-1][1]


def project_ndc(
    pt_w: Sequence[float],
    cam_pos: Sequence[float],
    tgt_pos: Sequence[float],
    lens_mm: float,
) -> tuple[float, float, float]:
    """Project a 3D world point into 9:16 portrait NDC [-1, +1] matching Blender sensor_fit='AUTO'."""
    fwd = _norm3(_sub3(tgt_pos, cam_pos))
    right = _norm3(_cross3(fwd, (0.0, 0.0, 1.0)))
    up = _cross3(right, fwd)
    rel = _sub3(pt_w, cam_pos)
    z = _dot3(rel, fwd)
    x = _dot3(rel, right)
    y = _dot3(rel, up)
    if z <= 0.01:
        return (99.0, 99.0, z)
    ndc_x = (x / z) * (lens_mm / (0.5 * SENSOR_W_EFF_MM))
    ndc_y = (y / z) * (lens_mm / (0.5 * SENSOR_H_EFF_MM))
    return (ndc_x, ndc_y, z)


# ==============================================================================
# MOTION FX LAYERS (CABLE TREMOR, CEILING RATTLE, PIT SLAM, BRAKE SHUDDER)
# ==============================================================================
def _motion_fx(f: float) -> tuple[Vec3, Vec3, float]:
    """Physical camera micro-impulses & tremors cleanly bounded within each beat."""
    cx = cy = cz = 0.0
    tx = ty = tz = 0.0
    lens_d = 0.0

    # Beat 1 (f=240..312): Subtle pre-snap hoist-cable tremor
    if 240.0 <= f <= 312.0:
        w = C.smoothstep(240.0, 265.0, f)
        cx += w * (0.0038 * math.sin(f * 0.92 + 0.3) + 0.0018 * math.cos(f * 2.15))
        cy += w * (0.0024 * math.cos(f * 0.78 + 0.9))
        cz += w * (0.0045 * math.sin(f * 1.25 + 1.6))
        tx += w * (0.0015 * math.sin(f * 0.92 + 0.3))
        tz += w * (0.0020 * math.sin(f * 1.25 + 1.6))

    # Beat 2 (f=325..355): Cable-snap recoil micro-jolt + subtle 50 mph shaft wind flutter
    if 325.0 <= f <= 355.0:
        dt = f - 325.0
        env = C.smoothstep(325.0, 327.0, f) * math.exp(-0.14 * dt) * (1.0 - C.smoothstep(348.0, 355.0, f))
        cz += -0.014 * env * math.sin(dt * 1.15)
        cx += 0.006 * env * math.cos(dt * 1.45)
        tz += -0.005 * env * math.sin(dt * 1.15)
    if 360.0 <= f <= 600.0:
        w = C.smoothstep(360.0, 420.0, f) * (1.0 - C.smoothstep(585.0, 600.0, f))
        cx += w * 0.0022 * math.sin(f * 0.55)
        cz += w * 0.0028 * math.cos(f * 0.67)

    # Beat 3 (f=750..778): Micro camera rattle on ceiling head-hit (EV['ceiling_hit']=750)
    f_hit = float(C.EV["ceiling_hit"])
    if f_hit <= f <= f_hit + 28.0:
        dt = f - f_hit
        env = C.smoothstep(f_hit, f_hit + 2.0, f) * math.exp(-0.16 * dt) * (1.0 - C.smoothstep(f_hit + 20.0, f_hit + 28.0, f))
        cz += 0.015 * env * math.sin(dt * 1.85 + 0.2)
        cx += 0.008 * env * math.cos(dt * 2.15)
        tz += 0.006 * env * math.sin(dt * 1.85 + 0.2)
        lens_d += 0.45 * env

    # Beat 4 (f=940..985): 28G pit impact camera slam pulse (EV['pit_impact']..EV['pit_peak_comp'])
    f_pit = float(C.EV["pit_impact"])
    if f_pit <= f <= 996.0:
        dt = f - f_pit
        rise = C.smoothstep(f_pit, f_pit + 3.0, f)
        fade = 1.0 - C.smoothstep(980.0, 996.0, f)
        env = rise * fade * math.exp(-0.075 * dt)
        cz += -0.022 * env * (0.65 + 0.35 * math.sin(dt * 1.55))
        cx += 0.010 * env * math.cos(dt * 1.95)
        cy += -0.008 * env * math.sin(dt * 1.35)
        tz += -0.008 * env * math.sin(dt * 1.55)
        lens_d += 0.75 * env

    # Beat 5a (f=1275..1355): Steel wedge rail-clamp arrest shudder
    f_clamp = float(C.EV["brake_clamp"])
    if f_clamp <= f <= 1355.0:
        w = C.smoothstep(f_clamp, f_clamp + 6.0, f) * (1.0 - C.smoothstep(1325.0, 1355.0, f))
        cx += w * (0.0075 * math.sin(f * 1.65) + 0.004 * math.cos(f * 2.85))
        cz += w * (0.0085 * math.cos(f * 1.92))
        tz += w * (0.0030 * math.cos(f * 1.92))

    return (cx, cy, cz), (tx, ty, tz), lens_d


# ==============================================================================
# 6-BEAT CAB-LOCAL CAMERA CHOREOGRAPHY
# ==============================================================================
# Beat 1 (1..312): Outside front-right glass doors (y < -0.70) looking into cab
_B1_CAM_KFS: list[tuple[float, Vec3]] = [
    (1.0,   (0.62, -2.64, 1.18)),
    (115.0, (0.60, -2.62, 1.17)),
    (190.0, (0.58, -2.60, 1.17)),
    (260.0, (0.56, -2.59, 1.16)),
    (312.0, (0.55, -2.58, 1.15)),
]
_B1_TGT_KFS: list[tuple[float, Vec3]] = [
    (1.0,   (-0.05, -0.04, 1.07)),
    (190.0, (-0.05, -0.03, 1.06)),
    (312.0, (-0.05, -0.02, 1.06)),
]
_B1_LENS_KFS: list[tuple[float, float]] = [
    (1.0,   20.0),
    (190.0, 20.1),
    (312.0, 20.2),
]

# Beat 2 (313..600): External cutaway left-front-above showing snapped hoist cables + plunging cab
_B2_CAM_KFS: list[tuple[float, Vec3]] = [
    (313.0, (-1.38, -2.65, 2.22)),
    (420.0, (-1.34, -2.60, 2.18)),
    (520.0, (-1.30, -2.56, 2.14)),
    (600.0, (-1.28, -2.54, 2.12)),
]
_B2_TGT_KFS: list[tuple[float, Vec3]] = [
    (313.0, (0.00, 0.00, 1.38)),
    (460.0, (0.00, 0.00, 1.35)),
    (600.0, (0.00, 0.00, 1.33)),
]
_B2_LENS_KFS: list[tuple[float, float]] = [
    (313.0, 23.0),
    (460.0, 23.4),
    (600.0, 23.8),
]

# Beat 3 (601..900): Side-front low-3/4 myth-jump camera (floor z=0 up to ceiling z=2.35)
_B3_CAM_KFS: list[tuple[float, Vec3]] = [
    (601.0, (1.68, -2.32, 1.24)),
    (700.0, (1.66, -2.30, 1.24)),
    (760.0, (1.65, -2.32, 1.26)),
    (900.0, (1.66, -2.30, 1.22)),
]
_B3_TGT_BASE_KFS: list[tuple[float, Vec3]] = [
    (601.0, (-0.04, 0.02, 0.99)),
    (750.0, (-0.04, 0.02, 1.00)),
    (900.0, (-0.04, 0.02, 0.98)),
]
_B3_LENS_KFS: list[tuple[float, float]] = [
    (601.0, 22.0),
    (750.0, 21.8),
    (900.0, 22.2),
]

# Beat 4 (901..1236): Tight X-ray biomechanics angle (ankles/buffer up to skull)
_B4_CAM_KFS: list[tuple[float, Vec3]] = [
    (901.0,  (1.58, -2.22, 0.85)),
    (940.0,  (1.56, -2.20, 0.85)),
    (985.0,  (1.54, -2.18, 0.84)),
    (1120.0, (1.56, -2.20, 0.85)),
    (1236.0, (1.58, -2.22, 0.86)),
]
_B4_TGT_KFS: list[tuple[float, Vec3]] = [
    (901.0,  (-0.02, 0.02, 0.68)),
    (985.0,  (-0.02, 0.02, 0.67)),
    (1236.0, (-0.02, 0.02, 0.69)),
]
_B4_LENS_KFS: list[tuple[float, float]] = [
    (901.0,  23.0),
    (985.0,  23.2),
    (1236.0, 23.0),
]

# Beat 5a (1237..1385): Underside/side rail brake view (steel safety clamps + T-rail + passenger)
_B5A_CAM_KFS: list[tuple[float, Vec3]] = [
    (1237.0, (1.88, -2.05, 0.34)),
    (1290.0, (1.90, -2.08, 0.36)),
    (1385.0, (1.92, -2.12, 0.38)),
]
_B5A_TGT_KFS: list[tuple[float, Vec3]] = [
    (1237.0, (0.48, 0.00, 0.42)),
    (1290.0, (0.47, 0.00, 0.43)),
    (1385.0, (0.46, 0.00, 0.44)),
]
_B5A_LENS_KFS: list[tuple[float, float]] = [
    (1237.0, 25.5),
    (1310.0, 25.8),
    (1385.0, 26.0),
]

# Beat 5b-6 (1386..1800): Elevated 3/4 interior survival reveal -> gentle orbit over 12x grid
_B56_CAM_KFS: list[tuple[float, Vec3]] = [
    (1386.0, (-0.42, -1.95, 1.52)),
    (1480.0, (-0.20, -1.94, 1.54)),
    (1580.0, (0.06,  -1.92, 1.55)),
    (1690.0, (0.30,  -1.88, 1.57)),
    (1800.0, (0.48,  -1.85, 1.58)),
]
_B56_TGT_KFS: list[tuple[float, Vec3]] = [
    (1386.0, (0.02, 0.02, 0.14)),
    (1545.0, (0.02, 0.02, 0.14)),
    (1800.0, (0.02, 0.02, 0.14)),
]
_B56_LENS_KFS: list[tuple[float, float]] = [
    (1386.0, 24.5),
    (1545.0, 24.8),
    (1800.0, 25.0),
]


def _focus_point_local(f: float, tgt_local: Vec3) -> Vec3:
    """Cab-local DOF focus plane keeping hero face/skeleton/clamps/grid razor-sharp."""
    px, py = C.PASSENGER["stand_xy"]
    if f <= 312.0:
        # Beat 1: Face/chest focus plane (drops subtly as passenger crouches f=190..312)
        z_face = C.lerp(1.52, 1.24, C.smootherstep(190.0, 300.0, f))
        return (px, py, z_face)
    if f <= 600.0:
        # Beat 2: Roof hitch & crouching passenger center plane
        return C.lerp3((0.0, 0.0, 1.85), (px, py, 1.35), C.smootherstep(325.0, 520.0, f))
    if f <= 900.0:
        # Beat 3: Jumping passenger plane
        return (px, py, 1.15 + 0.55 * C.passenger_jump_offset_z(f))
    if f <= 1236.0:
        # Beat 4: Translucent skeleton & spine plane
        return (px, py, 1.05)
    if f <= 1385.0:
        # Beat 5a: Steel safety clamp biting T-rail
        return (0.92, 0.0, float(C.CAB["clamp_z"]) + 0.10)
    # Beat 5b-6: Flat-back passenger & 12x floor force grid
    return C.PASSENGER["flat_pelvis"]


def camera_state_at(f: float, include_motion_fx: bool = True) -> dict:
    """
    Evaluate exact 9:16 portrait world-space camera state at frame `f` in [1, 1800].
    Returns {"cam_pos": (x,y,z), "tgt_pos": (x,y,z), "lens_mm": float, "fstop": float, "focus_dist": float}.
    """
    f = max(1.0, min(float(C.TOTAL_FRAMES), float(f)))

    if f <= 312.0:
        cam_local = _eval_piecewise_vec3(_B1_CAM_KFS, f)
        tgt_local = _eval_piecewise_vec3(_B1_TGT_KFS, f)
        lens_mm = _eval_piecewise_scalar(_B1_LENS_KFS, f)
        fstop = 4.8
    elif f <= 600.0:
        cam_local = _eval_piecewise_vec3(_B2_CAM_KFS, f)
        tgt_local = _eval_piecewise_vec3(_B2_TGT_KFS, f)
        lens_mm = _eval_piecewise_scalar(_B2_LENS_KFS, f)
        fstop = 6.3
    elif f <= 900.0:
        cam_local = _eval_piecewise_vec3(_B3_CAM_KFS, f)
        base_tgt = _eval_piecewise_vec3(_B3_TGT_BASE_KFS, f)
        tgt_local = (base_tgt[0], base_tgt[1], base_tgt[2] + 0.16 * C.passenger_jump_offset_z(f))
        lens_mm = _eval_piecewise_scalar(_B3_LENS_KFS, f)
        fstop = 5.6
    elif f <= 1236.0:
        cam_local = _eval_piecewise_vec3(_B4_CAM_KFS, f)
        tgt_local = _eval_piecewise_vec3(_B4_TGT_KFS, f)
        lens_mm = _eval_piecewise_scalar(_B4_LENS_KFS, f)
        fstop = 6.3
    elif f <= 1385.0:
        cam_local = _eval_piecewise_vec3(_B5A_CAM_KFS, f)
        tgt_local = _eval_piecewise_vec3(_B5A_TGT_KFS, f)
        lens_mm = _eval_piecewise_scalar(_B5A_LENS_KFS, f)
        fstop = 6.3
    else:
        cam_local = _eval_piecewise_vec3(_B56_CAM_KFS, f)
        tgt_local = _eval_piecewise_vec3(_B56_TGT_KFS, f)
        lens_mm = _eval_piecewise_scalar(_B56_LENS_KFS, f)
        fstop = 6.3

    cab_w = C.cab_world_location(f)
    cam_pos = _add3(cab_w, cam_local)
    tgt_pos = _add3(cab_w, tgt_local)
    focus_pt = _add3(cab_w, _focus_point_local(f, tgt_local))

    if include_motion_fx:
        fx_cam, fx_tgt, fx_lens = _motion_fx(f)
        cam_pos = _add3(cam_pos, fx_cam)
        tgt_pos = _add3(tgt_pos, fx_tgt)
        lens_mm += fx_lens

    fwd = _norm3(_sub3(tgt_pos, cam_pos))
    focus_dist = max(0.25, _dot3(_sub3(focus_pt, cam_pos), fwd))

    return {
        "cam_pos": cam_pos,
        "tgt_pos": tgt_pos,
        "lens_mm": float(lens_mm),
        "fstop": float(fstop),
        "focus_dist": float(focus_dist),
    }


# ==============================================================================
# BLENDER 4.2 SCENE BUILDER PUBLIC API
# ==============================================================================
def _link_to_collection(obj, col) -> None:
    target_col = col if col is not None else bpy.context.scene.collection
    if obj.name not in target_col.objects:
        target_col.objects.link(obj)


def _set_handles_with_hard_cuts(id_block) -> None:
    anim = getattr(id_block, "animation_data", None)
    if anim is None or anim.action is None:
        return
    for fcurve in C.iter_action_fcurves(anim.action):
        for kp in fcurve.keyframe_points:
            frame_i = int(round(kp.co.x))
            if frame_i in _HARD_CUT_PREV_FRAMES:
                kp.interpolation = "CONSTANT"
            else:
                kp.interpolation = "BEZIER"
                kp.handle_left_type = "AUTO_CLAMPED"
                kp.handle_right_type = "AUTO_CLAMPED"


def build_cinema_camera(col, total_frames: int = C.TOTAL_FRAMES) -> "bpy.types.Object":
    """
    Create and animate `Cam_Main` and `Cam_Target` (`TRACK_TO` -Z / UP_Y) across `1..total_frames`.
    Enables DOF with explicit per-frame `focus_distance` and `aperture_fstop` keyframes,
    sets `CONSTANT` interpolation across hard cuts, assigns `scene.camera`, and returns `Cam_Main`.
    """
    if bpy is None:
        raise RuntimeError("build_cinema_camera() must run inside Blender (bpy unavailable).")

    for name in ("Cam_Main", "Cam_Target"):
        old_obj = bpy.data.objects.get(name)
        if old_obj is not None:
            bpy.data.objects.remove(old_obj, do_unlink=True)

    cam_data = bpy.data.cameras.new("Cam_Main")
    cam_data.sensor_fit = "AUTO"
    cam_data.sensor_width = SENSOR_WIDTH_MM
    cam_data.clip_start = 0.02
    cam_data.clip_end = 1000.0
    cam_data.lens = 22.0

    cam_data.dof.use_dof = True
    cam_data.dof.focus_object = None
    cam_data.dof.aperture_blades = 7
    cam_data.dof.aperture_ratio = 1.0
    cam_data.dof.aperture_fstop = 4.8
    cam_data.dof.focus_distance = 2.40

    target = bpy.data.objects.new("Cam_Target", None)
    target.empty_display_type = "SPHERE"
    target.empty_display_size = 0.05
    _link_to_collection(target, col)

    cam_obj = bpy.data.objects.new("Cam_Main", cam_data)
    _link_to_collection(cam_obj, col)
    cam_obj["target_name"] = target.name
    bpy.context.scene.camera = cam_obj

    track = cam_obj.constraints.new(type="TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    for f in range(1, int(total_frames) + 1):
        st = camera_state_at(float(f), include_motion_fx=True)
        cam_obj.location = Vector(st["cam_pos"])
        cam_obj.keyframe_insert(data_path="location", frame=f)

        target.location = Vector(st["tgt_pos"])
        target.keyframe_insert(data_path="location", frame=f)

        cam_data.lens = st["lens_mm"]
        cam_data.keyframe_insert(data_path="lens", frame=f)

        cam_data.dof.focus_distance = st["focus_dist"]
        cam_data.dof.keyframe_insert(data_path="focus_distance", frame=f)

        cam_data.dof.aperture_fstop = st["fstop"]
        cam_data.dof.keyframe_insert(data_path="aperture_fstop", frame=f)

    _set_handles_with_hard_cuts(cam_obj)
    _set_handles_with_hard_cuts(target)
    _set_handles_with_hard_cuts(cam_data)

    st0 = camera_state_at(1.0, include_motion_fx=True)
    cam_obj.location = Vector(st0["cam_pos"])
    target.location = Vector(st0["tgt_pos"])
    cam_data.lens = st0["lens_mm"]
    cam_data.dof.focus_distance = st0["focus_dist"]
    cam_data.dof.aperture_fstop = st0["fstop"]

    return cam_obj


if __name__ == "__main__":
    if bpy is not None:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        test_col = bpy.data.collections.new("Cam_SelfTest")
        bpy.context.scene.collection.children.link(test_col)
        cam = build_cinema_camera(test_col, total_frames=C.TOTAL_FRAMES)
        assert cam.name == "Cam_Main"
        assert bpy.context.scene.camera is cam
        print(f"Cam_Main self-test PASSED in Blender ({len(C.iter_action_fcurves(cam.animation_data.action))} location fcurves).")
    else:
        for f_chk in (1, 180, 313, 450, 601, 750, 901, 985, 1237, 1300, 1386, 1545, 1750):
            st = camera_state_at(float(f_chk), include_motion_fx=True)
            print(
                f"f={f_chk:04d} cam={tuple(round(v, 2) for v in st['cam_pos'])} "
                f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f} focus={st['focus_dist']:.2f}m"
            )
