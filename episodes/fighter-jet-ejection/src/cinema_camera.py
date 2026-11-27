"""
Cinema Camera Engine for Episode #1:
"Why Ejecting From a Fighter Jet Shrinks You by 1 Inch" (60 fps, 1800 frames, 9:16 Portrait).

Pure-Python safe (`bpy` / `mathutils` imported conditionally) so `verify_framing.py` and
`labels_3d.py` can import `camera_state_at` and `project_ndc` directly outside Blender
with 0.000 mm trajectory, focus-plane, or lens drift.

Public API:
  - project_ndc(pt, cam_pos, tgt_pos, lens_mm: float) -> tuple[float, float, float]
  - camera_state_at(f: float, include_motion_fx: bool = True) -> dict
      returns {"cam_pos": (x, y, z), "tgt_pos": (x, y, z),
               "lens_mm": float, "fstop": float, "focus_dist": float}
  - build_cinema_camera(col: bpy.types.Collection, total_frames: int = 1800) -> bpy.types.Object
      creates `Cam_Main` + `Cam_Target` (`TRACK_TO` -Z / UP_Y), keys transform, lens,
      DOF `focus_distance` and `aperture_fstop` across `1..total_frames`, sets `scene.camera`.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Sequence

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT_SRC = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", "src"))
for _p in (_SCRIPT_DIR, _REPO_ROOT_SRC):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import ejection_contract as C  # noqa: E402

try:
    import bpy  # type: ignore[import-not-found]
    from mathutils import Euler, Matrix, Vector  # type: ignore[import-not-found]
except ImportError:  # Pure-Python mode (verify_framing.py, labels_3d static checks)
    bpy = None  # type: ignore[assignment]
    Euler = None  # type: ignore[assignment]
    Matrix = None  # type: ignore[assignment]
    Vector = None  # type: ignore[assignment]

Vec3 = C.Vec3

# ==============================================================================
# 9:16 PORTRAIT OPTICAL CONSTANTS (Blender sensor_fit='AUTO', 1080x1920)
# ==============================================================================
RES_X = 1080
RES_Y = 1920
ASPECT_W_OVER_H = RES_X / RES_Y        # 0.5625
SENSOR_WIDTH_MM = 36.0
SENSOR_H_EFF_MM = SENSOR_WIDTH_MM      # 36.0 mm (portrait height is major axis)
SENSOR_W_EFF_MM = SENSOR_WIDTH_MM * ASPECT_W_OVER_H  # 20.25 mm


# ==============================================================================
# PURE-PYTHON VECTOR & INTERPOLATION HELPERS
# ==============================================================================
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
    pt: Sequence[float],
    cam_pos: Sequence[float],
    tgt_pos: Sequence[float],
    lens_mm: float,
) -> tuple[float, float, float]:
    """Project a 3D world point into 9:16 portrait NDC [-1, +1] matching Blender sensor_fit='AUTO'."""
    fwd = _norm3(_sub3(tgt_pos, cam_pos))
    right = _norm3(_cross3(fwd, (0.0, 0.0, 1.0)))
    up = _cross3(right, fwd)
    rel = _sub3(pt, cam_pos)
    z = _dot3(rel, fwd)
    x = _dot3(rel, right)
    y = _dot3(rel, up)
    if z <= 0.01:
        return (99.0, 99.0, z)
    ndc_x = (x / z) * (lens_mm / (0.5 * SENSOR_W_EFF_MM))
    ndc_y = (y / z) * (lens_mm / (0.5 * SENSOR_H_EFF_MM))
    return (ndc_x, ndc_y, z)


# ==============================================================================
# HANDHELD, YANK & PROPULSION MOTION LAYERS
# ==============================================================================
def _beat1_handheld_and_yank(f: float) -> tuple[Vec3, Vec3, float]:
    """Subtle Beat 1 cockpit tremor (f=1..288) + visceral handle-yank micro-kick (f=150..172)."""
    cam_d = [0.0, 0.0, 0.0]
    tgt_d = [0.0, 0.0, 0.0]
    lens_d = 0.0

    # Panic handheld tremor fading out before Beat 2 bullet-time
    w_hh = 1.0 - C.smoothstep(270.0, 290.0, f)
    if w_hh > 1e-5:
        cam_d[0] += w_hh * (0.0040 * math.sin(f * 0.082 + 0.3) + 0.0020 * math.sin(f * 0.21 + 1.4))
        cam_d[1] += w_hh * (0.0030 * math.cos(f * 0.075 + 0.8))
        cam_d[2] += w_hh * (0.0042 * math.sin(f * 0.115 + 2.1) + 0.0018 * math.cos(f * 0.25))
        tgt_d[0] += w_hh * (0.0020 * math.sin(f * 0.095 + 1.0))
        tgt_d[2] += w_hh * (0.0022 * math.cos(f * 0.125 + 2.4))

    # Crisp yank micro-kick across EV['yank_start']=150..174 (peak f=153..160)
    y0 = float(C.EV["yank_start"])
    if y0 <= f <= 174.0:
        rise = C.smoothstep(y0, y0 + 4.0, f)
        fade = 1.0 - C.smoothstep(166.0, 174.0, f)
        dt = max(0.0, f - y0)
        env = rise * fade * math.exp(-0.08 * dt)
        osc = math.sin(dt * 1.15 + 0.4)
        cam_d[0] += 0.011 * env * math.cos(dt * 0.9)
        cam_d[1] += -0.014 * env * osc
        cam_d[2] += 0.016 * env * (0.7 + 0.3 * osc)
        tgt_d[0] += 0.004 * env * math.cos(dt * 1.1)
        tgt_d[2] += 0.010 * env * osc
        lens_d += 1.4 * env

    return (cam_d[0], cam_d[1], cam_d[2]), (tgt_d[0], tgt_d[1], tgt_d[2]), lens_d


def _impulse_motion_fx(f: float) -> tuple[Vec3, Vec3]:
    """Damped physical impulses for MDC detonation, catapult launch, rocket rumble, and chute snatch."""
    cx = cy = cz = 0.0
    tx = ty = tz = 0.0

    # Beat 2: MDC detonation micro-pulse (f=300..318)
    f_mdc = float(C.EV["mdc_detonate"])
    if f_mdc <= f <= f_mdc + 18.0:
        dt = f - f_mdc
        env = math.exp(-0.24 * dt) * (1.0 - C.smoothstep(f_mdc + 12.0, f_mdc + 18.0, f))
        cx += 0.004 * env * math.sin(dt * 1.5)
        cz += 0.005 * env * math.cos(dt * 1.3)

    # Beat 3: Catapult cartridge launch kick (f=700..726)
    f_cat = float(C.EV["catapult_ignite"])
    if f_cat <= f <= f_cat + 26.0:
        dt = f - f_cat
        env = C.smoothstep(f_cat, f_cat + 2.0, f) * math.exp(-0.16 * dt) * (1.0 - C.smoothstep(f_cat + 20.0, f_cat + 26.0, f))
        cz += -0.012 * env * math.sin(dt * 1.1)
        cy += 0.006 * env * math.cos(dt * 0.9)
        tz += -0.005 * env * math.sin(dt * 1.1)

    # Beat 5: Under-seat rocket burn thrust rumble (f=1250..1465)
    f_ri = float(C.EV["rocket_ignite"])
    f_rb = float(C.EV["rocket_burnout"])
    if f_ri <= f <= f_rb + 10.0:
        w = C.smoothstep(f_ri, f_ri + 12.0, f) * (1.0 - C.smoothstep(f_rb - 10.0, f_rb + 10.0, f))
        cx += 0.016 * w * (0.6 * math.sin(f * 1.47) + 0.4 * math.cos(f * 2.63))
        cy += 0.012 * w * math.sin(f * 1.91 + 0.8)
        cz += 0.018 * w * (0.6 * math.cos(f * 1.73 + 0.4) + 0.4 * math.sin(f * 2.97))
        tx += 0.006 * w * math.sin(f * 1.47)
        tz += 0.006 * w * math.cos(f * 1.73 + 0.4)

    # Beat 6: Separation kick (f=1545..1568) & main canopy blossom snatch (f=1646..1674)
    for ev_frame, amp in ((float(C.EV["separation"]), 0.025), (float(C.EV["main_full"]), 0.032)):
        if ev_frame <= f <= ev_frame + 24.0:
            dt = f - ev_frame
            env = C.smoothstep(ev_frame, ev_frame + 3.0, f) * math.exp(-0.14 * dt) * (1.0 - C.smoothstep(ev_frame + 18.0, ev_frame + 24.0, f))
            cz += -amp * env * math.sin(dt * 0.95)
            tz += -0.35 * amp * env * math.sin(dt * 0.95)

    return (cx, cy, cz), (tx, ty, tz)


# ==============================================================================
# 6-BEAT BASE CAMERA CHOREOGRAPHY
# ==============================================================================
_B1_CAM_KFS: list[tuple[float, Vec3]] = [
    (1.0,   (0.82, 1.26, 0.93)),
    (60.0,  (0.82, 1.26, 0.93)),
    (130.0, (0.82, 1.26, 0.93)),
    (152.0, (0.82, 1.26, 0.93)),
    (168.0, (0.76, 1.18, 0.92)),
    (220.0, (0.79, 1.21, 0.94)),
    (276.0, (0.82, 1.24, 0.95)),
    (288.0, (0.98, 1.34, 1.04)),
]
_B1_TGT_KFS: list[tuple[float, Vec3]] = [
    (1.0,   (-0.01, 0.19, 0.33)),
    (60.0,  (-0.01, 0.19, 0.33)),
    (130.0, (-0.01, 0.19, 0.33)),
    (152.0, (-0.01, 0.19, 0.33)),
    (168.0, (-0.01, 0.17, 0.44)),
    (220.0, (-0.01, 0.15, 0.46)),
    (276.0, (-0.01, 0.15, 0.47)),
    (288.0, (0.00,  0.15, 0.60)),
]
_B1_LENS_KFS: list[tuple[float, float]] = [
    (1.0,   28.5),
    (60.0,  28.5),
    (130.0, 28.5),
    (152.0, 28.5),
    (168.0, 32.0),
    (276.0, 31.2),
    (288.0, 30.0),
]

_B2_CAM_KFS: list[tuple[float, Vec3]] = [
    (288.0, (0.98, 1.34, 1.04)),
    (305.0, (1.28, 1.44, 1.16)),
    (360.0, (1.38, 1.40, 1.21)),
    (430.0, (1.50, 1.30, 1.25)),
    (500.0, (1.60, 1.18, 1.28)),
    (545.0, (1.65, 1.11, 1.28)),
    (588.0, (1.70, 1.06, 0.54)),
]
_B2_TGT_KFS: list[tuple[float, Vec3]] = [
    (288.0, (0.00, 0.15, 0.60)),
    (305.0, (0.00, 0.15, 0.90)),
    (430.0, (0.00, 0.15, 0.92)),
    (545.0, (0.00, 0.15, 0.90)),
    (588.0, (0.00, 0.28, 0.18)),
]
_B2_LENS_KFS: list[tuple[float, float]] = [
    (288.0, 30.0),
    (305.0, 29.0),
    (430.0, 29.0),
    (545.0, 30.0),
    (588.0, 31.0),
]

_B3_CAM_KFS: list[tuple[float, Vec3]] = [
    (588.0, (1.70, 1.06, 0.54)),
    (615.0, (1.72, 1.06, 0.45)),
    (650.0, (1.70, 1.04, 0.42)),
    (700.0, (1.70, 1.02, 0.40)),
    (760.0, (1.70, 1.00, 0.42)),
    (830.0, (1.71, 0.98, 0.46)),
    (888.0, (1.72, 0.96, 0.50)),
]
_B3_TGT_KFS: list[tuple[float, Vec3]] = [
    (588.0, (0.00, 0.28, 0.18)),
    (615.0, (0.00, 0.30, 0.14)),
    (650.0, (0.00, 0.22, 0.14)),
    (700.0, (0.00, 0.16, 0.15)),
    (760.0, (0.00, 0.12, 0.18)),
    (830.0, (0.00, 0.06, 0.24)),
    (888.0, (0.00, 0.02, 0.31)),
]
_B3_LENS_KFS: list[tuple[float, float]] = [
    (588.0, 31.0),
    (615.0, 31.0),
    (760.0, 32.0),
    (888.0, 32.0),
]

# Seat-relative offsets at exact Beat 3 exit (f=888) for zero-seam transition into Beat 4
_SEAT_888 = C.seat_world_location(888.0)
_B4_START_CAM_OFF: Vec3 = _sub3(_B3_CAM_KFS[-1][1], _SEAT_888)
_B4_START_TGT_OFF: Vec3 = _sub3(_B3_TGT_KFS[-1][1], _SEAT_888)

_B4_CAM_OFF_KFS: list[tuple[float, Vec3]] = [
    (888.0,  _B4_START_CAM_OFF),
    (930.0,  (1.12, 0.52, 0.45)),
    (1020.0, (1.11, 0.50, 0.45)),
    (1080.0, (1.10, 0.49, 0.45)),
    (1180.0, (1.11, 0.51, 0.45)),
    (1236.0, (1.12, 0.52, 0.45)),
]
_B4_TGT_OFF_KFS: list[tuple[float, Vec3]] = [
    (888.0,  _B4_START_TGT_OFF),
    (930.0,  (0.00, -0.06, 0.37)),
    (1080.0, (0.00, -0.06, 0.37)),
    (1236.0, (0.00, -0.06, 0.37)),
]
_B4_LENS_KFS: list[tuple[float, float]] = [
    (888.0,  32.0),
    (930.0,  42.0),
    (1080.0, 42.5),
    (1236.0, 42.0),
]

_B5_CAM_OFF_KFS: list[tuple[float, Vec3]] = [
    (1236.0, (1.12, 0.52, 0.45)),
    (1265.0, (3.20, 1.95, 0.80)),
    (1300.0, (4.35, 2.70, 0.85)),
    (1365.0, (5.35, 3.35, 0.35)),
    (1420.0, (4.90, 3.05, 1.45)),
    (1470.0, (5.15, 3.25, 1.95)),
    (1530.0, (5.20, 3.28, 2.00)),
]
_B5_TGT_OFF_KFS: list[tuple[float, Vec3]] = [
    (1236.0, (0.00, -0.06, 0.37)),
    (1265.0, (0.00,  0.00, 0.05)),
    (1300.0, (0.00,  0.10, -0.55)),
    (1365.0, (-0.05, 0.45, -3.10)),
    (1420.0, (0.10, -0.55, 1.35)),
    (1470.0, (0.12, -0.90, 1.65)),
    (1530.0, (0.12, -0.90, 1.65)),
]
_B5_LENS_KFS: list[tuple[float, float]] = [
    (1236.0, 42.0),
    (1265.0, 27.0),
    (1300.0, 25.0),
    (1365.0, 22.0),
    (1420.0, 24.5),
    (1470.0, 23.5),
    (1530.0, 23.5),
]

# Beat 6 polar sweep relative to pilot_carrier (radius, azimuth_deg from +Y toward +X, z_off)
_B6_R0 = math.hypot(_B5_CAM_OFF_KFS[-1][1][0], _B5_CAM_OFF_KFS[-1][1][1])
_B6_AZ0 = math.degrees(math.atan2(_B5_CAM_OFF_KFS[-1][1][0], _B5_CAM_OFF_KFS[-1][1][1]))
_B6_POLAR_KFS: list[tuple[float, Vec3]] = [
    (1530.0, (_B6_R0, _B6_AZ0, _B5_CAM_OFF_KFS[-1][1][2])),
    (1555.0, (8.60, 56.0, 2.65)),
    (1580.0, (11.90, 54.0, 3.20)),
    (1650.0, (12.68, 48.0, 3.15)),
    (1700.0, (12.72, 36.0, 3.05)),
    (1800.0, (12.75, 16.0, 2.95)),
]
_B6_TGT_OFF_KFS: list[tuple[float, Vec3]] = [
    (1530.0, _B5_TGT_OFF_KFS[-1][1]),
    (1555.0, (0.05, -0.40, 2.55)),
    (1580.0, (0.00, 0.00, 3.45)),
    (1650.0, (0.00, 0.00, 3.45)),
    (1800.0, (0.00, 0.00, 3.45)),
]
_B6_LENS_KFS: list[tuple[float, float]] = [
    (1530.0, 23.5),
    (1555.0, 25.5),
    (1580.0, 26.8),
    (1650.0, 28.0),
    (1800.0, 28.0),
]


def _focus_point_at(f: float, tgt_pos: Vec3) -> Vec3:
    """Continuous world-space optical focus plane target for sharp DOF on hero anatomy & 3D labels."""
    if f <= 288.0:
        return C.lerp3((0.0, 0.14, 0.45), (0.0, 0.15, 0.60), C.smootherstep(276.0, 288.0, f))
    if f <= 588.0:
        p = C.lerp3((0.0, 0.15, 0.60), (0.0, 0.15, 1.02), C.smootherstep(288.0, 305.0, f))
        return C.lerp3(p, (0.0, 0.28, 0.18), C.smootherstep(545.0, 588.0, f))
    if f <= 888.0:
        return tgt_pos
    if f <= 1236.0:
        seat = C.seat_world_location(f)
        spine_mid = C.add3(seat, (0.0, -0.085, 0.41))
        return C.lerp3(tgt_pos, spine_mid, C.smootherstep(888.0, 930.0, f))
    if f <= 1530.0:
        seat = C.seat_world_location(f)
        spine_mid = C.add3(seat, (0.0, -0.085, 0.41))
        b5_focus = C.add3(seat, (0.0, -0.20, 0.45))
        return C.lerp3(spine_mid, b5_focus, C.smootherstep(1236.0, 1295.0, f))
    pc = C.pilot_carrier_location(f)
    b5_focus = C.add3(pc, (0.0, -0.20, 0.45))
    return C.lerp3(b5_focus, tgt_pos, C.smootherstep(1530.0, 1580.0, f))


def camera_state_at(f: float, include_motion_fx: bool = True) -> dict:
    """
    Evaluate exact 9:16 portrait camera state at frame `f` in [1, 1800].
    Returns {"cam_pos": (x,y,z), "tgt_pos": (x,y,z), "lens_mm": float, "fstop": float, "focus_dist": float}.
    """
    f = max(1.0, min(float(C.TOTAL_FRAMES), float(f)))

    if f <= 288.0:
        cam_pos = _eval_piecewise_vec3(_B1_CAM_KFS, f)
        tgt_pos = _eval_piecewise_vec3(_B1_TGT_KFS, f)
        lens_mm = _eval_piecewise_scalar(_B1_LENS_KFS, f)
        fstop = _eval_piecewise_scalar([(1.0, 4.5), (168.0, 4.2), (255.0, 4.5), (288.0, 5.0)], f)
    elif f <= 588.0:
        cam_pos = _eval_piecewise_vec3(_B2_CAM_KFS, f)
        tgt_pos = _eval_piecewise_vec3(_B2_TGT_KFS, f)
        lens_mm = _eval_piecewise_scalar(_B2_LENS_KFS, f)
        fstop = _eval_piecewise_scalar([(288.0, 5.0), (305.0, 5.6), (545.0, 5.6), (588.0, 6.3)], f)
    elif f <= 888.0:
        cam_pos = _eval_piecewise_vec3(_B3_CAM_KFS, f)
        tgt_pos = _eval_piecewise_vec3(_B3_TGT_KFS, f)
        lens_mm = _eval_piecewise_scalar(_B3_LENS_KFS, f)
        fstop = 6.3
    elif f <= 1236.0:
        seat = C.seat_world_location(f)
        cam_pos = C.add3(seat, _eval_piecewise_vec3(_B4_CAM_OFF_KFS, f))
        tgt_pos = C.add3(seat, _eval_piecewise_vec3(_B4_TGT_OFF_KFS, f))
        lens_mm = _eval_piecewise_scalar(_B4_LENS_KFS, f)
        fstop = _eval_piecewise_scalar([(888.0, 6.3), (930.0, 5.6), (1236.0, 5.6)], f)
    elif f <= 1530.0:
        seat = C.seat_world_location(f)
        cam_pos = C.add3(seat, _eval_piecewise_vec3(_B5_CAM_OFF_KFS, f))
        tgt_pos = C.add3(seat, _eval_piecewise_vec3(_B5_TGT_OFF_KFS, f))
        lens_mm = _eval_piecewise_scalar(_B5_LENS_KFS, f)
        fstop = _eval_piecewise_scalar([(1236.0, 5.6), (1295.0, 7.1), (1530.0, 7.1)], f)
    else:
        pc = C.pilot_carrier_location(f)
        r_horiz, az_deg, z_off = _eval_piecewise_vec3(_B6_POLAR_KFS, f)
        az = math.radians(az_deg)
        cam_off = (r_horiz * math.sin(az), r_horiz * math.cos(az), z_off)
        cam_pos = C.add3(pc, cam_off)
        tgt_pos = C.add3(pc, _eval_piecewise_vec3(_B6_TGT_OFF_KFS, f))
        lens_mm = _eval_piecewise_scalar(_B6_LENS_KFS, f)
        fstop = _eval_piecewise_scalar([(1530.0, 7.1), (1650.0, 8.0), (1800.0, 8.0)], f)

    focus_pt = _focus_point_at(f, tgt_pos)

    if include_motion_fx:
        b1_cam, b1_tgt, b1_lens = _beat1_handheld_and_yank(f)
        imp_cam, imp_tgt = _impulse_motion_fx(f)
        cam_pos = C.add3(cam_pos, C.add3(b1_cam, imp_cam))
        tgt_pos = C.add3(tgt_pos, C.add3(b1_tgt, imp_tgt))
        lens_mm += b1_lens

    fwd = _norm3(_sub3(tgt_pos, cam_pos))
    focus_dist = max(0.15, _dot3(_sub3(focus_pt, cam_pos), fwd))

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


def _set_smooth_handles(id_block) -> None:
    anim = getattr(id_block, "animation_data", None)
    if anim is None or anim.action is None:
        return
    for fcurve in anim.action.fcurves:
        for kp in fcurve.keyframe_points:
            kp.interpolation = "BEZIER"
            kp.handle_left_type = "AUTO_CLAMPED"
            kp.handle_right_type = "AUTO_CLAMPED"


def build_cinema_camera(col, total_frames: int = C.TOTAL_FRAMES) -> "bpy.types.Object":
    """
    Create and animate `Cam_Main` and `Cam_Target` (`TRACK_TO` -Z / UP_Y) across `1..total_frames`.
    Enables DOF with explicit per-frame `focus_distance` and `aperture_fstop` keyframes,
    assigns `scene.camera = cam_obj`, and returns `Cam_Main`.
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
    cam_data.lens = 30.0

    cam_data.dof.use_dof = True
    cam_data.dof.focus_object = None
    cam_data.dof.aperture_blades = 7
    cam_data.dof.aperture_ratio = 1.0
    cam_data.dof.aperture_fstop = 4.5
    cam_data.dof.focus_distance = 1.40

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

    _set_smooth_handles(cam_obj)
    _set_smooth_handles(target)
    _set_smooth_handles(cam_data)

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
        print(f"Cam_Main self-test PASSED in Blender ({len(cam.animation_data.action.fcurves)} location fcurves).")
    else:
        for f_chk in C.PREVIEW_FRAMES:
            st = camera_state_at(float(f_chk), include_motion_fx=True)
            print(
                f"f={f_chk:04d} cam={tuple(round(v, 2) for v in st['cam_pos'])} "
                f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f} focus={st['focus_dist']:.2f}m"
            )
