"""
cinema_camera.py — 6-Beat 9:16 Portrait Cinema Camera Engine & Exact DOF Focus Tracking
for Episode #5: "Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock"
(60 fps, 1800 frames, 30.0 s, 1080x1920 Portrait).

Pure-Python safe (`bpy` / `mathutils` imported conditionally) so `verify_framing.py` and
`labels_3d.py` can import `camera_state_at` and `project_ndc` directly outside Blender
with 0.000 mm trajectory, focus-plane, or lens drift.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Sequence

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import heli_contract as C  # noqa: E402

try:
    import bpy  # type: ignore[import-not-found]
    from mathutils import Vector  # type: ignore[import-not-found]
except ImportError:
    bpy = None  # type: ignore[assignment]
    Vector = None  # type: ignore[assignment]

Vec3 = tuple[float, float, float]

# ==============================================================================
# 9:16 PORTRAIT OPTICAL CONSTANTS (Blender sensor_fit='AUTO', 1080x1920)
# ==============================================================================
RES_X = 1080
RES_Y = 1920
ASPECT_W_OVER_H = RES_X / RES_Y                      # 0.5625
SENSOR_WIDTH_MM = 36.0
SENSOR_H_EFF_MM = SENSOR_WIDTH_MM                    # 36.0 mm (portrait height is major axis)
SENSOR_W_EFF_MM = SENSOR_WIDTH_MM * ASPECT_W_OVER_H  # 20.25 mm

HARD_CUT_FRAMES: tuple[int, ...] = C.HARD_CUT_FRAMES  # (241, 571, 961, 1291, 1451, 1581)
_HARD_CUT_PREV_FRAMES: set[int] = {cut - 1 for cut in HARD_CUT_FRAMES}


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


def _lerp3(a: Sequence[float], b: Sequence[float], t: float) -> Vec3:
    return (
        C.lerp(float(a[0]), float(b[0]), t),
        C.lerp(float(a[1]), float(b[1]), t),
        C.lerp(float(a[2]), float(b[2]), t),
    )


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
            return _lerp3(v0, v1, u)
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
# SUBTLE AERODYNAMIC & MECHANICAL MOTION FX (BOUNDED CLEANLY INSIDE BEATS)
# ==============================================================================
def _motion_fx(f: float) -> tuple[Vec3, Vec3, float]:
    cx = cy = cz = 0.0
    tx = ty = tz = 0.0
    lens_d = 0.0

    # Beat 1 (f=14..65): Engine flame-out cockpit shudder impulse
    f_eng = float(C.EV["b1_engine_flameout"])
    if f_eng <= f <= f_eng + 50.0:
        dt = f - f_eng
        env = C.smoothstep(f_eng, f_eng + 3.0, f) * math.exp(-0.08 * dt) * (1.0 - C.smoothstep(f_eng + 36.0, f_eng + 50.0, f))
        cx += 0.010 * env * math.sin(dt * 1.9)
        cz += 0.008 * env * math.cos(dt * 2.3)
        lens_d += 0.35 * env

    # Beat 2 (f=340..375): Seized turbine shaft clunk micro-jolt
    f_sz = float(C.EV["b2_shaft_seize"])
    if f_sz <= f <= f_sz + 35.0:
        dt = f - f_sz
        env = C.smoothstep(f_sz, f_sz + 2.0, f) * math.exp(-0.12 * dt) * (1.0 - C.smoothstep(f_sz + 24.0, f_sz + 35.0, f))
        cx += -0.006 * env * math.sin(dt * 2.1)
        cz += 0.005 * env * math.cos(dt * 1.8)

    # Beat 5B (f=1465..1520): 30-foot flare air-bite cushion punch
    f_fl = float(C.EV["b5_flare_bite"])
    if f_fl <= f <= f_fl + 55.0:
        dt = f - f_fl
        env = C.smoothstep(f_fl, f_fl + 4.0, f) * math.exp(-0.07 * dt) * (1.0 - C.smoothstep(f_fl + 40.0, f_fl + 55.0, f))
        cz += 0.010 * env * math.sin(dt * 1.2)
        lens_d += 0.38 * env

    # Beat 6 (f=1685..1718): Gentle skid-on-helipad touchdown settle
    f_td = float(C.EV["b6_skid_touchdown"])
    if f_td <= f <= f_td + 33.0:
        dt = f - f_td
        env = C.smoothstep(f_td, f_td + 2.0, f) * math.exp(-0.14 * dt) * (1.0 - C.smoothstep(f_td + 22.0, f_td + 33.0, f))
        cz += -0.009 * env * math.sin(dt * 1.4)

    return (cx, cy, cz), (tx, ty, tz), lens_d


# ==============================================================================
# 6-BEAT CAMERA CHOREOGRAPHY (9:16 PORTRAIT, ZERO GIMBAL FLIP, CONTINUOUS MOTION)
# ==============================================================================
# Beat 1 (1..240): 1:1 Rescue Cockpit — Pilot (+0.36), Co-Pilot (-0.36), ENG OUT & Collective Slam!
_B1_CAM_KFS: list[tuple[float, Vec3]] = [
    (1.0, (-0.12, -1.62, 1.38)),
    (90.0, (-0.04, -1.56, 1.37)),
    (155.0, (0.04, -1.50, 1.35)),
    (240.0, (0.08, -1.46, 1.34)),
]
_B1_TGT_KFS: list[tuple[float, Vec3]] = [
    (1.0, (0.00, 0.14, 1.22)),
    (120.0, (0.03, 0.12, 1.18)),
    (240.0, (0.05, 0.10, 1.16)),
]
_B1_LENS_KFS: list[tuple[float, float]] = [
    (1.0, 22.5),
    (125.0, 23.4),
    (240.0, 24.2),
]

# Beat 2 (241..570, hard cut @ 241): Dorsal Macro X-Ray on White/Red Cowling, Seized Shaft & Sprag Clutch
_B2_CAM_KFS: list[tuple[float, Vec3]] = [
    (241.0, (-0.44, -1.72, 2.38)),
    (360.0, (-0.28, -1.58, 2.35)),
    (460.0, (-0.16, -1.48, 2.32)),
    (570.0, (-0.08, -1.42, 2.30)),
]
_B2_TGT_KFS: list[tuple[float, Vec3]] = [
    (241.0, (0.00, 0.04, 2.12)),
    (410.0, (0.00, 0.03, 2.13)),
    (570.0, (0.00, 0.02, 2.14)),
]
_B2_LENS_KFS: list[tuple[float, float]] = [
    (241.0, 29.5),
    (410.0, 31.8),
    (570.0, 33.5),
]

# Beat 3 (571..960, hard cut @ 571): Aerial Tracking Shot of Autorotation Glide & Upward Emerald Streamlines
_B3_CAM_KFS: list[tuple[float, Vec3]] = [
    (571.0, (-0.64, -3.98, 2.22)),
    (710.0, (-0.24, -3.86, 2.10)),
    (840.0, (0.14, -3.76, 1.98)),
    (960.0, (0.44, -3.68, 1.88)),
]
_B3_TGT_KFS: list[tuple[float, Vec3]] = [
    (571.0, (-0.26, 0.28, 2.34)),
    (765.0, (0.00, 0.20, 2.16)),
    (960.0, (0.22, 0.12, 1.98)),
]
_B3_LENS_KFS: list[tuple[float, float]] = [
    (571.0, 24.0),
    (765.0, 25.0),
    (960.0, 26.2),
]

# Beat 4 (961..1290, hard cut @ 961): Side-by-Side 3D Maple Seed (Left) vs Rescue Rotor (Right)
_B4_CAM_KFS: list[tuple[float, Vec3]] = [
    (961.0, (-0.12, -3.56, 2.20)),
    (1125.0, (-0.02, -3.44, 2.00)),
    (1290.0, (0.06, -3.34, 1.82)),
]
_B4_TGT_KFS: list[tuple[float, Vec3]] = [
    (961.0, (0.00, 0.14, 2.20)),
    (1125.0, (0.00, 0.14, 1.98)),
    (1290.0, (0.00, 0.14, 1.76)),
]
_B4_LENS_KFS: list[tuple[float, float]] = [
    (961.0, 22.5),
    (1125.0, 23.2),
    (1290.0, 23.8),
]

# Beat 5A (1291..1450, hard cut @ 1291): 1:1 Cockpit Final Flare Close-Up (Label-Free f/3.6 Bokeh!)
_B5A_CAM_KFS: list[tuple[float, Vec3]] = [
    (1291.0, (0.18, -1.50, 1.37)),
    (1370.0, (0.12, -1.44, 1.38)),
    (1450.0, (0.05, -1.40, 1.39)),
]
_B5A_TGT_KFS: list[tuple[float, Vec3]] = [
    (1291.0, (0.10, 0.18, 1.22)),
    (1370.0, (0.08, 0.18, 1.21)),
    (1450.0, (0.04, 0.18, 1.20)),
]
_B5A_LENS_KFS: list[tuple[float, float]] = [
    (1291.0, 23.8),
    (1370.0, 24.4),
    (1450.0, 25.0),
]

# Beat 5B (1451..1580, hard cut @ 1451): Crisp f/8.0 Exterior 30-Foot Nose-Up Flare beside Rooftop Helipad
_B5B_CAM_KFS: list[tuple[float, Vec3]] = [
    (1451.0, (-0.42, -4.35, 1.62)),
    (1515.0, (-0.24, -4.18, 1.56)),
    (1580.0, (-0.08, -4.02, 1.50)),
]
_B5B_TGT_KFS: list[tuple[float, Vec3]] = [
    (1451.0, (-0.22, 0.46, 1.72)),
    (1515.0, (-0.08, 0.45, 1.58)),
    (1580.0, (0.04, 0.45, 1.46)),
]
_B5B_LENS_KFS: list[tuple[float, float]] = [
    (1451.0, 23.8),
    (1515.0, 24.6),
    (1580.0, 25.4),
]

# Beat 6 (1581..1800, hard cut @ 1581): Golden-Hour Rooftop Helipad Touchdown Hero Arc
_B6_CAM_KFS: list[tuple[float, Vec3]] = [
    (1581.0, (0.54, -4.05, 1.88)),
    (1685.0, (0.12, -3.86, 1.72)),
    (1800.0, (-0.34, -3.72, 1.62)),
]
_B6_TGT_KFS: list[tuple[float, Vec3]] = [
    (1581.0, (0.06, 0.45, 1.36)),
    (1685.0, (0.08, 0.45, 1.20)),
    (1800.0, (0.08, 0.45, 1.18)),
]
_B6_LENS_KFS: list[tuple[float, float]] = [
    (1581.0, 24.5),
    (1690.0, 25.4),
    (1800.0, 26.2),
]


def _fstop_at(f: float) -> float:
    """f/7.5..f/8.0 on all labeled beats; shallow f/3.6 bokeh exclusively on label-free Beat 5A!"""
    if f <= 240.0:
        return 7.5
    if f <= 570.0:
        return 7.8
    if f <= 960.0:
        return 7.8
    if f <= 1290.0:
        return 7.5
    if f <= float(C.B5_COCKPIT_END):
        return 3.6
    if f <= 1580.0:
        return 8.0
    return 7.5


def camera_state_at(f: float, include_motion_fx: bool = True) -> dict:
    """Evaluate exact world-space camera state at frame `f` (1..1800)."""
    f = C.clamp(float(f), 1.0, float(C.TOTAL_FRAMES))
    if f <= 240.0:
        cp = _eval_piecewise_vec3(_B1_CAM_KFS, f)
        tp = _eval_piecewise_vec3(_B1_TGT_KFS, f)
        ln = _eval_piecewise_scalar(_B1_LENS_KFS, f)
    elif f <= 570.0:
        cp = _eval_piecewise_vec3(_B2_CAM_KFS, f)
        tp = _eval_piecewise_vec3(_B2_TGT_KFS, f)
        ln = _eval_piecewise_scalar(_B2_LENS_KFS, f)
    elif f <= 960.0:
        cp = _eval_piecewise_vec3(_B3_CAM_KFS, f)
        tp = _eval_piecewise_vec3(_B3_TGT_KFS, f)
        ln = _eval_piecewise_scalar(_B3_LENS_KFS, f)
    elif f <= 1290.0:
        cp = _eval_piecewise_vec3(_B4_CAM_KFS, f)
        tp = _eval_piecewise_vec3(_B4_TGT_KFS, f)
        ln = _eval_piecewise_scalar(_B4_LENS_KFS, f)
    elif f <= float(C.B5_COCKPIT_END):
        cp = _eval_piecewise_vec3(_B5A_CAM_KFS, f)
        tp = _eval_piecewise_vec3(_B5A_TGT_KFS, f)
        ln = _eval_piecewise_scalar(_B5A_LENS_KFS, f)
    elif f <= 1580.0:
        cp = _eval_piecewise_vec3(_B5B_CAM_KFS, f)
        tp = _eval_piecewise_vec3(_B5B_TGT_KFS, f)
        ln = _eval_piecewise_scalar(_B5B_LENS_KFS, f)
    else:
        cp = _eval_piecewise_vec3(_B6_CAM_KFS, f)
        tp = _eval_piecewise_vec3(_B6_TGT_KFS, f)
        ln = _eval_piecewise_scalar(_B6_LENS_KFS, f)

    if include_motion_fx:
        dc, dt, dl = _motion_fx(f)
        cp = _add3(cp, dc)
        tp = _add3(tp, dt)
        ln = ln + dl

    focus_dist = _len3(_sub3(tp, cp))
    return {
        "cam_pos": cp,
        "tgt_pos": tp,
        "lens_mm": ln,
        "fstop": _fstop_at(f),
        "focus_dist": focus_dist,
    }


def build_camera(col: bpy.types.Collection, total_frames: int = C.TOTAL_FRAMES) -> dict:
    """Build Blender 5.3.0 portrait camera + tracked Empty target with CONSTANT hard cuts."""
    if bpy is None:
        raise RuntimeError("build_camera() requires Blender bpy.")

    tgt = bpy.data.objects.new("Cinema_Cam_Target", None)
    tgt.empty_display_type = "PLAIN_AXES"
    tgt.empty_display_size = 0.15
    col.objects.link(tgt)

    cam_d = bpy.data.cameras.new("Cinema_Camera_Data")
    cam_d.sensor_fit = "AUTO"
    cam_d.sensor_width = SENSOR_WIDTH_MM
    cam_d.clip_start = 0.05
    cam_d.clip_end = 250.0
    cam_d.dof.use_dof = True
    cam_o = bpy.data.objects.new("Cinema_Camera", cam_d)
    col.objects.link(cam_o)

    trk = cam_o.constraints.new(type="TRACK_TO")
    trk.target = tgt
    trk.track_axis = "TRACK_NEGATIVE_Z"
    trk.up_axis = "UP_Y"

    # Dense clean keyframe sampling + exact hard-cut framing boundaries
    kfs: set[int] = set(range(1, total_frames + 1, 3)) | {1, total_frames}
    for c in HARD_CUT_FRAMES:
        kfs.add(c - 1)
        kfs.add(c)

    for f in sorted(kfs):
        st = camera_state_at(float(f), include_motion_fx=True)
        cam_o.location = Vector(st["cam_pos"])
        tgt.location = Vector(st["tgt_pos"])
        cam_d.lens = st["lens_mm"]
        cam_d.dof.focus_distance = st["focus_dist"]
        cam_d.dof.aperture_fstop = st["fstop"]
        cam_o.keyframe_insert("location", frame=f)
        tgt.keyframe_insert("location", frame=f)
        cam_d.keyframe_insert("lens", frame=f)
        cam_d.dof.keyframe_insert("focus_distance", frame=f)
        cam_d.dof.keyframe_insert("aperture_fstop", frame=f)

    # Enforce CONSTANT interpolation on pre-cut frames so hard cuts never slide across 1 frame!
    for anim_owner in (cam_o, tgt, cam_d):
        ad = anim_owner.animation_data
        if not ad or not ad.action:
            continue
        for fc in C.iter_action_fcurves(ad.action):
            for kp in fc.keyframe_points:
                if int(round(kp.co.x)) in _HARD_CUT_PREV_FRAMES:
                    kp.interpolation = "CONSTANT"
                else:
                    kp.interpolation = "LINEAR"

    return {"camera": cam_o, "target": tgt}
