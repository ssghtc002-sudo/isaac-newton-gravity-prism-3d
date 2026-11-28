"""
Cinema Camera Engine for Episode #4:
"How a 172-Foot Stealth Bomber Looks Like a Bumblebee on Radar" (60 fps, 1800 frames, 9:16 Portrait).

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

import stealth_contract as C  # noqa: E402

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

HARD_CUT_FRAMES: tuple[int, ...] = C.HARD_CUT_FRAMES  # (221, 541, 961, 1321, 1561)
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
# MOTION FX LAYERS (RADAR PULSE PUNCH, JET PASS RUMBLE, TURBINE HARMONIC BUZZ)
# ==============================================================================
def _motion_fx(f: float) -> tuple[Vec3, Vec3, float]:
    """Subtle aerodynamic turbulence & RF pulse micro-impulses cleanly bounded inside each beat."""
    cx = cy = cz = 0.0
    tx = ty = tz = 0.0
    lens_d = 0.0

    # Beat 1 (f=68..108): Radar pulse wing deflection micro-punch
    f_def1 = float(C.EV["b1_wing_deflect"])
    if f_def1 <= f <= f_def1 + 40.0:
        dt = f - f_def1
        env = (
            C.smoothstep(f_def1, f_def1 + 3.0, f)
            * math.exp(-0.10 * dt)
            * (1.0 - C.smoothstep(f_def1 + 28.0, f_def1 + 40.0, f))
        )
        cx += 0.008 * env * math.sin(dt * 1.4)
        cz += 0.007 * env * math.cos(dt * 1.7)
        lens_d += 0.35 * env

    # Beat 2 (f=318..355): Blazing conventional jet crimson echo shock pulse
    f_echo2 = float(C.EV["b2_conv_echo"])
    if f_echo2 <= f <= f_echo2 + 37.0:
        dt = f - f_echo2
        env = (
            C.smoothstep(f_echo2, f_echo2 + 2.5, f)
            * math.exp(-0.12 * dt)
            * (1.0 - C.smoothstep(f_echo2 + 25.0, f_echo2 + 37.0, f))
        )
        cx += -0.007 * env * math.sin(dt * 1.6)
        cz += 0.006 * env * math.cos(dt * 1.9)
        lens_d += 0.30 * env

    # Beat 3 (f=655..820): Subtle macro jet-turbine micro-vibration inside S-duct
    if 655.0 <= f <= 820.0:
        w = C.smoothstep(655.0, 675.0, f) * (1.0 - C.smoothstep(790.0, 820.0, f))
        cx += w * 0.0018 * math.sin((f - 655.0) * 0.85)
        cz += w * 0.0018 * math.cos((f - 655.0) * 1.05)

    # Beat 4 (f=1160..1210): RCS 100 m^2 -> 0.001 m^2 collapse kinetic pulse
    f_rcs = float(C.EV["b4_rcs_shrink"])
    if f_rcs <= f <= f_rcs + 45.0:
        dt = f - f_rcs
        env = (
            C.smoothstep(f_rcs, f_rcs + 3.0, f)
            * math.exp(-0.09 * dt)
            * (1.0 - C.smoothstep(f_rcs + 30.0, f_rcs + 45.0, f))
        )
        cy += -0.009 * env
        lens_d += 0.40 * env

    # Beat 6 (f=1600..1670): Overhead moonlight banking air-wake swell
    f_pass = float(C.EV["b6_moon_pass"])
    if f_pass <= f <= f_pass + 70.0:
        dt = f - f_pass
        env = C.smoothstep(f_pass, f_pass + 14.0, f) * (1.0 - C.smoothstep(f_pass + 45.0, f_pass + 70.0, f))
        cx += 0.006 * env * math.sin(dt * 0.25)
        cz += 0.005 * env * math.cos(dt * 0.30)

    return (cx, cy, cz), (tx, ty, tz), lens_d


# ==============================================================================
# 6-BEAT CAMERA CHOREOGRAPHY (9:16 Portrait, Continuously Moving, Zero Gimbal Flip)
# ==============================================================================
# Beat 1 (1..220): Inside Enemy Radar Command Room looking over blank console & through window at sunlit B-2
_B1_CAM_KFS: list[tuple[float, Vec3]] = [
    (1.0, (-0.26, -4.25, 1.62)),
    (75.0, (-0.08, -4.08, 1.64)),
    (150.0, (0.08, -3.96, 1.65)),
    (220.0, (0.18, -3.86, 1.66)),
]
_B1_TGT_KFS: list[tuple[float, Vec3]] = [
    (1.0, (-0.12, 0.28, 1.82)),
    (75.0, (-0.02, 0.22, 1.80)),
    (150.0, (0.06, 0.16, 1.79)),
    (220.0, (0.12, 0.10, 1.78)),
]
_B1_LENS_KFS: list[tuple[float, float]] = [
    (1.0, 24.0),
    (120.0, 24.8),
    (220.0, 25.5),
]

# Beat 2 (221..540, hard cut @ 221): Stacked vertical contrast — Conv Jet (top) vs Stealth Wing (bottom)
_B2_CAM_KFS: list[tuple[float, Vec3]] = [
    (221.0, (-0.24, -4.55, 2.02)),
    (340.0, (-0.06, -4.38, 2.00)),
    (440.0, (0.08, -4.24, 1.98)),
    (540.0, (0.18, -4.12, 1.96)),
]
_B2_TGT_KFS: list[tuple[float, Vec3]] = [
    (221.0, (-0.12, 0.10, 2.02)),
    (340.0, (-0.02, 0.10, 2.00)),
    (440.0, (0.04, 0.08, 1.98)),
    (540.0, (0.08, 0.08, 1.98)),
]
_B2_LENS_KFS: list[tuple[float, float]] = [
    (221.0, 24.0),
    (380.0, 24.8),
    (540.0, 25.6),
]

# Beat 3 (541..960, hard cut @ 541): Macro X-ray sweep over dorsal serpentine S-ducts & turbines
_B3_CAM_KFS: list[tuple[float, Vec3]] = [
    (541.0, (-0.26, -1.62, 2.04)),
    (660.0, (-0.18, -1.48, 2.06)),
    (810.0, (-0.10, -1.40, 2.07)),
    (960.0, (-0.04, -1.34, 2.06)),
]
_B3_TGT_KFS: list[tuple[float, Vec3]] = [
    (541.0, (-0.08, 0.00, 1.86)),
    (750.0, (-0.05, 0.00, 1.86)),
    (960.0, (-0.02, 0.00, 1.86)),
]
_B3_LENS_KFS: list[tuple[float, float]] = [
    (541.0, 34.0),
    (750.0, 36.5),
    (960.0, 38.0),
]

# Beat 4 (961..1320, hard cut @ 961): Macro RAM carbon honeycomb -> holographic 0.001 m^2 Bumblebee RCS
_B4_CAM_KFS: list[tuple[float, Vec3]] = [
    (961.0, (-0.18, -1.52, 1.95)),
    (1110.0, (-0.06, -1.44, 1.94)),
    (1200.0, (0.12, -1.34, 1.93)),
    (1320.0, (0.22, -1.26, 1.92)),
]
_B4_TGT_KFS: list[tuple[float, Vec3]] = [
    (961.0, (-0.08, 0.00, 1.85)),
    (1120.0, (-0.04, 0.00, 1.85)),
    (1200.0, (0.14, -0.02, 1.86)),
    (1320.0, (0.18, -0.03, 1.86)),
]
_B4_LENS_KFS: list[tuple[float, float]] = [
    (961.0, 34.0),
    (1140.0, 36.0),
    (1320.0, 39.0),
]

# Beat 5 (1321..1560, hard cut @ 1321): Main-Scene Daylight Real-Jet 90-deg vs 33-deg Deflection Explainer
_B5_CAM_KFS: list[tuple[float, Vec3]] = [
    (1321.0, (-0.22, -4.38, 1.90)),
    (1440.0, (-0.14, -4.20, 1.90)),
    (1560.0, (-0.06, -4.04, 1.90)),
]
_B5_TGT_KFS: list[tuple[float, Vec3]] = [
    (1321.0, (-0.14, 0.08, 1.88)),
    (1560.0, (-0.08, 0.08, 1.88)),
]
_B5_LENS_KFS: list[tuple[float, float]] = [
    (1321.0, 24.5),
    (1560.0, 26.0),
]

# Beat 6 (1561..1800, hard cut @ 1561): Inside Enemy Radar Command Room console blip + sunlit B-2 outside window
_B6_CAM_KFS: list[tuple[float, Vec3]] = [
    (1561.0, (-0.26, -3.64, 1.62)),
    (1680.0, (-0.08, -3.40, 1.68)),
    (1800.0, (0.08, -3.18, 1.74)),
]
_B6_TGT_KFS: list[tuple[float, Vec3]] = [
    (1561.0, (-0.02, 0.14, 1.86)),
    (1680.0, (0.01, 0.12, 1.84)),
    (1800.0, (0.04, 0.10, 1.82)),
]
_B6_LENS_KFS: list[tuple[float, float]] = [
    (1561.0, 24.5),
    (1800.0, 26.0),
]


def _fstop_at(f: float) -> float:
    """Keep f/7.5..8.0 whenever 3D plaques are onscreen; shallow macro f/3.8 ONLY during pure visual B3 intro."""
    if f <= 220.0:
        return 7.5
    if f <= 540.0:
        return 7.5
    if f <= 960.0:
        # B3_SDUCTS label appears at f=615; rack smoothly from f/3.8 -> f/7.5 before f=605
        if f <= 582.0:
            return 3.8
        if f < 606.0:
            return C.lerp(3.8, 7.5, C.smootherstep(582.0, 606.0, f))
        return 7.5
    if f <= 1320.0:
        return 7.5
    if f <= 1560.0:
        return 8.0
    return 7.5


def camera_state_at(f: float, include_motion_fx: bool = True) -> dict:
    """Deterministic camera position, target, focal length, f-stop, and exact DOF focus distance."""
    ff = C.clamp(float(f), 1.0, float(C.TOTAL_FRAMES))
    if ff <= 220.0:
        cam_p = _eval_piecewise_vec3(_B1_CAM_KFS, ff)
        tgt_p = _eval_piecewise_vec3(_B1_TGT_KFS, ff)
        lens = _eval_piecewise_scalar(_B1_LENS_KFS, ff)
    elif ff <= 540.0:
        cam_p = _eval_piecewise_vec3(_B2_CAM_KFS, ff)
        tgt_p = _eval_piecewise_vec3(_B2_TGT_KFS, ff)
        lens = _eval_piecewise_scalar(_B2_LENS_KFS, ff)
    elif ff <= 960.0:
        cam_p = _eval_piecewise_vec3(_B3_CAM_KFS, ff)
        tgt_p = _eval_piecewise_vec3(_B3_TGT_KFS, ff)
        lens = _eval_piecewise_scalar(_B3_LENS_KFS, ff)
    elif ff <= 1320.0:
        cam_p = _eval_piecewise_vec3(_B4_CAM_KFS, ff)
        tgt_p = _eval_piecewise_vec3(_B4_TGT_KFS, ff)
        lens = _eval_piecewise_scalar(_B4_LENS_KFS, ff)
    elif ff <= 1560.0:
        cam_p = _eval_piecewise_vec3(_B5_CAM_KFS, ff)
        tgt_p = _eval_piecewise_vec3(_B5_TGT_KFS, ff)
        lens = _eval_piecewise_scalar(_B5_LENS_KFS, ff)
    else:
        cam_p = _eval_piecewise_vec3(_B6_CAM_KFS, ff)
        tgt_p = _eval_piecewise_vec3(_B6_TGT_KFS, ff)
        lens = _eval_piecewise_scalar(_B6_LENS_KFS, ff)

    if include_motion_fx:
        dc, dt, dl = _motion_fx(ff)
        cam_p = _add3(cam_p, dc)
        tgt_p = _add3(tgt_p, dt)
        lens = C.clamp(lens + dl, 20.0, 85.0)

    focus_dist = max(0.25, _len3(_sub3(tgt_p, cam_p)))
    return {
        "cam_pos": cam_p,
        "tgt_pos": tgt_p,
        "lens_mm": float(lens),
        "fstop": float(_fstop_at(ff)),
        "focus_dist": float(focus_dist),
    }


def _set_hard_cut_constant(id_block) -> None:
    """Apply CONSTANT interpolation strictly on hard-cut boundary pre-frames via Blender 5.x safe iterator."""
    ad = getattr(id_block, "animation_data", None)
    if not ad or not ad.action:
        return
    for fc in C.iter_action_fcurves(ad.action):
        for kp in fc.keyframe_points:
            frame_i = int(round(kp.co.x))
            kp.interpolation = "CONSTANT" if frame_i in _HARD_CUT_PREV_FRAMES else "LINEAR"


def build_camera(col, total_frames: int = C.TOTAL_FRAMES) -> dict:
    """Create and keyframe `Cam_Main` + `Cam_Target` (`TRACK_TO` -Z / UP_Y) across `1..total_frames`."""
    if bpy is None:
        raise RuntimeError("build_camera() must run inside Blender (`bpy` unavailable).")

    cam_data = bpy.data.cameras.new("Cam_Main")
    cam_data.sensor_fit = "AUTO"
    cam_data.sensor_width = SENSOR_WIDTH_MM
    cam_data.clip_start = 0.05
    cam_data.clip_end = 120.0
    cam_data.dof.use_dof = True

    cam_obj = bpy.data.objects.new("Cam_Main", cam_data)
    col.objects.link(cam_obj)

    tgt_obj = bpy.data.objects.new("Cam_Target", None)
    tgt_obj.empty_display_type = "PLAIN_AXES"
    tgt_obj.empty_display_size = 0.12
    col.objects.link(tgt_obj)

    trk = cam_obj.constraints.new("TRACK_TO")
    trk.target = tgt_obj
    trk.track_axis = "TRACK_NEGATIVE_Z"
    trk.up_axis = "UP_Y"

    T = max(1, int(total_frames))
    for f in range(1, T + 1):
        st = camera_state_at(float(f), include_motion_fx=True)
        cam_obj.location = Vector(st["cam_pos"])
        tgt_obj.location = Vector(st["tgt_pos"])
        cam_data.lens = st["lens_mm"]
        cam_data.dof.focus_distance = st["focus_dist"]
        cam_data.dof.aperture_fstop = st["fstop"]

        cam_obj.keyframe_insert("location", frame=f)
        tgt_obj.keyframe_insert("location", frame=f)
        cam_data.keyframe_insert("lens", frame=f)
        cam_data.dof.keyframe_insert("focus_distance", frame=f)
        cam_data.dof.keyframe_insert("aperture_fstop", frame=f)

    _set_hard_cut_constant(cam_obj)
    _set_hard_cut_constant(tgt_obj)
    _set_hard_cut_constant(cam_data)

    bpy.context.scene.camera = cam_obj
    return {
        "cam_obj": cam_obj,
        "camera": cam_obj,
        "target": tgt_obj,
        "cam_data": cam_data,
    }
