"""
cinema_camera.py — Continuous 8-Segment 9:16 Portrait Cinema Camera & Exact DOF Focus Rig (V2 Ultimate)
for Episode #5: "Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock" (60 fps, 1800 frames).

Pure-Python safe (zero `bpy` required for `camera_state_at` & `project_ndc`) so `labels_3d.py` and
`verify_framing.py` validate exact 9:16 NDC coordinates & DOF sharpness before rendering!

V2 Ultimate Camera Grammar (`AGENTS.md` V12):
  - Beat 1A (1..115):     Tight dramatic exterior tracking shot beside the modern rescue chopper as its
                          dorsal turbine explodes in fire, sparks & black smoke (`f/7.5`, sharp label!)
  - Beat 1B (116..240):   Hard cut INSIDE the dark enclosed cockpit — urgent close-up on Pilot (+0.40),
                          Co-Pilot (-0.40), crashing 3D `ENG 0%` tachometer needle & collective slam (`f/7.5`)
  - Beat 2  (241..570):   Macro push-in on X-ray dorsal cowling, seized crimson turbine & golden cam sprags (`f/7.8`)
  - Beat 3  (571..960):   Dynamic aerial tracking arc following autorotation glide + wide emerald 3D arrows (`f/7.8`)
  - Beat 4  (961..1290):  Crisp side-by-side comparison: hero 3D Maple Seed (left) vs Rescue Rotor (right) (`f/7.5`)
  - Beat 5A (1291..1450): Label-free enclosed cockpit flare pull close-up with creamy `f/3.6` shallow bokeh!
  - Beat 5B (1451..1580): Dramatic low-angle exterior 30-foot nose-up flare over rooftop helipad (`f/8.0`)
  - Beat 6  (1581..1800): Golden-hour rooftop helipad touchdown hero arc with 0.000 mm flush skid contact (`f/7.5`)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import heli_contract as C  # noqa: E402

try:
    import bpy  # type: ignore[import-not-found]
    from mathutils import Vector  # type: ignore[import-not-found]
except ImportError:
    bpy = None  # type: ignore[assignment]
    Vector = None  # type: ignore[assignment]

RES_X: int = 1080
RES_Y: int = 1920
ASPECT: float = RES_X / float(RES_Y)  # 0.5625 (9:16 portrait)
SENSOR_WIDTH_MM: float = 36.0
# Blender sensor_fit='AUTO' on portrait 1080x1920 fits 36mm along vertical sensor height!
SENSOR_H_EFF_MM: float = SENSOR_WIDTH_MM
SENSOR_W_EFF_MM: float = SENSOR_WIDTH_MM * ASPECT  # 20.25 mm

Vec3 = tuple[float, float, float]


def _sub3(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add3(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _dot3(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross3(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _len3(a: Sequence[float]) -> float:
    return math.sqrt(_dot3(a, a))


def _norm3(a: Sequence[float]) -> Vec3:
    L = _len3(a)
    if L < 1e-9:
        return (0.0, 1.0, 0.0)
    return (a[0] / L, a[1] / L, a[2] / L)


def _lerp3(a: Vec3, b: Vec3, t: float) -> Vec3:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
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

    # Beat 1A (f=18..72): Explosive turbine compressor-stall camera shockwave!
    f_eng = float(C.EV["b1_engine_flameout"])
    if f_eng <= f <= f_eng + 54.0:
        dt = f - f_eng
        env = C.smoothstep(f_eng, f_eng + 2.0, f) * math.exp(-0.075 * dt) * (1.0 - C.smoothstep(f_eng + 40.0, f_eng + 54.0, f))
        cx += 0.022 * env * math.sin(dt * 2.1)
        cz += 0.018 * env * math.cos(dt * 2.4)
        lens_d += 0.55 * env

    # Beat 1B (f=118..165): Cockpit cabin shudder & collective slam impulse
    if 118.0 <= f <= 165.0:
        dt = f - 118.0
        env = C.smoothstep(118.0, 122.0, f) * (1.0 - C.smoothstep(150.0, 165.0, f))
        cx += 0.008 * env * math.sin(dt * 1.8)
        cz += 0.007 * env * math.cos(dt * 2.2)

    # Beat 2 (f=345..380): Seized turbine shaft clunk micro-jolt
    f_sz = float(C.EV["b2_shaft_seize"])
    if f_sz <= f <= f_sz + 35.0:
        dt = f - f_sz
        env = C.smoothstep(f_sz, f_sz + 2.0, f) * math.exp(-0.12 * dt) * (1.0 - C.smoothstep(f_sz + 24.0, f_sz + 35.0, f))
        cx += -0.006 * env * math.sin(dt * 2.1)
        cz += 0.005 * env * math.cos(dt * 1.8)

    # Beat 5B (f=1462..1520): 30-foot flare air-bite cushion punch
    f_fl = float(C.EV["b5_flare_bite"])
    if f_fl <= f <= f_fl + 58.0:
        dt = f - f_fl
        env = C.smoothstep(f_fl, f_fl + 4.0, f) * math.exp(-0.07 * dt) * (1.0 - C.smoothstep(f_fl + 42.0, f_fl + 58.0, f))
        cz += 0.012 * env * math.sin(dt * 1.2)
        lens_d += 0.42 * env

    # Beat 6 (f=1685..1718): Gentle skid-on-helipad touchdown settle
    f_td = float(C.EV["b6_skid_touchdown"])
    if f_td <= f <= f_td + 33.0:
        dt = f - f_td
        env = C.smoothstep(f_td, f_td + 2.0, f) * math.exp(-0.14 * dt) * (1.0 - C.smoothstep(f_td + 22.0, f_td + 33.0, f))
        cz += -0.010 * env * math.sin(dt * 1.4)

    return (cx, cy, cz), (tx, ty, tz), lens_d


# ==============================================================================
# 8-SEGMENT CINEMA CAMERA CHOREOGRAPHY (9:16 PORTRAIT, ZERO GIMBAL FLIP)
# ==============================================================================
# Beat 1A (1..115): Dramatic Exterior Tracking Shot — Turbine Fire Burst, Sparks & Black Smoke!
_B1A_CAM_KFS: list[tuple[float, Vec3]] = [
    (1.0, (-0.56, -3.55, 2.30)),
    (45.0, (-0.28, -3.35, 2.22)),
    (115.0, (0.16, -3.15, 2.08)),
]
_B1A_TGT_KFS: list[tuple[float, Vec3]] = [
    (1.0, (-0.24, 0.30, 2.40)),
    (55.0, (-0.04, 0.25, 2.28)),
    (115.0, (0.14, 0.20, 2.12)),
]
_B1A_LENS_KFS: list[tuple[float, float]] = [
    (1.0, 24.5),
    (55.0, 26.2),
    (115.0, 27.8),
]

# Beat 1B (116..240, hard cut @ 116): Dark Enclosed 1:1 Cockpit — Panic Crew, Crashing ENG 0% & Collective Slam!
_B1B_CAM_KFS: list[tuple[float, Vec3]] = [
    (116.0, (-0.10, -1.78, 1.37)),
    (175.0, (0.02, -1.68, 1.35)),
    (240.0, (0.10, -1.60, 1.34)),
]
_B1B_TGT_KFS: list[tuple[float, Vec3]] = [
    (116.0, (0.00, 0.14, 1.18)),
    (175.0, (0.04, 0.12, 1.14)),
    (240.0, (0.06, 0.10, 1.13)),
]
_B1B_LENS_KFS: list[tuple[float, float]] = [
    (116.0, 21.8),
    (175.0, 22.8),
    (240.0, 23.6),
]

# Beat 2 (241..570, hard cut @ 241): Dorsal Macro X-Ray on Modern Cowling, Seized Shaft & 12 Golden Cam Sprags
_B2_CAM_KFS: list[tuple[float, Vec3]] = [
    (241.0, (-0.42, -1.66, 2.42)),
    (360.0, (-0.26, -1.52, 2.38)),
    (460.0, (-0.14, -1.42, 2.34)),
    (570.0, (-0.06, -1.36, 2.32)),
]
_B2_TGT_KFS: list[tuple[float, Vec3]] = [
    (241.0, (0.00, 0.04, 2.13)),
    (410.0, (0.00, 0.03, 2.14)),
    (570.0, (0.00, 0.02, 2.15)),
]
_B2_LENS_KFS: list[tuple[float, float]] = [
    (241.0, 30.5),
    (410.0, 33.0),
    (570.0, 34.8),
]

# Beat 3 (571..960, hard cut @ 571): Aerial Tracking Shot of Autorotation Glide & Wide 3D Emerald Ribbon Arrows
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

# Beat 4 (961..1290, hard cut @ 961): Side-by-Side Hero 3D Maple Seed (Left) vs Modern Rescue Rotor (Right)
_B4_CAM_KFS: list[tuple[float, Vec3]] = [
    (961.0, (-0.12, -3.52, 2.20)),
    (1125.0, (-0.02, -3.40, 2.00)),
    (1290.0, (0.06, -3.30, 1.82)),
]
_B4_TGT_KFS: list[tuple[float, Vec3]] = [
    (961.0, (0.00, 0.14, 2.20)),
    (1125.0, (0.00, 0.14, 1.98)),
    (1290.0, (0.00, 0.14, 1.76)),
]
_B4_LENS_KFS: list[tuple[float, float]] = [
    (961.0, 22.8),
    (1125.0, 23.5),
    (1290.0, 24.2),
]

# Beat 5A (1291..1450, hard cut @ 1291): Enclosed Cockpit Final Flare Close-Up (Label-Free f/3.6 Bokeh!)
_B5A_CAM_KFS: list[tuple[float, Vec3]] = [
    (1291.0, (0.16, -1.68, 1.37)),
    (1370.0, (0.10, -1.62, 1.38)),
    (1450.0, (0.04, -1.56, 1.39)),
]
_B5A_TGT_KFS: list[tuple[float, Vec3]] = [
    (1291.0, (0.08, 0.18, 1.21)),
    (1370.0, (0.06, 0.18, 1.20)),
    (1450.0, (0.02, 0.18, 1.19)),
]
_B5A_LENS_KFS: list[tuple[float, float]] = [
    (1291.0, 22.2),
    (1370.0, 22.8),
    (1450.0, 23.4),
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
    if f <= float(C.B1_EXTERIOR_END):
        return 7.5
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


def camera_state_at(f: float, include_motion_fx: bool = True) -> dict[str, Any]:
    """Evaluate exact world-space camera state at frame `f` (1..1800)."""
    f = C.clamp(float(f), 1.0, float(C.TOTAL_FRAMES))
    if f <= float(C.B1_EXTERIOR_END):
        cp = _eval_piecewise_vec3(_B1A_CAM_KFS, f)
        tp = _eval_piecewise_vec3(_B1A_TGT_KFS, f)
        ln = _eval_piecewise_scalar(_B1A_LENS_KFS, f)
    elif f <= 240.0:
        cp = _eval_piecewise_vec3(_B1B_CAM_KFS, f)
        tp = _eval_piecewise_vec3(_B1B_TGT_KFS, f)
        ln = _eval_piecewise_scalar(_B1B_LENS_KFS, f)
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
        ln = max(16.0, ln + dl)

    fdist = max(0.45, _len3(_sub3(tp, cp)))
    return {
        "cam_pos": cp,
        "tgt_pos": tp,
        "lens_mm": ln,
        "focus_dist": fdist,
        "fstop": _fstop_at(f),
    }


def _mark_hard_cut_fcurves(obj: Any, cut_frames: Sequence[int]) -> None:
    ad = getattr(obj, "animation_data", None)
    if not ad or not ad.action:
        return
    pre_cuts = {int(c) - 1 for c in cut_frames}
    for fc in C.iter_action_fcurves(ad.action):
        for kp in fc.keyframe_points:
            if int(round(kp.co.x)) in pre_cuts:
                kp.interpolation = "CONSTANT"
            else:
                kp.interpolation = "LINEAR"


def build_camera(col: Any, total_frames: int = C.TOTAL_FRAMES) -> dict[str, Any]:
    """Create & keyframe the 8-segment portrait cinema camera + TRACK_TO target + exact DOF focus rig."""
    if bpy is None:
        raise RuntimeError("build_camera() requires Blender bpy.")

    cam_data = bpy.data.cameras.new("CinemaCam_Heli_Data")
    cam_data.sensor_fit = "AUTO"
    cam_data.sensor_width = SENSOR_WIDTH_MM
    cam_data.clip_start = 0.05
    cam_data.clip_end = 420.0
    cam_data.dof.use_dof = True

    cam_obj = bpy.data.objects.new("CinemaCam_Heli", cam_data)
    col.objects.link(cam_obj)

    tgt_obj = bpy.data.objects.new("CinemaCam_Heli_Target", None)
    tgt_obj.empty_display_type = "PLAIN_AXES"
    tgt_obj.empty_display_size = 0.25
    col.objects.link(tgt_obj)

    trk = cam_obj.constraints.new("TRACK_TO")
    trk.target = tgt_obj
    trk.track_axis = "TRACK_NEGATIVE_Z"
    trk.up_axis = "UP_Y"

    # Dense keyframe schedule + exact pre-cut/cut frame pairs for zero motion-blur smear across cuts!
    kfs: set[int] = set(range(1, total_frames + 1, 3)) | {1, total_frames}
    for cf in C.HARD_CUT_FRAMES:
        for df in (-2, -1, 0, 1):
            ff = cf + df
            if 1 <= ff <= total_frames:
                kfs.add(ff)
    for ev_f in C.EV.values():
        for ff in range(max(1, ev_f - 2), min(total_frames, ev_f + 46) + 1, 2):
            kfs.add(ff)

    for f in sorted(kfs):
        st = camera_state_at(float(f), include_motion_fx=True)
        cam_obj.location = Vector(st["cam_pos"])
        cam_obj.keyframe_insert("location", frame=f)
        tgt_obj.location = Vector(st["tgt_pos"])
        tgt_obj.keyframe_insert("location", frame=f)

        cam_data.lens = float(st["lens_mm"])
        cam_data.keyframe_insert("lens", frame=f)
        cam_data.dof.focus_distance = float(st["focus_dist"])
        cam_data.dof.keyframe_insert("focus_distance", frame=f)
        cam_data.dof.aperture_fstop = float(st["fstop"])
        cam_data.dof.keyframe_insert("aperture_fstop", frame=f)

    _mark_hard_cut_fcurves(cam_obj, C.HARD_CUT_FRAMES)
    _mark_hard_cut_fcurves(tgt_obj, C.HARD_CUT_FRAMES)
    _mark_hard_cut_fcurves(cam_data, C.HARD_CUT_FRAMES)

    return {"camera": cam_obj, "target": tgt_obj, "data": cam_data}
