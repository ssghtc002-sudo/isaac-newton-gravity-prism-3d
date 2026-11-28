"""
Cinema Camera Engine for Episode #3:
"Why Bulletproof Vests DON'T Send You Flying Backward" (60 fps, 1800 frames, 9:16 Portrait).

Pure-Python safe (`bpy` / `mathutils` imported conditionally) so `verify_framing.py` and
`labels_3d.py` can import `camera_state_at` and `project_ndc` directly outside Blender
with 0.000 mm trajectory, focus-plane, or lens drift.

Public API:
  - project_ndc(pt_w, cam_pos, tgt_pos, lens_mm: float) -> tuple[float, float, float]
  - camera_state_at(f: float, include_motion_fx: bool = True) -> dict
      returns {"cam_pos": (x, y, z), "tgt_pos": (x, y, z),
               "lens_mm": float, "fstop": float, "focus_dist": float}
  - build_camera(col: bpy.types.Collection, total_frames: int = 1800) -> dict
      creates `Cam_Main` + `Cam_Target` (`TRACK_TO` -Z / UP_Y), keys transform, lens,
      DOF `focus_distance` and `aperture_fstop` across `1..total_frames`, sets `CONSTANT`
      handles on hard-cut boundary frames via `C.iter_action_fcurves`, sets `scene.camera`,
      and returns {"cam_obj", "camera", "target", "cam_data"}.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Sequence

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import vest_contract as C  # noqa: E402

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
ASPECT_W_OVER_H = RES_X / RES_Y                      # 0.5625
SENSOR_WIDTH_MM = 36.0
SENSOR_H_EFF_MM = SENSOR_WIDTH_MM                    # 36.0 mm (portrait height is major axis)
SENSOR_W_EFF_MM = SENSOR_WIDTH_MM * ASPECT_W_OVER_H  # 20.25 mm

HARD_CUT_FRAMES: tuple[int, ...] = C.HARD_CUT_FRAMES  # (211, 511, 1261, 1561)
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
# MOTION FX LAYERS (IMPACT MICRO-PUNCH, MUZZLE JOLT, KEVLAR BUZZ, CERAMIC SHATTER)
# ==============================================================================
def _motion_fx(f: float) -> tuple[Vec3, Vec3, float]:
    """Physical camera micro-impulses & tremors cleanly bounded inside each beat."""
    cx = cy = cz = 0.0
    tx = ty = tz = 0.0
    lens_d = 0.0

    # Beat 1 (f=75..112): Crisp slow-mo chest-plate impact punch + spark burst micro-shake
    f_imp1 = float(C.EV["b1_impact"])
    if f_imp1 <= f <= f_imp1 + 37.0:
        dt = f - f_imp1
        env = (
            C.smoothstep(f_imp1, f_imp1 + 2.5, f)
            * math.exp(-0.11 * dt)
            * (1.0 - C.smoothstep(f_imp1 + 26.0, f_imp1 + 37.0, f))
        )
        cx += 0.008 * env * math.sin(dt * 1.65)
        cy += -0.006 * env * math.cos(dt * 1.35)
        cz += 0.009 * env * math.sin(dt * 1.95)
        tz += 0.003 * env * math.sin(dt * 1.95)
        lens_d += 0.45 * env

    # Beat 2 (f=245..268): Muzzle flash recoil + (f=335..365): Vest strike thump
    f_fire2 = float(C.EV["b2_muzzle_fire"])
    if f_fire2 <= f <= f_fire2 + 23.0:
        dt = f - f_fire2
        env = (
            C.smoothstep(f_fire2, f_fire2 + 2.0, f)
            * math.exp(-0.18 * dt)
            * (1.0 - C.smoothstep(f_fire2 + 15.0, f_fire2 + 23.0, f))
        )
        cx += -0.007 * env * math.sin(dt * 1.70)
        cz += 0.006 * env * math.cos(dt * 1.90)
        lens_d += 0.30 * env

    f_hit2 = float(C.EV["b2_vest_hit"])
    if f_hit2 <= f <= f_hit2 + 30.0:
        dt = f - f_hit2
        env = (
            C.smoothstep(f_hit2, f_hit2 + 2.5, f)
            * math.exp(-0.14 * dt)
            * (1.0 - C.smoothstep(f_hit2 + 20.0, f_hit2 + 30.0, f))
        )
        cx += 0.006 * env * math.sin(dt * 1.55)
        cz += -0.006 * env * math.cos(dt * 1.75)

    # Beat 3 (f=615..780): High-frequency taut Kevlar weave harmonic buzz (1.6 mm macro)
    f_kev = float(C.EV["b3_kevlar_impact"])
    if f_kev <= f <= f_kev + 165.0:
        w = C.smoothstep(f_kev, f_kev + 12.0, f) * (1.0 - C.smoothstep(f_kev + 125.0, f_kev + 165.0, f))
        cx += w * 0.0016 * math.sin((f - f_kev) * 0.90)
        cz += w * 0.0018 * math.cos((f - f_kev) * 1.10)

    # Beat 4 (f=965..1075): Ceramic strike shock + brittle fracture burst jolt
    f_cer = float(C.EV["b4_rifle_impact"])
    if f_cer <= f <= f_cer + 40.0:
        dt = f - f_cer
        env = (
            C.smoothstep(f_cer, f_cer + 2.0, f)
            * math.exp(-0.12 * dt)
            * (1.0 - C.smoothstep(f_cer + 28.0, f_cer + 40.0, f))
        )
        cx += 0.008 * env * math.sin(dt * 1.80)
        cy += -0.007 * env * math.cos(dt * 1.45)
        cz += 0.009 * env * math.sin(dt * 2.10)
        lens_d += 0.40 * env

    f_shat = float(C.EV["b4_ceramic_shatter"])
    if f_shat <= f <= f_shat + 38.0:
        dt = f - f_shat
        env = (
            C.smoothstep(f_shat, f_shat + 2.0, f)
            * math.exp(-0.13 * dt)
            * (1.0 - C.smoothstep(f_shat + 26.0, f_shat + 38.0, f))
        )
        cx += -0.007 * env * math.sin(dt * 1.95)
        cz += 0.008 * env * math.cos(dt * 2.20)
        tx += 0.002 * env * math.sin(dt * 1.95)
        tz += 0.002 * env * math.cos(dt * 2.20)

    # Beat 5 (f=1295..1318 & 1365..1388): Crisp blueprint vector snap pulses
    for f_ev, sign in ((float(C.EV["b5_dual_fire"]), -1.0), (float(C.EV["b5_vest_absorb"]), 1.0)):
        if f_ev <= f <= f_ev + 22.0:
            dt = f - f_ev
            env = (
                C.smoothstep(f_ev, f_ev + 2.0, f)
                * math.exp(-0.18 * dt)
                * (1.0 - C.smoothstep(f_ev + 14.0, f_ev + 22.0, f))
            )
            cx += sign * 0.006 * env * math.sin(dt * 1.60)
            cz += 0.004 * env * math.cos(dt * 1.75)

    return (cx, cy, cz), (tx, ty, tz), lens_d


# ==============================================================================
# 6-BEAT CAMERA CHOREOGRAPHY (9:16 Portrait, Always Moving, Zero Gimbal Flip)
# ==============================================================================
# Beat 1 (1..210): Sweeping 3/4 chest-level slow-mo tracking shot -> push into impact
_B1_CAM_KFS: list[tuple[float, Vec3]] = [
    (1.0, (0.04, -2.90, 1.34)),
    (22.0, (0.08, -2.86, 1.34)),
    (60.0, (0.38, -2.52, 1.35)),
    (95.0, (0.54, -2.32, 1.36)),
    (210.0, (0.64, -2.20, 1.36)),
]
_B1_TGT_KFS: list[tuple[float, Vec3]] = [
    (1.0, (0.20, 0.00, 1.24)),
    (22.0, (0.24, 0.00, 1.24)),
    (60.0, (0.80, 0.00, 1.25)),
    (95.0, (1.02, 0.00, 1.26)),
    (210.0, (1.08, 0.00, 1.26)),
]
_B1_LENS_KFS: list[tuple[float, float]] = [
    (1.0, 25.0),
    (22.0, 25.0),
    (95.0, 26.0),
    (210.0, 26.5),
]

# Beat 2 (211..510, hard cut @ 211): Range profile tracking shooter muzzle (-1.10) -> hero vest (+1.00)
_B2_CAM_KFS: list[tuple[float, Vec3]] = [
    (211.0, (-1.05, -3.55, 1.34)),
    (265.0, (-0.68, -3.25, 1.34)),
    (335.0, (0.36, -2.62, 1.34)),
    (425.0, (0.52, -2.46, 1.35)),
    (510.0, (0.60, -2.36, 1.35)),
]
_B2_TGT_KFS: list[tuple[float, Vec3]] = [
    (211.0, (-1.00, 0.00, 1.26)),
    (265.0, (-0.58, 0.00, 1.26)),
    (335.0, (0.42, 0.00, 1.26)),
    (425.0, (0.62, 0.00, 1.26)),
    (510.0, (0.72, 0.00, 1.26)),
]

# Beat 3-4 (511..1260, hard cut @ 511, SMOOTH continuous flow across 900 -> 901):
# Beat 3 (511..900): Action-locked macro Kevlar trampoline weave around Vest_Macro_Root (1.00, 0, 1.32)
# Beat 4 (901..1260): Smooth pull-back & orbit exposing SiC ceramic plate shatter + Dyneema backer
_B34_CAM_KFS: list[tuple[float, Vec3]] = [
    (511.0, (0.72, -0.82, 1.36)),
    (595.0, (0.78, -0.74, 1.36)),
    (660.0, (0.82, -0.68, 1.36)),
    (755.0, (0.85, -0.64, 1.37)),
    (855.0, (0.80, -0.80, 1.38)),
    (901.0, (0.74, -0.96, 1.38)),
    (930.0, (0.76, -0.96, 1.38)),
    (990.0, (0.82, -0.88, 1.39)),
    (1125.0, (0.88, -0.84, 1.40)),
    (1260.0, (0.94, -0.88, 1.41)),
]
_B34_TGT_KFS: list[tuple[float, Vec3]] = [
    (511.0, (0.88, 0.00, 1.31)),
    (595.0, (0.95, 0.00, 1.31)),
    (660.0, (0.99, 0.00, 1.31)),
    (755.0, (1.01, 0.00, 1.31)),
    (855.0, (0.96, 0.00, 1.31)),
    (901.0, (0.88, 0.00, 1.31)),
    (930.0, (0.90, 0.00, 1.31)),
    (990.0, (1.00, 0.00, 1.31)),
    (1125.0, (1.02, 0.00, 1.31)),
    (1260.0, (1.02, 0.00, 1.31)),
]
_B34_LENS_KFS: list[tuple[float, float]] = [
    (511.0, 33.0),
    (615.0, 35.0),
    (755.0, 36.0),
    (855.0, 32.0),
    (901.0, 29.0),
    (1030.0, 30.0),
    (1260.0, 29.0),
]
_B34_FSTOP_KFS: list[tuple[float, float]] = [
    (511.0, 4.0),
    (560.0, 4.0),
    (590.0, 7.5),
    (1260.0, 7.5),
]

# Beat 5 (1261..1560, hard cut @ 1261): Clean @zackdfilms studio blue blueprint F1 = -F2 symmetry
_B5_CAM_KFS: list[tuple[float, Vec3]] = [
    (1261.0, (0.00, -4.92, 1.28)),
    (1385.0, (0.00, -4.82, 1.28)),
    (1560.0, (0.00, -4.70, 1.28)),
]
_B5_TGT_KFS: list[tuple[float, Vec3]] = [
    (1261.0, (0.00, 0.00, 1.22)),
    (1385.0, (0.00, 0.00, 1.22)),
    (1560.0, (0.00, 0.00, 1.22)),
]

# Beat 6 (1561..1800, hard cut @ 1561): Hero 3/4 steady stance payoff & intact glass backdrop
_B6_CAM_KFS: list[tuple[float, Vec3]] = [
    (1561.0, (0.50, -2.22, 1.36)),
    (1680.0, (0.46, -2.32, 1.38)),
    (1800.0, (0.42, -2.44, 1.40)),
]
_B6_TGT_KFS: list[tuple[float, Vec3]] = [
    (1561.0, (1.10, 0.00, 1.24)),
    (1680.0, (1.13, 0.00, 1.25)),
    (1800.0, (1.16, 0.00, 1.26)),
]


def camera_state_at(f: float, include_motion_fx: bool = True) -> dict[str, object]:
    """Deterministic pure-Python camera state evaluator for frame `f` in `1..1800`."""
    f = C.clamp(float(f), 1.0, float(C.TOTAL_FRAMES))

    if f <= 210.0:
        cam_pos = _eval_piecewise_vec3(_B1_CAM_KFS, f)
        tgt_pos = _eval_piecewise_vec3(_B1_TGT_KFS, f)
        lens_mm = _eval_piecewise_scalar(_B1_LENS_KFS, f)
        fstop = 7.5
    elif f <= 510.0:
        cam_pos = _eval_piecewise_vec3(_B2_CAM_KFS, f)
        tgt_pos = _eval_piecewise_vec3(_B2_TGT_KFS, f)
        lens_mm = 26.0
        fstop = 7.5
    elif f <= 1260.0:
        cam_pos = _eval_piecewise_vec3(_B34_CAM_KFS, f)
        tgt_pos = _eval_piecewise_vec3(_B34_TGT_KFS, f)
        lens_mm = _eval_piecewise_scalar(_B34_LENS_KFS, f)
        fstop = _eval_piecewise_scalar(_B34_FSTOP_KFS, f)
    elif f <= 1560.0:
        cam_pos = _eval_piecewise_vec3(_B5_CAM_KFS, f)
        tgt_pos = _eval_piecewise_vec3(_B5_TGT_KFS, f)
        lens_mm = 24.0
        fstop = 8.0
    else:
        cam_pos = _eval_piecewise_vec3(_B6_CAM_KFS, f)
        tgt_pos = _eval_piecewise_vec3(_B6_TGT_KFS, f)
        lens_mm = 26.0
        fstop = 7.5

    if include_motion_fx:
        fx_cam, fx_tgt, fx_lens = _motion_fx(f)
        cam_pos = _add3(cam_pos, fx_cam)
        tgt_pos = _add3(tgt_pos, fx_tgt)
        lens_mm += fx_lens

    focus_dist = max(0.25, _len3(_sub3(tgt_pos, cam_pos)))

    return {
        "cam_pos": cam_pos,
        "tgt_pos": tgt_pos,
        "lens_mm": float(lens_mm),
        "fstop": float(fstop),
        "focus_dist": float(focus_dist),
    }


# ==============================================================================
# BLENDER 5.3.0 SCENE BUILDER PUBLIC API
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


def build_camera(col, total_frames: int = C.TOTAL_FRAMES) -> dict[str, object]:
    """
    Create and animate `Cam_Main` and `Cam_Target` (`TRACK_TO` -Z / UP_Y) across `1..total_frames`.
    Enables DOF with explicit per-frame `focus_distance` and `aperture_fstop` keyframes,
    sets `CONSTANT` interpolation across hard cuts via `C.iter_action_fcurves`,
    assigns `scene.camera`, and returns `{"cam_obj", "camera", "target", "cam_data"}`.
    """
    if bpy is None:
        raise RuntimeError("build_camera() must run inside Blender (bpy unavailable).")

    for name in ("Cam_Main", "Cam_Target"):
        old_obj = bpy.data.objects.get(name)
        if old_obj is not None:
            bpy.data.objects.remove(old_obj, do_unlink=True)

    cam_data = bpy.data.cameras.new("Cam_Main")
    cam_data.sensor_fit = "AUTO"
    cam_data.sensor_width = SENSOR_WIDTH_MM
    cam_data.clip_start = 0.02
    cam_data.clip_end = 1000.0
    cam_data.lens = 25.0

    cam_data.dof.use_dof = True
    cam_data.dof.focus_object = None
    cam_data.dof.aperture_blades = 7
    cam_data.dof.aperture_ratio = 1.0
    cam_data.dof.aperture_fstop = 7.5
    cam_data.dof.focus_distance = 2.80

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

        cam_data.lens = float(st["lens_mm"])
        cam_data.keyframe_insert(data_path="lens", frame=f)

        cam_data.dof.focus_distance = float(st["focus_dist"])
        cam_data.dof.keyframe_insert(data_path="focus_distance", frame=f)

        cam_data.dof.aperture_fstop = float(st["fstop"])
        cam_data.dof.keyframe_insert(data_path="aperture_fstop", frame=f)

    _set_handles_with_hard_cuts(cam_obj)
    _set_handles_with_hard_cuts(target)
    _set_handles_with_hard_cuts(cam_data)

    st0 = camera_state_at(1.0, include_motion_fx=True)
    cam_obj.location = Vector(st0["cam_pos"])
    target.location = Vector(st0["tgt_pos"])
    cam_data.lens = float(st0["lens_mm"])
    cam_data.dof.focus_distance = float(st0["focus_dist"])
    cam_data.dof.aperture_fstop = float(st0["fstop"])

    return {
        "cam_obj": cam_obj,
        "camera": cam_obj,
        "target": target,
        "cam_data": cam_data,
    }


if __name__ == "__main__":
    if bpy is not None:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        test_col = bpy.data.collections.new("Cam_SelfTest")
        bpy.context.scene.collection.children.link(test_col)
        res = build_camera(test_col, total_frames=C.TOTAL_FRAMES)
        assert res["camera"].name == "Cam_Main"
        assert bpy.context.scene.camera is res["camera"]
        fcs = list(C.iter_action_fcurves(res["camera"].animation_data.action))
        print(f"Cam_Main self-test PASSED in Blender ({len(fcs)} fcurves).")
    else:
        for f_chk in (1, 75, 211, 245, 335, 511, 615, 735, 901, 965, 1030, 1261, 1365, 1561, 1750):
            st = camera_state_at(float(f_chk), include_motion_fx=True)
            print(
                f"f={f_chk:04d} cam={tuple(round(v, 2) for v in st['cam_pos'])} "
                f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f} focus={st['focus_dist']:.2f}m"
            )
