"""
Cinema Camera Engine for Episode #2:
"Why Bullets Stop Dead Just 3 Feet Underwater" (60 fps, 1800 frames, 9:16 Portrait).

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

import bullet_contract as C  # noqa: E402

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

HARD_CUT_FRAMES: tuple[int, ...] = (211, 541, 1261, 1561)
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
# MOTION FX LAYERS (HYDRO-SHOCK, MUZZLE RECOIL, CAVITATION BUZZ, SHATTER JOLT)
# ==============================================================================
def _motion_fx(f: float) -> tuple[Vec3, Vec3, float]:
    """Physical camera micro-impulses & tremors cleanly bounded within each beat."""
    cx = cy = cz = 0.0
    tx = ty = tz = 0.0
    lens_d = 0.0

    # Beat 1 (f=128..165): Subtle hydrodynamic pressure pulse as round halts at 3 FT
    f_stop1 = float(C.EV["b1_bullet_stop"])
    if f_stop1 - 6.0 <= f <= f_stop1 + 30.0:
        dt = max(0.0, f - (f_stop1 - 6.0))
        env = C.smoothstep(f_stop1 - 6.0, f_stop1, f) * math.exp(-0.12 * dt) * (1.0 - C.smoothstep(f_stop1 + 20.0, f_stop1 + 30.0, f))
        cx += 0.004 * env * math.sin(dt * 1.4)
        cz += 0.005 * env * math.cos(dt * 1.8)
        tz += 0.002 * env * math.cos(dt * 1.8)

    # Beat 2 (f=248..272): Muzzle blast recoil + (f=300..336): Water-entry density slam
    f_fire = float(C.EV["b2_muzzle_fire"])
    if f_fire <= f <= f_fire + 24.0:
        dt = f - f_fire
        env = C.smoothstep(f_fire, f_fire + 2.0, f) * math.exp(-0.18 * dt) * (1.0 - C.smoothstep(f_fire + 16.0, f_fire + 24.0, f))
        cx += -0.007 * env * math.sin(dt * 1.65)
        cz += 0.006 * env * math.cos(dt * 1.90)
        lens_d += 0.35 * env

    f_entry = float(C.EV["b2_water_entry"])
    if f_entry <= f <= f_entry + 36.0:
        dt = f - f_entry
        env = C.smoothstep(f_entry, f_entry + 2.5, f) * math.exp(-0.13 * dt) * (1.0 - C.smoothstep(f_entry + 26.0, f_entry + 36.0, f))
        cx += 0.009 * env * math.sin(dt * 1.50)
        cy += -0.006 * env * math.cos(dt * 1.35)
        cz += -0.008 * env * math.sin(dt * 1.80)
        tz += -0.003 * env * math.sin(dt * 1.80)
        lens_d += 0.45 * env

    # Beat 3 (f=560..850): Subtle high-frequency supercavitation macro hum (2 mm)
    if 560.0 <= f <= 850.0:
        w = C.smoothstep(560.0, 595.0, f) * (1.0 - C.smoothstep(815.0, 850.0, f))
        cx += w * 0.0018 * math.sin(f * 0.85)
        cz += w * 0.0020 * math.cos(f * 1.05)

    # Beat 4 (f=1055..1098): Violent 8-shard fragmentation shockwave at 3 FT
    f_shatter = float(C.EV["b4_shatter_burst"])
    if f_shatter <= f <= f_shatter + 43.0:
        dt = f - f_shatter
        env = C.smoothstep(f_shatter, f_shatter + 2.0, f) * math.exp(-0.11 * dt) * (1.0 - C.smoothstep(f_shatter + 32.0, f_shatter + 43.0, f))
        cx += 0.011 * env * math.sin(dt * 1.75)
        cy += -0.010 * env * math.cos(dt * 1.45)
        cz += 0.012 * env * math.sin(dt * 2.05 + 0.4)
        tx += 0.004 * env * math.sin(dt * 1.75)
        tz += 0.004 * env * math.sin(dt * 2.05 + 0.4)
        lens_d += 0.60 * env

    return (cx, cy, cz), (tx, ty, tz), lens_d


# ==============================================================================
# 6-BEAT CAMERA CHOREOGRAPHY
# ==============================================================================
# Beat 1 (1..210): Sweeping subsurface 3/4 tracking shot (Y ~ -2.35..-2.05m, lens=26mm, f/7.5)
_B1_CAM_KFS: list[tuple[float, Vec3]] = [
    (1.0, (0.18, -2.36, -0.58)),
    (80.0, (0.36, -2.24, -0.58)),
    (145.0, (0.54, -2.12, -0.56)),
    (210.0, (0.64, -2.05, -0.56)),
]
_B1_TGT_KFS: list[tuple[float, Vec3]] = [
    (1.0, (0.68, 0.04, -0.80)),
    (80.0, (0.88, 0.04, -0.82)),
    (145.0, (1.08, 0.04, -0.83)),
    (210.0, (1.16, 0.04, -0.83)),
]

# Beat 2 (211..540, hard cut @ 211): Side-angle muzzle (-0.48) -> entry (0.0) -> density wave (0.48)
_B2_CAM_KFS: list[tuple[float, Vec3]] = [
    (211.0, (-0.26, -1.88, -0.48)),
    (300.0, (-0.12, -1.82, -0.49)),
    (420.0, (0.08, -1.78, -0.50)),
    (540.0, (0.20, -1.74, -0.50)),
]
_B2_TGT_KFS: list[tuple[float, Vec3]] = [
    (211.0, (-0.20, 0.00, -0.54)),
    (300.0, (-0.02, 0.00, -0.54)),
    (420.0, (0.16, 0.00, -0.54)),
    (540.0, (0.26, 0.00, -0.54)),
]

# Beat 3-4 (541..1260, hard cut @ 541, SMOOTH transition across 900->901):
# Beat 3 (541..900): Action-locked macro close-up alongside nose + vapor bubble (cam_y ~ -0.58m, lens=35mm)
# Beat 4 (901..1260): Smooth pull-back + subtle orbit (cam_y ~ -0.78..-0.95m) for cutaway + 8-shard burst
_B34_CAM_OFFSET_KFS: list[tuple[float, Vec3]] = [
    (541.0, (-0.11, -0.58, 0.03)),
    (720.0, (-0.10, -0.58, 0.03)),
    (865.0, (-0.10, -0.59, 0.03)),
    (901.0, (-0.12, -0.68, 0.04)),
    (945.0, (-0.14, -0.78, 0.05)),
    (1055.0, (0.02, -0.88, 0.06)),
    (1160.0, (0.12, -0.92, 0.07)),
    (1260.0, (0.18, -0.95, 0.08)),
]
_B34_TGT_OFFSET_KFS: list[tuple[float, Vec3]] = [
    (541.0, (-0.03, 0.00, 0.02)),
    (865.0, (-0.03, 0.00, 0.02)),
    (945.0, (-0.01, 0.00, 0.01)),
    (1055.0, (0.00, 0.00, 0.02)),
    (1260.0, (0.01, 0.00, 0.02)),
]
_B34_LENS_KFS: list[tuple[float, float]] = [
    (541.0, 35.0),
    (865.0, 35.0),
    (945.0, 32.0),
    (1055.0, 30.0),
    (1260.0, 28.0),
]
_B34_FSTOP_KFS: list[tuple[float, float]] = [
    (541.0, 4.2),
    (580.0, 4.2),
    (610.0, 7.5),
    (1260.0, 7.5),
]

# Beat 5 (1261..1560, hard cut @ 1261): Clean studio blueprint camera (cam_y ~ -2.55m, lens=28mm, f/8.0)
_B5_CAM_KFS: list[tuple[float, Vec3]] = [
    (1261.0, (0.76, -2.62, -0.80)),
    (1410.0, (0.94, -2.59, -0.80)),
    (1485.0, (1.04, -2.57, -0.80)),
    (1560.0, (1.10, -2.55, -0.80)),
]
_B5_TGT_KFS: list[tuple[float, Vec3]] = [
    (1261.0, (0.76, 0.00, -0.80)),
    (1410.0, (0.94, 0.00, -0.80)),
    (1485.0, (1.04, 0.00, -0.80)),
    (1560.0, (1.10, 0.00, -0.80)),
]

# Beat 6 (1561..1800, hard cut @ 1561): Hero 3/4 wide-medium pull-back (cam_y ~ -2.45m, lens=25mm, f/7.5)
_B6_CAM_KFS: list[tuple[float, Vec3]] = [
    (1561.0, (0.50, -2.38, -0.78)),
    (1680.0, (0.46, -2.45, -0.76)),
    (1800.0, (0.42, -2.52, -0.74)),
]
_B6_TGT_KFS: list[tuple[float, Vec3]] = [
    (1561.0, (1.16, 0.08, -1.00)),
    (1680.0, (1.20, 0.08, -1.00)),
    (1800.0, (1.24, 0.08, -0.98)),
]


def camera_state_at(f: float, include_motion_fx: bool = True) -> dict[str, object]:
    """Deterministic pure-Python camera state evaluator for frame `f` in `1..1800`."""
    f = C.clamp(float(f), 1.0, float(C.TOTAL_FRAMES))

    if f <= 210.0:
        cam_pos = _eval_piecewise_vec3(_B1_CAM_KFS, f)
        tgt_pos = _eval_piecewise_vec3(_B1_TGT_KFS, f)
        lens_mm = 26.0
        fstop = 7.5
    elif f <= 540.0:
        cam_pos = _eval_piecewise_vec3(_B2_CAM_KFS, f)
        tgt_pos = _eval_piecewise_vec3(_B2_TGT_KFS, f)
        lens_mm = 30.0
        fstop = 7.5
    elif f <= 1260.0:
        bx = C.supersonic_x(f)
        bz = C.supersonic_z(f)
        anchor = (bx, 0.0, bz)
        cam_pos = _add3(anchor, _eval_piecewise_vec3(_B34_CAM_OFFSET_KFS, f))
        tgt_pos = _add3(anchor, _eval_piecewise_vec3(_B34_TGT_OFFSET_KFS, f))
        lens_mm = _eval_piecewise_scalar(_B34_LENS_KFS, f)
        fstop = _eval_piecewise_scalar(_B34_FSTOP_KFS, f)
    elif f <= 1560.0:
        cam_pos = _eval_piecewise_vec3(_B5_CAM_KFS, f)
        tgt_pos = _eval_piecewise_vec3(_B5_TGT_KFS, f)
        lens_mm = 28.0
        fstop = 8.0
    else:
        cam_pos = _eval_piecewise_vec3(_B6_CAM_KFS, f)
        tgt_pos = _eval_piecewise_vec3(_B6_TGT_KFS, f)
        lens_mm = 25.0
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
# BLENDER 5.3 SCENE BUILDER PUBLIC API
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
    cam_data.lens = 26.0

    cam_data.dof.use_dof = True
    cam_data.dof.focus_object = None
    cam_data.dof.aperture_blades = 7
    cam_data.dof.aperture_ratio = 1.0
    cam_data.dof.aperture_fstop = 7.5
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
        for f_chk in (1, 135, 211, 300, 541, 720, 901, 1055, 1261, 1410, 1561, 1750):
            st = camera_state_at(float(f_chk), include_motion_fx=True)
            print(
                f"f={f_chk:04d} cam={tuple(round(v, 2) for v in st['cam_pos'])} "
                f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f} focus={st['focus_dist']:.2f}m"
            )
