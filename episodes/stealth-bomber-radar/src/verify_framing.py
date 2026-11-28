"""
Pre-flight Mathematical 9:16 Portrait Framing, Subtitle Safe-Zone & Gimbal QA
for Episode #4: "How a 172-Foot Stealth Bomber Looks Like a Bumblebee on Radar" (60 fps, 1800 frames).

Pure-Python (runs cleanly outside Blender via `python3 episodes/stealth-bomber-radar/src/verify_framing.py`).
"""
from __future__ import annotations

import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import stealth_contract as C  # noqa: E402
from cinema_camera import HARD_CUT_FRAMES, camera_state_at, project_ndc  # noqa: E402
from labels_3d import _target_world_pos  # noqa: E402

Vec3 = tuple[float, float, float]

SUBTITLE_ZONE_NDC_Y = -0.55
LABEL_MIN_NDC_Y = -0.45


def _sub3(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot3(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm3(a: Vec3) -> Vec3:
    length = math.sqrt(_dot3(a, a)) or 1e-9
    return (a[0] / length, a[1] / length, a[2] / length)


def _dist3(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((u - v) ** 2 for u, v in zip(a, b)))


def _check_point(
    errors: list[str],
    tag: str,
    pt_w: Vec3,
    st: dict,
    max_abs_x: float,
    min_y: float,
    max_y: float,
    min_depth: float = 0.25,
) -> tuple[float, float, float]:
    nx, ny, z = project_ndc(pt_w, st["cam_pos"], st["tgt_pos"], st["lens_mm"])
    ok = (
        math.isfinite(nx)
        and math.isfinite(ny)
        and z > min_depth
        and abs(nx) <= max_abs_x
        and min_y <= ny <= max_y
    )
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] {tag:28s} -> NDC=({nx:+.2f}, {ny:+.2f}) depth={z:.2f}m "
        f"(|X|<={max_abs_x:.2f}, Y in [{min_y:+.2f}, {max_y:+.2f}])"
    )
    if not ok:
        errors.append(f"{tag}: NDC=({nx:+.2f}, {ny:+.2f}) depth={z:.2f}m out of range")
    return (nx, ny, z)


def main() -> int:
    errors: list[str] = []
    print("=== Stealth Bomber Radar Ep#4 — 9:16 Portrait Framing & Safe-Zone QA (60fps, 1800f) ===")

    # 1. BEAT 1 DENSE AUDIT (1..220)
    print("\n[BEAT 1 — Hook: 172-Ft Stealth Bomber Glides Over Ground Radar Dish (1..220)]")
    for f in (1, 18, 68, 135, 220):
        st = camera_state_at(float(f), include_motion_fx=True)
        bpos, _ = C.bomber_pose_at(float(f))
        focus_err = abs(st["focus_dist"] - _dist3(st["cam_pos"], st["tgt_pos"]))
        dof_ok = abs(st["fstop"] - 7.5) < 1e-3 and focus_err < 1e-4
        if not dof_ok:
            errors.append(f"B1 f={f}: fstop={st['fstop']:.2f} focus_err={focus_err:.5f}")
        print(
            f"  f={f:03d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f} dof={'PASS' if dof_ok else 'FAIL'}"
        )
        _check_point(errors, f"B1_f{f:03d}_BomberCore", bpos, st, 0.85, -0.35, 0.65)
        _check_point(errors, f"B1_f{f:03d}_RadarDish", tuple(C.STAGE["radar_dish_b1"]), st, 0.90, -0.45, 0.45)
        _check_point(errors, f"B1_f{f:03d}_CmdConsole", tuple(C.STAGE["b1_console_scope_pos"]), st, 0.85, -0.45, 0.25)

    # 2. BEAT 2 DENSE AUDIT (221..540)
    print("\n[BEAT 2 — Radar Contrast: Conventional Jet vs Faceted Stealth Wing (221..540)]")
    for f in (221, 265, 318, 410, 540):
        st = camera_state_at(float(f), include_motion_fx=True)
        cpos, _ = C.conv_jet_pose_at(float(f))
        bpos, _ = C.bomber_pose_at(float(f))
        print(
            f"  f={f:03d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f}"
        )
        _check_point(errors, f"B2_f{f:03d}_ConvJet", cpos, st, 0.85, -0.10, 0.65)
        _check_point(errors, f"B2_f{f:03d}_StealthWing", bpos, st, 0.85, -0.45, 0.35)
        _check_point(errors, f"B2_f{f:03d}_RadarEmitter", tuple(C.STAGE["radar_dish_b2"]), st, 0.92, -0.40, 0.40)

    # 3. BEAT 3 DENSE AUDIT (541..960)
    print("\n[BEAT 3 — Macro X-Ray: Serpentine S-Shaped Ducts & Spinning Turbines (541..960)]")
    cx, cy, cz = C.STAGE["macro_center"]
    for f in (541, 575, 615, 655, 785, 960):
        st = camera_state_at(float(f), include_motion_fx=True)
        if f <= 582:
            fs_ok = abs(st["fstop"] - 3.8) < 1e-2
        elif f >= 606:
            fs_ok = abs(st["fstop"] - 7.5) < 1e-2
        else:
            fs_ok = 3.8 <= st["fstop"] <= 7.5
        if not fs_ok:
            errors.append(f"B3 f={f}: unexpected fstop={st['fstop']:.2f}")
        print(
            f"  f={f:03d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f} dof={'PASS' if fs_ok else 'FAIL'}"
        )
        _check_point(errors, f"B3_f{f:03d}_IntakeLip", (cx - 0.14, cy - 0.10, cz + 0.08), st, 0.85, -0.40, 0.55)
        _check_point(errors, f"B3_f{f:03d}_SDuctCore", (cx - 0.08, cy, cz + 0.04), st, 0.85, -0.40, 0.55)
        _check_point(errors, f"B3_f{f:03d}_TurbineFan", (cx - 0.14, cy + 0.18, cz + 0.032), st, 0.85, -0.42, 0.55)

    # 4. BEAT 4 DENSE AUDIT (961..1320)
    print("\n[BEAT 4 — RAM Honeycomb Absorption & 0.001 m^2 Bumblebee RCS Shrink (961..1320)]")
    for f in (961, 995, 1075, 1160, 1245, 1320):
        st = camera_state_at(float(f), include_motion_fx=True)
        print(
            f"  f={f:04d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f}"
        )
        _check_point(errors, f"B4_f{f:04d}_RAMTile", tuple(C.STAGE["ram_tile_pos"]), st, 0.85, -0.40, 0.55)
        _check_point(errors, f"B4_f{f:04d}_RCSBee", tuple(C.STAGE["rcs_bee_pos"]), st, 0.85, -0.40, 0.55)

    # 5. BEAT 5 DENSE AUDIT (1321..1560)
    print("\n[BEAT 5 — Main-Scene Daylight Explainer: Real Conv Jet 90-Deg vs Real B-2 33-Deg (1321..1560)]")
    for f in (1321, 1365, 1445, 1560):
        st = camera_state_at(float(f), include_motion_fx=True)
        cpos, _ = C.conv_jet_pose_at(float(f))
        bpos, _ = C.bomber_pose_at(float(f))
        print(
            f"  f={f:04d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f}"
        )
        _check_point(errors, f"B5_f{f:04d}_RealConvJet", cpos, st, 0.85, -0.10, 0.60)
        _check_point(errors, f"B5_f{f:04d}_RealStealthB2", bpos, st, 0.85, -0.45, 0.20)
        _check_point(errors, f"B5_f{f:04d}_RadarDish", tuple(C.STAGE["radar_dish_b5"]), st, 0.92, -0.40, 0.40)

    # 6. BEAT 6 DENSE AUDIT (1561..1800)
    print("\n[BEAT 6 — Enemy Radar Command Room Payoff + Sunlit B-2 Outside Window (1561..1800)]")
    for f in (1561, 1600, 1695, 1800):
        st = camera_state_at(float(f), include_motion_fx=True)
        bpos, _ = C.bomber_pose_at(float(f))
        print(
            f"  f={f:04d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f}"
        )
        _check_point(errors, f"B6_f{f:04d}_BomberDaylight", bpos, st, 0.85, 0.00, 0.65)
        _check_point(errors, f"B6_f{f:04d}_CmdRadarScope", tuple(C.STAGE["b6_scope_pos"]), st, 0.85, -0.45, 0.25)

    # 7. ALL C.LABELS ANCHOR SAFE-ZONE AUDIT (NDC Y >= -0.45)
    print("\n[3D LABELS SAFE-ZONE AUDIT (C.LABELS, NDC Y >= -0.45)]")
    for key, _text, (w0, w1) in C.LABELS:
        for f in (w0, (w0 + w1) // 2, w1):
            st = camera_state_at(float(f), include_motion_fx=True)
            pt = _target_world_pos(key, float(f))
            _check_point(errors, f"LBL_{key}_f{f:04d}", pt, st, 0.88, LABEL_MIN_NDC_Y, 0.85, min_depth=0.25)

    # 8. FULL 1800-FRAME ANTI-GIMBAL-FLIP & INTRA-BEAT CONTINUITY AUDIT
    max_abs_fwd_z = 0.0
    min_dist = 999.0
    max_step_cam = 0.0
    max_step_tgt = 0.0
    max_step_lens = 0.0
    prev_st = None
    cut_set = set(HARD_CUT_FRAMES)
    cut_jumps: dict[int, float] = {}

    for f in range(1, C.TOTAL_FRAMES + 1):
        st = camera_state_at(float(f), include_motion_fx=True)
        cp, tp = st["cam_pos"], st["tgt_pos"]
        ln, fs, fd = st["lens_mm"], st["fstop"], st["focus_dist"]
        d = _dist3(cp, tp)
        fwd = _norm3(_sub3(tp, cp))
        max_abs_fwd_z = max(max_abs_fwd_z, abs(fwd[2]))
        min_dist = min(min_dist, d)

        vals_ok = (
            all(math.isfinite(v) for v in (*cp, *tp, ln, fs, fd))
            and d > 0.25
            and 20.0 <= ln <= 85.0
            and 1.8 <= fs <= 11.0
            and fd > 0.25
        )
        if not vals_ok:
            errors.append(f"f={f}: invalid camera state dist={d:.3f} lens={ln:.2f} f/{fs:.2f} fd={fd:.2f}")

        if prev_st is not None:
            d_cam = _dist3(cp, prev_st["cam_pos"])
            d_tgt = _dist3(tp, prev_st["tgt_pos"])
            d_lens = abs(ln - prev_st["lens_mm"])
            if f in cut_set:
                cut_jumps[f] = d_cam
            else:
                max_step_cam = max(max_step_cam, d_cam)
                max_step_tgt = max(max_step_tgt, d_tgt)
                max_step_lens = max(max_step_lens, d_lens)
        prev_st = st

    gimbal_ok = max_abs_fwd_z < 0.85 and min_dist > 0.25
    cont_ok = max_step_cam <= 0.35 and max_step_tgt <= 0.35 and max_step_lens <= 1.0
    cuts_ok = len(cut_jumps) == len(HARD_CUT_FRAMES) and all(v > 0.25 for v in cut_jumps.values())
    if not gimbal_ok:
        errors.append(f"Gimbal/dist fail: max_|fwd.z|={max_abs_fwd_z:.3f}, min_dist={min_dist:.3f}")
    if not cont_ok:
        errors.append(f"Intra-beat continuity spike: dCam={max_step_cam:.3f}, dTgt={max_step_tgt:.3f}, dLens={max_step_lens:.2f}")
    if not cuts_ok:
        errors.append(f"Hard-cut audit fail: cut_jumps={cut_jumps}")

    print(
        f"\n[1800-FRAME GIMBAL & INTRA-BEAT CONTINUITY AUDIT]\n"
        f"  [{'PASS' if gimbal_ok else 'FAIL'}] min_target_dist={min_dist:.3f}m (> 0.25m), "
        f"max_|fwd.z|={max_abs_fwd_z:.3f} (< 0.85 guaranteed zero TRACK_TO flip!)\n"
        f"  [{'PASS' if cont_ok else 'FAIL'}] intra-beat max_step_cam={max_step_cam:.3f}m/f, "
        f"max_step_tgt={max_step_tgt:.3f}m/f, max_step_lens={max_step_lens:.2f}mm/f\n"
        f"  [{'PASS' if cuts_ok else 'FAIL'}] hard cuts verified @ {HARD_CUT_FRAMES}: "
        + ", ".join(f"f{k}={v:.2f}m" for k, v in sorted(cut_jumps.items()))
    )

    if errors:
        print(f"\nRESULT: FAILED — {len(errors)} violation(s):")
        for e in errors:
            print(f"  * {e}")
        return 1

    print(
        "\nRESULT: 100% PASSED — All 6 Beats + Stealth Bomber/Conv Jet/S-Ducts/RAM/Bumblebee/Scope + "
        "All 7 C.LABELS + 1800-Frame Gimbal & Continuity Audits PASSED!"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
