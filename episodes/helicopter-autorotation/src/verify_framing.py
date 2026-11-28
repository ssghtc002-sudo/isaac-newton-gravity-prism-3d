"""
verify_framing.py — Pre-flight Mathematical 9:16 Portrait Framing, Subtitle Safe-Zone & Gimbal QA
for Episode #5: "Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock" (60 fps, 1800 frames).

Pure-Python (runs cleanly outside Blender via `python3 episodes/helicopter-autorotation/src/verify_framing.py`).
"""
from __future__ import annotations

import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import heli_contract as C  # noqa: E402
from cinema_camera import HARD_CUT_FRAMES, camera_state_at, project_ndc  # noqa: E402
from labels_3d import _target_world_pos  # noqa: E402

Vec3 = tuple[float, float, float]
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
    print("=== Helicopter Autorotation Ep#5 — 9:16 Portrait Framing & Safe-Zone QA (60fps, 1800f) ===")

    # 1. BEAT 1 (1..240): 1:1 Rescue Cockpit — Pilot, Co-Pilot, ENG OUT Caution & Collective Slam
    print("\n[BEAT 1 — 1:1 Rescue Cockpit Engine Failure & Collective Slam (1..240)]")
    px, py, _ = C.CREW["pilot_pos"]
    cx, cy, _ = C.CREW["copilot_pos"]
    for f in (1, 14, 42, 120, 240):
        st = camera_state_at(float(f), include_motion_fx=True)
        focus_err = abs(st["focus_dist"] - _dist3(st["cam_pos"], st["tgt_pos"]))
        dof_ok = abs(st["fstop"] - 7.5) < 1e-3 and focus_err < 1e-4
        if not dof_ok:
            errors.append(f"B1 f={f}: fstop={st['fstop']:.2f} focus_err={focus_err:.5f}")
        print(
            f"  f={f:03d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f} dof={'PASS' if dof_ok else 'FAIL'}"
        )
        _check_point(errors, f"B1_f{f:03d}_PilotEyes", (px, py, float(C.CREW["pilot_eye_z"])), st, 0.85, -0.25, 0.50)
        _check_point(errors, f"B1_f{f:03d}_CoPilotEyes", (cx, cy, float(C.CREW["copilot_eye_z"])), st, 0.85, -0.25, 0.50)
        _check_point(errors, f"B1_f{f:03d}_EngOutPanel", tuple(C.STAGE["cockpit_caution_pos"]), st, 0.70, -0.42, 0.30)
        _check_point(errors, f"B1_f{f:03d}_CollectiveGrip", (0.10, -0.32, 0.80), st, 0.75, -0.48, 0.20)

    # 2. BEAT 2 (241..570): Dorsal Macro X-Ray on Seized Shaft & Freewheeling Sprag Clutch
    print("\n[BEAT 2 — Macro X-Ray: Seized Turbine Shaft & Freewheeling Sprag Clutch (241..570)]")
    for f in (241, 295, 340, 415, 570):
        st = camera_state_at(float(f), include_motion_fx=True)
        hpos, _ = C.heli_pose_at(float(f))
        print(
            f"  f={f:03d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f}"
        )
        _check_point(errors, f"B2_f{f:03d}_SpragClutch", tuple(C.STAGE["b2_clutch_world"]), st, 0.65, -0.35, 0.45)
        _check_point(errors, f"B2_f{f:03d}_TurbineShaft", (0.00, 0.24, 2.13), st, 0.75, -0.35, 0.45)
        _check_point(errors, f"B2_f{f:03d}_HeliCore", hpos, st, 0.75, -0.45, 0.35)

    # 3. BEAT 3 (571..960): Golden-Hour Autorotation Glide & Upward Emerald Airflow
    print("\n[BEAT 3 — Upward Airflow Windmill & Kinetic Flywheel Spin (571..960)]")
    for f in (571, 620, 740, 865, 960):
        st = camera_state_at(float(f), include_motion_fx=True)
        hpos, _ = C.heli_pose_at(float(f))
        print(
            f"  f={f:03d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f}"
        )
        _check_point(errors, f"B3_f{f:03d}_HeliCore", hpos, st, 0.75, -0.35, 0.45)
        _check_point(errors, f"B3_f{f:03d}_RotorHub", (hpos[0], hpos[1], hpos[2] + 0.44), st, 0.80, -0.25, 0.55)

    # 4. BEAT 4 (961..1290): Side-by-Side Sculpted 3D Maple Seed (Left) vs Rescue Rotor (Right)
    print("\n[BEAT 4 — 3D Winged Maple Seed vs Rescue Rotor Comparison (961..1290)]")
    for f in (961, 1005, 1125, 1290):
        st = camera_state_at(float(f), include_motion_fx=True)
        mpos, _ = C.maple_seed_pose_at(float(f))
        hpos, _ = C.heli_pose_at(float(f))
        print(
            f"  f={f:04d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f}"
        )
        _check_point(errors, f"B4_f{f:04d}_MapleSeed", mpos, st, 0.85, -0.35, 0.50)
        _check_point(errors, f"B4_f{f:04d}_RescueHeli", hpos, st, 0.85, -0.35, 0.50)

    # 5A. BEAT 5A (1291..1450): Label-Free 1:1 Cockpit Final Flare Pull (f/3.6 Bokeh!)
    print("\n[BEAT 5A — 1:1 Cockpit Final Flare Pull Close-Up (1291..1450, f/3.6 Bokeh)]")
    for f in (1291, 1350, 1400, C.B5_COCKPIT_END):
        st = camera_state_at(float(f), include_motion_fx=True)
        fs_ok = abs(st["fstop"] - 3.6) < 1e-2
        if not fs_ok:
            errors.append(f"B5A f={f}: expected f/3.6 bokeh, got f/{st['fstop']:.2f}")
        print(
            f"  f={f:04d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f} bokeh={'PASS' if fs_ok else 'FAIL'}"
        )
        _check_point(errors, f"B5A_f{f:04d}_PilotEyes", (px, py, float(C.CREW["pilot_eye_z"])), st, 0.85, -0.25, 0.50)
        _check_point(errors, f"B5A_f{f:04d}_CoPilotEyes", (cx, cy, float(C.CREW["copilot_eye_z"])), st, 0.90, -0.25, 0.50)
        _check_point(errors, f"B5A_f{f:04d}_CollectivePull", (0.10, -0.30, 0.94), st, 0.75, -0.48, 0.25)

    # 5B. BEAT 5B (1451..1580): Crisp f/8.0 Exterior 30-Foot Nose-Up Flare over Helipad
    print("\n[BEAT 5B — Exterior 30-Foot Nose-Up Flare over Rooftop Helipad (1451..1580)]")
    for f in (C.B5_EXTERIOR_START, 1468, 1525, 1580):
        st = camera_state_at(float(f), include_motion_fx=True)
        hpos, _ = C.heli_pose_at(float(f))
        fs_ok = abs(st["fstop"] - 8.0) < 1e-2
        if not fs_ok:
            errors.append(f"B5B f={f}: expected f/8.0, got f/{st['fstop']:.2f}")
        print(
            f"  f={f:04d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f}"
        )
        _check_point(errors, f"B5B_f{f:04d}_HeliFlare", hpos, st, 0.80, -0.25, 0.55)
        _check_point(errors, f"B5B_f{f:04d}_HelipadH", tuple(C.STAGE["helipad_center"]), st, 0.85, -0.50, 0.20)

    # 6. BEAT 6 (1581..1800): Smooth Rooftop Helipad Touchdown Payoff
    print("\n[BEAT 6 — Smooth Zero-Engine Rooftop Helipad Touchdown (1581..1800)]")
    for f in (1581, 1635, 1685, 1800):
        st = camera_state_at(float(f), include_motion_fx=True)
        hpos, _ = C.heli_pose_at(float(f))
        print(
            f"  f={f:04d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f}"
        )
        _check_point(errors, f"B6_f{f:04d}_HeliTouchdown", hpos, st, 0.75, -0.30, 0.45)
        _check_point(errors, f"B6_f{f:04d}_HelipadH", tuple(C.STAGE["helipad_center"]), st, 0.80, -0.45, 0.25)

    # Flush skid-contact verification at touchdown (f=1685 & f=1800)
    h_td, _ = C.heli_pose_at(1685.0)
    skid_world_z = h_td[2] + C.HELI_SKID_BOTTOM_LOCAL_Z
    skid_err = abs(skid_world_z - float(C.STAGE["helipad_z"]))
    if skid_err > 0.002:
        errors.append(f"Touchdown skid contact mismatch: skid_z={skid_world_z:.4f}m vs helipad_z={C.STAGE['helipad_z']:.4f}m")
    else:
        print(f"  [PASS] Touchdown Flush Skid Contact: skid_z={skid_world_z:.3f}m == helipad_z={C.STAGE['helipad_z']:.3f}m (err={skid_err*1000:.2f}mm)")

    # 7. ALL 6 C.LABELS SAFE-ZONE AUDIT (NDC Y >= -0.45)
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
        print(f"\nFAILED ({len(errors)} error(s)):")
        for e in errors:
            print("  -", e)
        return 1

    print(
        "\nRESULT: 100% PASSED — All 6 Beats + Rescue Crew/Cockpit/Sprag Clutch/Upward Airflow/Maple Seed/Helipad "
        "+ All 6 C.LABELS + 1800-Frame Gimbal & Continuity Audits PASSED!"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
