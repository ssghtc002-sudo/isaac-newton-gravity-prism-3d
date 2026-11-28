"""
Pre-flight Mathematical 9:16 Portrait Framing, Subtitle Safe-Zone & Gimbal QA
for Episode #16: "Why Jumping in a Falling Elevator WON'T Save You" (60 fps, 1800 frames).

Pure-Python (runs cleanly outside Blender via `python episodes/falling-elevator/src/verify_framing.py`).
Imports `elevator_contract` and `cinema_camera` directly so framing QA and Blender
camera generation share 100% identical trajectory, lens, and DOF math.
"""
from __future__ import annotations

import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import elevator_contract as C  # noqa: E402
from cinema_camera import HARD_CUT_FRAMES, camera_state_at, project_ndc  # noqa: E402

Vec3 = tuple[float, float, float]

SUBTITLE_ZONE_NDC_Y = -0.55
LABEL_MIN_NDC_Y = -0.45


def _add3(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub3(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot3(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm3(a: Vec3) -> Vec3:
    length = math.sqrt(_dot3(a, a)) or 1e-9
    return (a[0] / length, a[1] / length, a[2] / length)


def _dist3(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((u - v) ** 2 for u, v in zip(a, b)))


def _cab_pt(f: float, local_pt: Vec3) -> Vec3:
    return _add3(C.cab_world_location(f), local_pt)


def _label_anchor_world(key: str, f: float) -> Vec3:
    """Contract-aligned world-space anchor zone for each 3D engineering label in C.LABELS."""
    px, py = C.PASSENGER["stand_xy"]
    if key == "speed_cab":
        return _cab_pt(f, C.CAB["floor_led"])
    if key == "cables":
        return _cab_pt(f, (0.0, 0.0, float(C.CAB["hitch_z"]) + 0.25))
    if key == "speed_fall":
        return _cab_pt(f, C.CAB["floor_led"])
    if key == "jump_up":
        return _cab_pt(f, (px, py, 1.42 + C.passenger_jump_offset_z(f)))
    if key == "net_hit":
        return _cab_pt(f, (0.15, -0.42, 2.08))
    if key == "impact_g":
        return _cab_pt(f, (0.25, -0.18, 1.42))
    if key == "leg_load":
        return _cab_pt(f, (px, py, 0.72))
    if key == "safety_brakes":
        return _cab_pt(f, (0.92, 0.0, float(C.CAB["clamp_z"])))
    if key == "flat_pose":
        fp = C.PASSENGER["flat_pelvis"]
        return _cab_pt(f, (fp[0], fp[1], fp[2] + 0.22))
    if key == "force_spread":
        fp = C.PASSENGER["flat_pelvis"]
        return _cab_pt(f, (fp[0], fp[1], fp[2] + 0.30))
    raise KeyError(f"Unknown label key: {key}")


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
        f"  [{status}] {tag:24s} -> NDC=({nx:+.2f}, {ny:+.2f}) depth={z:.2f}m "
        f"(|X|<={max_abs_x:.2f}, Y in [{min_y:+.2f}, {max_y:+.2f}])"
    )
    if not ok:
        errors.append(f"{tag}: NDC=({nx:+.2f}, {ny:+.2f}) depth={z:.2f}m out of range")
    return (nx, ny, z)


def main() -> int:
    errors: list[str] = []
    px, py = C.PASSENGER["stand_xy"]
    print("=== Falling Elevator Ep#16 — 9:16 Portrait Framing & Safe-Zone QA (60fps, 1800f) ===")

    # ------------------------------------------------------------------
    # 1. BEAT 1 DENSE AUDIT (1..312)
    # ------------------------------------------------------------------
    print("\n[BEAT 1 — Glass Door Hook, Floor LED & Crouch (Dense Audit)]")
    for f in (1, 60, 115, 190, 240, 275, 300, 312):
        st = camera_state_at(float(f), include_motion_fx=True)
        cam_local = _sub3(st["cam_pos"], C.cab_world_location(f))
        out_doors = cam_local[1] < -float(C.CAB["half_d"]) - 0.50
        if not out_doors:
            errors.append(f"B1 f={f}: cam_local_y={cam_local[1]:.3f} too close to glass doors")
        print(
            f"  f={f:03d} | cam_local=({cam_local[0]:+.2f},{cam_local[1]:+.2f},{cam_local[2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f} outside_doors={'PASS' if out_doors else 'FAIL'}"
        )
        _check_point(errors, f"B1_f{f:03d}_HeadCrown", _cab_pt(f, (px, py, 1.78)), st, 0.75, -0.10, 0.82)
        _check_point(errors, f"B1_f{f:03d}_EyeMid", _cab_pt(f, (px, py, float(C.PASSENGER["eye_z"]))), st, 0.75, -0.10, 0.82)
        _check_point(errors, f"B1_f{f:03d}_Knees", _cab_pt(f, (px, py, 0.45)), st, 0.75, -0.45, 0.50)
        _check_point(errors, f"B1_f{f:03d}_FeetFloor", _cab_pt(f, (px, py, 0.00)), st, 0.75, -0.45, 0.20)
        _check_point(errors, f"B1_f{f:03d}_FloorLED", _cab_pt(f, C.CAB["floor_led"]), st, 0.80, -0.20, 0.85)

    # Verify Beat 1 cable-tremor shake (f=240..312)
    trem_disp_mm = _dist3(
        camera_state_at(295.0, include_motion_fx=True)["cam_pos"],
        camera_state_at(295.0, include_motion_fx=False)["cam_pos"],
    ) * 1000.0
    trem_ok = 2.0 <= trem_disp_mm <= 20.0
    print(f"  [{'PASS' if trem_ok else 'FAIL'}] Beat 1 cable-tremor @ f=295: disp={trem_disp_mm:.1f}mm")
    if not trem_ok:
        errors.append(f"Beat 1 cable-tremor disp={trem_disp_mm:.2f}mm out of [2, 20]mm")

    # ------------------------------------------------------------------
    # 2. BEAT 2 AUDIT (313..600)
    # ------------------------------------------------------------------
    print("\n[BEAT 2 — External Cutaway, Broken Hoist Cables & 50 mph Plunge]")
    for f in (313, 325, 420, 535, 595):
        st = camera_state_at(float(f), include_motion_fx=True)
        cam_local = _sub3(st["cam_pos"], C.cab_world_location(f))
        out_cab = cam_local[0] < -float(C.CAB["half_w"]) - 0.30
        if not out_cab:
            errors.append(f"B2 f={f}: cam_local_x={cam_local[0]:.3f} inside cab")
        print(
            f"  f={f:03d} | cam_local=({cam_local[0]:+.2f},{cam_local[1]:+.2f},{cam_local[2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm outside_cab={'PASS' if out_cab else 'FAIL'}"
        )
        _check_point(errors, f"B2_f{f:03d}_CablesTop", _cab_pt(f, (0.0, 0.0, 2.95)), st, 0.80, 0.10, 0.85)
        _check_point(errors, f"B2_f{f:03d}_Hitch", _cab_pt(f, (0.0, 0.0, float(C.CAB["hitch_z"]))), st, 0.80, 0.00, 0.80)
        _check_point(errors, f"B2_f{f:03d}_CrouchHead", _cab_pt(f, (px, py, 1.35)), st, 0.80, -0.35, 0.60)
        _check_point(errors, f"B2_f{f:03d}_FeetFloor", _cab_pt(f, (px, py, 0.00)), st, 0.80, -0.52, 0.25)
        _check_point(errors, f"B2_f{f:03d}_RailL", _cab_pt(f, (-float(C.CAB["rail_x"]), 0.0, 1.20)), st, 0.85, -0.45, 0.60)
        _check_point(errors, f"B2_f{f:03d}_RailR", _cab_pt(f, (float(C.CAB["rail_x"]), 0.0, 1.20)), st, 0.85, -0.45, 0.60)

    # ------------------------------------------------------------------
    # 3. BEAT 3 DENSE AUDIT (601..900)
    # ------------------------------------------------------------------
    print("\n[BEAT 3 — Myth Jump & Ceiling Strike (Dense Audit)]")
    for f in (601, 635, 665, 705, 750, 775, 830, 885, 900):
        st = camera_state_at(float(f), include_motion_fx=True)
        cam_local = _sub3(st["cam_pos"], C.cab_world_location(f))
        j_off = C.passenger_jump_offset_z(float(f))
        out_cab = cam_local[0] > float(C.CAB["half_w"]) + 0.50
        if not out_cab:
            errors.append(f"B3 f={f}: cam_local_x={cam_local[0]:.3f} inside cab")
        print(f"  f={f:03d} | jump_z=+{j_off:.2f}m lens={st['lens_mm']:.1f}mm outside_cab={'PASS' if out_cab else 'FAIL'}")
        _check_point(errors, f"B3_f{f:03d}_Ceiling", _cab_pt(f, (px, py, float(C.CAB["ceiling_z"]))), st, 0.75, -0.10, 0.78)
        _check_point(errors, f"B3_f{f:03d}_JumpHead", _cab_pt(f, (px, py, 1.78 + j_off)), st, 0.75, -0.10, 0.78)
        _check_point(errors, f"B3_f{f:03d}_Floor", _cab_pt(f, (px, py, 0.00)), st, 0.75, -0.46, 0.10)

    rattle_mm = _dist3(
        camera_state_at(754.0, include_motion_fx=True)["cam_pos"],
        camera_state_at(754.0, include_motion_fx=False)["cam_pos"],
    ) * 1000.0
    rattle_ok = rattle_mm >= 4.0
    print(f"  [{'PASS' if rattle_ok else 'FAIL'}] Beat 3 ceiling-hit rattle @ f=754: disp={rattle_mm:.1f}mm")
    if not rattle_ok:
        errors.append(f"Beat 3 ceiling rattle too weak ({rattle_mm:.1f}mm)")

    # ------------------------------------------------------------------
    # 4. BEAT 4 DENSE AUDIT (901..1236)
    # ------------------------------------------------------------------
    print("\n[BEAT 4 — 28G Pit Impact & X-Ray Skeleton Shockwave (Dense Audit)]")
    for f in (901, 920, 940, 962, 985, 1060, 1150, 1225):
        st = camera_state_at(float(f), include_motion_fx=True)
        print(f"  f={f:04d} | lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f} focus_dist={st['focus_dist']:.2f}m")
        _check_point(errors, f"B4_f{f:04d}_Skull", _cab_pt(f, (px, py, 1.74)), st, 0.75, -0.20, 0.78)
        _check_point(errors, f"B4_f{f:04d}_SpineMid", _cab_pt(f, (px, py, 1.15)), st, 0.75, -0.30, 0.78)
        _check_point(errors, f"B4_f{f:04d}_Ankles", _cab_pt(f, (px, py, 0.05)), st, 0.75, -0.50, 0.40)
        _check_point(errors, f"B4_f{f:04d}_PitBuffer", _cab_pt(f, (float(C.CAB["buffer_x"]), 0.0, -0.26)), st, 0.75, -0.50, 0.20)

    slam_mm = _dist3(
        camera_state_at(950.0, include_motion_fx=True)["cam_pos"],
        camera_state_at(950.0, include_motion_fx=False)["cam_pos"],
    ) * 1000.0
    slam_ok = slam_mm >= 6.0
    print(f"  [{'PASS' if slam_ok else 'FAIL'}] Beat 4 pit-impact slam @ f=950: disp={slam_mm:.1f}mm")
    if not slam_ok:
        errors.append(f"Beat 4 impact slam too weak ({slam_mm:.1f}mm)")

    # ------------------------------------------------------------------
    # 5. BEAT 5a DENSE AUDIT (1237..1385)
    # ------------------------------------------------------------------
    print("\n[BEAT 5a — Steel Safety Rail Clamps & Braced Passenger (Dense Audit)]")
    for f in (1237, 1275, 1310, 1345, 1380):
        st = camera_state_at(float(f), include_motion_fx=True)
        print(f"  f={f:04d} | lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f} focus={st['focus_dist']:.2f}m")
        _check_point(errors, f"B5a_f{f:04d}_ClampR", _cab_pt(f, (0.92, 0.0, float(C.CAB["clamp_z"]))), st, 0.80, -0.50, 0.50)
        _check_point(errors, f"B5a_f{f:04d}_RailR", _cab_pt(f, (float(C.CAB["rail_x"]), 0.0, 0.45)), st, 0.80, -0.45, 0.65)
        _check_point(errors, f"B5a_f{f:04d}_Ankles", _cab_pt(f, (px, py, float(C.PASSENGER["ankle_z"]))), st, 0.80, -0.45, 0.50)
        _check_point(errors, f"B5a_f{f:04d}_Torso", _cab_pt(f, (px, py, 1.25)), st, 0.80, -0.20, 0.78)

    # ------------------------------------------------------------------
    # 6. BEAT 5b–6 DENSE AUDIT (1386..1800)
    # ------------------------------------------------------------------
    print("\n[BEAT 5b-6 — Flat-Back Survival Reveal & 12x Force Grid (Dense Audit)]")
    for f in (1386, 1430, 1465, 1545, 1620, 1700, 1785, 1800):
        st = camera_state_at(float(f), include_motion_fx=True)
        print(f"  f={f:04d} | lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f} focus={st['focus_dist']:.2f}m")
        _check_point(errors, f"B6_f{f:04d}_FlatHead", _cab_pt(f, C.PASSENGER["flat_head"]), st, 0.85, -0.45, 0.75)
        _check_point(errors, f"B6_f{f:04d}_FlatPelvis", _cab_pt(f, C.PASSENGER["flat_pelvis"]), st, 0.85, -0.45, 0.75)
        _check_point(errors, f"B6_f{f:04d}_FlatFeet", _cab_pt(f, C.PASSENGER["flat_feet"]), st, 0.85, -0.45, 0.75)
        _check_point(errors, f"B6_f{f:04d}_GridFL", _cab_pt(f, (-0.60, -0.48, 0.02)), st, 0.88, -0.48, 0.60)
        _check_point(errors, f"B6_f{f:04d}_GridFR", _cab_pt(f, (0.60, -0.48, 0.02)), st, 0.88, -0.48, 0.60)

    # ------------------------------------------------------------------
    # 7. ALL 10 C.LABELS ANCHOR SAFE-ZONE AUDIT (NDC Y >= -0.45)
    # ------------------------------------------------------------------
    print("\n[3D LABELS SAFE-ZONE AUDIT (C.LABELS, NDC Y >= -0.45)]")
    for key, text, (w0, w1) in C.LABELS:
        for f in (w0, (w0 + w1) // 2, w1):
            st = camera_state_at(float(f), include_motion_fx=True)
            pt = _label_anchor_world(key, float(f))
            _check_point(errors, f"LBL_{key}_f{f:04d}", pt, st, 0.85, LABEL_MIN_NDC_Y, 0.85, min_depth=0.25)

    # ------------------------------------------------------------------
    # 8. FULL 1800-FRAME ANTI-GIMBAL-FLIP & INTRA-BEAT CONTINUITY AUDIT
    # ------------------------------------------------------------------
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
    cuts_ok = len(cut_jumps) == len(HARD_CUT_FRAMES) and all(v > 0.35 for v in cut_jumps.values())
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
        "\nRESULT: 100% PASSED — All 6 Beats + Dense Beat 1/3/4/5/6 + All 10 C.LABELS + "
        "1800-Frame Gimbal & Intra-Beat Continuity Audits PASSED!"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
