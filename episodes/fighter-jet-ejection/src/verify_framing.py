"""
Pre-flight Mathematical 9:16 Portrait Framing, Subtitle Safe-Zone & Gimbal QA
for Episode #1: "Why Ejecting From a Fighter Jet Shrinks You by 1 Inch" (60 fps, 1800 frames).

Pure-Python (runs cleanly outside Blender via `python src/verify_framing.py`).
Imports `ejection_contract` and `cinema_camera` directly so framing QA and Blender
camera generation share 100% identical trajectory, lens, and DOF math.
"""
from __future__ import annotations

import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import ejection_contract as C  # noqa: E402
from cinema_camera import camera_state_at, project_ndc  # noqa: E402


def _sub3(a: C.Vec3, b: C.Vec3) -> C.Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot3(a: C.Vec3, b: C.Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross3(a: C.Vec3, b: C.Vec3) -> C.Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm3(a: C.Vec3) -> C.Vec3:
    length = math.sqrt(_dot3(a, a)) or 1e-9
    return (a[0] / length, a[1] / length, a[2] / length)


def _dist3(a: C.Vec3, b: C.Vec3) -> float:
    return math.sqrt(sum((u - v) ** 2 for u, v in zip(a, b)))


def _label_anchor_world(key: str, f: float) -> C.Vec3:
    """Contract-aligned world-space anchor zone for each 3D engineering label in C.LABELS."""
    s = C.seat_world_location(f)
    if key == "handle":
        return C.add3(s, C.handle_local(f))
    if key == "mdc":
        return C.CANOPY["apex"]
    if key == "clock_mdc":
        return (0.0, 0.05, 1.18)
    if key == "garters":
        return C.add3(s, C.lerp3(C.SEAT["garter_anchor_R"], C.ankle_local("R", f), 0.5))
    if key == "catapult":
        return C.lerp3(C.RAIL_BASE, C.add3(s, (0.0, -0.22, -0.05)), 0.5)
    if key == "g_load":
        return C.add3(s, (0.0, -0.04, 0.58))
    if key == "spine_gauge":
        return C.add3(s, (0.0, -0.085, 0.41))
    if key == "rocket":
        return C.add3(s, C.SEAT["rocket_nozzles"][1])
    if key == "drogue":
        return C.drogue_world_location(f)
    if key == "clock_drogue":
        return C.add3(C.drogue_world_location(f), (0.0, 0.0, 0.50))
    raise KeyError(f"Unknown label key: {key}")


def _check_point(
    errors: list[str],
    tag: str,
    pt: C.Vec3,
    st: dict,
    max_abs_x: float,
    min_y: float,
    max_y: float,
    min_depth: float = 0.25,
) -> tuple[float, float, float]:
    nx, ny, z = project_ndc(pt, st["cam_pos"], st["tgt_pos"], st["lens_mm"])
    ok = (
        math.isfinite(nx)
        and math.isfinite(ny)
        and z > min_depth
        and abs(nx) <= max_abs_x
        and min_y <= ny <= max_y
    )
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] {tag:22s} -> NDC=({nx:+.2f}, {ny:+.2f}) depth={z:.2f}m "
        f"(|X|<={max_abs_x:.2f}, Y in [{min_y:+.2f}, {max_y:+.2f}])"
    )
    if not ok:
        errors.append(f"{tag}: NDC=({nx:+.2f}, {ny:+.2f}) depth={z:.2f}m out of range")
    return (nx, ny, z)


def main() -> int:
    errors: list[str] = []
    print("=== Fighter Jet Ejection Ep#1 — 9:16 Portrait Framing & Safe-Zone QA (60fps, 1800f) ===")

    # ------------------------------------------------------------------
    # 1. BEAT 1 DENSE HOOK + PREVIEW[0] AUDIT
    # ------------------------------------------------------------------
    b1_frames = sorted(set([1] + C.BEAT1_DENSE_FRAMES + [C.PREVIEW_FRAMES[0]]))
    print("\n[BEAT 1 — Viral Cockpit Hook, Eyes & Handle Yank (Dense Audit)]")
    for f in b1_frames:
        st = camera_state_at(float(f), include_motion_fx=True)
        out_canopy = st["cam_pos"][0] > C.CANOPY["half_width"] + 0.10
        if not out_canopy:
            errors.append(f"B1 f={f}: cam_x={st['cam_pos'][0]:.3f} inside canopy")
        print(
            f"  f={f:03d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f} outside_canopy={'PASS' if out_canopy else 'FAIL'}"
        )
        _check_point(errors, f"B1_f{f:03d}_EyeMid", C.PILOT["eye_mid"], st, 0.70, 0.20, 0.58)
        _check_point(errors, f"B1_f{f:03d}_HeadCrown", (0.0, -0.08, 0.96), st, 0.75, 0.10, 0.85)
        _check_point(errors, f"B1_f{f:03d}_Handle", C.handle_local(f), st, 0.70, -0.36, -0.10)

    # Verify Beat 1 yank micro-kick impulse at f=150..172
    st_y_fx = camera_state_at(158.0, include_motion_fx=True)
    st_y_base = camera_state_at(158.0, include_motion_fx=False)
    yank_disp_mm = _dist3(st_y_fx["cam_pos"], st_y_base["cam_pos"]) * 1000.0
    yank_lens_d = st_y_fx["lens_mm"] - st_y_base["lens_mm"]
    yank_ok = yank_disp_mm >= 5.0 and yank_lens_d >= 0.4
    print(
        f"  [{'PASS' if yank_ok else 'FAIL'}] Beat 1 yank micro-kick @ f=158: "
        f"cam_kick={yank_disp_mm:.1f}mm, lens_kick=+{yank_lens_d:.2f}mm"
    )
    if not yank_ok:
        errors.append("Beat 1 yank micro-kick too weak at f=158")

    # ------------------------------------------------------------------
    # 2. BEATS 2–6 + ALL PREVIEW FRAMES AUDIT
    # ------------------------------------------------------------------
    print("\n[BEAT 2 — MDC Canopy Detonation Bullet-Time]")
    for f in (C.EV["mdc_detonate"], C.PREVIEW_FRAMES[1], 540):
        st = camera_state_at(float(f), include_motion_fx=True)
        print(f"  f={f:03d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) lens={st['lens_mm']:.1f}mm")
        _check_point(errors, f"B2_f{f:03d}_CanopyApex", C.CANOPY["apex"], st, 0.75, -0.25, 0.65)
        _check_point(errors, f"B2_f{f:03d}_PilotEye", C.PILOT["eye_mid"], st, 0.75, -0.40, 0.50)
        _check_point(errors, f"B2_f{f:03d}_CanopySill", (C.CANOPY["half_width"], 0.15, C.CANOPY["sill_z"]), st, 0.85, C.SUBTITLE_ZONE_NDC_Y, 0.40)

    print("\n[BEAT 3 — Leg Garters & Catapult Rails X-Ray]")
    for f in (C.EV["garter_start"], C.EV["garter_end"], C.EV["catapult_ignite"], C.PREVIEW_FRAMES[2], 880):
        st = camera_state_at(float(f), include_motion_fx=True)
        seat = C.seat_world_location(f)
        print(f"  f={f:03d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) lens={st['lens_mm']:.1f}mm")
        _check_point(errors, f"B3_f{f:03d}_AnkleL", C.add3(seat, C.ankle_local("L", f)), st, 0.85, C.SUBTITLE_ZONE_NDC_Y, 0.75)
        _check_point(errors, f"B3_f{f:03d}_AnkleR", C.add3(seat, C.ankle_local("R", f)), st, 0.85, C.SUBTITLE_ZONE_NDC_Y, 0.75)
        _check_point(errors, f"B3_f{f:03d}_RailBase", C.RAIL_BASE, st, 0.85, C.SUBTITLE_ZONE_NDC_Y, 0.40)
        _check_point(errors, f"B3_f{f:03d}_SeatSRP", seat, st, 0.85, C.SUBTITLE_ZONE_NDC_Y, 0.65)

    print("\n[BEAT 4 — Spine X-Ray Compression & Exact DOF Focus Plane]")
    for f in (C.EV["spine_compress_start"], C.PREVIEW_FRAMES[3], 1200):
        st = camera_state_at(float(f), include_motion_fx=True)
        seat = C.seat_world_location(f)
        spine_mid = C.add3(seat, (0.0, -0.085, 0.41))
        print(f"  f={f:03d} | lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f} focus_dist={st['focus_dist']:.3f}m")
        _check_point(errors, f"B4_f{f:04d}_Sacrum", C.add3(seat, C.PILOT["sacrum"]), st, 0.65, C.SUBTITLE_ZONE_NDC_Y, 0.30)
        _, _, z_mid = _check_point(errors, f"B4_f{f:04d}_SpineMid", spine_mid, st, 0.50, -0.30, 0.45)
        _check_point(errors, f"B4_f{f:04d}_C1Base", C.add3(seat, C.PILOT["c1_skull_base"]), st, 0.65, -0.10, 0.75)
        dof_err = abs(st["focus_dist"] - z_mid)
        dof_ok = dof_err <= 0.03
        print(f"  [{'PASS' if dof_ok else 'FAIL'}] B4_f{f:04d}_DOFPlane    -> focus_err={dof_err * 1000:.1f}mm (<= 30mm)")
        if not dof_ok:
            errors.append(f"B4 f={f}: focus plane mismatch {dof_err:.3f}m")

    print("\n[BEAT 5 — Under-Seat Rocket, Tail-Fin Clearance & Drogue Chute]")
    for f in (C.EV["rocket_ignite"], C.EV["tail_clear"], C.PREVIEW_FRAMES[4], 1480, 1525):
        st = camera_state_at(float(f), include_motion_fx=True)
        seat = C.seat_world_location(f)
        dist_s = _dist3(st["cam_pos"], seat)
        print(f"  f={f:04d} | dist_from_seat={dist_s:.2f}m lens={st['lens_mm']:.1f}mm")
        _check_point(errors, f"B5_f{f:04d}_SeatSRP", seat, st, 0.85, C.SUBTITLE_ZONE_NDC_Y, 0.80)
        _check_point(errors, f"B5_f{f:04d}_RocketNoz", C.add3(seat, C.SEAT["rocket_nozzles"][0]), st, 0.85, C.SUBTITLE_ZONE_NDC_Y, 0.80)
        if f == C.EV["tail_clear"]:
            tail_pt = C.add3(C.jet_offset(f), (0.0, -7.5, 2.0))
            _check_point(errors, f"B5_f{f:04d}_JetTailFin", tail_pt, st, 0.85, C.SUBTITLE_ZONE_NDC_Y, 0.40)
            dist_ok = 4.6 <= dist_s <= 6.6
            print(f"  [{'PASS' if dist_ok else 'FAIL'}] B5_f{f:04d}_SeatDist     -> {dist_s:.2f}m (4.6..6.6m)")
            if not dist_ok:
                errors.append(f"B5 f={f}: dist_from_seat={dist_s:.2f}m")
        if f >= C.EV["drogue_fire"]:
            drogue = C.drogue_world_location(f)
            _check_point(errors, f"B5_f{f:04d}_Drogue", drogue, st, 0.85, -0.35, 0.85)

    print("\n[BEAT 6 — Heroic Pull-Back & 7.6m Main Canopy Finale]")
    for f in (C.EV["separation"], C.EV["main_line_stretch"], C.EV["main_full"], C.PREVIEW_FRAMES[5], 1800):
        st = camera_state_at(float(f), include_motion_fx=True)
        pc = C.pilot_carrier_location(f)
        mc = C.main_chute_world_location(f)
        dist_pc = _dist3(st["cam_pos"], pc)
        fwd = _norm3(_sub3(st["tgt_pos"], st["cam_pos"]))
        right = _norm3(_cross3(fwd, (0.0, 0.0, 1.0)))
        u = C.main_chute_open_fraction(f)
        r_eff = 0.5 * C.MAIN_CHUTE_DIAMETER_M * (max(0.08, (u - 0.25) / 0.75) if u > 0.25 else 0.08)
        print(f"  f={f:04d} | dist_from_pilot={dist_pc:.2f}m lens={st['lens_mm']:.1f}mm r_canopy={r_eff:.2f}m")
        _check_point(errors, f"B6_f{f:04d}_PilotCarry", pc, st, 0.85, -0.45, 0.85)
        _check_point(errors, f"B6_f{f:04d}_ChuteSkirt", mc, st, 0.85, -0.45, 0.85)
        _check_point(errors, f"B6_f{f:04d}_ChuteApex", C.add3(mc, (0.0, 0.0, 2.5)), st, 0.85, -0.45, 0.85)
        _check_point(errors, f"B6_f{f:04d}_CanopyEdgeL", C.add3(mc, (-r_eff * right[0], -r_eff * right[1], 0.0)), st, 0.85, -0.45, 0.85)
        _check_point(errors, f"B6_f{f:04d}_CanopyEdgeR", C.add3(mc, (r_eff * right[0], r_eff * right[1], 0.0)), st, 0.85, -0.45, 0.85)

    # ------------------------------------------------------------------
    # 3. ALL 10 C.LABELS ANCHOR SAFE-ZONE AUDIT (NDC Y >= -0.45)
    # ------------------------------------------------------------------
    print("\n[3D LABELS SAFE-ZONE AUDIT (C.LABELS, NDC Y >= -0.45)]")
    for key, text, (w0, w1) in C.LABELS:
        for f in (w0, (w0 + w1) // 2, w1):
            st = camera_state_at(float(f), include_motion_fx=True)
            pt = _label_anchor_world(key, float(f))
            _check_point(errors, f"LBL_{key}_f{f:04d}", pt, st, 0.85, C.LABEL_MIN_NDC_Y, 0.85, min_depth=0.15)

    # ------------------------------------------------------------------
    # 4. FULL 1800-FRAME CONTINUITY & ANTI-GIMBAL-FLIP AUDIT
    # ------------------------------------------------------------------
    max_abs_fwd_z = 0.0
    min_dist = 999.0
    max_step_cam = 0.0
    max_step_tgt = 0.0
    max_step_lens = 0.0
    prev_st = None

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
            max_step_cam = max(max_step_cam, _dist3(cp, prev_st["cam_pos"]))
            max_step_tgt = max(max_step_tgt, _dist3(tp, prev_st["tgt_pos"]))
            max_step_lens = max(max_step_lens, abs(ln - prev_st["lens_mm"]))
        prev_st = st

    gimbal_ok = max_abs_fwd_z < 0.85 and min_dist > 0.25
    cont_ok = max_step_cam <= 0.35 and max_step_tgt <= 0.35 and max_step_lens <= 1.0
    if not gimbal_ok:
        errors.append(f"Gimbal/dist fail: max_|fwd.z|={max_abs_fwd_z:.3f}, min_dist={min_dist:.3f}")
    if not cont_ok:
        errors.append(f"Continuity spike: dCam={max_step_cam:.3f}, dTgt={max_step_tgt:.3f}, dLens={max_step_lens:.2f}")

    print(
        f"\n[1800-FRAME CONTINUITY & ANTI-GIMBAL-FLIP AUDIT]\n"
        f"  [{'PASS' if gimbal_ok else 'FAIL'}] min_target_dist={min_dist:.3f}m (> 0.25m), "
        f"max_|fwd.z|={max_abs_fwd_z:.3f} (< 0.85 guaranteed zero TRACK_TO flip!)\n"
        f"  [{'PASS' if cont_ok else 'FAIL'}] max_step_cam={max_step_cam:.3f}m/f, "
        f"max_step_tgt={max_step_tgt:.3f}m/f, max_step_lens={max_step_lens:.2f}mm/f"
    )

    if errors:
        print(f"\nRESULT: FAILED — {len(errors)} violation(s):")
        for e in errors:
            print(f"  * {e}")
        return 1

    print(
        "\nRESULT: 100% PASSED — All 6 Beats + Beat 1 Dense Hook + All 10 C.LABELS + "
        "1800-Frame Gimbal & Continuity Audits PASSED!"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
