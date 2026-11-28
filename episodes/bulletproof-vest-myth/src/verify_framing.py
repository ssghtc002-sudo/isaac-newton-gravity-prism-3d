"""
Pre-flight Mathematical 9:16 Portrait Framing, Subtitle Safe-Zone & Gimbal QA
for Episode #3: "Why Bulletproof Vests DON'T Send You Flying Backward" (60 fps, 1800 frames).

Pure-Python (runs cleanly outside Blender via `python3 episodes/bulletproof-vest-myth/src/verify_framing.py`).
Imports `vest_contract` and `cinema_camera` directly so framing QA and Blender
camera generation share 100% identical trajectory, lens, and DOF math.
"""
from __future__ import annotations

import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import vest_contract as C  # noqa: E402
from cinema_camera import HARD_CUT_FRAMES, camera_state_at, project_ndc  # noqa: E402

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


def _label_anchor_world(key: str, f: float) -> Vec3:
    """Contract-aligned world-space anchor zone for each 3D engineering label in C.LABELS."""
    vx = float(C.RANGE["vest_surface_x"])
    cz = float(C.RANGE["chest_z"])
    if key == "shift_half_in":
        return (vx + C.hero_recoil_x(f), 0.0, cz + 0.16)
    if key == "tiny_momentum":
        return (C.pistol_bullet_x(f), 0.0, cz + 0.18)
    if key == "kevlar_net":
        return (vx + 0.018, 0.0, cz + 0.12)
    if key == "ceramic_dyneema":
        return (vx + 0.018, 0.0, cz + 0.14)
    if key == "newton_eq":
        return (float(C.RANGE["b5_shooter_x"]) - 0.25, 0.0, float(C.RANGE["b5_vector_z"]) + 0.22)
    if key == "equal_recoil":
        return (float(C.RANGE["b5_target_x"]) + 0.25, 0.0, float(C.RANGE["b5_vector_z"]) - 0.18)
    if key == "just_bruise":
        return (vx, 0.0, cz + 0.16)
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
        f"  [{status}] {tag:28s} -> NDC=({nx:+.2f}, {ny:+.2f}) depth={z:.2f}m "
        f"(|X|<={max_abs_x:.2f}, Y in [{min_y:+.2f}, {max_y:+.2f}])"
    )
    if not ok:
        errors.append(f"{tag}: NDC=({nx:+.2f}, {ny:+.2f}) depth={z:.2f}m out of range")
    return (nx, ny, z)


def main() -> int:
    errors: list[str] = []
    print("=== Bulletproof Vest Myth Ep#3 — 9:16 Portrait Framing & Safe-Zone QA (60fps, 1800f) ===")

    vx = float(C.RANGE["vest_surface_x"])
    cz = float(C.RANGE["chest_z"])
    ez = float(C.RANGE["eye_z"])

    # ------------------------------------------------------------------
    # 1. BEAT 1 DENSE AUDIT (1..210)
    # ------------------------------------------------------------------
    print("\n[BEAT 1 — Hook: Slow-Mo Chest Impact, 0.5-Inch Shift & Intact Glass Wall (1..210)]")
    hx1 = float(C.HERO["b1_pos"][0])
    for f in (1, 22, 55, 75, 92, 145, 210):
        st = camera_state_at(float(f), include_motion_fx=True)
        bx = C.pistol_bullet_x(float(f))
        rx = C.hero_recoil_x(float(f))
        focus_err = abs(st["focus_dist"] - _dist3(st["cam_pos"], st["tgt_pos"]))
        dof_ok = abs(st["fstop"] - 7.5) < 1e-3 and focus_err < 1e-4
        if not dof_ok:
            errors.append(f"B1 f={f}: fstop={st['fstop']:.2f} focus_err={focus_err:.5f}")
        print(
            f"  f={f:03d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f} focus={st['focus_dist']:.2f}m "
            f"recoil={rx*1000:.1f}mm dof={'PASS' if dof_ok else 'FAIL'}"
        )
        _check_point(errors, f"B1_f{f:03d}_Bullet", (bx, 0.0, cz), st, 0.90, -0.45, 0.65)
        _check_point(errors, f"B1_f{f:03d}_VestPlate", (vx + rx, 0.0, cz), st, 0.85, -0.45, 0.65)
        _check_point(errors, f"B1_f{f:03d}_HeroEye", (hx1 + rx, 0.0, ez), st, 0.88, -0.10, 0.72)
        if f >= 75:
            _check_point(errors, f"B1_f{f:03d}_GlassWall", (float(C.RANGE["glass_wall_x"]), 0.0, cz), st, 0.96, -0.45, 0.65)

    # ------------------------------------------------------------------
    # 2. BEAT 2 DENSE AUDIT (211..510)
    # ------------------------------------------------------------------
    print("\n[BEAT 2 — Momentum Myth: Muzzle Flash Across Range -> Hero Vest (211..510)]")
    hx2 = float(C.HERO["b2_pos"][0])
    for f in (211, 245, 290, 335, 420, 510):
        st = camera_state_at(float(f), include_motion_fx=True)
        bx = C.pistol_bullet_x(float(f))
        rx = C.hero_recoil_x(float(f))
        print(
            f"  f={f:03d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f}"
        )
        if f <= 265:
            _check_point(errors, f"B2_f{f:03d}_Muzzle", (float(C.RANGE["muzzle_x"]), 0.0, cz), st, 0.88, -0.45, 0.65)
        _check_point(errors, f"B2_f{f:03d}_Bullet", (bx, 0.0, cz), st, 0.88, -0.45, 0.65)
        if f >= 335:
            _check_point(errors, f"B2_f{f:03d}_VestHit", (vx + rx, 0.0, cz), st, 0.88, -0.45, 0.65)
            _check_point(errors, f"B2_f{f:03d}_HeroEye", (hx2 + rx, 0.0, ez), st, 0.90, -0.10, 0.72)

    # ------------------------------------------------------------------
    # 3. BEAT 3 DENSE AUDIT (511..900)
    # ------------------------------------------------------------------
    print("\n[BEAT 3 — Macro Vest Cutaway: Golden Kevlar Trampoline Ripple (511..900)]")
    for f in (511, 535, 560, 595, 615, 735, 880, 900):
        st = camera_state_at(float(f), include_motion_fx=True)
        bx = C.pistol_bullet_x(float(f))
        if f <= 560:
            fs_ok = abs(st["fstop"] - 4.0) < 1e-2
        elif f >= 590:
            fs_ok = abs(st["fstop"] - 7.5) < 1e-2
        else:
            fs_ok = 4.0 <= st["fstop"] <= 7.5
        if not fs_ok:
            errors.append(f"B3 f={f}: unexpected fstop={st['fstop']:.2f}")
        print(
            f"  f={f:03d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f} dof={'PASS' if fs_ok else 'FAIL'}"
        )
        _check_point(errors, f"B3_f{f:03d}_9mmRound", (bx, 0.0, cz), st, 0.88, -0.45, 0.65)
        _check_point(errors, f"B3_f{f:03d}_KevlarCore", (vx + 0.018, 0.0, cz), st, 0.85, -0.45, 0.65)
        _check_point(errors, f"B3_f{f:03d}_KevlarTop", (vx + 0.018, 0.0, cz + 0.14), st, 0.85, -0.25, 0.72)
        _check_point(errors, f"B3_f{f:03d}_KevlarBot", (vx + 0.018, 0.0, cz - 0.14), st, 0.85, -0.45, 0.45)

    # ------------------------------------------------------------------
    # 4. BEAT 4 DENSE AUDIT (901..1260)
    # ------------------------------------------------------------------
    print("\n[BEAT 4 — Smooth Macro Pull-Back: SiC Ceramic Shatter + Dyneema Catch (901..1260)]")
    for f in (901, 925, 965, 1030, 1125, 1260):
        st = camera_state_at(float(f), include_motion_fx=True)
        rx4 = C.rifle_bullet_x(float(f))
        print(
            f"  f={f:04d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f}"
        )
        _check_point(errors, f"B4_f{f:04d}_RifleRound", (rx4, 0.0, cz), st, 0.88, -0.45, 0.65)
        _check_point(errors, f"B4_f{f:04d}_SiC_Plate", (vx + 0.005, 0.0, cz), st, 0.85, -0.45, 0.65)
        _check_point(errors, f"B4_f{f:04d}_Dyneema", (vx + 0.036, 0.0, cz), st, 0.85, -0.45, 0.65)
        if f >= 1030:
            _check_point(errors, f"B4_f{f:04d}_DustTop", (vx + 0.010, 0.0, cz + 0.16), st, 0.85, -0.30, 0.72)
            _check_point(errors, f"B4_f{f:04d}_DustBot", (vx + 0.010, 0.0, cz - 0.16), st, 0.85, -0.45, 0.45)

    # ------------------------------------------------------------------
    # 5. BEAT 5 DENSE AUDIT (1261..1560)
    # ------------------------------------------------------------------
    print("\n[BEAT 5 — Studio Blueprint Newton's Third Law: Equal & Opposite F1 = -F2 (1261..1560)]")
    sx5_base = float(C.RANGE["b5_shooter_x"])
    tx5_base = float(C.RANGE["b5_target_x"])
    vlen = float(C.RANGE["b5_vector_len"])
    vz5 = float(C.RANGE["b5_vector_z"])
    for f in (1261, 1295, 1365, 1405, 1490, 1560):
        st = camera_state_at(float(f), include_motion_fx=True)
        bx5 = C.rifle_bullet_x(float(f))
        srx = C.shooter_recoil_x(float(f))
        hrx = C.hero_recoil_x(float(f))
        print(
            f"  f={f:04d} | bullet_x={bx5:+.2f}m shooter_rx={srx*1000:+.1f}mm hero_rx={hrx*1000:+.1f}mm "
            f"cam_y={st['cam_pos'][1]:.2f}m lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f}"
        )
        _check_point(errors, f"B5_f{f:04d}_ShooterChest", (sx5_base + srx, 0.0, cz), st, 0.82, -0.45, 0.65)
        _check_point(errors, f"B5_f{f:04d}_TargetChest", (tx5_base + hrx, 0.0, cz), st, 0.82, -0.45, 0.65)
        _check_point(errors, f"B5_f{f:04d}_Bullet", (bx5, 0.0, cz), st, 0.85, -0.45, 0.65)
        _check_point(errors, f"B5_f{f:04d}_VecF1Tip", (sx5_base - vlen, 0.0, vz5), st, 0.92, -0.45, 0.65)
        _check_point(errors, f"B5_f{f:04d}_VecF2Tip", (tx5_base + vlen, 0.0, vz5), st, 0.92, -0.45, 0.65)

    # ------------------------------------------------------------------
    # 6. BEAT 6 DENSE AUDIT (1561..1800)
    # ------------------------------------------------------------------
    print("\n[BEAT 6 — Hero 3/4 Steady Payoff: Intact Glass & Spent Dust (1561..1800)]")
    hx6 = float(C.HERO["b6_pos"][0])
    for f in (1561, 1620, 1690, 1760, 1800):
        st = camera_state_at(float(f), include_motion_fx=True)
        print(
            f"  f={f:04d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f}"
        )
        _check_point(errors, f"B6_f{f:04d}_HeroEye", (hx6, 0.0, ez), st, 0.85, -0.10, 0.72)
        _check_point(errors, f"B6_f{f:04d}_VestPlate", (vx, 0.0, cz), st, 0.85, -0.45, 0.65)
        _check_point(errors, f"B6_f{f:04d}_HeroWaist", (hx6, 0.0, 0.92), st, 0.85, -0.45, 0.40)
        _check_point(errors, f"B6_f{f:04d}_GlassWall", (float(C.RANGE["glass_wall_x"]), 0.0, cz), st, 0.95, -0.45, 0.65)

    # ------------------------------------------------------------------
    # 7. ALL C.LABELS ANCHOR SAFE-ZONE AUDIT (NDC Y >= -0.45)
    # ------------------------------------------------------------------
    print("\n[3D LABELS SAFE-ZONE AUDIT (C.LABELS, NDC Y >= -0.45)]")
    for key, _text, (w0, w1) in C.LABELS:
        for f in (w0, (w0 + w1) // 2, w1):
            st = camera_state_at(float(f), include_motion_fx=True)
            pt = _label_anchor_world(key, float(f))
            _check_point(errors, f"LBL_{key}_f{f:04d}", pt, st, 0.88, LABEL_MIN_NDC_Y, 0.85, min_depth=0.25)

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
        "\nRESULT: 100% PASSED — All 6 Beats + Hero/Shooter Bounds + Kevlar/Ceramic Cutaway + "
        "All 7 C.LABELS + 1800-Frame Gimbal & Continuity Audits PASSED!"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
