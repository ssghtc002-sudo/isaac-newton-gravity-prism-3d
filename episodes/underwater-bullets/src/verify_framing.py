"""
Pre-flight Mathematical 9:16 Portrait Framing, Subtitle Safe-Zone & Gimbal QA
for Episode #2: "Why Bullets Stop Dead Just 3 Feet Underwater" (60 fps, 1800 frames).

Pure-Python (runs cleanly outside Blender via `python episodes/underwater-bullets/src/verify_framing.py`).
Imports `bullet_contract` and `cinema_camera` directly so framing QA and Blender
camera generation share 100% identical trajectory, lens, and DOF math.
"""
from __future__ import annotations

import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import bullet_contract as C  # noqa: E402
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
    if key == "stop_3ft":
        return (C.THREE_FT_X, 0.0, float(C.TANK["flight_z"]) + 0.24)
    if key == "density_800":
        return (0.22, 0.0, float(C.TANK["flight_z"]) + 0.25)
    if key == "vapor_bubble":
        return (C.supersonic_x(f) - 0.04, 0.0, C.supersonic_z(f) + 0.14)
    if key == "shattered":
        return (C.THREE_FT_X, 0.0, C.supersonic_z(f) + 0.22)
    if key == "race_fast":
        return (C.THREE_FT_X - 0.08, 0.0, float(C.TANK["race_fast_z"]) + 0.20)
    if key == "race_slow":
        return (min(1.48, max(0.85, C.race_slow_x(f))), 0.0, float(C.TANK["race_slow_z"]) + 0.20)
    if key == "safe_depth":
        sx, sy, sz = C.SWIMMER["b6_pos"]
        return (sx - 0.22, sy, sz + 0.28)
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
        f"  [{status}] {tag:26s} -> NDC=({nx:+.2f}, {ny:+.2f}) depth={z:.2f}m "
        f"(|X|<={max_abs_x:.2f}, Y in [{min_y:+.2f}, {max_y:+.2f}])"
    )
    if not ok:
        errors.append(f"{tag}: NDC=({nx:+.2f}, {ny:+.2f}) depth={z:.2f}m out of range")
    return (nx, ny, z)


def main() -> int:
    errors: list[str] = []
    print("=== Underwater Bullets Ep#2 — 9:16 Portrait Framing & Safe-Zone QA (60fps, 1800f) ===")

    # ------------------------------------------------------------------
    # 1. BEAT 1 DENSE AUDIT (1..210)
    # ------------------------------------------------------------------
    print("\n[BEAT 1 — Hook: Incoming Bullet, 3 FT Datum Bar & Swimmer (1..210)]")
    sx1, sy1, sz1 = C.SWIMMER["b1_pos"]
    for f in (1, 35, 75, 118, 135, 170, 210):
        st = camera_state_at(float(f), include_motion_fx=True)
        bx, bz = C.supersonic_x(float(f)), C.supersonic_z(float(f))
        focus_err = abs(st["focus_dist"] - _dist3(st["cam_pos"], st["tgt_pos"]))
        dof_ok = abs(st["fstop"] - 7.5) < 1e-3 and focus_err < 1e-4
        if not dof_ok:
            errors.append(f"B1 f={f}: fstop={st['fstop']:.2f} focus_err={focus_err:.5f}")
        print(
            f"  f={f:03d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f} focus={st['focus_dist']:.2f}m "
            f"dof={'PASS' if dof_ok else 'FAIL'}"
        )
        _check_point(errors, f"B1_f{f:03d}_Bullet", (bx, 0.0, bz), st, 0.90, -0.45, 0.65)
        _check_point(errors, f"B1_f{f:03d}_3FT_BarMid", (C.THREE_FT_X, 0.0, float(C.TANK["flight_z"])), st, 0.85, -0.45, 0.65)
        _check_point(errors, f"B1_f{f:03d}_3FT_Ruler", (C.THREE_FT_X, 0.0, float(C.TANK["ruler_z"])), st, 0.85, -0.45, 0.25)
        _check_point(errors, f"B1_f{f:03d}_DiverHead", (sx1, sy1, -0.18), st, 0.90, -0.20, 0.78)
        _check_point(errors, f"B1_f{f:03d}_DiverTorso", (sx1, sy1, sz1), st, 0.90, -0.45, 0.55)

    # ------------------------------------------------------------------
    # 2. BEAT 2 DENSE AUDIT (211..540)
    # ------------------------------------------------------------------
    print("\n[BEAT 2 — Muzzle Blast, Water Entry Wall & 800x Density Shock (211..540)]")
    for f in (211, 248, 300, 360, 450, 540):
        st = camera_state_at(float(f), include_motion_fx=True)
        bx, bz = C.supersonic_x(float(f)), C.supersonic_z(float(f))
        print(
            f"  f={f:03d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f}"
        )
        if f <= 310:
            _check_point(errors, f"B2_f{f:03d}_Muzzle", (float(C.TANK["muzzle_x"]), 0.0, float(C.TANK["flight_z"])), st, 0.88, -0.45, 0.65)
        _check_point(errors, f"B2_f{f:03d}_WaterEntry", (float(C.TANK["entry_x"]), 0.0, float(C.TANK["flight_z"])), st, 0.88, -0.45, 0.65)
        _check_point(errors, f"B2_f{f:03d}_Bullet", (bx, 0.0, bz), st, 0.88, -0.45, 0.65)

    # ------------------------------------------------------------------
    # 3. BEAT 3 DENSE AUDIT (541..900)
    # ------------------------------------------------------------------
    print("\n[BEAT 3 — Action-Locked Macro Supercavitation & Vapor Bubble (541..900)]")
    for f in (541, 560, 580, 610, 720, 880, 900):
        st = camera_state_at(float(f), include_motion_fx=True)
        bx, bz = C.supersonic_x(float(f)), C.supersonic_z(float(f))
        if f <= 580:
            fs_ok = abs(st["fstop"] - 4.2) < 1e-2
        elif f >= 610:
            fs_ok = abs(st["fstop"] - 7.5) < 1e-2
        else:
            fs_ok = 4.2 <= st["fstop"] <= 7.5
        if not fs_ok:
            errors.append(f"B3 f={f}: unexpected fstop={st['fstop']:.2f}")
        print(
            f"  f={f:03d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f} dof={'PASS' if fs_ok else 'FAIL'}"
        )
        _check_point(errors, f"B3_f{f:03d}_Nose", (bx, 0.0, bz), st, 0.85, -0.45, 0.65)
        _check_point(errors, f"B3_f{f:03d}_BubbleTail", (bx - 0.08, 0.0, bz), st, 0.85, -0.45, 0.65)

    # ------------------------------------------------------------------
    # 4. BEAT 4 DENSE AUDIT (901..1260)
    # ------------------------------------------------------------------
    print("\n[BEAT 4 — Smooth Pull-Back, X-Ray Cutaway & 8-Shard Shatter at 3 FT (901..1260)]")
    for f in (901, 945, 1005, 1055, 1140, 1260):
        st = camera_state_at(float(f), include_motion_fx=True)
        bx, bz = C.supersonic_x(float(f)), C.supersonic_z(float(f))
        print(
            f"  f={f:04d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f}"
        )
        _check_point(errors, f"B4_f{f:04d}_Bullet3FT", (bx, 0.0, bz), st, 0.85, -0.45, 0.65)
        if f >= 1055:
            _check_point(errors, f"B4_f{f:04d}_ShardTop", (C.THREE_FT_X, 0.0, bz + 0.14), st, 0.85, -0.45, 0.72)
            _check_point(errors, f"B4_f{f:04d}_ShardBot", (C.THREE_FT_X, 0.0, bz - 0.14), st, 0.85, -0.45, 0.65)

    # ------------------------------------------------------------------
    # 5. BEAT 5 DENSE AUDIT (1261..1560)
    # ------------------------------------------------------------------
    print("\n[BEAT 5 — Studio Blueprint Paradox Race: Top 1900mph vs Bottom 750mph (1261..1560)]")
    for f in (1261, 1300, 1350, 1410, 1485, 1545, 1560):
        st = camera_state_at(float(f), include_motion_fx=True)
        fx_top = C.race_fast_x(float(f))
        fx_bot = C.race_slow_x(float(f))
        print(
            f"  f={f:04d} | fast_x={fx_top:.2f}m slow_x={fx_bot:.2f}m "
            f"cam_y={st['cam_pos'][1]:.2f}m lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f}"
        )
        _check_point(errors, f"B5_f{f:04d}_TopFast", (fx_top, 0.0, float(C.TANK["race_fast_z"])), st, 0.88, -0.25, 0.65)
        _check_point(errors, f"B5_f{f:04d}_BotSlow", (fx_bot, 0.0, float(C.TANK["race_slow_z"])), st, 0.88, -0.40, 0.45)
        _check_point(errors, f"B5_f{f:04d}_3FT_Ruler", (C.THREE_FT_X, 0.0, float(C.TANK["ruler_z"])), st, 0.88, -0.45, 0.25)
        if f >= 1485:
            _check_point(errors, f"B5_f{f:04d}_6FT_Ruler", (C.SIX_FT_X, 0.0, float(C.TANK["ruler_z"])), st, 0.90, -0.45, 0.25)

    # ------------------------------------------------------------------
    # 6. BEAT 6 DENSE AUDIT (1561..1800)
    # ------------------------------------------------------------------
    print("\n[BEAT 6 — Hero 3/4 Pull-Back: Peaceful Diver Below Z=-1.08m & Scrap (1561..1800)]")
    sx6, sy6, sz6 = C.SWIMMER["b6_pos"]
    for f in (1561, 1620, 1690, 1760, 1800):
        st = camera_state_at(float(f), include_motion_fx=True)
        print(
            f"  f={f:04d} | cam=({st['cam_pos'][0]:+.2f},{st['cam_pos'][1]:+.2f},{st['cam_pos'][2]:+.2f}) "
            f"lens={st['lens_mm']:.1f}mm f/{st['fstop']:.1f}"
        )
        _check_point(errors, f"B6_f{f:04d}_3FT_Zone", (C.THREE_FT_X, 0.0, float(C.TANK["flight_z"])), st, 0.85, -0.45, 0.65)
        _check_point(errors, f"B6_f{f:04d}_DiverTorso", (sx6, sy6, sz6), st, 0.85, -0.45, 0.55)
        _check_point(errors, f"B6_f{f:04d}_DiverLow", (sx6, sy6, -1.55), st, 0.85, -0.45, 0.30)
        _check_point(errors, f"B6_f{f:04d}_Scrap", (C.THREE_FT_X + 0.15, 0.0, -0.82), st, 0.85, -0.45, 0.55)

    # ------------------------------------------------------------------
    # 7. ALL C.LABELS ANCHOR SAFE-ZONE AUDIT (NDC Y >= -0.45)
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
        "\nRESULT: 100% PASSED — All 6 Beats + Diver Bounds + Bullet Trajectories + "
        "All 7 C.LABELS + 1800-Frame Gimbal & Continuity Audits PASSED!"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
