"""
Pre-flight Mathematical 9:16 Portrait Camera Framing, Subtitle Safe-Zone & Sightline QA
for `episodes/how-grenade-works` (Hollywood Action-Movie 6-Beat Desert Battlefield Edition).

Pure-Python (runs cleanly outside Blender!) and imports `camera_state_at`,
`grenade_pos_at`, and `GRENADE_FLIGHT_KFS` directly from `cinema_camera.py` so the
camera engine and framing verifier share 100% identical math with zero drift.

Verifies:
  1. All 9 key beat moments:
       - B1-Bite f=024: Heroic low-angle tight close-up on teeth biting chrome pull-ring
       - B1      f=035: Hero close-up pin rip-out (2mm micro-snap punch) + Hero <-> Grenade separation
       - B1-Spit f=068: Rajinikanth teeth pin spit-flick (56mm + Screen-Right kinetic whip-accent)
       - B1/2    f=134: Hero throw release & outstretched arm separation (35mm)
       - B2      f=195: Mid-air Matrix Bullet-Time X-ray grenade + 3D component labels + ~55% screen fill
       - B3      f=294: Striker hammer slamming primer cap + flying safety spoon
       - B4      f=420: 4-second mid-air delay column burn + 3D countdown callout
       - B5      f=540: Low-angle bunker ground-level macro + Enemy Sandbag Wall + diving Rival Sentry
       - B6      f=665: Wide bullet-time explosion climax (fireball, sandbags, BOTH diving sentries, shockwave)
     plus Beat 6 orbital start (f=634) & end (f=720) framing!
  2. 9:16 Portrait Subtitle Safe-Zone:
       - Beat 1 Hero eyes, mouth/teeth, grenade fist & Screen-Right spit-flick exit path strictly
         inside the upper-middle safe zone NDC Y in [-0.15, +0.62].
       - Primary Beats 2-6 hero subjects strictly inside NDC Y in [-0.25, +0.60].
       - Foreground grenade bottom edge cleanly above karaoke subtitle band (NDC Y > -0.55).
  3. Full 720-frame continuity & anti-gimbal-flip sweep (TRACK_TO -Z / UP_Y safety +
     exact `GRENADE_FLIGHT_KFS` anchor verification).
"""
from __future__ import annotations

import math
import os
import sys

# Ensure sibling import works regardless of current working directory
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from cinema_camera import (
    BUNKER_DEFAULT_POS,
    GRENADE_FLIGHT_KFS,
    HERO_DEFAULT_POS,
    HERO_EYE_LOCAL,
    HERO_GRENADE_HAND_LOCAL,
    HERO_MOUTH_LOCAL,
    HERO_THROW_RELEASE_LOCAL,
    R_OUTER_REAL,
    SANDBAG_WALL_Y,
    SENSOR_H_EFF_MM,
    SENSOR_W_EFF_MM,
    SENTRY_LEFT_POS,
    SENTRY_RIGHT_POS,
    camera_state_at,
    grenade_pos_at,
)


def _sub3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross3(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm3(a):
    length = math.sqrt(_dot3(a, a)) or 1e-9
    return (a[0] / length, a[1] / length, a[2] / length)


def project_ndc(pt, cam_pos, tgt_pos, lens_mm: float):
    """Projects a 3D world point into 9:16 portrait NDC [-1, +1] matching Blender sensor_fit='AUTO'."""
    fwd = _norm3(_sub3(tgt_pos, cam_pos))
    right = _norm3(_cross3(fwd, (0.0, 0.0, 1.0)))
    up = _cross3(right, fwd)
    rel = _sub3(pt, cam_pos)
    z = _dot3(rel, fwd)
    x = _dot3(rel, right)
    y = _dot3(rel, up)
    if z <= 0.01:
        return (99.0, 99.0, z)
    ndc_x = (x / z) * (lens_mm / (0.5 * SENSOR_W_EFF_MM))
    ndc_y = (y / z) * (lens_mm / (0.5 * SENSOR_H_EFF_MM))
    return (ndc_x, ndc_y, z)


def _screen_width_fraction(radius_m: float, depth_m: float, lens_mm: float) -> float:
    """Returns fraction of full 9:16 portrait screen width spanned by a sphere of given radius."""
    half_width_m = depth_m * (0.5 * SENSOR_W_EFF_MM) / lens_mm
    return (2.0 * radius_m) / max(1e-6, 2.0 * half_width_m)


def main() -> int:
    errors = 0

    print("=== How a Grenade Works V3 — Hollywood 9:16 Portrait Framing & Subtitle-Zone QA ===")

    # ------------------------------------------------------------------
    # 1. VERIFY GRENADE_FLIGHT_KFS EXACT ANCHOR SYNC
    # ------------------------------------------------------------------
    expected_anchors = {
        134: (0.240, 0.420, 1.620),
        152: (0.050, 2.350, 2.250),
        440: (0.000, 4.950, 2.350),
        495: (0.000, 7.250, 0.045),
    }
    kf_dict = dict(GRENADE_FLIGHT_KFS)
    print("\n[TRAJECTORY ANCHOR SYNC CHECK]")
    for f_anc, exp_p in expected_anchors.items():
        g_p = grenade_pos_at(f_anc)
        err = math.sqrt(sum((a - b) ** 2 for a, b in zip(g_p, exp_p)))
        in_kfs = f_anc in kf_dict
        ok = err < 1e-5 and in_kfs
        if not ok:
            errors += 1
        print(
            f"  [{'PASS' if ok else 'FAIL'}] f={f_anc:03d} | "
            f"pos=({g_p[0]:+.3f}, {g_p[1]:+.3f}, {g_p[2]:+.3f}) err={err:.5f}m"
        )

    # ------------------------------------------------------------------
    # 2. 9 KEY BEATS + BEAT 6 ORBIT BOUNDARY FRAMING CHECKS
    # ------------------------------------------------------------------
    g195 = grenade_pos_at(195)
    g294 = grenade_pos_at(294)
    g420 = grenade_pos_at(420)
    g540 = grenade_pos_at(540)

    # Format: (beat, frame, title, points_spec, sep_rules, macro_fill_target)
    # points_spec entry: (label, world_pt, max_abs_ndc_x, min_ndc_y, max_ndc_y)
    beat_checks = [
        (
            "B1-Bite",
            24,
            "Hero Low-Angle Tight Close-Up: Teeth Biting Chrome Pull-Ring",
            [
                ("HeroEyes", HERO_EYE_LOCAL, 0.72, -0.15, 0.62),
                ("HeroMouth", HERO_MOUTH_LOCAL, 0.72, -0.15, 0.62),
                ("HeroCrown", (0.0, 0.06, 1.80), 0.75, -0.10, 0.78),
                ("BitePinAtTeeth", (0.04, 0.19, 1.54), 0.45, -0.15, 0.62),
                ("GrenadeInHand", HERO_GRENADE_HAND_LOCAL, 0.75, -0.15, 0.62),
            ],
            [("HeroEyes", "GrenadeInHand", 0.40)],
            None,
        ),
        (
            "B1",
            35,
            "Hero Close-Up: Pin Rip-Out Micro-Snap Punch (Face & Grenade)",
            [
                ("HeroEyes", HERO_EYE_LOCAL, 0.75, -0.15, 0.62),
                ("HeroMouth", HERO_MOUTH_LOCAL, 0.75, -0.15, 0.62),
                ("HeroCrown", (0.0, 0.06, 1.80), 0.78, -0.10, 0.78),
                ("GrenadeInHand", HERO_GRENADE_HAND_LOCAL, 0.75, -0.15, 0.62),
                ("PullPinRing", (0.13, 0.30, 1.54), 0.82, -0.15, 0.62),
            ],
            [("HeroEyes", "GrenadeInHand", 0.40)],
            None,
        ),
        (
            "B1-Spit",
            68,
            "Rajinikanth Pin Spit-Flick Moment (56mm Kinetic Whip-Accent)",
            [
                ("HeroEyes", HERO_EYE_LOCAL, 0.60, -0.15, 0.62),
                ("HeroMouth", HERO_MOUTH_LOCAL, 0.60, -0.15, 0.62),
                ("HeroCrown", (0.0, 0.06, 1.80), 0.68, -0.10, 0.78),
                ("GrenadeInHand", HERO_GRENADE_HAND_LOCAL, 0.75, -0.15, 0.62),
                # Synced to build_grenade_scene.py Phase 3 spit ballistic launch:
                # v0=(-0.65, +0.85, +0.42) m/s, g=(0, 0, -4.2) m/s^2 from f=65 mouth release
                ("SpitPin_f68", (-0.081, 0.216, 1.560), 0.75, -0.15, 0.62),
                ("SpitPinExit", (-0.190, 0.358, 1.484), 0.85, -0.15, 0.62),
            ],
            [
                ("HeroEyes", "GrenadeInHand", 0.40),
                ("HeroMouth", "SpitPin_f68", 0.15),
                ("HeroMouth", "SpitPinExit", 0.40),
            ],
            None,
        ),
        (
            "B1/2",
            134,
            "Hero Throw Release: Dynamic 35mm Action Frame",
            [
                ("HeroEyes", HERO_EYE_LOCAL, 0.78, -0.25, 0.60),
                ("HeroCrown", (0.0, 0.06, 1.80), 0.80, -0.10, 0.78),
                ("ThrowRelease", HERO_THROW_RELEASE_LOCAL, 0.78, -0.25, 0.60),
            ],
            [("HeroEyes", "ThrowRelease", 0.60)],
            None,
        ),
        (
            "B2",
            195,
            "Mid-Air Bullet-Time X-Ray: Spring, Spoon & 3D Labels",
            [
                ("GrenadeCore", g195, 0.35, -0.25, 0.60),
                ("GrenadeBottom", (g195[0], g195[1], g195[2] - R_OUTER_REAL), 0.45, -0.52, 0.40),
                ("FuzeTop", (g195[0], g195[1], g195[2] + 0.070), 0.45, 0.05, 0.72),
                ("LabelLeft", (g195[0] - 0.055, g195[1], g195[2] + 0.035), 0.86, -0.20, 0.75),
                ("LabelRight", (g195[0] + 0.055, g195[1], g195[2] + 0.035), 0.86, -0.20, 0.75),
            ],
            [],
            ("GrenadeCore", 0.45, 0.68),
        ),
        (
            "B3",
            294,
            "Striker Hammer Slams Primer Cap (Macro Punch-In)",
            [
                ("GrenadeCore", g294, 0.35, -0.25, 0.60),
                ("GrenadeBottom", (g294[0], g294[1], g294[2] - R_OUTER_REAL), 0.45, -0.54, 0.35),
                ("PrimerImpact", (g294[0], g294[1], g294[2] + 0.055), 0.40, 0.05, 0.65),
                ("StrikerHammer", (g294[0], g294[1], g294[2] + 0.065), 0.45, 0.10, 0.72),
                ("SpoonFlying", (g294[0] + 0.045, g294[1], g294[2] + 0.035), 0.88, -0.20, 0.72),
            ],
            [],
            ("GrenadeCore", 0.50, 0.78),
        ),
        (
            "B4",
            420,
            "4-Second Mid-Air Delay Column Burn & 3D Timer",
            [
                ("GrenadeCore", g420, 0.35, -0.25, 0.60),
                ("GrenadeBottom", (g420[0], g420[1], g420[2] - R_OUTER_REAL), 0.45, -0.52, 0.40),
                ("DelayBurnMid", (g420[0], g420[1], g420[2] + 0.020), 0.35, -0.15, 0.60),
                ("TimerCallout", (g420[0], g420[1], g420[2] + 0.082), 0.50, 0.20, 0.82),
            ],
            [],
            ("GrenadeCore", 0.45, 0.68),
        ),
        (
            "B5",
            540,
            "Bunker Ground-Level Macro: Glowing Grenade + Sandbags + Diving Sentry",
            [
                ("GrenadeCore", g540, 0.35, -0.25, 0.60),
                ("GrenadeBottom", (g540[0], g540[1], g540[2] - R_OUTER_REAL), 0.45, -0.53, 0.35),
                ("FuzeTop", (g540[0], g540[1], g540[2] + 0.070), 0.45, -0.10, 0.60),
                ("SandbagWall", (0.18, SANDBAG_WALL_Y, 0.35), 0.82, 0.15, 0.85),
                ("SentryR_Dive", (SENTRY_RIGHT_POS[0], SENTRY_RIGHT_POS[1], 0.65), 0.85, 0.15, 0.88),
            ],
            [],
            ("GrenadeCore", 0.42, 0.65),
        ),
        (
            "B6",
            665,
            "Epic Wide Bullet-Time Explosion Climax (Mid-Orbit)",
            [
                ("GroundZero", BUNKER_DEFAULT_POS, 0.55, -0.25, 0.60),
                ("FireballCore", (0.0, 7.35, 0.85), 0.55, -0.15, 0.60),
                ("FireballTop", (0.0, 7.35, 1.85), 0.65, 0.10, 0.85),
                ("SandbagLeft", (-0.95, SANDBAG_WALL_Y, 0.55), 0.88, -0.35, 0.65),
                ("SandbagRight", (0.95, SANDBAG_WALL_Y, 0.55), 0.88, -0.35, 0.65),
                ("SentryL_Dive", (SENTRY_LEFT_POS[0], SENTRY_LEFT_POS[1], 0.75), 0.85, -0.25, 0.65),
                ("SentryR_Dive", (SENTRY_RIGHT_POS[0], SENTRY_RIGHT_POS[1], 0.75), 0.85, -0.25, 0.65),
                ("ShockwaveL", (-0.95, 7.25, 0.45), 0.90, -0.35, 0.55),
                ("ShockwaveR", (0.95, 7.25, 0.45), 0.90, -0.35, 0.55),
            ],
            [("SentryL_Dive", "SentryR_Dive", 0.75)],
            None,
        ),
        (
            "B6-Start",
            634,
            "Wide Explosion Orbit Entry (-1.45, 3.85, 1.25)",
            [
                ("GroundZero", BUNKER_DEFAULT_POS, 0.55, -0.25, 0.60),
                ("FireballCore", (0.0, 7.35, 0.85), 0.55, -0.15, 0.60),
                ("SentryL_Dive", (SENTRY_LEFT_POS[0], SENTRY_LEFT_POS[1], 0.75), 0.90, -0.25, 0.65),
                ("SentryR_Dive", (SENTRY_RIGHT_POS[0], SENTRY_RIGHT_POS[1], 0.75), 0.85, -0.25, 0.65),
            ],
            [],
            None,
        ),
        (
            "B6-End",
            720,
            "Wide Explosion Orbit Exit (+1.85, 4.40, 1.55)",
            [
                ("GroundZero", BUNKER_DEFAULT_POS, 0.55, -0.35, 0.60),
                ("FireballCore", (0.0, 7.35, 0.85), 0.55, -0.20, 0.60),
                ("SentryL_Dive", (SENTRY_LEFT_POS[0], SENTRY_LEFT_POS[1], 0.75), 0.90, -0.25, 0.65),
                ("SentryR_Dive", (SENTRY_RIGHT_POS[0], SENTRY_RIGHT_POS[1], 0.75), 0.90, -0.25, 0.65),
            ],
            [],
            None,
        ),
    ]

    for beat, frame, title, pts_spec, sep_rules, macro_fill in beat_checks:
        cam_p, tgt_p, lens, fstop = camera_state_at(float(frame), include_motion_fx=True)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(cam_p, tgt_p)))
        print(
            f"\n[{beat} f={frame:03d} — {title}]\n"
            f"  cam=({cam_p[0]:+.2f}, {cam_p[1]:+.2f}, {cam_p[2]:+.2f}) "
            f"tgt=({tgt_p[0]:+.2f}, {tgt_p[1]:+.2f}, {tgt_p[2]:+.2f}) "
            f"lens={lens:.1f}mm f/{fstop:.1f} dist={dist:.2f}m"
        )
        proj_map = {}
        for label, pt, max_x, min_y, max_y in pts_spec:
            nx, ny, z = project_ndc(pt, cam_p, tgt_p, lens)
            proj_map[label] = (nx, ny, z)
            ok = (z > 0.05) and (abs(nx) <= max_x) and (min_y <= ny <= max_y)
            if not ok:
                errors += 1
            print(
                f"  [{'PASS' if ok else 'FAIL'}] {label:16s} -> "
                f"NDC=({nx:+.2f}, {ny:+.2f}) depth={z:.2f}m "
                f"(limits |X|<={max_x:.2f}, Y in [{min_y:+.2f}, {max_y:+.2f}])"
            )

        for a_k, b_k, min_sep in sep_rules:
            sep = abs(proj_map[a_k][0] - proj_map[b_k][0])
            s_ok = sep >= min_sep
            if not s_ok:
                errors += 1
            print(
                f"  [{'PASS' if s_ok else 'FAIL'}] SEPARATION {a_k} <-> {b_k}: "
                f"|dX|={sep:.2f} (min {min_sep:.2f})"
            )

        if macro_fill is not None:
            key_name, min_fill, max_fill = macro_fill
            depth_g = proj_map[key_name][2]
            fill = _screen_width_fraction(R_OUTER_REAL, depth_g, lens)
            f_ok = min_fill <= fill <= max_fill
            if not f_ok:
                errors += 1
            print(
                f"  [{'PASS' if f_ok else 'FAIL'}] MACRO FILL ({key_name}, 8.4cm dia): "
                f"{fill * 100.0:.1f}% screen width (target {min_fill * 100:.0f}%..{max_fill * 100:.0f}%)"
            )

    # ------------------------------------------------------------------
    # 2b. BEAT 1 HEROIC LOW-ANGLE, PRIME PUSH & KINETIC FX AUDIT
    # ------------------------------------------------------------------
    b1_base_1 = camera_state_at(1.0, include_motion_fx=False)
    b1_base_26 = camera_state_at(26.0, include_motion_fx=False)
    b1_base_68 = camera_state_at(68.0, include_motion_fx=False)
    b1_base_104 = camera_state_at(104.0, include_motion_fx=False)
    b1_base_134 = camera_state_at(134.0, include_motion_fx=False)
    b1_fx_34 = camera_state_at(34.0, include_motion_fx=True)
    b1_base_34 = camera_state_at(34.0, include_motion_fx=False)
    b1_fx_66 = camera_state_at(66.0, include_motion_fx=True)
    b1_base_66 = camera_state_at(66.0, include_motion_fx=False)

    low_angle_ok = (
        1.435 <= b1_base_1[0][2] <= 1.445
        and 1.475 <= b1_base_104[0][2] <= 1.485
        and all(
            camera_state_at(float(f), include_motion_fx=False)[1][2]
            > camera_state_at(float(f), include_motion_fx=False)[0][2]
            for f in range(1, 109)
        )
    )
    lens_curve_ok = (
        abs(b1_base_1[2] - 48.0) < 1e-4
        and abs(b1_base_26[2] - 56.0) < 1e-4
        and abs(b1_base_68[2] - 56.0) < 1e-4
        and abs(b1_base_104[2] - 46.0) < 1e-4
        and abs(b1_base_134[2] - 35.0) < 1e-4
    )
    rip_punch_mm = b1_fx_34[2] - b1_base_34[2]
    rip_snap_ok = 1.5 <= rip_punch_mm <= 2.5
    whip_delta_x = b1_base_66[1][0] - b1_fx_66[1][0]  # Positive => Screen-Right (-X) whip lead
    whip_ok = whip_delta_x >= 0.012

    for flag in (low_angle_ok, lens_curve_ok, rip_snap_ok, whip_ok):
        if not flag:
            errors += 1

    print(
        "\n[BEAT 1 HEROIC LOW-ANGLE, LENS CURVE & KINETIC FX AUDIT]\n"
        f"  [{'PASS' if low_angle_ok else 'FAIL'}] Low-angle Z={b1_base_1[0][2]:.3f}m -> {b1_base_104[0][2]:.3f}m "
        f"(subtly looking upward across all f=1..108)\n"
        f"  [{'PASS' if lens_curve_ok else 'FAIL'}] Prime push: "
        f"f1={b1_base_1[2]:.1f}mm -> f26..68={b1_base_26[2]:.1f}..{b1_base_68[2]:.1f}mm -> "
        f"f104={b1_base_104[2]:.1f}mm -> f134={b1_base_134[2]:.1f}mm\n"
        f"  [{'PASS' if rip_snap_ok else 'FAIL'}] f=34 pin-rip micro-snap punch: +{rip_punch_mm:.2f}mm lens kick\n"
        f"  [{'PASS' if whip_ok else 'FAIL'}] f=66 spit-flick kinetic whip-accent: "
        f"Screen-Right tgt lead={whip_delta_x * 1000.0:.1f}mm"
    )

    # ------------------------------------------------------------------
    # 3. FULL 720-FRAME ANTI-GIMBAL-FLIP & CONTINUITY AUDIT
    # ------------------------------------------------------------------
    max_abs_pitch_sin = 0.0
    min_dist = 999.0
    for f in range(1, 721):
        c_p, t_p, l_mm, fs = camera_state_at(float(f), include_motion_fx=True)
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(c_p, t_p)))
        fwd = _norm3(_sub3(t_p, c_p))
        max_abs_pitch_sin = max(max_abs_pitch_sin, abs(fwd[2]))
        min_dist = min(min_dist, d)
        if not (math.isfinite(d) and d > 0.15 and 20.0 <= l_mm <= 85.0 and 2.0 <= fs <= 11.0):
            errors += 1
            print(f"  [FAIL] Continuity violation at f={f}: dist={d:.3f}m lens={l_mm:.1f}mm f/{fs:.1f}")

    gimbal_ok = max_abs_pitch_sin < 0.85
    if not gimbal_ok:
        errors += 1
    print(
        f"\n[720-FRAME CONTINUITY & GIMBAL-LOCK AUDIT]\n"
        f"  [{'PASS' if gimbal_ok else 'FAIL'}] min_target_dist={min_dist:.3f}m "
        f"max_|fwd.z|={max_abs_pitch_sin:.3f} (< 0.85 guaranteed zero TRACK_TO flip!)"
    )

    if errors:
        print(f"\nRESULT: FAILED — {errors} framing/continuity violation(s) detected!")
        return 1

    print(
        "\nRESULT: 100% PASSED — All 6 Beats + Beat 6 Orbit + 720-Frame Gimbal Audit "
        "CLEAR of frame edges & bottom subtitle safe-zone!"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
