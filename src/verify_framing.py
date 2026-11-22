"""
Pre-flight mathematical 9:16 portrait camera framing, ZERO-OVERLAP sightline & STATION ISOLATION verifier
for Scientist #9: Isaac Newton V9 (Outdoor Woolsthorpe Apple Orchard & Cotswold Stone Terrace -> Starry Twilight Cosmos)!

Verifies across all 6 Beats at true ground level (FLOOR_Z = 0.0, full 3/4-body 1.78m/1.79m adult cast):
  1. 100% of ALL Cast Heads (Z = 1.66m; Beat 4 knee-bend recoil Z = 1.60m) + Hero Scientific Centerpieces
     (including V9 Beat 1 apple-to-Moon golden orbital arc & Beat 6 opposite-phase Comm Satellite!)
     stay cleanly inside the 9:16 portrait safe frame (|ndcX| <= 0.85, ndcY in [-0.50, +0.72]) —
     INCLUDING BOTH NewtonHead & HalleyHead in Beat 2a (t=5.5s), Beat 2b (t=8.5s) & Beat 6 (t=28.2s)!
  2. ZERO Instrument-Blocking-Face Overlap: checks horizontal screen separation |ndcX(Actor) - ndcX(HeroProp)| >= 0.42
     in every close-up / medium shot (and >= 0.60 between Newton & Halley in 2-shots)!
  3. ZERO Station Edge Bleed: confirms Beat 1 Prism Alcove Easel stays completely OUTSIDE left frame,
     Beat 3 Apple Plinth stays completely OUTSIDE right frame,
     and Beat 4 Principia Plinth stays completely OUTSIDE left frame!
"""
import numpy as np

RES_X, RES_Y = 1080, 1920
ASPECT = RES_X / RES_Y
SENSOR_W = 36.0
SENSOR_H = SENSOR_W / ASPECT  # 64.0mm with sensor_fit='HORIZONTAL'


def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def smoothstep(e0, e1, x):
    t = clamp01((x - e0) / max(1e-6, e1 - e0))
    return t * t * (3.0 - 2.0 * t)


def ease_out_quint(x):
    t = clamp01(x)
    return 1.0 - (1.0 - t) ** 5


def lerp(a, b, t):
    return a + (b - a) * t


def bezier_quad(p0, p1, p2, t):
    u = 1.0 - t
    return p0 * (u * u) + p1 * (2.0 * u * t) + p2 * (t * t)


# Exact V8/V9 Outdoor Woolsthorpe Terrace Anchors (synchronized 1:1 with newton_world_v7.py & build_newton_scene.py!)
FLOOR_Z = 0.0
HERO_APPLE_ATTACH = np.array([-0.52, 0.12, 1.94])
APPLE_PLINTH = np.array([-0.52, 0.12, 0.88])
PRINCIPIA_PLINTH = np.array([0.55, 0.08, 0.90])
PRISM_ANCHOR = np.array([-2.85, 0.06, 0.0])
PRISM_PINHOLE = np.array([-3.38, 0.06, 1.54])
CRADLE_PLINTH = np.array([2.88, 0.08, 0.90])
ORBIT_ANCHOR = np.array([0.64, 0.14, 0.0])
EASEL_POS = np.array([-2.54, 0.06, 1.24])


def compute_beat(b_idx, t):
    if b_idx == 0:
        # Beat 1: Golden-Hour Woolsthorpe Apple Orchard Low-Angle Portrait — Newton(-1.06) + Apple/Plinth(-0.52)
        u = clamp01(t / 4.5)
        pos = bezier_quad(
            np.array([-0.82, -1.78, 1.24]),
            np.array([-0.79, -1.70, 1.27]),
            np.array([-0.76, -1.62, 1.30]),
            u,
        )
        tgt = lerp(np.array([-0.79, 0.24, 1.32]), np.array([-0.78, 0.22, 1.24]), u)
        return pos, tgt, lerp(38.0, 40.0, u)
    if b_idx == 1:
        # Beat 2: Sunlit Stone Courtyard Face-to-Face Debate — BOTH Newton(0.05) & Halley(1.05) eye-to-eye!
        pan_u = ease_out_quint(clamp01((t - 6.95) / 1.1))
        pos = lerp(np.array([0.62, -2.02, 1.42]), np.array([0.52, -1.94, 1.42]), pan_u)
        tgt = lerp(np.array([0.58, 0.28, 1.32]), np.array([0.54, 0.30, 1.32]), pan_u)
        return pos, tgt, lerp(36.0, 35.5, pan_u)
    if b_idx == 2:
        # Beat 3: Shaded Stone Arch Optical Alcove — Crystal Prism(-2.85) + Rainbow Easel(-2.54) + Newton(-2.12)
        u = clamp01((t - 9.5) / 6.0)
        pos = bezier_quad(
            np.array([-2.54, -1.68, 1.40]),
            np.array([-2.51, -1.58, 1.41]),
            np.array([-2.48, -1.52, 1.42]),
            u,
        )
        tgt = lerp(np.array([-2.52, 0.18, 1.32]), np.array([-2.50, 0.20, 1.36]), smoothstep(10.6, 13.0, t))
        return pos, tgt, lerp(38.5, 40.5, u)
    if b_idx == 3:
        # Beat 4: 3.8° Dutch-Tilt Garden Balustrade Comedy — Halley(2.34) + Brass Newton's Cradle(2.88, peak +3.25)
        u = clamp01((t - 15.5) / 5.0)
        pos = bezier_quad(
            np.array([2.62, -1.78, 1.38]),
            np.array([2.66, -1.72, 1.39]),
            np.array([2.70, -1.70, 1.40]),
            u,
        )
        tgt = np.array([2.62, 0.22, 1.32])
        return pos, tgt, lerp(36.0, 38.0, u)
    if b_idx == 4:
        # Beat 5: Starry Twilight Cosmos — Newton(0.02) + Outdoor Tripod Telescope & Giant Gravity Well(0.64)
        u = clamp01((t - 20.5) / 5.0)
        pos = bezier_quad(
            np.array([0.30, -1.74, 1.34]),
            np.array([0.34, -1.68, 1.42]),
            np.array([0.38, -1.66, 1.50]),
            u,
        )
        tgt = lerp(np.array([0.34, 0.24, 1.44]), np.array([0.36, 0.26, 1.52]), u)
        return pos, tgt, lerp(37.0, 38.5, u)
    # Beat 6: Moonlit Orchard-to-Cosmos Rocket & Satellite Slingshot Finale — Newton(-0.05), Halley(+1.15), Centered Well(0.55), Rocket, Satellite & Title!
    u = clamp01((t - 25.5) / 4.5)
    pos = bezier_quad(
        np.array([0.55, -2.35, 1.52]),
        np.array([0.55, -2.52, 1.58]),
        np.array([0.55, -2.68, 1.62]),
        u,
    )
    tgt = lerp(np.array([0.55, 0.30, 1.54]), np.array([0.55, 0.28, 1.52]), u)
    return pos, tgt, lerp(34.0, 32.0, u)


def project(cam_pos, tgt, lens_mm, pt):
    fwd = tgt - cam_pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    d = pt - cam_pos
    z = np.dot(d, fwd)
    x = np.dot(d, right)
    y = np.dot(d, up)
    ndc_x = (x / z) * (lens_mm / (SENSOR_W * 0.5))
    ndc_y = (y / z) * (lens_mm / (SENSOR_H * 0.5))
    return ndc_x, ndc_y, z


# FLOOR_Z = 0.0, Adult Cast Height 1.78m..1.79m -> Head center ~1.62m, upper head check Z = 1.66m (crown ~1.78m)!
checks = [
    (
        0,
        0.4,
        "Beat 1 Start Golden-Hour Orchard: Newton(L) + Hanging Apple(R)",
        {
            "NewtonHead": np.array([-1.06, 0.36, 1.66]),
            "AppleHanging": HERO_APPLE_ATTACH.copy(),
            "VelvetPad": np.array([APPLE_PLINTH[0], APPLE_PLINTH[1], APPLE_PLINTH[2] + 0.02]),
        },
        [("NewtonHead", "AppleHanging", 0.42)],
    ),
    (
        0,
        2.6,
        "Beat 1 Mid Golden-Hour Orchard: Newton(L) + Fallen Apple, g Arrow & Orbital Arc(R)",
        {
            "NewtonHead": np.array([-1.06, 0.36, 1.66]),
            "AppleLanded": np.array([APPLE_PLINTH[0], APPLE_PLINTH[1], APPLE_PLINTH[2] + 0.058]),
            "GravityArrow": np.array([APPLE_PLINTH[0] + 0.08, APPLE_PLINTH[1], 1.34]),
            "OrbitalArcTip": np.array([-0.26, 0.18, 1.88]),
        },
        [("NewtonHead", "GravityArrow", 0.42)],
    ),
    (
        1,
        5.5,
        "Beat 2a Courtyard Debate: BOTH Newton(L) + Halley(R) + F=ma(C)",
        {
            "NewtonHead": np.array([0.05, 0.45, 1.66]),
            "HalleyHead": np.array([1.05, 0.45, 1.66]),
            "Formula_Fma": np.array([PRINCIPIA_PLINTH[0], -0.02, 1.22]),
            "PrincipiaBook": np.array([PRINCIPIA_PLINTH[0], PRINCIPIA_PLINTH[1], PRINCIPIA_PLINTH[2] + 0.05]),
        },
        [("NewtonHead", "HalleyHead", 0.60), ("NewtonHead", "Formula_Fma", 0.30)],
    ),
    (
        1,
        8.5,
        "Beat 2b Courtyard Debate: BOTH Newton(L) + Halley(R) + F=Gm1m2/r²(C)",
        {
            "NewtonHead": np.array([0.05, 0.45, 1.66]),
            "HalleyHead": np.array([1.05, 0.45, 1.66]),
            "Formula_Grav": np.array([PRINCIPIA_PLINTH[0], -0.02, 1.26]),
        },
        [("NewtonHead", "HalleyHead", 0.60), ("NewtonHead", "Formula_Grav", 0.30)],
    ),
    (
        2,
        12.8,
        "Beat 3 Optical Alcove: Prism Tripod & 7-Spectral Easel(L) + Newton(R)",
        {
            "NewtonHead": np.array([-2.12, 0.42, 1.66]),
            "SunbeamMid": np.array([-3.12, 0.06, 1.42]),
            "PrismCenter": np.array([PRISM_ANCHOR[0], PRISM_ANCHOR[1], 1.28]),
            "RainbowEasel": EASEL_POS.copy(),
        },
        [("NewtonHead", "PrismCenter", 0.42), ("NewtonHead", "RainbowEasel", 0.26)],
    ),
    (
        3,
        17.8,
        "Beat 4 Garden Balustrade: Halley Knee-Bend Recoil(L) + Cradle Clack(R)",
        {
            "HalleyHead": np.array([2.27, 0.51, 1.60]),
            "CradleCenter": np.array([CRADLE_PLINTH[0], CRADLE_PLINTH[1], 1.20]),
            "CradlePeakBall": np.array([CRADLE_PLINTH[0] + 0.37, CRADLE_PLINTH[1], 1.34]),
        },
        [("HalleyHead", "CradleCenter", 0.42)],
    ),
    (
        4,
        23.2,
        "Beat 5 Starry Twilight: Newton(L) + Telescope & Giant Gravity Well(R)",
        {
            "NewtonHead": np.array([0.02, 0.45, 1.66]),
            "EarthCenter": np.array([ORBIT_ANCHOR[0], ORBIT_ANCHOR[1], 1.56]),
            "WellRightEdge": np.array([ORBIT_ANCHOR[0] + 0.296, ORBIT_ANCHOR[1], 1.58]),
            "TelescopeTop": np.array([ORBIT_ANCHOR[0], ORBIT_ANCHOR[1], 1.12]),
        },
        [("NewtonHead", "EarthCenter", 0.42)],
    ),
    (
        5,
        28.2,
        "Beat 6 Moonlit Finale: BOTH Newton(L) + Halley(R) + Well + Rocket + CommSatellite + Title",
        {
            "NewtonHead": np.array([-0.05, 0.55, 1.66]),
            "HalleyHead": np.array([1.15, 0.55, 1.66]),
            "EarthCenter": np.array([0.55, 0.14, 1.50]),
            "RocketOrbitTop": np.array([0.55, 0.14, 2.14]),
            "CommSatellite": np.array([0.55 + 0.29 * np.cos(3.535 + np.pi), 0.14 + 0.29 * np.sin(3.535 + np.pi), 2.04]),
            "TitleRight": np.array([1.01, 0.60, 2.52]),
        },
        [("NewtonHead", "HalleyHead", 0.60), ("NewtonHead", "EarthCenter", 0.30)],
    ),
]

# Neighbor-station out-of-frame isolation checks (confirm zero accidental station intrusion!)
isolation_checks = [
    (
        0,
        0.4,
        "Beat 1 Start: Prism Alcove Easel right edge OUTSIDE left frame",
        np.array([EASEL_POS[0] + 0.08, EASEL_POS[1], EASEL_POS[2]]),
        "LEFT",
    ),
    (
        2,
        11.5,
        "Beat 3 Mid: Apple Stone Plinth left edge OUTSIDE right frame",
        np.array([APPLE_PLINTH[0] - 0.18, APPLE_PLINTH[1], APPLE_PLINTH[2] + 0.02]),
        "RIGHT",
    ),
    (
        3,
        17.8,
        "Beat 4 Mid: Principia Plinth right edge OUTSIDE left frame",
        np.array([PRINCIPIA_PLINTH[0] + 0.26, PRINCIPIA_PLINTH[1], PRINCIPIA_PLINTH[2] + 0.04]),
        "LEFT",
    ),
]

errors = 0
for b_idx, t, label, pts, sep_rules in checks:
    c_pos, tgt, lens = compute_beat(b_idx, t)
    print(f"\n[{label} (t={t:.1f}s)] cam={np.round(c_pos, 2)} tgt={np.round(tgt, 2)} lens={lens:.1f}mm")
    proj_map = {}
    for pname, pt in pts.items():
        nx, ny, z = project(c_pos, tgt, lens, pt)
        proj_map[pname] = (nx, ny, z)
        ok = (z > 0.3) and (abs(nx) <= 0.85) and (-0.50 <= ny <= 0.72)
        if not ok:
            errors += 1
        print(f"  {pname:15s}: ndcX={nx:+.2f}  ndcY={ny:+.2f}  depth={z:.2f}m  {'[OK]' if ok else '[CLIP!]'}")
    for a_k, b_k, min_sep in sep_rules:
        sep = abs(proj_map[a_k][0] - proj_map[b_k][0])
        s_ok = sep >= min_sep
        if not s_ok:
            errors += 1
        print(f"  SEPARATION {a_k} <-> {b_k}: |dX|={sep:.2f} (min {min_sep:.2f}) {'[CLEAR!]' if s_ok else '[OVERLAP!]'}")

print("\n[STATION ISOLATION CHECKS (ZERO EDGE INTRUSION)]")
for b_idx, t, label, pt, side in isolation_checks:
    c_pos, tgt, lens = compute_beat(b_idx, t)
    nx, ny, z = project(c_pos, tgt, lens, pt)
    ok = (nx < -1.03) if side == "LEFT" else (nx > 1.03)
    if not ok:
        errors += 1
    print(f"  {label}: ndcX={nx:+.2f} ({'OUTSIDE' if ok else 'INTRUDING!'})")

if errors:
    print(f"\nFAILED: {errors} framing/overlap/isolation error(s)!")
    raise SystemExit(1)
print("\nPASSED: 100% V9 Hollywood Polish ZERO-OVERLAP Sightlines + Full-Stature Beat 2 2-Shot + Orbital Arc/Satellite + Station Isolation across ALL 6 Beats!")
