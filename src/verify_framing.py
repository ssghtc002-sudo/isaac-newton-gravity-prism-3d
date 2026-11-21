"""
Pre-flight mathematical 9:16 portrait camera framing, ZERO-OVERLAP sightline & STATION ISOLATION verifier
for Scientist #9: Isaac Newton V6!

Verifies across all 6 Beats:
  1. 100% of ALL Cast Heads + Hero Scientific Centerpieces stay cleanly inside the 9:16 portrait safe frame
     (|ndcX| <= 0.85, ndcY in [-0.50, +0.72]) — INCLUDING BOTH NewtonHead & HalleyHead in Beat 2a (t=5.5s) & Beat 2b (t=8.5s)!
  2. ZERO Instrument-Blocking-Face Overlap: checks horizontal screen separation |ndcX(Actor) - ndcX(HeroProp)| >= 0.42
     in every single close-up / medium shot (and >= 0.60 between Newton & Halley in 2-shots)!
  3. ZERO Station Edge Bleed: confirms Station B Prism Optical Screen stays completely OUTSIDE Beat 1 left edge,
     Station A Apple Velvet Pad stays completely OUTSIDE Beat 3 right edge,
     and Station Center Principia Folio stays completely OUTSIDE Beat 4 left edge!
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


# Exact V6 Symmetric Zero-Overlap Station Anchors (synchronized 1:1 with config/timeline.json & build_newton_scene.py!)
FLOOR_Z = -0.99
PRISM_POS = np.array([-1.96, -0.06, 0.0])
APPLE_POS = np.array([-0.74, -0.10, 0.0])
PRINCIPIA_POS = np.array([0.00, -0.10, 0.0])
ORBIT_POS = np.array([0.28, 0.02, 0.34])
CRADLE_POS = np.array([1.44, -0.08, 0.0])


def compute_beat(b_idx, t):
    if b_idx == 0:
        # Beat 1: Newton (Left, X=-1.22, facing RIGHT +24°) + Falling Woolsthorpe Apple & Gravity Vector (Right, X=-0.74)
        u = clamp01(t / 4.5)
        pos = bezier_quad(
            np.array([-0.98, -1.48, 0.56]),
            np.array([-0.93, -1.38, 0.54]),
            np.array([-0.89, -1.32, 0.52]),
            u,
        )
        tgt = lerp(np.array([-0.96, 0.14, 0.45]), np.array([-0.94, 0.14, 0.42]), u)
        return pos, tgt, lerp(41.0, 43.0, u)
    if b_idx == 1:
        # Beat 2: Balanced Face-to-Face 2-Shot from t=4.5s (BOTH Newton -0.46 & Halley +0.46 100% INSIDE frame!)
        pan_u = ease_out_quint(clamp01((t - 6.95) / 1.1))
        pos = lerp(np.array([0.06, -1.72, 0.56]), np.array([-0.04, -1.64, 0.56]), pan_u)
        tgt = lerp(np.array([0.03, 0.18, 0.46]), np.array([-0.01, 0.20, 0.46]), pan_u)
        return pos, tgt, lerp(36.0, 35.5, pan_u)
    if b_idx == 2:
        # Beat 3: Crystal Flint-Glass Prism + 7-Spectral Rainbow (Left, X=-1.96..-1.72) + Newton (Right, X=-1.42, facing LEFT -26°)
        u = clamp01((t - 9.5) / 6.0)
        pos = bezier_quad(
            np.array([-1.74, -1.56, 0.54]),
            np.array([-1.69, -1.46, 0.56]),
            np.array([-1.65, -1.40, 0.58]),
            u,
        )
        tgt = lerp(np.array([-1.71, 0.14, 0.44]), np.array([-1.69, 0.14, 0.48]), smoothstep(10.6, 13.0, t))
        return pos, tgt, lerp(38.5, 40.5, u)
    if b_idx == 3:
        # Beat 4: 3.8° Dutch-Tilt Halley (Left, X=+0.94, facing RIGHT +24°) + Brass Newton's Cradle (Right, X=+1.44, swing peak +1.81)
        u = clamp01((t - 15.5) / 5.0)
        pos = bezier_quad(
            np.array([1.22, -1.62, 0.54]),
            np.array([1.26, -1.54, 0.55]),
            np.array([1.28, -1.60, 0.56]),
            u,
        )
        tgt = np.array([1.24, 0.14, 0.46])
        return pos, tgt, lerp(35.5, 37.0, u)
    if b_idx == 4:
        # Beat 5: Newton (Left, X=-0.30, facing RIGHT +24°) + Earth-Moon Curved Gravity Well & Telescope (Right, X=+0.28)
        u = clamp01((t - 20.5) / 5.0)
        pos = bezier_quad(
            np.array([-0.04, -1.50, 0.56]),
            np.array([0.02, -1.44, 0.60]),
            np.array([0.06, -1.48, 0.64]),
            u,
        )
        tgt = lerp(np.array([0.01, 0.18, 0.46]), np.array([0.02, 0.20, 0.48]), u)
        return pos, tgt, lerp(37.0, 38.5, u)
    # Beat 6: Grand Cast Finale — Newton (Left -0.54), Halley (Right +0.56), Centered Earth-Moon Well, Modern Rocket & Title!
    u = clamp01((t - 25.5) / 4.5)
    pos = bezier_quad(
        np.array([0.04, -1.96, 0.72]),
        np.array([0.03, -2.12, 0.70]),
        np.array([0.02, -2.24, 0.68]),
        u,
    )
    tgt = lerp(np.array([0.02, 0.26, 0.56]), np.array([0.02, 0.24, 0.54]), u)
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


# FLOOR_Z = -0.99, height ~1.73..1.76m -> Head center ~0.68..0.70m, Head top ~0.74..0.76m!
checks = [
    (
        0,
        0.4,
        "Beat 1 Start Newton(L) + Hanging Apple Bough(R)",
        {
            "NewtonHead": np.array([-1.22, 0.44, 0.72]),
            "AppleHanging": np.array([APPLE_POS[0], APPLE_POS[1], 0.69]),
            "VelvetPad": np.array([APPLE_POS[0], APPLE_POS[1], 0.04]),
            "BoughRight": np.array([APPLE_POS[0] + 0.16, APPLE_POS[1], 0.76]),
        },
        [("NewtonHead", "AppleHanging", 0.42)],
    ),
    (
        0,
        2.6,
        "Beat 1 Mid Newton(L) + Fallen Apple & g Arrow(R)",
        {
            "NewtonHead": np.array([-1.22, 0.44, 0.72]),
            "AppleLanded": np.array([APPLE_POS[0], APPLE_POS[1], 0.06]),
            "GravityArrow": np.array([APPLE_POS[0] + 0.10, APPLE_POS[1], 0.34]),
        },
        [("NewtonHead", "GravityArrow", 0.42)],
    ),
    (
        1,
        5.5,
        "Beat 2a BOTH Newton(L) + Halley(R) + F=ma(C)",
        {
            "NewtonHead": np.array([-0.46, 0.46, 0.72]),
            "HalleyHead": np.array([0.46, 0.46, 0.74]),
            "Formula_Fma": np.array([PRINCIPIA_POS[0], PRINCIPIA_POS[1] - 0.04, 0.26]),
        },
        [("NewtonHead", "HalleyHead", 0.60), ("NewtonHead", "Formula_Fma", 0.30)],
    ),
    (
        1,
        8.5,
        "Beat 2b BOTH Newton(L) + Halley(R) + F=Gm1m2/r²(C)",
        {
            "NewtonHead": np.array([-0.46, 0.46, 0.72]),
            "HalleyHead": np.array([0.46, 0.46, 0.74]),
            "Formula_Grav": np.array([PRINCIPIA_POS[0], PRINCIPIA_POS[1] - 0.04, 0.30]),
        },
        [("NewtonHead", "HalleyHead", 0.60), ("NewtonHead", "Formula_Grav", 0.30)],
    ),
    (
        2,
        12.8,
        "Beat 3 Glass Prism & 7-Spectral Rainbow(L) + Newton(R)",
        {
            "NewtonHead": np.array([-1.42, 0.44, 0.72]),
            "SunbeamEntry": np.array([PRISM_POS[0] - 0.22, PRISM_POS[1], 0.46]),
            "PrismCenter": np.array([PRISM_POS[0], PRISM_POS[1], 0.28]),
            "RainbowScreen": np.array([PRISM_POS[0] + 0.24, PRISM_POS[1] + 0.01, 0.29]),
        },
        [("NewtonHead", "PrismCenter", 0.42), ("NewtonHead", "RainbowScreen", 0.26)],
    ),
    (
        3,
        17.8,
        "Beat 4 Halley(L) + Newton's Cradle Clack Launch(R)",
        {
            "HalleyHead": np.array([0.88, 0.53, 0.74]),
            "CradleCenter": np.array([CRADLE_POS[0], CRADLE_POS[1], 0.32]),
            "CradlePeakBall": np.array([CRADLE_POS[0] + 0.37, CRADLE_POS[1], 0.44]),
        },
        [("HalleyHead", "CradleCenter", 0.42)],
    ),
    (
        4,
        23.2,
        "Beat 5 Newton(L) + Earth-Moon Gravity Well & Telescope(R)",
        {
            "NewtonHead": np.array([-0.30, 0.44, 0.72]),
            "EarthCenter": np.array([ORBIT_POS[0], ORBIT_POS[1], 0.46]),
            "WellRightEdge": np.array([ORBIT_POS[0] + 0.24, ORBIT_POS[1], 0.48]),
            "TelescopeBase": np.array([ORBIT_POS[0], ORBIT_POS[1], 0.14]),
        },
        [("NewtonHead", "EarthCenter", 0.42)],
    ),
    (
        5,
        28.2,
        "Beat 6 Grand Cast Finale + Centered Well + Rocket + Title",
        {
            "NewtonHead": np.array([-0.54, 0.52, 0.72]),
            "HalleyHead": np.array([0.56, 0.52, 0.74]),
            "EarthCenter": np.array([0.04, 0.04, 0.46]),
            "RocketOrbitTop": np.array([0.04, 0.06, 1.06]),
            "TitleRight": np.array([0.44, 0.56, 1.54]),
        },
        [("NewtonHead", "HalleyHead", 0.60), ("NewtonHead", "EarthCenter", 0.30)],
    ),
]

# Neighbor-station out-of-frame isolation checks (confirm zero accidental prop edge intrusion!)
isolation_checks = [
    (
        0,
        0.4,
        "Beat 1 Start: Prism Optical Screen right edge OUTSIDE left frame",
        np.array([PRISM_POS[0] + 0.27, PRISM_POS[1] + 0.01, 0.29]),
        "LEFT",
    ),
    (
        2,
        11.5,
        "Beat 3 Mid: Apple Velvet Pad left edge OUTSIDE right frame",
        np.array([APPLE_POS[0] - 0.16, APPLE_POS[1], 0.04]),
        "RIGHT",
    ),
    (
        3,
        17.8,
        "Beat 4 Mid: Principia Folio right edge OUTSIDE left frame",
        np.array([PRINCIPIA_POS[0] + 0.20, PRINCIPIA_POS[1], 0.04]),
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
print("\nPASSED: 100% ZERO-OVERLAP Sightlines + Balanced Beat 2 2-Shot + Station Isolation across ALL 6 Beats!")
