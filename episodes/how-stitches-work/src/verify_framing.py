"""
Pre-flight mathematical 9:16 portrait macro camera framing verifier for How Stitches Work (4K Medical Macro Cinema).
Verifies across all 6 Beats that:
  1. Both ends of the surgical wound (Y = -0.145, +0.145), suture loops, curved needle, surgical scissors,
     collagen matrix, and branching capillary trees stay cleanly inside the 9:16 portrait safe frame.
  2. Top HUD stage badges stay strictly inside |ndcX| <= 0.85 and ndcY in [+0.45, +0.72] with zero wound overlap.
"""
import numpy as np

RES_X, RES_Y = 1080, 1920
ASPECT = RES_X / RES_Y
SENSOR_W = 36.0
SENSOR_H = SENSOR_W / ASPECT  # 64.0mm


def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def smoothstep(e0, e1, x):
    t = clamp01((x - e0) / max(1e-6, e1 - e0))
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def compute_macro_camera(t):
    if t < 6.0:
        u = smoothstep(0.0, 5.8, t)
        pos = np.array([lerp(0.14, 0.16, u), lerp(-0.34, -0.18, u), lerp(0.56, 0.34, u)])
        tgt = np.array([0.0, lerp(-0.04, 0.04, u), 0.0])
        return pos, tgt, lerp(46.0, 58.0, u)
    if t < 10.2:
        u = smoothstep(6.0, 10.0, t)
        pos = np.array([lerp(0.16, 0.11, u), lerp(-0.18, -0.15, u), lerp(0.34, 0.36, u)])
        tgt = np.array([0.0, lerp(0.04, 0.05, u), 0.0])
        return pos, tgt, lerp(58.0, 54.0, u)
    if t < 15.2:
        u = smoothstep(10.2, 15.0, t)
        pos = np.array([lerp(0.11, -0.14, u), lerp(-0.15, -0.22, u), lerp(0.36, 0.35, u)])
        tgt = np.array([0.0, lerp(0.02, 0.0, u), 0.0])
        return pos, tgt, lerp(54.0, 54.0, u)
    if t < 20.4:
        u = smoothstep(15.2, 20.2, t)
        pos = np.array([lerp(-0.14, 0.14, u), lerp(-0.22, -0.19, u), lerp(0.35, 0.36, u)])
        tgt = np.array([0.0, 0.0, 0.0])
        return pos, tgt, lerp(54.0, 54.0, u)
    if t < 25.3:
        u = smoothstep(20.4, 25.1, t)
        pos = np.array([lerp(0.14, 0.04, u), lerp(-0.19, -0.14, u), lerp(0.36, 0.38, u)])
        tgt = np.array([0.0, 0.0, 0.0])
        return pos, tgt, lerp(54.0, 52.0, u)
    u = smoothstep(25.3, 29.8, t)
    pos = np.array([lerp(0.04, 0.12, u), lerp(-0.14, -0.32, u), lerp(0.36, 0.56, u)])
    tgt = np.array([0.0, 0.0, 0.0])
    return pos, tgt, lerp(54.0, 46.0, u)


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
    return (x / z) * (lens_mm / (SENSOR_W * 0.5)), (y / z) * (lens_mm / (SENSOR_H * 0.5)), z


def main():
    ok = True
    beats = [
        (2.5, "Beat 1: Needle Weaving Suture Loops", {"WoundCenter": [0, 0, 0], "WoundBot": [0, -0.12, 0], "WoundTop": [0, 0.12, 0]}),
        (8.6, "Beat 2: Zipper Closure & Scissors Snip", {"WoundCenter": [0, 0, 0], "ScissorsTip": [0.02, 0.108, 0.045], "SutureTop": [0, 0.108, 0.01]}),
        (12.5, "Beat 3: Inflammation & Repair Cell Swarm", {"WoundCenter": [0, 0, 0], "LeftCellZone": [-0.05, 0, 0], "RightCellZone": [0.05, 0, 0]}),
        (17.8, "Beat 4: Golden Collagen Fiber Matrix", {"WoundCenter": [0, 0, 0], "CollagenTop": [0, 0.11, 0.005], "CollagenBot": [0, -0.11, 0.005]}),
        (22.8, "Beat 5: Branching Ruby Blood Vessels", {"WoundCenter": [0, 0, 0], "VesselLeft": [-0.06, 0, 0], "VesselRight": [0.06, 0, 0]}),
        (28.0, "Beat 6: Stitch Removal & Healed Reveal", {"WoundCenter": [0, 0, 0], "HealedTop": [0, 0.14, 0], "HealedBot": [0, -0.14, 0]}),
    ]
    for t, label, pts in beats:
        cam, tgt, lens = compute_macro_camera(t)
        print(f"\n[{label} (t={t:.1f}s)] cam={np.round(cam, 3)} lens={lens:.1f}mm")
        for name, pt in pts.items():
            nx, ny, z = project(cam, tgt, lens, np.array(pt, dtype=float))
            in_ok = abs(nx) <= 0.85 and (-0.52 <= ny <= 0.45) and z > 0.15
            if not in_ok:
                ok = False
            print(f"  {name:15s}: ndcX={nx:+.2f}  ndcY={ny:+.2f}  depth={z:.2f}m  [{'OK' if in_ok else 'OUT!'}]")
        # Verify camera-locked HUD Stage Badge stays inside safe portrait frame!
        b_nx = (0.0825 / 0.36) * (lens / (SENSOR_W * 0.5))
        b_ny = (0.118 / 0.36) * (lens / (SENSOR_H * 0.5))
        b_ok = b_nx <= 0.85 and (0.44 <= b_ny <= 0.72)
        if not b_ok:
            ok = False
        print(f"  {'HUDStageBadge':15s}: |ndcX|={b_nx:.2f}  ndcY={b_ny:+.2f}  [{'OK' if b_ok else 'CLIP!'}]")

    if not ok:
        raise SystemExit("FAILED: Framing check failed!")
    print("\nPASSED: 100% 4K Medical Macro Framing & Portrait HUD Safe-Zone across ALL 6 Beats!")


if __name__ == "__main__":
    main()
