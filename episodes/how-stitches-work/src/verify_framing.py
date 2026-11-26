"""
Pre-flight mathematical 9:16 portrait macro camera framing verifier for How Stitches Work V2
(Real MakeHuman 4K Lower Leg + Ankle + 5 Toes + 3D Multi-Layer Shin Incision).
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


def compute_v2_camera(t):
    if t < 6.0:
        u_zoom = smoothstep(0.15, 2.10, t)
        u_pan = smoothstep(1.20, 5.70, t)
        pos = np.array([
            lerp(0.16, 0.095, u_zoom),
            lerp(-0.44, -0.155, u_zoom) + lerp(-0.015, 0.025, u_pan),
            lerp(0.42, 0.215, u_zoom),
        ])
        tgt = np.array([
            lerp(-0.010, 0.0, u_zoom),
            lerp(-0.065, 0.0, u_zoom) + lerp(-0.018, 0.018, u_pan),
            lerp(0.005, 0.0, u_zoom),
        ])
        return pos, tgt, lerp(48.0, 62.0, u_zoom)
    if t < 10.2:
        u = smoothstep(6.0, 10.0, t)
        pos = np.array([lerp(0.095, 0.082, u), lerp(-0.130, -0.115, u), lerp(0.215, 0.205, u)])
        tgt = np.array([0.0, lerp(0.018, 0.012, u), 0.0])
        return pos, tgt, lerp(62.0, 64.0, u)
    if t < 15.2:
        u = smoothstep(10.2, 15.0, t)
        pos = np.array([lerp(0.082, -0.075, u), lerp(-0.115, -0.125, u), lerp(0.205, 0.178, u)])
        tgt = np.array([0.0, lerp(0.012, -0.004, u), -0.002])
        return pos, tgt, lerp(64.0, 68.0, u)
    if t < 20.4:
        u = smoothstep(15.2, 20.2, t)
        pos = np.array([lerp(-0.075, 0.078, u), lerp(-0.125, -0.118, u), lerp(0.178, 0.185, u)])
        tgt = np.array([0.0, 0.0, -0.002])
        return pos, tgt, lerp(68.0, 66.0, u)
    if t < 25.3:
        u = smoothstep(20.4, 25.1, t)
        pos = np.array([lerp(0.078, 0.055, u), lerp(-0.118, -0.130, u), lerp(0.185, 0.215, u)])
        tgt = np.array([0.0, 0.0, 0.0])
        return pos, tgt, lerp(66.0, 60.0, u)
    u_pull = smoothstep(27.0, 29.8, t)
    pos = np.array([lerp(0.055, 0.155, u_pull), lerp(-0.130, -0.430, u_pull), lerp(0.215, 0.415, u_pull)])
    tgt = np.array([lerp(0.0, -0.010, u_pull), lerp(0.0, -0.060, u_pull), lerp(0.0, 0.005, u_pull)])
    return pos, tgt, lerp(60.0, 48.0, u_pull)


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
    checks = [
        (0.29, "Beat 1a Hook: Full Lower Leg + Ankle + Foot", {"ShinWound": [0, 0, 0], "FootToes": [-0.02, -0.25, 0.07], "UpperShin": [0, 0.14, 0]}),
        (3.12, "Beat 1b Macro: Needle & Forceps Suturing", {"WoundCenter": [0, 0, 0], "StitchBot": [0, -0.04, 0], "StitchTop": [0, 0.04, 0]}),
        (8.54, "Beat 2: Corset Zipper Closure & Iris Scissors Snip", {"WoundCenter": [0, 0, 0], "TopKnot": [0.012, 0.04, 0], "ScissorsPivot": [0.0335, 0.04, 0.014]}),
        (12.95, "Beat 3: X-Ray Dermal Cutaway & Sculpted Cells", {"WoundCenter": [0, 0, -0.003], "LeftWall": [-0.016, 0, -0.003], "RightWall": [0.016, 0, -0.003]}),
        (17.80, "Beat 4: Double-Helix Golden Collagen Web", {"WoundCenter": [0, 0, -0.003], "ColTop": [0, 0.044, -0.003], "ColBot": [0, -0.044, -0.003]}),
        (22.65, "Beat 5: Flanking Arterioles & Capillary Sprouts", {"WoundCenter": [0, 0, -0.003], "LeftTrunk": [-0.016, 0, -0.003], "RightTrunk": [0.016, 0, -0.003]}),
        (29.33, "Beat 6: Full Healed Leg & Foot Pullback", {"HealedShin": [0, 0, 0], "FootToes": [-0.02, -0.25, 0.07], "UpperShin": [0, 0.14, 0]}),
    ]
    for t, label, pts in checks:
        cam, tgt, lens = compute_v2_camera(t)
        print(f"\n[{label} (t={t:.2f}s)] cam={np.round(cam, 3)} lens={lens:.1f}mm")
        for name, pt in pts.items():
            nx, ny, z = project(cam, tgt, lens, np.array(pt, dtype=float))
            in_ok = abs(nx) <= 0.85 and (-0.52 <= ny <= 0.65) and z > 0.12
            if not in_ok:
                ok = False
            print(f"  {name:15s}: ndcX={nx:+.2f}  ndcY={ny:+.2f}  depth={z:.2f}m  [{'OK' if in_ok else 'OUT!'}]")

    if not ok:
        raise SystemExit("FAILED: V2 framing check failed!")
    print("\nPASSED: 100% V2 Photoreal Human Leg, Foot & Macro Suturing Framing across ALL 7 Checkpoints!")


if __name__ == "__main__":
    main()
