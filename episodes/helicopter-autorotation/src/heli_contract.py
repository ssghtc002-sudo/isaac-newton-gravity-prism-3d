"""
heli_contract.py — Single Source of Truth Spatial, Timing, Physics & Blender 5.3.0 Contract
for Episode #5: "Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock"
(60 fps, 1800 frames, 30.0 s, 9:16 Portrait 1080x1920 -> True 4K 2160x3840).

Pure-Python module (zero `bpy` requirement) imported by all 3D scene builders, audio/subtitle
generators, and automated 1800-frame QA scripts so every frame, trajectory, socket, and
3D callout shares 0.000 mm consistency.
"""
from __future__ import annotations

import math
from typing import Any

# ==============================================================================
# 1. GLOBAL MASTER TIMING & BEAT ARCHITECTURE (60 fps, 1800 frames, 30.0 s)
# ==============================================================================
FPS: int = 60
TOTAL_FRAMES: int = 1800
DURATION_S: float = TOTAL_FRAMES / float(FPS)  # 30.0 s

# Sub-beat split for Beat 5:
#   5A (1291..1450): Label-free 1:1 Cockpit Final Flare close-up (f/3.6 bokeh!)
#   5B (1451..1580): Crisp f/8.0 Exterior 30-Foot Nose-Up Flare beside Rooftop Helipad
B5_COCKPIT_END: int = 1450
B5_EXTERIOR_START: int = 1451

BEATS: dict[str, tuple[int, int]] = {
    "B1_HOOK": (1, 240),
    "B2_CLUTCH": (241, 570),
    "B3_AIRFLOW": (571, 960),
    "B4_MAPLE": (961, 1290),
    "B5_FLARE": (1291, 1580),
    "B6_LANDING": (1581, 1800),
}

HARD_CUT_FRAMES: tuple[int, ...] = (241, 571, 961, 1291, 1451, 1581)

# Exact event milestones (1:1 synced with config/timeline.json sfx_cues)
EV: dict[str, int] = {
    "b1_engine_flameout": 14,
    "b1_caution_alarm": 42,
    "b1_collective_slam": 120,
    "b2_xray_sweep": 270,
    "b2_shaft_seize": 340,
    "b2_sprag_unlock": 415,
    "b3_airflow_surge": 605,
    "b3_rotor_windmill": 740,
    "b3_flywheel_charge": 865,
    "b4_maple_spin": 995,
    "b4_vortex_sync": 1125,
    "b5_cockpit_pull": 1350,
    "b5_flare_bite": 1465,
    "b5_cushion_wash": 1525,
    "b6_skid_touchdown": 1685,
    "b6_rotor_coast": 1735,
}

# ==============================================================================
# 2. SPATIAL STAGE, COCKPIT CREW & HERO DIMENSIONS
# ==============================================================================
# Exterior Rescue Helicopter (1:6 scale diorama):
#   - Fuselage length ~2.15 m, main rotor radius = 1.15 m (2.30 m diameter)
#   - Skid bottoms sit strictly at local Z = -0.34 m -> flush landing on Z = 0.90 m helipad at root Z = 1.24 m!
HELI_SKID_BOTTOM_LOCAL_Z: float = -0.34
HELI_ROTOR_HUB_LOCAL_Z: float = 0.44
HELI_CLUTCH_LOCAL_POS: tuple[float, float, float] = (0.00, 0.02, 0.30)
HELI_TURBINE_LOCAL_POS: tuple[float, float, float] = (0.00, 0.24, 0.28)
HELI_ROTOR_RADIUS: float = 1.15

STAGE: dict[str, Any] = {
    # Golden-hour rooftop helipad (Beats 5B & 6)
    "helipad_z": 0.90,
    "helipad_center": (0.08, 0.45, 0.90),
    "helipad_radius": 2.40,
    # Beat 1 & 5A 1:1 Rescue Cockpit key spatial anchors
    "cockpit_floor_z": 0.00,
    "cockpit_dash_pos": (0.00, -0.32, 1.06),
    "cockpit_caution_pos": (0.00, -0.36, 1.14),
    "cockpit_collective_pivot": (0.10, 0.20, 0.66),
    "cockpit_cyclic_pivot": (0.36, -0.04, 0.48),
    # Beat 2 Macro X-Ray Center (Rotor mast & sprag clutch)
    "b2_macro_center": (0.00, 0.00, 1.85),
    "b2_clutch_world": (0.00, 0.02, 2.15),
    # Beat 4 Maple Seed vs. Rescue Rotor Side-by-Side Comparison
    "b4_maple_start": (-0.56, -0.10, 2.26),
    "b4_maple_end": (-0.52, -0.06, 1.76),
    "b4_heli_start": (0.58, 0.40, 2.24),
    "b4_heli_end": (0.54, 0.34, 1.78),
    # Beat 6 final flush touchdown root position (1.24 - 0.34 = 0.90 m = EXACT helipad deck!)
    "b6_touchdown_root": (0.08, 0.45, 1.24),
}

# 2 Photoreal 4K MakeHuman MPFB2 Crew inside 1:1 Rescue Cockpit (Beats 1 & 5A)
CREW: dict[str, Any] = {
    # Standard helicopter PIC sits RIGHT seat (+X), Co-Pilot sits LEFT seat (-X), facing -Y toward windshield/camera!
    "pilot_pos": (0.36, 0.34, 0.00),
    "pilot_yaw_deg": 180.0,
    "pilot_eye_z": 1.38,
    "copilot_pos": (-0.36, 0.34, 0.00),
    "copilot_yaw_deg": 180.0,
    "copilot_eye_z": 1.38,
}

# ==============================================================================
# 3. 3D ENGINEERING CALLOUTS (6 Ultra-Short 1-3 Word Labels, V12 Razor-Sharp)
#    Beat 5A (1291..1450) intentionally has ZERO labels so f/3.6 bokeh shines!
# ==============================================================================
LABELS: tuple[tuple[str, str, tuple[int, int]], ...] = (
    ("engine_dead", "ENGINE DEAD", (64, 228)),
    ("sprag_clutch", "SPRAG CLUTCH", (295, 552)),
    ("upward_airflow", "UPWARD AIRFLOW", (620, 942)),
    ("maple_seed_lift", "MAPLE SEED LIFT", (1005, 1272)),
    ("flare_30ft", "30 FT FLARE", (1468, 1568)),
    ("zero_engine", "ZERO ENGINE", (1615, 1782)),
)


# ==============================================================================
# 4. MATH, SMOOTHING & BLENDER 5.3.0 SAFETY HELPERS
# ==============================================================================
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else (hi if x > hi else x)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(e0: float, e1: float, x: float) -> float:
    if abs(e1 - e0) < 1e-9:
        return 1.0 if x >= e1 else 0.0
    t = clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(e0: float, e1: float, x: float) -> float:
    if abs(e1 - e0) < 1e-9:
        return 1.0 if x >= e1 else 0.0
    t = clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def iter_action_fcurves(action: Any) -> list[Any]:
    """Blender 5.3.0 safe Slotted Action FCurve iterator (falls back gracefully to legacy .fcurves)."""
    if action is None:
        return []
    fcurves: list[Any] = []
    if hasattr(action, "layers"):
        for layer in action.layers:
            for strip in getattr(layer, "strips", ()):
                for bag in getattr(strip, "channelbags", ()):
                    fcurves.extend(list(getattr(bag, "fcurves", ())))
    if not fcurves and hasattr(action, "fcurves"):
        fcurves.extend(list(action.fcurves))
    return fcurves


def get_mix_sockets(mix_node: Any) -> tuple[Any, Any, Any]:
    """
    Blender 5.3.0 safe `ShaderNodeMix` socket resolver -> (Factor, A, B).
    Always indexes typed A/B sockets strictly from `inputs[2:]` (never catching Factor at inputs[0]!).
    """
    dtype = getattr(mix_node, "data_type", "FLOAT")
    fac = mix_node.inputs[0]
    if dtype == "FLOAT":
        return fac, mix_node.inputs[2], mix_node.inputs[3]
    if dtype == "VECTOR":
        return fac, mix_node.inputs[4], mix_node.inputs[5]
    if dtype == "RGBA":
        return fac, mix_node.inputs[6], mix_node.inputs[7]
    # Generic fallback strictly skipping inputs[0..1]
    typed = [s for s in list(mix_node.inputs)[2:] if s.type == dtype]
    if len(typed) >= 2:
        return fac, typed[0], typed[1]
    return fac, mix_node.inputs[2], mix_node.inputs[3]


# ==============================================================================
# 5. SHARED HERO KINEMATICS (HELICOPTER, ROTOR, SPRAG CLUTCH, COLLECTIVE, MAPLE)
# ==============================================================================
def collective_level_at(f: float) -> float:
    """
    Normalized collective pitch lever height in [0.0 (full down), 1.0 (full up)].
    - Beat 1 (1..240): Starts cruising at 0.72, pilot slams straight down to 0.05 at f=110..150!
    - Beats 2..4 (241..1290): Held low at 0.08 (autorotation flat/negative pitch).
    - Beat 5A (1291..1450): Pilot hauls lever strongly UP (0.08 -> 0.92) at f=1340..1425 for 30-ft flare!
    - Beat 5B (1451..1580): Held high at 0.94 cushion pull.
    - Beat 6 (1581..1800): Gently lowers back to 0.15 after flush rooftop touchdown.
    """
    if f <= 240.0:
        return lerp(0.72, 0.05, smootherstep(105.0, 150.0, f))
    if f <= 1290.0:
        return 0.08
    if f <= 1450.0:
        return lerp(0.08, 0.92, smootherstep(1340.0, 1425.0, f))
    if f <= 1580.0:
        return lerp(0.92, 0.95, smoothstep(1451.0, 1545.0, f))
    return lerp(0.95, 0.15, smootherstep(1680.0, 1760.0, f))


def blade_pitch_deg_at(f: float) -> float:
    """Physical main rotor airfoil collective pitch angle in degrees (-4 deg autorotation -> +13 deg flare)."""
    return lerp(-4.0, 13.0, collective_level_at(f))


def cowling_xray_t(f: float) -> float:
    """Local-space X-ray sweep progress across Beat 2 (0.0 solid -> 1.0 crystal hollow X-ray)."""
    if f < 241.0 or f > 570.0:
        return 0.0
    sweep_in = smootherstep(262.0, 325.0, f)
    sweep_out = 1.0 - smoothstep(545.0, 570.0, f)
    return sweep_in * sweep_out


def sprag_unlock_t(f: float) -> float:
    """Sprag clutch cam-roller decoupling progress (0.0 locked to engine -> 1.0 freewheeling open)."""
    if f < 241.0:
        return smoothstep(105.0, 145.0, f)
    if f <= 570.0:
        # Re-enact dramatic macro snap-open at f=395..435 right after shaft seizes!
        return smootherstep(392.0, 435.0, f)
    return 1.0


def rotor_angle_rad_at(f: float) -> float:
    """
    Cumulative main rotor azimuth angle (radians) across 1..1800 @ 60 fps.
    - Beats 1..2: Fast healthy spin (~360 RPM visual cadence) that NEVER stops even when turbine seizes!
    - Beat 3..4: Upward autorotation airflow accelerates stored flywheel spin slightly (+12%).
    - Beat 5B..6: 30-ft flare converts stored kinetic RPM into lift -> blades smoothly slow down after touchdown!
    """
    t = (f - 1.0) / float(FPS)
    # Piecewise smooth angular speed omega(t) in rad/s (visually crisp at 60fps without strobing)
    base_omega = 16.5  # ~2.63 rev/s (clean readable twin-blade sweep at 60fps with 0.25 shutter)
    if f <= 570.0:
        return base_omega * t
    if f <= 1290.0:
        # Slight upward-airflow flywheel spin boost during Beats 3 & 4
        t3 = (f - 570.0) / float(FPS)
        return base_omega * (569.0 / float(FPS)) + (base_omega * 1.12) * t3
    if f <= 1685.0:
        t5 = (f - 1290.0) / float(FPS)
        ang_1290 = base_omega * (569.0 / float(FPS)) + (base_omega * 1.12) * (720.0 / float(FPS))
        # High spin entering flare, gently bleeding energy during cushion pull
        return ang_1290 + (base_omega * 1.06) * t5 - 0.14 * (t5 * t5)
    # Post-touchdown freewheel coast-down (1685..1800)
    t_td = (1685.0 - 1290.0) / float(FPS)
    ang_1290 = base_omega * (569.0 / float(FPS)) + (base_omega * 1.12) * (720.0 / float(FPS))
    ang_td = ang_1290 + (base_omega * 1.06) * t_td - 0.14 * (t_td * t_td)
    omega_td = max(6.0, base_omega * 1.06 - 0.28 * t_td)
    dt = (f - 1685.0) / float(FPS)
    # Smooth exponential coast-down to ~2.8 rad/s by f=1800
    return ang_td + omega_td * (1.0 - math.exp(-0.75 * dt)) / 0.75


def engine_shaft_angle_rad_at(f: float) -> float:
    """
    Turbine engine drive shaft angle in Beat 2 macro X-ray:
    Spins rapidly at f=241..325, sputters, then SEIZES DEAD at f=345 (while rotor keeps flying!).
    """
    if f <= 241.0:
        return 0.0
    if f <= 345.0:
        u = (f - 241.0) / 104.0
        # Decelerating integral from omega0 -> 0 at u=1
        return 22.0 * (u - 0.5 * u * u)
    # Dead seized shaft with tiny 3-frame mechanical shudder at f=345..358
    shudder = 0.04 * math.exp(-0.28 * (f - 345.0)) * math.sin((f - 345.0) * 2.1)
    return 11.0 + shudder


def heli_pose_at(f: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    World-space (position_xyz, euler_deg_xyz) of `Heli_Root` across all exterior beats.
    (Hidden during 1:1 interior cockpit Beats 1 & 5A.)
    """
    # Beat 2 (241..570): Macro X-ray close-up of dorsal cowling, turbine shaft & sprag clutch
    if f <= 570.0:
        u = smootherstep(241.0, 570.0, f)
        bob = 0.015 * math.sin((f - 241.0) * 0.055)
        pos = (0.00, 0.00, 1.85 + bob)
        rot = (
            lerp(-7.0, -5.0, u),       # gentle nose-down pitch
            lerp(13.0, 10.0, u),       # tilted toward camera so dorsal mast & clutch shine!
            lerp(-28.0, -18.0, u),     # 3/4 rescue profile showing white/red nose & tail
        )
        return pos, rot

    # Beat 3 (571..960): Dynamic golden-hour autorotation glide with upward emerald airflow
    if f <= 960.0:
        u = smootherstep(571.0, 960.0, f)
        pos = (
            lerp(-0.34, 0.28, u),
            lerp(0.32, 0.12, u),
            lerp(2.44, 1.98, u),
        )
        rot = (
            lerp(-10.0, -8.0, u),      # controlled forward autorotation glide pitch
            lerp(-12.0, -7.0, u),      # sleek banking angle exposing full spinning rotor disk
            lerp(-34.0, -22.0, u),
        )
        return pos, rot

    # Beat 4 (961..1290): Right-hand half of side-by-side comparison with 3D Maple Seed
    if f <= 1290.0:
        u = smootherstep(961.0, 1290.0, f)
        hs = STAGE["b4_heli_start"]
        he = STAGE["b4_heli_end"]
        pos = (
            lerp(hs[0], he[0], u),
            lerp(hs[1], he[1], u),
            lerp(hs[2], he[2], u),
        )
        rot = (
            lerp(-8.0, -6.0, u),
            lerp(-14.0, -10.0, u),
            lerp(-22.0, -14.0, u),
        )
        return pos, rot

    # Beat 5B (1451..1580): Dramatic 30-foot nose-up flare over rooftop helipad!
    if f <= 1580.0:
        u = smootherstep(1451.0, 1580.0, f)
        flare_u = smootherstep(1455.0, 1555.0, f)
        pos = (
            lerp(-0.62, 0.02, u),
            lerp(0.54, 0.45, u),
            lerp(2.36, 1.66, u),
        )
        rot = (
            lerp(-5.0, 21.0, flare_u),  # dramatic nose-up flare killing forward & downward velocity!
            lerp(-9.0, -2.0, flare_u),
            lerp(-24.0, -10.0, u),
        )
        return pos, rot

    # Beat 6 (1581..1800): Smooth level-out & flush touchdown on Z = 0.90 m rooftop helipad!
    # Skid bottom local Z = -0.34 m -> root Z = 1.24 m gives EXACT 0.000 mm flush contact!
    td_u = smootherstep(1581.0, 1685.0, f)
    settle_u = smootherstep(1685.0, 1712.0, f)
    # Tiny 6mm cushioned skid-spring flex on touchdown
    skid_flex = -0.008 * math.sin(math.pi * settle_u) if 1685.0 <= f <= 1712.0 else 0.0
    pos = (
        lerp(0.02, 0.08, td_u),
        0.45,
        lerp(1.64, 1.24, td_u) + skid_flex,
    )
    rot = (
        lerp(14.0, 0.0, td_u),
        lerp(-3.0, 0.0, td_u),
        lerp(-12.0, -16.0, smoothstep(1581.0, 1760.0, f)),
    )
    return pos, rot


def maple_seed_pose_at(f: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    World-space (position_xyz, euler_deg_xyz) of `Heli_MapleSeed_Root` during Beat 4 (961..1290).
    Descends steadily on the left while autorotating smoothly at a readable 60fps spin rate!
    """
    u = smootherstep(961.0, 1290.0, f)
    ms = STAGE["b4_maple_start"]
    me = STAGE["b4_maple_end"]
    sway = 0.035 * math.sin((f - 961.0) * 0.05)
    pos = (
        lerp(ms[0], me[0], u) + sway,
        lerp(ms[1], me[1], u),
        lerp(ms[2], me[2], u),
    )
    # Coning tilt (~16 deg) + smooth continuous vertical-axis autorotation spin
    spin_deg = (f - 961.0) * 14.5
    rot = (14.0, 10.0, spin_deg)
    return pos, rot
