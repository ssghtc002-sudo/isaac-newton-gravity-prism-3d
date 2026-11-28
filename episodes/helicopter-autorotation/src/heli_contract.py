"""
heli_contract.py — Single Source of Truth Spatial, Timing, Physics & Blender 5.3.0 Contract (V2 Ultimate)
for Episode #5: "Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock"
(60 fps, 1800 frames, 30.0 s, 9:16 Portrait 1080x1920 -> True 4K 2160x3840).

V2 Ultimate Upgrade:
  - Splits Beat 1 into Beat 1A (1..115, Exterior Turbine Fire Flash, Sparks & Dark Smoke Plume)
    + Beat 1B (116..240, Dark Enclosed Panic Cockpit with Crashing 3D ENG 0% Needle & Collective Slam!)
  - Adds physical rotor blade coning angle `blade_coning_deg_at(f)`, cockpit `engine_rpm_pct_at(f)` /
    `rotor_rpm_pct_at(f)`, radar altitude `radar_alt_ft_at(f)`, and vertical cloud-parallax offset
    `descent_parallax_z_at(f)`.
  - Preserves 100% pure-Python safety (zero `bpy` import) and exact `0.000 mm` flush rooftop helipad
    skid contact (`HELI_SKID_BOTTOM_LOCAL_Z = -0.34`, `helipad_z = 0.90`, `b6_touchdown_root.z = 1.24`).
"""
from __future__ import annotations

import math
from typing import Any

# ==============================================================================
# 1. GLOBAL MASTER TIMING & 8-SEGMENT CINEMATIC ARCHITECTURE (60 fps, 1800 f)
# ==============================================================================
FPS: int = 60
TOTAL_FRAMES: int = 1800
DURATION_S: float = TOTAL_FRAMES / float(FPS)  # 30.0 s

# Sub-beat splits:
#   Beat 1A (1..115):     Exterior Modern Rescue Chopper — Dorsal Turbine Fire Flash, Sparks & Black Smoke!
#   Beat 1B (116..240):   Dark Enclosed 1:1 Cockpit — Red ENG OUT + Crashing ENG 0% Needle + Collective Slam!
#   Beat 5A (1291..1450): Dark Enclosed 1:1 Cockpit — Label-Free f/3.6 Bokeh Flare Pull + Altitude Drop!
#   Beat 5B (1451..1580): Crisp f/8.0 Exterior 30-Foot Nose-Up Flare over Weathered Rooftop Helipad!
B1_EXTERIOR_END: int = 115
B1_COCKPIT_START: int = 116
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

HARD_CUT_FRAMES: tuple[int, ...] = (116, 241, 571, 961, 1291, 1451, 1581)

# Exact event milestones (1:1 synced with config/timeline.json sfx_cues)
EV: dict[str, int] = {
    "b1_engine_flameout": 18,
    "b1_caution_alarm": 122,
    "b1_collective_slam": 152,
    "b2_xray_sweep": 265,
    "b2_shaft_seize": 345,
    "b2_sprag_unlock": 405,
    "b3_airflow_surge": 600,
    "b3_rotor_windmill": 735,
    "b3_flywheel_charge": 860,
    "b4_maple_spin": 995,
    "b4_vortex_sync": 1130,
    "b5_cockpit_pull": 1345,
    "b5_flare_bite": 1462,
    "b5_cushion_wash": 1522,
    "b6_skid_touchdown": 1685,
    "b6_rotor_coast": 1735,
}

# ==============================================================================
# 2. SPATIAL STAGE, COCKPIT CREW & HERO DIMENSIONS
# ==============================================================================
# Modern Twin-Engine Rescue Helicopter (Airbus H145 / Dauphin silhouette, 1:6 diorama):
#   - Fuselage + shrouded Fenestron tail length ~2.35 m, main rotor radius = 1.22 m
#   - Stepped rescue skids strictly flush at local Z = -0.340 m -> Z = 0.900 m helipad @ root Z = 1.240 m!
HELI_SKID_BOTTOM_LOCAL_Z: float = -0.34
HELI_ROTOR_HUB_LOCAL_Z: float = 0.45
HELI_CLUTCH_LOCAL_POS: tuple[float, float, float] = (0.00, 0.02, 0.30)
HELI_TURBINE_LOCAL_POS: tuple[float, float, float] = (0.00, 0.28, 0.27)
HELI_EXHAUST_LOCAL_POS: tuple[float, float, float] = (0.00, 0.42, 0.26)
HELI_ROTOR_RADIUS: float = 1.22

STAGE: dict[str, Any] = {
    # Golden-hour rooftop helipad (Beats 5B & 6)
    "helipad_z": 0.90,
    "helipad_center": (0.08, 0.45, 0.90),
    "helipad_radius": 2.45,
    # Beat 1B & 5A Enclosed 1:1 Rescue Cockpit key spatial anchors
    "cockpit_floor_z": 0.00,
    "cockpit_dash_pos": (0.00, -0.34, 1.02),
    "cockpit_caution_pos": (0.00, -0.34, 1.11),
    "cockpit_rpm_gauge_pos": (0.00, -0.26, 1.06),
    "cockpit_collective_pivot": (0.12, 0.16, 0.64),
    "cockpit_cyclic_pivot": (0.40, -0.02, 0.46),
    # Beat 2 Macro X-Ray Center (Rotor mast & sprag clutch)
    "b2_macro_center": (0.00, 0.00, 1.85),
    "b2_clutch_world": (0.00, 0.02, 2.15),
    # Beat 4 Hero 3D Maple Seed vs. Rescue Rotor Side-by-Side Comparison
    "b4_maple_start": (-0.54, -0.18, 2.24),
    "b4_maple_end": (-0.50, -0.14, 1.76),
    "b4_heli_start": (0.58, 0.42, 2.24),
    "b4_heli_end": (0.54, 0.36, 1.78),
    # Beat 6 final flush touchdown root position (1.24 - 0.34 = 0.90 m = EXACT helipad deck!)
    "b6_touchdown_root": (0.08, 0.45, 1.24),
}

# 2 Photoreal 4K MakeHuman MPFB2 Crew inside Enclosed 1:1 Rescue Cockpit (Beats 1B & 5A)
# Widened to X = +/-0.40 m for zero shoulder/arm crossover + clean physical stick/collective grips!
CREW: dict[str, Any] = {
    "pilot_pos": (0.40, 0.34, 0.00),
    "pilot_yaw_deg": 0.0,
    "pilot_eye_z": 1.38,
    "copilot_pos": (-0.40, 0.34, 0.00),
    "copilot_yaw_deg": 0.0,
    "copilot_eye_z": 1.38,
}

# ==============================================================================
# 3. 3D ENGINEERING CALLOUTS (6 Ultra-Short 1-3 Word Labels, V12 Razor-Sharp)
#    Beat 1A gets `ENGINE DEAD` pointing right at the flaming dorsal turbine;
#    Beat 1B & Beat 5A stay clean inside the cockpit so alarms & faces shine!
# ==============================================================================
LABELS: tuple[tuple[str, str, tuple[int, int]], ...] = (
    ("engine_dead", "ENGINE DEAD", (26, 108)),
    ("sprag_clutch", "SPRAG CLUTCH", (290, 552)),
    ("upward_airflow", "UPWARD AIRFLOW", (615, 942)),
    ("maple_seed_lift", "MAPLE SEED LIFT", (1000, 1272)),
    ("flare_30ft", "30 FT FLARE", (1466, 1568)),
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
    typed = [s for s in list(mix_node.inputs)[2:] if s.type == dtype]
    if len(typed) >= 2:
        return fac, typed[0], typed[1]
    return fac, mix_node.inputs[2], mix_node.inputs[3]


# ==============================================================================
# 5. SHARED HERO KINEMATICS & LIVE INSTRUMENT TELEMETRY
# ==============================================================================
def engine_rpm_pct_at(f: float) -> float:
    """Engine turbine N2 RPM percentage: 100% -> crashes violently to 0% during Beat 1A/1B!"""
    if f <= 18.0:
        return 100.0
    if f <= 115.0:
        return lerp(100.0, 28.0, smootherstep(18.0, 115.0, f))
    if f <= 168.0:
        # Visible crashing crimson needle inside Beat 1B cockpit (28% -> 0%!)
        return lerp(28.0, 0.0, smootherstep(116.0, 168.0, f))
    return 0.0


def rotor_rpm_pct_at(f: float) -> float:
    """Main rotor NR RPM percentage: dips slightly on stall, then stays healthy in green arc (98-104%)!"""
    if f <= 115.0:
        return lerp(100.0, 91.0, smoothstep(18.0, 115.0, f))
    if f <= 240.0:
        # Recovers cleanly into the green arc once pilot slams collective down!
        return lerp(91.0, 102.0, smootherstep(145.0, 215.0, f))
    if f <= 1450.0:
        return 103.0
    if f <= 1685.0:
        return lerp(103.0, 74.0, smootherstep(1465.0, 1685.0, f))
    return lerp(74.0, 22.0, smootherstep(1685.0, 1800.0, f))


def radar_alt_ft_at(f: float) -> float:
    """Radar altimeter altitude (feet) for Beat 5A cockpit HUD & Beat 5B/6 touchdown."""
    if f <= 1290.0:
        return 180.0
    if f <= 1450.0:
        return lerp(125.0, 32.0, smootherstep(1291.0, 1450.0, f))
    if f <= 1685.0:
        return lerp(32.0, 0.0, smootherstep(1451.0, 1685.0, f))
    return 0.0


def descent_parallax_z_at(f: float) -> float:
    """Upward cloud-layer drift offset (meters) creating visceral falling parallax in Beats 3 & 4!"""
    if f <= 570.0:
        return 0.0
    if f <= 960.0:
        return lerp(0.0, 5.4, (f - 570.0) / 390.0)
    if f <= 1290.0:
        return lerp(5.4, 9.8, (f - 960.0) / 330.0)
    return 9.8


def collective_level_at(f: float) -> float:
    """
    Normalized collective pitch lever height in [0.0 (full down), 1.0 (full up)].
    - Beat 1A (1..115): Cruising at 0.72.
    - Beat 1B (116..240): Pilot SLAMS lever straight down (0.72 -> 0.05) at f=135..178!
    - Beats 2..4 (241..1290): Held low at 0.08 (autorotation flat/negative pitch).
    - Beat 5A (1291..1450): Pilot hauls lever strongly UP (0.08 -> 0.94) at f=1335..1418 for 30-ft flare!
    - Beat 5B (1451..1580): Held high at 0.95 cushion pull.
    - Beat 6 (1581..1800): Gently lowers back to 0.14 after flush rooftop touchdown.
    """
    if f <= 115.0:
        return 0.72
    if f <= 240.0:
        return lerp(0.72, 0.05, smootherstep(135.0, 178.0, f))
    if f <= 1290.0:
        return 0.08
    if f <= 1450.0:
        return lerp(0.08, 0.94, smootherstep(1335.0, 1418.0, f))
    if f <= 1580.0:
        return lerp(0.94, 0.96, smoothstep(1451.0, 1545.0, f))
    return lerp(0.96, 0.14, smootherstep(1680.0, 1765.0, f))


def blade_pitch_deg_at(f: float) -> float:
    """Physical main rotor airfoil collective pitch angle (-4 deg autorotation -> +14 deg flare)."""
    return lerp(-4.0, 14.0, collective_level_at(f))


def blade_coning_deg_at(f: float) -> float:
    """Aerodynamic upward blade coning flex (+1.2 deg cruise -> +4.2 deg high-G 30-ft flare -> 0.2 deg parked!)."""
    if f <= 1450.0:
        return 1.3
    if f <= 1580.0:
        return lerp(1.3, 4.2, smootherstep(1455.0, 1540.0, f))
    return lerp(4.2, 0.2, smootherstep(1615.0, 1730.0, f))


def cowling_xray_t(f: float) -> float:
    """Local-space X-ray sweep progress across Beat 2 (0.0 solid -> 1.0 crystal hollow X-ray)."""
    if f < 241.0 or f > 570.0:
        return 0.0
    sweep_in = smootherstep(258.0, 315.0, f)
    sweep_out = 1.0 - smoothstep(545.0, 570.0, f)
    return sweep_in * sweep_out


def sprag_unlock_t(f: float) -> float:
    """Sprag clutch cam-roller decoupling progress (0.0 locked to engine -> 1.0 freewheeling open)."""
    if f < 241.0:
        return smoothstep(135.0, 175.0, f)
    if f <= 570.0:
        return smootherstep(388.0, 432.0, f)
    return 1.0


def rotor_angle_rad_at(f: float) -> float:
    """Cumulative main rotor azimuth angle (radians) across 1..1800 @ 60 fps."""
    t = (f - 1.0) / float(FPS)
    base_omega = 17.5
    if f <= 570.0:
        return base_omega * t
    if f <= 1290.0:
        t3 = (f - 570.0) / float(FPS)
        return base_omega * (569.0 / float(FPS)) + (base_omega * 1.14) * t3
    if f <= 1685.0:
        t5 = (f - 1290.0) / float(FPS)
        ang_1290 = base_omega * (569.0 / float(FPS)) + (base_omega * 1.14) * (720.0 / float(FPS))
        return ang_1290 + (base_omega * 1.08) * t5 - 0.14 * (t5 * t5)
    t_td = (1685.0 - 1290.0) / float(FPS)
    ang_1290 = base_omega * (569.0 / float(FPS)) + (base_omega * 1.14) * (720.0 / float(FPS))
    ang_td = ang_1290 + (base_omega * 1.08) * t_td - 0.14 * (t_td * t_td)
    omega_td = max(6.2, base_omega * 1.08 - 0.28 * t_td)
    dt = (f - 1685.0) / float(FPS)
    return ang_td + omega_td * (1.0 - math.exp(-0.78 * dt)) / 0.78


def engine_shaft_angle_rad_at(f: float) -> float:
    """Turbine engine drive shaft angle in Beat 2 macro X-ray: spins -> sputters -> SEIZES DEAD at f=345!"""
    if f <= 241.0:
        return 0.0
    if f <= 345.0:
        u = (f - 241.0) / 104.0
        return 24.0 * (u - 0.5 * u * u)
    shudder = 0.05 * math.exp(-0.26 * (f - 345.0)) * math.sin((f - 345.0) * 2.2)
    return 12.0 + shudder


def heli_pose_at(f: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    World-space (position_xyz, euler_deg_xyz) of `Heli_Root` across all exterior beats:
      - Beat 1A (1..115): Dynamic banking flight over glass skyline with sudden engine stall jolt @ f=18!
      - Beat 2  (241..570): Dorsal macro X-ray close-up of seized turbine shaft & sprag clutch
      - Beat 3  (571..960): Smooth autorotation glide with clean upward emerald 3D ribbon arrows
      - Beat 4  (961..1290): Right-hand half of side-by-side comparison with hero 3D Maple Seed
      - Beat 5B (1451..1580): Dramatic 30-foot nose-up flare over weathered rooftop helipad
      - Beat 6  (1581..1800): Smooth level-out & flush 0.000 mm touchdown on rooftop helipad (Z = 0.90 m)
    """
    # Beat 1A (1..115): Exterior Engine Compressor-Stall Fire Burst & Dark Smoke Plume!
    if f <= float(B1_EXTERIOR_END):
        u = smootherstep(1.0, float(B1_EXTERIOR_END), f)
        # Sharp mechanical yaw/roll shudder at f=18 when the turbine explodes!
        jolt = 0.0
        if f >= 18.0:
            dt = f - 18.0
            jolt = math.exp(-0.08 * dt) * math.sin(dt * 1.15)
        pos = (
            lerp(-0.28, 0.16, u),
            lerp(0.32, 0.20, u),
            lerp(2.38, 2.02, u) - 0.04 * max(0.0, jolt),
        )
        rot = (
            lerp(-5.0, -11.5, u) + 3.2 * jolt,
            lerp(-8.0, -14.0, u) + 4.5 * jolt,
            lerp(-34.0, -24.0, u) - 3.0 * jolt,
        )
        return pos, rot

    # Beat 2 (241..570): Dorsal Macro X-Ray on White/Red Cowling, Seized Shaft & Sprag Clutch
    if f <= 570.0:
        u = smootherstep(241.0, 570.0, f)
        bob = 0.014 * math.sin((f - 241.0) * 0.055)
        pos = (0.00, 0.00, 1.85 + bob)
        rot = (
            lerp(-8.0, -5.5, u),
            lerp(16.0, 13.0, u),       # slightly stronger tilt exposing dorsal clutch assembly clearly!
            lerp(-30.0, -18.0, u),
        )
        return pos, rot

    # Beat 3 (571..960): Dynamic Golden-Hour Autorotation Glide with Wide Upward Emerald Ribbon Arrows
    if f <= 960.0:
        u = smootherstep(571.0, 960.0, f)
        pos = (
            lerp(-0.34, 0.28, u),
            lerp(0.32, 0.12, u),
            lerp(2.46, 1.98, u),
        )
        rot = (
            lerp(-10.5, -8.0, u),
            lerp(-12.0, -7.0, u),
            lerp(-34.0, -22.0, u),
        )
        return pos, rot

    # Beat 4 (961..1290): Right-hand half of side-by-side comparison with hero 3D Maple Seed
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

    # Beat 5B (1451..1580): Dramatic 30-Foot Nose-Up Flare over Rooftop Helipad!
    if f <= 1580.0:
        u = smootherstep(1451.0, 1580.0, f)
        flare_u = smootherstep(1455.0, 1555.0, f)
        pos = (
            lerp(-0.62, 0.02, u),
            lerp(0.54, 0.45, u),
            lerp(2.38, 1.66, u),
        )
        rot = (
            lerp(-5.0, 22.0, flare_u),
            lerp(-9.0, -2.0, flare_u),
            lerp(-24.0, -10.0, u),
        )
        return pos, rot

    # Beat 6 (1581..1800): Smooth Level-Out & Exact 0.000 mm Flush Touchdown on Z = 0.90 m Helipad!
    td_u = smootherstep(1581.0, 1685.0, f)
    settle_u = smootherstep(1685.0, 1712.0, f)
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
    """World-space (position_xyz, euler_deg_xyz) of `Heli_MapleSeed_Root` during Beat 4 (961..1290)."""
    u = smootherstep(961.0, 1290.0, f)
    ms = STAGE["b4_maple_start"]
    me = STAGE["b4_maple_end"]
    sway = 0.032 * math.sin((f - 961.0) * 0.05)
    pos = (
        lerp(ms[0], me[0], u) + sway,
        lerp(ms[1], me[1], u),
        lerp(ms[2], me[2], u),
    )
    spin_deg = (f - 961.0) * 14.5
    rot = (16.0, 10.0, spin_deg)
    return pos, rot
