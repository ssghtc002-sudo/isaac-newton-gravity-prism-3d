"""
Single source of truth for Episode #16 ("Why Jumping in a Falling Elevator WON'T Save You").
60 fps, 1800 frames (30.0 s), 1080x1920 Cycles -> True 4K (2160x3840 High@L5.2).

EVERY crew module imports `elevator_contract.py` instead of guessing world or cab coordinates.
World axes: +X = Cab Right, +Y = Cab Back Wall (Shaft Rear), -Y = Glass Doors (Viewer Front), +Z = Up.
"""
from __future__ import annotations

import math

FPS: int = 60
TOTAL_FRAMES: int = 1800
DURATION_S: float = TOTAL_FRAMES / float(FPS)  # 30.0s

# ---------------------------------------------------------------------------
# 6-BEAT MILESTONES & KEY EVENTS
# ---------------------------------------------------------------------------
BEATS: dict[int, tuple[int, int]] = {
    1: (1, 312),       # 0.00s - 5.20s  Hook: floor display drop + jump myth crouch
    2: (313, 600),     # 5.20s - 10.00s Cable snap & 50 mph shaft plunge cutaway
    3: (601, 900),     # 10.00s - 15.00s Myth jump (+0.42m, -2 mph vs 50 mph -> dramatic ceiling near-miss)
    4: (901, 1236),    # 15.00s - 20.60s 28G pit impact & full-body X-ray skeleton shockwave
    5: (1237, 1530),   # 20.60s - 25.50s Safety rail clamps (1237..1385) + flat-back survival pose (1386..1530)
    6: (1531, 1800),   # 25.50s - 30.00s 12x floor force-distribution grid hero reveal
}

EV: dict[str, int] = {
    "alarm_start": 110,
    "crouch_start": 190,
    "cable_snap": 325,
    "jump_crouch": 620,
    "jump_launch": 665,
    "ceiling_hit": 750,
    "pit_impact": 940,
    "pit_peak_comp": 985,
    "xray_in": 910,
    "xray_full_in": 938,
    "xray_full_out": 1205,
    "xray_out": 1235,
    "brake_cut_start": 1237,
    "brake_clamp": 1275,
    "brake_cut_end": 1385,
    "survival_cut_start": 1386,
    "lie_flat_settled": 1465,
    "grid_reveal": 1545,
}

# ---------------------------------------------------------------------------
# CAB, SHAFT & STRUCTURAL GEOMETRY (Cab-Local: floor surface = z 0.0)
# ---------------------------------------------------------------------------
CAB: dict[str, object] = {
    "half_w": 0.85,          # X in [-0.85, +0.85] (1.70 m wide)
    "half_d": 0.70,          # Y in [-0.70, +0.70] (1.40 m deep)
    "ceiling_z": 2.44,       # clear ceiling height inside cab
    "roof_z": 2.54,          # outer roof slab top
    "hitch_z": 2.64,         # cable hitch block center (0, 0, 2.64)
    "handrail_y": 0.61,      # rear stainless handrail bar Y
    "handrail_z": 0.92,      # rear handrail bar height
    "floor_led": (0.0, -0.66, 2.34),
    "cop_panel": (0.76, -0.52, 1.24),
    "rail_x": 0.96,          # vertical steel T-guide rails at X = +-0.96
    "clamp_z": -0.14,        # under-cab wedge safety clamps (X = +-0.92, Y = 0, Z = -0.14)
    "buffer_x": 0.42,        # twin hydraulic pit buffer springs at X = +-0.42
}

SHAFT: dict[str, object] = {
    "half_w": 1.25,          # concrete shaft X in [-1.25, +1.25]
    "half_d": 1.05,          # shaft Y in [-1.05, +1.05]
    "pit_floor_z": -1.15,    # pit concrete base slab top
    "buffer_top_z": -0.02,   # uncompressed spring buffer pad top when cab rests at Z=0
    "ceiling_sheave_z": 27.5,  # upper shaft traction sheave height
}

SUN_TOWARD: tuple[float, float, float] = (-0.52, -0.78, 0.35)

# Passenger reference metrics (1.78 m standing adult in cab-local coordinates)
PASSENGER: dict[str, object] = {
    "height_m": 1.78,
    "stand_xy": (-0.08, 0.04),
    "stand_yaw_deg": -14.0,   # slightly angled toward front-right Beat 1 camera
    "ankle_z": 0.075,
    "pelvis_z": 0.94,
    "eye_z": 1.65,
    "c1_z": 1.54,
    # Diagonal flat-on-back survival layout (38 deg diagonal inside 1.70x1.40m cab -> zero wall clip!)
    "flat_head": (-0.48, 0.34, 0.135),
    "flat_pelvis": (0.02, 0.02, 0.115),
    "flat_feet": (0.50, -0.32, 0.115),
    "flat_yaw_deg": 146.0,    # body axis along (-0.48,+0.34) -> (+0.50,-0.32)
}

LABELS: list[tuple[str, str, tuple[int, int]]] = [
    ("speed_cab", "ELEVATOR: 50 mph ↓", (115, 300)),
    ("cables", "HOIST CABLES", (318, 535)),
    ("speed_fall", "00 mph ↓", (320, 595)),
    ("jump_up", "YOUR JUMP: −2 mph ↑", (635, 885)),
    ("net_hit", "NET IMPACT: 48 mph ↓", (670, 895)),
    ("impact_g", "IMPACT: 28 G", (920, 1225)),
    ("leg_load", "SHIN & SPINE SHOCK", (940, 1225)),
    ("safety_brakes", "SAFETY CLAMPS", (1246, 1380)),
    ("flat_pose", "LIE FLAT ON BACK", (1405, 1545)),
    ("force_spread", "FORCE SPREAD: 12×", (1555, 1785)),
]


# ---------------------------------------------------------------------------
# PURE-MATH CURVES & TRAJECTORY EVALUATORS
# ---------------------------------------------------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp3(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def smootherstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def cab_world_z(f: float) -> float:
    """Exact world-space Z of `Cab_Root` across all 6 beats (pit impact at Z=0.0)."""
    if f <= 312.0:
        # Beat 1: High floor cruise tremor building before cable snap
        u_trem = smootherstep(230.0, 312.0, f)
        tremor = 0.010 * u_trem * math.sin(f * 0.85)
        return 18.0 + tremor
    if f <= 600.0:
        # Beat 2: Cable snaps at f=325 -> cab plunges downward (18.0m -> 9.0m; shaft scrolls too)
        u = smoothstep(325.0, 600.0, f)
        return lerp(18.0, 9.0, u ** 1.35)
    if f <= 900.0:
        # Beat 3: Free-fall plunge 9.0m -> 1.1m approaching shaft pit
        u = (f - 600.0) / 300.0
        return lerp(9.0, 1.10, u)
    if f <= 940.0:
        # Approaching pit buffer contact (1.10m -> 0.0m at f=940)
        u = (f - 900.0) / 40.0
        return lerp(1.10, 0.0, u)
    if f <= 1236.0:
        # Beat 4: Pit impact! Compresses buffer -0.26m (f=940..985), rebounds damped to 0.0m
        if f <= 985.0:
            u = math.sin(smootherstep(940.0, 985.0, f) * math.pi * 0.5)
            return -0.26 * u
        u_reb = smootherstep(985.0, 1085.0, f)
        damped = -0.26 * (1.0 - u_reb) * math.cos((f - 985.0) * 0.12) * math.exp(-0.025 * (f - 985.0))
        return clamp(damped, -0.26, 0.04)
    if f <= 1385.0:
        # Beat 5a (hard cut): Mid-shaft safety brake clamping demonstration (4.2m -> 2.4m brake arrest)
        u = smootherstep(1237.0, 1345.0, f)
        shudder = 0.008 * math.sin(f * 1.4) * (1.0 - smootherstep(1340.0, 1380.0, f))
        return lerp(4.2, 2.4, 1.0 - (1.0 - u) ** 2.2) + shudder
    # Beat 5b - Beat 6 (hard cut at 1386): Stabilized cab at pit level Z = 0.0 for flat-back reveal
    return 0.0


def cab_world_location(f: float) -> tuple[float, float, float]:
    return (0.0, 0.0, cab_world_z(f))


def shaft_scroll_z(f: float) -> float:
    """Cumulative apparent upward motion of mid-shaft wall rings/lights to convey 50 mph speed."""
    if f <= 325.0:
        return 0.0
    if f <= 600.0:
        u = (f - 325.0) / 275.0
        return 26.0 * (u ** 1.65)
    if f <= 900.0:
        u = (f - 600.0) / 300.0
        return 26.0 + 44.0 * u
    if f <= 940.0:
        u = (f - 900.0) / 40.0
        return 70.0 + 4.8 * u
    if f <= 1236.0:
        return 74.8
    if f <= 1385.0:
        u = smootherstep(1237.0, 1350.0, f)
        return 74.8 + 7.5 * (1.0 - (1.0 - u) ** 2.0)
    return 82.3


def speed_mph_at(f: float) -> float:
    """Cab downward speed in mph (ticks 0 -> 50 mph in Beat 2, holds 50 -> 48 net, hits 0 at impact)."""
    if f <= 325.0:
        return 0.0 if f < 100 else 18.0 + 32.0 * smoothstep(100.0, 312.0, f)
    if f <= 600.0:
        return lerp(14.0, 50.0, smoothstep(325.0, 560.0, f))
    if f <= 940.0:
        return 50.0
    if f <= 985.0:
        return lerp(50.0, 0.0, smootherstep(940.0, 985.0, f))
    if 1237.0 <= f <= 1355.0:
        return lerp(42.0, 0.0, smootherstep(1245.0, 1355.0, f))
    return 0.0


def passenger_jump_offset_z(f: float) -> float:
    """Cab-local vertical lift of passenger during Beat 3 myth jump (+0.42m ceiling near-miss)."""
    if f < 665.0 or f > 905.0:
        return 0.0
    if f <= 750.0:
        # Explosive spring up to near-miss apex (1.78 + 0.42 = 2.20m vs 2.44m ceiling)
        u = (f - 665.0) / 85.0
        return 0.42 * (1.0 - (1.0 - u) ** 2.1)
    if f <= 795.0:
        # Brief hang-time settle at apex before drop (0.42 -> 0.40m)
        return 0.42 - 0.02 * smootherstep(750.0, 795.0, f)
    # Fall back down to cab floor by f=905
    u = (f - 795.0) / 110.0
    return 0.40 * (1.0 - u * u)


def spine_impact_comp_cm(f: float) -> float:
    """Axial spinal + leg shock compression (cm, up to 3.5 cm at 28 G Beat 4 pit impact)."""
    if f < 935.0 or f > 1236.0:
        return 0.0
    if f <= 985.0:
        return 3.5 * smootherstep(935.0, 985.0, f)
    if f <= 1200.0:
        return 3.5 - 0.6 * smootherstep(985.0, 1200.0, f)
    return 2.9 * (1.0 - smootherstep(1200.0, 1236.0, f))


def xray_cab_weight(f: float) -> float:
    """Transparent X-ray cutaway on cab walls & front shaft during Beats 2b and 4."""
    if 345.0 <= f <= 585.0:
        return smootherstep(345.0, 370.0, f) * (1.0 - smootherstep(565.0, 585.0, f)) * 0.78
    if 910.0 <= f <= 1230.0:
        return smootherstep(910.0, 935.0, f) * (1.0 - smootherstep(1205.0, 1230.0, f)) * 0.88
    if f >= 1535.0:
        return 0.0
    return 0.0


def xray_body_weight(f: float) -> float:
    """Translucent X-ray character scan window in Beat 4 (f=910..1235)."""
    if f < 910.0 or f > 1235.0:
        return 0.0
    return smootherstep(910.0, 938.0, f) * (1.0 - smootherstep(1205.0, 1235.0, f))


def g_load_at(f: float) -> float:
    """Axial G-force waveform peaking cleanly at 28.0 G during Beat 4 impact."""
    if f < 918.0 or f > 1236.0:
        return 1.0
    if f <= 985.0:
        return 1.0 + 27.0 * smootherstep(925.0, 985.0, f)
    if f <= 1185.0:
        return 28.0 - 16.0 * smootherstep(985.0, 1185.0, f)
    return max(1.0, 12.0 * (1.0 - smootherstep(1185.0, 1236.0, f)))


def iter_action_fcurves(action) -> list:
    """Return F-Curves across Blender <5.0 (`action.fcurves`) and Blender 5.0+ Slotted Actions."""
    if action is None:
        return []
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        return list(legacy)
    fcurves: list = []
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for bag in getattr(strip, "channelbags", ()):
                fcurves.extend(getattr(bag, "fcurves", ()))
    return fcurves
