"""
Single Source of Truth Contract for Episode #2:
"Why Bullets Stop Dead Just 3 Feet Underwater" (30.0 s @ 60 fps = 1800 frames).

Every 3D scene builder, camera rig, FX shader, audio stem, and subtitle script MUST
read shared dimensions, milestone frames, labels, and trajectories exclusively here.
Follows AGENTS.md V12 (Blender 5.3.0 + @zackdfilms blueprint).
"""
from __future__ import annotations

import math
from typing import Any, Iterator

FPS: int = 60
DURATION_S: float = 30.0
TOTAL_FRAMES: int = 1800

FT_TO_M: float = 0.3048
THREE_FT_X: float = 3.0 * FT_TO_M   # 0.9144 m
SIX_FT_X: float = 6.0 * FT_TO_M     # 1.8288 m

BEATS: dict[str, tuple[int, int]] = {
    "B1_HOOK": (1, 210),
    "B2_ENTRY": (211, 540),
    "B3_CAVITATION": (541, 900),
    "B4_SHATTER": (901, 1260),
    "B5_RACE": (1261, 1560),
    "B6_SAFE": (1561, 1800),
}

EV: dict[str, int] = {
    "b1_bullet_start": 10,
    "b1_bullet_stop": 135,
    "b2_muzzle_fire": 248,
    "b2_water_entry": 300,
    "b3_bubble_peak": 720,
    "b4_xray_on": 925,
    "b4_tumble_start": 970,
    "b4_shatter_burst": 1055,
    "b5_race_start": 1285,
    "b5_fast_stop": 1410,
    "b5_slow_cross": 1525,
    "b6_safe_settle": 1620,
}

TANK: dict[str, Any] = {
    "x_min": -1.8,
    "x_max": 3.2,
    "y_min": -1.6,
    "y_max": 2.0,
    "water_z": 0.0,
    "floor_z": -2.15,
    "flight_z": -0.55,
    "race_fast_z": -0.28,
    "race_slow_z": -0.84,
    "ruler_z": -1.42,
    "muzzle_x": -0.48,
    "entry_x": 0.0,
    "three_ft_x": THREE_FT_X,
    "six_ft_x": SIX_FT_X,
}

SWIMMER: dict[str, Any] = {
    "height_m": 1.78,
    "b1_pos": (1.58, 0.06, -0.78),
    "b1_yaw_deg": -82.0,    # Facing -X toward incoming bullet & front-left 3/4 camera
    "b6_pos": (1.42, 0.12, -1.08),
    "b6_yaw_deg": -72.0,
}

# Strict V12 @zackdfilms ultra-short 1-3 word callouts only (NDC Y >= -0.45)
LABELS: list[tuple[str, str, tuple[int, int]]] = [
    ("stop_3ft", "3 FT STOP", (118, 204)),
    ("density_800", "800× DENSER", (325, 525)),
    ("vapor_bubble", "VAPOR BUBBLE", (585, 880)),
    ("shattered", "SHATTERED", (1055, 1245)),
    ("race_fast", "FASTER: 3 FT", (1345, 1545)),
    ("race_slow", "SLOWER: 6 FT", (1385, 1545)),
    ("safe_depth", "SAFE BELOW 3 FT", (1615, 1785)),
]


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def iter_action_fcurves(action: Any) -> Iterator[Any]:
    """Blender 4.x & 5.x Slotted Actions safe FCurve iterator."""
    if action is None:
        return
    if hasattr(action, "fcurves"):
        for fc in action.fcurves:
            yield fc
        return
    if hasattr(action, "layers"):
        for layer in action.layers:
            for strip in getattr(layer, "strips", ()):
                for cbag in getattr(strip, "channelbags", ()):
                    for fc in getattr(cbag, "fcurves", ()):
                        yield fc


# ---------------------------------------------------------------------------
# DETERMINISTIC BALLISTIC TRAJECTORY CURVES (Single Source of Truth)
# ---------------------------------------------------------------------------
def supersonic_x(f: float) -> float:
    """World X of primary hero supersonic round across Beats 1-4 (0.0m -> 0.9144m = 3 ft)."""
    if f <= 210.0:
        # Beat 1: streaks in fast from X=-0.10m, screeches dead at 3 FT (0.9144m) by f=135
        u = 1.0 - (1.0 - smoothstep(10.0, 135.0, f)) ** 2.6
        return lerp(-0.10, THREE_FT_X, u)
    if f <= 540.0:
        # Beat 2: fires from muzzle (-0.48) at f=248 -> slams water (0.0) at f=300 -> 0.48m
        if f < 248.0:
            return -0.48
        if f <= 300.0:
            return lerp(-0.48, 0.0, (f - 248.0) / 52.0)
        u = 1.0 - (1.0 - smoothstep(300.0, 540.0, f)) ** 1.45
        return lerp(0.0, 0.48, u)
    if f <= 900.0:
        # Beat 3: cavitation glide 0.48m -> 0.78m
        u = smootherstep(541.0, 900.0, f)
        return lerp(0.48, 0.78, u)
    if f <= 1260.0:
        # Beat 4: tumbles & shatters dead at THREE_FT_X (0.9144m) around f=1055
        u = 1.0 - (1.0 - smoothstep(901.0, 1110.0, f)) ** 2.0
        return lerp(0.78, THREE_FT_X, u)
    # Beat 5-6 hero lane handled cleanly by race_fast_x / shard drift
    return THREE_FT_X


def supersonic_z(f: float) -> float:
    """World Z of primary hero round (subtle Beat 1 post-stop settle & Beat 4 tumble drift)."""
    base_z = float(TANK["flight_z"])
    if f <= 135.0:
        return base_z
    if f <= 210.0:
        return base_z - 0.16 * smootherstep(135.0, 210.0, f)
    if f <= 970.0:
        return base_z
    if f <= 1260.0:
        return base_z - 0.08 * smootherstep(1055.0, 1260.0, f)
    return base_z


def supersonic_tumble_deg(f: float) -> tuple[float, float]:
    """(pitch_deg, yaw_deg) aerodynamic tumble before Beat 4 shatter."""
    if f < 955.0 or f > 1260.0:
        if 122.0 <= f <= 210.0:
            u = smootherstep(122.0, 165.0, f)
            return (14.0 * u, -24.0 * u)
        return (0.0, 0.0)
    u = smootherstep(955.0, 1055.0, f)
    return (18.0 * u, -34.0 * u)


def supersonic_shatter_t(f: float) -> float:
    """Fragment separation factor [0.0 intact -> 1.0 shattered shards]."""
    if 1045.0 <= f <= 1260.0:
        return smootherstep(1045.0, 1185.0, f)
    if 1398.0 <= f <= 1560.0:
        return smootherstep(1398.0, 1500.0, f)
    if f > 1560.0:
        return 1.0
    return 0.0


def xray_bullet_weight(f: float) -> float:
    """Cutaway reveal inside supersonic bullet jacket in Beat 4 (`f=915..1245`)."""
    if 915.0 <= f <= 1245.0:
        return smootherstep(915.0, 945.0, f) * (1.0 - smootherstep(1220.0, 1245.0, f))
    return 0.0


def race_fast_x(f: float) -> float:
    """Beat 5 Top Lane (Supersonic): blasts 0.0 -> halts violently at 3 FT (0.9144m)."""
    if f < 1285.0:
        return 0.0
    u = 1.0 - (1.0 - smoothstep(1285.0, 1410.0, f)) ** 2.6
    return lerp(0.0, THREE_FT_X, u)


def race_slow_x(f: float) -> float:
    """Beat 5 Bottom Lane (Heavy Subsonic): steady laminar glide 0.0 -> 1.83m (6 FT)."""
    if f < 1285.0:
        return 0.0
    u = smootherstep(1285.0, 1555.0, f)
    return lerp(0.0, SIX_FT_X, u)


def blueprint_bg_weight(f: float) -> float:
    """Clean @zackdfilms Studio Blue/Cyan Blueprint mix during Beat 5 comparison race."""
    if 1261.0 <= f <= 1560.0:
        return smootherstep(1261.0, 1285.0, f) * (1.0 - smootherstep(1540.0, 1560.0, f))
    return 0.0


def primary_subject_world_pos(f: float) -> tuple[float, float, float]:
    """Primary optical focus anchor tracked by camera & framing QA across Beats 1-6."""
    if f <= 210.0:
        # Track toward bullet-vs-swimmer convergence near 3 FT
        bx = supersonic_x(f)
        return (lerp(bx, 1.05, 0.35), 0.02, -0.60)
    if f <= 1260.0:
        return (supersonic_x(f), 0.0, supersonic_z(f))
    if f <= 1560.0:
        # Midpoint between fast & slow racing bullets
        mid_x = 0.5 * (race_fast_x(f) + race_slow_x(f))
        return (clamp(mid_x, 0.35, 1.35), 0.0, -0.56)
    return (1.28, 0.08, -0.95)
