"""
Single Source of Truth Contract for Episode #3:
"Why Bulletproof Vests DON'T Send You Flying Backward" (30.0 s @ 60 fps = 1800 frames).

Every 3D scene builder, camera rig, FX shader, character animator, audio stem, and
subtitle script MUST read shared dimensions, milestone frames, labels, camera targets,
and deterministic trajectories exclusively from this module.
Follows AGENTS.md V12 (Blender 5.3.0 + @zackdfilms blueprint).
"""
from __future__ import annotations

import math
from typing import Any, Iterator

# ==============================================================================
# GLOBAL TIMING & METRIC CONSTANTS
# ==============================================================================
FPS: int = 60
DURATION_S: float = 30.0
TOTAL_FRAMES: int = 1800

INCH_TO_M: float = 0.0254
HALF_INCH_M: float = 0.5 * INCH_TO_M  # 0.0127 m (actual chest recoil shift)

# ==============================================================================
# 6-BEAT STRUCTURE (1..1800 @ 60 fps = 30.0 s)
# ==============================================================================
BEATS: dict[str, tuple[int, int]] = {
    "B1_HOOK": (1, 210),          # 0.00 - 3.50 s: Chest impact slow-mo, 0.5-inch shift
    "B2_MOMENTUM": (211, 510),    # 3.50 - 8.50 s: Range wide shot, bullet vs fastball momentum
    "B3_KEVLAR": (511, 900),      # 8.50 - 15.00 s: Macro X-ray golden Kevlar trampoline weave
    "B4_CERAMIC": (901, 1260),    # 15.00 - 21.00 s: Rifle tip shatters on SiC ceramic + Dyneema catch
    "B5_NEWTON": (1261, 1560),    # 21.00 - 26.00 s: Studio blueprint F1 = -F2 equal recoil proof
    "B6_FINAL": (1561, 1800),     # 26.00 - 30.00 s: Hero stands steady, intact glass, bruise payoff
}

HARD_CUT_FRAMES: tuple[int, ...] = (211, 511, 1261, 1561)

# ==============================================================================
# EXACT FRAME MILESTONES (Synchronized 1:1 with config/timeline.json sfx_cues)
# ==============================================================================
EV: dict[str, int] = {
    "b1_fire": 22,
    "b1_impact": 75,
    "b1_sparks_peak": 92,
    "b2_muzzle_fire": 245,
    "b2_vest_hit": 335,
    "b3_xray_peel": 535,
    "b3_kevlar_impact": 615,
    "b3_ripple_peak": 735,
    "b4_rifle_fire": 925,
    "b4_rifle_impact": 965,
    "b4_ceramic_shatter": 1030,
    "b4_dyneema_catch": 1125,
    "b5_dual_fire": 1295,
    "b5_vest_absorb": 1365,
    "b5_vectors_lock": 1405,
    "b6_hero_steady": 1620,
}

# ==============================================================================
# BALLISTIC RANGE & STUDIO GEOMETRY (1:1 Metric Scale)
# ==============================================================================
RANGE: dict[str, Any] = {
    "x_min": -3.20,
    "x_max": 3.20,
    "y_min": -2.80,
    "y_max": 2.80,
    "floor_z": 0.00,
    "ceiling_z": 3.40,
    "chest_z": 1.32,
    "eye_z": 1.62,
    "muzzle_x": -1.10,
    "vest_surface_x": 1.00,
    "glass_wall_x": 1.85,
    # Beat 5 symmetrical Newton's Third Law blueprint arena
    "b5_shooter_x": -1.20,
    "b5_muzzle_x": -0.82,
    "b5_vest_x": 1.05,
    "b5_target_x": 1.20,
    "b5_vector_len": 0.55,
    "b5_vector_z": 1.32,
}

# ==============================================================================
# MACRO MULTI-LAYER ARMOR CUTAWAY (Local-Space Origin = Vest_Macro_Root)
# ==============================================================================
VEST: dict[str, Any] = {
    "root_pos": (1.00, 0.00, 1.32),
    "width_y": 0.28,
    "height_z": 0.34,
    # Local X spans from front strike face (0.000) toward wearer sternum (+0.044)
    "carrier_x": (-0.006, 0.000),
    "ceramic_x": (0.000, 0.010),      # 10 mm Silicon-Carbide (SiC) strike plate
    "kevlar_x": (0.010, 0.028),       # 18 mm Woven golden Kevlar aramid layers
    "dyneema_x": (0.028, 0.044),      # 16 mm UHMWPE Dyneema trauma backer
    "max_kevlar_bulge_m": 0.026,
    "max_dyneema_bulge_m": 0.018,
    "pistol_caliber_m": 0.009,        # 9mm FMJ (Beats 1, 2, 3)
    "rifle_caliber_m": 0.00762,       # 7.62 / 5.56 steel-core penetrator (Beats 4, 5)
}

# ==============================================================================
# MPFB2 CHARACTER CONTRACTS (Hero Target & Shooter)
# ==============================================================================
HERO: dict[str, Any] = {
    "height_m": 1.78,
    "b1_pos": (1.15, 0.00, 0.00),
    "b1_yaw_deg": -82.0,     # Facing -X toward incoming shot & 3/4 toward -Y camera
    "b2_pos": (1.15, 0.00, 0.00),
    "b2_yaw_deg": -84.0,
    "b5_pos": (1.20, 0.00, 0.00),
    "b5_yaw_deg": -90.0,     # Pure profile for symmetrical F1 = -F2 comparison
    "b6_pos": (1.15, 0.00, 0.00),
    "b6_yaw_deg": -76.0,     # Relaxed 3/4 hero payoff toward camera
    "eye_z": 1.62,
    "chest_z": 1.32,
    "recoil_shift_m": HALF_INCH_M,
}

SHOOTER: dict[str, Any] = {
    "height_m": 1.78,
    "b2_pos": (-1.60, 0.00, 0.00),
    "b2_yaw_deg": 84.0,      # Facing +X toward hero target
    "b5_pos": (-1.20, 0.00, 0.00),
    "b5_yaw_deg": 90.0,      # Pure profile facing +X
    "eye_z": 1.62,
    "shoulder_z": 1.38,
    "recoil_shift_m": HALF_INCH_M,
}

# ==============================================================================
# ULTRA-SHORT 3D ENGINEERING CALLOUTS (V12: 1-3 Words Max, NDC Y >= -0.45)
# ==============================================================================
LABELS: list[tuple[str, str, tuple[int, int]]] = [
    ("shift_half_in", "0.5 INCH SHIFT", (92, 204)),
    ("tiny_momentum", "TINY MOMENTUM", (330, 495)),
    ("kevlar_net", "KEVLAR NET", (595, 880)),
    ("ceramic_dyneema", "CERAMIC + DYNEEMA", (990, 1245)),
    ("newton_eq", "F₁ = −F₂", (1320, 1545)),
    ("equal_recoil", "EQUAL RECOIL", (1385, 1545)),
    ("just_bruise", "JUST A BRUISE", (1615, 1785)),
]

# ==============================================================================
# HIGH-LEVEL CAMERA TARGET PLAN (Consumed by cinema_camera.py & verify_framing.py)
# ==============================================================================
CAMERA_TARGETS: dict[str, dict[str, Any]] = {
    "B1_HOOK": {
        "cam_start": (0.38, -2.05, 1.38),
        "cam_end": (0.58, -1.82, 1.36),
        "tgt_start": (0.98, 0.00, 1.32),
        "tgt_end": (1.10, 0.00, 1.34),
        "lens_mm": 32.0,
        "fstop": 7.5,
        "hard_cut_in": False,
    },
    "B2_MOMENTUM": {
        "cam_start": (-0.30, -2.85, 1.34),
        "cam_end": (0.28, -2.70, 1.34),
        "tgt_start": (-0.25, 0.00, 1.30),
        "tgt_end": (0.35, 0.00, 1.30),
        "lens_mm": 26.0,
        "fstop": 7.5,
        "hard_cut_in": True,
    },
    "B3_KEVLAR": {
        "cam_offset_start": (-0.18, -0.52, 0.04),
        "cam_offset_end": (-0.14, -0.54, 0.05),
        "tgt_offset": (0.01, 0.00, 0.00),
        "lens_mm": 38.0,
        "fstop_pre_label": 4.0,
        "fstop_label": 7.5,
        "hard_cut_in": True,
    },
    "B4_CERAMIC": {
        "cam_offset_start": (-0.14, -0.54, 0.05),
        "cam_offset_end": (-0.22, -0.78, 0.07),
        "tgt_offset": (0.01, 0.00, 0.00),
        "lens_start_mm": 38.0,
        "lens_end_mm": 30.0,
        "fstop": 7.5,
        "hard_cut_in": False,  # Smooth continuous pull-back from Beat 3
    },
    "B5_NEWTON": {
        "cam_start": (0.00, -3.55, 1.32),
        "cam_end": (0.00, -3.35, 1.32),
        "tgt_start": (0.00, 0.00, 1.24),
        "tgt_end": (0.00, 0.00, 1.24),
        "lens_mm": 26.0,
        "fstop": 8.0,
        "hard_cut_in": True,
    },
    "B6_FINAL": {
        "cam_start": (0.48, -2.08, 1.36),
        "cam_end": (0.40, -2.26, 1.38),
        "tgt_start": (1.12, 0.00, 1.28),
        "tgt_end": (1.14, 0.00, 1.30),
        "lens_mm": 28.0,
        "fstop": 7.5,
        "hard_cut_in": True,
    },
}


# ==============================================================================
# PURE-PYTHON MATH & BLENDER 5.3.0 SAFETY HELPERS
# ==============================================================================
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
    """Blender 4.x & 5.3.0 Slotted Actions safe FCurve iterator (never access action.fcurves blindly)."""
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


def get_mix_sockets(node: Any) -> tuple[Any, Any, Any]:
    """
    Blender 5.3.0 ShaderNodeMix safe (Factor, A, B) socket resolver.
    Slices `node.inputs[2:]` and maps `data_type == 'FLOAT'` to socket type `'VALUE'`
    so Factor (`inputs[0]`) is NEVER misidentified as input A.
    """
    dtype = getattr(node, "data_type", "RGBA")
    sock_type = "VALUE" if dtype == "FLOAT" else dtype
    fac = node.inputs[0]
    typed = [s for s in node.inputs[2:] if s.type == sock_type]
    if len(typed) >= 2:
        return fac, typed[0], typed[1]
    if dtype == "FLOAT":
        return fac, node.inputs[2], node.inputs[3]
    if dtype == "VECTOR":
        return fac, node.inputs[4], node.inputs[5]
    return fac, node.inputs[6], node.inputs[7]


# ==============================================================================
# DETERMINISTIC BALLISTIC, RECOIL & ARMOR-RESPONSE TRAJECTORIES
# ==============================================================================
def pistol_bullet_x(f: float) -> float:
    """World X of 9mm handgun round across Beats 1, 2, and 3."""
    vx = float(RANGE["vest_surface_x"])
    if f <= 210.0:
        # Beat 1: fires from X=-0.55m at f=22 -> slams chest vest (+1.00m) at f=75 -> embeds +1.012m
        if f < 22.0:
            return -0.55
        if f <= 75.0:
            u = (f - 22.0) / 53.0
            return lerp(-0.55, vx, u)
        u_stop = 1.0 - (1.0 - smoothstep(75.0, 102.0, f)) ** 2.4
        return vx + 0.012 * u_stop

    if f <= 510.0:
        # Beat 2: fires from shooter muzzle (-1.10m) at f=245 -> hits vest (+1.00m) at f=335
        mx = float(RANGE["muzzle_x"])
        if f < 245.0:
            return mx
        if f <= 335.0:
            u = (f - 245.0) / 90.0
            return lerp(mx, vx, u)
        u_stop = 1.0 - (1.0 - smoothstep(335.0, 365.0, f)) ** 2.2
        return vx + 0.012 * u_stop

    if f <= 900.0:
        # Beat 3: macro shot enters from +0.72m -> strikes Kevlar face (+1.010m) at f=615
        # -> stretches trampoline to +1.036m at f=735 -> elastic slight rebound to +1.028m
        kx = vx + float(VEST["kevlar_x"][0])
        if f < 535.0:
            return vx - 0.28
        if f <= 615.0:
            u = (f - 535.0) / 80.0
            return lerp(vx - 0.28, kx, u)
        if f <= 735.0:
            u_stretch = 1.0 - (1.0 - smoothstep(615.0, 735.0, f)) ** 2.1
            return kx + float(VEST["max_kevlar_bulge_m"]) * u_stretch
        u_reb = smootherstep(735.0, 860.0, f)
        return kx + lerp(float(VEST["max_kevlar_bulge_m"]), 0.018, u_reb)

    return vx + 0.018


def rifle_bullet_x(f: float) -> float:
    """World X of high-velocity pointed rifle round across Beat 4 (ceramic) & Beat 5 (Newton)."""
    vx = float(RANGE["vest_surface_x"])
    if f <= 1260.0:
        # Beat 4: approaches from +0.68m (f=925) -> hits SiC ceramic (1.000m) at f=965
        # -> nose shatters through ceramic into Dyneema (+1.030m) by f=1125
        if f < 925.0:
            return vx - 0.32
        if f <= 965.0:
            u = (f - 925.0) / 40.0
            return lerp(vx - 0.32, vx, u)
        u_pen = 1.0 - (1.0 - smoothstep(965.0, 1125.0, f)) ** 2.3
        return vx + 0.030 * u_pen

    if f <= 1560.0:
        # Beat 5: fires from blueprint shooter muzzle (-0.82m) at f=1295 -> target vest (+1.05m) at f=1365
        mx5 = float(RANGE["b5_muzzle_x"])
        vx5 = float(RANGE["b5_vest_x"])
        if f < 1295.0:
            return mx5
        if f <= 1365.0:
            u = (f - 1295.0) / 70.0
            return lerp(mx5, vx5, u)
        u_stop = 1.0 - (1.0 - smoothstep(1365.0, 1390.0, f)) ** 2.4
        return vx5 + 0.010 * u_stop

    return float(RANGE["b5_vest_x"]) + 0.010


def bullet_world_pos(f: float) -> tuple[float, float, float]:
    """Active hero projectile world coordinate `(x, y, z)` for Beats 1-5."""
    cz = float(RANGE["chest_z"])
    if f <= 900.0:
        return (pistol_bullet_x(f), 0.0, cz)
    return (rifle_bullet_x(f), 0.0, cz)


def _damped_recoil_pulse(f: float, f_hit: float, amp_m: float = HALF_INCH_M) -> float:
    """Physically realistic 0.5-inch (12.7 mm) damped elastic torso/shoulder recoil pulse."""
    if f < f_hit:
        return 0.0
    dt = f - f_hit
    rise = 1.0 - (1.0 - smoothstep(f_hit, f_hit + 14.0, f)) ** 2.2
    settle = math.exp(-0.018 * max(0.0, dt - 14.0))
    return amp_m * rise * (0.68 + 0.32 * settle)


def hero_recoil_x(f: float) -> float:
    """World +X displacement of armored hero on impact (strictly <= 0.5 inch = 0.0127 m)."""
    if f <= 210.0:
        return _damped_recoil_pulse(f, float(EV["b1_impact"]), HALF_INCH_M)
    if f <= 510.0:
        return _damped_recoil_pulse(f, float(EV["b2_vest_hit"]), HALF_INCH_M)
    if 1261.0 <= f <= 1560.0:
        return _damped_recoil_pulse(f, float(EV["b5_vest_absorb"]), HALF_INCH_M)
    return 0.0


def shooter_recoil_x(f: float) -> float:
    """World -X displacement of shooter on muzzle fire (equal & opposite -0.0127 m)."""
    if 211.0 <= f <= 510.0:
        return -_damped_recoil_pulse(f, float(EV["b2_muzzle_fire"]), HALF_INCH_M)
    if 1261.0 <= f <= 1560.0:
        return -_damped_recoil_pulse(f, float(EV["b5_dual_fire"]), HALF_INCH_M)
    return 0.0


def xray_vest_weight(f: float) -> float:
    """Local-space outer carrier X-ray cutaway reveal across Beats 3-4 (`520..1250`)."""
    if 520.0 <= f <= 1250.0:
        return smootherstep(520.0, 560.0, f) * (1.0 - smootherstep(1225.0, 1250.0, f))
    return 0.0


def kevlar_ripple_t(f: float) -> float:
    """Normalized `[0..1]` Kevlar trampoline deflection & radial weave wave in Beat 3."""
    if 615.0 <= f <= 900.0:
        rise = 1.0 - (1.0 - smoothstep(615.0, 735.0, f)) ** 2.0
        hold = 1.0 - 0.25 * smootherstep(735.0, 900.0, f)
        return clamp(rise * hold, 0.0, 1.0)
    return 0.0


def ceramic_shatter_t(f: float) -> float:
    """Normalized `[0..1]` silicon-carbide ceramic fracture & rifle tip pulverization in Beat 4."""
    if 965.0 <= f <= 1260.0:
        return smootherstep(965.0, 1110.0, f)
    return 0.0


def dyneema_bulge_t(f: float) -> float:
    """Normalized `[0..1]` rear UHMWPE Dyneema fragment-catch flex in Beat 4."""
    if 1030.0 <= f <= 1260.0:
        rise = smootherstep(1030.0, 1145.0, f)
        relax = 1.0 - 0.18 * smootherstep(1145.0, 1260.0, f)
        return clamp(rise * relax, 0.0, 1.0)
    return 0.0


def spark_burst_t(f: float) -> float:
    """Normalized `[0..1]` impact spark burst intensity on Beats 1, 2, and 4."""
    for f0, peak_dt, fade_dt in (
        (float(EV["b1_impact"]), 17.0, 68.0),
        (float(EV["b2_vest_hit"]), 12.0, 48.0),
        (float(EV["b4_rifle_impact"]), 18.0, 75.0),
    ):
        if f0 <= f <= f0 + fade_dt:
            rise = smoothstep(f0, f0 + peak_dt * 0.45, f)
            decay = 1.0 - smootherstep(f0 + peak_dt, f0 + fade_dt, f)
            return clamp(rise * decay, 0.0, 1.0)
    return 0.0


def newton_vector_t(f: float) -> float:
    """Normalized `[0..1]` opposing `F₁` / `−F₂` 3D arrow reveal in Beat 5 (`1295..1555`)."""
    if 1295.0 <= f <= 1555.0:
        return smootherstep(1295.0, 1405.0, f) * (1.0 - smootherstep(1535.0, 1555.0, f))
    return 0.0


def blueprint_bg_weight(f: float) -> float:
    """Clean `@zackdfilms` blue/cyan studio blueprint mix during Beat 5 (`1261..1560`)."""
    if 1261.0 <= f <= 1560.0:
        return smootherstep(1261.0, 1285.0, f) * (1.0 - smootherstep(1540.0, 1560.0, f))
    return 0.0


def primary_subject_world_pos(f: float) -> tuple[float, float, float]:
    """Primary optical focus anchor tracked by camera & framing QA across Beats 1-6."""
    cz = float(RANGE["chest_z"])
    if f <= 210.0:
        bx = pistol_bullet_x(f)
        return (lerp(bx, float(RANGE["vest_surface_x"]) + 0.08, 0.55), 0.0, cz)
    if f <= 510.0:
        bx = pistol_bullet_x(f)
        return (clamp(lerp(bx, 0.05, 0.45), -0.35, 0.55), 0.0, cz - 0.02)
    if f <= 1260.0:
        rx, ry, rz = VEST["root_pos"]
        return (rx + 0.012, ry, rz)
    if f <= 1560.0:
        return (0.0, 0.0, cz - 0.08)
    hx, hy, _ = HERO["b6_pos"]
    return (hx - 0.03, hy, cz - 0.02)
