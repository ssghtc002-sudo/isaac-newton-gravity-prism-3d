"""
Single Source of Truth Contract for Episode #4:
"How a 172-Foot Stealth Bomber Looks Like a Bumblebee on Radar" (30.0 s @ 60 fps = 1800 frames).

Action Thriller & 3-Character MPFB2 Edition (V3):
- Beat 1 (1..220): Inside Enemy Air-Defense Radar Bunker with TWO 4K MPFB2 characters
  (Commander pointing urgently out the window + Radar Operator scanning a BLANK scope)
  while the photo-accurate 172-ft B-2 Spirit banks dramatically across sunlit sky outside!
- Beat 2 (221..540): High-Altitude Intercept Action — Normal Jet hit by crimson missile-lock
  reticle vs B-2 Spirit's razor-angled 33-deg facets & W-sawtooth trailing edges!
- Beat 3 (541..960): Dynamic Banking Macro X-Ray revealing twin serpentine S-shaped air tunnels
  & spinning 20-blade titanium turbofans trapping radar pulses inside!
- Beat 4 (961..1320): Micro-Action Carbon-Iron RAM Honeycomb Heat Trap + Live Shrinking
  100 m² -> 0.001 m² Counter & Flapping 3D Bumblebee!
- Beat 5A (1321..1430): 1:1 Scale B-2 Cockpit Action Close-Up starring 4K MPFB2 Stealth Pilot
  (HGU-55 helmet, reflective gold visor, oxygen mask, glowing HUD glass, banking flight stick)!
- Beat 5B (1431..1560): Daylight SAM Radar Deflection Showdown on the real Conventional Jet
  (90-deg crimson return arrow) vs real B-2 Spirit (33-deg golden deflection arrow into space)!
- Beat 6 (1561..1800): Chilling Payoff Twist Back in Enemy Bunker — Commander & Operator spot
  the tiny 0.001 m² bumblebee speck right as the massive B-2 looms overhead opening its belly bay!

Strictly follows AGENTS.md V12 (Blender 5.3.0 + @zackdfilms blueprint).
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

REAL_WINGSPAN_FT: float = 172.0
REAL_RCS_JET_M2: float = 100.0
REAL_RCS_STEALTH_M2: float = 0.001  # Bumblebee radar cross-section (~0.001 m^2)

# ==============================================================================
# 6-BEAT STRUCTURE (1..1800 @ 60 fps = 30.0 s)
# ==============================================================================
BEATS: dict[str, tuple[int, int]] = {
    "B1_HOOK": (1, 220),          # 0.00 - 3.67 s: Enemy Bunker Crew (Commander + Operator) vs Sunlit B-2
    "B2_FACETS": (221, 540),      # 3.67 - 9.00 s: Intercept Action: Conv Jet Missile Lock vs B-2 33-deg Facets
    "B3_SDUCTS": (541, 960),      # 9.00 - 16.00 s: Banking Macro X-Ray: S-Ducts & Spinning Titanium Turbines
    "B4_RAM_BEE": (961, 1320),    # 16.00 - 22.00 s: Carbon RAM Heat Trap + Live RCS Shrink -> 3D Bumblebee
    "B5_BLUEPRINT": (1321, 1560), # 22.00 - 26.00 s: B5A Pilot Cockpit Close-Up (1321..1430) + B5B SAM Showdown (1431..1560)
    "B6_FINAL": (1561, 1800),     # 26.00 - 30.00 s: Bunker Twist: Tiny Blip on Scope as B-2 Looms Overhead
}

# Beat 5A -> 5B split frame for Pilot Cockpit Close-Up -> SAM Radar Deflection Showdown
B5_COCKPIT_END: int = 1430
B5_SHOWDOWN_START: int = 1431

HARD_CUT_FRAMES: tuple[int, ...] = (221, 541, 961, 1321, 1431, 1561)

# ==============================================================================
# EXACT FRAME MILESTONES (Synchronized 1:1 with config/timeline.json sfx_cues)
# ==============================================================================
EV: dict[str, int] = {
    "b1_radar_ping": 18,
    "b1_wing_deflect": 68,
    "b1_blank_scope": 135,
    "b2_conv_ping": 265,
    "b2_conv_echo": 318,
    "b2_stealth_deflect": 410,
    "b3_xray_sweep": 575,
    "b3_duct_enter": 655,
    "b3_duct_trap": 785,
    "b4_ram_wave": 995,
    "b4_heat_absorb": 1075,
    "b4_rcs_shrink": 1160,
    "b5_mirror_echo": 1365,
    "b5_facet_scatter": 1445,
    "b6_moon_pass": 1600,
    "b6_final_blip": 1695,
}

# ==============================================================================
# 3-CHARACTER MPFB2 CREW ANCHORS (COMMANDER, RADAR OPERATOR, STEALTH PILOT)
# ==============================================================================
CREW: dict[str, Any] = {
    # Enemy Radar Bunker — Radar Operator (seated right of console, angled 3/4 toward scope & window)
    "operator_b1_pos": (0.76, -1.78, -0.34),
    "operator_b1_yaw_deg": 136.0,
    "operator_b6_pos": (0.70, -1.62, -0.34),
    "operator_b6_yaw_deg": 128.0,
    "operator_eye_z": 1.28,
    "operator_height_m": 1.76,

    # Enemy Radar Bunker — Air-Defense Commander (standing left/rear, pointing out window & peering at scope)
    "commander_b1_pos": (-0.80, -1.84, -0.22),
    "commander_b1_yaw_deg": -128.0,
    "commander_b6_pos": (-0.72, -1.68, -0.22),
    "commander_b6_yaw_deg": -122.0,
    "commander_eye_z": 1.44,
    "commander_height_m": 1.80,

    # Beat 5A (1321..1430) — 1:1 Scale B-2 Spirit Cockpit Hero Stage with MPFB2 Stealth Pilot
    "pilot_cockpit_origin": (0.00, 0.00, 1.85),
    "pilot_pos": (0.00, 0.14, 0.42),
    "pilot_yaw_deg": -12.0,
    "pilot_eye_z": 2.04,
    "pilot_height_m": 1.78,
}

# ==============================================================================
# AIRSPACE, ENEMY RADAR COMMAND ROOM & DAYLIGHT STAGE GEOMETRY (1:15 Scale)
# ==============================================================================
STAGE: dict[str, Any] = {
    "floor_z": 0.00,
    "backdrop_y": 8.50,
    "backdrop_x_span": 24.0,
    "backdrop_z_min": -0.50,
    "backdrop_z_max": 12.00,
    "floor_x_span": 24.0,
    "floor_y_span": 20.0,
    # Outdoor military SAM radar emitter tower (Beats 1, 2 & 5B)
    "radar_dish_b1": (-0.98, 1.12, 1.44),
    "radar_dish_b2": (-1.36, 0.15, 1.95),
    "radar_dish_b5": (-1.32, 0.08, 1.86),
    # Enemy Radar Command Room console anchors (Beats 1 & 6 — elevated cleanly above desk bolster!)
    "b1_console_scope_pos": (0.14, -1.16, 1.18),
    "b6_scope_pos": (0.00, -0.88, 1.18),
    "b6_scope_radius": 0.34,
    # Beat 2 vertical split comparison heights
    "b2_conv_pos": (0.16, 0.20, 2.72),
    "b2_stealth_pos": (0.16, 0.08, 1.42),
    # Beat 3 & 4 macro stage center
    "macro_center": (0.00, 0.00, 1.85),
    "sduct_intake_local": (-0.25, -0.22, 0.064),
    "sduct_turbine_local": (-0.25, +0.22, 0.032),
    # Beat 4 RAM macro tile & bumblebee RCS showcase
    "ram_tile_pos": (-0.08, 0.00, 1.85),
    "rcs_bee_pos": (0.20, -0.04, 1.86),
    # Beat 5B (1431..1560) Main-Scene Daylight SAM Showdown: Real Conv Jet (90-deg) vs Real B-2 (33-deg)
    "b5_top_mirror_pos": (0.34, 0.08, 2.50),
    "b5_top_emitter_pos": (-0.98, 0.08, 2.50),
    "b5_bot_facet_pos": (0.30, 0.06, 1.32),
    "b5_bot_emitter_pos": (-0.98, 0.06, 1.32),
}

# ==============================================================================
# STEALTH BOMBER & CONVENTIONAL JET DIMENSIONS (1:15 Tactical Scale)
# ==============================================================================
BOMBER: dict[str, Any] = {
    "wingspan_m": 3.20,          # Scaled 172-ft B-2 Spirit flying wing
    "chord_m": 1.38,             # Hawk-beak nose to center W-sawtooth beaver-tail apex
    "thickness_m": 0.23,         # Central blended fuselage lifting body height
    "sweep_deg": 33.0,           # Authentic B-2 33-degree leading-edge sweep
    "duct_radius_m": 0.056,      # Internal serpentine S-duct bore radius
    "fan_radius_m": 0.052,       # Buried spinning titanium compressor fan radius
}

CONV_JET: dict[str, Any] = {
    "length_m": 1.56,
    "wingspan_m": 1.46,
    "fuselage_r_m": 0.092,
    "tail_height_m": 0.36,
}

# ==============================================================================
# ULTRA-SHORT 3D ENGINEERING CALLOUTS (V12: 1-3 Words Max, NDC Y >= -0.45)
# ==============================================================================
LABELS: list[tuple[str, str, tuple[int, int]]] = [
    ("wing_span", "0 TARGETS", (70, 212)),
    ("facet_deflect", "33° DEFLECTION", (260, 528)),
    ("s_ducts", "HIDDEN TURBINES", (615, 945)),
    ("ram_coating", "HEAT ABSORPTION", (990, 1145)),
    ("rcs_blip", "0.001 m² BLIP", (1155, 1308)),
    ("tilted_mirror", "MISSILE BLIND", (1445, 1550)),
    ("bumblebee_size", "BUMBLEBEE SIZE", (1595, 1785)),
]


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
# DETERMINISTIC FLIGHT, RADAR WAVE, X-RAY & RAM PHYSICS TRAJECTORIES
# ==============================================================================
def bomber_pose_at(f: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Return `(location_xyz, euler_deg_xyz)` for `Stealth_Bomber_Root` across all 6 beats.
    Local bomber convention: Nose points toward `-Y` (forward), wings span along `X`, dorsal `+Z`.
    Positive `rx` (`+24..+38 deg`) tilts the sculpted 3D dorsal cockpit, 4-pane windshield,
    twin dorsal engine nacelles & W-sawtooth wings toward the camera (like the USAF B-2 photo!).
    """
    if f <= 220.0:
        # Beat 1: Dramatic banking flyby outside the Enemy Bunker panoramic window!
        u = smootherstep(1.0, 220.0, f)
        pos = (
            lerp(-0.46, 0.32, u),
            lerp(1.22, 0.56, u),
            lerp(2.58, 2.34, u),
        )
        rot_deg = (
            lerp(+34.0, +28.0, u),   # Shows sculpted dorsal cockpit, 4-pane glass & nacelles like USAF reference!
            lerp(+24.0, +12.0, u),   # Hard cinematic banking roll
            lerp(-40.0, -20.0, u),
        )
        return pos, rot_deg

    if f <= 540.0:
        # Beat 2: High-altitude intercept contrast — sleek banking B-2 slicing below normal jet
        u = smootherstep(221.0, 540.0, f)
        bx, by, bz = STAGE["b2_stealth_pos"]
        pos = (bx + lerp(-0.14, 0.16, u), by, bz + 0.05 * math.sin(u * math.pi))
        rot_deg = (+30.0, +18.0 - 6.0 * u, -30.0 + 10.0 * u)
        return pos, rot_deg

    if f <= 960.0:
        # Beat 3: Banking dorsal macro pose exposing internal serpentine S-ducts & spinning turbofans
        u = smootherstep(541.0, 960.0, f)
        cx, cy, cz = STAGE["macro_center"]
        pos = (cx + 0.12, cy + 0.04, cz - 0.02)
        rot_deg = (
            lerp(+32.0, +38.0, u),
            lerp(-8.0, -14.0, u),
            lerp(-16.0, -6.0, u),
        )
        return pos, rot_deg

    if f <= 1320.0:
        # Beat 4: Bomber banks high in sunlit sky behind RAM honeycomb & Bumblebee implosion
        u = smootherstep(961.0, 1320.0, f)
        pos = (lerp(-0.62, -0.46, u), 0.92, lerp(2.38, 2.44, u))
        rot_deg = (+30.0, +18.0, -34.0 + 8.0 * u)
        return pos, rot_deg

    if f <= 1560.0:
        # Beat 5B (1431..1560): Daylight SAM Radar Deflection Showdown (hidden during Beat 5A 1:1 cockpit!)
        u = smootherstep(float(B5_SHOWDOWN_START), 1560.0, max(float(B5_SHOWDOWN_START), f))
        fx, fy, fz = STAGE["b5_bot_facet_pos"]
        pos = (fx + lerp(-0.04, 0.05, u), fy, fz + 0.02 * math.sin(u * math.pi))
        rot_deg = (
            lerp(+28.0, +24.0, u),
            lerp(+20.0, +15.0, u),
            lerp(-54.0, -46.0, u),
        )
        return pos, rot_deg

    # Beat 6: Looms massive & dramatic right outside the Enemy Bunker panoramic window!
    u = smootherstep(1561.0, 1800.0, f)
    pos = (
        lerp(-0.50, 0.44, u),
        lerp(1.12, 0.62, u),
        lerp(2.74, 2.50, u),
    )
    rot_deg = (
        lerp(+32.0, +24.0, u),
        lerp(+24.0, +10.0, u),
        lerp(-38.0, -16.0, u),
    )
    return pos, rot_deg


def conv_jet_pose_at(f: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Conventional Jet pose across Beat 2 (intercept action) AND Beat 5B (SAM 90-deg echo showdown)."""
    if f <= 540.0:
        u = smootherstep(221.0, 540.0, f)
        cx, cy, cz = STAGE["b2_conv_pos"]
        pos = (cx + lerp(-0.16, 0.14, u), cy, cz - 0.03 * math.sin(u * math.pi))
        rot_deg = (+18.0, +12.0 - 4.0 * u, -52.0 + 10.0 * u)
        return pos, rot_deg

    # Beat 5B (1431..1560): Clean side-3/4 profile exposing tall 90-deg vertical tail fin
    u = smootherstep(float(B5_SHOWDOWN_START), 1560.0, max(float(B5_SHOWDOWN_START), f))
    mx, my, mz = STAGE["b5_top_mirror_pos"]
    pos = (mx + lerp(-0.04, 0.05, u), my, mz + 0.02 * math.sin(u * math.pi))
    rot_deg = (+16.0, +10.0, lerp(-68.0, -60.0, u))
    return pos, rot_deg


def xray_skin_weight(f: float) -> float:
    """Normalized `[0..1]` local-space dorsal skin X-ray cutaway reveal across Beat 3 (`555..955`)."""
    if 555.0 <= f <= 955.0:
        return smootherstep(555.0, 615.0, f) * (1.0 - smootherstep(925.0, 955.0, f))
    return 0.0


def xray_scan_progress(f: float) -> float:
    """Normalized `[0..1]` sweeping X-ray front moving aft along bomber chord in Beat 3."""
    if f < 555.0:
        return 0.0
    if f <= 665.0:
        return smootherstep(555.0, 665.0, f)
    if f <= 920.0:
        return 1.0
    return 1.0 - smootherstep(920.0, 955.0, f)


def sduct_wave_progress(f: float) -> float:
    """Normalized `[0..1]` progress of trapped cyan RF pulse traveling into S-duct and dying out."""
    if f < 640.0 or f > 955.0:
        return 0.0
    dt = (f - 640.0) % 150.0
    return clamp(dt / 135.0, 0.0, 1.0)


def ram_heat_t(f: float) -> float:
    """Normalized `[0..1]` warm thermal glow inside hexagonal RAM lattice during Beat 4 (`995..1220`)."""
    if 995.0 <= f <= 1220.0:
        return smootherstep(995.0, 1085.0, f) * (1.0 - smootherstep(1155.0, 1220.0, f))
    return 0.0


def rcs_shrink_t(f: float) -> float:
    """Normalized `[0..1]` collapse from 100 m^2 red jet RCS orb to 0.001 m^2 bumblebee (`1135..1320`)."""
    if f < 1135.0:
        return 0.0
    if f <= 1235.0:
        return smootherstep(1135.0, 1235.0, f)
    return 1.0 if f <= 1320.0 else 0.0


def bomb_bay_open_t(f: float) -> float:
    """Normalized `[0..1]` dramatic Beat 6 twist: belly bay glow & doors cracking open (`1665..1800`)."""
    if f < 1665.0:
        return 0.0
    return smootherstep(1665.0, 1755.0, f)


def blueprint_bg_weight(f: float) -> float:
    _ = f
    return 0.0


def radar_pulse_phase(f: float, start_f: float, period_f: float) -> float:
    if f < start_f:
        return 0.0
    return ((f - start_f) % period_f) / period_f


def radar_scope_angle_rad(f: float) -> float:
    if f < 1561.0:
        return 0.0
    u = (f - 1561.0) / 192.0
    return -2.0 * math.pi * u


def primary_subject_world_pos(f: float) -> tuple[float, float, float]:
    """Primary optical focus anchor tracked by camera & framing QA across Beats 1-6."""
    if f <= 220.0:
        pos, _ = bomber_pose_at(f)
        return (pos[0], pos[1], pos[2] - 0.22)
    if f <= 540.0:
        return (0.04, 0.12, 2.02)
    if f <= 960.0:
        cx, cy, cz = STAGE["macro_center"]
        return (cx - 0.05, cy - 0.02, cz + 0.02)
    if f <= 1320.0:
        u = smootherstep(1125.0, 1185.0, f)
        rx, ry, rz = STAGE["ram_tile_pos"]
        bx, by, bz = STAGE["rcs_bee_pos"]
        return (lerp(rx, bx, u), lerp(ry, by, u), lerp(rz, bz, u))
    if f <= float(B5_COCKPIT_END):
        return (0.00, 0.10, float(CREW["pilot_eye_z"]) - 0.04)
    if f <= 1560.0:
        return (-0.10, 0.06, 1.88)
    return (0.00, 0.00, 1.84)
