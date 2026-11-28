"""
Single Source of Truth Contract for Episode #4:
"How a 172-Foot Stealth Bomber Looks Like a Bumblebee on Radar" (30.0 s @ 60 fps = 1800 frames).

Daylight Hollywood-VFX & Enemy Radar Command Room Edition:
- Bright sunlit daytime sky & cloud horizon across all 6 beats (maximum contrast for dark stealth bomber!)
- Architectural Enemy Air-Defense Radar Command Room looking through a panoramic armored window
  at the sunlit airspace in Beat 1 (0 TARGETS blank console) and Beat 6 (0.001 m² bumblebee speck)
- Beat 5 explains 90-degree echo vs 33-degree tilted stealth deflection directly on the REAL
  Conventional Jet and REAL Stealth Bomber in the main daylight sky (zero confusing blue grid wall!)

Every 3D scene builder, camera rig, FX shader, aircraft asset, audio stem, and
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

REAL_WINGSPAN_FT: float = 172.0
REAL_RCS_JET_M2: float = 100.0
REAL_RCS_STEALTH_M2: float = 0.001  # Bumblebee radar cross-section (~0.001 m^2)

# ==============================================================================
# 6-BEAT STRUCTURE (1..1800 @ 60 fps = 30.0 s)
# ==============================================================================
BEATS: dict[str, tuple[int, int]] = {
    "B1_HOOK": (1, 220),          # 0.00 - 3.67 s: Enemy Command Room window -> daylight B-2 over blank radar
    "B2_FACETS": (221, 540),      # 3.67 - 9.00 s: Daylight sky: Realistic Conv Jet echo vs Stealth B-2 facets
    "B3_SDUCTS": (541, 960),      # 9.00 - 16.00 s: Macro X-ray serpentine S-ducts & detailed titanium turbofans
    "B4_RAM_BEE": (961, 1320),    # 16.00 - 22.00 s: Hex RAM heat absorption + live RCS counter -> 3D bumblebee
    "B5_BLUEPRINT": (1321, 1560), # 22.00 - 26.00 s: Main-scene daylight explainer: Real Jet 90-deg vs B-2 33-deg
    "B6_FINAL": (1561, 1800),     # 26.00 - 30.00 s: Enemy Command Room console blip + sunlit B-2 outside window
}

HARD_CUT_FRAMES: tuple[int, ...] = (221, 541, 961, 1321, 1561)

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
    # Outdoor military radar emitter tower (Beats 1, 2 & 5)
    "radar_dish_b1": (-0.98, 1.05, 1.38),
    "radar_dish_b2": (-1.38, 0.15, 1.95),
    "radar_dish_b5": (-1.32, 0.08, 1.84),
    # Enemy Radar Command Room console anchors (Beats 1 & 6)
    "b1_console_scope_pos": (0.44, -1.25, 1.12),
    "b6_scope_pos": (0.00, -0.82, 1.16),
    "b6_scope_radius": 0.36,
    # Beat 2 vertical split comparison heights
    "b2_conv_pos": (0.16, 0.22, 2.74),
    "b2_stealth_pos": (0.16, 0.08, 1.40),
    # Beat 3 & 4 macro stage center
    "macro_center": (0.00, 0.00, 1.85),
    "sduct_intake_local": (-0.25, -0.20, 0.068),
    "sduct_turbine_local": (-0.27, +0.26, -0.018),
    # Beat 4 RAM macro tile & bumblebee RCS showcase
    "ram_tile_pos": (-0.08, 0.00, 1.85),
    "rcs_bee_pos": (0.20, -0.04, 1.86),
    # Beat 5 Main-Scene Daylight Explainer: Real Conv Jet (90-deg tail) vs Real B-2 (33-deg wing)
    "b5_top_mirror_pos": (0.34, 0.08, 2.48),    # Real Conv Jet 90-deg tail reflector
    "b5_top_emitter_pos": (-0.98, 0.08, 2.48),  # Incoming wave origin from left radar
    "b5_bot_facet_pos": (0.30, 0.06, 1.28),     # Real B-2 Stealth Bomber 33-deg tilted wing
    "b5_bot_emitter_pos": (-0.98, 0.06, 1.28),  # Incoming wave origin from left radar
}

# ==============================================================================
# STEALTH BOMBER & CONVENTIONAL JET DIMENSIONS (1:15 Tactical Scale)
# ==============================================================================
BOMBER: dict[str, Any] = {
    "wingspan_m": 3.20,          # Scaled 172-ft B-2 Spirit flying wing
    "chord_m": 1.32,             # Nose to center W-sawtooth trailing apex
    "thickness_m": 0.20,         # Central blended fuselage lifting body height
    "sweep_deg": 33.0,           # Authentic B-2 33-degree leading-edge sweep
    "duct_radius_m": 0.060,      # Internal serpentine S-duct bore radius
    "fan_radius_m": 0.056,       # Buried spinning titanium compressor fan radius
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
    ("wing_span", "172 FT WING", (70, 212)),
    ("facet_deflect", "FACETED DEFLECTION", (260, 528)),
    ("s_ducts", "S-SHAPED DUCTS", (615, 945)),
    ("ram_coating", "RAM COATING", (990, 1145)),
    ("rcs_blip", "0.001 m² BLIP", (1155, 1308)),
    ("tilted_mirror", "TILTED FACETS", (1345, 1545)),
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
    """
    if f <= 220.0:
        # Beat 1: Seen through the wide armored Enemy Command Room window soaring in bright daylight sky!
        u = smootherstep(1.0, 220.0, f)
        pos = (
            lerp(-0.46, 0.36, u),
            lerp(0.95, 0.58, u),
            lerp(2.44, 2.34, u),
        )
        rot_deg = (
            lerp(-22.0, -18.0, u),   # Shows iconic sculpted flying-wing dorsal planform clearly
            lerp(-20.0, -12.0, u),
            lerp(-32.0, -22.0, u),
        )
        return pos, rot_deg

    if f <= 540.0:
        # Beat 2: Lower-half flying wing cruising smoothly in sunlit sky opposite conventional jet
        u = smootherstep(221.0, 540.0, f)
        bx, by, bz = STAGE["b2_stealth_pos"]
        pos = (bx + lerp(-0.14, 0.16, u), by, bz + 0.04 * math.sin(u * math.pi))
        rot_deg = (-24.0, -16.0 + 4.0 * u, -26.0 + 10.0 * u)
        return pos, rot_deg

    if f <= 960.0:
        # Beat 3: Stabilized dorsal macro pose exposing left/right S-shaped intake ducts & titanium fans
        u = smootherstep(541.0, 960.0, f)
        cx, cy, cz = STAGE["macro_center"]
        pos = (cx + 0.14, cy + 0.02, cz)
        rot_deg = (
            lerp(-38.0, -46.0, u),
            lerp(10.0, 16.0, u),
            lerp(-18.0, -6.0, u),
        )
        return pos, rot_deg

    if f <= 1320.0:
        # Beat 4: Bomber glides upper-left in bright sunlit sky behind RAM tile & Bumblebee RCS reveal
        u = smootherstep(961.0, 1320.0, f)
        pos = (lerp(-0.62, -0.46, u), 0.92, lerp(2.38, 2.44, u))
        rot_deg = (-26.0, -18.0, -32.0 + 8.0 * u)
        return pos, rot_deg

    if f <= 1560.0:
        # Beat 5: MAIN-SCENE DAYLIGHT EXPLAINER! Real B-2 angled cleanly in lower half showing
        # razor-sharp 33-deg tilted stealth wing facet deflecting incoming radar wave UP & AWAY!
        u = smootherstep(1321.0, 1560.0, f)
        fx, fy, fz = STAGE["b5_bot_facet_pos"]
        pos = (fx + lerp(-0.04, 0.05, u), fy, fz + 0.02 * math.sin(u * math.pi))
        rot_deg = (
            lerp(-26.0, -22.0, u),   # Shows sleek tilted faceted upper wing + sharp leading edge
            lerp(-22.0, -18.0, u),   # Tilted toward incoming left radar wave so deflection reads instantly
            lerp(-62.0, -54.0, u),   # Side-3/4 profile facing left radar dish
        )
        return pos, rot_deg

    # Beat 6: Seen through the Enemy Command Room panoramic window banking majestically in bright daylight!
    u = smootherstep(1561.0, 1800.0, f)
    pos = (
        lerp(-0.56, 0.48, u),
        lerp(1.10, 0.72, u),
        lerp(2.74, 2.58, u),
    )
    rot_deg = (
        lerp(-24.0, -18.0, u),
        lerp(-24.0, -10.0, u),
        lerp(-36.0, -18.0, u),
    )
    return pos, rot_deg


def conv_jet_pose_at(f: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Conventional Jet pose across Beat 2 (aerial contrast) AND Beat 5 (main-scene 90-deg tail echo explainer).
    """
    if f <= 540.0:
        u = smootherstep(221.0, 540.0, f)
        cx, cy, cz = STAGE["b2_conv_pos"]
        pos = (cx + lerp(-0.16, 0.14, u), cy, cz - 0.03 * math.sin(u * math.pi))
        rot_deg = (-12.0, -10.0 + 3.0 * u, -24.0 + 8.0 * u)
        return pos, rot_deg

    # Beat 5 (1321..1560): Clean side-3/4 profile exposing tall 90-deg vertical tail fin to incoming left radar
    u = smootherstep(1321.0, 1560.0, f)
    mx, my, mz = STAGE["b5_top_mirror_pos"]
    pos = (mx + lerp(-0.04, 0.05, u), my, mz + 0.02 * math.sin(u * math.pi))
    rot_deg = (-8.0, -6.0, lerp(-76.0, -68.0, u))
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


def blueprint_bg_weight(f: float) -> float:
    """Disabled (returns 0.0): Beat 5 now stays in the crisp main sunlit daylight sky with real jets."""
    _ = f
    return 0.0


def radar_pulse_phase(f: float, start_f: float, period_f: float) -> float:
    """Repeating `[0..1]` wave phase for crisp radar emissions."""
    if f < start_f:
        return 0.0
    return ((f - start_f) % period_f) / period_f


def radar_scope_angle_rad(f: float) -> float:
    """Continuous clockwise phosphor sweep angle on Beat 6 Enemy Command Room radar scope."""
    if f < 1561.0:
        return 0.0
    u = (f - 1561.0) / 192.0
    return -2.0 * math.pi * u


def primary_subject_world_pos(f: float) -> tuple[float, float, float]:
    """Primary optical focus anchor tracked by camera & framing QA across Beats 1-6."""
    if f <= 220.0:
        pos, _ = bomber_pose_at(f)
        return (pos[0], pos[1], pos[2] - 0.18)
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
    if f <= 1560.0:
        return (-0.10, 0.06, 1.88)
    return (0.00, 0.00, 1.84)
