"""
ejection_contract.py — SINGLE SOURCE OF TRUTH for Episode #1
"Why Ejecting From a Fighter Jet Shrinks You by 1 Inch" (30.0 s @ 60 fps = 1800 frames).

Owned by the Lead Director. ALL specialist modules (environment, canopy, seat, pilot, spine,
labels, camera, audio) MUST import timings / coordinates / trajectories from here and MUST NOT
redefine them. Pure Python (no bpy / mathutils) so it also runs inside verify_framing.py and
audio scripts.

WORLD AXES (metres):  +Y = jet nose / flight direction, +Z = up, +X = pilot's RIGHT.
SEAT-LOCAL frame == world frame while the seat sits in the cockpit (seat root at SRP, identity rot).
SRP (Seat Reference Point) = top-centre of the seat-pan cushion = world (0, 0, 0) at rest.

Return types: every *_location / *_local function returns a plain tuple (x, y, z) of floats;
every *_rotation_euler returns a plain tuple (rx, ry, rz) in RADIANS, Euler order 'XYZ'.
Wrap with mathutils.Vector(...) / Euler(..., 'XYZ') inside Blender.
"""
from __future__ import annotations

import math

Vec3 = tuple[float, float, float]

# ---------------------------------------------------------------------------
# 1. GLOBAL TIMING
# ---------------------------------------------------------------------------
FPS = 60
TOTAL_FRAMES = 1800
DURATION_S = 30.0

BEATS: dict[int, tuple[int, int]] = {
    1: (1, 288),      # 0.00–4.80 s  Viral cockpit panic hook + handle YANK
    2: (289, 588),    # 4.80–9.80 s  MDC canopy-breaker detonation (bullet-time)
    3: (589, 888),    # 9.80–14.80 s Leg-restraint garters + catapult ignition
    4: (889, 1236),   # 14.80–20.60 s Spine X-ray compression (−2.5 cm ≈ 1 inch)
    5: (1237, 1530),  # 20.60–25.50 s Under-seat rocket, tail-fin clearance, drogue chute
    6: (1531, 1800),  # 25.50–30.00 s Seat-man separation + main parachute blossom finale
}

# Representative QA frames (one per beat) + dense Beat-1 hook frames
PREVIEW_FRAMES = [100, 430, 760, 1080, 1420, 1700]
BEAT1_DENSE_FRAMES = [20, 70, 120, 140, 152, 160, 168, 200, 250]

EV: dict[str, int] = {
    # Beat 1 — cockpit panic + handle
    "alarm_start": 1,            # master-caution red flashing starts immediately
    "reach_start": 60,           # both hands leave stick/throttle, dive down to handle
    "grip": 130,                 # both hands closed on yellow-black handle
    "yank_start": 150,           # VIOLENT upward yank begins (≈2.5 s — inside the 3 s hook!)
    "yank_end": 168,             # handle fully pulled (cable visible)
    "brace_start": 176,          # head slams back to headbox, elbows tuck
    "brace_end": 230,
    "cord_spark": 282,           # MDC cord ignition spark travels along canopy
    # Beat 2 — canopy
    "mdc_detonate": 300,         # canopy glass fractures along the cord (flash)
    "shards_fly_end": 560,       # bullet-time shard drift ends (off-frame by now)
    "shards_hidden": 589,        # ALL canopy shards CONSTANT-hidden from here on
    # Beat 3 — garters + catapult
    "garter_start": 610,         # leg-restraint garters start reeling boots back
    "garter_end": 650,
    "catapult_ignite": 700,      # catapult cartridge fires, seat starts moving up rails
    # Beat 4 — spine
    "spine_compress_start": 930,
    "spine_compress_end": 1080,
    "rail_exit": 1236,           # seat leaves the rails
    # Beat 5 — rocket + drogue
    "rocket_ignite": 1250,
    "tail_clear": 1365,          # seat passes over vertical tail fin (approx, see jet_offset)
    "drogue_fire": 1400,
    "drogue_full": 1440,
    "rocket_burnout": 1460,
    # Beat 6 — separation + main chute
    "separation": 1545,          # seat-man separation; handle hidden; pilot -> chute
    "main_line_stretch": 1580,
    "main_full": 1650,
    "seat_hidden": 1740,         # falling seat (+drogue) CONSTANT-hidden from here on
}

# X-ray windows: (ramp_in_start, ramp_in_end, ramp_out_start, ramp_out_end)
XRAY_FUSELAGE = (320, 350, 1200, 1236)   # fuselage / cockpit tub turns to X-ray glass
XRAY_BODY = (895, 930, 1200, 1236)       # pilot flight-suit + skin turn translucent (spine visible)

# ---------------------------------------------------------------------------
# 2. FIXED GEOMETRY (jet-local == world while jet_offset == 0)
# ---------------------------------------------------------------------------
SRP: Vec3 = (0.0, 0.0, 0.0)
RAIL_TILT_DEG = 15.0                                  # rails & seat back lean 15° aft
RAIL_AXIS: Vec3 = (0.0, -math.sin(math.radians(RAIL_TILT_DEG)), math.cos(math.radians(RAIL_TILT_DEG)))
RAIL_LENGTH_M = 2.0
RAIL_X = (-0.20, 0.20)                                # twin guide rails behind seat back
RAIL_BASE: Vec3 = (0.0, -0.30, -0.25)                 # catapult tube base (behind/below seat)

COCKPIT = {
    "floor_z": -0.30,
    "sill_z": 0.55,                    # canopy rail / cockpit side-wall top
    "inner_half_width": 0.46,
    "panel_y": 0.95,                   # instrument panel face
    "panel_z_range": (0.18, 0.62),
    "glareshield_z": 0.66,
    "hud_combiner": (0.0, 0.98, 0.78),
    "stick": (0.0, 0.55, 0.05),        # centre stick grip (pilot right hand pre-reach)
    "throttle": (-0.36, 0.20, 0.28),   # left console throttle (pilot left hand pre-reach)
    "pedals_y": 0.80,
}
CANOPY = {
    "y_range": (-1.40, 1.60),
    "half_width": 0.48,
    "apex": (0.0, 0.15, 1.25),
    "sill_z": 0.55,
    "mdc_cord_pattern": "zigzag",      # thin detonating cord embedded in glass (canopy agent)
}
JET = {
    "nose_tip_y": 7.5,
    "tail_end_y": -9.5,
    "fuselage_center_z": -0.25,
    "fuselage_max_radius": 0.85,
    "wing_z": -0.40, "wing_half_span": 5.5, "wing_y_range": (-6.5, -1.5),
    "fin_y_range": (-9.0, -6.0), "fin_top_z": 3.2,
    "intake_y": 1.8,
}

# Pilot anatomy targets in SEAT-LOCAL coordinates (seated, torso leaning back 15° with seat back).
PILOT = {
    "pelvis": (0.0, 0.02, 0.10),
    "sacrum": (0.0, -0.04, 0.12),
    "c1_skull_base": (0.0, -0.13, 0.70),
    "head_center": (0.0, -0.08, 0.80),
    "eye_mid": (0.0, 0.02, 0.81),           # ≈ design eye point
    "shoulder_L": (-0.20, -0.08, 0.55),
    "shoulder_R": (0.20, -0.08, 0.55),
    "knee_L": (-0.11, 0.48, 0.30),
    "knee_R": (0.11, 0.48, 0.30),
}
SEAT = {
    "pan_front_y": 0.32,
    "headbox_top": (0.0, -0.30, 1.00),
    "back_y": -0.12,
    "garter_anchor_L": (-0.10, 0.30, -0.04),
    "garter_anchor_R": (0.10, 0.30, -0.04),
    "rocket_nozzles": [(-0.12, 0.05, -0.22), (0.12, 0.05, -0.22)],
    "drogue_mortar": (0.18, -0.34, 0.92),
    "pitot_booms": [(-0.24, -0.20, 0.98), (0.24, -0.20, 0.98)],
}
RISER_ATTACH_L: Vec3 = (-0.17, -0.10, 0.60)   # carrier-local shoulder harness riser points
RISER_ATTACH_R: Vec3 = (0.17, -0.10, 0.60)
MAIN_CHUTE_DIAMETER_M = 7.6
MAIN_CHUTE_LINE_LEN_M = 6.4
DROGUE_DIAMETER_M = 1.5

SUN_TOWARD: Vec3 = (-0.42, 0.80, 0.43)        # unit-ish vector pointing TOWARD the golden-hour sun
CLOUD_DECK_Z = -60.0
SUBTITLE_ZONE_NDC_Y = -0.55                   # nothing important below this NDC Y (karaoke subs)
LABEL_MIN_NDC_Y = -0.45                       # 3D labels must stay above this


# ---------------------------------------------------------------------------
# 3. EASING HELPERS
# ---------------------------------------------------------------------------
def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def smoothstep(e0: float, e1: float, x: float) -> float:
    t = clamp01((x - e0) / (e1 - e0)) if e1 != e0 else (1.0 if x >= e1 else 0.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(e0: float, e1: float, x: float) -> float:
    t = clamp01((x - e0) / (e1 - e0)) if e1 != e0 else (1.0 if x >= e1 else 0.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp3(a: Vec3, b: Vec3, t: float) -> Vec3:
    return (lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t))


def add3(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale3(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _hermite(p0: float, p1: float, v0: float, v1: float, span: float, t: float) -> float:
    t = clamp01(t)
    t2, t3 = t * t, t * t * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * p0 + h10 * span * v0 + h01 * p1 + h11 * span * v1


def window_weight(win: tuple[int, int, int, int], f: float) -> float:
    """0→1 ramp over (win[0], win[1]), hold 1, 1→0 ramp over (win[2], win[3])."""
    return smoothstep(win[0], win[1], f) * (1.0 - smoothstep(win[2], win[3], f))


def xray_fuselage_weight(f: float) -> float:
    return window_weight(XRAY_FUSELAGE, f)


def xray_body_weight(f: float) -> float:
    return window_weight(XRAY_BODY, f)


# ---------------------------------------------------------------------------
# 4. PHYSICAL READOUTS (labels / gauges / audio)
# ---------------------------------------------------------------------------
SPINE_MAX_COMPRESSION_CM = 2.5   # ≈ 1 inch (title claim)


def spine_compression_cm(f: float) -> float:
    """0.0 → 2.5 cm across Beat 4 (smootherstep), holds 2.5 afterwards."""
    return SPINE_MAX_COMPRESSION_CM * smootherstep(EV["spine_compress_start"], EV["spine_compress_end"], f)


def g_load(f: float) -> float:
    """Vertical G on the pilot: 1 G cruise → 16 G catapult peak (bullet-time hold) → rocket 10 G → chute 1 G."""
    if f < EV["catapult_ignite"]:
        return 1.0
    if f < 1180:
        return lerp(1.0, 16.0, smoothstep(EV["catapult_ignite"], 740, f))
    if f < EV["rocket_ignite"]:
        return lerp(16.0, 6.0, smoothstep(1180, EV["rocket_ignite"], f))
    if f < EV["rocket_burnout"]:
        return lerp(6.0, 10.0, smoothstep(EV["rocket_ignite"], 1280, f))
    if f < EV["main_full"]:
        return lerp(10.0, 1.0, smoothstep(EV["rocket_burnout"], EV["main_full"], f))
    return 1.0


def mission_clock_s(f: float) -> float:
    """Real-world elapsed time since handle pull (s), for the live millisecond timer label.
    Bullet-time: 0.000 s at yank_end → 0.050 s at mdc_detonate+10 → 0.20 s at rail_exit →
    0.25 s at drogue_full → 1.8 s at main_full → 2.4 s at end."""
    keys = [
        (EV["yank_end"], 0.000), (EV["mdc_detonate"], 0.040), (310, 0.050),
        (EV["catapult_ignite"], 0.120), (EV["rail_exit"], 0.200), (EV["drogue_fire"], 0.230),
        (EV["drogue_full"], 0.250), (EV["separation"], 1.100), (EV["main_full"], 1.800),
        (TOTAL_FRAMES, 2.400),
    ]
    if f <= keys[0][0]:
        return 0.0
    for (f0, v0), (f1, v1) in zip(keys, keys[1:]):
        if f <= f1:
            return lerp(v0, v1, (f - f0) / float(f1 - f0))
    return keys[-1][1]


# ---------------------------------------------------------------------------
# 5. TRAJECTORIES
# ---------------------------------------------------------------------------
_JET_ACCEL = 3.4          # m/s² relative "streak away" in B5–B6 (stylised slow-mo)
_JET_DIVE_RATIO = 0.35    # jet also sinks 0.35 m per metre forward (ends up BELOW the pilot)


def jet_offset(f: float) -> Vec3:
    """World translation of JET_ROOT. Zero until rail_exit, then the unmanned jet streaks forward/down."""
    if f <= EV["rail_exit"]:
        return (0.0, 0.0, 0.0)
    t = (f - EV["rail_exit"]) / FPS
    d = 0.5 * _JET_ACCEL * t * t
    return (0.0, d, -_JET_DIVE_RATIO * d)


def seat_rail_travel_m(f: float) -> float:
    """Distance travelled up the rails (bullet-time slow-mo)."""
    f0, f1 = EV["catapult_ignite"], EV["rail_exit"]
    if f <= f0:
        return 0.0
    u = clamp01((f - f0) / float(f1 - f0))
    return RAIL_LENGTH_M * (u ** 1.35)


_RAIL_EXIT_POS: Vec3 = scale3(RAIL_AXIS, RAIL_LENGTH_M)
_RAIL_EXIT_VEL_PER_FRAME = RAIL_LENGTH_M * 1.35 / float(EV["rail_exit"] - EV["catapult_ignite"])
_APEX_FRAME = 1530
_APEX_POS: Vec3 = (0.35, -1.60, 12.0)


def _seat_pre_separation_location(f: float) -> Vec3:
    if f <= EV["rail_exit"]:
        return scale3(RAIL_AXIS, seat_rail_travel_m(f))
    span = float(_APEX_FRAME - EV["rail_exit"])
    t = (min(f, _APEX_FRAME) - EV["rail_exit"]) / span
    v = scale3(RAIL_AXIS, _RAIL_EXIT_VEL_PER_FRAME)
    p = tuple(_hermite(_RAIL_EXIT_POS[i], _APEX_POS[i], v[i], 0.0, span, t) for i in range(3))
    if f > _APEX_FRAME:   # gentle hover / start of fall before separation
        dt = (f - _APEX_FRAME) / FPS
        p = (p[0], p[1], p[2] - 0.5 * 1.2 * dt * dt)
    return p  # type: ignore[return-value]


_SEAT_FALL_G = 2.6        # stylised slow-mo gravity for the discarded seat (m/s²)


def seat_world_location(f: float) -> Vec3:
    """World location of EjectionSeat_Root (origin == SRP)."""
    sep = EV["separation"]
    if f <= sep:
        return _seat_pre_separation_location(f)
    p = _seat_pre_separation_location(sep)
    dt = (f - sep) / FPS
    return (p[0] + 0.15 * dt, p[1] - 0.55 * dt, p[2] - 0.5 * _SEAT_FALL_G * dt * dt)


def seat_world_rotation_euler(f: float) -> Vec3:
    """Identity on the rails; gyro-damped pitch wobble under rocket; tumbles away after separation."""
    rx = ry = 0.0
    if f > EV["rail_exit"]:
        t = (f - EV["rail_exit"]) / FPS
        damp = math.exp(-1.6 * max(0.0, (f - EV["drogue_full"]) / FPS)) if f > EV["drogue_full"] else 1.0
        rx = math.radians(7.0) * math.sin(t * 2.2) * damp
        ry = math.radians(3.0) * math.sin(t * 1.7 + 0.6) * damp
    if f > EV["separation"]:
        dt = (f - EV["separation"]) / FPS
        rx += math.radians(55.0) * dt
        ry += math.radians(-20.0) * dt
    return (rx, ry, 0.0)


def pilot_carrier_location(f: float) -> Vec3:
    """World location of Pilot_Carrier (pilot armature parent; origin == SRP while seated).
    Follows the seat until separation, then the main chute pulls him up slightly and he descends slowly."""
    sep = EV["separation"]
    if f <= sep:
        return seat_world_location(f)
    p = _seat_pre_separation_location(sep)
    lift = 0.7 * smoothstep(EV["main_line_stretch"], EV["main_full"], f)
    sink = 0.35 * max(0.0, (f - EV["main_full"]) / FPS)
    drop_before_open = 0.5 * 2.0 * (min(f, EV["main_line_stretch"]) - sep) ** 2 / (FPS * FPS)
    return (p[0], p[1] - 0.10 * (f - sep) / FPS, p[2] - drop_before_open + lift - sink)


def pilot_carrier_rotation_euler(f: float) -> Vec3:
    """Seated orientation (identity) → hanging under canopy with damped pendulum sway."""
    if f <= EV["separation"]:
        return seat_world_rotation_euler(f)
    rx0, ry0, _ = seat_world_rotation_euler(EV["separation"])
    w = smoothstep(EV["separation"], EV["main_full"], f)
    t = (f - EV["main_line_stretch"]) / FPS
    sway = math.exp(-0.45 * max(0.0, t)) if t > 0 else 0.0
    rx = lerp(rx0, 0.0, w) + math.radians(9.0) * math.sin(t * 2.4) * sway * w
    ry = lerp(ry0, 0.0, w) + math.radians(5.0) * math.sin(t * 1.9 + 1.0) * sway * w
    return (rx, ry, 0.0)


def handle_local(f: float) -> Vec3:
    """Seat-local position of the yellow-black ejection handle grip centre (both hands hold it ±0.045 m in X)."""
    rest: Vec3 = (0.0, 0.36, 0.06)
    pulled: Vec3 = (0.0, 0.31, 0.24)
    return lerp3(rest, pulled, smootherstep(EV["yank_start"], EV["yank_end"], f))


def handle_visible(f: float) -> bool:
    return f < EV["separation"]


def ankle_local(side: str, f: float) -> Vec3:
    """Seat-local ankle target ('L'/'R'): on rudder pedals → reeled back against seat front by garters."""
    sx = -1.0 if side.upper().startswith("L") else 1.0
    on_pedals: Vec3 = (0.13 * sx, 0.74, -0.08)
    reeled: Vec3 = (0.12 * sx, 0.40, -0.16)
    return lerp3(on_pedals, reeled, smootherstep(EV["garter_start"], EV["garter_end"], f))


def drogue_open_fraction(f: float) -> float:
    return smootherstep(EV["drogue_fire"], EV["drogue_full"], f)


def drogue_world_location(f: float) -> Vec3:
    """Drogue canopy centre: stowed in mortar → trails above/behind the seat headbox."""
    seat = seat_world_location(f)
    mortar = add3(seat, SEAT["drogue_mortar"])
    full_off: Vec3 = (0.10, -1.6, 4.2)
    return add3(mortar, scale3(full_off, drogue_open_fraction(f)))


def main_chute_open_fraction(f: float) -> float:
    """0 = packed; 0..0.25 line stretch; 0.25..1.0 canopy blossom (inflation)."""
    a = 0.25 * smoothstep(EV["separation"] + 5, EV["main_line_stretch"], f)
    b = 0.75 * smootherstep(EV["main_line_stretch"], EV["main_full"], f)
    return a + b


def main_chute_world_location(f: float) -> Vec3:
    """Centre of the main canopy skirt ring; directly above the pilot once inflated."""
    pc = pilot_carrier_location(f)
    u = main_chute_open_fraction(f)
    stretch = smoothstep(0.0, 0.25, u)
    off_packed: Vec3 = (0.0, -0.35, 0.95)
    off_full: Vec3 = (0.0, 0.0, 0.60 + MAIN_CHUTE_LINE_LEN_M)
    return add3(pc, lerp3(off_packed, off_full, stretch))


# ---------------------------------------------------------------------------
# 6. LABEL TEXTS (3D Typography agent uses these exact strings; DejaVuSans-Bold glyph-verified)
# ---------------------------------------------------------------------------
LABELS = [
    # key,            text,                    visible window (frames), anchor (world-ish, see labels_3d.py)
    ("handle",        "EJECTION HANDLE",       (120, 280)),
    ("mdc",           "CANOPY BREAKER CORD",   (300, 560)),
    ("clock_mdc",     "0.05 s",                (300, 560)),
    ("garters",       "LEG GARTERS",           (600, 760)),
    ("catapult",      "CATAPULT CHARGE",       (690, 880)),
    ("g_load",        "16 G",                  (720, 1230)),
    ("spine_gauge",   "SPINE: −0.0 cm",        (900, 1230)),   # live: −0.0 → −2.5 cm (≈ 1 INCH)
    ("rocket",        "ROCKET MOTOR",          (1250, 1420)),
    ("drogue",        "DROGUE CHUTE",          (1400, 1525)),
    ("clock_drogue",  "0.25 s",                (1400, 1525)),
]
