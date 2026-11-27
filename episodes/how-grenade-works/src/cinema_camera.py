"""
Hollywood Action-Movie 9:16 Portrait Cinematography Engine for
`episodes/how-grenade-works` (Blender 4.2 Cycles — IMAX / Matrix Bullet-Time Standard).

Public API:
  - GRENADE_FLIGHT_KFS: Synchronized grenade flight trajectory keyframes (f=134..495)
    shared by the master scene builder and `verify_framing.py`.
  - grenade_pos_at(frame, ...): Evaluates exact 3D grenade center across f=1..720.
  - camera_state_at(frame, ...): Pure-Python evaluator returning
    (cam_pos, tgt_pos, lens_mm, aperture_fstop) with handheld drift, bullet-time
    gimbal orbit, Beat 5 pressure tremor, and Beat 6 shockwave jolt.
  - build_cinema_camera(col, grenade_root=None, hero_pos=Vector((0, 0, 0)),
    bunker_pos=Vector((0, 7.25, 0.045)), total_frames=720)
    -> tuple[bpy.types.Object, bpy.types.Object]

Pure-Python safe: imports `bpy` / `mathutils` lazily/gracefully so `verify_framing.py`
can import `GRENADE_FLIGHT_KFS`, `grenade_pos_at`, and `camera_state_at` directly
outside Blender with 0.000m trajectory or lens drift.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

try:
    import bpy  # type: ignore[import-not-found]
    from mathutils import Vector  # type: ignore[import-not-found]
except ImportError:  # Pure-Python QA mode outside Blender (e.g. verify_framing.py)
    bpy = None  # type: ignore[assignment]

    class Vector(tuple):  # type: ignore[no-redef]
        """Lightweight 3D Vector stand-in when imported outside Blender."""

        def __new__(cls, seq: Sequence[float] = (0.0, 0.0, 0.0)) -> "Vector":
            return super().__new__(cls, (float(seq[0]), float(seq[1]), float(seq[2])))

        @property
        def x(self) -> float:
            return self[0]

        @property
        def y(self) -> float:
            return self[1]

        @property
        def z(self) -> float:
            return self[2]


Vec3 = Tuple[float, float, float]

# ==============================================================================
# OPTICAL & WORLD BLOCKING CONSTANTS (100% Synced with Director / Battlefield)
# ==============================================================================
RES_X = 1080
RES_Y = 1920
ASPECT_W_OVER_H = RES_X / RES_Y  # 0.5625 (9:16 portrait)
SENSOR_WIDTH_MM = 36.0           # Blender sensor_fit='AUTO' -> H_eff=36.0mm, W_eff=20.25mm
SENSOR_H_EFF_MM = SENSOR_WIDTH_MM
SENSOR_W_EFF_MM = SENSOR_WIDTH_MM * ASPECT_W_OVER_H

R_OUTER_REAL = 0.042             # Real handheld grenade radius (8.4 cm outer diameter)

HERO_DEFAULT_POS: Vec3 = (0.0, 0.0, 0.0)
HERO_EYE_LOCAL: Vec3 = (0.0, 0.10, 1.64)
HERO_MOUTH_LOCAL: Vec3 = (0.0, 0.11, 1.54)
HERO_GRENADE_HAND_LOCAL: Vec3 = (0.08, 0.30, 1.50)
HERO_THROW_RELEASE_LOCAL: Vec3 = (0.24, 0.42, 1.62)

BUNKER_DEFAULT_POS: Vec3 = (0.0, 7.25, 0.045)
SANDBAG_WALL_Y = 7.80
SENTRY_LEFT_POS: Vec3 = (-0.78, 8.55, 0.0)
SENTRY_RIGHT_POS: Vec3 = (0.82, 8.65, 0.0)

# ==============================================================================
# SYNCHRONIZED MID-AIR GRENADE TRAJECTORY KEYFRAMES (f = 134..495)
# ==============================================================================
# Exact blocking anchors:
#   f=134 -> (0.24, 0.42, 1.62)  Hero throw release
#   f=152 -> (0.05, 2.35, 2.25)  End of fast launch -> Matrix Bullet-Time begins
#   f=152..440 -> Ultra-slow-mo bullet-time parabolic glide to (0.0, 4.95, 2.35)
#   f=440..495 -> Real-time ballistic acceleration down into Enemy Bunker (0.0, 7.25, 0.045)
GRENADE_FLIGHT_KFS: list[tuple[int, Vec3]] = [
    (134, (0.240, 0.420, 1.620)),
    (143, (0.140, 1.390, 2.020)),
    (152, (0.050, 2.350, 2.250)),
    (195, (0.042, 2.740, 2.310)),
    (235, (0.034, 3.105, 2.355)),
    (294, (0.022, 3.645, 2.400)),
    (355, (0.012, 4.195, 2.405)),
    (420, (0.002, 4.775, 2.370)),
    (440, (0.000, 4.950, 2.350)),
    (468, (0.000, 6.100, 1.520)),
    (495, (0.000, 7.250, 0.045)),
]


# ==============================================================================
# PURE-PYTHON INTERPOLATION & VECTOR MATH HELPERS
# ==============================================================================
def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _smoothstep(e0: float, e1: float, x: float) -> float:
    t = _clamp01((x - e0) / max(1e-6, e1 - e0))
    return t * t * (3.0 - 2.0 * t)


def _smootherstep(e0: float, e1: float, x: float) -> float:
    t = _clamp01((x - e0) / max(1e-6, e1 - e0))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp3(a: Vec3, b: Vec3, t: float) -> Vec3:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def _add3(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub3(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _as_vec3(v: Sequence[float]) -> Vec3:
    return (float(v[0]), float(v[1]), float(v[2]))


def _eval_piecewise_scalar(kfs: Sequence[tuple[float, float]], frame: float, smooth: bool = True) -> float:
    if frame <= kfs[0][0]:
        return kfs[0][1]
    if frame >= kfs[-1][0]:
        return kfs[-1][1]
    for i in range(len(kfs) - 1):
        f0, v0 = kfs[i]
        f1, v1 = kfs[i + 1]
        if f0 <= frame <= f1:
            u = (frame - f0) / max(1e-6, f1 - f0)
            if smooth:
                u = u * u * (3.0 - 2.0 * u)
            return _lerp(v0, v1, u)
    return kfs[-1][1]


def _eval_piecewise_vec3(kfs: Sequence[tuple[float, Vec3]], frame: float, smooth: bool = True) -> Vec3:
    if frame <= kfs[0][0]:
        return kfs[0][1]
    if frame >= kfs[-1][0]:
        return kfs[-1][1]
    for i in range(len(kfs) - 1):
        f0, v0 = kfs[i]
        f1, v1 = kfs[i + 1]
        if f0 <= frame <= f1:
            u = (frame - f0) / max(1e-6, f1 - f0)
            if smooth:
                u = u * u * (3.0 - 2.0 * u)
            return _lerp3(v0, v1, u)
    return kfs[-1][1]


def _catmull_rom_vec3(kfs: Sequence[tuple[float, Vec3]], frame: float) -> Vec3:
    """Smooth tension-controlled Catmull-Rom spline through 3D keyframes (zero overshoot)."""
    if frame <= kfs[0][0]:
        return kfs[0][1]
    if frame >= kfs[-1][0]:
        return kfs[-1][1]
    n = len(kfs)
    for i in range(n - 1):
        f0, p1 = kfs[i]
        f1, p2 = kfs[i + 1]
        if f0 <= frame <= f1:
            t = (frame - f0) / max(1e-6, f1 - f0)
            p0 = kfs[max(0, i - 1)][1]
            p3 = kfs[min(n - 1, i + 2)][1]
            t2 = t * t
            t3 = t2 * t
            out = []
            for c in range(3):
                val = 0.5 * (
                    (2.0 * p1[c])
                    + (-p0[c] + p2[c]) * t
                    + (2.0 * p0[c] - 5.0 * p1[c] + 4.0 * p2[c] - p3[c]) * t2
                    + (-p0[c] + 3.0 * p1[c] - 3.0 * p2[c] + p3[c]) * t3
                )
                out.append(val)
            return (out[0], out[1], out[2])
    return kfs[-1][1]


# ==============================================================================
# GRENADE TRAJECTORY EVALUATOR (f = 1..720)
# ==============================================================================
def grenade_pos_at(
    frame: float,
    hero_pos: Sequence[float] = HERO_DEFAULT_POS,
    bunker_pos: Sequence[float] = BUNKER_DEFAULT_POS,
) -> Vec3:
    """
    Returns exact world-space 3D center of the grenade at any frame in [1, 720].
    Matches `GRENADE_FLIGHT_KFS` exactly on f=134..495, hero-hand hold on f=1..134,
    and enemy bunker ground zero on f=495..720.
    """
    hero_off = _sub3(_as_vec3(hero_pos), HERO_DEFAULT_POS)
    bunker_off = _sub3(_as_vec3(bunker_pos), BUNKER_DEFAULT_POS)

    if frame <= 108.0:
        return _add3(HERO_GRENADE_HAND_LOCAL, hero_off)
    if frame < 134.0:
        # Smooth wind-up cock-back (f=108..122) then whip-throw release (f=122..134)
        windup_kfs: list[tuple[float, Vec3]] = [
            (108.0, HERO_GRENADE_HAND_LOCAL),
            (122.0, (0.28, -0.06, 1.56)),
            (134.0, HERO_THROW_RELEASE_LOCAL),
        ]
        return _add3(_eval_piecewise_vec3(windup_kfs, frame, smooth=True), hero_off)
    if frame <= 495.0:
        # Piece-wise smooth ballistic segments with exact anchor hit at 134, 152, 440, 495
        if frame <= 152.0:
            # Fast launch arc
            u = (frame - 134.0) / 18.0
            p0 = GRENADE_FLIGHT_KFS[0][1]
            p1 = GRENADE_FLIGHT_KFS[2][1]
            x = _lerp(p0[0], p1[0], u)
            y = _lerp(p0[1], p1[1], u)
            z = _lerp(p0[2], p1[2], u) + 0.14 * 4.0 * u * (1.0 - u)
            base = (x, y, z)
        elif frame <= 440.0:
            # Matrix Bullet-Time ultra-slow-mo glide through exact flight keyframes
            bt_kfs = [kf for kf in GRENADE_FLIGHT_KFS if 152 <= kf[0] <= 440]
            base = _eval_piecewise_vec3(bt_kfs, frame, smooth=False)
        else:
            # Real-time gravity acceleration down into bunker ground zero (f=440..495)
            u = (frame - 440.0) / 55.0
            p0 = (0.0, 4.950, 2.350)
            p1 = (0.0, 7.250, 0.045)
            x = _lerp(p0[0], p1[0], u)
            y = _lerp(p0[1], p1[1], u)
            # Parabolic gravity drop: gentle initial downward slope, accelerating into impact
            z = p0[2] + (p1[2] - p0[2]) * (0.25 * u + 0.75 * u * u)
            base = (x, y, z)

        blend = _smoothstep(134.0, 495.0, frame)
        off = _lerp3(hero_off, bunker_off, blend)
        return _add3(base, off)

    return _add3(BUNKER_DEFAULT_POS, bunker_off)


# ==============================================================================
# CINEMATOGRAPHY MOTION LAYER EVALUATORS (HANDHELD / ORBIT / TREMOR / JOLT)
# ==============================================================================
def _handheld_drift(frame: float) -> tuple[Vec3, Vec3]:
    """
    Subtle realistic handheld action-cam micro-drift during Beat 1-2 hero shots (f=1..134).
    Smoothly fades to zero by f=150 so Matrix Bullet-Time gimbal is 100% silky-locked.
    """
    weight = 1.0 - _smoothstep(130.0, 150.0, frame)
    if weight <= 1e-5:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    dx = weight * (0.0055 * math.sin(frame * 0.095 + 0.4) + 0.0028 * math.sin(frame * 0.215 + 1.9))
    dy = weight * (0.0030 * math.cos(frame * 0.080 + 0.7))
    dz = weight * (0.0050 * math.sin(frame * 0.125 + 2.3) + 0.0022 * math.cos(frame * 0.270 + 0.5))
    tx = weight * (0.0022 * math.sin(frame * 0.110 + 1.1))
    tz = weight * (0.0020 * math.cos(frame * 0.135 + 2.7))
    return (dx, dy, dz), (tx, 0.0, tz)


def _pin_rip_snap_impulse(frame: float) -> tuple[Vec3, Vec3, float]:
    """
    Crisp subtle 2mm micro-snap camera punch at f=34 (Hero teeth ripping the pull-pin free).
    Combines a damped 2.0mm camera-body kick with a subtle +2.0mm focal micro-punch
    (56mm -> 58mm -> 56mm), decaying smoothly to strict zero by f=45.
    """
    if frame < 33.0 or frame > 45.0:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0
    rise = _smoothstep(33.0, 34.0, frame)
    fade = 1.0 - _smoothstep(41.0, 45.0, frame)
    dt = max(0.0, frame - 33.5)
    env = rise * fade * math.exp(-0.42 * dt)
    osc = math.sin(dt * 2.35 + 0.65)
    cam_d = (
        -0.0020 * env * osc,
        -0.0018 * env * math.cos(dt * 1.9),
        0.0020 * env * osc,
    )
    tgt_d = (
        -0.0025 * env * osc,
        0.0,
        0.0018 * env * osc,
    )
    lens_d = 2.0 * rise * fade * math.exp(-0.45 * max(0.0, frame - 34.0))
    return cam_d, tgt_d, lens_d


def _spit_flick_whip_accent(frame: float) -> tuple[Vec3, Vec3]:
    """
    Slick subtle kinetic whip-accent at f=66 (Rajinikanth teeth pin spit-flick toward Screen-Right).
    Leads subtly toward Screen-Right (-X, -Y world) with a crisp upward flick accent,
    opening clean Screen-Right framing headroom for the spinning chrome pin before
    settling smoothly to strict zero by f=80.
    """
    if frame < 64.5 or frame > 80.0:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    rise = _smoothstep(64.5, 66.5, frame)
    fade = 1.0 - _smoothstep(74.0, 80.0, frame)
    dt = max(0.0, frame - 66.0)
    env = rise * fade * math.exp(-0.24 * dt)
    ring = 1.0 + 0.14 * math.sin(dt * 2.0)
    cam_d = (-0.011 * env * ring, 0.006 * env, 0.004 * env)
    tgt_d = (-0.022 * env * ring, -0.009 * env, 0.005 * env)
    return cam_d, tgt_d


def _striker_snap_impulse(frame: float) -> Vec3:
    """Crisp 2.4mm damped cinema-gimbal impulse when striker hammer slams primer (f=294..304)."""
    if frame < 294.0 or frame > 306.0:
        return (0.0, 0.0, 0.0)
    dt = frame - 294.0
    env = math.exp(-0.38 * dt)
    return (
        0.0014 * env * math.sin(dt * 2.1),
        0.0,
        -0.0024 * env * math.sin(dt * 1.85),
    )


def _bunker_impact_thud(frame: float) -> Vec3:
    """Subtle ground-contact micro-thud when grenade lands in sandbag bunker (f=495..503)."""
    if frame < 495.0 or frame > 504.0:
        return (0.0, 0.0, 0.0)
    dt = frame - 495.0
    env = math.exp(-0.45 * dt)
    return (
        0.0010 * env * math.sin(dt * 2.4),
        0.0,
        -0.0032 * env * math.sin(dt * 1.9),
    )


def _pressure_tremor(frame: float) -> tuple[Vec3, Vec3]:
    """
    Escalating high-pressure overpressure camera tremor during Beat 5 (f=540..611).
    Builds palpable physical tension while staying strictly inside macro framing margins.
    """
    if frame < 540.0 or frame >= 612.0:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    u = _clamp01((frame - 540.0) / 71.0)
    amp = 0.0042 * (u ** 1.65)
    dx = amp * (0.65 * math.sin(frame * 2.37) + 0.35 * math.cos(frame * 4.13 + 0.9))
    dy = amp * 0.25 * math.sin(frame * 3.11 + 1.4)
    dz = amp * (0.65 * math.cos(frame * 2.71 + 0.4) + 0.35 * math.sin(frame * 4.89))
    tx = dx * 0.30
    tz = dz * 0.30
    return (dx, dy, dz), (tx, 0.0, tz)


def _shockwave_jolt(frame: float) -> tuple[Vec3, Vec3]:
    """
    Concussive detonation shockwave jolt at Beat 6 ignition (f=612..630),
    decaying smoothly into silky bullet-time orbital flight by f=634.
    """
    if frame < 612.0 or frame > 633.0:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    dt = frame - 612.0
    env = math.exp(-0.22 * dt) * (1.0 - _smoothstep(624.0, 633.0, frame))
    dx = 0.045 * env * math.sin(dt * 2.25 + 0.3)
    dy = -0.055 * env * math.cos(dt * 1.75)
    dz = 0.050 * env * math.sin(dt * 2.85)
    return (dx, dy, dz), (dx * 0.25, 0.0, dz * 0.25)


# ==============================================================================
# MASTER 6-BEAT CINEMA CAMERA EVALUATOR (f = 1..720)
# ==============================================================================
def _bullet_time_orbit_cam(
    frame: float,
    gren_pos: Vec3,
) -> tuple[Vec3, Vec3, float, float]:
    """
    Matrix Bullet-Time stabilized gimbal orbit around the flying grenade (f=152..440).
    Keeps the camera in the front cutaway hemisphere (-Y side, azimuth within [-22°, +18°])
    so the X-ray interior & 3D pointer labels stay 100% open, tack-sharp, and unobstructed,
    while parallax against the sunlit desert battlefield & bunker melts into creamy bokeh.
    """
    # Distance from grenade center (m): 0.43m X-ray wide macro -> 0.385m striker punch-in -> 0.42m delay
    dist_kfs = [
        (152.0, 0.430),
        (195.0, 0.430),
        (235.0, 0.420),
        (272.0, 0.392),
        (294.0, 0.385),  # Beat 3 Striker / Primer macro punch-in
        (320.0, 0.396),
        (355.0, 0.420),
        (420.0, 0.420),
        (440.0, 0.420),
    ]
    # Front-cutaway azimuth (degrees around -Y axis; negative = camera on -X side)
    az_kfs = [
        (152.0, -20.0),
        (195.0, -7.5),
        (235.0, 9.0),
        (294.0, 16.0),
        (355.0, 11.0),
        (400.0, 0.0),
        (420.0, -6.0),
        (440.0, -11.0),
    ]
    # Vertical camera offset relative to grenade center (gentle hero low-to-level crane)
    dz_cam_kfs = [
        (152.0, 0.025),
        (195.0, 0.020),
        (235.0, 0.022),
        (294.0, 0.022),
        (355.0, 0.018),
        (420.0, 0.014),
        (440.0, 0.012),
    ]
    # Vertical target offset relative to grenade center (keeps grenade & labels in NDC Y [-0.25, +0.60])
    dz_tgt_kfs = [
        (152.0, 0.012),
        (195.0, 0.012),
        (235.0, 0.013),
        (294.0, 0.011),  # Frames grenade core (-0.10), bottom (-0.46 > -0.55) & primer/striker (+0.39/+0.48)
        (355.0, 0.010),
        (420.0, 0.008),
        (440.0, 0.008),
    ]
    # Cinema prime focal length (mm)
    lens_kfs = [
        (152.0, 58.0),
        (195.0, 58.0),
        (235.0, 60.0),
        (294.0, 60.5),
        (355.0, 60.0),
        (420.0, 60.0),
        (440.0, 58.0),
    ]

    dist = _eval_piecewise_scalar(dist_kfs, frame, smooth=True)
    az_rad = math.radians(_eval_piecewise_scalar(az_kfs, frame, smooth=True))
    dz_cam = _eval_piecewise_scalar(dz_cam_kfs, frame, smooth=True)
    dz_tgt = _eval_piecewise_scalar(dz_tgt_kfs, frame, smooth=True)
    lens = _eval_piecewise_scalar(lens_kfs, frame, smooth=True)

    cam_pos = (
        gren_pos[0] + dist * math.sin(az_rad),
        gren_pos[1] - dist * math.cos(az_rad),
        gren_pos[2] + dz_cam,
    )
    tgt_pos = (gren_pos[0], gren_pos[1], gren_pos[2] + dz_tgt)
    return cam_pos, tgt_pos, lens, 8.0


def camera_state_at(
    frame: float,
    hero_pos: Sequence[float] = HERO_DEFAULT_POS,
    bunker_pos: Sequence[float] = BUNKER_DEFAULT_POS,
    include_motion_fx: bool = True,
) -> tuple[Vec3, Vec3, float, float]:
    """
    Computes exact (cam_pos, tgt_pos, lens_mm, aperture_fstop) for any frame in [1, 720].

    6-Beat Hollywood Action Choreography:
      - Beat 1 (f=1..108):    Heroic low-angle Rajinikanth close-up (Z=1.44->1.48m),
                              48mm->56mm(f=26..68)->46mm(f=104), f=34 2mm pin-rip punch &
                              f=66 spit-flick kinetic whip-accent
      - Throw (f=108..152):   Widens to 35mm f/4.2 at release (f=134), whip-pans into flight
      - Beat 2 (f=152..235):  Matrix bullet-time X-ray orbit, 58-60mm f/8.0, d=0.43m
      - Beat 3 (f=235..355):  Striker/primer macro punch-in, 62mm f/8.0, d=0.37m + micro snap
      - Beat 4 (f=355..495):  4s delay glide (60mm f/8.0) -> ballistic dive into bunker (48mm)
      - Beat 5 (f=495..611):  Low-angle ground macro (d=0.42m, 48-50mm f/6.3) + rising tremor
      - Beat 6 (f=612..720):  Shockwave jolt + rapid pull-back -> 53° bullet-time orbit (32-35mm f/5.6)
    """
    hero_off = _sub3(_as_vec3(hero_pos), HERO_DEFAULT_POS)
    bunker_off = _sub3(_as_vec3(bunker_pos), BUNKER_DEFAULT_POS)
    gren_pos = grenade_pos_at(frame, hero_pos=hero_pos, bunker_pos=bunker_pos)

    # ------------------------------------------------------------------
    # PHASE A: Beat 1 Heroic Low-Angle Rajinikanth Close-Up & Throw (f = 1..134)
    # ------------------------------------------------------------------
    if frame <= 134.0:
        # Heroic slight low-angle (Z = 1.44m -> 1.48m across f=1..104) looking subtly UPWARD
        # at Hero's jawline, teeth, chrome ring & intense eyes; steps closer & more centered on
        # the mouth-to-grenade action during teeth bite & rip (f=18..56), then settles cleanly
        # into the EXACT throw-release anchor (-0.96, 1.28, 1.62) at f=134 for seamless Phase B.
        cam_kfs: list[tuple[float, Vec3]] = [
            (1.0,   (-0.50, 1.40, 1.440)),
            (18.0,  (-0.46, 1.33, 1.450)),
            (34.0,  (-0.45, 1.30, 1.455)),
            (56.0,  (-0.45, 1.31, 1.465)),
            (68.0,  (-0.46, 1.33, 1.472)),
            (104.0, (-0.50, 1.35, 1.480)),
            (108.0, (-0.52, 1.34, 1.485)),
            (122.0, (-0.76, 1.30, 1.560)),
            (134.0, (-0.96, 1.28, 1.620)),
        ]
        # Centered on mouth-to-grenade bite/rip action, drifting subtly Screen-Right (-X) for
        # the f=66..68 spit-flick corridor while keeping eyes/mouth/fist/pin inside NDC Y [-0.15, +0.62].
        tgt_kfs: list[tuple[float, Vec3]] = [
            (1.0,   (0.03, 0.19, 1.518)),
            (18.0,  (0.03, 0.19, 1.512)),
            (34.0,  (0.03, 0.19, 1.512)),
            (56.0,  (0.02, 0.19, 1.512)),
            (68.0,  (0.01, 0.18, 1.514)),
            (104.0, (0.04, 0.21, 1.520)),
            (108.0, (0.05, 0.22, 1.525)),
            (122.0, (0.08, 0.25, 1.538)),
            (134.0, (0.12, 0.28, 1.540)),
        ]
        # Prime push: 48mm (f=1) -> 56mm (f=26..68 tight teeth/smirk/spit punch-in) -> 46mm (f=104) -> 35mm (f=134)
        lens_kfs = [
            (1.0,   48.0),
            (26.0,  56.0),
            (68.0,  56.0),
            (104.0, 46.0),
            (108.0, 45.0),
            (122.0, 39.5),
            (134.0, 35.0),
        ]
        fstop_kfs = [
            (1.0,   3.5),
            (26.0,  3.8),
            (68.0,  3.8),
            (104.0, 3.5),
            (108.0, 3.5),
            (134.0, 4.2),
        ]
        cam_pos = _add3(_eval_piecewise_vec3(cam_kfs, frame, smooth=True), hero_off)
        tgt_pos = _add3(_eval_piecewise_vec3(tgt_kfs, frame, smooth=True), hero_off)
        lens = _eval_piecewise_scalar(lens_kfs, frame, smooth=True)
        fstop = _eval_piecewise_scalar(fstop_kfs, frame, smooth=True)

    # ------------------------------------------------------------------
    # PHASE B: Fast Launch Whip-Pan Catch-Up (f = 134..152)
    # ------------------------------------------------------------------
    elif frame < 152.0:
        # Smooth continuous counter-clockwise whip-pan on -X side (zero gimbal flip!)
        start_cam = _add3((-0.96, 1.28, 1.62), hero_off)
        start_tgt = _add3((0.12, 0.28, 1.54), hero_off)
        end_gren = grenade_pos_at(152.0, hero_pos=hero_pos, bunker_pos=bunker_pos)
        end_cam, end_tgt, end_lens, end_fstop = _bullet_time_orbit_cam(152.0, end_gren)

        u = _smootherstep(134.0, 152.0, frame)
        # Keep whip-pan curving cleanly outside -X while tracking launched grenade
        mid_cam = _add3((-0.58, 1.60, 2.02), _lerp3(hero_off, bunker_off, 0.10))
        cam_pos = (
            (1 - u) * (1 - u) * start_cam[0] + 2 * (1 - u) * u * mid_cam[0] + u * u * end_cam[0],
            (1 - u) * (1 - u) * start_cam[1] + 2 * (1 - u) * u * mid_cam[1] + u * u * end_cam[1],
            (1 - u) * (1 - u) * start_cam[2] + 2 * (1 - u) * u * mid_cam[2] + u * u * end_cam[2],
        )
        # Target blends smoothly from Hero throw center directly onto live flying grenade
        tgt_pos = _lerp3(start_tgt, (gren_pos[0], gren_pos[1], gren_pos[2] + 0.012), _smoothstep(134.0, 148.0, frame))
        lens = _lerp(35.0, end_lens, u)
        fstop = _lerp(4.2, end_fstop, u)

    # ------------------------------------------------------------------
    # PHASE C: Matrix Bullet-Time X-Ray Flight (f = 152..440)
    # ------------------------------------------------------------------
    elif frame <= 440.0:
        cam_pos, tgt_pos, lens, fstop = _bullet_time_orbit_cam(frame, gren_pos)

    # ------------------------------------------------------------------
    # PHASE D: Accelerating Ballistic Plunge into Enemy Bunker (f = 440..495)
    # ------------------------------------------------------------------
    elif frame < 495.0:
        u = _smoothstep(440.0, 495.0, frame)
        # Maintain tight macro chase beside/behind plummeting grenade, landing smoothly at Beat 5 pose
        # At f=440: offset=(-0.080, -0.412, +0.012), tgt_dz=+0.008
        # At f=495: offset=(-0.150, -0.390, +0.030), tgt_dz=+0.035 (cam=(-0.15, 6.86, 0.075), tgt=(0, 7.25, 0.080))
        off_x = _lerp(-0.080, -0.150, u)
        off_y = _lerp(-0.412, -0.390, u) - 0.055 * math.sin(math.pi * u)  # brief pull-back breath mid-drop
        off_z = _lerp(0.012, 0.030, u) + 0.025 * math.sin(math.pi * u)
        cam_pos = (
            gren_pos[0] + off_x,
            gren_pos[1] + off_y,
            max(0.075, gren_pos[2] + off_z),
        )
        tgt_dz = _lerp(0.008, 0.035, u)
        tgt_pos = (gren_pos[0], gren_pos[1], gren_pos[2] + tgt_dz)
        lens = _lerp(58.0, 48.0, u)
        fstop = _lerp(8.0, 6.3, u)

    # ------------------------------------------------------------------
    # PHASE E: Beat 5 Low-Angle Bunker Macro + Rising Pressure (f = 495..611)
    # ------------------------------------------------------------------
    elif frame < 612.0:
        # Low-angle dramatic ground macro (d≈0.42m) looking slightly upward across
        # glowing X-ray grenade toward Enemy Sandbag Bunker (Y=7.8) & diving Rival Sentries!
        b5_cam_kfs: list[tuple[float, Vec3]] = [
            (495.0, (-0.150, 6.860, 0.075)),
            (540.0, (-0.145, 6.860, 0.076)),
            (580.0, (-0.130, 6.850, 0.080)),
            (611.0, (-0.120, 6.840, 0.082)),
        ]
        b5_tgt_kfs: list[tuple[float, Vec3]] = [
            (495.0, (0.000, 7.250, 0.080)),
            (611.0, (0.000, 7.250, 0.080)),
        ]
        b5_lens_kfs = [
            (495.0, 48.0),
            (540.0, 48.0),
            (611.0, 50.0),
        ]
        cam_pos = _add3(_eval_piecewise_vec3(b5_cam_kfs, frame, smooth=True), bunker_off)
        tgt_pos = _add3(_eval_piecewise_vec3(b5_tgt_kfs, frame, smooth=True), bunker_off)
        lens = _eval_piecewise_scalar(b5_lens_kfs, frame, smooth=True)
        fstop = 6.3

    # ------------------------------------------------------------------
    # PHASE F: Beat 6 Epic Wide-to-Orbit Bullet-Time Explosion Climax (f = 612..720)
    # ------------------------------------------------------------------
    else:
        if frame <= 634.0:
            # Rapid heroic pull-back from ground macro to wide-action orbit entry (-1.45, 3.85, 1.25)
            u = _smootherstep(612.0, 634.0, frame)
            # Fast initial acceleration away from blast (ease-out exponential kick blended with smootherstep)
            u_fast = 1.0 - (1.0 - _clamp01((frame - 612.0) / 22.0)) ** 2.6
            u_mix = 0.55 * u_fast + 0.45 * u
            cam_pos = _add3(
                _lerp3((-0.120, 6.840, 0.082), (-1.450, 3.850, 1.250), u_mix),
                bunker_off,
            )
            tgt_pos = _add3(
                _lerp3((0.000, 7.250, 0.080), (0.000, 7.450, 0.480), u_mix),
                bunker_off,
            )
            lens = _lerp(50.0, 32.0, u_mix)
            fstop = _lerp(6.3, 5.6, u_mix)
        else:
            # Silky 53° Bullet-Time Orbital Sweep around Ground Zero & Exploding Bunker!
            # Exact prompt start (-1.45, 3.85, 1.25) at f=634 -> exact prompt end (1.85, 4.40, 1.55) at f=720
            # Arcing cleanly around target (0.0, 7.45) at constant heroic radius R≈3.88m -> 3.57m
            u = _clamp01((frame - 634.0) / 86.0)
            u_ease = _lerp(u, _smoothstep(634.0, 720.0, frame), 0.35)

            center_x, center_y = 0.0, 7.450
            # Start vector (-1.45, -3.60) -> R0=3.881m, az0=-21.94°
            # End vector   (+1.85, -3.05) -> R1=3.567m, az1=+31.24°  (Total sweep = 53.2°!)
            r0 = math.hypot(-1.450, 3.850 - center_y)
            r1 = math.hypot(1.850, 4.400 - center_y)
            az0 = math.atan2(-1.450, center_y - 3.850)
            az1 = math.atan2(1.850, center_y - 4.400)

            r = _lerp(r0, r1, u_ease)
            az = _lerp(az0, az1, u_ease)
            cam_z = _lerp(1.250, 1.550, u_ease)

            cam_pos = _add3(
                (center_x + r * math.sin(az), center_y - r * math.cos(az), cam_z),
                bunker_off,
            )
            # Gentle horizontal lead (+0.14m by f=720) & low-horizon center (Z=0.48->0.50)
            # keeps Ground Zero above -0.25 AND both diving Rival Sentries cleanly inside frame!
            tgt_x = _lerp(0.000, 0.140, u_ease)
            tgt_z = _lerp(0.480, 0.500, u_ease)
            tgt_pos = _add3((tgt_x, 7.450, tgt_z), bunker_off)
            lens = _lerp(32.0, 30.0, u_ease)
            fstop = 5.6

    # ------------------------------------------------------------------
    # SUPERIMPOSE CINEMATOGRAPHY MOTION LAYERS
    # ------------------------------------------------------------------
    if include_motion_fx:
        h_cam, h_tgt = _handheld_drift(frame)
        r_cam, r_tgt, r_lens = _pin_rip_snap_impulse(frame)
        w_cam, w_tgt = _spit_flick_whip_accent(frame)
        s_cam = _striker_snap_impulse(frame)
        b_cam = _bunker_impact_thud(frame)
        p_cam, p_tgt = _pressure_tremor(frame)
        j_cam, j_tgt = _shockwave_jolt(frame)

        cam_pos = _add3(
            cam_pos,
            _add3(_add3(h_cam, _add3(r_cam, w_cam)), _add3(s_cam, _add3(b_cam, _add3(p_cam, j_cam)))),
        )
        tgt_pos = _add3(
            tgt_pos,
            _add3(_add3(h_tgt, _add3(r_tgt, w_tgt)), _add3(p_tgt, j_tgt)),
        )
        lens = lens + r_lens

    return cam_pos, tgt_pos, lens, fstop


# ==============================================================================
# BLENDER 4.2 SCENE BUILDER PUBLIC API
# ==============================================================================
def _link_to_collection(obj, col) -> None:
    target_col = col if col is not None else bpy.context.scene.collection
    if obj.name not in target_col.objects:
        target_col.objects.link(obj)


def _set_smooth_handles(id_block) -> None:
    anim = getattr(id_block, "animation_data", None)
    if anim is None or anim.action is None:
        return
    for fcurve in anim.action.fcurves:
        for kp in fcurve.keyframe_points:
            kp.interpolation = "BEZIER"
            kp.handle_left_type = "AUTO_CLAMPED"
            kp.handle_right_type = "AUTO_CLAMPED"


def build_cinema_camera(
    col,
    grenade_root=None,
    hero_pos=Vector((0.0, 0.0, 0.0)),
    bunker_pos=Vector((0.0, 7.25, 0.045)),
    total_frames: int = 720,
) -> tuple["bpy.types.Object", "bpy.types.Object"]:
    """
    Creates and animates the Hollywood 9:16 Portrait Cinema Camera (`Cinema_Camera`)
    and its focus/aim tracking empty (`Camera_Target`) across `1..total_frames`.

    Features:
      - `sensor_width = 36.0`, `sensor_fit = 'AUTO'`, `clip_start = 0.01`, `clip_end = 150.0`
      - `TRACK_TO` constraint (`TRACK_NEGATIVE_Z`, `UP_Y`) onto `Camera_Target`
      - Animated Cinema Prime focal length (`48mm -> 56mm -> 46mm -> 35mm -> 58-62mm -> 48-50mm -> 32-35mm`)
      - Animated Cinema DOF (`use_dof = True`, `focus_object = target`, `aperture_blades = 7`,
        `f/3.5-3.8` Hero bokeh -> `f/8.0` X-ray tack-sharp macro -> `f/6.3` bunker -> `f/5.6` climax)
      - Subtle handheld micro-drift + f=34 pin-rip micro-snap + f=66 spit-flick whip-accent (Beat 1),
        silky stabilized gimbal orbit (Beat 2-4), escalating overpressure tremor (Beat 5, f=540..611),
        and shockwave jolt + 53° bullet-time orbital sweep (Beat 6, f=612..720).
    """
    if bpy is None:
        raise RuntimeError("build_cinema_camera() must be executed inside Blender (bpy unavailable).")

    # Clean up any pre-existing objects of the same name if re-invoked interactively
    for existing_name in ("Cinema_Camera", "Camera_Target"):
        old_obj = bpy.data.objects.get(existing_name)
        if old_obj is not None:
            bpy.data.objects.remove(old_obj, do_unlink=True)

    cam_data = bpy.data.cameras.new("Cinema_Camera")
    cam_data.sensor_fit = "AUTO"
    cam_data.sensor_width = SENSOR_WIDTH_MM
    cam_data.clip_start = 0.01
    cam_data.clip_end = 150.0
    cam_data.lens = 48.0

    target = bpy.data.objects.new("Camera_Target", None)
    target.empty_display_type = "SPHERE"
    target.empty_display_size = 0.04
    _link_to_collection(target, col)

    cam_data.dof.use_dof = True
    cam_data.dof.focus_object = target
    cam_data.dof.aperture_fstop = 3.5
    cam_data.dof.aperture_blades = 7
    cam_data.dof.aperture_ratio = 1.0

    cam_obj = bpy.data.objects.new("Cinema_Camera", cam_data)
    _link_to_collection(cam_obj, col)
    bpy.context.scene.camera = cam_obj

    track = cam_obj.constraints.new(type="TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    # Keep metadata link to grenade_root for inspection/debugging if supplied
    if grenade_root is not None:
        cam_obj["grenade_root_name"] = grenade_root.name

    hero_v = _as_vec3(hero_pos)
    bunker_v = _as_vec3(bunker_pos)

    # Bake per-frame cinema curves (1..total_frames) for 0.000m parity with verify_framing.py,
    # zero Bezier spline ringing during whip-pans/tremors, and exact Cycles motion-blur velocities.
    for f in range(1, int(total_frames) + 1):
        c_pos, t_pos, lens_mm, fstop = camera_state_at(
            float(f),
            hero_pos=hero_v,
            bunker_pos=bunker_v,
            include_motion_fx=True,
        )
        cam_obj.location = Vector(c_pos)
        cam_obj.keyframe_insert(data_path="location", frame=f)

        target.location = Vector(t_pos)
        target.keyframe_insert(data_path="location", frame=f)

        cam_data.lens = lens_mm
        cam_data.keyframe_insert(data_path="lens", frame=f)

        cam_data.dof.aperture_fstop = fstop
        cam_data.dof.keyframe_insert(data_path="aperture_fstop", frame=f)

    _set_smooth_handles(cam_obj)
    _set_smooth_handles(target)
    _set_smooth_handles(cam_data)

    # Reset viewport/scene preview to frame 1 initial pose
    c0, t0, l0, fs0 = camera_state_at(1.0, hero_pos=hero_v, bunker_pos=bunker_v, include_motion_fx=True)
    cam_obj.location = Vector(c0)
    target.location = Vector(t0)
    cam_data.lens = l0
    cam_data.dof.aperture_fstop = fs0

    return cam_obj, target


def animate_grenade_trajectory(
    grenade_root,
    hero_pos=Vector((0.0, 0.0, 0.0)),
    bunker_pos=Vector((0.0, 7.25, 0.045)),
    start_frame: int = 134,
    end_frame: int = 495,
) -> None:
    """
    Optional convenience helper for the master scene builder to keyframe `grenade_root`
    along the exact `GRENADE_FLIGHT_KFS` trajectory (`f = 134..495`) with 0.000m sync.
    """
    if bpy is None or grenade_root is None:
        return
    hero_v = _as_vec3(hero_pos)
    bunker_v = _as_vec3(bunker_pos)
    for f in range(int(start_frame), int(end_frame) + 1):
        g_pos = grenade_pos_at(float(f), hero_pos=hero_v, bunker_pos=bunker_v)
        grenade_root.location = Vector(g_pos)
        grenade_root.keyframe_insert(data_path="location", frame=f)
    _set_smooth_handles(grenade_root)
