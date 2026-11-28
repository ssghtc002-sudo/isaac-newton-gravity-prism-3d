"""
Foley SFX Designer for Episode #16
"Why Jumping in a Falling Elevator WON'T Save You" (30.0 s @ 60 fps = 1800 frames).

Generates:
    episodes/falling-elevator/audio/output/sfx_stem.wav
    - 48000 Hz, Stereo, PCM_16, exact 30.000 s (1,440,000 samples)
    - Synthesizes all 26 cues defined in config/timeline.json
    - Synced to elevator_contract.py timings + continuous cab-to-shaft-to-grid ambient bed
    - Zero NaNs, peak <= 0.90, zero clipping
"""
from __future__ import annotations

import json
import math
import sys
import wave
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from scipy import signal

AUDIO_DIR = Path(__file__).resolve().parent
EPISODE_ROOT = AUDIO_DIR.parent
SRC_DIR = EPISODE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import elevator_contract as C  # noqa: E402

TIMELINE_PATH = EPISODE_ROOT / "config" / "timeline.json"
OUTPUT_PATH = AUDIO_DIR / "output" / "sfx_stem.wav"

SR = 48000
CHANNELS = 2
DURATION_S = float(C.DURATION_S)
TOTAL_SAMPLES = int(round(DURATION_S * SR))


# ---------------------------------------------------------------------------
# DSP & Spatial Utilities
# ---------------------------------------------------------------------------
def beat_span(beat_idx: int) -> tuple[float, float]:
    """Return (start_s, end_s) for beat_idx from elevator_contract.BEATS."""
    f0, f1 = C.BEATS[beat_idx]
    return ((f0 - 1) / float(C.FPS), f1 / float(C.FPS))


def ev_s(event_key: str) -> float:
    """Return event timestamp in seconds from elevator_contract.EV."""
    return C.EV[event_key] / float(C.FPS)


def pan_stereo(mono: np.ndarray, pan: float) -> np.ndarray:
    """Constant-power stereo panner (-1.0 left .. +1.0 right)."""
    pan = float(np.clip(pan, -1.0, 1.0))
    angle = (pan + 1.0) * (math.pi / 4.0)
    return np.stack([mono * math.cos(angle), mono * math.sin(angle)], axis=-1).astype(np.float32)


def pan_sweep_stereo(mono: np.ndarray, p0: float, p1: float) -> np.ndarray:
    """Dynamic constant-power stereo sweep from p0 to p1."""
    p = np.linspace(
        float(np.clip(p0, -1.0, 1.0)),
        float(np.clip(p1, -1.0, 1.0)),
        len(mono),
        dtype=np.float32,
    )
    ang = (p + 1.0) * (math.pi / 4.0)
    return np.stack([mono * np.cos(ang), mono * np.sin(ang)], axis=-1).astype(np.float32)


def add_stereo_event(bus: np.ndarray, event: np.ndarray, start_time: float) -> None:
    """Sample-accurately mix a mono or stereo event into bus at start_time (seconds)."""
    if event.ndim == 1:
        event = pan_stereo(event, 0.0)
    idx = int(round(start_time * SR))
    if idx >= len(bus) or idx + len(event) <= 0:
        return
    src_s = max(0, -idx)
    dst_s = max(0, idx)
    clen = min(len(event) - src_s, len(bus) - dst_s)
    if clen > 0:
        bus[dst_s : dst_s + clen] += event[src_s : src_s + clen]


def butter_filter(
    audio: np.ndarray,
    btype: str,
    cutoff: float | tuple[float, float],
    order: int = 2,
) -> np.ndarray:
    """Butterworth SOS filter along time axis=0."""
    nyq = 0.5 * SR
    if isinstance(cutoff, tuple):
        wn: float | list[float] = [float(np.clip(c / nyq, 1e-4, 0.999)) for c in cutoff]
    else:
        wn = float(np.clip(cutoff / nyq, 1e-4, 0.999))
    sos = signal.butter(order, wn, btype=btype, output="sos")
    return signal.sosfilt(sos, audio, axis=0).astype(np.float32)


def peaking_eq(audio: np.ndarray, freq: float, q: float, gain_db: float) -> np.ndarray:
    """RBJ peaking EQ biquad filter along time axis=0."""
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / SR
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    norm = 1.0 + alpha / a_lin
    b = np.array([1.0 + alpha * a_lin, -2.0 * cw, 1.0 - alpha * a_lin], dtype=np.float64) / norm
    a = np.array([1.0, (-2.0 * cw) / norm, (1.0 - alpha / a_lin) / norm], dtype=np.float64)
    return signal.lfilter(b, a, audio.astype(np.float64), axis=0).astype(np.float32)


def make_env(n: int, attack_s: float, release_s: float) -> np.ndarray:
    """Smooth attack/release window for a buffer of length n."""
    t = np.arange(n, dtype=np.float32) / SR
    dur = n / float(SR)
    att = np.minimum(1.0, t / max(1e-4, attack_s))
    rel = np.minimum(1.0, (dur - t) / max(1e-4, release_s))
    return (att * rel).astype(np.float32)


def phase_integral(freq_hz: np.ndarray) -> np.ndarray:
    """Instantaneous phase integral 2*pi*cumsum(f)/SR in float64 precision."""
    return 2.0 * math.pi * np.cumsum(freq_hz.astype(np.float64)) / SR


def add_room_reflections(
    stereo: np.ndarray,
    taps_ms: tuple[float, ...] = (7.0, 15.0, 27.0),
    gains: tuple[float, ...] = (0.22, 0.13, 0.07),
    lp_hz: float = 4000.0,
) -> np.ndarray:
    """Short early-reflection taps simulating brushed-steel elevator cab or concrete shaft."""
    out = stereo.copy()
    refl = np.zeros_like(stereo)
    for i, (ms, g) in enumerate(zip(taps_ms, gains)):
        d = int(round(ms * 0.001 * SR))
        if 0 < d < len(stereo):
            if i % 2 == 0:
                refl[d:, 0] += stereo[:-d, 0] * g
                refl[d:, 1] += stereo[:-d, 1] * (g * 0.84)
            else:
                refl[d:, 0] += stereo[:-d, 1] * (g * 0.86)
                refl[d:, 1] += stereo[:-d, 0] * g
    out += butter_filter(refl, "low", lp_hz, order=2)
    return out.astype(np.float32)


def soft_limit_peak(audio: np.ndarray, ceiling: float = 0.88) -> np.ndarray:
    """Lookahead smooth gain limiter guaranteeing peak <= ceiling with zero clipping."""
    abs_pk = np.max(np.abs(audio), axis=1).astype(np.float32)
    step = 48  # 1 ms blocks
    n_blk = (len(abs_pk) + step - 1) // step
    blk = np.array(
        [float(np.max(abs_pk[i * step : min(len(abs_pk), (i + 1) * step)])) for i in range(n_blk)],
        dtype=np.float32,
    )
    att = math.exp(-step / max(1.0, 0.0007 * SR))
    rel = math.exp(-step / max(1.0, 0.045 * SR))
    env = np.zeros(n_blk, dtype=np.float32)
    cur = 0.0
    for i in range(n_blk):
        v = float(blk[i])
        cur = (att * cur + (1.0 - att) * v) if v > cur else (rel * cur + (1.0 - rel) * v)
        env[i] = max(cur, v)
    env_full = np.interp(
        np.linspace(0.0, max(0, n_blk - 1), len(abs_pk), dtype=np.float32),
        np.arange(n_blk, dtype=np.float32),
        env,
    ).astype(np.float32)
    gain = np.where(env_full > ceiling, ceiling / np.maximum(env_full, 1e-6), 1.0).astype(np.float32)
    out = (audio * gain[:, None]).astype(np.float32)
    pk = float(np.max(np.abs(out)))
    if pk > ceiling:
        out *= float(ceiling / pk)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Individual Foley Synthesizers (all 26 timeline.json SFX cues)
# ---------------------------------------------------------------------------
def _sfx_elevator_hum_ambient(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Metallic elevator cab hum, ballast buzz & strained steel hoist cable creaks (0.12 -> 5.40s)."""
    dur = 5.28
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    prog = np.clip(t / dur, 0.0, 1.0)

    # Traction motor harmonic hum + fluorescent overhead ballast
    hum_f = 58.0 + 18.0 * (prog ** 1.5)
    motor_hum = (
        0.55 * np.sin(phase_integral(hum_f))
        + 0.32 * np.sin(phase_integral(hum_f * 2.0))
        + 0.14 * np.sin(2.0 * math.pi * 120.0 * t)
        + 0.07 * np.sin(2.0 * math.pi * 240.0 * t)
    ).astype(np.float32) * (0.28 + 0.25 * prog)

    # Cab ventilation duct airflow
    vent = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (140.0, 920.0), order=2) * 0.18

    # Strained braided steel cable creaks building toward the 5.40s snap
    creak_layer = np.zeros(n, dtype=np.float32)
    for tc, c_dur, f_base, amp in [
        (0.85, 0.42, 310.0, 0.14),
        (2.15, 0.48, 365.0, 0.18),
        (3.45, 0.52, 420.0, 0.24),
        (4.52, 0.62, 485.0, 0.34),
    ]:
        cn = int(c_dur * SR)
        ct = np.arange(cn, dtype=np.float32) / SR
        cenv = np.maximum(0.0, np.sin(math.pi * ct / c_dur)) ** 1.4
        # Stick-slip tension groans
        slip = 1.0 + 0.14 * np.sin(2.0 * math.pi * 19.0 * ct + 2.0 * np.sin(2.0 * math.pi * 5.0 * ct))
        cf = (f_base + 110.0 * (ct / c_dur)) * slip
        ctone = (
            0.55 * np.sin(phase_integral(cf))
            + 0.35 * np.sin(phase_integral(cf * 1.63))
            + 0.20 * np.sin(phase_integral(cf * 2.41))
        ).astype(np.float32)
        cgrit = butter_filter(rng.standard_normal(cn).astype(np.float32), "band", (650.0, 2600.0), order=2)
        csig = ((ctone * 0.45 + cgrit * 0.55) * cenv * amp).astype(np.float32)
        i0 = int(round(tc * SR))
        i1 = min(n, i0 + cn)
        if i1 > i0:
            creak_layer[i0:i1] += csig[: i1 - i0]

    mono = (motor_hum * 0.28 + vent * 0.32 + creak_layer * 0.52).astype(np.float32)
    stereo = add_room_reflections(pan_stereo(mono, pan), taps_ms=(6.0, 13.0, 22.0), gains=(0.20, 0.12, 0.06), lp_hz=2800.0)
    return (stereo * make_env(n, 0.12, 0.08)[:, None]).astype(np.float32)


def _sfx_alarm_start_chime(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Crisp electronic floor-drop chime + initial red fault warning tone (1.82s)."""
    dur = 0.62
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # Relay micro-click at 0.00s
    click = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1800.0, 7200.0), order=2)
    click *= np.exp(-t * 210.0) * 0.42

    # Crisp dual-tone descending elevator floor-drop chime (G5 -> D5, 783.99 Hz -> 587.33 Hz)
    c1_env = np.minimum(1.0, t / 0.004) * np.exp(-t * 9.5)
    chime1 = (
        0.72 * np.sin(2.0 * math.pi * 783.99 * t)
        + 0.28 * np.sin(2.0 * math.pi * 1567.98 * t)
        + 0.12 * np.sin(2.0 * math.pi * 2351.97 * t)
    ) * c1_env * 0.48

    t2 = np.maximum(0.0, t - 0.145)
    m2 = (t >= 0.145).astype(np.float32)
    c2_env = np.minimum(1.0, t2 / 0.004) * np.exp(-t2 * 7.8) * m2
    chime2 = (
        0.74 * np.sin(2.0 * math.pi * 587.33 * t2)
        + 0.25 * np.sin(2.0 * math.pi * 1174.66 * t2)
        + 0.10 * np.sin(2.0 * math.pi * 1761.99 * t2)
    ) * c2_env * 0.46

    # Sharp red LED fault beep at 0.34s
    t3 = np.maximum(0.0, t - 0.34)
    m3 = ((t >= 0.34) & (t < 0.51)).astype(np.float32)
    e3 = np.minimum(1.0, t3 / 0.004) * np.minimum(1.0, np.maximum(0.0, 0.51 - t) / 0.008) * m3
    beep = (
        0.65 * np.sin(2.0 * math.pi * 1318.5 * t3)
        + 0.28 * np.sin(2.0 * math.pi * 2637.0 * t3)
        + 0.14 * np.sin(2.0 * math.pi * 3955.5 * t3)
    ) * e3 * 0.38

    mono = np.tanh((click + chime1 + chime2 + beep) * 1.02).astype(np.float32) * 0.54
    stereo = add_room_reflections(pan_stereo(mono, pan), taps_ms=(5.5, 11.5), gains=(0.22, 0.11), lp_hz=4500.0)
    return (stereo * make_env(n, 0.002, 0.025)[:, None]).astype(np.float32)


def _sfx_alarm_pulse(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Double red emergency cab alarm pulse as floor indicator speeds down (2.48s)."""
    dur = 0.56
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    mono = np.zeros(n, dtype=np.float32)

    for ts, te, f0 in [(0.00, 0.16, 1396.9), (0.25, 0.43, 1396.9)]:
        dt = np.maximum(0.0, t - ts)
        mask = ((t >= ts) & (t < te)).astype(np.float32)
        env = np.minimum(1.0, dt / 0.004) * np.minimum(1.0, np.maximum(0.0, te - t) / 0.010) * mask
        tone = (
            0.66 * np.sin(2.0 * math.pi * f0 * dt)
            + 0.28 * np.sin(2.0 * math.pi * (f0 * 2.0) * dt)
            + 0.16 * np.sin(2.0 * math.pi * (f0 * 3.0) * dt)
        )
        click = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1800.0, 5600.0), order=2)
        mono += (tone * env + click * np.exp(-dt * 220.0) * mask * 0.20).astype(np.float32)

    mono = np.tanh(mono * 1.05).astype(np.float32) * 0.44
    stereo = add_room_reflections(pan_stereo(mono, pan), taps_ms=(5.0, 12.0), gains=(0.20, 0.10), lp_hz=4200.0)
    return (stereo * make_env(n, 0.002, 0.02)[:, None]).astype(np.float32)


def _sfx_crouch_cloth_creak(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Passenger crouching into jump stance: jacket rustle, sneaker pivot & floor thump (3.15s)."""
    dur = 0.68
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # Denim & jacket fabric swish
    cloth_env = np.maximum(0.0, np.sin(math.pi * np.clip(t / 0.58, 0.0, 1.0))) ** 1.1
    cloth = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (620.0, 3400.0), order=2) * cloth_env * 0.36

    # Sneaker rubber squeak on steel cab floor at 0.18s
    ts = np.maximum(0.0, t - 0.18)
    ms = (t >= 0.18).astype(np.float32)
    squeak_f = 1650.0 + 480.0 * np.sin(2.0 * math.pi * 12.0 * ts)
    squeak = np.sin(phase_integral(squeak_f)).astype(np.float32) * np.exp(-ts * 28.0) * ms * 0.22

    # Cab floor subtle weight shift resonance
    shift = np.sin(phase_integral(96.0 * np.exp(-ts * 14.0) + 48.0)).astype(np.float32) * np.exp(-ts * 12.0) * ms * 0.32

    mono = np.tanh((cloth + squeak + shift) * 0.95).astype(np.float32) * 0.46
    stereo = add_room_reflections(pan_stereo(mono, pan), taps_ms=(6.0, 12.0), gains=(0.18, 0.09), lp_hz=3000.0)
    return (stereo * make_env(n, 0.015, 0.04)[:, None]).astype(np.float32)


def _sfx_cab_tremor_rattle(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Escalating steel cab tremor, guide-shoe chatter & cable strand groans before snap (3.85s)."""
    dur = 1.52
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    prog = t / dur
    env = np.minimum(1.0, t / 0.10) * (0.35 + 0.65 * (prog ** 1.4)) * np.minimum(1.0, (dur - t) / 0.06)

    # Low structural shudder + panel rattle bursts (~14-22 Hz chatter)
    shudder = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (36.0, 145.0), order=2) * 0.68
    chatter_mod = np.maximum(0.0, np.sin(2.0 * math.pi * (11.0 + 10.0 * prog) * t)) ** 2.0
    panel_rattle = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (320.0, 2100.0), order=2)
    panel_rattle *= (0.35 + 0.65 * chatter_mod) * 0.56

    # Strained wire-rope overtone whine climbing right up to failure
    wire_f = 380.0 + 540.0 * (prog ** 2.2)
    wire = (
        0.52 * np.sin(phase_integral(wire_f))
        + 0.30 * np.sin(phase_integral(wire_f * 1.51))
    ).astype(np.float32) * (0.20 + 0.80 * (prog ** 2.0)) * 0.34

    # Sharp individual wire-strand micro-pops before main rupture
    strand_pops = np.zeros(n, dtype=np.float32)
    for pt in (0.72, 1.08, 1.31, 1.42):
        tp = np.maximum(0.0, t - pt)
        mp = (t >= pt).astype(np.float32)
        strand_pops += (
            np.sin(2.0 * math.pi * 2450.0 * tp) * np.exp(-tp * 65.0) * mp * 0.24
            + butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1800.0, 7500.0), order=2)
            * np.exp(-tp * 90.0)
            * mp
            * 0.38
        )

    mono = np.tanh((shudder + panel_rattle + wire) * env + strand_pops).astype(np.float32) * 0.58
    stereo = add_room_reflections(pan_sweep_stereo(mono, pan - 0.12, pan + 0.12), taps_ms=(7.0, 16.0), gains=(0.20, 0.11))
    return stereo.astype(np.float32)


def _sfx_cable_snap_whip(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Razor-sharp steel cable SNAP + high-voltage spark crack + metallic whip echo (5.40s)."""
    dur = 1.65
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # 1. Violent steel rupture transient (0.00s) + sub-bass gut-drop release
    crack = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (900.0, 14500.0), order=2)
    crack *= np.exp(-t * 48.0) * 1.45

    snap_ring = (
        0.58 * np.sin(2.0 * math.pi * 1840.0 * t)
        + 0.44 * np.sin(2.0 * math.pi * 3190.0 * t)
        + 0.32 * np.sin(2.0 * math.pi * 5420.0 * t)
        + 0.22 * np.sin(2.0 * math.pi * 8110.0 * t)
    ) * np.exp(-t * 19.0) * 0.72

    gut_drop = np.sin(phase_integral(165.0 * np.exp(-t * 8.5) + 34.0)).astype(np.float32)
    gut_drop *= np.minimum(1.0, t / 0.004) * np.exp(-t * 3.8) * 0.88

    # 2. High-voltage traction motor / roof junction spark arc crackle (0.01..0.38s)
    spark_env = np.minimum(1.0, t / 0.004) * np.exp(-t * 9.0)
    impulses = (rng.random(n) > 0.968).astype(np.float32) * rng.uniform(0.6, 1.8, n).astype(np.float32)
    spark_burst = butter_filter(impulses, "band", (2600.0, 14000.0), order=2) * 2.8 * spark_env
    arc_buzz = (
        0.55 * np.sin(2.0 * math.pi * 120.0 * t)
        + 0.35 * np.sin(2.0 * math.pi * 360.0 * t)
        + 0.22 * np.sin(2.0 * math.pi * 840.0 * t)
    ).astype(np.float32) * spark_env * 0.42

    # 3. Severed steel cable WHIP lash sweeping across stereo shaft + metallic clangs
    tw = np.maximum(0.0, t - 0.03)
    mw = (t >= 0.03).astype(np.float32)
    whip_f = 1950.0 * np.exp(-tw * 3.8) + 280.0
    whip_tone = np.sin(phase_integral(whip_f)).astype(np.float32) * np.exp(-tw * 5.2) * mw * 0.45
    whip_air = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (750.0, 6200.0), order=2)
    whip_air *= np.exp(-((tw - 0.11) / 0.08) ** 2) * mw * 0.92

    # Secondary shaft wall whip slap at 0.24s and 0.48s
    wall_slaps = np.zeros(n, dtype=np.float32)
    for ts, s_pan, amp in [(0.23, -0.55, 0.44), (0.46, 0.50, 0.26)]:
        td = np.maximum(0.0, t - ts)
        md = (t >= ts).astype(np.float32)
        slap = (
            butter_filter(rng.standard_normal(n).astype(np.float32), "band", (500.0, 4800.0), order=2) * np.exp(-td * 22.0)
            + np.sin(2.0 * math.pi * 1420.0 * td) * np.exp(-td * 18.0) * 0.5
        ) * md * amp
        wall_slaps += slap

    core = np.tanh((crack + snap_ring + gut_drop + spark_burst + arc_buzz) * 1.05).astype(np.float32) * 0.92
    stereo = pan_stereo(core, pan)
    stereo += pan_sweep_stereo((whip_tone + whip_air).astype(np.float32) * 0.72, -0.65, 0.72)
    stereo += pan_stereo(wall_slaps, 0.25)

    # Metallic concrete shaft whip echo taps
    stereo = add_room_reflections(
        stereo,
        taps_ms=(18.0, 42.0, 88.0, 145.0),
        gains=(0.28, 0.19, 0.12, 0.07),
        lp_hz=3600.0,
    )
    return (np.tanh(stereo * 0.98) * 0.92 * make_env(n, 0.001, 0.15)[:, None]).astype(np.float32)


def _sfx_xray_scan_whoosh(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """High-tech cyan X-ray cutaway scan sweep revealing the shaft plunge (5.65s)."""
    dur = 0.85
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    prog = t / dur
    bell = np.maximum(0.0, np.sin(math.pi * prog)) ** 1.25

    air = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (600.0, 5600.0), order=2)
    f_sw = 380.0 + 1280.0 * (prog ** 1.15)
    shimmer = 0.80 + 0.20 * np.sin(2.0 * math.pi * 26.0 * t)
    tone = (
        0.48 * np.sin(phase_integral(f_sw))
        + 0.28 * np.sin(phase_integral(f_sw * 1.498))
        + 0.16 * np.sin(phase_integral(f_sw * 2.0))
    ).astype(np.float32) * shimmer

    mono = (air * 0.42 + tone * 0.34) * bell * 0.54
    p0 = float(np.clip(pan - 0.52, -0.90, 0.90))
    p1 = float(np.clip(pan + 0.52, -0.90, 0.90))
    return (pan_sweep_stereo(mono, p0, p1) * make_env(n, 0.015, 0.06)[:, None]).astype(np.float32)


def _sfx_shaft_wind_ramp(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Surging concrete shaft wind & accelerating bay-ring Doppler whooshes (6.10 -> 8.20s)."""
    dur = 2.15
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    prog = t / dur
    env = np.minimum(1.0, t / 0.16) * (0.28 + 0.72 * (prog ** 1.2)) * np.minimum(1.0, (dur - t) / 0.25)

    # Wide stereo rushing shaft air
    l_air = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (110.0, 1800.0), order=2)
    r_air = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (115.0, 1850.0), order=2)

    # Resonant shaft pipe pitch climbing as speed increases
    pipe_f = 135.0 + 190.0 * (prog ** 1.3)
    pipe = np.sin(phase_integral(pipe_f)).astype(np.float32) * 0.18

    # Accelerating concrete floor-ring Doppler whooshes passing outside the cab
    bay_hits = np.zeros(n, dtype=np.float32)
    for bt in (0.42, 0.95, 1.36, 1.68, 1.94):
        dt = t - bt
        w_env = np.exp(-(dt / 0.085) ** 2).astype(np.float32)
        f_dop = 420.0 * np.where(dt < 0.0, 1.18, 0.82)
        dop_tone = np.sin(phase_integral(f_dop)).astype(np.float32) * 0.16
        dop_air = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (160.0, 1250.0), order=2) * 0.65
        bay_hits += (dop_tone + dop_air) * w_env * 0.42

    stereo = np.stack(
        [(l_air * 0.45 + pipe + bay_hits) * env, (r_air * 0.45 + pipe + bay_hits) * env],
        axis=-1,
    ).astype(np.float32) * 0.56
    if abs(pan) > 1e-3:
        stereo *= pan_stereo(np.ones(n, dtype=np.float32), pan) * math.sqrt(2.0)
    return stereo.astype(np.float32)


def _sfx_shaft_wind_howl(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Sustained 50 mph elevator shaft wind howl & rapid concrete bay flybys (8.00 -> 11.40s)."""
    dur = 3.45
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    env = np.minimum(1.0, t / 0.30) * np.minimum(1.0, (dur - t) / 0.40)

    # Twin-channel howling shaft turbulence + resonant hollow shaft tone (~240 Hz)
    swirl_l = 0.72 + 0.28 * np.sin(2.0 * math.pi * 1.15 * t)
    swirl_r = 0.72 + 0.28 * np.sin(2.0 * math.pi * 1.22 * t + 1.1)
    wind_l = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (120.0, 2350.0), order=2) * swirl_l
    wind_r = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (125.0, 2400.0), order=2) * swirl_r
    howl = (
        0.55 * np.sin(phase_integral(235.0 + 28.0 * np.sin(2.0 * math.pi * 0.85 * t)))
        + 0.25 * np.sin(phase_integral(470.0 + 52.0 * np.sin(2.0 * math.pi * 0.85 * t)))
    ).astype(np.float32) * 0.16

    # Rapid 50 mph concrete floor-bay Doppler whooshes (~every 0.27s)
    bays = np.zeros((n, 2), dtype=np.float32)
    t_bay = 0.22
    idx = 0
    while t_bay < dur - 0.22:
        dt = t - t_bay
        b_env = np.exp(-(dt / 0.055) ** 2).astype(np.float32)
        b_air = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (180.0, 1450.0), order=2)
        b_pan = -0.35 if (idx % 2 == 0) else 0.35
        bays += pan_stereo(b_air * b_env * 0.38, b_pan)
        t_bay += 0.27
        idx += 1

    stereo = np.stack([(wind_l * 0.42 + howl) * env, (wind_r * 0.42 + howl) * env], axis=-1).astype(np.float32)
    stereo = (stereo + bays * env[:, None]) * 0.56
    if abs(pan) > 1e-3:
        stereo *= pan_stereo(np.ones(n, dtype=np.float32), pan) * math.sqrt(2.0)
    return stereo.astype(np.float32)


def _sfx_jump_crouch_coil(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Athletic squat coil before jump: cloth tension & sneaker sole grip (10.32s)."""
    dur = 0.62
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    coil_env = np.maximum(0.0, np.sin(math.pi * np.clip(t / 0.56, 0.0, 1.0))) ** 1.2
    fabric = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (580.0, 3200.0), order=2) * coil_env * 0.38

    # Sneaker grip creak at 0.24s
    tg = np.maximum(0.0, t - 0.24)
    mg = (t >= 0.24).astype(np.float32)
    grip = np.sin(phase_integral(1420.0 - 280.0 * np.clip(tg / 0.14, 0.0, 1.0))).astype(np.float32)
    grip *= np.exp(-tg * 26.0) * mg * 0.22
    sole = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (380.0, 1900.0), order=2) * np.exp(-tg * 24.0) * mg * 0.28

    mono = np.tanh((fabric + grip + sole) * 0.95).astype(np.float32) * 0.48
    stereo = add_room_reflections(pan_stereo(mono, pan), taps_ms=(6.0, 12.0), gains=(0.16, 0.08), lp_hz=3000.0)
    return (stereo * make_env(n, 0.015, 0.03)[:, None]).astype(np.float32)


def _sfx_jump_launch_whoosh(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Athletic sneaker jump push-off pop, cab floor rebound & upward body whoosh (11.07s)."""
    dur = 0.68
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # 1. Crisp sneaker rubber push-off snap + hollow steel cab floor thump
    push_thump = np.sin(phase_integral(135.0 * np.exp(-t * 28.0) + 54.0)).astype(np.float32) * np.exp(-t * 18.0) * 0.68
    rubber_pop = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (950.0, 4800.0), order=2)
    rubber_pop *= np.exp(-t * 42.0) * 0.72
    floor_ring = (
        0.35 * np.sin(2.0 * math.pi * 195.0 * t)
        + 0.20 * np.sin(2.0 * math.pi * 340.0 * t)
    ) * np.exp(-t * 14.0) * 0.34

    # 2. Upward air-displacement whoosh (-2 mph effort)
    tw = np.maximum(0.0, t - 0.04)
    mw = (t >= 0.04).astype(np.float32)
    w_env = np.minimum(1.0, tw / 0.06) * np.exp(-tw * 4.5) * mw
    whoosh_air = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (220.0, 2200.0), order=2) * w_env * 0.64
    whoosh_tone = np.sin(phase_integral(220.0 + 340.0 * np.clip(tw / 0.35, 0.0, 1.0))).astype(np.float32) * w_env * 0.16

    mono = np.tanh((push_thump + rubber_pop + floor_ring + whoosh_air + whoosh_tone) * 1.02).astype(np.float32) * 0.68
    stereo = add_room_reflections(pan_stereo(mono, pan), taps_ms=(6.5, 13.5), gains=(0.20, 0.10), lp_hz=3400.0)
    return (stereo * make_env(n, 0.002, 0.04)[:, None]).astype(np.float32)


def _sfx_ceiling_hit_thud(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Visceral dull metallic skull-on-ceiling thud + cab header panel rattle (12.48s)."""
    dur = 0.78
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # 1. Heavy visceral skull/body dull thud (140 -> 46 Hz)
    skull_thud = np.sin(phase_integral(148.0 * np.exp(-t * 30.0) + 46.0)).astype(np.float32) * np.exp(-t * 13.5) * 0.96
    flesh_slap = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (140.0, 1150.0), order=2)
    flesh_slap *= np.exp(-t * 36.0) * 0.85

    # 2. Hollow brushed-steel ceiling header drum ring (218 Hz & 372 Hz)
    metal_drum = (
        0.58 * np.sin(2.0 * math.pi * 218.0 * t) * np.exp(-t * 11.0)
        + 0.38 * np.sin(2.0 * math.pi * 372.0 * t) * np.exp(-t * 15.0)
        + 0.22 * np.sin(2.0 * math.pi * 645.0 * t) * np.exp(-t * 22.0)
    ).astype(np.float32) * 0.56

    # 3. Overhead ceiling LED diffuser / metal trim micro-rattle (0.02..0.28s)
    tr = np.maximum(0.0, t - 0.015)
    mr = (t >= 0.015).astype(np.float32)
    rattle_mod = 0.55 + 0.45 * np.sin(2.0 * math.pi * 34.0 * tr)
    rattle = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1200.0, 5400.0), order=2)
    rattle *= np.exp(-tr * 16.0) * rattle_mod * mr * 0.36

    mono = np.tanh((skull_thud + flesh_slap + metal_drum + rattle) * 1.08).astype(np.float32) * 0.86
    stereo = add_room_reflections(pan_stereo(mono, pan), taps_ms=(5.5, 11.5, 19.0), gains=(0.24, 0.14, 0.08), lp_hz=3200.0)
    return (stereo * make_env(n, 0.002, 0.05)[:, None]).astype(np.float32)


def _sfx_freefall_drop_rush(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """50 mph pit-approach wind surge + sneaker floor contact before 28G impact (13.60 -> 15.55s)."""
    dur = 1.98
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    prog = t / dur
    env = np.minimum(1.0, t / 0.14) * (0.35 + 0.65 * (prog ** 1.3)) * np.minimum(1.0, (dur - t) / 0.06)

    # Accelerating cab airflow + rising tension sub-sweep into pit impact at 15.65s
    air_l = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (140.0, 2600.0), order=2)
    air_r = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (145.0, 2650.0), order=2)
    riser_f = 55.0 + 115.0 * (prog ** 2.2)
    riser = np.sin(phase_integral(riser_f)).astype(np.float32) * (prog ** 1.8) * 0.28

    # Soft sneaker touchdown on cab floor at ~1.42s (f~900)
    td = np.maximum(0.0, t - 1.42)
    md = (t >= 1.42).astype(np.float32)
    foot_thump = (
        np.sin(phase_integral(115.0 * np.exp(-td * 22.0) + 48.0)).astype(np.float32) * np.exp(-td * 20.0) * 0.42
        + butter_filter(rng.standard_normal(n).astype(np.float32), "band", (220.0, 1400.0), order=2) * np.exp(-td * 30.0) * 0.36
    ) * md

    stereo = np.stack([(air_l * 0.42 + riser) * env + foot_thump, (air_r * 0.42 + riser) * env + foot_thump], axis=-1)
    stereo = stereo.astype(np.float32) * 0.56
    if abs(pan) > 1e-3:
        stereo *= pan_stereo(np.ones(n, dtype=np.float32), pan) * math.sqrt(2.0)
    return stereo.astype(np.float32)


def _sfx_xray_in_sweep(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Shimmering cyan X-ray bone scan activation sweep into Beat 4 (15.15s)."""
    dur = 0.92
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    prog = t / dur
    bell = np.maximum(0.0, np.sin(math.pi * prog)) ** 1.2

    air = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (650.0, 6800.0), order=2)
    f_scan = 420.0 + 1650.0 * (prog ** 1.1)
    shimmer = 0.76 + 0.24 * np.sin(2.0 * math.pi * 32.0 * t)
    cyan_laser = (
        0.50 * np.sin(phase_integral(f_scan))
        + 0.32 * np.sin(phase_integral(f_scan * 1.498))
        + 0.22 * np.sin(phase_integral(f_scan * 2.0))
    ).astype(np.float32) * shimmer

    # Crystalline bone-scan telemetry lock chime around t=0.45s
    ch_env = np.exp(-((t - 0.45) / 0.14) ** 2).astype(np.float32)
    chime = (
        0.48 * np.sin(2.0 * math.pi * 1318.51 * t)  # E6
        + 0.26 * np.sin(2.0 * math.pi * 1975.53 * t)  # B6
    ) * ch_env * 0.28

    mono = ((air * 0.40 + cyan_laser * 0.38) * bell + chime).astype(np.float32) * 0.58
    p0 = float(np.clip(pan - 0.56, -0.90, 0.90))
    p1 = float(np.clip(pan + 0.56, -0.90, 0.90))
    return (pan_sweep_stereo(mono, p0, p1) * make_env(n, 0.012, 0.06)[:, None]).astype(np.float32)


def _sfx_pit_impact_boom(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """28 G Hydraulic Pit Impact: massive 32-55 Hz floor slam, buffer strike & steel reverb (15.65s)."""
    dur = 1.85
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # 1. Massive sub-bass floor slam (115 -> 33 Hz)
    sub_f = 115.0 * np.exp(-t * 6.5) + 33.5
    sub_boom = np.sin(phase_integral(sub_f)).astype(np.float32) * np.minimum(1.0, t / 0.003) * np.exp(-t * 1.85) * 1.05
    sub_45 = np.sin(2.0 * math.pi * 46.0 * t).astype(np.float32) * np.exp(-t * 2.6) * 0.45

    # 2. Heavy steel chassis slam transient + concrete pit crunch
    slam_crack = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (180.0, 7200.0), order=2)
    slam_crack *= np.exp(-t * 24.0) * 1.18
    concrete_rumble = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (34.0, 220.0), order=3)
    concrete_rumble *= np.minimum(1.0, t / 0.006) * np.exp(-t * 2.1) * 0.92

    # 3. Metallic cab chassis modal reverberation + spring buffer ring
    chassis_ring = (
        0.52 * np.sin(2.0 * math.pi * 138.0 * t) * np.exp(-t * 4.5)
        + 0.36 * np.sin(2.0 * math.pi * 244.0 * t) * np.exp(-t * 6.8)
        + 0.24 * np.sin(2.0 * math.pi * 412.0 * t) * np.exp(-t * 9.5)
        + 0.16 * np.sin(2.0 * math.pi * 890.0 * t) * np.exp(-t * 14.0)
    ).astype(np.float32) * 0.48

    mono = np.tanh((sub_boom + sub_45 + slam_crack + concrete_rumble + chassis_ring) * 0.96).astype(np.float32) * 0.92
    stereo = add_room_reflections(
        pan_stereo(mono, pan),
        taps_ms=(11.0, 24.0, 44.0, 76.0),
        gains=(0.24, 0.16, 0.10, 0.05),
        lp_hz=3200.0,
    )
    return (np.tanh(stereo * 0.98) * 0.92 * make_env(n, 0.002, 0.12)[:, None]).astype(np.float32)


def _sfx_pit_peak_comp_crunch(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Hydraulic pit buffer bottom-out spring crunch & 28 G chassis compression groan (16.40s)."""
    dur = 1.25
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # 1. Twin heavy coil-spring compression crunch + hydraulic fluid hiss
    spring_env = np.minimum(1.0, t / 0.015) * np.exp(-t * 2.8)
    spring_f = 185.0 - 75.0 * np.clip(t / 0.65, 0.0, 1.0)
    spring_groan = (
        0.56 * np.sin(phase_integral(spring_f))
        + 0.32 * np.sin(phase_integral(spring_f * 1.62))
        + 0.20 * np.sin(phase_integral(spring_f * 2.54))
    ).astype(np.float32) * spring_env * 0.58

    # Hydraulic oil valve squeeze hiss
    hyd_hiss = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1400.0, 6800.0), order=2)
    hyd_hiss *= np.minimum(1.0, t / 0.025) * np.exp(-t * 4.5) * 0.46

    # 2. Structural metallic crunch & bone-load creaks
    crunch_imp = (rng.random(n) > 0.982).astype(np.float32) * rng.uniform(0.4, 1.5, n).astype(np.float32)
    crunch = butter_filter(crunch_imp, "band", (380.0, 4200.0), order=2) * 2.4 * np.exp(-t * 3.6)

    # Secondary buffer thud
    sub_thud = np.sin(phase_integral(78.0 * np.exp(-t * 8.0) + 36.0)).astype(np.float32) * np.exp(-t * 3.2) * 0.55

    mono = np.tanh((spring_groan + hyd_hiss + crunch + sub_thud) * 0.98).astype(np.float32) * 0.72
    stereo = add_room_reflections(pan_stereo(mono, pan), taps_ms=(9.0, 19.0, 35.0), gains=(0.20, 0.12, 0.06), lp_hz=3500.0)
    return (stereo * make_env(n, 0.005, 0.08)[:, None]).astype(np.float32)


def _sfx_spine_shockwave_pulse(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Shimmering cyan X-ray bone shockwave & spine-shock crystalline crackle (17.40s)."""
    dur = 1.85
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # 1. Resonant upward cyan X-ray shockwave pulse (shins -> pelvis -> spine)
    pulse_env = np.minimum(1.0, t / 0.04) * np.exp(-t * 1.9)
    sw_f = 160.0 + 920.0 * (1.0 - np.exp(-t * 2.4))
    xray_wave = (
        0.48 * np.sin(phase_integral(sw_f))
        + 0.30 * np.sin(phase_integral(sw_f * 1.498))
        + 0.18 * np.sin(phase_integral(sw_f * 2.0))
    ).astype(np.float32) * (0.78 + 0.22 * np.sin(2.0 * math.pi * 24.0 * t)) * pulse_env * 0.44

    # Sub-bass axial shock ripple
    sub_rip = np.sin(phase_integral(64.0 * np.exp(-t * 2.2) + 34.0)).astype(np.float32) * np.exp(-t * 2.2) * 0.48

    # 2. Crystalline osseous vertebra compression crackles (staggered L/R stereo micro-shocks)
    stereo = pan_sweep_stereo((xray_wave + sub_rip).astype(np.float32), pan - 0.28, pan + 0.28)
    for k, tc in enumerate((0.08, 0.22, 0.39, 0.58, 0.80, 1.04)):
        sn = int(0.16 * SR)
        st = np.arange(sn, dtype=np.float32) / SR
        cf = 2100.0 + k * 420.0
        cryst = (
            0.52 * np.sin(2.0 * math.pi * cf * st)
            + 0.32 * np.sin(2.0 * math.pi * (cf * 1.54) * st)
        ) * np.exp(-st * 28.0) * np.minimum(1.0, st / 0.002)
        crk = butter_filter(rng.standard_normal(sn).astype(np.float32), "band", (1800.0, 9500.0), order=2)
        crk *= np.exp(-st * 55.0) * 0.72
        amp = 0.30 * math.exp(-k * 0.18)
        c_pan = (-0.42 if (k % 2 == 0) else 0.42) + pan * 0.4
        add_stereo_event(stereo, pan_stereo(((cryst * 0.55 + crk) * amp).astype(np.float32), c_pan), tc)

    return (np.tanh(stereo * 0.95) * 0.74 * make_env(n, 0.008, 0.12)[:, None]).astype(np.float32)


def _sfx_g_load_sub_rumble(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Damped hydraulic rebound sub-rumble & stressed steel cab settle (18.80 -> 20.50s)."""
    dur = 1.70
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    env = np.minimum(1.0, t / 0.12) * np.exp(-t * 1.55) * np.minimum(1.0, (dur - t) / 0.15)

    # Low structural rebound oscillation (~6 Hz damped spring modulation)
    bounce = 0.72 + 0.28 * np.cos(2.0 * math.pi * 3.8 * t) * np.exp(-t * 1.8)
    rumble = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (34.0, 150.0), order=2) * bounce * 0.72
    sub_tone = np.sin(2.0 * math.pi * 44.0 * t).astype(np.float32) * 0.32

    # Residual X-ray fade chime tail
    fade_tone = np.sin(2.0 * math.pi * 587.33 * t).astype(np.float32) * np.exp(-t * 2.4) * 0.12

    mono = ((rumble + sub_tone) * env + fade_tone).astype(np.float32) * 0.56
    return add_room_reflections(pan_stereo(mono, pan), taps_ms=(10.0, 22.0), gains=(0.18, 0.10), lp_hz=1800.0)


def _sfx_mechanical_cut_whoosh(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Crisp mechanical cutaway transition whoosh into mid-shaft brake clamps (20.60s)."""
    dur = 0.48
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    prog = t / dur
    env = np.maximum(0.0, np.sin(math.pi * prog)) ** 1.15

    air = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (220.0, 3200.0), order=2)
    sw = np.sin(phase_integral(260.0 + 680.0 * prog)).astype(np.float32) * 0.24
    mono = (air * 0.52 + sw) * env * 0.58
    return pan_sweep_stereo(mono, float(np.clip(pan - 0.55, -0.9, 0.9)), float(np.clip(pan + 0.55, -0.9, 0.9)))


def _synth_brake_clamp_slam(rng: np.random.Generator, pan: float, f_bias: float) -> np.ndarray:
    """Heavy hydraulic wedge latch clunk & spring-loaded steel clamp slam onto T-guide rail."""
    dur = 0.62
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # 1. Heavy hydraulic latch release clack (0.00s) -> main wedge anvil SLAM (0.025s)
    latch = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1100.0, 6200.0), order=2)
    latch *= np.exp(-t * 95.0) * 0.75

    ts = np.maximum(0.0, t - 0.022)
    ms = (t >= 0.022).astype(np.float32)
    anvil_sub = np.sin(phase_integral(155.0 * np.exp(-ts * 32.0) + 52.0)).astype(np.float32) * np.exp(-ts * 18.0) * ms * 0.92
    clamp_crack = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (420.0, 9800.0), order=2)
    clamp_crack *= np.exp(-ts * 38.0) * ms * 1.18

    # High-Q forged steel jaw resonance
    steel_ring = (
        0.55 * np.sin(2.0 * math.pi * (890.0 * f_bias) * ts) * np.exp(-ts * 14.0)
        + 0.38 * np.sin(2.0 * math.pi * (1620.0 * f_bias) * ts) * np.exp(-ts * 18.0)
        + 0.26 * np.sin(2.0 * math.pi * (2840.0 * f_bias) * ts) * np.exp(-ts * 24.0)
    ).astype(np.float32) * ms * 0.56

    mono = np.tanh((latch + anvil_sub + clamp_crack + steel_ring) * 1.05).astype(np.float32) * 0.84
    stereo = add_room_reflections(pan_stereo(mono, pan), taps_ms=(8.0, 18.0, 32.0), gains=(0.22, 0.12, 0.06), lp_hz=4200.0)
    return (stereo * make_env(n, 0.001, 0.04)[:, None]).astype(np.float32)


def _sfx_brake_clamp_slam_left(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    return _synth_brake_clamp_slam(rng, pan, f_bias=0.97)


def _sfx_brake_clamp_slam_right(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    return _synth_brake_clamp_slam(rng, pan, f_bias=1.04)


def _sfx_rail_friction_screech(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Biting steel-on-rail friction screech & sparks decelerating cleanly to stop (21.55 -> 23.05s)."""
    dur = 1.52
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    prog = t / dur

    # Decelerating speed curve: high energy initially, dropping cleanly to zero arrest
    decel = np.maximum(0.0, 1.0 - (prog ** 1.15))
    env = np.minimum(1.0, t / 0.03) * (0.20 + 0.80 * decel) * np.minimum(1.0, (dur - t) / 0.10)

    # Stick-slip chatter slowing down in frequency as cab halts
    slip_hz = 46.0 - 28.0 * prog
    jitter = 0.72 + 0.28 * np.sin(2.0 * math.pi * slip_hz * t + 1.4 * np.sin(2.0 * math.pi * 7.5 * t))

    def rail_channel(detune: float, seed_off: int) -> np.ndarray:
        lrng = np.random.default_rng(9100 + seed_off)
        # Biting high-frequency steel screech sliding down from ~2.6 kHz -> ~950 Hz
        f_screech = (2550.0 - 1580.0 * (prog ** 0.9)) * detune
        screech = (
            0.48 * np.sin(phase_integral(f_screech))
            + 0.34 * np.sin(phase_integral(f_screech * 1.41))
            + 0.22 * np.sin(phase_integral(f_screech * 2.18))
        ).astype(np.float32) * jitter
        grit = butter_filter(lrng.standard_normal(n).astype(np.float32), "band", (1800.0, 8200.0), order=2)
        sparks_imp = (lrng.random(n) > (0.978 + 0.018 * prog)).astype(np.float32) * lrng.uniform(0.5, 1.4, n).astype(np.float32)
        sparks = butter_filter(sparks_imp, "band", (3800.0, 14000.0), order=2) * 2.2
        return ((screech * 0.36 + grit * 0.44 + sparks * 0.35) * env).astype(np.float32)

    stereo = np.stack([rail_channel(0.985, 1), rail_channel(1.018, 2)], axis=-1).astype(np.float32)
    # Final arrest shudder clunk near 1.28s
    ta = np.maximum(0.0, t - 1.28)
    ma = (t >= 1.28).astype(np.float32)
    stop_clunk = np.sin(phase_integral(105.0 * np.exp(-ta * 26.0) + 46.0)).astype(np.float32) * np.exp(-ta * 18.0) * ma * 0.42
    stereo += pan_stereo(stop_clunk, pan)

    if abs(pan) > 1e-3:
        stereo *= pan_stereo(np.ones(n, dtype=np.float32), pan) * math.sqrt(2.0)
    return (np.tanh(stereo * 0.96) * 0.62 * make_env(n, 0.01, 0.06)[:, None]).astype(np.float32)


def _sfx_survival_cut_whoosh(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Clean transition whoosh + cloth settle into flat-back survival pose (23.08s)."""
    dur = 0.54
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    prog = t / dur
    env = np.maximum(0.0, np.sin(math.pi * prog)) ** 1.2

    air = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (180.0, 2600.0), order=2)
    tone = np.sin(phase_integral(440.0 - 210.0 * prog)).astype(np.float32) * 0.18
    mono = (air * 0.48 + tone) * env * 0.50
    return pan_sweep_stereo(mono, float(np.clip(pan + 0.48, -0.9, 0.9)), float(np.clip(pan - 0.48, -0.9, 0.9)))


def _sfx_lie_flat_body_thump(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Cloth rustle + firm flat-back body & palm settle onto brushed steel floor (24.40s)."""
    dur = 0.68
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # Pre-settle jacket/denim sweep (0..0.18s)
    cloth = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (480.0, 2800.0), order=2)
    cloth *= np.exp(-((t - 0.10) / 0.08) ** 2) * 0.44

    # Firm dorsal torso thump on cab floor at 0.14s + subtle shoe heel tap at 0.23s
    tb = np.maximum(0.0, t - 0.14)
    mb = (t >= 0.14).astype(np.float32)
    body_thud = np.sin(phase_integral(108.0 * np.exp(-tb * 22.0) + 46.0)).astype(np.float32) * np.exp(-tb * 15.0) * mb * 0.72
    floor_mat = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (110.0, 950.0), order=2)
    floor_mat *= np.exp(-tb * 26.0) * mb * 0.54

    th = np.maximum(0.0, t - 0.23)
    mh = (t >= 0.23).astype(np.float32)
    heel = np.sin(phase_integral(132.0 * np.exp(-th * 30.0) + 58.0)).astype(np.float32) * np.exp(-th * 22.0) * mh * 0.34

    mono = np.tanh((cloth + body_thud + floor_mat + heel) * 0.95).astype(np.float32) * 0.62
    stereo = add_room_reflections(pan_stereo(mono, pan), taps_ms=(6.0, 12.5), gains=(0.18, 0.09), lp_hz=2800.0)
    return (stereo * make_env(n, 0.005, 0.04)[:, None]).astype(np.float32)


def _sfx_grid_reveal_chime(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Serene crystalline emerald 12x force-grid reveal harmonic chime (25.73s)."""
    dur = 1.65
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # Soft airy shimmer onset
    air = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (900.0, 6200.0), order=2)
    air *= np.minimum(1.0, t / 0.08) * np.exp(-t * 3.8) * 0.22

    # Warm lush emerald chord arpeggio (C5 - G5 - D6 - E6)
    notes = [(0.00, 523.25, 0.42), (0.06, 783.99, 0.36), (0.12, 1174.66, 0.28), (0.18, 1318.51, 0.24)]
    stereo = pan_stereo(air, pan)
    for t_n, freq, amp in notes:
        tn = np.maximum(0.0, t - t_n)
        mn = (t >= t_n).astype(np.float32)
        nenv = np.minimum(1.0, tn / 0.012) * np.exp(-tn * 2.3) * mn
        bell = (
            0.70 * np.sin(2.0 * math.pi * freq * tn)
            + 0.24 * np.sin(2.0 * math.pi * (freq * 2.0) * tn) * np.exp(-tn * 4.5)
            + 0.08 * np.sin(2.0 * math.pi * (freq * 3.0) * tn) * np.exp(-tn * 7.0)
        ) * nenv * amp
        n_pan = float(np.clip(pan + (freq - 850.0) * 0.00045, -0.55, 0.55))
        stereo += pan_stereo(bell.astype(np.float32), n_pan)

    stereo = add_room_reflections(stereo, taps_ms=(14.0, 29.0, 48.0), gains=(0.18, 0.11, 0.06), lp_hz=4200.0)
    return (np.tanh(stereo * 0.92) * 0.54 * make_env(n, 0.005, 0.15)[:, None]).astype(np.float32)


def _sfx_force_spread_hum(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Serene harmonic emerald force-grid hum sustaining cleanly to 30.00s (26.80 -> 29.95s)."""
    dur = 3.15
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    env = np.minimum(1.0, t / 0.35) * np.minimum(1.0, (dur - t) / 0.65)

    # Warm emerald resonant harmonic pad (C3=130.81 Hz, G3=196.0 Hz, C4=261.63 Hz, E4=329.63 Hz)
    shimmer_l = 0.80 + 0.20 * np.sin(2.0 * math.pi * 1.4 * t)
    shimmer_r = 0.80 + 0.20 * np.sin(2.0 * math.pi * 1.35 * t + 1.2)
    pad_core = (
        0.46 * np.sin(2.0 * math.pi * 130.81 * t)
        + 0.32 * np.sin(2.0 * math.pi * 196.00 * t)
        + 0.22 * np.sin(2.0 * math.pi * 261.63 * t)
        + 0.12 * np.sin(2.0 * math.pi * 329.63 * t)
    ).astype(np.float32)

    air_silk = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (320.0, 2100.0), order=2) * 0.08
    left = (pad_core * shimmer_l + air_silk) * env * 0.30
    right = (pad_core * shimmer_r + air_silk) * env * 0.30
    stereo = np.stack([left, right], axis=-1).astype(np.float32)
    if abs(pan) > 1e-3:
        stereo *= pan_stereo(np.ones(n, dtype=np.float32), pan) * math.sqrt(2.0)
    return stereo.astype(np.float32)


SFX_REGISTRY: dict[str, Callable[[np.random.Generator, float, float], np.ndarray]] = {
    "elevator_hum_ambient": _sfx_elevator_hum_ambient,
    "alarm_start_chime": _sfx_alarm_start_chime,
    "alarm_pulse": _sfx_alarm_pulse,
    "crouch_cloth_creak": _sfx_crouch_cloth_creak,
    "cab_tremor_rattle": _sfx_cab_tremor_rattle,
    "cable_snap_whip": _sfx_cable_snap_whip,
    "xray_scan_whoosh": _sfx_xray_scan_whoosh,
    "shaft_wind_ramp": _sfx_shaft_wind_ramp,
    "shaft_wind_howl": _sfx_shaft_wind_howl,
    "jump_crouch_coil": _sfx_jump_crouch_coil,
    "jump_launch_whoosh": _sfx_jump_launch_whoosh,
    "ceiling_hit_thud": _sfx_ceiling_hit_thud,
    "freefall_drop_rush": _sfx_freefall_drop_rush,
    "xray_in_sweep": _sfx_xray_in_sweep,
    "pit_impact_boom": _sfx_pit_impact_boom,
    "pit_peak_comp_crunch": _sfx_pit_peak_comp_crunch,
    "spine_shockwave_pulse": _sfx_spine_shockwave_pulse,
    "g_load_sub_rumble": _sfx_g_load_sub_rumble,
    "mechanical_cut_whoosh": _sfx_mechanical_cut_whoosh,
    "brake_clamp_slam_left": _sfx_brake_clamp_slam_left,
    "brake_clamp_slam_right": _sfx_brake_clamp_slam_right,
    "rail_friction_screech": _sfx_rail_friction_screech,
    "survival_cut_whoosh": _sfx_survival_cut_whoosh,
    "lie_flat_body_thump": _sfx_lie_flat_body_thump,
    "grid_reveal_chime": _sfx_grid_reveal_chime,
    "force_spread_hum": _sfx_force_spread_hum,
}


def synth_sfx(kind: str, pan: float, event_t: float) -> np.ndarray:
    """Dispatch and synthesize a single Foley event with deterministic per-cue seed."""
    if kind not in SFX_REGISTRY:
        raise ValueError(f"Unknown SFX type in timeline.json: {kind!r}")
    seed = int(round(event_t * 1000)) + (sum(ord(ch) for ch in kind) * 131)
    rng = np.random.default_rng(seed)
    return SFX_REGISTRY[kind](rng, pan, event_t)


# ---------------------------------------------------------------------------
# Continuous Ambient Bed (Cab Hum -> Shaft Free-Fall Airflow -> Quiet Cab)
# ---------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Overflow-safe float32 sigmoid."""
    return (1.0 / (1.0 + np.exp(np.clip(-x, -60.0, 60.0)))).astype(np.float32)


def build_ambient_bed() -> np.ndarray:
    """Subtle continuous cab-to-shaft ambient bed sculpted cleanly under narration."""
    n = TOTAL_SAMPLES
    t = np.arange(n, dtype=np.float32) / SR
    rng_l = np.random.default_rng(1601)
    rng_r = np.random.default_rng(1602)

    raw_l = rng_l.standard_normal(n).astype(np.float32)
    raw_r = rng_r.standard_normal(n).astype(np.float32)

    t_snap = ev_s("cable_snap")
    t_impact = ev_s("pit_impact")
    t_brake_s = ev_s("brake_cut_start")
    t_surv_s = ev_s("survival_cut_start")

    # 1. Closed elevator cab electrical/vent bed (0.0 -> 5.4s, fading at snap)
    cab_env = np.minimum(1.0, t / 0.12) * _sigmoid((t_snap - t) / 0.15)
    cab_l = butter_filter(raw_l, "band", (65.0, 380.0), order=2) * 0.55
    cab_r = butter_filter(raw_r, "band", (65.0, 380.0), order=2) * 0.55
    cab_bed = np.stack([cab_l * cab_env, cab_r * cab_env], axis=-1).astype(np.float32) * 0.085

    # 2. Hollow concrete shaft airflow bed (ramping 5.4 -> 15.65s, dipping after pit impact, brief burst in 20.6..22.8s)
    fall_ramp = _sigmoid((t - (t_snap + 0.15)) / 0.35) * _sigmoid((t_impact - t) / 0.12)
    brake_air = np.where((t >= t_brake_s) & (t <= t_surv_s), np.exp(-np.maximum(0.0, t - (t_brake_s + 0.4)) * 1.4) * 0.65, 0.0)
    calm_room = np.where(t >= t_surv_s, 0.18 * np.minimum(1.0, (DURATION_S - t) / 0.35), 0.0)
    air_env = np.clip(fall_ramp * 0.90 + brake_air + calm_room, 0.0, 1.0).astype(np.float32)

    air_l = butter_filter(raw_l, "band", (110.0, 1900.0), order=2) * (0.78 + 0.22 * np.sin(2.0 * math.pi * 0.55 * t))
    air_r = butter_filter(raw_r, "band", (110.0, 1900.0), order=2) * (0.78 + 0.22 * np.sin(2.0 * math.pi * 0.51 * t + 1.3))
    air_bed = np.stack([air_l * air_env, air_r * air_env], axis=-1).astype(np.float32) * 0.082

    bed = cab_bed + air_bed
    # Vocal-pocket dip so en-US-BrianNeural remains crisp
    bed = peaking_eq(bed, 1150.0, 0.55, -3.8)
    return (bed * make_env(n, 0.02, 0.10)[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# Assembly, Verification & Export
# ---------------------------------------------------------------------------
def build_sfx_stem() -> tuple[np.ndarray, list[dict]]:
    """Synthesize all timeline.json SFX cues + ambient bed into 48kHz stereo stem."""
    tl = json.loads(TIMELINE_PATH.read_text(encoding="utf-8"))
    cues: list[dict] = tl["sfx"]

    bus = build_ambient_bed()
    for ev in cues:
        ev_type = str(ev["type"])
        ev_time = float(ev["t"])
        ev_pan = float(ev.get("pan", 0.0))
        add_stereo_event(bus, synth_sfx(ev_type, ev_pan, ev_time), ev_time)

    # Remove subsonic DC mud below 26 Hz and normalize cleanly <= 0.90 (zero clipping)
    bus = butter_filter(bus, "high", 26.0, order=2)
    bus = soft_limit_peak(bus, ceiling=0.88)

    # Click-free 5ms fade boundaries
    micro = int(round(0.005 * SR))
    bus[:micro] *= np.linspace(0.0, 1.0, micro, dtype=np.float32)[:, None]
    bus[-micro:] *= np.linspace(1.0, 0.0, micro, dtype=np.float32)[:, None]
    return bus.astype(np.float32), cues


def verify_sfx_stem(path: Path, audio: np.ndarray, cues: list[dict]) -> None:
    """Validate format, exact 30.000s sample count, zero NaNs, peak <= 0.90, and per-beat energy."""
    assert np.isfinite(audio).all(), "SFX stem contains NaN or Inf values"

    info = sf.info(str(path))
    assert info.samplerate == SR, f"Expected SR={SR}, got {info.samplerate}"
    assert info.channels == CHANNELS, f"Expected {CHANNELS}ch, got {info.channels}"
    assert info.subtype == "PCM_16", f"Expected PCM_16, got {info.subtype}"
    assert info.frames == TOTAL_SAMPLES, f"Expected {TOTAL_SAMPLES} samples, got {info.frames}"
    assert abs(info.duration - DURATION_S) < 1e-6, f"Expected {DURATION_S}s, got {info.duration}"

    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == SR
        assert wf.getnchannels() == CHANNELS
        assert wf.getsampwidth() == 2
        assert wf.getnframes() == TOTAL_SAMPLES

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    assert 0.01 < rms < 0.50, f"Unexpected SFX RMS: {rms:.5f}"
    assert peak <= 0.90, f"Peak exceeded <= 0.90 limit: {peak:.5f}"

    print(
        f"[SFX] Wrote {path.name}: {info.samplerate} Hz, {info.channels}ch, "
        f"{info.frames:,} samples ({info.duration:.3f}s), cues={len(cues)}, "
        f"peak={peak:.4f}, rms={rms:.4f}, nan=0"
    )
    for b_idx in sorted(C.BEATS):
        s0, s1 = beat_span(b_idx)
        seg = audio[int(round(s0 * SR)) : int(round(s1 * SR))]
        seg_rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        seg_pk = float(np.max(np.abs(seg)))
        print(f"  Beat {b_idx} ({s0:5.2f}-{s1:5.2f}s): rms={seg_rms:.4f}, peak={seg_pk:.4f}")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audio, cues = build_sfx_stem()
    sf.write(str(OUTPUT_PATH), audio, SR, subtype="PCM_16")
    verify_sfx_stem(OUTPUT_PATH, audio, cues)


if __name__ == "__main__":
    main()
