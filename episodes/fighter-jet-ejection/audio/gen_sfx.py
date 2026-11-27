"""
Foley SFX Designer for Episode #1
"Why Ejecting From a Fighter Jet Shrinks You by 1 Inch" (30.0 s @ 60 fps = 1800 frames).

Generates:
    episodes/fighter-jet-ejection/audio/output/sfx_stem.wav
    - 48000 Hz, Stereo, PCM_16, exact 30.000 s (1,440,000 samples)
    - Synthesizes all 22 cues (18 Foley types) defined in config/timeline.json
    - Synced to ejection_contract.py timings + continuous cockpit-to-sky ambient airflow bed
    - Zero NaNs, peak strictly < 0.96
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

import ejection_contract as C  # noqa: E402

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
    """Return (start_s, end_s) for beat_idx from ejection_contract.BEATS."""
    f0, f1 = C.BEATS[beat_idx]
    return ((f0 - 1) / float(C.FPS), f1 / float(C.FPS))


def ev_s(event_key: str) -> float:
    """Return event timestamp in seconds from ejection_contract.EV."""
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


def stereo_spatialize(mono: np.ndarray, delay_ms: float = 2.2, wet: float = 0.28) -> np.ndarray:
    """Subtle Haas micro-delay stereo reflection for natural acoustic depth."""
    d_samp = max(1, int(round(delay_ms * 0.001 * SR)))
    left = mono.copy()
    right = mono.copy()
    if d_samp < len(mono):
        left[d_samp:] += mono[:-d_samp] * (wet * 0.65)
        right[d_samp:] -= mono[:-d_samp] * (wet * 0.50)
    return np.stack([left, right], axis=-1).astype(np.float32)


def add_room_reflections(
    stereo: np.ndarray,
    taps_ms: tuple[float, ...] = (9.0, 17.0, 29.0),
    gains: tuple[float, ...] = (0.20, 0.12, 0.07),
    lp_hz: float = 4200.0,
) -> np.ndarray:
    """Short damped early-reflection taps for metallic/cockpit or air-space realism."""
    out = stereo.copy()
    refl = np.zeros_like(stereo)
    for i, (ms, g) in enumerate(zip(taps_ms, gains)):
        d = int(round(ms * 0.001 * SR))
        if 0 < d < len(stereo):
            if i % 2 == 0:
                refl[d:, 0] += stereo[:-d, 0] * g
                refl[d:, 1] += stereo[:-d, 1] * (g * 0.82)
            else:
                refl[d:, 0] += stereo[:-d, 1] * (g * 0.85)
                refl[d:, 1] += stereo[:-d, 0] * g
    out += butter_filter(refl, "low", lp_hz, order=2)
    return out.astype(np.float32)


def soft_limit_peak(audio: np.ndarray, ceiling: float = 0.92) -> np.ndarray:
    """Lookahead-style smooth gain reduction guaranteeing peak < ceiling without harsh clipping."""
    abs_pk = np.max(np.abs(audio), axis=1).astype(np.float32)
    step = 48  # 1 ms blocks
    n_blk = (len(abs_pk) + step - 1) // step
    blk = np.array(
        [float(np.max(abs_pk[i * step : min(len(abs_pk), (i + 1) * step)])) for i in range(n_blk)],
        dtype=np.float32,
    )
    att = math.exp(-step / max(1.0, 0.0008 * SR))
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
# Individual Foley Synthesizers (all 18 timeline.json SFX types)
# ---------------------------------------------------------------------------
def _sfx_cockpit_alarm(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Two-tone fighter Master Caution cockpit warning annunciator (2 urgency pulses)."""
    dur = 0.50
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    mono = np.zeros(n, dtype=np.float32)

    # Authentic alternating avionics two-tone Master Caution (1046.5 Hz <-> 784.0 Hz)
    pulses = [(0.00, 0.105, 1046.5), (0.105, 0.210, 784.0), (0.245, 0.350, 1046.5), (0.350, 0.455, 784.0)]
    for t_start, t_end, f0 in pulses:
        mask = ((t >= t_start) & (t < t_end)).astype(np.float32)
        dt = np.maximum(0.0, t - t_start)
        env = np.minimum(1.0, dt / 0.004) * np.minimum(1.0, np.maximum(0.0, t_end - t) / 0.007) * mask
        # Piezo cockpit speaker harmonic profile (fundamental + odd harmonics, band-limited)
        tone = (
            0.70 * np.sin(2.0 * math.pi * f0 * dt)
            + 0.24 * np.sin(2.0 * math.pi * (2.0 * f0) * dt)
            + 0.18 * np.sin(2.0 * math.pi * (3.0 * f0) * dt)
            + 0.08 * np.sin(2.0 * math.pi * (5.0 * f0) * dt)
        )
        # Micro speaker relay click on pulse onset
        click = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1600.0, 5200.0), order=2)
        click_env = np.exp(-dt * 240.0) * mask * 0.25
        mono += (tone * env + click * click_env).astype(np.float32)

    mono = butter_filter(mono, "band", (340.0, 4800.0), order=2)
    mono = np.tanh(mono * 1.05).astype(np.float32) * 0.41
    stereo = add_room_reflections(pan_stereo(mono, pan), taps_ms=(6.0, 13.0), gains=(0.18, 0.09), lp_hz=3200.0)
    return (stereo * make_env(n, 0.002, 0.015)[:, None]).astype(np.float32)


def _sfx_handle_grip_click(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Flight-glove grab thump + taut steel ejection-handle detent lock click."""
    dur = 0.28
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # Gloved hand slap onto striped rubber-steel grip
    glove_thump = np.sin(phase_integral(148.0 * np.exp(-t * 36.0) + 64.0)) * np.exp(-t * 42.0) * 0.42
    leather = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (360.0, 2400.0), order=2)
    leather *= np.exp(-t * 58.0) * 0.46

    # Primary steel safety detent latch click at 45ms + secondary lock tick at 95ms
    t1 = np.maximum(0.0, t - 0.045)
    m1 = (t >= 0.045).astype(np.float32)
    latch1 = (
        0.55 * np.sin(2.0 * math.pi * 3480.0 * t1)
        + 0.36 * np.sin(2.0 * math.pi * 5920.0 * t1)
        + 0.22 * np.sin(2.0 * math.pi * 8940.0 * t1)
    ) * np.exp(-t1 * 88.0) * m1
    snap1 = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (2200.0, 11500.0), order=2)
    snap1 *= np.exp(-t1 * 120.0) * m1 * 0.85

    t2 = np.maximum(0.0, t - 0.095)
    m2 = (t >= 0.095).astype(np.float32)
    latch2 = (0.42 * np.sin(2.0 * math.pi * 4220.0 * t2) + 0.24 * np.sin(2.0 * math.pi * 7140.0 * t2))
    latch2 *= np.exp(-t2 * 96.0) * m2 * 0.65

    mono = np.tanh((glove_thump + leather + latch1 + snap1 + latch2) * 1.05).astype(np.float32) * 0.62
    mono *= make_env(n, 0.002, 0.018)
    return add_room_reflections(pan_stereo(mono, pan), taps_ms=(7.0, 15.0), gains=(0.15, 0.08))


def _sfx_handle_yank_snap(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Taut steel cable yank (0..0.18s) -> violent firing-sear pin SNAP (0.16s) -> cartridge hiss."""
    dur = 0.46
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # 1. Taut steel lanyard ripping through fairlead (0..0.19s)
    yank_env = np.where(
        t < 0.20,
        np.maximum(0.0, np.sin(math.pi * np.clip(t, 0.0, 0.20) / 0.20)) ** 0.7,
        0.0,
    ).astype(np.float32)
    cable_f = 620.0 + 1350.0 * np.clip(t / 0.18, 0.0, 1.0)
    cable_whine = np.sin(phase_integral(cable_f)).astype(np.float32) * yank_env * 0.24
    cable_scrape = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1300.0, 6200.0), order=2)
    cable_scrape *= yank_env * 0.54

    # 2. Violent sear release & initiator firing pin SNAP at 0.165s
    t_snap = np.maximum(0.0, t - 0.165)
    m_snap = (t >= 0.165).astype(np.float32)
    steel_punch = np.sin(phase_integral(225.0 * np.exp(-t_snap * 40.0) + 74.0)) * np.exp(-t_snap * 32.0) * m_snap * 0.70
    ping = (
        0.58 * np.sin(2.0 * math.pi * 2980.0 * t_snap)
        + 0.40 * np.sin(2.0 * math.pi * 4890.0 * t_snap)
        + 0.26 * np.sin(2.0 * math.pi * 7820.0 * t_snap)
    ) * np.exp(-t_snap * 28.0) * m_snap * 0.58
    crack = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (2200.0, 13000.0), order=2)
    crack *= np.exp(-t_snap * 64.0) * m_snap * 1.15

    # 3. Initiator gas charge micro-hiss tail
    hiss = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (3400.0, 10500.0), order=2)
    hiss *= np.exp(-t_snap * 14.0) * m_snap * 0.30

    mono = np.tanh((cable_whine + cable_scrape + steel_punch + ping + crack + hiss) * 0.98).astype(np.float32) * 0.74
    mono *= make_env(n, 0.003, 0.025)
    return add_room_reflections(pan_sweep_stereo(mono, pan - 0.08, pan + 0.14), taps_ms=(8.0, 16.0), gains=(0.18, 0.09))


def _sfx_brace_thump(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Helmet & shoulders slamming back into padded headbox + inertia harness lock."""
    dur = 0.42
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # Dense padded headbox / chest impact
    thud = np.sin(phase_integral(142.0 * np.exp(-t * 26.0) + 46.0)) * np.exp(-t * 17.5) * 0.85
    shell = np.sin(2.0 * math.pi * 285.0 * t) * np.exp(-t * 42.0) * 0.28
    pad_hit = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (190.0, 1350.0), order=2)
    pad_hit *= np.exp(-t * 28.0) * 0.62

    # Shoulder harness nylon cinch + 3 inertia-reel ratchet clicks
    harness = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (550.0, 2900.0), order=2)
    harness *= np.exp(-((t - 0.11) / 0.075) ** 2) * 0.34
    ratchet = np.zeros(n, dtype=np.float32)
    for rt in (0.055, 0.082, 0.110):
        tr = np.maximum(0.0, t - rt)
        mr = (t >= rt).astype(np.float32)
        ratchet += np.sin(2.0 * math.pi * 3450.0 * tr) * np.exp(-tr * 140.0) * mr * 0.22

    mono = np.tanh((thud + shell + pad_hit + harness + ratchet) * 0.95).astype(np.float32) * 0.64
    mono *= make_env(n, 0.002, 0.025)
    return add_room_reflections(pan_stereo(mono, pan), taps_ms=(8.0, 17.0), gains=(0.14, 0.07))


def _sfx_mdc_cord_spark(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """High-energy MDC plasma crackle racing zig-zag across canopy glass into detonation."""
    dur = 0.36
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    prog = np.clip(t / 0.30, 0.0, 1.0)

    # Rising ionized plasma arc hum (220 -> 760 Hz) + dense micro-arc impulses
    arc_f = 220.0 + 540.0 * (prog ** 1.4)
    arc_hum = (
        0.55 * np.sin(phase_integral(arc_f))
        + 0.30 * np.sin(phase_integral(arc_f * 2.02))
    ).astype(np.float32) * (0.35 + 0.65 * prog)

    impulses = (rng.random(n) > (0.986 - 0.012 * prog)).astype(np.float32) * rng.uniform(0.5, 1.6, n).astype(np.float32)
    sparks = butter_filter(impulses, "band", (2600.0, 14000.0), order=2) * 2.6
    plasma_sizz = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (3200.0, 12800.0), order=2)
    plasma_sizz *= (0.30 + 0.70 * prog) * 0.48

    # Zippy ramp into 0.30s detonation handoff
    env = np.minimum(1.0, t / 0.012) * np.where(t <= 0.30, 0.45 + 0.55 * prog, np.exp(-(t - 0.30) * 42.0))
    mono = np.tanh((arc_hum * 0.42 + sparks * 0.55 + plasma_sizz) * env).astype(np.float32) * 0.68

    # Zigzag spatial sweep along canopy frame
    zig_pan = np.clip(pan + 0.68 * np.sin(2.0 * math.pi * 14.0 * t), -0.92, 0.92).astype(np.float32)
    ang = (zig_pan + 1.0) * (math.pi / 4.0)
    stereo = np.stack([mono * np.cos(ang), mono * np.sin(ang)], axis=-1).astype(np.float32)
    return (stereo * make_env(n, 0.002, 0.02)[:, None]).astype(np.float32)


def _sfx_canopy_shatter_slowmo(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """MDC detonation crack + bullet-time sub-bass freeze + crystalline acrylic shard cascade."""
    dur = 2.85
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # 1. Razor-sharp initial MDC explosive crack + sub-bass bullet-time drop
    crack = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (450.0, 12000.0), order=2)
    crack *= np.exp(-t * 28.0) * 1.15
    f_sub = 145.0 * np.exp(-t * 2.1) + 31.0
    sub_drop = np.sin(phase_integral(f_sub)).astype(np.float32) * np.minimum(1.0, t / 0.008) * np.exp(-t * 1.25) * 0.92

    # 2. Time-dilated supersonic air rush around breached cockpit
    air_rush = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (130.0, 1150.0), order=2)
    air_rush *= np.minimum(1.0, t / 0.03) * np.exp(-t * 1.15) * 0.52

    base = np.tanh((crack + sub_drop + air_rush) * 0.96).astype(np.float32) * 0.84
    stereo = add_room_reflections(pan_stereo(base, pan), taps_ms=(14.0, 26.0, 41.0), gains=(0.20, 0.12, 0.06))

    # 3. 16 staggered crystalline polycarbonate/glass shard tinkles drifting outward in slow-mo
    n_shards = 16
    for k in range(n_shards):
        t_sh = 0.015 + (k ** 1.25) * 0.072
        if t_sh >= dur - 0.15:
            break
        s_dur = float(rng.uniform(0.32, 0.68))
        sn = int(s_dur * SR)
        st = np.arange(sn, dtype=np.float32) / SR
        f_base = float(rng.uniform(2400.0, 7200.0)) * (1.0 - 0.18 * (st / s_dur))  # slight slow-mo downward drift
        sh_sig = (
            0.52 * np.sin(phase_integral(f_base))
            + 0.32 * np.sin(phase_integral(f_base * 1.47))
            + 0.18 * np.sin(phase_integral(f_base * 2.19))
        ).astype(np.float32) * np.exp(-st * float(rng.uniform(5.5, 10.5))) * np.minimum(1.0, st / 0.002)
        hi_spark = butter_filter(rng.standard_normal(sn).astype(np.float32), "band", (4200.0, 13500.0), order=2)
        hi_spark *= np.exp(-st * 16.0) * 0.25
        p0 = float(rng.uniform(-0.45, 0.45))
        p1 = float(np.clip(p0 + rng.choice([-1.0, 1.0]) * rng.uniform(0.25, 0.50), -0.92, 0.92))
        amp = 0.22 * math.exp(-t_sh * 0.55)
        add_stereo_event(stereo, pan_sweep_stereo((sh_sig + hi_spark).astype(np.float32) * amp, p0, p1), t_sh)

    return (stereo * make_env(n, 0.002, 0.12)[:, None]).astype(np.float32)


def _sfx_xray_scan_whoosh(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Futuristic high-tech cyan medical/engineering X-ray laser sweep."""
    dur = 0.95
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    prog = t / dur
    bell = np.maximum(0.0, np.sin(math.pi * prog)) ** 1.3

    # Airy glassy laser sweep noise + resonant upward sweep
    air = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (550.0, 5800.0), order=2)
    f_sweep = 360.0 + 1320.0 * (prog ** 1.15)
    am_shimmer = 0.78 + 0.22 * np.sin(2.0 * math.pi * 28.0 * t)
    cyan_tone = (
        0.48 * np.sin(phase_integral(f_sweep))
        + 0.30 * np.sin(phase_integral(f_sweep * 1.498))   # clean fifth shimmer
        + 0.18 * np.sin(phase_integral(f_sweep * 2.0))
    ).astype(np.float32) * am_shimmer

    # Subtle medical imaging telemetry chime peak near center of sweep
    chime_env = np.exp(-((t - 0.46) / 0.16) ** 2).astype(np.float32)
    chime = (
        0.45 * np.sin(2.0 * math.pi * 1174.66 * t)  # D6
        + 0.25 * np.sin(2.0 * math.pi * 1760.00 * t)  # A6
    ) * chime_env * 0.24

    mono = ((air * 0.44 + cyan_tone * 0.34) * bell + chime).astype(np.float32) * 0.56
    p0 = float(np.clip(pan - 0.58, -0.92, 0.92))
    p1 = float(np.clip(pan + 0.58, -0.92, 0.92))
    return (pan_sweep_stereo(mono, p0, p1) * make_env(n, 0.015, 0.06)[:, None]).astype(np.float32)


def _sfx_garter_cable_snap(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Taut Kevlar leg-restraint reel whip (0..0.44s) + twin stereo boot ankle snaps."""
    dur = 0.70
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # 1. Rapid reel zip pulling ankles aft (0..0.44s)
    zip_env = np.where(
        t < 0.44,
        np.maximum(0.0, np.sin(math.pi * np.clip(t, 0.0, 0.44) / 0.44)) ** 0.75,
        0.0,
    ).astype(np.float32)
    reel_f = 680.0 + 2100.0 * np.clip(t / 0.42, 0.0, 1.0) ** 1.3
    reel_whip = np.sin(phase_integral(reel_f)).astype(np.float32) * zip_env * 0.24
    kevlar = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1350.0, 6800.0), order=2)
    kevlar *= zip_env * 0.48
    stereo = pan_stereo((reel_whip + kevlar).astype(np.float32), pan)

    # 2. Twin boot-heel impacts & stainless garter clamp snaps (L at 0.38s, R at 0.44s)
    for t_boot, b_pan, f_ring in [(0.38, -0.45, 3420.0), (0.44, 0.45, 3780.0)]:
        bn = int(0.24 * SR)
        bt = np.arange(bn, dtype=np.float32) / SR
        thud = np.sin(phase_integral(122.0 * np.exp(-bt * 24.0) + 52.0)).astype(np.float32) * np.exp(-bt * 20.0) * 0.65
        snap = butter_filter(rng.standard_normal(bn).astype(np.float32), "band", (1900.0, 9600.0), order=2)
        snap *= np.exp(-bt * 65.0) * 0.90
        ring = (
            0.48 * np.sin(2.0 * math.pi * f_ring * bt)
            + 0.28 * np.sin(2.0 * math.pi * (f_ring * 1.62) * bt)
        ) * np.exp(-bt * 44.0) * 0.35
        boot_ev = np.tanh((thud + snap + ring) * 0.90).astype(np.float32) * 0.48
        add_stereo_event(stereo, pan_stereo(boot_ev, float(np.clip(pan + b_pan, -0.9, 0.9))), t_boot)

    return (stereo * make_env(n, 0.004, 0.03)[:, None]).astype(np.float32)


def _sfx_catapult_fire_boom(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Solid ballistic catapult cartridge blast + pressurized steel tube thrust whoosh."""
    dur = 1.25
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # 1. Cartridge firing crack
    crack = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (420.0, 10800.0), order=2)
    crack *= np.exp(-t * 32.0) * 1.12

    # 2. Deep ballistic mortar punch (118 -> 33 Hz) + propellant gas rumble
    sub = np.sin(phase_integral(118.0 * np.exp(-t * 4.2) + 33.0)).astype(np.float32) * np.exp(-t * 2.4) * 0.95
    rumble = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (38.0, 290.0), order=3)
    rumble *= np.exp(-t * 2.2) * 0.88

    # 3. Pressurized telescoping tube upward whoosh
    tube_f = 340.0 + 980.0 * np.clip(t / 0.55, 0.0, 1.0)
    tube_tone = np.sin(phase_integral(tube_f)).astype(np.float32) * np.exp(-t * 4.6) * 0.22
    gas_jet = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (260.0, 3100.0), order=2)
    gas_jet *= np.minimum(1.0, t / 0.015) * np.exp(-t * 3.0) * 0.64

    mono = np.tanh((crack + sub + rumble + tube_tone + gas_jet) * 0.96).astype(np.float32) * 0.92
    stereo = add_room_reflections(pan_stereo(mono, pan), taps_ms=(11.0, 21.0, 34.0), gains=(0.18, 0.10, 0.05))
    return (stereo * make_env(n, 0.002, 0.08)[:, None]).astype(np.float32)


def _sfx_rail_slide_screech(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """High-G metallic steel-on-steel guide-rail glide & harmonic screech."""
    dur = 1.45
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    prog = t / dur
    env = (np.minimum(1.0, t / 0.08) * np.minimum(1.0, (dur - t) / 0.22) * (0.45 + 0.55 * np.sin(math.pi * prog))).astype(np.float32)

    # Stick-slip micro-jitter modulation
    jitter = 0.78 + 0.22 * np.sin(2.0 * math.pi * 38.0 * t + 1.5 * np.sin(2.0 * math.pi * 6.0 * t))

    def rail_voice(detune: float, seed_off: int) -> np.ndarray:
        local_rng = np.random.default_rng(7700 + seed_off)
        f1 = (720.0 + 540.0 * (prog ** 0.85)) * detune
        tones = (
            0.46 * np.sin(phase_integral(f1))
            + 0.30 * np.sin(phase_integral(f1 * 2.04))
            + 0.16 * np.sin(phase_integral(f1 * 3.12))
        ).astype(np.float32) * jitter
        grit = butter_filter(local_rng.standard_normal(n).astype(np.float32), "band", (1500.0, 5600.0), order=2)
        return ((tones * 0.36 + grit * 0.44) * env).astype(np.float32)

    left = rail_voice(0.985, 1)
    right = rail_voice(1.015, 2)
    stereo = np.stack([left, right], axis=-1).astype(np.float32)
    # Apply requested pan bias subtly while preserving twin-rail width
    if abs(pan) > 1e-3:
        bal = pan_stereo(np.ones(n, dtype=np.float32), pan) * math.sqrt(2.0)
        stereo *= bal
    return (np.tanh(stereo * 0.95) * 0.48 * make_env(n, 0.02, 0.08)[:, None]).astype(np.float32)


def _sfx_spine_compress_creak(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Visceral bone/cartilage hydraulic compression groans & disc squish (2.30s)."""
    dur = 2.30
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    prog = t / dur
    env = np.maximum(0.0, np.sin(math.pi * prog)) ** 0.85

    # Downward-pitching cartilaginous compression groan (125 -> 62 Hz + 240 -> 130 Hz)
    stick_slip = 0.65 + 0.35 * np.sin(2.0 * math.pi * (8.5 + 5.5 * prog) * t)
    f_low = 124.0 - 62.0 * (prog ** 0.8) + 12.0 * np.sin(2.0 * math.pi * 7.2 * t)
    groan = (
        0.58 * np.sin(phase_integral(f_low))
        + 0.32 * np.sin(phase_integral(f_low * 1.96))
        + 0.16 * np.sin(phase_integral(f_low * 3.10))
    ).astype(np.float32) * stick_slip

    # Wet viscous spinal disc compression squish
    squish_mod = 0.55 + 0.45 * np.sin(2.0 * math.pi * 3.2 * t) ** 2
    squish = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (280.0, 1250.0), order=2)
    squish *= squish_mod * 0.46

    # 6 subtle dry micro-creak clicks across compression stroke
    clicks = np.zeros(n, dtype=np.float32)
    for k, ct in enumerate([0.28, 0.56, 0.88, 1.18, 1.52, 1.84]):
        tc = np.maximum(0.0, t - ct)
        mc = (t >= ct).astype(np.float32)
        cf = 1280.0 + (k % 3) * 340.0
        clicks += (
            0.55 * np.sin(2.0 * math.pi * cf * tc)
            + butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1100.0, 3400.0), order=2) * 0.45
        ) * np.exp(-tc * 110.0) * mc * 0.34

    # Keep vocal pocket (800-3000 Hz) smooth so Beat-4 narration stays ultra-clear
    mono = peaking_eq(((groan * 0.48 + squish + clicks) * env).astype(np.float32), 1400.0, 0.7, -3.5)
    mono = np.tanh(mono * 1.05).astype(np.float32) * 0.56
    return (stereo_spatialize(mono, delay_ms=2.4, wet=0.25) * make_env(n, 0.03, 0.09)[:, None]).astype(np.float32)


def _sfx_g_load_pulse(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Low-frequency 16-G heartbeat sub thump (LUB-DUB) + muffled arterial blood-rush."""
    dur = 0.78
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # Primary LUB (0.00s) + secondary DUB (0.19s)
    lub = np.sin(phase_integral(82.0 * np.exp(-t * 14.0) + 34.0)).astype(np.float32) * np.exp(-t * 9.5) * 0.90
    t2 = np.maximum(0.0, t - 0.19)
    m2 = (t >= 0.19).astype(np.float32)
    dub = np.sin(phase_integral(70.0 * np.exp(-t2 * 16.0) + 31.0)).astype(np.float32) * np.exp(-t2 * 11.0) * m2 * 0.72

    # Muffled 16G blood-rush pressure wave in pilot's ears
    rush = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (75.0, 380.0), order=2)
    rush *= (np.exp(-((t - 0.12) / 0.16) ** 2) + 0.55 * np.exp(-((t - 0.32) / 0.18) ** 2)) * 0.42

    mono = np.tanh((lub + dub + rush) * 0.95).astype(np.float32) * 0.68
    return (pan_stereo(mono, pan) * make_env(n, 0.004, 0.06)[:, None]).astype(np.float32)


def _sfx_rocket_ignite_roar(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Sustained twin solid-rocket motor roar (20.83s -> 24.33s burnout, 3.50s)."""
    dur = 3.50
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # Ignition squib pop + pressure surge (0..0.14s)
    ign_crack = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (320.0, 9200.0), order=2)
    ign_crack *= np.exp(-t * 24.0) * 0.95
    ign_sub = np.sin(phase_integral(128.0 * np.exp(-t * 7.5) + 40.0)).astype(np.float32) * np.exp(-t * 5.2) * 0.85

    # Sustained thrust envelope: fierce ignition -> steady burn -> sputter burnout around 3.15..3.50s
    burn_env = (
        np.minimum(1.0, t / 0.045)
        * np.where(t < 3.05, 1.0, np.exp(-((np.maximum(0.0, t - 3.05)) / 0.24) ** 1.6))
        * make_env(n, 0.01, 0.10)
    ).astype(np.float32)

    # Twin nozzles (L and R slightly asymmetric turbulence & flutter for wide rocket power)
    def nozzle_stream(seed_off: int, f_res: float) -> np.ndarray:
        lrng = np.random.default_rng(9100 + seed_off)
        lo_roar = butter_filter(lrng.standard_normal(n).astype(np.float32), "band", (45.0, 260.0), order=3) * 0.82
        mid_jet = butter_filter(lrng.standard_normal(n).astype(np.float32), "band", (260.0, 1750.0), order=2) * 0.62
        crackle = butter_filter(lrng.standard_normal(n).astype(np.float32), "band", (2200.0, 7800.0), order=2)
        flutter = 0.72 + 0.28 * np.sin(2.0 * math.pi * 19.0 * t + 0.8 * np.sin(2.0 * math.pi * 4.3 * t))
        motor_tone = np.sin(2.0 * math.pi * f_res * t) * 0.14
        return ((lo_roar + mid_jet + crackle * 0.36 + motor_tone) * flutter * burn_env).astype(np.float32)

    left = nozzle_stream(1, 182.0) + (ign_crack + ign_sub) * 0.55
    right = nozzle_stream(2, 196.0) + (ign_crack + ign_sub) * 0.55
    stereo = np.stack([left, right], axis=-1).astype(np.float32)
    stereo = peaking_eq(stereo, 1250.0, 0.65, -2.5)  # protect vocal legibility during Beat 5
    stereo = np.tanh(stereo * 0.82).astype(np.float32) * 0.68
    if abs(pan) > 1e-3:
        stereo *= pan_stereo(np.ones(n, dtype=np.float32), pan) * math.sqrt(2.0)
    return stereo.astype(np.float32)


def _sfx_tail_flyby_whoosh(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Doppler-shifted fighter jet airframe & vertical tail fin flyby sweep L -> R."""
    dur = 1.05
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    t_peak = 0.28  # passes close beneath seat shortly after cue
    env = (np.exp(-((t - t_peak) / 0.21) ** 2) * np.minimum(1.0, t / 0.025)).astype(np.float32)

    # Classic Doppler jet turbine pitch drop (1620 Hz -> 520 Hz across pass point)
    doppler = 0.5 * (1.0 - np.tanh((t - t_peak) / 0.13))
    f_jet = 520.0 + 1100.0 * doppler
    turbine = (
        0.52 * np.sin(phase_integral(f_jet))
        + 0.28 * np.sin(phase_integral(f_jet * 1.98))
    ).astype(np.float32) * 0.30

    wake_hi = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (340.0, 4400.0), order=2) * 0.68
    wake_lo = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (70.0, 320.0), order=2) * 0.62

    mono = np.tanh((turbine + wake_hi + wake_lo) * env * 1.05).astype(np.float32) * 0.68
    # Sweep L -> R centered squarely at t_peak so jet crosses past the viewer at peak energy
    sweep_p = np.clip(pan + 0.80 * np.tanh((t - t_peak) / 0.22), -0.95, 0.95).astype(np.float32)
    ang = (sweep_p + 1.0) * (math.pi / 4.0)
    stereo = np.stack([mono * np.cos(ang), mono * np.sin(ang)], axis=-1).astype(np.float32)
    return (stereo * make_env(n, 0.01, 0.06)[:, None]).astype(np.float32)


def _sfx_drogue_mortar_pop(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Headbox drogue mortar thump + whistling bridle zip + small canopy snap."""
    dur = 0.85
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # 1. Mortar tube pop (0.00s)
    pop_sub = np.sin(phase_integral(190.0 * np.exp(-t * 28.0) + 52.0)).astype(np.float32) * np.exp(-t * 18.0) * 0.78
    pop_air = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (580.0, 6200.0), order=2)
    pop_air *= np.exp(-t * 36.0) * 0.88

    # 2. Bridle zip whistle (0.04..0.52s)
    t_zip = np.maximum(0.0, t - 0.04)
    m_zip = ((t >= 0.04) & (t < 0.54)).astype(np.float32)
    zip_f = 920.0 + 1350.0 * np.clip(t_zip / 0.45, 0.0, 1.0)
    zip_sig = np.sin(phase_integral(zip_f)).astype(np.float32) * np.sin(math.pi * np.clip(t_zip / 0.50, 0.0, 1.0)) * m_zip * 0.16

    # 3. Drogue deceleration fabric snap at 0.52s
    t_sn = np.maximum(0.0, t - 0.52)
    m_sn = (t >= 0.52).astype(np.float32)
    snap_air = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (360.0, 4800.0), order=2)
    snap_air *= np.exp(-t_sn * 14.0) * m_sn * 0.62
    snap_tug = np.sin(phase_integral(98.0 * np.exp(-t_sn * 20.0) + 46.0)).astype(np.float32) * np.exp(-t_sn * 16.0) * m_sn * 0.54

    mono = np.tanh((pop_sub + pop_air + zip_sig + snap_air + snap_tug) * 0.92).astype(np.float32) * 0.68
    return (pan_stereo(mono, pan) * make_env(n, 0.002, 0.04)[:, None]).astype(np.float32)


def _sfx_separation_latch_clunk(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Pyrotechnic seat-man separation shackle CLACK-CLUNK + strap spring recoil."""
    dur = 0.45
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # Heavy shackle release double-clack (0.00s + 0.042s)
    clunk1 = np.sin(phase_integral(162.0 * np.exp(-t * 32.0) + 56.0)).astype(np.float32) * np.exp(-t * 28.0) * 0.72
    metal1 = (
        0.48 * np.sin(2.0 * math.pi * 2050.0 * t)
        + 0.30 * np.sin(2.0 * math.pi * 3420.0 * t)
        + 0.18 * np.sin(2.0 * math.pi * 5560.0 * t)
    ) * np.exp(-t * 42.0) * 0.45
    hit1 = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1400.0, 9200.0), order=2)
    hit1 *= np.exp(-t * 62.0) * 0.85

    t2 = np.maximum(0.0, t - 0.042)
    m2 = (t >= 0.042).astype(np.float32)
    clunk2 = np.sin(2.0 * math.pi * 2520.0 * t2) * np.exp(-t2 * 48.0) * m2 * 0.36
    recoil = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (850.0, 5200.0), order=2)
    recoil *= np.exp(-((t - 0.12) / 0.07) ** 2) * 0.38

    mono = np.tanh((clunk1 + metal1 + hit1 + clunk2 + recoil) * 0.96).astype(np.float32) * 0.66
    p1 = float(np.clip(pan + 0.24, -0.92, 0.92))
    return add_room_reflections(pan_sweep_stereo(mono, pan, p1), taps_ms=(9.0, 19.0), gains=(0.14, 0.07))


def _sfx_main_chute_blossom(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """Riser line whip + massive ripstop fabric canopy air-gulp CRACK and billow (1.45s)."""
    dur = 1.45
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # 1. Nylon suspension lines whipping taut (0..0.28s)
    line_env = np.exp(-((t - 0.09) / 0.07) ** 2).astype(np.float32)
    line_whip = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1100.0, 6800.0), order=2)
    line_whip *= line_env * 0.56

    # 2. Massive canopy air-capture CRACK + booming inflation pressure wave (at 0.24s)
    tb = np.maximum(0.0, t - 0.24)
    mb = (t >= 0.24).astype(np.float32)
    air_boom = np.sin(phase_integral(96.0 * np.exp(-tb * 5.2) + 32.0)).astype(np.float32) * np.exp(-tb * 2.8) * mb * 0.88
    sail_crack = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (260.0, 4200.0), order=2)
    sail_crack *= np.exp(-tb * 11.0) * mb * 0.95

    # 3. Wide stereo billowing ripstop nylon flutter (0.28..1.40s)
    billow_env = np.minimum(1.0, tb / 0.09) * np.exp(-tb * 1.95) * mb

    def sail_side(seed_off: int, flutter_hz: float) -> np.ndarray:
        lrng = np.random.default_rng(5100 + seed_off)
        nylon = butter_filter(lrng.standard_normal(n).astype(np.float32), "band", (160.0, 2600.0), order=2)
        flutter = 0.65 + 0.35 * np.sin(2.0 * math.pi * flutter_hz * tb)
        return (nylon * flutter * billow_env * 0.72).astype(np.float32)

    base = np.tanh((line_whip + air_boom + sail_crack) * 1.12).astype(np.float32) * 0.92
    stereo = pan_stereo(base, pan)
    stereo[:, 0] += sail_side(1, 6.2) * 1.25
    stereo[:, 1] += sail_side(2, 7.1) * 1.25
    return (np.tanh(stereo * 1.02) * 0.90 * make_env(n, 0.005, 0.14)[:, None]).astype(np.float32)


def _sfx_wind_canopy_outro(rng: np.random.Generator, pan: float, _t0: float) -> np.ndarray:
    """High-altitude serene canopy wind glide & soft harness sway (27.50 -> 30.00s)."""
    dur = 2.48
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    env = np.minimum(1.0, t / 0.25) * np.minimum(1.0, (dur - t) / 0.45)

    def breeze(seed_off: int, mod_hz: float, phase_off: float) -> np.ndarray:
        lrng = np.random.default_rng(6200 + seed_off)
        air = butter_filter(lrng.standard_normal(n).astype(np.float32), "band", (180.0, 2200.0), order=2)
        sail = butter_filter(lrng.standard_normal(n).astype(np.float32), "band", (95.0, 340.0), order=2)
        sway = 0.72 + 0.28 * np.sin(2.0 * math.pi * mod_hz * t + phase_off)
        return ((air * 0.42 + sail * 0.48) * sway * env).astype(np.float32)

    stereo = np.stack([breeze(1, 0.42, 0.0), breeze(2, 0.38, 1.3)], axis=-1).astype(np.float32) * 0.28
    # Gentle parachute riser webbing micro-creak at 0.85s
    tc = np.maximum(0.0, t - 0.85)
    mc = (t >= 0.85).astype(np.float32)
    creak = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (650.0, 1900.0), order=2)
    creak *= np.exp(-((tc - 0.12) / 0.08) ** 2) * mc * 0.08
    stereo += pan_stereo(creak, float(np.clip(pan - 0.15, -0.8, 0.8)))
    return stereo.astype(np.float32)


SFX_REGISTRY: dict[str, Callable[[np.random.Generator, float, float], np.ndarray]] = {
    "cockpit_alarm": _sfx_cockpit_alarm,
    "handle_grip_click": _sfx_handle_grip_click,
    "handle_yank_snap": _sfx_handle_yank_snap,
    "brace_thump": _sfx_brace_thump,
    "mdc_cord_spark": _sfx_mdc_cord_spark,
    "canopy_shatter_slowmo": _sfx_canopy_shatter_slowmo,
    "xray_scan_whoosh": _sfx_xray_scan_whoosh,
    "garter_cable_snap": _sfx_garter_cable_snap,
    "catapult_fire_boom": _sfx_catapult_fire_boom,
    "rail_slide_screech": _sfx_rail_slide_screech,
    "spine_compress_creak": _sfx_spine_compress_creak,
    "g_load_pulse": _sfx_g_load_pulse,
    "rocket_ignite_roar": _sfx_rocket_ignite_roar,
    "tail_flyby_whoosh": _sfx_tail_flyby_whoosh,
    "drogue_mortar_pop": _sfx_drogue_mortar_pop,
    "separation_latch_clunk": _sfx_separation_latch_clunk,
    "main_chute_blossom": _sfx_main_chute_blossom,
    "wind_canopy_outro": _sfx_wind_canopy_outro,
}


def synth_sfx(kind: str, pan: float, event_t: float) -> np.ndarray:
    """Dispatch and synthesize a single Foley event with deterministic per-cue seed."""
    if kind not in SFX_REGISTRY:
        raise ValueError(f"Unknown SFX type in timeline.json: {kind!r}")
    seed = int(round(event_t * 1000)) + (sum(ord(ch) for ch in kind) * 131)
    rng = np.random.default_rng(seed)
    return SFX_REGISTRY[kind](rng, pan, event_t)


# ---------------------------------------------------------------------------
# Continuous Ambient Bed (Mach-1 Cockpit Jet Hum -> Open-Air Rushing Wind)
# ---------------------------------------------------------------------------
def build_ambient_bed() -> np.ndarray:
    """Subtle continuous jet cockpit hum (B1-B2) -> open-air rushing wind (B5-B6).

    Strictly sculpted below narration levels (-28 to -36 dBFS) with a gentle
    vocal-pocket dip around 1100 Hz so en-US-BrianNeural stays razor-crisp.
    """
    n = TOTAL_SAMPLES
    t = np.arange(n, dtype=np.float32) / SR
    rng_l = np.random.default_rng(4041)
    rng_r = np.random.default_rng(4042)

    raw_l = rng_l.standard_normal(n).astype(np.float32)
    raw_r = rng_r.standard_normal(n).astype(np.float32)

    # 1. Closed-cockpit Mach-1 jet engine hum + ECS airflow (Beats 1-2, dropping at MDC shatter 5.00s)
    t_mdc = ev_s("mdc_detonate")
    b2_end = beat_span(2)[1]
    cockpit_env = np.minimum(1.0, t / 0.15) * (1.0 - 0.65 * (1.0 / (1.0 + np.exp(-(t - t_mdc) / 0.12))))
    cockpit_env *= np.where(t <= b2_end, 1.0, np.exp(-np.maximum(0.0, t - b2_end) * 2.2))

    jet_hum_l = butter_filter(raw_l, "band", (55.0, 320.0), order=2) * 0.65
    jet_hum_r = butter_filter(raw_r, "band", (55.0, 320.0), order=2) * 0.65
    turbine_whine = (
        0.55 * np.sin(2.0 * math.pi * 92.0 * t)
        + 0.35 * np.sin(2.0 * math.pi * 184.0 * t)
        + 0.12 * np.sin(2.0 * math.pi * 1380.0 * t) * np.where(t < t_mdc, 1.0, 0.15)
    ).astype(np.float32) * 0.18
    ecs_air_l = butter_filter(raw_l, "band", (420.0, 2400.0), order=2) * 0.24
    ecs_air_r = butter_filter(raw_r, "band", (420.0, 2400.0), order=2) * 0.24

    cockpit_bed = np.stack(
        [
            (jet_hum_l + ecs_air_l + turbine_whine) * cockpit_env,
            (jet_hum_r + ecs_air_r + turbine_whine) * cockpit_env,
        ],
        axis=-1,
    ).astype(np.float32) * 0.120

    # 2. Bullet-time slipstream & open-air high-altitude wind (swells in B5-B6)
    b5_s, _ = beat_span(5)
    b6_e = DURATION_S
    wind_env = (
        0.16 * np.exp(-((t - 7.4) / 2.2) ** 2)                  # B2 bullet-time open canopy airflow
        + 0.28 * np.exp(-((t - 12.6) / 1.8) ** 2)                # B3 rail slipstream
        + 0.12 * np.exp(-((t - 17.5) / 2.4) ** 2)                # B4 subtle high-G tremor air
        + 0.85 * np.where(t >= b5_s - 0.4, np.minimum(1.0, (t - (b5_s - 0.4)) / 0.85), 0.0)
    ).astype(np.float32)

    # Gentle dip across 25.44-25.50s breath cue and smooth taper to 30.00s
    breath_dip = 1.0 - 0.55 * np.exp(-((t - 25.46) / 0.06) ** 2)
    outro_fade = np.where(t <= 27.6, 1.0, np.maximum(0.0, (b6_e - t) / (b6_e - 27.6)) ** 0.7)
    wind_env *= (breath_dip * outro_fade).astype(np.float32)

    swirl_l = 0.76 + 0.24 * np.sin(2.0 * math.pi * 0.45 * t)
    swirl_r = 0.76 + 0.24 * np.sin(2.0 * math.pi * 0.41 * t + 1.4)
    wind_l = butter_filter(raw_l, "band", (130.0, 2100.0), order=2) * swirl_l
    wind_r = butter_filter(raw_r, "band", (130.0, 2100.0), order=2) * swirl_r
    wind_bed = np.stack([wind_l * wind_env, wind_r * wind_env], axis=-1).astype(np.float32) * 0.105

    bed = cockpit_bed + wind_bed
    bed = peaking_eq(bed, 1150.0, 0.55, -3.5)
    return (bed * make_env(n, 0.02, 0.08)[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# Assembly, Verification & Export
# ---------------------------------------------------------------------------
def build_sfx_stem() -> tuple[np.ndarray, list[dict]]:
    """Synthesize all timeline.json SFX cues + ambient airflow bed into 48kHz stereo stem."""
    tl = json.loads(TIMELINE_PATH.read_text(encoding="utf-8"))
    cues: list[dict] = tl["sfx"]

    bus = build_ambient_bed()
    for ev in cues:
        ev_type = str(ev["type"])
        ev_time = float(ev["t"])
        ev_pan = float(ev.get("pan", 0.0))
        add_stereo_event(bus, synth_sfx(ev_type, ev_pan, ev_time), ev_time)

    # Remove DC / subsonic mud below 26 Hz and guarantee clean peak < 0.96
    bus = butter_filter(bus, "high", 26.0, order=2)
    bus = soft_limit_peak(bus, ceiling=0.90)

    # Click-free 5ms fade boundaries
    micro = int(round(0.005 * SR))
    bus[:micro] *= np.linspace(0.0, 1.0, micro, dtype=np.float32)[:, None]
    bus[-micro:] *= np.linspace(1.0, 0.0, micro, dtype=np.float32)[:, None]
    return bus.astype(np.float32), cues


def verify_sfx_stem(path: Path, audio: np.ndarray, cues: list[dict]) -> None:
    """Validate format, exact 30.000s sample count, zero NaNs, peak < 0.96, and per-beat energy."""
    assert np.isfinite(audio).all(), "SFX stem contains NaN or Inf values"

    info = sf.info(str(path))
    assert info.samplerate == SR, f"Expected SR={SR}, got {info.samplerate}"
    assert info.channels == CHANNELS, f"Expected {CHANNELS}ch, got {info.channels}"
    assert info.frames == TOTAL_SAMPLES, f"Expected {TOTAL_SAMPLES} samples, got {info.frames}"
    assert abs(info.duration - DURATION_S) < 1e-6, f"Expected {DURATION_S}s, got {info.duration}"

    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == SR
        assert wf.getnchannels() == CHANNELS
        assert wf.getnframes() == TOTAL_SAMPLES

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    assert 0.01 < rms < 0.50, f"Unexpected SFX RMS: {rms:.5f}"
    assert peak < 0.96, f"Peak exceeded < 0.96 limit: {peak:.5f}"

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
