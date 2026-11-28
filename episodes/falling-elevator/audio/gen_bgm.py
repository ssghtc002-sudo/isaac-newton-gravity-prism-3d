"""
BGM Composer for Episode #16 ("Why Jumping in a Falling Elevator WON'T Save You", 30.0 s @ 60 fps).

Generates:
    episodes/falling-elevator/audio/output/bgm_stem.wav
    - 48000 Hz, Stereo, PCM_16, exact 30.000 s (1,440,000 samples)
    - Synced strictly to elevator_contract.BEATS and elevator_contract.EV
    - Permanent -3.5 dB vocal-pocket EQ notch across 250-3500 Hz for en-US-BrianNeural clarity
    - Clean 45 ms breath gap (25.455-25.500 s) right before Beat 6 heroic F#maj7 -> Dbmaj7 uplift
    - Peak normalized cleanly (~0.80, <= 0.82), zero clipping
"""
from __future__ import annotations

import math
import sys
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

AUDIO_DIR = Path(__file__).resolve().parent
EPISODE_ROOT = AUDIO_DIR.parent
SRC_DIR = EPISODE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import elevator_contract as C  # noqa: E402

SR = 48000
CHANNELS = 2
DURATION_S = float(C.DURATION_S)
TOTAL_SAMPLES = int(round(DURATION_S * SR))

BPM = 120.0
BEAT_S = 60.0 / BPM  # 0.500 s

# Exact 45 ms breath gap right before Beat 6 (Beat 6 starts at frame 1531 -> 25.500 s)
BREATH_DUR_S = 0.045
BREATH_END_S = (C.BEATS[6][0] - 1) / float(C.FPS)  # 25.500 s
BREATH_START_S = BREATH_END_S - BREATH_DUR_S       # 25.455 s

OUTPUT_PATH = AUDIO_DIR / "output" / "bgm_stem.wav"


def beat_span(beat_idx: int) -> tuple[float, float]:
    """Return (start_s, end_s) for beat_idx from elevator_contract.BEATS."""
    f0, f1 = C.BEATS[beat_idx]
    return ((f0 - 1) / float(C.FPS), f1 / float(C.FPS))


def ev_s(event_key: str) -> float:
    """Return event timestamp in seconds from elevator_contract.EV."""
    return C.EV[event_key] / float(C.FPS)


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def pan_stereo(mono: np.ndarray, pan: float) -> np.ndarray:
    """Constant-power stereo panner (-1.0 left .. +1.0 right)."""
    pan = float(np.clip(pan, -1.0, 1.0))
    angle = (pan + 1.0) * (math.pi / 4.0)
    return np.stack([mono * math.cos(angle), mono * math.sin(angle)], axis=-1).astype(np.float32)


def add_stereo_event(bus: np.ndarray, event: np.ndarray, start_time: float) -> None:
    """Mix a mono or stereo event into the stereo bus at start_time (seconds)."""
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
    """RBJ peaking EQ biquad filter (works on 1D or 2D audio along axis=0)."""
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / SR
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    norm = 1.0 + alpha / a_lin
    b = np.array([1.0 + alpha * a_lin, -2.0 * cw, 1.0 - alpha * a_lin], dtype=np.float64) / norm
    a = np.array([1.0, (-2.0 * cw) / norm, (1.0 - alpha / a_lin) / norm], dtype=np.float64)
    return signal.lfilter(b, a, audio.astype(np.float64), axis=0).astype(np.float32)


def band_limited_saw(
    freq: float | np.ndarray,
    t: np.ndarray,
    max_harmonics: int = 16,
    rolloff: float = 0.85,
) -> np.ndarray:
    """Warm additive Fourier sawtooth clamped safely below Nyquist (zero aliasing)."""
    f_ref = float(np.max(freq)) if isinstance(freq, np.ndarray) else float(freq)
    if f_ref <= 1e-3:
        return np.zeros_like(t, dtype=np.float32)
    n_harm = max(1, min(max_harmonics, int(0.42 * SR / f_ref)))
    if isinstance(freq, np.ndarray):
        phase = 2.0 * math.pi * np.cumsum(freq.astype(np.float64)) / SR
    else:
        phase = 2.0 * math.pi * float(freq) * t.astype(np.float64)
    out = np.zeros(len(t), dtype=np.float64)
    for k in range(1, n_harm + 1):
        amp = (1.0 / (k ** rolloff)) * (1.0 if k % 2 == 1 else 0.88)
        out += amp * np.sin(k * phase)
    out *= 0.62
    return out.astype(np.float32)


def make_env(n: int, attack_s: float, release_s: float) -> np.ndarray:
    """Smooth fade-in/out envelope for a buffer of length n."""
    t = np.arange(n, dtype=np.float32) / SR
    dur = n / float(SR)
    att = np.minimum(1.0, t / max(1e-4, attack_s))
    rel = np.minimum(1.0, (dur - t) / max(1e-4, release_s))
    return (att * rel).astype(np.float32)


def stereo_ensemble_pad(
    t: np.ndarray,
    midi_notes: list[float],
    weights: list[float],
    detune_cents: float = 8.5,
    max_harmonics: int = 12,
    rolloff: float = 0.95,
    vibrato_hz: float = 0.0,
    vibrato_cents: float = 0.0,
) -> np.ndarray:
    """Detuned L/R lush analog brass/string ensemble pad across a list of MIDI notes."""
    ratio_up = 2.0 ** (detune_cents / 1200.0)
    ratio_dn = 2.0 ** (-detune_cents / 1200.0)
    vib_mod = 1.0
    if vibrato_hz > 0.0 and vibrato_cents > 0.0:
        vib_mod = 2.0 ** ((vibrato_cents * np.sin(2.0 * math.pi * vibrato_hz * t)) / 1200.0)

    left = np.zeros(len(t), dtype=np.float32)
    right = np.zeros(len(t), dtype=np.float32)
    for i, (m, w) in enumerate(zip(midi_notes, weights)):
        f0 = midi_to_hz(m) * vib_mod
        f_l = f0 * (ratio_up if i % 2 == 0 else ratio_dn)
        f_r = f0 * (ratio_dn if i % 2 == 0 else ratio_up)
        left += w * band_limited_saw(f_l, t, max_harmonics=max_harmonics, rolloff=rolloff)
        right += w * band_limited_saw(f_r, t, max_harmonics=max_harmonics, rolloff=rolloff)
    return np.stack([left, right], axis=-1).astype(np.float32)


def synth_kick(amp: float = 0.24, punch_hz: float = 96.0, tail_hz: float = 40.0, decay: float = 14.0) -> np.ndarray:
    """Tight thriller cinema sub-kick."""
    n = int(0.22 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    f_inst = punch_hz * np.exp(-t * 26.0) + tail_hz
    phase = 2.0 * math.pi * np.cumsum(f_inst.astype(np.float64)) / SR
    click = np.sin(2.0 * math.pi * 230.0 * t) * np.exp(-t * 82.0) * 0.20
    body = np.sin(phase).astype(np.float32) * np.exp(-t * decay)
    sig = np.tanh((body + click) * 1.15) * make_env(n, 0.0015, 0.035) * amp
    return pan_stereo(sig.astype(np.float32), 0.0)


def synth_sub_boom(amp: float = 0.34, start_hz: float = 84.0, end_hz: float = 34.0, dur_s: float = 1.25) -> np.ndarray:
    """Deep cinematic sub-boom impact for cable snap / pit crash."""
    n = int(round(dur_s * SR))
    t = np.arange(n, dtype=np.float32) / SR
    f_inst = end_hz + (start_hz - end_hz) * np.exp(-t * 14.0)
    phase = 2.0 * math.pi * np.cumsum(f_inst.astype(np.float64)) / SR
    sub = 0.78 * np.sin(phase).astype(np.float32) + 0.22 * np.sin(2.0 * phase).astype(np.float32) * np.exp(-t * 4.5)
    transient = np.sin(2.0 * math.pi * 140.0 * t) * np.exp(-t * 38.0) * 0.28
    env = np.minimum(1.0, t / 0.003) * np.exp(-t * 2.4) * np.minimum(1.0, (dur_s - t) / 0.08)
    sig = np.tanh((sub + transient) * 1.18) * env * amp
    return pan_stereo(sig.astype(np.float32), 0.0)


def synth_clock_tick(rng: np.random.Generator, amp: float = 0.085, pan: float = 0.0, accent: bool = False) -> np.ndarray:
    """High metallic clockwork escapement tick (>3.5 kHz shimmer + resonant partials)."""
    n = int(0.065 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    base = 4680.0 if accent else 4120.0
    metal = (
        0.50 * np.sin(2.0 * math.pi * base * t) * np.exp(-t * 68.0)
        + 0.32 * np.sin(2.0 * math.pi * (base * 1.47) * t) * np.exp(-t * 88.0)
        + 0.18 * np.sin(2.0 * math.pi * (base * 2.13) * t) * np.exp(-t * 110.0)
    )
    burst = butter_filter(rng.standard_normal(n).astype(np.float32), "high", 5400.0, order=2) * np.exp(-t * 130.0) * 0.45
    sig = (metal + burst) * np.minimum(1.0, t / 0.0008) * np.minimum(1.0, (0.065 - t) / 0.008) * amp
    return pan_stereo(sig.astype(np.float32), pan)


def synth_industrial_clang(
    rng: np.random.Generator,
    amp: float = 0.16,
    pan: float = 0.0,
    base_hz: float = 285.0,
    dur_s: float = 0.28,
) -> np.ndarray:
    """Tight industrial steel brake/rail percussive strike."""
    n = int(round(dur_s * SR))
    t = np.arange(n, dtype=np.float32) / SR
    ratios = [1.0, 1.48, 2.19, 3.31, 4.76]
    metal = np.zeros(n, dtype=np.float32)
    for idx, r in enumerate(ratios):
        metal += (0.42 / (idx + 1)) * np.sin(2.0 * math.pi * (base_hz * r) * t) * np.exp(-t * (12.0 + idx * 7.5))
    grit = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1400.0, 6200.0), order=2)
    grit *= np.exp(-t * 42.0) * 0.45
    sig = np.tanh((metal + grit) * 1.25) * np.minimum(1.0, t / 0.0012) * np.minimum(1.0, (dur_s - t) / 0.025) * amp
    return pan_stereo(sig.astype(np.float32), pan)


def synth_pluck(
    midi: float,
    dur_s: float = 0.19,
    amp: float = 0.15,
    decay: float = 12.0,
    pan: float = 0.0,
    brightness: float = 1.0,
) -> np.ndarray:
    """Tactile analog synth ostinato stab with smooth exponential envelope."""
    n = int(round(dur_s * SR))
    t = np.arange(n, dtype=np.float32) / SR
    f = midi_to_hz(midi)
    body = (
        0.52 * np.sin(2.0 * math.pi * f * t)
        + 0.24 * np.sin(4.0 * math.pi * f * t) * np.exp(-t * 13.0)
        + 0.24 * brightness * band_limited_saw(f, t, max_harmonics=12, rolloff=0.95) * np.exp(-t * 18.0)
    )
    env = np.minimum(1.0, t / 0.003) * np.exp(-t * decay) * np.minimum(1.0, (dur_s - t) / 0.015)
    return pan_stereo((body * env * amp).astype(np.float32), pan)


def synth_xray_ping(midi: float, dur_s: float = 0.95, amp: float = 0.095, pan: float = 0.0) -> np.ndarray:
    """Pure crystalline X-ray harmonic shimmer ping."""
    n = int(round(dur_s * SR))
    t = np.arange(n, dtype=np.float32) / SR
    f = midi_to_hz(midi)
    ping = (
        0.66 * np.sin(2.0 * math.pi * f * t)
        + 0.24 * np.sin(2.0 * math.pi * (f * 2.0015) * t) * np.exp(-t * 4.2)
        + 0.10 * np.sin(2.0 * math.pi * (f * 3.0) * t) * np.exp(-t * 7.5)
    )
    env = np.minimum(1.0, t / 0.008) * np.exp(-t * 3.4) * np.minimum(1.0, (dur_s - t) / 0.05)
    return pan_stereo((ping * env * amp).astype(np.float32), pan)


def synth_riser_noise(
    rng: np.random.Generator,
    dur_s: float,
    f_start: float,
    f_end: float,
    amp: float,
    curve_pow: float = 1.6,
) -> np.ndarray:
    """Sweeping stereo filtered noise riser."""
    n = int(round(dur_s * SR))
    if n <= 0:
        return np.zeros((0, CHANNELS), dtype=np.float32)
    t = np.arange(n, dtype=np.float32) / SR
    u = np.clip(t / max(1e-4, dur_s), 0.0, 1.0) ** curve_pow

    bands = [
        (f_start, f_start * 1.9),
        (f_start * 1.6, (f_start + f_end) * 0.55),
        ((f_start + f_end) * 0.45, f_end * 0.85),
        (f_end * 0.70, min(9500.0, f_end * 1.35)),
    ]
    st = np.stack(
        [rng.standard_normal(n).astype(np.float32), rng.standard_normal(n).astype(np.float32)],
        axis=-1,
    )
    out = np.zeros_like(st)
    centers = np.linspace(0.0, 1.0, len(bands))
    for c, (lo, hi) in zip(centers, bands):
        lo_c = float(np.clip(lo, 40.0, 11000.0))
        hi_c = float(np.clip(max(lo_c + 40.0, hi), 80.0, 13000.0))
        filtered = butter_filter(st, "band", (lo_c, hi_c), order=2)
        w = np.exp(-((u - c) / 0.34) ** 2).astype(np.float32)
        out += filtered * w[:, None]

    env = np.minimum(1.0, t / 0.12) * (0.20 + 0.80 * u) * np.minimum(1.0, (dur_s - t) / 0.025)
    return (out * env[:, None] * amp).astype(np.float32)


def synth_shepard_fall(
    t: np.ndarray,
    fall_progress: np.ndarray,
    octaves_drop: float = 2.2,
    center_log2_hz: float = math.log2(340.0),
    sigma_oct: float = 1.35,
    n_layers: int = 7,
) -> np.ndarray:
    """
    Seamless stereo Shepard tone plunge synced to elevator fall speed:
    multi-octave partials glide downward while weighted by a stationary log-frequency bell.
    """
    n = len(t)
    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)
    base_min_hz = 42.0
    span_oct = float(n_layers)

    for k in range(n_layers):
        # Wrap octave position seamlessly across [0, span_oct)
        oct_pos = (k - octaves_drop * fall_progress) % span_oct
        freq = base_min_hz * (2.0 ** oct_pos)
        log2_f = np.log2(np.maximum(freq, 20.0))
        weight = np.exp(-0.5 * ((log2_f - center_log2_hz) / sigma_oct) ** 2)
        # Edge fade near top/bottom octave wrap boundary to ensure zero clicks
        wrap_fade = np.sin(math.pi * np.clip(oct_pos / span_oct, 0.0, 1.0)) ** 0.65
        w = weight * wrap_fade

        det = 1.0 + (0.0028 if k % 2 == 0 else -0.0028)
        phase_l = 2.0 * math.pi * np.cumsum(freq.astype(np.float64)) / SR
        phase_r = 2.0 * math.pi * np.cumsum((freq * det).astype(np.float64)) / SR
        left += w * (0.78 * np.sin(phase_l) + 0.22 * np.sin(2.0 * phase_l))
        right += w * (0.78 * np.sin(phase_r) + 0.22 * np.sin(2.0 * phase_r))

    return np.stack([left, right], axis=-1).astype(np.float32) * 0.28


def compose_beat1(bus: np.ndarray, rng: np.random.Generator) -> None:
    """Beat 1 (0.0-5.2s): Tense sub-bass pulse (44 Hz / 88 Hz) + high metallic clockwork tick anticipation."""
    t0, t1 = beat_span(1)
    dur = t1 - t0
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    t_alarm = ev_s("alarm_start")   # 1.833 s
    t_crouch = ev_s("crouch_start") # 3.167 s

    # 44 Hz / 88 Hz tense cinematic sub-bass bed with pulsing throb
    pulse_lfo = 0.62 + 0.38 * np.sin(2.0 * math.pi * (1.0 / BEAT_S) * t - math.pi * 0.5)
    sub_44 = np.sin(2.0 * math.pi * 44.0 * t)
    sub_88 = np.sin(2.0 * math.pi * 88.0 * t)
    harm_132 = np.sin(2.0 * math.pi * 132.0 * t) * 0.16
    bed = 0.60 * sub_44 + 0.34 * sub_88 + harm_132
    swell = 0.85 + 0.35 * np.clip((t - t_crouch) / max(0.2, dur - t_crouch), 0.0, 1.0)
    bed_env = make_env(n, 0.06, 0.12) * pulse_lfo * swell * 0.23
    add_stereo_event(bus, pan_stereo((bed * bed_env).astype(np.float32), 0.0), t0)

    # Dark minor suspense pad creeping in from alarm_start (1.833s)
    dark_pad = stereo_ensemble_pad(
        t,
        midi_notes=[41, 53, 60, 68],  # F2, F3, C4, Ab4 (tense F minor)
        weights=[0.30, 0.28, 0.22, 0.18],
        detune_cents=7.0,
        max_harmonics=8,
        rolloff=1.05,
    )
    pad_gate = np.clip((t - (t_alarm - 0.15)) / 1.4, 0.0, 1.0)
    pad_env = (make_env(n, 0.25, 0.15) * pad_gate * 0.085)[:, None]
    add_stereo_event(bus, (dark_pad * pad_env).astype(np.float32), t0)

    # Rhythmic 44/88 Hz sub-bass pulse hits + high metallic clockwork ticks
    step = BEAT_S * 0.5  # 0.250 s (8th notes at 120 BPM)
    st = t0
    k = 0
    while st < t1 - 0.08:
        pan = -0.34 if k % 2 == 0 else 0.34
        is_accent = (k % 2 == 0)
        # Metallic clockwork escapement ticks
        add_stereo_event(
            bus,
            synth_clock_tick(rng, amp=0.095 if is_accent else 0.062, pan=pan, accent=is_accent),
            st,
        )
        # Double-time 16th clockwork ticks once crouch_start tension ramps
        if st >= t_crouch:
            add_stereo_event(
                bus,
                synth_clock_tick(rng, amp=0.052, pan=-pan * 0.75, accent=False),
                st + step * 0.5,
            )
        # Sub-bass 44/88 Hz pulse stabs on quarter notes (and 8ths after crouch)
        if k % 2 == 0 or st >= t_crouch:
            n_p = int(0.22 * SR)
            tp = np.arange(n_p, dtype=np.float32) / SR
            p_sig = (
                0.64 * np.sin(2.0 * math.pi * 44.0 * tp)
                + 0.36 * np.sin(2.0 * math.pi * 88.0 * tp)
            ) * np.exp(-tp * 10.5) * np.minimum(1.0, tp / 0.004)
            p_amp = 0.24 if k % 4 == 0 else (0.17 if k % 2 == 0 else 0.12)
            add_stereo_event(bus, pan_stereo((p_sig * p_amp).astype(np.float32), 0.0), st)

        st += step
        k += 1

    # Subtle alarm chime harmonic ping at alarm_start (1.833 s)
    add_stereo_event(bus, synth_xray_ping(77, dur_s=0.75, amp=0.070, pan=0.28), t_alarm)

    # Pre-snap anticipation riser into Beat 2 (3.80 -> 5.20 s)
    add_stereo_event(bus, synth_riser_noise(rng, 1.40, 380.0, 4400.0, amp=0.11, curve_pow=1.8), t1 - 1.40)


def compose_beat2(bus: np.ndarray, rng: np.random.Generator) -> None:
    """Beat 2 (5.2-10.0s): Cable snap + escalating falling Shepard/pitch riser + taut low strings swell to 50 mph."""
    t0, t1 = beat_span(2)
    dur = t1 - t0
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    t_snap = ev_s("cable_snap")  # 5.417 s

    # Evaluate physical speed curve directly from elevator_contract
    frames = (t0 + t) * C.FPS
    speed_norm = np.array([min(1.0, max(0.0, C.speed_mph_at(float(f)) / 50.0)) for f in frames], dtype=np.float32)
    u = np.clip(t / max(1e-4, dur), 0.0, 1.0)

    # Cable snap sub-boom impact at cable_snap (5.417 s)
    add_stereo_event(bus, synth_sub_boom(amp=0.32, start_hz=110.0, end_hz=34.0, dur_s=1.05), t_snap)
    add_stereo_event(bus, synth_industrial_clang(rng, amp=0.14, pan=0.0, base_hz=370.0, dur_s=0.25), t_snap)

    # Falling Shepard illusion synced to plunge acceleration
    fall_prog = np.cumsum(0.25 + 1.35 * speed_norm.astype(np.float64)) / SR
    shepard = synth_shepard_fall(t, fall_prog, octaves_drop=1.55)
    shep_env = make_env(n, 0.22, 0.16) * (0.14 + 0.16 * (u ** 1.15))
    add_stereo_event(bus, (shepard * shep_env[:, None]).astype(np.float32), t0)

    # Escalating counter-motion pitch riser + stereo wind-shear noise riser as speed reaches 50 mph
    r_freq = midi_to_hz(53) * (2.0 ** (1.35 * (u ** 1.45)))  # F3 -> ~G#4+ escalating tension!
    r_phase = 2.0 * math.pi * np.cumsum(r_freq.astype(np.float64)) / SR
    riser_tone = (0.65 * np.sin(r_phase) + 0.35 * np.sin(1.5 * r_phase)).astype(np.float32)
    riser_env = make_env(n, 0.35, 0.05) * (u ** 1.65) * 0.095
    add_stereo_event(bus, pan_stereo((riser_tone * riser_env).astype(np.float32), 0.0), t0)
    add_stereo_event(bus, synth_riser_noise(rng, dur, 240.0, 5600.0, amp=0.14, curve_pow=1.75), t0)

    # Taut low strings swell building toward 50 mph peak (F1, C2, F2, G2, Ab2, C3)
    low_strings = stereo_ensemble_pad(
        t,
        midi_notes=[29, 36, 41, 43, 44, 48],
        weights=[0.30, 0.24, 0.26, 0.18, 0.20, 0.18],
        detune_cents=9.0,
        max_harmonics=12,
        rolloff=0.92,
        vibrato_hz=4.4,
        vibrato_cents=6.0,
    )
    low_warm = butter_filter(low_strings, "low", 950.0, order=2)
    low_taut = butter_filter(low_strings, "low", 3800.0, order=2)
    bright_u = (speed_norm ** 1.4)[:, None]
    strings_blend = low_warm * (1.0 - 0.72 * bright_u) + low_taut * (0.72 * bright_u)
    str_env = (make_env(n, 0.35, 0.10) * (0.09 + 0.18 * (speed_norm ** 1.35)))[:, None]
    add_stereo_event(bus, (strings_blend * str_env).astype(np.float32), t0)

    # Driving low string staccato pulse beneath the swell
    step = BEAT_S * 0.5
    st = t_snap + 0.25
    k = 0
    while st < t1 - 0.10:
        prog = (st - t0) / dur
        pan = -0.28 if k % 2 == 0 else 0.28
        add_stereo_event(
            bus,
            synth_pluck(41 if k % 4 != 3 else 44, dur_s=0.17, amp=0.10 + 0.07 * prog, decay=11.0, pan=pan, brightness=0.85 + 0.4 * prog),
            st,
        )
        if k % 2 == 0:
            add_stereo_event(bus, synth_kick(amp=0.16 + 0.08 * prog, punch_hz=88.0, tail_hz=38.0), st)
        st += step
        k += 1


def compose_beat3(bus: np.ndarray, rng: np.random.Generator) -> None:
    """Beat 3 (10.0-15.0s): Half-time bullet-time suspense drop + pulsing synth ostinato up to 15.0s."""
    t0, t1 = beat_span(3)
    dur = t1 - t0
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    t_launch = ev_s("jump_launch")   # 11.083 s
    t_ceil = ev_s("ceiling_hit")     # 12.500 s

    # Bullet-time half-time sub-drop right at 10.00 s (F2 -> Db1 glide)
    add_stereo_event(bus, synth_sub_boom(amp=0.30, start_hz=76.0, end_hz=34.65, dur_s=1.35), t0)
    f_glide = midi_to_hz(41) * (1.0 - 0.21 * np.clip(t / 1.4, 0.0, 1.0))  # F2 -> ~Db2
    g_phase = 2.0 * math.pi * np.cumsum(f_glide.astype(np.float64)) / SR
    sub_bed = 0.64 * np.sin(g_phase).astype(np.float32) + 0.24 * np.sin(2.0 * g_phase).astype(np.float32)
    sub_env = make_env(n, 0.08, 0.18) * (0.14 + 0.05 * np.clip((t - (t_ceil - t0)) / 1.5, 0.0, 1.0))
    add_stereo_event(bus, pan_stereo((sub_bed * sub_env).astype(np.float32), 0.0), t0)

    # Half-time heavy pulse (every 1.0 s = 60 BPM bullet-time feel!)
    st = t0
    while st < t1 - 0.15:
        add_stereo_event(bus, synth_kick(amp=0.24, punch_hz=74.0, tail_hz=34.6, decay=10.5), st)
        add_stereo_event(bus, synth_clock_tick(rng, amp=0.058, pan=0.40, accent=False), st + 0.50)
        st += 1.00

    # Jump launch whoosh cue & ceiling strike impact thud
    add_stereo_event(bus, synth_pluck(53, dur_s=0.35, amp=0.14, decay=6.5, pan=0.0, brightness=1.0), t_launch)
    add_stereo_event(bus, synth_sub_boom(amp=0.28, start_hz=104.0, end_hz=36.0, dur_s=0.65), t_ceil)
    add_stereo_event(bus, synth_industrial_clang(rng, amp=0.12, pan=-0.18, base_hz=210.0, dur_s=0.22), t_ceil)

    # Pulsing synth ostinato up to 15.0s (16th-note tense analog pattern)
    ost_notes = [41, 53, 48, 56, 41, 53, 49, 55]  # Fm / Phrygian tension ostinato
    step = BEAT_S * 0.25  # 0.125 s (16th notes)
    st = t0 + 0.25
    k = 0
    while st < t1 - 0.06:
        prog = (st - t0) / dur
        pan = -0.34 if k % 2 == 0 else 0.34
        is_bar_accent = (k % 4 == 0)
        amp = (0.135 if is_bar_accent else 0.095) * (0.80 + 0.45 * prog)
        bright = 0.72 + 0.65 * prog
        add_stereo_event(
            bus,
            synth_pluck(ost_notes[k % len(ost_notes)], dur_s=0.14, amp=amp, decay=15.0, pan=pan, brightness=bright),
            st,
        )
        st += step
        k += 1

    # Building suspense sweep into 28G pit impact (13.40 -> 15.00 s)
    add_stereo_event(bus, synth_riser_noise(rng, 1.60, 260.0, 4200.0, amp=0.12, curve_pow=1.8), t1 - 1.60)


def compose_beat4(bus: np.ndarray, rng: np.random.Generator) -> None:
    """Beat 4 (15.0-20.6s): Heavy dramatic low brass/sub-boom chord + high crystalline X-ray shimmer pad."""
    t0, t1 = beat_span(4)
    dur = t1 - t0
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    t_impact = ev_s("pit_impact")     # 15.667 s
    t_peak = ev_s("pit_peak_comp")    # 16.417 s

    # Evaluate physical 28G load & spine compression curves from elevator_contract
    frames = (t0 + t) * C.FPS
    comp_norm = np.array([C.spine_impact_comp_cm(float(f)) / 3.5 for f in frames], dtype=np.float32)
    g_norm = np.array([min(1.0, max(0.0, (C.g_load_at(float(f)) - 1.0) / 27.0)) for f in frames], dtype=np.float32)
    xray_w = np.array([C.xray_body_weight(float(f)) for f in frames], dtype=np.float32)

    # Massive sub-boom at pit impact (15.667 s) & secondary compression shock at pit_peak_comp (16.417 s)
    add_stereo_event(bus, synth_sub_boom(amp=0.40, start_hz=92.0, end_hz=32.0, dur_s=1.85), t_impact)
    add_stereo_event(bus, synth_kick(amp=0.28, punch_hz=72.0, tail_hz=32.7, decay=8.5), t_peak)

    # Heavy dramatic low brass / sub-boom sustained chord (F1, C2, F2, Gb2, Ab2, C3) bending down under 28G!
    bend_hz = 1.0 - 0.022 * comp_norm
    brass_chord = stereo_ensemble_pad(
        t,
        midi_notes=[29, 36, 41, 42, 44, 48],
        weights=[0.34, 0.24, 0.28, 0.14, 0.20, 0.18],
        detune_cents=10.5,
        max_harmonics=11,
        rolloff=0.90,
    )
    brass_filtered = butter_filter(brass_chord, "low", 2100.0, order=2)
    impact_gate = np.clip((t - (t_impact - t0 - 0.18)) / 0.22, 0.05, 1.0)
    brass_env = (make_env(n, 0.10, 0.25) * impact_gate * (0.14 + 0.13 * g_norm))[:, None]
    add_stereo_event(bus, (brass_filtered * brass_env).astype(np.float32), t0)

    # Sub-boom 34-43 Hz bedrock weighted by 28G curve
    f_sub = midi_to_hz(29) * bend_hz  # ~43.65 Hz F1 bending slightly under spinal shock
    p_sub = 2.0 * math.pi * np.cumsum(f_sub.astype(np.float64)) / SR
    sub_bed = 0.72 * np.sin(p_sub).astype(np.float32) + 0.28 * np.sin(0.5 * p_sub).astype(np.float32)
    sub_env = make_env(n, 0.12, 0.24) * (0.15 + 0.09 * g_norm)
    add_stereo_event(bus, pan_stereo((sub_bed * sub_env).astype(np.float32), 0.0), t0)

    # High crystalline X-ray shimmer pad (F5, C6, Eb6, F6, G6) cleanly shimmering above vocal pocket
    shimmer_lfo = 1.0 + 0.18 * np.sin(2.0 * math.pi * 1.65 * t)
    xray_pad = stereo_ensemble_pad(
        t,
        midi_notes=[77, 84, 87, 89, 91],
        weights=[0.32, 0.26, 0.22, 0.22, 0.16],
        detune_cents=12.0,
        max_harmonics=4,
        rolloff=1.25,
        vibrato_hz=5.2,
        vibrato_cents=5.0,
    )
    xray_pad = butter_filter(xray_pad, "high", 2600.0, order=2)
    xray_env = (make_env(n, 0.35, 0.30) * (0.04 + 0.08 * xray_w) * shimmer_lfo)[:, None]
    add_stereo_event(bus, (xray_pad * xray_env).astype(np.float32), t0)

    # Crystalline X-ray sonar pings + restrained shockwave sub pulses
    sonar_notes = [77, 84, 80, 89, 84, 87]
    st = t_impact + 0.25
    k = 0
    while st < t1 - 0.30:
        pan = -0.46 if k % 2 == 0 else 0.46
        add_stereo_event(bus, synth_xray_ping(sonar_notes[k % len(sonar_notes)], dur_s=0.90, amp=0.088, pan=pan), st)
        if k > 0:
            add_stereo_event(bus, synth_kick(amp=0.19, punch_hz=66.0, tail_hz=33.0), st)
        add_stereo_event(bus, synth_clock_tick(rng, amp=0.042, pan=-pan, accent=False), st + 0.40)
        st += 0.80
        k += 1


def compose_beat5(bus: np.ndarray, rng: np.random.Generator) -> None:
    """
    Beat 5 (20.6-25.5s): Clean pivot — percussive industrial drive during safety brakes (20.60-23.08s),
    softening cleanly (23.08-25.455s) into a crisp 45 ms breath gap (25.455-25.500s) right before Beat 6!
    """
    t0, _ = beat_span(5)
    t_brake_end = ev_s("brake_cut_end")       # 23.083 s
    t_clamp = ev_s("brake_clamp")             # 21.250 s
    t_settled = ev_s("lie_flat_settled")      # 24.417 s
    t_end = BREATH_START_S                    # 25.455 s (45 ms breath before 25.500 s)

    # --- Phase 5A (20.60 -> 23.08 s): Percussive industrial drive during safety clamps ---
    dur_a = t_brake_end - t0
    na = int(round(dur_a * SR))
    ta = np.arange(na, dtype=np.float32) / SR
    # Gritty C#1/C#2 industrial bass bed setting up the F# / Db tonal pivot
    bass_a = 0.62 * np.sin(2.0 * math.pi * midi_to_hz(37) * ta) + 0.32 * band_limited_saw(midi_to_hz(37), ta, max_harmonics=8, rolloff=1.0)
    env_a = make_env(na, 0.04, 0.14) * 0.18
    add_stereo_event(bus, pan_stereo((bass_a * env_a).astype(np.float32), 0.0), t0)

    # Double metallic rail-clamp slams at brake_clamp (21.25 s)
    add_stereo_event(bus, synth_industrial_clang(rng, amp=0.22, pan=-0.38, base_hz=310.0, dur_s=0.35), t_clamp)
    add_stereo_event(bus, synth_industrial_clang(rng, amp=0.22, pan=0.38, base_hz=328.0, dur_s=0.35), t_clamp + 0.025)
    add_stereo_event(bus, synth_sub_boom(amp=0.28, start_hz=88.0, end_hz=36.0, dur_s=0.55), t_clamp)

    # Industrial percussive groove (8th/16th kicks, metallic clangs, tight bass plucks)
    step_a = BEAT_S * 0.25  # 0.125 s
    notes_a = [37, 49, 44, 49, 37, 51, 49, 44]  # C#2 / G#2 / D#3 industrial pivot
    st = t0
    k = 0
    while st < t_brake_end - 0.06:
        pan = -0.35 if k % 2 == 0 else 0.35
        if k % 2 == 0:
            add_stereo_event(
                bus,
                synth_pluck(notes_a[(k // 2) % len(notes_a)], dur_s=0.14, amp=0.14, decay=14.0, pan=pan, brightness=1.25),
                st,
            )
        if k % 4 == 0:
            add_stereo_event(bus, synth_kick(amp=0.25, punch_hz=98.0, tail_hz=37.0), st)
        elif k % 4 == 2:
            add_stereo_event(bus, synth_industrial_clang(rng, amp=0.12, pan=-pan * 0.6, base_hz=260.0, dur_s=0.18), st)
        else:
            add_stereo_event(bus, synth_clock_tick(rng, amp=0.055, pan=pan, accent=(k % 4 == 1)), st)
        st += step_a
        k += 1

    # --- Phase 5B (23.08 -> 25.455 s): Softening cleanly into calm anticipation ---
    dur_b = t_end - t_brake_end
    nb = int(round(dur_b * SR))
    tb = np.arange(nb, dtype=np.float32) / SR
    ub = np.clip(tb / max(1e-4, dur_b), 0.0, 1.0)

    # Warm softening suspended pad (C#3, G#3, C#4, D#4 -> gentle anticipation of F#maj7!)
    soft_pad = stereo_ensemble_pad(
        tb,
        midi_notes=[49, 56, 61, 63],
        weights=[0.30, 0.26, 0.24, 0.18],
        detune_cents=8.0,
        max_harmonics=6,
        rolloff=1.15,
    )
    soft_pad = butter_filter(soft_pad, "low", 1800.0, order=2)
    # Softens smoothly down to near-zero right at 25.455 s
    soft_fade = (1.0 - 0.72 * (ub ** 1.35)) * np.minimum(1.0, (dur_b - tb) / 0.055)
    soft_env = (np.minimum(1.0, tb / 0.18) * soft_fade * 0.12)[:, None]
    add_stereo_event(bus, (soft_pad * soft_env).astype(np.float32), t_brake_end)

    # Gentle muted pulse at survival cut & lie-flat settled (23.10s, 23.75s, 24.42s)
    add_stereo_event(bus, synth_kick(amp=0.15, punch_hz=68.0, tail_hz=36.0, decay=11.0), t_brake_end + 0.02)
    add_stereo_event(bus, synth_xray_ping(73, dur_s=0.95, amp=0.065, pan=-0.25), t_settled)
    add_stereo_event(bus, synth_kick(amp=0.11, punch_hz=62.0, tail_hz=36.0, decay=12.0), t_settled)


def compose_beat6(bus: np.ndarray, rng: np.random.Generator) -> None:
    """
    Beat 6 (25.5-30.0s): Inspiring heroic major-7th resolution swell (F#maj7 -> Dbmaj7 cinematic uplift),
    cleanly fading tail at 29.6-30.0s.
    """
    t0, t1 = beat_span(6)
    t_enter = max(BREATH_END_S, t0)  # Exact entry right after the 45 ms breath gap (25.500 s)
    dur = t1 - t_enter
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    t_grid = ev_s("grid_reveal")     # 25.750 s

    # Clean raised-cosine tail fade over 29.60 -> 30.00 s (last 0.40 s)
    tail_start = 29.60 - t_enter
    tail_dur = max(0.05, dur - tail_start)
    tail_u = np.clip((t - tail_start) / tail_dur, 0.0, 1.0)
    outro_fade = (0.5 * (1.0 + np.cos(math.pi * tail_u))).astype(np.float32)

    # Heroic chord progression: F#maj7 (0..2.1s) -> Dbmaj7 (2.1..4.5s)
    # Crossfade smoothly centered at t=2.15 s
    prog_u = 0.5 * (1.0 + np.tanh((t - 2.15) / 0.55))

    # Foundation bass: F#1/F#2 (MIDI 30/42) -> Db2/Db3 (MIDI 37/49)
    f_bass_1 = midi_to_hz(30) * (1.0 - prog_u) + midi_to_hz(37) * prog_u
    f_bass_2 = midi_to_hz(42) * (1.0 - prog_u) + midi_to_hz(49) * prog_u
    p_b1 = 2.0 * math.pi * np.cumsum(f_bass_1.astype(np.float64)) / SR
    p_b2 = 2.0 * math.pi * np.cumsum(f_bass_2.astype(np.float64)) / SR
    bass = 0.58 * np.sin(p_b1).astype(np.float32) + 0.34 * np.sin(p_b2).astype(np.float32) + 0.14 * np.sin(1.5 * p_b2).astype(np.float32)
    bass_env = np.minimum(1.0, t / 0.045) * outro_fade * 0.24
    add_stereo_event(bus, pan_stereo((bass * bass_env).astype(np.float32), 0.0), t_enter)

    # F#maj7 lush brass/strings: F#2(42), C#3(49), F#3(54), A#3(58), C#4(61), F4(65), A#4(70)
    pad_fsharp = stereo_ensemble_pad(
        t,
        midi_notes=[42, 49, 54, 58, 61, 65, 70],
        weights=[0.26, 0.22, 0.24, 0.22, 0.20, 0.22, 0.16],
        detune_cents=9.5,
        max_harmonics=13,
        rolloff=0.92,
        vibrato_hz=4.5,
        vibrato_cents=5.0,
    )
    # Dbmaj7 lush brass/strings: Db3(49), Ab3(56), Db4(61), F4(65), Ab4(68), C5(72), Eb5(75)
    pad_dbmaj = stereo_ensemble_pad(
        t,
        midi_notes=[49, 56, 61, 65, 68, 72, 75],
        weights=[0.26, 0.22, 0.24, 0.22, 0.22, 0.20, 0.16],
        detune_cents=9.5,
        max_harmonics=13,
        rolloff=0.92,
        vibrato_hz=4.6,
        vibrato_cents=5.5,
    )
    chord_pad = pad_fsharp * (1.0 - prog_u[:, None]) + pad_dbmaj * prog_u[:, None]

    # Cinematic filter bloom peaking through 27.5s -> 29.0s
    pad_warm = butter_filter(chord_pad, "low", 2100.0, order=2)
    pad_bright = butter_filter(chord_pad, "low", 6400.0, order=2)
    bloom_u = np.clip(t / 2.2, 0.0, 1.0)[:, None]
    brass = pad_warm * (1.0 - 0.70 * bloom_u) + pad_bright * (0.70 * bloom_u)
    sw_arch = 0.80 + 0.28 * np.exp(-((t - 2.35) / 1.35) ** 2)
    brass_env = (np.minimum(1.0, t / 0.14) * outro_fade * sw_arch * 0.24)[:, None]
    add_stereo_event(bus, (brass * brass_env).astype(np.float32), t_enter)

    # Soaring upper crystalline shimmer (F#5/A#5/C#6/F6 -> Db5/F5/Ab5/C6)
    sky_fs = stereo_ensemble_pad(t, [78, 82, 85, 89], [0.30, 0.28, 0.26, 0.22], detune_cents=11.0, max_harmonics=5)
    sky_db = stereo_ensemble_pad(t, [73, 77, 80, 84], [0.30, 0.28, 0.28, 0.24], detune_cents=11.0, max_harmonics=5)
    sky = sky_fs * (1.0 - prog_u[:, None]) + sky_db * prog_u[:, None]
    sky_env = (np.minimum(1.0, t / 0.32) * outro_fade * 0.115)[:, None]
    add_stereo_event(bus, (sky * sky_env).astype(np.float32), t_enter)

    # Majestic half-time cinema impacts & crystalline grid reveal shimmer
    add_stereo_event(bus, synth_kick(amp=0.30, punch_hz=86.0, tail_hz=46.2), t_enter)
    add_stereo_event(bus, synth_xray_ping(78, dur_s=1.40, amp=0.095, pan=-0.30), t_grid)
    add_stereo_event(bus, synth_xray_ping(85, dur_s=1.40, amp=0.085, pan=0.35), t_grid + 0.12)
    add_stereo_event(bus, synth_kick(amp=0.24, punch_hz=76.0, tail_hz=34.6), t_enter + 2.15)
    add_stereo_event(bus, synth_xray_ping(84, dur_s=1.60, amp=0.085, pan=0.0), t_enter + 2.15)
    add_stereo_event(bus, synth_clock_tick(rng, amp=0.042, pan=0.0, accent=True), t_enter + 2.15)


def enforce_breath_window(bus: np.ndarray, start_s: float, end_s: float, fade_s: float = 0.008) -> None:
    """Guarantee a razor-clean silent breath window [start_s, end_s] with click-free edges."""
    i0 = max(0, int(round(start_s * SR)))
    i1 = min(len(bus), int(round(end_s * SR)))
    nf = max(1, int(round(fade_s * SR)))
    if i0 > nf:
        bus[i0 - nf : i0] *= np.linspace(1.0, 0.0, nf, dtype=np.float32)[:, None]
    if i1 > i0:
        bus[i0:i1] = 0.0
    if i1 + nf < len(bus):
        bus[i1 : i1 + nf] *= np.linspace(0.0, 1.0, nf, dtype=np.float32)[:, None]


def apply_vocal_pocket_eq(bus: np.ndarray) -> np.ndarray:
    """
    Carve a smooth -3.5 dB vocal-pocket EQ notch across 250-3500 Hz
    so en-US-BrianNeural narration cuts through with razor clarity.
    """
    bus = peaking_eq(bus, 520.0, 0.68, -3.1)
    bus = peaking_eq(bus, 1850.0, 0.62, -3.2)
    return bus


def compose_bgm() -> np.ndarray:
    """Compose, EQ, and normalize the 30.000 s 48 kHz stereo BGM stem."""
    bus = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)
    rng = np.random.default_rng(20260716)

    compose_beat1(bus, rng)
    compose_beat2(bus, rng)
    compose_beat3(bus, rng)
    compose_beat4(bus, rng)
    compose_beat5(bus, rng)
    compose_beat6(bus, rng)

    # Clean subsonic rumble cut below 26 Hz
    bus = butter_filter(bus, "high", 26.0, order=2)

    # Permanent -3.5 dB vocal-pocket EQ notch across 250-3500 Hz
    bus = apply_vocal_pocket_eq(bus)

    # Warm analog saturation + headroom normalization (peak ~ 0.80, <= 0.82, zero clipping)
    bus = np.tanh(bus * 1.12).astype(np.float32)
    peak = float(np.max(np.abs(bus)))
    if peak > 1e-6:
        bus *= 0.80 / peak

    # Re-assert crisp 45 ms breath gap (25.455-25.500 s) after IIR EQ rings
    enforce_breath_window(bus, BREATH_START_S, BREATH_END_S, fade_s=0.006)

    # Clean raised-cosine tail fade at 29.60-30.00 s + 5 ms opening fade
    micro = int(round(0.005 * SR))
    bus[:micro] *= np.linspace(0.0, 1.0, micro, dtype=np.float32)[:, None]
    i_tail = int(round(29.60 * SR))
    tail_len = len(bus) - i_tail
    if tail_len > 0:
        u_tail = np.linspace(0.0, 1.0, tail_len, dtype=np.float32)
        bus[i_tail:] *= (0.5 * (1.0 + np.cos(math.pi * u_tail)))[:, None]
    bus[-1, :] = 0.0
    return bus.astype(np.float32)


def verify_stem(path: Path, audio: np.ndarray) -> None:
    """Verify exact 48 kHz stereo 30.000 s WAV specs, vocal-pocket EQ, breath gap, and zero clipping."""
    info = sf.info(str(path))
    assert info.samplerate == SR, f"Expected SR={SR}, got {info.samplerate}"
    assert info.channels == CHANNELS, f"Expected {CHANNELS} channels, got {info.channels}"
    assert info.frames == TOTAL_SAMPLES, f"Expected {TOTAL_SAMPLES} frames, got {info.frames}"
    assert abs(info.duration - DURATION_S) < 1e-6, f"Expected {DURATION_S}s, got {info.duration}"

    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == SR
        assert wf.getnchannels() == CHANNELS
        assert wf.getnframes() == TOTAL_SAMPLES
        assert wf.getsampwidth() == 2, f"Expected 16-bit PCM_16, got {wf.getsampwidth() * 8}-bit"

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    assert 0.02 < rms < 0.60, f"Unexpected RMS: {rms:.5f}"
    assert peak <= 0.82, f"Peak exceeded 0.82 ceiling: {peak:.5f}"

    b0 = int(round((BREATH_START_S + 0.004) * SR))
    b1 = int(round((BREATH_END_S - 0.004) * SR))
    breath_rms = float(np.sqrt(np.mean(audio[b0:b1].astype(np.float64) ** 2)))
    assert breath_rms < 1e-4, f"Breath window not clean: {breath_rms:.6f}"

    tail_end_rms = float(np.sqrt(np.mean(audio[int(round(29.97 * SR)) :].astype(np.float64) ** 2)))
    assert tail_end_rms < 0.01, f"Outro tail did not fade cleanly: {tail_end_rms:.6f}"

    print(
        f"[BGM] Wrote {path.name}: {info.samplerate} Hz, {info.channels}ch, PCM_16, "
        f"{info.frames} samples ({info.duration:.3f}s), peak={peak:.4f}, rms={rms:.4f}, "
        f"breath_rms={breath_rms:.6f}, tail_end_rms={tail_end_rms:.6f}"
    )
    for b_idx in sorted(C.BEATS):
        s0, s1 = beat_span(b_idx)
        seg = audio[int(round(s0 * SR)) : int(round(s1 * SR))]
        seg_rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        seg_pk = float(np.max(np.abs(seg)))
        print(f"  Beat {b_idx} ({s0:5.2f}-{s1:5.2f}s): rms={seg_rms:.4f}, peak={seg_pk:.4f}")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audio = compose_bgm()
    sf.write(str(OUTPUT_PATH), audio, SR, subtype="PCM_16")
    verify_stem(OUTPUT_PATH, audio)


if __name__ == "__main__":
    main()
