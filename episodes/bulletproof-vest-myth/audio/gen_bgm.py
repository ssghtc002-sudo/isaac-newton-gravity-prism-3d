"""
BGM Composer for Episode #3
("Why Bulletproof Vests DON'T Send You Flying Backward", 30.0 s @ 60 fps).

Generates:
    episodes/bulletproof-vest-myth/audio/output/bgm_stem.wav
    - 48000 Hz, Stereo, PCM_16, exact 30.000 s (1,440,000 samples)
    - Synced strictly to vest_contract.BEATS and vest_contract.EV
    - @zackdfilms-style taut, dark ballistic-lab rhythmic suspense bed in D minor (124 BPM):
      * Deep D1/D2 kinetic pulse + clean precision escapement ticks (>3.6 kHz, zero vocal clutter)
      * Crystalline X-ray shimmer during Beats 3-4 (Kevlar weave & SiC ceramic shatter)
      * Crisp analytical Studio Blueprint pulse in Beat 5 (Newton's Third Law proof)
      * Clean 45 ms breath gap before Beat 6 grounded resolution (Bbmaj7 -> Dm(add9))
    - Permanent vocal-pocket EQ notch cut (250-3500 Hz, -3.5 dB peak cut) for BrianNeural clarity
    - Smooth 20 ms fade-in/out and peak strictly <= 0.82 (zero clipping)
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

import vest_contract as C  # noqa: E402

SR = 48000
CHANNELS = 2
DURATION_S = float(C.DURATION_S)
TOTAL_SAMPLES = int(round(DURATION_S * SR))

BPM = 124.0
BEAT_S = 60.0 / BPM  # 0.4839 s

# Crisp 45 ms breath gap right before Beat 6 grounded finale (B6_FINAL starts at f=1561 -> 26.000 s)
BREATH_DUR_S = 0.045
BREATH_END_S = (C.BEATS["B6_FINAL"][0] - 1) / float(C.FPS)  # 26.000 s
BREATH_START_S = BREATH_END_S - BREATH_DUR_S                # 25.955 s

FADE_IO_S = 0.020
OUTPUT_PATH = AUDIO_DIR / "output" / "bgm_stem.wav"


def beat_span(beat_key: str) -> tuple[float, float]:
    f0, f1 = C.BEATS[beat_key]
    return ((f0 - 1) / float(C.FPS), f1 / float(C.FPS))


def ev_s(event_key: str) -> float:
    return C.EV[event_key] / float(C.FPS)


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def pan_stereo(mono: np.ndarray, pan: float) -> np.ndarray:
    pan = float(np.clip(pan, -1.0, 1.0))
    angle = (pan + 1.0) * (math.pi / 4.0)
    return np.stack([mono * math.cos(angle), mono * math.sin(angle)], axis=-1).astype(np.float32)


def add_stereo_event(bus: np.ndarray, event: np.ndarray, start_time: float) -> None:
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
    nyq = 0.5 * SR
    if isinstance(cutoff, tuple):
        wn: float | list[float] = [float(np.clip(c / nyq, 1e-4, 0.999)) for c in cutoff]
    else:
        wn = float(np.clip(cutoff / nyq, 1e-4, 0.999))
    sos = signal.butter(order, wn, btype=btype, output="sos")
    return signal.sosfilt(sos, audio, axis=0).astype(np.float32)


def peaking_eq_coeffs(freq: float, q: float, gain_db: float) -> tuple[np.ndarray, np.ndarray]:
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / SR
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    norm = 1.0 + alpha / a_lin
    b = np.array([1.0 + alpha * a_lin, -2.0 * cw, 1.0 - alpha * a_lin], dtype=np.float64) / norm
    a = np.array([1.0, (-2.0 * cw) / norm, (1.0 - alpha / a_lin) / norm], dtype=np.float64)
    return b, a


def peaking_eq(audio: np.ndarray, freq: float, q: float, gain_db: float) -> np.ndarray:
    b, a = peaking_eq_coeffs(freq, q, gain_db)
    return signal.lfilter(b, a, audio.astype(np.float64), axis=0).astype(np.float32)


def make_env(n: int, attack_s: float, release_s: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SR
    dur = n / float(SR)
    att = np.minimum(1.0, t / max(1e-4, attack_s))
    rel = np.minimum(1.0, (dur - t) / max(1e-4, release_s))
    return (att * rel).astype(np.float32)


def band_limited_wave(
    freq: float | np.ndarray,
    t: np.ndarray,
    max_harmonics: int = 10,
    rolloff: float = 1.15,
) -> np.ndarray:
    f_ref = float(np.max(freq)) if isinstance(freq, np.ndarray) else float(freq)
    if f_ref <= 1e-3:
        return np.zeros_like(t, dtype=np.float32)
    n_harm = max(1, min(max_harmonics, int(0.40 * SR / f_ref)))
    if isinstance(freq, np.ndarray):
        phase = 2.0 * math.pi * np.cumsum(freq.astype(np.float64)) / SR
    else:
        phase = 2.0 * math.pi * float(freq) * t.astype(np.float64)
    out = np.zeros(len(t), dtype=np.float64)
    for k in range(1, n_harm + 1):
        amp = (1.0 / (k ** rolloff)) * (1.0 if k % 2 == 1 else 0.72)
        out += amp * np.sin(k * phase)
    return (out * 0.58).astype(np.float32)


def dark_pad(
    t: np.ndarray,
    midi_notes: list[float],
    weights: list[float],
    detune_cents: float = 8.0,
    cutoff_hz: float = 900.0,
    vibrato_hz: float = 0.35,
    vibrato_cents: float = 4.0,
) -> np.ndarray:
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
        left += w * band_limited_wave(f_l, t, max_harmonics=10, rolloff=1.10)
        right += w * band_limited_wave(f_r, t, max_harmonics=10, rolloff=1.10)
    stereo = np.stack([left, right], axis=-1).astype(np.float32)
    return butter_filter(stereo, "low", cutoff_hz, order=2)


def synth_sub_drone(
    rng: np.random.Generator,
    dur_s: float,
    root_midi: float = 38.0,  # D2 (73.4 Hz) / D1 (36.7 Hz)
    amp: float = 0.22,
    drift_hz: float = 0.30,
) -> np.ndarray:
    n = int(round(dur_s * SR))
    t = np.arange(n, dtype=np.float32) / SR
    f_root = midi_to_hz(root_midi)
    f_sub = f_root * 0.5
    lfo_l = 0.74 + 0.26 * np.sin(2.0 * math.pi * drift_hz * t - math.pi * 0.30)
    lfo_r = 0.74 + 0.26 * np.sin(2.0 * math.pi * (drift_hz * 1.11) * t + 0.45)

    sub_tone = (
        0.64 * np.sin(2.0 * math.pi * f_sub * t)
        + 0.30 * np.sin(2.0 * math.pi * f_root * t)
        + 0.08 * np.sin(2.0 * math.pi * (f_root * 1.5) * t)
    )
    noise_l = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (38.0, 140.0), order=2)
    noise_r = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (38.0, 140.0), order=2)
    env = make_env(n, 0.14, 0.18) * amp
    return np.stack([(sub_tone + 0.42 * noise_l) * lfo_l * env, (sub_tone + 0.42 * noise_r) * lfo_r * env], axis=-1).astype(np.float32)


def synth_sub_pulse(
    midi: float = 38.0,
    dur_s: float = 0.26,
    amp: float = 0.23,
    cutoff_hz: float = 250.0,
    pan: float = 0.0,
) -> np.ndarray:
    n = int(round(dur_s * SR))
    t = np.arange(n, dtype=np.float32) / SR
    f = midi_to_hz(midi)
    f_inst = f * (1.0 + 0.16 * np.exp(-t * 38.0))
    phase = 2.0 * math.pi * np.cumsum(f_inst.astype(np.float64)) / SR
    sig = (
        0.68 * np.sin(phase).astype(np.float32)
        + 0.24 * np.sin(2.0 * phase).astype(np.float32) * np.exp(-t * 9.0)
        + 0.12 * band_limited_wave(f, t, max_harmonics=6, rolloff=1.2) * np.exp(-t * 14.0)
    )
    sig = butter_filter(sig, "low", cutoff_hz, order=2)
    env = np.minimum(1.0, t / 0.004) * np.exp(-t * 9.8) * np.minimum(1.0, (dur_s - t) / 0.022)
    return pan_stereo((sig * env * amp).astype(np.float32), pan)


def synth_kinetic_boom(
    rng: np.random.Generator,
    amp: float = 0.34,
    start_hz: float = 96.0,
    end_hz: float = 36.7,  # D1
    dur_s: float = 1.05,
) -> np.ndarray:
    n = int(round(dur_s * SR))
    t = np.arange(n, dtype=np.float32) / SR
    f_inst = end_hz + (start_hz - end_hz) * np.exp(-t * 17.0)
    phase = 2.0 * math.pi * np.cumsum(f_inst.astype(np.float64)) / SR
    sub = 0.78 * np.sin(phase).astype(np.float32) + 0.22 * np.sin(2.0 * phase).astype(np.float32) * np.exp(-t * 5.0)
    thump = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (45.0, 210.0), order=2) * np.exp(-t * 18.0) * 0.50
    env = np.minimum(1.0, t / 0.004) * np.exp(-t * 2.8) * np.minimum(1.0, (dur_s - t) / 0.08)
    return pan_stereo((np.tanh((sub + thump) * 1.12) * env * amp).astype(np.float32), 0.0)


def synth_clean_tick(
    rng: np.random.Generator,
    amp: float = 0.078,
    pan: float = 0.0,
    accent: bool = False,
    bright: float = 1.0,
) -> np.ndarray:
    """Precision ballistic-timer escapement tick (>3.7 kHz, completely clear of vocal pocket)."""
    n = int(0.052 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    base = (4640.0 if accent else 4080.0) * bright
    tone = (
        0.54 * np.sin(2.0 * math.pi * base * t) * np.exp(-t * 80.0)
        + 0.30 * np.sin(2.0 * math.pi * (base * 1.49) * t) * np.exp(-t * 108.0)
        + 0.16 * np.sin(2.0 * math.pi * (base * 2.16) * t) * np.exp(-t * 138.0)
    )
    click = butter_filter(rng.standard_normal(n).astype(np.float32), "high", 4800.0, order=2) * np.exp(-t * 155.0) * 0.38
    sig = (tone + click) * np.minimum(1.0, t / 0.0006) * np.minimum(1.0, (0.052 - t) / 0.008) * amp
    return pan_stereo(sig.astype(np.float32), pan)


def synth_glass_ping(
    midi: float,
    dur_s: float = 1.00,
    amp: float = 0.080,
    pan: float = 0.0,
) -> np.ndarray:
    n = int(round(dur_s * SR))
    t = np.arange(n, dtype=np.float32) / SR
    f = midi_to_hz(midi)
    ping = (
        0.72 * np.sin(2.0 * math.pi * f * t)
        + 0.20 * np.sin(2.0 * math.pi * (f * 2.001) * t) * np.exp(-t * 4.4)
        + 0.08 * np.sin(2.0 * math.pi * (f * 3.0) * t) * np.exp(-t * 7.8)
    )
    env = np.minimum(1.0, t / 0.008) * np.exp(-t * 3.3) * np.minimum(1.0, (dur_s - t) / 0.06)
    return pan_stereo((ping * env * amp).astype(np.float32), pan)


def synth_muted_pluck(
    midi: float,
    dur_s: float = 0.20,
    amp: float = 0.105,
    cutoff_hz: float = 780.0,
    pan: float = 0.0,
) -> np.ndarray:
    n = int(round(dur_s * SR))
    t = np.arange(n, dtype=np.float32) / SR
    f = midi_to_hz(midi)
    body = (
        0.58 * np.sin(2.0 * math.pi * f * t)
        + 0.26 * np.sin(4.0 * math.pi * f * t) * np.exp(-t * 10.0)
        + 0.22 * band_limited_wave(f, t, max_harmonics=8, rolloff=1.05) * np.exp(-t * 12.0)
    )
    body = butter_filter(body, "low", cutoff_hz, order=2)
    env = np.minimum(1.0, t / 0.004) * np.exp(-t * 9.5) * np.minimum(1.0, (dur_s - t) / 0.016)
    return pan_stereo((body * env * amp).astype(np.float32), pan)


def synth_tension_riser(
    rng: np.random.Generator,
    dur_s: float,
    f_start: float,
    f_end: float,
    amp: float,
    curve_pow: float = 1.70,
) -> np.ndarray:
    n = int(round(dur_s * SR))
    if n <= 0:
        return np.zeros((0, CHANNELS), dtype=np.float32)
    t = np.arange(n, dtype=np.float32) / SR
    u = np.clip(t / max(1e-4, dur_s), 0.0, 1.0) ** curve_pow
    bands = [
        (f_start, f_start * 1.8),
        (f_start * 1.5, (f_start + f_end) * 0.52),
        ((f_start + f_end) * 0.42, f_end * 0.85),
        (f_end * 0.68, min(6800.0, f_end * 1.25)),
    ]
    st = np.stack([rng.standard_normal(n).astype(np.float32), rng.standard_normal(n).astype(np.float32)], axis=-1)
    out = np.zeros_like(st)
    for c, (lo, hi) in zip(np.linspace(0.0, 1.0, len(bands)), bands):
        lo_c = float(np.clip(lo, 45.0, 7500.0))
        hi_c = float(np.clip(max(lo_c + 40.0, hi), 90.0, 8800.0))
        w = np.exp(-((u - c) / 0.34) ** 2).astype(np.float32)
        out += butter_filter(st, "band", (lo_c, hi_c), order=2) * w[:, None]
    env = np.minimum(1.0, t / 0.12) * (0.18 + 0.82 * u) * np.minimum(1.0, (dur_s - t) / 0.03)
    return (out * env[:, None] * amp).astype(np.float32)


def compose_beat1(bus: np.ndarray, rng: np.random.Generator) -> None:
    """Beat 1 (0.00-3.50s): Slow-mo chest hit hook — taut D minor pulse + impact sub-boom."""
    t0, t1 = beat_span("B1_HOOK")
    dur = t1 - t0
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    t_hit = ev_s("b1_impact")

    add_stereo_event(bus, synth_sub_drone(rng, dur, root_midi=38.0, amp=0.22, drift_hz=0.35), t0)
    pad = dark_pad(t, [50, 57, 65, 64], [0.34, 0.28, 0.22, 0.16], detune_cents=7.5, cutoff_hz=720.0)
    add_stereo_event(bus, (pad * (make_env(n, 0.24, 0.16) * 0.13)[:, None]).astype(np.float32), t0)

    pulse_notes = [38, 38, 45, 41, 38, 38, 43, 45]  # D2, A2, F2, G2
    st, k = t0 + 0.10, 0
    while st < t1 - 0.10:
        add_stereo_event(bus, synth_sub_pulse(pulse_notes[k % len(pulse_notes)], dur_s=0.26, amp=0.20 if k % 2 == 0 else 0.13), st)
        if k % 2 == 1:
            add_stereo_event(bus, synth_muted_pluck(62 if k % 4 == 1 else 65, dur_s=0.22, amp=0.075, cutoff_hz=640.0, pan=0.24 if k % 4 == 1 else -0.24), st)
        st += BEAT_S * 0.5
        k += 1

    add_stereo_event(bus, synth_kinetic_boom(rng, amp=0.28, start_hz=90.0, end_hz=36.7, dur_s=0.90), t_hit)
    add_stereo_event(bus, synth_glass_ping(81, dur_s=0.90, amp=0.074, pan=0.20), t_hit)  # A5
    add_stereo_event(bus, synth_tension_riser(rng, 0.95, 160.0, 2600.0, amp=0.095), t1 - 0.95)


def compose_beat2(bus: np.ndarray, rng: np.random.Generator) -> None:
    """Beat 2 (3.50-8.50s): Momentum Myth — tight ticking urgency + dry D minor kinetic groove."""
    t0, t1 = beat_span("B2_MOMENTUM")
    dur = t1 - t0
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    t_fire = ev_s("b2_muzzle_fire")
    t_hit = ev_s("b2_vest_hit")

    add_stereo_event(bus, synth_sub_drone(rng, dur, root_midi=38.0, amp=0.23, drift_hz=0.40), t0)
    prog = np.clip((t - (t_hit - t0)) / max(0.5, dur - (t_hit - t0)), 0.0, 1.0)[:, None]
    pad_a = dark_pad(t, [50, 57, 62, 65], [0.32, 0.26, 0.24, 0.20], detune_cents=8.0, cutoff_hz=800.0)
    pad_b = dark_pad(t, [46, 55, 62, 65], [0.30, 0.26, 0.24, 0.22], detune_cents=9.0, cutoff_hz=900.0)
    pad = pad_a * (1.0 - prog) + pad_b * prog
    add_stereo_event(bus, (pad * (make_env(n, 0.20, 0.16) * (0.11 + 0.05 * prog[:, 0]))[:, None]).astype(np.float32), t0)

    add_stereo_event(bus, synth_sub_pulse(38, dur_s=0.22, amp=0.22, cutoff_hz=320.0), t_fire)
    add_stereo_event(bus, synth_kinetic_boom(rng, amp=0.30, start_hz=96.0, end_hz=36.7, dur_s=1.05), t_hit)

    step = BEAT_S * 0.5
    arp_notes = [50, 57, 65, 57, 53, 60, 65, 60]
    st, k = t0, 0
    while st < t1 - 0.08:
        pan = -0.32 if k % 2 == 0 else 0.32
        is_acc = (k % 2 == 0)
        add_stereo_event(bus, synth_clean_tick(rng, amp=0.082 if is_acc else 0.050, pan=pan, accent=is_acc), st)
        if is_acc:
            add_stereo_event(bus, synth_sub_pulse(38 if k % 4 == 0 else 36, dur_s=0.24, amp=0.21 if k % 4 == 0 else 0.15), st)
        if st >= t_fire - 0.02:
            add_stereo_event(bus, synth_muted_pluck(arp_notes[k % len(arp_notes)], dur_s=0.19, amp=0.092, cutoff_hz=760.0, pan=-pan * 0.7), st)
        st += step
        k += 1


def compose_beat3(bus: np.ndarray, rng: np.random.Generator) -> None:
    """Beat 3 (8.50-15.00s): Golden Kevlar Trampoline X-Ray — driving ostinato + airy X-ray shimmer."""
    t0, t1 = beat_span("B3_KEVLAR")
    dur = t1 - t0
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    t_peak = ev_s("b3_ripple_peak")
    u = np.clip(t / max(1e-4, dur), 0.0, 1.0)

    add_stereo_event(bus, synth_sub_drone(rng, dur, root_midi=38.0, amp=0.24, drift_hz=0.52), t0)
    shimmer = butter_filter(
        dark_pad(t, [74, 81, 86, 89], [0.28, 0.26, 0.24, 0.18], detune_cents=12.0, cutoff_hz=6200.0, vibrato_hz=4.0, vibrato_cents=6.0),
        "high", 3400.0, order=2,
    )
    bell = np.exp(-((t - (t_peak - t0)) / 2.1) ** 2)
    add_stereo_event(bus, (shimmer * (make_env(n, 0.28, 0.20) * (0.045 + 0.062 * bell))[:, None]).astype(np.float32), t0)

    low_pad = dark_pad(t, [38, 45, 50, 53, 57], [0.30, 0.24, 0.24, 0.20, 0.18], detune_cents=9.0, cutoff_hz=690.0)
    add_stereo_event(bus, (low_pad * (make_env(n, 0.24, 0.18) * (0.14 + 0.05 * u))[:, None]).astype(np.float32), t0)

    step = BEAT_S * 0.25
    ost_notes = [50, 57, 53, 57, 50, 58, 53, 57]
    st, k = t0, 0
    while st < t1 - 0.06:
        prog = (st - t0) / dur
        pan = -0.35 if k % 2 == 0 else 0.35
        if k % 2 == 0:
            is_acc = (k % 4 == 0)
            add_stereo_event(bus, synth_clean_tick(rng, amp=0.086 if is_acc else 0.054, pan=pan, accent=is_acc, bright=1.0 + 0.06 * prog), st)
            add_stereo_event(bus, synth_muted_pluck(ost_notes[(k // 2) % len(ost_notes)], dur_s=0.16, amp=0.105 + 0.028 * prog, cutoff_hz=720.0 + 180.0 * prog, pan=-pan * 0.65), st)
        elif st >= t_peak - 0.02:
            add_stereo_event(bus, synth_clean_tick(rng, amp=0.046, pan=-pan * 0.8, accent=False, bright=1.08), st)
        if k % 4 == 0:
            add_stereo_event(bus, synth_sub_pulse(38 if (k // 4) % 4 != 3 else 34, dur_s=0.26, amp=0.22 + 0.04 * prog), st)
        st += step
        k += 1

    add_stereo_event(bus, synth_glass_ping(86, dur_s=1.05, amp=0.072, pan=0.0), t_peak)
    add_stereo_event(bus, synth_tension_riser(rng, 1.30, 180.0, 3200.0, amp=0.11), t1 - 1.30)


def compose_beat4(bus: np.ndarray, rng: np.random.Generator) -> None:
    """Beat 4 (15.00-21.00s): Ceramic Shatter & Dyneema — peak Phrygian urgency & impact boom."""
    t0, t1 = beat_span("B4_CERAMIC")
    dur = t1 - t0
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    t_hit = ev_s("b4_rifle_impact")
    t_shat = ev_s("b4_ceramic_shatter")
    t_catch = ev_s("b4_dyneema_catch")

    add_stereo_event(bus, synth_sub_drone(rng, dur, root_midi=38.0, amp=0.25, drift_hz=0.45), t0)
    dark_chord = dark_pad(t, [38, 45, 50, 51, 53, 57], [0.30, 0.24, 0.24, 0.14, 0.20, 0.18], detune_cents=10.5, cutoff_hz=1120.0)
    dark_env = (make_env(n, 0.14, 0.22) * (0.14 + 0.08 * np.sin(math.pi * np.clip(t / dur, 0.0, 1.0))))[:, None]
    add_stereo_event(bus, (dark_chord * dark_env).astype(np.float32), t0)

    xray_pad = butter_filter(
        dark_pad(t, [74, 81, 84, 89, 93], [0.30, 0.26, 0.22, 0.20, 0.16], detune_cents=13.0, cutoff_hz=6800.0, vibrato_hz=5.0, vibrato_cents=5.0),
        "high", 3200.0, order=2,
    )
    add_stereo_event(bus, (xray_pad * (make_env(n, 0.25, 0.35) * 0.095)[:, None]).astype(np.float32), t0)

    add_stereo_event(bus, synth_kinetic_boom(rng, amp=0.38, start_hz=112.0, end_hz=35.0, dur_s=1.45), t_shat)
    add_stereo_event(bus, synth_glass_ping(81, dur_s=1.00, amp=0.082, pan=-0.28), t_hit)
    add_stereo_event(bus, synth_glass_ping(74, dur_s=1.15, amp=0.088, pan=0.0), t_shat)
    add_stereo_event(bus, synth_glass_ping(86, dur_s=0.85, amp=0.070, pan=0.30), t_catch)

    step = BEAT_S * 0.25
    ost_notes = [50, 57, 51, 57, 50, 58, 57, 53]
    st, k = t0, 0
    while st < t1 - 0.08:
        pan = -0.36 if k % 2 == 0 else 0.36
        is_acc = (k % 4 == 0)
        fade_end = 1.0 - 0.35 * max(0.0, (st - (t1 - 0.9)) / 0.9)
        if k % 2 == 0 or (t_hit - 0.02 <= st <= t_catch + 1.1):
            add_stereo_event(bus, synth_clean_tick(rng, amp=(0.090 if is_acc else 0.054) * fade_end, pan=pan, accent=is_acc, bright=1.06), st)
        if k % 2 == 0:
            add_stereo_event(bus, synth_muted_pluck(ost_notes[(k // 2) % len(ost_notes)], dur_s=0.16, amp=0.105 * fade_end, cutoff_hz=760.0, pan=-pan * 0.6), st)
        if k % 4 == 0:
            add_stereo_event(bus, synth_sub_pulse(38 if (k // 4) % 2 == 0 else 37, dur_s=0.24, amp=0.23 * fade_end), st)
        st += step
        k += 1


def compose_beat5(bus: np.ndarray, rng: np.random.Generator) -> None:
    """Beat 5 (21.00-25.955s): Newton's Third Law Blueprint — analytical pulse tapering into 45 ms breath gap."""
    t0, _ = beat_span("B5_NEWTON")
    t_fire = ev_s("b5_dual_fire")
    t_hit = ev_s("b5_vest_absorb")
    t_lock = ev_s("b5_vectors_lock")
    t_end = BREATH_START_S
    dur = t_end - t0
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR

    soft_tail = (1.0 - 0.65 * np.clip((t - 3.2) / max(0.2, dur - 3.2), 0.0, 1.0)) * np.minimum(1.0, (dur - t) / 0.06)
    add_stereo_event(bus, (synth_sub_drone(rng, dur, root_midi=38.0, amp=0.20, drift_hz=0.30) * soft_tail[:, None]).astype(np.float32), t0)

    prog_u = np.clip((t - (t_lock - t0)) / 1.2, 0.0, 1.0)[:, None]
    pad_a = dark_pad(t, [50, 57, 60, 65], [0.30, 0.26, 0.22, 0.22], detune_cents=7.0, cutoff_hz=920.0)
    pad_b = dark_pad(t, [48, 55, 62, 67], [0.28, 0.26, 0.24, 0.22], detune_cents=7.5, cutoff_hz=980.0)
    pad = pad_a * (1.0 - prog_u) + pad_b * prog_u
    add_stereo_event(bus, (pad * (make_env(n, 0.14, 0.08) * soft_tail * 0.15)[:, None]).astype(np.float32), t0)

    notes = [50, 57, 60, 65, 50, 57, 64, 60]
    st, k = t0, 0
    while st < t_end - 0.45:
        fade_out = 1.0 - 0.72 * np.clip((st - (t_end - 1.50)) / 1.15, 0.0, 1.0)
        pan = -0.28 if k % 2 == 0 else 0.28
        add_stereo_event(bus, synth_muted_pluck(notes[k % len(notes)], dur_s=0.20, amp=0.105 * fade_out, cutoff_hz=860.0, pan=pan), st)
        if k % 2 == 0:
            add_stereo_event(bus, synth_sub_pulse(38 if k % 4 == 0 else 36, dur_s=0.24, amp=0.19 * fade_out), st)
        if st < t_lock + 0.60:
            add_stereo_event(bus, synth_clean_tick(rng, amp=0.060 * fade_out, pan=-pan, accent=(k % 2 == 0)), st)
        st += BEAT_S * 0.5
        k += 1

    add_stereo_event(bus, synth_sub_pulse(38, dur_s=0.25, amp=0.22, cutoff_hz=260.0), t_fire)
    add_stereo_event(bus, synth_kinetic_boom(rng, amp=0.24, start_hz=88.0, end_hz=36.7, dur_s=0.75), t_hit)
    add_stereo_event(bus, synth_glass_ping(81, dur_s=0.90, amp=0.068, pan=0.0), t_lock)


def compose_beat6(bus: np.ndarray, rng: np.random.Generator) -> None:
    """Beat 6 (26.00-30.00s): Reality Payoff — calm Bbmaj7 -> Dm(add9) resolution after 45 ms breath gap."""
    t0, t1 = beat_span("B6_FINAL")
    t_enter = max(BREATH_END_S, t0)
    t_steady = ev_s("b6_hero_steady")
    dur = t1 - t_enter
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR

    tail_start = 29.55 - t_enter
    tail_u = np.clip((t - tail_start) / max(0.05, dur - tail_start), 0.0, 1.0)
    outro_fade = (0.5 * (1.0 + np.cos(math.pi * tail_u))).astype(np.float32)

    add_stereo_event(bus, (synth_sub_drone(rng, dur, root_midi=38.0, amp=0.19, drift_hz=0.22) * outro_fade[:, None]).astype(np.float32), t_enter)

    prog_u = 0.5 * (1.0 + np.tanh((t - 1.85) / 0.50))
    f_b1 = midi_to_hz(34) * (1.0 - prog_u) + midi_to_hz(38) * prog_u
    f_b2 = midi_to_hz(46) * (1.0 - prog_u) + midi_to_hz(50) * prog_u
    p_b1 = 2.0 * math.pi * np.cumsum(f_b1.astype(np.float64)) / SR
    p_b2 = 2.0 * math.pi * np.cumsum(f_b2.astype(np.float64)) / SR
    bass = (0.62 * np.sin(p_b1).astype(np.float32) + 0.34 * np.sin(p_b2).astype(np.float32)) * np.minimum(1.0, t / 0.05) * outro_fade * 0.22
    add_stereo_event(bus, pan_stereo(bass.astype(np.float32), 0.0), t_enter)

    pad_bb = dark_pad(t, [46, 53, 57, 62, 65, 69], [0.28, 0.24, 0.22, 0.22, 0.20, 0.16], detune_cents=9.0, cutoff_hz=1650.0, vibrato_hz=3.2, vibrato_cents=4.5)
    pad_dm = dark_pad(t, [50, 57, 64, 65, 69, 72], [0.28, 0.24, 0.22, 0.22, 0.20, 0.16], detune_cents=9.0, cutoff_hz=1750.0, vibrato_hz=3.2, vibrato_cents=4.5)
    pad = pad_bb * (1.0 - prog_u[:, None]) + pad_dm * prog_u[:, None]
    add_stereo_event(bus, (pad * (np.minimum(1.0, t / 0.14) * outro_fade * 0.21)[:, None]).astype(np.float32), t_enter)

    sky_bb = dark_pad(t, [81, 86, 89], [0.34, 0.32, 0.28], detune_cents=10.0, cutoff_hz=5800.0)
    sky_dm = dark_pad(t, [81, 84, 88], [0.34, 0.32, 0.28], detune_cents=10.0, cutoff_hz=5800.0)
    sky = butter_filter(sky_bb * (1.0 - prog_u[:, None]) + sky_dm * prog_u[:, None], "high", 3300.0, order=2)
    add_stereo_event(bus, (sky * (np.minimum(1.0, t / 0.25) * outro_fade * 0.082)[:, None]).astype(np.float32), t_enter)

    add_stereo_event(bus, synth_sub_pulse(46, dur_s=0.45, amp=0.22, cutoff_hz=210.0), t_enter)
    add_stereo_event(bus, synth_glass_ping(81, dur_s=1.35, amp=0.072, pan=-0.26), t_steady)
    add_stereo_event(bus, synth_glass_ping(86, dur_s=1.40, amp=0.066, pan=0.28), t_steady + 0.14)
    add_stereo_event(bus, synth_sub_pulse(38, dur_s=0.50, amp=0.18, cutoff_hz=190.0), t_enter + 1.85)


def enforce_breath_window(bus: np.ndarray, start_s: float, end_s: float, fade_s: float = 0.008) -> None:
    i0 = max(0, int(round(start_s * SR)))
    i1 = min(len(bus), int(round(end_s * SR)))
    nf = max(1, int(round(fade_s * SR)))
    if i0 > nf:
        u = np.linspace(0.0, 1.0, nf, dtype=np.float32)
        bus[i0 - nf : i0] *= (0.5 * (1.0 + np.cos(math.pi * u)))[:, None]
    if i1 > i0:
        bus[i0:i1] = 0.0
    if i1 + nf < len(bus):
        u = np.linspace(0.0, 1.0, nf, dtype=np.float32)
        bus[i1 : i1 + nf] *= (0.5 * (1.0 - np.cos(math.pi * u)))[:, None]


def vocal_pocket_filters() -> list[tuple[float, float, float]]:
    """Vocal-pocket EQ bands (250-3500 Hz, -3.5 dB peak cut) for BrianNeural clarity."""
    return [
        (460.0, 0.90, -2.75),
        (1200.0, 0.95, -1.85),
        (2500.0, 0.90, -2.75),
    ]


def apply_vocal_pocket_eq(bus: np.ndarray) -> np.ndarray:
    for freq, q, gain_db in vocal_pocket_filters():
        bus = peaking_eq(bus, freq, q, gain_db)
    return bus


def apply_smooth_20ms_edges(bus: np.ndarray) -> None:
    n_fade = max(1, int(round(FADE_IO_S * SR)))
    u = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
    bus[:n_fade] *= (0.5 * (1.0 - np.cos(math.pi * u)))[:, None]
    bus[-n_fade:] *= (0.5 * (1.0 + np.cos(math.pi * u)))[:, None]
    bus[0, :] = 0.0
    bus[-1, :] = 0.0


def compose_bgm() -> np.ndarray:
    bus = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)
    rng = np.random.default_rng(20261129)

    compose_beat1(bus, rng)
    compose_beat2(bus, rng)
    compose_beat3(bus, rng)
    compose_beat4(bus, rng)
    compose_beat5(bus, rng)
    compose_beat6(bus, rng)

    bus = butter_filter(bus, "high", 26.0, order=2)
    bus = apply_vocal_pocket_eq(bus)
    bus = np.tanh(bus * 1.10).astype(np.float32)
    peak = float(np.max(np.abs(bus)))
    if peak > 1e-6:
        bus *= 0.80 / peak

    enforce_breath_window(bus, BREATH_START_S, BREATH_END_S, fade_s=0.006)
    i_tail = int(round(29.55 * SR))
    tail_len = len(bus) - i_tail
    if tail_len > 0:
        u_tail = np.linspace(0.0, 1.0, tail_len, dtype=np.float32)
        bus[i_tail:] *= (0.5 * (1.0 + np.cos(math.pi * u_tail)))[:, None]
    apply_smooth_20ms_edges(bus)
    return bus.astype(np.float32)


def measure_vocal_pocket_peak_cut_db() -> float:
    freqs = np.geomspace(250.0, 3500.0, 512)
    w = 2.0 * math.pi * freqs / SR
    mag_db = np.zeros_like(freqs)
    for freq, q, gain_db in vocal_pocket_filters():
        b, a = peaking_eq_coeffs(freq, q, gain_db)
        _, h = signal.freqz(b, a, worN=w)
        mag_db += 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))
    return float(np.min(mag_db))


def verify_stem(path: Path, audio: np.ndarray) -> None:
    info = sf.info(str(path))
    assert info.samplerate == SR and info.channels == CHANNELS and info.frames == TOTAL_SAMPLES
    assert info.subtype == "PCM_16" and abs(info.duration - DURATION_S) < 1e-6

    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == SR and wf.getnchannels() == CHANNELS and wf.getsampwidth() == 2

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    assert 0.02 < rms < 0.60 and peak <= 0.82

    min_eq_db = measure_vocal_pocket_peak_cut_db()
    assert -3.9 <= min_eq_db <= -3.1, f"Vocal-pocket EQ out of spec: {min_eq_db:.2f} dB"

    b0 = int(round((BREATH_START_S + 0.004) * SR))
    b1 = int(round((BREATH_END_S - 0.004) * SR))
    breath_rms = float(np.sqrt(np.mean(audio[b0:b1].astype(np.float64) ** 2)))
    assert breath_rms < 1e-4, f"Breath window not clean: {breath_rms:.6f}"
    assert float(np.max(np.abs(audio[:4]))) < 1e-4 and float(np.max(np.abs(audio[-4:]))) < 1e-4

    print(
        f"[BGM] Wrote {path.name}: {info.samplerate} Hz, {info.channels}ch, {info.subtype}, "
        f"{info.frames} samples ({info.duration:.3f}s), peak={peak:.4f}, rms={rms:.4f}, "
        f"eq_peak_cut={min_eq_db:.2f} dB, breath_rms={breath_rms:.6f}"
    )


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audio = compose_bgm()
    sf.write(str(OUTPUT_PATH), audio, SR, subtype="PCM_16")
    verify_stem(OUTPUT_PATH, audio)


if __name__ == "__main__":
    main()
