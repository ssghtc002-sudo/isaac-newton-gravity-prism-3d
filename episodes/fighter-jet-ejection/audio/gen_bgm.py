"""
BGM Composer for Episode #1 ("Why Ejecting From a Fighter Jet Shrinks You by 1 Inch", 30.0 s @ 60 fps).

Generates:
    episodes/fighter-jet-ejection/audio/output/bgm_stem.wav
    - 48000 Hz, Stereo, PCM_16, exact 30.000 s (1,440,000 samples)
    - Synced strictly to ejection_contract.BEATS and ejection_contract.EV
    - Broadcast vocal-pocket EQ attenuation across 250-3500 Hz (-4.5 dB @ 880 Hz, -3.5 dB @ 2450 Hz)
    - Peak safely bounded <= 0.95 (target <= 0.85), zero clipping
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

import ejection_contract as C  # noqa: E402

SR = 48000
CHANNELS = 2
DURATION_S = float(C.DURATION_S)
TOTAL_SAMPLES = int(round(DURATION_S * SR))

BPM = 128.0
BEAT_S = 60.0 / BPM

BREATH_START_S = 25.440
BREATH_DUR_S = 0.042
BREATH_END_S = BREATH_START_S + BREATH_DUR_S

OUTPUT_PATH = AUDIO_DIR / "output" / "bgm_stem.wav"


def beat_span(beat_idx: int) -> tuple[float, float]:
    """Return (start_s, end_s) for beat_idx from ejection_contract.BEATS."""
    f0, f1 = C.BEATS[beat_idx]
    return ((f0 - 1) / float(C.FPS), f1 / float(C.FPS))


def ev_s(event_key: str) -> float:
    """Return event timestamp in seconds from ejection_contract.EV."""
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
    """Zero-surprise Butterworth SOS filter along time axis=0."""
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
    rolloff: float = 0.80,
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


def stereo_brass_pad(
    t: np.ndarray,
    midi_notes: list[float],
    weights: list[float],
    detune_cents: float = 7.5,
    max_harmonics: int = 14,
) -> np.ndarray:
    """Detuned L/R lush analog brass/string pad across a list of MIDI notes."""
    ratio_up = 2.0 ** (detune_cents / 1200.0)
    ratio_dn = 2.0 ** (-detune_cents / 1200.0)
    left = np.zeros(len(t), dtype=np.float32)
    right = np.zeros(len(t), dtype=np.float32)
    for i, (m, w) in enumerate(zip(midi_notes, weights)):
        f0 = midi_to_hz(m)
        f_l = f0 * (ratio_up if i % 2 == 0 else ratio_dn)
        f_r = f0 * (ratio_dn if i % 2 == 0 else ratio_up)
        left += w * band_limited_saw(f_l, t, max_harmonics=max_harmonics, rolloff=0.92)
        right += w * band_limited_saw(f_r, t, max_harmonics=max_harmonics, rolloff=0.92)
    return np.stack([left, right], axis=-1).astype(np.float32)


def synth_kick(amp: float = 0.22, punch_hz: float = 92.0, tail_hz: float = 36.7) -> np.ndarray:
    """Tight tactical cinema sub-kick tuned to D1/D0."""
    n = int(0.18 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    f_inst = punch_hz * np.exp(-t * 28.0) + tail_hz
    phase = 2.0 * math.pi * np.cumsum(f_inst.astype(np.float64)) / SR
    click = np.sin(2.0 * math.pi * 240.0 * t) * np.exp(-t * 85.0) * 0.22
    body = np.sin(phase).astype(np.float32) * np.exp(-t * 15.0)
    sig = np.tanh((body + click) * 1.15) * make_env(n, 0.0015, 0.03) * amp
    return pan_stereo(sig.astype(np.float32), 0.0)


def synth_hat(rng: np.random.Generator, amp: float = 0.05, pan: float = 0.0, decay: float = 85.0) -> np.ndarray:
    """Crisp high-passed aviation telemetry tick / air-hat."""
    n = int(0.055 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    noise = rng.standard_normal(n).astype(np.float32)
    tick = butter_filter(noise, "high", 5800.0, order=2) * np.exp(-t * decay) * amp
    tick *= np.minimum(1.0, t / 0.001)
    return pan_stereo(tick.astype(np.float32), pan)


def synth_pluck(
    midi: float,
    dur_s: float = 0.19,
    amp: float = 0.15,
    decay: float = 12.0,
    pan: float = 0.0,
    brightness: float = 1.0,
) -> np.ndarray:
    """Tactile analog synth pulse stab with smooth exponential envelope and crisp top harmonics."""
    n = int(dur_s * SR)
    t = np.arange(n, dtype=np.float32) / SR
    f = midi_to_hz(midi)
    body = (
        0.52 * np.sin(2.0 * math.pi * f * t)
        + 0.24 * np.sin(4.0 * math.pi * f * t) * np.exp(-t * 13.0)
        + 0.24 * brightness * band_limited_saw(f, t, max_harmonics=12, rolloff=0.95) * np.exp(-t * 18.0)
    )
    env = np.minimum(1.0, t / 0.003) * np.exp(-t * decay) * np.minimum(1.0, (dur_s - t) / 0.015)
    return pan_stereo((body * env * amp).astype(np.float32), pan)


def synth_sonar_ping(midi: float, dur_s: float = 0.85, amp: float = 0.11, pan: float = 0.0) -> np.ndarray:
    """Pure medical/X-ray harmonic sonar ping with subtle octave shimmer."""
    n = int(dur_s * SR)
    t = np.arange(n, dtype=np.float32) / SR
    f = midi_to_hz(midi)
    ping = (
        0.72 * np.sin(2.0 * math.pi * f * t)
        + 0.20 * np.sin(2.0 * math.pi * (f * 2.001) * t) * np.exp(-t * 5.0)
        + 0.08 * np.sin(2.0 * math.pi * (f * 3.0) * t) * np.exp(-t * 9.0)
    )
    env = np.minimum(1.0, t / 0.006) * np.exp(-t * 3.8) * np.minimum(1.0, (dur_s - t) / 0.04)
    return pan_stereo((ping * env * amp).astype(np.float32), pan)


def synth_riser_noise(
    rng: np.random.Generator,
    dur_s: float,
    f_start: float,
    f_end: float,
    amp: float,
    curve_pow: float = 1.6,
) -> np.ndarray:
    """Sweeping stereo filtered noise riser / jet-afterburner propulsion wash."""
    n = int(round(dur_s * SR))
    if n <= 0:
        return np.zeros((0, CHANNELS), dtype=np.float32)
    t = np.arange(n, dtype=np.float32) / SR
    u = np.clip(t / max(1e-4, dur_s), 0.0, 1.0) ** curve_pow

    # Crossfade between 4 progressively brighter bandpass bands to create a clean smooth sweep
    bands = [
        (f_start, f_start * 1.9),
        (f_start * 1.6, (f_start + f_end) * 0.55),
        ((f_start + f_end) * 0.45, f_end * 0.85),
        (f_end * 0.70, min(9500.0, f_end * 1.35)),
    ]
    noise_l = rng.standard_normal(n).astype(np.float32)
    noise_r = rng.standard_normal(n).astype(np.float32)
    st = np.stack([noise_l, noise_r], axis=-1)

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


def compose_beat1(bus: np.ndarray, rng: np.random.Generator) -> None:
    """Beat 1 (0.0-4.8s): 128 BPM tactical aviation pulse in D minor + riser into 2.5s handle yank."""
    t0, t1 = beat_span(1)
    dur = t1 - t0
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    t_yank = ev_s("yank_start")
    t_reach = ev_s("reach_start")
    t_spark = ev_s("cord_spark")

    # D1/D2 tactical bass bed + rhythmic drive
    f_d1 = midi_to_hz(38)  # D2 = 73.42 Hz
    f_d0 = midi_to_hz(26)  # D1 = 36.71 Hz
    bed = (
        0.54 * np.sin(2.0 * math.pi * f_d0 * t)
        + 0.34 * band_limited_saw(f_d1, t, max_harmonics=8, rolloff=1.0)
        + 0.14 * np.sin(2.0 * math.pi * midi_to_hz(45) * t)
    )
    bed_env = make_env(n, 0.05, 0.14) * (0.95 + 0.28 * np.clip((t - t_reach) / 1.5, 0.0, 1.0))
    add_stereo_event(bus, pan_stereo((bed * bed_env * 0.22).astype(np.float32), 0.0), t0)

    # 128 BPM 8th-note driving pulse + 16th syncopations
    step = BEAT_S * 0.5
    notes = [50, 62, 53, 57, 50, 60, 57, 53]  # Dm9 tactical motif
    st = t0
    k = 0
    while st < t1 - 0.10:
        amp_scale = 1.0 if st < t_yank else 0.90
        m = notes[k % len(notes)]
        pan = -0.32 if k % 2 == 0 else 0.32
        add_stereo_event(
            bus,
            synth_pluck(m, dur_s=0.18, amp=(0.20 if k % 4 == 0 else 0.14) * amp_scale, decay=13.0, pan=pan, brightness=1.25),
            st,
        )
        # Driving sub-bass octave pulse on off-beats
        if k % 2 == 1:
            add_stereo_event(
                bus,
                synth_pluck(38 if k % 4 == 1 else 41, dur_s=0.16, amp=0.18 * amp_scale, decay=12.0, pan=0.0, brightness=0.95),
                st,
            )
        if k % 2 == 0:
            add_stereo_event(bus, synth_kick(amp=(0.28 if k % 4 == 0 else 0.17) * amp_scale), st)
        add_stereo_event(bus, synth_hat(rng, amp=0.068 if k % 2 == 1 else 0.036, pan=-pan), st)
        if st >= t_reach and k % 2 == 1:
            # Extra 16th-note tension double-hat building into handle yank
            add_stereo_event(bus, synth_hat(rng, amp=0.044, pan=pan * 0.8, decay=105.0), st + step * 0.5)
        st += step
        k += 1

    # Tension riser from reach_start (1.0s) into violent yank_start (2.5s)
    riser_dur = max(0.2, t_yank - t_reach)
    add_stereo_event(bus, synth_riser_noise(rng, riser_dur, 320.0, 4200.0, amp=0.13, curve_pow=1.8), t_reach)
    rn = int(round(riser_dur * SR))
    rt = np.arange(rn, dtype=np.float32) / SR
    ru = rt / riser_dur
    r_freq = midi_to_hz(50) * (2.0 ** ru)
    r_phase = 2.0 * math.pi * np.cumsum(r_freq.astype(np.float64)) / SR
    r_sig = np.sin(r_phase).astype(np.float32) * (ru ** 1.7) * np.minimum(1.0, (riser_dur - rt) / 0.02) * 0.10
    add_stereo_event(bus, pan_stereo(r_sig, 0.0), t_reach)

    # Impact hit at yank_start (2.50s)
    add_stereo_event(bus, synth_kick(amp=0.32, punch_hz=112.0, tail_hz=35.0), t_yank)
    add_stereo_event(bus, synth_pluck(50, dur_s=0.32, amp=0.22, decay=6.5, pan=0.0, brightness=1.3), t_yank)

    # MDC spark anticipation shimmer into Beat 2 (4.70s -> 4.80s)
    spark_dur = max(0.08, t1 - t_spark + 0.04)
    add_stereo_event(bus, synth_riser_noise(rng, spark_dur, 1800.0, 7200.0, amp=0.11, curve_pow=1.2), t_spark - 0.04)


def compose_beat2(bus: np.ndarray, rng: np.random.Generator) -> None:
    """Beat 2 (4.8-9.8s): Matrix bullet-time glass-shatter atmospheric drop + crystalline wide stereo pad."""
    t0, t1 = beat_span(2)
    dur = t1 - t0
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    t_det = ev_s("mdc_detonate")

    # Deep bullet-time sub-drop glide (D2 -> Bb1) right at canopy detonation (5.0s)
    f_drop = midi_to_hz(38) * (1.0 - 0.206 * np.clip((t - (t_det - t0)) / 1.2, 0.0, 1.0))  # D2 -> Bb1
    drop_phase = 2.0 * math.pi * np.cumsum(f_drop.astype(np.float64)) / SR
    sub = 0.74 * np.sin(drop_phase).astype(np.float32) + 0.20 * np.sin(2.0 * drop_phase).astype(np.float32)
    sub_env = make_env(n, 0.06, 0.30) * (0.60 + 0.42 * np.exp(-np.maximum(0.0, t - (t_det - t0)) * 0.8))
    add_stereo_event(bus, pan_stereo((sub * sub_env * 0.18).astype(np.float32), 0.0), t0)

    # Bullet-time drop impact thud at mdc_detonate (5.0s)
    add_stereo_event(bus, synth_kick(amp=0.25, punch_hz=76.0, tail_hz=32.0), t_det)

    # Wide crystalline stereo synth pad (Bbmaj7 / Dm9 shimmer with bright >3.5 kHz air)
    pad_notes = [(62, 0.28), (69, 0.28), (74, 0.28), (76, 0.24), (81, 0.22), (86, 0.16)]  # up to D6 crystalline shimmer!
    pad_l = np.zeros(n, dtype=np.float32)
    pad_r = np.zeros(n, dtype=np.float32)
    shimmer = 1.0 + 0.16 * np.sin(2.0 * math.pi * 1.45 * t)
    for idx, (m, w) in enumerate(pad_notes):
        f = midi_to_hz(m)
        det = 1.0 + (0.0042 if idx % 2 == 0 else -0.0042)
        pad_l += w * (0.62 * np.sin(2.0 * math.pi * f * t) + 0.28 * np.sin(2.0 * math.pi * (f * 2.0 * det) * t) + 0.10 * np.sin(2.0 * math.pi * (f * 3.0) * t))
        pad_r += w * (0.62 * np.sin(2.0 * math.pi * (f * det) * t) + 0.28 * np.sin(2.0 * math.pi * (f * 2.0) * t) + 0.10 * np.sin(2.0 * math.pi * (f * 3.0 * det) * t))
    pad_env = make_env(n, 0.35, 0.35) * shimmer * 0.13
    add_stereo_event(bus, np.stack([pad_l * pad_env, pad_r * pad_env], axis=-1).astype(np.float32), t0)

    # Half-time crystalline glass-chime pulses floating across stereo field
    step = BEAT_S * 1.0
    chimes = [74, 86, 81, 77, 88, 81]
    st = t0 + 0.25
    k = 0
    while st < t1 - 0.35:
        pan = -0.60 if k % 2 == 0 else 0.60
        add_stereo_event(bus, synth_sonar_ping(chimes[k % len(chimes)], dur_s=0.95, amp=0.075, pan=pan), st)
        if k % 2 == 0:
            add_stereo_event(bus, synth_pluck(34 if k >= 2 else 38, dur_s=0.45, amp=0.10, decay=4.5, pan=0.0, brightness=0.45), st)
        st += step
        k += 1

    # Subtle glass-air shimmer tail leading into Beat 3
    add_stereo_event(bus, synth_riser_noise(rng, 1.2, 800.0, 4800.0, amp=0.075, curve_pow=1.4), t1 - 1.2)


def compose_beat3(bus: np.ndarray, rng: np.random.Generator) -> None:
    """Beat 3 (9.8-14.8s): Accelerating mechanical rhythmic drive (garters + catapult ignition)."""
    t0, t1 = beat_span(3)
    t_cat = ev_s("catapult_ignite")  # 11.667 s
    dur = t1 - t0
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR

    # Mechanical bass bed shifting F1 (MIDI 41) -> G1 (MIDI 43) at catapult ignition
    f_pre = midi_to_hz(41)
    f_post = midi_to_hz(43)
    blend = np.clip((t - (t_cat - t0)) / 0.18, 0.0, 1.0)
    f_bass = f_pre * (1.0 - blend) + f_post * blend
    b_phase = 2.0 * math.pi * np.cumsum(f_bass.astype(np.float64)) / SR
    drone = (
        0.56 * np.sin(b_phase).astype(np.float32)
        + 0.28 * band_limited_saw(f_bass, t, max_harmonics=7, rolloff=1.05)
        + 0.16 * np.sin(1.5 * b_phase).astype(np.float32)
    )
    drone_env = make_env(n, 0.12, 0.18) * (0.14 + 0.05 * blend)
    add_stereo_event(bus, pan_stereo((drone * drone_env).astype(np.float32), 0.0), t0)

    # Phase 3A (9.80 -> 11.67s): tight 8th-note mechanical garter lock groove
    st = t0
    k = 0
    step_a = BEAT_S * 0.5
    notes_a = [41, 53, 48, 53, 41, 55, 53, 48]
    while st < t_cat - 0.06:
        pan = -0.34 if k % 2 == 0 else 0.34
        add_stereo_event(
            bus,
            synth_pluck(notes_a[k % len(notes_a)], dur_s=0.16, amp=0.15 if k % 2 == 0 else 0.11, decay=15.0, pan=pan, brightness=1.15),
            st,
        )
        if k % 2 == 0:
            add_stereo_event(bus, synth_kick(amp=0.22 if k % 4 == 0 else 0.14), st)
        add_stereo_event(bus, synth_hat(rng, amp=0.058, pan=-pan, decay=115.0), st)
        st += step_a
        k += 1

    # Catapult ignition impact + tight ramp
    add_stereo_event(bus, synth_kick(amp=0.29, punch_hz=104.0, tail_hz=36.0), t_cat)
    add_stereo_event(bus, synth_riser_noise(rng, t1 - t_cat, 300.0, 3800.0, amp=0.095, curve_pow=1.3), t_cat)

    # Phase 3B (11.67 -> 14.80s): accelerating 16th-note catapult drive (0.42 -> 0.26 beat step)
    st = t_cat
    k = 0
    notes_b = [43, 55, 50, 58, 55, 62, 58, 55]
    while st < t1 - 0.08:
        prog = (st - t_cat) / max(0.1, t1 - t_cat)
        step_b = BEAT_S * (0.42 - 0.16 * prog)
        pan = -0.38 if k % 2 == 0 else 0.38
        add_stereo_event(
            bus,
            synth_pluck(notes_b[k % len(notes_b)], dur_s=0.14, amp=0.15 if k % 2 == 0 else 0.11, decay=16.0, pan=pan, brightness=1.25),
            st,
        )
        if k % 2 == 0:
            add_stereo_event(bus, synth_kick(amp=0.22 if k % 4 == 0 else 0.14), st)
        add_stereo_event(bus, synth_hat(rng, amp=0.060, pan=-pan, decay=105.0), st)
        st += step_b
        k += 1


def compose_beat4(bus: np.ndarray, rng: np.random.Generator) -> None:
    """Beat 4 (14.8-20.6s): Heavy 16-G sub-bass pressure drone + tense sonar/X-ray harmonics."""
    t0, t1 = beat_span(4)
    dur = t1 - t0
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR

    # Evaluate physical spine compression & G-load curves directly from ejection_contract!
    frames = (t0 + t) * C.FPS
    comp_norm = np.array([C.spine_compression_cm(float(f)) / C.SPINE_MAX_COMPRESSION_CM for f in frames], dtype=np.float32)
    g_norm = np.array([min(1.0, max(0.0, (C.g_load(float(f)) - 6.0) / 10.0)) for f in frames], dtype=np.float32)

    # 16-G sub-bass pressure drone: D1 sub (36.7 Hz) bending slightly downward under spinal compression!
    f_sub = midi_to_hz(26) * (1.0 - 0.024 * comp_norm)
    f_mid = midi_to_hz(38) * (1.0 - 0.024 * comp_norm)
    p_sub = 2.0 * math.pi * np.cumsum(f_sub.astype(np.float64)) / SR
    p_mid = 2.0 * math.pi * np.cumsum(f_mid.astype(np.float64)) / SR
    pressure_drone = (
        0.58 * np.sin(p_sub).astype(np.float32)
        + 0.30 * np.sin(p_mid).astype(np.float32)
        + 0.14 * band_limited_saw(f_mid, t, max_harmonics=5, rolloff=1.2)
    )
    drone_env = make_env(n, 0.16, 0.18) * (0.17 + 0.07 * g_norm)
    add_stereo_event(bus, pan_stereo((pressure_drone * drone_env).astype(np.float32), 0.0), t0)

    # Eerie wide medical X-ray harmonic shimmer pad (D5 + G5 + Bb5 + high D6 tension cluster)
    xray_pad = stereo_brass_pad(t, [74, 79, 82, 86], [0.34, 0.28, 0.24, 0.16], detune_cents=9.5, max_harmonics=4)
    xray_env = make_env(n, 0.45, 0.30) * (0.045 + 0.032 * comp_norm)
    add_stereo_event(bus, (xray_pad * xray_env[:, None]).astype(np.float32), t0)

    # Tense medical sonar pings + heavy restrained 16-G heartbeat pressure pulses
    step = BEAT_S * 1.0
    sonar_notes = [74, 82, 79, 81, 86, 82]
    st = t0 + 0.15
    k = 0
    while st < t1 - 0.25:
        pan = -0.45 if k % 2 == 0 else 0.45
        add_stereo_event(bus, synth_sonar_ping(sonar_notes[k % len(sonar_notes)], dur_s=0.82, amp=0.095, pan=pan), st)
        add_stereo_event(bus, synth_kick(amp=0.22, punch_hz=68.0, tail_hz=33.0), st)
        if st + 0.19 < t1 - 0.15:
            add_stereo_event(bus, synth_kick(amp=0.14, punch_hz=62.0, tail_hz=31.0), st + 0.19)
        add_stereo_event(bus, synth_hat(rng, amp=0.042, pan=-pan, decay=120.0), st + step * 0.5)
        st += step
        k += 1


def compose_beat5(bus: np.ndarray, rng: np.random.Generator) -> None:
    """Beat 5 (20.6-25.44s): Roaring afterburner propulsion riser clearing tail fin + 40ms crisp breath at 25.44s."""
    t0, _ = beat_span(5)
    t_end = BREATH_START_S  # 25.440 s -> crisp 40ms breath right before Beat 6 / separation!
    dur = t_end - t0
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    u = np.clip(t / max(1e-4, dur), 0.0, 1.0)

    # Rising afterburner propulsion brass/saw swarm: A1 (55 Hz) -> A2 (110 Hz) controlled headroom
    f_rise = midi_to_hz(33) * (2.0 ** (0.95 * (u ** 1.35)))
    f_fifth = f_rise * 1.498
    swarm_l = band_limited_saw(f_rise, t, max_harmonics=10, rolloff=1.0) + 0.45 * band_limited_saw(f_fifth * 1.004, t, max_harmonics=8, rolloff=1.1)
    swarm_r = band_limited_saw(f_rise * 1.005, t, max_harmonics=10, rolloff=1.0) + 0.45 * band_limited_saw(f_fifth * 0.996, t, max_harmonics=8, rolloff=1.1)
    swarm = np.stack([swarm_l, swarm_r], axis=-1).astype(np.float32)
    swarm_env = make_env(n, 0.14, 0.025) * (0.075 + 0.105 * (u ** 1.4))
    add_stereo_event(bus, (swarm * swarm_env[:, None]).astype(np.float32), t0)

    # Deep sub-propulsion floor (A0/D1)
    sub_env = make_env(n, 0.12, 0.025) * (0.14 + 0.06 * u)
    sub = 0.72 * np.sin(2.0 * math.pi * midi_to_hz(33) * t) + 0.28 * np.sin(2.0 * math.pi * midi_to_hz(45) * t)
    add_stereo_event(bus, pan_stereo((sub * sub_env).astype(np.float32), 0.0), t0)

    # Roaring aerodynamic jet-afterburner noise riser peaking cleanly into 25.44s cutoff
    add_stereo_event(bus, synth_riser_noise(rng, dur, 200.0, 5200.0, amp=0.14, curve_pow=1.45), t0)

    # High-energy propulsion pulse stabs clearing tail fin (22.75s) & drogue fire (23.33s)
    step = BEAT_S * 0.375
    st = t0
    k = 0
    notes = [45, 57, 52, 60, 57, 64, 62, 69]
    while st < t_end - 0.12:
        prog = (st - t0) / max(0.1, dur)
        pan = -0.34 if k % 2 == 0 else 0.34
        add_stereo_event(
            bus,
            synth_pluck(notes[k % len(notes)], dur_s=0.12, amp=0.10 + 0.04 * prog, decay=15.0, pan=pan, brightness=1.25),
            st,
        )
        if k % 2 == 0:
            add_stereo_event(bus, synth_kick(amp=0.19 + 0.04 * prog, punch_hz=96.0, tail_hz=36.0), st)
        add_stereo_event(bus, synth_hat(rng, amp=0.058, pan=-pan, decay=95.0), st)
        st += step
        k += 1


def compose_beat6(bus: np.ndarray, rng: np.random.Generator) -> None:
    """Beat 6 (25.5-30.0s): Soaring open-sky triumphant D Major lift / heroic brass-pad finale chord."""
    t0, t1 = beat_span(6)
    t_enter = max(BREATH_END_S, t0 - 0.015)
    dur = t1 - t_enter
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    t_main_full = ev_s("main_full")  # 27.50s

    # Heroic D Major foundation bass (D1 + D2 + A2)
    bass = (
        0.58 * np.sin(2.0 * math.pi * midi_to_hz(26) * t)
        + 0.30 * np.sin(2.0 * math.pi * midi_to_hz(38) * t)
        + 0.16 * np.sin(2.0 * math.pi * midi_to_hz(45) * t)
    )
    outro_fade = np.minimum(1.0, (dur - t) / 0.55)
    bass_env = np.minimum(1.0, t / 0.04) * outro_fade * 0.25
    add_stereo_event(bus, pan_stereo((bass * bass_env).astype(np.float32), 0.0), t_enter)

    # Triumphant Dadd9 -> Gmaj7/D -> Dadd9 harmonic lift peaking as main chute blossoms at 27.50s!
    sw_bloom = 0.78 + 0.36 * np.exp(-((t - (t_main_full - t_enter)) / 1.4) ** 2)
    brass_d = stereo_brass_pad(
        t,
        midi_notes=[50, 57, 62, 66, 69, 76],  # D3, A3, D4, F#4, A4, E5 (heroic Dadd9!)
        weights=[0.30, 0.24, 0.26, 0.24, 0.22, 0.16],
        detune_cents=9.0,
        max_harmonics=14,
    )
    brass_g = stereo_brass_pad(
        t,
        midi_notes=[50, 55, 62, 67, 71, 78],  # Gmaj7/D lift!
        weights=[0.28, 0.24, 0.24, 0.22, 0.22, 0.16],
        detune_cents=9.0,
        max_harmonics=14,
    )
    # Smooth chord arch: Dadd9 (0..1.4s) -> Gmaj7/D (1.4..2.8s) -> Dadd9 resolution (2.8..4.5s)
    lift_blend = np.exp(-((t - 1.95) / 0.85) ** 2)[:, None].astype(np.float32) * 0.65
    brass_chord = brass_d * (1.0 - lift_blend) + brass_g * lift_blend

    brass_warm = butter_filter(brass_chord, "low", 2200.0, order=2)
    brass_bright = butter_filter(brass_chord, "low", 6200.0, order=2)
    bright_mix = np.clip(t / max(0.5, t_main_full - t_enter), 0.0, 1.0)[:, None]
    brass = brass_warm * (1.0 - 0.65 * bright_mix) + brass_bright * (0.65 * bright_mix)
    brass_env = (np.minimum(1.0, t / 0.16) * outro_fade * sw_bloom * 0.24)[:, None]
    add_stereo_event(bus, (brass * brass_env).astype(np.float32), t_enter)

    # Bright sunlit open-sky choir/string octave shimmer (D5, F#5, A5, D6)
    sky_pad = stereo_brass_pad(t, [74, 78, 81, 86], [0.34, 0.32, 0.28, 0.20], detune_cents=11.0, max_harmonics=5)
    sky_env = (np.minimum(1.0, t / 0.35) * outro_fade * 0.11)[:, None]
    add_stereo_event(bus, (sky_pad * sky_env).astype(np.float32), t_enter)

    # Majestic slow-motion half-time cinema impacts (Beat 6 launch @ 25.50s, separation @ 25.75s, main_full @ 27.50s)
    add_stereo_event(bus, synth_kick(amp=0.30, punch_hz=86.0, tail_hz=36.7), t0)
    add_stereo_event(bus, synth_pluck(66, dur_s=0.65, amp=0.14, decay=4.2, pan=-0.25, brightness=1.0), ev_s("separation"))
    add_stereo_event(bus, synth_kick(amp=0.27, punch_hz=78.0, tail_hz=36.7), t_main_full)
    add_stereo_event(bus, synth_sonar_ping(78, dur_s=1.4, amp=0.09, pan=0.35), t_main_full)
    add_stereo_event(bus, synth_hat(rng, amp=0.045, pan=0.0, decay=55.0), t_main_full)


def enforce_breath_window(bus: np.ndarray, start_s: float, end_s: float, fade_s: float = 0.012) -> None:
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


def compose_bgm() -> np.ndarray:
    """Compose, EQ, and normalize the 30.000s 48kHz stereo BGM stem."""
    bus = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)
    rng = np.random.default_rng(20260601)

    compose_beat1(bus, rng)
    compose_beat2(bus, rng)
    compose_beat3(bus, rng)
    compose_beat4(bus, rng)
    compose_beat5(bus, rng)
    compose_beat6(bus, rng)

    # Clean subsonic rumble cut below 28 Hz
    bus = butter_filter(bus, "high", 28.0, order=2)

    # Broadcast vocal-pocket EQ attenuation across 250-3500 Hz
    bus = peaking_eq(bus, 880.0, 0.55, -4.5)
    bus = peaking_eq(bus, 2450.0, 0.80, -3.5)

    # Warm analog saturation + headroom scaling (peak safely <= 0.85, zero clipping)
    bus = np.tanh(bus * 1.15).astype(np.float32)
    peak = float(np.max(np.abs(bus)))
    if peak > 1e-6:
        target_peak = 0.80
        bus *= target_peak / peak

    # Re-assert crisp 40ms breath at 25.44s after IIR EQ rings so breath stays ultra-clean
    enforce_breath_window(bus, BREATH_START_S, BREATH_END_S, fade_s=0.008)

    # Click-free 5ms boundary micro-fades
    micro = int(round(0.005 * SR))
    bus[:micro] *= np.linspace(0.0, 1.0, micro, dtype=np.float32)[:, None]
    bus[-micro:] *= np.linspace(1.0, 0.0, micro, dtype=np.float32)[:, None]
    return bus.astype(np.float32)


def verify_stem(path: Path, audio: np.ndarray) -> None:
    """Verify exact 48kHz stereo 30.000s WAV specs, non-zero RMS, breath window, and zero clipping."""
    info = sf.info(str(path))
    assert info.samplerate == SR, f"Expected SR={SR}, got {info.samplerate}"
    assert info.channels == CHANNELS, f"Expected {CHANNELS} channels, got {info.channels}"
    assert info.frames == TOTAL_SAMPLES, f"Expected {TOTAL_SAMPLES} frames, got {info.frames}"
    assert abs(info.duration - DURATION_S) < 1e-6, f"Expected {DURATION_S}s, got {info.duration}"

    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == SR
        assert wf.getnchannels() == CHANNELS
        assert wf.getnframes() == TOTAL_SAMPLES

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    assert 0.02 < rms < 0.60, f"Unexpected RMS: {rms:.5f}"
    assert peak <= 0.95, f"Peak exceeded 0.95 ceiling: {peak:.5f}"

    b0 = int(round((BREATH_START_S + 0.005) * SR))
    b1 = int(round((BREATH_END_S - 0.005) * SR))
    breath_rms = float(np.sqrt(np.mean(audio[b0:b1].astype(np.float64) ** 2)))
    assert breath_rms < 1e-4, f"Breath window not clean: {breath_rms:.6f}"

    print(
        f"[BGM] Wrote {path.name}: {info.samplerate} Hz, {info.channels}ch, "
        f"{info.frames} samples ({info.duration:.3f}s), peak={peak:.4f}, rms={rms:.4f}, "
        f"breath_rms={breath_rms:.6f}"
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
