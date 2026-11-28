"""
BGM Composer for Episode #2 ("Why Bullets Stop Dead Just 3 Feet Underwater", 30.0 s @ 60 fps).

Generates:
    episodes/underwater-bullets/audio/output/bgm_stem.wav
    - 48000 Hz, Stereo, PCM_16, exact 30.000 s (1,440,000 samples)
    - Synced strictly to bullet_contract.BEATS and bullet_contract.EV
    - @zackdfilms-style subtle, submerged rhythmic suspense bed in C minor:
      * C minor atmospheric synth pulse + soft hydrophone deep-water sub tone
      * Clean ticking urgency through Beats 2-4
      * Smooth breath gap before Beat 6 survival resolution
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

import bullet_contract as C  # noqa: E402

SR = 48000
CHANNELS = 2
DURATION_S = float(C.DURATION_S)
TOTAL_SAMPLES = int(round(DURATION_S * SR))

BPM = 120.0
BEAT_S = 60.0 / BPM  # 0.500 s

# Smooth 45 ms breath gap right before Beat 6 survival resolution (B6_SAFE starts at frame 1561 -> 26.000 s)
BREATH_DUR_S = 0.045
BREATH_END_S = (C.BEATS["B6_SAFE"][0] - 1) / float(C.FPS)  # 26.000 s
BREATH_START_S = BREATH_END_S - BREATH_DUR_S               # 25.955 s

FADE_IO_S = 0.020  # Mandatory smooth 20 ms fade-in / fade-out

OUTPUT_PATH = AUDIO_DIR / "output" / "bgm_stem.wav"


def beat_span(beat_key: str) -> tuple[float, float]:
    """Return (start_s, end_s) for beat_key from bullet_contract.BEATS."""
    f0, f1 = C.BEATS[beat_key]
    return ((f0 - 1) / float(C.FPS), f1 / float(C.FPS))


def ev_s(event_key: str) -> float:
    """Return event timestamp in seconds from bullet_contract.EV."""
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


def peaking_eq_coeffs(freq: float, q: float, gain_db: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute RBJ peaking EQ biquad coefficients."""
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / SR
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    norm = 1.0 + alpha / a_lin
    b = np.array([1.0 + alpha * a_lin, -2.0 * cw, 1.0 - alpha * a_lin], dtype=np.float64) / norm
    a = np.array([1.0, (-2.0 * cw) / norm, (1.0 - alpha / a_lin) / norm], dtype=np.float64)
    return b, a


def peaking_eq(audio: np.ndarray, freq: float, q: float, gain_db: float) -> np.ndarray:
    """Apply RBJ peaking EQ biquad filter along axis=0."""
    b, a = peaking_eq_coeffs(freq, q, gain_db)
    return signal.lfilter(b, a, audio.astype(np.float64), axis=0).astype(np.float32)


def make_env(n: int, attack_s: float, release_s: float) -> np.ndarray:
    """Smooth fade-in/out envelope for a buffer of length n."""
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
    """Warm submerged additive synth wave clamped safely below Nyquist."""
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
    out *= 0.58
    return out.astype(np.float32)


def submerged_pad(
    t: np.ndarray,
    midi_notes: list[float],
    weights: list[float],
    detune_cents: float = 8.0,
    cutoff_hz: float = 950.0,
    vibrato_hz: float = 0.35,
    vibrato_cents: float = 4.0,
) -> np.ndarray:
    """Lush detuned L/R underwater synth pad filtered through a warm low-pass."""
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


def synth_hydrophone_bed(
    rng: np.random.Generator,
    dur_s: float,
    root_midi: float = 36.0,  # C2 (65.4 Hz) / C1 (32.7 Hz)
    amp: float = 0.22,
    drift_hz: float = 0.28,
) -> np.ndarray:
    """
    Soft hydrophone deep-water sub tone:
    pure sub-octave C1/C2 resonance + softly undulating bandpass ocean-pressure murmur.
    """
    n = int(round(dur_s * SR))
    t = np.arange(n, dtype=np.float32) / SR
    f_root = midi_to_hz(root_midi)
    f_sub = f_root * 0.5
    # Slow underwater pressure swell modulation
    lfo = 0.72 + 0.28 * np.sin(2.0 * math.pi * drift_hz * t - math.pi * 0.35)
    lfo_r = 0.72 + 0.28 * np.sin(2.0 * math.pi * (drift_hz * 1.13) * t + 0.4)

    sub_tone = (
        0.64 * np.sin(2.0 * math.pi * f_sub * t)
        + 0.32 * np.sin(2.0 * math.pi * f_root * t)
        + 0.08 * np.sin(2.0 * math.pi * (f_root * 1.5) * t)
    )
    # Soft stereo hydrophone water-pressure texture (38-145 Hz)
    noise_l = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (36.0, 145.0), order=2)
    noise_r = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (36.0, 145.0), order=2)
    water_l = (sub_tone + 0.55 * noise_l) * lfo
    water_r = (sub_tone + 0.55 * noise_r) * lfo_r
    env = make_env(n, 0.14, 0.18) * amp
    return np.stack([water_l * env, water_r * env], axis=-1).astype(np.float32)


def synth_sub_pulse(
    midi: float = 36.0,
    dur_s: float = 0.28,
    amp: float = 0.24,
    cutoff_hz: float = 240.0,
    pan: float = 0.0,
) -> np.ndarray:
    """Submerged muted synth bass pulse (warm C minor low-passed thump)."""
    n = int(round(dur_s * SR))
    t = np.arange(n, dtype=np.float32) / SR
    f = midi_to_hz(midi)
    # Subtle downward pitch-settle for round underwater weight
    f_inst = f * (1.0 + 0.18 * np.exp(-t * 36.0))
    phase = 2.0 * math.pi * np.cumsum(f_inst.astype(np.float64)) / SR
    sig = (
        0.68 * np.sin(phase).astype(np.float32)
        + 0.24 * np.sin(2.0 * phase).astype(np.float32) * np.exp(-t * 9.0)
        + 0.12 * band_limited_wave(f, t, max_harmonics=6, rolloff=1.2) * np.exp(-t * 14.0)
    )
    sig = butter_filter(sig, "low", cutoff_hz, order=2)
    env = np.minimum(1.0, t / 0.005) * np.exp(-t * 9.5) * np.minimum(1.0, (dur_s - t) / 0.025)
    return pan_stereo((sig * env * amp).astype(np.float32), pan)


def synth_water_boom(
    rng: np.random.Generator,
    amp: float = 0.34,
    start_hz: float = 92.0,
    end_hz: float = 32.7,  # C1
    dur_s: float = 1.15,
) -> np.ndarray:
    """Deep submerged water-wall impact sub-boom (warm muffled transient + deep C1 bloom)."""
    n = int(round(dur_s * SR))
    t = np.arange(n, dtype=np.float32) / SR
    f_inst = end_hz + (start_hz - end_hz) * np.exp(-t * 16.0)
    phase = 2.0 * math.pi * np.cumsum(f_inst.astype(np.float64)) / SR
    sub = 0.78 * np.sin(phase).astype(np.float32) + 0.22 * np.sin(2.0 * phase).astype(np.float32) * np.exp(-t * 5.0)
    thump = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (45.0, 210.0), order=2) * np.exp(-t * 18.0) * 0.55
    env = np.minimum(1.0, t / 0.004) * np.exp(-t * 2.7) * np.minimum(1.0, (dur_s - t) / 0.08)
    sig = np.tanh((sub + thump) * 1.12) * env * amp
    return pan_stereo(sig.astype(np.float32), 0.0)


def synth_clean_tick(
    rng: np.random.Generator,
    amp: float = 0.080,
    pan: float = 0.0,
    accent: bool = False,
    bright: float = 1.0,
) -> np.ndarray:
    """
    Clean precision ticking urgency (>3.6 kHz crisp escapement click, zero clutter in vocal pocket).
    """
    n = int(0.055 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    base = (4520.0 if accent else 3980.0) * bright
    tone = (
        0.54 * np.sin(2.0 * math.pi * base * t) * np.exp(-t * 78.0)
        + 0.30 * np.sin(2.0 * math.pi * (base * 1.49) * t) * np.exp(-t * 105.0)
        + 0.16 * np.sin(2.0 * math.pi * (base * 2.18) * t) * np.exp(-t * 135.0)
    )
    click = butter_filter(rng.standard_normal(n).astype(np.float32), "high", 4800.0, order=2) * np.exp(-t * 155.0) * 0.40
    sig = (tone + click) * np.minimum(1.0, t / 0.0006) * np.minimum(1.0, (0.055 - t) / 0.008) * amp
    return pan_stereo(sig.astype(np.float32), pan)


def synth_sonar_ping(
    midi: float,
    dur_s: float = 1.05,
    amp: float = 0.085,
    pan: float = 0.0,
) -> np.ndarray:
    """Soft submerged hydrophone / sonar chime harmonic."""
    n = int(round(dur_s * SR))
    t = np.arange(n, dtype=np.float32) / SR
    f = midi_to_hz(midi)
    ping = (
        0.72 * np.sin(2.0 * math.pi * f * t)
        + 0.20 * np.sin(2.0 * math.pi * (f * 2.001) * t) * np.exp(-t * 4.2)
        + 0.08 * np.sin(2.0 * math.pi * (f * 3.0) * t) * np.exp(-t * 7.5)
    )
    env = np.minimum(1.0, t / 0.010) * np.exp(-t * 3.2) * np.minimum(1.0, (dur_s - t) / 0.06)
    return pan_stereo((ping * env * amp).astype(np.float32), pan)


def synth_muted_arp_pluck(
    midi: float,
    dur_s: float = 0.22,
    amp: float = 0.11,
    cutoff_hz: float = 780.0,
    pan: float = 0.0,
) -> np.ndarray:
    """Submerged atmospheric C minor synth pulse pluck."""
    n = int(round(dur_s * SR))
    t = np.arange(n, dtype=np.float32) / SR
    f = midi_to_hz(midi)
    body = (
        0.58 * np.sin(2.0 * math.pi * f * t)
        + 0.26 * np.sin(4.0 * math.pi * f * t) * np.exp(-t * 10.0)
        + 0.22 * band_limited_wave(f, t, max_harmonics=8, rolloff=1.05) * np.exp(-t * 12.0)
    )
    body = butter_filter(body, "low", cutoff_hz, order=2)
    env = np.minimum(1.0, t / 0.004) * np.exp(-t * 9.0) * np.minimum(1.0, (dur_s - t) / 0.018)
    return pan_stereo((body * env * amp).astype(np.float32), pan)


def synth_water_riser(
    rng: np.random.Generator,
    dur_s: float,
    f_start: float,
    f_end: float,
    amp: float,
    curve_pow: float = 1.65,
) -> np.ndarray:
    """Smooth submerged pressure sweep riser."""
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
    st = np.stack(
        [rng.standard_normal(n).astype(np.float32), rng.standard_normal(n).astype(np.float32)],
        axis=-1,
    )
    out = np.zeros_like(st)
    centers = np.linspace(0.0, 1.0, len(bands))
    for c, (lo, hi) in zip(centers, bands):
        lo_c = float(np.clip(lo, 45.0, 7500.0))
        hi_c = float(np.clip(max(lo_c + 40.0, hi), 90.0, 8800.0))
        filtered = butter_filter(st, "band", (lo_c, hi_c), order=2)
        w = np.exp(-((u - c) / 0.34) ** 2).astype(np.float32)
        out += filtered * w[:, None]

    env = np.minimum(1.0, t / 0.14) * (0.18 + 0.82 * u) * np.minimum(1.0, (dur_s - t) / 0.03)
    return (out * env[:, None] * amp).astype(np.float32)


def compose_beat1(bus: np.ndarray, rng: np.random.Generator) -> None:
    """
    Beat 1 (0.00-3.50s): Hook — soft hydrophone deep-water sub tone + C minor submerged synth pulse
    as bullet streaks in and halts dead at 3 FT (2.250s).
    """
    t0, t1 = beat_span("B1_HOOK")
    dur = t1 - t0
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    t_stop = ev_s("b1_bullet_stop")  # 2.250 s

    # Soft hydrophone deep-water sub bed (C1/C2)
    add_stereo_event(bus, synth_hydrophone_bed(rng, dur, root_midi=36.0, amp=0.22, drift_hz=0.35), t0)

    # C minor atmospheric submerged pad (C3, G3, Eb4, D4 -> Cm9 suspense)
    pad = submerged_pad(t, [48, 55, 63, 62], [0.34, 0.28, 0.22, 0.16], detune_cents=7.5, cutoff_hz=720.0)
    pad_env = (make_env(n, 0.28, 0.18) * 0.13)[:, None]
    add_stereo_event(bus, (pad * pad_env).astype(np.float32), t0)

    # Subtle rhythmic C minor pulse on quarter/eighth beats
    pulse_notes = [36, 36, 43, 39, 36, 36, 41, 43]  # C2, G2, Eb2, F2
    st = t0 + 0.125
    k = 0
    while st < t1 - 0.12:
        amp = 0.20 if k % 2 == 0 else 0.13
        add_stereo_event(bus, synth_sub_pulse(pulse_notes[k % len(pulse_notes)], dur_s=0.28, amp=amp, cutoff_hz=230.0), st)
        if k % 2 == 1:
            add_stereo_event(bus, synth_muted_arp_pluck(60 if k % 4 == 1 else 63, dur_s=0.24, amp=0.075, cutoff_hz=640.0, pan=0.24 if k % 4 == 1 else -0.24), st)
        st += 0.25
        k += 1

    # Bullet stops dead at 3 FT (2.250s): muffled hydrophone thud + high sonar lock ping
    add_stereo_event(bus, synth_water_boom(rng, amp=0.26, start_hz=82.0, end_hz=32.7, dur_s=0.85), t_stop)
    add_stereo_event(bus, synth_sonar_ping(79, dur_s=0.95, amp=0.078, pan=0.22), t_stop)  # G5

    # Smooth water riser into Beat 2 hard cut
    add_stereo_event(bus, synth_water_riser(rng, 1.05, 160.0, 2600.0, amp=0.095, curve_pow=1.75), t1 - 1.05)


def compose_beat2(bus: np.ndarray, rng: np.random.Generator) -> None:
    """
    Beat 2 (3.50-9.00s): 800x Density Wall — muzzle fire (4.133s), water entry slam (5.000s),
    heavy submerged C minor pulse + clean ticking urgency enters cleanly.
    """
    t0, t1 = beat_span("B2_ENTRY")
    dur = t1 - t0
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    t_fire = ev_s("b2_muzzle_fire")    # 4.133 s
    t_entry = ev_s("b2_water_entry")   # 5.000 s

    # Deep hydrophone pressure bed intensifying after water entry
    hydro = synth_hydrophone_bed(rng, dur, root_midi=36.0, amp=0.24, drift_hz=0.40)
    gate_entry = np.clip((t - (t_entry - t0 - 0.1)) / 0.35, 0.72, 1.05)[:, None]
    add_stereo_event(bus, (hydro * gate_entry).astype(np.float32), t0)

    # C minor submerged pad shifting Cm -> Ab/C -> Fm7/C
    prog = np.clip((t - (t_entry - t0)) / max(0.5, dur - (t_entry - t0)), 0.0, 1.0)[:, None]
    pad_a = submerged_pad(t, [48, 55, 60, 63], [0.32, 0.26, 0.24, 0.20], detune_cents=8.0, cutoff_hz=820.0)
    pad_b = submerged_pad(t, [44, 53, 60, 63], [0.30, 0.26, 0.24, 0.22], detune_cents=9.0, cutoff_hz=920.0)
    pad = pad_a * (1.0 - prog) + pad_b * prog
    pad_env = (make_env(n, 0.22, 0.16) * (0.11 + 0.05 * prog[:, 0]))[:, None]
    add_stereo_event(bus, (pad * pad_env).astype(np.float32), t0)

    # Muzzle fire dry tension cue (4.133s) -> Water entry sub-boom (5.000s)
    add_stereo_event(bus, synth_sub_pulse(36, dur_s=0.22, amp=0.22, cutoff_hz=320.0), t_fire)
    add_stereo_event(bus, synth_water_boom(rng, amp=0.35, start_hz=104.0, end_hz=32.7, dur_s=1.25), t_entry)
    add_stereo_event(bus, synth_sonar_ping(72, dur_s=0.90, amp=0.072, pan=-0.25), t_entry)  # C5

    # Clean ticking urgency (8th notes from t0, accented quarters after water entry)
    step = BEAT_S * 0.5  # 0.250 s
    arp_notes = [48, 55, 63, 55, 51, 58, 63, 58]  # Cm / Eb tension
    st = t0
    k = 0
    while st < t1 - 0.08:
        pan = -0.32 if k % 2 == 0 else 0.32
        is_post_entry = st >= t_entry - 0.02
        is_acc = (k % 2 == 0)
        tick_amp = (0.082 if is_acc else 0.052) * (1.05 if is_post_entry else 0.85)
        add_stereo_event(bus, synth_clean_tick(rng, amp=tick_amp, pan=pan, accent=is_acc), st)

        if k % 2 == 0:
            add_stereo_event(bus, synth_sub_pulse(36 if k % 4 == 0 else 34, dur_s=0.26, amp=0.21 if k % 4 == 0 else 0.16, cutoff_hz=240.0), st)
        if is_post_entry:
            add_stereo_event(
                bus,
                synth_muted_arp_pluck(arp_notes[k % len(arp_notes)], dur_s=0.20, amp=0.095, cutoff_hz=760.0, pan=-pan * 0.7),
                st,
            )
        st += step
        k += 1


def compose_beat3(bus: np.ndarray, rng: np.random.Generator) -> None:
    """
    Beat 3 (9.00-15.00s): Supercavitation — bubbling vapor shimmer + C minor rhythmic synth ostinato
    + relentless clean ticking urgency peaking at b3_bubble_peak (12.00s).
    """
    t0, t1 = beat_span("B3_CAVITATION")
    dur = t1 - t0
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    t_peak = ev_s("b3_bubble_peak")  # 12.000 s
    u = np.clip(t / max(1e-4, dur), 0.0, 1.0)

    # Deep hydrophone cavitation hum (C1/G1)
    add_stereo_event(bus, synth_hydrophone_bed(rng, dur, root_midi=36.0, amp=0.25, drift_hz=0.55), t0)

    # Submerged vapor shimmer pad above vocal pocket (>3.5 kHz airy harmonic mist)
    shimmer = submerged_pad(t, [72, 79, 84, 87], [0.28, 0.26, 0.24, 0.18], detune_cents=12.0, cutoff_hz=6200.0, vibrato_hz=4.2, vibrato_cents=6.0)
    shimmer = butter_filter(shimmer, "high", 3400.0, order=2)
    bell = np.exp(-((t - (t_peak - t0)) / 2.2) ** 2)
    shim_env = (make_env(n, 0.30, 0.20) * (0.045 + 0.065 * bell))[:, None]
    add_stereo_event(bus, (shimmer * shim_env).astype(np.float32), t0)

    # Warm C minor / Fm submerged chord swell underneath
    low_pad = submerged_pad(t, [36, 43, 48, 51, 55], [0.30, 0.24, 0.24, 0.20, 0.18], detune_cents=9.0, cutoff_hz=680.0)
    low_env = (make_env(n, 0.25, 0.18) * (0.14 + 0.05 * u))[:, None]
    add_stereo_event(bus, (low_pad * low_env).astype(np.float32), t0)

    # Continuous clean ticking urgency (8ths -> 16th double-ticks after bubble peak 12.0s)
    # + driving C minor submerged ostinato pulse
    step = BEAT_S * 0.25  # 0.125 s (16th grid)
    ost_notes = [48, 55, 51, 55, 48, 56, 51, 55]  # C3, G3, Eb3, G3, C3, Ab3, Eb3, G3
    st = t0
    k = 0
    while st < t1 - 0.06:
        prog = (st - t0) / dur
        pan = -0.35 if k % 2 == 0 else 0.35
        # Ticking on every 8th (even k) + urgent 16th off-ticks after t_peak
        if k % 2 == 0:
            is_acc = (k % 4 == 0)
            add_stereo_event(
                bus,
                synth_clean_tick(rng, amp=0.088 if is_acc else 0.056, pan=pan, accent=is_acc, bright=1.0 + 0.06 * prog),
                st,
            )
            add_stereo_event(
                bus,
                synth_muted_arp_pluck(ost_notes[(k // 2) % len(ost_notes)], dur_s=0.17, amp=0.105 + 0.03 * prog, cutoff_hz=720.0 + 180.0 * prog, pan=-pan * 0.65),
                st,
            )
        elif st >= t_peak - 0.02:
            add_stereo_event(bus, synth_clean_tick(rng, amp=0.048, pan=-pan * 0.8, accent=False, bright=1.08), st)

        if k % 4 == 0:
            sub_m = 36 if (k // 4) % 4 != 3 else 32  # C2 -> Ab1 turnaround
            add_stereo_event(bus, synth_sub_pulse(sub_m, dur_s=0.28, amp=0.22 + 0.04 * prog, cutoff_hz=235.0), st)
        st += step
        k += 1

    # Sonar peak accent & submerged swell riser into Beat 4 shatter
    add_stereo_event(bus, synth_sonar_ping(84, dur_s=1.10, amp=0.075, pan=0.0), t_peak)
    add_stereo_event(bus, synth_water_riser(rng, 1.35, 180.0, 3200.0, amp=0.11, curve_pow=1.75), t1 - 1.35)


def compose_beat4(bus: np.ndarray, rng: np.random.Generator) -> None:
    """
    Beat 4 (15.00-21.00s): Violent Shred & X-Ray — X-ray crystalline shimmer (15.417s),
    tumble tension (16.167s), shatter sub-boom at 3 FT (17.583s) + peak ticking urgency.
    """
    t0, t1 = beat_span("B4_SHATTER")
    dur = t1 - t0
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    t_xray = ev_s("b4_xray_on")       # 15.417 s
    t_tumble = ev_s("b4_tumble_start")  # 16.167 s
    t_shatter = ev_s("b4_shatter_burst")  # 17.583 s

    # Deep hydrophone pressure bedrock
    add_stereo_event(bus, synth_hydrophone_bed(rng, dur, root_midi=36.0, amp=0.26, drift_hz=0.45), t0)

    # Dark C minor / Phrygian tension chord bending slightly at tumble -> shatter
    shat_u = np.clip((t - (t_shatter - t0)) / 1.2, 0.0, 1.0)
    dark_chord = submerged_pad(t, [36, 43, 48, 49, 51, 55], [0.30, 0.24, 0.24, 0.14, 0.20, 0.18], detune_cents=10.5, cutoff_hz=1150.0)
    dark_env = (make_env(n, 0.14, 0.22) * (0.14 + 0.08 * np.sin(math.pi * np.clip(t / dur, 0.0, 1.0))))[:, None]
    add_stereo_event(bus, (dark_chord * dark_env).astype(np.float32), t0)

    # High crystalline X-ray shimmer (>3.2 kHz) activated during xray_on (15.417s .. 20.5s)
    xray_gate = np.clip((t - (t_xray - t0)) / 0.35, 0.0, 1.0) * np.clip((dur - 0.35 - t) / 0.45, 0.0, 1.0)
    xray_pad = submerged_pad(t, [72, 79, 82, 87, 91], [0.30, 0.26, 0.22, 0.20, 0.16], detune_cents=13.0, cutoff_hz=6800.0, vibrato_hz=5.0, vibrato_cents=5.0)
    xray_pad = butter_filter(xray_pad, "high", 3200.0, order=2)
    add_stereo_event(bus, (xray_pad * (xray_gate * 0.105)[:, None]).astype(np.float32), t0)

    # Key event hits: X-ray sonar ping, tumble pulse, massive 3 FT shatter boom
    add_stereo_event(bus, synth_sonar_ping(79, dur_s=1.05, amp=0.085, pan=-0.30), t_xray)
    add_stereo_event(bus, synth_sonar_ping(87, dur_s=0.85, amp=0.072, pan=0.32), t_tumble)
    add_stereo_event(bus, synth_water_boom(rng, amp=0.40, start_hz=112.0, end_hz=31.0, dur_s=1.65), t_shatter)
    add_stereo_event(bus, synth_sonar_ping(72, dur_s=1.20, amp=0.090, pan=0.0), t_shatter)

    # Urgent ticking pattern (16th-note ticking urgency through shatter peak, calming slightly in last 0.8s)
    step = BEAT_S * 0.25  # 0.125 s
    ost_notes = [48, 55, 49, 55, 48, 56, 55, 51]  # Cm / Db Phrygian tension
    st = t0
    k = 0
    while st < t1 - 0.08:
        pan = -0.36 if k % 2 == 0 else 0.36
        is_acc = (k % 4 == 0)
        fade_end = 1.0 - 0.35 * max(0.0, (st - (t1 - 0.9)) / 0.9)
        if k % 2 == 0 or (t_tumble - 0.02 <= st <= t_shatter + 2.2):
            add_stereo_event(
                bus,
                synth_clean_tick(rng, amp=(0.092 if is_acc else 0.056) * fade_end, pan=pan, accent=is_acc, bright=1.06),
                st,
            )
        if k % 2 == 0:
            add_stereo_event(
                bus,
                synth_muted_arp_pluck(ost_notes[(k // 2) % len(ost_notes)], dur_s=0.16, amp=0.105 * fade_end, cutoff_hz=760.0, pan=-pan * 0.6),
                st,
            )
        if k % 4 == 0:
            add_stereo_event(bus, synth_sub_pulse(36 if (k // 4) % 2 == 0 else 35, dur_s=0.25, amp=0.23 * fade_end, cutoff_hz=230.0), st)
        st += step
        k += 1


def compose_beat5(bus: np.ndarray, rng: np.random.Generator) -> None:
    """
    Beat 5 (21.00-26.00s): Paradox Race — analytical C minor studio blueprint pulse
    (21.00-24.60s), softening cleanly (24.60-25.955s) into a crisp 45 ms breath gap
    (25.955-26.000s) right before Beat 6 survival resolution!
    """
    t0, _ = beat_span("B5_RACE")
    t_start = ev_s("b5_race_start")   # 21.417 s
    t_fstop = ev_s("b5_fast_stop")    # 23.500 s
    t_scross = ev_s("b5_slow_cross")  # 25.417 s
    t_end = BREATH_START_S            # 25.955 s
    dur = t_end - t0
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    u = np.clip(t / max(1e-4, dur), 0.0, 1.0)

    # Clean hydrophone undercurrent softening smoothly toward the breath window
    hydro = synth_hydrophone_bed(rng, dur, root_midi=36.0, amp=0.20, drift_hz=0.30)
    soft_tail = (1.0 - 0.65 * np.clip((t - 3.2) / max(0.2, dur - 3.2), 0.0, 1.0)) * np.minimum(1.0, (dur - t) / 0.06)
    add_stereo_event(bus, (hydro * soft_tail[:, None]).astype(np.float32), t0)

    # Studio blueprint harmonic pad: Cm7 (21.0-23.5s) -> Fsus4/Bb (23.5-25.955s) preparing resolution
    prog_u = np.clip((t - (t_fstop - t0)) / 1.2, 0.0, 1.0)[:, None]
    pad_a = submerged_pad(t, [48, 55, 58, 63], [0.30, 0.26, 0.22, 0.22], detune_cents=7.0, cutoff_hz=920.0)
    pad_b = submerged_pad(t, [46, 53, 60, 65], [0.28, 0.26, 0.24, 0.22], detune_cents=7.5, cutoff_hz=980.0)
    pad = pad_a * (1.0 - prog_u) + pad_b * prog_u
    pad_env = (make_env(n, 0.14, 0.08) * soft_tail * 0.15)[:, None]
    add_stereo_event(bus, (pad * pad_env).astype(np.float32), t0)

    # Clean analytical comparison pulse (8th notes through 24.7s, then tapering into calm)
    race_notes = [48, 55, 58, 63, 48, 55, 62, 58]  # Cm9 clean studio pulse
    st = t0
    k = 0
    while st < t_end - 0.45:
        fade_out = 1.0 - 0.72 * np.clip((st - (t_end - 1.50)) / 1.15, 0.0, 1.0)
        pan = -0.28 if k % 2 == 0 else 0.28
        add_stereo_event(
            bus,
            synth_muted_arp_pluck(race_notes[k % len(race_notes)], dur_s=0.22, amp=0.105 * fade_out, cutoff_hz=860.0, pan=pan),
            st,
        )
        if k % 2 == 0:
            add_stereo_event(bus, synth_sub_pulse(36 if k % 4 == 0 else 34, dur_s=0.24, amp=0.19 * fade_out, cutoff_hz=230.0), st)
        if st < t_fstop + 0.60:
            add_stereo_event(bus, synth_clean_tick(rng, amp=0.062 * fade_out, pan=-pan, accent=(k % 2 == 0)), st)
        st += 0.25
        k += 1

    # Key race comparison cues
    add_stereo_event(bus, synth_sub_pulse(36, dur_s=0.25, amp=0.22, cutoff_hz=260.0), t_start)
    add_stereo_event(bus, synth_water_boom(rng, amp=0.25, start_hz=88.0, end_hz=34.0, dur_s=0.75), t_fstop)
    add_stereo_event(bus, synth_sonar_ping(79, dur_s=0.90, amp=0.068, pan=-0.28), t_fstop)
    add_stereo_event(bus, synth_sonar_ping(75, dur_s=0.80, amp=0.055, pan=0.28), t_scross)  # Eb5 soft laminar ping


def compose_beat6(bus: np.ndarray, rng: np.random.Generator) -> None:
    """
    Beat 6 (26.00-30.00s): Survival Payoff — enters cleanly after the 45 ms breath gap
    with warm C minor -> Abmaj7 -> Cm(add9) calm underwater resolution, settling gently
    with a smooth 20 ms+ tail fade at 30.000 s.
    """
    t0, t1 = beat_span("B6_SAFE")
    t_enter = max(BREATH_END_S, t0)  # 26.000 s
    t_settle = ev_s("b6_safe_settle")  # 27.000 s
    dur = t1 - t_enter
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR

    # Smooth outro fade over 29.55 -> 30.00 s
    tail_start = 29.55 - t_enter
    tail_dur = max(0.05, dur - tail_start)
    tail_u = np.clip((t - tail_start) / tail_dur, 0.0, 1.0)
    outro_fade = (0.5 * (1.0 + np.cos(math.pi * tail_u))).astype(np.float32)

    # Soft hydrophone deep-water warmth (C1/C2)
    hydro = synth_hydrophone_bed(rng, dur, root_midi=36.0, amp=0.19, drift_hz=0.22)
    add_stereo_event(bus, (hydro * outro_fade[:, None]).astype(np.float32), t_enter)

    # Warm resolution chord progression: Abmaj7 (0..1.9s) -> Cm(add9) / Eb6 (1.9..4.0s)
    prog_u = 0.5 * (1.0 + np.tanh((t - 1.85) / 0.50))

    # Foundation bass: Ab1/Ab2 (32/44) -> C2/C3 (36/48)
    f_b1 = midi_to_hz(32) * (1.0 - prog_u) + midi_to_hz(36) * prog_u
    f_b2 = midi_to_hz(44) * (1.0 - prog_u) + midi_to_hz(48) * prog_u
    p_b1 = 2.0 * math.pi * np.cumsum(f_b1.astype(np.float64)) / SR
    p_b2 = 2.0 * math.pi * np.cumsum(f_b2.astype(np.float64)) / SR
    bass = 0.62 * np.sin(p_b1).astype(np.float32) + 0.34 * np.sin(p_b2).astype(np.float32)
    bass_env = np.minimum(1.0, t / 0.05) * outro_fade * 0.22
    add_stereo_event(bus, pan_stereo((bass * bass_env).astype(np.float32), 0.0), t_enter)

    # Warm Abmaj7 (Ab2, Eb3, G3, C4, Eb4, G4) -> Cm(add9) (C3, G3, D4, Eb4, G4, Bb4)
    pad_ab = submerged_pad(t, [44, 51, 55, 60, 63, 67], [0.28, 0.24, 0.22, 0.22, 0.20, 0.16], detune_cents=9.0, cutoff_hz=1650.0, vibrato_hz=3.2, vibrato_cents=4.5)
    pad_cm = submerged_pad(t, [48, 55, 62, 63, 67, 70], [0.28, 0.24, 0.22, 0.22, 0.20, 0.16], detune_cents=9.0, cutoff_hz=1750.0, vibrato_hz=3.2, vibrato_cents=4.5)
    pad = pad_ab * (1.0 - prog_u[:, None]) + pad_cm * prog_u[:, None]
    pad_env = (np.minimum(1.0, t / 0.14) * outro_fade * 0.21)[:, None]
    add_stereo_event(bus, (pad * pad_env).astype(np.float32), t_enter)

    # Gentle submerged shimmer (>3.5 kHz)
    sky_ab = submerged_pad(t, [79, 84, 87], [0.34, 0.32, 0.28], detune_cents=10.0, cutoff_hz=5800.0)
    sky_cm = submerged_pad(t, [79, 82, 86], [0.34, 0.32, 0.28], detune_cents=10.0, cutoff_hz=5800.0)
    sky = butter_filter(sky_ab * (1.0 - prog_u[:, None]) + sky_cm * prog_u[:, None], "high", 3300.0, order=2)
    sky_env = (np.minimum(1.0, t / 0.25) * outro_fade * 0.085)[:, None]
    add_stereo_event(bus, (sky * sky_env).astype(np.float32), t_enter)

    # Soft arrival sub pulse + calm survival sonar chimes
    add_stereo_event(bus, synth_sub_pulse(44, dur_s=0.45, amp=0.22, cutoff_hz=210.0), t_enter)
    add_stereo_event(bus, synth_sonar_ping(79, dur_s=1.40, amp=0.075, pan=-0.26), t_settle)       # G5
    add_stereo_event(bus, synth_sonar_ping(84, dur_s=1.45, amp=0.068, pan=0.30), t_settle + 0.14) # C6
    add_stereo_event(bus, synth_sub_pulse(36, dur_s=0.50, amp=0.18, cutoff_hz=190.0), t_enter + 1.85)
    add_stereo_event(bus, synth_sonar_ping(74, dur_s=1.50, amp=0.062, pan=0.0), t_enter + 1.85)   # D5 (add9 warmth)


def enforce_breath_window(bus: np.ndarray, start_s: float, end_s: float, fade_s: float = 0.008) -> None:
    """Guarantee a razor-clean silent breath window [start_s, end_s] with click-free edges."""
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
    """
    Vocal-pocket EQ bands (freq_hz, Q, gain_db) carved smoothly across 250-3500 Hz
    with an exact -3.5 dB peak cut for BrianNeural narration clarity.
    """
    return [
        (460.0, 0.90, -2.75),
        (1200.0, 0.95, -1.85),
        (2500.0, 0.90, -2.75),
    ]


def apply_vocal_pocket_eq(bus: np.ndarray) -> np.ndarray:
    """Apply the -3.5 dB vocal-pocket EQ notch across 250-3500 Hz."""
    for freq, q, gain_db in vocal_pocket_filters():
        bus = peaking_eq(bus, freq, q, gain_db)
    return bus


def apply_smooth_20ms_edges(bus: np.ndarray) -> None:
    """Apply smooth 20 ms raised-cosine fade-in and fade-out at start/end of stem."""
    n_fade = max(1, int(round(FADE_IO_S * SR)))  # 960 samples @ 48 kHz
    u = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
    fade_in = (0.5 * (1.0 - np.cos(math.pi * u)))[:, None]
    fade_out = (0.5 * (1.0 + np.cos(math.pi * u)))[:, None]
    bus[:n_fade] *= fade_in
    bus[-n_fade:] *= fade_out
    bus[0, :] = 0.0
    bus[-1, :] = 0.0


def compose_bgm() -> np.ndarray:
    """Compose, EQ, and normalize the 30.000 s 48 kHz stereo BGM stem."""
    bus = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)
    rng = np.random.default_rng(20261128)

    compose_beat1(bus, rng)
    compose_beat2(bus, rng)
    compose_beat3(bus, rng)
    compose_beat4(bus, rng)
    compose_beat5(bus, rng)
    compose_beat6(bus, rng)

    # Clean subsonic rumble cut below 26 Hz
    bus = butter_filter(bus, "high", 26.0, order=2)

    # Permanent vocal-pocket EQ notch cut (250-3500 Hz, -3.5 dB peak cut)
    bus = apply_vocal_pocket_eq(bus)

    # Warm analog soft saturation + strict peak normalization (0.80 <= 0.82)
    bus = np.tanh(bus * 1.10).astype(np.float32)
    peak = float(np.max(np.abs(bus)))
    if peak > 1e-6:
        bus *= 0.80 / peak

    # Re-assert crisp 45 ms breath gap before Beat 6 after IIR EQ rings
    enforce_breath_window(bus, BREATH_START_S, BREATH_END_S, fade_s=0.006)

    # Clean musical outro tail (29.55-30.00s) + explicit smooth 20 ms fade-in/out
    i_tail = int(round(29.55 * SR))
    tail_len = len(bus) - i_tail
    if tail_len > 0:
        u_tail = np.linspace(0.0, 1.0, tail_len, dtype=np.float32)
        bus[i_tail:] *= (0.5 * (1.0 + np.cos(math.pi * u_tail)))[:, None]
    apply_smooth_20ms_edges(bus)

    return bus.astype(np.float32)


def measure_vocal_pocket_peak_cut_db() -> float:
    """Measure exact combined vocal-pocket peak attenuation (dB) across 250-3500 Hz."""
    freqs = np.geomspace(250.0, 3500.0, 512)
    w = 2.0 * math.pi * freqs / SR
    mag_db = np.zeros_like(freqs)
    for freq, q, gain_db in vocal_pocket_filters():
        b, a = peaking_eq_coeffs(freq, q, gain_db)
        _, h = signal.freqz(b, a, worN=w)
        mag_db += 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))
    return float(np.min(mag_db))


def verify_stem(path: Path, audio: np.ndarray) -> None:
    """Verify exact 48 kHz stereo PCM_16 30.000 s WAV specs, vocal-pocket EQ, breath gap, and zero clipping."""
    info = sf.info(str(path))
    assert info.samplerate == SR, f"Expected SR={SR}, got {info.samplerate}"
    assert info.channels == CHANNELS, f"Expected {CHANNELS} channels, got {info.channels}"
    assert info.frames == TOTAL_SAMPLES, f"Expected {TOTAL_SAMPLES} frames, got {info.frames}"
    assert info.subtype == "PCM_16", f"Expected PCM_16, got {info.subtype}"
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

    min_eq_db = measure_vocal_pocket_peak_cut_db()
    assert -3.9 <= min_eq_db <= -3.1, f"Vocal-pocket EQ peak cut out of spec: {min_eq_db:.2f} dB"

    b0 = int(round((BREATH_START_S + 0.004) * SR))
    b1 = int(round((BREATH_END_S - 0.004) * SR))
    breath_rms = float(np.sqrt(np.mean(audio[b0:b1].astype(np.float64) ** 2)))
    assert breath_rms < 1e-4, f"Breath window not clean: {breath_rms:.6f}"

    head_start = float(np.max(np.abs(audio[:4])))
    tail_end = float(np.max(np.abs(audio[-4:])))
    assert head_start < 1e-4, f"20ms fade-in start not clean: {head_start:.6f}"
    assert tail_end < 1e-4, f"20ms fade-out end not clean: {tail_end:.6f}"

    print(
        f"[BGM] Wrote {path.name}: {info.samplerate} Hz, {info.channels}ch, {info.subtype}, "
        f"{info.frames} samples ({info.duration:.3f}s), peak={peak:.4f}, rms={rms:.4f}, "
        f"eq_peak_cut={min_eq_db:.2f} dB, breath_rms={breath_rms:.6f}"
    )
    for b_key in C.BEATS:
        s0, s1 = beat_span(b_key)
        seg = audio[int(round(s0 * SR)) : int(round(s1 * SR))]
        seg_rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        seg_pk = float(np.max(np.abs(seg)))
        print(f"  {b_key:14s} ({s0:5.2f}-{s1:5.2f}s): rms={seg_rms:.4f}, peak={seg_pk:.4f}")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audio = compose_bgm()
    sf.write(str(OUTPUT_PATH), audio, SR, subtype="PCM_16")
    verify_stem(OUTPUT_PATH, audio)


if __name__ == "__main__":
    main()
