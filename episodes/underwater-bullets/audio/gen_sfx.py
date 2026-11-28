"""
Foley SFX Synthesizer for Episode #2:
"Why Bullets Stop Dead Just 3 Feet Underwater" (30.0 s @ 60 fps = 1800 frames).

Reads milestone frames and trajectories from `src/bullet_contract.py` (`EV`, `FPS`,
`DURATION_S`, `BEATS`) and outputs:
    episodes/underwater-bullets/audio/output/sfx_stem.wav
    - 48000 Hz, Stereo, PCM_16, exact 30.000 s (1,440,000 samples)
    - Zero NaNs/Infs, zero clipping, peak <= 0.88
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

OUTPUT_PATH = AUDIO_DIR / "output" / "sfx_stem.wav"

SR: int = 48000
CHANNELS: int = 2
DURATION_S: float = float(C.DURATION_S)
TOTAL_SAMPLES: int = int(round(DURATION_S * SR))
PEAK_CEILING: float = 0.86


# ---------------------------------------------------------------------------
# Timing & Spatial DSP Helpers
# ---------------------------------------------------------------------------
def ev_s(event_key: str) -> float:
    """Return event timestamp in seconds from bullet_contract.EV."""
    return float(C.EV[event_key]) / float(C.FPS)


def frame_s(frame: float) -> float:
    """Convert 1-based animation frame to seconds."""
    return float(frame) / float(C.FPS)


def beat_span_s(beat_key: str) -> tuple[float, float]:
    """Return (start_s, end_s) for a beat key in bullet_contract.BEATS."""
    f0, f1 = C.BEATS[beat_key]
    return ((f0 - 1) / float(C.FPS), f1 / float(C.FPS))


def pan_stereo(mono: np.ndarray, pan: float) -> np.ndarray:
    """Constant-power stereo panner (-1.0 left .. +1.0 right)."""
    pan = float(np.clip(pan, -1.0, 1.0))
    angle = (pan + 1.0) * (math.pi / 4.0)
    return np.stack([mono * math.cos(angle), mono * math.sin(angle)], axis=-1).astype(np.float32)


def pan_sweep_stereo(mono: np.ndarray, pan0: float, pan1: float) -> np.ndarray:
    """Dynamic constant-power stereo pan sweep from pan0 to pan1."""
    p = np.linspace(
        float(np.clip(pan0, -1.0, 1.0)),
        float(np.clip(pan1, -1.0, 1.0)),
        len(mono),
        dtype=np.float32,
    )
    ang = (p + 1.0) * (math.pi / 4.0)
    return np.stack([mono * np.cos(ang), mono * np.sin(ang)], axis=-1).astype(np.float32)


def add_stereo_event(bus: np.ndarray, event: np.ndarray, start_time: float) -> None:
    """Sample-accurately mix a mono or stereo buffer into bus at start_time (s)."""
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
    """Butterworth SOS filter along axis=0."""
    nyq = 0.5 * SR
    if isinstance(cutoff, tuple):
        wn: float | list[float] = [float(np.clip(c / nyq, 1e-4, 0.995)) for c in cutoff]
    else:
        wn = float(np.clip(cutoff / nyq, 1e-4, 0.995))
    sos = signal.butter(order, wn, btype=btype, output="sos")
    return signal.sosfilt(sos, audio, axis=0).astype(np.float32)


def peaking_eq(audio: np.ndarray, freq: float, q: float, gain_db: float) -> np.ndarray:
    """RBJ peaking EQ biquad filter along axis=0."""
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / SR
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    norm = 1.0 + alpha / a_lin
    b = np.array([1.0 + alpha * a_lin, -2.0 * cw, 1.0 - alpha * a_lin], dtype=np.float64) / norm
    a = np.array([1.0, (-2.0 * cw) / norm, (1.0 - alpha / a_lin) / norm], dtype=np.float64)
    return signal.lfilter(b, a, audio.astype(np.float64), axis=0).astype(np.float32)


def make_env(n: int, attack_s: float, release_s: float) -> np.ndarray:
    """Smooth linear attack/release envelope of length n."""
    t = np.arange(n, dtype=np.float32) / SR
    dur = n / float(SR)
    att = np.minimum(1.0, t / max(1e-4, attack_s))
    rel = np.minimum(1.0, (dur - t) / max(1e-4, release_s))
    return (att * rel).astype(np.float32)


def phase_integral(freq_hz: np.ndarray) -> np.ndarray:
    """Instantaneous phase integral 2*pi*cumsum(f)/SR in float64."""
    return 2.0 * math.pi * np.cumsum(freq_hz.astype(np.float64)) / SR


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))).astype(np.float32)


def add_tank_reflections(
    stereo: np.ndarray,
    taps_ms: tuple[float, ...] = (9.0, 21.0, 38.0),
    gains: tuple[float, ...] = (0.20, 0.12, 0.06),
    lp_hz: float = 2800.0,
) -> np.ndarray:
    """Short underwater ballistic-tank acoustic reflections."""
    out = stereo.copy()
    refl = np.zeros_like(stereo)
    for i, (ms, g) in enumerate(zip(taps_ms, gains)):
        d = int(round(ms * 0.001 * SR))
        if 0 < d < len(stereo):
            if i % 2 == 0:
                refl[d:, 0] += stereo[:-d, 0] * g
                refl[d:, 1] += stereo[:-d, 1] * (g * 0.85)
            else:
                refl[d:, 0] += stereo[:-d, 1] * (g * 0.85)
                refl[d:, 1] += stereo[:-d, 0] * g
    out += butter_filter(refl, "low", lp_hz, order=2)
    return out.astype(np.float32)


def synth_bubble_pop(
    rng: np.random.Generator,
    f0: float,
    dur_s: float = 0.016,
    rise_ratio: float = 1.65,
    amp: float = 1.0,
) -> np.ndarray:
    """Minnaert-inspired underwater bubble chirp (rapid upward pitch snap)."""
    n = max(64, int(round(dur_s * SR)))
    t = np.arange(n, dtype=np.float64) / SR
    tau = dur_s * 0.28
    freq = f0 * (1.0 + (rise_ratio - 1.0) * (1.0 - np.exp(-t / max(1e-4, tau))))
    phi = phase_integral(freq)
    env = np.exp(-t / max(1e-4, dur_s * 0.34)) * np.minimum(1.0, t / 0.0008)
    wav = np.sin(phi + rng.uniform(0.0, 2.0 * math.pi)) * env * amp
    return wav.astype(np.float32)


def soft_limit_peak(audio: np.ndarray, ceiling: float = PEAK_CEILING) -> np.ndarray:
    """Lookahead smooth gain limiter guaranteeing peak <= ceiling with zero clipping."""
    abs_pk = np.max(np.abs(audio), axis=1).astype(np.float32)
    step = 48  # 1 ms lookahead blocks
    n_blk = (len(abs_pk) + step - 1) // step
    blk = np.array(
        [float(np.max(abs_pk[i * step : min(len(abs_pk), (i + 1) * step)])) for i in range(n_blk)],
        dtype=np.float32,
    )
    att = math.exp(-step / max(1.0, 0.0006 * SR))
    rel = math.exp(-step / max(1.0, 0.050 * SR))
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
# Continuous Ambient Bed (Tank Underwater + Air-Cut + Blueprint + Deep Calm)
# ---------------------------------------------------------------------------
def build_ambient_bed() -> np.ndarray:
    """Continuous stereo underwater ambience shaped per beat, vocal-pocket EQ'd."""
    n = TOTAL_SAMPLES
    t = np.arange(n, dtype=np.float32) / SR
    rng_l = np.random.default_rng(201)
    rng_r = np.random.default_rng(202)
    raw_l = rng_l.standard_normal(n).astype(np.float32)
    raw_r = rng_r.standard_normal(n).astype(np.float32)

    t_b2_start, _ = beat_span_s("B2_ENTRY")
    t_entry = ev_s("b2_water_entry")
    t_b5_start, t_b5_end = beat_span_s("B5_RACE")

    # Subsurface low water pressure rumble (drops slightly during air-side muzzle beat)
    air_dip = 1.0 - 0.65 * _sigmoid((t - t_b2_start) / 0.06) * _sigmoid((t_entry - t) / 0.06)
    blueprint_dip = 1.0 - 0.35 * _sigmoid((t - t_b5_start) / 0.12) * _sigmoid((t_b5_end - t) / 0.12)
    sub_mod_l = 0.84 + 0.16 * np.sin(2.0 * math.pi * 0.28 * t)
    sub_mod_r = 0.84 + 0.16 * np.sin(2.0 * math.pi * 0.31 * t + 1.4)

    deep_l = butter_filter(raw_l, "band", (42.0, 210.0), order=2) * sub_mod_l
    deep_r = butter_filter(raw_r, "band", (42.0, 210.0), order=2) * sub_mod_r
    deep_bed = np.stack([deep_l, deep_r], axis=-1) * (0.068 * air_dip * blueprint_dip)[:, None]

    # Subtle mid-tank water circulation texture
    mid_l = butter_filter(raw_l, "band", (220.0, 820.0), order=2) * (0.80 + 0.20 * np.sin(2.0 * math.pi * 0.43 * t + 0.5))
    mid_r = butter_filter(raw_r, "band", (220.0, 820.0), order=2) * (0.80 + 0.20 * np.sin(2.0 * math.pi * 0.39 * t + 2.1))
    mid_bed = np.stack([mid_l, mid_r], axis=-1) * (0.022 * air_dip * blueprint_dip)[:, None]

    bed = deep_bed + mid_bed
    bed = peaking_eq(bed, 1250.0, 0.55, -3.5)
    return (bed * make_env(n, 0.03, 0.25)[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# Beat 1 (f=10..135): Hydro-Streak Hiss -> Dead-Stop Thud + Metallic Plink
# ---------------------------------------------------------------------------
def render_beat1(bus: np.ndarray) -> None:
    t0 = ev_s("b1_bullet_start")   # f=10 (0.167 s)
    t1 = ev_s("b1_bullet_stop")    # f=135 (2.250 s)
    dur_streak = (t1 - t0) + 0.10
    n = int(round(dur_streak * SR))
    t = np.arange(n, dtype=np.float64) / SR
    rng = np.random.default_rng(101)
    noise = rng.standard_normal(n).astype(np.float32)

    # Decelerating streak velocity profile matching bullet_contract.supersonic_x
    u = np.clip(t / max(1e-3, t1 - t0), 0.0, 1.0)
    vel = np.maximum(0.06, (1.0 - u) ** 1.45)

    # Layered hydro-streak hiss: split-band swept shear + cavitation whistle
    hi_band = butter_filter(noise, "band", (1800.0, 6800.0), order=2)
    mid_band = butter_filter(noise, "band", (420.0, 1750.0), order=2)
    sub_wake = butter_filter(noise, "band", (75.0, 260.0), order=2)

    # Decelerating resonant hydrodynamic pitch sweep (1450 Hz -> 290 Hz)
    whistle_f = 290.0 + 1160.0 * (1.0 - u) ** 1.35
    whistle = np.sin(phase_integral(whistle_f)) * (0.14 * vel + 0.03)

    streak_env = make_env(n, 0.04, 0.12) * (0.30 + 0.85 * vel)
    # Ramp intensity subtly right into the violent stop
    streak_mono = (hi_band * (0.28 * vel + 0.06) + mid_band * (0.24 * vel + 0.08) + sub_wake * 0.22 + whistle) * streak_env
    streak_stereo = pan_sweep_stereo(streak_mono.astype(np.float32), -0.78, 0.26)
    add_stereo_event(bus, add_tank_reflections(streak_stereo, gains=(0.16, 0.09, 0.04)), t0)

    # Sharp dead-stop thud at f=135 (2.250 s)
    n_stop = int(round(0.45 * SR))
    ts = np.arange(n_stop, dtype=np.float64) / SR
    thud_freq = 36.0 + 135.0 * np.exp(-ts / 0.022)
    thud_body = np.sin(phase_integral(thud_freq)) * np.exp(-ts / 0.095) * 0.72
    snap_noise = butter_filter(rng.standard_normal(n_stop).astype(np.float32), "band", (650.0, 3200.0), order=2)
    snap_hit = snap_noise * np.exp(-ts / 0.018) * 0.48
    stop_mono = (thud_body + snap_hit) * make_env(n_stop, 0.001, 0.08)
    add_stereo_event(bus, add_tank_reflections(pan_stereo(stop_mono.astype(np.float32), 0.26)), t1)

    # Descending metallic plink + settling bubble ticks right at/after f=135
    n_plink = int(round(0.85 * SR))
    tp = np.arange(n_plink, dtype=np.float64) / SR
    f_p1 = 1520.0 + 580.0 * np.exp(-tp / 0.14)
    f_p2 = 2480.0 + 980.0 * np.exp(-tp / 0.11)
    f_p3 = 3820.0 + 1240.0 * np.exp(-tp / 0.08)
    plink = (
        0.48 * np.sin(phase_integral(f_p1)) * np.exp(-tp / 0.24)
        + 0.34 * np.sin(phase_integral(f_p2)) * np.exp(-tp / 0.17)
        + 0.20 * np.sin(phase_integral(f_p3)) * np.exp(-tp / 0.11)
    ) * make_env(n_plink, 0.0015, 0.12) * 0.36
    add_stereo_event(bus, add_tank_reflections(pan_sweep_stereo(plink.astype(np.float32), 0.26, 0.34)), t1 + 0.018)

    for k in range(5):
        bt = t1 + 0.09 + k * 0.13 + rng.uniform(-0.01, 0.02)
        pop = synth_bubble_pop(rng, f0=float(rng.uniform(720.0, 1550.0)), dur_s=0.022, amp=0.14)
        add_stereo_event(bus, pan_stereo(pop, float(rng.uniform(0.15, 0.42))), bt)


# ---------------------------------------------------------------------------
# Beat 2 (f=248, f=300..520): Barrel Crack -> Hydro Impact Wall + Water Roar
# ---------------------------------------------------------------------------
def render_beat2(bus: np.ndarray) -> None:
    t_fire = ev_s("b2_muzzle_fire")   # f=248 (4.133 s)
    t_entry = ev_s("b2_water_entry")  # f=300 (5.000 s)
    t_roar_end = frame_s(535)         # through f=520 tail into 540
    rng = np.random.default_rng(2020)

    # 1. Supersonic / suppressed punchy barrel crack at f=248
    n_fire = int(round(0.55 * SR))
    tf = np.arange(n_fire, dtype=np.float64) / SR
    # Mechanical pin + razor transient crack
    pin_click = np.sin(2.0 * math.pi * 3200.0 * tf) * np.exp(-tf / 0.0025) * 0.42
    crack_noise = butter_filter(rng.standard_normal(n_fire).astype(np.float32), "band", (1400.0, 7400.0), order=2)
    crack_burst = crack_noise * np.exp(-tf / 0.022) * 0.85
    # Punchy barrel chest thump (210 Hz -> 52 Hz)
    muzzle_f = 52.0 + 165.0 * np.exp(-tf / 0.016)
    muzzle_punch = np.sin(phase_integral(muzzle_f)) * np.exp(-tf / 0.065) * 0.80
    # Crisp suppressed chamber tail
    tail_noise = butter_filter(rng.standard_normal(n_fire).astype(np.float32), "band", (260.0, 1500.0), order=2)
    tail = tail_noise * np.exp(-tf / 0.085) * 0.35

    fire_mono = (pin_click + crack_burst + muzzle_punch + tail) * make_env(n_fire, 0.0005, 0.06)
    add_stereo_event(bus, pan_stereo(fire_mono.astype(np.float32), -0.68), t_fire)

    # Brief supersonic air-whip between muzzle (f=248) and water surface (f=300)
    dur_air = max(0.20, t_entry - t_fire)
    n_air = int(round(dur_air * SR))
    ta = np.arange(n_air, dtype=np.float64) / SR
    air_noise = butter_filter(rng.standard_normal(n_air).astype(np.float32), "band", (1600.0, 6200.0), order=2)
    air_whip = (air_noise * 0.24 + np.sin(phase_integral(np.linspace(2600.0, 1450.0, n_air))) * 0.08) * make_env(n_air, 0.01, 0.04)
    add_stereo_event(bus, pan_sweep_stereo(air_whip.astype(np.float32), -0.65, -0.26), t_fire + 0.02)

    # 2. Massive hydro-density impact wall thump at f=300 (5.000 s)
    n_imp = int(round(0.95 * SR))
    ti = np.arange(n_imp, dtype=np.float64) / SR
    wall_f = 30.0 + 150.0 * np.exp(-ti / 0.026)
    wall_sub = np.sin(phase_integral(wall_f)) * np.exp(-ti / 0.16) * 0.88
    wall_mid = np.sin(phase_integral(58.0 + 210.0 * np.exp(-ti / 0.014))) * np.exp(-ti / 0.07) * 0.45
    slap_hi = butter_filter(rng.standard_normal(n_imp).astype(np.float32), "band", (900.0, 4800.0), order=2) * np.exp(-ti / 0.032) * 0.65
    imp_mono = (wall_sub + wall_mid + slap_hi) * make_env(n_imp, 0.001, 0.14)
    add_stereo_event(bus, add_tank_reflections(pan_stereo(imp_mono.astype(np.float32), -0.22)), t_entry)

    # 3. Pressurized water roar (f=300..520+)
    dur_roar = t_roar_end - t_entry
    n_roar = int(round(dur_roar * SR))
    tr = np.arange(n_roar, dtype=np.float32) / SR
    roar_raw_l = rng.standard_normal(n_roar).astype(np.float32)
    roar_raw_r = rng.standard_normal(n_roar).astype(np.float32)

    roar_lo_l = butter_filter(roar_raw_l, "band", (65.0, 320.0), order=2)
    roar_lo_r = butter_filter(roar_raw_r, "band", (65.0, 320.0), order=2)
    roar_mid_l = butter_filter(roar_raw_l, "band", (340.0, 1450.0), order=2)
    roar_mid_r = butter_filter(roar_raw_r, "band", (340.0, 1450.0), order=2)
    roar_hi_l = butter_filter(roar_raw_l, "band", (1600.0, 4600.0), order=2)
    roar_hi_r = butter_filter(roar_raw_r, "band", (1600.0, 4600.0), order=2)

    # Swell right after entry, churn through f=520, taper cleanly into Beat 3
    churn_l = 0.80 + 0.20 * np.sin(2.0 * math.pi * 2.4 * tr)
    churn_r = 0.80 + 0.20 * np.sin(2.0 * math.pi * 2.3 * tr + 1.1)
    roar_env = make_env(n_roar, 0.04, 0.45) * (0.48 + 0.65 * np.exp(-tr / 1.35))

    roar_l = (roar_lo_l * 0.44 + roar_mid_l * 0.28 + roar_hi_l * 0.14) * churn_l * roar_env
    roar_r = (roar_lo_r * 0.44 + roar_mid_r * 0.28 + roar_hi_r * 0.14) * churn_r * roar_env
    roar_st = np.stack([roar_l, roar_r], axis=-1).astype(np.float32)
    roar_st = peaking_eq(roar_st, 1200.0, 0.6, -2.5)
    add_stereo_event(bus, roar_st, t_entry + 0.015)

    # Burst of heavy displacement bubbles in wake (f=315..490)
    for idx in range(22):
        bt = t_entry + 0.12 + idx * 0.13 + float(rng.uniform(-0.04, 0.04))
        pop = synth_bubble_pop(rng, f0=float(rng.uniform(320.0, 950.0)), dur_s=float(rng.uniform(0.018, 0.036)), amp=float(rng.uniform(0.10, 0.22)))
        add_stereo_event(bus, pan_stereo(pop, float(rng.uniform(-0.45, 0.35))), bt)


# ---------------------------------------------------------------------------
# Beat 3 (f=541..900): Sizzling Supercavitation Vapor Boil & Swirling Crackle
# ---------------------------------------------------------------------------
def render_beat3(bus: np.ndarray) -> None:
    t_s, t_e = beat_span_s("B3_CAVITATION")
    t_peak = ev_s("b3_bubble_peak")  # f=720 (12.000 s)
    dur = t_e - t_s
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    rng_l = np.random.default_rng(301)
    rng_r = np.random.default_rng(302)

    raw_l = rng_l.standard_normal(n).astype(np.float32)
    raw_r = rng_r.standard_normal(n).astype(np.float32)

    # Sizzling supercavitation steam/vapor envelope
    sizzle_l = butter_filter(raw_l, "band", (2200.0, 7600.0), order=2)
    sizzle_r = butter_filter(raw_r, "band", (2200.0, 7600.0), order=2)
    boil_l = butter_filter(raw_l, "band", (420.0, 1550.0), order=2)
    boil_r = butter_filter(raw_r, "band", (420.0, 1550.0), order=2)
    sheath_l = butter_filter(raw_l, "band", (110.0, 340.0), order=2)
    sheath_r = butter_filter(raw_r, "band", (110.0, 340.0), order=2)

    # Swirling L/R spin modulation matching bullet rifling spin inside gas cavity
    swirl_l = 0.72 + 0.28 * np.sin(2.0 * math.pi * 1.15 * t)
    swirl_r = 0.72 + 0.28 * np.sin(2.0 * math.pi * 1.15 * t + 1.85)
    peak_gauss = 0.62 + 0.55 * np.exp(-((t + t_s - t_peak) ** 2) / (2.0 * 1.55 ** 2))
    env = make_env(n, 0.22, 0.28) * peak_gauss

    cav_l = (sizzle_l * 0.22 + boil_l * 0.20 + sheath_l * 0.18) * swirl_l * env
    cav_r = (sizzle_r * 0.22 + boil_r * 0.20 + sheath_r * 0.18) * swirl_r * env
    cav_st = np.stack([cav_l, cav_r], axis=-1).astype(np.float32)
    cav_st = peaking_eq(cav_st, 1300.0, 0.6, -2.8)
    add_stereo_event(bus, cav_st, t_s)

    # Swirling micro-bubble crackle field (~140 tiny crisp pops panning dynamically)
    rng_pop = np.random.default_rng(303)
    num_pops = 140
    for idx in range(num_pops):
        frac = idx / float(num_pops)
        rel_t = 0.10 + frac * (dur - 0.24) + float(rng_pop.uniform(-0.025, 0.025))
        rel_t = float(np.clip(rel_t, 0.02, dur - 0.08))
        # Density/energy bell around Beat 3 peak
        density_gain = 0.55 + 0.65 * math.exp(-((t_s + rel_t - t_peak) ** 2) / (2.0 * 1.6 ** 2))
        f0 = float(rng_pop.uniform(850.0, 3600.0))
        pop = synth_bubble_pop(rng_pop, f0=f0, dur_s=float(rng_pop.uniform(0.006, 0.017)), rise_ratio=1.55, amp=0.14 * density_gain)
        swirl_pan = 0.58 * math.sin(2.0 * math.pi * 0.85 * rel_t + idx * 0.27)
        add_stereo_event(bus, pan_stereo(pop, swirl_pan), t_s + rel_t)


# ---------------------------------------------------------------------------
# Beat 4 (f=925..1180): X-Ray Chime -> Torque Groan -> Jacket Shatter + Shards
# ---------------------------------------------------------------------------
def render_beat4(bus: np.ndarray) -> None:
    t_xray = ev_s("b4_xray_on")         # f=925 (15.417 s)
    t_tumble = ev_s("b4_tumble_start")  # f=970 (16.167 s)
    t_shatter = ev_s("b4_shatter_burst")# f=1055 (17.583 s)
    rng = np.random.default_rng(404)

    # 1. Brief X-ray scan chime at f=925
    n_xr = int(round(0.60 * SR))
    tx = np.arange(n_xr, dtype=np.float64) / SR
    scan_sweep = np.sin(phase_integral(np.linspace(780.0, 1560.0, n_xr))) * np.exp(-tx / 0.14) * 0.18
    chime = (
        0.24 * np.sin(2.0 * math.pi * 1320.0 * tx) * np.exp(-tx / 0.22)
        + 0.18 * np.sin(2.0 * math.pi * 1980.0 * tx) * np.exp(-tx / 0.18)
        + 0.10 * np.sin(2.0 * math.pi * 2640.0 * tx) * np.exp(-tx / 0.14)
    )
    xr_mono = (scan_sweep + chime) * make_env(n_xr, 0.006, 0.10)
    add_stereo_event(bus, add_tank_reflections(pan_sweep_stereo(xr_mono.astype(np.float32), -0.25, 0.25)), t_xray)

    # 2. Metallic stress torque groan (f=970 -> f=1055)
    dur_groan = max(0.35, t_shatter - t_tumble + 0.05)
    n_gr = int(round(dur_groan * SR))
    tg = np.arange(n_gr, dtype=np.float64) / SR
    u_gr = np.clip(tg / max(1e-3, dur_groan), 0.0, 1.0)
    # Non-linear FM metallic groaning torque rising under hydro-shear
    mod = np.sin(2.0 * math.pi * (24.0 + 38.0 * u_gr) * tg) * (35.0 + 70.0 * u_gr)
    f_base = 162.0 + 68.0 * (u_gr ** 1.4) + mod
    groan = (
        0.34 * np.sin(phase_integral(f_base))
        + 0.26 * np.sin(phase_integral(f_base * 1.51))
        + 0.18 * np.sin(phase_integral(f_base * 2.38))
        + 0.10 * np.sin(phase_integral(f_base * 3.74))
    )
    # Add stick-slip metallic micro-creak impulses
    creak_noise = butter_filter(rng.standard_normal(n_gr).astype(np.float32), "band", (900.0, 2900.0), order=2)
    creak_mod = 0.5 + 0.5 * np.sin(2.0 * math.pi * (14.0 + 20.0 * u_gr) * tg)
    groan_mono = (groan * (0.45 + 0.55 * u_gr) + creak_noise * creak_mod * 0.20) * make_env(n_gr, 0.06, 0.05) * 0.55
    add_stereo_event(bus, add_tank_reflections(pan_sweep_stereo(groan_mono.astype(np.float32), -0.05, 0.20)), t_tumble)

    # 3. Crisp brittle copper-jacket shatter crack at f=1055
    n_sh = int(round(0.70 * SR))
    ts = np.arange(n_sh, dtype=np.float64) / SR
    # Glass-sharp metallic brittle fracture transient
    crack_hi = butter_filter(rng.standard_normal(n_sh).astype(np.float32), "band", (2200.0, 8200.0), order=2) * np.exp(-ts / 0.022) * 0.92
    crack_mid = butter_filter(rng.standard_normal(n_sh).astype(np.float32), "band", (850.0, 2200.0), order=2) * np.exp(-ts / 0.036) * 0.55
    shock_sub = np.sin(phase_integral(38.0 + 130.0 * np.exp(-ts / 0.020))) * np.exp(-ts / 0.11) * 0.65
    # Crisp metallic ring bursts
    ring = (
        0.28 * np.sin(2.0 * math.pi * 2840.0 * ts) * np.exp(-ts / 0.09)
        + 0.22 * np.sin(2.0 * math.pi * 4120.0 * ts) * np.exp(-ts / 0.065)
    )
    shatter_mono = (crack_hi + crack_mid + shock_sub + ring) * make_env(n_sh, 0.0005, 0.09)
    add_stereo_event(bus, add_tank_reflections(pan_stereo(shatter_mono.astype(np.float32), 0.20)), t_shatter)

    # 4. Scattered fragment chimes (8 copper/lead shards decelerating out through f=1180)
    shard_freqs = [1940.0, 2380.0, 2790.0, 3150.0, 3580.0, 4120.0, 2610.0, 4680.0]
    shard_pans = [-0.48, 0.52, -0.24, 0.66, -0.60, 0.34, 0.10, -0.35]
    for i, (sfreq, span) in enumerate(zip(shard_freqs, shard_pans)):
        dt = 0.018 + i * 0.125 + float(rng.uniform(-0.015, 0.02))
        n_frag = int(round(0.48 * SR))
        tf = np.arange(n_frag, dtype=np.float64) / SR
        f_drop = sfreq * (1.0 - 0.16 * (1.0 - np.exp(-tf / 0.18)))
        frag = (
            0.55 * np.sin(phase_integral(f_drop)) * np.exp(-tf / 0.16)
            + 0.35 * np.sin(phase_integral(f_drop * 1.47)) * np.exp(-tf / 0.11)
        ) * make_env(n_frag, 0.001, 0.08) * 0.22
        add_stereo_event(bus, add_tank_reflections(pan_stereo(frag.astype(np.float32), span)), t_shatter + dt)
        pop = synth_bubble_pop(rng, f0=float(rng.uniform(950.0, 2100.0)), dur_s=0.016, amp=0.11)
        add_stereo_event(bus, pan_stereo(pop, span), t_shatter + dt + 0.02)


# ---------------------------------------------------------------------------
# Beat 5 (f=1285..1560): Twin Ticks -> Screeching Halt -> Laminar Sonar Whoosh
# ---------------------------------------------------------------------------
def render_beat5(bus: np.ndarray) -> None:
    t_launch = ev_s("b5_race_start")   # f=1285 (21.417 s)
    t_halt = ev_s("b5_fast_stop")      # f=1410 (23.500 s)
    t_whoosh0 = frame_s(1460)          # 24.333 s
    t_cross = ev_s("b5_slow_cross")    # f=1525 (25.417 s)
    t_whoosh1 = frame_s(1545)          # 25.750 s
    rng = np.random.default_rng(505)

    # 1. Twin launch ticks at f=1285 (staggered 0ms and +52ms for top/bottom lanes)
    for k, (dt, freq_a, freq_b, pan) in enumerate([(0.0, 1760.0, 2940.0, -0.45), (0.052, 1320.0, 2240.0, -0.38)]):
        n_tk = int(round(0.16 * SR))
        tt = np.arange(n_tk, dtype=np.float64) / SR
        tick = (
            0.48 * np.sin(2.0 * math.pi * freq_a * tt) * np.exp(-tt / 0.016)
            + 0.34 * np.sin(2.0 * math.pi * freq_b * tt) * np.exp(-tt / 0.010)
            + 0.40 * np.sin(phase_integral(65.0 + 110.0 * np.exp(-tt / 0.012))) * np.exp(-tt / 0.032)
        ) * make_env(n_tk, 0.0005, 0.03) * 0.58
        add_stereo_event(bus, pan_stereo(tick.astype(np.float32), pan), t_launch + dt)

    # 2. Fast supersonic lane sprint & screeching halt (f=1285..1410 -> halt @ 23.500 s)
    dur_fast = max(0.3, t_halt - t_launch)
    n_fast = int(round(dur_fast * SR))
    tf = np.arange(n_fast, dtype=np.float64) / SR
    uf = np.clip(tf / max(1e-3, dur_fast), 0.0, 1.0)
    z_noise = butter_filter(rng.standard_normal(n_fast).astype(np.float32), "band", (1400.0, 5600.0), order=2)
    z_whine = np.sin(phase_integral(1900.0 - 1100.0 * (uf ** 0.7))) * (0.14 * (1.0 - 0.5 * uf))
    fast_streak = (z_noise * 0.24 * (1.0 - 0.45 * uf) + z_whine) * make_env(n_fast, 0.02, 0.04)
    add_stereo_event(bus, pan_sweep_stereo(fast_streak.astype(np.float32), -0.45, 0.14), t_launch + 0.02)

    # Screeching hydro-halt burst at f=1410
    n_halt = int(round(0.48 * SR))
    th = np.arange(n_halt, dtype=np.float64) / SR
    screech_f = 2850.0 * np.exp(-th / 0.075) + 520.0
    screech = (
        0.38 * np.sin(phase_integral(screech_f)) * np.exp(-th / 0.11)
        + 0.26 * np.sin(phase_integral(screech_f * 1.41)) * np.exp(-th / 0.08)
    )
    halt_snap = butter_filter(rng.standard_normal(n_halt).astype(np.float32), "band", (1100.0, 5200.0), order=2) * np.exp(-th / 0.024) * 0.62
    halt_sub = np.sin(phase_integral(42.0 + 140.0 * np.exp(-th / 0.018))) * np.exp(-th / 0.085) * 0.62
    halt_mono = (screech + halt_snap + halt_sub) * make_env(n_halt, 0.001, 0.07)
    add_stereo_event(bus, add_tank_reflections(pan_stereo(halt_mono.astype(np.float32), 0.14)), t_halt)

    # 3. Smooth laminar sonar whoosh for slower subsonic round (f=1460..1535+, crossing at f=1525)
    dur_wh = (t_whoosh1 - t_whoosh0) + 0.22
    n_wh = int(round(dur_wh * SR))
    tw = np.arange(n_wh, dtype=np.float64) / SR
    uw = np.clip(tw / max(1e-3, dur_wh), 0.0, 1.0)
    lam_lo = butter_filter(rng.standard_normal(n_wh).astype(np.float32), "band", (140.0, 520.0), order=2)
    lam_mid = butter_filter(rng.standard_normal(n_wh).astype(np.float32), "band", (540.0, 1650.0), order=2)
    lam_tone = np.sin(phase_integral(260.0 + 210.0 * np.sin(math.pi * uw))) * 0.14
    lam_env = np.sin(math.pi * uw) ** 1.15
    lam_mono = (lam_lo * 0.34 + lam_mid * 0.20 + lam_tone) * lam_env * make_env(n_wh, 0.08, 0.14)
    add_stereo_event(bus, pan_sweep_stereo(lam_mono.astype(np.float32), -0.25, 0.72), t_whoosh0)

    # Clean sonar telemetry ping as slower round clears 3 FT -> 6 FT (f=1525)
    n_sonar = int(round(0.65 * SR))
    tsn = np.arange(n_sonar, dtype=np.float64) / SR
    sonar = (
        0.26 * np.sin(2.0 * math.pi * 740.0 * tsn) * np.exp(-tsn / 0.26)
        + 0.14 * np.sin(2.0 * math.pi * 1110.0 * tsn) * np.exp(-tsn / 0.18)
    ) * make_env(n_sonar, 0.008, 0.12)
    add_stereo_event(bus, add_tank_reflections(pan_stereo(sonar.astype(np.float32), 0.48)), t_cross - 0.06)


# ---------------------------------------------------------------------------
# Beat 6 (f=1561..1800): Calm Deep-Water Ambience + Gentle Copper-Scrap Shimmer
# ---------------------------------------------------------------------------
def render_beat6(bus: np.ndarray) -> None:
    t_s, t_e = beat_span_s("B6_SAFE")
    t_settle = ev_s("b6_safe_settle")  # f=1620 (27.000 s)
    dur = t_e - t_s
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    rng = np.random.default_rng(606)

    # Calm deep-water swell layer
    raw_l = rng.standard_normal(n).astype(np.float32)
    raw_r = rng.standard_normal(n).astype(np.float32)
    swell_l = butter_filter(raw_l, "band", (52.0, 280.0), order=2) * (0.75 + 0.25 * np.sin(2.0 * math.pi * 0.35 * t))
    swell_r = butter_filter(raw_r, "band", (52.0, 280.0), order=2) * (0.75 + 0.25 * np.sin(2.0 * math.pi * 0.31 * t + 1.2))
    swell_env = make_env(n, 0.35, 0.45) * 0.085
    add_stereo_event(bus, np.stack([swell_l * swell_env, swell_r * swell_env], axis=-1).astype(np.float32), t_s)

    # Gentle copper-scrap shimmer glints + soft drifting bubbles as crushed scrap settles
    shimmer_times = [t_s + 0.38, t_settle, t_settle + 0.68, t_settle + 1.35, t_settle + 1.98]
    shimmer_pans = [0.16, 0.28, 0.20, 0.34, 0.24]
    shimmer_bases = [1840.0, 2160.0, 1680.0, 2340.0, 1960.0]

    for st, sp, sb in zip(shimmer_times, shimmer_pans, shimmer_bases):
        n_sh = int(round(0.85 * SR))
        ts = np.arange(n_sh, dtype=np.float64) / SR
        glint = (
            0.16 * np.sin(2.0 * math.pi * sb * ts) * np.exp(-ts / 0.32)
            + 0.11 * np.sin(2.0 * math.pi * (sb * 1.498) * ts) * np.exp(-ts / 0.26)
            + 0.06 * np.sin(2.0 * math.pi * (sb * 2.02) * ts) * np.exp(-ts / 0.20)
        ) * make_env(n_sh, 0.045, 0.18) * 0.45
        add_stereo_event(bus, add_tank_reflections(pan_sweep_stereo(glint.astype(np.float32), sp - 0.06, sp + 0.08)), st)

    for k in range(10):
        bt = t_s + 0.25 + k * 0.34 + float(rng.uniform(-0.04, 0.04))
        pop = synth_bubble_pop(rng, f0=float(rng.uniform(480.0, 1080.0)), dur_s=0.024, rise_ratio=1.45, amp=0.08)
        add_stereo_event(bus, pan_stereo(pop, float(rng.uniform(-0.30, 0.42))), bt)


# ---------------------------------------------------------------------------
# Master Build & Verification
# ---------------------------------------------------------------------------
def build_sfx_stem() -> np.ndarray:
    """Synthesize all 6 beats into an exact 30.000s 48kHz stereo float32 bus."""
    bus = build_ambient_bed()
    render_beat1(bus)
    render_beat2(bus)
    render_beat3(bus)
    render_beat4(bus)
    render_beat5(bus)
    render_beat6(bus)

    # Clean subsonic DC cut below 24 Hz & zero-clipping smooth peak ceiling <= 0.88
    bus = butter_filter(bus, "high", 24.0, order=2)
    bus = soft_limit_peak(bus, ceiling=PEAK_CEILING)

    # Click-free 5ms boundary fades
    micro = int(round(0.005 * SR))
    bus[:micro] *= np.linspace(0.0, 1.0, micro, dtype=np.float32)[:, None]
    bus[-micro:] *= np.linspace(1.0, 0.0, micro, dtype=np.float32)[:, None]
    return bus.astype(np.float32)


def verify_sfx_stem(path: Path, audio: np.ndarray) -> None:
    """Assert 48000 Hz, Stereo, PCM_16, exact 30.000s, zero NaNs, peak <= 0.88."""
    assert np.isfinite(audio).all(), "SFX stem contains NaN or Inf values"

    info = sf.info(str(path))
    assert info.samplerate == SR, f"Expected {SR} Hz, got {info.samplerate}"
    assert info.channels == CHANNELS, f"Expected {CHANNELS}ch, got {info.channels}"
    assert info.subtype == "PCM_16", f"Expected PCM_16, got {info.subtype}"
    assert info.frames == TOTAL_SAMPLES, f"Expected {TOTAL_SAMPLES} samples, got {info.frames}"
    assert abs(info.duration - DURATION_S) < 1e-6, f"Expected {DURATION_S:.3f}s, got {info.duration:.6f}s"

    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == SR
        assert wf.getnchannels() == CHANNELS
        assert wf.getsampwidth() == 2
        assert wf.getnframes() == TOTAL_SAMPLES

    # Re-read quantized PCM_16 sample values to verify true quantized peak <= 0.88
    pcm_float, sr_read = sf.read(str(path), dtype="float32")
    assert sr_read == SR
    peak = float(np.max(np.abs(pcm_float)))
    rms = float(np.sqrt(np.mean(pcm_float.astype(np.float64) ** 2)))
    assert peak <= 0.88, f"Peak exceeded 0.88: {peak:.5f}"
    assert 0.01 < rms < 0.45, f"Unexpected SFX RMS: {rms:.5f}"

    print(
        f"[SFX] {path.as_posix()}: sample_rate={info.samplerate} Hz, "
        f"channels={info.channels}, subtype={info.subtype}, "
        f"frames={info.frames}, duration={info.duration:.3f}s, "
        f"peak={peak:.4f} (<= 0.88), rms={rms:.4f}"
    )
    for b_key in C.BEATS:
        s0, s1 = beat_span_s(b_key)
        seg = pcm_float[int(round(s0 * SR)) : int(round(s1 * SR))]
        seg_rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        seg_pk = float(np.max(np.abs(seg)))
        print(f"  {b_key:14s} ({s0:5.2f}-{s1:5.2f}s): rms={seg_rms:.4f}, peak={seg_pk:.4f}")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audio = build_sfx_stem()
    sf.write(str(OUTPUT_PATH), audio, SR, subtype="PCM_16")
    verify_sfx_stem(OUTPUT_PATH, audio)


if __name__ == "__main__":
    main()
