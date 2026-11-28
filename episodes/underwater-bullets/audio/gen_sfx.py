"""
Foley SFX Synthesizer & Real CC0/PD Sample Engine for Episode #2:
"Why Bullets Stop Dead Just 3 Feet Underwater" (30.0 s @ 60 fps = 1800 frames).

Combines verified Public Domain / CC0 48 kHz recordings (`audio/samples/`) with
physical sub-bass impacts, Doppler sweeps, cavitation bubbles, and underwater
tank acoustics:
  - Beat 1 (f=4..210): Supersonic underwater gunshot discharge + hydro whip ->
    heavy sub-braam 3 FT halt impact + metallic casing settle + rising bubbles
  - Beat 2 (f=248..540): Layered high-caliber rifle muzzle crack + mechanical bolt
    cycle -> supersonic air whip -> massive 800x hydraulic splash cannon + wake boil
  - Beat 3 (f=541..900): Real turbulent boiling cavitation bubbles spinning in L/R
    macro stereo + pressurized vapor sheath fizz peaking at f=720
  - Beat 4 (f=925..1240): Sleek X-ray scan pulse -> strained metal torque creak ->
    brittle copper-jacket shatter explosion + ricocheting shard cracks & implosion
  - Beat 5 (f=1285..1560): Twin staggered REAL gunshots (crack rifle vs deep heavy
    subsonic thump) -> 3 FT screech halt -> smooth laminar water glide + sonar ping
  - Beat 6 (f=1561..1800): Calm deep-water bubble drift + soft copper scrap clinks
    + warm sub-bass resolve

Outputs:
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
SAMPLES_DIR = AUDIO_DIR / "samples"
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
PEAK_CEILING: float = 0.87

_SAMPLE_CACHE: dict[str, np.ndarray] = {}


# ---------------------------------------------------------------------------
# Timing, Sample Loading & Spatial DSP Helpers
# ---------------------------------------------------------------------------
def ev_s(event_key: str) -> float:
    return float(C.EV[event_key]) / float(C.FPS)


def frame_s(frame: float) -> float:
    return float(frame) / float(C.FPS)


def beat_span_s(beat_key: str) -> tuple[float, float]:
    f0, f1 = C.BEATS[beat_key]
    return ((f0 - 1) / float(C.FPS), f1 / float(C.FPS))


def load_sample(filename: str) -> np.ndarray:
    """Load a 48 kHz float32 stereo WAV sample from `audio/samples/` with caching."""
    if filename not in _SAMPLE_CACHE:
        p = SAMPLES_DIR / filename
        data, sr = sf.read(str(p), dtype="float32")
        if sr != SR:
            n_out = int(round(len(data) * (SR / float(sr))))
            data = signal.resample(data, n_out, axis=0).astype(np.float32)
        if data.ndim == 1:
            data = np.stack([data, data], axis=-1)
        _SAMPLE_CACHE[filename] = data.astype(np.float32)
    return _SAMPLE_CACHE[filename].copy()


def pitch_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    """Resample audio to shift pitch/speed smoothly (`speed < 1.0` = deeper/heavier)."""
    speed = float(np.clip(speed, 0.25, 3.0))
    n_out = max(16, int(round(len(audio) / speed)))
    idx = np.linspace(0.0, len(audio) - 1, n_out, dtype=np.float64)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(len(audio) - 1, i0 + 1)
    frac = (idx - i0).astype(np.float32)
    if audio.ndim == 2:
        return (audio[i0] * (1.0 - frac[:, None]) + audio[i1] * frac[:, None]).astype(np.float32)
    return (audio[i0] * (1.0 - frac) + audio[i1] * frac).astype(np.float32)


def pan_stereo(mono_or_st: np.ndarray, pan: float) -> np.ndarray:
    """Constant-power stereo panner (-1.0 left .. +1.0 right)."""
    mono = np.mean(mono_or_st, axis=1) if mono_or_st.ndim == 2 else mono_or_st
    pan = float(np.clip(pan, -1.0, 1.0))
    angle = (pan + 1.0) * (math.pi / 4.0)
    return np.stack([mono * math.cos(angle), mono * math.sin(angle)], axis=-1).astype(np.float32)


def pan_sweep_stereo(mono_or_st: np.ndarray, pan0: float, pan1: float) -> np.ndarray:
    """Dynamic constant-power stereo pan sweep from pan0 to pan1."""
    mono = np.mean(mono_or_st, axis=1) if mono_or_st.ndim == 2 else mono_or_st
    p = np.linspace(float(np.clip(pan0, -1.0, 1.0)), float(np.clip(pan1, -1.0, 1.0)), len(mono), dtype=np.float32)
    ang = (p + 1.0) * (math.pi / 4.0)
    return np.stack([mono * np.cos(ang), mono * np.sin(ang)], axis=-1).astype(np.float32)


def add_stereo_event(bus: np.ndarray, event: np.ndarray, start_time: float, gain: float = 1.0) -> None:
    """Sample-accurately mix a mono or stereo buffer into bus at start_time (s)."""
    if event.ndim == 1:
        event = np.stack([event, event], axis=-1).astype(np.float32)
    idx = int(round(start_time * SR))
    if idx >= len(bus) or idx + len(event) <= 0:
        return
    src_s = max(0, -idx)
    dst_s = max(0, idx)
    clen = min(len(event) - src_s, len(bus) - dst_s)
    if clen > 0:
        bus[dst_s : dst_s + clen] += event[src_s : src_s + clen] * float(gain)


def butter_filter(
    audio: np.ndarray,
    btype: str,
    cutoff: float | tuple[float, float],
    order: int = 2,
) -> np.ndarray:
    nyq = 0.5 * SR
    if isinstance(cutoff, tuple):
        wn: float | list[float] = [float(np.clip(c / nyq, 1e-4, 0.995)) for c in cutoff]
    else:
        wn = float(np.clip(cutoff / nyq, 1e-4, 0.995))
    sos = signal.butter(order, wn, btype=btype, output="sos")
    return signal.sosfilt(sos, audio, axis=0).astype(np.float32)


def peaking_eq(audio: np.ndarray, freq: float, q: float, gain_db: float) -> np.ndarray:
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / SR
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    norm = 1.0 + alpha / a_lin
    b = np.array([1.0 + alpha * a_lin, -2.0 * cw, 1.0 - alpha * a_lin], dtype=np.float64) / norm
    a = np.array([1.0, (-2.0 * cw) / norm, (1.0 - alpha / a_lin) / norm], dtype=np.float64)
    return signal.lfilter(b, a, audio.astype(np.float64), axis=0).astype(np.float32)


def make_env(n: int, attack_s: float, release_s: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SR
    dur = n / float(SR)
    att = np.minimum(1.0, t / max(1e-4, attack_s))
    rel = np.minimum(1.0, (dur - t) / max(1e-4, release_s))
    return (att * rel).astype(np.float32)


def phase_integral(freq_hz: np.ndarray) -> np.ndarray:
    return 2.0 * math.pi * np.cumsum(freq_hz.astype(np.float64)) / SR


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))).astype(np.float32)


def add_tank_reflections(
    stereo: np.ndarray,
    taps_ms: tuple[float, ...] = (11.0, 24.0, 44.0, 72.0),
    gains: tuple[float, ...] = (0.24, 0.15, 0.09, 0.05),
    lp_hz: float = 2400.0,
) -> np.ndarray:
    """Ballistic tank spatial early reflections."""
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


def build_realistic_rifle_shot(
    crack_file: str = "pd_gunshot_a.wav",
    include_body: bool = True,
    pitch_scale: float = 1.0,
    sub_weight: float = 0.85,
    underwater_lp: float | None = None,
    pan: float = -0.5,
) -> np.ndarray:
    """
    Build a multi-layered cinematic rifle discharge strictly from clean Public Domain gunshots:
      1. Dry crisp supersonic crack (`pd_gunshot_a/b/c.wav`)
      2. Pitched-down (0.54x) chest-hitting acoustic boom tail (`pd_gunshot_b/c.wav`, 0% voice)
      3. Propellant sub-harmonic punch (42-150 Hz) + tape saturation grit
      4. Mechanical bolt snap / metallic action transient
    """
    crack = load_sample(crack_file)
    if abs(pitch_scale - 1.0) > 1e-3:
        crack = pitch_speed(crack, pitch_scale)

    n = max(len(crack), int(round(0.90 * SR)))
    shot = np.zeros((n, 2), dtype=np.float32)
    shot[: len(crack)] += crack * 0.96

    if include_body:
        body_src = "pd_gunshot_c.wav" if crack_file != "pd_gunshot_c.wav" else "pd_gunshot_b.wav"
        body = pitch_speed(load_sample(body_src), 0.54 * pitch_scale)
        body = butter_filter(body, "low", 1150.0, order=2)
        nb = min(n, len(body))
        shot[:nb] += body[:nb] * 0.78

    # Chest-rattling propellant sub-punch + mechanical bolt click
    t = np.arange(n, dtype=np.float64) / SR
    sub_f = (38.0 + 135.0 * np.exp(-t / 0.018)) * min(1.15, pitch_scale)
    sub_kick = (np.sin(phase_integral(sub_f)) * np.exp(-t / 0.075) * sub_weight).astype(np.float32)
    bolt_snap = (np.sin(2.0 * math.pi * 4100.0 * t) * np.exp(-t / 0.0022) * 0.28).astype(np.float32)
    shot += pan_stereo(sub_kick + bolt_snap, 0.0)

    # Subtle harmonic drive for theatrical punch without harsh clipping
    shot = (np.tanh(shot * 1.35) * 0.82).astype(np.float32)

    if underwater_lp is not None:
        shot = butter_filter(shot, "low", underwater_lp, order=2)

    panned = pan_stereo(shot, pan) * 0.70 + shot * 0.30
    return add_tank_reflections(panned * make_env(n, 0.0005, 0.14)[:, None])


def synth_bubble_pop(
    rng: np.random.Generator,
    f0: float,
    dur_s: float = 0.016,
    rise_ratio: float = 1.65,
    amp: float = 1.0,
) -> np.ndarray:
    n = max(64, int(round(dur_s * SR)))
    t = np.arange(n, dtype=np.float64) / SR
    tau = dur_s * 0.28
    freq = f0 * (1.0 + (rise_ratio - 1.0) * (1.0 - np.exp(-t / max(1e-4, tau))))
    phi = phase_integral(freq)
    env = np.exp(-t / max(1e-4, dur_s * 0.34)) * np.minimum(1.0, t / 0.0008)
    wav = np.sin(phi + rng.uniform(0.0, 2.0 * math.pi)) * env * amp
    return wav.astype(np.float32)


def soft_limit_peak(audio: np.ndarray, ceiling: float = PEAK_CEILING) -> np.ndarray:
    abs_pk = np.max(np.abs(audio), axis=1).astype(np.float32)
    step = 48
    n_blk = (len(abs_pk) + step - 1) // step
    blk = np.array(
        [float(np.max(abs_pk[i * step : min(len(abs_pk), (i + 1) * step)])) for i in range(n_blk)],
        dtype=np.float32,
    )
    att = math.exp(-step / max(1.0, 0.0005 * SR))
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
# Continuous Subtle Underwater Tank Ambience
# ---------------------------------------------------------------------------
def build_ambient_bed() -> np.ndarray:
    n = TOTAL_SAMPLES
    t = np.arange(n, dtype=np.float32) / SR
    rng_l = np.random.default_rng(201)
    rng_r = np.random.default_rng(202)
    raw_l = rng_l.standard_normal(n).astype(np.float32)
    raw_r = rng_r.standard_normal(n).astype(np.float32)

    t_b2_start, _ = beat_span_s("B2_ENTRY")
    t_entry = ev_s("b2_water_entry")
    t_b5_start, t_b5_end = beat_span_s("B5_RACE")

    air_dip = 1.0 - 0.70 * _sigmoid((t - t_b2_start) / 0.06) * _sigmoid((t_entry - t) / 0.06)
    blueprint_dip = 1.0 - 0.55 * _sigmoid((t - t_b5_start) / 0.12) * _sigmoid((t_b5_end - t) / 0.12)
    sub_mod_l = 0.84 + 0.16 * np.sin(2.0 * math.pi * 0.28 * t)
    sub_mod_r = 0.84 + 0.16 * np.sin(2.0 * math.pi * 0.31 * t + 1.4)

    deep_l = butter_filter(raw_l, "band", (44.0, 195.0), order=2) * sub_mod_l
    deep_r = butter_filter(raw_r, "band", (44.0, 195.0), order=2) * sub_mod_r
    deep_bed = np.stack([deep_l, deep_r], axis=-1) * (0.048 * air_dip * blueprint_dip)[:, None]
    return (deep_bed * make_env(n, 0.03, 0.25)[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# Beat 1 (f=2..180): Crisp Muffled Gun Discharge -> Hydro Whip -> Sub-Braam Stop
# ---------------------------------------------------------------------------
def render_beat1(bus: np.ndarray) -> None:
    t_shot = frame_s(4)            # Immediate opening gunshot discharge underwater (0.067 s)
    t0 = ev_s("b1_bullet_start")   # f=10 (0.167 s)
    t1 = ev_s("b1_bullet_stop")    # f=135 (2.250 s)
    rng = np.random.default_rng(101)

    # 1. Opening underwater gunshot (punchy bass + damped supersonic crack) right on frame 4!
    uw_shot = build_realistic_rifle_shot(
        crack_file="pd_gunshot_b.wav",
        include_body=True,
        pitch_scale=0.92,
        sub_weight=0.95,
        underwater_lp=2200.0,
        pan=-0.65,
    )
    add_stereo_event(bus, uw_shot, t_shot, gain=0.82)

    # 2. Fast supersonic hydro-streak whoosh panning left -> 3 FT stop
    dur_streak = max(0.3, (t1 - t0) + 0.08)
    n = int(round(dur_streak * SR))
    t = np.arange(n, dtype=np.float64) / SR
    u = np.clip(t / max(1e-3, t1 - t0), 0.0, 1.0)
    vel = np.maximum(0.08, (1.0 - u) ** 1.45)

    noise = rng.standard_normal(n).astype(np.float32)
    hi_shear = butter_filter(noise, "band", (1400.0, 5200.0), order=2)
    mid_rush = butter_filter(noise, "band", (380.0, 1350.0), order=2)
    lo_rush = butter_filter(noise, "band", (85.0, 340.0), order=2)
    streak_env = make_env(n, 0.02, 0.10) * (0.18 + 0.92 * (vel ** 1.1))
    streak = (hi_shear * (0.34 * vel) + mid_rush * (0.28 * vel + 0.04) + lo_rush * (0.30 * vel + 0.08)) * streak_env
    add_stereo_event(bus, add_tank_reflections(pan_sweep_stereo(streak, -0.78, 0.24)), t0, gain=0.88)

    # 3. Heavy Dead-Stop Wall Impact + Sub-Braam at f=135 ("3 FT STOP")
    braam = load_sample("cc0_sub_braam.wav")[: int(round(1.15 * SR))]
    braam *= make_env(len(braam), 0.002, 0.45)[:, None]
    braam_sub = butter_filter(braam, "low", 260.0, order=2) * 0.72 + butter_filter(braam, "band", (280.0, 1400.0), order=2) * 0.28
    add_stereo_event(bus, pan_stereo(braam_sub, 0.12) * 0.65 + braam_sub * 0.35, t1 - 0.015, gain=0.88)

    # Crisp brittle water-slam snap at f=135 + rising real PD bubbles
    crack_stop = load_sample("pd_gunshot_c.wav")[: int(round(0.22 * SR))]
    crack_stop = butter_filter(crack_stop, "band", (320.0, 3200.0), order=2) * make_env(len(crack_stop), 0.001, 0.14)[:, None]
    add_stereo_event(bus, pan_stereo(crack_stop, 0.20), t1, gain=0.55)

    bubbles = load_sample("pd_water_bubbles.wav")[: int(round(1.15 * SR))]
    bubbles = butter_filter(bubbles, "band", (260.0, 2200.0), order=2) * make_env(len(bubbles), 0.04, 0.35)[:, None]
    add_stereo_event(bus, pan_sweep_stereo(bubbles, 0.18, 0.35), t1 + 0.03, gain=0.45)


# ---------------------------------------------------------------------------
# Beat 2 (f=248..535): FULL RIFLE GUNSHOT -> Air Whip -> Hydraulic Cannon Splash
# ---------------------------------------------------------------------------
def render_beat2(bus: np.ndarray) -> None:
    t_fire = ev_s("b2_muzzle_fire")   # f=248 (4.133 s)
    t_entry = ev_s("b2_water_entry")  # f=300 (5.000 s)
    t_roar_end = frame_s(535)
    rng = np.random.default_rng(2020)

    # 1. FULL HIGH-CALIBER RIFLE SHOT right on muzzle flash (f=248)
    rifle_blast = build_realistic_rifle_shot(
        crack_file="pd_gunshot_a.wav",
        include_body=True,
        pitch_scale=1.0,
        sub_weight=1.0,
        underwater_lp=None,  # Full crisp supersonic air muzzle crack!
        pan=-0.52,
    )
    add_stereo_event(bus, rifle_blast, t_fire, gain=1.05)

    # Metallic shell casing ping + action cycling (~140 ms after fire)
    n_sh = int(round(0.32 * SR))
    t_sh = np.arange(n_sh, dtype=np.float64) / SR
    casing = (
        0.22 * np.sin(2.0 * math.pi * 5240.0 * t_sh) * np.exp(-t_sh / 0.045)
        + 0.16 * np.sin(2.0 * math.pi * 7820.0 * t_sh) * np.exp(-t_sh / 0.032)
    ) * make_env(n_sh, 0.0006, 0.08)
    add_stereo_event(bus, pan_stereo(casing.astype(np.float32), -0.62), t_fire + 0.13)

    # 2. Supersonic air bullet whip into the water wall (f=254..300)
    dur_air = max(0.18, t_entry - t_fire)
    n_air = int(round(dur_air * SR))
    air_n = butter_filter(rng.standard_normal(n_air).astype(np.float32), "band", (1500.0, 6200.0), order=2)
    air_whip = air_n * make_env(n_air, 0.015, 0.04) * 0.28
    add_stereo_event(bus, pan_sweep_stereo(air_whip, -0.58, -0.22), t_fire + 0.04)

    # 3. MASSIVE 800x WATER WALL SLAM & PLUNGE (f=300)
    plunge_hi = load_sample("pd_water_plunge.wav")
    plunge_heavy = pitch_speed(plunge_hi, 0.62)[: int(round(2.4 * SR))]  # Deep hydraulic cannon weight!
    add_stereo_event(bus, pan_stereo(plunge_hi, -0.20), t_entry, gain=0.88)
    add_stereo_event(bus, pan_stereo(plunge_heavy, -0.15) * 0.7 + plunge_heavy * 0.3, t_entry, gain=0.92)

    # 32 Hz Deep Hydro Wall Thump + tight splash shockwave
    n_wall = int(round(0.85 * SR))
    tw = np.arange(n_wall, dtype=np.float64) / SR
    wall_sub = np.sin(phase_integral(32.0 + 160.0 * np.exp(-tw / 0.022))) * np.exp(-tw / 0.15) * 0.82
    add_stereo_event(bus, pan_stereo(wall_sub.astype(np.float32), -0.18), t_entry)

    # Turbulent underwater wake roar + real bubbles through f=535
    dur_roar = max(0.5, t_roar_end - t_entry)
    bub_roar = load_sample("pd_water_bubbles.wav")[: int(round(dur_roar * SR))]
    bub_roar = (
        butter_filter(bub_roar, "band", (120.0, 2600.0), order=2)
        * make_env(len(bub_roar), 0.06, 0.40)[:, None]
    )
    add_stereo_event(bus, pan_sweep_stereo(bub_roar, -0.28, 0.18), t_entry + 0.12, gain=0.72)


# ---------------------------------------------------------------------------
# Beat 3 (f=541..900): Real Boiling Supercavitation Vortex & Micro-Fizziness
# ---------------------------------------------------------------------------
def render_beat3(bus: np.ndarray) -> None:
    t_s, t_e = beat_span_s("B3_CAVITATION")
    t_peak = ev_s("b3_bubble_peak")  # f=720 (12.000 s)
    dur = t_e - t_s
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    rng_l = np.random.default_rng(301)
    rng_r = np.random.default_rng(302)

    # Real boiling bubble sample tiled seamlessly + slightly sped up (1.28x) for fierce cavitation hiss
    bub_src = load_sample("pd_water_bubbles.wav")
    bub_fast = pitch_speed(bub_src, 1.28)
    reps = (n // len(bub_fast)) + 2
    bub_tiled = np.tile(bub_fast, (reps, 1))[:n]
    bub_bp = butter_filter(bub_tiled, "band", (220.0, 3800.0), order=2)

    # Subtle deep boiling pressure sheath rumble underneath
    raw_l = rng_l.standard_normal(n).astype(np.float32)
    raw_r = rng_r.standard_normal(n).astype(np.float32)
    rumble_l = butter_filter(raw_l, "band", (75.0, 280.0), order=2)
    rumble_r = butter_filter(raw_r, "band", (75.0, 280.0), order=2)
    fizz_l = butter_filter(raw_l, "band", (2800.0, 7400.0), order=2)
    fizz_r = butter_filter(raw_r, "band", (2800.0, 7400.0), order=2)

    swirl_l = 0.68 + 0.32 * np.sin(2.0 * math.pi * 1.25 * t)
    swirl_r = 0.68 + 0.32 * np.sin(2.0 * math.pi * 1.25 * t + 1.9)
    peak_gauss = 0.58 + 0.62 * np.exp(-((t + t_s - t_peak) ** 2) / (2.0 * 1.45 ** 2))
    env = make_env(n, 0.22, 0.28) * peak_gauss

    cav_l = (bub_bp[:, 0] * 0.56 + rumble_l * 0.22 + fizz_l * 0.14) * swirl_l * env
    cav_r = (bub_bp[:, 1] * 0.56 + rumble_r * 0.22 + fizz_r * 0.14) * swirl_r * env
    cav_st = peaking_eq(np.stack([cav_l, cav_r], axis=-1).astype(np.float32), 1400.0, 0.65, -2.5)
    add_stereo_event(bus, cav_st, t_s, gain=0.92)


# ---------------------------------------------------------------------------
# Beat 4 (f=925..1240): X-Ray Lock -> Metal Stress Snap -> Violent Hydro Shatter
# ---------------------------------------------------------------------------
def render_beat4(bus: np.ndarray) -> None:
    t_xray = ev_s("b4_xray_on")          # f=925 (15.417 s)
    t_tumble = ev_s("b4_tumble_start")   # f=970 (16.167 s)
    t_shatter = ev_s("b4_shatter_burst") # f=1055 (17.583 s)
    rng = np.random.default_rng(404)

    # 1. Clean subtle X-ray imaging lock click/pulse at f=925 (NOT loud synth bell)
    n_xr = int(round(0.32 * SR))
    tx = np.arange(n_xr, dtype=np.float64) / SR
    xray_pulse = (
        np.sin(phase_integral(np.linspace(1100.0, 2200.0, n_xr))) * np.exp(-tx / 0.06) * 0.14
        + np.sin(2.0 * math.pi * 2400.0 * tx) * np.exp(-tx / 0.09) * 0.10
    ) * make_env(n_xr, 0.004, 0.08)
    add_stereo_event(bus, pan_sweep_stereo(xray_pulse.astype(np.float32), -0.2, 0.2), t_xray)

    # 2. Rising metallic stress ratchets & hydraulic pressure build (f=970..1055)
    dur_build = max(0.35, t_shatter - t_tumble)
    n_b = int(round(dur_build * SR))
    tb = np.arange(n_b, dtype=np.float64) / SR
    ub = np.clip(tb / max(1e-3, dur_build), 0.0, 1.0)
    # Filtered metallic strain noise + low sub-tension riser (NO cheesy FM organ!)
    strain_n = butter_filter(rng.standard_normal(n_b).astype(np.float32), "band", (450.0, 2600.0), order=2)
    ratchet = np.maximum(0.0, np.sin(2.0 * math.pi * (12.0 + 26.0 * (ub ** 1.5)) * tb)) ** 2
    sub_riser = np.sin(phase_integral(55.0 + 85.0 * (ub ** 2))) * (0.18 + 0.30 * ub)
    build_mono = (strain_n * (0.18 + 0.35 * ratchet) * ub + sub_riser) * make_env(n_b, 0.06, 0.04)
    add_stereo_event(bus, add_tank_reflections(pan_sweep_stereo(build_mono.astype(np.float32), -0.1, 0.18)), t_tumble)

    # 3. VIOLENT JACKET SHATTER BLAST at f=1055!
    # Combine sharp crack + CC0 sub-braam impact + heavy plunge collapse + shrapnel snaps
    crack_sh = load_sample("pd_gunshot_c.wav")[: int(round(0.38 * SR))]
    crack_sh_hi = butter_filter(crack_sh, "high", 900.0, order=2) * 1.15
    add_stereo_event(bus, pan_stereo(crack_sh_hi, 0.14), t_shatter, gain=0.95)

    braam_sh = load_sample("cc0_sub_braam.wav")[: int(round(1.25 * SR))]
    braam_sh *= make_env(len(braam_sh), 0.001, 0.45)[:, None]
    add_stereo_event(bus, pan_stereo(braam_sh, 0.10) * 0.65 + braam_sh * 0.35, t_shatter - 0.01, gain=0.95)

    implode = pitch_speed(load_sample("pd_water_plunge.wav"), 0.72)[: int(round(1.35 * SR))]
    implode = butter_filter(implode, "low", 2400.0, order=2) * make_env(len(implode), 0.005, 0.35)[:, None]
    add_stereo_event(bus, implode, t_shatter + 0.02, gain=0.72)

    # 8 fast brittle metallic copper/lead shard snaps radiating outward across L/R
    shard_pans = [-0.68, 0.72, -0.45, 0.55, -0.82, 0.40, 0.18, -0.28]
    shard_pitches = [2650.0, 3420.0, 4180.0, 5120.0, 3150.0, 4680.0, 2940.0, 5840.0]
    for i, (span, spitch) in enumerate(zip(shard_pans, shard_pitches)):
        dt = 0.012 + i * 0.075 + float(rng.uniform(-0.01, 0.015))
        n_f = int(round(0.18 * SR))
        tf = np.arange(n_f, dtype=np.float64) / SR
        ping = (
            butter_filter(rng.standard_normal(n_f).astype(np.float32), "band", (1800.0, 6800.0), order=2) * np.exp(-tf / 0.012) * 0.55
            + np.sin(2.0 * math.pi * spitch * tf) * np.exp(-tf / 0.045) * 0.32
        ) * make_env(n_f, 0.0005, 0.05)
        add_stereo_event(bus, add_tank_reflections(pan_stereo(ping.astype(np.float32), span)), t_shatter + dt, gain=0.75)


# ---------------------------------------------------------------------------
# Beat 5 (f=1285..1560): TWIN REAL GUNSHOTS -> 3 FT Snap Halt -> Subsonic Glide
# ---------------------------------------------------------------------------
def render_beat5(bus: np.ndarray) -> None:
    t_launch = ev_s("b5_race_start")   # f=1285 (21.417 s)
    t_halt = ev_s("b5_fast_stop")      # f=1410 (23.500 s)
    t_whoosh0 = frame_s(1435)          # 23.917 s
    t_cross = ev_s("b5_slow_cross")    # f=1525 (25.417 s)
    t_whoosh1 = frame_s(1550)          # 25.833 s
    rng = np.random.default_rng(505)

    # 1. TWIN STAGGERED REAL GUNSHOTS AT LAUNCH (f=1285):
    # Top Lane (FASTER supersonic rifle round) @ +0ms, pitch=1.12x sharp high crack
    shot_fast = build_realistic_rifle_shot(
        crack_file="pd_gunshot_a.wav",
        include_body=True,
        pitch_scale=1.12,
        sub_weight=0.80,
        underwater_lp=None,
        pan=-0.52,
    )
    add_stereo_event(bus, shot_fast, t_launch, gain=0.92)

    # Bottom Lane (SLOWER heavy slug) @ +110ms, pitch=0.72x deep heavy thud cannon!
    shot_slow = build_realistic_rifle_shot(
        crack_file="pd_gunshot_b.wav",
        include_body=True,
        pitch_scale=0.72,
        sub_weight=1.05,
        underwater_lp=2800.0,
        pan=-0.42,
    )
    add_stereo_event(bus, shot_slow, t_launch + 0.11, gain=0.90)

    # 2. Fast supersonic cavitation sprint -> dead-stop water slam at 3 FT (f=1410)
    dur_fast = max(0.3, t_halt - (t_launch + 0.14))
    n_fast = int(round(dur_fast * SR))
    zf = butter_filter(rng.standard_normal(n_fast).astype(np.float32), "band", (1100.0, 4800.0), order=2)
    z_env = make_env(n_fast, 0.04, 0.05) * np.linspace(1.0, 0.35, n_fast, dtype=np.float32)
    add_stereo_event(bus, pan_sweep_stereo(zf * z_env * 0.35, -0.45, 0.12), t_launch + 0.14)

    # Dead-stop slam at f=1410 ("FASTER: 3 FT")
    halt_crack = load_sample("pd_gunshot_c.wav")[: int(round(0.28 * SR))]
    halt_crack = butter_filter(halt_crack, "band", (220.0, 3600.0), order=2) * make_env(len(halt_crack), 0.001, 0.15)[:, None]
    n_hs = int(round(0.45 * SR))
    th = np.arange(n_hs, dtype=np.float64) / SR
    halt_sub = (np.sin(phase_integral(40.0 + 145.0 * np.exp(-th / 0.018))) * np.exp(-th / 0.095) * 0.72).astype(np.float32)
    add_stereo_event(bus, pan_stereo(halt_crack, 0.14), t_halt, gain=0.78)
    add_stereo_event(bus, pan_stereo(halt_sub, 0.14), t_halt, gain=0.88)

    # 3. Clean laminar underwater glide whoosh + crisp telemetry lock chime as slow round hits 6 FT
    dur_wh = max(0.3, (t_whoosh1 - t_whoosh0) + 0.15)
    n_wh = int(round(dur_wh * SR))
    uw = np.linspace(0.0, 1.0, n_wh, dtype=np.float32)
    lam = butter_filter(rng.standard_normal(n_wh).astype(np.float32), "band", (160.0, 920.0), order=2)
    lam_env = (np.maximum(0.0, np.sin(math.pi * uw)) ** 1.1).astype(np.float32)
    add_stereo_event(bus, pan_sweep_stereo(lam * lam_env * 0.42, -0.18, 0.72), t_whoosh0)

    n_son = int(round(0.55 * SR))
    tsn = np.arange(n_son, dtype=np.float64) / SR
    sonar = (0.24 * np.sin(2.0 * math.pi * 880.0 * tsn) * np.exp(-tsn / 0.22) + 0.14 * np.sin(2.0 * math.pi * 1320.0 * tsn) * np.exp(-tsn / 0.16)) * make_env(n_son, 0.006, 0.10)
    add_stereo_event(bus, add_tank_reflections(pan_stereo(sonar.astype(np.float32), 0.52)), t_cross - 0.04)


# ---------------------------------------------------------------------------
# Beat 6 (f=1561..1800): Calm Underwater Bubbles + Soft Copper Scrap Clinks
# ---------------------------------------------------------------------------
def render_beat6(bus: np.ndarray) -> None:
    t_s, t_e = beat_span_s("B6_SAFE")
    t_settle = ev_s("b6_safe_settle")
    dur = t_e - t_s

    # Gentle low-passed real underwater bubble drift (ambient peace under surface)
    bub_calm = load_sample("pd_water_bubbles.wav")[: int(round(dur * SR))]
    bub_calm = butter_filter(bub_calm, "band", (95.0, 1150.0), order=2) * make_env(len(bub_calm), 0.35, 0.55)[:, None]
    add_stereo_event(bus, pan_sweep_stereo(bub_calm, -0.12, 0.22) * 0.45 + bub_calm * 0.25, t_s, gain=0.45)

    # Subtle realistic copper-scrap metallic tinkles as crushed flakes drift harmlessly
    for st, sp, pitch in [
        (t_s + 0.45, 0.14, 2420.0),
        (t_settle + 0.05, 0.26, 2980.0),
        (t_settle + 0.78, 0.18, 2150.0),
        (t_settle + 1.55, 0.30, 2680.0),
    ]:
        n_fl = int(round(0.45 * SR))
        tf = np.arange(n_fl, dtype=np.float64) / SR
        flake = (
            0.16 * np.sin(2.0 * math.pi * pitch * tf) * np.exp(-tf / 0.14)
            + 0.10 * np.sin(2.0 * math.pi * (pitch * 1.52) * tf) * np.exp(-tf / 0.10)
        ) * make_env(n_fl, 0.015, 0.12)
        add_stereo_event(bus, add_tank_reflections(pan_stereo(flake.astype(np.float32), sp)), st, gain=0.65)


# ---------------------------------------------------------------------------
# Master Build & Verification
# ---------------------------------------------------------------------------
def build_sfx_stem() -> np.ndarray:
    bus = build_ambient_bed()
    render_beat1(bus)
    render_beat2(bus)
    render_beat3(bus)
    render_beat4(bus)
    render_beat5(bus)
    render_beat6(bus)

    bus = butter_filter(bus, "high", 26.0, order=2)
    bus = soft_limit_peak(bus, ceiling=PEAK_CEILING)

    micro = int(round(0.005 * SR))
    bus[:micro] *= np.linspace(0.0, 1.0, micro, dtype=np.float32)[:, None]
    bus[-micro:] *= np.linspace(1.0, 0.0, micro, dtype=np.float32)[:, None]
    return bus.astype(np.float32)


def verify_sfx_stem(path: Path, audio: np.ndarray) -> None:
    assert np.isfinite(audio).all(), "SFX stem contains NaN or Inf values"
    info = sf.info(str(path))
    assert info.samplerate == SR and info.channels == CHANNELS and info.subtype == "PCM_16"
    assert info.frames == TOTAL_SAMPLES and abs(info.duration - DURATION_S) < 1e-6

    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == SR and wf.getnchannels() == CHANNELS and wf.getsampwidth() == 2

    pcm_float, _ = sf.read(str(path), dtype="float32")
    peak = float(np.max(np.abs(pcm_float)))
    rms = float(np.sqrt(np.mean(pcm_float.astype(np.float64) ** 2)))
    assert peak <= 0.88, f"Peak exceeded 0.88: {peak:.5f}"
    assert 0.01 < rms < 0.45, f"Unexpected SFX RMS: {rms:.5f}"

    print(
        f"[SFX] {path.as_posix()}: {info.samplerate} Hz, {info.channels}ch, "
        f"{info.subtype}, {info.duration:.3f}s, peak={peak:.4f}, rms={rms:.4f}"
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
