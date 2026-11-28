"""
Foley SFX Synthesizer & Real CC0/PD Sample Engine for Episode #3:
"Why Bulletproof Vests DON'T Send You Flying Backward" (30.0 s @ 60 fps = 1800 frames).

Strictly layers verified Public Domain / CC0 48 kHz recordings (`pd_gunshot_a.wav`,
`pd_gunshot_b.wav`, `pd_gunshot_c.wav`, `cc0_sub_braam.wav`) with physical acoustic
sub-punch (40-140 Hz), ballistic range reflections, Kevlar membrane tension, and
Silicon-Carbide ceramic shatter acoustics — ZERO toy sine beeps for gunshots/impacts!

Synced 1:1 to all 16 `vest_contract.EV` milestones:
  - Beat 1 (f=22..210): Slow-mo range gunshot -> ballistic zip -> dead-center plate
    chest slam (`cc0_sub_braam` + `pd_gunshot_c` + 40-140 Hz sub-punch) -> spark shower
  - Beat 2 (f=245..510): Dry tactical range gunshot (`pd_gunshot_a`) + shell casing ping
    -> air crack -> short contained vest thud (proving tiny fastball-level momentum)
  - Beat 3 (f=535..900): Optical X-ray cutaway peel -> golden Kevlar snare catch ->
    deep elastic trampoline weave ripple resonance & fiber stress creak
  - Beat 4 (f=925..1260): High-velocity rifle crack -> SiC ceramic plate strike boom ->
    brittle ceramic fracture crunch + 10 porcelain/metal shards -> UHMWPE Dyneema catch
  - Beat 5 (f=1295..1560): Studio Blueprint symmetrical shooter discharge (-0.45 pan) ->
    mirror-intensity vest absorb thud (+0.45 pan) -> Newton Third Law lock-in sub-braam
  - Beat 6 (f=1561..1800): Settling spent ceramic/brass dust tinkles + calm range resolve

Outputs:
    episodes/bulletproof-vest-myth/audio/output/sfx_stem.wav
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
FALLBACK_SAMPLES_DIR = EPISODE_ROOT.parent / "underwater-bullets" / "audio" / "samples"
SRC_DIR = EPISODE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import vest_contract as C  # noqa: E402

OUTPUT_PATH = AUDIO_DIR / "output" / "sfx_stem.wav"

SR: int = 48000
CHANNELS: int = 2
DURATION_S: float = float(C.DURATION_S)
TOTAL_SAMPLES: int = int(round(DURATION_S * SR))
PEAK_CEILING: float = 0.87

_SAMPLE_CACHE: dict[str, np.ndarray] = {}


# ---------------------------------------------------------------------------
# Timing, Real CC0/PD Sample Loading & Spatial DSP Helpers
# ---------------------------------------------------------------------------
def ev_s(event_key: str) -> float:
    return float(C.EV[event_key]) / float(C.FPS)


def frame_s(frame: float) -> float:
    return float(frame) / float(C.FPS)


def beat_span_s(beat_key: str) -> tuple[float, float]:
    f0, f1 = C.BEATS[beat_key]
    return ((f0 - 1) / float(C.FPS), f1 / float(C.FPS))


def load_sample(filename: str) -> np.ndarray:
    """Load a verified 48 kHz float32 stereo CC0/PD WAV sample with fallback to shared samples."""
    if filename not in _SAMPLE_CACHE:
        p = SAMPLES_DIR / filename
        if not p.is_file():
            p = FALLBACK_SAMPLES_DIR / filename
        if not p.is_file():
            raise FileNotFoundError(f"Missing required CC0/PD WAV sample: {filename}")
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


def add_range_reflections(
    stereo: np.ndarray,
    taps_ms: tuple[float, ...] = (9.5, 21.0, 38.0, 64.0),
    gains: tuple[float, ...] = (0.26, 0.16, 0.10, 0.055),
    lp_hz: float = 3400.0,
) -> np.ndarray:
    """Indoor concrete/steel ballistic test range spatial early reflections."""
    out = stereo.copy()
    refl = np.zeros_like(stereo)
    for i, (ms, g) in enumerate(zip(taps_ms, gains)):
        d = int(round(ms * 0.001 * SR))
        if 0 < d < len(stereo):
            if i % 2 == 0:
                refl[d:, 0] += stereo[:-d, 0] * g
                refl[d:, 1] += stereo[:-d, 1] * (g * 0.86)
            else:
                refl[d:, 0] += stereo[:-d, 1] * (g * 0.86)
                refl[d:, 1] += stereo[:-d, 0] * g
    out += butter_filter(refl, "low", lp_hz, order=2)
    return out.astype(np.float32)


def build_acoustic_sub_punch(
    dur_s: float = 0.28,
    f_start: float = 136.0,
    f_end: float = 42.0,
    tau_pitch: float = 0.019,
    tau_amp: float = 0.078,
    amp: float = 0.88,
) -> np.ndarray:
    """Physical 40-140 Hz acoustic chest/muzzle sub-punch layered underneath real PD recordings."""
    n = max(64, int(round(dur_s * SR)))
    t = np.arange(n, dtype=np.float64) / SR
    f_lo = float(np.clip(f_end, 40.0, 75.0))
    f_hi = float(np.clip(f_start, 95.0, 140.0))
    freq = f_lo + (f_hi - f_lo) * np.exp(-t / max(0.005, tau_pitch))
    sub = np.sin(phase_integral(freq)) * np.exp(-t / max(0.015, tau_amp))
    sub *= make_env(n, 0.0008, 0.045)
    return (sub * amp).astype(np.float32)


def build_realistic_gunshot(
    crack_file: str = "pd_gunshot_a.wav",
    pitch_scale: float = 1.0,
    sub_weight: float = 0.90,
    lp_hz: float | None = None,
    pan: float = -0.45,
) -> np.ndarray:
    """
    Build a multi-layered ballistic range gunshot strictly from real Public Domain WAVs:
      1. Crisp dry muzzle crack (`pd_gunshot_a/b/c.wav`)
      2. Pitched-down (0.56x) indoor concrete range boom tail (`pd_gunshot_b/c.wav`)
      3. Physical 40-140 Hz propellant sub-punch + subtle analog drive
      4. Indoor concrete ballistic bay spatial reflections
    """
    crack = load_sample(crack_file)
    if abs(pitch_scale - 1.0) > 1e-3:
        crack = pitch_speed(crack, pitch_scale)

    n = max(len(crack), int(round(0.88 * SR)))
    shot = np.zeros((n, 2), dtype=np.float32)
    shot[: len(crack)] += crack * 0.96

    body_src = "pd_gunshot_c.wav" if crack_file != "pd_gunshot_c.wav" else "pd_gunshot_b.wav"
    body = pitch_speed(load_sample(body_src), 0.56 * pitch_scale)
    body = butter_filter(body, "low", 1250.0, order=2)
    nb = min(n, len(body))
    shot[:nb] += body[:nb] * 0.76

    sub_kick = build_acoustic_sub_punch(
        dur_s=min(0.45, n / float(SR)),
        f_start=min(140.0, 134.0 * pitch_scale),
        f_end=42.0,
        tau_pitch=0.018,
        tau_amp=0.075,
        amp=sub_weight,
    )
    shot[: len(sub_kick)] += pan_stereo(sub_kick, 0.0)

    shot = (np.tanh(shot * 1.32) * 0.82).astype(np.float32)
    if lp_hz is not None:
        shot = butter_filter(shot, "low", lp_hz, order=2)

    panned = pan_stereo(shot, pan) * 0.72 + shot * 0.28
    return add_range_reflections(panned * make_env(n, 0.0005, 0.14)[:, None])


def build_armor_plate_impact(
    pan: float = 0.15,
    include_braam: bool = True,
    sub_amp: float = 0.92,
    bright_lp: float = 4200.0,
) -> np.ndarray:
    """
    Realistic ballistic armor strike impact built from real PD crack + CC0 sub-braam
    + 40-140 Hz chest-thud acoustic punch (ZERO toy synthetic beeps).
    """
    n = int(round(0.95 * SR))
    out = np.zeros((n, 2), dtype=np.float32)

    # 1. Sharp kinetic strike slap from real PD recording
    slap = load_sample("pd_gunshot_c.wav")[: int(round(0.30 * SR))]
    slap = butter_filter(slap, "band", (220.0, bright_lp), order=2) * make_env(len(slap), 0.0005, 0.14)[:, None]
    out[: len(slap)] += slap * 0.85

    # 2. Pitched-down composite/ceramic plate mass thump
    mass_hit = pitch_speed(load_sample("pd_gunshot_b.wav"), 0.48)[: int(round(0.48 * SR))]
    mass_hit = butter_filter(mass_hit, "low", 680.0, order=2) * make_env(len(mass_hit), 0.001, 0.20)[:, None]
    out[: len(mass_hit)] += mass_hit * 0.78

    # 3. Physical 42-132 Hz chest-plate acoustic punch
    sub = build_acoustic_sub_punch(dur_s=0.38, f_start=132.0, f_end=42.0, tau_pitch=0.022, tau_amp=0.095, amp=sub_amp)
    out[: len(sub)] += pan_stereo(sub, 0.0)

    if include_braam:
        braam = load_sample("cc0_sub_braam.wav")[: int(round(0.90 * SR))]
        braam = butter_filter(braam, "low", 340.0, order=2) * make_env(len(braam), 0.002, 0.38)[:, None]
        out[: len(braam)] += braam * 0.64

    out = (np.tanh(out * 1.25) * 0.84).astype(np.float32)
    panned = pan_stereo(out, pan) * 0.68 + out * 0.32
    return add_range_reflections(panned * make_env(n, 0.0005, 0.18)[:, None])


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
# Continuous Subtle Indoor Ballistic Range Room Tone
# ---------------------------------------------------------------------------
def build_ambient_bed() -> np.ndarray:
    n = TOTAL_SAMPLES
    t = np.arange(n, dtype=np.float32) / SR
    rng_l = np.random.default_rng(301)
    rng_r = np.random.default_rng(302)
    raw_l = rng_l.standard_normal(n).astype(np.float32)
    raw_r = rng_r.standard_normal(n).astype(np.float32)

    t_b5_s, t_b5_e = beat_span_s("B5_NEWTON")
    blueprint_dip = 1.0 - 0.58 * _sigmoid((t - t_b5_s) / 0.10) * _sigmoid((t_b5_e - t) / 0.10)

    hvac_l = butter_filter(raw_l, "band", (48.0, 210.0), order=2)
    hvac_r = butter_filter(raw_r, "band", (48.0, 210.0), order=2)
    bed = np.stack([hvac_l, hvac_r], axis=-1) * (0.036 * blueprint_dip)[:, None]
    return (bed * make_env(n, 0.03, 0.25)[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# Beat 1 (f=1..210): Slow-Mo Gunfire -> Chest Plate Slam -> Spark Shower
# ---------------------------------------------------------------------------
def render_beat1(bus: np.ndarray) -> None:
    t_fire = ev_s("b1_fire")           # f=22 (0.367 s)
    t_hit = ev_s("b1_impact")          # f=75 (1.250 s)
    t_sparks = ev_s("b1_sparks_peak")  # f=92 (1.533 s)
    rng = np.random.default_rng(1001)

    # 1. Slow-mo range gunshot discharge at f=22
    shot1 = build_realistic_gunshot(
        crack_file="pd_gunshot_b.wav",
        pitch_scale=0.84,
        sub_weight=0.92,
        lp_hz=5400.0,
        pan=-0.42,
    )
    add_stereo_event(bus, shot1, t_fire, gain=0.92)

    # 2. Fast supersonic bullet air-zip panning left -> Hero chest (f=24..75)
    dur_zip = max(0.25, t_hit - t_fire)
    n_zip = int(round(dur_zip * SR))
    nz = rng.standard_normal(n_zip).astype(np.float32)
    zip_hi = butter_filter(nz, "band", (1600.0, 6400.0), order=2)
    zip_mid = butter_filter(nz, "band", (380.0, 1400.0), order=2)
    zip_env = make_env(n_zip, 0.025, 0.04) * np.linspace(0.35, 1.0, n_zip, dtype=np.float32)
    add_stereo_event(bus, pan_sweep_stereo((zip_hi * 0.28 + zip_mid * 0.20) * zip_env, -0.45, 0.16), t_fire + 0.03, gain=0.88)

    # 3. Dead-center chest strike plate impact at f=75 (real PD slap + CC0 sub-braam + 40-140 Hz punch)
    plate_hit = build_armor_plate_impact(pan=0.15, include_braam=True, sub_amp=0.98, bright_lp=4600.0)
    add_stereo_event(bus, plate_hit, t_hit, gain=1.02)

    # 4. Hot tungsten/copper spark shower & scatter crackling peaking at f=92
    n_sp = int(round(0.95 * SR))
    sp_noise = butter_filter(rng.standard_normal(n_sp).astype(np.float32), "band", (2600.0, 8800.0), order=2)
    sp_crackle = (np.abs(rng.standard_normal(n_sp).astype(np.float32)) > 2.25).astype(np.float32)
    sp_sizzle = (sp_noise * (0.24 + 0.72 * sp_crackle)) * make_env(n_sp, 0.012, 0.42)
    add_stereo_event(bus, add_range_reflections(pan_sweep_stereo(sp_sizzle, 0.14, 0.45)), t_hit + 0.04, gain=0.72)

    for i, (pan_s, freq_s) in enumerate(
        [(-0.35, 3450.0), (0.48, 4120.0), (0.12, 5280.0), (-0.52, 3880.0), (0.62, 4640.0), (0.28, 6100.0)]
    ):
        dt = i * 0.055 + float(rng.uniform(0.0, 0.015))
        nf = int(round(0.14 * SR))
        tf = np.arange(nf, dtype=np.float64) / SR
        ping = (
            butter_filter(rng.standard_normal(nf).astype(np.float32), "band", (2400.0, 7400.0), order=2)
            * np.exp(-tf / 0.010) * 0.48
            + np.sin(2.0 * math.pi * freq_s * tf) * np.exp(-tf / 0.036) * 0.24
        ) * make_env(nf, 0.0005, 0.04)
        add_stereo_event(bus, add_range_reflections(pan_stereo(ping.astype(np.float32), pan_s)), t_sparks - 0.12 + dt, gain=0.62)


# ---------------------------------------------------------------------------
# Beat 2 (f=211..510): Dry Range Shot -> Shell Ping -> Contained Momentum Thud
# ---------------------------------------------------------------------------
def render_beat2(bus: np.ndarray) -> None:
    t_fire = ev_s("b2_muzzle_fire")  # f=245 (4.083 s)
    t_hit = ev_s("b2_vest_hit")      # f=335 (5.583 s)
    rng = np.random.default_rng(2002)

    # 1. Full dry indoor range gunshot right on muzzle flash (f=245)
    shot2 = build_realistic_gunshot(
        crack_file="pd_gunshot_a.wav",
        pitch_scale=1.0,
        sub_weight=0.95,
        lp_hz=None,
        pan=-0.45,
    )
    add_stereo_event(bus, shot2, t_fire, gain=1.02)

    # Ejected brass shell casing ping (~160 ms after discharge)
    n_sh = int(round(0.32 * SR))
    t_sh = np.arange(n_sh, dtype=np.float64) / SR
    casing = (
        0.22 * np.sin(2.0 * math.pi * 5320.0 * t_sh) * np.exp(-t_sh / 0.042)
        + 0.16 * np.sin(2.0 * math.pi * 7940.0 * t_sh) * np.exp(-t_sh / 0.030)
    ) * make_env(n_sh, 0.0005, 0.08)
    add_stereo_event(bus, add_range_reflections(pan_stereo(casing.astype(np.float32), -0.58)), t_fire + 0.16, gain=0.72)

    # 2. Air crack sweep across range bay (f=250..335)
    dur_air = max(0.25, t_hit - t_fire)
    n_air = int(round(dur_air * SR))
    air_n = butter_filter(rng.standard_normal(n_air).astype(np.float32), "band", (1300.0, 5600.0), order=2)
    add_stereo_event(bus, pan_sweep_stereo(air_n * make_env(n_air, 0.02, 0.05) * 0.26, -0.45, 0.35), t_fire + 0.04)

    # 3. Short, dry, contained vest thud at f=335 (proves tiny fastball momentum — no cinematic blast!)
    thud = build_armor_plate_impact(pan=0.35, include_braam=False, sub_amp=0.82, bright_lp=2400.0)
    add_stereo_event(bus, thud, t_hit, gain=0.92)


# ---------------------------------------------------------------------------
# Beat 3 (f=511..900): X-Ray Peel -> Kevlar Snare Catch -> Trampoline Ripple
# ---------------------------------------------------------------------------
def render_beat3(bus: np.ndarray) -> None:
    t_peel = ev_s("b3_xray_peel")       # f=535 (8.917 s)
    t_hit = ev_s("b3_kevlar_impact")    # f=615 (10.250 s)
    t_peak = ev_s("b3_ripple_peak")     # f=735 (12.250 s)
    rng = np.random.default_rng(3003)

    # 1. Clean X-ray cutaway peel pulse at f=535
    n_xr = int(round(0.45 * SR))
    tx = np.arange(n_xr, dtype=np.float64) / SR
    xray_sw = butter_filter(rng.standard_normal(n_xr).astype(np.float32), "band", (1200.0, 4800.0), order=2) * make_env(n_xr, 0.04, 0.20) * 0.22
    xray_tone = np.sin(phase_integral(np.linspace(920.0, 1840.0, n_xr))) * np.exp(-tx / 0.08) * 0.14
    add_stereo_event(bus, pan_sweep_stereo((xray_sw + xray_tone).astype(np.float32), -0.30, 0.30), t_peel, gain=0.85)

    # 2. Taut Kevlar weave bullet snare-catch at f=615
    snare_crack = load_sample("pd_gunshot_c.wav")[: int(round(0.24 * SR))]
    snare_crack = butter_filter(snare_crack, "band", (380.0, 3400.0), order=2) * make_env(len(snare_crack), 0.001, 0.12)[:, None]
    add_stereo_event(bus, pan_stereo(snare_crack, 0.05), t_hit, gain=0.78)

    kev_sub = build_acoustic_sub_punch(dur_s=0.42, f_start=122.0, f_end=46.0, tau_pitch=0.028, tau_amp=0.14, amp=0.92)
    add_stereo_event(bus, pan_stereo(kev_sub, 0.05), t_hit, gain=0.96)

    # 3. Elastic golden Kevlar trampoline membrane ripple & taut aramid tension wave (f=615..880)
    dur_rip = max(0.8, (t_peak - t_hit) + 2.2)
    n_rip = int(round(dur_rip * SR))
    tr = np.arange(n_rip, dtype=np.float64) / SR
    peak_g = 0.45 + 0.55 * np.exp(-((tr + t_hit - t_peak) ** 2) / (2.0 * 0.95 ** 2))
    # Elastic trampoline drum-head harmonics (64 Hz + 128 Hz + 196 Hz modulated by ripple oscillation)
    wob = 1.0 + 0.08 * np.sin(2.0 * math.pi * 6.5 * tr)
    mem = (
        0.52 * np.sin(phase_integral(64.0 * wob))
        + 0.26 * np.sin(phase_integral(128.0 * wob))
        + 0.14 * np.sin(phase_integral(194.0 * wob))
    )
    # Taut woven aramid fiber stretch friction (filtered stereo noise)
    fib_l = butter_filter(rng.standard_normal(n_rip).astype(np.float32), "band", (680.0, 2600.0), order=2)
    fib_r = butter_filter(rng.standard_normal(n_rip).astype(np.float32), "band", (680.0, 2600.0), order=2)
    rip_env = make_env(n_rip, 0.04, 0.55) * peak_g
    rip_st = np.stack(
        [(mem * 0.34 + fib_l * 0.18) * rip_env, (mem * 0.34 + fib_r * 0.18) * rip_env],
        axis=-1,
    ).astype(np.float32)
    add_stereo_event(bus, rip_st, t_hit + 0.03, gain=0.84)


# ---------------------------------------------------------------------------
# Beat 4 (f=901..1260): High-Velocity Rifle -> SiC Ceramic Shatter -> Dyneema
# ---------------------------------------------------------------------------
def render_beat4(bus: np.ndarray) -> None:
    t_fire = ev_s("b4_rifle_fire")        # f=925 (15.417 s)
    t_hit = ev_s("b4_rifle_impact")       # f=965 (16.083 s)
    t_shat = ev_s("b4_ceramic_shatter")   # f=1030 (17.167 s)
    t_catch = ev_s("b4_dyneema_catch")    # f=1125 (18.750 s)
    rng = np.random.default_rng(4004)

    # 1. High-velocity rifle shot discharge at f=925 (sharp supersonic crack)
    rifle = build_realistic_gunshot(
        crack_file="pd_gunshot_a.wav",
        pitch_scale=1.14,
        sub_weight=0.90,
        lp_hz=None,
        pan=-0.35,
    )
    add_stereo_event(bus, rifle, t_fire, gain=0.98)

    # 2. Silicon-Carbide ceramic strike face boom at f=965
    strike = build_armor_plate_impact(pan=0.0, include_braam=True, sub_amp=0.96, bright_lp=5800.0)
    add_stereo_event(bus, strike, t_hit, gain=0.98)

    # 3. Brittle ceramic tile fracture crunch + exploding SiC / rifle-tip shards at f=1030
    shat_crack = load_sample("pd_gunshot_c.wav")[: int(round(0.36 * SR))]
    shat_hi = butter_filter(shat_crack, "high", 1100.0, order=2) * make_env(len(shat_crack), 0.0005, 0.16)[:, None]
    add_stereo_event(bus, pan_stereo(shat_hi, 0.08), t_shat, gain=0.95)

    braam_sh = load_sample("cc0_sub_braam.wav")[: int(round(1.10 * SR))]
    braam_sh = butter_filter(braam_sh, "low", 420.0, order=2) * make_env(len(braam_sh), 0.002, 0.42)[:, None]
    add_stereo_event(bus, braam_sh, t_shat - 0.01, gain=0.78)

    # Granular porcelain/ceramic crunch burst + 10 brittle shard snaps across L/R
    n_gr = int(round(0.65 * SR))
    grit = butter_filter(rng.standard_normal(n_gr).astype(np.float32), "band", (1800.0, 7600.0), order=2)
    grit *= (np.abs(rng.standard_normal(n_gr).astype(np.float32)) > 1.35).astype(np.float32) * make_env(n_gr, 0.005, 0.35)
    add_stereo_event(bus, add_range_reflections(pan_sweep_stereo(grit, -0.15, 0.35)), t_shat, gain=0.68)

    shard_pans = [-0.72, 0.68, -0.48, 0.54, -0.80, 0.36, -0.22, 0.78, 0.15, -0.58]
    shard_freqs = [2850.0, 3680.0, 4420.0, 5260.0, 3290.0, 4950.0, 6120.0, 3910.0, 5580.0, 4190.0]
    for i, (sp, sfq) in enumerate(zip(shard_pans, shard_freqs)):
        dt = 0.010 + i * 0.068 + float(rng.uniform(-0.008, 0.012))
        nf = int(round(0.16 * SR))
        tf = np.arange(nf, dtype=np.float64) / SR
        shard = (
            butter_filter(rng.standard_normal(nf).astype(np.float32), "band", (2100.0, 7800.0), order=2)
            * np.exp(-tf / 0.009) * 0.56
            + np.sin(2.0 * math.pi * sfq * tf) * np.exp(-tf / 0.032) * 0.28
        ) * make_env(nf, 0.0005, 0.04)
        add_stereo_event(bus, add_range_reflections(pan_stereo(shard.astype(np.float32), sp)), t_shat + dt, gain=0.72)

    # 4. Dense UHMWPE Dyneema backer fiber catch thump at f=1125
    dyn_sub = build_acoustic_sub_punch(dur_s=0.38, f_start=110.0, f_end=44.0, tau_pitch=0.025, tau_amp=0.11, amp=0.86)
    n_df = int(round(0.45 * SR))
    dyn_fib = butter_filter(rng.standard_normal(n_df).astype(np.float32), "band", (220.0, 1450.0), order=2) * make_env(n_df, 0.010, 0.28) * 0.34
    add_stereo_event(bus, pan_stereo(dyn_sub, 0.15), t_catch, gain=0.92)
    add_stereo_event(bus, pan_stereo(dyn_fib, 0.15), t_catch, gain=0.82)


# ---------------------------------------------------------------------------
# Beat 5 (f=1261..1560): Equal & Opposite Shooter (-F2) vs Target (F1) Proof
# ---------------------------------------------------------------------------
def render_beat5(bus: np.ndarray) -> None:
    t_fire = ev_s("b5_dual_fire")     # f=1295 (21.583 s)
    t_hit = ev_s("b5_vest_absorb")    # f=1365 (22.750 s)
    t_lock = ev_s("b5_vectors_lock")  # f=1405 (23.417 s)

    # 1. Left-side Shooter firing recoil (-0.45 pan)
    shot5 = build_realistic_gunshot(
        crack_file="pd_gunshot_a.wav",
        pitch_scale=0.98,
        sub_weight=0.92,
        lp_hz=6400.0,
        pan=-0.45,
    )
    add_stereo_event(bus, shot5, t_fire, gain=0.94)

    # 2. Right-side Target chest plate equal-and-opposite impulse (+0.45 pan)
    hit5 = build_armor_plate_impact(pan=0.45, include_braam=False, sub_amp=0.92, bright_lp=3400.0)
    add_stereo_event(bus, hit5, t_hit, gain=0.94)

    # 3. Symmetrical F1 = -F2 vector lock-in braam + clean telemetry harmonic chime at f=1405
    braam5 = load_sample("cc0_sub_braam.wav")[: int(round(1.10 * SR))]
    braam5 = butter_filter(braam5, "low", 310.0, order=2) * make_env(len(braam5), 0.003, 0.48)[:, None]
    add_stereo_event(bus, braam5, t_lock - 0.01, gain=0.76)

    n_lk = int(round(0.65 * SR))
    tlk = np.arange(n_lk, dtype=np.float64) / SR
    lock_tone = (
        0.18 * np.sin(2.0 * math.pi * 587.33 * tlk) * np.exp(-tlk / 0.24)  # D5
        + 0.14 * np.sin(2.0 * math.pi * 880.00 * tlk) * np.exp(-tlk / 0.18)  # A5
    ) * make_env(n_lk, 0.006, 0.12)
    add_stereo_event(bus, pan_stereo(lock_tone.astype(np.float32), 0.0), t_lock, gain=0.85)


# ---------------------------------------------------------------------------
# Beat 6 (f=1561..1800): Spent Dust Tinkles & Calm Ballistic Bay Resolve
# ---------------------------------------------------------------------------
def render_beat6(bus: np.ndarray) -> None:
    t_s, _ = beat_span_s("B6_FINAL")
    t_steady = ev_s("b6_hero_steady")  # f=1620 (27.000 s)

    # Subtle spent ceramic/brass dust grains settling on the concrete floor
    for st, sp, pitch in [
        (t_s + 0.35, -0.18, 2680.0),
        (t_steady + 0.08, 0.24, 3340.0),
        (t_steady + 0.85, -0.10, 2410.0),
        (t_steady + 1.62, 0.28, 2960.0),
    ]:
        n_fl = int(round(0.36 * SR))
        tf = np.arange(n_fl, dtype=np.float64) / SR
        flake = (
            0.14 * np.sin(2.0 * math.pi * pitch * tf) * np.exp(-tf / 0.10)
            + 0.09 * np.sin(2.0 * math.pi * (pitch * 1.48) * tf) * np.exp(-tf / 0.07)
        ) * make_env(n_fl, 0.008, 0.10)
        add_stereo_event(bus, add_range_reflections(pan_stereo(flake.astype(np.float32), sp)), st, gain=0.58)

    # Warm low 44 Hz grounded resolve thump right as Hero stands steady
    res_sub = build_acoustic_sub_punch(dur_s=0.65, f_start=96.0, f_end=42.0, tau_pitch=0.04, tau_amp=0.22, amp=0.46)
    add_stereo_event(bus, pan_stereo(res_sub, 0.0), t_steady, gain=0.80)


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
