"""
Cinema Foley & Tactical Radar SFX Engineer for Episode #4 (Upgraded Hollywood Sound Design Edition)
("How a 172-Foot Stealth Bomber Looks Like a Bumblebee on Radar", 30.0 s @ 60 fps).

Generates:
    episodes/stealth-bomber-radar/audio/output/sfx_stem.wav
        - 48000 Hz, Stereo, PCM_16, exact 30.000 s (1,440,000 samples)
        - Frame-locked 1:1 to all 16 `stealth_contract.EV` milestones & `config/timeline.json` sfx_cues
        - Dynamic stratospheric high-altitude airflow bed + military radar dish rotation servo hum
        - Crisp 180 ms cinema transition whoosh risers leading into every hard beat cut
        - Authentic military RWR (Radar Warning Receiver) triple-beep target-lock flare in Beat 2
        - Synced digital telemetry counter ratchets (`100 m² -> 10 -> 1.0 -> 0.01 -> 0.001 m²`) +
          cinematic sub-drop + hyper-realistic stereo bumblebee wingbeat harmonics in Beat 4
        - Automatically layers shared CC0 sub-braam sample if present in repo
"""
from __future__ import annotations

import json
import math
import sys
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

AUDIO_DIR = Path(__file__).resolve().parent
EPISODE_ROOT = AUDIO_DIR.parent
REPO_ROOT = EPISODE_ROOT.parent.parent
SRC_DIR = EPISODE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import stealth_contract as C  # noqa: E402

TIMELINE_PATH = EPISODE_ROOT / "config" / "timeline.json"
OUTPUT_DIR = AUDIO_DIR / "output"
SFX_STEM_PATH = OUTPUT_DIR / "sfx_stem.wav"
CC0_SUB_BRAAM = REPO_ROOT / "episodes" / "underwater-bullets" / "audio" / "samples" / "cc0_sub_braam.wav"

SR = 48000
CHANNELS = 2
DURATION_S = float(C.DURATION_S)
TOTAL_SAMPLES = int(round(DURATION_S * SR))

RNG = np.random.default_rng(4042026)


def pan_stereo(mono: np.ndarray, pan: float) -> np.ndarray:
    pan = float(np.clip(pan, -1.0, 1.0))
    angle = (pan + 1.0) * (math.pi / 4.0)
    return np.stack([mono * math.cos(angle), mono * math.sin(angle)], axis=-1).astype(np.float32)


def add_stereo(bus: np.ndarray, event: np.ndarray, start_s: float, gain: float = 1.0) -> None:
    if event.ndim == 1:
        event = pan_stereo(event, 0.0)
    idx = int(round(start_s * SR))
    if idx >= len(bus) or idx + len(event) <= 0:
        return
    src_s = max(0, -idx)
    dst_s = max(0, idx)
    clen = min(len(event) - src_s, len(bus) - dst_s)
    if clen > 0:
        bus[dst_s : dst_s + clen] += event[src_s : src_s + clen] * float(gain)


def bp_filter(x: np.ndarray, lo_hz: float, hi_hz: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * SR
    lo = float(np.clip(lo_hz / nyq, 1e-4, 0.995))
    hi = float(np.clip(hi_hz / nyq, lo + 1e-4, 0.999))
    sos = signal.butter(order, [lo, hi], btype="band", output="sos")
    return signal.sosfilt(sos, x, axis=0).astype(np.float32)


def lp_filter(x: np.ndarray, cutoff_hz: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * SR
    sos = signal.butter(order, float(np.clip(cutoff_hz / nyq, 1e-4, 0.999)), btype="low", output="sos")
    return signal.sosfilt(sos, x, axis=0).astype(np.float32)


def hp_filter(x: np.ndarray, cutoff_hz: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * SR
    sos = signal.butter(order, float(np.clip(cutoff_hz / nyq, 1e-4, 0.999)), btype="high", output="sos")
    return signal.sosfilt(sos, x, axis=0).astype(np.float32)


def env_ad(n: int, att_s: float, dec_s: float, pow_dec: float = 2.0) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / SR
    att = np.clip(t / max(1e-4, att_s), 0.0, 1.0)
    dec = np.exp(-((np.maximum(0.0, t - att_s) / max(1e-4, dec_s)) ** pow_dec))
    return (att * dec).astype(np.float32)


def load_cc0_braam(target_dur: float = 1.65) -> np.ndarray | None:
    if not CC0_SUB_BRAAM.is_file():
        return None
    y, sr_in = sf.read(str(CC0_SUB_BRAAM), dtype="float32")
    if y.ndim == 2:
        y = np.mean(y, axis=1)
    if sr_in != SR:
        g = math.gcd(SR, int(sr_in))
        y = signal.resample_poly(y.astype(np.float64), SR // g, int(sr_in) // g).astype(np.float32)
    n_tgt = int(round(target_dur * SR))
    if len(y) > n_tgt:
        y = y[:n_tgt]
    fade = min(int(0.18 * SR), len(y) // 3)
    if fade > 0:
        y[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    pk = float(np.max(np.abs(y))) + 1e-8
    return (y / pk).astype(np.float32)


# ==============================================================================
# CINEMA TRANSITION WHOOSH & PROCEDURAL FOLEY SYNTHESIZERS
# ==============================================================================
def sfx_cut_whoosh(dur: float = 0.22, pan_dir: float = 1.0) -> np.ndarray:
    """Crisp 220 ms trailer transition air-whoosh leading tightly into a hard beat cut."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    noise = RNG.standard_normal(n).astype(np.float32)
    band = bp_filter(noise, 380.0, 4200.0, order=3)
    sub = np.sin(2.0 * math.pi * np.cumsum(np.linspace(55.0, 125.0, n)) / SR)
    # Ramps up sharply into the cut frame, tiny 18ms tail after cut
    env = (np.clip(t / (dur * 0.86), 0.0, 1.0) ** 2.1) * np.clip((dur - t) / (dur * 0.14), 0.0, 1.0)
    mono = (0.55 * band + 0.35 * sub) * env
    pan_c = np.linspace(-0.45 * pan_dir, +0.45 * pan_dir, n, dtype=np.float32)
    ang = (pan_c + 1.0) * (math.pi / 4.0)
    return np.stack([mono * np.cos(ang), mono * np.sin(ang)], axis=-1).astype(np.float32) * 0.56


def sfx_radar_dish_ping(freq: float = 1640.0, dur: float = 0.72) -> np.ndarray:
    """Crisp tactical ground-radar pulse ping + hydraulic dish servo tick + resonant sonar tail."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    chirp_f = np.linspace(freq * 1.18, freq * 0.96, n)
    phase = 2.0 * math.pi * np.cumsum(chirp_f) / SR
    ping = np.sin(phase) * np.exp(-t / 0.15)
    ping += 0.34 * np.sin(2.0 * phase + 0.4) * np.exp(-t / 0.075)
    ping += 0.16 * np.sin(0.5 * phase) * np.exp(-t / 0.24)
    click_n = int(0.020 * SR)
    click = bp_filter(RNG.standard_normal(click_n).astype(np.float32), 1400.0, 6800.0) * np.exp(
        -np.linspace(0, 8, click_n)
    )
    out = (ping * env_ad(n, 0.004, 0.25, 1.3)).astype(np.float32)
    out[:click_n] += click * 0.48
    return out * 0.80


def sfx_rf_deflect_whoosh(dur: float = 0.80, sweep_hi: float = 2900.0, sweep_lo: float = 400.0) -> np.ndarray:
    """Wide stereo RF deflection scatter whoosh sliding sideways off stealth facets."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    noise = RNG.standard_normal(n).astype(np.float32)
    f_sweep = sweep_hi * ((sweep_lo / sweep_hi) ** (t / dur))
    carrier = np.sin(2.0 * math.pi * np.cumsum(f_sweep) / SR)
    band = bp_filter(noise, 550.0, 5000.0, order=3)
    shimmer = bp_filter(noise, 2400.0, 8600.0, order=2) * (0.45 + 0.55 * np.sin(2.0 * math.pi * 21.0 * t))
    env = np.sin(math.pi * (t / dur)) ** 1.35 * np.exp(-t / 0.56)
    mono = (0.36 * carrier + 0.48 * band + 0.36 * shimmer) * env
    pan_curve = np.linspace(-0.40, 0.88, n, dtype=np.float32)
    ang = (pan_curve + 1.0) * (math.pi / 4.0)
    return np.stack([mono * np.cos(ang), mono * np.sin(ang)], axis=-1).astype(np.float32) * 0.84


def sfx_stealth_flyover_rumble(dur: float = 1.40) -> np.ndarray:
    """Eerie low-frequency flying-wing Doppler air displacement & distant muffled sub-jet glide."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    sub_f = np.linspace(56.0, 34.0, n)
    sub = np.sin(2.0 * math.pi * np.cumsum(sub_f) / SR)
    air = lp_filter(RNG.standard_normal(n).astype(np.float32), 260.0, order=3)
    whoosh = bp_filter(RNG.standard_normal(n).astype(np.float32), 180.0, 720.0, order=2)
    bell = np.sin(math.pi * (t / dur)) ** 1.45
    mono = (0.60 * sub + 0.46 * air + 0.28 * whoosh) * bell
    pan = np.linspace(-0.60, 0.60, n, dtype=np.float32)
    ang = (pan + 1.0) * (math.pi / 4.0)
    return np.stack([mono * np.cos(ang), mono * np.sin(ang)], axis=-1).astype(np.float32) * 0.88


def sfx_high_rcs_alarm_return(dur: float = 0.92) -> np.ndarray:
    """Military RWR (Radar Warning Receiver) triple-beep target-lock return + heavy metallic echo thump."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    # Authentic crisp 3-pulse military RWR lock alert
    out = np.zeros(n, dtype=np.float32)
    for k in range(3):
        b0 = int(round((0.02 + k * 0.145) * SR))
        blen = int(round(0.088 * SR))
        if b0 + blen < n:
            bt = np.arange(blen, dtype=np.float64) / SR
            be = np.sin(math.pi * bt / 0.088) ** 0.55
            beep = (
                0.55 * np.sin(2.0 * math.pi * 1245.0 * bt)
                + 0.30 * np.sin(2.0 * math.pi * 2490.0 * bt)
                + 0.18 * np.sin(2.0 * math.pi * 622.5 * bt)
            ) * be
            out[b0 : b0 + blen] += beep.astype(np.float32) * 0.65
    # Heavy low-frequency radar echo strike impact
    thump = np.sin(2.0 * math.pi * np.cumsum(np.linspace(145.0, 38.0, n)) / SR) * np.exp(-t / 0.14)
    return (out + thump.astype(np.float32) * 0.72) * 0.86


def sfx_sawtooth_scatter_chirp(dur: float = 0.75) -> np.ndarray:
    """Triple staggered crystalline RF pings bouncing sideways off sawtooth trailing-edge chevrons."""
    n = int(round(dur * SR))
    out = np.zeros((n, 2), dtype=np.float32)
    for k, (delay_s, f0, pan) in enumerate([(0.00, 2150.0, -0.50), (0.09, 2580.0, 0.15), (0.19, 3120.0, 0.75)]):
        sub_n = int(0.32 * SR)
        t = np.arange(sub_n, dtype=np.float64) / SR
        f = np.linspace(f0, f0 * 0.66, sub_n)
        w = np.sin(2.0 * math.pi * np.cumsum(f) / SR) * np.exp(-t / 0.085)
        w += 0.28 * bp_filter(RNG.standard_normal(sub_n).astype(np.float32), 2400.0, 7400.0) * np.exp(-t / 0.06)
        add_stereo(out, pan_stereo(w.astype(np.float32), pan), delay_s, gain=0.70 - 0.08 * k)
    return out


def sfx_xray_harmonic_sweep(dur: float = 1.08) -> np.ndarray:
    """Sleek high-tech X-ray cutaway scan sweep revealing internal serpentine ducts."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    f1 = np.linspace(260.0, 1520.0, n)
    f2 = np.linspace(520.0, 3040.0, n)
    p1 = 2.0 * math.pi * np.cumsum(f1) / SR
    p2 = 2.0 * math.pi * np.cumsum(f2) / SR
    glass = (0.50 * np.sin(p1) + 0.28 * np.sin(p2) + 0.15 * np.sin(1.5 * p1)) * np.sin(math.pi * t / dur) ** 1.2
    air = bp_filter(RNG.standard_normal(n).astype(np.float32), 1800.0, 7200.0) * np.sin(math.pi * t / dur) ** 1.5
    sub = np.sin(2.0 * math.pi * 62.0 * t) * np.exp(-t / 0.45)
    return (0.48 * glass + 0.30 * air + 0.36 * sub).astype(np.float32) * 0.80


def sfx_turbine_whine_duct_bounce(dur: float = 1.50) -> np.ndarray:
    """Buried jet compressor fan whine + hollow metallic RF ricochets inside curved S-duct."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    whine_f = 1850.0 + 280.0 * np.sin(2.0 * math.pi * 0.72 * t)
    whine = (
        0.36 * np.sin(2.0 * math.pi * np.cumsum(whine_f) / SR)
        + 0.22 * np.sin(2.0 * math.pi * np.cumsum(whine_f * 1.5) / SR)
    )
    whine = bp_filter(whine.astype(np.float32), 900.0, 3600.0, order=2) * np.sin(math.pi * t / dur) ** 1.1
    out = pan_stereo(whine * 0.56, 0.0)
    for delay_s, f_b, g_b, p_b in [
        (0.05, 1980.0, 0.74, -0.45),
        (0.34, 1420.0, 0.54, +0.40),
        (0.64, 940.0, 0.36, -0.25),
    ]:
        bn = int(0.28 * SR)
        bt = np.arange(bn, dtype=np.float64) / SR
        bw = np.sin(2.0 * math.pi * f_b * bt) * np.exp(-bt / 0.065)
        bw += 0.35 * np.sin(2.0 * math.pi * f_b * 0.5 * bt) * np.exp(-bt / 0.09)
        add_stereo(out, pan_stereo(bw.astype(np.float32), p_b), delay_s, gain=g_b)
    return out * 0.86


def sfx_rf_wave_damped_kill(dur: float = 0.88) -> np.ndarray:
    """Low damped absorption thud as trapped RF energy dies out inside the curved S-duct."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    f = np.linspace(380.0, 44.0, n)
    w = np.sin(2.0 * math.pi * np.cumsum(f) / SR) * np.exp(-t / 0.23)
    hiss = lp_filter(RNG.standard_normal(n).astype(np.float32), 780.0) * np.exp(-t / 0.19)
    return ((0.70 * w + 0.32 * hiss) * env_ad(n, 0.015, 0.36, 1.4)).astype(np.float32) * 0.84


def sfx_ram_rf_impact_buzz(dur: float = 0.78) -> np.ndarray:
    """Electromagnetic buzz + resonant ferrite pulse as RF hits the hexagonal RAM honeycomb."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    mod = 1.0 + 0.35 * np.sin(2.0 * math.pi * 64.0 * t)
    buzz = np.sin(2.0 * math.pi * 240.0 * t * mod) * 0.45 + np.sin(2.0 * math.pi * 720.0 * t) * 0.25
    crackle = bp_filter(RNG.standard_normal(n).astype(np.float32), 1800.0, 5800.0) * 0.38
    env = env_ad(n, 0.01, 0.28, 1.3)
    return ((buzz + crackle) * env).astype(np.float32) * 0.78


def sfx_thermal_dissipation_hiss(dur: float = 0.98) -> np.ndarray:
    """Warm sizzle/hiss of microwave energy turning into tiny harmless heat."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    noise = RNG.standard_normal(n).astype(np.float32)
    sizzle = bp_filter(noise, 2600.0, 9400.0, order=2) * np.sin(math.pi * t / dur) ** 1.3
    warm = lp_filter(noise, 420.0, order=2) * np.sin(math.pi * t / dur) ** 1.6
    return (0.50 * sizzle + 0.42 * warm).astype(np.float32) * 0.74


def sfx_rcs_collapse_sub_drop_bee(dur: float = 1.85) -> np.ndarray:
    """
    Signature viral payoff SFX synced to the live 3D counter (`100 -> 10 -> 1.0 -> 0.01 -> 0.001 m^2`):
    - 4 crisp descending digital telemetry ticks + deep 145->34 Hz sub-drop
    - Final crystalline lock chime at 0.001 m^2
    - Realistic stereo-fluttering bumblebee wingbeat harmonics!
    """
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    out = np.zeros((n, 2), dtype=np.float32)

    # Power-down sub-drop (145 Hz -> 34 Hz)
    drop_f = 145.0 * ((34.0 / 145.0) ** np.clip(t / 0.75, 0.0, 1.0))
    sub_drop = (np.sin(2.0 * math.pi * np.cumsum(drop_f) / SR) * np.exp(-t / 0.50)).astype(np.float32)
    add_stereo(out, pan_stereo(sub_drop * 0.74, 0.0), 0.0)

    # 4 descending digital counter ticks synced to visual milestones (f=1160, 1186, 1213, 1239)
    for k, (dt_s, freq_t) in enumerate([(0.00, 1980.0), (0.43, 1660.0), (0.88, 1380.0), (1.31, 2480.0)]):
        tn = int(0.18 * SR)
        tt = np.arange(tn, dtype=np.float64) / SR
        dec = 0.095 if k == 3 else 0.032
        tk = (np.sin(2.0 * math.pi * freq_t * tt) * np.exp(-tt / dec)).astype(np.float32)
        if k == 3:
            tk += (0.35 * np.sin(2.0 * math.pi * freq_t * 0.5 * tt) * np.exp(-tt / 0.14)).astype(np.float32)
        add_stereo(out, pan_stereo(tk, -0.15 + 0.12 * k), dt_s, gain=0.46 if k == 3 else 0.34)

    # Realistic stereo hovering bumblebee wingbeat harmonic (f=1190..1315)
    bee_mask = np.clip((t - 0.55) / 0.16, 0.0, 1.0) * np.clip((dur - t) / 0.24, 0.0, 1.0)
    bee_f = 214.0 + 16.0 * np.sin(2.0 * math.pi * 5.4 * t)
    bp = 2.0 * math.pi * np.cumsum(bee_f) / SR
    bee = (0.45 * np.sin(bp) + 0.36 * np.sin(2.0 * bp) + 0.24 * np.sin(3.0 * bp) + 0.14 * np.sin(4.0 * bp))
    bee = bp_filter(bee.astype(np.float32), 180.0, 1550.0, order=2) * bee_mask * 0.44
    pan_bee = 0.28 * np.sin(2.0 * math.pi * 2.1 * t)
    ang = (pan_bee + 1.0) * (math.pi / 4.0)
    out[:, 0] += bee * np.cos(ang).astype(np.float32)
    out[:, 1] += bee * np.sin(ang).astype(np.float32)
    return out * 0.90


def sfx_blueprint_mirror_ping(dur: float = 0.62) -> np.ndarray:
    """Clean CAD laboratory double-ping illustrating 90-degree mirror bounce back."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    p1 = np.sin(2.0 * math.pi * 1320.0 * t) * np.exp(-t / 0.07)
    d2 = int(0.16 * SR)
    p2 = np.zeros(n, dtype=np.float64)
    p2[d2:] = np.sin(2.0 * math.pi * 1760.0 * t[:-d2]) * np.exp(-t[:-d2] / 0.11)
    return (0.55 * p1 + 0.65 * p2).astype(np.float32) * 0.80


def sfx_blueprint_tilt_deflect(dur: float = 0.68) -> np.ndarray:
    """Clean upward CAD deflection sweep illustrating 33-degree tilted facet scattering."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    f = np.linspace(980.0, 3050.0, n)
    w = np.sin(2.0 * math.pi * np.cumsum(f) / SR) * np.sin(math.pi * t / dur) ** 1.2 * np.exp(-t / 0.38)
    air = bp_filter(RNG.standard_normal(n).astype(np.float32), 2000.0, 7200.0) * np.exp(-t / 0.24) * 0.26
    return (0.64 * w + air).astype(np.float32) * 0.82


def sfx_stealth_glide_sub_braam(dur: float = 1.70) -> np.ndarray:
    """Deep theatrical moonlight finale sub-braam (layers CC0 braam sample if available)."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    f = np.linspace(69.3, 34.65, n)  # C#2 -> C#1
    synth = (
        0.58 * np.sin(2.0 * math.pi * np.cumsum(f) / SR)
        + 0.28 * np.sin(2.0 * math.pi * np.cumsum(f * 2.0) / SR)
        + 0.14 * np.sin(2.0 * math.pi * np.cumsum(f * 3.0) / SR)
    )
    synth = lp_filter(np.tanh(synth * 1.45).astype(np.float32), 340.0, order=3) * env_ad(n, 0.04, 0.74, 1.35)
    cc0 = load_cc0_braam(dur)
    if cc0 is not None:
        m = min(len(cc0), n)
        synth[:m] = 0.52 * synth[:m] + 0.58 * cc0[:m]
    return synth * 0.90


def sfx_radar_scope_tiny_blip(dur: float = 0.98) -> np.ndarray:
    """Solitary crystalline phosphor scope ping resolving on the tiny 0.001 m^2 bumblebee speck."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    ping = np.sin(2.0 * math.pi * 2217.5 * t) * np.exp(-t / 0.20)
    ping += 0.28 * np.sin(2.0 * math.pi * 1108.7 * t) * np.exp(-t / 0.30)
    env = env_ad(n, 0.003, 0.36, 1.2)
    return (ping * env).astype(np.float32) * 0.76


def build_sfx_stem(tl: dict) -> np.ndarray:
    bus = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)

    # 1. Dynamic stratospheric night airflow + tactical radar command-room bed (quiet in Beat 5 CAD)
    t_all = np.arange(TOTAL_SAMPLES, dtype=np.float64) / SR
    air_mask = np.where((t_all >= 22.0) & (t_all < 26.0), 0.22, 1.0).astype(np.float32)
    air_l = bp_filter(RNG.standard_normal(TOTAL_SAMPLES).astype(np.float32), 65.0, 290.0, order=2) * 0.028 * air_mask
    air_r = bp_filter(RNG.standard_normal(TOTAL_SAMPLES).astype(np.float32), 65.0, 290.0, order=2) * 0.028 * air_mask
    hum = np.sin(2.0 * math.pi * 55.0 * t_all).astype(np.float32) * 0.012
    bus[:, 0] += air_l + hum
    bus[:, 1] += air_r + hum

    # 2. Subtle cinema transition whooshes tightly bridging every hard cut!
    for ci, cut_f in enumerate(C.HARD_CUT_FRAMES):
        t_cut = float(cut_f) / float(C.FPS)
        add_stereo(bus, sfx_cut_whoosh(0.22, pan_dir=1.0 if (ci % 2 == 0) else -1.0), t_cut - 0.18, gain=0.72)

    # 3. Crisp tactical CRT scope sweep ticks in Beat 1 (blank monitor) & Beat 6 (solitary speck)
    for ts in (1.05, 2.15, 26.85, 27.95, 29.05):
        tn = int(0.055 * SR)
        tt = np.arange(tn, dtype=np.float64) / SR
        tk = (np.sin(2.0 * math.pi * 2793.8 * tt) * np.exp(-tt / 0.014)).astype(np.float32)
        add_stereo(bus, pan_stereo(tk, 0.25 if ts < 5.0 else 0.0), ts, gain=0.22)

    # 4. All 16 frame-synced hero Foley cues from stealth_contract.EV
    builders = {
        "radar_dish_sweep_ping": lambda: sfx_radar_dish_ping(1640.0, 0.72),
        "rf_facet_deflect_whoosh": lambda: sfx_rf_deflect_whoosh(0.80, 2900.0, 400.0),
        "stealth_flyover_sub_rumble": lambda: sfx_stealth_flyover_rumble(1.40),
        "radar_pulse_burst": lambda: sfx_radar_dish_ping(1880.0, 0.58),
        "high_rcs_alarm_return": lambda: sfx_high_rcs_alarm_return(0.92),
        "rf_sawtooth_scatter_chirp": lambda: sfx_sawtooth_scatter_chirp(0.75),
        "xray_scan_harmonic_sweep": lambda: sfx_xray_harmonic_sweep(1.08),
        "turbine_whine_duct_bounce": lambda: sfx_turbine_whine_duct_bounce(1.50),
        "rf_wave_damped_kill": lambda: sfx_rf_wave_damped_kill(0.88),
        "ram_rf_impact_buzz": lambda: sfx_ram_rf_impact_buzz(0.78),
        "thermal_dissipation_hiss": lambda: sfx_thermal_dissipation_hiss(0.98),
        "rcs_collapse_sub_drop_bee": lambda: sfx_rcs_collapse_sub_drop_bee(1.85),
        "blueprint_mirror_ping_back": lambda: sfx_blueprint_mirror_ping(0.62),
        "blueprint_tilt_deflect_sweep": lambda: sfx_blueprint_tilt_deflect(0.68),
        "stealth_glide_sub_braam": lambda: sfx_stealth_glide_sub_braam(1.70),
        "radar_scope_tiny_blip_resolve": lambda: sfx_radar_scope_tiny_blip(0.98),
    }

    for cue in tl["sfx_cues"]:
        ev_key = str(cue["ev"])
        assert ev_key in C.EV, f"SFX cue {ev_key!r} missing from stealth_contract.EV"
        t_s = float(C.EV[ev_key]) / float(C.FPS)
        pan = float(cue.get("pan", 0.0))
        ctype = str(cue["type"])
        if ctype not in builders:
            raise ValueError(f"Unknown SFX cue type: {ctype!r}")
        ev_audio = builders[ctype]()
        if ev_audio.ndim == 1:
            ev_audio = pan_stereo(ev_audio, pan)
        add_stereo(bus, ev_audio, t_s, gain=0.92)

    bus = hp_filter(bus, 26.0, order=2)
    pk = float(np.max(np.abs(bus)))
    if pk > 0.88:
        bus *= 0.88 / pk

    micro = int(round(0.008 * SR))
    bus[:micro] *= np.linspace(0.0, 1.0, micro, dtype=np.float32)[:, None]
    bus[-micro:] *= np.linspace(1.0, 0.0, micro, dtype=np.float32)[:, None]
    return bus.astype(np.float32)


def verify_sfx(path: Path) -> None:
    info = sf.info(str(path))
    assert info.samplerate == SR and info.channels == CHANNELS and info.frames == TOTAL_SAMPLES
    assert info.subtype == "PCM_16" and abs(info.duration - DURATION_S) < 1e-6
    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == SR and wf.getnchannels() == CHANNELS and wf.getsampwidth() == 2
    y, _ = sf.read(str(path), dtype="float32")
    assert np.isfinite(y).all()
    pk = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    assert 0.015 < rms < 0.45 and pk < 0.94
    print(f"[SFX] Verified {path.name}: {SR}Hz 2ch PCM_16, {TOTAL_SAMPLES} samples, rms={rms:.4f}, peak={pk:.4f}")


def main() -> None:
    tl = json.loads(TIMELINE_PATH.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = build_sfx_stem(tl)
    sf.write(str(SFX_STEM_PATH), stem, SR, subtype="PCM_16")
    verify_sfx(SFX_STEM_PATH)


if __name__ == "__main__":
    main()
