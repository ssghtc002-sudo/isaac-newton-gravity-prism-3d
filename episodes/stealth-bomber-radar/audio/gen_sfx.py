"""
Cinema Foley & Tactical Radar SFX Engineer for Episode #4
("How a 172-Foot Stealth Bomber Looks Like a Bumblebee on Radar", 30.0 s @ 60 fps).

Generates:
    episodes/stealth-bomber-radar/audio/output/sfx_stem.wav
        - 48000 Hz, Stereo, PCM_16, exact 30.000 s (1,440,000 samples)
        - Frame-locked 1:1 to all 16 `stealth_contract.EV` milestones & `config/timeline.json` sfx_cues
        - Multi-layered procedural radar sonar pings, RF scatter whooshes, high-RCS warning echo,
          local-space X-ray harmonic sweep, buried turbine spool & damped S-duct ricochet,
          RAM carbon-ferrite thermal hiss, 100 m^2 -> bumblebee sub-drop + subtle wing buzz,
          CAD blueprint mirror/tilt pulses, and stealth moonlight flyover sub-braam
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
# INDIVIDUAL PROCEDURAL CINEMA FOLEY SYNTHESIZERS (16 CUES)
# ==============================================================================
def sfx_radar_dish_ping(freq: float = 1640.0, dur: float = 0.72) -> np.ndarray:
    """Crisp tactical ground-radar pulse ping + servo click + subtle sonar tail."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    # Sharp RF trigger chirp
    chirp_f = np.linspace(freq * 1.18, freq * 0.96, n)
    phase = 2.0 * math.pi * np.cumsum(chirp_f) / SR
    ping = np.sin(phase) * np.exp(-t / 0.14)
    ping += 0.32 * np.sin(2.0 * phase + 0.4) * np.exp(-t / 0.07)
    # Mechanical dish relay transient
    click_n = int(0.018 * SR)
    click = bp_filter(RNG.standard_normal(click_n).astype(np.float32), 1400.0, 6200.0) * np.exp(
        -np.linspace(0, 8, click_n)
    )
    out = (ping * env_ad(n, 0.004, 0.24, 1.3)).astype(np.float32)
    out[:click_n] += click * 0.45
    return out * 0.78


def sfx_rf_deflect_whoosh(dur: float = 0.78, sweep_hi: float = 2800.0, sweep_lo: float = 420.0) -> np.ndarray:
    """Wide stereo RF deflection scatter whoosh sliding sideways off stealth facets."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    noise = RNG.standard_normal(n).astype(np.float32)
    # Dual sweeping resonant bands
    f_sweep = sweep_hi * ((sweep_lo / sweep_hi) ** (t / dur))
    carrier = np.sin(2.0 * math.pi * np.cumsum(f_sweep) / SR)
    band = bp_filter(noise, 550.0, 4800.0, order=3)
    shimmer = bp_filter(noise, 2400.0, 8200.0, order=2) * (0.45 + 0.55 * np.sin(2.0 * math.pi * 19.0 * t))
    env = np.sin(math.pi * (t / dur)) ** 1.35 * np.exp(-t / 0.55)
    mono = (0.34 * carrier + 0.48 * band + 0.35 * shimmer) * env
    # Pan sweep L -> R to accentuate sideways scatter!
    pan_curve = np.linspace(-0.35, 0.85, n, dtype=np.float32)
    ang = (pan_curve + 1.0) * (math.pi / 4.0)
    st = np.stack([mono * np.cos(ang), mono * np.sin(ang)], axis=-1).astype(np.float32)
    return st * 0.82


def sfx_stealth_flyover_rumble(dur: float = 1.35) -> np.ndarray:
    """Eerie low-frequency flying-wing air displacement & distant muffled sub-jet pass."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    sub_f = np.linspace(54.0, 36.0, n)
    sub = np.sin(2.0 * math.pi * np.cumsum(sub_f) / SR)
    air = lp_filter(RNG.standard_normal(n).astype(np.float32), 260.0, order=3)
    whoosh = bp_filter(RNG.standard_normal(n).astype(np.float32), 180.0, 680.0, order=2)
    bell = np.sin(math.pi * (t / dur)) ** 1.5
    mono = (0.58 * sub + 0.45 * air + 0.25 * whoosh) * bell
    pan = np.linspace(-0.55, 0.55, n, dtype=np.float32)
    ang = (pan + 1.0) * (math.pi / 4.0)
    return np.stack([mono * np.cos(ang), mono * np.sin(ang)], axis=-1).astype(np.float32) * 0.86


def sfx_high_rcs_alarm_return(dur: float = 0.85) -> np.ndarray:
    """Blazing red conventional-jet radar echo return: sharp resonant metallic lock + urgent warning pulse."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    # Double radar-lock strobe tone + low impact thump
    pulse_gate = (np.sin(2.0 * math.pi * 8.5 * t) > -0.15).astype(np.float32)
    tone = (
        0.55 * np.sin(2.0 * math.pi * 1120.0 * t)
        + 0.35 * np.sin(2.0 * math.pi * 1680.0 * t)
        + 0.20 * np.sin(2.0 * math.pi * 560.0 * t)
    ) * (0.45 + 0.55 * pulse_gate)
    thump = np.sin(2.0 * math.pi * np.cumsum(np.linspace(135.0, 42.0, n)) / SR) * np.exp(-t / 0.12)
    env = env_ad(n, 0.008, 0.36, 1.4)
    return ((tone * 0.58 + thump * 0.65) * env).astype(np.float32) * 0.85


def sfx_sawtooth_scatter_chirp(dur: float = 0.72) -> np.ndarray:
    """Triple staggered crystalline RF pings bouncing off sawtooth trailing-edge chevrons."""
    n = int(round(dur * SR))
    out = np.zeros((n, 2), dtype=np.float32)
    for k, (delay_s, f0, pan) in enumerate([(0.00, 2150.0, -0.45), (0.09, 2580.0, 0.15), (0.19, 3050.0, 0.72)]):
        sub_n = int(0.32 * SR)
        t = np.arange(sub_n, dtype=np.float64) / SR
        f = np.linspace(f0, f0 * 0.68, sub_n)
        w = np.sin(2.0 * math.pi * np.cumsum(f) / SR) * np.exp(-t / 0.085)
        w += 0.25 * bp_filter(RNG.standard_normal(sub_n).astype(np.float32), 2400.0, 7200.0) * np.exp(-t / 0.06)
        add_stereo(out, pan_stereo(w.astype(np.float32), pan), delay_s, gain=0.68 - 0.08 * k)
    return out


def sfx_xray_harmonic_sweep(dur: float = 1.05) -> np.ndarray:
    """Sleek high-tech X-ray cutaway scan sweep revealing internal serpentine ducts."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    f1 = np.linspace(260.0, 1480.0, n)
    f2 = np.linspace(520.0, 2960.0, n)
    p1 = 2.0 * math.pi * np.cumsum(f1) / SR
    p2 = 2.0 * math.pi * np.cumsum(f2) / SR
    glass = (0.50 * np.sin(p1) + 0.28 * np.sin(p2) + 0.14 * np.sin(1.5 * p1)) * np.sin(math.pi * t / dur) ** 1.2
    air = bp_filter(RNG.standard_normal(n).astype(np.float32), 1800.0, 6800.0) * np.sin(math.pi * t / dur) ** 1.5
    sub = np.sin(2.0 * math.pi * 62.0 * t) * np.exp(-t / 0.45)
    return (0.48 * glass + 0.28 * air + 0.35 * sub).astype(np.float32) * 0.78


def sfx_turbine_whine_duct_bounce(dur: float = 1.45) -> np.ndarray:
    """Buried jet compressor fan whine + hollow metallic RF ricochets inside S-duct."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    # High-RPM compressor blade whine (muffled inside duct)
    whine_f = 1850.0 + 260.0 * np.sin(2.0 * math.pi * 0.7 * t)
    whine = (
        0.35 * np.sin(2.0 * math.pi * np.cumsum(whine_f) / SR)
        + 0.20 * np.sin(2.0 * math.pi * np.cumsum(whine_f * 1.5) / SR)
    )
    whine = bp_filter(whine.astype(np.float32), 900.0, 3400.0, order=2) * np.sin(math.pi * t / dur) ** 1.1
    # 3 rapid RF wall-bounce pings decreasing in pitch & energy
    out = pan_stereo(whine * 0.55, 0.0)
    for delay_s, f_b, g_b, p_b in [(0.05, 1920.0, 0.72, -0.4), (0.34, 1380.0, 0.52, 0.35), (0.64, 920.0, 0.34, -0.2)]:
        bn = int(0.28 * SR)
        bt = np.arange(bn, dtype=np.float64) / SR
        bw = np.sin(2.0 * math.pi * f_b * bt) * np.exp(-bt / 0.065)
        bw += 0.35 * np.sin(2.0 * math.pi * f_b * 0.5 * bt) * np.exp(-bt / 0.09)
        add_stereo(out, pan_stereo(bw.astype(np.float32), p_b), delay_s, gain=g_b)
    return out * 0.84


def sfx_rf_wave_damped_kill(dur: float = 0.85) -> np.ndarray:
    """Low damped absorption thud as trapped RF energy dies out inside the curved S-duct."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    f = np.linspace(380.0, 48.0, n)
    w = np.sin(2.0 * math.pi * np.cumsum(f) / SR) * np.exp(-t / 0.22)
    hiss = lp_filter(RNG.standard_normal(n).astype(np.float32), 750.0) * np.exp(-t / 0.18)
    return ((0.68 * w + 0.32 * hiss) * env_ad(n, 0.015, 0.35, 1.4)).astype(np.float32) * 0.82


def sfx_ram_rf_impact_buzz(dur: float = 0.75) -> np.ndarray:
    """Electromagnetic buzz + resonant ferrite pulse as RF hits the hexagonal RAM honeycomb."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    mod = 1.0 + 0.35 * np.sin(2.0 * math.pi * 64.0 * t)
    buzz = np.sin(2.0 * math.pi * 240.0 * t * mod) * 0.45 + np.sin(2.0 * math.pi * 720.0 * t) * 0.25
    crackle = bp_filter(RNG.standard_normal(n).astype(np.float32), 1800.0, 5600.0) * 0.35
    env = env_ad(n, 0.01, 0.28, 1.3)
    return ((buzz + crackle) * env).astype(np.float32) * 0.76


def sfx_thermal_dissipation_hiss(dur: float = 0.95) -> np.ndarray:
    """Warm sizzle/hiss of microwave energy turning into tiny harmless heat."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    noise = RNG.standard_normal(n).astype(np.float32)
    sizzle = bp_filter(noise, 2600.0, 9200.0, order=2) * np.sin(math.pi * t / dur) ** 1.3
    warm = lp_filter(noise, 420.0, order=2) * np.sin(math.pi * t / dur) ** 1.6
    return (0.48 * sizzle + 0.42 * warm).astype(np.float32) * 0.72


def sfx_rcs_collapse_sub_drop_bee(dur: float = 1.45) -> np.ndarray:
    """Signature viral payoff SFX: 100 m^2 -> 0.001 m^2 cinematic sub-drop resolving into a tiny bumblebee wing buzz!"""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    # Smooth power-down pitch drop (140 Hz -> 34 Hz)
    drop_f = 140.0 * ((34.0 / 140.0) ** np.clip(t / 0.62, 0.0, 1.0))
    sub_drop = np.sin(2.0 * math.pi * np.cumsum(drop_f) / SR) * np.exp(-t / 0.44)
    # Crisp lock-in ping at t=0.52s
    lock_n = int(0.24 * SR)
    lt = np.arange(lock_n, dtype=np.float64) / SR
    lock_ping = np.sin(2.0 * math.pi * 2460.0 * lt) * np.exp(-lt / 0.065)
    # Realistic gentle bumblebee wingbeat harmonic (215 Hz fundamental + 2x/3x overtones) from t=0.55s..1.40s
    bee_mask = np.clip((t - 0.48) / 0.14, 0.0, 1.0) * np.clip((dur - t) / 0.22, 0.0, 1.0)
    bee_f = 212.0 + 14.0 * np.sin(2.0 * math.pi * 5.2 * t)
    bp = 2.0 * math.pi * np.cumsum(bee_f) / SR
    bee = (0.45 * np.sin(bp) + 0.35 * np.sin(2.0 * bp) + 0.22 * np.sin(3.0 * bp) + 0.12 * np.sin(4.0 * bp))
    bee = bp_filter(bee.astype(np.float32), 180.0, 1450.0, order=2) * bee_mask * 0.42

    mono = (0.72 * sub_drop + bee).astype(np.float32)
    idx_lock = int(0.50 * SR)
    mono[idx_lock : idx_lock + lock_n] += (lock_ping * 0.42).astype(np.float32)
    return mono * 0.90


def sfx_blueprint_mirror_ping(dur: float = 0.62) -> np.ndarray:
    """Clean CAD laboratory double-ping illustrating 90-degree mirror bounce back."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    p1 = np.sin(2.0 * math.pi * 1320.0 * t) * np.exp(-t / 0.07)
    d2 = int(0.16 * SR)
    p2 = np.zeros(n, dtype=np.float64)
    p2[d2:] = np.sin(2.0 * math.pi * 1760.0 * t[:-d2]) * np.exp(-t[:-d2] / 0.11)
    return (0.55 * p1 + 0.65 * p2).astype(np.float32) * 0.78


def sfx_blueprint_tilt_deflect(dur: float = 0.68) -> np.ndarray:
    """Clean upward CAD deflection sweep illustrating 33-degree tilted facet scattering."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    f = np.linspace(980.0, 2950.0, n)
    w = np.sin(2.0 * math.pi * np.cumsum(f) / SR) * np.sin(math.pi * t / dur) ** 1.2 * np.exp(-t / 0.38)
    air = bp_filter(RNG.standard_normal(n).astype(np.float32), 2000.0, 7000.0) * np.exp(-t / 0.24) * 0.25
    return (0.62 * w + air).astype(np.float32) * 0.80


def sfx_stealth_glide_sub_braam(dur: float = 1.65) -> np.ndarray:
    """Deep theatrical moonlight finale sub-braam (layers CC0 braam sample if available)."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    f = np.linspace(69.3, 34.65, n)  # C#2 -> C#1
    synth = (
        0.58 * np.sin(2.0 * math.pi * np.cumsum(f) / SR)
        + 0.28 * np.sin(2.0 * math.pi * np.cumsum(f * 2.0) / SR)
        + 0.14 * np.sin(2.0 * math.pi * np.cumsum(f * 3.0) / SR)
    )
    synth = lp_filter(np.tanh(synth * 1.4).astype(np.float32), 340.0, order=3) * env_ad(n, 0.04, 0.72, 1.35)
    cc0 = load_cc0_braam(dur)
    if cc0 is not None:
        m = min(len(cc0), n)
        synth[:m] = 0.55 * synth[:m] + 0.55 * cc0[:m]
    return synth * 0.88


def sfx_radar_scope_tiny_blip(dur: float = 0.95) -> np.ndarray:
    """Solitary crystalline phosphor scope ping resolving on the tiny 0.001 m^2 bumblebee speck."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    ping = np.sin(2.0 * math.pi * 2217.5 * t) * np.exp(-t / 0.19)  # C#7 crystal ping
    ping += 0.25 * np.sin(2.0 * math.pi * 1108.7 * t) * np.exp(-t / 0.28)
    env = env_ad(n, 0.003, 0.34, 1.2)
    return (ping * env).astype(np.float32) * 0.72


def build_sfx_stem(tl: dict) -> np.ndarray:
    bus = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)

    # Subtle ambient dark tactical command-room air floor (non-intrusive, well below voice)
    t_all = np.arange(TOTAL_SAMPLES, dtype=np.float64) / SR
    room = lp_filter(RNG.standard_normal(TOTAL_SAMPLES).astype(np.float32), 160.0, order=2) * 0.018
    hum = np.sin(2.0 * math.pi * 55.0 * t_all).astype(np.float32) * 0.012
    bus[:, 0] += room + hum
    bus[:, 1] += np.roll(room, 97) + hum

    builders = {
        "radar_dish_sweep_ping": lambda: sfx_radar_dish_ping(1640.0, 0.72),
        "rf_facet_deflect_whoosh": lambda: sfx_rf_deflect_whoosh(0.78, 2800.0, 420.0),
        "stealth_flyover_sub_rumble": lambda: sfx_stealth_flyover_rumble(1.35),
        "radar_pulse_burst": lambda: sfx_radar_dish_ping(1880.0, 0.58),
        "high_rcs_alarm_return": lambda: sfx_high_rcs_alarm_return(0.85),
        "rf_sawtooth_scatter_chirp": lambda: sfx_sawtooth_scatter_chirp(0.72),
        "xray_scan_harmonic_sweep": lambda: sfx_xray_harmonic_sweep(1.05),
        "turbine_whine_duct_bounce": lambda: sfx_turbine_whine_duct_bounce(1.45),
        "rf_wave_damped_kill": lambda: sfx_rf_wave_damped_kill(0.85),
        "ram_rf_impact_buzz": lambda: sfx_ram_rf_impact_buzz(0.75),
        "thermal_dissipation_hiss": lambda: sfx_thermal_dissipation_hiss(0.95),
        "rcs_collapse_sub_drop_bee": lambda: sfx_rcs_collapse_sub_drop_bee(1.45),
        "blueprint_mirror_ping_back": lambda: sfx_blueprint_mirror_ping(0.62),
        "blueprint_tilt_deflect_sweep": lambda: sfx_blueprint_tilt_deflect(0.68),
        "stealth_glide_sub_braam": lambda: sfx_stealth_glide_sub_braam(1.65),
        "radar_scope_tiny_blip_resolve": lambda: sfx_radar_scope_tiny_blip(0.95),
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

    # Clean HPF + soft peak protection (< 0.92)
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
