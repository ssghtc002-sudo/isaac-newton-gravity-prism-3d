"""
gen_sfx.py — Procedural 48 kHz Stereo Helicopter Autorotation Foley & Mechanical SFX Designer
for Episode #5: "Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock" (30.0 s @ 60 fps).

Synthesizes all 16 action-synced events from `config/timeline.json`:
  - Turbine flame-out spool-down + dual-tone Master Caution cockpit alarm + collective lever slam
  - Crystalline X-ray sweep + seized engine shaft clunk + unmistakable bicycle-style sprag clutch ratchet
  - Upward emerald airflow rush + accelerating twin-blade rotor whop-whop + kinetic flywheel rise
  - Fluttering maple-seed samara whirl + matched helical vortex shimmer
  - Cockpit collective pull-up click + heavy 30-ft flare blade-bite air chop + downwash cushion sub-boom
  - Cushioned rooftop helipad skid touchdown thud + coasting freewheel rotor blade fade
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

AUDIO_DIR = Path(__file__).resolve().parent
EPISODE_ROOT = AUDIO_DIR.parent
SRC_DIR = EPISODE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import heli_contract as C  # noqa: E402

TIMELINE_PATH = EPISODE_ROOT / "config" / "timeline.json"
SFX_STEM_PATH = AUDIO_DIR / "output" / "sfx_stem.wav"

SR = 48000
CHANNELS = 2
DURATION_S = float(C.DURATION_S)
TOTAL_SAMPLES = int(round(DURATION_S * SR))


def _pan(mono: np.ndarray, pan: float) -> np.ndarray:
    a = (float(np.clip(pan, -1.0, 1.0)) + 1.0) * (math.pi / 4.0)
    return np.stack([mono * math.cos(a), mono * math.sin(a)], axis=-1).astype(np.float32)


def _bp(x: np.ndarray, f_lo: float, f_hi: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * SR
    sos = signal.butter(order, [max(1e-4, f_lo / nyq), min(0.999, f_hi / nyq)], btype="band", output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def _lp(x: np.ndarray, f_cut: float, order: int = 3) -> np.ndarray:
    sos = signal.butter(order, min(0.999, f_cut / (0.5 * SR)), btype="low", output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def _add(bus: np.ndarray, ev: np.ndarray, t0: float, pan: float = 0.0, gain: float = 1.0) -> None:
    st = _pan(ev * gain, pan) if ev.ndim == 1 else (ev * gain).astype(np.float32)
    idx = int(round(t0 * SR))
    if idx >= len(bus) or idx + len(st) <= 0:
        return
    s0 = max(0, -idx)
    d0 = max(0, idx)
    n = min(len(st) - s0, len(bus) - d0)
    if n > 0:
        bus[d0 : d0 + n] += st[s0 : s0 + n]


def _synth_cue(cue_type: str, rng: np.random.Generator) -> np.ndarray:
    if cue_type == "turbine_flameout_spool_down":
        t = np.linspace(0.0, 1.85, int(1.85 * SR), dtype=np.float64)
        freq = 1480.0 * np.exp(-1.55 * t) + 140.0
        phase = 2.0 * np.pi * np.cumsum(freq) / SR
        whine = (0.48 * np.sin(phase) + 0.24 * np.sin(2.0 * phase)) * np.exp(-1.1 * t)
        cough = _bp(rng.standard_normal(len(t)), 90.0, 420.0) * np.exp(-6.5 * t) * 1.15
        env = np.minimum(1.0, t / 0.03) * np.exp(-0.9 * t)
        return ((whine + cough) * env * 0.72).astype(np.float32)

    if cue_type == "cockpit_master_caution_beeps":
        t = np.linspace(0.0, 1.45, int(1.45 * SR), dtype=np.float64)
        out = np.zeros_like(t)
        for k in range(5):
            ts = k * 0.28
            m = (t >= ts) & (t < ts + 0.14)
            loc = t[m] - ts
            f0 = 880.0 if (k % 2 == 0) else 660.0
            e = np.sin(np.pi * loc / 0.14) ** 0.35
            out[m] += (0.65 * np.sin(2.0 * np.pi * f0 * loc) + 0.25 * np.sin(4.0 * np.pi * f0 * loc)) * e
        return (out * 0.42).astype(np.float32)

    if cue_type == "collective_lever_mechanical_slam":
        t = np.linspace(0.0, 0.38, int(0.38 * SR), dtype=np.float64)
        thud = np.sin(2.0 * np.pi * (145.0 * np.exp(-12.0 * t) + 48.0) * t) * np.exp(-14.0 * t)
        clack = _bp(rng.standard_normal(len(t)), 900.0, 3600.0) * np.exp(-32.0 * t)
        return ((0.80 * thud + 0.65 * clack) * 0.85).astype(np.float32)

    if cue_type == "xray_scan_harmonic_sweep":
        t = np.linspace(0.0, 1.10, int(1.10 * SR), dtype=np.float64)
        freq = 320.0 + 1350.0 * (t / 1.10) ** 1.4
        ph = 2.0 * np.pi * np.cumsum(freq) / SR
        env = np.sin(np.pi * t / 1.10) ** 0.7
        return ((0.45 * np.sin(ph) + 0.22 * np.sin(2.0 * ph)) * env * 0.55).astype(np.float32)

    if cue_type == "engine_shaft_seize_clunk":
        t = np.linspace(0.0, 0.55, int(0.55 * SR), dtype=np.float64)
        ring = (np.sin(2.0 * np.pi * 240.0 * t) + 0.5 * np.sin(2.0 * np.pi * 610.0 * t)) * np.exp(-11.0 * t)
        sub = np.sin(2.0 * np.pi * (95.0 * np.exp(-9.0 * t) + 38.0) * t) * np.exp(-8.5 * t)
        grind = _bp(rng.standard_normal(len(t)), 600.0, 2400.0) * np.exp(-18.0 * t)
        return ((0.55 * ring + 0.85 * sub + 0.50 * grind) * 0.82).astype(np.float32)

    if cue_type == "sprag_clutch_bicycle_ratchet":
        t = np.linspace(0.0, 1.65, int(1.65 * SR), dtype=np.float64)
        out = np.zeros_like(t)
        # Authentic bicycle freewheel ratchet clicks (~26 clicks/sec crisp metallic ticks!)
        n_clicks = 42
        for k in range(n_clicks):
            ts = k * 0.036
            if ts >= 1.55:
                break
            m = (t >= ts) & (t < ts + 0.018)
            loc = t[m] - ts
            tick = (np.sin(2.0 * np.pi * 2650.0 * loc) + 0.6 * np.sin(2.0 * np.pi * 4300.0 * loc)) * np.exp(-260.0 * loc)
            out[m] += tick * (1.0 - 0.35 * (k / float(n_clicks)))
        return (out * 0.78).astype(np.float32)

    if cue_type == "upward_airflow_rush_whoosh":
        t = np.linspace(0.0, 2.40, int(2.40 * SR), dtype=np.float64)
        n = rng.standard_normal(len(t))
        air = _bp(n, 220.0, 1400.0) * (0.4 + 0.6 * np.sin(np.pi * t / 2.40))
        whist = np.sin(2.0 * np.pi * np.cumsum(380.0 + 340.0 * (t / 2.40)) / SR) * 0.16 * np.sin(np.pi * t / 2.40)
        return ((air * 0.45 + whist) * 0.72).astype(np.float32)

    if cue_type == "rotor_blade_slap_windmill":
        t = np.linspace(0.0, 3.20, int(3.20 * SR), dtype=np.float64)
        # Accelerating twin-blade helicopter whop-whop (~5.5 Hz -> 7.2 Hz)
        blade_hz = 5.4 + 1.8 * (t / 3.20)
        ph = 2.0 * np.pi * np.cumsum(blade_hz) / SR
        pulse = (0.5 + 0.5 * np.sin(ph)) ** 3.2
        wind = _bp(rng.standard_normal(len(t)), 75.0, 480.0)
        sub = np.sin(2.0 * np.pi * 58.0 * t) * 0.42
        env = np.sin(np.pi * t / 3.20) ** 0.35
        return (((wind * 0.95 + sub) * (0.28 + 0.72 * pulse)) * env * 0.78).astype(np.float32)

    if cue_type == "kinetic_flywheel_harmonic_rise":
        t = np.linspace(0.0, 1.50, int(1.50 * SR), dtype=np.float64)
        freq = 180.0 + 440.0 * (t / 1.50) ** 1.3
        ph = 2.0 * np.pi * np.cumsum(freq) / SR
        env = np.sin(np.pi * t / 1.50) ** 0.65
        return ((0.55 * np.sin(ph) + 0.25 * np.sin(2.0 * ph)) * env * 0.55).astype(np.float32)

    if cue_type == "maple_seed_flutter_whirl":
        t = np.linspace(0.0, 2.20, int(2.20 * SR), dtype=np.float64)
        flut = (0.5 + 0.5 * np.sin(2.0 * np.pi * 11.5 * t)) ** 2.2
        air = _bp(rng.standard_normal(len(t)), 520.0, 2600.0)
        env = np.sin(np.pi * t / 2.20) ** 0.5
        return (air * (0.3 + 0.7 * flut) * env * 0.55).astype(np.float32)

    if cue_type == "autorotation_vortex_pulse":
        t = np.linspace(0.0, 1.80, int(1.80 * SR), dtype=np.float64)
        air = _bp(rng.standard_normal(len(t)), 160.0, 920.0) * (0.4 + 0.6 * np.sin(2.0 * np.pi * 6.5 * t) ** 2)
        tone = np.sin(2.0 * np.pi * 293.66 * t) * 0.14
        env = np.sin(np.pi * t / 1.80) ** 0.5
        return ((air * 0.55 + tone) * env * 0.65).astype(np.float32)

    if cue_type == "collective_pull_ratchet_click":
        t = np.linspace(0.0, 0.42, int(0.42 * SR), dtype=np.float64)
        click = _bp(rng.standard_normal(len(t)), 1100.0, 4200.0) * np.exp(-38.0 * t)
        pull = np.sin(2.0 * np.pi * (120.0 + 180.0 * t) * t) * np.exp(-7.5 * t)
        return ((0.75 * click + 0.65 * pull) * 0.80).astype(np.float32)

    if cue_type == "flare_blade_bite_air_thump":
        t = np.linspace(0.0, 1.35, int(1.35 * SR), dtype=np.float64)
        # Heavy positive-pitch blade-slap chop biting dense air at 30 ft!
        chop = (0.5 + 0.5 * np.sin(2.0 * np.pi * 7.8 * t)) ** 4.0
        air = _bp(rng.standard_normal(len(t)), 65.0, 620.0)
        sub = np.sin(2.0 * np.pi * (72.0 * np.exp(-1.8 * t) + 38.0) * t) * 0.65
        env = np.minimum(1.0, t / 0.04) * np.exp(-1.1 * t)
        return (((air * 1.1 + sub) * (0.35 + 0.65 * chop)) * env * 0.86).astype(np.float32)

    if cue_type == "downwash_cushion_sub_boom":
        t = np.linspace(0.0, 1.25, int(1.25 * SR), dtype=np.float64)
        sub = np.sin(2.0 * np.pi * (82.0 * np.exp(-2.8 * t) + 34.0) * t) * np.exp(-2.2 * t)
        wash = _lp(rng.standard_normal(len(t)), 360.0) * np.exp(-1.8 * t)
        return ((0.85 * sub + 0.55 * wash) * 0.82).astype(np.float32)

    if cue_type == "helipad_skid_touchdown_thud":
        t = np.linspace(0.0, 0.65, int(0.65 * SR), dtype=np.float64)
        thud = np.sin(2.0 * np.pi * (108.0 * np.exp(-14.0 * t) + 36.0) * t) * np.exp(-9.5 * t)
        scrape = _bp(rng.standard_normal(len(t)), 350.0, 1800.0) * np.exp(-20.0 * t)
        return ((0.90 * thud + 0.42 * scrape) * 0.85).astype(np.float32)

    # coasting_blade_freewheel_fade
    t = np.linspace(0.0, 1.85, int(1.85 * SR), dtype=np.float64)
    hz = 5.2 * np.exp(-0.65 * t) + 1.4
    ph = 2.0 * np.pi * np.cumsum(hz) / SR
    chop = (0.5 + 0.5 * np.sin(ph)) ** 2.6
    air = _bp(rng.standard_normal(len(t)), 90.0, 440.0)
    env = np.exp(-1.15 * t)
    return (air * (0.25 + 0.75 * chop) * env * 0.65).astype(np.float32)


def main() -> None:
    tl = json.loads(TIMELINE_PATH.read_text(encoding="utf-8"))
    rng = np.random.default_rng(5050)
    bus = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)

    # Subtle continuous airborne cockpit/skyline airflow bed (quietly anchors realism!)
    t_all = np.linspace(0.0, DURATION_S, TOTAL_SAMPLES, dtype=np.float64)
    amb = _bp(rng.standard_normal(TOTAL_SAMPLES), 85.0, 320.0) * 0.045
    blade_bed_hz = np.where(t_all < 28.0, 5.8, 3.2)
    blade_mod = 0.65 + 0.35 * np.sin(2.0 * np.pi * np.cumsum(blade_bed_hz) / SR)
    bus[:, 0] += (amb * blade_mod).astype(np.float32)
    bus[:, 1] += (amb * blade_mod * 0.96).astype(np.float32)

    for cue in tl.get("sfx_cues", []):
        ev_audio = _synth_cue(str(cue["type"]), rng)
        _add(bus, ev_audio, float(cue["t"]), pan=float(cue.get("pan", 0.0)), gain=0.92)

    peak = float(np.max(np.abs(bus)))
    if peak > 0.88:
        bus *= 0.88 / peak
    micro = int(round(0.008 * SR))
    bus[:micro] *= np.linspace(0.0, 1.0, micro, dtype=np.float32)[:, None]
    bus[-micro:] *= np.linspace(1.0, 0.0, micro, dtype=np.float32)[:, None]

    SFX_STEM_PATH.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(SFX_STEM_PATH), bus, SR, subtype="PCM_16")
    print(f"[SFX] Generated {SFX_STEM_PATH.name} ({len(tl.get('sfx_cues', []))} helicopter cues, peak={float(np.max(np.abs(bus))):.3f})")


if __name__ == "__main__":
    main()
