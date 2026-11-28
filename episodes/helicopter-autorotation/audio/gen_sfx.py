"""
gen_sfx.py — Blockbuster Multi-Layered 48 kHz Stereo Aviation Foley & Mechanical SFX Designer (V2 Ultimate)
for Episode #5: "Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock" (30.0 s @ 60 fps).

V2 Ultimate Audio Overhaul:
  - Beat 1A (0.00-1.92s): Explosive turbine compressor-stall BANG + metallic backfire crack + flame whoosh
    + realistic high-RPM turbine spool-down whine (2400 Hz -> 160 Hz).
  - Beat 1B (1.92-4.00s): Acoustic shift into dark enclosed cockpit + authentic dual-tone Master Caution horn
    & low-RPM wailer (intercom bandpass + cabin reflections) + visceral 108 BPM cockpit heartbeat sub-thump
    + heavy aluminum collective lever slam & hydraulic detent snap.
  - Beat 2  (4.00-9.50s): Holographic X-ray sweep + hardened-steel gear-tooth screech & dead-stop seize clunk
    + hyper-real metallic bicycle-freewheel sprag clutch ratchet (~32 clicks/sec multi-modal steel ring).
  - Beat 3  (9.50-16.00s): Sculpted upward airflow column rush + authentic BVI helicopter rotor slap
    (`WHOP-WHOP-WHOP` blade-vortex interaction!) + kinetic flywheel harmonic charge.
  - Beat 4  (16.00-21.50s): Organic papery maple-seed samara flutter + synchronized stereo vortex shimmer.
  - Beat 5A (21.50-24.17s): Enclosed cockpit heartbeat acceleration + heavy collective haul + 0.35s pre-flare
    ambient vacuum drop (`24.02..24.37s`) for maximum Beat 5B goosebump impact!
  - Beat 5B (24.17-26.33s): Massive post-vacuum 30-foot flare blade-bite air chop + chest-hitting sub boom.
  - Beat 6  (26.33-30.00s): Steel-skid-on-concrete rooftop touchdown thud + slowing freewheel rotor coast-down.
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


def _hp(x: np.ndarray, f_cut: float, order: int = 2) -> np.ndarray:
    sos = signal.butter(order, max(1e-4, f_cut / (0.5 * SR)), btype="high", output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def _soft_clip(x: np.ndarray, drive: float = 1.6) -> np.ndarray:
    return (np.tanh(x * drive) / math.tanh(drive)).astype(np.float32)


def _short_room(mono: np.ndarray, wet: float = 0.22) -> np.ndarray:
    """Tiny metallic cockpit/fuselage early-reflection stereo widener (zero mud)."""
    d1, d2 = int(0.011 * SR), int(0.019 * SR)
    l = mono.copy()
    r = mono.copy()
    if len(mono) > d2:
        l[d1:] += mono[:-d1] * wet
        r[d2:] += mono[:-d2] * (wet * 0.92)
    return np.stack([l, r], axis=-1).astype(np.float32)


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


def _synth_heartbeat(t_start: float, t_end: float, bpm: float, bus: np.ndarray, gain: float = 0.34) -> None:
    """Visceral enclosed-cockpit low-frequency heartbeat (`lub-dub`) amplifying pilot tension."""
    step_s = 60.0 / bpm
    t_cur = t_start
    while t_cur < t_end - 0.25:
        for offset, f0, amp in ((0.0, 56.0, 1.0), (0.17, 46.0, 0.72)):
            dur = 0.14
            n = int(dur * SR)
            lt = np.linspace(0.0, dur, n, dtype=np.float64)
            f = f0 * np.exp(-9.0 * lt) + 28.0
            ph = 2.0 * np.pi * np.cumsum(f) / SR
            thump = np.sin(ph) * np.minimum(1.0, lt / 0.012) * np.exp(-18.0 * lt) * amp
            _add(bus, thump.astype(np.float32), t_cur + offset, pan=0.0, gain=gain)
        t_cur += step_s


def _synth_cue(cue_type: str, rng: np.random.Generator) -> np.ndarray:
    if cue_type == "turbine_flameout_spool_down":
        # Beat 1A: Explosive compressor-stall BANG + metallic crack + roaring flame burst + turbine spool-down!
        t = np.linspace(0.0, 1.68, int(1.68 * SR), dtype=np.float64)
        # 1) Sub-bass compressor stall explosion punch (95 Hz -> 30 Hz)
        f_sub = 95.0 * np.exp(-12.0 * t) + 30.0
        sub_bang = np.sin(2.0 * np.pi * np.cumsum(f_sub) / SR) * np.exp(-5.8 * t) * 1.15
        # 2) Sharp metallic turbine-blade snap / backfire crack
        crack = _bp(rng.standard_normal(len(t)), 900.0, 5800.0) * np.exp(-26.0 * t) * 1.20
        # 3) Roaring kerosene flame whoosh (mid-low combustion turbulence)
        fire = _bp(rng.standard_normal(len(t)), 140.0, 1300.0) * np.minimum(1.0, t / 0.025) * np.exp(-2.8 * t) * 0.95
        # 4) High-RPM jet turbine spool-down whine (2450 Hz -> 165 Hz)
        f_whine = 2280.0 * np.exp(-1.75 * t) + 165.0
        ph_w = 2.0 * np.pi * np.cumsum(f_whine) / SR
        whine = (0.45 * np.sin(ph_w) + 0.22 * np.sin(2.0 * ph_w) + 0.10 * np.sin(3.0 * ph_w)) * np.exp(-1.15 * t)
        # 5) Secondary exhaust sputter coughs at +0.28s & +0.58s
        sputter = np.zeros_like(t)
        for tc in (0.28, 0.58):
            m = t >= tc
            lt = t[m] - tc
            sputter[m] += _bp(rng.standard_normal(len(lt)), 120.0, 540.0) * np.exp(-14.0 * lt) * 0.65
        mix = _soft_clip((sub_bang + crack + fire + whine * 0.72 + sputter) * 0.62, drive=1.5)
        return _short_room(mix, wet=0.18)

    if cue_type == "cockpit_master_caution_beeps":
        # Beat 1B: Authentic helicopter dual-tone Master Caution horn + urgent low-RPM warbler!
        t = np.linspace(0.0, 1.85, int(1.85 * SR), dtype=np.float64)
        out = np.zeros_like(t)
        # Dual-tone cockpit horn bursts
        for k in range(6):
            ts = k * 0.27
            m = (t >= ts) & (t < ts + 0.145)
            loc = t[m] - ts
            f0 = 920.0 if (k % 2 == 0) else 690.0
            e = np.minimum(1.0, loc / 0.008) * np.minimum(1.0, (0.145 - loc) / 0.012)
            # Slightly square/buzzy aviation buzzer harmonic profile
            sig = (
                0.62 * np.sin(2.0 * np.pi * f0 * loc)
                + 0.28 * np.sin(6.0 * np.pi * f0 * loc)
                + 0.15 * np.sin(10.0 * np.pi * f0 * loc)
            )
            out[m] += sig * e
        # Background low-RPM urgent 4.5 Hz subtle wailer under the horn
        wail_f = 1320.0 + 180.0 * np.sin(2.0 * np.pi * 4.8 * t)
        wail = np.sin(2.0 * np.pi * np.cumsum(wail_f) / SR) * 0.16 * np.sin(np.pi * t / 1.85) ** 0.4
        # Intercom bandpass (520..3800 Hz) for real aviation headset grit!
        filtered = _bp((out + wail).astype(np.float32), 520.0, 3800.0) * 0.56
        return _short_room(filtered, wet=0.25)

    if cue_type == "collective_lever_mechanical_slam":
        # Heavy aluminum collective rod slam + mechanical bottom detent latch + hydraulic hiss!
        t = np.linspace(0.0, 0.48, int(0.48 * SR), dtype=np.float64)
        thud = np.sin(2.0 * np.pi * (155.0 * np.exp(-15.0 * t) + 46.0) * t) * np.exp(-13.0 * t)
        clack = _bp(rng.standard_normal(len(t)), 1100.0, 4800.0) * np.exp(-36.0 * t)
        metal_ring = (np.sin(2.0 * np.pi * 840.0 * t) + 0.5 * np.sin(2.0 * np.pi * 1690.0 * t)) * np.exp(-22.0 * t)
        hiss = _hp(rng.standard_normal(len(t)), 2400.0) * np.minimum(1.0, t / 0.03) * np.exp(-8.5 * t) * 0.22
        return _short_room(_soft_clip((0.85 * thud + 0.72 * clack + 0.35 * metal_ring + hiss) * 0.78), wet=0.20)

    if cue_type == "xray_scan_harmonic_sweep":
        # Crystalline high-tech holographic cutaway sweep with shimmer stereo harmonics
        t = np.linspace(0.0, 1.15, int(1.15 * SR), dtype=np.float64)
        freq = 280.0 + 1580.0 * (t / 1.15) ** 1.35
        ph = 2.0 * np.pi * np.cumsum(freq) / SR
        env = np.sin(np.pi * t / 1.15) ** 0.65
        tone = (0.42 * np.sin(ph) + 0.24 * np.sin(1.5 * ph) + 0.14 * np.sin(2.0 * ph)) * env
        air = _bp(rng.standard_normal(len(t)), 1800.0, 6200.0) * env * 0.18
        return _short_room(((tone + air) * 0.54).astype(np.float32), wet=0.30)

    if cue_type == "engine_shaft_seize_clunk":
        # Hardened-steel turbine gear jam: metallic screech -> brutal iron dead-stop clunk -> sub resonance
        t = np.linspace(0.0, 0.68, int(0.68 * SR), dtype=np.float64)
        screech_f = 1850.0 * np.exp(-8.0 * t) + 520.0
        screech = np.sin(2.0 * np.pi * np.cumsum(screech_f) / SR) * np.exp(-16.0 * t) * 0.36
        ring = (
            np.sin(2.0 * np.pi * 265.0 * t)
            + 0.55 * np.sin(2.0 * np.pi * 635.0 * t)
            + 0.30 * np.sin(2.0 * np.pi * 1410.0 * t)
        ) * np.exp(-11.5 * t)
        sub = np.sin(2.0 * np.pi * (102.0 * np.exp(-11.0 * t) + 34.0) * t) * np.exp(-7.8 * t)
        grind = _bp(rng.standard_normal(len(t)), 700.0, 3200.0) * np.exp(-18.0 * t)
        return _short_room(_soft_clip((0.55 * ring + 0.95 * sub + 0.52 * grind + screech) * 0.78), wet=0.22)

    if cue_type == "sprag_clutch_bicycle_ratchet":
        # Hyper-real metallic bicycle-freewheel sprag clutch ratchet (~32 crisp clicks/sec!)
        t = np.linspace(0.0, 1.90, int(1.90 * SR), dtype=np.float64)
        out_l = np.zeros_like(t)
        out_r = np.zeros_like(t)
        n_clicks = 58
        ts = 0.0
        for k in range(n_clicks):
            if ts >= 1.82:
                break
            m = (t >= ts) & (t < ts + 0.016)
            loc = t[m] - ts
            # Multi-modal hardened bearing steel click (2.8 kHz + 4.9 kHz + 7.6 kHz crisp transient!)
            f_jitter = 1.0 + 0.03 * math.sin(k * 1.7)
            tick = (
                0.70 * np.sin(2.0 * np.pi * 2820.0 * f_jitter * loc)
                + 0.55 * np.sin(2.0 * np.pi * 4920.0 * f_jitter * loc)
                + 0.32 * np.sin(2.0 * np.pi * 7650.0 * f_jitter * loc)
            ) * np.exp(-310.0 * loc)
            imp = rng.standard_normal(len(loc)) * np.exp(-620.0 * loc) * 0.45
            amp = (0.85 + 0.15 * math.sin(k * 0.9)) * (1.0 - 0.32 * (k / float(n_clicks)))
            pan_k = 0.12 * math.sin(k * 0.65)
            out_l[m] += (tick + imp) * amp * (1.0 - pan_k)
            out_r[m] += (tick + imp) * amp * (1.0 + pan_k)
            # Slight authentic acceleration as rotor breaks free cleanly!
            ts += 0.034 - 0.006 * (k / float(n_clicks))
        return (np.stack([out_l, out_r], axis=-1) * 0.84).astype(np.float32)

    if cue_type == "upward_airflow_rush_whoosh":
        # Sweeping vertical airflow column rushing UP through the negatively pitched blades!
        t = np.linspace(0.0, 2.55, int(2.55 * SR), dtype=np.float64)
        n = rng.standard_normal(len(t))
        air_lo = _bp(n, 150.0, 720.0) * (0.45 + 0.55 * np.sin(np.pi * t / 2.55))
        air_hi = _bp(rng.standard_normal(len(t)), 750.0, 2600.0) * (np.sin(np.pi * t / 2.55) ** 1.3) * 0.48
        whist = np.sin(2.0 * np.pi * np.cumsum(340.0 + 420.0 * (t / 2.55)) / SR) * 0.15 * np.sin(np.pi * t / 2.55)
        return _short_room(((air_lo * 0.52 + air_hi * 0.35 + whist) * 0.78).astype(np.float32), wet=0.28)

    if cue_type == "rotor_blade_slap_windmill":
        # Authentic helicopter Blade-Vortex Interaction (`WHOP-WHOP-WHOP`) accelerating in autorotation!
        t = np.linspace(0.0, 3.30, int(3.30 * SR), dtype=np.float64)
        # 4-blade rescue chopper BVI slap cadence (8.2 Hz -> 11.4 Hz)
        blade_hz = 8.2 + 3.2 * (t / 3.30)
        ph = 2.0 * np.pi * np.cumsum(blade_hz) / SR
        slap = (0.5 + 0.5 * np.sin(ph)) ** 5.2
        sub_whop = np.sin(2.0 * np.pi * 54.0 * t) * slap * 0.72
        mid_chop = _bp(rng.standard_normal(len(t)), 90.0, 580.0) * (0.22 + 0.78 * slap) * 1.05
        tip_whip = _bp(rng.standard_normal(len(t)), 900.0, 3200.0) * (slap ** 1.6) * 0.42
        env = np.sin(np.pi * t / 3.30) ** 0.30
        # Subtle stereo sweeping pan across blade rotation
        pan_l = 0.85 + 0.15 * np.sin(0.5 * ph)
        pan_r = 0.85 - 0.15 * np.sin(0.5 * ph)
        mono = _soft_clip((sub_whop + mid_chop + tip_whip) * env * 0.74, drive=1.4)
        return np.stack([mono * pan_l, mono * pan_r], axis=-1).astype(np.float32)

    if cue_type == "kinetic_flywheel_harmonic_rise":
        # Stored flywheel gyro energy spin-up harmonic rise
        t = np.linspace(0.0, 1.55, int(1.55 * SR), dtype=np.float64)
        freq = 165.0 + 480.0 * (t / 1.55) ** 1.35
        ph = 2.0 * np.pi * np.cumsum(freq) / SR
        env = np.sin(np.pi * t / 1.55) ** 0.65
        sig = (0.52 * np.sin(ph) + 0.28 * np.sin(1.5 * ph) + 0.18 * np.sin(2.0 * ph)) * env * 0.56
        return _short_room(sig.astype(np.float32), wet=0.25)

    if cue_type == "maple_seed_flutter_whirl":
        # Organic papery maple samara wing flutter + silky spinning air vortex
        t = np.linspace(0.0, 2.30, int(2.30 * SR), dtype=np.float64)
        flut = (0.5 + 0.5 * np.sin(2.0 * np.pi * 13.5 * t)) ** 2.6
        air = _bp(rng.standard_normal(len(t)), 580.0, 3400.0)
        soft_whoosh = _bp(rng.standard_normal(len(t)), 240.0, 900.0) * 0.45
        env = np.sin(np.pi * t / 2.30) ** 0.45
        return ((air * (0.25 + 0.75 * flut) + soft_whoosh) * env * 0.58).astype(np.float32)

    if cue_type == "autorotation_vortex_pulse":
        # Matched stereo aerodynamic shimmer tying maple seed (L) & rescue rotor (R)
        t = np.linspace(0.0, 1.90, int(1.90 * SR), dtype=np.float64)
        mod = 0.35 + 0.65 * (np.sin(2.0 * np.pi * 7.2 * t) ** 2)
        air = _bp(rng.standard_normal(len(t)), 180.0, 1150.0) * mod
        tone = (np.sin(2.0 * np.pi * 293.66 * t) + 0.4 * np.sin(2.0 * np.pi * 440.0 * t)) * 0.13
        env = np.sin(np.pi * t / 1.90) ** 0.48
        return _short_room(((air * 0.55 + tone) * env * 0.66).astype(np.float32), wet=0.30)

    if cue_type == "collective_pull_ratchet_click":
        # Heavy collective pull-up haul + servo hydraulic groan + mechanical latch click
        t = np.linspace(0.0, 0.52, int(0.52 * SR), dtype=np.float64)
        click = _bp(rng.standard_normal(len(t)), 1200.0, 4800.0) * np.exp(-42.0 * t)
        servo_f = 135.0 + 260.0 * (t / 0.52)
        servo = np.sin(2.0 * np.pi * np.cumsum(servo_f) / SR) * np.exp(-5.8 * t) * 0.58
        latch = _bp(rng.standard_normal(len(t)), 2200.0, 5600.0) * np.exp(-30.0 * np.maximum(0.0, t - 0.14)) * (t >= 0.14)
        return _short_room(_soft_clip((0.75 * click + 0.72 * servo + 0.55 * latch) * 0.78), wet=0.22)

    if cue_type == "flare_blade_bite_air_thump":
        # MASSIVE post-vacuum Beat 5B 30-foot flare: heavy high-pitch rotor chop biting dense air!
        t = np.linspace(0.0, 1.45, int(1.45 * SR), dtype=np.float64)
        chop = (0.5 + 0.5 * np.sin(2.0 * np.pi * 9.2 * t)) ** 4.8
        air = _bp(rng.standard_normal(len(t)), 60.0, 760.0) * 1.25
        whip = _bp(rng.standard_normal(len(t)), 900.0, 3400.0) * (chop ** 1.5) * 0.55
        sub_drop = np.sin(2.0 * np.pi * np.cumsum(84.0 * np.exp(-2.6 * t) + 34.0) / SR) * 0.92
        env = np.minimum(1.0, t / 0.025) * np.exp(-0.95 * t)
        sig = _soft_clip(((air + sub_drop) * (0.35 + 0.65 * chop) + whip) * env * 0.82, drive=1.6)
        return _short_room(sig, wet=0.24)

    if cue_type == "downwash_cushion_sub_boom":
        # Ground-effect rooftop dust blast + deep sub air-cushion shockwave!
        t = np.linspace(0.0, 1.35, int(1.35 * SR), dtype=np.float64)
        sub = np.sin(2.0 * np.pi * np.cumsum(78.0 * np.exp(-3.0 * t) + 32.0) / SR) * np.exp(-2.0 * t)
        wash = _lp(rng.standard_normal(len(t)), 480.0) * np.exp(-1.5 * t)
        dust_grit = _bp(rng.standard_normal(len(t)), 1400.0, 4800.0) * np.minimum(1.0, t / 0.08) * np.exp(-2.6 * t) * 0.34
        return _short_room(_soft_clip((0.92 * sub + 0.60 * wash + dust_grit) * 0.80), wet=0.25)

    if cue_type == "helipad_skid_touchdown_thud":
        # Authentic steel-skid-on-concrete cushioned touchdown + suspension settle!
        t = np.linspace(0.0, 0.72, int(0.72 * SR), dtype=np.float64)
        thud = np.sin(2.0 * np.pi * np.cumsum(112.0 * np.exp(-15.0 * t) + 34.0) / SR) * np.exp(-8.8 * t)
        concrete = _bp(rng.standard_normal(len(t)), 320.0, 1950.0) * np.exp(-22.0 * t)
        strut = np.sin(2.0 * np.pi * 210.0 * t) * np.exp(-18.0 * t) * 0.35
        return _short_room(_soft_clip((0.95 * thud + 0.48 * concrete + strut) * 0.84), wet=0.20)

    # coasting_blade_freewheel_fade
    t = np.linspace(0.0, 1.95, int(1.95 * SR), dtype=np.float64)
    hz = 6.4 * np.exp(-0.72 * t) + 1.5
    ph = 2.0 * np.pi * np.cumsum(hz) / SR
    chop = (0.5 + 0.5 * np.sin(ph)) ** 3.2
    air = _bp(rng.standard_normal(len(t)), 85.0, 480.0)
    env = np.exp(-1.05 * t)
    return _short_room((air * (0.20 + 0.80 * chop) * env * 0.68).astype(np.float32), wet=0.22)


def main() -> None:
    tl = json.loads(TIMELINE_PATH.read_text(encoding="utf-8"))
    rng = np.random.default_rng(5052)
    bus = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)

    # Continuous scene-matched helicopter & golden-hour wind bed (with Beat 1B/5A enclosed cabin acoustics
    # AND a crisp 0.35s pre-flare ambient vacuum drop right before t=24.37s!)
    t_all = np.linspace(0.0, DURATION_S, TOTAL_SAMPLES, dtype=np.float64)
    ext_air = _bp(rng.standard_normal(TOTAL_SAMPLES), 75.0, 380.0) * 0.055
    cab_hum = _bp(rng.standard_normal(TOTAL_SAMPLES), 95.0, 260.0) * 0.042

    in_cockpit = ((t_all >= 1.92) & (t_all < 4.00)) | ((t_all >= 21.50) & (t_all < 24.17))
    # 0.35s dramatic pre-flare silence vacuum (24.02..24.37s)!
    pre_flare_vac = np.where((t_all >= 24.02) & (t_all <= 24.36), 0.14, 1.0)
    blade_bed_hz = np.where(t_all < 28.08, 9.6, 3.6)
    blade_mod = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(2.0 * np.pi * np.cumsum(blade_bed_hz) / SR)) ** 2.4

    amb = np.where(in_cockpit, cab_hum, ext_air * blade_mod) * pre_flare_vac
    bus[:, 0] += amb.astype(np.float32)
    bus[:, 1] += (amb * 0.95).astype(np.float32)

    # Visceral cockpit heartbeat sub-thump inside Beat 1B (panic) & Beat 5A (final 30-ft countdown!)
    _synth_heartbeat(2.05, 3.85, bpm=108.0, bus=bus, gain=0.30)
    _synth_heartbeat(21.65, 23.95, bpm=118.0, bus=bus, gain=0.34)

    for cue in tl.get("sfx_cues", []):
        ev_audio = _synth_cue(str(cue["type"]), rng)
        _add(bus, ev_audio, float(cue["t"]), pan=float(cue.get("pan", 0.0)), gain=0.95)

    peak = float(np.max(np.abs(bus)))
    if peak > 0.90:
        bus *= 0.90 / peak
    micro = int(round(0.008 * SR))
    bus[:micro] *= np.linspace(0.0, 1.0, micro, dtype=np.float32)[:, None]
    bus[-micro:] *= np.linspace(1.0, 0.0, micro, dtype=np.float32)[:, None]

    SFX_STEM_PATH.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(SFX_STEM_PATH), bus, SR, subtype="PCM_16")
    print(
        f"[SFX] Generated V2 blockbuster {SFX_STEM_PATH.name} "
        f"({len(tl.get('sfx_cues', []))} layered aviation cues + cockpit heartbeat, peak={float(np.max(np.abs(bus))):.3f})"
    )


if __name__ == "__main__":
    main()
