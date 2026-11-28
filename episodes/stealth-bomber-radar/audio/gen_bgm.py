"""
Cinematic Stealth-Thriller BGM Composer for Episode #4 (Hans Zimmer / Tenet Inspired Edition)
("How a 172-Foot Stealth Bomber Looks Like a Bumblebee on Radar", 30.0 s @ 60 fps).

Generates:
    episodes/stealth-bomber-radar/audio/output/bgm_stem.wav
        - 48000 Hz, Stereo, PCM_16, exact 30.000 s (1,440,000 samples)
        - Dark tactical C#-minor aerospace pulse at 122 BPM
        - Deep analog sub-bass ostinato (< 200 Hz) + tight filtered stealth arpeggio (C#m9)
          + crisp tactical sonar hi-ticks (> 3.8 kHz)
        - Permanent vocal-pocket EQ carve (250 - 3500 Hz) so calm narration stays crystal clear
        - Dramatic Beat 1->2 & Beat 5->6 low-end tension swells + 55 ms breath drop before Beat 6
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

import stealth_contract as C  # noqa: E402

OUTPUT_DIR = AUDIO_DIR / "output"
BGM_STEM_PATH = OUTPUT_DIR / "bgm_stem.wav"

SR = 48000
CHANNELS = 2
DURATION_S = float(C.DURATION_S)
TOTAL_SAMPLES = int(round(DURATION_S * SR))
BPM = 122.0
BEAT_S = 60.0 / BPM
STEP_S = BEAT_S / 4.0  # 16th-note pulse

RNG = np.random.default_rng(774122)


def peaking_eq(audio: np.ndarray, freq: float, q: float, gain_db: float) -> np.ndarray:
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / SR
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    norm = 1.0 + alpha / a_lin
    b = np.array([1.0 + alpha * a_lin, -2.0 * cw, 1.0 - alpha * a_lin], dtype=np.float64) / norm
    a = np.array([1.0, (-2.0 * cw) / norm, (1.0 - alpha / a_lin) / norm], dtype=np.float64)
    return signal.lfilter(b, a, audio.astype(np.float64), axis=0).astype(np.float32)


def butter_filter(audio: np.ndarray, btype: str, cutoff: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * SR
    sos = signal.butter(order, float(np.clip(cutoff / nyq, 1e-4, 0.999)), btype=btype, output="sos")
    return signal.sosfilt(sos, audio, axis=0).astype(np.float32)


def build_bgm_stem() -> np.ndarray:
    t = np.arange(TOTAL_SAMPLES, dtype=np.float64) / SR
    bus = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)

    # C#-minor root progression across the 6 visual beats (Hz)
    root_f = np.full(TOTAL_SAMPLES, 34.65, dtype=np.float64)
    root_f[(t >= 3.67) & (t < 6.40)] = 41.20  # E1
    root_f[(t >= 6.40) & (t < 9.00)] = 51.91  # G#1
    root_f[(t >= 9.00) & (t < 16.00)] = 34.65  # C#1
    root_f[(t >= 16.00) & (t < 22.00)] = 46.25  # F#1
    root_f[(t >= 22.00) & (t < 24.00)] = 41.20  # E1
    root_f[(t >= 24.00) & (t < 26.00)] = 51.91  # G#1
    root_f[t >= 26.00] = 34.65  # C#1 resolution

    b_sm, a_sm = signal.butter(2, 6.0 / (0.5 * SR), btype="low")
    root_f = signal.filtfilt(b_sm, a_sm, root_f)

    # 1) Deep stealth analog sub-pulse (8th-note pumping)
    phase_sub = 2.0 * math.pi * np.cumsum(root_f) / SR
    phase_oct = 2.0 * math.pi * np.cumsum(root_f * 2.0) / SR
    pulse_8th = 0.44 + 0.56 * (np.sin(math.pi * (t / (BEAT_S * 0.5))) ** 2)
    bass_mono = (0.62 * np.sin(phase_sub) + 0.25 * np.sin(phase_oct) + 0.11 * np.sin(phase_oct * 1.5)) * pulse_8th
    bass_mono = butter_filter(bass_mono.astype(np.float32), "low", 215.0, order=3)
    bus[:, 0] += bass_mono * 0.62
    bus[:, 1] += bass_mono * 0.62

    # 2) Tight Tenet-style stealth analog 16th-note low-passed arpeggio (Beats 2-4 & 6)
    arp_notes = [138.59, 207.65, 164.81, 155.56]  # C#3, G#3, E3, D#3
    arp_len = int(round(STEP_S * 0.85 * SR))
    at = np.arange(arp_len, dtype=np.float64) / SR
    arp_env = (np.clip(at / 0.008, 0.0, 1.0) * np.exp(-at / 0.065)).astype(np.float32)
    n_steps = int(DURATION_S / STEP_S)
    for s in range(n_steps):
        ts = s * STEP_S
        if ts < 3.65 or (22.0 <= ts < 26.0) or ts >= 29.2:
            continue
        nf = arp_notes[s % len(arp_notes)]
        # Warm muted saw-sine hybrid filtered cleanly under 240 Hz + airy harmonic shimmer
        note = (0.65 * np.sin(2.0 * math.pi * nf * at) + 0.28 * np.sin(4.0 * math.pi * nf * at)) * arp_env
        note = butter_filter(note.astype(np.float32), "low", 240.0, order=2)
        idx = int(round(ts * SR))
        cl = min(arp_len, TOTAL_SAMPLES - idx)
        if cl > 0:
            pan = -0.28 if (s % 2 == 0) else 0.28
            ang = (pan + 1.0) * (math.pi / 4.0)
            g = 0.22 if ts < 26.0 else 0.16
            bus[idx : idx + cl, 0] += note[:cl] * math.cos(ang) * g
            bus[idx : idx + cl, 1] += note[:cl] * math.sin(ang) * g

    # 3) Wide cold tactical atmospheric pad (C#m9 airy shimmer)
    pad_l = np.zeros(TOTAL_SAMPLES, dtype=np.float64)
    pad_r = np.zeros(TOTAL_SAMPLES, dtype=np.float64)
    for freq_mul, detune, amp in [(4.0, -0.35, 0.14), (6.0, +0.40, 0.11), (12.0, -0.55, 0.06), (15.0, +0.60, 0.05)]:
        fl = root_f * freq_mul + detune
        fr = root_f * freq_mul - detune
        pad_l += amp * np.sin(2.0 * math.pi * np.cumsum(fl) / SR)
        pad_r += amp * np.sin(2.0 * math.pi * np.cumsum(fr) / SR)
    pad = np.stack([pad_l, pad_r], axis=-1).astype(np.float32)
    pad *= (0.72 + 0.28 * np.sin(2.0 * math.pi * 0.22 * t)).astype(np.float32)[:, None]
    bus += pad * 0.44

    # 4) Crisp tactical sonar 16th-note clock ticks (> 4.2 kHz, ultra-clean stereo width)
    tick_len = int(0.032 * SR)
    tt = np.arange(tick_len, dtype=np.float64) / SR
    for s in range(n_steps):
        ts = s * STEP_S
        if (22.0 <= ts < 22.5) or (ts >= 28.8):
            continue
        accent = 1.0 if (s % 4 == 0) else (0.55 if (s % 2 == 0) else 0.28)
        if ts >= 26.0:
            accent *= 0.45
        tick = butter_filter(RNG.standard_normal(tick_len).astype(np.float32), "high", 4200.0, order=3)
        tick *= np.exp(-tt / 0.0085).astype(np.float32) * accent * 0.19
        idx = int(round(ts * SR))
        cl = min(tick_len, TOTAL_SAMPLES - idx)
        if cl > 0:
            pan = -0.34 if (s % 2 == 0) else 0.34
            ang = (pan + 1.0) * (math.pi / 4.0)
            bus[idx : idx + cl, 0] += tick[:cl] * math.cos(ang)
            bus[idx : idx + cl, 1] += tick[:cl] * math.sin(ang)

    # 5) Soft muffled tactical heartbeat kick on beat 1 & 3 (Beats 1-5)
    kick_len = int(0.18 * SR)
    kt = np.arange(kick_len, dtype=np.float64) / SR
    kf = 94.0 * ((36.0 / 94.0) ** np.clip(kt / 0.08, 0.0, 1.0))
    kick_wave = (np.sin(2.0 * math.pi * np.cumsum(kf) / SR) * np.exp(-kt / 0.065)).astype(np.float32) * 0.38
    b_idx = 0
    while b_idx * BEAT_S < 25.85:
        ts = b_idx * BEAT_S
        if b_idx % 2 == 0:
            gain_k = 0.65 if ts >= 22.0 else 1.0
            i0 = int(round(ts * SR))
            cl = min(kick_len, TOTAL_SAMPLES - i0)
            if cl > 0:
                bus[i0 : i0 + cl, 0] += kick_wave[:cl] * gain_k
                bus[i0 : i0 + cl, 1] += kick_wave[:cl] * gain_k
        b_idx += 1

    # 6) Subtle low-end tension swell rising into Beat 6 moonlight payoff (24.80..25.92s)
    sw0, sw1 = int(round(24.80 * SR)), int(round(25.92 * SR))
    st = np.linspace(0.0, 1.0, sw1 - sw0, dtype=np.float64)
    swell = (np.sin(2.0 * math.pi * np.cumsum(34.65 + 34.65 * st) / SR) * (st ** 2.2) * 0.28).astype(np.float32)
    bus[sw0:sw1, 0] += swell
    bus[sw0:sw1, 1] += swell

    # 7) Permanent vocal-pocket EQ carve (250 - 3500 Hz) + clean 28 Hz sub HPF
    bus = butter_filter(bus, "high", 28.0, order=2)
    bus = peaking_eq(bus, 520.0, 0.85, -6.5)
    bus = peaking_eq(bus, 1450.0, 0.75, -8.0)
    bus = peaking_eq(bus, 2850.0, 0.90, -5.5)

    # 8) Dramatic 55 ms tension breath dip right before Beat 6 moonlight payoff (25.925..25.995s)
    g0 = int(round(25.925 * SR))
    g1 = int(round(25.995 * SR))
    bus[g0:g1] *= 0.14

    fin = int(round(0.18 * SR))
    fout = int(round(0.55 * SR))
    bus[:fin] *= np.linspace(0.0, 1.0, fin, dtype=np.float32)[:, None]
    bus[-fout:] *= np.linspace(1.0, 0.0, fout, dtype=np.float32)[:, None]

    rms = float(np.sqrt(np.mean(bus.astype(np.float64) ** 2) + 1e-12))
    bus *= 0.16 / max(1e-5, rms)
    pk = float(np.max(np.abs(bus)))
    if pk > 0.78:
        bus *= 0.78 / pk
    return bus.astype(np.float32)


def verify_bgm(path: Path) -> None:
    info = sf.info(str(path))
    assert info.samplerate == SR and info.channels == CHANNELS and info.frames == TOTAL_SAMPLES
    assert info.subtype == "PCM_16" and abs(info.duration - DURATION_S) < 1e-6
    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == SR and wf.getnchannels() == CHANNELS and wf.getsampwidth() == 2
    y, _ = sf.read(str(path), dtype="float32")
    assert np.isfinite(y).all()
    pk = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    assert 0.05 < rms < 0.35 and pk < 0.85
    print(f"[BGM] Verified {path.name}: {SR}Hz 2ch PCM_16, {TOTAL_SAMPLES} samples, rms={rms:.4f}, peak={pk:.4f}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = build_bgm_stem()
    sf.write(str(BGM_STEM_PATH), stem, SR, subtype="PCM_16")
    verify_bgm(BGM_STEM_PATH)


if __name__ == "__main__":
    main()
