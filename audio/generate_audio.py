"""
Isaac Newton (1666–1687) — The Falling Apple, Rainbow Prism & Universal Gravity
V9 Theatrical IMAX / @zackdfilms Broadcast-Mastered Studio Audio:
  - 112 BPM Cambridge Celestial Baroque & Cosmic Orchestral Score (D Minor -> D Major/Lydian -> Soaring Climax)
  - 3 Distinct British Kokoro ONNX Voices (Narrator bm_george, Newton bm_daniel, Halley bm_lewis)
  - Broadcast Vocal Chain: 85 Hz HPF + Anti-Mud Dip + +2.2 dB Clarity/Air Shelf (3.5-5.5 kHz) + Smooth Optical Compressor
  - Airy Outdoor Stone-Courtyard Early Reflections + Clean Spatial Stereo Panning (0.0, -0.22, +0.24)
  - Living Outdoor Ambience Bed (-34 dB): Golden-Hour English Meadow Breeze & Distant Birdsong -> Night Crickets & Cosmic Hum
  - 26 Phone-Speaker-Tested Spatial Foley SFX (Sub+Woody Apple Thud, 7-Note Glass Harmonica/Harp Prism, High-Q Steel Cradle Clacks, Gravity Well, Rocket)
  - -5.5 dB Smooth Sidechain Ducking + ITU-R BS.1770-4 (-14.0 LUFS) + 4x True-Peak Brickwall Limiter (<= -1.5 dBFS)
  - Exact 48,000 Hz Stereo (1,440,000 samples = 30.000s)
  - Exact 30 Hz (900-sample) Normalized Dual-Axis Lip-Sync Manifest (lipSync + lipNarrow)
  - Clause-Aware Balanced Boxless Karaoke ASS + SRT Subtitles

Run with:
    /home/antigravity/shorts-engine/.venv/bin/python audio/generate_audio.py
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import kokoro_onnx
import numpy as np
import soundfile as sf
from scipy import signal

# ============================================================================
# GLOBAL BROADCAST & PIPELINE CONSTANTS (48 kHz / 30.000s / 30 Hz Lip-Sync)
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TIMELINE_PATH = PROJECT_ROOT / "config" / "timeline.json"

KOKORO_MODEL_PATH = Path("/home/antigravity/.cache/kokoro-onnx/kokoro-v1.0.int8.onnx")
KOKORO_VOICES_PATH = Path("/home/antigravity/.cache/kokoro-onnx/voices-v1.0.bin")

SR = 48000
CHANNELS = 2
DURATION = 30.0
TOTAL_SAMPLES = int(round(DURATION * SR))  # 1,440,000 exact samples
LIP_FPS = 30
LIP_TOTAL_FRAMES = int(round(DURATION * LIP_FPS))  # 900 exact samples @ 30 Hz

TARGET_LUFS = -14.0
TRUE_PEAK_CEILING_DBFS = -1.62  # Strictly keeps FFmpeg ebur128 true peak <= -1.55 dBFS & max_volume <= -1.5 dB
TRUE_PEAK_CEILING_LIN = 10.0 ** (TRUE_PEAK_CEILING_DBFS / 20.0)
AMBIENCE_TARGET_DBFS = -34.0  # Subtle living outdoor bed (-34 dB RMS in mastered mix)

ATEMPO_MIN = 0.88
ATEMPO_MAX = 1.24

BPM = 112.0
BEAT = 60.0 / BPM
EIGHTH = BEAT * 0.5
SIXTEENTH = BEAT * 0.25

# Natural spoken phrasing overrides (display text stays verbatim from timeline.json)
TTS_OVERRIDES: dict[str, str] = {
    "L1": "One falling apple sparked the wildest question: does the Moon fall too?",
    "L2": "Impossible, Newton! What holds the Moon up?",
    "L3": "Invisible gravity! Every mass pulls every other mass!",
    "L4": "Next, Newton shot sunlight through a glass prism, splitting white light into seven rainbow colors!",
    "L5": "Every action triggers an equal, opposite reaction.",
    "L6": "Whoa! That steel ball nearly knocked me over!",
    "L7": "Gravity bends a fall into orbit, uniting falling apples and moons under one law!",
    "L8": "Rockets, satellites, Moon landings today, every launch flies on Isaac Newton's laws.",
}

VOICE_FALLBACKS: dict[str, list[str]] = {
    "bm_george": ["bm_fable", "am_michael"],
    "bm_daniel": ["bm_fable", "am_adam", "bm_george"],
    "bm_lewis": ["bm_fable", "am_michael", "bm_george"],
}


# ============================================================================
# CORE DSP UTILITIES (48 kHz)
# ============================================================================
def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def pan_stereo(mono: np.ndarray, pan: float) -> np.ndarray:
    pan = float(np.clip(pan, -1.0, 1.0))
    angle = (pan + 1.0) * (math.pi / 4.0)
    return np.stack([mono * math.cos(angle), mono * math.sin(angle)], axis=-1).astype(np.float32)


def pan_sweep_stereo(mono: np.ndarray, pan_start: float, pan_end: float) -> np.ndarray:
    n = len(mono)
    p = np.linspace(float(np.clip(pan_start, -1.0, 1.0)), float(np.clip(pan_end, -1.0, 1.0)), n, dtype=np.float32)
    ang = (p + 1.0) * (math.pi / 4.0)
    return np.stack([mono * np.cos(ang), mono * np.sin(ang)], axis=-1).astype(np.float32)


def add_stereo_event(bus: np.ndarray, event: np.ndarray, start_time: float) -> None:
    if event.ndim == 1:
        event = pan_stereo(event, 0.0)
    idx = int(round(start_time * SR))
    if idx >= len(bus) or idx + len(event) <= 0:
        return
    src_start = max(0, -idx)
    dst_start = max(0, idx)
    copy_len = min(len(event) - src_start, len(bus) - dst_start)
    if copy_len > 0:
        bus[dst_start : dst_start + copy_len] += event[src_start : src_start + copy_len]


def adsr_env(n: int, attack: float = 0.01, decay: float = 0.08, sustain: float = 0.6, release: float = 0.06) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    a = min(n, max(1, int(attack * SR)))
    r = min(n - a, max(1, int(release * SR)))
    d = min(n - a - r, max(0, int(decay * SR)))
    s = max(0, n - a - d - r)
    parts = [np.linspace(0.0, 1.0, a, endpoint=False, dtype=np.float32)]
    if d > 0:
        parts.append(np.linspace(1.0, sustain, d, endpoint=False, dtype=np.float32))
    if s > 0:
        parts.append(np.full(s, sustain, dtype=np.float32))
    if r > 0:
        sv = sustain if (d > 0 or s > 0) else 1.0
        parts.append(np.linspace(sv, 0.0, r, endpoint=True, dtype=np.float32))
    env = np.concatenate(parts)
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    return env[:n]


def butter_filter(audio: np.ndarray, btype: str, cutoff: float | tuple[float, float], order: int = 2) -> np.ndarray:
    nyq = 0.5 * SR
    if isinstance(cutoff, tuple):
        wn = [float(np.clip(c / nyq, 1e-4, 0.999)) for c in cutoff]
    else:
        wn = float(np.clip(cutoff / nyq, 1e-4, 0.999))
    sos = signal.butter(order, wn, btype=btype, output="sos")
    return signal.sosfilt(sos, audio, axis=0).astype(np.float32)


def peaking_eq(mono: np.ndarray, freq: float, q: float, gain_db: float) -> np.ndarray:
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / SR
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)
    b0 = 1.0 + alpha * a_lin
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a_lin
    a0 = 1.0 + alpha / a_lin
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a_lin
    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return signal.lfilter(b, a, mono).astype(np.float32)


def high_shelf_eq(mono: np.ndarray, freq: float, gain_db: float, slope: float = 0.72) -> np.ndarray:
    """RBJ Audio EQ Cookbook smooth high-shelf filter for broadcast vocal air & clarity."""
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / SR
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    s_term = max(1e-6, (a_lin + 1.0 / a_lin) * (1.0 / slope - 1.0) + 2.0)
    alpha = 0.5 * sin_w0 * math.sqrt(s_term)
    two_sqrt_a_alpha = 2.0 * math.sqrt(a_lin) * alpha

    b0 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w0 + two_sqrt_a_alpha)
    b1 = -2.0 * a_lin * ((a_lin - 1.0) + (a_lin + 1.0) * cos_w0)
    b2 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w0 - two_sqrt_a_alpha)
    a0 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w0 + two_sqrt_a_alpha
    a1 = 2.0 * ((a_lin - 1.0) - (a_lin + 1.0) * cos_w0)
    a2 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w0 - two_sqrt_a_alpha

    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return signal.lfilter(b, a, mono).astype(np.float32)


def optical_vocal_compressor(
    mono: np.ndarray,
    thresh_db: float = -19.0,
    ratio: float = 3.2,
    attack_ms: float = 8.5,
    release_ms: float = 95.0,
    knee_db: float = 6.0,
) -> np.ndarray:
    """LA-2A-inspired smooth optical vocal compressor (1 kHz control envelope, zero pumping)."""
    n = len(mono)
    if n < 128:
        return mono
    hop = 48  # 1 ms control frames @ 48 kHz
    win = 192  # 4 ms RMS sensing window
    sq = mono.astype(np.float64) ** 2
    csum = np.concatenate(([0.0], np.cumsum(sq)))
    centers = np.arange(0, n, hop)
    lo = np.maximum(0, centers - win // 2)
    hi = np.minimum(n, centers + win // 2)
    rms = np.sqrt(np.maximum(1e-12, (csum[hi] - csum[lo]) / np.maximum(1, hi - lo)))
    lvl_db = 20.0 * np.log10(rms)

    # Smooth optical attack & gentle program-dependent release
    att = math.exp(-1.0 / max(0.5, attack_ms))
    rel = math.exp(-1.0 / max(5.0, release_ms))
    env_db = np.empty_like(lvl_db)
    e = float(lvl_db[0])
    for i, x in enumerate(lvl_db):
        xv = float(x)
        if xv > e:
            e = att * e + (1.0 - att) * xv
        else:
            e = rel * e + (1.0 - rel) * xv
        env_db[i] = e

    # Soft-knee gain reduction curve
    half_k = 0.5 * knee_db
    over = env_db - thresh_db
    gr_db = np.zeros_like(env_db)
    slope = 1.0 - 1.0 / ratio
    in_knee = (over > -half_k) & (over < half_k)
    above = over >= half_k
    gr_db[in_knee] = -slope * ((over[in_knee] + half_k) ** 2) / (2.0 * knee_db)
    gr_db[above] = -slope * over[above]

    gain_ctrl = (10.0 ** (gr_db / 20.0)).astype(np.float32)
    gain_full = np.interp(np.arange(n, dtype=np.float32), centers.astype(np.float32), gain_ctrl).astype(np.float32)
    return (mono * gain_full).astype(np.float32)


def add_outdoor_courtyard_reverb(stereo: np.ndarray, mix: float = 0.115) -> np.ndarray:
    """Airy English stone-courtyard early reflections (crisp outdoor stone slap, ZERO boomy cave echo!)."""
    out = np.copy(stereo)
    # Short open-sky stone courtyard reflections: ground bounce -> low garden wall -> stone arch -> distant facade
    taps = [
        (0.0068, 0.22, 0.14, 0.05),
        (0.0155, 0.14, 0.21, 0.07),
        (0.0278, 0.12, 0.10, 0.06),
        (0.0435, 0.08, 0.10, 0.05),
        (0.0615, 0.05, 0.06, 0.04),
    ]
    wet = np.zeros_like(stereo)
    for dt, gl, gr, gx in taps:
        d = int(round(dt * SR))
        if d < len(stereo):
            wet[d:, 0] += stereo[:-d, 0] * gl + stereo[:-d, 1] * gx
            wet[d:, 1] += stereo[:-d, 1] * gr + stereo[:-d, 0] * gx
    # Open-air high-passed reflection band (260 Hz - 6500 Hz) avoids any indoor low-mid cave boom
    wet = butter_filter(wet, "band", (260.0, 6500.0), order=2)
    return (out * (1.0 - mix * 0.20) + wet * mix).astype(np.float32)


def add_cambridge_chamber_reverb(stereo: np.ndarray, mix: float = 0.22) -> np.ndarray:
    """Lush concert-hall / Cambridge chapel orchestral reverb for cinema score width."""
    out = np.copy(stereo)
    taps = [
        (0.0142, 0.26, 0.17, 0.09),
        (0.0285, 0.18, 0.26, 0.11),
        (0.0468, 0.16, 0.13, 0.10),
        (0.0695, 0.12, 0.15, 0.09),
        (0.0980, 0.09, 0.10, 0.07),
        (0.1340, 0.06, 0.07, 0.05),
    ]
    wet = np.zeros_like(stereo)
    for dt, gl, gr, gx in taps:
        d = int(round(dt * SR))
        if d < len(stereo):
            wet[d:, 0] += stereo[:-d, 0] * gl + stereo[:-d, 1] * gx
            wet[d:, 1] += stereo[:-d, 1] * gr + stereo[:-d, 0] * gx
    wet = butter_filter(wet, "band", (150.0, 4400.0), order=2)
    return (out * (1.0 - mix * 0.26) + wet * mix).astype(np.float32)


def wsola_time_stretch(mono: np.ndarray, rate: float) -> np.ndarray:
    """Fast pitch-preserving in-memory WSOLA overlap-add time stretch (rate > 1 = faster)."""
    mono = np.asarray(mono, dtype=np.float32)
    if abs(rate - 1.0) < 0.012 or len(mono) < 1024:
        return mono

    win_len = 512
    hop_syn = 256
    search_rad = 160
    hann = np.hanning(win_len).astype(np.float32)

    out_len = max(win_len, int(round(len(mono) / rate)))
    out = np.zeros(out_len + win_len, dtype=np.float32)
    norm = np.zeros(out_len + win_len, dtype=np.float32)

    out[:win_len] += mono[:win_len] * hann
    norm[:win_len] += hann

    syn_pos = hop_syn
    while syn_pos < out_len - win_len:
        target_ana = int(round(syn_pos * rate))
        if target_ana + win_len >= len(mono):
            break

        ref = out[syn_pos : syn_pos + hop_syn]
        lo = max(0, target_ana - search_rad)
        hi = min(len(mono) - win_len, target_ana + search_rad)

        best_offset = target_ana
        best_score = -1e18
        if hi > lo:
            for cand in range(lo, hi + 1, 4):
                seg = mono[cand : cand + hop_syn]
                score = float(np.dot(ref, seg))
                if score > best_score:
                    best_score = score
                    best_offset = cand

        grain = mono[best_offset : best_offset + win_len]
        out[syn_pos : syn_pos + win_len] += grain * hann
        norm[syn_pos : syn_pos + win_len] += hann
        syn_pos += hop_syn

    mask = norm > 1e-4
    out[mask] /= norm[mask]
    return out[:out_len].astype(np.float32)


def resample_to_48k(mono: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == SR:
        return mono.astype(np.float32)
    g = math.gcd(SR, sr_in)
    return signal.resample_poly(mono, SR // g, sr_in // g).astype(np.float32)


def trim_silence(wav: np.ndarray, sr: int, thresh: float = 0.008) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    idx = np.where(np.abs(wav) > thresh)[0]
    if len(idx) == 0:
        return wav
    pad = int(0.015 * sr)
    lo = max(0, int(idx[0]) - pad)
    hi = min(len(wav), int(idx[-1]) + pad)
    return wav[lo:hi].astype(np.float32)


# ============================================================================
# KOKORO MULTI-VOICE SYNTHESIS, COURTYARD ACOUSTICS & 30 Hz DUAL-AXIS LIP-SYNC
# ============================================================================
def resolve_voice(available: set[str], desired: str) -> str:
    if desired in available:
        return desired
    for fb in VOICE_FALLBACKS.get(desired, ["bm_george", "bm_fable", "am_adam"]):
        if fb in available:
            print(f"  [Voice Fallback] '{desired}' unavailable -> using '{fb}'")
            return fb
    return sorted(available)[0]


def polish_voice(mono: np.ndarray, speaker: str) -> np.ndarray:
    """V9 Broadcast Vocal Chain: 85Hz HPF + De-Mud + +2.2dB Air/Clarity Shelf (3.5-5.5kHz) + Optical Compressor."""
    # 1. Clean 85 Hz high-pass rumble removal (preserves warm British baritone chest resonance)
    y = butter_filter(mono, "high", 85.0, order=3)
    # 2. Gentle anti-mud dip so vocals never mask cello/strings
    y = peaking_eq(y, 310.0, 1.25, -1.3)

    if speaker == "narrator":
        # Authoritative documentary narrator (bm_george): rich chest + crisp articulation + +2.2 dB clarity shelf
        y = peaking_eq(y, 180.0, 1.0, 1.5)
        y = peaking_eq(y, 3100.0, 1.15, 1.6)
        y = high_shelf_eq(y, 4100.0, 2.2, slope=0.72)
    elif speaker == "newton":
        # Articulated dignified Cambridge scholar (bm_daniel): focused presence + +2.2 dB air/clarity shelf
        y = peaking_eq(y, 215.0, 1.0, 1.2)
        y = peaking_eq(y, 3400.0, 1.15, 1.8)
        y = high_shelf_eq(y, 4350.0, 2.2, slope=0.72)
    else:
        # Spirited Royal Society interlocutor Halley (bm_lewis): agile crispness + +2.2 dB air/clarity shelf
        y = peaking_eq(y, 175.0, 1.0, 1.0)
        y = peaking_eq(y, 3650.0, 1.15, 1.9)
        y = high_shelf_eq(y, 4600.0, 2.2, slope=0.72)

    # 3. Pre-calibrate into smooth optical vocal compressor so every syllable cuts through BGM on phone speakers
    rms_pre = float(np.sqrt(np.mean(y ** 2) + 1e-10))
    if rms_pre > 1e-4:
        y = y * (0.185 / rms_pre)
    y = optical_vocal_compressor(y, thresh_db=-19.0, ratio=3.2, attack_ms=8.5, release_ms=92.0, knee_db=6.0)

    # 4. Makeup gain & warm transparent peak polish
    rms_post = float(np.sqrt(np.mean(y ** 2) + 1e-10))
    if rms_post > 1e-4:
        y = y * (0.215 / rms_post)
    y = (np.tanh(y * 1.05) * 0.86).astype(np.float32)

    fade = min(len(y), int(0.008 * SR))
    if fade > 2:
        y[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        y[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return y


def tokenize_display_words(text: str) -> list[str]:
    tokens = []
    for raw in text.strip().split():
        if "—" in raw and not raw.endswith("—"):
            parts = raw.split("—")
            for idx, part in enumerate(parts):
                if idx < len(parts) - 1:
                    tokens.append(part + "—")
                elif part:
                    tokens.append(part)
        else:
            tokens.append(raw)
    return [t for t in tokens if t]


def align_words_to_audio(words: list[str], audio: np.ndarray, t0: float, t1: float) -> list[dict[str, Any]]:
    """Punctuation-weighted & vocal-dip-nudged monotonic word alignment."""
    if not words:
        return []
    weights_list = []
    for w in words:
        core = len(re.sub(r"[^A-Za-z0-9]", "", w))
        wt = max(1.8, float(core))
        # Spoken pause weight after punctuation
        if w.endswith(("!", "?", ".")):
            wt += 1.6
        elif w.endswith(("—", ",", ";", ":")):
            wt += 1.1
        weights_list.append(wt)
    weights = np.array(weights_list, dtype=np.float64)
    weights /= weights.sum()

    dur = max(0.05, t1 - t0)
    raw_bounds = [t0]
    acc = 0.0
    for wt in weights[:-1]:
        acc += float(wt)
        raw_bounds.append(t0 + dur * acc)
    raw_bounds.append(t1)

    # Gently nudge internal boundaries toward nearby vocal energy valleys (+/- 45ms) for tighter karaoke lock
    n_audio = len(audio)
    if n_audio > int(0.12 * SR) and len(words) > 1:
        frame_step = int(0.010 * SR)
        env = np.array(
            [
                float(np.sqrt(np.mean(audio[i : min(n_audio, i + frame_step)] ** 2) + 1e-12))
                for i in range(0, n_audio, frame_step)
            ],
            dtype=np.float32,
        )
        for i in range(1, len(raw_bounds) - 1):
            rel = raw_bounds[i] - t0
            center_f = int(round(rel / 0.010))
            lo_f = max(1, center_f - 4)
            hi_f = min(len(env) - 2, center_f + 4)
            if hi_f > lo_f:
                best_f = lo_f + int(np.argmin(env[lo_f : hi_f + 1]))
                cand = t0 + best_f * 0.010
                min_t = raw_bounds[i - 1] + 0.06
                max_t = raw_bounds[i + 1] - 0.06
                if min_t < max_t:
                    raw_bounds[i] = float(np.clip(cand, min_t, max_t))

    out = []
    for i, w in enumerate(words):
        out.append({
            "word": w,
            "start": round(raw_bounds[i], 4),
            "end": round(raw_bounds[i + 1], 4),
        })
    out[-1]["end"] = round(t1, 4)
    return out


def compute_lip_sync(
    speaker_mono_buses: dict[str, np.ndarray], fps: int = LIP_FPS, total_frames: int = LIP_TOTAL_FRAMES
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Compute crisp, normalized (0.0..1.0) 30Hz (900-sample) dual-axis phoneme lip-sync curves:
    - lipSync: jaw openness (RMS envelope with responsive attack/natural release)
    - lipNarrow: rounded vowel formant ratio (220-820 Hz vs 1100-3400 Hz) smoothly gated by speech
    """
    lip_map: dict[str, list[float]] = {}
    narrow_map: dict[str, list[float]] = {}
    hop = int(round(SR / fps))  # 1600 samples exact @ 48 kHz / 30 fps
    win = int(round(SR / 18))
    for spk, bus in speaker_mono_buses.items():
        has_sig = bool(np.any(bus))
        low_bus = butter_filter(bus, "band", (220.0, 820.0), order=2) if has_sig else bus
        hi_bus = butter_filter(bus, "band", (1100.0, 3400.0), order=2) if has_sig else bus
        vals = np.zeros(total_frames, dtype=np.float32)
        nvals = np.zeros(total_frames, dtype=np.float32)
        for f in range(total_frames):
            c = min(len(bus) - 1, f * hop + hop // 2)
            lo = max(0, c - win // 2)
            hi = min(len(bus), c + win // 2)
            seg = bus[lo:hi]
            rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12)) if len(seg) > 0 else 0.0
            if rms > 0.012:
                vals[f] = rms
                e_lo = float(np.sqrt(np.mean(low_bus[lo:hi] ** 2) + 1e-12))
                e_hi = float(np.sqrt(np.mean(hi_bus[lo:hi] ** 2) + 1e-12))
                ratio = float(np.clip((e_lo / (e_lo + e_hi + 1e-6) - 0.40) * 2.35, 0.0, 1.0))
                nvals[f] = ratio

        active = vals > 0.012
        p92 = float(np.percentile(vals[active], 92)) if np.any(active) else 0.1
        normed = np.clip(vals / max(0.04, p92), 0.0, 1.0)
        # Gate lipNarrow smoothly by speech activity so mouth puckering tracks real vowels cleanly
        n_gated = np.clip(nvals * np.minimum(1.0, normed * 1.45), 0.0, 1.0)

        sm = np.copy(normed)
        nsm = np.copy(n_gated)
        for i in range(1, total_frames):
            # Snappy jaw open (0.65), natural jaw close (0.52)
            a_lip = 0.65 if normed[i] > sm[i - 1] else 0.52
            sm[i] = (1.0 - a_lip) * sm[i - 1] + a_lip * normed[i]
            a_nar = 0.56 if n_gated[i] > nsm[i - 1] else 0.46
            nsm[i] = (1.0 - a_nar) * nsm[i - 1] + a_nar * n_gated[i]

        # Re-normalize peak cleanly into [0.0, 1.0] & zero imperceptible silent tail (<0.008)
        if np.max(sm) > 0.05:
            sm = np.clip(sm / max(1.0, float(np.max(sm))), 0.0, 1.0)
        if np.max(nsm) > 0.05:
            nsm = np.clip(nsm / max(0.85, float(np.max(nsm))), 0.0, 1.0)
        sm[sm < 0.008] = 0.0
        nsm[nsm < 0.008] = 0.0

        lip_map[spk] = [round(float(v), 4) for v in sm]
        narrow_map[spk] = [round(float(v), 4) for v in nsm]
    return lip_map, narrow_map


def synthesize_dialogue(
    timeline: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    kokoro = kokoro_onnx.Kokoro(str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH))
    available_voices = set(kokoro.get_voices())
    speakers_cfg = timeline["speakers"]

    narrator_bus = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)
    courtyard_voice_bus = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)
    speaker_mono_buses = {spk: np.zeros(TOTAL_SAMPLES, dtype=np.float32) for spk in speakers_cfg}
    manifest_lines: list[dict[str, Any]] = []

    for item in timeline["dialogue"]:
        lid = item["id"]
        spk = item["speaker"]
        cfg = speakers_cfg[spk]
        voice_id = resolve_voice(available_voices, cfg["voice"])
        lang = cfg.get("lang", "en-gb")
        base_speed = float(cfg.get("speed", 1.10))
        pan = float(cfg.get("pan", 0.0))

        slot_start = float(item["start"])
        slot_end = float(item["end"])
        slot_dur = slot_end - slot_start
        tts_text = TTS_OVERRIDES.get(lid, item["text"].replace("—", ", "))

        raw_wav, k_sr = kokoro.create(tts_text, voice=voice_id, speed=base_speed, lang=lang, trim=True)
        comp = trim_silence(raw_wav, k_sr)
        comp_dur = len(comp) / k_sr
        target_fit = slot_dur * 0.94
        used_speed = base_speed

        # Adaptive native speed pass if natural take runs longer than slot
        if comp_dur / target_fit > 1.14:
            desired_speed = float(np.clip(used_speed * (comp_dur / target_fit) * 0.96, used_speed + 0.05, 1.44))
            raw2, k_sr = kokoro.create(tts_text, voice=voice_id, speed=desired_speed, lang=lang, trim=True)
            comp2 = trim_silence(raw2, k_sr)
            if len(comp2) < len(comp):
                comp = comp2
                comp_dur = len(comp) / k_sr
                used_speed = desired_speed

        raw_atempo = 1.0
        if comp_dur > target_fit:
            raw_atempo = comp_dur / target_fit
        elif comp_dur < slot_dur * 0.78:
            raw_atempo = comp_dur / (slot_dur * 0.82)
        atempo = float(np.clip(raw_atempo, ATEMPO_MIN, ATEMPO_MAX))

        stretched_24k = wsola_time_stretch(comp, atempo)
        fitted_48k = resample_to_48k(stretched_24k, k_sr)

        max_samples = int(math.floor(slot_dur * 0.985 * SR))
        if len(fitted_48k) > max_samples:
            fitted_48k = fitted_48k[:max_samples]
            fade_tail = min(len(fitted_48k), int(0.012 * SR))
            fitted_48k[-fade_tail:] *= np.linspace(1.0, 0.0, fade_tail, dtype=np.float32)

        polished = polish_voice(fitted_48k, spk)
        actual_dur = len(polished) / SR
        slack = max(0.0, slot_dur - actual_dur)
        actual_start = slot_start + min(0.025, slack * 0.25)
        actual_end = min(slot_end - 0.01, actual_start + actual_dur)
        start_idx = int(round(actual_start * SR))

        copy_len = min(len(polished), TOTAL_SAMPLES - start_idx)
        # Keep speaker_mono_buses dry for razor-sharp 30Hz dual-axis lip-sync!
        speaker_mono_buses[spk][start_idx : start_idx + copy_len] += polished[:copy_len]

        panned = pan_stereo(polished[:copy_len], pan)
        if spk == "narrator":
            add_stereo_event(narrator_bus, panned, actual_start)
        else:
            add_stereo_event(courtyard_voice_bus, panned, actual_start)

        display_words = tokenize_display_words(item["text"])
        aligned_words = align_words_to_audio(display_words, polished[:copy_len], actual_start, actual_end)
        manifest_lines.append({
            "id": lid,
            "beatId": item["beatId"],
            "speaker": spk,
            "voice": voice_id,
            "lang": lang,
            "pan": pan,
            "text": item["text"],
            "start": round(actual_start, 4),
            "end": round(actual_end, 4),
            "duration": round(actual_dur, 4),
            "kokoroSpeed": round(used_speed, 3),
            "atempo": round(atempo, 4),
            "words": aligned_words,
        })
        print(
            f"  {lid} [{spk:8s} | {voice_id:9s} | pan={pan:+.2f}] "
            f"slot=[{slot_start:5.2f}..{slot_end:5.2f}] act=[{actual_start:5.2f}..{actual_end:5.2f}] "
            f"sp={used_speed:.2f} atempo={atempo:.2f} w={len(aligned_words)}"
        )

    # Apply subtle natural outdoor-courtyard stone early reflections on Newton (-0.22) & Halley (+0.24),
    # plus whisper-subtle air on Narrator (0.0) so voices feel grounded in Woolsthorpe/Cambridge open air!
    courtyard_wet = add_outdoor_courtyard_reverb(courtyard_voice_bus, mix=0.115)
    narrator_air = add_outdoor_courtyard_reverb(narrator_bus, mix=0.03)
    dialogue_bus = (narrator_air + courtyard_wet).astype(np.float32)
    return dialogue_bus, speaker_mono_buses, manifest_lines


# ============================================================================
# 112 BPM CAMBRIDGE CELESTIAL BAROQUE & COSMIC ORCHESTRAL SCORE (100% UNIQUE)
# ============================================================================
def synth_harpsichord_lute(freq: float, dur: float, vel: float = 0.20) -> np.ndarray:
    """Delicate Baroque harpsichord/lute hybrid with crisp quill pluck and warm oak body."""
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    # Plucked bright harmonic series with frequency-dependent damping
    w = (
        0.56 * np.sin(2 * math.pi * freq * t) * np.exp(-t * 4.8)
        + 0.34 * np.sin(2 * math.pi * 2.0 * freq * t) * np.exp(-t * 7.2)
        + 0.22 * np.sin(2 * math.pi * 3.0 * freq * t) * np.exp(-t * 10.5)
        + 0.14 * np.sin(2 * math.pi * 4.0 * freq * t) * np.exp(-t * 14.5)
        + 0.08 * np.sin(2 * math.pi * 6.0 * freq * t) * np.exp(-t * 22.0)
    )
    quill = np.sin(2 * math.pi * (freq * 5.2) * t) * np.exp(-t * 140.0) * 0.12
    attack = np.minimum(1.0, t / 0.0025)
    body = peaking_eq(((w + quill) * attack * vel).astype(np.float32), 310.0, 1.4, 2.5)
    return butter_filter(body, "low", 5800.0, order=2)


def synth_bowed_cello(freq: float, dur: float, vel: float = 0.26, attack: float = 0.18, release: float = 0.24) -> np.ndarray:
    """Warm bowed Baroque cello ensemble with subtle wood resonance and expressive vibrato."""
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    vib = 1.0 + 0.0042 * np.sin(2 * math.pi * 5.1 * t) * np.minimum(1.0, t / 0.28)
    ph1 = 2 * math.pi * freq * np.cumsum(vib) / SR
    ph2 = 2 * math.pi * (freq * 1.0022) * np.cumsum(vib) / SR
    w = (
        0.58 * np.sin(ph1) + 0.30 * np.sin(2 * ph1) + 0.18 * np.sin(3 * ph1) + 0.10 * np.sin(4 * ph1) + 0.05 * np.sin(5 * ph1)
        + 0.36 * np.sin(ph2) + 0.18 * np.sin(2 * ph2) + 0.09 * np.sin(3 * ph2)
    ) * 0.65
    env = adsr_env(n, attack=min(attack, dur * 0.35), decay=0.12, sustain=0.84, release=min(release, dur * 0.35))
    y = butter_filter((w * env * vel).astype(np.float32), "low", 1450.0, order=2)
    return peaking_eq(y, 240.0, 1.3, 2.2)


def synth_cello_ostinato(freq: float, dur: float, vel: float = 0.32) -> np.ndarray:
    """Tight, tense Baroque spiccato cello ostinato stroke."""
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    w = (
        0.68 * np.sin(2 * math.pi * freq * t)
        + 0.36 * np.sin(2 * math.pi * 2 * freq * t)
        + 0.20 * np.sin(2 * math.pi * 3 * freq * t)
        + 0.11 * np.sin(2 * math.pi * 4 * freq * t)
    )
    env = np.minimum(1.0, t / 0.008) * np.exp(-t * 6.4)
    return butter_filter((w * env * vel).astype(np.float32), "low", 1350.0, order=2)


def synth_brass_stab(freqs: list[float], dur: float, vel: float = 0.22) -> np.ndarray:
    """Regal Baroque brass chord stab with authentic brassy flare."""
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    ol = np.zeros(n, dtype=np.float32)
    orr = np.zeros(n, dtype=np.float32)
    env = adsr_env(n, attack=0.025, decay=0.09, sustain=0.68, release=min(0.14, dur * 0.35))
    for i, f in enumerate(freqs):
        fl = f * (1.0 - 0.0018 * ((i % 2) * 2 - 1))
        fr = f * (1.0 + 0.0018 * ((i % 2) * 2 - 1))
        ol += 0.50 * np.sin(2 * math.pi * fl * t) + 0.34 * np.sin(4 * math.pi * fl * t) + 0.22 * np.sin(6 * math.pi * fl * t) + 0.12 * np.sin(8 * math.pi * fl * t)
        orr += 0.50 * np.sin(2 * math.pi * fr * t) + 0.34 * np.sin(4 * math.pi * fr * t) + 0.22 * np.sin(6 * math.pi * fr * t) + 0.12 * np.sin(8 * math.pi * fr * t)
    norm = max(1, len(freqs))
    st = np.stack([ol / norm, orr / norm], axis=-1) * env[:, None] * vel
    return butter_filter(st.astype(np.float32), "low", 2500.0, order=2)


def synth_french_horn(freq: float, dur: float, vel: float = 0.25) -> np.ndarray:
    """Warm, soaring cinematic French horn solo voice."""
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    vib = 1.0 + 0.0040 * np.sin(2 * math.pi * 4.9 * t) * np.minimum(1.0, t / 0.24)
    ph = 2 * math.pi * freq * np.cumsum(vib) / SR
    flare = np.minimum(1.0, t / 0.09)
    w = (
        0.54 * np.sin(ph)
        + 0.36 * np.sin(2 * ph) * (0.65 + 0.35 * flare)
        + 0.24 * np.sin(3 * ph) * flare
        + 0.13 * np.sin(4 * ph) * flare
        + 0.06 * np.sin(5 * ph) * flare
    )
    env = adsr_env(n, attack=min(0.085, dur * 0.22), decay=0.15, sustain=0.80, release=min(0.32, dur * 0.30))
    return butter_filter((w * env * vel).astype(np.float32), "low", 2250.0, order=2)


def synth_glass_celesta(freq: float, dur: float, vel: float = 0.17) -> np.ndarray:
    """Crystalline prism glass-celesta chime with pure delicate bell overtones."""
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    w = (
        0.74 * np.sin(2 * math.pi * freq * t) * np.exp(-t * 3.1)
        + 0.20 * np.sin(2 * math.pi * 2.0 * freq * t) * np.exp(-t * 5.8)
        + 0.11 * np.sin(2 * math.pi * 3.005 * freq * t) * np.exp(-t * 9.5)
        + 0.06 * np.sin(2 * math.pi * 5.01 * freq * t) * np.exp(-t * 16.0)
    )
    attack = np.minimum(1.0, t / 0.004)
    return (w * attack * vel).astype(np.float32)


def synth_concert_harp(freq: float, dur: float, vel: float = 0.16) -> np.ndarray:
    """Warm glistening orchestral concert harp pluck."""
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    w = (
        0.68 * np.sin(2 * math.pi * freq * t) * np.exp(-t * 2.6)
        + 0.28 * np.sin(2 * math.pi * 2.0 * freq * t) * np.exp(-t * 4.9)
        + 0.12 * np.sin(2 * math.pi * 3.0 * freq * t) * np.exp(-t * 8.2)
    )
    attack = np.minimum(1.0, t / 0.005)
    return butter_filter((w * attack * vel).astype(np.float32), "low", 4600.0, order=2)


def synth_pizzicato(freq: float, dur: float, vel: float = 0.25) -> np.ndarray:
    """Playful staccato orchestral pizzicato string pluck with wooden fingerbody."""
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    pitch_env = freq * (1.0 + 0.018 * np.exp(-t * 65.0))
    ph = 2 * math.pi * np.cumsum(pitch_env) / SR
    w = (
        0.74 * np.sin(ph) * np.exp(-t * 9.5)
        + 0.28 * np.sin(2.0 * ph) * np.exp(-t * 15.0)
        + 0.12 * np.sin(3.0 * ph) * np.exp(-t * 22.0)
    )
    thump = np.sin(2 * math.pi * 160.0 * t) * np.exp(-t * 85.0) * 0.18
    attack = np.minimum(1.0, t / 0.003)
    return butter_filter(((w + thump) * attack * vel).astype(np.float32), "low", 2600.0, order=2)


def synth_orchestral_strings(freqs: list[float], dur: float, vel: float = 0.20, attack: float = 0.38, release: float = 0.45) -> np.ndarray:
    """Wide lush Hans-Zimmer-meets-Handel stereo orchestral string choir."""
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    ol = np.zeros(n, dtype=np.float32)
    orr = np.zeros(n, dtype=np.float32)
    env = adsr_env(n, attack=min(attack, dur * 0.28), decay=0.22, sustain=0.84, release=min(release, dur * 0.28))
    for i, f in enumerate(freqs):
        # 3 detuned voices per pitch distributed left/right for rich cinematic width
        sign = 1.0 if i % 2 == 0 else -1.0
        vib = 1.0 + 0.0028 * np.sin(2 * math.pi * (4.6 + 0.3 * (i % 3)) * t + i)
        fl1 = f * (1.0 - 0.0024 * sign) * vib
        fr1 = f * (1.0 + 0.0024 * sign) * vib
        fl2 = f * (1.0 + 0.0011 * sign)
        fr2 = f * (1.0 - 0.0011 * sign)
        pl1 = 2 * math.pi * np.cumsum(fl1) / SR
        pr1 = 2 * math.pi * np.cumsum(fr1) / SR
        pl2 = 2 * math.pi * fl2 * t
        pr2 = 2 * math.pi * fr2 * t
        ol += (0.52 * np.sin(pl1) + 0.24 * np.sin(2 * pl1) + 0.11 * np.sin(3 * pl1) + 0.34 * np.sin(pl2) + 0.14 * np.sin(2 * pl2)) * 0.65
        orr += (0.52 * np.sin(pr1) + 0.24 * np.sin(2 * pr1) + 0.11 * np.sin(3 * pr1) + 0.34 * np.sin(pr2) + 0.14 * np.sin(2 * pr2)) * 0.65
    norm = max(1, len(freqs))
    st = np.stack([ol / norm, orr / norm], axis=-1) * env[:, None] * vel
    return butter_filter(st.astype(np.float32), "low", 3400.0, order=2)


def synth_cosmic_choir(freqs: list[float], dur: float, vel: float = 0.18) -> np.ndarray:
    """Ethereal space-time vocal choir pad shaped with warm 'Ah/Oh' vocal formants."""
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    ol = np.zeros(n, dtype=np.float32)
    orr = np.zeros(n, dtype=np.float32)
    env = adsr_env(n, attack=min(0.65, dur * 0.30), decay=0.25, sustain=0.86, release=min(0.70, dur * 0.30))
    for i, f in enumerate(freqs):
        fl = f * (1.0 - 0.0026)
        fr = f * (1.0 + 0.0026)
        for h in range(1, 7):
            amp = 1.0 / (h ** 1.15)
            ol += amp * np.sin(2 * math.pi * fl * h * t + i * 0.7)
            orr += amp * np.sin(2 * math.pi * fr * h * t + i * 1.1)
    norm = max(1, len(freqs)) * 1.8
    raw = np.stack([ol / norm, orr / norm], axis=-1).astype(np.float32)
    # Blend warm 'Oh'/'Ah' choral formant bands (500Hz + 1020Hz + airy 2550Hz)
    f1 = butter_filter(raw, "band", (380.0, 720.0), order=2) * 0.72
    f2 = butter_filter(raw, "band", (850.0, 1320.0), order=2) * 0.42
    f3 = butter_filter(raw, "band", (2200.0, 3100.0), order=2) * 0.18
    choir = (f1 + f2 + f3) * env[:, None] * vel
    return choir.astype(np.float32)


def synth_timpani(freq: float, dur: float, vel: float = 0.34) -> np.ndarray:
    """Orchestral pitched kettle drum resonance."""
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    f_inst = freq * (1.0 + 0.14 * np.exp(-t * 28.0))
    ph = 2 * math.pi * np.cumsum(f_inst) / SR
    w = (
        0.78 * np.sin(ph) * np.exp(-t * 3.4)
        + 0.28 * np.sin(1.50 * ph) * np.exp(-t * 6.5)
        + 0.14 * np.sin(2.02 * ph) * np.exp(-t * 10.0)
    )
    attack = np.minimum(1.0, t / 0.006)
    return butter_filter((w * attack * vel).astype(np.float32), "low", 480.0, order=2)


def compose_newton_score() -> np.ndarray:
    """Compose 30.000s unique Cambridge Celestial Baroque & Cosmic Orchestral Score."""
    bgm = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)

    # ------------------------------------------------------------------
    # BEAT 1 (0.00–4.50s): Woolsthorpe Orchard Curiosity (D Minor)
    # Curious plucked Baroque lute/harpsichord arpeggios + warm low cello drone
    # ------------------------------------------------------------------
    b1_arp = [62, 65, 69, 72, 69, 65, 64, 69, 58, 62, 65, 69, 67, 65, 64, 62, 57, 61, 64, 69, 73, 69, 64, 61]
    t_cur = 0.0
    idx = 0
    while t_cur < 4.35:
        m = b1_arp[idx % len(b1_arp)]
        pan = math.sin(idx * 0.72) * 0.38
        vel = 0.19 if t_cur < 1.45 else 0.17
        add_stereo_event(bgm, pan_stereo(synth_harpsichord_lute(midi_to_hz(m), 0.44, vel=vel), pan), t_cur)
        t_cur += EIGHTH * 0.85
        idx += 1

    # Warm rising cello drone under Beat 1 (D2 + A2 -> Bb1 -> A1 suspended question)
    add_stereo_event(bgm, pan_stereo(synth_bowed_cello(midi_to_hz(38), 2.55, vel=0.28, attack=0.35), -0.14), 0.0)
    add_stereo_event(bgm, pan_stereo(synth_bowed_cello(midi_to_hz(45), 2.55, vel=0.17, attack=0.45), 0.14), 0.05)
    add_stereo_event(bgm, pan_stereo(synth_bowed_cello(midi_to_hz(34), 1.05, vel=0.26, attack=0.18), -0.12), 2.45)
    add_stereo_event(bgm, pan_stereo(synth_bowed_cello(midi_to_hz(33), 1.10, vel=0.25, attack=0.16), -0.10), 3.40)
    add_stereo_event(bgm, synth_orchestral_strings([midi_to_hz(m) for m in (50, 57, 62, 65)], 2.45, vel=0.15, attack=0.45), 0.05)
    add_stereo_event(bgm, synth_orchestral_strings([midi_to_hz(m) for m in (46, 53, 58, 62)], 2.05, vel=0.16, attack=0.30), 2.40)
    # Curious lunar celesta notes as "does the Moon fall too?" arrives
    for k, (tm, mn) in enumerate([(3.12, 81), (3.44, 86), (3.82, 88)]):
        add_stereo_event(bgm, pan_stereo(synth_glass_celesta(midi_to_hz(mn), 0.85, vel=0.13), 0.18 + k * 0.12), tm)

    # ------------------------------------------------------------------
    # BEAT 2 (4.50–9.50s): Tense Halley vs Newton Debate -> Bright Gravity Lift (t=6.95s)
    # ------------------------------------------------------------------
    # 4.50–6.90s: Dramatic tense cello ostinato + regal brass stabs (D Minor)
    ost_notes = [38, 38, 41, 38, 43, 41, 40, 38, 45, 46, 45, 43]
    t_cur = 4.52
    o_idx = 0
    while t_cur < 6.86:
        m = ost_notes[o_idx % len(ost_notes)]
        add_stereo_event(bgm, pan_stereo(synth_cello_ostinato(midi_to_hz(m), 0.34, vel=0.31), -0.12), t_cur)
        if o_idx % 2 == 0:
            add_stereo_event(bgm, pan_stereo(synth_harpsichord_lute(midi_to_hz(m + 24), 0.28, vel=0.14), 0.26), t_cur)
        t_cur += EIGHTH * 0.80
        o_idx += 1

    # Regal Baroque brass stabs punctuating Halley's challenge
    add_stereo_event(bgm, synth_brass_stab([midi_to_hz(m) for m in (50, 57, 62)], 0.36, vel=0.22), 4.54)
    add_stereo_event(bgm, synth_brass_stab([midi_to_hz(m) for m in (53, 57, 65)], 0.32, vel=0.20), 5.34)
    add_stereo_event(bgm, synth_brass_stab([midi_to_hz(m) for m in (57, 61, 69)], 0.44, vel=0.22), 6.18)

    # t=6.95s: RADIANT BRIGHT HARMONIC CHORD LIFT INTO D MAJOR as Newton reveals gravity!
    add_stereo_event(bgm, pan_stereo(synth_timpani(midi_to_hz(38), 1.15, vel=0.36), 0.0), 6.95)
    add_stereo_event(bgm, synth_orchestral_strings([midi_to_hz(m) for m in (50, 57, 62, 66, 69)], 2.55, vel=0.24, attack=0.22), 6.95)
    add_stereo_event(bgm, pan_stereo(synth_bowed_cello(midi_to_hz(38), 2.50, vel=0.28, attack=0.12), -0.10), 6.95)
    for th, mh, dh in [(6.98, 66, 0.52), (7.52, 69, 0.56), (8.10, 74, 0.65), (8.78, 76, 0.65)]:
        add_stereo_event(bgm, pan_stereo(synth_french_horn(midi_to_hz(mh), dh, vel=0.21), -0.16), th)

    # ------------------------------------------------------------------
    # BEAT 3 (9.50–15.50s): 7-Spectral Rainbow Prism Bloom (D Major / Lydian)
    # Shimmering crystalline glass-celesta & high harp arpeggios across 7 spectral notes!
    # ------------------------------------------------------------------
    add_stereo_event(bgm, synth_orchestral_strings([midi_to_hz(m) for m in (50, 57, 62, 66, 73)], 2.20, vel=0.20, attack=0.38), 9.50)
    add_stereo_event(bgm, synth_orchestral_strings([midi_to_hz(m) for m in (55, 62, 66, 71, 74)], 2.15, vel=0.22, attack=0.35), 11.55)
    add_stereo_event(bgm, synth_orchestral_strings([midi_to_hz(m) for m in (57, 64, 69, 73, 78)], 1.95, vel=0.21, attack=0.35), 13.55)

    # 7-note radiant spectral scale (D5, E5, F#5, G#5, A5, B5, C#6 -> D6)
    spectral_7 = [74, 76, 78, 80, 81, 83, 85, 86, 85, 83, 81, 78, 76, 74]
    t_cur = 9.65
    s_idx = 0
    while t_cur < 15.25:
        m = spectral_7[s_idx % len(spectral_7)]
        # Sweep rainbow stereo from left to right across the 7 colors!
        pos = (s_idx % 7) / 6.0
        pan = -0.52 + 1.04 * pos
        add_stereo_event(bgm, pan_stereo(synth_glass_celesta(midi_to_hz(m), 0.72, vel=0.16), pan), t_cur)
        if s_idx % 2 == 0:
            add_stereo_event(bgm, pan_stereo(synth_concert_harp(midi_to_hz(m - 12), 0.90, vel=0.15), -pan * 0.7), t_cur + 0.04)
        t_cur += SIXTEENTH * 1.45
        s_idx += 1

    # ------------------------------------------------------------------
    # BEAT 4 (15.50–20.50s): Playful Staccato Pizzicato -> CRISP COMEDIC STOP @ 17.45s!
    # ------------------------------------------------------------------
    pizz_seq = [62, 66, 69, 66, 67, 71, 69, 66, 64, 67, 69]
    for k, m in enumerate(pizz_seq):
        tp = 15.56 + k * (EIGHTH * 0.64)
        if tp < 17.34:
            add_stereo_event(bgm, pan_stereo(synth_pizzicato(midi_to_hz(m), 0.24, vel=0.27), -0.26 if k % 2 == 0 else 0.26), tp)
            if k % 2 == 0:
                add_stereo_event(bgm, pan_stereo(synth_pizzicato(midi_to_hz(m - 24), 0.28, vel=0.28), 0.0), tp)

    # *** CRISP COMEDIC ORCHESTRAL STOP: 17.38s .. 18.92s ***
    # Music drops completely silent right on the 17.45s steel-sphere CLACK & Halley's "Whoa!"

    # Witty pizzicato rebound into Beat 5 (18.96..20.32s)
    rebound_seq = [57, 59, 61, 62, 64, 66, 67, 69]
    for k, m in enumerate(rebound_seq):
        tp = 18.96 + k * (EIGHTH * 0.65)
        add_stereo_event(bgm, pan_stereo(synth_pizzicato(midi_to_hz(m), 0.26, vel=0.24), -0.18 + k * 0.05), tp)
        if k % 2 == 0:
            add_stereo_event(bgm, pan_stereo(synth_harpsichord_lute(midi_to_hz(m + 12), 0.28, vel=0.14), 0.20), tp)

    # ------------------------------------------------------------------
    # BEAT 5 (20.50–25.50s): Soaring Zimmer-meets-Handel Cosmic Gravity Well Climax!
    # Orchestral strings + soaring French horn + regal brass + ethereal cosmic choir pad
    # ------------------------------------------------------------------
    add_stereo_event(bgm, pan_stereo(synth_timpani(midi_to_hz(38), 1.25, vel=0.42), 0.0), 20.52)
    add_stereo_event(bgm, pan_stereo(synth_timpani(midi_to_hz(34), 1.05, vel=0.36), -0.08), 22.12)
    add_stereo_event(bgm, pan_stereo(synth_timpani(midi_to_hz(36), 1.05, vel=0.38), 0.08), 23.25)
    add_stereo_event(bgm, pan_stereo(synth_timpani(midi_to_hz(38), 1.35, vel=0.44), 0.0), 24.38)

    # Epic bVI - bVII - I orchestral string progression (D -> Bb -> C -> D)
    add_stereo_event(bgm, synth_orchestral_strings([midi_to_hz(m) for m in (50, 57, 62, 66, 69, 74)], 1.85, vel=0.28, attack=0.30), 20.50)
    add_stereo_event(bgm, synth_orchestral_strings([midi_to_hz(m) for m in (46, 53, 58, 62, 65, 70)], 1.35, vel=0.29, attack=0.22), 22.10)
    add_stereo_event(bgm, synth_orchestral_strings([midi_to_hz(m) for m in (48, 55, 60, 64, 67, 72)], 1.35, vel=0.31, attack=0.22), 23.25)
    add_stereo_event(bgm, synth_orchestral_strings([midi_to_hz(m) for m in (50, 57, 62, 66, 69, 74, 78)], 1.50, vel=0.33, attack=0.20), 24.35)

    # Deep sub-cello pedal underpinning the gravity well
    add_stereo_event(bgm, pan_stereo(synth_bowed_cello(midi_to_hz(38), 1.75, vel=0.33, attack=0.18), -0.06), 20.50)
    add_stereo_event(bgm, pan_stereo(synth_bowed_cello(midi_to_hz(34), 1.25, vel=0.33, attack=0.15), -0.06), 22.10)
    add_stereo_event(bgm, pan_stereo(synth_bowed_cello(midi_to_hz(36), 1.25, vel=0.34, attack=0.15), 0.06), 23.25)
    add_stereo_event(bgm, pan_stereo(synth_bowed_cello(midi_to_hz(38), 1.40, vel=0.35, attack=0.15), 0.0), 24.35)

    # Cosmic choir pad blooming as the Earth-Moon gravity well unfolds
    add_stereo_event(bgm, synth_cosmic_choir([midi_to_hz(m) for m in (62, 69, 74, 78)], 2.40, vel=0.23), 20.95)
    add_stereo_event(bgm, synth_cosmic_choir([midi_to_hz(m) for m in (60, 67, 72, 76)], 1.45, vel=0.25), 23.15)
    add_stereo_event(bgm, synth_cosmic_choir([midi_to_hz(m) for m in (62, 69, 74, 78)], 1.45, vel=0.27), 24.35)

    # Soaring heroic French horn theme across Beat 5 (with warm lower octave harmonic reinforcement on climax)
    horn_b5 = [
        (20.68, 69, 0.62), (21.34, 74, 0.72), (22.12, 77, 0.56), (22.72, 74, 0.50),
        (23.28, 76, 0.56), (23.88, 79, 0.48), (24.40, 81, 1.05),
    ]
    for th, mh, dh in horn_b5:
        add_stereo_event(bgm, pan_stereo(synth_french_horn(midi_to_hz(mh), dh, vel=0.25), -0.10), th)
    add_stereo_event(bgm, pan_stereo(synth_french_horn(midi_to_hz(69), 1.05, vel=0.19), 0.14), 24.40)

    # ------------------------------------------------------------------
    # BEAT 6 (25.50–30.00s): Soaring Triumphant Brass & Strings Crescendo
    # Crowns Newton's legacy & lands cleanly at 29.85s for seamless loop!
    # ------------------------------------------------------------------
    add_stereo_event(bgm, pan_stereo(synth_timpani(midi_to_hz(43), 1.05, vel=0.38), 0.0), 25.52)
    add_stereo_event(bgm, pan_stereo(synth_timpani(midi_to_hz(45), 0.90, vel=0.35), 0.08), 26.64)
    add_stereo_event(bgm, pan_stereo(synth_timpani(midi_to_hz(38), 1.80, vel=0.44), 0.0), 27.60)

    # Grand rising strings crescendo: G -> A7sus4 -> radiant wide D major resolution!
    add_stereo_event(bgm, synth_orchestral_strings([midi_to_hz(m) for m in (55, 62, 67, 71, 74, 79)], 1.35, vel=0.30, attack=0.20), 25.50)
    add_stereo_event(bgm, synth_orchestral_strings([midi_to_hz(m) for m in (57, 64, 69, 73, 76, 81)], 1.25, vel=0.32, attack=0.18), 26.62)
    add_stereo_event(bgm, synth_orchestral_strings([midi_to_hz(m) for m in (50, 57, 62, 66, 69, 74, 78, 86)], 2.20, vel=0.34, attack=0.20, release=0.78), 27.60)
    add_stereo_event(bgm, pan_stereo(synth_bowed_cello(midi_to_hz(38), 2.10, vel=0.34, attack=0.14, release=0.65), 0.0), 27.60)

    # Soaring triumphant resolving brass & octave French horns crescendo
    horn_b6 = [
        (25.58, 79, 0.56), (26.16, 78, 0.48), (26.66, 76, 0.88), (27.60, 74, 1.95),
    ]
    for th, mh, dh in horn_b6:
        add_stereo_event(bgm, pan_stereo(synth_french_horn(midi_to_hz(mh), dh, vel=0.26), 0.12), th)
    add_stereo_event(bgm, pan_stereo(synth_french_horn(midi_to_hz(86), 1.85, vel=0.17), -0.14), 27.62)
    add_stereo_event(bgm, synth_brass_stab([midi_to_hz(m) for m in (55, 62, 67, 71)], 0.95, vel=0.16), 25.55)
    add_stereo_event(bgm, synth_brass_stab([midi_to_hz(m) for m in (57, 64, 69, 73)], 0.90, vel=0.18), 26.65)
    add_stereo_event(bgm, synth_brass_stab([midi_to_hz(m) for m in (50, 57, 62, 66, 74)], 1.75, vel=0.23), 27.62)
    add_stereo_event(bgm, synth_cosmic_choir([midi_to_hz(m) for m in (62, 66, 69, 74)], 2.00, vel=0.22), 27.58)

    # Sparkling celesta & concert harp resolution cascade settling cleanly before 29.85s
    cadence_cel = [86, 81, 78, 74, 81, 86, 90, 86, 81, 74]
    for k, m in enumerate(cadence_cel):
        tf = 26.12 + k * 0.27
        if tf < 28.92:
            pan_c = math.sin(k * 0.9) * 0.52
            add_stereo_event(bgm, pan_stereo(synth_glass_celesta(midi_to_hz(m), 0.82, vel=0.17), pan_c), tf)
            if k % 2 == 0:
                add_stereo_event(bgm, pan_stereo(synth_concert_harp(midi_to_hz(m - 12), 0.90, vel=0.14), -pan_c * 0.7), tf + 0.03)

    # Warm lush concert-hall ambience on score
    bgm = add_cambridge_chamber_reverb(bgm, mix=0.22)

    # Smooth cosine fade-out ending cleanly at 29.85s (zero click on seamless YouTube Short loop!)
    fade_start = int(round(29.15 * SR))
    fade_end = int(round(29.85 * SR))
    fade_len = fade_end - fade_start
    if fade_len > 0:
        curve = 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, fade_len, dtype=np.float32)))
        bgm[fade_start:fade_end] *= curve[:, None]
    bgm[fade_end:] = 0.0
    return bgm


# ============================================================================
# LIVING OUTDOOR AMBIENCE BED (-34 dB SUBTLE CINEMA IMMERSION)
# Beats 1–4: English Golden-Hour Meadow Breeze & Distant Birdsong Whisper
# Crossfade @ t=20.0s -> Beats 5–6: Nighttime Crickets & Celestial Cosmic Hum
# ============================================================================
def _synth_distant_birdsong(rng: np.random.Generator, variant: int) -> np.ndarray:
    """Delicate far-field English orchard songbird whisper (blackbird/robin soft melodic FM)."""
    n = int(0.85 * SR)
    out = np.zeros(n, dtype=np.float32)
    patterns = [
        [(0.02, 0.14, 2850.0, 3450.0), (0.20, 0.19, 3550.0, 3050.0), (0.44, 0.16, 3150.0, 3680.0)],
        [(0.04, 0.17, 3250.0, 2720.0), (0.26, 0.22, 2900.0, 3520.0)],
        [(0.02, 0.11, 3650.0, 3920.0), (0.16, 0.12, 3780.0, 3380.0), (0.33, 0.20, 3100.0, 3580.0)],
    ]
    notes = patterns[variant % len(patterns)]
    for t0, dur, f0, f1 in notes:
        i0 = int(t0 * SR)
        m = min(n - i0, int(dur * SR))
        if m <= 8:
            continue
        tt = np.arange(m, dtype=np.float32) / SR
        u = tt / max(1e-4, dur)
        # Smooth organic syrinx pitch glide + delicate vibrato
        f_inst = f0 + (f1 - f0) * (0.5 - 0.5 * np.cos(math.pi * u)) + 28.0 * np.sin(2.0 * math.pi * 22.0 * tt)
        ph = 2.0 * math.pi * np.cumsum(f_inst) / SR
        env = np.maximum(0.0, np.sin(math.pi * u)) ** 1.4
        chirp = (0.85 * np.sin(ph) + 0.15 * np.sin(2.0 * ph)) * env
        out[i0 : i0 + m] += chirp.astype(np.float32)
    # Soften & distance-filter so birdsong whispers gently 40m away across Woolsthorpe meadow
    out = butter_filter(out, "band", (2200.0, 5200.0), order=2)
    return (out * 0.24).astype(np.float32)


def compose_living_ambience_bed() -> np.ndarray:
    """Synthesize 30.000s living outdoor ambience bed with smooth equal-power crossfade at t=20.0s."""
    rng = np.random.default_rng(16661687)
    t = np.arange(TOTAL_SAMPLES, dtype=np.float32) / SR

    # ------------------------------------------------------------------
    # PART A (Beats 1–4, 0.00–21.00s): Golden-Hour English Meadow Breeze & Distant Birdsong
    # ------------------------------------------------------------------
    # Decorrelated stereo warm breeze + delicate orchard leaf air
    n_l = rng.standard_normal(TOTAL_SAMPLES).astype(np.float32)
    n_r = rng.standard_normal(TOTAL_SAMPLES).astype(np.float32)
    breeze_l = butter_filter(n_l, "band", (160.0, 880.0), order=2)
    breeze_r = butter_filter(n_r, "band", (160.0, 880.0), order=2)
    foliage_l = butter_filter(n_l, "band", (2100.0, 5000.0), order=2) * 0.24
    foliage_r = butter_filter(n_r, "band", (2100.0, 5000.0), order=2) * 0.24

    # Slow natural breathing wind sway (asymmetric L/R phase for wide open-air feel)
    sway_l = 0.68 + 0.32 * np.sin(2.0 * math.pi * 0.16 * t + 0.4) * np.cos(2.0 * math.pi * 0.07 * t)
    sway_r = 0.68 + 0.32 * np.sin(2.0 * math.pi * 0.19 * t + 1.9) * np.cos(2.0 * math.pi * 0.09 * t)
    meadow = np.stack([(breeze_l + foliage_l) * sway_l, (breeze_r + foliage_r) * sway_r], axis=-1).astype(np.float32)

    # Layer subtle distant English birdsong whispers in Beats 1–4
    bird_cues = [
        (0.46, -0.66, 0),
        (2.38, 0.68, 1),
        (4.92, -0.72, 2),
        (8.62, 0.64, 0),
        (10.22, -0.68, 1),
        (14.85, 0.70, 2),
        (16.08, -0.64, 0),
    ]
    birds = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)
    for bt, bpan, bvar in bird_cues:
        add_stereo_event(birds, pan_stereo(_synth_distant_birdsong(rng, bvar), bpan), bt)
    birds = add_outdoor_courtyard_reverb(birds, mix=0.22)
    day_bed = (meadow * 0.85 + birds * 0.48).astype(np.float32)

    # ------------------------------------------------------------------
    # PART B (Beats 5–6, 19.00–29.85s): Gentle Nighttime Crickets & Celestial Cosmic Hum
    # ------------------------------------------------------------------
    # Two delicate distant summer-night English field crickets (left & right meadow)
    cr_gate_l = (np.maximum(0.0, np.sin(2.0 * math.pi * 2.15 * t + 0.2)) ** 3.0) * (
        0.5 + 0.5 * np.sin(2.0 * math.pi * 34.0 * t)
    )
    cr_gate_r = (np.maximum(0.0, np.sin(2.0 * math.pi * 2.42 * t + 1.7)) ** 3.0) * (
        0.5 + 0.5 * np.sin(2.0 * math.pi * 37.0 * t)
    )
    cricket_l = np.sin(2.0 * math.pi * 4120.0 * t + 0.35 * np.sin(2.0 * math.pi * 34.0 * t)) * cr_gate_l * 0.14
    cricket_r = np.sin(2.0 * math.pi * 4460.0 * t + 0.35 * np.sin(2.0 * math.pi * 37.0 * t)) * cr_gate_r * 0.12
    crickets = butter_filter(
        np.stack([cricket_l * 0.82 + cricket_r * 0.22, cricket_r * 0.82 + cricket_l * 0.22], axis=-1).astype(np.float32),
        "band",
        (3200.0, 5600.0),
        order=2,
    )

    # Velvety celestial cosmic hum (warm D2+A2+D3 fifth harmonic bed + airy starlight breath)
    cosmic_l = (
        0.48 * np.sin(2.0 * math.pi * 73.42 * t)
        + 0.30 * np.sin(2.0 * math.pi * 110.00 * t + 0.3)
        + 0.18 * np.sin(2.0 * math.pi * 146.83 * t + 0.8)
    ) * (0.85 + 0.15 * np.sin(2.0 * math.pi * 0.22 * t))
    cosmic_r = (
        0.48 * np.sin(2.0 * math.pi * 73.60 * t + 0.5)
        + 0.30 * np.sin(2.0 * math.pi * 110.22 * t + 1.1)
        + 0.18 * np.sin(2.0 * math.pi * 146.50 * t + 1.6)
    ) * (0.85 + 0.15 * np.sin(2.0 * math.pi * 0.25 * t + 1.2))
    star_air_l = butter_filter(rng.standard_normal(TOTAL_SAMPLES).astype(np.float32), "band", (1350.0, 2900.0), order=2) * 0.16
    star_air_r = butter_filter(rng.standard_normal(TOTAL_SAMPLES).astype(np.float32), "band", (1350.0, 2900.0), order=2) * 0.16
    night_bed = np.stack([cosmic_l * 0.42 + star_air_l, cosmic_r * 0.42 + star_air_r], axis=-1).astype(np.float32) + crickets * 0.65

    # Normalize day and night halves to matched gentle energy before equal-power crossfade @ t=20.0s
    day_rms = float(np.sqrt(np.mean(day_bed[: int(19.0 * SR)] ** 2) + 1e-12))
    night_rms = float(np.sqrt(np.mean(night_bed[int(21.0 * SR) : int(29.0 * SR)] ** 2) + 1e-12))
    day_bed = day_bed / max(1e-6, day_rms)
    night_bed = night_bed / max(1e-6, night_rms)

    # Smooth equal-power cosine crossfade across 19.00s..21.00s (centered cleanly at t=20.00s!)
    xf0, xf1 = 19.0, 21.0
    u = np.clip((t - xf0) / (xf1 - xf0), 0.0, 1.0)
    w_day = np.cos(0.5 * math.pi * u).astype(np.float32)
    w_night = np.sin(0.5 * math.pi * u).astype(np.float32)
    amb = (day_bed * w_day[:, None] + night_bed * w_night[:, None]).astype(np.float32)

    # Smooth fade-in (0.00..0.35s) and fade-out ending cleanly at 29.85s
    fi = int(round(0.35 * SR))
    amb[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)[:, None]
    fo0 = int(round(29.15 * SR))
    fo1 = int(round(29.85 * SR))
    if fo1 > fo0:
        curve = 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, fo1 - fo0, dtype=np.float32)))
        amb[fo0:fo1] *= curve[:, None]
    amb[fo1:] = 0.0
    return amb


# ============================================================================
# 26 BESPOKE SPATIAL FOLEY SFX (V9 PHONE-SPEAKER-TESTED IMAX IMPACT PUNCH!)
# ============================================================================
def synth_sfx(kind: str, pan: float, event_t: float) -> np.ndarray:
    seed = abs(hash((kind, round(event_t, 2)))) % (2**31 - 1)
    rng = np.random.default_rng(seed)

    if kind == "twig_snap":
        # Crisp dry orchard twig dual-snap + delicate rustle of leaves + golden-hour breeze whisper!
        n = int(0.52 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        raw = rng.standard_normal(n).astype(np.float32)
        crack = butter_filter(raw, "band", (1350.0, 6400.0), order=3) * np.exp(-t * 105.0) * 1.05
        # Secondary tiny wood-fiber splinter 14 ms later
        s2_idx = int(0.014 * SR)
        splinter = np.zeros(n, dtype=np.float32)
        t2 = np.arange(n - s2_idx, dtype=np.float32) / SR
        splinter[s2_idx:] = (
            butter_filter(rng.standard_normal(n - s2_idx).astype(np.float32), "band", (1800.0, 7200.0), order=2)
            * np.exp(-t2 * 125.0)
            * 0.55
        )
        wood = (
            0.55 * np.sin(2 * math.pi * 720.0 * t)
            + 0.36 * np.sin(2 * math.pi * 1480.0 * t)
        ) * np.exp(-t * 62.0) * 0.48
        # Delicate fluttering orchard leaf rustle (amplitude-modulated 2.4-8.8 kHz)
        leaf_mod = 0.55 + 0.45 * np.sin(2 * math.pi * 19.0 * t) * np.cos(2 * math.pi * 7.5 * t)
        rustle = (
            butter_filter(rng.standard_normal(n).astype(np.float32), "band", (2400.0, 8800.0), order=2)
            * leaf_mod
            * np.minimum(1.0, t / 0.030)
            * np.exp(-t * 6.8)
            * 0.34
        )
        # Soft orchard breeze whisper swelling right behind the detached apple
        breeze = (
            butter_filter(rng.standard_normal(n).astype(np.float32), "band", (320.0, 1400.0), order=2)
            * (np.maximum(0.0, np.sin(math.pi * np.clip(t / 0.50, 0.0, 1.0))) ** 1.3)
            * 0.24
        )
        return pan_stereo((crack + splinter + wood + rustle + breeze) * 0.90, pan)

    if kind == "apple_fall_whoosh":
        # Downward-accelerating airy orchard breeze whoosh (pitch falls gently with gravity)
        n = int(0.50 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        noise = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (320.0, 2400.0), order=2)
        f_drop = 620.0 * np.exp(-t * 2.4) + 160.0
        tone = np.sin(2 * math.pi * np.cumsum(f_drop) / SR) * 0.16
        env = np.maximum(0.0, np.sin(math.pi * (t / 0.50))) ** 1.30
        return pan_stereo((noise * 0.48 + tone) * env * 0.78, pan)

    if kind == "apple_velvet_thud":
        # V9 Phone-Speaker-Tested Apple Thud:
        # Warm sub-bass thump (65-90 Hz) + phone-speaker 2nd harmonic + crisp mid-woody velvet tap + micro-bounce!
        n = int(0.42 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        # 1. Warm sub-bass thump strictly in 65-90 Hz sweet spot (92 Hz -> 66 Hz pitch drop)
        f_sub = 66.0 + 26.0 * np.exp(-t * 26.0)
        ph_sub = 2.0 * math.pi * np.cumsum(f_sub) / SR
        sub_thump = (
            0.88 * np.sin(ph_sub) * np.exp(-t * 14.5)
            + 0.36 * np.sin(2.0 * ph_sub) * np.exp(-t * 22.0)  # Phone-speaker chest punch (132-184 Hz)
        )
        # 2. Crisp mid-woody oak/velvet tap (tactile wooden table resonance + damped velvet slap)
        wood_tap = (
            0.34 * np.sin(2.0 * math.pi * 285.0 * t) * np.exp(-t * 36.0)
            + 0.24 * np.sin(2.0 * math.pi * 525.0 * t) * np.exp(-t * 54.0)
            + 0.16 * np.sin(2.0 * math.pi * 910.0 * t) * np.exp(-t * 82.0)
        )
        velvet_slap = (
            butter_filter(rng.standard_normal(n).astype(np.float32), "band", (360.0, 2150.0), order=2)
            * np.exp(-t * 48.0)
            * 0.52
        )
        # 3. Delicate micro-bounce settle 105 ms later + tiny leaf/velvet brush
        b_idx = int(0.105 * SR)
        bounce = np.zeros(n, dtype=np.float32)
        m = n - b_idx
        tb = np.arange(m, dtype=np.float32) / SR
        bounce[b_idx:] = (
            0.26 * np.sin(2.0 * math.pi * 88.0 * tb) * np.exp(-tb * 24.0)
            + 0.14 * np.sin(2.0 * math.pi * 320.0 * tb) * np.exp(-tb * 48.0)
        )
        attack = np.minimum(1.0, t / 0.0022)
        return pan_stereo(((sub_thump + wood_tap + velvet_slap) * attack + bounce) * 0.98, pan)

    if kind == "gravity_pulse_chime":
        # Golden gravitational pull resonance + harmonic D-major fifth chime
        n = int(1.05 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        sub = np.sin(2 * math.pi * (95.0 * np.exp(-t * 2.5) + 73.42) * t) * np.exp(-t * 3.8) * 0.44
        chime = (
            0.42 * np.sin(2 * math.pi * 587.33 * t)
            + 0.30 * np.sin(2 * math.pi * 880.00 * t)
            + 0.20 * np.sin(2 * math.pi * 1174.66 * t)
            + 0.12 * np.sin(2 * math.pi * 1760.00 * t)
        ) * np.minimum(1.0, t / 0.012) * np.exp(-t * 3.3)
        return pan_stereo((sub + chime) * 0.66, pan)

    if kind == "celestial_shimmer":
        # Lunar ethereal harmonic shimmer sweeping softly across stereo
        n = int(1.15 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        env = adsr_env(n, attack=0.22, decay=0.20, sustain=0.58, release=0.50)
        w = (
            0.34 * np.sin(2 * math.pi * 880.0 * t)
            + 0.26 * np.sin(2 * math.pi * 1174.66 * t)
            + 0.22 * np.sin(2 * math.pi * 1479.98 * t)
            + 0.16 * np.sin(2 * math.pi * 2349.32 * t) * (0.6 + 0.4 * np.sin(2 * math.pi * 6.5 * t))
        )
        air = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (3200.0, 8200.0), order=2) * 0.12
        return pan_sweep_stereo((w + air) * env * 0.54, pan - 0.22, pan + 0.28)

    if kind == "whip_whoosh":
        # Fast cinematic camera whip-pan whoosh with directional stereo travel & sub-air body
        n = int(0.36 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        noise = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (240.0, 3100.0), order=2)
        env = np.maximum(0.0, np.sin(math.pi * (t / 0.36))) ** 1.45
        f_sw = 165.0 + 560.0 * (t / 0.36)
        tone = np.sin(2 * math.pi * np.cumsum(f_sw) / SR) * 0.16
        return pan_sweep_stereo((noise * 0.46 + tone) * env * 0.74, pan * 0.28, np.clip(pan * 1.45, -0.82, 0.82))

    if kind == "parchment_flip":
        # Authentic crisp 17th-century vellum/parchment page turn + gentle folio settle
        n = int(0.38 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        paper1 = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1400.0, 6400.0), order=2)
        env1 = np.exp(-((t - 0.09) / 0.045) ** 2) + 0.65 * np.exp(-((t - 0.21) / 0.055) ** 2)
        thump = np.sin(2 * math.pi * 165.0 * t) * np.exp(-((t - 0.24) / 0.025) ** 2) * 0.28
        return pan_stereo((paper1 * env1 * 0.48 + thump) * 0.84, pan)

    if kind == "formula_gold_chime":
        # Radiant golden bell-harp chord as Universal Gravity F = G m1 m2 / r^2 illuminates
        n = int(1.30 * SR)
        freqs = [587.33, 739.99, 880.00, 1174.66, 1479.98]
        out = np.zeros(n, dtype=np.float32)
        for i, f in enumerate(freqs):
            delay = int(i * 0.032 * SR)
            m = n - delay
            if m > 0:
                tt = np.arange(m, dtype=np.float32) / SR
                note = (
                    np.sin(2 * math.pi * f * tt)
                    + 0.28 * np.sin(2 * math.pi * 2.0 * f * tt)
                ) * np.minimum(1.0, tt / 0.006) * np.exp(-tt * 3.1)
                out[delay:] += note * (0.34 / (1 + i * 0.22))
        return pan_sweep_stereo(out * 0.80, pan - 0.20, pan + 0.24)

    if kind == "gravity_bass_swell":
        # Deep cinematic gravitational well sub-swell (D1 -> D2 bloom with phone-speaker harmonics)
        n = int(1.45 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        env = adsr_env(n, attack=0.45, decay=0.25, sustain=0.65, release=0.55)
        f_sw = 55.0 + 18.42 * (t / 1.45)
        ph = 2 * math.pi * np.cumsum(f_sw) / SR
        w = 0.65 * np.sin(ph) + 0.36 * np.sin(2 * ph) + 0.18 * np.sin(3 * ph)
        return pan_stereo(butter_filter(w * env * 0.72, "low", 400.0, order=2), pan)

    if kind == "glass_prism_ring":
        # Authentic crystal flint-glass prism ring with singing glass-harmonica overtones
        n = int(1.15 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        modes = [(1760.0, 0.46, 3.6), (2845.0, 0.30, 5.2), (4410.0, 0.18, 7.8), (6720.0, 0.11, 13.0)]
        w = np.zeros(n, dtype=np.float32)
        for f, amp, dec in modes:
            w += amp * np.sin(2 * math.pi * f * t) * np.exp(-t * dec)
        tap = butter_filter(rng.standard_normal(n).astype(np.float32), "high", 3600.0, order=2) * np.exp(-t * 145.0) * 0.30
        return pan_stereo((w * np.minimum(1.0, t / 0.002) + tap) * 0.66, pan)

    if kind == "prism_rainbow_bloom":
        # V9 Lush Wide Stereo 7-Note Crystalline Glass-Harmonica + Concert Harp Shimmer!
        # Sweeps cleanly left-to-right (-0.78 -> +0.78) in exact sync with the 7 glowing spectral beams!
        n = int(1.90 * SR)
        st = np.zeros((n, CHANNELS), dtype=np.float32)
        rainbow_midis = [74, 76, 78, 80, 81, 83, 85]  # 7 radiant spectral notes (D5..C#6 Lydian)
        for k, m in enumerate(rainbow_midis):
            t_off = k * 0.082
            idx0 = int(t_off * SR)
            rem = n - idx0
            if rem > 0:
                tt = np.arange(rem, dtype=np.float32) / SR
                f = midi_to_hz(m)
                # Layer A: Singing crystal glass-harmonica (silky bowed attack + beating glass-rim modes)
                shimmer_lfo = 0.86 + 0.14 * np.sin(2.0 * math.pi * (5.2 + 0.35 * k) * tt)
                glass = (
                    0.56 * np.sin(2.0 * math.pi * f * tt) * shimmer_lfo * np.exp(-tt * 2.0)
                    + 0.22 * np.sin(2.0 * math.pi * 2.0 * f * tt) * np.exp(-tt * 3.6)
                    + 0.13 * np.sin(2.0 * math.pi * 2.756 * f * tt) * np.exp(-tt * 6.5)
                ) * np.minimum(1.0, tt / 0.032)
                # Layer B: Sparkling concert harp glissando pluck locked to each spectral ray
                harp = (
                    0.46 * np.sin(2.0 * math.pi * (f * 0.5) * tt) * np.exp(-tt * 3.2)
                    + 0.24 * np.sin(2.0 * math.pi * f * tt) * np.exp(-tt * 5.0)
                    + 0.12 * np.sin(2.0 * math.pi * 2.0 * f * tt) * np.exp(-tt * 8.5)
                ) * np.minimum(1.0, tt / 0.004)
                p_col = -0.78 + (k / 6.0) * 1.56  # Ultra-wide clean left-to-right spectral sweep!
                st[idx0:] += pan_stereo((glass * 0.28 + harp * 0.20), p_col)

        # Warm crystalline prism body halo (D4 + A4) + sweeping prismatic air sparkle
        t = np.arange(n, dtype=np.float32) / SR
        halo_env = adsr_env(n, attack=0.28, decay=0.30, sustain=0.55, release=0.75)
        halo = (0.20 * np.sin(2.0 * math.pi * 293.66 * t) + 0.15 * np.sin(2.0 * math.pi * 440.0 * t)) * halo_env
        air = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (3200.0, 10500.0), order=2)
        air_env = adsr_env(n, attack=0.30, decay=0.25, sustain=0.50, release=0.72)
        st += pan_sweep_stereo((air * air_env * 0.16 + halo * 0.45), -0.68, 0.68)
        return (st * 0.92).astype(np.float32)

    if kind == "spectral_crystal_shimmer":
        # Sparkling prismatic glass-harmonica & harp micro-chimes
        n = int(0.92 * SR)
        st = np.zeros((n, CHANNELS), dtype=np.float32)
        base_notes = [86, 90, 93, 98] if event_t < 13.5 else [88, 92, 95, 100]
        for k, m in enumerate(base_notes):
            idx0 = int(k * 0.062 * SR)
            rem = n - idx0
            if rem > 0:
                tt = np.arange(rem, dtype=np.float32) / SR
                f = midi_to_hz(m)
                g = (
                    0.72 * np.sin(2 * math.pi * f * tt) * np.exp(-tt * 4.4)
                    + 0.24 * np.sin(2 * math.pi * 2.0 * f * tt) * np.exp(-tt * 7.8)
                ) * np.minimum(1.0, tt / 0.006)
                st[idx0:] += pan_stereo(g * 0.21, np.clip(pan + (k - 1.5) * 0.22, -0.82, 0.82))
        return (st * 0.84).astype(np.float32)

    if kind == "brass_pendulum_pull":
        # Taut metallic wire & brass pivot draw-back building comedic anticipation
        n = int(0.72 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        f_wire = 420.0 + 360.0 * (t / 0.72) + 18.0 * np.sin(2 * math.pi * 16.0 * t)
        ph = 2 * math.pi * np.cumsum(f_wire) / SR
        creak = (0.55 * np.sin(ph) + 0.35 * np.sin(2.45 * ph)) * adsr_env(n, 0.12, 0.12, 0.72, 0.14)
        satin = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1800.0, 5200.0), order=2) * adsr_env(n, 0.15, 0.1, 0.5, 0.2) * 0.12
        return pan_stereo(butter_filter((creak * 0.32 + satin), "band", (350.0, 4800.0), order=2) * 0.74, pan)

    if kind == "cradle_steel_clack":
        # V9 Hyper-Realistic Hardened-Steel Ball-Bearing CLACK!
        # Exact high-Q metallic ringing modes (2.8 kHz, 5.4 kHz, 8.9 kHz) + punchy 120 Hz kinetic transient!
        if event_t < 17.8:
            vel_scale, detune, alt_pan = 1.00, 1.000, pan
        elif event_t < 18.7:
            vel_scale, detune, alt_pan = 0.85, 1.007, pan - 0.04  # Clean alternating return clack @ 18.25s
        else:
            vel_scale, detune, alt_pan = 0.72, 0.994, pan + 0.03  # Alternating rebound clack @ 19.10s

        n = int(0.32 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        # 1. Razor-sharp Hertzian hardened-steel micro-contact impulse (< 1 ms burst)
        imp = (
            butter_filter(rng.standard_normal(n).astype(np.float32), "band", (2600.0, 11800.0), order=3)
            * np.exp(-t * 340.0)
            * 1.25
        )
        # 2. Crisp high-Q hardened chrome-steel ball-bearing eigenmodes: 2.8 kHz, 5.4 kHz, 8.9 kHz (+ 12.2 kHz spark)
        steel = (
            0.60 * np.sin(2.0 * math.pi * (2800.0 * detune) * t) * np.exp(-t * 36.0)
            + 0.48 * np.sin(2.0 * math.pi * (5400.0 * detune) * t) * np.exp(-t * 50.0)
            + 0.36 * np.sin(2.0 * math.pi * (8900.0 * detune) * t) * np.exp(-t * 72.0)
            + 0.16 * np.sin(2.0 * math.pi * (12200.0 * detune) * t) * np.exp(-t * 115.0)
        )
        # 3. Punchy 120 Hz kinetic impact transient + solid brass-frame transfer knock (cuts on phone speakers!)
        kinetic_120 = (
            0.72 * np.sin(2.0 * math.pi * (120.0 + 38.0 * np.exp(-t * 110.0)) * t) * np.exp(-t * 32.0)
            + 0.32 * np.sin(2.0 * math.pi * 245.0 * t) * np.exp(-t * 68.0)
        )
        attack = np.minimum(1.0, t / 0.0005)
        return pan_stereo(((imp + steel + kinetic_120) * attack) * 0.96 * vel_scale, alt_pan)

    if kind == "kinetic_whoosh_launch":
        # Snappy outward ballistic steel-ball launch whoosh right after impact!
        n = int(0.36 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        noise = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (420.0, 4600.0), order=2)
        f_up = 340.0 + 1220.0 * (1.0 - np.exp(-t * 9.5))
        tone = np.sin(2 * math.pi * np.cumsum(f_up) / SR) * 0.22
        env = np.minimum(1.0, t / 0.015) * np.exp(-t * 7.2)
        return pan_sweep_stereo((noise * 0.50 + tone) * env * 0.84, pan - 0.12, min(0.86, pan + 0.38))

    if kind == "orbital_well_bloom":
        # V9 Deep Cinematic IMAX Sub-Bass Cosmic Swell + Space-Time Harmonic Bloom!
        n = int(2.05 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        env = adsr_env(n, attack=0.38, decay=0.32, sustain=0.72, release=0.78)
        # Deep IMAX D1 (36.7Hz) + D2 (73.4Hz) sub-bass + phone-speaker A2/D3 harmonic weight
        sub = (
            0.52 * np.sin(2.0 * math.pi * 36.71 * t)
            + 0.64 * np.sin(2.0 * math.pi * 73.42 * t)
            + 0.30 * np.sin(2.0 * math.pi * 110.00 * t)
            + 0.28 * np.sin(2.0 * math.pi * 146.83 * t)
        ) * env
        harm = (
            0.28 * np.sin(2.0 * math.pi * 293.66 * t)
            + 0.24 * np.sin(2.0 * math.pi * 440.00 * t)
            + 0.20 * np.sin(2.0 * math.pi * 587.33 * t)
            + 0.14 * np.sin(2.0 * math.pi * 880.00 * t)
        ) * adsr_env(n, attack=0.46, decay=0.28, sustain=0.66, release=0.80)
        swirl = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (550.0, 3200.0), order=2)
        swirl *= (np.maximum(0.0, np.sin(math.pi * (t / 2.05))) ** 1.35) * 0.18
        st = pan_sweep_stereo((harm * 0.56 + swirl), -0.48, 0.48) + pan_stereo(sub * 0.66, 0.0)
        return (st * 0.80).astype(np.float32)

    if kind == "cosmic_choir_swell":
        # Ethereal deep-space angelic harmonic swell
        n = int(1.80 * SR)
        choir = synth_cosmic_choir([midi_to_hz(m) for m in (62, 69, 74, 81)], 1.80, vel=0.55)
        t = np.arange(n, dtype=np.float32) / SR
        halo = np.sin(2 * math.pi * 1760.0 * t) * adsr_env(n, 0.55, 0.25, 0.50, 0.68) * 0.09
        return (choir + pan_stereo(halo, pan)).astype(np.float32)

    if kind == "rocket_ignition_whoosh":
        # V9 Thrilling Modern Space-Launch Rocket Thruster Ignition & Supersonic Flyby Whoosh!
        n = int(1.75 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        env = adsr_env(n, attack=0.14, decay=0.26, sustain=0.72, release=0.68)
        raw_n = rng.standard_normal(n).astype(np.float32)
        # Crisp ignition spark crackle transient (0..90ms)
        crackle = (
            butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1800.0, 7600.0), order=2)
            * np.exp(-t * 32.0)
            * (0.65 + 0.35 * np.sin(2.0 * math.pi * 95.0 * t))
            * 0.55
        )
        # Deep roaring rocket thruster combustion rumble (48-240 Hz with organic combustion flutter)
        flutter = 0.82 + 0.18 * np.sin(2.0 * math.pi * 24.0 * t) * np.cos(2.0 * math.pi * 11.0 * t)
        rumble = butter_filter(raw_n, "band", (48.0, 240.0), order=3) * flutter * 1.32
        sub_thrust = np.sin(2.0 * math.pi * (62.0 + 28.0 * (t / 1.75)) * t) * 0.36
        # Fiery supersonic exhaust plume sweeping wide across stereo
        plume = butter_filter(raw_n, "band", (420.0, 4800.0), order=2) * 0.64
        jet_pitch = 170.0 + 610.0 * ((t / 1.75) ** 1.25)
        jet = np.sin(2.0 * math.pi * np.cumsum(jet_pitch) / SR) * 0.15
        st = (
            pan_sweep_stereo((plume + jet + crackle) * env, -0.52, min(0.84, pan + 0.60))
            + pan_sweep_stereo((rumble + sub_thrust) * env, -0.20, 0.25)
        )
        return (st * 0.74).astype(np.float32)

    if kind == "orbital_triumph_chime":
        # Glorious crowning harmonic bell-and-celesta chord for Newton's legacy
        n = int(1.90 * SR)
        st = np.zeros((n, CHANNELS), dtype=np.float32)
        triumph = [(74, -0.40), (81, -0.18), (86, 0.0), (90, 0.20), (93, 0.42)]
        for k, (m, p) in enumerate(triumph):
            idx0 = int(k * 0.052 * SR)
            rem = n - idx0
            if rem > 0:
                tt = np.arange(rem, dtype=np.float32) / SR
                f = midi_to_hz(m)
                bell = (
                    0.58 * np.sin(2 * math.pi * f * tt)
                    + 0.25 * np.sin(2 * math.pi * 2.0 * f * tt) * np.exp(-tt * 3.8)
                    + 0.12 * np.sin(2 * math.pi * 3.0 * f * tt) * np.exp(-tt * 6.5)
                ) * np.minimum(1.0, tt / 0.010) * np.exp(-tt * 2.0)
                st[idx0:] += pan_stereo(bell * 0.29, p)
        return (st * 0.88).astype(np.float32)

    raise ValueError(f"Unhandled SFX type in timeline.json: {kind}")


# ============================================================================
# BROADCAST MASTERING: ITU-R BS.1770-4 (-14.0 LUFS) + 4x TRUE-PEAK LIMITER
# ============================================================================
def measure_lufs_48k(stereo: np.ndarray) -> float:
    """Exact ITU-R BS.1770-4 K-weighted integrated loudness (LUFS) at 48 kHz."""
    # Stage 1: High-shelf pre-filter (+4 dB @ 1681.97 Hz, exact 48 kHz coefficients from BS.1770-4)
    b1 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285], dtype=np.float64)
    a1 = np.array([1.0, -1.69065929318241, 0.73248077421585], dtype=np.float64)
    # Stage 2: RLB high-pass filter (38.13 Hz, exact 48 kHz coefficients from BS.1770-4)
    b2 = np.array([1.0, -2.0, 1.0], dtype=np.float64)
    a2 = np.array([1.0, -1.99004745483398, 0.99007225036621], dtype=np.float64)

    y = signal.lfilter(b1, a1, stereo.astype(np.float64), axis=0)
    y = signal.lfilter(b2, a2, y, axis=0)

    block_len = int(0.400 * SR)  # 400 ms gating block
    hop_len = int(0.100 * SR)    # 75% overlap (100 ms step)
    powers = []
    for start in range(0, len(y) - block_len + 1, hop_len):
        blk = y[start : start + block_len]
        # Stereo channel weights: L=1.0, R=1.0
        z = float(np.mean(blk[:, 0] ** 2) + np.mean(blk[:, 1] ** 2))
        powers.append(z)
    if not powers:
        return -70.0
    p_arr = np.array(powers, dtype=np.float64)
    l_blocks = -0.691 + 10.0 * np.log10(np.maximum(p_arr, 1e-12))

    # Absolute gate: > -70 LUFS
    abs_mask = l_blocks > -70.0
    if not np.any(abs_mask):
        return -70.0
    ungated_mean = float(np.mean(p_arr[abs_mask]))
    rel_thresh = -0.691 + 10.0 * math.log10(max(ungated_mean, 1e-12)) - 10.0

    # Relative gate: > rel_thresh
    rel_mask = abs_mask & (l_blocks > rel_thresh)
    if not np.any(rel_mask):
        return -70.0
    gated_mean = float(np.mean(p_arr[rel_mask]))
    return -0.691 + 10.0 * math.log10(max(gated_mean, 1e-12))


def brickwall_true_peak_limiter(stereo: np.ndarray, ceiling: float = TRUE_PEAK_CEILING_LIN) -> np.ndarray:
    """Clean lookahead brickwall true-peak limiter with 4x inter-sample peak verification."""
    # Compute 4x oversampled inter-sample peak envelope per 48 kHz sample
    up = signal.resample_poly(stereo.astype(np.float64), 4, 1, axis=0)
    up_abs = np.max(np.abs(up), axis=1)
    # Max peak within each original sample's 4 sub-samples
    n_orig = len(stereo)
    tp_per_samp = np.max(up_abs[: n_orig * 4].reshape(n_orig, 4), axis=1).astype(np.float32)

    # Lookahead window (5 ms) maximum detector
    look = int(round(0.005 * SR))
    from scipy.ndimage import maximum_filter1d
    peak_env = maximum_filter1d(tp_per_samp, size=2 * look + 1, mode="nearest")

    # Smooth release envelope (45 ms)
    rel_coeff = math.exp(-1.0 / (0.045 * SR))
    gain_req = np.minimum(1.0, ceiling / np.maximum(peak_env, 1e-9)).astype(np.float32)
    gain = np.ones(n_orig, dtype=np.float32)
    g = 1.0
    for i in range(n_orig):
        gr = float(gain_req[i])
        if gr < g:
            g = gr  # Instant brickwall attack on lookahead window
        else:
            g = rel_coeff * g + (1.0 - rel_coeff) * gr
        gain[i] = g

    # Gentle attack smoothing (1.5 ms) centered on lookahead so transients stay distortion-free
    sm_len = int(round(0.0015 * SR))
    if sm_len >= 3:
        kern = np.hanning(sm_len).astype(np.float32)
        kern /= kern.sum()
        gain = np.minimum(gain, np.convolve(gain, kern, mode="same"))

    limited = (stereo * gain[:, None]).astype(np.float32)

    # Final safety true-peak ceiling verification @ 4x oversampling
    up_check = signal.resample_poly(limited.astype(np.float64), 4, 1, axis=0)
    max_tp = float(np.max(np.abs(up_check)))
    if max_tp > ceiling:
        limited *= float(ceiling / max_tp)
    return limited


def sidechain_duck_envelope(dialogue_bus: np.ndarray, duck_db: float = -5.5) -> np.ndarray:
    """Smooth studio sidechain envelope reducing BGM by duck_db (-5.5 dB) under speech."""
    speech_rms = np.sqrt(np.mean(dialogue_bus ** 2, axis=1) + 1e-12).astype(np.float32)
    # 60 ms RMS window
    win = int(round(0.060 * SR))
    kernel = np.ones(win, dtype=np.float32) / win
    smooth = np.convolve(speech_rms, kernel, mode="same")

    # Asymmetric attack (25 ms) / release (190 ms) follower for zero pumping
    att = math.exp(-1.0 / (0.025 * SR))
    rel = math.exp(-1.0 / (0.190 * SR))
    env = np.zeros_like(smooth)
    e = 0.0
    for i in range(len(smooth)):
        x = float(smooth[i])
        if x > e:
            e = att * e + (1.0 - att) * x
        else:
            e = rel * e + (1.0 - rel) * x
        env[i] = e

    duck_depth = 1.0 - (10.0 ** (duck_db / 20.0))  # ~0.469 for -5.5 dB
    amount = np.clip((env - 0.008) / 0.042, 0.0, 1.0)
    return (1.0 - duck_depth * amount).astype(np.float32)


def master_and_export(
    timeline: dict[str, Any],
    dialogue_bus: np.ndarray,
    bgm_bus: np.ndarray,
    sfx_bus: np.ndarray,
    ambience_bus: np.ndarray,
    speaker_mono_buses: dict[str, np.ndarray],
    manifest_lines: list[dict[str, Any]],
) -> dict[str, Any]:
    # 1. Smooth -5.5 dB sidechain BGM ducking + gentle -2.5 dB ambience ducking under dialogue
    duck_bgm = sidechain_duck_envelope(dialogue_bus, duck_db=-5.5)
    duck_amb = sidechain_duck_envelope(dialogue_bus, duck_db=-2.5)
    bgm_ducked = bgm_bus * duck_bgm[:, None]
    amb_ducked = ambience_bus * duck_amb[:, None]

    # 2. Balanced studio stem summing (crystal-clear British dialogue + lush orchestral bed + punchy IMAX Foley)
    core_mix = dialogue_bus * 1.10 + bgm_ducked * 0.62 + sfx_bus * 0.90
    core_lufs = measure_lufs_48k(core_mix)
    est_master_gain = 10.0 ** ((TARGET_LUFS - core_lufs) / 20.0)

    # Calibrate subtle living outdoor ambience bed to hit exact -34.0 dBFS RMS in the finished master!
    amb_rms = float(np.sqrt(np.mean(amb_ducked[: int(29.0 * SR)] ** 2) + 1e-12))
    target_amb_lin = 10.0 ** (AMBIENCE_TARGET_DBFS / 20.0)
    amb_scale = (target_amb_lin / max(1e-6, amb_rms)) / max(1e-4, est_master_gain)
    mix = core_mix + amb_ducked * float(amb_scale)
    mix = butter_filter(mix, "high", 28.0, order=2)  # Tight sub-rumble protection

    # 3. Clean 2-pass ITU-R BS.1770-4 (-14.0 LUFS) + 4x True-Peak Brickwall Limiter (<= -1.5 dBFS)
    total_gain = 1.0
    for _ in range(2):
        cur_lufs = measure_lufs_48k(mix)
        gain_db = TARGET_LUFS - cur_lufs
        g_lin = 10.0 ** (gain_db / 20.0)
        total_gain *= g_lin
        mix = (mix * g_lin).astype(np.float32)
        mix = brickwall_true_peak_limiter(mix, ceiling=TRUE_PEAK_CEILING_LIN)

    # Exact 1,440,000 samples safety check + finite guard + tiny micro-fades on endpoints for loop cleanliness
    assert mix.shape == (TOTAL_SAMPLES, CHANNELS), f"Unexpected mix shape: {mix.shape}"
    assert np.all(np.isfinite(mix)), "Non-finite (NaN/Inf) samples detected in master mix!"
    micro = int(round(0.004 * SR))
    mix[:micro] *= np.linspace(0.0, 1.0, micro, dtype=np.float32)[:, None]
    mix[-micro:] *= np.linspace(1.0, 0.0, micro, dtype=np.float32)[:, None]

    final_lufs = measure_lufs_48k(mix)
    final_sample_peak_db = 20.0 * math.log10(max(1e-9, float(np.max(np.abs(mix)))))
    up_tp = signal.resample_poly(mix.astype(np.float64), 4, 1, axis=0)
    final_true_peak_db = 20.0 * math.log10(max(1e-9, float(np.max(np.abs(up_tp)))))
    final_amb_db = 20.0 * math.log10(max(1e-9, amb_rms * amb_scale * total_gain))

    master_path = PROJECT_ROOT / timeline["audio"]["masterPath"]
    manifest_path = PROJECT_ROOT / timeline["audio"]["manifestPath"]
    srt_path = PROJECT_ROOT / timeline["audio"]["srtPath"]
    master_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(str(master_path), mix, SR, subtype="PCM_16")

    lip_map, narrow_map = compute_lip_sync(speaker_mono_buses, LIP_FPS, LIP_TOTAL_FRAMES)
    manifest = {
        "title": timeline.get("title", ""),
        "duration": DURATION,
        "fps": LIP_FPS,
        "totalFrames": LIP_TOTAL_FRAMES,
        "sampleRate": SR,
        "totalSamples": TOTAL_SAMPLES,
        "integratedLufs": round(final_lufs, 3),
        "samplePeakDbfs": round(final_sample_peak_db, 3),
        "truePeakDbfs": round(final_true_peak_db, 3),
        "peakDbfs": round(final_sample_peak_db, 3),
        "ambienceBedDbfs": round(final_amb_db, 2),
        "lines": manifest_lines,
        "lipSync": lip_map,
        "lipNarrow": narrow_map,
        "sfxEvents": timeline.get("sfxEvents", []),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_srt(manifest_lines, srt_path)
    print(
        f"[Master] {master_path.name}: {TOTAL_SAMPLES:,} samples @ {SR}Hz stereo (30.000s) | "
        f"LUFS={final_lufs:.2f} | SamplePeak={final_sample_peak_db:.2f} dBFS | "
        f"TruePeak={final_true_peak_db:.2f} dBFS | AmbienceBed={final_amb_db:.1f} dBFS"
    )
    return manifest


# ============================================================================
# EXACT WORD-SYNCED BOXLESS KARAOKE SUBTITLES (ASS + SRT)
# ============================================================================
HANGING_WORDS = {
    "a", "an", "the", "of", "on", "in", "to", "with", "and", "or", "as", "so",
    "its", "his", "my", "our", "your", "through", "into", "under", "for", "by", "at", "seven",
}


def join_tokens(tokens: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for i, tok in enumerate(tokens):
        w = tok["word"]
        if i == 0:
            parts.append(w)
        else:
            sep = "" if (tokens[i - 1]["word"].endswith("—") or w.startswith("—")) else " "
            parts.append(sep + w)
    return "".join(parts)


def split_clause_balanced(clause: list[dict[str, Any]], max_words: int = 7, max_chars: int = 44) -> list[list[dict[str, Any]]]:
    """Split a clause into balanced cards (2..max_words words, <=max_chars chars) avoiding hanging words & 1-word orphans."""
    text = join_tokens(clause)
    if len(clause) <= max_words and len(text) <= max_chars:
        return [clause]

    # Choose best split point k in [2 .. len(clause) - 2] so neither half is a 1-word orphan
    best_k = None
    best_score = 1e9
    n = len(clause)
    for k in range(2, n - 1):
        left = clause[:k]
        right = clause[k:]
        lt = join_tokens(left)
        rt = join_tokens(right)
        score = abs(len(lt) - len(rt)) * 1.0
        # Strong preference for keeping left card within limits
        if len(left) > max_words:
            score += (len(left) - max_words) * 40.0
        if len(lt) > max_chars:
            score += (len(lt) - max_chars) * 6.0
        # Penalize ending left half on a hanging article/preposition/determiner
        last_l = left[-1]["word"].lower().strip(".,!?—':;\"")
        if last_l in HANGING_WORDS:
            score += 35.0
        # Reward splitting right after a comma or clause punctuation
        if left[-1]["word"].endswith((",", "—", ";", ":")):
            score -= 18.0
        if score < best_score:
            best_score = score
            best_k = k

    if best_k is None:
        best_k = max(2, min(n - 2, max_words))
    return split_clause_balanced(clause[:best_k], max_words, max_chars) + split_clause_balanced(clause[best_k:], max_words, max_chars)


def chunk_words(words: list[dict[str, Any]], max_words: int = 7, max_chars: int = 44) -> list[list[dict[str, Any]]]:
    """Clause-aware balanced subtitle chunker: <=7 words, <=44 chars, zero 1-word orphan cards."""
    # First segment at strong clause boundaries (em-dash, sentence enders, or commas when chunk >= 2 words)
    clauses: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    for idx, w in enumerate(words):
        cur.append(w)
        wt = w["word"]
        rem = len(words) - 1 - idx
        is_strong_end = wt.endswith(("—", ".", "!", "?", ";"))
        is_comma_end = wt.endswith(",") and len(cur) >= 3 and len(join_tokens(cur)) >= 20
        if (is_strong_end or is_comma_end) and len(cur) >= 2 and rem != 1:
            clauses.append(cur)
            cur = []
    if cur:
        if len(cur) == 1 and clauses:
            clauses[-1].extend(cur)
        else:
            clauses.append(cur)

    # Balance any long clause into clean sub-cards
    cards: list[list[dict[str, Any]]] = []
    for cl in clauses:
        cards.extend(split_clause_balanced(cl, max_words=max_words, max_chars=max_chars))

    # Guarantee zero 1-word orphan cards anywhere
    merged: list[list[dict[str, Any]]] = []
    for card in cards:
        if len(card) == 1 and merged and len(merged[-1]) < max_words:
            merged[-1].extend(card)
        else:
            merged.append(card)
    return merged


def choose_line_break(ch: list[dict[str, Any]], max_single_line_chars: int = 23) -> int:
    """Return index where '\\N' goes (or len(ch) for clean 1-line card). Prevents hanging words & 1-word line-2 orphans."""
    full = join_tokens(ch)
    n = len(ch)
    if n <= 2 or (len(full) <= max_single_line_chars and n <= 4):
        return n

    best_k = n
    best_score = 1e9
    for k in range(1, n):
        l1 = join_tokens(ch[:k])
        l2 = join_tokens(ch[k:])
        # Minimize imbalance while keeping both lines well inside 920px safe width (~24 chars @ 66pt bold)
        score = abs(len(l1) - len(l2)) * 1.2 + max(0, len(l1) - 23) * 15.0 + max(0, len(l2) - 23) * 15.0
        # Avoid 1-word line-2 orphan whenever n >= 4
        if n - k == 1 and n >= 4:
            score += 24.0
        if k == 1 and n >= 4:
            score += 18.0
        last_w = ch[k - 1]["word"].lower().strip(".,!?—':;\"")
        if last_w in HANGING_WORDS:
            score += 30.0
        if ch[k - 1]["word"].endswith((",", "—", "!", "?")):
            score -= 12.0
        if score < best_score:
            best_score = score
            best_k = k
    return best_k


def fmt_ass_time(sec: float) -> str:
    cs = int(round(max(0.0, sec) * 100))
    h = cs // 360000
    m = (cs % 360000) // 6000
    s = (cs % 6000) // 100
    rem = cs % 100
    return f"{h}:{m:02d}:{s:02d}.{rem:02d}"


def write_srt(manifest_lines: list[dict[str, Any]], srt_path: Path) -> None:
    def fmt_ts(sec: float) -> str:
        ms = int(round(max(0.0, sec) * 1000))
        h = ms // 3600000
        m = (ms % 3600000) // 60000
        s = (ms % 60000) // 1000
        rem = ms % 1000
        return f"{h:02d}:{m:02d}:{s:02d},{rem:03d}"

    blocks = []
    idx = 1
    prev_end_ms = 0
    for line in manifest_lines:
        chunks = chunk_words(line["words"], max_words=7, max_chars=44)
        reconstructed = join_tokens([w for ch in chunks for w in ch])
        assert reconstructed == line["text"], f"SRT verbatim mismatch: {reconstructed!r} != {line['text']!r}"
        for ch in chunks:
            assert 2 <= len(ch) <= 7, f"SRT orphan/overflow card ({len(ch)} words): {join_tokens(ch)}"
            t0_ms = int(round(max(0.0, ch[0]["start"]) * 1000))
            t1_ms = int(round(max(0.0, ch[-1]["end"]) * 1000))
            if t0_ms < prev_end_ms:
                t0_ms = prev_end_ms
            assert t1_ms > t0_ms, f"Invalid SRT card timing: {t0_ms}..{t1_ms}"
            prev_end_ms = t1_ms
            split_at = choose_line_break(ch)
            if split_at < len(ch):
                txt = join_tokens(ch[:split_at]) + "\n" + join_tokens(ch[split_at:])
            else:
                txt = join_tokens(ch)
            blocks.append(f"{idx}\n{fmt_ts(t0_ms / 1000.0)} --> {fmt_ts(t1_ms / 1000.0)}\n{txt}\n")
            idx += 1
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.write_text("\n".join(blocks), encoding="utf-8")


def write_word_karaoke_ass(manifest_lines: list[dict[str, Any]], ass_path: Path) -> None:
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,66,&H00FFFFFF,&H00F8BD38,&H00120803,&H96000000,-1,0,0,0,100,100,0,0,1,8,3.0,2,80,80,275,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    raw_events: list[tuple[int, int, str]] = []
    prev_end_cs = 0

    for line in manifest_lines:
        chunks = chunk_words(line["words"], max_words=7, max_chars=44)
        reconstructed = join_tokens([w for ch in chunks for w in ch])
        assert reconstructed == line["text"], f"ASS verbatim mismatch: {reconstructed!r} != {line['text']!r}"
        for ch in chunks:
            card_text = join_tokens(ch)
            assert 2 <= len(ch) <= 7, f"Subtitle card word count violation ({len(ch)}): {card_text}"
            assert len(card_text) <= 44, f"Subtitle card char limit violation ({len(card_text)}): {card_text}"
            split_at = choose_line_break(ch)

            for idx, w in enumerate(ch):
                t0_cs = int(round(max(0.0, w["start"] if idx > 0 else ch[0]["start"]) * 100))
                t1_cs = int(round(max(0.0, ch[idx + 1]["start"] if idx + 1 < len(ch) else w["end"]) * 100))
                if t0_cs < prev_end_cs:
                    t0_cs = prev_end_cs
                if t1_cs <= t0_cs:
                    t1_cs = t0_cs + 3  # Minimum 30ms highlight visibility
                prev_end_cs = t1_cs

                parts = []
                for j, cw in enumerate(ch):
                    token = cw["word"]
                    styled = f"{{\\c&H00F8BD38&}}{token}{{\\c&H00FFFFFF&}}" if j == idx else token
                    if j == split_at:
                        parts.append("\\N" + styled)
                    elif j == 0:
                        parts.append(styled)
                    else:
                        sep = "" if (ch[j - 1]["word"].endswith("—") or token.startswith("—")) else " "
                        parts.append(sep + styled)
                raw_events.append((t0_cs, t1_cs, "".join(parts)))

    # Strict zero-overlap verification in centiseconds
    for i in range(1, len(raw_events)):
        assert raw_events[i][0] >= raw_events[i - 1][1], (
            f"ASS overlap detected at event {i}: {raw_events[i - 1][1]} > {raw_events[i][0]}"
        )

    event_lines = [
        f"Dialogue: 0,{fmt_ass_time(t0 / 100.0)},{fmt_ass_time(t1 / 100.0)},Default,,0,0,0,,{txt}"
        for t0, t1, txt in raw_events
    ]
    ass_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path.write_text(header + "\n".join(event_lines) + "\n", encoding="utf-8")
    print(f"[Subtitles] Exported word-synced boxless ASS: {ass_path} ({len(event_lines)} highlight events, 0 overlaps, 100% verbatim)")


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main() -> None:
    raw_tl = json.loads(TIMELINE_PATH.read_text(encoding="utf-8"))
    tl = json.loads(json.dumps(raw_tl))
    tl["sfxEvents"] = [{**ev, "time": float(ev["t"])} for ev in tl.get("sfx", [])]

    print("[1/5] Synthesizing Kokoro ONNX British multi-voice dialogue (V9 Broadcast Presence + Outdoor Courtyard)...")
    dialogue_bus, speaker_mono_buses, manifest_lines = synthesize_dialogue(tl)

    print("[2/5] Composing 112 BPM Cambridge Celestial Baroque & Cosmic Orchestral Score...")
    bgm_bus = compose_newton_score()

    print("[3/5] Synthesizing Living Outdoor Ambience Bed (Golden-Hour Meadow & Birdsong -> t=20s -> Night Crickets & Cosmic Hum)...")
    ambience_bus = compose_living_ambience_bed()

    print(f"[4/5] Synthesizing {len(tl['sfxEvents'])} V9 phone-speaker-tested spatial Foley SFX...")
    sfx_bus = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)
    for ev in tl["sfxEvents"]:
        add_stereo_event(
            sfx_bus,
            synth_sfx(ev["type"], float(ev.get("pan", 0.0)), float(ev["time"])),
            float(ev["time"]),
        )
    # Crisp airy outdoor stone-courtyard cohesion on Foley (preserves razor-sharp metallic transients!)
    sfx_bus = add_outdoor_courtyard_reverb(sfx_bus, mix=0.08)

    print("[5/5] Mastering 30.000s 48kHz stereo WAV (-14 LUFS, True Peak <= -1.5 dBFS) + Boxless ASS/SRT...")
    master_and_export(tl, dialogue_bus, bgm_bus, sfx_bus, ambience_bus, speaker_mono_buses, manifest_lines)
    write_word_karaoke_ass(manifest_lines, PROJECT_ROOT / raw_tl["audio"]["assPath"])
    print("DONE! Isaac Newton V9 Theatrical IMAX Master Audio, 30Hz Dual-Axis Lip-Sync & Boxless Karaoke Subtitles ready.")


if __name__ == "__main__":
    main()
