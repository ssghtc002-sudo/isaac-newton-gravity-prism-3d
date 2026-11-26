"""
How a Grenade Actually Works — 4K Hollywood Action + Slow-Mo X-Ray Studio Audio (AGENTS.md V10)
  - Studio Microsoft Neural Narration (en-US-BrianNeural — razor-crisp viral documentary pacing, zero truncation)
  - 124 BPM Tactical Action & Matrix Bullet-Time X-Ray Score (vocal-pocket EQ 250-3500 Hz + -9.5 dB sidechain ducking)
  - 19 Tactile Battlefield & Mechanical Foley SFX:
      * B1: pin_pull_click (metallic tooth-tug & cotter pin pop), pin_ring_jingle (chrome ring jingle)
      * B2: hero_throw_whoosh (wind-up & throw), bullet_time_drop (Matrix freeze drop),
            xray_slice_whoosh (cyan X-ray laser sweep), spring_tension_creak (coiled steel tension)
      * B3: spoon_ping_fly (iconic lever PING!), striker_hammer_snap (steel slam), primer_spark_pop (brass primer crack)
      * B4: fuse_burn_sizzle (organic crackling 4s delay burn), 4x timer_tick (crisp countdown ticks)
      * B5: grenade_dirt_bounce (cast-iron thud/bounce on dusty earth), core_ignition_flash, pressure_swell_sub
      * B6: shell_fracture_slowmo (colossal bunker detonation + shrapnel zings + slow-mo bass boom), cinema_outro_hit
  - Broadcast Mastered: 48 kHz Stereo (30.000s), -14.0 LUFS, True Peak <= -1.5 dBFS, Boxless Word-Synced ASS/SRT
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TIMELINE_PATH = PROJECT_ROOT / "config" / "timeline.json"

SR = 48000
CHANNELS = 2
DURATION = 30.0
TOTAL_SAMPLES = int(round(DURATION * SR))

TARGET_LUFS = -14.0
TRUE_PEAK_CEILING_DBFS = -1.62
TRUE_PEAK_CEILING_LIN = 10.0 ** (TRUE_PEAK_CEILING_DBFS / 20.0)
BPM = 124.0
BEAT = 60.0 / BPM

# Natural comma phrasing for Edge-TTS (removes excessive em-dash / double-comma silence pauses while keeping exact script words)
TTS_OVERRIDES: dict[str, str] = {
    "L1": "In movies soldiers pull a grenade pin with their teeth, that would rip your teeth out and it doesn't even start the timer!",
    "L2": "Watch what happens the second he throws it, inside a coiled steel spring is locked down by this curved spoon.",
    "L3": "The instant it leaves his hand, the spoon flies off and a steel striker hammer slams into the brass primer!",
    "L4": "That spark lights a four-second delay fuse, burning steadily down the center as it flies toward the enemy bunker!",
    "L5": "The moment it lands by the sandbags, the flame hits the core and superheated gas swells to extreme pressure!",
    "L6": "In a millisecond, the pre-scored cast-iron shell shatters outward in a massive shockwave!",
}


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def pan_stereo(mono: np.ndarray, pan: float) -> np.ndarray:
    pan = float(np.clip(pan, -1.0, 1.0))
    angle = (pan + 1.0) * (math.pi / 4.0)
    return np.stack([mono * math.cos(angle), mono * math.sin(angle)], axis=-1).astype(np.float32)


def pan_sweep_stereo(mono: np.ndarray, p0: float, p1: float) -> np.ndarray:
    p = np.linspace(float(np.clip(p0, -1.0, 1.0)), float(np.clip(p1, -1.0, 1.0)), len(mono), dtype=np.float32)
    ang = (p + 1.0) * (math.pi / 4.0)
    return np.stack([mono * np.cos(ang), mono * np.sin(ang)], axis=-1).astype(np.float32)


def add_stereo_event(bus: np.ndarray, event: np.ndarray, start_time: float) -> None:
    if event.ndim == 1:
        event = pan_stereo(event, 0.0)
    idx = int(round(start_time * SR))
    if idx >= len(bus) or idx + len(event) <= 0:
        return
    src_s, dst_s = max(0, -idx), max(0, idx)
    clen = min(len(event) - src_s, len(bus) - dst_s)
    if clen > 0:
        bus[dst_s : dst_s + clen] += event[src_s : src_s + clen]


def butter_filter(audio: np.ndarray, btype: str, cutoff: float | tuple[float, float], order: int = 2) -> np.ndarray:
    nyq = 0.5 * SR
    wn = [float(np.clip(c / nyq, 1e-4, 0.999)) for c in cutoff] if isinstance(cutoff, tuple) else float(np.clip(cutoff / nyq, 1e-4, 0.999))
    return signal.sosfilt(signal.butter(order, wn, btype=btype, output="sos"), audio, axis=0).astype(np.float32)


def peaking_eq(mono: np.ndarray, freq: float, q: float, gain_db: float) -> np.ndarray:
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / SR
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    b = np.array([1.0 + alpha * a_lin, -2.0 * cw, 1.0 - alpha * a_lin], dtype=np.float64) / (1.0 + alpha / a_lin)
    a = np.array([1.0, (-2.0 * cw) / (1.0 + alpha / a_lin), (1.0 - alpha / a_lin) / (1.0 + alpha / a_lin)], dtype=np.float64)
    return signal.lfilter(b, a, mono.astype(np.float64), axis=0).astype(np.float32)


def smooth_envelope(sig: np.ndarray, attack_ms: float, release_ms: float) -> np.ndarray:
    x = np.max(np.abs(sig), axis=1) if sig.ndim == 2 else np.abs(sig)
    x = x.astype(np.float32)
    step = 64
    n_blk = (len(x) + step - 1) // step
    blk = np.array([float(np.max(x[i * step : min(len(x), (i + 1) * step)])) for i in range(n_blk)], dtype=np.float32)
    att = math.exp(-step / max(1.0, attack_ms * 0.001 * SR))
    rel = math.exp(-step / max(1.0, release_ms * 0.001 * SR))
    env = np.zeros(n_blk, dtype=np.float32)
    cur = 0.0
    for i in range(n_blk):
        v = float(blk[i])
        cur = (att * cur + (1.0 - att) * v) if v > cur else (rel * cur + (1.0 - rel) * v)
        env[i] = cur
    return np.interp(np.linspace(0, n_blk - 1, len(x), dtype=np.float32), np.arange(n_blk, dtype=np.float32), env).astype(np.float32)


def trim_silence(mono: np.ndarray, thresh_db: float = -42.0) -> np.ndarray:
    thr = float(np.max(np.abs(mono)) + 1e-12) * (10.0 ** (thresh_db / 20.0))
    idx = np.where(np.abs(mono) > thr)[0]
    if len(idx) == 0:
        return mono
    return mono[max(0, int(idx[0]) - int(0.012 * SR)) : min(len(mono), int(idx[-1]) + int(0.022 * SR))].astype(np.float32)


def broadcast_vocal_chain(mono: np.ndarray) -> np.ndarray:
    y = butter_filter(mono, "high", 85.0, order=2)
    y = peaking_eq(y, 260.0, 1.3, -1.6)
    y = peaking_eq(y, 3400.0, 1.0, +2.4)
    y = peaking_eq(y, 7000.0, 0.9, +1.5)
    env = smooth_envelope(y, 6.0, 95.0)
    thr, ratio = 0.14, 3.0
    gain = np.where(env > thr, (thr + (env - thr) / ratio) / np.maximum(env, 1e-6), 1.0)
    y = (y * gain).astype(np.float32)
    y *= 0.20 / max(1e-5, float(np.sqrt(np.mean(y ** 2) + 1e-12)))
    return np.tanh(y * 0.95).astype(np.float32)


def ffmpeg_atempo_48k(mono: np.ndarray, rate: float) -> np.ndarray:
    if abs(rate - 1.0) < 0.008:
        return mono.astype(np.float32)
    rate = float(np.clip(rate, 0.88, 1.12))
    with tempfile.TemporaryDirectory() as td:
        ip, op = Path(td) / "in.wav", Path(td) / "out.wav"
        sf.write(str(ip), mono.astype(np.float32), SR, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-y", "-i", str(ip), "-filter:a", f"atempo={rate:.4f}", "-ar", str(SR), "-ac", "1", str(op), "-loglevel", "error"], check=True)
        y, _ = sf.read(str(op), dtype="float32")
    return y.astype(np.float32)


async def _edge_tts_synth_async(text: str, voice: str, rate_pct: int, pitch: str) -> np.ndarray:
    import edge_tts

    comm = edge_tts.Communicate(text=text, voice=voice, rate=f"{rate_pct:+d}%", pitch=pitch)
    mp3 = bytearray()
    async for ch in comm.stream():
        if ch.get("type") == "audio":
            mp3.extend(ch["data"])
    with tempfile.TemporaryDirectory() as td:
        mp, wp = Path(td) / "v.mp3", Path(td) / "v.wav"
        mp.write_bytes(bytes(mp3))
        subprocess.run(["ffmpeg", "-y", "-i", str(mp), "-ar", str(SR), "-ac", "1", str(wp), "-loglevel", "error"], check=True)
        y, _ = sf.read(str(wp), dtype="float32")
    return y.astype(np.float32)


def tokenize_display_words(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in text.strip().split():
        if "—" in raw and not raw.endswith("—"):
            parts = raw.split("—")
            for i, p in enumerate(parts):
                if i < len(parts) - 1:
                    tokens.append(p + "—")
                elif p:
                    tokens.append(p)
        else:
            tokens.append(raw)
    return [t for t in tokens if t]


def align_words(words: list[str], t0: float, t1: float) -> list[dict[str, Any]]:
    weights = np.array(
        [max(1.8, float(len(re.sub(r"[^A-Za-z0-9]", "", w)))) + (1.5 if w.endswith(("!", "?", ".")) else (1.0 if w.endswith(("—", ",")) else 0.0)) for w in words],
        dtype=np.float64,
    )
    weights /= weights.sum()
    bounds = [t0]
    acc = 0.0
    for wt in weights[:-1]:
        acc += float(wt)
        bounds.append(t0 + (t1 - t0) * acc)
    bounds.append(t1)
    return [{"word": w, "start": round(bounds[i], 4), "end": round(bounds[i + 1], 4)} for i, w in enumerate(words)]


def synthesize_dialogue(tl: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    bus = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)
    lines: list[dict[str, Any]] = []
    voice = "en-US-BrianNeural"
    dlg_items = tl["dialogue"]
    for idx, item in enumerate(dlg_items):
        lid = item["id"]
        s0, s1 = float(item["start"]), float(item["end"])
        next_start = float(dlg_items[idx + 1]["start"]) if idx + 1 < len(dlg_items) else DURATION - 0.06
        slot = s1 - s0
        max_end = min(s1 + 0.04, next_start - 0.08, DURATION - 0.06)
        tts_txt = TTS_OVERRIDES.get(lid, item["text"].replace("—", ", "))
        rate_pct = 0
        comp = trim_silence(asyncio.run(_edge_tts_synth_async(tts_txt, voice, rate_pct, "-1Hz")))
        cdur = len(comp) / float(SR)
        if cdur > slot * 0.96:
            rate_pct = max(4, min(18, int(math.ceil((cdur / (slot * 0.93) - 1.0) * 100.0))))
            comp2 = trim_silence(asyncio.run(_edge_tts_synth_async(tts_txt, voice, rate_pct, "-1Hz")))
            if len(comp2) < len(comp):
                comp, cdur = comp2, len(comp2) / float(SR)
        atempo = float(np.clip(cdur / (slot * 0.96), 1.0, 1.12)) if cdur > slot * 0.98 else 1.0
        proc = broadcast_vocal_chain(ffmpeg_atempo_48k(comp, atempo))
        proc_dur = len(proc) / float(SR)
        if s0 + proc_dur > max_end:
            raise RuntimeError(
                f"{lid} overflowed window without truncation allowed: {s0 + proc_dur:.3f}s > {max_end:.3f}s "
                f"(rate={rate_pct:+d}%, atempo={atempo:.3f}x)"
            )
        t0, t1 = s0, s0 + proc_dur
        fn = min(int(0.012 * SR), len(proc) // 4)
        if fn > 0:
            proc[:fn] *= np.linspace(0.0, 1.0, fn, dtype=np.float32)
            proc[-fn:] *= np.linspace(1.0, 0.0, fn, dtype=np.float32)
        add_stereo_event(bus, pan_stereo(proc, 0.0), t0)
        lines.append({
            "id": lid,
            "beatId": item["beatId"],
            "speaker": "narrator",
            "text": item["text"],
            "start": round(t0, 4),
            "end": round(t1, 4),
            "ratePct": rate_pct,
            "atempo": round(atempo, 4),
            "words": align_words(tokenize_display_words(item["text"]), t0, t1),
        })
        print(
            f"  [EdgeTTS] {lid} (rate={rate_pct:+d}% atempo={atempo:.2f}x): "
            f"{t0:5.2f}s..{t1:5.2f}s / window={s0:.2f}..{s1:.2f}s (margin={s1 - t1:+.2f}s)"
        )
    return bus, lines


def compose_engineering_tension_score() -> np.ndarray:
    """High-retention 124 BPM Hollywood Action + Matrix Bullet-Time X-Ray score with 250-3500 Hz vocal-pocket EQ."""
    bus = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)
    rng = np.random.default_rng(194507)
    sections = [
        (0.0, 4.8, 38, "hook"),       # B1: D1 Hero Pin-Pull Hook
        (4.8, 6.0, 38, "throw"),      # B2a: Hero Throw build
        (6.0, 9.8, 38, "bullet_time"),# B2b: Matrix Bullet-Time Freeze + X-Ray Reveal
        (9.8, 14.8, 41, "striker"),   # B3: F1 Slow-Mo Spoon & Striker Slam
        (14.8, 20.6, 43, "fuse"),     # B4: G1 4-Second Delay Countdown Escalation
        (20.6, 25.46, 45, "pressure"),# B5: A1 Sandbag Landing & Extreme Pressure Riser (40ms breath before 25.52s blast!)
        (25.52, 29.95, 38, "finale"), # B6: D1 Colossal Bullet-Time Bunker Shockwave Finale
    ]
    for t0, t1, bass_m, mode in sections:
        dur = t1 - t0
        n = int(dur * SR)
        if n <= 0:
            continue
        t = np.arange(n, dtype=np.float32) / SR
        f_b = midi_to_hz(bass_m)
        env = np.minimum(1.0, t / 0.14) * np.minimum(1.0, (dur - t) / (0.04 if mode == "pressure" else 0.22))

        # Deep cinema sub-drone + warm fifth pad
        drone_amp = 0.22 if mode in ("pressure", "finale") else (0.14 if mode == "bullet_time" else 0.17)
        drone = (
            0.58 * np.sin(2.0 * math.pi * f_b * t)
            + 0.26 * np.sin(4.0 * math.pi * f_b * t)
            + 0.14 * np.sin(2.0 * math.pi * midi_to_hz(bass_m + 7) * t)
        ) * env * drone_amp
        if mode == "pressure":
            # Escalating intensity into detonation
            drone *= 0.72 + 0.58 * ((t / dur) ** 1.5)
        add_stereo_event(bus, pan_stereo(drone.astype(np.float32), 0.0), t0)

        # Ethereal wide sci-fi X-ray pad during Matrix bullet-time freeze (6.0-9.8s)
        if mode == "bullet_time":
            f_hi = midi_to_hz(bass_m + 24)
            f_5th = midi_to_hz(bass_m + 31)
            pad_l = (0.45 * np.sin(2.0 * math.pi * f_hi * t) + 0.30 * np.sin(2.0 * math.pi * (f_5th * 1.003) * t)) * env * 0.09
            pad_r = (0.45 * np.sin(2.0 * math.pi * (f_hi * 0.997) * t) + 0.30 * np.sin(2.0 * math.pi * f_5th * t)) * env * 0.09
            add_stereo_event(bus, np.stack([pad_l, pad_r], axis=-1).astype(np.float32), t0)

        # Cinematic dark brassy sustain in finale (B6)
        if mode == "finale":
            brass = (
                0.42 * np.sin(2.0 * math.pi * midi_to_hz(50) * t)
                + 0.28 * np.sin(2.0 * math.pi * midi_to_hz(57) * t)
                + 0.20 * np.sin(2.0 * math.pi * midi_to_hz(62) * t)
                + 0.14 * np.sin(2.0 * math.pi * midi_to_hz(65) * t)
            ) * np.minimum(1.0, t / 0.12) * np.exp(-t * 0.55) * 0.20
            add_stereo_event(bus, pan_stereo(brass.astype(np.float32), 0.0), t0)

        # Driving tactical pulse (half-time sonar pulses during bullet_time, escalating 8th/16th pulses in fuse & pressure)
        step = BEAT * (1.0 if mode == "bullet_time" else (0.5 if mode != "pressure" else 0.375))
        if mode == "finale":
            step = BEAT * 1.0
        st = t0
        k = 0
        while st < t1 - 0.12:
            pn = int(0.18 * SR)
            pt = np.arange(pn, dtype=np.float32) / SR
            m_note = bass_m + (12 if k % 4 == 0 else (19 if k % 4 == 2 else 24))
            f = midi_to_hz(m_note)
            plk = (
                0.62 * np.sin(2.0 * math.pi * f * pt)
                + 0.26 * np.sin(4.0 * math.pi * f * pt) * np.exp(-pt * 18.0)
            ) * np.minimum(1.0, pt / 0.004) * np.exp(-pt * (8.0 if mode == "bullet_time" else 13.0))
            amp = (0.15 if k % 4 == 0 else 0.085) * (0.70 if mode == "bullet_time" else (1.15 if mode in ("fuse", "pressure") else 1.0))
            add_stereo_event(bus, pan_stereo((plk * amp).astype(np.float32), -0.28 if k % 2 == 0 else 0.28), st)

            # Tactical sub-kick & metallic percussion tick on strong beats (gives real Hollywood trailer drive)
            if mode in ("hook", "throw", "striker", "fuse", "pressure") and k % 2 == 0:
                kn = int(0.16 * SR)
                kt = np.arange(kn, dtype=np.float32) / SR
                kick = np.sin(2.0 * math.pi * (88.0 * np.exp(-kt * 26.0) + 36.0) * kt) * np.exp(-kt * 16.0) * (0.20 if k % 4 == 0 else 0.11)
                add_stereo_event(bus, pan_stereo(kick.astype(np.float32), 0.0), st)
            if mode in ("hook", "striker", "fuse", "pressure"):
                hn = int(0.045 * SR)
                ht = np.arange(hn, dtype=np.float32) / SR
                hat = butter_filter(rng.standard_normal(hn).astype(np.float32), "high", 6800.0, order=2) * np.exp(-ht * 95.0) * (0.048 if k % 2 == 1 else 0.025)
                add_stereo_event(bus, pan_stereo(hat.astype(np.float32), 0.22 if k % 2 == 0 else -0.22), st)
            st += step
            k += 1

    # Broadcast vocal-pocket EQ cut across 250-3500 Hz (keeps sub-bass punch and crisp highs while clearing narration)
    for ch in range(CHANNELS):
        bus[:, ch] = peaking_eq(bus[:, ch], 880.0, 0.55, -4.6)
        bus[:, ch] = peaking_eq(bus[:, ch], 2450.0, 0.80, -3.4)
    return (bus * 0.48).astype(np.float32)


def synth_sfx(kind: str, pan: float, event_t: float) -> np.ndarray:
    rng = np.random.default_rng(int(round(event_t * 1000)) + abs(hash(kind)) % 10000)

    if kind == "pin_pull_click":
        # Metallic tooth-tug friction (0..0.15s) -> crisp bent cotter-pin spring POP (0.16s)
        n = int(0.46 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        tug_env = np.where(t < 0.16, np.sin(math.pi * np.clip(t, 0.0, 0.16) / 0.16), 0.0)
        tug = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (2100.0, 8800.0), order=2) * tug_env * 0.44
        t_pop = np.maximum(0.0, t - 0.155)
        pop_mask = (t >= 0.155).astype(np.float32)
        clink = (
            0.55 * np.sin(2 * math.pi * 3850.0 * t_pop)
            + 0.38 * np.sin(2 * math.pi * 6120.0 * t_pop)
            + 0.24 * np.sin(2 * math.pi * 8740.0 * t_pop)
        ) * np.exp(-t_pop * 26.0) * pop_mask * 0.65
        snap = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (2800.0, 11500.0), order=2) * np.exp(-t_pop * 55.0) * pop_mask * 0.72
        return pan_stereo(np.tanh((tug + clink + snap) * 0.85).astype(np.float32) * 0.72, pan)

    if kind == "pin_ring_jingle":
        # Chrome pull-ring double-clink jingle
        n = int(0.52 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        r1 = (0.50 * np.sin(2 * math.pi * 4680.0 * t) + 0.35 * np.sin(2 * math.pi * 7240.0 * t) + 0.22 * np.sin(2 * math.pi * 10420.0 * t)) * np.exp(-t * 14.0)
        t2 = np.maximum(0.0, t - 0.11)
        m2 = (t >= 0.11).astype(np.float32)
        r2 = (0.45 * np.sin(2 * math.pi * 5120.0 * t2) + 0.32 * np.sin(2 * math.pi * 7890.0 * t2)) * np.exp(-t2 * 16.0) * m2 * 0.72
        return pan_stereo(((r1 + r2) * 0.42).astype(np.float32), pan)

    if kind == "hero_throw_whoosh":
        # Heroic wind-up & forceful overhead throw whoosh sweeping L -> R
        n = int(0.72 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        raw = rng.standard_normal(n).astype(np.float32)
        cloth = butter_filter(raw, "band", (380.0, 1800.0), order=2) * np.exp(-((t - 0.14) / 0.10) ** 2) * 0.45
        whoosh_hi = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (450.0, 3200.0), order=2) * np.exp(-((t - 0.38) / 0.15) ** 2) * 0.92
        whoosh_lo = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (110.0, 480.0), order=2) * np.exp(-((t - 0.36) / 0.16) ** 2) * 0.78
        thump = np.sin(2.0 * math.pi * (92.0 * np.exp(-np.maximum(0.0, t - 0.28) * 14.0) + 42.0) * np.maximum(0.0, t - 0.28)) * (t >= 0.28) * np.exp(-np.maximum(0.0, t - 0.28) * 12.0) * 0.42
        return pan_sweep_stereo(np.tanh(cloth + whoosh_hi + whoosh_lo + thump).astype(np.float32) * 0.68, -0.65, 0.65)

    if kind == "bullet_time_drop":
        # Matrix bullet-time sub-bass freeze drop + glassy time-dilation downward warp
        n = int(1.25 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        f_sub = 132.0 * np.exp(-t * 3.1) + 33.0
        sub_drop = np.sin(2.0 * math.pi * np.cumsum(f_sub) / SR) * np.minimum(1.0, t / 0.015) * np.exp(-t * 1.45) * 0.92
        f_warp = 1850.0 * np.exp(-t * 2.4) + 320.0
        shimmer = np.sin(2.0 * math.pi * np.cumsum(f_warp) / SR) * np.sin(math.pi * np.minimum(1.0, t / 1.10)) * np.exp(-t * 2.2) * 0.24
        air = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (180.0, 1400.0), order=2) * np.exp(-t * 3.2) * 0.48
        # Subtle stereo width on the warp shimmer
        l_sig = np.tanh(sub_drop + air + shimmer * 1.05).astype(np.float32) * 0.78
        r_sig = np.tanh(sub_drop + air + np.roll(shimmer, 90) * 0.95).astype(np.float32) * 0.78
        return np.stack([l_sig, r_sig], axis=-1).astype(np.float32)

    if kind == "xray_slice_whoosh":
        # Glowing cyan X-ray laser scan sweep
        n = int(0.95 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        w = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (360.0, 4400.0), order=2)
        f = 310.0 + 1180.0 * (t / 0.95)
        sweep = (np.sin(2.0 * math.pi * np.cumsum(f) / SR) * 0.22 + np.sin(4.0 * math.pi * np.cumsum(f) / SR) * 0.11)
        env = np.sin(math.pi * t / 0.95) ** 1.35
        return pan_sweep_stereo(((w * 0.46 + sweep) * env * 0.56).astype(np.float32), -0.52, 0.52)

    if kind == "spring_tension_creak":
        # High-tension coiled steel spring creak
        n = int(0.65 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        f = 540.0 + 360.0 * np.sin(2 * math.pi * 8.5 * t)
        creak = (np.sin(2.0 * math.pi * np.cumsum(f) / SR) * 0.26 + np.sin(6.0 * math.pi * np.cumsum(f) / SR) * 0.10)
        grit = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1800.0, 5200.0), order=2) * 0.14
        env = np.sin(math.pi * t / 0.65)
        return pan_stereo(((creak + grit) * env).astype(np.float32), pan)

    if kind == "spoon_ping_fly":
        # Crisp metallic safety spoon PING + spinning Doppler flight whoosh
        n = int(0.82 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        ping = (
            0.58 * np.sin(2 * math.pi * 2940.0 * t) * np.exp(-t * 5.0)
            + 0.36 * np.sin(2 * math.pi * 4980.0 * t) * np.exp(-t * 7.8)
            + 0.26 * np.sin(2 * math.pi * 8240.0 * t) * np.exp(-t * 13.0)
        ) * np.minimum(1.0, t / 0.002)
        flutter = 0.65 + 0.35 * np.sin(2 * math.pi * 15.0 * t)
        whoosh = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (650.0, 4400.0), order=2) * np.exp(-t * 5.2) * flutter * 0.30
        return pan_sweep_stereo(((ping + whoosh) * 0.68).astype(np.float32), 0.0, 0.76)

    if kind == "striker_hammer_snap":
        # Heavy steel striker hammer slam
        n = int(0.28 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        hit = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1400.0, 9600.0), order=2) * np.exp(-t * 40.0) * 1.15
        body = np.sin(2 * math.pi * 162.0 * t) * np.exp(-t * 26.0) * 0.62
        ring = np.sin(2 * math.pi * 3420.0 * t) * np.exp(-t * 32.0) * 0.28
        return pan_stereo(np.tanh(hit + body + ring).astype(np.float32) * 0.74, pan)

    if kind == "primer_spark_pop":
        # Sharp brass primer ignition spark crack
        n = int(0.48 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        pop = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1800.0, 11800.0), order=2) * np.exp(-t * 24.0) * 1.05
        brass = np.sin(2 * math.pi * 5450.0 * t) * np.exp(-t * 28.0) * 0.25
        sizz = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (3400.0, 12400.0), order=2) * np.exp(-t * 6.2) * 0.35
        return pan_stereo(np.tanh(pop + brass + sizz).astype(np.float32) * 0.68, pan)

    if kind == "fuse_burn_sizzle":
        # Continuous 4-second delay fuse sizzle with organic powder crackles (14.95s -> 20.50s)
        dur = 5.55
        n = int(dur * SR)
        t = np.arange(n, dtype=np.float32) / SR
        raw = rng.standard_normal(n).astype(np.float32)
        sizz = butter_filter(raw, "band", (2300.0, 10200.0), order=2) * (0.74 + 0.26 * np.sin(2 * math.pi * 17.0 * t))
        # Sparse pyrotechnic micro-crackles
        impulses = (rng.random(n) > 0.9992).astype(np.float32) * rng.uniform(0.6, 1.4, n).astype(np.float32)
        crackles = butter_filter(impulses, "band", (3200.0, 11500.0), order=2) * 3.8
        env = np.minimum(1.0, t / 0.14) * np.minimum(1.0, (dur - t) / 0.22)
        return pan_stereo(((sizz * 0.28 + crackles * 0.18) * env).astype(np.float32), pan)

    if kind == "timer_tick":
        # Crisp tactical mechanical countdown tick
        n = int(0.12 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        tick = (
            0.62 * np.sin(2 * math.pi * 1950.0 * t)
            + 0.35 * np.sin(2 * math.pi * 3900.0 * t)
            + 0.18 * np.sin(2 * math.pi * 7200.0 * t)
        ) * np.exp(-t * 48.0)
        return pan_stereo((tick * 0.48).astype(np.float32), pan)

    if kind == "grenade_dirt_bounce":
        # Metallic cast-iron grenade thud + dusty earth/gravel bounce by enemy sandbags
        n = int(0.75 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        # Primary impact (0.00s)
        thud1 = np.sin(2 * math.pi * (115.0 * np.exp(-t * 18.0) + 46.0) * t) * np.exp(-t * 14.0) * 0.85
        iron1 = (0.45 * np.sin(2 * math.pi * 920.0 * t) + 0.28 * np.sin(2 * math.pi * 1840.0 * t)) * np.exp(-t * 32.0) * 0.42
        dirt1 = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (650.0, 5800.0), order=2) * np.exp(-t * 11.0) * 0.62
        # Secondary bounce (0.24s)
        t2 = np.maximum(0.0, t - 0.24)
        m2 = (t >= 0.24).astype(np.float32)
        thud2 = np.sin(2 * math.pi * (128.0 * np.exp(-t2 * 22.0) + 54.0) * t2) * np.exp(-t2 * 18.0) * m2 * 0.44
        iron2 = np.sin(2 * math.pi * 960.0 * t2) * np.exp(-t2 * 36.0) * m2 * 0.24
        dirt2 = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (900.0, 6500.0), order=2) * np.exp(-t2 * 14.0) * m2 * 0.36
        return pan_stereo(np.tanh(thud1 + iron1 + dirt1 + thud2 + iron2 + dirt2).astype(np.float32) * 0.72, pan)

    if kind == "core_ignition_flash":
        # Explosive internal core ignition flash
        n = int(1.15 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        sub = np.sin(2 * math.pi * (98.0 * np.exp(-t * 2.5) + 38.0) * t) * np.exp(-t * 2.6) * 0.76
        burst = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (240.0, 5200.0), order=2) * np.exp(-t * 5.8) * 0.70
        return pan_stereo(np.tanh(sub + burst).astype(np.float32) * 0.74, pan)

    if kind == "pressure_swell_sub":
        # Rumbling internal pressure swell rising straight into 25.52s detonation (21.75s -> 25.50s)
        dur = 3.75
        n = int(dur * SR)
        t = np.arange(n, dtype=np.float32) / SR
        prog = t / dur
        f = 40.0 + 155.0 * (prog ** 2.2)
        swell = np.sin(2.0 * math.pi * np.cumsum(f) / SR) * (0.22 + 0.78 * (prog ** 1.5))
        strain = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (80.0, 520.0), order=2) * (prog ** 2.0) * 0.42
        creak = np.sin(2.0 * math.pi * np.cumsum(380.0 + 620.0 * prog) / SR) * (prog ** 2.6) * 0.14
        env = np.minimum(1.0, t / 0.15) * np.minimum(1.0, (dur - t) / 0.04)
        return pan_stereo(np.tanh((swell + strain + creak) * env).astype(np.float32) * 0.56, pan)

    if kind == "shell_fracture_slowmo":
        # Colossal layered Hollywood bunker detonation + stereo shrapnel zings + slow-mo bass boom!
        dur = 3.20
        n = int(dur * SR)
        t = np.arange(n, dtype=np.float32) / SR
        raw = rng.standard_normal(n).astype(np.float32)
        # 1. Razor-sharp initial shockwave crack
        crack = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (350.0, 9500.0), order=2) * np.exp(-t * 22.0) * 1.25
        # 2. Colossal slow-mo pitch-diving seismic sub-bass boom
        f_boom = 94.0 * np.exp(-t * 1.45) + 28.0
        boom = np.sin(2.0 * math.pi * np.cumsum(f_boom) / SR) * np.exp(-t * 0.92) * 1.05
        # 3. Massive bunker blast & rubble roar
        rumble = butter_filter(raw, "band", (34.0, 260.0), order=3) * np.exp(-t * 1.05) * 0.96
        debris = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1800.0, 8800.0), order=2) * np.exp(-t * 2.2) * 0.48
        base_stereo = pan_stereo(np.tanh(crack + boom + rumble + debris).astype(np.float32) * 0.84, pan)
        # 4. 4 staggered stereo Doppler cast-iron shrapnel zings whipping across the soundfield!
        zings = [
            (0.06, 0.55, 4200.0, 1150.0, -0.85, 0.75),
            (0.18, 0.62, 5400.0, 1480.0, 0.85, -0.70),
            (0.34, 0.58, 3600.0, 920.0, -0.60, 0.90),
            (0.52, 0.65, 4800.0, 1300.0, 0.70, -0.85),
        ]
        for z_t0, z_dur, f0, f1, p0, p1 in zings:
            zn = int(z_dur * SR)
            zt = np.arange(zn, dtype=np.float32) / SR
            zf = f0 * ((f1 / f0) ** (zt / z_dur))
            z_sig = (np.sin(2.0 * math.pi * np.cumsum(zf) / SR) * 0.65 + np.sin(4.0 * math.pi * np.cumsum(zf) / SR) * 0.25) * np.sin(math.pi * zt / z_dur) ** 1.3 * 0.24
            add_stereo_event(base_stereo, pan_sweep_stereo(z_sig.astype(np.float32), p0, p1), z_t0)
        return np.tanh(base_stereo * 1.05).astype(np.float32) * 0.86

    if kind == "cinema_outro_hit":
        # Final cinema sub impact + dark heroic chord tail
        n = int(1.60 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        sub = np.sin(2 * math.pi * (64.0 * np.exp(-t * 3.8) + 34.0) * t) * np.exp(-t * 2.1) * 0.65
        chord = (
            0.44 * np.sin(2 * math.pi * midi_to_hz(50) * t)
            + 0.32 * np.sin(2 * math.pi * midi_to_hz(57) * t)
            + 0.24 * np.sin(2 * math.pi * midi_to_hz(62) * t)
        ) * np.minimum(1.0, t / 0.018) * np.exp(-t * 1.75) * 0.45
        return pan_stereo(np.tanh(sub + chord).astype(np.float32) * 0.62, pan)

    raise ValueError(f"Unknown SFX: {kind}")


def measure_lufs_48k(stereo: np.ndarray) -> float:
    b1 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285], dtype=np.float64)
    a1 = np.array([1.0, -1.69065929318241, 0.73248077421585], dtype=np.float64)
    b2 = np.array([1.0, -2.0, 1.0], dtype=np.float64)
    a2 = np.array([1.0, -1.99004745483398, 0.99007225036621], dtype=np.float64)
    y = signal.lfilter(b2, a2, signal.lfilter(b1, a1, stereo.astype(np.float64), axis=0), axis=0)
    bl, hl = int(0.400 * SR), int(0.100 * SR)
    p_arr = np.array([float(np.mean(y[s : s + bl, 0] ** 2) + np.mean(y[s : s + bl, 1] ** 2)) for s in range(0, len(y) - bl + 1, hl)], dtype=np.float64)
    lb = -0.691 + 10.0 * np.log10(np.maximum(p_arr, 1e-12))
    am = lb > -70.0
    if not np.any(am):
        return -70.0
    rt = -0.691 + 10.0 * math.log10(max(float(np.mean(p_arr[am])), 1e-12)) - 10.0
    rm = am & (lb > rt)
    return -0.691 + 10.0 * math.log10(max(float(np.mean(p_arr[rm])), 1e-12))


def brickwall_limiter(stereo: np.ndarray, ceiling: float = TRUE_PEAK_CEILING_LIN) -> np.ndarray:
    out = stereo.copy().astype(np.float32)
    for _ in range(3):
        up = signal.resample_poly(out.astype(np.float64), 4, 1, axis=0)
        pk = np.max(np.abs(up), axis=1).reshape(-1, 4).max(axis=1)[: len(out)].astype(np.float32)
        env = np.maximum(smooth_envelope(pk, 0.8, 45.0), pk)
        out *= np.where(env > ceiling, ceiling / np.maximum(env, 1e-6), 1.0).astype(np.float32)[:, None]
    tp = float(np.max(np.abs(signal.resample_poly(out.astype(np.float64), 4, 1, axis=0))))
    if tp > ceiling:
        out *= float(ceiling / tp)
    return out.astype(np.float32)


HANGING = {"a", "an", "the", "of", "on", "in", "to", "with", "and", "or", "as", "your", "into", "for", "by", "at", "this", "he", "his", "it"}


def join_tokens(tokens: list[dict[str, Any]]) -> str:
    return "".join(t["word"] if i == 0 else (("" if (tokens[i - 1]["word"].endswith("—") or t["word"].startswith("—")) else " ") + t["word"]) for i, t in enumerate(tokens))


def split_clause(cl: list[dict[str, Any]], mw: int = 7, mc: int = 44) -> list[list[dict[str, Any]]]:
    if len(cl) <= mw and len(join_tokens(cl)) <= mc:
        return [cl]
    bk, bs, n = max(2, min(len(cl) - 2, mw)), 1e9, len(cl)
    for k in range(2, n - 1):
        lt, rt = join_tokens(cl[:k]), join_tokens(cl[k:])
        sc = abs(len(lt) - len(rt)) + max(0, k - mw) * 40 + max(0, len(lt) - mc) * 6 + (35 if cl[k - 1]["word"].lower().strip(".,!?—") in HANGING else 0) - (18 if cl[k - 1]["word"].endswith((",", "—")) else 0)
        if sc < bs:
            bs, bk = sc, k
    return split_clause(cl[:bk], mw, mc) + split_clause(cl[bk:], mw, mc)


def chunk_words(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    clauses, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        rem = len(words) - 1 - i
        if ((w["word"].endswith(("—", ".", "!", "?"))) or (w["word"].endswith(",") and len(cur) >= 3 and len(join_tokens(cur)) >= 20)) and len(cur) >= 2 and rem != 1:
            clauses.append(cur)
            cur = []
    if cur:
        (clauses[-1].extend(cur) if (len(cur) == 1 and clauses) else clauses.append(cur))
    cards = [c for cl in clauses for c in split_clause(cl)]
    merged = []
    for c in cards:
        (merged[-1].extend(c) if (len(c) == 1 and merged and len(merged[-1]) < 7) else merged.append(c))
    return merged


def choose_break(ch: list[dict[str, Any]]) -> int:
    n = len(ch)
    if n <= 2 or (len(join_tokens(ch)) <= 23 and n <= 4):
        return n
    bk, bs = n, 1e9
    for k in range(1, n):
        l1, l2 = join_tokens(ch[:k]), join_tokens(ch[k:])
        sc = abs(len(l1) - len(l2)) * 1.2 + max(0, len(l1) - 23) * 15 + max(0, len(l2) - 23) * 15 + (24 if (n - k == 1 and n >= 4) else 0) + (30 if ch[k - 1]["word"].lower().strip(".,!?—") in HANGING else 0)
        if sc < bs:
            bs, bk = sc, k
    return bk


def fmt_ass_t(sec: float) -> str:
    cs = int(round(max(0.0, sec) * 100))
    return f"{cs // 360000}:{(cs % 360000) // 6000:02d}:{(cs % 6000) // 100:02d}.{cs % 100:02d}"


def write_subtitles(lines: list[dict[str, Any]], srt_path: Path, ass_path: Path) -> int:
    def fmt_srt(sec: float) -> str:
        ms = int(round(max(0.0, sec) * 1000))
        return f"{ms // 3600000:02d}:{(ms % 3600000) // 60000:02d}:{(ms % 60000) // 1000:02d},{ms % 1000:03d}"

    srt_blocks, ass_ev, idx, prev_ms, prev_cs = [], [], 1, 0, 0
    for ln in lines:
        for ch in chunk_words(ln["words"]):
            sp = choose_break(ch)
            t0m = max(prev_ms, int(round(ch[0]["start"] * 1000)))
            t1m = max(t0m + 50, int(round(ch[-1]["end"] * 1000)))
            prev_ms = t1m
            txt = (join_tokens(ch[:sp]) + "\n" + join_tokens(ch[sp:])) if sp < len(ch) else join_tokens(ch)
            srt_blocks.append(f"{idx}\n{fmt_srt(t0m / 1000.0)} --> {fmt_srt(t1m / 1000.0)}\n{txt}\n")
            idx += 1
            for i, w in enumerate(ch):
                t0c = max(prev_cs, int(round((w["start"] if i > 0 else ch[0]["start"]) * 100)))
                t1c = max(t0c + 3, int(round((ch[i + 1]["start"] if i + 1 < len(ch) else w["end"]) * 100)))
                prev_cs = t1c
                parts = []
                for j, cw in enumerate(ch):
                    tok = cw["word"]
                    st = f"{{\\c&H0038C8FF&}}{tok}{{\\c&H00FFFFFF&}}" if j == i else tok
                    if j == sp:
                        parts.append("\\N" + st)
                    elif j == 0:
                        parts.append(st)
                    else:
                        parts.append(("" if (ch[j - 1]["word"].endswith("—") or tok.startswith("—")) else " ") + st)
                ass_ev.append(f"Dialogue: 0,{fmt_ass_t(t0c/100.0)},{fmt_ass_t(t1c/100.0)},Default,,0,0,0,,{''.join(parts)}")

    srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,66,&H00FFFFFF,&H0038C8FF,&H00080A10,&H96000000,-1,0,0,0,100,100,0,0,1,8,3.0,2,80,80,260,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ass_path.write_text(header + "\n".join(ass_ev) + "\n", encoding="utf-8")
    print(f"[Subtitles] Exported {len(ass_ev)} word-synced karaoke events ({len(srt_blocks)} SRT cards, 0 overlaps)")
    return len(ass_ev)


def main() -> None:
    tl = json.loads(TIMELINE_PATH.read_text(encoding="utf-8"))
    print("[1/4] Synthesizing Studio Neural Narration (en-US-BrianNeural)...")
    dlg_bus, lines = synthesize_dialogue(tl)
    print("[2/4] Composing 124 BPM Hollywood Action + Matrix Bullet-Time Score...")
    bgm_bus = compose_engineering_tension_score()
    print(f"[3/4] Synthesizing {len(tl['sfx'])} Battlefield & Mechanical Foley SFX...")
    sfx_bus = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)
    for ev in tl["sfx"]:
        add_stereo_event(sfx_bus, synth_sfx(ev["type"], float(ev.get("pan", 0.0)), float(ev["t"])), float(ev["t"]))

    print("[4/4] Mastering 30.000s 48kHz Stereo WAV (-14.0 LUFS, -9.5 dB Sidechain Ducking, TP <= -1.5 dBFS)...")
    env = smooth_envelope(dlg_bus, 15.0, 240.0)
    duck = np.clip(1.0 - np.clip(env / 0.06, 0.0, 1.0) * (1.0 - 10.0 ** (-9.5 / 20.0)), 0.30, 1.0)
    mix = butter_filter(dlg_bus * 1.18 + bgm_bus * duck[:, None] * 0.56 + sfx_bus * 0.80, "high", 28.0, order=2)
    for _ in range(2):
        g = 10.0 ** ((TARGET_LUFS - measure_lufs_48k(mix)) / 20.0)
        mix = brickwall_limiter((mix * g).astype(np.float32), TRUE_PEAK_CEILING_LIN)
    micro = int(round(0.004 * SR))
    mix[:micro] *= np.linspace(0.0, 1.0, micro, dtype=np.float32)[:, None]
    mix[-micro:] *= np.linspace(1.0, 0.0, micro, dtype=np.float32)[:, None]

    lufs = measure_lufs_48k(mix)
    tp = 20.0 * math.log10(max(1e-9, float(np.max(np.abs(signal.resample_poly(mix.astype(np.float64), 4, 1, axis=0))))))
    mp = PROJECT_ROOT / tl["audio"]["masterPath"]
    mp.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(mp), mix, SR, subtype="PCM_16")
    ev_count = write_subtitles(lines, PROJECT_ROOT / tl["audio"]["srtPath"], PROJECT_ROOT / tl["audio"]["assPath"])
    (PROJECT_ROOT / tl["audio"]["manifestPath"]).write_text(
        json.dumps(
            {
                "title": tl["title"],
                "duration": DURATION,
                "integratedLufs": round(lufs, 3),
                "truePeakDbfs": round(tp, 3),
                "subtitleEvents": ev_count,
                "lines": lines,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Master] master_audio.wav: LUFS={lufs:.2f} | TruePeak={tp:.2f} dBFS | Events={ev_count} | 30.000s READY!")


if __name__ == "__main__":
    main()
