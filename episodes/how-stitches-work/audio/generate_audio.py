"""
How Stitches Work — 4K Medical Macro Cinema Studio Audio (AGENTS.md V10 Standard)
  - Studio Microsoft Neural Narration (en-US-BrianNeural — warm, crisp Zack-D-Films style cadence)
  - Clean Modern Medical-Curiosity Pluck & Warm Ambient Pad Score (108 BPM, vocal-pocket EQ + -9.5 dB sidechain ducking)
  - Tactile ASMR Surgical & Cellular Foley SFX:
      * needle_pass (subtle silky suture thread pull)
      * thread_zip_pull (ultra-satisfying taut zipper closure tension!)
      * scissors_snip (razor-crisp stainless-steel surgical snip!)
      * heartbeat_pulse (warm sub-bass organic pulse)
      * cell_swarm_shimmer & collagen_weave_chime (delicate crystalline bio-growth)
      * vessel_pulse & suture_slide_out & healed_resolve
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
BPM = 108.0
BEAT = 60.0 / BPM

TTS_OVERRIDES: dict[str, str] = {
    "L1": "When you get a deep cut, a doctor weaves a curved surgical needle through both edges of your skin,",
    "L2": "pulling the thread tight like a zipper to lock the wound shut, then snipping the ends clean.",
    "L3": "Inside, the skin turns red and inflamed, as millions of repair cells rush in to rebuild the damage.",
    "L4": "Next, golden collagen fibers weave across the gap like living scaffolding, spinning brand new skin!",
    "L5": "Tiny new blood vessels branch through the seam, pumping oxygen to nourish the fresh tissue,",
    "L6": "and a few weeks later, the stitches slide right out, leaving your skin healed back together!",
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
    y = peaking_eq(y, 260.0, 1.3, -1.5)
    y = peaking_eq(y, 3300.0, 1.0, +2.3)
    y = peaking_eq(y, 7000.0, 0.9, +1.4)
    env = smooth_envelope(y, 6.0, 95.0)
    thr, ratio = 0.14, 3.0
    gain = np.where(env > thr, (thr + (env - thr) / ratio) / np.maximum(env, 1e-6), 1.0)
    y = (y * gain).astype(np.float32)
    y *= 0.20 / max(1e-5, float(np.sqrt(np.mean(y ** 2) + 1e-12)))
    return np.tanh(y * 0.95).astype(np.float32)


def ffmpeg_atempo_48k(mono: np.ndarray, rate: float) -> np.ndarray:
    if abs(rate - 1.0) < 0.008:
        return mono.astype(np.float32)
    rate = float(np.clip(rate, 0.88, 1.15))
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
    for item in tl["dialogue"]:
        lid = item["id"]
        s0, s1 = float(item["start"]), float(item["end"])
        slot = s1 - s0
        tts_txt = TTS_OVERRIDES.get(lid, item["text"].replace("—", ", "))
        rate_pct = 0
        comp = trim_silence(asyncio.run(_edge_tts_synth_async(tts_txt, voice, rate_pct, "-1Hz")))
        cdur = len(comp) / float(SR)
        if cdur > slot * 0.96:
            rate_pct = max(4, min(18, int(math.ceil((cdur / (slot * 0.93) - 1.0) * 100.0))))
            comp2 = trim_silence(asyncio.run(_edge_tts_synth_async(tts_txt, voice, rate_pct, "-1Hz")))
            if len(comp2) < len(comp):
                comp, cdur = comp2, len(comp2) / float(SR)
        atempo = float(np.clip(cdur / (slot * 0.95), 1.0, 1.12)) if cdur > slot * 0.97 else 1.0
        proc = broadcast_vocal_chain(ffmpeg_atempo_48k(comp, atempo))
        t0, t1 = s0, min(DURATION - 0.04, s0 + len(proc) / float(SR))
        ml = int(round((t1 - t0) * SR))
        proc = proc[:ml]
        fn = min(int(0.012 * SR), len(proc) // 4)
        if fn > 0:
            proc[:fn] *= np.linspace(0.0, 1.0, fn, dtype=np.float32)
            proc[-fn:] *= np.linspace(1.0, 0.0, fn, dtype=np.float32)
        add_stereo_event(bus, pan_stereo(proc, 0.0), t0)
        lines.append({"id": lid, "beatId": item["beatId"], "speaker": "narrator", "text": item["text"], "start": round(t0, 4), "end": round(t1, 4), "words": align_words(tokenize_display_words(item["text"]), t0, t1)})
        print(f"  [EdgeTTS] {lid} (rate={rate_pct:+d}% atempo={atempo:.2f}x): {t0:5.2f}s..{t1:5.2f}s / slot={slot:.2f}s")
    return bus, lines


def compose_medical_curiosity_score() -> np.ndarray:
    """Clean, addictive Zack-D-Films style marimba/pluck curiosity groove + warm atmospheric pad."""
    bus = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)
    chords = [
        (0.0, 6.0, [48, 55, 60, 63]),   # Cm clinical intrigue
        (6.0, 10.2, [44, 51, 56, 63]),  # Ab focus
        (10.2, 15.2, [41, 53, 56, 60]), # Fm cellular pulse
        (15.2, 20.4, [46, 53, 58, 62]), # Bb collagen bloom
        (20.4, 25.3, [44, 51, 56, 63]), # Ab rising warmth
        (25.3, 30.0, [48, 55, 60, 64]), # C Major healed resolution
    ]
    for t0, t1, notes in chords:
        dur = t1 - t0
        n = int(dur * SR)
        t = np.arange(n, dtype=np.float32) / SR
        env = np.minimum(1.0, t / 0.25) * np.minimum(1.0, (dur - t) / 0.25)
        pad = sum(np.sin(2.0 * math.pi * midi_to_hz(m) * t) for m in notes[:3]) * 0.08 * env
        add_stereo_event(bus, pan_stereo(pad.astype(np.float32), 0.0), t0)

        st = t0
        k = 0
        while st < t1 - 0.12:
            m = notes[1 + (k % 3)] + (12 if k % 4 == 2 else 0)
            pn = int(0.28 * SR)
            pt = np.arange(pn, dtype=np.float32) / SR
            f = midi_to_hz(m)
            # Crisp wooden marimba / felt pluck
            pluck = (0.70 * np.sin(2.0 * math.pi * f * pt) + 0.22 * np.sin(8.0 * math.pi * f * pt) * np.exp(-pt * 24.0)) * np.minimum(1.0, pt / 0.004) * np.exp(-pt * 7.2)
            add_stereo_event(bus, pan_stereo(pluck.astype(np.float32) * 0.14, -0.26 if k % 2 == 0 else 0.26), st)
            st += BEAT * 0.5
            k += 1

    for ch in range(CHANNELS):
        bus[:, ch] = peaking_eq(bus[:, ch], 1200.0, 0.65, -4.5)
    return (bus * 0.44).astype(np.float32)


def synth_sfx(kind: str, pan: float, event_t: float) -> np.ndarray:
    rng = np.random.default_rng(int(round(event_t * 1000)) + abs(hash(kind)) % 10000)
    if kind == "needle_pass":
        n = int(0.32 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        silk = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (2600.0, 7800.0), order=2)
        env = np.sin(math.pi * t / 0.32) ** 1.5
        tink = np.sin(2.0 * math.pi * 3800.0 * t) * np.exp(-t * 26.0) * 0.18
        return pan_stereo(((silk * 0.36 * env + tink) * 0.48).astype(np.float32), pan)
    if kind == "thread_zip_pull":
        n = int(1.35 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        f = 420.0 + 1150.0 * (t / 1.35) ** 1.3
        zip_tone = np.sin(2.0 * math.pi * np.cumsum(f) / SR) * (0.20 + 0.15 * np.sin(2.0 * math.pi * 24.0 * t))
        silk = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1600.0, 6800.0), order=2) * 0.42
        env = np.minimum(1.0, t / 0.12) * np.minimum(1.0, (1.35 - t) / 0.18)
        return pan_sweep_stereo(((zip_tone + silk) * env * 0.46).astype(np.float32), -0.25, 0.25)
    if kind == "scissors_snip":
        n = int(0.28 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        metal = (
            0.55 * np.sin(2.0 * math.pi * 4420.0 * t) * np.exp(-t * 28.0)
            + 0.42 * np.sin(2.0 * math.pi * 6850.0 * t) * np.exp(-t * 36.0)
            + butter_filter(rng.standard_normal(n).astype(np.float32), "band", (3200.0, 12000.0), order=2) * np.exp(-t * 48.0) * 0.9
        )
        return pan_stereo((metal * 0.65).astype(np.float32), pan)
    if kind in ("heartbeat_pulse", "vessel_pulse"):
        n = int(0.55 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        lub = np.sin(2.0 * math.pi * 62.0 * t) * np.exp(-t * 14.0) * 0.55
        dt = np.maximum(0.0, t - 0.19)
        dub = (t > 0.19).astype(np.float32) * np.sin(2.0 * math.pi * 54.0 * dt) * np.exp(-dt * 16.0) * 0.42
        return pan_stereo(((lub + dub) * 0.65).astype(np.float32), pan)
    if kind in ("cell_swarm_shimmer", "collagen_weave_chime"):
        n = int(0.95 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        freqs = [880.0, 1318.5, 1760.0] if kind == "collagen_weave_chime" else [659.25, 987.77, 1318.5]
        chime = sum((0.35 / (i + 1)) * np.sin(2.0 * math.pi * f * t) * np.exp(-t * (2.6 + i)) for i, f in enumerate(freqs))
        env = np.minimum(1.0, t / 0.04)
        return pan_stereo((chime * env * 0.36).astype(np.float32), pan)
    if kind == "suture_slide_out":
        n = int(0.85 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        sw = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (2200.0, 7200.0), order=2)
        env = np.sin(math.pi * t / 0.85) ** 1.3
        return pan_sweep_stereo((sw * env * 0.40).astype(np.float32), -0.2, 0.35)
    if kind == "healed_resolve":
        n = int(1.8 * SR)
        st = np.zeros((n, CHANNELS), dtype=np.float32)
        for k, (m, p) in enumerate([(72, -0.32), (76, -0.10), (79, 0.12), (84, 0.34)]):
            idx0 = int(k * 0.06 * SR)
            tt = np.arange(n - idx0, dtype=np.float32) / SR
            f = midi_to_hz(m)
            bell = np.sin(2.0 * math.pi * f * tt) * np.minimum(1.0, tt / 0.012) * np.exp(-tt * 2.2)
            st[idx0:] += pan_stereo(bell.astype(np.float32) * 0.24, p)
        return st
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


HANGING = {"a", "an", "the", "of", "on", "in", "to", "with", "and", "or", "as", "your", "into", "for", "by", "at"}


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


def write_subtitles(lines: list[dict[str, Any]], srt_path: Path, ass_path: Path) -> None:
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
                    st = f"{{\\c&H00FFE040&}}{tok}{{\\c&H00FFFFFF&}}" if j == i else tok
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
Style: Default,DejaVu Sans,66,&H00FFFFFF,&H00FFE040,&H00120803,&H96000000,-1,0,0,0,100,100,0,0,1,8,3.0,2,80,80,260,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ass_path.write_text(header + "\n".join(ass_ev) + "\n", encoding="utf-8")
    print(f"[Subtitles] Exported {len(ass_ev)} word-synced karaoke events (0 overlaps)")


def main() -> None:
    tl = json.loads(TIMELINE_PATH.read_text(encoding="utf-8"))
    print("[1/4] Synthesizing Studio Neural Narration (en-US-BrianNeural)...")
    dlg_bus, lines = synthesize_dialogue(tl)
    print("[2/4] Composing 108 BPM Medical Curiosity Score...")
    bgm_bus = compose_medical_curiosity_score()
    print(f"[3/4] Synthesizing {len(tl['sfx'])} Tactile Surgical & Cellular ASMR Foley SFX...")
    sfx_bus = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)
    for ev in tl["sfx"]:
        add_stereo_event(sfx_bus, synth_sfx(ev["type"], float(ev.get("pan", 0.0)), float(ev["t"])), float(ev["t"]))

    print("[4/4] Mastering 30.000s 48kHz Stereo WAV (-14.0 LUFS, -9.5 dB Sidechain Ducking, TP <= -1.5 dBFS)...")
    env = smooth_envelope(dlg_bus, 15.0, 240.0)
    duck = np.clip(1.0 - np.clip(env / 0.06, 0.0, 1.0) * (1.0 - 10.0 ** (-9.5 / 20.0)), 0.30, 1.0)
    mix = butter_filter(dlg_bus * 1.18 + bgm_bus * duck[:, None] * 0.58 + sfx_bus * 0.78, "high", 28.0, order=2)
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
    (PROJECT_ROOT / tl["audio"]["manifestPath"]).write_text(json.dumps({"title": tl["title"], "duration": DURATION, "integratedLufs": round(lufs, 3), "truePeakDbfs": round(tp, 3), "lines": lines}, indent=2), encoding="utf-8")
    write_subtitles(lines, PROJECT_ROOT / tl["audio"]["srtPath"], PROJECT_ROOT / tl["audio"]["assPath"])
    print(f"[Master] master_audio.wav: LUFS={lufs:.2f} | TruePeak={tp:.2f} dBFS | 30.000s READY!")


if __name__ == "__main__":
    main()
