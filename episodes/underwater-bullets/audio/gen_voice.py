"""
TTS Voice Engineer for Episode #2 ("Why Bullets Stop Dead Just 3 Feet Underwater", 30.0 s @ 60 fps).

Generates:
    episodes/underwater-bullets/audio/output/voice_stem.wav
        - 48000 Hz, Stereo, PCM_16, exact 30.000 s (1,440,000 samples)
        - Microsoft Neural Voice (en-US-BrianNeural, -1Hz pitch, +4% base rate) via edge-tts
        - Calm deadpan @zackdfilms authority
        - Gentle adaptive ffmpeg atempo (<= 1.08x target, hard cap <= 1.12x) ONLY if clip exceeds beat window
        - Broadcast vocal chain: 75 Hz 24 dB/oct HPF + 140 Hz chest warmth + 3.2 kHz presence + 3:1 comp
        - ZERO word truncation & peak < 0.90

    episodes/underwater-bullets/audio/output/voice_manifest.json
        - Exact line + word-level timestamps (`words` list with `text`, `t0`, `t1`, line `beat`)
          synchronized cleanly to actual rendered audio duration and WordBoundary events
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

AUDIO_DIR = Path(__file__).resolve().parent
EPISODE_ROOT = AUDIO_DIR.parent
SRC_DIR = EPISODE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import bullet_contract as C  # noqa: E402

TIMELINE_PATH = EPISODE_ROOT / "config" / "timeline.json"
OUTPUT_DIR = AUDIO_DIR / "output"
VOICE_STEM_PATH = OUTPUT_DIR / "voice_stem.wav"
MANIFEST_PATH = OUTPUT_DIR / "voice_manifest.json"

SR = 48000
CHANNELS = 2
DURATION_S = float(C.DURATION_S)
TOTAL_SAMPLES = int(round(DURATION_S * SR))

DEFAULT_VOICE = "en-US-BrianNeural"
DEFAULT_PITCH = "-1Hz"
DEFAULT_RATE_PCT = 4
TARGET_ATEMPO = 1.08
MAX_ATEMPO = 1.12


def pan_stereo(mono: np.ndarray, pan: float) -> np.ndarray:
    """Constant-power stereo panner (-1.0 left .. +1.0 right)."""
    pan = float(np.clip(pan, -1.0, 1.0))
    angle = (pan + 1.0) * (math.pi / 4.0)
    return np.stack([mono * math.cos(angle), mono * math.sin(angle)], axis=-1).astype(np.float32)


def add_stereo_event(bus: np.ndarray, event: np.ndarray, start_time: float) -> None:
    """Mix a mono or stereo event into the stereo bus at start_time (seconds)."""
    if event.ndim == 1:
        event = pan_stereo(event, 0.0)
    idx = int(round(start_time * SR))
    if idx >= len(bus) or idx + len(event) <= 0:
        return
    src_s = max(0, -idx)
    dst_s = max(0, idx)
    clen = min(len(event) - src_s, len(bus) - dst_s)
    if clen > 0:
        bus[dst_s : dst_s + clen] += event[src_s : src_s + clen]


def butter_filter(
    audio: np.ndarray,
    btype: str,
    cutoff: float | tuple[float, float],
    order: int = 4,
) -> np.ndarray:
    """Apply Butterworth SOS filter along axis=0 (order=4 gives 24 dB/oct rolloff)."""
    nyq = 0.5 * SR
    if isinstance(cutoff, tuple):
        wn: float | list[float] = [float(np.clip(c / nyq, 1e-4, 0.999)) for c in cutoff]
    else:
        wn = float(np.clip(cutoff / nyq, 1e-4, 0.999))
    sos = signal.butter(order, wn, btype=btype, output="sos")
    return signal.sosfilt(sos, audio, axis=0).astype(np.float32)


def peaking_eq(mono: np.ndarray, freq: float, q: float, gain_db: float) -> np.ndarray:
    """RBJ peaking biquad EQ filter."""
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / SR
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    norm = 1.0 + alpha / a_lin
    b = np.array([1.0 + alpha * a_lin, -2.0 * cw, 1.0 - alpha * a_lin], dtype=np.float64) / norm
    a = np.array([1.0, (-2.0 * cw) / norm, (1.0 - alpha / a_lin) / norm], dtype=np.float64)
    return signal.lfilter(b, a, mono.astype(np.float64), axis=0).astype(np.float32)


def smooth_envelope(sig: np.ndarray, attack_ms: float, release_ms: float) -> np.ndarray:
    """Block-peak follower with smooth attack/release interpolated back to sample rate."""
    x = np.max(np.abs(sig), axis=1) if sig.ndim == 2 else np.abs(sig)
    x = x.astype(np.float32)
    step = 64
    n_blk = (len(x) + step - 1) // step
    blk = np.array(
        [float(np.max(x[i * step : min(len(x), (i + 1) * step)])) for i in range(n_blk)],
        dtype=np.float32,
    )
    att = math.exp(-step / max(1.0, attack_ms * 0.001 * SR))
    rel = math.exp(-step / max(1.0, release_ms * 0.001 * SR))
    env = np.zeros(n_blk, dtype=np.float32)
    cur = 0.0
    for i in range(n_blk):
        v = float(blk[i])
        cur = (att * cur + (1.0 - att) * v) if v > cur else (rel * cur + (1.0 - rel) * v)
        env[i] = cur
    return np.interp(
        np.linspace(0, n_blk - 1, len(x), dtype=np.float32),
        np.arange(n_blk, dtype=np.float32),
        env,
    ).astype(np.float32)


def trim_silence(mono: np.ndarray, thresh_db: float = -48.0) -> tuple[np.ndarray, float]:
    """Trim leading/trailing silence while preserving soft fricative onsets; returns (trimmed, lead_s)."""
    thr = float(np.max(np.abs(mono)) + 1e-12) * (10.0 ** (thresh_db / 20.0))
    idx = np.where(np.abs(mono) > thr)[0]
    if len(idx) == 0:
        return mono.astype(np.float32), 0.0
    s0 = max(0, int(idx[0]) - int(0.012 * SR))
    s1 = min(len(mono), int(idx[-1]) + int(0.022 * SR))
    return mono[s0:s1].astype(np.float32), s0 / float(SR)


def broadcast_vocal_chain(mono: np.ndarray) -> np.ndarray:
    """Broadcast voice EQ (75 Hz 24 dB/oct HPF, 140 Hz warmth, 3.2 kHz presence), 3:1 compression & peak < 0.90."""
    y = butter_filter(mono, "high", 75.0, order=4)
    y = peaking_eq(y, 140.0, 1.1, +1.5)
    y = peaking_eq(y, 280.0, 1.4, -1.0)
    y = peaking_eq(y, 3200.0, 1.0, +2.3)
    y = peaking_eq(y, 7200.0, 0.9, +1.0)
    env = smooth_envelope(y, 6.0, 95.0)
    thr, ratio = 0.14, 3.0
    gain = np.where(env > thr, (thr + (env - thr) / ratio) / np.maximum(env, 1e-6), 1.0)
    y = (y * gain).astype(np.float32)
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2) + 1e-12))
    y *= 0.19 / max(1e-5, rms)
    y = np.tanh(y * 0.85).astype(np.float32)
    y = (y - np.mean(y, dtype=np.float64)).astype(np.float32)
    peak = float(np.max(np.abs(y)))
    if peak > 0.88:
        y *= 0.88 / peak
    return y.astype(np.float32)


def apply_cosine_fades(mono: np.ndarray, fade_s: float = 0.012) -> np.ndarray:
    """Apply smooth raised-cosine fade-in/out to eliminate clicks at line boundaries."""
    out = mono.astype(np.float32, copy=True)
    fn = min(int(round(fade_s * SR)), len(out) // 4)
    if fn > 0:
        t = np.linspace(0.0, math.pi, fn, dtype=np.float32)
        ramp = 0.5 * (1.0 - np.cos(t))
        out[:fn] *= ramp
        out[-fn:] *= ramp[::-1]
    return out


def ffmpeg_atempo_48k(mono: np.ndarray, rate: float) -> np.ndarray:
    """Time-scale mono audio via ffmpeg WSOLA atempo (clamped <= MAX_ATEMPO)."""
    if abs(rate - 1.0) < 0.004:
        return mono.astype(np.float32)
    rate = float(np.clip(rate, 0.88, MAX_ATEMPO))
    with tempfile.TemporaryDirectory() as td:
        ip, op = Path(td) / "in.wav", Path(td) / "out.wav"
        sf.write(str(ip), mono.astype(np.float32), SR, subtype="FLOAT")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(ip),
                "-filter:a",
                f"atempo={rate:.5f}",
                "-ar",
                str(SR),
                "-ac",
                "1",
                str(op),
                "-loglevel",
                "error",
            ],
            check=True,
        )
        y, _ = sf.read(str(op), dtype="float32")
    return y.astype(np.float32)


async def _edge_tts_synth_async(
    text: str,
    voice: str,
    rate_pct: int,
    pitch: str,
) -> tuple[np.ndarray, list[tuple[str, float, float]]]:
    """Synthesize speech with Edge-TTS and return (mono_48k, word_boundaries)."""
    import edge_tts

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            comm = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=f"{rate_pct:+d}%",
                pitch=pitch,
                boundary="WordBoundary",
            )
            mp3 = bytearray()
            wb: list[tuple[str, float, float]] = []
            async for ch in comm.stream():
                ctype = ch.get("type")
                if ctype == "audio":
                    mp3.extend(ch["data"])
                elif ctype == "WordBoundary":
                    wb.append(
                        (
                            str(ch.get("text", "")),
                            float(ch["offset"]) / 1e7,
                            float(ch["duration"]) / 1e7,
                        )
                    )
            if not mp3:
                raise RuntimeError("Edge-TTS returned empty audio stream")
            with tempfile.TemporaryDirectory() as td:
                mp, wp = Path(td) / "v.mp3", Path(td) / "v.wav"
                mp.write_bytes(bytes(mp3))
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(mp), "-ar", str(SR), "-ac", "1", str(wp), "-loglevel", "error"],
                    check=True,
                )
                y, _ = sf.read(str(wp), dtype="float32")
            return y.astype(np.float32), wb
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < 2:
                await asyncio.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Edge-TTS synthesis failed after 3 attempts: {last_err}") from last_err


def tokenize_display_words(text: str) -> list[str]:
    """Split display text into words preserving trailing punctuation for subtitle chunking."""
    tokens: list[str] = []
    for raw in text.strip().split():
        if "\u2014" in raw and not raw.endswith("\u2014"):
            parts = raw.split("\u2014")
            for i, p in enumerate(parts):
                if i < len(parts) - 1:
                    tokens.append(p + "\u2014")
                elif p:
                    tokens.append(p)
        else:
            tokens.append(raw)
    return [t for t in tokens if t]


def align_words_weighted(words: list[str], t0: float, t1: float) -> list[dict[str, Any]]:
    """Fallback proportional character+punctuation word alignment."""
    weights = np.array(
        [
            max(1.8, float(len(re.sub(r"[^A-Za-z0-9]", "", w))))
            + (1.5 if w.endswith(("!", "?", ".")) else (1.0 if w.endswith(("\u2014", ",")) else 0.0))
            for w in words
        ],
        dtype=np.float64,
    )
    weights /= weights.sum()
    bounds = [t0]
    acc = 0.0
    for wt in weights[:-1]:
        acc += float(wt)
        bounds.append(t0 + (t1 - t0) * acc)
    bounds.append(t1)
    return [{"text": w, "t0": round(bounds[i], 4), "t1": round(bounds[i + 1], 4)} for i, w in enumerate(words)]


def _merge_boundaries_to_display(
    display_words: list[str],
    word_bounds: list[tuple[str, float, float]],
) -> list[tuple[float, float]] | None:
    """Group consecutive Edge-TTS WordBoundary tokens (e.g. hyphenated words) to match display_words."""
    if not display_words or not word_bounds:
        return None
    norm = lambda s: re.sub(r"[^A-Za-z0-9]", "", s).lower()  # noqa: E731
    grouped: list[tuple[float, float]] = []
    j = 0
    for dw in display_words:
        target = norm(dw)
        if not target or j >= len(word_bounds):
            return None
        acc = ""
        start_s = word_bounds[j][1]
        end_s = start_s + word_bounds[j][2]
        while j < len(word_bounds):
            tok, off_s, dur_s = word_bounds[j]
            acc += norm(tok)
            end_s = off_s + dur_s
            j += 1
            if acc == target:
                break
            if len(acc) > len(target) or not target.startswith(acc):
                return None
        if acc != target:
            return None
        grouped.append((start_s, max(0.015, end_s - start_s)))
    if j != len(word_bounds):
        return None
    return grouped


def align_words_from_boundaries(
    display_words: list[str],
    word_bounds: list[tuple[str, float, float]],
    lead_s: float,
    scale: float,
    t0: float,
    t1: float,
) -> list[dict[str, Any]]:
    """Map Edge-TTS WordBoundary timings (shifted for trim_silence and scaled for atempo) onto display words."""
    grouped = _merge_boundaries_to_display(display_words, word_bounds)
    if grouped is None:
        return align_words_weighted(display_words, t0, t1)

    aligned: list[dict[str, Any]] = []
    prev_end = t0
    n = len(display_words)
    for i, (dw, (off_s, dur_s)) in enumerate(zip(display_words, grouped)):
        ws = t0 + max(0.0, off_s - lead_s) * scale
        we = t0 + max(0.0, (off_s + dur_s) - lead_s) * scale
        if i == 0:
            ws = t0
        else:
            ws = max(prev_end, min(ws, t1 - 0.015 * (n - i)))
        if i == n - 1:
            we = t1
        else:
            we = max(ws + 0.015, min(we, t1 - 0.015 * (n - 1 - i)))
        prev_end = we
        aligned.append({"text": dw, "t0": round(ws, 4), "t1": round(we, 4)})
    return aligned


def synthesize_narration(tl: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Synthesize all narration lines with en-US-BrianNeural, gentle adaptive atempo <= 1.12x, zero truncation."""
    bus = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)
    lines: list[dict[str, Any]] = []
    speakers = tl.get("speakers", {})
    nar_items = tl["narration_lines"]

    for idx, item in enumerate(nar_items):
        lid = str(item["id"])
        beat = str(item["beat"])
        if beat not in C.BEATS:
            raise ValueError(f"{lid} references unknown beat {beat!r} not in bullet_contract.BEATS")
        spk_id = str(item.get("speaker", "narrator"))
        spk_cfg = speakers.get(spk_id, {})
        voice = str(spk_cfg.get("voice", DEFAULT_VOICE))
        pan = float(spk_cfg.get("pan", 0.0))

        s0, s1 = float(item["t0"]), float(item["t1"])
        slot = s1 - s0
        next_start = float(nar_items[idx + 1]["t0"]) if idx + 1 < len(nar_items) else DURATION_S - 0.06
        max_end = min(s1, next_start - 0.06, DURATION_S - 0.05)

        tts_txt = str(item["text"]).replace("\u2014", ", ")
        rate_pct = DEFAULT_RATE_PCT
        raw_audio, wb = asyncio.run(_edge_tts_synth_async(tts_txt, voice, rate_pct, DEFAULT_PITCH))
        comp, lead_s = trim_silence(raw_audio)
        cdur = len(comp) / float(SR)

        # Keep ffmpeg atempo <= MAX_ATEMPO (1.12x, targeting <= 1.08x) by adapting Edge-TTS rate if needed.
        if cdur > slot * (TARGET_ATEMPO * 0.96):
            dur_base = cdur * (1.0 + rate_pct / 100.0)
            dur_target = slot * (TARGET_ATEMPO * 0.955)
            cand_rate = max(rate_pct + 2, min(65, int(math.ceil((dur_base / dur_target - 1.0) * 100.0))))
            raw2, wb2 = asyncio.run(_edge_tts_synth_async(tts_txt, voice, cand_rate, DEFAULT_PITCH))
            comp2, lead2 = trim_silence(raw2)
            if len(comp2) < len(comp):
                comp, lead_s, wb, rate_pct = comp2, lead2, wb2, cand_rate
                cdur = len(comp) / float(SR)

        if cdur > slot * (MAX_ATEMPO * 0.955):
            dur_base = cdur * (1.0 + rate_pct / 100.0)
            dur_target = slot * (TARGET_ATEMPO * 0.95)
            cand_rate = max(rate_pct + 2, min(75, int(math.ceil((dur_base / dur_target - 1.0) * 100.0))))
            raw3, wb3 = asyncio.run(_edge_tts_synth_async(tts_txt, voice, cand_rate, DEFAULT_PITCH))
            comp3, lead3 = trim_silence(raw3)
            if len(comp3) < len(comp):
                comp, lead_s, wb, rate_pct = comp3, lead3, wb3, cand_rate
                cdur = len(comp) / float(SR)

        atempo = float(np.clip(cdur / (slot * 0.96), 1.0, MAX_ATEMPO)) if cdur > slot * 0.97 else 1.0
        scaled = ffmpeg_atempo_48k(comp, atempo)
        proc = apply_cosine_fades(broadcast_vocal_chain(scaled), fade_s=0.012)
        proc_dur = len(proc) / float(SR)

        if s0 + proc_dur > max_end:
            raise RuntimeError(
                f"{lid} exceeded window without truncation allowed: {s0 + proc_dur:.3f}s > {max_end:.3f}s "
                f"(rate={rate_pct:+d}%, atempo={atempo:.4f}x)"
            )

        t0 = s0
        t1 = s0 + proc_dur
        time_scale = proc_dur / max(1e-6, cdur)
        add_stereo_event(bus, pan_stereo(proc, pan), t0)

        display_words = tokenize_display_words(str(item["text"]))
        words = align_words_from_boundaries(display_words, wb, lead_s, time_scale, t0, t1)

        lines.append(
            {
                "id": lid,
                "beat": beat,
                "speaker": spk_id,
                "text": str(item["text"]),
                "t0": round(t0, 4),
                "t1": round(t1, 4),
                "ratePct": int(rate_pct),
                "atempo": round(atempo, 4),
                "words": words,
            }
        )
        print(
            f"  [EdgeTTS] {lid} ({beat}, {voice}, rate={rate_pct:+d}%, atempo={atempo:.4f}x): "
            f"{t0:5.2f}s..{t1:5.2f}s / window={s0:.2f}..{s1:.2f}s (margin={s1 - t1:+.3f}s, {len(words)} words)"
        )

    # Click-free 5 ms boundary micro-fades on the full 30.000 s stem
    micro = int(round(0.005 * SR))
    bus[:micro] *= np.linspace(0.0, 1.0, micro, dtype=np.float32)[:, None]
    bus[-micro:] *= np.linspace(1.0, 0.0, micro, dtype=np.float32)[:, None]
    return bus, lines


def verify_outputs(stem_path: Path, manifest_path: Path, tl: dict[str, Any]) -> None:
    """Strictly verify 48kHz stereo 30.000s WAV specs, peak < 0.90, zero truncation, and manifest schema."""
    info = sf.info(str(stem_path))
    assert info.samplerate == SR, f"Expected SR={SR}, got {info.samplerate}"
    assert info.channels == CHANNELS, f"Expected {CHANNELS} channels, got {info.channels}"
    assert info.frames == TOTAL_SAMPLES, f"Expected {TOTAL_SAMPLES} samples, got {info.frames}"
    assert info.subtype == "PCM_16", f"Expected PCM_16, got {info.subtype}"
    assert abs(info.duration - DURATION_S) < 1e-6, f"Expected {DURATION_S}s, got {info.duration}"

    with wave.open(str(stem_path), "rb") as wf:
        assert wf.getframerate() == SR
        assert wf.getnchannels() == CHANNELS
        assert wf.getnframes() == TOTAL_SAMPLES
        assert wf.getsampwidth() == 2, f"Expected PCM_16 (2 bytes), got {wf.getsampwidth()}"

    audio, _ = sf.read(str(stem_path), dtype="float32")
    assert np.isfinite(audio).all(), "Voice stem contains NaN or Inf"
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    assert 0.03 < rms < 0.50, f"Unexpected voice stem RMS: {rms:.5f}"
    assert peak < 0.90, f"Voice stem peak exceeded 0.90: {peak:.5f}"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["title"] == tl["title"]
    assert abs(float(manifest["duration"]) - DURATION_S) < 1e-6
    assert len(manifest["lines"]) == len(tl["narration_lines"])

    prev_end = 0.0
    total_words_manifest = 0
    total_words_timeline = 0
    for src_item, ln in zip(tl["narration_lines"], manifest["lines"]):
        w0, w1 = float(src_item["t0"]), float(src_item["t1"])
        ls, le = float(ln["t0"]), float(ln["t1"])
        assert ln["id"] == src_item["id"]
        assert ln["beat"] == src_item["beat"]
        assert ln["text"] == src_item["text"]
        assert 1.0 <= float(ln["atempo"]) <= MAX_ATEMPO, f"{ln['id']} atempo {ln['atempo']} > {MAX_ATEMPO}"
        assert w0 - 1e-4 <= ls < le <= w1 + 1e-4, f"{ln['id']} [{ls}, {le}] outside window [{w0}, {w1}]"
        assert ls >= prev_end - 1e-4, f"{ln['id']} overlaps previous line ({ls} < {prev_end})"
        prev_end = le

        expected_tokens = tokenize_display_words(src_item["text"])
        words = ln["words"]
        assert len(words) == len(expected_tokens), (
            f"{ln['id']} word count {len(words)} != {len(expected_tokens)}"
        )
        total_words_manifest += len(words)
        total_words_timeline += len(expected_tokens)

        w_prev = ls
        for w in words:
            assert w_prev - 1e-4 <= float(w["t0"]) < float(w["t1"]) <= le + 1e-4
            w_prev = float(w["t1"])
        joined = "".join(
            w["text"]
            if j == 0
            else (("" if (words[j - 1]["text"].endswith("\u2014") or w["text"].startswith("\u2014")) else " ") + w["text"])
            for j, w in enumerate(words)
        )
        assert joined == src_item["text"], f"{ln['id']} text mismatch: {joined!r} != {src_item['text']!r}"

    assert total_words_manifest == total_words_timeline

    print(
        f"[Voice] Verified {stem_path.name}: {info.samplerate} Hz, {info.channels}ch {info.subtype}, "
        f"{info.frames} samples ({info.duration:.3f}s), peak={peak:.4f}, rms={rms:.4f} | "
        f"{len(manifest['lines'])} lines, {total_words_manifest} words (== timeline.json), 0 truncations"
    )


def main() -> None:
    tl = json.loads(TIMELINE_PATH.read_text(encoding="utf-8"))
    assert abs(float(tl["duration"]) - DURATION_S) < 1e-6, "timeline.json duration mismatch"
    assert int(tl["fps"]) == int(C.FPS), "timeline.json fps mismatch"
    assert int(tl["totalFrames"]) == int(C.TOTAL_FRAMES), "timeline.json totalFrames mismatch"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[Voice] Synthesizing {len(tl['narration_lines'])} lines via Edge-TTS "
        f"({DEFAULT_VOICE}, pitch={DEFAULT_PITCH}, base_rate={DEFAULT_RATE_PCT:+d}%)..."
    )
    stem_audio, lines = synthesize_narration(tl)

    sf.write(str(VOICE_STEM_PATH), stem_audio, SR, subtype="PCM_16")
    manifest = {
        "episode": tl.get("episode", "underwater-bullets"),
        "title": tl["title"],
        "duration": DURATION_S,
        "fps": int(C.FPS),
        "voice": DEFAULT_VOICE,
        "lines": lines,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    verify_outputs(VOICE_STEM_PATH, MANIFEST_PATH, tl)


if __name__ == "__main__":
    main()
