"""
generate_audio.py — Broadcast Master Mix Engineer for Episode #5
("Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock", 30.0 s @ 60 fps = 1800 frames).

Mixes voice_stem.wav + sfx_stem.wav + bgm_stem.wav into:
    episodes/helicopter-autorotation/audio/output/master_audio.wav
        - 48000 Hz, Stereo, PCM_16, exact 30.000 s (1,440,000 samples)
        - -9.5 dB smooth vocal sidechain ducking on BGM
        - 360 ms transient guard protecting all helicopter Foley impacts
        - ITU-R BS.1770-4 integrated loudness normalized to -14.0 +/- 0.15 LUFS
        - 4x oversampled lookahead True Peak limiter (True Peak <= -1.50 dBFS)
        - `--rebuild-all` cleanly runs gen_voice, gen_sfx, gen_bgm, and gen_subtitles.main([])
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

AUDIO_DIR = Path(__file__).resolve().parent
EPISODE_ROOT = AUDIO_DIR.parent
SRC_DIR = EPISODE_ROOT / "src"
for _p in (AUDIO_DIR, SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import heli_contract as C  # noqa: E402

TIMELINE_PATH = EPISODE_ROOT / "config" / "timeline.json"
OUTPUT_DIR = AUDIO_DIR / "output"
VOICE_STEM_PATH = OUTPUT_DIR / "voice_stem.wav"
SFX_STEM_PATH = OUTPUT_DIR / "sfx_stem.wav"
BGM_STEM_PATH = OUTPUT_DIR / "bgm_stem.wav"
MANIFEST_PATH = OUTPUT_DIR / "voice_manifest.json"
MASTER_PATH = OUTPUT_DIR / "master_audio.wav"

SR: int = 48000
CHANNELS: int = 2
DURATION_S: float = float(C.DURATION_S)
TOTAL_SAMPLES: int = int(round(DURATION_S * SR))

TARGET_LUFS: float = -14.0
LUFS_TOLERANCE: float = 0.15
LIMITER_CEILING_DBFS: float = -1.62
LIMITER_CEILING_LIN: float = 10.0 ** (LIMITER_CEILING_DBFS / 20.0)
MAX_TRUE_PEAK_DBFS: float = -1.50

VOICE_GAIN: float = 1.22
BGM_GAIN: float = 0.50
SFX_GAIN: float = 1.12

BGM_DUCK_DB: float = -9.5
SFX_BED_DUCK_DB: float = -1.8
SFX_REST_DUCK_DB: float = -0.8
SFX_MID_BAND_HZ: tuple[float, float] = (250.0, 3800.0)
RUMBLE_HP_HZ: float = 28.0
TRANSIENT_GUARD_PRE_S: float = 0.035
TRANSIENT_GUARD_POST_S: float = 0.360

KEY_TRANSIENT_EVENTS: tuple[str, ...] = tuple(C.EV.keys())


def butter_filter(audio: np.ndarray, btype: str, cutoff: float | tuple[float, float], order: int = 2) -> np.ndarray:
    nyq = 0.5 * SR
    if isinstance(cutoff, tuple):
        wn: float | list[float] = [float(np.clip(c / nyq, 1e-4, 0.999)) for c in cutoff]
    else:
        wn = float(np.clip(cutoff / nyq, 1e-4, 0.999))
    sos = signal.butter(order, wn, btype=btype, output="sos")
    return signal.sosfilt(sos, audio, axis=0).astype(np.float32)


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


def measure_lufs_48k(stereo: np.ndarray) -> float:
    b1 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285], dtype=np.float64)
    a1 = np.array([1.0, -1.69065929318241, 0.73248077421585], dtype=np.float64)
    b2 = np.array([1.0, -2.0, 1.0], dtype=np.float64)
    a2 = np.array([1.0, -1.99004745483398, 0.99007225036621], dtype=np.float64)
    y = signal.lfilter(b2, a2, signal.lfilter(b1, a1, stereo.astype(np.float64), axis=0), axis=0)
    bl, hl = int(0.400 * SR), int(0.100 * SR)
    p_arr = np.array(
        [float(np.mean(y[s : s + bl, 0] ** 2) + np.mean(y[s : s + bl, 1] ** 2)) for s in range(0, len(y) - bl + 1, hl)],
        dtype=np.float64,
    )
    lb = -0.691 + 10.0 * np.log10(np.maximum(p_arr, 1e-12))
    am = lb > -70.0
    if not np.any(am):
        return -70.0
    rt = -0.691 + 10.0 * math.log10(max(float(np.mean(p_arr[am])), 1e-12)) - 10.0
    rm = am & (lb > rt)
    return -0.691 + 10.0 * math.log10(max(float(np.mean(p_arr[rm])), 1e-12))


def measure_true_peak_dbfs(stereo: np.ndarray) -> float:
    up = signal.resample_poly(stereo.astype(np.float64), 4, 1, axis=0)
    return 20.0 * math.log10(max(1e-9, float(np.max(np.abs(up)))))


def lookahead_peak_limiter(stereo: np.ndarray, ceiling: float = LIMITER_CEILING_LIN) -> np.ndarray:
    out = stereo.copy().astype(np.float32)
    la_n = int(round(0.0025 * SR))
    for _ in range(3):
        up = signal.resample_poly(out.astype(np.float64), 4, 1, axis=0)
        pk = np.max(np.abs(up), axis=1).reshape(-1, 4).max(axis=1)[: len(out)].astype(np.float32)
        padded = np.pad(pk, (0, la_n), mode="edge")
        la_pk = np.maximum.reduce([padded[k : k + len(pk)] for k in range(0, la_n + 1, 16)])
        env = np.maximum(smooth_envelope(la_pk, 0.8, 55.0), pk)
        gain_curve = np.where(env > ceiling, ceiling / np.maximum(env, 1e-6), 1.0).astype(np.float32)
        out *= gain_curve[:, None]
    tp = float(np.max(np.abs(signal.resample_poly(out.astype(np.float64), 4, 1, axis=0))))
    if tp > ceiling:
        out *= float(ceiling / tp)
    return out.astype(np.float32)


def load_verified_stem(path: Path, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name} stem: {path}")
    info = sf.info(str(path))
    if info.samplerate != SR or info.channels != CHANNELS or info.frames != TOTAL_SAMPLES:
        raise ValueError(f"{name} stem format mismatch")
    audio, _ = sf.read(str(path), dtype="float32")
    return audio.astype(np.float32)


def master_mix(voice_stem: np.ndarray, sfx_stem: np.ndarray, bgm_stem: np.ndarray) -> np.ndarray:
    voice_env = smooth_envelope(voice_stem, attack_ms=14.0, release_ms=230.0)
    activity = np.clip(voice_env / 0.055, 0.0, 1.0).astype(np.float32)

    bgm_floor = 10.0 ** (BGM_DUCK_DB / 20.0)
    bgm_duck = np.clip(1.0 - activity * (1.0 - bgm_floor), bgm_floor, 1.0).astype(np.float32)
    bgm_ducked = (bgm_stem * bgm_duck[:, None]).astype(np.float32)

    ev_guard = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
    for ev_key in KEY_TRANSIENT_EVENTS:
        c_idx = int(round((float(C.EV[ev_key]) / float(C.FPS)) * SR))
        w0 = max(0, c_idx - int(round(TRANSIENT_GUARD_PRE_S * SR)))
        w1 = min(TOTAL_SAMPLES, c_idx + int(round(TRANSIENT_GUARD_POST_S * SR)))
        if w1 > w0:
            ev_guard[w0:w1] = 1.0
    ev_guard = smooth_envelope(ev_guard, attack_ms=4.0, release_ms=95.0)
    bed_activity = np.clip(activity * (1.0 - 0.96 * ev_guard), 0.0, 1.0).astype(np.float32)

    nyq = 0.5 * SR
    bp_sos = signal.butter(2, [SFX_MID_BAND_HZ[0] / nyq, SFX_MID_BAND_HZ[1] / nyq], btype="band", output="sos")
    sfx_mid = signal.sosfiltfilt(bp_sos, sfx_stem.astype(np.float64), axis=0).astype(np.float32)
    sfx_rest = (sfx_stem - sfx_mid).astype(np.float32)

    mid_fl = 10.0 ** (SFX_BED_DUCK_DB / 20.0)
    rest_fl = 10.0 ** (SFX_REST_DUCK_DB / 20.0)
    sfx_ducked = (
        sfx_rest * np.clip(1.0 - bed_activity * (1.0 - rest_fl), rest_fl, 1.0)[:, None]
        + sfx_mid * np.clip(1.0 - bed_activity * (1.0 - mid_fl), mid_fl, 1.0)[:, None]
    ).astype(np.float32)

    summed = (voice_stem * VOICE_GAIN + bgm_ducked * BGM_GAIN + sfx_ducked * SFX_GAIN).astype(np.float32)
    mix = butter_filter(summed, "high", RUMBLE_HP_HZ, order=2)

    for _ in range(5):
        cur_lufs = measure_lufs_48k(mix)
        cur_tp = measure_true_peak_dbfs(mix)
        if abs(TARGET_LUFS - cur_lufs) <= 0.02 and cur_tp <= LIMITER_CEILING_DBFS + 0.02:
            break
        gain = 10.0 ** ((TARGET_LUFS - cur_lufs) / 20.0)
        mix = lookahead_peak_limiter((mix * gain).astype(np.float32), LIMITER_CEILING_LIN)

    micro = int(round(0.004 * SR))
    mix[:micro] *= np.linspace(0.0, 1.0, micro, dtype=np.float32)[:, None]
    mix[-micro:] *= np.linspace(1.0, 0.0, micro, dtype=np.float32)[:, None]
    tp_lin = float(np.max(np.abs(signal.resample_poly(mix.astype(np.float64), 4, 1, axis=0))))
    if tp_lin > LIMITER_CEILING_LIN:
        mix = (mix * float(LIMITER_CEILING_LIN / tp_lin)).astype(np.float32)
    return mix


def verify_master_output(master_path: Path) -> dict[str, Any]:
    info = sf.info(str(master_path))
    assert info.samplerate == SR and info.channels == CHANNELS and info.frames == TOTAL_SAMPLES
    assert info.subtype == "PCM_16" and abs(info.duration - DURATION_S) < 1e-6
    with wave.open(str(master_path), "rb") as wf:
        assert wf.getframerate() == SR and wf.getnchannels() == CHANNELS and wf.getsampwidth() == 2
    audio, _ = sf.read(str(master_path), dtype="float32")
    lufs = measure_lufs_48k(audio)
    tp = measure_true_peak_dbfs(audio)
    assert abs(lufs - TARGET_LUFS) <= LUFS_TOLERANCE, f"LUFS {lufs:.3f} out of spec"
    assert tp <= MAX_TRUE_PEAK_DBFS, f"True Peak {tp:.3f} dBFS > {MAX_TRUE_PEAK_DBFS}"
    print(f"[Master] Verified {master_path.name}: {SR} Hz 2ch PCM_16, {DURATION_S:.3f}s | LUFS={lufs:.3f} | TP={tp:.3f} dBFS")
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        res = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-nostats", "-i", str(master_path), "-af", "ebur128=peak=true", "-f", "null", "-"],
            capture_output=True, text=True, check=False,
        )
        if res.returncode == 0:
            summary = [ln.strip() for ln in res.stderr.splitlines() if ln.strip().startswith(("I:", "LRA:", "Peak:"))]
            if summary:
                print(f"[Master] FFmpeg ebur128 cross-check: {' | '.join(summary)}")
    return {"lufs": lufs, "true_peak_dbfs": tp}


def rebuild_all_components() -> None:
    for mod_name in ("gen_voice", "gen_sfx", "gen_bgm"):
        print(f"[Rebuild] Running {mod_name}.main()...")
        mod = importlib.import_module(mod_name)
        mod.main()
    sub_mod = importlib.import_module("gen_subtitles")
    sub_mod.main([])


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Master audio mixer for Episode #5 (-14.0 LUFS, TP <= -1.5 dBFS).")
    parser.add_argument("--rebuild-all", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)

    if args.rebuild_all:
        rebuild_all_components()

    tl = json.loads(TIMELINE_PATH.read_text(encoding="utf-8"))
    master_out = EPISODE_ROOT / tl["audio"]["masterPath"]
    master_out.parent.mkdir(parents=True, exist_ok=True)
    if args.verify_only:
        return verify_master_output(master_out)

    voice_stem = load_verified_stem(VOICE_STEM_PATH, "voice")
    sfx_stem = load_verified_stem(SFX_STEM_PATH, "sfx")
    bgm_stem = load_verified_stem(BGM_STEM_PATH, "bgm")
    mix = master_mix(voice_stem, sfx_stem, bgm_stem)
    sf.write(str(master_out), mix, SR, subtype="PCM_16")
    return verify_master_output(master_out)


if __name__ == "__main__":
    main()
