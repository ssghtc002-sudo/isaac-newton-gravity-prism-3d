"""
Master Mix Engineer for Episode #1
"Why Ejecting From a Fighter Jet Shrinks You by 1 Inch" (30.0 s @ 60 fps = 1800 frames).

Mixes the 3 pre-generated stems (voice_stem.wav, sfx_stem.wav, bgm_stem.wav) into:
    episodes/fighter-jet-ejection/audio/output/master_audio.wav
        - 48000 Hz, Stereo, PCM_16, exact 30.000 s (1,440,000 samples)
        - Smooth -9.5 dB sidechain ducking onto bgm_stem driven by narration envelope
        - Subtle -3.5 dB smart mid-range (320-3800 Hz) ducking onto sfx_stem while dialogue is active
        - 28 Hz high-pass rumble safety filter
        - ITU-R BS.1770 integrated loudness normalized to -14.0 +/- 0.15 LUFS
        - 4x oversampled True Peak brickwall limiter (<= -1.60 dBFS, strictly <= -1.50 dBFS)
        - Verified against timeline.json & voice_manifest.json

Default usage mixes existing stems without modifying them:
    python audio/generate_audio.py

Pass --rebuild-all to optionally re-run gen_voice, gen_sfx, gen_bgm, and gen_subtitles first:
    python audio/generate_audio.py --rebuild-all
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

import ejection_contract as C  # noqa: E402

TIMELINE_PATH = EPISODE_ROOT / "config" / "timeline.json"
OUTPUT_DIR = AUDIO_DIR / "output"
VOICE_STEM_PATH = OUTPUT_DIR / "voice_stem.wav"
SFX_STEM_PATH = OUTPUT_DIR / "sfx_stem.wav"
BGM_STEM_PATH = OUTPUT_DIR / "bgm_stem.wav"
MANIFEST_PATH = OUTPUT_DIR / "voice_manifest.json"
MASTER_PATH = OUTPUT_DIR / "master_audio.wav"

SR = 48000
CHANNELS = 2
DURATION_S = float(C.DURATION_S)
TOTAL_SAMPLES = int(round(DURATION_S * SR))

TARGET_LUFS = -14.0
LUFS_TOLERANCE = 0.15
LIMITER_CEILING_DBFS = -1.62
LIMITER_CEILING_LIN = 10.0 ** (LIMITER_CEILING_DBFS / 20.0)
MAX_TRUE_PEAK_DBFS = -1.50

VOICE_GAIN = 1.18
BGM_GAIN = 0.56
SFX_GAIN = 0.80

BGM_DUCK_DB = -9.5
SFX_MID_DUCK_DB = -3.5
SFX_MID_BAND_HZ = (320.0, 3800.0)
RUMBLE_HP_HZ = 28.0


def butter_filter(
    audio: np.ndarray,
    btype: str,
    cutoff: float | tuple[float, float],
    order: int = 2,
) -> np.ndarray:
    """Apply a Butterworth SOS filter along the time axis."""
    nyq = 0.5 * SR
    if isinstance(cutoff, tuple):
        wn: float | list[float] = [float(np.clip(c / nyq, 1e-4, 0.999)) for c in cutoff]
    else:
        wn = float(np.clip(cutoff / nyq, 1e-4, 0.999))
    sos = signal.butter(order, wn, btype=btype, output="sos")
    return signal.sosfilt(sos, audio, axis=0).astype(np.float32)


def smooth_envelope(sig: np.ndarray, attack_ms: float, release_ms: float) -> np.ndarray:
    """Fast block-peak attack/release envelope follower for sidechain and limiting."""
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


def measure_lufs_48k(stereo: np.ndarray) -> float:
    """Measure ITU-R BS.1770-4 K-weighted gated integrated loudness (LUFS) at 48 kHz."""
    b1 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285], dtype=np.float64)
    a1 = np.array([1.0, -1.69065929318241, 0.73248077421585], dtype=np.float64)
    b2 = np.array([1.0, -2.0, 1.0], dtype=np.float64)
    a2 = np.array([1.0, -1.99004745483398, 0.99007225036621], dtype=np.float64)
    y = signal.lfilter(b2, a2, signal.lfilter(b1, a1, stereo.astype(np.float64), axis=0), axis=0)
    bl, hl = int(0.400 * SR), int(0.100 * SR)
    p_arr = np.array(
        [
            float(np.mean(y[s : s + bl, 0] ** 2) + np.mean(y[s : s + bl, 1] ** 2))
            for s in range(0, len(y) - bl + 1, hl)
        ],
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
    """Measure 4x oversampled True Peak in dBFS."""
    up = signal.resample_poly(stereo.astype(np.float64), 4, 1, axis=0)
    tp_lin = max(1e-9, float(np.max(np.abs(up))))
    return 20.0 * math.log10(tp_lin)


def brickwall_limiter(stereo: np.ndarray, ceiling: float = LIMITER_CEILING_LIN) -> np.ndarray:
    """4x oversampled lookahead-style brickwall True Peak limiter."""
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


def load_verified_stem(path: Path, name: str) -> np.ndarray:
    """Load a stereo 48 kHz 30.000s stem and validate sample count and finiteness."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name} stem: {path} (run with --rebuild-all or generate stems first)")
    info = sf.info(str(path))
    if info.samplerate != SR or info.channels != CHANNELS or info.frames != TOTAL_SAMPLES:
        raise ValueError(
            f"{name} stem format mismatch ({path}): expected {SR}Hz {CHANNELS}ch {TOTAL_SAMPLES} samples, "
            f"got {info.samplerate}Hz {info.channels}ch {info.frames} samples"
        )
    audio, _ = sf.read(str(path), dtype="float32")
    if audio.shape != (TOTAL_SAMPLES, CHANNELS):
        raise ValueError(f"{name} shape mismatch: {audio.shape} != {(TOTAL_SAMPLES, CHANNELS)}")
    if not np.isfinite(audio).all():
        raise ValueError(f"{name} stem contains NaN or Inf values")
    return audio.astype(np.float32)


def apply_smart_ducking(
    voice_stem: np.ndarray,
    bgm_stem: np.ndarray,
    sfx_stem: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute dialogue activity envelope and apply:
      - Smooth -9.5 dB sidechain ducking onto bgm_stem
      - Smooth -3.5 dB zero-phase mid-range (320-3800 Hz) ducking onto sfx_stem
        so narration remains 100% intelligible during rocket/catapult SFX while
        preserving sub-bass punch and high mechanical transients.
    """
    env = smooth_envelope(voice_stem, attack_ms=15.0, release_ms=240.0)
    activity = np.clip(env / 0.06, 0.0, 1.0)

    bgm_floor = 10.0 ** (BGM_DUCK_DB / 20.0)
    bgm_duck = np.clip(1.0 - activity * (1.0 - bgm_floor), bgm_floor, 1.0).astype(np.float32)
    bgm_ducked = (bgm_stem * bgm_duck[:, None]).astype(np.float32)

    nyq = 0.5 * SR
    bp_sos = signal.butter(
        2,
        [SFX_MID_BAND_HZ[0] / nyq, SFX_MID_BAND_HZ[1] / nyq],
        btype="band",
        output="sos",
    )
    sfx_mid = signal.sosfiltfilt(bp_sos, sfx_stem.astype(np.float64), axis=0).astype(np.float32)
    sfx_rest = (sfx_stem - sfx_mid).astype(np.float32)

    sfx_mid_floor = 10.0 ** (SFX_MID_DUCK_DB / 20.0)
    sfx_mid_duck = np.clip(1.0 - activity * (1.0 - sfx_mid_floor), sfx_mid_floor, 1.0).astype(np.float32)
    sfx_ducked = (sfx_rest + sfx_mid * sfx_mid_duck[:, None]).astype(np.float32)

    return bgm_ducked, sfx_ducked, activity


def master_mix(voice_stem: np.ndarray, sfx_stem: np.ndarray, bgm_stem: np.ndarray) -> np.ndarray:
    """Duck stems, sum buses, apply 28 Hz HPF, normalize to -14.0 LUFS, and brickwall-limit."""
    bgm_ducked, sfx_ducked, _ = apply_smart_ducking(voice_stem, bgm_stem, sfx_stem)

    summed = (
        voice_stem * VOICE_GAIN
        + bgm_ducked * BGM_GAIN
        + sfx_ducked * SFX_GAIN
    ).astype(np.float32)

    mix = butter_filter(summed, "high", RUMBLE_HP_HZ, order=2)

    for _ in range(4):
        cur_lufs = measure_lufs_48k(mix)
        if abs(TARGET_LUFS - cur_lufs) <= 0.02 and measure_true_peak_dbfs(mix) <= -1.60:
            break
        gain = 10.0 ** ((TARGET_LUFS - cur_lufs) / 20.0)
        mix = brickwall_limiter((mix * gain).astype(np.float32), LIMITER_CEILING_LIN)

    # Click-free 4 ms boundary micro-fades
    micro = int(round(0.004 * SR))
    mix[:micro] *= np.linspace(0.0, 1.0, micro, dtype=np.float32)[:, None]
    mix[-micro:] *= np.linspace(1.0, 0.0, micro, dtype=np.float32)[:, None]

    tp_lin = float(np.max(np.abs(signal.resample_poly(mix.astype(np.float64), 4, 1, axis=0))))
    if tp_lin > LIMITER_CEILING_LIN:
        mix = (mix * float(LIMITER_CEILING_LIN / tp_lin)).astype(np.float32)

    return mix


def load_and_validate_metadata(timeline_path: Path, manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load timeline.json and voice_manifest.json and validate exact duration and line sync."""
    tl = json.loads(timeline_path.read_text(encoding="utf-8"))
    if abs(float(tl["duration"]) - DURATION_S) >= 1e-6:
        raise ValueError(f"timeline.json duration {tl['duration']} != {DURATION_S}")

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing voice_manifest.json: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if abs(float(manifest["duration"]) - DURATION_S) >= 1e-6:
        raise ValueError(f"voice_manifest.json duration {manifest['duration']} != {DURATION_S}")

    if len(manifest["lines"]) != len(tl["dialogue"]):
        raise ValueError(
            f"voice_manifest.json line count {len(manifest['lines'])} != timeline dialogue {len(tl['dialogue'])}"
        )

    prev_end = 0.0
    for src_item, ln in zip(tl["dialogue"], manifest["lines"]):
        w0, w1 = float(src_item["start"]), float(src_item["end"])
        ls, le = float(ln["start"]), float(ln["end"])
        if not (w0 - 1e-4 <= ls < le <= w1 + 1e-4):
            raise ValueError(f"Manifest line {ln['id']} [{ls}, {le}] outside timeline window [{w0}, {w1}]")
        if ls < prev_end - 1e-4:
            raise ValueError(f"Manifest line {ln['id']} overlaps previous line ({ls} < {prev_end})")
        prev_end = le

    return tl, manifest


def verify_master_output(
    master_path: Path,
    tl: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Read back PCM_16 master WAV and strictly verify specs, LUFS, True Peak, and manifest sync."""
    info = sf.info(str(master_path))
    assert info.samplerate == SR, f"Expected {SR} Hz, got {info.samplerate}"
    assert info.channels == CHANNELS, f"Expected {CHANNELS} channels, got {info.channels}"
    assert info.frames == TOTAL_SAMPLES, f"Expected {TOTAL_SAMPLES} samples, got {info.frames}"
    assert info.subtype == "PCM_16", f"Expected PCM_16, got {info.subtype}"
    assert abs(info.duration - DURATION_S) < 1e-6, f"Duration {info.duration:.6f}s != {DURATION_S:.6f}s"
    assert abs(info.duration - float(manifest["duration"])) < 1e-6, (
        f"Master duration {info.duration:.6f}s != voice_manifest.json duration {manifest['duration']}"
    )

    with wave.open(str(master_path), "rb") as wf:
        assert wf.getframerate() == SR
        assert wf.getnchannels() == CHANNELS
        assert wf.getnframes() == TOTAL_SAMPLES
        assert wf.getsampwidth() == 2

    audio, _ = sf.read(str(master_path), dtype="float32")
    assert audio.shape == (TOTAL_SAMPLES, CHANNELS)
    assert np.isfinite(audio).all(), "Master audio contains NaN or Inf"

    lufs = measure_lufs_48k(audio)
    tp_dbfs = measure_true_peak_dbfs(audio)
    sample_peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    rms_dbfs = 20.0 * math.log10(max(1e-9, rms))

    assert abs(lufs - TARGET_LUFS) <= LUFS_TOLERANCE, (
        f"Integrated loudness {lufs:.3f} LUFS outside {TARGET_LUFS:.1f} +/- {LUFS_TOLERANCE:.2f} LUFS"
    )
    assert tp_dbfs <= MAX_TRUE_PEAK_DBFS, (
        f"True Peak {tp_dbfs:.3f} dBFS exceeded <= {MAX_TRUE_PEAK_DBFS:.2f} dBFS"
    )

    print(
        f"[Master] Verified {master_path.name}: {info.samplerate} Hz, {info.channels}ch {info.subtype}, "
        f"{info.frames:,} samples ({info.duration:.3f}s) | "
        f"LUFS={lufs:.3f} | TruePeak={tp_dbfs:.3f} dBFS | "
        f"RMS={rms:.4f} ({rms_dbfs:.2f} dBFS) | SamplePeak={sample_peak:.4f}"
    )

    for bt in tl["beats"]:
        i0 = int(round(float(bt["start"]) * SR))
        i1 = int(round(float(bt["end"]) * SR))
        seg = audio[i0:i1]
        seg_rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        seg_pk = float(np.max(np.abs(seg)))
        print(
            f"  {bt['id']} ({float(bt['start']):5.2f}-{float(bt['end']):5.2f}s): "
            f"rms={seg_rms:.4f} ({20.0 * math.log10(max(1e-9, seg_rms)):+6.2f} dBFS), peak={seg_pk:.4f}"
        )

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        res = subprocess.run(
            [
                ffmpeg_bin,
                "-hide_banner",
                "-nostats",
                "-i",
                str(master_path),
                "-af",
                "ebur128=peak=true",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            summary = [ln.strip() for ln in res.stderr.splitlines() if ln.strip().startswith(("I:", "LRA:", "Peak:"))]
            if summary:
                print(f"[Master] FFmpeg ebur128 cross-check: {' | '.join(summary)}")

    return {
        "samplerate": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,
        "samples": info.frames,
        "duration": info.duration,
        "lufs": lufs,
        "true_peak_dbfs": tp_dbfs,
        "rms": rms,
        "rms_dbfs": rms_dbfs,
        "sample_peak": sample_peak,
    }


def rebuild_all_components() -> None:
    """Optionally regenerate voice, sfx, bgm, and subtitles via their standalone modules."""
    for mod_name in ("gen_voice", "gen_sfx", "gen_bgm"):
        print(f"[Rebuild] Running {mod_name}.main()...")
        mod = importlib.import_module(mod_name)
        mod.main()

    sub_script = AUDIO_DIR / "gen_subtitles.py"
    if sub_script.is_file():
        print("[Rebuild] Running gen_subtitles.main()...")
        sub_mod = importlib.import_module("gen_subtitles")
        sub_mod.main()
    else:
        print("[Rebuild] Note: audio/gen_subtitles.py not present yet; skipping subtitle regeneration.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Master audio mixer for Episode #1 (48kHz Stereo PCM_16, -14.0 LUFS, TP <= -1.5 dBFS)."
    )
    parser.add_argument(
        "--rebuild-all",
        action="store_true",
        help="Regenerate voice, sfx, bgm, and subtitles before mixing master_audio.wav.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.rebuild_all:
        rebuild_all_components()

    tl, manifest = load_and_validate_metadata(TIMELINE_PATH, MANIFEST_PATH)
    master_out = EPISODE_ROOT / tl["audio"]["masterPath"]
    master_out.parent.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading verified 48 kHz stereo stems (voice, sfx, bgm)...")
    voice_stem = load_verified_stem(VOICE_STEM_PATH, "voice")
    sfx_stem = load_verified_stem(SFX_STEM_PATH, "sfx")
    bgm_stem = load_verified_stem(BGM_STEM_PATH, "bgm")

    print(
        f"[2/3] Mixing with {BGM_DUCK_DB:+.1f} dB BGM sidechain ducking, "
        f"{SFX_MID_DUCK_DB:+.1f} dB SFX mid-band ({SFX_MID_BAND_HZ[0]:.0f}-{SFX_MID_BAND_HZ[1]:.0f} Hz) "
        f"clarity ducking, {RUMBLE_HP_HZ:.0f} Hz HPF, -14.0 LUFS norm & 4x TP limiter ({LIMITER_CEILING_DBFS:.2f} dBFS)..."
    )
    mix = master_mix(voice_stem, sfx_stem, bgm_stem)

    print(f"[3/3] Writing {master_out} (48000 Hz, Stereo, PCM_16) and verifying...")
    sf.write(str(master_out), mix, SR, subtype="PCM_16")
    return verify_master_output(master_out, tl, manifest)


if __name__ == "__main__":
    main()
