"""
Master Mix Engineer for Episode #4
"How a 172-Foot Stealth Bomber Looks Like a Bumblebee on Radar" (30.0 s @ 60 fps = 1800 frames).

Mixes the 3 pre-generated 48 kHz stereo stems (voice_stem.wav, bgm_stem.wav, sfx_stem.wav) into:
    episodes/stealth-bomber-radar/audio/output/master_audio.wav
        - 48000 Hz, Stereo, PCM_16, exact 30.000 s (1,440,000 samples)
        - Smooth -9.5 dB sidechain ducking onto bgm_stem driven by voice_stem envelope
        - Impact-preserving SFX gain (`SFX_GAIN = 1.10`) + 360 ms transient guard
          protecting every radar pulse, echo alarm, X-ray scan, RCS sub-drop & scope blip
        - 28 Hz high-pass rumble safety filter
        - ITU-R BS.1770-4 integrated loudness normalized to -14.0 +/- 0.15 LUFS
        - 4x oversampled soft lookahead peak limiter (True Peak <= -1.50 dBFS, zero clipping)
        - Verified against stealth_contract.py, timeline.json, voice_manifest.json,
          and FFmpeg ebur128=peak=true

Usage:
    python3 episodes/stealth-bomber-radar/audio/generate_audio.py --rebuild-all
    python3 episodes/stealth-bomber-radar/audio/generate_audio.py
    python3 episodes/stealth-bomber-radar/audio/generate_audio.py --verify-only
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

import stealth_contract as C  # noqa: E402

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
SFX_GAIN: float = 1.10

BGM_DUCK_DB: float = -9.5
SFX_BED_DUCK_DB: float = -1.8
SFX_REST_DUCK_DB: float = -0.8
SFX_MID_BAND_HZ: tuple[float, float] = (250.0, 3800.0)
RUMBLE_HP_HZ: float = 28.0

TRANSIENT_GUARD_PRE_S: float = 0.035
TRANSIENT_GUARD_POST_S: float = 0.360

KEY_TRANSIENT_EVENTS: tuple[str, ...] = (
    "b1_radar_ping",
    "b1_wing_deflect",
    "b2_conv_ping",
    "b2_conv_echo",
    "b2_stealth_deflect",
    "b3_xray_sweep",
    "b3_duct_enter",
    "b3_duct_trap",
    "b4_ram_wave",
    "b4_rcs_shrink",
    "b5_mirror_echo",
    "b5_facet_scatter",
    "b6_moon_pass",
    "b6_final_blip",
)


def butter_filter(
    audio: np.ndarray,
    btype: str,
    cutoff: float | tuple[float, float],
    order: int = 2,
) -> np.ndarray:
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
    up = signal.resample_poly(stereo.astype(np.float64), 4, 1, axis=0)
    tp_lin = max(1e-9, float(np.max(np.abs(up))))
    return 20.0 * math.log10(tp_lin)


def lookahead_peak_limiter(stereo: np.ndarray, ceiling: float = LIMITER_CEILING_LIN) -> np.ndarray:
    """4x oversampled soft lookahead brickwall True Peak limiter (zero clipping)."""
    out = stereo.copy().astype(np.float32)
    lookahead_samples = int(round(0.0025 * SR))
    for _ in range(3):
        up = signal.resample_poly(out.astype(np.float64), 4, 1, axis=0)
        pk = np.max(np.abs(up), axis=1).reshape(-1, 4).max(axis=1)[: len(out)].astype(np.float32)
        padded = np.pad(pk, (0, lookahead_samples), mode="edge")
        la_pk = np.maximum.reduce([padded[k : k + len(pk)] for k in range(0, lookahead_samples + 1, 16)])
        env = np.maximum(smooth_envelope(la_pk, 0.8, 55.0), pk)
        gain_curve = np.where(env > ceiling, ceiling / np.maximum(env, 1e-6), 1.0).astype(np.float32)
        out *= gain_curve[:, None]
    tp = float(np.max(np.abs(signal.resample_poly(out.astype(np.float64), 4, 1, axis=0))))
    if tp > ceiling:
        out *= float(ceiling / tp)
    return out.astype(np.float32)


def load_verified_stem(path: Path, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name} stem: {path} (run with --rebuild-all first)")
    info = sf.info(str(path))
    if info.samplerate != SR or info.channels != CHANNELS or info.frames != TOTAL_SAMPLES:
        raise ValueError(
            f"{name} stem format mismatch ({path}): expected {SR}Hz {CHANNELS}ch {TOTAL_SAMPLES} samples, "
            f"got {info.samplerate}Hz {info.channels}ch {info.frames} samples"
        )
    audio, _ = sf.read(str(path), dtype="float32")
    if audio.shape != (TOTAL_SAMPLES, CHANNELS) or not np.isfinite(audio).all():
        raise ValueError(f"{name} stem invalid shape or non-finite values")
    return audio.astype(np.float32)


def apply_smart_ducking(
    voice_stem: np.ndarray,
    bgm_stem: np.ndarray,
    sfx_stem: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Smooth -9.5 dB BGM sidechain ducking + transient-preserving SFX ducking
    with 360 ms impact guard on all KEY_TRANSIENT_EVENTS.
    """
    voice_env = smooth_envelope(voice_stem, attack_ms=14.0, release_ms=230.0)
    activity = np.clip(voice_env / 0.055, 0.0, 1.0).astype(np.float32)

    bgm_floor = 10.0 ** (BGM_DUCK_DB / 20.0)
    bgm_duck = np.clip(1.0 - activity * (1.0 - bgm_floor), bgm_floor, 1.0).astype(np.float32)
    bgm_ducked = (bgm_stem * bgm_duck[:, None]).astype(np.float32)

    sfx_fast = smooth_envelope(sfx_stem, attack_ms=1.2, release_ms=55.0)
    sfx_slow = smooth_envelope(sfx_stem, attack_ms=55.0, release_ms=280.0)
    raw_trans = np.clip((sfx_fast - 1.25 * sfx_slow) / 0.12, 0.0, 1.0).astype(np.float32)

    ev_guard = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
    for ev_key in KEY_TRANSIENT_EVENTS:
        c_idx = int(round((float(C.EV[ev_key]) / float(C.FPS)) * SR))
        w0 = max(0, c_idx - int(round(TRANSIENT_GUARD_PRE_S * SR)))
        w1 = min(TOTAL_SAMPLES, c_idx + int(round(TRANSIENT_GUARD_POST_S * SR)))
        if w1 > w0:
            ev_guard[w0:w1] = 1.0
    ev_guard = smooth_envelope(ev_guard, attack_ms=4.0, release_ms=95.0)

    transient_guard = np.maximum(
        smooth_envelope(raw_trans, attack_ms=1.0, release_ms=85.0),
        ev_guard,
    ).astype(np.float32)

    bed_activity = np.clip(activity * (1.0 - 0.96 * transient_guard), 0.0, 1.0).astype(np.float32)

    nyq = 0.5 * SR
    bp_sos = signal.butter(
        2,
        [SFX_MID_BAND_HZ[0] / nyq, SFX_MID_BAND_HZ[1] / nyq],
        btype="band",
        output="sos",
    )
    sfx_mid = signal.sosfiltfilt(bp_sos, sfx_stem.astype(np.float64), axis=0).astype(np.float32)
    sfx_rest = (sfx_stem - sfx_mid).astype(np.float32)

    sfx_mid_floor = 10.0 ** (SFX_BED_DUCK_DB / 20.0)
    sfx_rest_floor = 10.0 ** (SFX_REST_DUCK_DB / 20.0)
    sfx_mid_duck = np.clip(1.0 - bed_activity * (1.0 - sfx_mid_floor), sfx_mid_floor, 1.0).astype(np.float32)
    sfx_rest_duck = np.clip(1.0 - bed_activity * (1.0 - sfx_rest_floor), sfx_rest_floor, 1.0).astype(np.float32)
    sfx_ducked = (sfx_rest * sfx_rest_duck[:, None] + sfx_mid * sfx_mid_duck[:, None]).astype(np.float32)

    return bgm_ducked, sfx_ducked, {
        "activity": activity,
        "bgm_duck": bgm_duck,
        "transient_guard": transient_guard,
        "sfx_mid_duck": sfx_mid_duck,
    }


def master_mix(
    voice_stem: np.ndarray,
    sfx_stem: np.ndarray,
    bgm_stem: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    bgm_ducked, sfx_ducked, telemetry = apply_smart_ducking(voice_stem, bgm_stem, sfx_stem)
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
    return mix, telemetry


def load_and_validate_metadata(timeline_path: Path, manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    tl = json.loads(timeline_path.read_text(encoding="utf-8"))
    if abs(float(tl["duration"]) - DURATION_S) >= 1e-6:
        raise ValueError(f"timeline.json duration {tl['duration']} != {DURATION_S}")
    if int(tl["fps"]) != int(C.FPS) or int(tl["totalFrames"]) != int(C.TOTAL_FRAMES):
        raise ValueError("timeline.json FPS/frames mismatch with stealth_contract")

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing voice_manifest.json: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if abs(float(manifest["duration"]) - DURATION_S) >= 1e-6:
        raise ValueError(f"voice_manifest.json duration {manifest['duration']} != {DURATION_S}")

    tl_lines = tl.get("narration_lines", [])
    if len(manifest["lines"]) != len(tl_lines):
        raise ValueError("voice_manifest.json line count mismatch")

    prev_end = 0.0
    for src_item, ln in zip(tl_lines, manifest["lines"]):
        w0, w1 = float(src_item["t0"]), float(src_item["t1"])
        ls, le = float(ln["t0"]), float(ln["t1"])
        if not (w0 - 1e-4 <= ls < le <= w1 + 1e-4):
            raise ValueError(f"Manifest line {ln['id']} [{ls}, {le}] outside [{w0}, {w1}]")
        if ls < prev_end - 1e-4:
            raise ValueError(f"Manifest line {ln['id']} overlaps previous line")
        prev_end = le
    return tl, manifest


def log_transient_audit(telemetry: dict[str, np.ndarray], mix: np.ndarray) -> None:
    print("[Mix] Transient preservation audit (stealth_contract.EV):")
    for ev_key in KEY_TRANSIENT_EVENTS:
        t_ev = float(C.EV[ev_key]) / float(C.FPS)
        i0 = max(0, int(round((t_ev - 0.025) * SR)))
        i1 = min(TOTAL_SAMPLES, int(round((t_ev + 0.140) * SR)))
        tg = float(np.max(telemetry["transient_guard"][i0:i1]))
        sfx_db = 20.0 * math.log10(max(1e-6, float(np.max(telemetry["sfx_mid_duck"][i0:i1]))))
        bgm_db = 20.0 * math.log10(max(1e-6, float(np.min(telemetry["bgm_duck"][i0:i1]))))
        seg_pk = float(np.max(np.abs(mix[i0:i1])))
        print(
            f"  {ev_key:20s} @ {t_ev:5.2f}s (f={C.EV[ev_key]:4d}): "
            f"guard={tg:.3f}, sfx_duck={sfx_db:+5.2f} dB, bgm_duck={bgm_db:+5.2f} dB, pk={seg_pk:.3f}"
        )


def verify_master_output(master_path: Path, tl: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    info = sf.info(str(master_path))
    assert info.samplerate == SR and info.channels == CHANNELS and info.frames == TOTAL_SAMPLES
    assert info.subtype == "PCM_16" and abs(info.duration - DURATION_S) < 1e-6
    assert abs(info.duration - float(manifest["duration"])) < 1e-6

    with wave.open(str(master_path), "rb") as wf:
        assert wf.getframerate() == SR and wf.getnchannels() == CHANNELS and wf.getsampwidth() == 2

    audio, _ = sf.read(str(master_path), dtype="float32")
    assert audio.shape == (TOTAL_SAMPLES, CHANNELS) and np.isfinite(audio).all()

    lufs = measure_lufs_48k(audio)
    tp_dbfs = measure_true_peak_dbfs(audio)
    sample_peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))

    assert abs(lufs - TARGET_LUFS) <= LUFS_TOLERANCE, f"LUFS {lufs:.3f} out of spec"
    assert tp_dbfs <= MAX_TRUE_PEAK_DBFS, f"True Peak {tp_dbfs:.3f} dBFS > {MAX_TRUE_PEAK_DBFS}"
    assert sample_peak < 1.0

    print(
        f"[Master] Verified {master_path.name}: {info.samplerate} Hz, {info.channels}ch {info.subtype}, "
        f"{info.frames:,} samples ({info.duration:.3f}s) | LUFS={lufs:.3f} | TruePeak={tp_dbfs:.3f} dBFS"
    )

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

    return {
        "samplerate": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,
        "samples": info.frames,
        "duration": info.duration,
        "lufs": lufs,
        "true_peak_dbfs": tp_dbfs,
        "rms": rms,
        "sample_peak": sample_peak,
    }


def rebuild_all_components() -> None:
    for mod_name in ("gen_voice", "gen_sfx", "gen_bgm"):
        print(f"[Rebuild] Running {mod_name}.main()...")
        mod = importlib.import_module(mod_name)
        mod.main()

    sub_path = AUDIO_DIR / "gen_subtitles.py"
    if sub_path.is_file():
        print("[Rebuild] Running audio/gen_subtitles.py...")
        sub_mod = importlib.import_module("gen_subtitles")
        sub_mod.main()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Master audio mixer for Episode #4 (48kHz Stereo PCM_16, -14.0 LUFS, TP <= -1.5 dBFS)."
    )
    parser.add_argument("--rebuild-all", action="store_true", help="Regenerate voice, sfx, bgm, and subtitles first.")
    parser.add_argument("--verify-only", action="store_true", help="Validate stems and existing master_audio.wav.")
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

    if args.verify_only:
        return verify_master_output(master_out, tl, manifest)

    print(
        f"[2/3] Mixing with {BGM_DUCK_DB:+.1f} dB BGM ducking, SFX_GAIN={SFX_GAIN:.2f}, "
        f"transient_guard={int(TRANSIENT_GUARD_POST_S * 1000)}ms, -14.0 LUFS & 4x lookahead limiter..."
    )
    mix, telemetry = master_mix(voice_stem, sfx_stem, bgm_stem)
    log_transient_audit(telemetry, mix)

    print(f"[3/3] Writing {master_out} (48000 Hz, Stereo, PCM_16) and verifying...")
    sf.write(str(master_out), mix, SR, subtype="PCM_16")
    return verify_master_output(master_out, tl, manifest)


if __name__ == "__main__":
    main()
