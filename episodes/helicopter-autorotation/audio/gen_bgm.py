"""
gen_bgm.py — Cinematic Golden-Hour Aviation Tension Pulse BGM Composer for Episode #5
("Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock", 30.0 s @ 60 fps).

Composes a clean 48 kHz stereo score in D minor (108 BPM) with:
  - Urgent muted low-register tension pulse during Beat 1–2 engine failure & X-ray clutch reveal
  - Soaring aerodynamic harmonic lift through Beat 3–4 upward airflow & maple seed glide
  - High-stakes rhythmic swell into Beat 5 30-foot flare, resolving cleanly into a warm triumphant
    D-major/sus2 sunset chord on Beat 6 rooftop touchdown!
  - Built-in vocal pocket EQ carve (250–3500 Hz) so deadpan narration cuts through effortlessly.
"""
from __future__ import annotations

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

BGM_STEM_PATH = AUDIO_DIR / "output" / "bgm_stem.wav"

SR = 48000
CHANNELS = 2
DURATION_S = float(C.DURATION_S)
TOTAL_SAMPLES = int(round(DURATION_S * SR))


def _peaking_eq_stereo(audio: np.ndarray, freq: float, q: float, gain_db: float) -> np.ndarray:
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / SR
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    norm = 1.0 + alpha / a_lin
    b = np.array([1.0 + alpha * a_lin, -2.0 * cw, 1.0 - alpha * a_lin], dtype=np.float64) / norm
    a = np.array([1.0, (-2.0 * cw) / norm, (1.0 - alpha / a_lin) / norm], dtype=np.float64)
    return signal.lfilter(b, a, audio.astype(np.float64), axis=0).astype(np.float32)


def main() -> None:
    t = np.linspace(0.0, DURATION_S, TOTAL_SAMPLES, dtype=np.float64)
    left = np.zeros(TOTAL_SAMPLES, dtype=np.float64)
    right = np.zeros(TOTAL_SAMPLES, dtype=np.float64)

    # Chord progression across the 6 beats (D minor -> Bb maj7 -> C sus2 -> A min7 -> Bb -> D major sunset payoff!)
    sections = [
        (0.0, 4.0, (73.42, 146.83, 220.00, 261.63)),    # Beat 1: Dm7 tension
        (4.0, 9.5, (58.27, 116.54, 174.61, 220.00)),    # Beat 2: Bbmaj7 mechanical awe
        (9.5, 16.0, (65.41, 130.81, 196.00, 293.66)),   # Beat 3: Csus2 soaring lift
        (16.0, 21.5, (55.00, 110.00, 164.81, 261.63)),  # Beat 4: Am7 maple-seed grace
        (21.5, 26.3, (58.27, 116.54, 174.61, 261.63)),  # Beat 5: Bbadd9 flare climax
        (26.3, 30.0, (73.42, 146.83, 220.00, 293.66)),  # Beat 6: D5/Dmaj warm sunset touchdown!
    ]

    for t0, t1, freqs in sections:
        i0, i1 = int(t0 * SR), min(TOTAL_SAMPLES, int(t1 * SR))
        loc = t[i0:i1] - t0
        dur = max(0.1, t1 - t0)
        fade = np.minimum(1.0, loc / 0.35) * np.minimum(1.0, (dur - loc) / 0.35)
        pad_l = np.zeros_like(loc)
        pad_r = np.zeros_like(loc)
        for k, f0 in enumerate(freqs):
            det = 1.0 + (0.0022 if k % 2 == 0 else -0.0022)
            w_l = np.sin(2.0 * np.pi * f0 * loc) + 0.28 * np.sin(4.0 * np.pi * f0 * loc)
            w_r = np.sin(2.0 * np.pi * (f0 * det) * loc + 0.4) + 0.28 * np.sin(4.0 * np.pi * (f0 * det) * loc)
            wt = 0.30 if k == 0 else (0.18 if k == 1 else 0.12)
            pad_l += w_l * wt
            pad_r += w_r * wt
        left[i0:i1] += pad_l * fade
        right[i0:i1] += pad_r * fade

        # Rhythmic 108 BPM eighth-note muted tension pulse (drops gently on Beat 6 smooth landing!)
        if t0 < 26.0:
            step_s = 60.0 / 108.0 / 2.0
            n_steps = int(dur / step_s)
            f_bass = freqs[0]
            for s in range(n_steps):
                ps = i0 + int(s * step_s * SR)
                pe = min(i1, ps + int(0.16 * SR))
                if pe <= ps:
                    continue
                lt = np.linspace(0.0, (pe - ps) / float(SR), pe - ps, dtype=np.float64)
                acc = 1.0 if (s % 4 == 0) else 0.58
                plk = np.sin(2.0 * np.pi * f_bass * lt) * np.exp(-16.0 * lt) * 0.36 * acc
                left[ps:pe] += plk
                right[ps:pe] += plk

    stereo = np.stack([left, right], axis=-1).astype(np.float32)
    # Broadcast vocal-pocket EQ carve (250-3500 Hz) so neural narration sits front-and-center!
    stereo = _peaking_eq_stereo(stereo, 1150.0, 0.75, -5.5)
    sos_lp = signal.butter(2, 6500.0 / (0.5 * SR), btype="low", output="sos")
    stereo = signal.sosfilt(sos_lp, stereo, axis=0).astype(np.float32)

    rms = float(np.sqrt(np.mean(stereo.astype(np.float64) ** 2) + 1e-12))
    stereo *= 0.14 / max(1e-5, rms)
    peak = float(np.max(np.abs(stereo)))
    if peak > 0.72:
        stereo *= 0.72 / peak

    fade_in = int(0.12 * SR)
    fade_out = int(0.65 * SR)
    stereo[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)[:, None]
    stereo[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)[:, None]

    BGM_STEM_PATH.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(BGM_STEM_PATH), stereo, SR, subtype="PCM_16")
    print(f"[BGM] Generated {BGM_STEM_PATH.name} (D minor -> D major sunset payoff, peak={float(np.max(np.abs(stereo))):.3f})")


if __name__ == "__main__":
    main()
