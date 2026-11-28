"""
gen_bgm.py — Hollywood Cinematic Suspense Orchestral-Hybrid Score Composer (V2 Ultimate)
for Episode #5: "Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock" (30.0 s @ 60 fps).

V2 Ultimate Score Architecture (D minor -> D major9 golden-hour resolution, 112 BPM):
  - Formant-filtered multi-detuned orchestral strings (dark cellos + warm violas + high suspense tremolo)
  - Massive cinematic Hans Zimmer-style sub-bass & brass BRAAM impacts synced to:
      * t = 0.30 s (Beat 1A Dorsal Turbine Explosion!)
      * t = 24.37 s (Beat 5B 30-Foot Nose-Up Flare!)
  - Driving 112 BPM hybrid tension ostinato + crisp mechanical ticking percussion (`0.0..24.02 s`)
  - Dramatic 0.35-second musical VACUUM DROP (`t = 24.02..24.37 s`) right before the 30-foot flare
    so the flare blade-bite + brass Braam slams in with pure cinema goosebumps!
  - Lush open D-major9 golden-hour touchdown resolution chord on Beat 6 (`26.33..30.00 s`)
  - Broadcast vocal-pocket EQ carve (250-3500 Hz) so neural narration cuts through crystal clear.
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
BPM = 112.0


def _peaking_eq_stereo(audio: np.ndarray, freq: float, q: float, gain_db: float) -> np.ndarray:
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / SR
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    norm = 1.0 + alpha / a_lin
    b = np.array([1.0 + alpha * a_lin, -2.0 * cw, 1.0 - alpha * a_lin], dtype=np.float64) / norm
    a = np.array([1.0, (-2.0 * cw) / norm, (1.0 - alpha / a_lin) / norm], dtype=np.float64)
    return signal.lfilter(b, a, audio.astype(np.float64), axis=0).astype(np.float32)


def _rich_string_voice(t: np.ndarray, freq: float, vib_rate: float = 4.6, vib_cents: float = 6.0) -> np.ndarray:
    """Band-limited bowed-string harmonic series (1..7) with natural vibrato (zero harsh aliasing)."""
    vib = 1.0 + (vib_cents / 1200.0) * np.sin(2.0 * np.pi * vib_rate * t)
    ph = 2.0 * np.pi * np.cumsum(freq * vib) / SR
    out = np.zeros_like(t)
    # Warm cello/string harmonic falloff
    for h, w in ((1, 1.00), (2, 0.56), (3, 0.34), (4, 0.22), (5, 0.14), (6, 0.08)):
        if freq * h < 7200.0:
            out += w * np.sin(h * ph)
    return out


def _add_braam(left: np.ndarray, right: np.ndarray, t0: float, f_root: float, dur: float, gain: float) -> None:
    """Cinematic low-brass / sub-bass BRAAM impact with warm filter sweep & stereo chorus."""
    i0 = int(round(t0 * SR))
    n = int(round(dur * SR))
    if i0 >= len(left) or n <= 0:
        return
    n = min(n, len(left) - i0)
    lt = np.linspace(0.0, n / float(SR), n, dtype=np.float64)
    env = np.minimum(1.0, lt / 0.045) * np.exp(-1.85 * lt)
    # Deep sub + detuned brass-saw harmonic stack
    sub = np.sin(2.0 * np.pi * f_root * lt) * 0.95
    brass_l = np.zeros_like(lt)
    brass_r = np.zeros_like(lt)
    for h in range(1, 9):
        wt = 0.55 / (h ** 0.72)
        brass_l += wt * np.sin(2.0 * np.pi * (f_root * 2.0 * h * 0.9985) * lt)
        brass_r += wt * np.sin(2.0 * np.pi * (f_root * 2.0 * h * 1.0015) * lt + 0.3)
    sos = signal.butter(3, 620.0 / (0.5 * SR), btype="low", output="sos")
    bl = signal.sosfilt(sos, np.tanh(brass_l * 1.45)) * 0.75 + sub
    br = signal.sosfilt(sos, np.tanh(brass_r * 1.45)) * 0.75 + sub
    left[i0 : i0 + n] += bl * env * gain
    right[i0 : i0 + n] += br * env * gain


def main() -> None:
    rng = np.random.default_rng(5059)
    t = np.linspace(0.0, DURATION_S, TOTAL_SAMPLES, dtype=np.float64)
    left = np.zeros(TOTAL_SAMPLES, dtype=np.float64)
    right = np.zeros(TOTAL_SAMPLES, dtype=np.float64)

    # ------------------------------------------------------------------
    # 1. ORCHESTRAL STRING ENSEMBLE PROGRESSION (D minor -> D major9!)
    # ------------------------------------------------------------------
    sections = [
        # (t0, t1, chord_freqs_hz, intensity, add_high_tremolo)
        (0.00, 4.00, (73.42, 146.83, 220.00, 261.63, 349.23), 1.05, True),    # Beat 1: Dm7 high-stakes emergency!
        (4.00, 9.50, (58.27, 116.54, 174.61, 220.00, 293.66), 0.92, False),   # Beat 2: Bbmaj7 mechanical reveal
        (9.50, 16.00, (65.41, 130.81, 196.00, 293.66, 392.00), 1.00, False),  # Beat 3: Csus2 soaring upward lift
        (16.00, 21.50, (55.00, 110.00, 164.81, 261.63, 329.63), 0.95, False), # Beat 4: Am7 graceful maple glide
        (21.50, 24.04, (58.27, 116.54, 174.61, 233.08, 349.23), 1.15, True),  # Beat 5A: Bb5 urgent 30-ft countdown!
        (24.36, 26.35, (65.41, 130.81, 196.00, 293.66, 440.00), 1.18, False), # Beat 5B: C6/9 huge flare lift!
        (26.35, 30.00, (73.42, 146.83, 220.00, 293.66, 369.99, 659.25), 1.08, False),  # Beat 6: Lush Dmaj9 sunset!
    ]

    for t0, t1, freqs, intensity, trem in sections:
        i0 = int(round(t0 * SR))
        i1 = min(TOTAL_SAMPLES, int(round(t1 * SR)))
        if i1 <= i0:
            continue
        loc = t[i0:i1] - t0
        dur = max(0.1, t1 - t0)
        fade_in_s = 0.08 if t0 == 24.36 else 0.28
        fade_out_s = 0.10 if t1 == 24.04 else 0.32
        fade = np.minimum(1.0, loc / fade_in_s) * np.minimum(1.0, (dur - loc) / fade_out_s)

        str_l = np.zeros_like(loc)
        str_r = np.zeros_like(loc)
        for k, f0 in enumerate(freqs):
            wt = 0.32 if k == 0 else (0.20 if k == 1 else (0.14 if k <= 3 else 0.10))
            v_l = _rich_string_voice(loc, f0 * (1.0 - 0.0018), vib_rate=4.3 + 0.3 * k)
            v_r = _rich_string_voice(loc, f0 * (1.0 + 0.0018), vib_rate=4.7 + 0.25 * k)
            str_l += v_l * wt
            str_r += v_r * wt

        # Urgent high-register bowed tremolo violin pedal during Beat 1B & Beat 5A!
        if trem:
            trem_mod = 0.65 + 0.35 * np.sin(2.0 * np.pi * 11.5 * loc)
            hi_l = _rich_string_voice(loc, 587.33, vib_rate=5.8, vib_cents=8.0) * trem_mod * 0.09
            hi_r = _rich_string_voice(loc, 588.10, vib_rate=6.1, vib_cents=8.0) * trem_mod * 0.09
            str_l += hi_l
            str_r += hi_r

        left[i0:i1] += str_l * fade * intensity
        right[i0:i1] += str_r * fade * intensity

    # ------------------------------------------------------------------
    # 2. DRIVING 112 BPM HYBRID OSTINATO & MECHANICAL TICKING PERCUSSION
    # ------------------------------------------------------------------
    step_s = 60.0 / BPM / 2.0  # crisp eighth-note tension drive (~0.268 s)
    # Build bass root lookup by time
    def _root_at(sec: float) -> float:
        if sec < 4.0:
            return 73.42
        if sec < 9.5:
            return 58.27
        if sec < 16.0:
            return 65.41
        if sec < 21.5:
            return 55.00
        if sec < 24.04:
            return 58.27
        return 65.41

    n_steps = int(26.20 / step_s)
    for s in range(n_steps):
        ts = s * step_s
        # Respect the dramatic 0.35s pre-flare musical vacuum (24.02..24.36s)!
        if 24.00 <= ts <= 24.36:
            continue
        ps = int(round(ts * SR))
        pe = min(TOTAL_SAMPLES, ps + int(0.18 * SR))
        if pe <= ps:
            continue
        lt = np.linspace(0.0, (pe - ps) / float(SR), pe - ps, dtype=np.float64)
        f_b = _root_at(ts)
        acc = 1.0 if (s % 4 == 0) else (0.72 if (s % 2 == 0) else 0.48)
        # Tight analog-hybrid bass pulse (fundamental + warm octave pluck)
        plk = (
            0.78 * np.sin(2.0 * np.pi * f_b * lt)
            + 0.32 * np.sin(4.0 * np.pi * f_b * lt)
        ) * np.exp(-15.5 * lt) * 0.42 * acc
        # Subtle Hans Zimmer-style mechanical ticking percussion (high-passed micro-click)
        tick_len = min(len(lt), int(0.025 * SR))
        tick = np.zeros_like(lt)
        if tick_len > 8:
            t_loc = lt[:tick_len]
            tick[:tick_len] = (
                rng.standard_normal(tick_len) * np.exp(-190.0 * t_loc) * (0.14 if s % 2 == 0 else 0.08)
            )
        pan = 0.22 if (s % 2 == 0) else -0.22
        left[ps:pe] += plk + tick * (1.0 - pan)
        right[ps:pe] += plk + tick * (1.0 + pan)

    # ------------------------------------------------------------------
    # 3. CINEMATIC BRASS / SUB-BASS BRAAM IMPACTS + REVERSED PRE-FLARE INTAKE
    # ------------------------------------------------------------------
    # Beat 1A Turbine Explosion Braam (t=0.30s)
    _add_braam(left, right, t0=0.30, f_root=36.71, dur=1.55, gain=0.62)
    # Beat 2 X-Ray Reveal low-pulse anchor (t=4.05s)
    _add_braam(left, right, t0=4.05, f_root=58.27, dur=1.10, gain=0.36)
    # Beat 5B Post-Vacuum 30-Foot Flare Heroic Braam (t=24.37s!)
    _add_braam(left, right, t0=24.37, f_root=43.65, dur=1.65, gain=0.72)

    # Subtle reversed air-intake riser inside the 0.35s vacuum drop (`24.06..24.36s`)
    i_v0, i_v1 = int(24.06 * SR), int(24.36 * SR)
    if i_v1 > i_v0:
        n_v = i_v1 - i_v0
        lt_v = np.linspace(0.0, 1.0, n_v, dtype=np.float64)
        sos_v = signal.butter(2, [450.0 / (0.5 * SR), 2200.0 / (0.5 * SR)], btype="band", output="sos")
        intake = signal.sosfilt(sos_v, rng.standard_normal(n_v)) * (lt_v ** 3.2) * 0.22
        left[i_v0:i_v1] += intake
        right[i_v0:i_v1] += intake

    # Hard musical vacuum duck across 24.02..24.36s so the Beat 5B drop hits like an IMAX trailer!
    vac_env = np.ones(TOTAL_SAMPLES, dtype=np.float64)
    v_a, v_b, v_c, v_d = int(24.00 * SR), int(24.05 * SR), int(24.34 * SR), int(24.38 * SR)
    vac_env[v_a:v_b] = np.linspace(1.0, 0.06, v_b - v_a)
    vac_env[v_b:v_c] = 0.06
    vac_env[v_c:v_d] = np.linspace(0.06, 1.0, v_d - v_c)
    left *= vac_env
    right *= vac_env

    # ------------------------------------------------------------------
    # 4. ORCHESTRAL BODY WARMTH + BROADCAST VOCAL-POCKET EQ CARVE
    # ------------------------------------------------------------------
    stereo = np.stack([left, right], axis=-1).astype(np.float32)
    # Warm orchestral cello/brass body resonance (+2.0 dB @ 210 Hz)
    stereo = _peaking_eq_stereo(stereo, 210.0, 0.85, 2.0)
    # Broadcast vocal-pocket EQ carve (250-3500 Hz) so deadpan neural narration sits crystal clear!
    stereo = _peaking_eq_stereo(stereo, 1180.0, 0.72, -5.8)
    sos_lp = signal.butter(2, 7200.0 / (0.5 * SR), btype="low", output="sos")
    stereo = signal.sosfilt(sos_lp, stereo, axis=0).astype(np.float32)

    rms = float(np.sqrt(np.mean(stereo.astype(np.float64) ** 2) + 1e-12))
    stereo *= 0.155 / max(1e-5, rms)
    peak = float(np.max(np.abs(stereo)))
    if peak > 0.78:
        stereo *= 0.78 / peak

    fade_in = int(0.08 * SR)
    fade_out = int(0.75 * SR)
    stereo[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)[:, None]
    stereo[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)[:, None]

    BGM_STEM_PATH.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(BGM_STEM_PATH), stereo, SR, subtype="PCM_16")
    print(
        f"[BGM] Generated V2 Hollywood score {BGM_STEM_PATH.name} "
        f"(112 BPM orchestral suspense + 0.35s pre-flare vacuum + Dmaj9 landing, peak={float(np.max(np.abs(stereo))):.3f})"
    )


if __name__ == "__main__":
    main()
