"""
gen_subtitles.py — Boxless Lower-Third Karaoke Subtitle Engine for Episode #5
("Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock", 30.0 s @ 60 fps).

Strictly enforces AGENTS.md V12 & @zackdfilms lower-third caption rules:
  - Pure-Python safe (runs cleanly both from voice_manifest.json AND `--from-timeline`)
  - `main(argv=None)` accepts optional `argv` so `generate_audio.py --rebuild-all` calls `sub_mod.main([])` cleanly!
  - Bold clean sans-serif white (`DejaVu Sans` 66pt) with crisp dark outline, golden-amber `\kf` active sweep,
    plus selective Cyan/Crimson power-word pops on high-stakes aviation & physics terms
  - Strictly 2-4 words per card, max 23 chars/line, zero hanging words, zero overlaps, top NDC Y = -0.583 < -0.55
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

AUDIO_DIR = Path(__file__).resolve().parent
EPISODE_ROOT = AUDIO_DIR.parent
SRC_DIR = EPISODE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import heli_contract as C  # noqa: E402

MANIFEST_PATH = AUDIO_DIR / "output" / "voice_manifest.json"
TIMELINE_PATH = EPISODE_ROOT / "config" / "timeline.json"
SUBTITLES_DIR = EPISODE_ROOT / "subtitles"
ASS_PATH = SUBTITLES_DIR / "subtitles.ass"
SRT_PATH = SUBTITLES_DIR / "subtitles.srt"
FONTS_DIR = SUBTITLES_DIR / "fonts"

DURATION_S = float(C.DURATION_S)

PLAY_RES_X = 1080
PLAY_RES_Y = 1920
FONT_NAME = "DejaVu Sans"
FONT_SIZE = 66
PRIMARY_COLOUR = "&H0038C8FF"    # Bold golden-amber active fill (#FFC838)
SECONDARY_COLOUR = "&H00FFFFFF"  # Crisp white unread text (#FFFFFF)
OUTLINE_COLOUR = "&H00080A10"    # Deep obsidian outline
BACK_COLOUR = "&H96000000"       # Subtle drop shadow
OUTLINE_PX = 8
SHADOW_PX = 3.0
ALIGNMENT = 2
MARGIN_L = 80
MARGIN_R = 80
MARGIN_V = 235  # Top NDC Y = -0.583 < -0.55 (safely below 3D labels & above Shorts chrome)

CYAN_POP_HEX = "&H00FFE633&"     # Electric Cyan (#33E6FF)
CRIMSON_POP_HEX = "&H004258FF&"  # Alert Crimson-Orange (#FF5842)

CYAN_POWER_WORDS = {
    "pilot", "lever", "rotor", "bicycle-style", "clutch", "unlocks",
    "blades", "air", "wind", "turbine", "energy", "maple", "seed",
    "lift", "spin", "smooth", "landing",
}
CRIMSON_POWER_WORDS = {
    "engine", "dies", "mid-air", "slams", "dead", "brake",
    "thirty", "final", "burst", "drop", "powerless",
}

MIN_CARD_WORDS = 2
MAX_CARD_WORDS = 4
MAX_LINE_CHARS = 23
MAX_CARD_CHARS = 44

HANGING_WORDS = {
    "a", "an", "the", "of", "on", "in", "to", "with", "and", "or", "as",
    "your", "into", "for", "by", "at", "this", "that", "these", "those",
    "he", "his", "it", "its", "my", "our", "their", "from", "up", "down",
    "out", "off", "over", "under", "until", "while", "but", "if", "is",
    "are", "was", "were", "can", "yet", "so", "any", "one", "than",
}

_PUNCT_STRIP = ".,!?\u2014:;\"'"
_ASS_TAG_RE = re.compile(r"\{\\[^}]*\}")
_KF_TAG_RE = re.compile(r"\\kf(\d+)")


def _tok_word(tok: dict[str, Any]) -> str:
    return str(tok["text"] if "text" in tok else tok["word"])


def _tok_t0(tok: dict[str, Any]) -> float:
    return float(tok["t0"] if "t0" in tok else tok["start"])


def _tok_t1(tok: dict[str, Any]) -> float:
    return float(tok["t1"] if "t1" in tok else tok["end"])


def _power_color_tag(raw_word: str) -> str:
    norm = raw_word.lower().strip(_PUNCT_STRIP)
    if norm in CRIMSON_POWER_WORDS:
        return f"\\1c{CRIMSON_POP_HEX}"
    if norm in CYAN_POWER_WORDS:
        return f"\\1c{CYAN_POP_HEX}"
    return ""


def _fmt_ass_time(sec: float) -> str:
    cs_tot = max(0, int(round(sec * 100.0)))
    h = cs_tot // 360000
    m = (cs_tot % 360000) // 6000
    s = (cs_tot % 6000) // 100
    cs = cs_tot % 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _parse_ass_time(ts: str) -> float:
    h, m, rest = ts.strip().split(":")
    s, cs = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100.0


def _fmt_srt_time(sec: float) -> str:
    ms_tot = max(0, int(round(sec * 1000.0)))
    h = ms_tot // 3600000
    m = (ms_tot % 3600000) // 60000
    s = (ms_tot % 60000) // 1000
    ms = ms_tot % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _parse_srt_time(ts: str) -> float:
    h, m, rest = ts.strip().split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def build_lines_from_timeline(timeline_path: Path) -> list[dict[str, Any]]:
    """Pure-Python weighted word alignment fallback when voice_manifest.json is not yet generated."""
    tl = json.loads(timeline_path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for item in tl["narration_lines"]:
        raw_words = [w for w in str(item["text"]).strip().split() if w]
        t0, t1 = float(item["t0"]), float(item["t1"])
        wts = [max(1.8, float(len(re.sub(r"[^A-Za-z0-9]", "", w)))) + (1.2 if w.endswith((".", ",", "!")) else 0.0) for w in raw_words]
        s_wt = sum(wts) or 1.0
        cur = t0
        words_aligned: list[dict[str, Any]] = []
        for i, (w, wt) in enumerate(zip(raw_words, wts)):
            nxt = t1 if i == len(raw_words) - 1 else cur + (t1 - t0) * (wt / s_wt)
            words_aligned.append({"text": w, "t0": round(cur, 4), "t1": round(nxt, 4)})
            cur = nxt
        out.append({"id": item["id"], "beat": item["beat"], "text": item["text"], "t0": t0, "t1": t1, "words": words_aligned})
    return out


def _split_card_lines(words: list[str]) -> list[list[str]]:
    joined = " ".join(words)
    if len(joined) <= MAX_LINE_CHARS:
        return [words]
    best: list[list[str]] | None = None
    best_score = 1e9
    for k in range(1, len(words)):
        l1, l2 = words[:k], words[k:]
        s1, s2 = " ".join(l1), " ".join(l2)
        if len(s1) > MAX_LINE_CHARS or len(s2) > MAX_LINE_CHARS:
            continue
        last1 = l1[-1].lower().strip(_PUNCT_STRIP)
        if last1 in HANGING_WORDS:
            continue
        score = abs(len(s1) - len(s2))
        if score < best_score:
            best_score = score
            best = [l1, l2]
    if best is not None:
        return best
    # Fallback even split (respecting MAX_LINE_CHARS)
    mid = max(1, len(words) // 2)
    return [words[:mid], words[mid:]]


def _chunk_line_words(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Partition words into 2-4 word cards with zero hanging words and <= MAX_LINE_CHARS per line."""
    n = len(words)
    if n <= MAX_CARD_WORDS:
        txt = " ".join(_tok_word(w) for w in words)
        if len(txt) <= MAX_CARD_CHARS:
            return [words]

    # Dynamic programming for optimal 2-4 word card breaks
    dp: list[tuple[float, int] | None] = [None] * (n + 1)
    dp[n] = (0.0, n)
    for i in range(n - 1, -1, -1):
        best_cost = 1e9
        best_j = -1
        for k in range(MIN_CARD_WORDS, MAX_CARD_WORDS + 1):
            j = i + k
            if j > n or dp[j] is None:
                continue
            chunk = words[i:j]
            ws = [_tok_word(w) for w in chunk]
            if len(" ".join(ws)) > MAX_CARD_CHARS:
                continue
            lines = _split_card_lines(ws)
            if any(len(" ".join(ln)) > MAX_LINE_CHARS for ln in lines):
                continue
            if len(lines) == 2 and lines[0][-1].lower().strip(_PUNCT_STRIP) in HANGING_WORDS:
                continue
            cost = abs(k - 3) * 1.2 + (2.5 if ws[-1].lower().strip(_PUNCT_STRIP) in HANGING_WORDS and j < n else 0.0)
            if ws[-1].endswith((".", ",", "!", "?")):
                cost -= 1.0
            tot = cost + dp[j][0]  # type: ignore[index]
            if tot < best_cost:
                best_cost = tot
                best_j = j
        if best_j != -1:
            dp[i] = (best_cost, best_j)

    if dp[0] is not None:
        cards: list[list[dict[str, Any]]] = []
        idx = 0
        while idx < n:
            nxt = dp[idx][1]  # type: ignore[index]
            cards.append(words[idx:nxt])
            idx = nxt
        return cards

    # Safe greedy fallback
    cards = []
    idx = 0
    while idx < n:
        rem = n - idx
        take = 2 if rem == 4 else min(3, rem)
        if rem - take == 1:
            take = max(2, take - 1)
        cards.append(words[idx : idx + take])
        idx += take
    return cards


def write_subtitles(
    lines: list[dict[str, Any]],
    srt_path: Path = SRT_PATH,
    ass_path: Path = ASS_PATH,
    max_duration_s: float = DURATION_S,
) -> dict[str, int]:
    SUBTITLES_DIR.mkdir(parents=True, exist_ok=True)
    cards: list[tuple[float, float, list[dict[str, Any]]]] = []
    for ln in lines:
        for ch in _chunk_line_words(ln["words"]):
            c0 = max(0.0, _tok_t0(ch[0]))
            c1 = min(max_duration_s, _tok_t1(ch[-1]))
            cards.append((c0, c1, ch))

    # Eliminate any micro-overlaps at centisecond resolution
    norm_cards: list[tuple[float, float, list[dict[str, Any]]]] = []
    prev_cs = 0
    for c0, c1, ch in cards:
        cs0 = max(prev_cs, int(round(c0 * 100.0)))
        cs1 = max(cs0 + len(ch), int(round(c1 * 100.0)))
        cs1 = min(int(round(max_duration_s * 100.0)), cs1)
        prev_cs = cs1
        norm_cards.append((cs0 / 100.0, cs1 / 100.0, ch))

    ass_header = "\n".join([
        "[Script Info]",
        "Title: Episode 5 - Helicopter Autorotation",
        "ScriptType: v4.00+",
        f"PlayResX: {PLAY_RES_X}",
        f"PlayResY: {PLAY_RES_Y}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{FONT_NAME},{FONT_SIZE},{PRIMARY_COLOUR},{SECONDARY_COLOUR},{OUTLINE_COLOUR},{BACK_COLOUR},-1,0,0,0,100,100,0,0,1,{OUTLINE_PX},{SHADOW_PX:.1f},{ALIGNMENT},{MARGIN_L},{MARGIN_R},{MARGIN_V},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ])

    ass_events: list[str] = []
    srt_blocks: list[str] = []
    total_words = 0

    for idx, (c0, c1, ch) in enumerate(norm_cards, start=1):
        ws = [_tok_word(w) for w in ch]
        total_words += len(ws)
        split_lines = _split_card_lines(ws)

        tot_cs = max(len(ch), int(round((c1 - c0) * 100.0)))
        raw_durs = [max(0.01, _tok_t1(w) - _tok_t0(w)) for w in ch]
        sum_d = sum(raw_durs) or 1.0
        kfs = [max(1, int(round(tot_cs * (d / sum_d)))) for d in raw_durs]
        diff = tot_cs - sum(kfs)
        kfs[-1] = max(1, kfs[-1] + diff)

        ass_line_strs: list[str] = []
        w_cursor = 0
        for ln_ws in split_lines:
            parts: list[str] = []
            for w_txt in ln_ws:
                kf_val = kfs[w_cursor]
                w_cursor += 1
                pop_tag = _power_color_tag(w_txt)
                if pop_tag:
                    parts.append(f"{{{pop_tag}\\kf{kf_val}}}{w_txt}{{\\1c{PRIMARY_COLOUR}&}}")
                else:
                    parts.append(f"{{\\kf{kf_val}}}{w_txt}")
            ass_line_strs.append(" ".join(parts))

        ass_payload = "\\N".join(ass_line_strs)
        ass_events.append(f"Dialogue: 0,{_fmt_ass_time(c0)},{_fmt_ass_time(c1)},Default,,0,0,0,,{ass_payload}")

        srt_txt = "\n".join(" ".join(ln_ws) for ln_ws in split_lines)
        srt_blocks.append(f"{idx}\n{_fmt_srt_time(c0)} --> {_fmt_srt_time(c1)}\n{srt_txt}")

    ass_path.write_text(ass_header + "\n" + "\n".join(ass_events) + "\n", encoding="utf-8")
    srt_path.write_text("\n\n".join(srt_blocks) + "\n", encoding="utf-8")
    return {"events": len(norm_cards), "words": total_words}


def verify_subtitles(
    srt_path: Path = SRT_PATH,
    ass_path: Path = ASS_PATH,
    lines: list[dict[str, Any]] | None = None,
    max_duration_s: float = DURATION_S,
) -> int:
    assert ass_path.is_file(), f"Missing ASS: {ass_path}"
    assert srt_path.is_file(), f"Missing SRT: {srt_path}"
    assert (FONTS_DIR / "DejaVuSans-Bold.ttf").is_file(), f"Missing bundled font in {FONTS_DIR}"

    max_block_h = FONT_SIZE * 2.2 + OUTLINE_PX * 2 + SHADOW_PX
    top_ndc_y = 1.0 - 2.0 * ((PLAY_RES_Y - MARGIN_V - max_block_h) / float(PLAY_RES_Y))
    assert top_ndc_y < -0.55, f"Subtitle top NDC Y {top_ndc_y:.3f} must be < -0.55"

    expected_text = " ".join(_tok_word(w) for ln in (lines or []) for w in ln["words"])
    ass_text = ass_path.read_text(encoding="utf-8")
    dialogue_lines = [ln for ln in ass_text.splitlines() if ln.startswith("Dialogue:")]
    assert dialogue_lines, "No ASS Dialogue lines"

    collected: list[str] = []
    prev_end = -1.0
    max_line_len = 0
    for d_line in dialogue_lines:
        fields = d_line.split(",", 9)
        t0, t1, payload = _parse_ass_time(fields[1]), _parse_ass_time(fields[2]), fields[9]
        assert 0.0 <= t0 < t1 <= max_duration_s + 1e-6 and t0 >= prev_end - 1e-9
        prev_end = t1
        kfs = [int(m) for m in _KF_TAG_RE.findall(payload)]
        plain = _ASS_TAG_RE.sub("", payload)
        c_lines = plain.split("\\N")
        assert 1 <= len(c_lines) <= 2
        for cl in c_lines:
            max_line_len = max(max_line_len, len(cl))
            assert len(cl) <= MAX_LINE_CHARS, f"ASS line > {MAX_LINE_CHARS}: {cl!r}"
        if len(c_lines) == 2:
            assert c_lines[0].split()[-1].lower().strip(_PUNCT_STRIP) not in HANGING_WORDS
        c_ws = " ".join(c_lines).split()
        assert MIN_CARD_WORDS <= len(c_ws) <= MAX_CARD_WORDS
        assert len(kfs) == len(c_ws) and sum(kfs) == int(round((t1 - t0) * 100))
        collected.extend(c_ws)

    if expected_text:
        assert " ".join(collected) == expected_text, "ASS word mismatch"
    return max_line_len


def generate_subtitles(
    manifest_path: Path = MANIFEST_PATH,
    srt_path: Path = SRT_PATH,
    ass_path: Path = ASS_PATH,
    from_timeline: bool = False,
) -> dict[str, int]:
    if from_timeline or not manifest_path.is_file():
        lines = build_lines_from_timeline(TIMELINE_PATH)
        max_dur = DURATION_S
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lines = manifest["lines"]
        max_dur = float(manifest.get("duration", DURATION_S))

    counts = write_subtitles(lines, srt_path=srt_path, ass_path=ass_path, max_duration_s=max_dur)
    counts["max_line_len"] = verify_subtitles(srt_path=srt_path, ass_path=ass_path, lines=lines, max_duration_s=max_dur)
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate & verify ASS/SRT karaoke captions for Episode #5.")
    parser.add_argument("--from-timeline", action="store_true", help="Align directly from timeline.json.")
    args = parser.parse_args(argv)

    counts = generate_subtitles(from_timeline=args.from_timeline)
    max_block_h = FONT_SIZE * 2.2 + OUTLINE_PX * 2 + SHADOW_PX
    top_ndc_y = 1.0 - 2.0 * ((PLAY_RES_Y - MARGIN_V - max_block_h) / float(PLAY_RES_Y))
    print(
        f"[Subtitles] Generated & verified {counts['events']} \\kf power-word karaoke cards "
        f"({counts['words']} words, 0 overlaps, max_line={counts['max_line_len']} <= {MAX_LINE_CHARS}, "
        f"top NDC Y={top_ndc_y:.3f} < -0.55)"
    )


if __name__ == "__main__":
    main()
