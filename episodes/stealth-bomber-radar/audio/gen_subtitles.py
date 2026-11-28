"""
Subtitles Engineer for Episode #4
("How a 172-Foot Stealth Bomber Looks Like a Bumblebee on Radar", 30.0 s @ 60 fps).

Upgraded Viral-Retention Edition:
    - Boxless word-synced ASS karaoke (`\\kf` sweep, 1080x1920, DejaVu Sans Bold 66pt)
    - BorderStyle=1, Outline=8, Shadow=3.0, Alignment=2, MarginV=235 (top NDC Y = -0.583 < -0.55)
    - Golden-amber (&H0038C8FF) active sweep over crisp white (&H00FFFFFF)
    - Selective Power-Word color pops (Electric Cyan &H00FFE633 for stealth tech terms,
      Alert Crimson-Orange &H004258FF for high-stakes viral punch words!)
    - Strictly 2-4 words per card, max 23 chars/line, zero overlaps, 100% word fidelity
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

import stealth_contract as C  # noqa: E402

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
OUTLINE_COLOUR = "&H00080A10"    # Deep obsidian keyline outline
BACK_COLOUR = "&H96000000"       # Soft drop shadow
OUTLINE_PX = 8
SHADOW_PX = 3.0
ALIGNMENT = 2
MARGIN_L = 80
MARGIN_R = 80
MARGIN_V = 235  # Bottom = 1685px (NDC Y -0.755), 2-line top ~1520px (NDC Y -0.583 < -0.55)

# Selective high-retention power-word highlight palette (ASS BGR format)
CYAN_POP_HEX = "&H00FFE633&"     # Electric Cyan (#33E6FF)
CRIMSON_POP_HEX = "&H004258FF&"  # Alert Crimson-Orange (#FF5842)

CYAN_POWER_WORDS = {
    "172-foot", "radar", "stealth", "origami", "sideways",
    "s-shaped", "tunnels", "carbon", "tilted", "scattering",
}
CRIMSON_POWER_WORDS = {
    "eighty", "tons", "shrinks", "bumblebee", "hidden",
    "trapped", "dies", "swallows", "heat", "sci-fi", "already", "gone",
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
    return f"\\1c{PRIMARY_COLOUR}&"


def join_tokens(tokens: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for i, tok in enumerate(tokens):
        word = _tok_word(tok)
        if i == 0:
            parts.append(word)
        elif _tok_word(tokens[i - 1]).endswith("\u2014") or word.startswith("\u2014"):
            parts.append(word)
        else:
            parts.append(" " + word)
    return "".join(parts)


def escape_ass_text(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _is_hanging(token: dict[str, Any]) -> bool:
    return _tok_word(token).lower().strip(_PUNCT_STRIP) in HANGING_WORDS


def choose_break(card: list[dict[str, Any]], max_line_chars: int = MAX_LINE_CHARS) -> int:
    n = len(card)
    full_txt = join_tokens(card)
    if n <= 2 and len(full_txt) <= max_line_chars:
        return n
    if n == 3 and len(full_txt) <= max_line_chars:
        return n
    if n == 4 and len(full_txt) <= max_line_chars - 2:
        return n

    best_k = n
    best_score = float("inf")
    for k in range(1, n):
        line1 = join_tokens(card[:k])
        line2 = join_tokens(card[k:])
        if len(line1) > max_line_chars or len(line2) > max_line_chars:
            continue
        score = (
            abs(len(line1) - len(line2)) * 1.2
            + (24.0 if (n - k == 1 and n >= 4) else 0.0)
            + (36.0 if _is_hanging(card[k - 1]) else 0.0)
            - (12.0 if _tok_word(card[k - 1]).endswith((",", "\u2014")) else 0.0)
        )
        if score < best_score:
            best_score = score
            best_k = k

    if best_k < n:
        return best_k

    best_score = float("inf")
    for k in range(1, n):
        line1 = join_tokens(card[:k])
        line2 = join_tokens(card[k:])
        score = (
            abs(len(line1) - len(line2)) * 1.2
            + max(0, len(line1) - max_line_chars) * 50
            + max(0, len(line2) - max_line_chars) * 50
            + (36 if _is_hanging(card[k - 1]) else 0)
        )
        if score < best_score:
            best_score = score
            best_k = k
    return best_k


def _card_lines(card: list[dict[str, Any]]) -> list[str]:
    sp = choose_break(card)
    if sp < len(card):
        return [join_tokens(card[:sp]), join_tokens(card[sp:])]
    return [join_tokens(card)]


def _score_card(card: list[dict[str, Any]], is_clause_end: bool) -> float:
    lines = _card_lines(card)
    if any(len(ln) > MAX_LINE_CHARS for ln in lines):
        return float("inf")
    if len(lines) == 2 and _is_hanging(card[choose_break(card) - 1]):
        return 1e5

    n = len(card)
    if len(join_tokens(card)) > MAX_CARD_CHARS:
        return float("inf")

    score = 9.0 if n == 2 else (0.0 if n == 3 else 2.0)
    if len(lines) == 2:
        score += abs(len(lines[0]) - len(lines[1])) * 0.45

    last_w = _tok_word(card[-1])
    if _is_hanging(card[-1]) and not is_clause_end:
        score += 40.0
    if last_w.endswith((",", ".", "!", "?", "\u2014")):
        score -= 22.0
    if any(_tok_word(w).endswith((",", "\u2014")) for w in card[:-1]):
        score += 38.0
    return score


def chunk_words(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    n = len(words)
    if n <= MIN_CARD_WORDS:
        return [words]

    dp: list[float] = [float("inf")] * (n + 1)
    nxt: list[int] = [-1] * (n + 1)
    dp[n] = 0.0

    for i in range(n - 2, -1, -1):
        for size in range(MIN_CARD_WORDS, MAX_CARD_WORDS + 1):
            j = i + size
            if j > n or dp[j] == float("inf"):
                continue
            cost = _score_card(words[i:j], is_clause_end=(j == n)) + dp[j]
            if cost < dp[i]:
                dp[i] = cost
                nxt[i] = j

    if dp[0] == float("inf"):
        raise ValueError(f"Unable to partition {n} words into valid 2-4 word cards <= {MAX_LINE_CHARS} chars/line")

    cards: list[list[dict[str, Any]]] = []
    idx = 0
    while idx < n:
        j = nxt[idx]
        assert j > idx
        cards.append(words[idx:j])
        idx = j
    return cards


def fmt_ass_cs(cs: int) -> str:
    cs = max(0, int(cs))
    return f"{cs // 360000}:{(cs % 360000) // 6000:02d}:{(cs % 6000) // 100:02d}.{cs % 100:02d}"


def fmt_srt_ms(ms: int) -> str:
    ms = max(0, int(ms))
    return f"{ms // 3600000:02d}:{(ms % 3600000) // 60000:02d}:{(ms % 60000) // 1000:02d},{ms % 1000:03d}"


def _parse_ass_time(ts: str) -> float:
    h_str, m_str, sec_str = ts.strip().split(":")
    return int(h_str) * 3600.0 + int(m_str) * 60.0 + float(sec_str)


def _parse_srt_time(ts: str) -> float:
    hms, ms_str = ts.strip().split(",")
    h_str, m_str, s_str = hms.split(":")
    return int(h_str) * 3600.0 + int(m_str) * 60.0 + int(s_str) + int(ms_str) / 1000.0


def build_ass_header() -> str:
    style_row = (
        f"Style: Default,{FONT_NAME},{FONT_SIZE},"
        f"{PRIMARY_COLOUR},{SECONDARY_COLOUR},{OUTLINE_COLOUR},{BACK_COLOUR},"
        f"-1,0,0,0,100,100,0,0,1,{OUTLINE_PX},{SHADOW_PX},"
        f"{ALIGNMENT},{MARGIN_L},{MARGIN_R},{MARGIN_V},1"
    )
    return "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {PLAY_RES_X}",
            f"PlayResY: {PLAY_RES_Y}",
            "WrapStyle: 0",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding",
            style_row,
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
            "",
        ]
    )


def _compute_kf_durations_cs(card: list[dict[str, Any]], t0c: int, t1c: int) -> list[int]:
    n = len(card)
    min_cs = 2
    if t1c - t0c < n * min_cs:
        raise ValueError(f"Card duration {t1c - t0c}cs too short for {n} words")

    durs: list[int] = []
    cur = t0c
    for i in range(n):
        if i == n - 1:
            dur = t1c - cur
        else:
            next_start_cs = int(round(_tok_t0(card[i + 1]) * 100))
            max_allowed = (t1c - cur) - (n - 1 - i) * min_cs
            dur = max(min_cs, min(max_allowed, next_start_cs - cur))
        durs.append(dur)
        cur += dur

    assert sum(durs) == (t1c - t0c)
    return durs


def build_lines_from_timeline(timeline_path: Path = TIMELINE_PATH) -> list[dict[str, Any]]:
    """Pure-Python proportional word alignment from timeline.json for offline subtitle/FFmpeg QA."""
    tl = json.loads(timeline_path.read_text(encoding="utf-8"))
    out_lines: list[dict[str, Any]] = []
    for item in tl["narration_lines"]:
        raw_words = [w for w in str(item["text"]).strip().split() if w]
        t0, t1 = float(item["t0"]), float(item["t1"])
        eff_t1 = max(t0 + 0.4, t1 - 0.12)
        weights = [
            max(1.8, float(len(re.sub(r"[^A-Za-z0-9]", "", w))))
            + (1.5 if w.endswith(("!", "?", ".")) else (1.0 if w.endswith(("\u2014", ",")) else 0.0))
            for w in raw_words
        ]
        total_w = sum(weights)
        bounds = [t0]
        acc = 0.0
        for wt in weights[:-1]:
            acc += wt / total_w
            bounds.append(t0 + (eff_t1 - t0) * acc)
        bounds.append(eff_t1)
        words = [{"text": w, "t0": round(bounds[i], 4), "t1": round(bounds[i + 1], 4)} for i, w in enumerate(raw_words)]
        out_lines.append({"id": item["id"], "beat": item["beat"], "text": item["text"], "t0": t0, "t1": eff_t1, "words": words})
    return out_lines


def write_subtitles(
    lines: list[dict[str, Any]],
    srt_path: Path = SRT_PATH,
    ass_path: Path = ASS_PATH,
    max_duration_s: float = DURATION_S,
) -> dict[str, int]:
    max_ms = int(round(max_duration_s * 1000))
    max_cs = int(round(max_duration_s * 100))

    srt_blocks: list[str] = []
    ass_events: list[str] = []
    srt_idx = 1
    prev_ms = 0
    prev_cs = 0
    max_line_len = 0

    for ln in lines:
        words = ln["words"]
        for card in chunk_words(words):
            sp = choose_break(card)
            for cl in _card_lines(card):
                max_line_len = max(max_line_len, len(cl))

            t0m = max(prev_ms, int(round(_tok_t0(card[0]) * 1000)))
            t1m = min(max_ms, max(t0m + 50, int(round(_tok_t1(card[-1]) * 1000))))
            if t1m <= t0m:
                raise ValueError(f"SRT card timing overflow near {_tok_word(card[0])!r}: {t0m}..{t1m}")
            prev_ms = t1m

            card_txt = (
                join_tokens(card[:sp]) + "\n" + join_tokens(card[sp:])
                if sp < len(card)
                else join_tokens(card)
            )
            srt_blocks.append(f"{srt_idx}\n{fmt_srt_ms(t0m)} --> {fmt_srt_ms(t1m)}\n{card_txt}\n")
            srt_idx += 1

            t0c = max(prev_cs, int(round(_tok_t0(card[0]) * 100)))
            t1c = min(max_cs, max(t0c + 2 * len(card), int(round(_tok_t1(card[-1]) * 100))))
            if t1c <= t0c:
                raise ValueError(f"ASS card timing overflow near {_tok_word(card[0])!r}: {t0c}..{t1c}")
            prev_cs = t1c

            kf_durs = _compute_kf_durations_cs(card, t0c, t1c)
            parts: list[str] = []
            for j, cw in enumerate(card):
                tok_raw = _tok_word(cw)
                tok = escape_ass_text(tok_raw)
                ctag = _power_color_tag(tok_raw)
                tagged = f"{{{ctag}\\kf{kf_durs[j]}}}{tok}"
                if j == sp:
                    parts.append("\\N" + tagged)
                elif j == 0:
                    parts.append(tagged)
                else:
                    sep = (
                        ""
                        if (_tok_word(card[j - 1]).endswith("\u2014") or tok_raw.startswith("\u2014"))
                        else " "
                    )
                    parts.append(sep + tagged)

            ass_events.append(
                f"Dialogue: 0,{fmt_ass_cs(t0c)},{fmt_ass_cs(t1c)},Default,,0,0,0,,{''.join(parts)}"
            )

    srt_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")
    ass_path.write_text(build_ass_header() + "\n".join(ass_events) + "\n", encoding="utf-8")

    return {
        "cards": len(srt_blocks),
        "events": len(ass_events),
        "words": sum(len(ln["words"]) for ln in lines),
        "max_line_len": max_line_len,
    }


def verify_subtitles(
    srt_path: Path,
    ass_path: Path,
    lines: list[dict[str, Any]],
    max_duration_s: float = DURATION_S,
) -> int:
    assert srt_path.is_file(), f"Missing SRT file: {srt_path}"
    assert ass_path.is_file(), f"Missing ASS file: {ass_path}"
    font_file = FONTS_DIR / "DejaVuSans-Bold.ttf"
    assert font_file.is_file(), f"Missing bundled font: {font_file}"

    max_block_height_px = FONT_SIZE * 2.2 + OUTLINE_PX * 2 + SHADOW_PX
    top_px = PLAY_RES_Y - MARGIN_V - max_block_height_px
    top_ndc_y = 1.0 - 2.0 * (top_px / float(PLAY_RES_Y))
    assert top_ndc_y < -0.55, f"Subtitle top NDC Y {top_ndc_y:.3f} must be < -0.55"

    expected_words = [_tok_word(w) for ln in lines for w in ln["words"]]
    expected_text = " ".join(expected_words)

    ass_text = ass_path.read_text(encoding="utf-8")
    assert f"PlayResX: {PLAY_RES_X}" in ass_text and f"PlayResY: {PLAY_RES_Y}" in ass_text
    dialogue_lines = [ln for ln in ass_text.splitlines() if ln.startswith("Dialogue:")]
    assert len(dialogue_lines) >= len(lines)

    ass_words_collected: list[str] = []
    prev_end = -1.0
    max_line_len = 0
    for d_idx, d_line in enumerate(dialogue_lines, start=1):
        fields = d_line.split(",", 9)
        assert len(fields) == 10
        t0, t1, payload = _parse_ass_time(fields[1]), _parse_ass_time(fields[2]), fields[9]
        assert 0.0 <= t0 < t1 <= max_duration_s + 1e-6
        assert t0 >= prev_end - 1e-9
        prev_end = t1

        kf_matches = [int(m) for m in _KF_TAG_RE.findall(payload)]
        plain_card = _ASS_TAG_RE.sub("", payload)
        card_lines = plain_card.split("\\N")
        assert 1 <= len(card_lines) <= 2
        for cl_txt in card_lines:
            max_line_len = max(max_line_len, len(cl_txt))
            assert len(cl_txt) <= MAX_LINE_CHARS, f"ASS line > {MAX_LINE_CHARS} chars: {cl_txt!r}"
        if len(card_lines) == 2:
            last_w1 = card_lines[0].split()[-1].lower().strip(_PUNCT_STRIP)
            assert last_w1 not in HANGING_WORDS, f"Hanging word {last_w1!r} at end of line 1: {plain_card!r}"

        card_words = " ".join(card_lines).split()
        assert MIN_CARD_WORDS <= len(card_words) <= MAX_CARD_WORDS
        assert len(kf_matches) == len(card_words)
        assert sum(kf_matches) == int(round((t1 - t0) * 100))
        ass_words_collected.extend(card_words)

    assert " ".join(ass_words_collected) == expected_text, "ASS word sequence mismatch"

    srt_raw = srt_path.read_text(encoding="utf-8").strip()
    blocks = [b.strip() for b in re.split(r"\r?\n\r?\n", srt_raw) if b.strip()]
    assert len(blocks) == len(dialogue_lines)

    srt_words_collected: list[str] = []
    prev_srt_end = -1.0
    for b_idx, blk in enumerate(blocks, start=1):
        blk_lines = blk.splitlines()
        assert int(blk_lines[0]) == b_idx
        ts_a, ts_b = blk_lines[1].split(" --> ")
        t0, t1 = _parse_srt_time(ts_a), _parse_srt_time(ts_b)
        assert 0.0 <= t0 < t1 <= max_duration_s + 1e-6 and t0 >= prev_srt_end - 1e-9
        prev_srt_end = t1
        txt_lines = blk_lines[2:]
        assert 1 <= len(txt_lines) <= 2
        for tl in txt_lines:
            max_line_len = max(max_line_len, len(tl))
            assert len(tl) <= MAX_LINE_CHARS
        if len(txt_lines) == 2:
            last_w1 = txt_lines[0].split()[-1].lower().strip(_PUNCT_STRIP)
            assert last_w1 not in HANGING_WORDS
        card_ws = " ".join(txt_lines).split()
        assert MIN_CARD_WORDS <= len(card_ws) <= MAX_CARD_WORDS
        srt_words_collected.extend(card_ws)

    assert " ".join(srt_words_collected) == expected_text, "SRT word sequence mismatch"
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
    verified_max_line = verify_subtitles(srt_path=srt_path, ass_path=ass_path, lines=lines, max_duration_s=max_dur)
    counts["max_line_len"] = verified_max_line
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate & verify ASS/SRT karaoke captions for Episode #4.")
    parser.add_argument(
        "--from-timeline",
        action="store_true",
        help="Align words directly from timeline.json if voice_manifest.json is not yet built.",
    )
    args = parser.parse_args()

    counts = generate_subtitles(from_timeline=args.from_timeline)
    max_block_height_px = FONT_SIZE * 2.2 + OUTLINE_PX * 2 + SHADOW_PX
    top_ndc_y = 1.0 - 2.0 * ((PLAY_RES_Y - MARGIN_V - max_block_height_px) / float(PLAY_RES_Y))
    print(
        f"[Subtitles] Generated & verified {counts['events']} \\kf power-word karaoke cards "
        f"({counts['words']} words, 0 overlaps, 2-4 words/card, max_line={counts['max_line_len']} <= {MAX_LINE_CHARS} chars/line, "
        f"<= {DURATION_S:.3f}s, top NDC Y={top_ndc_y:.3f} < -0.55)"
    )
    print(f"  ASS: {ASS_PATH}")
    print(f"  SRT: {SRT_PATH}")


if __name__ == "__main__":
    main()
