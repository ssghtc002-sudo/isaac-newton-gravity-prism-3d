"""
Subtitles Engineer for Episode #1
("Why Ejecting From a Fighter Jet Shrinks You by 1 Inch", 30.0 s @ 60 fps).

Reads word-level timestamps from:
    episodes/fighter-jet-ejection/audio/output/voice_manifest.json

Generates:
    episodes/fighter-jet-ejection/subtitles/subtitles.ass
        - Boxless word-synced ASS karaoke (1080x1920, DejaVu Sans Bold 66pt)
        - BorderStyle=1, Outline=8, Shadow=3.0, Alignment=2, MarginV=260
        - Gold-amber (&H0038C8FF&) active word highlight on crisp white (&H00FFFFFF&)
    episodes/fighter-jet-ejection/subtitles/subtitles.srt
        - Card-level SRT captions (max 2 lines/card, max 23 chars/line)

Guarantees:
    - Zero time overlaps (t0 >= prev_end), monotonic non-decreasing, t1 <= 30.000s
    - Zero single-word orphan cards, zero hanging preposition/article orphans
    - 100% word fidelity against voice_manifest.json
"""
from __future__ import annotations

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

import ejection_contract as C  # noqa: E402

MANIFEST_PATH = AUDIO_DIR / "output" / "voice_manifest.json"
SUBTITLES_DIR = EPISODE_ROOT / "subtitles"
ASS_PATH = SUBTITLES_DIR / "subtitles.ass"
SRT_PATH = SUBTITLES_DIR / "subtitles.srt"
FONTS_DIR = SUBTITLES_DIR / "fonts"

DURATION_S = float(C.DURATION_S)

PLAY_RES_X = 1080
PLAY_RES_Y = 1920
FONT_NAME = "DejaVu Sans"
FONT_SIZE = 66
PRIMARY_COLOUR = "&H00FFFFFF"
SECONDARY_COLOUR = "&H0038C8FF"
OUTLINE_COLOUR = "&H00080A10"
BACK_COLOUR = "&H96000000"
OUTLINE_PX = 8
SHADOW_PX = 3.0
ALIGNMENT = 2
MARGIN_L = 80
MARGIN_R = 80
MARGIN_V = 260

MAX_LINE_CHARS = 23
MAX_CARD_WORDS = 7
MAX_CARD_CHARS = 44

HANGING_WORDS = {
    "a",
    "an",
    "the",
    "of",
    "on",
    "in",
    "to",
    "with",
    "and",
    "or",
    "as",
    "your",
    "into",
    "for",
    "by",
    "at",
    "this",
    "that",
    "these",
    "those",
    "he",
    "his",
    "it",
    "its",
    "my",
    "our",
    "their",
    "from",
    "up",
    "down",
    "out",
    "off",
    "over",
    "under",
    "until",
    "while",
    "but",
    "if",
    "is",
    "are",
    "was",
    "were",
}

_PUNCT_STRIP = ".,!?\u2014:;\"'"
_ASS_TAG_RE = re.compile(r"\{\\[^}]*\}")


def join_tokens(tokens: list[dict[str, Any]]) -> str:
    """Join word tokens cleanly, attaching em-dashes without extra spaces."""
    parts: list[str] = []
    for i, tok in enumerate(tokens):
        word = str(tok["word"])
        if i == 0:
            parts.append(word)
        elif str(tokens[i - 1]["word"]).endswith("\u2014") or word.startswith("\u2014"):
            parts.append(word)
        else:
            parts.append(" " + word)
    return "".join(parts)


def escape_ass_text(text: str) -> str:
    """Escape ASS control braces/backslashes in raw display tokens."""
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _is_hanging(token: dict[str, Any]) -> bool:
    return str(token["word"]).lower().strip(_PUNCT_STRIP) in HANGING_WORDS


def split_clause(
    clause: list[dict[str, Any]],
    max_words: int = MAX_CARD_WORDS,
    max_chars: int = MAX_CARD_CHARS,
) -> list[list[dict[str, Any]]]:
    """Recursively split a long clause into balanced 1-2 line subtitle cards."""
    if len(clause) <= max_words and len(join_tokens(clause)) <= max_chars:
        return [clause]
    n = len(clause)
    if n < 4:
        return [clause]

    best_k = max(2, min(n - 2, max_words))
    best_score = float("inf")
    for k in range(2, n - 1):
        left_txt = join_tokens(clause[:k])
        right_txt = join_tokens(clause[k:])
        score = (
            abs(len(left_txt) - len(right_txt))
            + max(0, k - max_words) * 40
            + max(0, (n - k) - max_words) * 40
            + max(0, len(left_txt) - max_chars) * 6
            + max(0, len(right_txt) - max_chars) * 6
            + (35 if _is_hanging(clause[k - 1]) else 0)
            - (18 if str(clause[k - 1]["word"]).endswith((",", "\u2014")) else 0)
        )
        if score < best_score:
            best_score = score
            best_k = k
    return split_clause(clause[:best_k], max_words, max_chars) + split_clause(
        clause[best_k:], max_words, max_chars
    )


def chunk_words(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split a dialogue line into balanced cards with zero single-word orphans."""
    clauses: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    for i, word_item in enumerate(words):
        cur.append(word_item)
        rem = len(words) - 1 - i
        token = str(word_item["word"])
        hard_end = token.endswith(("\u2014", ".", "!", "?"))
        comma_end = token.endswith(",") and len(cur) >= 3 and len(join_tokens(cur)) >= 20
        if (hard_end or comma_end) and len(cur) >= 2 and rem != 1:
            clauses.append(cur)
            cur = []
    if cur:
        if len(cur) == 1 and clauses:
            clauses[-1].extend(cur)
        else:
            clauses.append(cur)

    cards = [card for cl in clauses for card in split_clause(cl)]
    merged: list[list[dict[str, Any]]] = []
    for card in cards:
        if len(card) == 1 and merged:
            merged[-1].extend(card)
        else:
            merged.append(card)
    return merged


def choose_break(card: list[dict[str, Any]], max_line_chars: int = MAX_LINE_CHARS) -> int:
    """Choose line break index inside a card (returns len(card) for 1-line cards)."""
    n = len(card)
    if n <= 2 or (len(join_tokens(card)) <= max_line_chars and n <= 4):
        return n

    best_k = n
    best_score = float("inf")
    for k in range(1, n):
        line1 = join_tokens(card[:k])
        line2 = join_tokens(card[k:])
        score = (
            abs(len(line1) - len(line2)) * 1.2
            + max(0, len(line1) - max_line_chars) * 15
            + max(0, len(line2) - max_line_chars) * 15
            + (24 if (n - k == 1 and n >= 4) else 0)
            + (30 if _is_hanging(card[k - 1]) else 0)
        )
        if score < best_score:
            best_score = score
            best_k = k
    return best_k


def fmt_ass_cs(cs: int) -> str:
    """Format integer centiseconds as ASS timestamp H:MM:SS.cc."""
    cs = max(0, int(cs))
    return f"{cs // 360000}:{(cs % 360000) // 6000:02d}:{(cs % 6000) // 100:02d}.{cs % 100:02d}"


def fmt_srt_ms(ms: int) -> str:
    """Format integer milliseconds as SRT timestamp HH:MM:SS,mmm."""
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
    """Return ASS v4.00+ header configured for 1080x1920 boxless portrait captions."""
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


def write_subtitles(
    lines: list[dict[str, Any]],
    srt_path: Path = SRT_PATH,
    ass_path: Path = ASS_PATH,
    max_duration_s: float = DURATION_S,
) -> dict[str, int]:
    """Generate synchronized .srt cards and word-highlighted .ass karaoke events."""
    max_ms = int(round(max_duration_s * 1000))
    max_cs = int(round(max_duration_s * 100))

    srt_blocks: list[str] = []
    ass_events: list[str] = []
    srt_idx = 1
    prev_ms = 0
    prev_cs = 0

    for ln in lines:
        words = ln["words"]
        for card in chunk_words(words):
            sp = choose_break(card)
            t0m = max(prev_ms, int(round(float(card[0]["start"]) * 1000)))
            t1m = min(max_ms, max(t0m + 50, int(round(float(card[-1]["end"]) * 1000))))
            if t1m <= t0m:
                raise ValueError(f"SRT card timing overflow near {card[0]['word']!r}: {t0m}..{t1m}")
            prev_ms = t1m

            card_txt = (
                join_tokens(card[:sp]) + "\n" + join_tokens(card[sp:])
                if sp < len(card)
                else join_tokens(card)
            )
            srt_blocks.append(f"{srt_idx}\n{fmt_srt_ms(t0m)} --> {fmt_srt_ms(t1m)}\n{card_txt}\n")
            srt_idx += 1

            for i, w in enumerate(card):
                raw_start = float(w["start"]) if i > 0 else float(card[0]["start"])
                raw_end = float(card[i + 1]["start"]) if i + 1 < len(card) else float(w["end"])
                t0c = max(prev_cs, int(round(raw_start * 100)))
                t1c = min(max_cs, max(t0c + 3, int(round(raw_end * 100))))
                if t1c <= t0c:
                    raise ValueError(f"ASS event timing overflow on word {w['word']!r}: {t0c}..{t1c}")
                prev_cs = t1c

                parts: list[str] = []
                for j, cw in enumerate(card):
                    tok_raw = str(cw["word"])
                    tok = escape_ass_text(tok_raw)
                    styled = (
                        f"{{\\c{SECONDARY_COLOUR}&}}{tok}{{\\c{PRIMARY_COLOUR}&}}"
                        if j == i
                        else tok
                    )
                    if j == sp:
                        parts.append("\\N" + styled)
                    elif j == 0:
                        parts.append(styled)
                    else:
                        sep = (
                            ""
                            if (str(card[j - 1]["word"]).endswith("\u2014") or tok_raw.startswith("\u2014"))
                            else " "
                        )
                        parts.append(sep + styled)

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
    }


def verify_subtitles(
    srt_path: Path,
    ass_path: Path,
    lines: list[dict[str, Any]],
    max_duration_s: float = DURATION_S,
) -> None:
    """Strictly verify generated ASS & SRT files for timing, line limits, orphans, and word parity."""
    assert srt_path.is_file(), f"Missing SRT file: {srt_path}"
    assert ass_path.is_file(), f"Missing ASS file: {ass_path}"

    expected_words = [str(w["word"]) for ln in lines for w in ln["words"]]
    expected_text = " ".join(expected_words)

    # 1. Verify ASS
    ass_text = ass_path.read_text(encoding="utf-8")
    assert f"PlayResX: {PLAY_RES_X}" in ass_text
    assert f"PlayResY: {PLAY_RES_Y}" in ass_text
    assert "BorderStyle" in ass_text
    assert (
        f"Style: Default,{FONT_NAME},{FONT_SIZE},{PRIMARY_COLOUR},{SECONDARY_COLOUR},"
        f"{OUTLINE_COLOUR},{BACK_COLOUR},-1,0,0,0,100,100,0,0,1,{OUTLINE_PX},{SHADOW_PX},"
        f"{ALIGNMENT},{MARGIN_L},{MARGIN_R},{MARGIN_V},1"
    ) in ass_text

    dialogue_lines = [ln for ln in ass_text.splitlines() if ln.startswith("Dialogue:")]
    assert len(dialogue_lines) == len(expected_words), (
        f"ASS event count {len(dialogue_lines)} != manifest word count {len(expected_words)}"
    )

    prev_end = -1.0
    for d_idx, d_line in enumerate(dialogue_lines):
        fields = d_line.split(",", 9)
        assert len(fields) == 10, f"Malformed ASS dialogue line {d_idx + 1}: {d_line}"
        t0 = _parse_ass_time(fields[1])
        t1 = _parse_ass_time(fields[2])
        payload = fields[9]
        assert 0.0 <= t0 < t1 <= max_duration_s + 1e-6, (
            f"ASS event {d_idx + 1} out of bounds: {t0:.2f}..{t1:.2f}s (max={max_duration_s:.2f}s)"
        )
        assert t0 >= prev_end - 1e-9, (
            f"ASS overlap at event {d_idx + 1}: t0={t0:.2f} < prev_end={prev_end:.2f}"
        )
        prev_end = t1

        assert f"{{\\c{SECONDARY_COLOUR}&}}" in payload, f"Missing active gold-amber tag in event {d_idx + 1}"
        assert f"{{\\c{PRIMARY_COLOUR}&}}" in payload, f"Missing white reset tag in event {d_idx + 1}"
        plain_card = _ASS_TAG_RE.sub("", payload)
        card_lines = plain_card.split("\\N")
        assert 1 <= len(card_lines) <= 2, f"ASS card has {len(card_lines)} lines: {plain_card!r}"
        for cl_txt in card_lines:
            assert len(cl_txt) <= MAX_LINE_CHARS, (
                f"ASS line exceeds {MAX_LINE_CHARS} chars ({len(cl_txt)}): {cl_txt!r}"
            )
        if len(card_lines) == 2:
            last_w1 = card_lines[0].split()[-1].lower().strip(_PUNCT_STRIP)
            assert last_w1 not in HANGING_WORDS, f"Hanging word {last_w1!r} at end of line 1: {plain_card!r}"
        card_words = " ".join(card_lines).split()
        assert len(card_words) >= 2, f"Single-word orphan ASS card: {plain_card!r}"

    # 2. Verify SRT
    srt_raw = srt_path.read_text(encoding="utf-8").strip()
    blocks = [b.strip() for b in re.split(r"\r?\n\r?\n", srt_raw) if b.strip()]
    assert len(blocks) >= len(lines), f"Too few SRT cards: {len(blocks)}"

    srt_words_collected: list[str] = []
    prev_srt_end = -1.0
    for b_idx, blk in enumerate(blocks, start=1):
        blk_lines = blk.splitlines()
        assert len(blk_lines) in (3, 4), f"SRT card {b_idx} has {len(blk_lines) - 2} text lines: {blk!r}"
        assert int(blk_lines[0]) == b_idx, f"SRT index mismatch: expected {b_idx}, got {blk_lines[0]!r}"
        ts_a, ts_b = blk_lines[1].split(" --> ")
        t0 = _parse_srt_time(ts_a)
        t1 = _parse_srt_time(ts_b)
        assert 0.0 <= t0 < t1 <= max_duration_s + 1e-6, (
            f"SRT card {b_idx} out of bounds: {t0:.3f}..{t1:.3f}s"
        )
        assert t0 >= prev_srt_end - 1e-9, (
            f"SRT overlap at card {b_idx}: t0={t0:.3f} < prev_end={prev_srt_end:.3f}"
        )
        prev_srt_end = t1

        txt_lines = blk_lines[2:]
        assert 1 <= len(txt_lines) <= 2
        for tl in txt_lines:
            assert len(tl) <= MAX_LINE_CHARS, f"SRT line > {MAX_LINE_CHARS} chars ({len(tl)}): {tl!r}"
        if len(txt_lines) == 2:
            last_w1 = txt_lines[0].split()[-1].lower().strip(_PUNCT_STRIP)
            assert last_w1 not in HANGING_WORDS, f"Hanging word {last_w1!r} in SRT card {b_idx}: {txt_lines}"
        card_ws = " ".join(txt_lines).split()
        assert len(card_ws) >= 2, f"Single-word orphan SRT card {b_idx}: {txt_lines!r}"
        srt_words_collected.extend(card_ws)

    assert " ".join(srt_words_collected) == expected_text, "SRT word sequence does not match manifest!"


def generate_subtitles(
    manifest_path: Path = MANIFEST_PATH,
    srt_path: Path = SRT_PATH,
    ass_path: Path = ASS_PATH,
) -> dict[str, int]:
    """Load voice_manifest.json, write .srt/.ass subtitles, and run strict verification."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lines = manifest["lines"]
    max_dur = float(manifest.get("duration", DURATION_S))
    counts = write_subtitles(lines, srt_path=srt_path, ass_path=ass_path, max_duration_s=max_dur)
    verify_subtitles(srt_path=srt_path, ass_path=ass_path, lines=lines, max_duration_s=max_dur)
    return counts


def main() -> None:
    counts = generate_subtitles()
    print(
        f"[Subtitles] Generated & verified {counts['events']} word-synced ASS events "
        f"({counts['cards']} SRT cards, 0 overlaps, <= {MAX_LINE_CHARS} chars/line, <= {DURATION_S:.3f}s)"
    )
    print(f"  ASS: {ASS_PATH}")
    print(f"  SRT: {SRT_PATH}")


if __name__ == "__main__":
    main()
