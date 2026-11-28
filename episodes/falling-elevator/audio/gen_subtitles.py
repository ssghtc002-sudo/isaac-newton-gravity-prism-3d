"""
Subtitles Engineer for Episode #16
("Why Jumping in a Falling Elevator WON'T Save You", 30.0 s @ 60 fps).

Reads word-level timestamps from:
    episodes/falling-elevator/audio/output/voice_manifest.json

Generates:
    episodes/falling-elevator/subtitles/subtitles.ass
        - Boxless word-synced ASS karaoke (`\\kf` sweep, 1080x1920, DejaVu Sans Bold 66pt)
        - BorderStyle=1, Outline=8, Shadow=3.0, Alignment=2, MarginV=235 (strictly NDC Y < -0.55)
        - Bold golden-amber (&H0038C8FF&) active highlight over crisp white (&H00FFFFFF&)
    episodes/falling-elevator/subtitles/subtitles.srt
        - Card-level SRT captions (2-4 words/card, max 2 lines/card, max 23 chars/line)

Guarantees:
    - Zero time overlaps (t0 >= prev_end), monotonic non-decreasing, 0.0 <= t0 < t1 <= 30.000s
    - Strictly 2-4 words per card (zero single-word orphans)
    - Max 23 characters per line, zero hanging preposition/article line-1 breaks
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

import elevator_contract as C  # noqa: E402

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
# In ASS karaoke (\kf), SecondaryColour renders before sweep and PrimaryColour renders after sweep.
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

MIN_CARD_WORDS = 2
MAX_CARD_WORDS = 4
MAX_LINE_CHARS = 23
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
    "can",
}

_PUNCT_STRIP = ".,!?\u2014:;\"'"
_ASS_TAG_RE = re.compile(r"\{\\[^}]*\}")
_KF_TAG_RE = re.compile(r"\{\\kf(\d+)\}")


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


def choose_break(card: list[dict[str, Any]], max_line_chars: int = MAX_LINE_CHARS) -> int:
    """Choose line break index inside a 2-4 word card (returns len(card) for 1-line cards)."""
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
            - (12.0 if str(card[k - 1]["word"]).endswith((",", "\u2014")) else 0.0)
        )
        if score < best_score:
            best_score = score
            best_k = k

    if best_k < n:
        return best_k

    # Fallback if 1-line exceeded max_line_chars and no valid split found above
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
        return 1e5  # Avoid hanging preposition at end of line 1

    n = len(card)
    total_chars = len(join_tokens(card))
    if total_chars > MAX_CARD_CHARS:
        return float("inf")

    score = 0.0
    # Prefer crisp 3-word and 4-word cards over 2-word cards unless punctuation/fit favors 2
    if n == 2:
        score += 9.0
    elif n == 3:
        score += 0.0
    elif n == 4:
        score += 2.0

    # Prefer balanced line lengths if 2 lines
    if len(lines) == 2:
        score += abs(len(lines[0]) - len(lines[1])) * 0.45

    # Strongly avoid ending a card on a hanging article/preposition unless end of clause
    last_w = str(card[-1]["word"])
    if _is_hanging(card[-1]) and not is_clause_end:
        score += 40.0

    # Reward ending a card cleanly on punctuation; penalize mid-card commas
    if last_w.endswith((",", ".", "!", "?", "\u2014")):
        score -= 22.0
    if any(str(w["word"]).endswith((",", "\u2014")) for w in card[:-1]):
        score += 38.0

    return score


def chunk_words(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Partition a dialogue line into crisp 2-4 word cards with <= 23 chars/line."""
    n = len(words)
    if n <= MIN_CARD_WORDS:
        return [words]

    # Exact DP partitioning into cards of length 2..4
    dp: list[float] = [float("inf")] * (n + 1)
    nxt: list[int] = [-1] * (n + 1)
    dp[n] = 0.0

    for i in range(n - 2, -1, -1):
        for size in range(MIN_CARD_WORDS, MAX_CARD_WORDS + 1):
            j = i + size
            if j > n:
                continue
            if dp[j] == float("inf"):
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
    """Return ASS v4.00+ header configured for 1080x1920 boxless portrait \\kf karaoke."""
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
    """Allocate integer centisecond \\kf durations summing exactly to (t1c - t0c)."""
    n = len(card)
    min_cs = 2
    if t1c - t0c < n * min_cs:
        raise ValueError(f"Card duration {t1c - t0c}cs too short for {n} words")

    durs: list[int] = []
    cur = t0c
    for i, w in enumerate(card):
        if i == n - 1:
            dur = t1c - cur
        else:
            next_start_cs = int(round(float(card[i + 1]["start"]) * 100))
            max_allowed = (t1c - cur) - (n - 1 - i) * min_cs
            dur = max(min_cs, min(max_allowed, next_start_cs - cur))
        durs.append(dur)
        cur += dur

    assert sum(durs) == (t1c - t0c)
    return durs


def write_subtitles(
    lines: list[dict[str, Any]],
    srt_path: Path = SRT_PATH,
    ass_path: Path = ASS_PATH,
    max_duration_s: float = DURATION_S,
) -> dict[str, int]:
    """Generate synchronized .srt cards and word-highlighted \\kf .ass karaoke events."""
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

            t0c = max(prev_cs, int(round(float(card[0]["start"]) * 100)))
            t1c = min(max_cs, max(t0c + 2 * len(card), int(round(float(card[-1]["end"]) * 100))))
            if t1c <= t0c:
                raise ValueError(f"ASS card timing overflow near {card[0]['word']!r}: {t0c}..{t1c}")
            prev_cs = t1c

            kf_durs = _compute_kf_durations_cs(card, t0c, t1c)
            parts: list[str] = []
            for j, cw in enumerate(card):
                tok_raw = str(cw["word"])
                tok = escape_ass_text(tok_raw)
                tagged = f"{{\\kf{kf_durs[j]}}}{tok}"
                if j == sp:
                    parts.append("\\N" + tagged)
                elif j == 0:
                    parts.append(tagged)
                else:
                    sep = (
                        ""
                        if (str(card[j - 1]["word"]).endswith("\u2014") or tok_raw.startswith("\u2014"))
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
    }


def verify_subtitles(
    srt_path: Path,
    ass_path: Path,
    lines: list[dict[str, Any]],
    max_duration_s: float = DURATION_S,
) -> None:
    """Strictly verify generated ASS & SRT files for timing, NDC Y zone, line limits, and word parity."""
    assert srt_path.is_file(), f"Missing SRT file: {srt_path}"
    assert ass_path.is_file(), f"Missing ASS file: {ass_path}"
    font_file = FONTS_DIR / "DejaVuSans-Bold.ttf"
    assert font_file.is_file(), f"Missing bundled font: {font_file}"

    # Verify MarginV places even 2-line cards + outline/shadow strictly inside NDC Y < -0.55
    max_block_height_px = FONT_SIZE * 2.2 + OUTLINE_PX * 2 + SHADOW_PX
    top_px = PLAY_RES_Y - MARGIN_V - max_block_height_px
    top_ndc_y = 1.0 - 2.0 * (top_px / float(PLAY_RES_Y))
    assert top_ndc_y < -0.55, f"Subtitle top NDC Y {top_ndc_y:.3f} must be < -0.55"

    expected_words = [str(w["word"]) for ln in lines for w in ln["words"]]
    expected_text = " ".join(expected_words)

    # 1. Verify ASS
    ass_text = ass_path.read_text(encoding="utf-8")
    assert f"PlayResX: {PLAY_RES_X}" in ass_text
    assert f"PlayResY: {PLAY_RES_Y}" in ass_text
    assert (
        f"Style: Default,{FONT_NAME},{FONT_SIZE},{PRIMARY_COLOUR},{SECONDARY_COLOUR},"
        f"{OUTLINE_COLOUR},{BACK_COLOUR},-1,0,0,0,100,100,0,0,1,{OUTLINE_PX},{SHADOW_PX},"
        f"{ALIGNMENT},{MARGIN_L},{MARGIN_R},{MARGIN_V},1"
    ) in ass_text

    dialogue_lines = [ln for ln in ass_text.splitlines() if ln.startswith("Dialogue:")]
    assert len(dialogue_lines) >= len(lines), f"Too few ASS cards: {len(dialogue_lines)}"

    ass_words_collected: list[str] = []
    prev_end = -1.0
    for d_idx, d_line in enumerate(dialogue_lines, start=1):
        fields = d_line.split(",", 9)
        assert len(fields) == 10, f"Malformed ASS dialogue line {d_idx}: {d_line}"
        t0 = _parse_ass_time(fields[1])
        t1 = _parse_ass_time(fields[2])
        payload = fields[9]
        assert 0.0 <= t0 < t1 <= max_duration_s + 1e-6, (
            f"ASS card {d_idx} out of bounds: {t0:.2f}..{t1:.2f}s (max={max_duration_s:.2f}s)"
        )
        assert t0 >= prev_end - 1e-9, (
            f"ASS overlap at card {d_idx}: t0={t0:.2f} < prev_end={prev_end:.2f}"
        )
        prev_end = t1

        kf_matches = [int(m) for m in _KF_TAG_RE.findall(payload)]
        plain_card = _ASS_TAG_RE.sub("", payload)
        card_lines = plain_card.split("\\N")
        assert 1 <= len(card_lines) <= 2, f"ASS card {d_idx} has {len(card_lines)} lines: {plain_card!r}"
        for cl_txt in card_lines:
            assert len(cl_txt) <= MAX_LINE_CHARS, (
                f"ASS line exceeds {MAX_LINE_CHARS} chars ({len(cl_txt)}): {cl_txt!r}"
            )
        if len(card_lines) == 2:
            last_w1 = card_lines[0].split()[-1].lower().strip(_PUNCT_STRIP)
            assert last_w1 not in HANGING_WORDS, f"Hanging word {last_w1!r} at end of line 1: {plain_card!r}"

        card_words = " ".join(card_lines).split()
        assert MIN_CARD_WORDS <= len(card_words) <= MAX_CARD_WORDS, (
            f"ASS card {d_idx} must have {MIN_CARD_WORDS}..{MAX_CARD_WORDS} words, got {len(card_words)}: {plain_card!r}"
        )
        assert len(kf_matches) == len(card_words), (
            f"ASS card {d_idx} \\kf tag count {len(kf_matches)} != word count {len(card_words)}"
        )
        expected_cs = int(round((t1 - t0) * 100))
        assert sum(kf_matches) == expected_cs, (
            f"ASS card {d_idx} \\kf sum {sum(kf_matches)}cs != duration {expected_cs}cs"
        )
        ass_words_collected.extend(card_words)

    assert " ".join(ass_words_collected) == expected_text, "ASS word sequence does not match manifest!"

    # 2. Verify SRT
    srt_raw = srt_path.read_text(encoding="utf-8").strip()
    blocks = [b.strip() for b in re.split(r"\r?\n\r?\n", srt_raw) if b.strip()]
    assert len(blocks) == len(dialogue_lines), (
        f"SRT card count {len(blocks)} != ASS card count {len(dialogue_lines)}"
    )

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
        assert MIN_CARD_WORDS <= len(card_ws) <= MAX_CARD_WORDS, (
            f"SRT card {b_idx} word count {len(card_ws)} not in {MIN_CARD_WORDS}..{MAX_CARD_WORDS}: {txt_lines!r}"
        )
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
    max_block_height_px = FONT_SIZE * 2.2 + OUTLINE_PX * 2 + SHADOW_PX
    top_ndc_y = 1.0 - 2.0 * ((PLAY_RES_Y - MARGIN_V - max_block_height_px) / float(PLAY_RES_Y))
    print(
        f"[Subtitles] Generated & verified {counts['events']} \\kf karaoke cards "
        f"({counts['words']} words, 0 overlaps, 2-4 words/card, <= {MAX_LINE_CHARS} chars/line, "
        f"<= {DURATION_S:.3f}s, top NDC Y={top_ndc_y:.3f} < -0.55)"
    )
    print(f"  ASS: {ASS_PATH}")
    print(f"  SRT: {SRT_PATH}")


if __name__ == "__main__":
    main()
