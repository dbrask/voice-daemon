"""Spoken-text normalisation: markdown/symbols/units in, a speakable script out.

Portions of this file (the markdown/symbol regexes and the unit/currency tables)
are adapted from ``tools/tts_text_normalize.py`` in Hermes Agent,
Copyright (c) 2025 Nous Research, MIT License. See THIRD_PARTY_NOTICES.md.

Phase 3b. The rules here are a deliberate re-implementation of the behaviour of
Hermes' ``tools/tts_text_normalize.py`` — nothing from hermes-agent is imported
(invariant 6), and nothing here knows about engines, PCM, ffmpeg or HTTP.

One difference from the Hermes version, and it is the point of the phase: the
last step does **not** flatten the whole script onto one line. Paragraph breaks
survive as ``\\n\\n`` so the daemon can put real silence between paragraphs
instead of speaking a pause. Inside a paragraph the text is a single line,
because newline-sensitive engines truncate at the first ``\\n``.

Pipeline (``prepare_spoken_text``):

    non-spoken blocks -> markdown -> symbols/units -> line smoothing ->
    per-paragraph flattening

Non-ASCII characters are written as escapes on purpose, so this file stays free
of invisible / look-alike glyphs.
"""

from __future__ import annotations

import html
import re
from typing import List

# Sentinel appended to former heading lines so _smooth folds the heading into
# the sentence after it ("Weather, it will be sunny") instead of leaving a bare
# "Weather." label.
_HEAD = "\x00"

_MD_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^()]|\([^)]*\))*\)")
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((?:[^()]|\([^)]*\))*\)")
_MD_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", flags=re.DOTALL)
_MD_UNDERSCORE_BOLD_RE = re.compile(r"__(.+?)__", flags=re.DOTALL)
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", flags=re.DOTALL)
_MD_UNDERSCORE_ITALIC_RE = re.compile(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", flags=re.DOTALL)
_MD_STRIKE_RE = re.compile(r"~~(.+?)~~", flags=re.DOTALL)
_MD_HEADING_LINE_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", flags=re.MULTILINE)
_MD_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?", flags=re.MULTILINE)
_MD_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", flags=re.MULTILINE)
_MD_HR_RE = re.compile(r"^\s*[-*_]{3,}\s*$", flags=re.MULTILINE)
_MD_TABLE_PIPE_RE = re.compile(r"\s*\|\s*")
_URL_RE = re.compile(r"https?://\S+")

# ``<think>`` reasoning blocks are a UI affordance: readers want to see them,
# never to hear them. An unterminated block (a cut-off stream) is silenced too.
_THINK_BLOCK_RE = re.compile(r"<think[\s>].*?</think>", flags=re.DOTALL | re.IGNORECASE)
_THINK_BLOCK_OPEN_RE = re.compile(r"<think[\s>].*\Z", flags=re.DOTALL | re.IGNORECASE)

_DEGREE_UNITS = (("C", "Celsius"), ("F", "Fahrenheit"))
# Unit suffix (regex, after a digit) -> spoken word; km/h variants before bare "m".
_UNIT_WORDS = (
    (r"km\s*/\s*h", "kilometres per hour"), (r"km/h", "kilometres per hour"),
    (r"mm", "millimetres"), (r"cm", "centimetres"), (r"km", "kilometres"), (r"m", "metres"),
)
# Currency prefix (regex) -> spoken word; order matters (NZ$/A$/US$ before bare $).
_CURRENCY_WORDS = (
    (r"NZ\$", "New Zealand dollars", re.IGNORECASE), (r"A\$", "Australian dollars", re.IGNORECASE),
    (r"US\$", "US dollars", re.IGNORECASE),
    ("\u20ac", "euros", 0), ("\u00a3", "pounds", 0), (r"\$", "dollars", 0),
)

# Broad emoji / pictograph cleanup: TTS engines read emoji as awkward labels.
_EMOJI_RE = re.compile(
    "[\U0001F1E6-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF\u2600-\u27bf]+",
    flags=re.UNICODE)
_VARIATION_SELECTOR_RE = re.compile("[\ufe0e\ufe0f]")

_PARAGRAPH_SPLIT_RE = re.compile(r"\n{2,}")


def strip_nonspoken_blocks(text: str) -> str:
    """Remove blocks that are for the eye only (``<think>`` reasoning)."""
    if not text:
        return ""
    for pattern in (_THINK_BLOCK_RE, _THINK_BLOCK_OPEN_RE):
        text = pattern.sub(" ", text)
    return text


def strip_markdown(text: str) -> str:
    """Strip Markdown formatting while keeping the readable words.

    Code fences go entirely (a spoken diff is noise); inline code keeps its
    text; link text is kept and the target dropped; a bare URL is dropped
    rather than spelled out character by character.
    """
    if not text:
        return ""
    text = html.unescape(str(text))
    text = _MD_CODE_BLOCK_RE.sub(" ", text)
    text = _MD_IMAGE_RE.sub(lambda m: f" {m.group(1)} " if m.group(1) else " ", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub("", text)
    text = _MD_INLINE_CODE_RE.sub(r"\1", text)
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_UNDERSCORE_BOLD_RE.sub(r"\1", text)
    text = _MD_ITALIC_RE.sub(r"\1", text)
    text = _MD_UNDERSCORE_ITALIC_RE.sub(r"\1", text)
    text = _MD_STRIKE_RE.sub(r"\1", text)
    # Mark headings rather than deleting the marker: see _HEAD.
    text = _MD_HEADING_LINE_RE.sub(lambda m: m.group(1).rstrip() + _HEAD, text)
    text = _MD_BLOCKQUOTE_RE.sub("", text)
    text = _MD_LIST_ITEM_RE.sub("", text)
    text = _MD_HR_RE.sub("", text)
    # Leftover table pipes become pauses, not a spoken "vertical bar".
    return _MD_TABLE_PIPE_RE.sub("; ", text)


def _temperature_ranges(text: str) -> str:
    """``11-17\u00b0C`` -> ``11 to 17 degrees Celsius`` (hyphen, en/em dash, unicode minus)."""
    number = r"([-+\u2212]?\d+(?:\.\d+)?)"
    for unit, word in _DEGREE_UNITS:
        text = re.sub(
            r"(?<!\w)" + number + r"\s*[\u2013\u2014-]\s*" + number + r"\s*\u00b0\s*" + unit + r"\b",
            lambda m, w=word: (
                f"{m.group(1).replace(chr(0x2212), '-')} to "
                f"{m.group(2).replace(chr(0x2212), '-')} degrees {w}"
            ),
            text, flags=re.IGNORECASE)
    return text


def normalize_symbols(text: str) -> str:
    """Expand symbols, units, money and percentages into words."""
    if not text:
        return ""
    text = re.sub("[\u00a0\u2007\u202f\u2009]", " ", str(text))  # nbsp / thin spaces
    text = text.replace("\u2212", "-").replace("\u2026", "...")  # minus sign, ellipsis
    text = _temperature_ranges(text)
    # Number then unit, then a bare unit ("measured in degrees C"), then any
    # stray degree symbol (angles).
    for unit, word in _DEGREE_UNITS:
        text = re.sub(r"(?<!\w)([-+]?\d+(?:\.\d+)?)\s*\u00b0\s*" + unit + r"\b",
                      r"\1 degrees " + word, text, flags=re.IGNORECASE)
    for unit, word in _DEGREE_UNITS:
        text = re.sub(r"\u00b0\s*" + unit + r"\b", "degrees " + word, text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)([-+]?\d+(?:\.\d+)?)\s*\u00b0", r"\1 degrees", text).replace("\u00b0", " degrees")
    for pattern, word in _UNIT_WORDS:
        text = re.sub(r"(?<=\d)\s*" + pattern + r"\b", " " + word, text, flags=re.IGNORECASE)
    # Numeric rates only ("5/month" -> "5 per month"). Requiring digit-then-letter
    # keeps "and/or", "N/A", "TCP/IP" and dates like "2026/06" intact.
    text = re.sub(r"(?<=\d)\s*/\s*(?=[A-Za-z])", " per ", text)
    # Money: the integer part must END in a digit so a trailing comma
    # ("A$50, then") is not swallowed. Prefixed currencies first, so "$" does
    # not eat the "NZ" of "NZ$".
    for symbol, word, flags in _CURRENCY_WORDS:
        text = re.sub(symbol + r"\s*([\d,]*\d(?:\.\d+)?)", r"\1 " + word, text, flags=flags)
    text = re.sub(r"(?<=\d)\s*%", " percent", text)
    text = re.sub("[\u2022\u25e6\u25aa\u25ab]", " ", text.replace("&", " and "))  # bullet glyphs
    for symbol, word in (("\u2192", " to "), ("\u21d2", " to "),
                         ("\u2248", " about "), ("~", " about ")):
        text = text.replace(symbol, word)
    return _EMOJI_RE.sub("", _VARIATION_SELECTOR_RE.sub("", text))


def smooth_lines(text: str) -> str:
    """Collapse visual layout into calm sentences, keeping blank-line paragraphs.

    A ``_HEAD``-marked heading folds into the next content line as a lead-in
    ("Weather, it will be sunny."); a heading with nothing after it becomes its
    own sentence.
    """
    if not text:
        return ""
    raw_lines = text.splitlines()
    add_sentence_pauses = sum(1 for raw in raw_lines if raw.replace(_HEAD, "").strip()) > 1
    lines: List[str] = []
    pending_heading = None

    def flush_pending() -> None:
        nonlocal pending_heading
        if pending_heading is not None:
            lines.append(pending_heading.rstrip(".:;,") + ".")
            pending_heading = None

    for raw_line in raw_lines:
        is_heading = raw_line.rstrip().endswith(_HEAD)
        line = raw_line.replace(_HEAD, "").strip()
        if not line:
            # Hold a pending heading across blank lines so it still folds into
            # the next content line.
            if pending_heading is None and lines and lines[-1] != "":
                lines.append("")
            continue
        if is_heading:
            flush_pending()
            pending_heading = line.rstrip(".:;,")
            continue
        if pending_heading is not None:
            line = f"{pending_heading.rstrip('.:;,')}, {line}"
            pending_heading = None
        if add_sentence_pauses and line[-1] not in ".!?;:":
            line += "."
        lines.append(line)
    flush_pending()
    text = "\n".join(lines)
    for pattern, repl in ((r"\n{3,}", "\n\n"), (r"[ \t]{2,}", " "), (r"\s+([,.;:!?])", r"\1"),
                          (r"([,.;:!?])([A-Za-z])", r"\1 \2"), (r"\.{4,}", "...")):
        text = re.sub(pattern, repl, text)
    return text.strip()


def flatten_paragraphs(text: str) -> str:
    """One line per paragraph, blank line between paragraphs.

    Engines such as Kokoro truncate a payload at the first newline, so a
    paragraph must be a single line. The paragraph break itself survives: the
    daemon turns it into silence, never into a spoken word.
    """
    if not text:
        return ""
    out: List[str] = []
    for para in _PARAGRAPH_SPLIT_RE.split(text):
        for pattern, repl in ((r"(?<=[.!?;:,])\n", " "), (r"\n", ". "), (r"\.\s*\.", "."),
                              (r"[ \t]{2,}", " ")):
            para = re.sub(pattern, repl, para)
        para = para.strip()
        if para:
            out.append(para)
    return "\n\n".join(out)


def split_paragraphs(text: str) -> List[str]:
    """Paragraphs of an already-prepared script (blank line separated)."""
    return [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text or "") if p.strip()]


def prepare_spoken_text(text: str, max_chars: int | None = None) -> str:
    """Deterministic cleanup of *text* into a spoken script. Not a rewrite.

    Returns paragraphs separated by a blank line; each paragraph is one line.
    May return "" (e.g. the input was only a code fence or a bare URL) — the
    caller decides whether that is an error.
    """
    spoken = text or ""
    for step in (strip_nonspoken_blocks, strip_markdown, normalize_symbols,
                 smooth_lines, flatten_paragraphs):
        spoken = step(spoken)
    if max_chars is not None and max_chars > 0 and len(spoken) > max_chars:
        spoken = spoken[:max_chars].rstrip()
    return spoken
