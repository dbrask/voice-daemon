#!/usr/bin/env python3
"""Build a `/tts/long` `chapters` payload from a text file.

Only used by `bin/hermes-tts-long --auto-chapter`. The daemon already splits a
document on markdown headings, but extracted text (a PDF through `pdftotext`, a
scraped article) frequently has no headings at all, so the whole book becomes
one 40-minute file. This splits any chapter longer than `--max-chars` at
paragraph boundaries — never mid-sentence — and emits the explicit `chapters`
array the daemon accepts, so the split is visible in the job snapshot and in
the per-chapter filenames.

Heading chaptering itself is `jobs.split_chapters`, imported rather than
re-implemented, so the two paths cannot drift.

    chapterize.py --split headings --max-chars 4000 book.txt > payload.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jobs import split_chapters  # noqa: E402

_PARA = re.compile(r"\n[ \t]*\n")


def split_long(title: str, text: str, max_chars: int):
    """Yield `(title, text)` pieces no longer than `max_chars` where possible.

    Greedy over paragraphs: a paragraph is never cut, so a single paragraph
    longer than `max_chars` stays whole (the daemon chunks it for the engine
    anyway; only the file boundary is at stake here). Parts after the first are
    titled `<title> (2)` so the job snapshot and filenames stay readable.
    """
    if len(text) <= max_chars:
        return [(title, text)]
    parts, buf = [], ""
    for para in _PARA.split(text):
        para = para.strip()
        if not para:
            continue
        if buf and len(buf) + 2 + len(para) > max_chars:
            parts.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        parts.append(buf)
    if len(parts) == 1:
        return [(title, parts[0])]
    base = title or "part"
    return [(base if i == 1 else f"{base} ({i})", body)
            for i, body in enumerate(parts, start=1)]


def chapterize(text: str, split: str = "headings", max_chars: int = 4000):
    out = []
    for title, body in split_chapters(text, split):
        out.extend(split_long(title, body, max_chars))
    if all(not t or t.startswith("part") for t, _ in out) and len(out) > 1:
        # Nothing in the source was a real heading (extracted PDF text): number
        # the parts so the filenames and the job snapshot stay navigable.
        out = [(f"Part {i}", body) for i, (_t, body) in enumerate(out, start=1)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="text file, or - for stdin")
    ap.add_argument("--split", default="headings", choices=("headings", "none"))
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--voice", default=None)
    ap.add_argument("--format", default=None, dest="fmt")
    args = ap.parse_args()

    text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8", errors="replace")
    chapters = chapterize(text, args.split, args.max_chars)
    if not chapters:
        print("chapterize: nothing to speak", file=sys.stderr)
        return 1
    payload = {"chapters": [{"title": t, "text": b} for t, b in chapters]}
    if args.voice:
        payload["voice"] = args.voice
    if args.fmt:
        payload["format"] = args.fmt
    json.dump(payload, sys.stdout, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
