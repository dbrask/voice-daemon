"""Satellite wire protocol v1 — message types, validation, mood parsing.

Imported by both ends (relay on the GPU host, client on the Pi), so it stays
stdlib-only: no aiohttp, no daemon imports. See the README's satellite section
for the authoritative table; this module is that table in code.

One socket, one turn in flight per device. Text frames are JSON objects with
a `type`; binary frames are audio belonging to the current turn.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple

V = 1

# Audio the client sends up: raw little-endian signed 16-bit mono PCM.
FORMAT = "s16le"
RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2

# 30 s cap at 16 kHz mono s16le = 960 000 bytes.
MAX_AUDIO_SECONDS = 30
MAX_AUDIO_BYTES = RATE * CHANNELS * SAMPLE_WIDTH * MAX_AUDIO_SECONDS
# Client-side record cap before auto-stop (shorter than the relay's hard cap).
RECORD_CAP_SECONDS = 20
# Per-binary-frame ceiling; the client chunks while the user speaks.
MAX_CHUNK_BYTES = 64 * 1024

STATES = ("idle", "listening", "thinking", "speaking", "error", "offline")
MOODS = ("neutral", "happy", "curious", "concerned", "playful")
DEFAULT_MOOD = "neutral"

# Close code for a `hello` whose device does not match the token's device.
CLOSE_DEVICE_MISMATCH = 4401

ERROR_CODES = (
    "bad_message",
    "busy",
    "too_long",
    "empty_transcript",
    "stt_failed",
    "hermes_failed",
    "hermes_timeout",
    "tts_failed",
)

# Hermes run statuses. Non-terminal ones mean "keep polling".
TERMINAL_RUN_STATUSES = ("completed", "failed", "cancelled", "interrupted")
NONTERMINAL_RUN_STATUSES = ("queued", "running", "waiting_for_approval", "stopping")

TIMING_KEYS = ("audio_ms", "stt_ms", "hermes_ms", "tts_ms", "total_ms")

_MOOD_RE = re.compile(r"\[\[mood:([a-z]+)\]\]\s*$")


# ---- schema -------------------------------------------------------------
#
# field -> (python type, required). `None` type means "any JSON value".

_STR = (str, True)
_INT = (int, True)
_BOOL = (bool, True)
_DICT = (dict, True)

CLIENT_SCHEMA: Dict[str, Dict[str, tuple]] = {
    "hello": {"v": _INT, "device": _STR, "client": _STR},
    "turn.start": {"turn_id": _STR, "format": _STR, "rate": _INT, "channels": _INT},
    "turn.end": {"turn_id": _STR},
    "cancel": {"turn_id": _STR},
}

RELAY_SCHEMA: Dict[str, Dict[str, tuple]] = {
    "ready": {"v": _INT, "device": _STR, "session_id": _STR},
    "transcript": {"turn_id": _STR, "text": _STR, "stt_ms": (int, False)},
    "reply": {"turn_id": _STR, "text": _STR, "mood": _STR},
    "audio": {"turn_id": _STR, "format": _STR},
    "turn.done": {"turn_id": _STR, "cancelled": _BOOL, "timing": _DICT},
    "error": {"turn_id": (str, False), "code": _STR, "message": _STR},
}

CLIENT_TYPES = tuple(CLIENT_SCHEMA)
RELAY_TYPES = tuple(RELAY_SCHEMA)


class ProtocolError(ValueError):
    """A frame that violates the v1 schema. Carries a protocol error code."""

    def __init__(self, message: str, code: str = "bad_message") -> None:
        super().__init__(message)
        self.code = code


def encode(type_: str, **fields: Any) -> str:
    """Serialise one text frame. Validates against the schema on the way out."""
    msg = {"type": type_, **fields}
    validate(msg)
    return json.dumps(msg, separators=(",", ":"))


def decode(raw: str, *, schema: Dict[str, Dict[str, tuple]] | None = None) -> Dict[str, Any]:
    """Parse and validate one text frame. Raises ProtocolError on anything odd."""
    try:
        msg = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ProtocolError(f"not JSON: {exc}") from exc
    if not isinstance(msg, dict):
        raise ProtocolError("frame must be a JSON object")
    return validate(msg, schema=schema)


def validate(msg: Dict[str, Any], *, schema: Dict[str, Dict[str, tuple]] | None = None) -> Dict[str, Any]:
    """Check one decoded frame. Returns it unchanged so callers can chain.

    `schema` restricts the accepted directions: pass CLIENT_SCHEMA on the relay
    so a client cannot send relay-only frames, and vice versa. Default accepts
    both (used by encode() and by tests).
    """
    if not isinstance(msg, dict):
        raise ProtocolError("frame must be a JSON object")
    type_ = msg.get("type")
    if not isinstance(type_, str) or not type_:
        raise ProtocolError("missing 'type'")
    if schema is None:
        fields = CLIENT_SCHEMA.get(type_) or RELAY_SCHEMA.get(type_)
    else:
        fields = schema.get(type_)
    if fields is None:
        raise ProtocolError(f"unknown type {type_!r}")
    for name, (expected, required) in fields.items():
        if name not in msg or msg[name] is None:
            if required:
                raise ProtocolError(f"{type_}: missing field {name!r}")
            continue
        value = msg[name]
        # bool is a subclass of int; keep them apart so rate=True is rejected.
        if expected is int and isinstance(value, bool):
            raise ProtocolError(f"{type_}: field {name!r} must be int, got bool")
        if expected is not None and not isinstance(value, expected):
            raise ProtocolError(
                f"{type_}: field {name!r} must be {expected.__name__}, got {type(value).__name__}")
    if type_ == "hello" and msg["v"] != V:
        raise ProtocolError(f"unsupported protocol version {msg['v']!r}")
    if type_ == "turn.done":
        missing = [k for k in TIMING_KEYS if k not in msg["timing"]]
        if missing:
            raise ProtocolError(f"turn.done: timing missing {missing}")
    return msg


def turn_start(turn_id: str) -> str:
    """The client's `turn.start` with the only audio format v1 defines."""
    return encode("turn.start", turn_id=turn_id, format=FORMAT, rate=RATE, channels=CHANNELS)


def timing(**stages: Any) -> Dict[str, Any]:
    """A full timing dict: every key present, stages that did not run are None."""
    return {key: stages.get(key) for key in TIMING_KEYS}


def parse_mood(text: str) -> Tuple[str, str]:
    """Split a reply into (spoken_text, mood).

    The tag is `[[mood:X]]` on its own line at the very end. A value outside
    MOODS, or no tag at all, is `neutral`. A tag mid-text is left alone: the
    daemon's normaliser would read it aloud, but stripping mid-text tags would
    silently mangle a reply that legitimately quotes one.
    """
    if not isinstance(text, str):
        return "", DEFAULT_MOOD
    match = _MOOD_RE.search(text)
    if not match:
        return text.strip(), DEFAULT_MOOD
    mood = match.group(1)
    if mood not in MOODS:
        mood = DEFAULT_MOOD
    return text[: match.start()].strip(), mood


def truncate_for_tts(text: str, limit: int) -> str:
    """Trim a reply to `limit` chars at the last sentence end before the cap.

    The daemon 400s over VOICE_TTS_MAX_TEXT_CHARS, so the relay truncates for
    speech only; `reply.text` on the wire stays full.
    """
    if len(text) <= limit:
        return text
    head = text[:limit]
    cut = max(head.rfind(c) for c in ".!?…")
    if cut > limit // 4:
        return head[: cut + 1].strip()
    cut = head.rfind(" ")
    return (head[:cut] if cut > 0 else head).strip()
