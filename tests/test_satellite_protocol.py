"""Satellite wire protocol v1: round-trips, rejection, mood parsing.

stdlib unittest only. Run: .venv/bin/python -m unittest discover -s tests
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satellite import protocol as P  # noqa: E402


class TestRoundTrip(unittest.TestCase):
    """Every message type in §2 encodes and decodes unchanged."""

    CASES = [
        ("hello", {"v": 1, "device": "desk", "client": "satellite-client/0"}),
        ("turn.start", {"turn_id": "t1", "format": "s16le", "rate": 16000, "channels": 1}),
        ("turn.end", {"turn_id": "t1"}),
        ("cancel", {"turn_id": "t1"}),
        ("ready", {"v": 1, "device": "desk", "session_id": "sat-desk-20260929"}),
        ("transcript", {"turn_id": "t1", "text": "hello there", "stt_ms": 240}),
        ("reply", {"turn_id": "t1", "text": "hi", "mood": "happy"}),
        ("audio", {"turn_id": "t1", "format": "wav"}),
        ("turn.done", {"turn_id": "t1", "cancelled": False,
                       "timing": {k: 1 for k in P.TIMING_KEYS}}),
        ("error", {"turn_id": "t1", "code": "busy", "message": "already talking"}),
    ]

    def test_every_type_round_trips(self):
        for type_, fields in self.CASES:
            with self.subTest(type=type_):
                decoded = P.decode(P.encode(type_, **fields))
                self.assertEqual(decoded, {"type": type_, **fields})

    def test_all_types_are_covered(self):
        covered = {t for t, _ in self.CASES}
        self.assertEqual(covered, set(P.CLIENT_TYPES) | set(P.RELAY_TYPES))

    def test_optional_fields_may_be_omitted(self):
        # error without turn_id (socket-scoped), transcript without stt_ms.
        self.assertEqual(
            P.decode(P.encode("error", code="bad_message", message="?"))["code"], "bad_message")
        self.assertNotIn("stt_ms", P.decode(P.encode("transcript", turn_id="t", text="x")))

    def test_turn_start_helper_pins_the_only_v1_format(self):
        frame = json.loads(P.turn_start("t9"))
        self.assertEqual(
            frame, {"type": "turn.start", "turn_id": "t9", "format": "s16le",
                    "rate": 16000, "channels": 1})


class TestRejection(unittest.TestCase):
    def test_unknown_type(self):
        with self.assertRaises(P.ProtocolError) as ctx:
            P.decode('{"type":"turn.pause","turn_id":"t1"}')
        self.assertEqual(ctx.exception.code, "bad_message")

    def test_missing_type(self):
        with self.assertRaises(P.ProtocolError):
            P.decode('{"turn_id":"t1"}')

    def test_missing_required_field(self):
        with self.assertRaises(P.ProtocolError):
            P.decode('{"type":"turn.end"}')

    def test_null_counts_as_missing(self):
        with self.assertRaises(P.ProtocolError):
            P.decode('{"type":"turn.end","turn_id":null}')

    def test_wrong_field_type(self):
        with self.assertRaises(P.ProtocolError):
            P.decode('{"type":"turn.end","turn_id":7}')

    def test_bool_is_not_an_int(self):
        with self.assertRaises(P.ProtocolError):
            P.decode('{"type":"turn.start","turn_id":"t","format":"s16le",'
                     '"rate":true,"channels":1}')

    def test_not_json_and_not_an_object(self):
        for raw in ("not json", "[1,2]", '"hello"'):
            with self.subTest(raw=raw), self.assertRaises(P.ProtocolError):
                P.decode(raw)

    def test_wrong_protocol_version(self):
        with self.assertRaises(P.ProtocolError):
            P.decode('{"type":"hello","v":2,"device":"desk","client":"c"}')

    def test_turn_done_needs_every_timing_key(self):
        with self.assertRaises(P.ProtocolError):
            P.decode('{"type":"turn.done","turn_id":"t","cancelled":false,'
                     '"timing":{"total_ms":5}}')

    def test_direction_schemas_are_enforced(self):
        # A client must not be able to send relay frames, and vice versa.
        ready = P.encode("ready", v=1, device="desk", session_id="s")
        with self.assertRaises(P.ProtocolError):
            P.decode(ready, schema=P.CLIENT_SCHEMA)
        with self.assertRaises(P.ProtocolError):
            P.decode(P.encode("turn.end", turn_id="t"), schema=P.RELAY_SCHEMA)
        self.assertEqual(P.decode(ready, schema=P.RELAY_SCHEMA)["type"], "ready")


class TestParseMood(unittest.TestCase):
    def test_valid_tag_is_stripped(self):
        self.assertEqual(P.parse_mood("All set.\n[[mood:happy]]"), ("All set.", "happy"))

    def test_every_documented_mood_is_accepted(self):
        for mood in P.MOODS:
            with self.subTest(mood=mood):
                self.assertEqual(P.parse_mood(f"x\n[[mood:{mood}]]")[1], mood)

    def test_invalid_value_falls_back_to_neutral_and_is_still_stripped(self):
        self.assertEqual(P.parse_mood("x\n[[mood:furious]]"), ("x", "neutral"))

    def test_missing_tag_is_neutral(self):
        self.assertEqual(P.parse_mood("just a reply"), ("just a reply", "neutral"))

    def test_trailing_whitespace_after_the_tag_is_tolerated(self):
        self.assertEqual(P.parse_mood("x\n[[mood:curious]]  \n\n"), ("x", "curious"))

    def test_tag_mid_text_is_left_alone(self):
        text = "I would say [[mood:happy]] but not at the end"
        self.assertEqual(P.parse_mood(text), (text, "neutral"))

    def test_only_the_trailing_tag_is_removed(self):
        spoken, mood = P.parse_mood("quoting [[mood:happy]] inline\n[[mood:playful]]")
        self.assertEqual(spoken, "quoting [[mood:happy]] inline")
        self.assertEqual(mood, "playful")

    def test_non_string_input(self):
        self.assertEqual(P.parse_mood(None), ("", "neutral"))  # type: ignore[arg-type]


class TestTruncateForTTS(unittest.TestCase):
    def test_short_text_untouched(self):
        self.assertEqual(P.truncate_for_tts("hello", 100), "hello")

    def test_cuts_at_the_last_sentence_end(self):
        text = "One. Two. Three. " + "x" * 50
        out = P.truncate_for_tts(text, 20)
        self.assertEqual(out, "One. Two. Three.")
        self.assertLessEqual(len(out), 20)

    def test_falls_back_to_a_word_boundary(self):
        out = P.truncate_for_tts("alpha beta gamma delta", 14)
        self.assertEqual(out, "alpha beta")

    def test_never_exceeds_the_limit(self):
        self.assertLessEqual(len(P.truncate_for_tts("a" * 500, 40)), 40)


class TestConstants(unittest.TestCase):
    def test_audio_cap_matches_thirty_seconds(self):
        self.assertEqual(P.MAX_AUDIO_BYTES, 960_000)
        self.assertEqual(P.MAX_AUDIO_BYTES, P.RATE * P.CHANNELS * P.SAMPLE_WIDTH * 30)

    def test_client_record_cap_is_below_the_relay_hard_cap(self):
        self.assertLess(P.RECORD_CAP_SECONDS, P.MAX_AUDIO_SECONDS)

    def test_run_status_sets_are_disjoint(self):
        self.assertFalse(set(P.TERMINAL_RUN_STATUSES) & set(P.NONTERMINAL_RUN_STATUSES))

    def test_timing_fills_every_key_with_none(self):
        self.assertEqual(P.timing(stt_ms=3),
                         {"audio_ms": None, "stt_ms": 3, "hermes_ms": None,
                          "tts_ms": None, "total_ms": None})


if __name__ == "__main__":
    unittest.main()
