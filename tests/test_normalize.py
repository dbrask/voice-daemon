"""Phase 3b: the spoken-text normaliser and the paragraph-pause plan.

Pure functions, no GPU, no engine. Run:
    .venv/bin/python -m unittest discover -s tests
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import voice_daemon as vd  # noqa: E402
from normalize import (  # noqa: E402
    normalize_symbols,
    prepare_spoken_text,
    smooth_lines,
    split_paragraphs,
    strip_markdown,
    strip_nonspoken_blocks,
)


class MarkdownTests(unittest.TestCase):
    def test_code_fences_are_not_spoken(self):
        spoken = prepare_spoken_text("Here it is.\n\n```python\nx = 1\nprint(x)\n```\n\nDone.")
        self.assertNotIn("print", spoken)
        self.assertNotIn("```", spoken)
        self.assertIn("Here it is.", spoken)
        self.assertIn("Done.", spoken)

    def test_inline_code_keeps_its_words(self):
        self.assertIn("cargo test", prepare_spoken_text("Run `cargo test` first."))
        self.assertNotIn("`", prepare_spoken_text("Run `cargo test` first."))

    def test_links_keep_the_label_and_drop_the_target(self):
        spoken = prepare_spoken_text("See [the docs](https://example.com/a_b_c) now.")
        self.assertIn("the docs", spoken)
        self.assertNotIn("example.com", spoken)
        self.assertNotIn("https", spoken)

    def test_bare_urls_are_dropped_not_spelled_out(self):
        spoken = prepare_spoken_text("Go to https://example.com/x?y=1 for more.")
        self.assertNotIn("example", spoken)
        self.assertNotIn("http", spoken)

    def test_emphasis_and_lists_and_rules_lose_their_markers(self):
        spoken = prepare_spoken_text("**Bold** and _italic_ and ~~gone~~.\n\n- one\n- two\n\n---\n")
        for marker in ("**", "~~", "---", "- "):
            self.assertNotIn(marker, spoken)
        self.assertIn("Bold", spoken)
        self.assertIn("italic", spoken)

    def test_headings_fold_into_the_following_sentence(self):
        spoken = prepare_spoken_text("# Weather\n\nIt will be sunny.")
        self.assertEqual(spoken, "Weather, It will be sunny.")
        self.assertNotIn("#", spoken)

    def test_heading_with_no_body_becomes_its_own_sentence(self):
        self.assertEqual(prepare_spoken_text("## The End\n"), "The End.")

    def test_table_pipes_become_pauses(self):
        spoken = prepare_spoken_text("| a | b |")
        self.assertNotIn("|", spoken)
        self.assertIn(";", spoken)

    def test_html_entities_are_unescaped(self):
        self.assertIn("Tom & Jerry", strip_markdown("Tom &amp; Jerry"))

    def test_think_blocks_are_never_spoken(self):
        self.assertNotIn("secret", prepare_spoken_text("<think>secret</think>Hello."))
        # An unterminated block (a cut-off stream) is silenced too.
        self.assertNotIn("secret", strip_nonspoken_blocks("Hi. <think>secret and on"))


class SymbolTests(unittest.TestCase):
    def test_temperatures(self):
        self.assertIn("11 to 17 degrees Celsius", normalize_symbols("11-17\u00b0C"))
        self.assertIn("21 degrees Celsius", normalize_symbols("21\u00b0C"))
        self.assertIn("70 degrees Fahrenheit", normalize_symbols("70\u00b0F"))
        self.assertIn("45 degrees", normalize_symbols("45\u00b0"))

    def test_units(self):
        self.assertIn("30 kilometres per hour", normalize_symbols("30 km/h"))
        self.assertIn("5 millimetres", normalize_symbols("5mm"))
        self.assertIn("12 centimetres", normalize_symbols("12cm"))
        self.assertIn("400 metres", normalize_symbols("400m"))
        self.assertIn("7 kilometres", normalize_symbols("7km"))

    def test_money_and_percent(self):
        self.assertIn("50 dollars", normalize_symbols("$50"))
        self.assertIn("1,200.50 euros", normalize_symbols("\u20ac1,200.50"))
        self.assertIn("30 pounds", normalize_symbols("\u00a330"))
        self.assertIn("99 percent", normalize_symbols("99%"))

    def test_prefixed_currencies_win_over_the_bare_dollar(self):
        spoken = normalize_symbols("NZ$40, A$50 and US$60.")
        self.assertIn("40 New Zealand dollars", spoken)
        self.assertIn("50 Australian dollars", spoken)
        self.assertIn("60 US dollars", spoken)
        self.assertNotIn("$", spoken)
        self.assertIn("40 New Zealand dollars,", spoken)  # trailing comma survives

    def test_numeric_rates_only(self):
        self.assertIn("5 per month", normalize_symbols("5/month"))
        for intact in ("and/or", "N/A", "TCP/IP", "2026/06"):
            self.assertIn(intact, normalize_symbols(intact))

    def test_arrows_ampersand_and_emoji(self):
        self.assertIn(" to ", normalize_symbols("A \u2192 B"))
        self.assertIn(" about ", normalize_symbols("~20"))
        self.assertIn(" and ", normalize_symbols("this & that"))
        self.assertEqual(normalize_symbols("nice \U0001F600").strip(), "nice")

    def test_empty_input_is_empty_output(self):
        for fn in (strip_markdown, normalize_symbols, smooth_lines, prepare_spoken_text):
            self.assertEqual(fn(""), "")


class ParagraphTests(unittest.TestCase):
    def test_paragraph_breaks_survive_as_blank_lines(self):
        spoken = prepare_spoken_text("First para line one.\nStill one.\n\nSecond para.")
        self.assertEqual(len(split_paragraphs(spoken)), 2)
        self.assertNotIn("pause", spoken.lower())

    def test_a_paragraph_is_a_single_line(self):
        spoken = prepare_spoken_text("Line one\nline two\n\nNext.")
        for para in split_paragraphs(spoken):
            self.assertNotIn("\n", para)

    def test_single_paragraph_has_no_break(self):
        self.assertEqual(split_paragraphs(prepare_spoken_text("Just one sentence.")),
                         ["Just one sentence."])

    def test_max_chars_caps_the_script(self):
        self.assertLessEqual(len(prepare_spoken_text("word " * 500, max_chars=100)), 100)


class PlanTests(unittest.TestCase):
    """The daemon's side: pauses are silence items, not spoken text."""

    def setUp(self):
        self._pause = vd.PARAGRAPH_PAUSE_S
        self._normalize = vd.NORMALIZE
        vd.PARAGRAPH_PAUSE_S = 0.4
        vd.NORMALIZE = True

    def tearDown(self):
        vd.PARAGRAPH_PAUSE_S = self._pause
        vd.NORMALIZE = self._normalize

    def test_plan_inserts_a_pause_between_paragraphs_only(self):
        plan = vd.plan_speech("One.\n\nTwo.\n\nThree.", 200)
        self.assertEqual([k for k, _ in plan],
                         ["text", "pause", "text", "pause", "text"])
        self.assertEqual([v for k, v in plan if k == "text"], ["One.", "Two.", "Three."])

    def test_no_pause_for_a_single_paragraph(self):
        self.assertEqual([k for k, _ in vd.plan_speech("Only one.", 200)], ["text"])

    def test_a_chunk_never_straddles_a_paragraph_break(self):
        plan = vd.plan_speech("Short.\n\nAlso short.", 200)
        for kind, value in plan:
            if kind == "text":
                self.assertNotIn("\n", value)
        self.assertIn("pause", [k for k, _ in plan])

    def test_first_limit_applies_once_to_the_whole_script(self):
        text = "Short lead. " + "A longer sentence that keeps going a while. " * 3 \
               + "\n\n" + "Another paragraph sentence here. " * 3
        chunks = [v for k, v in vd.plan_speech(text, 200, first_limit=40)]
        self.assertEqual(chunks[0], "Short lead.")
        # The second paragraph's first chunk uses the full limit, not 40.
        after_pause = [v for k, v in vd.plan_speech(text, 200, first_limit=40)][1:]
        self.assertTrue(any(isinstance(v, str) and len(v) > 40 for v in after_pause))

    def test_pause_disabled_by_zero(self):
        vd.PARAGRAPH_PAUSE_S = 0.0
        self.assertEqual([k for k, _ in vd.plan_speech("One.\n\nTwo.", 200)], ["text", "text"])

    def test_silence_pcm_is_int16_zeros_of_the_right_length(self):
        pcm = vd.silence_pcm(0.5, 16000)
        self.assertEqual(pcm.dtype, np.int16)
        self.assertEqual(pcm.size, 8000)
        self.assertFalse(pcm.any())
        self.assertEqual(vd.silence_pcm(-1, 16000).size, 0)

    def test_spoken_script_normalises_and_can_be_switched_off(self):
        self.assertEqual(vd.spoken_script("**Hi** at 20\u00b0C."), "Hi at 20 degrees Celsius.")
        vd.NORMALIZE = False
        self.assertEqual(vd.spoken_script("**Hi**"), "**Hi**")

    def test_script_that_normalises_to_nothing_falls_back_to_raw(self):
        # A bare URL survives as itself rather than becoming a silent request.
        self.assertEqual(vd.spoken_script("https://example.com"), "https://example.com")


if __name__ == "__main__":
    unittest.main()
