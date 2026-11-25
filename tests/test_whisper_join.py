"""Pure-CPU tests for the whisper_ct2 transcript join and gazetteer prompt.

Importing ``engines.whisper_ct2`` must not pull faster_whisper/torch — the
import lives inside ``__init__`` — so these run anywhere.

Run:  .venv/bin/python -m unittest discover -s tests -v
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engines import whisper_ct2 as w  # noqa: E402


class FakeSegment:
    """Stands in for faster_whisper.Segment: only ``.text`` is read."""

    def __init__(self, text):
        self.text = text


class JoinSegmentsTests(unittest.TestCase):
    def test_import_is_cheap(self):
        # Importing engines.whisper_ct2 must not pull faster_whisper/torch.
        # Checked in a fresh interpreter: GPU tests in this process may already
        # have loaded the real model, which would make a sys.modules assertion
        # here fail even when the module itself is still lazy.
        repo = str(Path(__file__).resolve().parents[1])
        script = (
            "import sys; sys.path.insert(0, %r);"
            "import engines.whisper_ct2;"
            "assert 'faster_whisper' not in sys.modules, 'faster_whisper imported at module scope';"
            "assert 'torch' not in sys.modules, 'torch imported at module scope';"
            "print('cheap')" % repo
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            cwd=repo,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("cheap", proc.stdout)

    def test_punctuation_and_case_survive_the_join(self):
        segs = [FakeSegment(" Pokrovsk is holding."),
                FakeSegment(" Witkoff flew to Moscow;"),
                FakeSegment(" Ushakov denied it.")]
        self.assertEqual(
            w.join_segments(segs),
            "Pokrovsk is holding. Witkoff flew to Moscow; Ushakov denied it.")

    def test_segments_without_leading_space_do_not_glue(self):
        # The batched pipeline strips some leading spaces; "".join used to
        # produce "holding.Witkoff".
        segs = [FakeSegment(" Pokrovsk is holding."), FakeSegment("Witkoff flew.")]
        self.assertEqual(w.join_segments(segs), "Pokrovsk is holding. Witkoff flew.")

    def test_whitespace_is_normalised_not_the_text(self):
        segs = [FakeSegment("  Sloviansk\nand   Kramatorsk. "), FakeSegment("\t")]
        self.assertEqual(w.join_segments(segs), "Sloviansk and Kramatorsk.")

    def test_empty_and_non_string_segments(self):
        self.assertEqual(w.join_segments([]), "")
        self.assertEqual(w.join_segments([FakeSegment(""), FakeSegment("   ")]), "")
        self.assertEqual(w.join_segments([FakeSegment(None), FakeSegment(" ok.")]), "ok.")

    def test_accepts_bare_strings(self):
        self.assertEqual(w.join_segments([" one.", "two."]), "one. two.")


class InitialPromptTests(unittest.TestCase):
    def setUp(self):
        for key in ("WHISPER_GAZETTEER", "WHISPER_INITIAL_PROMPT"):
            self.addCleanup(os.environ.pop, key, None)
            os.environ.pop(key, None)

    def test_off_by_default(self):
        self.assertIsNone(w.initial_prompt())

    def test_env_gate_enables_the_gazetteer(self):
        os.environ["WHISPER_GAZETTEER"] = "1"
        self.assertEqual(w.initial_prompt(), w.DEFAULT_GAZETTEER)
        os.environ["WHISPER_GAZETTEER"] = "0"
        self.assertIsNone(w.initial_prompt())

    def test_custom_prompt_wins_and_implies_enabled(self):
        os.environ["WHISPER_INITIAL_PROMPT"] = "  Bakhmut, Avdiivka.  "
        self.assertEqual(w.initial_prompt(), "Bakhmut, Avdiivka.")
        os.environ["WHISPER_INITIAL_PROMPT"] = "   "
        self.assertIsNone(w.initial_prompt())

    def test_gazetteer_names_the_hard_words_and_stays_short(self):
        for name in ("Dobropillia", "Sloviansk", "Kramatorsk", "Pokrovsk",
                     "Svyatohirsk", "Orikhiv", "Witkoff", "Ushakov", "Zelensky"):
            self.assertIn(name, w.DEFAULT_GAZETTEER)
        # 50-80 tokens; ~1.3 tokens/word is a safe upper bound for this text.
        words = len(w.DEFAULT_GAZETTEER.split())
        self.assertLessEqual(words * 1.3, 80, "gazetteer prompt is turning into a novel")


if __name__ == "__main__":
    unittest.main()
