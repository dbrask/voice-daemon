"""Chapterize tests: paragraph splitting for headingless extracted text.

No GPU, no daemon, no HTTP — `scripts/chapterize.py` is pure text handling
over `jobs.split_chapters`.

Run:  .venv/bin/python -m unittest discover -s tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from chapterize import chapterize, split_long  # noqa: E402


class SplitLong(unittest.TestCase):
    def test_short_text_is_one_part(self):
        self.assertEqual(split_long("T", "short body", 100), [("T", "short body")])

    def test_splits_on_paragraph_boundaries(self):
        text = "\n\n".join(["a" * 90] * 5)
        parts = split_long("Ch", text, 200)
        self.assertGreater(len(parts), 1)
        for _title, body in parts:
            # Never cut inside a paragraph.
            for para in body.split("\n\n"):
                self.assertEqual(para, "a" * 90)

    def test_parts_are_numbered_after_the_first(self):
        parts = split_long("Ch", "\n\n".join(["x" * 90] * 4), 200)
        self.assertEqual(parts[0][0], "Ch")
        self.assertEqual(parts[1][0], "Ch (2)")

    def test_untitled_chapter_gets_a_fallback_base(self):
        parts = split_long("", "\n\n".join(["x" * 90] * 4), 200)
        self.assertEqual(parts[0][0], "part")

    def test_one_huge_paragraph_is_not_cut(self):
        # The daemon chunks for the engine anyway; only the file boundary is
        # at stake, and a mid-sentence file boundary is worse than a long file.
        body = "y" * 5000
        self.assertEqual(split_long("T", body, 1000), [("T", body)])

    def test_respects_max_chars_when_it_can(self):
        parts = split_long("T", "\n\n".join(["z" * 300] * 10), 1000)
        self.assertTrue(all(len(b) <= 1000 for _t, b in parts))


class Chapterize(unittest.TestCase):
    def test_headings_still_chapter(self):
        out = chapterize("# One\n\nbody one\n\n# Two\n\nbody two", max_chars=10000)
        self.assertEqual([t for t, _ in out], ["One", "Two"])

    def test_headingless_text_is_split_by_length(self):
        text = "\n\n".join(["p" * 400] * 10)
        out = chapterize(text, max_chars=1000)
        self.assertGreater(len(out), 3)

    def test_headingless_parts_are_numbered(self):
        out = chapterize("\n\n".join(["p" * 400] * 10), max_chars=1000)
        self.assertEqual([t for t, _ in out[:3]], ["Part 1", "Part 2", "Part 3"])

    def test_real_headings_keep_their_titles(self):
        text = "# Alpha\n\n" + "a" * 100 + "\n\n# Beta\n\n" + "b" * 100
        out = chapterize(text, max_chars=1000)
        self.assertEqual([t for t, _ in out], ["Alpha", "Beta"])

    def test_long_chapter_under_a_heading_is_subdivided(self):
        text = "# Big\n\n" + "\n\n".join(["q" * 400] * 6)
        out = chapterize(text, max_chars=1000)
        self.assertGreater(len(out), 1)
        self.assertEqual(out[0][0], "Big")
        self.assertEqual(out[1][0], "Big (2)")

    def test_split_none_keeps_one_logical_chapter(self):
        out = chapterize("# One\n\nbody", split="none", max_chars=10000)
        self.assertEqual(len(out), 1)
        self.assertIn("# One", out[0][1])

    def test_empty_text_yields_nothing(self):
        self.assertEqual(chapterize("   \n\n  "), [])


if __name__ == "__main__":
    unittest.main()
