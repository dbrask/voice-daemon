"""Real-audio STT language checks. Needs the GPU model and the Common Voice corpus.

Skipped unless ``VOICEBOT_GPU_TESTS=1`` and the corpus is present, because it
loads large-v3-turbo onto CUDA and decodes real clips. This is the test that
actually proves the fix: with ``language='en'`` the Swedish reference sentence
comes back as English word-salad; on auto it comes back as Swedish.

Run:  VOICEBOT_GPU_TESTS=1 .venv/bin/python -m unittest tests.test_stt_language_gpu -v
"""

import csv
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SR = 16000
CORPUS = ROOT / "testdata/cv-corpus-26.0-2026-06-12/sv-SE"
EN_CLIP = ROOT / "bench/clips/01-short.ogg"
EN_TEXT = ROOT / "bench/clips/01-short.txt"

# Two validated Common Voice sv-SE clips. The first is the regression witness:
# under the en pin it transcribes as "She had laid the arm on the tree...".
SV_CLIPS = {
    "common_voice_sv-SE_25158638.mp3": "Hon hade lagt armarna på trälocket och dolt huvudet i dem.",
    "common_voice_sv-SE_25158633.mp3": "Kaptenen berättade och jag gjorde anteckningar.",
}

_ENABLED = os.environ.get("VOICEBOT_GPU_TESTS") == "1"
_HAVE_CORPUS = (CORPUS / "clips").is_dir()


def _decode(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
         "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
        check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def _reference(name: str) -> str:
    """The validated sentence for a clip, read from the corpus TSVs."""
    for tsv in ("test.tsv", "dev.tsv", "train.tsv", "other.tsv"):
        path = CORPUS / tsv
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                if row["path"] == name:
                    return row["sentence"]
    raise KeyError(name)


def _words(text: str) -> set:
    return {"".join(ch for ch in tok.lower() if ch.isalnum()) for tok in text.split()} - {""}


def _overlap(got: str, want: str) -> float:
    want_words = _words(want)
    return len(_words(got) & want_words) / max(len(want_words), 1)


@unittest.skipUnless(_ENABLED, "set VOICEBOT_GPU_TESTS=1 to run GPU STT tests")
@unittest.skipUnless(_HAVE_CORPUS, f"Common Voice sv-SE corpus not at {CORPUS}")
class RealAudioLanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from engines.whisper_ct2 import WhisperCT2Engine
        cls.engine = WhisperCT2Engine()
        cls.sv = {name: _decode(CORPUS / "clips" / name) for name in SV_CLIPS}
        cls.en = _decode(EN_CLIP)

    @classmethod
    def tearDownClass(cls):
        cls.engine.unload()

    def test_corpus_references_match_the_pinned_sentences(self):
        # Guards the fixtures themselves: a corpus re-download that moved these
        # clips would otherwise silently weaken every assertion below.
        for name, sentence in SV_CLIPS.items():
            self.assertEqual(_reference(name), sentence, name)

    def test_swedish_audio_transcribes_as_swedish(self):
        for name, sentence in SV_CLIPS.items():
            with self.subTest(clip=name):
                text, lang = self.engine.transcribe(self.sv[name])
                self.assertEqual(lang, "sv")
                self.assertGreaterEqual(
                    _overlap(text, sentence), 0.6,
                    f"{text!r} does not match reference {sentence!r}")

    def test_forcing_a_language_actually_forces_it(self):
        # Not asserting *what* comes out: forcing `en` on Swedish audio is
        # whisper's own unstable behaviour (float16 greedy decode returns
        # English word-salad most runs and leaked Swedish on some). What the
        # engine owes the caller is that the request is honoured and reported.
        name = "common_voice_sv-SE_25158638.mp3"
        text, lang = self.engine.transcribe(self.sv[name], language="en")
        self.assertEqual(lang, "en")
        self.assertTrue(text.strip())

    def test_english_audio_is_unchanged_by_auto_detect(self):
        # Same reference, same language, whether pinned or detected. Exact
        # string equality is not asserted: the GPU float16 decode is not
        # bit-reproducible run to run.
        pinned, pinned_lang = self.engine.transcribe(self.en, language="en")
        auto, lang = self.engine.transcribe(self.en)
        self.assertEqual((pinned_lang, lang), ("en", "en"))
        reference = EN_TEXT.read_text()
        self.assertGreaterEqual(_overlap(pinned, reference), 0.8, pinned)
        self.assertGreaterEqual(_overlap(auto, reference), 0.8, auto)

    def test_mixed_swedish_then_english_keeps_both_languages(self):
        name = "common_voice_sv-SE_25158638.mp3"
        gap = np.zeros(int(2.5 * SR), dtype=np.float32)
        mixed = np.concatenate([self.sv[name], gap, self.en])
        text, lang = self.engine.transcribe(mixed)
        self.assertEqual(lang, "sv+en")
        self.assertGreaterEqual(_overlap(text, SV_CLIPS[name]), 0.6, text)
        self.assertGreaterEqual(_overlap(text, EN_TEXT.read_text()), 0.6, text)

    def test_explicit_swedish_on_mixed_audio_is_honoured(self):
        name = "common_voice_sv-SE_25158638.mp3"
        gap = np.zeros(int(2.5 * SR), dtype=np.float32)
        mixed = np.concatenate([self.sv[name], gap, self.en])
        _text, lang = self.engine.transcribe(mixed, language="sv")
        self.assertEqual(lang, "sv")


if __name__ == "__main__":
    unittest.main()
