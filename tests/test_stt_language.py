"""Language selection for the whisper STT path — CPU only, no GPU, no model.

Covers the `stt-language-detect` change: auto-detect by default, an
explicit request winning over detection, per-utterance detection on
mixed-language audio, and the language actually used being reported back.

The engine's faster_whisper import lives inside ``__init__``, so these tests
drive the module-level helpers and an engine instance built without ever
touching CUDA (see ``_engine``).

Run:  .venv/bin/python -m unittest discover -s tests -v
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engines import whisper_ct2 as w  # noqa: E402

SR = w.SAMPLE_RATE


class FakeSegment:
    def __init__(self, text):
        self.text = text


class FakeInfo:
    def __init__(self, language):
        self.language = language
        self.language_probability = 1.0


class FakeModel:
    """Stands in for faster_whisper.WhisperModel.

    Language is a property of the audio, not of the model: each fake sample
    carries its language in its value (1.0 -> sv, 2.0 -> en), so detection and
    decode agree with each other the way the real model does.

    ``detect_prob`` fakes a low-confidence detection: it is consulted per
    detect call (indexed by call number) and defaults to 1.0.
    ``silent_languages`` names run languages whose decode yields no segments.
    """

    LANGS = {1.0: "sv", 2.0: "en"}
    WORDS = {"sv": "svenska", "en": "english"}

    def __init__(self, detect_probs=None, silent_languages=()):
        self.detect_calls = []
        self.transcribe_calls = []
        self.detect_probs = list(detect_probs or [])
        self.silent_languages = set(silent_languages)

    def _lang_of(self, pcm):
        values = {round(float(v), 1) for v in np.unique(pcm) if v}
        langs = {self.LANGS[v] for v in values if v in self.LANGS}
        return sorted(langs)[0] if len(langs) == 1 else "en"

    def detect_language(self, audio=None, **kwargs):
        index = len(self.detect_calls)
        self.detect_calls.append(len(audio))
        prob = self.detect_probs[index] if index < len(self.detect_probs) else 1.0
        return self._lang_of(audio), prob, []

    def transcribe(self, pcm, language=None, **kwargs):
        self.transcribe_calls.append({"samples": len(pcm), "language": language, **kwargs})
        used = language or self._lang_of(pcm)
        if used in self.silent_languages:
            return iter([]), FakeInfo(used)
        # Mimic whisper: a forced language rewrites the content in that language.
        text = f"{self.WORDS[used]}({len(pcm) // SR}s)"
        return iter([FakeSegment(text)]), FakeInfo(used)


def _tone(seconds, value):
    return np.full(int(seconds * SR), value, dtype=np.float32)


SILENCE = np.zeros(int(2.5 * SR), dtype=np.float32)


def _chunks_from(pcm):
    """VAD stand-in: contiguous runs of non-zero samples."""
    voiced = pcm != 0
    out, start = [], None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append({"start": start, "end": i})
            start = None
    if start is not None:
        out.append({"start": start, "end": len(voiced)})
    return out


def _engine(model=None, **env):
    """A WhisperCT2Engine with the model injected — __init__ never runs, so no CUDA."""
    eng = w.WhisperCT2Engine.__new__(w.WhisperCT2Engine)
    eng.model_name = "fake"
    eng.device = "cpu"
    eng.compute_type = "float32"
    eng.vram_mb = None
    eng.beam_size = 1
    eng.condition_on_previous_text = False
    eng.batch_above_s = 30.0
    eng.batch_size = 8
    eng.initial_prompt = None
    eng.multilingual = env.pop("multilingual", True)
    eng.lang_min_prob = env.pop("lang_min_prob", w.DEFAULT_LANG_MIN_PROB)
    eng._model = model or FakeModel()
    eng._batched = model or eng._model
    return eng


# ---- language_runs: pure grouping ---------------------------------------

class LanguageRunsTests(unittest.TestCase):
    def test_adjacent_chunks_of_one_language_collapse_to_one_run(self):
        chunks = [{"start": 0, "end": 10}, {"start": 20, "end": 30}]
        runs = w.language_runs(chunks, lambda s, e: ("sv", 1.0))
        self.assertEqual(runs, [(0, 30, "sv")])

    def test_language_switch_starts_a_new_run(self):
        chunks = [{"start": 0, "end": 10}, {"start": 20, "end": 30}]
        langs = iter([("sv", 1.0), ("en", 1.0)])
        runs = w.language_runs(chunks, lambda s, e: next(langs))
        self.assertEqual(runs, [(0, 10, "sv"), (20, 30, "en")])

    def test_run_spans_from_first_to_last_chunk_of_the_group(self):
        chunks = [{"start": 0, "end": 5}, {"start": 8, "end": 12}, {"start": 40, "end": 50}]
        langs = iter([("sv", 1.0), ("sv", 1.0), ("en", 1.0)])
        runs = w.language_runs(chunks, lambda s, e: next(langs))
        self.assertEqual(runs, [(0, 12, "sv"), (40, 50, "en")])

    def test_no_chunks_yields_no_runs(self):
        self.assertEqual(w.language_runs([], lambda s, e: ("sv", 1.0)), [])


# ---- language_runs: low-confidence detections ---------------------------

class LowConfidenceRunTests(unittest.TestCase):
    """A short interjection ("ok", "ja") detects wrongly at p~0.4 and would
    otherwise cut one utterance into three decode runs."""

    CHUNKS = [{"start": 0, "end": 10}, {"start": 20, "end": 30}, {"start": 40, "end": 50}]

    def _runs(self, detections, **kwargs):
        it = iter(detections)
        return w.language_runs(self.CHUNKS, lambda s, e: next(it), **kwargs)

    def test_low_confidence_chunk_inherits_the_previous_run_language(self):
        runs = self._runs([("sv", 0.99), ("en", 0.4), ("sv", 0.99)], min_prob=0.6)
        self.assertEqual(runs, [(0, 50, "sv")], "one run, not three")

    def test_confident_switch_still_cuts_a_run(self):
        runs = self._runs([("sv", 0.99), ("en", 0.9), ("en", 0.9)], min_prob=0.6)
        self.assertEqual(runs, [(0, 10, "sv"), (20, 50, "en")])

    def test_first_chunk_has_nothing_to_inherit_so_its_detection_stands(self):
        runs = self._runs([("sv", 0.2), ("sv", 0.99), ("sv", 0.99)], min_prob=0.6)
        self.assertEqual(runs, [(0, 50, "sv")])

    def test_zero_floor_trusts_every_detection(self):
        runs = self._runs([("sv", 0.99), ("en", 0.4), ("sv", 0.99)], min_prob=0.0)
        self.assertEqual(runs, [(0, 10, "sv"), (20, 30, "en"), (40, 50, "sv")])

    def test_min_prob_is_env_gated_with_a_default(self):
        with mock.patch.dict(os.environ, {"WHISPER_LANG_MIN_PROB": "0.9"}):
            self.assertEqual(w.lang_min_prob(), 0.9)
        with mock.patch.dict(os.environ, {"WHISPER_LANG_MIN_PROB": "nonsense"}):
            self.assertEqual(w.lang_min_prob(), w.DEFAULT_LANG_MIN_PROB)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WHISPER_LANG_MIN_PROB", None)
            self.assertEqual(w.lang_min_prob(), w.DEFAULT_LANG_MIN_PROB)
            self.assertEqual(w.DEFAULT_LANG_MIN_PROB, 0.6)


# ---- engine.transcribe --------------------------------------------------

class TranscribeLanguageTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.eng = _engine(self.model)
        patcher = mock.patch.object(w, "speech_chunks", side_effect=lambda pcm: _chunks_from(pcm))
        self.vad = patcher.start()
        self.addCleanup(patcher.stop)

    def test_returns_text_and_the_language_used(self):
        text, lang = self.eng.transcribe(_tone(2, 1.0))
        self.assertEqual(lang, "sv")
        self.assertIn("svenska", text)

    def test_english_clip_still_decodes_as_english(self):
        text, lang = self.eng.transcribe(_tone(2, 2.0))
        self.assertEqual(lang, "en")
        self.assertIn("english", text)

    def test_explicit_language_wins_over_detection(self):
        text, lang = self.eng.transcribe(_tone(2, 1.0), language="en")
        self.assertEqual(lang, "en")
        self.assertIn("english", text)
        self.assertEqual(self.model.detect_calls, [], "no detection when a language is requested")

    def test_explicit_language_skips_per_utterance_detection(self):
        mixed = np.concatenate([_tone(2, 1.0), SILENCE, _tone(2, 2.0)])
        text, lang = self.eng.transcribe(mixed, language="sv")
        self.assertEqual(lang, "sv")
        self.assertEqual(self.model.detect_calls, [])
        self.assertEqual(len(self.model.transcribe_calls), 1, "one whole-clip pass")

    def test_mixed_clip_decodes_each_utterance_in_its_own_language(self):
        mixed = np.concatenate([_tone(2, 1.0), SILENCE, _tone(2, 2.0)])
        text, lang = self.eng.transcribe(mixed)
        self.assertIn("svenska", text)
        self.assertIn("english", text)
        self.assertLess(text.index("svenska"), text.index("english"), "time order preserved")
        self.assertEqual(lang, "sv+en")
        self.assertEqual([c["language"] for c in self.model.transcribe_calls], ["sv", "en"])

    def test_run_slices_are_padded_so_onsets_are_not_clipped(self):
        # The VAD marks onset slightly late; decoding exactly on the boundary
        # ate the first consonant of the second language live.
        mixed = np.concatenate([_tone(2, 1.0), SILENCE, _tone(2, 2.0)])
        self.eng.transcribe(mixed)
        chunks = _chunks_from(mixed)
        for call, chunk in zip(self.model.transcribe_calls, chunks):
            self.assertGreater(call["samples"], chunk["end"] - chunk["start"])

    def test_single_language_clip_is_decoded_in_one_pass(self):
        two_utterances = np.concatenate([_tone(2, 1.0), SILENCE, _tone(2, 1.0)])
        text, lang = self.eng.transcribe(two_utterances)
        self.assertEqual(lang, "sv")
        self.assertEqual(len(self.model.transcribe_calls), 1)
        # The detected language must be *forced* on the whole-clip decode, not
        # left to a second detection that could disagree with the first.
        self.assertEqual(self.model.transcribe_calls[0]["language"], "sv")

    def test_kill_switch_restores_whole_clip_detection(self):
        eng = _engine(self.model, multilingual=False)
        mixed = np.concatenate([_tone(2, 1.0), SILENCE, _tone(2, 2.0)])
        text, lang = eng.transcribe(mixed)
        self.assertEqual(self.model.detect_calls, [], "no per-chunk detection")
        self.assertEqual(len(self.model.transcribe_calls), 1)
        self.assertEqual(lang, "en")

    def test_long_clip_takes_the_batched_path_without_per_chunk_detection(self):
        eng = _engine(self.model)
        eng.batch_above_s = 1.0
        text, lang = eng.transcribe(np.concatenate([_tone(2, 1.0), SILENCE, _tone(2, 2.0)]))
        self.assertEqual(self.model.detect_calls, [])
        self.assertEqual(len(self.model.transcribe_calls), 1)
        self.assertIn("batch_size", self.model.transcribe_calls[0])
        # The batched pipeline detects for itself; the language it reports must
        # reach the caller rather than being dropped on the floor.
        self.assertEqual(lang, "en")
        self.assertIn("english", text)

    def test_long_clip_with_an_explicit_language_reports_that_language(self):
        eng = _engine(self.model)
        eng.batch_above_s = 1.0
        _text, lang = eng.transcribe(_tone(4, 2.0), language="sv")
        self.assertEqual(lang, "sv")
        self.assertEqual(self.model.transcribe_calls[0]["language"], "sv")

    def test_a_run_that_decodes_to_nothing_is_not_named_in_the_language(self):
        # The VAD occasionally marks silence as speech; that run detects a
        # language but decodes to no segments. Reporting "sv+en" for text that
        # is only English is a lie about what was transcribed.
        model = FakeModel(silent_languages={"sv"})
        eng = _engine(model)
        with mock.patch.object(w, "speech_chunks", side_effect=_chunks_from):
            text, lang = eng.transcribe(
                np.concatenate([_tone(2, 1.0), SILENCE, _tone(2, 2.0)]))
        self.assertEqual([c["language"] for c in model.transcribe_calls], ["sv", "en"])
        self.assertEqual(lang, "en")
        self.assertEqual(text, "english(2s)")

    def test_all_runs_silent_reports_no_language_at_all(self):
        model = FakeModel(silent_languages={"sv", "en"})
        eng = _engine(model)
        with mock.patch.object(w, "speech_chunks", side_effect=_chunks_from):
            text, lang = eng.transcribe(
                np.concatenate([_tone(2, 1.0), SILENCE, _tone(2, 2.0)]))
        self.assertEqual(text, "")
        self.assertIsNone(lang)

    def test_low_confidence_utterance_does_not_fragment_the_decode(self):
        # Second utterance is English audio detected at p=0.4 — below the floor,
        # so it stays in the Swedish run and the clip decodes in one pass.
        model = FakeModel(detect_probs=[0.99, 0.4])
        eng = _engine(model, lang_min_prob=0.6)
        with mock.patch.object(w, "speech_chunks", side_effect=_chunks_from):
            _text, lang = eng.transcribe(
                np.concatenate([_tone(2, 1.0), SILENCE, _tone(2, 2.0)]))
        self.assertEqual(len(model.detect_calls), 2, "both chunks are still detected")
        self.assertEqual(lang, "sv")
        self.assertEqual([c["language"] for c in model.transcribe_calls], ["sv"])

    def test_describe_reports_the_confidence_floor(self):
        self.assertEqual(self.eng.describe()["lang_min_prob"], w.DEFAULT_LANG_MIN_PROB)

    def test_multilingual_is_env_gated(self):
        with mock.patch.dict(os.environ, {"WHISPER_MULTILINGUAL": "0"}):
            self.assertFalse(w.multilingual_enabled())
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WHISPER_MULTILINGUAL", None)
            self.assertTrue(w.multilingual_enabled())

    def test_describe_reports_multilingual(self):
        self.assertIs(self.eng.describe()["multilingual"], True)


if __name__ == "__main__":
    unittest.main()
