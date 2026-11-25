"""int8_float16 vs float16 on the real model. Needs the GPU; loads two models.

Skipped unless ``VOICEBOT_GPU_TESTS=1``. This is the test that earns the default
flip in the `stt-int8-quantization` change: everything else is plumbing,
and the quantization claim ("half the VRAM, no measurable quality loss") is only
true if it is true on this card with these clips.

The two engines are loaded and measured **sequentially**, never both resident,
for two reasons: the host has one 16 GB GPU, and a VRAM figure taken while the
other model is loaded is not a footprint, it is noise.

Run:  VOICEBOT_GPU_TESTS=1 .venv/bin/python -m unittest tests.test_stt_compute_type_gpu -v
"""

import csv
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SR = 16000
CLIPS = ROOT / "bench/clips"
CORPUS = ROOT / "testdata/cv-corpus-26.0-2026-06-12/sv-SE"

BASELINE = "float16"
CANDIDATE = "int8_float16"

# The footprint must actually shrink; the proposal claims roughly half, the spec
# demands at most 75% so a CTranslate2 version that quantizes less aggressively
# fails loudly instead of passing on a rounding error.
MAX_VRAM_RATIO = 0.75
# WER is a ratio in [0,1]; 0.05 absolute is one word in twenty. Anything worse
# than that on these short clips is a real regression, not sampling noise.
MAX_WER_REGRESSION = 0.05
MAX_LATENCY_RATIO = 1.15

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


def _words(text: str) -> list:
    return [t for t in ("".join(ch if ch.isalnum() else " " for ch in text.lower())).split() if t]


def _wer(reference: str, hypothesis: str) -> float:
    """Levenshtein word error rate. Same shape as scripts/bench.py's."""
    r, h = _words(reference), _words(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


def _bench_clips() -> dict:
    """``name -> (pcm, reference)`` for every bench clip that has a reference."""
    out = {}
    for path in sorted(CLIPS.iterdir()):
        if path.suffix not in (".ogg", ".wav"):
            continue
        ref = path.with_suffix(".txt")
        if not ref.exists():
            continue
        out[path.name] = (_decode(path), ref.read_text().strip())
    return out


def _measure(compute_type: str, audio: dict, runs: int = 3) -> dict:
    """Load the model at *compute_type*, transcribe everything, unload.

    Returns ``{"vram_mb", "clips": {name: {"text", "language", "ms"}}}``. The
    engine is unloaded before returning so the caller can load the other one.
    """
    from engines.whisper_ct2 import WhisperCT2Engine
    import torch

    engine = WhisperCT2Engine(compute_type=compute_type)
    engine.warmup()
    results = {}
    try:
        for name, pcm in audio.items():
            samples = []
            text = language = None
            for _ in range(runs):
                t0 = time.perf_counter()
                text, language = engine.transcribe(pcm)
                samples.append((time.perf_counter() - t0) * 1000)
            results[name] = {"text": text, "language": language,
                            "ms": sorted(samples)[len(samples) // 2]}
        vram = engine.vram_mb
    finally:
        engine.unload()
        torch.cuda.empty_cache()
    return {"vram_mb": vram, "clips": results}


@unittest.skipUnless(_ENABLED, "set VOICEBOT_GPU_TESTS=1 to run GPU STT tests")
class QuantizedComputeTypeTests(unittest.TestCase):
    """One shared measurement pass; each test asserts one property of it.

    Loading the model four times would take minutes per assertion, so both
    compute types are measured once in ``setUpClass`` and the tests read the
    result. They still fail independently.
    """

    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.audio = {}
        cls.references = {}
        for name, (pcm, ref) in _bench_clips().items():
            cls.audio[name] = pcm
            cls.references[name] = ref
        if _HAVE_CORPUS:
            for name, sentence in SV_CLIPS.items():
                cls.audio[name] = _decode(CORPUS / "clips" / name)
                cls.references[name] = sentence
        if not cls.audio:
            raise unittest.SkipTest(f"no reference clips under {CLIPS}")
        # Sequential on purpose: one model resident at a time.
        cls.baseline = _measure(BASELINE, cls.audio)
        cls.candidate = _measure(CANDIDATE, cls.audio)

    def test_quantized_model_footprint_is_materially_smaller(self):
        base, cand = self.baseline["vram_mb"], self.candidate["vram_mb"]
        self.assertIsNotNone(base, "no VRAM figure for the float16 load")
        self.assertIsNotNone(cand, "no VRAM figure for the int8_float16 load")
        self.assertLessEqual(
            cand, base * MAX_VRAM_RATIO,
            f"{CANDIDATE} used {cand} MB vs {BASELINE} {base} MB "
            f"(ratio {cand / base:.2f}, need <= {MAX_VRAM_RATIO})")

    def test_word_error_rate_does_not_regress_on_any_clip(self):
        for name, reference in self.references.items():
            if not reference:
                continue  # silence: covered by its own test
            with self.subTest(clip=name):
                base = _wer(reference, self.baseline["clips"][name]["text"])
                cand = _wer(reference, self.candidate["clips"][name]["text"])
                self.assertLessEqual(
                    cand, base + MAX_WER_REGRESSION,
                    f"{name}: WER {base:.3f} -> {cand:.3f}\n"
                    f"  {BASELINE}: {self.baseline['clips'][name]['text']!r}\n"
                    f"  {CANDIDATE}: {self.candidate['clips'][name]['text']!r}")

    @unittest.skipUnless(_HAVE_CORPUS, f"Common Voice sv-SE corpus not at {CORPUS}")
    def test_swedish_clips_still_detect_and_transcribe_as_swedish(self):
        for name in SV_CLIPS:
            with self.subTest(clip=name):
                got = self.candidate["clips"][name]
                self.assertEqual(got["language"], "sv", got["text"])

    def test_detected_language_is_unchanged_by_quantization(self):
        for name in self.audio:
            with self.subTest(clip=name):
                self.assertEqual(self.candidate["clips"][name]["language"],
                                 self.baseline["clips"][name]["language"])

    def test_silence_stays_silent(self):
        silent = [n for n, ref in self.references.items() if not ref]
        if not silent:
            self.skipTest("no silence clip in bench/clips")
        for name in silent:
            with self.subTest(clip=name):
                self.assertEqual(self.candidate["clips"][name]["text"].strip(), "")

    def test_decode_latency_does_not_regress(self):
        for name in self.audio:
            with self.subTest(clip=name):
                base = self.baseline["clips"][name]["ms"]
                cand = self.candidate["clips"][name]["ms"]
                self.assertLessEqual(
                    cand, base * MAX_LATENCY_RATIO,
                    f"{name}: {base:.0f} ms -> {cand:.0f} ms "
                    f"(ratio {cand / base:.2f}, need <= {MAX_LATENCY_RATIO})")


if __name__ == "__main__":
    unittest.main()
