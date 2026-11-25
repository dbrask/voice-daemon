"""Piper CPU TTS engine — CPU-only tests, no GPU, no torch.

Covers the `piper-cpu-tts` change: the shared engine contract, voice
discovery from a directory, sidecar-driven audio parameters, and the guarantee
that nothing here allocates GPU memory.

Most tests drive the engine against a stub ONNX session and hand-written
sidecars in a tmp dir, so they run with no model files at all. Two tests use
the real voices at ~/piper-voices and skip when they are absent; the
nvidia-smi delta test additionally needs VOICEBOT_GPU_TESTS=1.

Run:  .venv/bin/python -m unittest discover -s tests -v
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import engines  # noqa: E402
from engines import TTSEngine, VoiceParams  # noqa: E402

def _voice_dir_candidates():
    """Where the real voices might be.

    ``~`` is not always the login home: agent sandboxes and some service
    managers override ``HOME``. Fall back to the passwd entry so the
    acceptance tests run rather than silently skipping.
    """
    import pwd

    seen, out = set(), []
    for cand in (
        os.environ.get("PIPER_VOICE_DIR"),
        os.path.expanduser("~/piper-voices"),
        os.path.join(pwd.getpwuid(os.getuid()).pw_dir, "piper-voices"),
    ):
        if cand and cand not in seen:
            seen.add(cand)
            out.append(Path(cand))
    return out


def _find_real_voice_dir():
    for cand in _voice_dir_candidates():
        try:
            if cand.is_dir() and any(
                p.with_suffix(".onnx.json").exists() for p in cand.glob("*.onnx")
            ):
                return cand
        except OSError:  # unreadable candidate (e.g. another user's home)
            continue
    return None


REAL_VOICE_DIR = _find_real_voice_dir()


def _has_real_voice() -> bool:
    return REAL_VOICE_DIR is not None


# ---- stubs ---------------------------------------------------------------

# A tiny but complete phoneme inventory: the four Piper specials plus the
# symbols the fake phonemizer emits. Ids are arbitrary but distinct.
STUB_PHONEMES = {
    "_": [0], "^": [1], "$": [2], " ": [3],
    "h": [10], "ə": [11], "l": [12], "o": [13], "ʊ": [14], ".": [15],
}


def _sidecar(sample_rate=22050, espeak_voice="en-us", **inference):
    inf = {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8}
    inf.update(inference)
    return {
        "audio": {"sample_rate": sample_rate, "quality": "medium"},
        "espeak": {"voice": espeak_voice},
        "inference": inf,
        "num_speakers": 1,
        "phoneme_type": "espeak",
        "phoneme_id_map": dict(STUB_PHONEMES),
    }


def _write_voice(directory: Path, name: str, sidecar=None, with_sidecar=True):
    """Create NAME.onnx (+ optional NAME.onnx.json) in *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.onnx").write_bytes(b"not-a-real-onnx")
    if with_sidecar:
        (directory / f"{name}.onnx.json").write_text(
            json.dumps(sidecar if sidecar is not None else _sidecar())
        )
    return directory / f"{name}.onnx"


class StubSession:
    """Stands in for onnxruntime.InferenceSession.

    Records the ids and scales it was called with so the tests can assert on
    the encoding, and returns a fixed non-silent waveform.
    """

    providers = ["CPUExecutionProvider"]
    frames = 4410

    def __init__(self, path, providers=None, **kw):
        self.path = path
        self.requested_providers = providers
        self.calls: list[dict] = []

    def get_providers(self):
        return list(self.providers)

    def run(self, outputs, feed):
        self.calls.append(feed)
        wave = 0.3 * np.sin(
            np.linspace(0, 40 * np.pi, self.frames, dtype=np.float32)
        )
        return [wave.reshape(1, 1, 1, -1)]


class CudaSession(StubSession):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]


class StubBackend:
    """Stands in for phonemizer's EspeakBackend."""

    last_voice = None
    emit = "həloʊ."

    def __init__(self, voice, **kw):
        StubBackend.last_voice = voice
        self.voice = voice

    def phonemize(self, texts, **kw):
        return [self.emit for _ in texts]


def _engine(tmpdir, session_cls=StubSession, backend_cls=StubBackend, **kwargs):
    """Build a PiperEngine with the ONNX session and phonemizer stubbed out."""
    from engines import piper as p

    with mock.patch.object(p, "_make_session", lambda path: session_cls(path, providers=["CPUExecutionProvider"])), \
         mock.patch.object(p, "_make_phonemizer", lambda voice: backend_cls(voice)):
        eng = p.PiperEngine(voice_dir=str(tmpdir), **kwargs)
        # Keep the stubs in place for synthesize() too — voices load lazily.
        eng._patches = (session_cls, backend_cls)
        with mock.patch.object(p, "_make_session", lambda path: session_cls(path, providers=["CPUExecutionProvider"])), \
             mock.patch.object(p, "_make_phonemizer", lambda voice: backend_cls(voice)):
            eng.warmup()
        return eng


class _TmpVoiceCase(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


# ---- registry ------------------------------------------------------------

class PiperRegistryTests(unittest.TestCase):
    def test_registry_lists_piper(self):
        self.assertIn("piper", engines.tts_engine_names())

    def test_naming_piper_does_not_import_onnxruntime(self):
        # The fallback must not cost import time in configs that never use it,
        # exactly like the kokoro/chatterbox laziness. Checked in a fresh
        # interpreter: within this process another test may already have
        # imported onnxruntime, which would make the assertion vacuous.
        script = (
            "import sys; sys.path.insert(0, %r);"
            "import engines;"
            "assert 'piper' in engines.tts_engine_names(), engines.tts_engine_names();"
            "assert 'onnxruntime' not in sys.modules, 'listing engines imported onnxruntime';"
            "assert 'engines.piper' not in sys.modules, 'listing engines imported the module';"
            "print('lazy')" % str(Path(__file__).resolve().parents[1])
        )
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("lazy", proc.stdout)


# ---- discovery -----------------------------------------------------------

class PiperDiscoveryTests(_TmpVoiceCase):
    def test_a_voice_is_a_model_plus_its_sidecar(self):
        _write_voice(self.tmpdir, "en_US-amy-medium")
        _write_voice(self.tmpdir, "en_US-lessac-medium")
        eng = _engine(self.tmpdir)
        self.assertEqual(
            sorted(eng.voices()), ["en_US-amy-medium", "en_US-lessac-medium"]
        )

    def test_model_without_sidecar_is_ignored(self):
        _write_voice(self.tmpdir, "good")
        _write_voice(self.tmpdir, "orphan", with_sidecar=False)
        eng = _engine(self.tmpdir)
        self.assertEqual(eng.voices(), ["good"])

    def test_voice_dir_is_configurable_by_env(self):
        from engines import piper as p

        _write_voice(self.tmpdir, "envvoice")
        with mock.patch.dict(os.environ, {"PIPER_VOICE_DIR": str(self.tmpdir)}), \
             mock.patch.object(p, "_make_session", lambda path: StubSession(path)), \
             mock.patch.object(p, "_make_phonemizer", lambda voice: StubBackend(voice)):
            eng = p.PiperEngine()
        self.assertEqual(eng.voices(), ["envvoice"])

    def test_empty_voice_dir_raises_naming_the_path(self):
        from engines import piper as p

        missing = self.tmpdir / "nothing-here"
        with self.assertRaises(Exception) as ctx:
            p.PiperEngine(voice_dir=str(missing))
        self.assertIn(str(missing), str(ctx.exception))

    def test_per_request_voice_selection(self):
        _write_voice(self.tmpdir, "alpha")
        _write_voice(self.tmpdir, "beta")
        eng = _engine(self.tmpdir, default_voice="alpha")
        from engines import piper as p

        with mock.patch.object(p, "_make_session", lambda path: StubSession(path)), \
             mock.patch.object(p, "_make_phonemizer", lambda voice: StubBackend(voice)):
            eng.synthesize("hello.", VoiceParams(voice="beta"))
        loaded = eng._loaded_voice_names()
        self.assertIn("beta", loaded)
        self.assertTrue(eng._session_for("beta").calls)

    def test_unknown_voice_names_the_known_list(self):
        _write_voice(self.tmpdir, "alpha")
        eng = _engine(self.tmpdir)
        with self.assertRaises(ValueError) as ctx:
            eng.synthesize("hello.", VoiceParams(voice="nope"))
        msg = str(ctx.exception)
        self.assertIn("nope", msg)
        self.assertIn("alpha", msg)


# ---- contract ------------------------------------------------------------

class PiperContractTests(_TmpVoiceCase):
    def setUp(self):
        super().setUp()
        _write_voice(self.tmpdir, "en_US-amy-medium")
        self.eng = _engine(self.tmpdir)

    def test_engine_satisfies_the_tts_protocol(self):
        self.assertIsInstance(self.eng, TTSEngine)
        self.assertEqual(self.eng.name, "piper")
        self.assertIsInstance(self.eng.max_chars, int)
        self.assertGreater(self.eng.max_chars, 0)

    def test_text_in_valid_audio_out(self):
        from engines import piper as p

        with mock.patch.object(p, "_make_session", lambda path: StubSession(path)), \
             mock.patch.object(p, "_make_phonemizer", lambda voice: StubBackend(voice)):
            pcm, sr = self.eng.synthesize("hello.", VoiceParams())
        self.assertIsInstance(pcm, np.ndarray)
        self.assertEqual(pcm.dtype, np.int16)
        self.assertEqual(pcm.ndim, 1)
        self.assertGreater(len(pcm), 0)
        self.assertGreater(int(np.abs(pcm).max()), 0)  # not silence
        self.assertEqual(sr, self.eng.sample_rate)

    def test_empty_text_yields_empty_audio(self):
        for text in ("", "   ", "\n\t"):
            pcm, sr = self.eng.synthesize(text, VoiceParams())
            self.assertEqual(len(pcm), 0)
            self.assertEqual(pcm.dtype, np.int16)
            self.assertEqual(sr, self.eng.sample_rate)

    def test_unload_then_synthesize_raises(self):
        self.eng.unload()
        with self.assertRaises(RuntimeError):
            self.eng.synthesize("hello.", VoiceParams())

    def test_describe_reports_cpu(self):
        d = self.eng.describe()
        self.assertEqual(d["engine"], "piper")
        self.assertEqual(d["device"], "cpu")
        self.assertEqual(d["sample_rate"], self.eng.sample_rate)


# ---- sidecar-driven parameters -------------------------------------------

class PiperSidecarTests(_TmpVoiceCase):
    def test_sample_rate_follows_the_voice(self):
        _write_voice(self.tmpdir, "lowrate", sidecar=_sidecar(sample_rate=16000))
        eng = _engine(self.tmpdir)
        self.assertEqual(eng.sample_rate, 16000)
        from engines import piper as p

        with mock.patch.object(p, "_make_session", lambda path: StubSession(path)), \
             mock.patch.object(p, "_make_phonemizer", lambda voice: StubBackend(voice)):
            _, sr = eng.synthesize("hello.", VoiceParams())
        self.assertEqual(sr, 16000)

    def test_espeak_voice_comes_from_the_sidecar(self):
        _write_voice(self.tmpdir, "swede", sidecar=_sidecar(espeak_voice="sv"))
        _engine(self.tmpdir)
        self.assertEqual(StubBackend.last_voice, "sv")

    def test_inference_scales_are_passed_through(self):
        _write_voice(
            self.tmpdir,
            "scaled",
            sidecar=_sidecar(noise_scale=0.5, length_scale=1.5, noise_w=0.9),
        )
        eng = _engine(self.tmpdir)
        session = eng._session_for("scaled")
        session.calls.clear()
        from engines import piper as p

        with mock.patch.object(p, "_make_session", lambda path: StubSession(path)), \
             mock.patch.object(p, "_make_phonemizer", lambda voice: StubBackend(voice)):
            eng.synthesize("hello.", VoiceParams())
        scales = session.calls[-1]["scales"]
        np.testing.assert_allclose(scales, [0.5, 1.5, 0.9], rtol=1e-6)

    def test_ids_are_bracketed_by_bos_and_eos(self):
        # Golden vector from Piper's own encoder, not from our implementation:
        # BOS, PAD, (phoneme_ids, PAD)*, EOS. StubBackend.emit = "həloʊ."
        # https://github.com/rhasspy/piper-phonemize/blob/master/src/phoneme_ids.cpp
        # https://github.com/OHF-Voice/piper1-gpl/blob/main/src/piper/phoneme_ids.py
        _write_voice(self.tmpdir, "amy")
        eng = _engine(self.tmpdir)
        session = eng._session_for("amy")
        session.calls.clear()
        from engines import piper as p

        with mock.patch.object(p, "_make_session", lambda path: StubSession(path)), \
             mock.patch.object(p, "_make_phonemizer", lambda voice: StubBackend(voice)):
            eng.synthesize("hello.", VoiceParams())
        ids = list(session.calls[-1]["input"][0])
        self.assertEqual(
            ids,
            [1, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 2],
        )
        self.assertEqual(int(session.calls[-1]["input_lengths"][0]), len(ids))

    def test_phoneme_outside_the_inventory_is_skipped(self):
        _write_voice(self.tmpdir, "amy")
        from engines import piper as p

        class OddBackend(StubBackend):
            emit = "hZZeloʊ."  # Z is not in STUB_PHONEMES

        # Voice is cached at construct, so OddBackend must be in place then —
        # patching only around synthesize would keep StubBackend.emit.
        with mock.patch.object(p, "_make_session", lambda path: StubSession(path)), \
             mock.patch.object(p, "_make_phonemizer", lambda voice: OddBackend(voice)):
            eng = p.PiperEngine(voice_dir=str(self.tmpdir))
            pcm, _ = eng.synthesize("hello.", VoiceParams())  # must not KeyError
        self.assertGreater(len(pcm), 0)
        # "hZZeloʊ." with Z dropped → "heloʊ."; 'e' is not in STUB_PHONEMES so
        # it is skipped too. Remaining ids must not shift.
        ids = list(eng._session_for("amy").calls[-1]["input"][0])
        self.assertEqual(
            ids,
            [1, 0, 10, 0, 12, 0, 13, 0, 14, 0, 15, 0, 2],
        )


# ---- CPU-only guarantee --------------------------------------------------

class PiperCpuOnlyTests(_TmpVoiceCase):
    def test_make_session_requests_only_the_cpu_provider(self):
        # Exercises the REAL _make_session, not a stub: whatever it hands to
        # onnxruntime must name CPUExecutionProvider and nothing else.
        from engines import piper as p

        seen = {}

        class Spy:
            def __init__(self, path, providers=None, **kw):
                seen["path"], seen["providers"] = path, providers

            def get_providers(self):
                return ["CPUExecutionProvider"]

        import onnxruntime

        with mock.patch.object(onnxruntime, "InferenceSession", Spy):
            p._make_session("/nonexistent/voice.onnx")
        self.assertEqual(seen["providers"], ["CPUExecutionProvider"])

    def test_a_cuda_session_makes_construction_raise(self):
        from engines import piper as p

        _write_voice(self.tmpdir, "amy")
        with mock.patch.object(p, "_make_session", lambda path: CudaSession(path)), \
             mock.patch.object(p, "_make_phonemizer", lambda voice: StubBackend(voice)):
            with self.assertRaises(RuntimeError) as ctx:
                p.PiperEngine(voice_dir=str(self.tmpdir))
        self.assertIn("CUDAExecutionProvider", str(ctx.exception))


# ---- real voice ----------------------------------------------------------

@unittest.skipUnless(_has_real_voice(), f"no piper voices at {REAL_VOICE_DIR}")
class PiperRealVoiceTests(unittest.TestCase):
    """The acceptance test: a real ONNX voice, text in, valid audio out."""

    def test_real_synthesis_produces_speech_shaped_audio(self):
        from engines.piper import PiperEngine

        eng = PiperEngine(voice_dir=str(REAL_VOICE_DIR))
        pcm, sr = eng.synthesize(
            "The quick brown fox jumps over the lazy dog.", VoiceParams()
        )
        self.assertEqual(pcm.dtype, np.int16)
        self.assertEqual(pcm.ndim, 1)
        self.assertEqual(sr, eng.sample_rate)
        self.assertIn(sr, (16000, 22050))  # piper's x_low / medium rates
        duration = len(pcm) / sr
        self.assertGreater(duration, 1.0)  # that sentence is not sub-second
        self.assertLess(duration, 15.0)
        self.assertGreater(int(np.abs(pcm).max()), 1000)  # audible, not silence
        eng.unload()

    def test_real_synthesis_does_not_import_torch(self):
        # Piper is the engine for when CUDA is unusable; importing torch would
        # both cost seconds and risk touching the driver.
        script = (
            "import sys; sys.path.insert(0, %r);"
            "from engines.piper import PiperEngine;"
            "from engines import VoiceParams;"
            "e = PiperEngine(voice_dir=%r);"
            "e.synthesize('Hello there.', VoiceParams());"
            "assert 'torch' not in sys.modules, 'torch was imported';"
            "print('ok')"
            % (str(Path(__file__).resolve().parents[1]), str(REAL_VOICE_DIR))
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ok", proc.stdout)


def _gpu_used_bytes():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip().splitlines()[0])


@unittest.skipUnless(
    os.environ.get("VOICEBOT_GPU_TESTS") == "1" and _has_real_voice(),
    "needs VOICEBOT_GPU_TESTS=1 and piper voices on disk",
)
class PiperZeroVramTests(unittest.TestCase):
    """The card's acceptance criterion: nvidia-smi delta = 0."""

    def test_construct_warmup_synthesize_allocates_no_vram(self):
        from engines.piper import PiperEngine

        before = _gpu_used_bytes()
        eng = PiperEngine(voice_dir=str(REAL_VOICE_DIR))
        eng.warmup()
        pcm, _ = eng.synthesize(
            "Piper runs entirely on the processor.", VoiceParams()
        )
        self.assertGreater(len(pcm), 0)
        after = _gpu_used_bytes()
        eng.unload()
        self.assertEqual(
            after, before,
            f"GPU memory changed {before} -> {after} MiB; piper must allocate none",
        )


if __name__ == "__main__":
    unittest.main()
