"""Daemon tests that need no GPU: engine seam, body reader, HTTP surface.

stdlib unittest + aiohttp.test_utils only — no pytest, so it runs in the
project venv with nothing extra installed.

Run:  .venv/bin/python -m unittest discover -s tests -v
GPU / live-daemon checks are the opt-in *_gpu tests and the README smoke test.
"""

import asyncio
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from aiohttp.test_utils import AioHTTPTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import engines  # noqa: E402
import voice_daemon as vd  # noqa: E402
from engines import STTEngine, TTSEngine, VoiceParams  # noqa: E402


# ---- fakes ---------------------------------------------------------------

class FakeSTT:
    name = "fake_stt"
    sample_rate = 16000

    def __init__(self):
        self.unloaded = False
        self.warmed = 0

    def transcribe(self, pcm, language=None):
        # Echo back what the daemon decoded so tests can check the ffmpeg path.
        # Contract: (text, language actually used) — "sv" when asked to detect.
        used = language or "sv"
        return f"fake:{pcm.dtype}:{len(pcm) // self.sample_rate}s:{language}", used

    def warmup(self):
        self.warmed += 1

    def unload(self):
        self.unloaded = True

    def describe(self):
        # Mirrors the real engine's shape closely enough to prove /health does
        # not drop engine-reported fields it does not know about.
        return {"engine": self.name, "compute_type": "int8_float16", "vram_mb": 1600}


class FakeTTS:
    name = "fake_tts"
    sample_rate = 16000
    max_chars = 40
    voice = "fake_default"

    def __init__(self):
        self.unloaded = False
        self.calls: list[str] = []
        self.params: list[VoiceParams] = []

    def voices(self):
        return [self.voice, "fake_other"]

    def synthesize(self, text, params: VoiceParams):
        voice = params.voice or self.voice
        if voice not in self.voices():
            raise ValueError(f"unknown voice {voice!r}")
        self.calls.append(text)
        self.params.append(params)
        return np.zeros(self.sample_rate // 10, dtype=np.int16), self.sample_rate

    def warmup(self):
        pass

    def unload(self):
        self.unloaded = True


def _wav_bytes(seconds: float, rate: int = 8000) -> bytes:
    """A real WAV container so the daemon's ffmpeg decode has something to chew."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(np.zeros(int(seconds * rate), dtype=np.int16).tobytes())
    return buf.getvalue()


def _fake_engines(test_case):
    """Point the daemon at the fakes for the duration of *test_case*."""
    saved = dict(vd._engines)
    saved_stt, saved_tts = vd.load_stt_engine, vd.load_tts_engine
    vd._engines.update({"stt": None, "tts": None})
    vd.load_stt_engine = lambda name: FakeSTT()
    vd.load_tts_engine = lambda name: FakeTTS()

    def restore():
        vd._engines.update(saved)
        vd.load_stt_engine, vd.load_tts_engine = saved_stt, saved_tts

    test_case.addCleanup(restore)


# ---- engine seam ---------------------------------------------------------

class EngineSeamTests(unittest.TestCase):
    def test_fakes_satisfy_the_protocols(self):
        self.assertIsInstance(FakeSTT(), STTEngine)
        self.assertIsInstance(FakeTTS(), TTSEngine)

    def test_registry_rejects_unknown_engines(self):
        with self.assertRaisesRegex(ValueError, "unknown STT engine"):
            engines.load_stt_engine("nope")
        with self.assertRaisesRegex(ValueError, "unknown TTS engine"):
            engines.load_tts_engine("nope")

    def test_registry_lists_the_shipped_engines(self):
        self.assertIn("whisper_ct2", engines.stt_engine_names())
        self.assertIn("chatterbox", engines.tts_engine_names())
        self.assertIn("kokoro", engines.tts_engine_names())

    def test_tts_factories_are_lazy(self):
        # Naming an engine must not import torch/kokoro/chatterbox; the daemon
        # runs STT-only configs and the import costs seconds of startup.
        self.assertNotIn("kokoro", sys.modules)
        self.assertIn("kokoro", engines.tts_engine_names())
        self.assertNotIn("kokoro", sys.modules)

    def test_ensure_and_release_engine(self):
        _fake_engines(self)
        first = vd.ensure_engine("stt")
        self.assertIs(vd.ensure_engine("stt"), first)  # loaded once
        self.assertEqual(first.warmed, 1)  # warm-up ran exactly once, on load
        self.assertTrue(vd.release_engine("stt"))
        self.assertTrue(first.unloaded)
        self.assertFalse(vd.release_engine("stt"))  # idempotent
        with self.assertRaises(ValueError):
            vd.ensure_engine("bogus")


class ChunkerTests(unittest.TestCase):
    """Pure function; importing it must not pull torch."""

    def test_chunker_respects_the_limit(self):
        split = vd.split_sentences

        self.assertEqual(split("", 50), [])
        self.assertEqual(split("Short one.", 50), ["Short one."])
        text = "One sentence here. " * 20
        chunks = split(text, 60)
        self.assertTrue(chunks)
        self.assertTrue(all(len(c) <= 60 for c in chunks))
        self.assertEqual("".join(chunks).replace(" ", ""), text.replace(" ", ""))
        monster = "a" * 130  # one word longer than the limit
        self.assertTrue(all(len(c) <= 50 for c in split(monster, 50)))

    def test_chunker_caps_only_the_first_chunk(self):
        text = "Short lead. " + "A longer sentence that goes on a bit. " * 6
        chunks = vd.split_sentences(text, 200, first_limit=40)
        self.assertEqual(chunks[0], "Short lead.")
        self.assertTrue(all(len(c) <= 200 for c in chunks))
        self.assertTrue(any(len(c) > 40 for c in chunks[1:]))  # later chunks use the full limit
        self.assertEqual(" ".join(chunks), text.strip())
        # A first sentence longer than first_limit is cut on a space, not mid-word.
        chunks = vd.split_sentences("word " * 30, 200, first_limit=22)
        self.assertLessEqual(len(chunks[0]), 22)
        self.assertEqual(chunks[0], "word word word word")


# ---- HTTP surface --------------------------------------------------------

class DaemonHTTPTests(AioHTTPTestCase):
    async def get_application(self):
        _fake_engines(self)
        return vd.build_app()

    async def test_health_reports_engines(self):
        body = await (await self.client.get("/health")).json()
        self.assertTrue(body["ok"])
        self.assertFalse(body["engines"]["stt"]["loaded"])
        self.assertEqual(set(body["vram_mb"]), {"free", "total", "used"})

    async def test_health_surfaces_the_stt_engines_vram_and_compute_type(self):
        # Regression guard on the passthrough, not red-green evidence: the
        # daemon already merges whatever describe() returns. It exists so a
        # future refactor to an explicit field list cannot silently drop the
        # compute-type reporting the spec requires.
        await self.client.post("/load?engine=stt")
        stt = (await (await self.client.get("/health")).json())["engines"]["stt"]
        self.assertTrue(stt["loaded"])
        self.assertEqual(stt["compute_type"], "int8_float16")
        self.assertEqual(stt["vram_mb"], 1600)

    async def test_stt_raw_body_passes_language_through(self):
        resp = await self.client.post("/stt?language=sv", data=_wav_bytes(2.0))
        self.assertEqual(resp.status, 200)
        body = await resp.json()
        # ffmpeg resampled 8 kHz -> the engine's 16 kHz float32; language passed.
        self.assertEqual(body["text"], "fake:float32:2s:sv")
        self.assertAlmostEqual(body["audio_s"], 2.0, places=1)
        self.assertTrue({"decode", "model", "encode", "total", "request"} <= set(body["timing_ms"]))

    async def test_stt_reports_the_language_actually_used(self):
        body = await (await self.client.post("/stt", data=_wav_bytes(1.0))).json()
        # The fake engine detects "sv" when asked to auto-detect; the daemon must
        # report what came back, not the "auto" that was requested.
        self.assertEqual(body["language"], "sv")

    async def test_stt_auto_sentinels_reach_the_engine_as_no_language(self):
        for value in ("", "auto", "none"):
            with self.subTest(language=value):
                body = await (await self.client.post(
                    f"/stt?language={value}", data=_wav_bytes(1.0))).json()
                self.assertEqual(body["text"], "fake:float32:1s:None")
                self.assertEqual(body["language"], "sv")

    async def test_stt_without_language_parameter_auto_detects(self):
        body = await (await self.client.post("/stt", data=_wav_bytes(1.0))).json()
        self.assertEqual(body["text"], "fake:float32:1s:None")

    async def test_stt_accepts_multipart(self):
        resp = await self.client.post("/stt", data={"file": _wav_bytes(0.5)})
        self.assertEqual(resp.status, 200)
        self.assertTrue((await resp.json())["ok"])

    async def test_stt_rejects_undecodable_audio(self):
        resp = await self.client.post("/stt", data=b"this is not audio at all, sorry")
        self.assertEqual(resp.status, 400)
        self.assertIn("decode", (await resp.json())["error"])

    async def test_stt_rejects_empty_body(self):
        self.assertEqual((await self.client.post("/stt", data=b"")).status, 400)

    async def test_stt_rejects_oversize_body(self):
        saved = vd.MAX_UPLOAD_BYTES
        vd.MAX_UPLOAD_BYTES = 16
        try:
            resp = await self.client.post("/stt", data=b"x" * 4096)
        finally:
            vd.MAX_UPLOAD_BYTES = saved
        self.assertEqual(resp.status, 400)

    async def test_tts_returns_ogg_by_default(self):
        resp = await self.client.post("/tts", data="hello there".encode())
        self.assertEqual(resp.status, 200)
        self.assertTrue(resp.headers["Content-Type"].startswith("audio/ogg"))
        self.assertGreater(len(await resp.read()), 0)
        timing = json.loads(resp.headers["X-Timing-Ms"])
        self.assertTrue({"model", "encode", "total", "first_chunk", "request"} <= set(timing))
        self.assertAlmostEqual(float(resp.headers["X-Audio-Seconds"]), 0.1, places=2)

    async def test_tts_chunks_long_text_at_the_engine_limit(self):
        text = "First sentence here. Second sentence follows. Third one closes it out."
        resp = await self.client.post("/tts", data=text.encode())
        self.assertEqual(resp.status, 200)
        calls = vd._engines["tts"].calls
        self.assertGreater(len(calls), 1)
        self.assertTrue(all(len(c) <= 40 for c in calls))
        self.assertEqual(" ".join(calls), text)
        self.assertAlmostEqual(float(resp.headers["X-Audio-Seconds"]), 0.1 * len(calls), places=2)

    async def test_tts_rejects_text_over_the_request_cap(self):
        saved = vd.MAX_TTS_TEXT_CHARS
        vd.MAX_TTS_TEXT_CHARS = 20
        try:
            resp = await self.client.post("/tts", data=b"x" * 21)
        finally:
            vd.MAX_TTS_TEXT_CHARS = saved
        self.assertEqual(resp.status, 400)

    async def test_tts_honours_format_wav_and_json_body(self):
        resp = await self.client.post("/tts?format=wav", json={"text": "hello there"})
        self.assertEqual(resp.status, 200)
        self.assertTrue(resp.headers["Content-Type"].startswith("audio/wav"))

    async def test_tts_rejects_bad_input(self):
        self.assertEqual((await self.client.post("/tts", data=b"   ")).status, 400)
        self.assertEqual((await self.client.post("/tts?format=mp3", data=b"hi")).status, 400)

    async def test_tts_normalises_the_text_before_the_engine_sees_it(self):
        """Phase 3b: markdown/symbols/units are expanded, not spoken literally."""
        resp = await self.client.post("/tts", json={"text": "**Warm** at 21\u00b0C."})
        self.assertEqual(resp.status, 200)
        spoken = " ".join(vd._engines["tts"].calls)
        self.assertIn("21 degrees Celsius", spoken)
        self.assertNotIn("**", spoken)
        self.assertNotIn("\u00b0", spoken)

    async def test_tts_paragraph_break_is_silence_not_a_spoken_word(self):
        engine_rate = FakeTTS.sample_rate
        saved = vd.PARAGRAPH_PAUSE_S
        vd.PARAGRAPH_PAUSE_S = 0.5
        try:
            resp = await self.client.post("/tts?format=wav", json={"text": "One.\n\nTwo."})
        finally:
            vd.PARAGRAPH_PAUSE_S = saved
        self.assertEqual(resp.status, 200)
        calls = vd._engines["tts"].calls
        self.assertEqual(calls, ["One.", "Two."])  # the pause is never spoken
        # Two 0.1 s chunks plus 0.5 s of silence.
        self.assertAlmostEqual(float(resp.headers["X-Audio-Seconds"]), 0.7, places=2)
        self.assertEqual(engine_rate, 16000)

    async def test_tts_long_normalises_and_pauses_too(self):
        saved = vd.PARAGRAPH_PAUSE_S
        vd.PARAGRAPH_PAUSE_S = 0.5
        try:
            resp = await self.client.post("/tts/long", json={
                "text": "# Chapter\n\n**Warm** at 21\u00b0C.\n\nSecond paragraph.",
                "format": "wav"})
            self.assertEqual(resp.status, 202)
            job_id = (await resp.json())["job"]["id"]
            job = (await (await self.client.get(f"/jobs/{job_id}")).json())["job"]
            for _ in range(200):
                job = (await (await self.client.get(f"/jobs/{job_id}")).json())["job"]
                if job["status"] in ("done", "error", "cancelled"):
                    break
                await asyncio.sleep(0.02)
        finally:
            vd.PARAGRAPH_PAUSE_S = saved
        self.assertEqual(job["status"], "done", job.get("error"))
        spoken = " ".join(vd._engines["tts"].calls)
        self.assertIn("21 degrees Celsius", spoken)
        self.assertNotIn("**", spoken)
        # One 0.5 s silence between the two paragraphs, on top of the chunks.
        self.assertGreaterEqual(job["audio_s"], 0.5)

    async def test_voices_lists_the_engine_voices(self):
        body = await (await self.client.get("/voices")).json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["engine"], "fake_tts")
        self.assertEqual(body["default"], "fake_default")
        self.assertIn("fake_other", body["voices"])

    async def test_tts_honours_a_named_voice(self):
        resp = await self.client.post("/tts?voice=fake_other", data=b"hello there")
        self.assertEqual(resp.status, 200)
        self.assertEqual(vd._engines["tts"].params[-1].voice, "fake_other")
        resp = await self.client.post("/tts", json={"text": "hello", "voice": "fake_other"})
        self.assertEqual(resp.status, 200)
        self.assertEqual(vd._engines["tts"].params[-1].voice, "fake_other")

    async def test_tts_rejects_an_unknown_voice_instead_of_falling_back(self):
        resp = await self.client.post("/tts?voice=nope", data=b"hello there")
        self.assertEqual(resp.status, 400)
        self.assertIn("unknown voice", (await resp.json())["error"])
        self.assertEqual(vd._engines["tts"].calls, [])  # no silent fallback audio

    async def test_load_and_release_roundtrip(self):
        body = await (await self.client.post("/load?engine=stt")).json()
        self.assertTrue(body["engines"]["stt"]["loaded"])
        self.assertFalse(body["engines"]["tts"]["loaded"])
        body = await (await self.client.post("/release?engine=stt")).json()
        self.assertEqual(body["released"], ["stt"])
        self.assertEqual((await self.client.post("/load?engine=bogus")).status, 400)


# ---- invariants ----------------------------------------------------------

class InvariantTests(unittest.TestCase):
    def test_loopback_only_is_enforced_in_main(self):
        saved = sys.argv
        sys.argv = ["voice_daemon.py", "--host", "0.0.0.0"]
        try:
            with self.assertRaisesRegex(SystemExit, "loopback-only"):
                vd.main()
        finally:
            sys.argv = saved

    def test_no_chat_platform_leaks_into_the_daemon(self):
        """Invariant 2: this daemon knows nothing about chat surfaces.

        Checks code only — the module docstring states the invariant and names
        the banned concepts on purpose.
        """
        import ast

        source = Path(vd.__file__).read_text()
        tree = ast.parse(source)
        tree.body = [n for n in tree.body
                     if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                             and isinstance(n.value.value, str))]
        code = ast.unparse(tree).lower()
        for banned in ("telegram", "chat_id", "bot_token", "openclaw"):
            self.assertNotIn(banned, code, f"{banned!r} leaked into the daemon")


if __name__ == "__main__":
    unittest.main()
