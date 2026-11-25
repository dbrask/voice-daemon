"""Long-form TTS job tests: chaptering, the store, and the HTTP job surface.

No GPU: a fake engine returns silence, so the whole /tts/long path — chapter
split, per-chapter ffmpeg encode to disk, progress, cancel, TTL sweep — runs
in CI. ffmpeg IS required (it is the daemon's only audio boundary).

Run:  .venv/bin/python -m unittest discover -s tests -v
"""

import asyncio
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from aiohttp.test_utils import AioHTTPTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jobs as jobs_mod  # noqa: E402
import voice_daemon as vd  # noqa: E402
from jobs import JobStore, normalize_chapters, split_chapters, valid_job_id  # noqa: E402

CHAPTERED = """# One

First chapter body. It has two sentences.

# Two

Second chapter body.

# Three

Third chapter body.
"""


# ---- chaptering ----------------------------------------------------------

class ChapterSplitTests(unittest.TestCase):
    def test_headings_become_chapters(self):
        chapters = split_chapters(CHAPTERED)
        self.assertEqual([t for t, _ in chapters], ["One", "Two", "Three"])
        self.assertIn("First chapter body", chapters[0][1])
        self.assertNotIn("#", chapters[0][1])

    def test_preamble_before_the_first_heading_is_kept(self):
        chapters = split_chapters("Front matter here.\n\n# One\n\nBody.")
        self.assertEqual([t for t, _ in chapters], ["", "One"])
        self.assertEqual(chapters[0][1], "Front matter here.")

    def test_empty_and_unheaded_text(self):
        self.assertEqual(split_chapters(""), [])
        self.assertEqual(split_chapters("   \n\n "), [])
        self.assertEqual(split_chapters("Just prose."), [("", "Just prose.")])
        # split=none keeps headings in the spoken text as one chapter
        self.assertEqual(len(split_chapters(CHAPTERED, split="none")), 1)

    def test_heading_with_no_body_is_dropped(self):
        chapters = split_chapters("# Empty\n\n# Real\n\nBody.")
        self.assertEqual([t for t, _ in chapters], ["Real"])

    def test_explicit_chapters_accept_strings_and_objects(self):
        chapters = normalize_chapters(["plain", {"title": "T", "text": "body"}, {"text": "  "}])
        self.assertEqual(chapters, [("", "plain"), ("T", "body")])
        with self.assertRaisesRegex(ValueError, "string or"):
            normalize_chapters([123])


# ---- store ---------------------------------------------------------------

class JobStoreTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="voice-jobs-test-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.store = JobStore(self.root, ttl_s=60)

    def test_create_lays_out_a_job_dir_and_snapshot(self):
        job = self.store.create([("One", "a"), ("Two", "bb")], voice="af_heart")
        self.assertTrue(valid_job_id(job.id))
        self.assertTrue((job.dir / "job.json").is_file())
        snap = job.snapshot()
        self.assertEqual(snap["status"], "queued")
        self.assertEqual(snap["voice"], "af_heart")
        self.assertEqual(snap["progress"]["chapters_total"], 2)
        self.assertEqual(snap["progress"]["chars_total"], 3)
        self.assertIs(self.store.get(job.id), job)

    def test_chapter_filenames_are_numbered_and_slugged(self):
        job = self.store.create([("Chapter One: A Start!", "a"), ("", "b")])
        names = [job.chapter_path(c).name for c in job.chapters]
        self.assertEqual(names, ["001-chapter-one-a-start.ogg", "002-chapter-2.ogg"])

    def test_cancel_of_a_queued_job_is_terminal(self):
        job = self.store.create([("", "a")])
        self.assertTrue(job.cancel())
        self.assertEqual(job.status, "cancelled")
        self.assertEqual(job.chapters[0].status, "cancelled")
        self.assertFalse(job.cancel())  # already terminal

    def test_unknown_and_malformed_ids_are_not_found(self):
        self.assertIsNone(self.store.get("nope"))
        self.assertIsNone(self.store.get("../../etc"))  # no path traversal
        self.assertFalse(self.store.delete("deadbeef"))

    def test_delete_removes_the_audio(self):
        job = self.store.create([("", "a")])
        self.assertTrue(self.store.delete(job.id))
        self.assertFalse(job.dir.exists())
        self.assertIsNone(self.store.get(job.id))

    def test_purge_drops_expired_jobs_and_orphan_dirs(self):
        fresh = self.store.create([("", "a")])
        old = self.store.create([("", "a")])
        old.finish("done")
        old.finished_at = time.time() - 3600  # past the 60 s TTL
        orphan = self.root / "0123456789ab"  # what a daemon restart leaves
        orphan.mkdir()
        import os
        os.utime(orphan, (time.time() - 3600, time.time() - 3600))

        removed = self.store.purge()
        self.assertIn(old.id, removed)
        self.assertIn(orphan.name, removed)
        self.assertFalse(orphan.exists())
        self.assertIsNotNone(self.store.get(fresh.id))  # running job untouched

    def test_trim_keeps_the_cap(self):
        store = JobStore(self.root, ttl_s=60, max_jobs=2)
        first = store.create([("", "a")])
        first.finish("done")
        second = store.create([("", "a")])
        second.finish("done")
        store.create([("", "a")])
        self.assertIsNone(store.get(first.id))  # oldest terminal job evicted
        self.assertEqual(len(store.list()), 2)


# ---- HTTP ----------------------------------------------------------------

class SlowFakeTTS:
    """Fake engine: 0.1 s of silence per chunk, optionally with a delay."""

    name = "fake_tts"
    sample_rate = 16000
    max_chars = 40
    voice = "fake_default"
    delay = 0.0

    def __init__(self):
        self.calls: list[str] = []
        self.params: list = []
        self.unloaded = False

    def voices(self):
        return [self.voice, "fake_other"]

    def synthesize(self, text, params):
        if params.voice and params.voice not in self.voices():
            raise ValueError(f"unknown voice {params.voice!r}")
        if self.delay:
            time.sleep(self.delay)
        self.calls.append(text)
        self.params.append(params)
        return np.zeros(self.sample_rate // 10, dtype=np.int16), self.sample_rate

    def warmup(self):
        pass

    def unload(self):
        self.unloaded = True


class LongJobHTTPTests(AioHTTPTestCase):
    async def get_application(self):
        self.root = Path(tempfile.mkdtemp(prefix="voice-jobs-http-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

        saved_engines, saved_store = dict(vd._engines), vd._jobs
        saved_loader, saved_dir = vd.load_tts_engine, vd.JOBS_DIR
        vd._engines.update({"stt": None, "tts": None})
        vd.load_tts_engine = lambda name: SlowFakeTTS()
        vd.JOBS_DIR = self.root
        vd._jobs = JobStore(self.root, ttl_s=60)

        def restore():
            vd._engines.update(saved_engines)
            vd.load_tts_engine, vd._jobs, vd.JOBS_DIR = saved_loader, saved_store, saved_dir

        self.addCleanup(restore)
        return vd.build_app()

    async def _await_status(self, job_id, statuses=("done", "cancelled", "error"), timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            body = await (await self.client.get(f"/jobs/{job_id}")).json()
            if body["job"]["status"] in statuses:
                return body["job"]
            await asyncio.sleep(0.02)
        self.fail(f"job {job_id} never reached {statuses}")

    async def test_multi_chapter_job_writes_one_file_per_chapter(self):
        resp = await self.client.post("/tts/long", json={"text": CHAPTERED, "voice": "fake_other"})
        self.assertEqual(resp.status, 202)
        job = (await resp.json())["job"]
        self.assertEqual(job["progress"]["chapters_total"], 3)

        done = await self._await_status(job["id"])
        self.assertEqual(done["status"], "done")
        self.assertEqual(done["progress"]["chapters_done"], 3)
        self.assertEqual(done["progress"]["percent"], 100.0)
        self.assertGreater(done["audio_s"], 0)

        paths = [Path(c["path"]) for c in done["chapters"]]
        self.assertEqual([p.name for p in paths],
                         ["001-one.ogg", "002-two.ogg", "003-three.ogg"])
        for path in paths:
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(path.parent.parent, self.root)  # under the jobs dir
        self.assertTrue((paths[0].parent / "job.json").is_file())
        # voice honoured on every chunk, no silent fallback
        self.assertTrue(all(p.voice == "fake_other" for p in vd._engines["tts"].params))

    async def test_job_reuses_split_sentences_per_chapter(self):
        text = "# A\n\n" + "One sentence here. " * 5
        resp = await self.client.post("/tts/long", json={"text": text})
        await self._await_status((await resp.json())["job"]["id"])
        calls = vd._engines["tts"].calls
        self.assertGreater(len(calls), 1)  # chunked, not one giant call
        self.assertTrue(all(len(c) <= 40 for c in calls))  # at the engine limit

    async def test_explicit_chapters_and_wav_format(self):
        resp = await self.client.post("/tts/long?format=wav", json={
            "chapters": [{"title": "Intro", "text": "Hello there."}, "Second bit."],
        })
        self.assertEqual(resp.status, 202)
        done = await self._await_status((await resp.json())["job"]["id"])
        self.assertEqual(done["status"], "done")
        names = [Path(c["path"]).name for c in done["chapters"]]
        self.assertEqual(names, ["001-intro.wav", "002-chapter-2.wav"])

    async def test_cancel_stops_the_job_and_keeps_finished_chapters(self):
        vd._engines["tts"] = SlowFakeTTS()
        vd._engines["tts"].delay = 0.05
        text = "# A\n\n" + "Sentence one here. " * 8 + "\n\n# B\n\n" + "Sentence two here. " * 8
        resp = await self.client.post("/tts/long", json={"text": text})
        job_id = (await resp.json())["job"]["id"]

        await asyncio.sleep(0.15)  # let it get into chapter one
        body = await (await self.client.post(f"/jobs/{job_id}/cancel")).json()
        self.assertTrue(body["cancelled"])

        done = await self._await_status(job_id)
        self.assertEqual(done["status"], "cancelled")
        self.assertLess(done["progress"]["chapters_done"], 2)
        # the interrupted chapter left no half file behind
        for chapter in done["chapters"]:
            if chapter["status"] != "done":
                self.assertIsNone(chapter["path"])

    async def test_cancel_of_a_finished_job_reports_false(self):
        resp = await self.client.post("/tts/long", json={"text": "Short one."})
        job_id = (await resp.json())["job"]["id"]
        await self._await_status(job_id)
        body = await (await self.client.post(f"/jobs/{job_id}/cancel")).json()
        self.assertFalse(body["cancelled"])

    async def test_delete_removes_job_and_files(self):
        resp = await self.client.post("/tts/long", json={"text": "Short one."})
        job_id = (await resp.json())["job"]["id"]
        done = await self._await_status(job_id)
        job_dir = Path(done["dir"])
        self.assertTrue(job_dir.is_dir())
        self.assertEqual((await self.client.delete(f"/jobs/{job_id}")).status, 200)
        self.assertFalse(job_dir.exists())
        self.assertEqual((await self.client.get(f"/jobs/{job_id}")).status, 404)

    async def test_jobs_list_and_health_report_jobs(self):
        await self.client.post("/tts/long", json={"text": "Short one."})
        body = await (await self.client.get("/jobs")).json()
        self.assertEqual(len(body["jobs"]), 1)
        health = await (await self.client.get("/health")).json()
        self.assertEqual(health["jobs"]["dir"], str(self.root))
        self.assertEqual(health["jobs"]["total"], 1)

    async def test_bad_requests(self):
        self.assertEqual((await self.client.post("/tts/long", data=b"   ")).status, 400)
        self.assertEqual((await self.client.post("/tts/long?format=mp3",
                                                 json={"text": "hi"})).status, 400)
        self.assertEqual((await self.client.post("/tts/long?split=weird",
                                                 json={"text": "hi"})).status, 400)
        self.assertEqual((await self.client.post("/tts/long",
                                                 json={"chapters": "not a list"})).status, 400)
        self.assertEqual((await self.client.get("/jobs/deadbeef")).status, 404)
        self.assertEqual((await self.client.post("/jobs/deadbeef/cancel")).status, 404)

    async def test_unknown_voice_is_rejected_at_submit_not_mid_job(self):
        resp = await self.client.post("/tts/long", json={"text": CHAPTERED, "voice": "nope"})
        self.assertEqual(resp.status, 400)
        self.assertIn("unknown voice", (await resp.json())["error"])
        self.assertEqual(vd._engines["tts"].calls, [])
        self.assertEqual((await (await self.client.get("/jobs")).json())["jobs"], [])

    async def test_oversize_text_is_rejected(self):
        saved = vd.MAX_LONG_TEXT_CHARS
        vd.MAX_LONG_TEXT_CHARS = 20
        try:
            resp = await self.client.post("/tts/long", json={"text": "x " * 40})
        finally:
            vd.MAX_LONG_TEXT_CHARS = saved
        self.assertEqual(resp.status, 400)

    async def test_batch_tts_still_works_alongside_jobs(self):
        """Phase 3a must not change POST /tts."""
        resp = await self.client.post("/tts", data=b"hello there")
        self.assertEqual(resp.status, 200)
        self.assertTrue(resp.headers["Content-Type"].startswith("audio/ogg"))


class NoWholeDocumentPCMTests(unittest.IsolatedAsyncioTestCase):
    """Invariant: a job never holds more than one chunk of PCM in memory."""

    async def test_encoder_streams_each_chunk_to_ffmpeg(self):
        tmp = Path(tempfile.mkdtemp(prefix="voice-enc-test-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        out = tmp / "chapter.ogg"
        encoder = vd._ChapterEncoder("ogg", out)
        for _ in range(5):
            encoder.write(np.zeros(16000, dtype=np.int16), 16000)
        # ffmpeg opened the file and is writing while we still hold no PCM:
        # the chapter exists on disk before close(), not after a big concat.
        deadline = time.monotonic() + 5.0
        while not out.exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        self.assertTrue(out.exists(), "ffmpeg never opened the chapter file")
        audio_s = encoder.close()
        self.assertAlmostEqual(audio_s, 5.0, places=1)
        self.assertGreater(out.stat().st_size, 0)

    async def test_encoder_abort_leaves_no_file(self):
        tmp = Path(tempfile.mkdtemp(prefix="voice-enc-test-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        out = tmp / "chapter.ogg"
        encoder = vd._ChapterEncoder("ogg", out)
        encoder.write(np.zeros(16000, dtype=np.int16), 16000)
        encoder.abort()
        self.assertFalse(out.exists())

    async def test_synthesize_never_accumulates_chapter_pcm(self):
        """The daemon holds one chunk at a time: assert on source, not vibes."""
        import inspect

        source = inspect.getsource(vd._synthesize_chapter)
        self.assertNotIn("concatenate", source)
        self.assertIn("encoder.write", source)


if __name__ == "__main__":
    unittest.main()
