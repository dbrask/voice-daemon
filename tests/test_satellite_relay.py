"""Satellite relay: auth, protocol order, error codes, Hermes contract.

No GPU, no network, no real Hermes: the voice daemon and the api_server are
stub aiohttp servers on ephemeral ports that record every request, so the tests
assert the exact wire contract the relay promises (headers, bodies, order).

Run: .venv/bin/python -m unittest discover -s tests
"""

import asyncio
import contextlib
import datetime as dt
import io
import json
import sys
import unittest
import unittest.mock
import wave
from pathlib import Path

import aiohttp
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satellite import protocol as P  # noqa: E402
from satellite import relay as R  # noqa: E402

TOKEN = "a" * 64
DEVICE = "desk"
TOKENS = {DEVICE: R.token_digest(TOKEN)}
AUTH = {"Authorization": f"Bearer {TOKEN}"}

WAV_REPLY = b"RIFF" + b"\0" * 40  # opaque to the relay; it only forwards bytes


def pcm(seconds: float = 0.1) -> bytes:
    return b"\0" * int(P.RATE * P.SAMPLE_WIDTH * seconds)


# ---- stubs ---------------------------------------------------------------

class VoiceStub:
    """Stands in for voice_daemon on :9877. Records /stt and /tts requests."""

    def __init__(self, *, transcript="turn on the lamp", stt_ok=True, stt_status=200,
                 tts_status=200):
        self.transcript = transcript
        self.stt_ok = stt_ok
        self.stt_status = stt_status
        self.tts_status = tts_status
        self.stt_requests = []
        self.tts_requests = []

    def app(self) -> web.Application:
        app = web.Application(client_max_size=1024 * 1024 * 8)
        app.router.add_get("/health", self._health)
        app.router.add_post("/stt", self._stt)
        app.router.add_post("/tts", self._tts)
        return app

    async def _health(self, request):
        return web.json_response({"ok": True})

    async def _stt(self, request):
        body = await request.read()
        self.stt_requests.append({"language": request.query.get("language"), "body": body})
        if self.stt_status != 200:
            return web.json_response({"ok": False, "error": "boom"}, status=self.stt_status)
        if not self.stt_ok:
            return web.json_response({"ok": False, "error": "decode failed"})
        return web.json_response({
            "ok": True, "text": self.transcript, "language": "en", "audio_s": 1.0,
            "timing_ms": {"total": 120.0}})

    async def _tts(self, request):
        payload = await request.json()
        self.tts_requests.append({
            "content_type": request.headers.get("Content-Type"), "payload": payload})
        if self.tts_status != 200:
            return web.json_response({"ok": False, "error": "boom"}, status=self.tts_status)
        return web.Response(body=WAV_REPLY, content_type="audio/wav")


class HermesStub:
    """Stands in for the api_server's /v1/runs. `statuses` is the poll script."""

    def __init__(self, *, statuses=None, output="Done.\n[[mood:happy]]", error=None,
                 poll_delay_s=0.0):
        # Each GET pops the next status; the last one repeats.
        self.statuses = list(statuses or ["completed"])
        self.output = output
        self.error = error
        self.poll_delay_s = poll_delay_s
        self.runs = []
        self.polls = 0
        self.stops = []
        self.approvals = []

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/runs", self._start)
        app.router.add_get("/v1/runs/{run_id}", self._get)
        app.router.add_post("/v1/runs/{run_id}/stop", self._stop)
        app.router.add_post("/v1/runs/{run_id}/approval", self._approval)
        return app

    async def _start(self, request):
        self.runs.append({
            "payload": await request.json(),
            "idempotency_key": request.headers.get("Idempotency-Key"),
            "authorization": request.headers.get("Authorization"),
            "content_type": request.headers.get("Content-Type")})
        return web.json_response({"run_id": "run_1", "status": "queued", "replayed": False},
                                 status=202)

    async def _get(self, request):
        self.polls += 1
        if self.poll_delay_s:
            await asyncio.sleep(self.poll_delay_s)
        status = self.statuses[0] if len(self.statuses) == 1 else self.statuses.pop(0)
        body = {"object": "hermes.run", "run_id": "run_1", "status": status}
        if status == "completed":
            body["output"] = self.output
        if status in ("failed", "interrupted"):
            body["error"] = self.error or "the model exploded"
        return web.json_response(body)

    async def _stop(self, request):
        self.stops.append(request.match_info["run_id"])
        # Real api_server answers `stopping`, then the run settles to cancelled.
        self.statuses = ["cancelled"]
        return web.json_response({"run_id": "run_1", "status": "stopping"})

    async def _approval(self, request):
        self.approvals.append(await request.json())
        # A denied approval lets the run finish.
        self.statuses = ["completed"]
        return web.json_response({"run_id": "run_1", "choice": "deny", "resolved": 1})


# ---- harness -------------------------------------------------------------

class RelayCase(unittest.IsolatedAsyncioTestCase):
    """Boots the voice stub, the Hermes stub and the relay under test."""

    voice_kwargs: dict = {}
    hermes_kwargs: dict = {}
    hermes_timeout_s: float = 30.0

    async def asyncSetUp(self):
        self.voice_stub = VoiceStub(**self.voice_kwargs)
        self.hermes_stub = HermesStub(**self.hermes_kwargs)
        self.voice_server = TestServer(self.voice_stub.app())
        self.hermes_server = TestServer(self.hermes_stub.app())
        await self.voice_server.start_server()
        await self.hermes_server.start_server()

        self.relay = R.Relay(
            tokens=dict(TOKENS),
            voice=R.VoiceClient(str(self.voice_server.make_url("")).rstrip("/")),
            hermes=R.HermesClient(
                str(self.hermes_server.make_url("")).rstrip("/"), "hermes-key",
                timeout_s=self.hermes_timeout_s, poll_interval_s=0.01),
            heartbeat=None)
        self.client = TestClient(TestServer(self.relay.build_app()))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)
        self.addAsyncCleanup(self.voice_server.close)
        self.addAsyncCleanup(self.hermes_server.close)

    async def connect(self, *, hello=True, device=DEVICE, path="/ws", headers=AUTH):
        ws = await self.client.ws_connect(path, headers=headers)
        if hello:
            await ws.send_str(P.encode("hello", v=P.V, device=device, client="test"))
        return ws

    async def collect(self, ws, *, until="turn.done", limit=20):
        """Read frames until `until` arrives. Binary frames land as ('binary', bytes)."""
        frames = []
        for _ in range(limit):
            msg = await asyncio.wait_for(ws.receive(), timeout=10)
            if msg.type == WSMsgType.BINARY:
                frames.append(("binary", msg.data))
                continue
            if msg.type != WSMsgType.TEXT:
                break
            frame = P.decode(msg.data, schema=P.RELAY_SCHEMA)
            frames.append(frame)
            if frame["type"] == until:
                break
        return frames

    @staticmethod
    def order(frames):
        return [f[0] if isinstance(f, tuple) else f["type"] for f in frames]

    async def one_turn(self, ws, *, audio=None, turn_id="t1", end=True):
        await ws.send_str(P.turn_start(turn_id))
        await ws.send_bytes(audio if audio is not None else pcm())
        if end:
            await ws.send_str(P.encode("turn.end", turn_id=turn_id))
        return await self.collect(ws)


# ---- auth ----------------------------------------------------------------

class TestAuth(RelayCase):
    async def test_missing_token_is_401_before_upgrade(self):
        resp = await self.client.get("/ws")
        self.assertEqual(resp.status, 401)

    async def test_wrong_token_is_401(self):
        resp = await self.client.get("/ws", headers={"Authorization": f"Bearer {'b' * 64}"})
        self.assertEqual(resp.status, 401)

    async def test_non_bearer_scheme_is_401(self):
        resp = await self.client.get("/ws", headers={"Authorization": f"Basic {TOKEN}"})
        self.assertEqual(resp.status, 401)

    async def test_hello_device_mismatch_closes_4401(self):
        ws = await self.connect(hello=False)
        await ws.send_str(P.encode("hello", v=P.V, device="kitchen", client="test"))
        msg = await asyncio.wait_for(ws.receive(), timeout=5)
        self.assertEqual(msg.type, WSMsgType.CLOSE)
        self.assertEqual(ws.close_code, P.CLOSE_DEVICE_MISMATCH)

    async def test_good_token_gets_ready_with_the_session_id(self):
        ws = await self.connect()
        frame = P.decode((await ws.receive()).data, schema=P.RELAY_SCHEMA)
        self.assertEqual(frame["type"], "ready")
        self.assertEqual(frame["device"], DEVICE)
        self.assertRegex(frame["session_id"], r"^sat-desk-\d{8}$")

    async def test_turn_before_hello_is_rejected(self):
        ws = await self.connect(hello=False)
        await ws.send_str(P.turn_start("t1"))
        frame = P.decode((await ws.receive()).data, schema=P.RELAY_SCHEMA)
        self.assertEqual((frame["type"], frame["code"]), ("error", "bad_message"))

    async def test_health_needs_no_token_and_reports_the_daemon(self):
        for path in ("/health", "/satellite/health"):
            with self.subTest(path=path):
                resp = await self.client.get(path)
                self.assertEqual(resp.status, 200)
                self.assertEqual(await resp.json(), {"ok": True, "v": 1, "voice": True})

    async def test_the_satellite_prefixed_ws_route_exists(self):
        # A reverse proxy mounted at /satellite may or may not strip the prefix.
        resp = await self.client.get("/satellite/ws")
        self.assertEqual(resp.status, 401)  # routed, and then auth-checked


# ---- happy path ----------------------------------------------------------

class TestHappyPath(RelayCase):
    async def test_frame_order_and_full_timing(self):
        ws = await self.connect()
        frames = await self.collect(ws, until="ready")
        frames += await self.one_turn(ws)
        self.assertEqual(
            self.order(frames),
            ["ready", "transcript", "reply", "audio", "binary", "turn.done"])

        done = frames[-1]
        self.assertFalse(done["cancelled"])
        for key in P.TIMING_KEYS:
            with self.subTest(key=key):
                self.assertIsInstance(done["timing"][key], (int, float), f"{key} did not run")

    async def test_audio_never_precedes_reply(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        order = self.order(await self.one_turn(ws))
        self.assertLess(order.index("reply"), order.index("audio"))
        self.assertLess(order.index("audio"), order.index("binary"))

    async def test_transcript_and_reply_content(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        frames = await self.one_turn(ws)
        by_type = {f["type"]: f for f in frames if not isinstance(f, tuple)}
        self.assertEqual(by_type["transcript"]["text"], "turn on the lamp")
        # The mood tag is parsed out; reply.text is what gets spoken.
        self.assertEqual(by_type["reply"]["text"], "Done.")
        self.assertEqual(by_type["reply"]["mood"], "happy")

    async def test_the_reply_wav_is_forwarded_verbatim(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        frames = await self.one_turn(ws)
        binary = [f[1] for f in frames if isinstance(f, tuple)]
        self.assertEqual(binary, [WAV_REPLY])

    async def test_stt_gets_a_valid_wav_at_the_pinned_language(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        await self.one_turn(ws, audio=pcm(0.5))
        self.assertEqual(len(self.voice_stub.stt_requests), 1)
        req = self.voice_stub.stt_requests[0]
        self.assertEqual(req["language"], "en")
        with wave.open(io.BytesIO(req["body"]), "rb") as wav:
            self.assertEqual(wav.getframerate(), P.RATE)
            self.assertEqual(wav.getnchannels(), P.CHANNELS)
            self.assertEqual(wav.getsampwidth(), P.SAMPLE_WIDTH)
            self.assertEqual(wav.getnframes(), int(P.RATE * 0.5))

    async def test_tts_is_called_as_json_asking_for_wav(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        await self.one_turn(ws)
        self.assertEqual(len(self.voice_stub.tts_requests), 1)
        req = self.voice_stub.tts_requests[0]
        # Without this content type the daemon speaks the raw body and emits ogg.
        self.assertTrue(req["content_type"].startswith("application/json"))
        self.assertEqual(req["payload"], {"text": "Done.", "format": "wav"})

    async def test_hermes_contract(self):
        ws = await self.connect()
        ready = (await self.collect(ws, until="ready"))[0]
        await self.one_turn(ws, turn_id="abc-123")
        self.assertEqual(len(self.hermes_stub.runs), 1)
        run = self.hermes_stub.runs[0]
        self.assertEqual(run["idempotency_key"], f"sat-{DEVICE}-abc-123")
        self.assertEqual(run["authorization"], "Bearer hermes-key")
        self.assertEqual(run["payload"]["input"], "turn on the lamp")
        self.assertEqual(run["payload"]["session_id"], ready["session_id"])
        self.assertEqual(run["payload"]["instructions"], R.VOICE_INSTRUCTIONS)

    async def test_two_turns_reuse_the_session_and_get_distinct_keys(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        await self.one_turn(ws, turn_id="t1")
        self.hermes_stub.statuses = ["completed"]
        await self.one_turn(ws, turn_id="t2")
        keys = [r["idempotency_key"] for r in self.hermes_stub.runs]
        sessions = {r["payload"]["session_id"] for r in self.hermes_stub.runs}
        self.assertEqual(keys, [f"sat-{DEVICE}-t1", f"sat-{DEVICE}-t2"])
        self.assertEqual(len(sessions), 1)


class TestPolling(RelayCase):
    hermes_kwargs = {"statuses": ["queued", "running", "running", "completed"]}

    async def test_non_terminal_statuses_are_polled_through(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        frames = await self.one_turn(ws)
        self.assertEqual(self.order(frames)[-1], "turn.done")
        self.assertGreaterEqual(self.hermes_stub.polls, 4)


class TestApprovalAutoDeny(RelayCase):
    hermes_kwargs = {"statuses": ["waiting_for_approval"]}

    async def test_approval_is_denied_for_the_whole_run(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        frames = await self.one_turn(ws)
        self.assertEqual(self.hermes_stub.approvals, [{"choice": "deny", "all": True}])
        # And the run still completes and speaks.
        self.assertIn("binary", self.order(frames))


# ---- failures ------------------------------------------------------------

class ErrorMixin(unittest.TestCase):
    """Shared assertion: an error is always followed by exactly one turn.done."""

    def assert_error_then_done(self, frames, code, *, cancelled=False):
        order = self.order(frames)
        self.assertEqual(order[-2:], ["error", "turn.done"], order)
        errors = [f for f in frames if not isinstance(f, tuple) and f["type"] == "error"]
        self.assertEqual(errors[-1]["code"], code)
        self.assertEqual(frames[-1]["cancelled"], cancelled)
        # turn.done always carries every timing key, nulls for stages that ran not.
        self.assertEqual(set(frames[-1]["timing"]), set(P.TIMING_KEYS))
        self.assertEqual(order.count("turn.done"), 1, "turn.done must be sent exactly once")


class TestEmptyTranscript(RelayCase, ErrorMixin):
    voice_kwargs = {"transcript": "   "}

    async def test_hermes_is_not_called(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        frames = await self.one_turn(ws)
        self.assert_error_then_done(frames, "empty_transcript")
        self.assertEqual(self.hermes_stub.runs, [], "Hermes must not see an empty transcript")
        self.assertEqual(self.voice_stub.tts_requests, [])


class TestSTTFailures(RelayCase, ErrorMixin):
    voice_kwargs = {"stt_ok": False}

    async def test_ok_false_is_stt_failed(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        self.assert_error_then_done(await self.one_turn(ws), "stt_failed")
        self.assertEqual(self.hermes_stub.runs, [])


class TestSTTHTTPError(RelayCase, ErrorMixin):
    voice_kwargs = {"stt_status": 500}

    async def test_non_2xx_is_stt_failed(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        self.assert_error_then_done(await self.one_turn(ws), "stt_failed")


class TestTTSFailure(RelayCase, ErrorMixin):
    voice_kwargs = {"tts_status": 500}

    async def test_tts_failure_still_delivers_the_reply_text(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        frames = await self.one_turn(ws)
        self.assert_error_then_done(frames, "tts_failed")
        # The reply text got through even though the audio did not.
        self.assertIn("reply", self.order(frames))
        self.assertNotIn("binary", self.order(frames))


class TestHermesFailed(RelayCase, ErrorMixin):
    hermes_kwargs = {"statuses": ["failed"], "error": "provider 500"}

    async def test_failed_run_skips_tts(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        frames = await self.one_turn(ws)
        self.assert_error_then_done(frames, "hermes_failed")
        self.assertEqual(self.voice_stub.tts_requests, [], "no TTS call on a failed run")
        errors = [f for f in frames if not isinstance(f, tuple) and f["type"] == "error"]
        self.assertIn("provider 500", errors[-1]["message"])


class TestHermesEmptyOutput(RelayCase, ErrorMixin):
    hermes_kwargs = {"output": "[[mood:neutral]]"}

    async def test_empty_after_mood_strip_is_hermes_failed_with_no_tts(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        frames = await self.one_turn(ws)
        self.assert_error_then_done(frames, "hermes_failed")
        self.assertEqual(self.voice_stub.tts_requests, [])


class TestHermesTimeout(RelayCase, ErrorMixin):
    hermes_kwargs = {"statuses": ["running"]}
    hermes_timeout_s = 0.05

    async def test_timeout_stops_the_run_then_errors(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        frames = await self.one_turn(ws)
        self.assert_error_then_done(frames, "hermes_timeout")
        self.assertEqual(self.hermes_stub.stops, ["run_1"])
        self.assertEqual(self.voice_stub.tts_requests, [])


# ---- turn lifecycle ------------------------------------------------------

class TestCancelWhileThinking(RelayCase, ErrorMixin):
    """The regression that the live run caught: a `cancel` arriving WHILE Hermes
    is still running must be read immediately, not after the pipeline finishes.
    If the reader awaits the pipeline, the turn completes and speaks anyway."""

    hermes_kwargs = {"statuses": ["running"], "poll_delay_s": 0.3}
    hermes_timeout_s = 30.0

    async def test_cancel_mid_run_stops_hermes_and_speaks_nothing(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        await ws.send_str(P.turn_start("t1"))
        await ws.send_bytes(pcm())
        await ws.send_str(P.encode("turn.end", turn_id="t1"))
        # Wait for the transcript so we know the run has actually started.
        first = P.decode((await asyncio.wait_for(ws.receive(), timeout=5)).data,
                         schema=P.RELAY_SCHEMA)
        self.assertEqual(first["type"], "transcript")

        await ws.send_str(P.encode("cancel", turn_id="t1"))
        frames = await self.collect(ws)
        self.assertEqual(self.order(frames), ["turn.done"], "no reply/audio after a cancel")
        self.assertTrue(frames[-1]["cancelled"])
        self.assertEqual(self.hermes_stub.stops, ["run_1"], "the Hermes run must be stopped")
        self.assertEqual(self.voice_stub.tts_requests, [], "nothing is synthesised")

    async def test_the_socket_survives_a_cancel_and_takes_a_new_turn(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        await ws.send_str(P.turn_start("t1"))
        await ws.send_bytes(pcm())
        await ws.send_str(P.encode("turn.end", turn_id="t1"))
        await asyncio.wait_for(ws.receive(), timeout=5)  # transcript
        await ws.send_str(P.encode("cancel", turn_id="t1"))
        await self.collect(ws)

        self.hermes_stub.statuses = ["completed"]
        self.hermes_stub.poll_delay_s = 0.0
        frames = await self.one_turn(ws, turn_id="t2")
        self.assertEqual(
            self.order(frames), ["transcript", "reply", "audio", "binary", "turn.done"])


class TestTurnLifecycle(RelayCase, ErrorMixin):
    async def test_second_turn_start_while_busy_is_busy(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        await ws.send_str(P.turn_start("t1"))
        await ws.send_str(P.turn_start("t2"))
        frame = P.decode((await ws.receive()).data, schema=P.RELAY_SCHEMA)
        self.assertEqual((frame["type"], frame["code"], frame["turn_id"]), ("error", "busy", "t2"))

    async def test_audio_over_the_cap_is_too_long_then_done(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        await ws.send_str(P.turn_start("t1"))
        # 30 s cap; send 31 s in 64 KiB frames.
        chunk = b"\0" * P.MAX_CHUNK_BYTES
        sent = 0
        while sent <= P.MAX_AUDIO_BYTES:
            await ws.send_bytes(chunk)
            sent += len(chunk)
        frames = await self.collect(ws)
        self.assert_error_then_done(frames, "too_long")
        self.assertEqual(self.voice_stub.stt_requests, [], "over-long audio never reaches STT")

    async def test_a_new_turn_works_after_too_long(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        await ws.send_str(P.turn_start("t1"))
        chunk = b"\0" * P.MAX_CHUNK_BYTES
        sent = 0
        while sent <= P.MAX_AUDIO_BYTES:
            await ws.send_bytes(chunk)
            sent += len(chunk)
        await self.collect(ws)
        frames = await self.one_turn(ws, turn_id="t2")
        self.assertEqual(self.order(frames)[-1], "turn.done")
        self.assertNotIn("error", self.order(frames))

    async def test_cancel_stops_the_run_and_reports_cancelled(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        await ws.send_str(P.turn_start("t1"))
        await ws.send_bytes(pcm())
        await ws.send_str(P.encode("cancel", turn_id="t1"))
        frames = await self.collect(ws)
        self.assertEqual(self.order(frames), ["turn.done"])
        self.assertTrue(frames[-1]["cancelled"])
        self.assertEqual(self.voice_stub.stt_requests, [], "a cancelled turn is not transcribed")

    async def test_binary_outside_a_turn_is_bad_message(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        await ws.send_bytes(pcm())
        frame = P.decode((await ws.receive()).data, schema=P.RELAY_SCHEMA)
        self.assertEqual((frame["type"], frame["code"]), ("error", "bad_message"))

    async def test_turn_end_with_a_mismatched_id_is_rejected(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        await ws.send_str(P.turn_start("t1"))
        await ws.send_str(P.encode("turn.end", turn_id="nope"))
        frame = P.decode((await ws.receive()).data, schema=P.RELAY_SCHEMA)
        self.assertEqual((frame["type"], frame["code"]), ("error", "bad_message"))

    async def test_unknown_frame_type_keeps_the_socket_open(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        await ws.send_str('{"type":"turn.pause","turn_id":"t1"}')
        frame = P.decode((await ws.receive()).data, schema=P.RELAY_SCHEMA)
        self.assertEqual((frame["type"], frame["code"]), ("error", "bad_message"))
        self.assertFalse(ws.closed)
        # And a real turn still works afterwards.
        self.assertEqual(self.order(await self.one_turn(ws))[-1], "turn.done")

    async def test_a_wrong_audio_format_is_refused(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        await ws.send_str(P.encode(
            "turn.start", turn_id="t1", format="s16le", rate=48000, channels=1))
        frame = P.decode((await ws.receive()).data, schema=P.RELAY_SCHEMA)
        self.assertEqual((frame["type"], frame["code"]), ("error", "bad_message"))

    async def test_empty_audio_turn_is_empty_transcript(self):
        ws = await self.connect()
        await self.collect(ws, until="ready")
        await ws.send_str(P.turn_start("t1"))
        await ws.send_str(P.encode("turn.end", turn_id="t1"))
        frames = await self.collect(ws)
        self.assert_error_then_done(frames, "empty_transcript")
        self.assertEqual(self.voice_stub.stt_requests, [])


class TestTTSTruncation(RelayCase):
    hermes_kwargs = {"output": "word. " * 400 + "\n[[mood:neutral]]"}

    async def test_long_reply_is_truncated_for_speech_but_not_on_the_wire(self):
        self.relay.tts_max_chars = 200
        ws = await self.connect()
        await self.collect(ws, until="ready")
        frames = await self.one_turn(ws)
        reply = next(f for f in frames if not isinstance(f, tuple) and f["type"] == "reply")
        spoken = self.voice_stub.tts_requests[0]["payload"]["text"]
        self.assertLessEqual(len(spoken), 200)
        self.assertGreater(len(reply["text"]), 200, "reply.text stays full")


# ---- units ---------------------------------------------------------------

class TestUnits(unittest.TestCase):
    def test_relay_refuses_a_non_loopback_bind(self):
        for host in ("0.0.0.0", "203.0.113.7", "relay.example.invalid"):
            with self.subTest(host=host), self.assertRaises(SystemExit) as ctx:
                R.main(["--host", host])
            self.assertIn("loopback", str(ctx.exception))

    def test_session_id_shape(self):
        sid = R.session_id_for("desk", today=dt.date(2026, 9, 29))
        self.assertEqual(sid, "sat-desk-20260929")
        self.assertRegex(sid, r"^sat-desk-\d{8}$")

    def test_tokens_file_parsing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokens"
            path.write_text(
                "# a comment\n\n"
                f"desk {R.token_digest(TOKEN)}\n"
                "malformed-line\n"
                "short abc\n"
                f"kitchen {R.token_digest('x')}\n")
            tokens = R.load_tokens(path)
        self.assertEqual(sorted(tokens), ["desk", "kitchen"])
        self.assertEqual(R.device_for_token(tokens, TOKEN), "desk")
        self.assertEqual(R.device_for_token(tokens, "x"), "kitchen")
        self.assertIsNone(R.device_for_token(tokens, "nope"))
        self.assertIsNone(R.device_for_token(tokens, ""))

    def test_missing_tokens_file_is_empty_not_a_crash(self):
        self.assertEqual(R.load_tokens("/nonexistent/satellite-tokens"), {})

    def test_pcm_to_wav_is_a_readable_wav(self):
        data = R.pcm_to_wav(b"\x01\x02" * 1000)
        with wave.open(io.BytesIO(data), "rb") as wav:
            self.assertEqual(wav.getnframes(), 1000)
            self.assertEqual(wav.getframerate(), P.RATE)

    def test_pcm_duration(self):
        self.assertEqual(R.pcm_duration_ms(P.RATE * P.SAMPLE_WIDTH), 1000.0)
        self.assertEqual(R.pcm_duration_ms(0), 0.0)

    def test_voice_instructions_ask_for_the_mood_tag(self):
        self.assertIn("[[mood:X]]", R.VOICE_INSTRUCTIONS)
        for mood in P.MOODS:
            self.assertIn(mood, R.VOICE_INSTRUCTIONS)


# ---- installation: the documented tokens path, end to end ------------------

REPO = Path(__file__).resolve().parents[1]


def env_example_value(name: str) -> str:
    """The literal value .env.example assigns to `name`, as systemd's
    EnvironmentFile= would pass it: no shell, so no tilde expansion."""
    for line in (REPO / ".env.example").read_text().splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1]
    raise AssertionError(f"{name} is not set in .env.example")


class TestInstalledTokensPath(unittest.IsolatedAsyncioTestCase):
    """Mint a token with scripts/satellite-token and authenticate with it, using
    the SAT_TOKENS_FILE value from .env.example verbatim (a literal `~`)."""

    async def asyncSetUp(self):
        import os
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name) / "home"
        self.cwd = Path(tmp.name) / "cwd"
        self.home.mkdir()
        self.cwd.mkdir()
        self.tokens_value = env_example_value("SAT_TOKENS_FILE")
        env_patch = unittest.mock.patch.dict(os.environ, {"HOME": str(self.home)})
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def mint(self) -> str:
        import os
        import subprocess
        client_token = self.home / "client-token"
        env = dict(os.environ, HOME=str(self.home), SAT_TOKENS_FILE=self.tokens_value,
                   SAT_TOKEN_FILE=str(client_token))
        subprocess.run([str(REPO / "scripts" / "satellite-token"), DEVICE, "--write-local"],
                       cwd=self.cwd, env=env, check=True, capture_output=True)
        return client_token.read_text()

    async def test_the_helper_writes_under_home_not_a_literal_tilde_dir(self):
        self.mint()
        self.assertFalse((self.cwd / "~").exists(), "helper created a directory named '~'")
        expected = self.home / ".config" / "voice-bot" / "satellite-tokens"
        self.assertTrue(expected.is_file())
        self.assertEqual(expected.stat().st_mode & 0o777, 0o600)

    async def test_the_relay_loads_the_env_example_path(self):
        token = self.mint()
        tokens = R.load_tokens(self.tokens_value)
        self.assertEqual(R.device_for_token(tokens, token), DEVICE)

    async def test_a_minted_token_upgrades_against_the_env_example_path(self):
        token = self.mint()
        relay = R.Relay(tokens_file=self.tokens_value, heartbeat=None)
        client = TestClient(TestServer(relay.build_app()))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        ws = await client.ws_connect("/ws", headers={"Authorization": f"Bearer {token}"})
        await ws.send_str(P.encode("hello", v=P.V, device=DEVICE, client="test"))
        frame = P.decode((await asyncio.wait_for(ws.receive(), timeout=5)).data,
                         schema=P.RELAY_SCHEMA)
        self.assertEqual(frame["type"], "ready")
        await ws.close()


# ---- the real client against the real relay, in the real wire order ---------

class TestClientAgainstRelay(RelayCase):
    """satellite.client driving satellite.relay over a real WebSocket. Recorder
    and player are fakes; the player blocks like a reply that is still audible."""

    async def test_tap_after_turn_done_during_playback_stops_it_cleanly(self):
        from satellite import client as C

        class Recorder:
            def __init__(self):
                self.chunks = [pcm(0.1)]

            async def start(self):
                pass

            async def read(self, nbytes=P.MAX_CHUNK_BYTES):
                return self.chunks.pop(0) if self.chunks else b""

            async def stop(self):
                pass

        class Player:
            def __init__(self):
                self.played, self.stopped = [], 0
                self.release = asyncio.Event()

            async def play(self, wav):
                self.played.append(wav)
                await self.release.wait()

            async def stop(self):
                self.stopped += 1
                self.release.set()

        recorder, player = Recorder(), Player()
        sat = C.SatelliteClient(
            relay_url=str(self.client.make_url("/ws")), device=DEVICE, token=TOKEN,
            recorder=recorder, player=player)
        seen = []
        handle_frame = sat.handle_frame

        async def record_frame(frame):
            await handle_frame(frame)
            seen.append(frame["type"])
        sat.handle_frame = record_frame

        async def until(pred, what):
            for _ in range(500):
                if pred():
                    return
                await asyncio.sleep(0.01)
            self.fail(f"timed out waiting for {what}; frames={seen} state={sat.face.state}")

        async with aiohttp.ClientSession() as session:
            conn = asyncio.create_task(sat._connect_once(session))
            try:
                await until(lambda: "ready" in seen, "ready")
                self.assertEqual(await sat.tap(), "start")
                await until(lambda: not recorder.chunks, "the pump to send the audio")
                await asyncio.sleep(0.05)
                self.assertEqual(await sat.tap(), "end")
                await until(lambda: "turn.done" in seen, "turn.done")
                self.assertEqual(player.played, [WAV_REPLY])
                self.assertEqual(sat.face.state, "speaking", "the reply is still playing")
                self.assertEqual(await sat.tap(), "cancel")
                self.assertEqual(player.stopped, 1)
                await asyncio.sleep(0.2)
                self.assertNotIn("error", seen, "the relay must not be sent a stale cancel")
                self.assertEqual(sat.face.state, "idle")
            finally:
                conn.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await conn


if __name__ == "__main__":
    unittest.main()
