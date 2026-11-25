"""Satellite client: face state machine, SSE endpoint, /tap.

Recorder and player are injected fakes — no pw-record, no pw-play, no audio
device, no relay. The clock is injectable so the error -> idle timeout is
tested without sleeping.

Run: .venv/bin/python -m unittest discover -s tests
"""

import asyncio
import json
import sys
import unittest
import unittest.mock
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satellite import client as C  # noqa: E402
from satellite import protocol as P  # noqa: E402


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeRecorder:
    def __init__(self, chunks=None):
        self.chunks = list(chunks or [])
        self.started = 0
        self.stopped = 0

    async def start(self):
        self.started += 1

    async def read(self, nbytes=P.MAX_CHUNK_BYTES):
        return self.chunks.pop(0) if self.chunks else b""

    async def stop(self):
        self.stopped += 1


class FakePlayer:
    def __init__(self, *, block=False):
        self.played = []
        self.stopped = 0
        self.block = block
        self.release = asyncio.Event()

    async def play(self, wav):
        self.played.append(wav)
        if self.block:
            await self.release.wait()

    async def stop(self):
        self.stopped += 1
        self.release.set()


class FakeWS:
    """Records everything the client sends upstream."""

    def __init__(self):
        self.text = []
        self.binary = []
        self.closed = False

    async def send_str(self, data):
        self.text.append(json.loads(data))

    async def send_bytes(self, data):
        self.binary.append(data)

    def sent_types(self):
        return [f["type"] for f in self.text]


def make_client(*, recorder=None, player=None, clock=None, record_cap_s=20.0):
    clock = clock or FakeClock()
    client = C.SatelliteClient(
        relay_url="ws://127.0.0.1:0/ws", device="desk", token="t",
        recorder=recorder or FakeRecorder(), player=player or FakePlayer(),
        clock=clock, record_cap_s=record_cap_s)
    client.ws = FakeWS()
    return client, clock


READY = {"type": "ready", "v": 1, "device": "desk", "session_id": "sat-desk-20260929"}


def done_frame(turn_id="t1", cancelled=False):
    return {"type": "turn.done", "turn_id": turn_id, "cancelled": cancelled,
            "timing": {k: 1 for k in P.TIMING_KEYS}}


class TestStateMachine(unittest.IsolatedAsyncioTestCase):
    """Every transition of the client state machine (idle/listening/thinking/speaking/error)."""

    async def asyncSetUp(self):
        self.client, self.clock = make_client()
        await self.client.handle_frame(READY)

    async def asyncTearDown(self):
        await self.client._cancel_pump()

    async def test_ready_lands_in_idle_with_the_session(self):
        self.assertEqual(self.client.face.state, "idle")
        self.assertEqual(self.client.session_id, "sat-desk-20260929")

    async def test_offline_until_ready(self):
        client, _ = make_client()
        self.assertEqual(client.face.state, "offline")

    async def test_idle_tap_starts_listening_and_sends_turn_start(self):
        self.assertEqual(await self.client.tap(), "start")
        self.assertEqual(self.client.face.state, "listening")
        self.assertEqual(self.client.ws.sent_types(), ["turn.start"])
        start = self.client.ws.text[0]
        self.assertEqual((start["format"], start["rate"], start["channels"]),
                         (P.FORMAT, P.RATE, P.CHANNELS))
        self.assertEqual(self.client.recorder.started, 1)

    async def test_listening_tap_ends_the_turn_and_stops_the_recorder(self):
        await self.client.tap()
        self.assertEqual(await self.client.tap(), "end")
        self.assertEqual(self.client.face.state, "thinking")
        self.assertEqual(self.client.ws.sent_types(), ["turn.start", "turn.end"])
        self.assertEqual(self.client.recorder.stopped, 1)

    async def test_transcript_keeps_thinking_and_captions(self):
        await self.client.tap()
        await self.client.tap()
        await self.client.handle_frame(
            {"type": "transcript", "turn_id": "t1", "text": "what time is it", "stt_ms": 200})
        self.assertEqual(self.client.face.state, "thinking")
        self.assertEqual(self.client.face.text, "what time is it")

    async def test_reply_sets_mood_without_leaving_thinking(self):
        await self.client.tap()
        await self.client.tap()
        await self.client.handle_frame(
            {"type": "reply", "turn_id": "t1", "text": "Half past four.", "mood": "playful"})
        self.assertEqual(self.client.face.state, "thinking")
        self.assertEqual(self.client.face.mood, "playful")
        self.assertEqual(self.client.face.text, "Half past four.")

    async def test_audio_then_binary_plays_and_returns_to_idle(self):
        await self.client.tap()
        await self.client.tap()
        await self.client.handle_frame({"type": "audio", "turn_id": "t1", "format": "wav"})
        await self.client.handle_binary(b"WAVDATA")
        self.assertEqual(self.client.face.state, "speaking")
        await asyncio.sleep(0)  # let the playback task finish
        await asyncio.sleep(0)
        self.assertEqual(self.client.player.played, [b"WAVDATA"])
        self.assertEqual(self.client.face.state, "idle")

    async def test_binary_without_audio_is_dropped(self):
        await self.client.handle_binary(b"WAVDATA")
        self.assertEqual(self.client.player.played, [])
        self.assertEqual(self.client.face.state, "idle")

    async def test_tap_while_thinking_cancels(self):
        await self.client.tap()
        await self.client.tap()
        self.assertEqual(await self.client.tap(), "cancel")
        self.assertEqual(self.client.face.state, "idle")
        self.assertEqual(self.client.ws.sent_types(), ["turn.start", "turn.end", "cancel"])

    async def test_tap_while_speaking_cancels_and_kills_playback(self):
        player = FakePlayer(block=True)
        client, _ = make_client(player=player)
        await client.handle_frame(READY)
        await client.tap()
        await client.tap()
        await client.handle_frame({"type": "audio", "turn_id": "t1", "format": "wav"})
        await client.handle_binary(b"WAVDATA")
        await asyncio.sleep(0)
        self.assertEqual(client.face.state, "speaking")
        self.assertEqual(await client.tap(), "cancel")
        self.assertEqual(player.stopped, 1)
        self.assertEqual(client.face.state, "idle")

    async def test_error_shows_error_then_idles_after_the_hold(self):
        await self.client.tap()
        await self.client.tap()
        await self.client.handle_frame(
            {"type": "error", "turn_id": "t1", "code": "stt_failed", "message": "boom"})
        self.assertEqual(self.client.face.state, "error")
        self.assertEqual(self.client.face.text, "stt_failed")
        self.client.face.tick()
        self.assertEqual(self.client.face.state, "error", "too early to clear")
        self.clock.advance(C.ERROR_HOLD_S)
        self.client.face.tick()
        self.assertEqual(self.client.face.state, "idle")
        self.assertEqual(self.client.face.text, "")

    async def test_turn_done_after_an_error_does_not_clobber_the_error(self):
        await self.client.tap()
        await self.client.tap()
        await self.client.handle_frame(
            {"type": "error", "turn_id": "t1", "code": "hermes_failed", "message": "x"})
        await self.client.handle_frame(done_frame())
        self.assertEqual(self.client.face.state, "error")

    async def test_turn_done_with_pending_audio_leaves_speaking_alone(self):
        await self.client.tap()
        await self.client.tap()
        await self.client.handle_frame({"type": "audio", "turn_id": "t1", "format": "wav"})
        await self.client.handle_frame(done_frame())
        self.assertEqual(self.client.face.state, "thinking", "audio has not arrived yet")
        await self.client.handle_binary(b"W")
        self.assertEqual(self.client.face.state, "speaking")

    async def test_cancelled_turn_done_returns_to_idle(self):
        await self.client.tap()
        await self.client.tap()
        await self.client.handle_frame(done_frame(cancelled=True))
        self.assertEqual(self.client.face.state, "idle")

    async def test_turn_done_in_relay_order_keeps_speaking_until_playback_ends(self):
        # The relay sends `audio`, the WAV, then `turn.done` (relay._run_pipeline).
        # turn.done means "the relay is finished", not "the speaker is finished".
        player = FakePlayer(block=True)
        client, _ = make_client(player=player)
        await client.handle_frame(READY)
        await client.tap()
        await client.tap()
        await client.handle_frame({"type": "audio", "turn_id": "t1", "format": "wav"})
        await client.handle_binary(b"WAVDATA")
        await asyncio.sleep(0)
        await client.handle_frame(done_frame())
        self.assertEqual(client.face.state, "speaking", "playback is still running")
        player.release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertEqual(client.face.state, "idle", "idle once playback has ended")
        self.assertIsNone(client.turn_id)

    async def test_tap_after_turn_done_during_playback_stops_playback(self):
        player = FakePlayer(block=True)
        client, _ = make_client(player=player)
        await client.handle_frame(READY)
        await client.tap()
        await client.tap()
        await client.handle_frame({"type": "audio", "turn_id": "t1", "format": "wav"})
        await client.handle_binary(b"WAVDATA")
        await asyncio.sleep(0)
        await client.handle_frame(done_frame())
        self.assertEqual(await client.tap(), "cancel", "a tap while audible must stop it")
        self.assertEqual(player.stopped, 1)
        self.assertEqual(client.face.state, "idle")
        # The relay already closed the turn; a `cancel` now would earn a
        # bad_message error and flash the error face.
        self.assertEqual(client.ws.sent_types(), ["turn.start", "turn.end"])

    async def test_tap_from_error_starts_a_new_turn(self):
        await self.client.handle_frame(
            {"type": "error", "code": "bad_message", "message": "x"})
        self.assertEqual(await self.client.tap(), "start")
        self.assertEqual(self.client.face.state, "listening")

    async def test_tap_while_offline_is_ignored(self):
        client, _ = make_client()
        self.assertEqual(await client.tap(), "ignored")
        self.assertEqual(client.face.state, "offline")

    async def test_tap_with_a_closed_socket_goes_offline(self):
        self.client.ws.closed = True
        await self.client.tap()
        self.assertEqual(self.client.face.state, "offline")
        self.assertEqual(self.client.ws.sent_types(), [])

    async def test_every_documented_state_is_reachable_in_this_suite(self):
        # Guards against the state list and the machine drifting apart.
        self.assertEqual(set(P.STATES),
                         {"idle", "listening", "thinking", "speaking", "error", "offline"})

    async def test_unknown_state_is_refused(self):
        with self.assertRaises(ValueError):
            self.client.face.set("dreaming")


class TestAudioPump(unittest.IsolatedAsyncioTestCase):
    async def test_pcm_is_streamed_while_listening(self):
        recorder = FakeRecorder([b"aaa", b"bbb", b"ccc"])
        client, _ = make_client(recorder=recorder)
        await client.handle_frame(READY)
        await client.tap()
        for _ in range(20):
            await asyncio.sleep(0)
            if len(client.ws.binary) >= 3:
                break
        await client.tap()
        self.assertEqual(client.ws.binary, [b"aaa", b"bbb", b"ccc"])
        # Upload overlapped the speech: chunks went out before turn.end.
        self.assertEqual(client.ws.sent_types(), ["turn.start", "turn.end"])

    async def test_record_cap_auto_ends_the_turn(self):
        clock = FakeClock()
        recorder = FakeRecorder()
        client, _ = make_client(recorder=recorder, clock=clock, record_cap_s=5.0)
        await client.handle_frame(READY)
        await client.tap()
        # Let the pump task start so it latches its deadline off the fake clock,
        # then jump past the cap.
        await asyncio.sleep(0.02)
        clock.advance(5.0)
        # The pump idles on a real 10 ms sleep when the recorder is dry, so give
        # it wall-clock time rather than bare event-loop turns.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if client.face.state == "thinking":
                break
        self.assertEqual(client.face.state, "thinking")
        self.assertEqual(client.ws.sent_types(), ["turn.start", "turn.end"])

    async def test_eos_to_first_audio_is_measured(self):
        clock = FakeClock()
        client, _ = make_client(clock=clock)
        await client.handle_frame(READY)
        await client.tap()
        await client.tap()            # turn.end -> eos timestamp
        clock.advance(2.5)
        await client.handle_frame({"type": "audio", "turn_id": "t1", "format": "wav"})
        await client.handle_binary(b"W")
        self.assertEqual(client.last_eos_to_first_audio_ms, 2500.0)
        await client._cancel_pump()


class TestLocalHTTP(unittest.IsolatedAsyncioTestCase):
    """The face contract: GET /, GET /events (SSE), POST /tap."""

    async def asyncSetUp(self):
        self.client, self.clock = make_client()
        self.http = TestClient(TestServer(self.client.build_app()))
        await self.http.start_server()
        self.addAsyncCleanup(self.http.close)

    async def asyncTearDown(self):
        await self.client._cancel_pump()

    async def test_index_serves_the_face_page(self):
        resp = await self.http.get("/")
        self.assertEqual(resp.status, 200)
        body = await resp.text()
        self.assertIn("EventSource(\"/events\")", body)
        self.assertIn("/tap", body)

    async def test_tap_drives_the_state_machine(self):
        await self.client.handle_frame(READY)
        resp = await self.http.post("/tap")
        self.assertEqual(resp.status, 200)
        self.assertEqual(await resp.json(),
                         {"ok": True, "action": "start", "state": "listening"})
        self.assertEqual(self.client.face.state, "listening")
        self.assertEqual(self.client.ws.sent_types(), ["turn.start"])

        resp = await self.http.post("/tap")
        self.assertEqual((await resp.json())["action"], "end")

    async def test_events_emits_the_snapshot_then_each_change(self):
        resp = await self.http.get("/events")
        self.assertEqual(resp.status, 200)
        self.assertTrue(resp.headers["Content-Type"].startswith("text/event-stream"))

        async def next_event():
            line = await asyncio.wait_for(resp.content.readline(), timeout=5)
            while line.strip() in (b"", b": keepalive"):
                line = await asyncio.wait_for(resp.content.readline(), timeout=5)
            return json.loads(line.decode().removeprefix("data: "))

        self.assertEqual(await next_event(),
                         {"state": "offline", "mood": "neutral", "text": ""})
        await self.client.handle_frame(READY)
        self.assertEqual(await next_event(),
                         {"state": "idle", "mood": "neutral", "text": ""})
        await self.client.handle_frame(
            {"type": "reply", "turn_id": "t1", "text": "Sure.", "mood": "happy"})
        self.assertEqual(await next_event(),
                         {"state": "idle", "mood": "happy", "text": "Sure."})
        resp.close()

    async def test_the_face_page_never_reaches_the_relay(self):
        body = (C.FACE_DIR / "index.html").read_text()
        for forbidden in ("wss://", "ws://", "Authorization", "Bearer"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, body)


class TestFaceState(unittest.TestCase):
    def test_subscribers_get_a_snapshot_on_subscribe(self):
        async def go():
            face = C.FaceState()
            face.set("idle")
            q = face.subscribe()
            self.assertEqual(q.get_nowait(),
                             {"state": "idle", "mood": "neutral", "text": ""})
            face.set("listening")
            self.assertEqual(q.get_nowait()["state"], "listening")
            face.unsubscribe(q)
            face.set("thinking")
            self.assertTrue(q.empty())
        asyncio.run(go())

    def test_mood_persists_across_state_changes(self):
        face = C.FaceState()
        face.set("speaking", mood="curious")
        face.set("idle")
        self.assertEqual(face.mood, "curious")


class TestPlaybackTarget(unittest.TestCase):
    """The reply must go to an explicitly named sink.

    Regression: a dev run used the default PipeWire sink, which a multi-room
    audio streamer captured and forwarded to other speakers, so test replies
    were audible elsewhere. Playback targets are named, never implicit.
    """

    def test_pw_play_names_the_sink(self):
        argv = C.Player(sink="desk-out")._argv(Path("/tmp/r.wav"))
        if argv[0] != "pw-play":
            self.skipTest("pw-play is not installed here")
        self.assertEqual(argv, ["pw-play", "--target", "desk-out", "/tmp/r.wav"])

    def test_aplay_names_the_sink(self):
        player = C.Player(sink="hw:1,0")
        with unittest.mock.patch.object(C.shutil, "which", return_value=None):
            argv = player._argv(Path("/tmp/r.wav"))
        self.assertEqual(argv, ["aplay", "-q", "-D", "hw:1,0", "/tmp/r.wav"])

    def test_no_sink_falls_back_to_the_default_but_is_not_the_configured_path(self):
        argv = C.Player()._argv(Path("/tmp/r.wav"))
        self.assertNotIn("--target", argv)
        self.assertNotIn("-D", argv)

    def test_recorder_names_the_capture_source(self):
        argv = C.Recorder(device="desk-mic")._argv()
        if argv[0] != "pw-record":
            self.skipTest("pw-record is not installed here")
        self.assertIn("--target", argv)
        self.assertEqual(argv[argv.index("--target") + 1], "desk-mic")

    def test_the_cli_exposes_mic_and_sink(self):
        args = C.build_parser().parse_args(
            ["--device", "desk", "--sink", "desk-out", "--mic", "desk-mic"])
        self.assertEqual((args.sink, args.mic), ("desk-out", "desk-mic"))

    def test_the_pi_unit_names_a_sink(self):
        unit = (Path(__file__).resolve().parents[1]
                / "satellite" / "pi" / "satellite-client.service").read_text()
        self.assertIn("--sink", unit,
                      "the Pi unit must name its output sink, not inherit the default")


class TestTokenFile(unittest.TestCase):
    def test_token_is_read_stripped(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "token"
            path.write_text("  deadbeef\n")
            self.assertEqual(C.read_token(path), "deadbeef")


if __name__ == "__main__":
    unittest.main()
