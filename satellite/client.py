"""Desk satellite client — the Pi side.

Thin: holds one per-device token, records mic PCM, streams it to the relay over
one WebSocket, plays the reply WAV, and serves the face page a state stream on
loopback. No STT, no TTS, no LLM, no Hermes key.

    python3 -m satellite.client \
        --relay wss://relay.example.invalid/satellite/ws \
        --device desk --token-file ~/.config/satellite/token [--keyboard]

For dev on the relay host, --relay ws://127.0.0.1:9878/ws (the token check still applies).

Imports only stdlib + aiohttp + satellite.protocol, so it installs on Pi OS
with `apt install python3-aiohttp` and nothing else.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import aiohttp
from aiohttp import web

from satellite import protocol as P

log = logging.getLogger("satellite.client")

DEFAULT_RELAY = os.environ.get("SAT_RELAY_URL", "ws://127.0.0.1:9878/ws")
DEFAULT_DEVICE = os.environ.get("SAT_DEVICE", "desk")
DEFAULT_TOKEN_FILE = os.environ.get(
    "SAT_TOKEN_FILE", str(Path.home() / ".config" / "satellite" / "token"))
DEFAULT_UI_PORT = int(os.environ.get("SAT_UI_PORT", "8765"))

ERROR_HOLD_S = 3.0
RECONNECT_MIN_S = 1.0
RECONNECT_MAX_S = 30.0
FACE_DIR = Path(__file__).resolve().parent / "face"


def read_token(path: str | os.PathLike) -> str:
    return Path(path).expanduser().read_text(encoding="utf-8").strip()


# ---- audio I/O -----------------------------------------------------------
#
# Both are injectable: the tests drive the state machine with fakes and never
# spawn a subprocess.

class Recorder:
    """pw-record (PipeWire), falling back to arecord, as raw s16le mono PCM."""

    def __init__(self, *, device: Optional[str] = None) -> None:
        self.device = device
        self._proc: Optional[asyncio.subprocess.Process] = None

    def _argv(self) -> List[str]:
        if shutil.which("pw-record"):
            argv = ["pw-record", "--rate", str(P.RATE), "--channels", str(P.CHANNELS),
                    "--format", "s16", "--"]
            if self.device:
                argv[-1:-1] = ["--target", self.device]
            return argv + ["-"]
        if shutil.which("arecord"):
            argv = ["arecord", "-q", "-t", "raw", "-f", "S16_LE",
                    "-r", str(P.RATE), "-c", str(P.CHANNELS)]
            if self.device:
                argv += ["-D", self.device]
            return argv
        raise RuntimeError("neither pw-record nor arecord is installed")

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *self._argv(), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)

    async def read(self, nbytes: int = P.MAX_CHUNK_BYTES) -> bytes:
        if self._proc is None or self._proc.stdout is None:
            return b""
        return await self._proc.stdout.read(nbytes)

    async def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=2)


class Player:
    """pw-play (falling back to aplay) on a temp WAV in $XDG_RUNTIME_DIR.

    `sink` names the output explicitly. Leaving it None uses the box's default
    sink, which is the wrong default anywhere the default is captured by
    something else (for example a multi-room audio streamer that forwards it
    to other speakers). The Pi passes --sink; dev runs should too.
    """

    def __init__(self, *, sink: Optional[str] = None) -> None:
        self.sink = sink
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._path: Optional[Path] = None

    @staticmethod
    def _runtime_dir() -> Path:
        return Path(os.environ.get("XDG_RUNTIME_DIR") or os.environ.get("TMPDIR") or "/tmp")

    def _argv(self, path: Path) -> List[str]:
        if shutil.which("pw-play"):
            argv = ["pw-play"]
            if self.sink:
                argv += ["--target", self.sink]
            return argv + [str(path)]
        argv = ["aplay", "-q"]
        if self.sink:
            argv += ["-D", self.sink]
        return argv + [str(path)]

    async def play(self, wav: bytes) -> None:
        """Write, play to completion, remove. Returns when playback ends."""
        self._path = self._runtime_dir() / f"satellite-reply-{uuid.uuid4().hex}.wav"
        self._path.write_bytes(wav)
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self._argv(self._path),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            await self._proc.wait()
        finally:
            self._proc = None
            with contextlib.suppress(OSError):
                self._path.unlink()
            self._path = None

    async def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
        if self._path is not None:
            with contextlib.suppress(OSError):
                self._path.unlink()
            self._path = None


# ---- face state ----------------------------------------------------------

class FaceState:
    """The state the face shows. Owned by the client, derived from the socket.

    idle -tap-> listening -tap/cap-> thinking -audio-> speaking -end-> idle
    tap while thinking/speaking -> cancel -> idle;  error -3s-> idle
    socket down -> offline -> (reconnect) idle
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.state = "offline"
        self.mood = P.DEFAULT_MOOD
        self.text = ""
        self.clock = clock
        self._error_at: Optional[float] = None
        self._subscribers: List[asyncio.Queue] = []

    def snapshot(self) -> Dict[str, Any]:
        return {"state": self.state, "mood": self.mood, "text": self.text}

    def set(self, state: str, *, mood: Optional[str] = None, text: Optional[str] = None) -> None:
        if state not in P.STATES:
            raise ValueError(f"unknown face state {state!r}")
        self.state = state
        if mood is not None:
            self.mood = mood
        if text is not None:
            self.text = text
        self._error_at = self.clock() if state == "error" else None
        self._publish()

    def tick(self) -> None:
        """Drop back to idle once an error has been shown for ERROR_HOLD_S."""
        if self.state == "error" and self._error_at is not None:
            if self.clock() - self._error_at >= ERROR_HOLD_S:
                self.set("idle", text="")

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=32)
        q.put_nowait(self.snapshot())
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with contextlib.suppress(ValueError):
            self._subscribers.remove(q)

    def _publish(self) -> None:
        snap = self.snapshot()
        for q in list(self._subscribers):
            try:
                q.put_nowait(snap)
            except asyncio.QueueFull:
                log.warning("face subscriber is not draining; dropping an event")


# ---- the client ----------------------------------------------------------

class SatelliteClient:
    """One socket to the relay, one face state, one turn at a time."""

    def __init__(
        self,
        *,
        relay_url: str,
        device: str,
        token: str,
        recorder: Optional[Recorder] = None,
        player: Optional[Player] = None,
        clock: Callable[[], float] = time.monotonic,
        record_cap_s: float = P.RECORD_CAP_SECONDS,
    ) -> None:
        self.relay_url = relay_url
        self.device = device
        self.token = token
        self.recorder = recorder if recorder is not None else Recorder()
        self.player = player if player is not None else Player()
        self.clock = clock
        self.record_cap_s = record_cap_s
        self.face = FaceState(clock=clock)
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.session_id = ""
        self.turn_id: Optional[str] = None
        self._eos_at: Optional[float] = None
        self._pending_audio = False
        # True from turn.start until the relay's turn.done (or our cancel): only
        # then does the relay still have a turn that a `cancel` frame can stop.
        self._relay_turn_open = False
        self._pump: Optional[asyncio.Task] = None
        self._playback: Optional[asyncio.Task] = None
        self.last_eos_to_first_audio_ms: Optional[float] = None

    # -- tap ---------------------------------------------------------------

    async def tap(self) -> str:
        """One tap. Returns the action taken, for logs and for /tap's response."""
        state = self.face.state
        if state == "idle" or state == "error":
            await self._start_turn()
            return "start"
        if state == "listening":
            await self._end_turn()
            return "end"
        if state in ("thinking", "speaking"):
            await self._cancel_turn()
            return "cancel"
        return "ignored"  # offline

    async def _start_turn(self) -> None:
        if self.ws is None or self.ws.closed:
            self.face.set("offline")
            return
        self.turn_id = str(uuid.uuid4())
        self._eos_at = None
        self._pending_audio = False
        self._relay_turn_open = True
        self.face.set("listening", text="")
        await self.ws.send_str(P.turn_start(self.turn_id))
        await self.recorder.start()
        self._pump = asyncio.create_task(self._pump_audio(self.turn_id))

    async def _pump_audio(self, turn_id: str) -> None:
        """Stream PCM while recording, so the upload overlaps the speech."""
        deadline = self.clock() + self.record_cap_s
        try:
            while self.face.state == "listening" and self.turn_id == turn_id:
                if self.clock() >= deadline:
                    log.info("record cap %.0fs reached; ending turn", self.record_cap_s)
                    # We ARE the pump: clear the handle first so _end_turn does
                    # not cancel this task and deadlock waiting on itself.
                    self._pump = None
                    await self._end_turn()
                    return
                chunk = await self.recorder.read()
                if not chunk:
                    await asyncio.sleep(0.01)
                    continue
                if self.ws is None or self.ws.closed:
                    return
                await self.ws.send_bytes(chunk)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("audio pump died")

    async def _end_turn(self) -> None:
        turn_id = self.turn_id
        if turn_id is None:
            return
        self.face.set("thinking")
        await self.recorder.stop()
        await self._cancel_pump()
        self._eos_at = self.clock()
        if self.ws is not None and not self.ws.closed:
            await self.ws.send_str(P.encode("turn.end", turn_id=turn_id))

    async def _cancel_turn(self) -> None:
        turn_id = self.turn_id
        await self.recorder.stop()
        await self._cancel_pump()
        await self.player.stop()
        if self._playback is not None:
            self._playback.cancel()
            self._playback = None
        if turn_id and self._relay_turn_open and self.ws is not None and not self.ws.closed:
            await self.ws.send_str(P.encode("cancel", turn_id=turn_id))
        self.turn_id = None
        self._relay_turn_open = False
        self.face.set("idle", text="")

    async def _cancel_pump(self) -> None:
        pump, self._pump = self._pump, None
        if pump is not None:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump

    # -- inbound -----------------------------------------------------------

    async def handle_frame(self, frame: Dict[str, Any]) -> None:
        """Apply one relay frame to the state machine."""
        type_ = frame["type"]
        if type_ == "ready":
            self.session_id = frame["session_id"]
            self._relay_turn_open = False  # a fresh socket has no turn on the relay
            self.face.set("idle", text="")
            log.info("ready: device=%s session=%s", frame["device"], self.session_id)
        elif type_ == "transcript":
            # Informational: the face already shows `thinking` from turn.end on.
            self.face.set("thinking", text=frame["text"])
        elif type_ == "reply":
            self.face.set(self.face.state, mood=frame["mood"], text=frame["text"])
        elif type_ == "audio":
            self._pending_audio = True
        elif type_ == "turn.done":
            # The relay is done with the turn; the speaker may not be. While the
            # reply is still playing the turn stays ours (speaking, tap = stop),
            # and _play_then_idle returns to idle when playback ends.
            self._relay_turn_open = False
            if not self._pending_audio and self._playback is None:
                self.turn_id = None
                if self.face.state != "error":
                    self.face.set("idle", text="")
            log.info("turn.done %s", json.dumps(
                {"turn_id": frame["turn_id"], "cancelled": frame["cancelled"],
                 "eos_to_first_audio_ms": self.last_eos_to_first_audio_ms,
                 **frame["timing"]}, separators=(",", ":")))
        elif type_ == "error":
            log.warning("relay error %s: %s", frame["code"], frame.get("message"))
            self._pending_audio = False
            self.face.set("error", text=frame["code"])

    async def handle_binary(self, wav: bytes) -> None:
        """The reply WAV. `audio` always arrives first, per §2."""
        if not self._pending_audio:
            log.warning("binary frame with no preceding `audio`; dropped")
            return
        self._pending_audio = False
        if self._eos_at is not None:
            self.last_eos_to_first_audio_ms = round((self.clock() - self._eos_at) * 1000, 1)
            log.info("eos_to_first_audio_ms=%s", self.last_eos_to_first_audio_ms)
        self.face.set("speaking")
        self._playback = asyncio.create_task(self._play_then_idle(wav))

    async def _play_then_idle(self, wav: bytes) -> None:
        try:
            await self.player.play(wav)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("playback failed")
        finally:
            self._playback = None
            self.turn_id = None
            if self.face.state == "speaking":
                self.face.set("idle", text="")

    # -- socket ------------------------------------------------------------

    async def run(self, session: aiohttp.ClientSession) -> None:
        """Connect, serve, reconnect with 1 -> 30 s backoff, forever."""
        backoff = RECONNECT_MIN_S
        while True:
            try:
                await self._connect_once(session)
                backoff = RECONNECT_MIN_S
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("relay connection failed: %s", exc)
            self.ws = None
            self.face.set("offline")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_MAX_S)

    async def _connect_once(self, session: aiohttp.ClientSession) -> None:
        headers = {"Authorization": f"Bearer {self.token}"}
        async with session.ws_connect(
            self.relay_url, headers=headers, heartbeat=20, max_msg_size=0
        ) as ws:
            self.ws = ws
            await ws.send_str(P.encode("hello", v=P.V, device=self.device, client="satellite-client/0"))
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        frame = P.decode(msg.data, schema=P.RELAY_SCHEMA)
                    except P.ProtocolError as exc:
                        log.warning("bad frame from relay: %s", exc)
                        continue
                    await self.handle_frame(frame)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    await self.handle_binary(msg.data)
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                    break

    # -- local HTTP (face) -------------------------------------------------

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self.handle_index)
        app.router.add_get("/events", self.handle_events)
        app.router.add_post("/tap", self.handle_tap)
        return app

    async def handle_index(self, request: web.Request) -> web.StreamResponse:
        index = FACE_DIR / "index.html"
        if not index.exists():
            return web.Response(text="face/index.html is missing", status=500)
        return web.FileResponse(index)

    async def handle_tap(self, request: web.Request) -> web.Response:
        action = await self.tap()
        return web.json_response({"ok": True, "action": action, "state": self.face.state})

    async def handle_events(self, request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-store",
            "Connection": "keep-alive",
        })
        await resp.prepare(request)
        queue = self.face.subscribe()
        try:
            while True:
                try:
                    snap = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    await resp.write(b": keepalive\n\n")  # keep proxies and Chromium happy
                    continue
                await resp.write(f"data: {json.dumps(snap, separators=(',', ':'))}\n\n".encode())
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self.face.unsubscribe(queue)
        return resp

    async def tick_loop(self, *, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        """Drives the error -> idle timeout without a per-event timer."""
        while True:
            self.face.tick()
            await sleep(0.5)


async def keyboard_loop(client: SatelliteClient) -> None:
    """Dev mode without a touch screen: Enter (or space + Enter) is a tap, q quits."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    print("tap = Enter (start / stop / cancel), q + Enter = quit", flush=True)
    while True:
        line = await reader.readline()
        if not line:
            return
        if line.strip().lower() == b"q":
            return
        action = await client.tap()
        print(f"[{action}] state={client.face.state}", flush=True)


async def amain(args: argparse.Namespace) -> None:
    client = SatelliteClient(
        relay_url=args.relay, device=args.device, token=read_token(args.token_file),
        recorder=Recorder(device=args.mic), player=Player(sink=args.sink))
    log.info("audio: mic=%s sink=%s", args.mic or "(default)", args.sink or "(default)")
    app = client.build_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", args.ui_port)
    await site.start()
    log.info("face UI on http://127.0.0.1:%d (GET /, /events, POST /tap)", args.ui_port)

    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(client.run(session)),
                 asyncio.create_task(client.tick_loop())]
        if args.keyboard:
            tasks.append(asyncio.create_task(keyboard_loop(client)))
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                with contextlib.suppress(asyncio.CancelledError):
                    task.result()
        finally:
            await runner.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Desk satellite client (Pi side)")
    parser.add_argument("--relay", default=DEFAULT_RELAY)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--token-file", default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--ui-port", type=int, default=DEFAULT_UI_PORT)
    parser.add_argument("--mic", default=os.environ.get("SAT_MIC") or None,
                        metavar="SOURCE", help="capture source (pw-record --target / arecord -D)")
    parser.add_argument("--sink", default=os.environ.get("SAT_SINK") or None,
                        metavar="SINK",
                        help="playback sink (pw-play --target / aplay -D). Name it explicitly: "
                             "the default sink may be captured by something else, e.g. a "
                             "multi-room audio streamer)")
    parser.add_argument("--keyboard", action="store_true", help="tap with Enter, for dev without a touch screen")
    parser.add_argument("--log-level", default=os.environ.get("SAT_LOG_LEVEL", "INFO"))
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(amain(args))


if __name__ == "__main__":
    main()
