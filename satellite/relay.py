"""Satellite relay — the only component that knows both halves.

Binds 127.0.0.1:9878 (loopback only, same refuse-other-bind rule as the voice
daemon) and is reached through a TLS-terminating reverse proxy you put in
front of it, e.g. at wss://relay.example.invalid/satellite/ws. Per-device bearer token on the
WS upgrade.

For one turn it: wraps the buffered PCM in a WAV header, POSTs :9877/stt,
POSTs the transcript to Hermes /v1/runs and polls to a terminal status, parses
and strips the mood tag, POSTs the spoken text to :9877/tts, and pushes the
reply WAV back down the socket. See satellite/protocol.py.

It never imports voice_daemon or Hermes code — HTTP on both sides.

Run:  python -m satellite.relay [--host 127.0.0.1] [--port 9878]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import hashlib
import hmac
import io
import json
import logging
import os
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import aiohttp
from aiohttp import web

from satellite import protocol as P

log = logging.getLogger("satellite.relay")

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")

DEFAULT_PORT = int(os.environ.get("SAT_RELAY_PORT", "9878"))
DEFAULT_HOST = os.environ.get("SAT_RELAY_HOST", "127.0.0.1")
DEFAULT_TOKENS_FILE = os.environ.get(
    "SAT_TOKENS_FILE", str(Path.home() / ".config" / "voice-bot" / "satellite-tokens"))
DEFAULT_VOICE_URL = os.environ.get("VOICE_URL", "http://127.0.0.1:9877")
DEFAULT_HERMES_URL = os.environ.get("HERMES_API_URL", "http://127.0.0.1:8642")
STT_LANGUAGE = os.environ.get("SAT_STT_LANGUAGE", "en")
HERMES_TIMEOUT_S = float(os.environ.get("SAT_HERMES_TIMEOUT_S", "120"))
# The daemon's own ceiling (VOICE_TTS_MAX_TEXT_CHARS); replies are trimmed for
# speech at the last sentence end below it.
TTS_MAX_TEXT_CHARS = int(os.environ.get("VOICE_TTS_MAX_TEXT_CHARS", "4000"))

POLL_INTERVAL_S = 0.5
STOP_GRACE_S = 10.0
WS_HEARTBEAT_S = 20.0

# The shared outbound HTTP session, keyed the way aiohttp 3.9+ wants.
HTTP: web.AppKey[aiohttp.ClientSession] = web.AppKey("http", aiohttp.ClientSession)

VOICE_INSTRUCTIONS = (
    "You are answering through a small desk speaker. Reply in plain spoken "
    "sentences, usually one to three, no markdown, lists, code or URLs. "
    "If the answer needs a screen, say so briefly. "
    "End with one tag on its own line: [[mood:X]] where X is one of "
    "neutral, happy, curious, concerned, playful."
)


class PipelineError(Exception):
    """A turn-scoped failure. `code` is one of P.ERROR_CODES."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ---- auth ----------------------------------------------------------------

def load_tokens(path: str | os.PathLike) -> Dict[str, str]:
    """Read `<device> <sha256hex>` lines. Missing file is an empty map.

    A leading `~` is expanded: systemd's EnvironmentFile= passes it literally.
    """
    tokens: Dict[str, str] = {}
    path = Path(path).expanduser()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        log.warning("tokens file %s does not exist; every upgrade will 401", path)
        return tokens
    for lineno, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2 or len(parts[1]) != 64:
            log.warning("tokens file %s line %d is malformed; skipped", path, lineno)
            continue
        tokens[parts[0]] = parts[1].lower()
    return tokens


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def device_for_token(tokens: Dict[str, str], token: str) -> Optional[str]:
    """Constant-time lookup of the device a plaintext token belongs to."""
    if not token:
        return None
    digest = token_digest(token)
    found = None
    for device, expected in tokens.items():
        # Compare every entry so timing does not leak which device matched.
        if hmac.compare_digest(digest, expected):
            found = device
    return found


def bearer_token(request: web.Request) -> str:
    header = request.headers.get("Authorization") or ""
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return value.strip()


# ---- audio ---------------------------------------------------------------

def pcm_to_wav(pcm: bytes, *, rate: int = P.RATE, channels: int = P.CHANNELS) -> bytes:
    """Wrap raw s16le PCM in a WAV container (stdlib `wave`, no ffmpeg here)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(P.SAMPLE_WIDTH)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buf.getvalue()


def pcm_duration_ms(nbytes: int, *, rate: int = P.RATE, channels: int = P.CHANNELS) -> float:
    frame = P.SAMPLE_WIDTH * channels
    return round(nbytes / frame / rate * 1000.0, 1)


# ---- backend clients -----------------------------------------------------

class VoiceClient:
    """The voice daemon over loopback. Base URL injectable so tests stub it."""

    def __init__(self, base_url: str = DEFAULT_VOICE_URL, *, language: str = STT_LANGUAGE) -> None:
        self.base_url = base_url.rstrip("/")
        self.language = language

    async def stt(self, session: aiohttp.ClientSession, wav: bytes) -> Tuple[str, Dict[str, Any]]:
        url = f"{self.base_url}/stt"
        params = {"language": self.language} if self.language else {}
        try:
            async with session.post(url, params=params, data=wav) as resp:
                body = await resp.read()
                if resp.status >= 300:
                    raise PipelineError("stt_failed", f"/stt HTTP {resp.status}: {body[:200]!r}")
                payload = json.loads(body)
        except PipelineError:
            raise
        except Exception as exc:  # noqa: BLE001 — network / malformed JSON
            raise PipelineError("stt_failed", f"/stt call failed: {exc}") from exc
        if not payload.get("ok"):
            raise PipelineError("stt_failed", str(payload.get("error") or "stt reported ok:false"))
        return str(payload.get("text") or ""), payload

    async def tts(self, session: aiohttp.ClientSession, text: str) -> bytes:
        url = f"{self.base_url}/tts"
        try:
            async with session.post(
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps({"text": text, "format": "wav"}),
            ) as resp:
                body = await resp.read()
                if resp.status >= 300:
                    raise PipelineError("tts_failed", f"/tts HTTP {resp.status}: {body[:200]!r}")
        except PipelineError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PipelineError("tts_failed", f"/tts call failed: {exc}") from exc
        if not body:
            raise PipelineError("tts_failed", "/tts returned an empty body")
        return body


class HermesClient:
    """Hermes api_server /v1/runs. Base URL and key injectable for tests."""

    def __init__(
        self,
        base_url: str = DEFAULT_HERMES_URL,
        api_key: str = "",
        *,
        timeout_s: float = HERMES_TIMEOUT_S,
        poll_interval_s: float = POLL_INTERVAL_S,
        clock=time.monotonic,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self.clock = clock

    def _headers(self, **extra: str) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", **extra}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def start(
        self, session: aiohttp.ClientSession, *, text: str, session_id: str, idempotency_key: str,
        on_run_id=None,
    ) -> str:
        """Admit a run. `on_run_id` fires the moment the id is known, before this
        coroutine returns, so a `cancel` racing the admission can still stop it."""
        payload = {"input": text, "session_id": session_id, "instructions": VOICE_INSTRUCTIONS}
        try:
            async with session.post(
                f"{self.base_url}/v1/runs",
                headers=self._headers(**{"Idempotency-Key": idempotency_key}),
                data=json.dumps(payload),
            ) as resp:
                body = await resp.read()
                if resp.status not in (200, 202):
                    raise PipelineError("hermes_failed", f"/v1/runs HTTP {resp.status}: {body[:200]!r}")
                run = json.loads(body)
        except PipelineError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PipelineError("hermes_failed", f"/v1/runs call failed: {exc}") from exc
        run_id = run.get("run_id")
        if not run_id:
            raise PipelineError("hermes_failed", "/v1/runs returned no run_id")
        run_id = str(run_id)
        if on_run_id is not None:
            on_run_id(run_id)
        return run_id

    async def get(self, session: aiohttp.ClientSession, run_id: str) -> Dict[str, Any]:
        try:
            async with session.get(
                f"{self.base_url}/v1/runs/{run_id}", headers=self._headers()
            ) as resp:
                body = await resp.read()
                if resp.status >= 300:
                    raise PipelineError(
                        "hermes_failed", f"GET /v1/runs/{run_id} HTTP {resp.status}: {body[:200]!r}")
                return json.loads(body)
        except PipelineError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PipelineError("hermes_failed", f"run status poll failed: {exc}") from exc

    async def stop(self, session: aiohttp.ClientSession, run_id: str) -> None:
        try:
            async with session.post(
                f"{self.base_url}/v1/runs/{run_id}/stop", headers=self._headers(), data="{}"
            ) as resp:
                await resp.read()
        except Exception as exc:  # noqa: BLE001 — best effort, never masks the real error
            log.warning("stop of run %s failed: %s", run_id, exc)

    async def deny_approval(self, session: aiohttp.ClientSession, run_id: str) -> None:
        """v0 has no voice approve: deny everything, for the whole run."""
        try:
            async with session.post(
                f"{self.base_url}/v1/runs/{run_id}/approval",
                headers=self._headers(),
                data=json.dumps({"choice": "deny", "all": True}),
            ) as resp:
                await resp.read()
        except Exception as exc:  # noqa: BLE001
            log.warning("approval deny for run %s failed: %s", run_id, exc)

    async def run_to_completion(
        self, session: aiohttp.ClientSession, run_id: str, *, sleep=asyncio.sleep,
        cancelled=None, stop=None,
    ) -> str:
        """Poll to a terminal status and return the reply text.

        `waiting_for_approval` is denied immediately. The timeout stops the run
        and keeps polling briefly for `cancelled` before raising hermes_timeout.
        `cancelled` is a predicate checked each poll: when it turns true the run
        is stopped and asyncio.CancelledError is raised, so a `cancel` frame
        that raced the admission still kills the run. `stop` overrides how the
        run is stopped (the caller dedupes so /stop is not called twice).
        """
        stop_run = stop or (lambda: self.stop(session, run_id))
        deadline = self.clock() + self.timeout_s
        denied = False
        while True:
            if cancelled is not None and cancelled():
                await stop_run()
                raise asyncio.CancelledError()
            status_body = await self.get(session, run_id)
            status = str(status_body.get("status") or "")
            if status in P.TERMINAL_RUN_STATUSES:
                return self._terminal_output(run_id, status, status_body)
            if status == "waiting_for_approval" and not denied:
                denied = True
                await self.deny_approval(session, run_id)
            if self.clock() >= deadline:
                break
            await sleep(self.poll_interval_s)

        await stop_run()
        stop_deadline = self.clock() + STOP_GRACE_S
        while self.clock() < stop_deadline:
            await sleep(self.poll_interval_s)
            status_body = await self.get(session, run_id)
            if str(status_body.get("status") or "") in P.TERMINAL_RUN_STATUSES:
                break
        raise PipelineError("hermes_timeout", f"run {run_id} did not settle in {self.timeout_s:g}s")

    @staticmethod
    def _terminal_output(run_id: str, status: str, body: Dict[str, Any]) -> str:
        if status != "completed":
            reason = body.get("error") or f"run {status}"
            raise PipelineError("hermes_failed", str(reason))
        return str(body.get("output") or "")


# ---- one turn ------------------------------------------------------------

class Turn:
    """Buffered audio for the turn currently in flight on one socket."""

    def __init__(self, turn_id: str, *, rate: int = P.RATE, channels: int = P.CHANNELS) -> None:
        self.turn_id = turn_id
        self.rate = rate
        self.channels = channels
        self.chunks: list[bytes] = []
        self.nbytes = 0
        self.ended = False
        self.cancelled = False
        self.run_id: Optional[str] = None
        # Set by whoever calls /stop first, so the reader and the pipeline do
        # not both stop the same run.
        self.run_stopped = False
        self.started_at = time.perf_counter()

    def append(self, chunk: bytes) -> None:
        """Buffer one PCM chunk. Raises `too_long` past the 30 s cap."""
        if self.nbytes + len(chunk) > P.MAX_AUDIO_BYTES:
            raise PipelineError(
                "too_long", f"audio exceeds {P.MAX_AUDIO_SECONDS}s ({P.MAX_AUDIO_BYTES} bytes)")
        self.chunks.append(chunk)
        self.nbytes += len(chunk)

    @property
    def pcm(self) -> bytes:
        return b"".join(self.chunks)

    @property
    def audio_ms(self) -> float:
        return pcm_duration_ms(self.nbytes, rate=self.rate, channels=self.channels)


def session_id_for(device: str, *, today: Optional[dt.date] = None) -> str:
    """`sat-<device>-<YYYYMMDD>` on the local date — the desk's daily thread."""
    day = today or dt.date.today()
    return f"sat-{device}-{day:%Y%m%d}"


# ---- the socket ----------------------------------------------------------

class Relay:
    """aiohttp app factory plus the per-socket state machine.

    `voice` and `hermes` are injected so tests point them at stub servers on
    ephemeral ports instead of the real daemon and Hermes.
    """

    def __init__(
        self,
        *,
        tokens_file: str | os.PathLike = DEFAULT_TOKENS_FILE,
        voice: Optional[VoiceClient] = None,
        hermes: Optional[HermesClient] = None,
        tokens: Optional[Dict[str, str]] = None,
        heartbeat: Optional[float] = WS_HEARTBEAT_S,
        tts_max_chars: int = TTS_MAX_TEXT_CHARS,
    ) -> None:
        self.tokens_file = tokens_file
        self.tokens = tokens if tokens is not None else load_tokens(tokens_file)
        self.voice = voice or VoiceClient()
        self.hermes = hermes or HermesClient(api_key=os.environ.get("HERMES_API_KEY", ""))
        self.heartbeat = heartbeat
        self.tts_max_chars = tts_max_chars
        # One turn in flight per device, across sockets.
        self._busy: set[str] = set()

    # -- app ---------------------------------------------------------------

    def build_app(self) -> web.Application:
        app = web.Application(client_max_size=P.MAX_CHUNK_BYTES * 4)
        # A reverse proxy mounted at /satellite may or may not strip the
        # prefix depending on version, so both shapes are routed.
        for prefix in ("", "/satellite"):
            app.router.add_get(f"{prefix}/health", self.handle_health)
            app.router.add_get(f"{prefix}/ws", self.handle_ws)
        app.cleanup_ctx.append(self._http_session_ctx)
        return app

    async def _http_session_ctx(self, app: web.Application):
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            app[HTTP] = session
            yield

    # -- health ------------------------------------------------------------

    async def handle_health(self, request: web.Request) -> web.Response:
        """Unauthenticated liveness, plus whether the voice daemon answers."""
        voice_ok = False
        try:
            async with request.app[HTTP].get(
                f"{self.voice.base_url}/health", timeout=aiohttp.ClientTimeout(total=3)
            ) as resp:
                voice_ok = resp.status == 200
        except Exception:  # noqa: BLE001
            voice_ok = False
        return web.json_response({"ok": True, "v": P.V, "voice": voice_ok})

    # -- websocket ---------------------------------------------------------

    async def handle_ws(self, request: web.Request) -> web.StreamResponse:
        device = device_for_token(self.tokens, bearer_token(request))
        if device is None:
            return web.json_response({"error": "unauthorized"}, status=401)

        ws = web.WebSocketResponse(heartbeat=self.heartbeat, max_msg_size=P.MAX_CHUNK_BYTES * 4)
        await ws.prepare(request)
        try:
            await self._serve(ws, request.app[HTTP], device)
        finally:
            self._busy.discard(device)
        return ws

    async def _serve(self, ws: web.WebSocketResponse, http: aiohttp.ClientSession, device: str) -> None:
        session_id = session_id_for(device)
        greeted = False
        turn: Optional[Turn] = None
        # The pipeline runs as a task so this loop keeps reading the socket:
        # `cancel` must be seen WHILE Hermes is thinking, not after.
        pipeline: Optional[asyncio.Task] = None

        def clear() -> None:
            nonlocal turn, pipeline
            turn = None
            pipeline = None
            self._busy.discard(device)

        try:
            async for msg in ws:
                if pipeline is not None and pipeline.done():
                    clear()

                if msg.type == aiohttp.WSMsgType.BINARY:
                    if turn is None or turn.ended:
                        await self._send_error(ws, None, "bad_message", "binary frame outside a turn")
                        continue
                    try:
                        turn.append(msg.data)
                    except PipelineError as exc:
                        await self._fail_turn(
                            ws, turn, exc, timing=P.timing(audio_ms=turn.audio_ms),
                            device=device, session_id=session_id)
                        clear()
                    continue
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    frame = P.decode(msg.data, schema=P.CLIENT_SCHEMA)
                except P.ProtocolError as exc:
                    await self._send_error(ws, None, exc.code, str(exc))
                    continue

                type_ = frame["type"]
                if type_ == "hello":
                    if frame["device"] != device:
                        await ws.close(
                            code=P.CLOSE_DEVICE_MISMATCH, message=b"device does not match token")
                        return
                    greeted = True
                    await self._send(ws, "ready", v=P.V, device=device, session_id=session_id)
                    continue
                if not greeted:
                    await self._send_error(ws, None, "bad_message", "expected hello first")
                    continue

                if type_ == "turn.start":
                    if turn is not None or device in self._busy:
                        await self._send_error(
                            ws, frame["turn_id"], "busy", "a turn is already in flight")
                        continue
                    if (frame["format"] != P.FORMAT or frame["rate"] != P.RATE
                            or frame["channels"] != P.CHANNELS):
                        await self._send_error(
                            ws, frame["turn_id"], "bad_message",
                            f"v1 accepts only {P.FORMAT} {P.RATE}Hz mono")
                        continue
                    self._busy.add(device)
                    turn = Turn(frame["turn_id"], rate=frame["rate"], channels=frame["channels"])
                    continue

                if type_ == "cancel":
                    if turn is None:
                        await self._send_error(ws, frame["turn_id"], "bad_message", "no turn to cancel")
                        continue
                    # Set the flag first: the pipeline's poll loop sees it and
                    # stops the run itself, even if the run_id is not known yet.
                    turn.cancelled = True
                    if pipeline is not None and not pipeline.done():
                        with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                            await asyncio.wait_for(asyncio.shield(pipeline), timeout=5)
                        if not pipeline.done():
                            pipeline.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await pipeline
                    if turn.run_id and not turn.run_stopped:
                        turn.run_stopped = True
                        await self.hermes.stop(http, turn.run_id)
                    await self._send(
                        ws, "turn.done", turn_id=turn.turn_id, cancelled=True,
                        timing=P.timing(audio_ms=turn.audio_ms))
                    self._log_turn(device, turn, session_id, status="cancelled")
                    clear()
                    continue

                if type_ == "turn.end":
                    if turn is None or frame["turn_id"] != turn.turn_id:
                        await self._send_error(ws, frame["turn_id"], "bad_message", "no matching turn")
                        continue
                    turn.ended = True
                    pipeline = asyncio.create_task(
                        self._run_pipeline(ws, http, device, turn, session_id))
                    continue
        finally:
            # Socket closed: drop the buffer, stop any run we started.
            if pipeline is not None and not pipeline.done():
                pipeline.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pipeline
            if turn is not None and turn.run_id and not turn.run_stopped:
                turn.run_stopped = True
                await self.hermes.stop(http, turn.run_id)
            self._busy.discard(device)

    # -- pipeline ----------------------------------------------------------

    async def _run_pipeline(
        self,
        ws: web.WebSocketResponse,
        http: aiohttp.ClientSession,
        device: str,
        turn: Turn,
        session_id: str,
    ) -> None:
        """§2 steps 1–6. Always emits exactly one turn.done, error or not."""
        t0 = time.perf_counter()
        stt_ms = hermes_ms = tts_ms = None
        audio_ms = turn.audio_ms
        try:
            if turn.nbytes == 0:
                raise PipelineError("empty_transcript", "no audio in the turn")

            t = time.perf_counter()
            transcript, _ = await self.voice.stt(http, pcm_to_wav(turn.pcm, rate=turn.rate, channels=turn.channels))
            stt_ms = round((time.perf_counter() - t) * 1000, 1)
            transcript = transcript.strip()
            if not transcript:
                raise PipelineError("empty_transcript", "transcript was empty")
            await self._send(ws, "transcript", turn_id=turn.turn_id, text=transcript, stt_ms=int(stt_ms))

            t = time.perf_counter()

            def record_run_id(run_id: str) -> None:
                turn.run_id = run_id

            await self.hermes.start(
                http, text=transcript, session_id=session_id,
                idempotency_key=f"sat-{device}-{turn.turn_id}", on_run_id=record_run_id)

            async def stop_once() -> None:
                if turn.run_id and not turn.run_stopped:
                    turn.run_stopped = True
                    await self.hermes.stop(http, turn.run_id)

            reply_raw = await self.hermes.run_to_completion(
                http, turn.run_id or "", cancelled=lambda: turn.cancelled, stop=stop_once)
            hermes_ms = round((time.perf_counter() - t) * 1000, 1)

            spoken, mood = P.parse_mood(reply_raw)
            if not spoken:
                raise PipelineError("hermes_failed", "reply was empty after the mood tag was stripped")
            await self._send(ws, "reply", turn_id=turn.turn_id, text=spoken, mood=mood)

            t = time.perf_counter()
            wav = await self.voice.tts(http, P.truncate_for_tts(spoken, self.tts_max_chars))
            tts_ms = round((time.perf_counter() - t) * 1000, 1)
            await self._send(ws, "audio", turn_id=turn.turn_id, format="wav")
            await ws.send_bytes(wav)
        except PipelineError as exc:
            await self._fail_turn(
                ws, turn, exc,
                timing=P.timing(audio_ms=audio_ms, stt_ms=stt_ms, hermes_ms=hermes_ms,
                                tts_ms=tts_ms, total_ms=round((time.perf_counter() - t0) * 1000, 1)),
                device=device, session_id=session_id)
            return
        except asyncio.CancelledError:
            # A `cancel` frame killed this task; the reader sends turn.done
            # {cancelled:true} itself, so say nothing more on the wire.
            raise

        timing = P.timing(
            audio_ms=audio_ms, stt_ms=stt_ms, hermes_ms=hermes_ms, tts_ms=tts_ms,
            total_ms=round((time.perf_counter() - t0) * 1000, 1))
        await self._send(ws, "turn.done", turn_id=turn.turn_id, cancelled=False, timing=timing)
        self._log_turn(device, turn, session_id, status="ok", timing=timing,
                       transcript_chars=len(transcript))

    # -- wire helpers ------------------------------------------------------

    async def _send(self, ws: web.WebSocketResponse, type_: str, **fields: Any) -> None:
        await ws.send_str(P.encode(type_, **fields))

    async def _send_error(
        self, ws: web.WebSocketResponse, turn_id: Optional[str], code: str, message: str
    ) -> None:
        fields: Dict[str, Any] = {"code": code, "message": message}
        if turn_id:
            fields["turn_id"] = turn_id
        await self._send(ws, "error", **fields)

    async def _fail_turn(
        self,
        ws: web.WebSocketResponse,
        turn: Turn,
        exc: PipelineError,
        *,
        timing: Dict[str, Any],
        device: str = "?",
        session_id: str = "?",
    ) -> None:
        """Every turn-scoped error is followed by exactly one turn.done."""
        log.info("turn %s failed: %s %s", turn.turn_id, exc.code, exc.message)
        await self._send_error(ws, turn.turn_id, exc.code, exc.message)
        await self._send(ws, "turn.done", turn_id=turn.turn_id, cancelled=False, timing=timing)
        self._log_turn(device, turn, session_id, status=exc.code, timing=timing)

    def _log_turn(
        self,
        device: str,
        turn: Turn,
        session_id: str,
        *,
        status: str,
        timing: Optional[Dict[str, Any]] = None,
        transcript_chars: int = 0,
    ) -> None:
        """One JSON line per turn. Never audio; transcripts only at DEBUG."""
        timing = timing or P.timing(audio_ms=turn.audio_ms)
        line = {
            "device": device,
            "turn_id": turn.turn_id,
            "session_id": session_id,
            "run_id": turn.run_id,
            "transcript_chars": transcript_chars,
            "status": status,
            **{k: timing.get(k) for k in P.TIMING_KEYS},
        }
        log.info("turn %s", json.dumps(line, separators=(",", ":")))


# ---- entry point ---------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Desk satellite relay (Pi <-> voice daemon + Hermes)")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--tokens-file", default=DEFAULT_TOKENS_FILE)
    parser.add_argument("--voice-url", default=DEFAULT_VOICE_URL)
    parser.add_argument("--hermes-url", default=DEFAULT_HERMES_URL)
    parser.add_argument("--log-level", default=os.environ.get("SAT_LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.host not in LOOPBACK_HOSTS:
        raise SystemExit(
            f"refusing to bind non-loopback host {args.host!r}; the relay is reached "
            "through a reverse proxy, never directly")

    api_key = os.environ.get("HERMES_API_KEY", "")
    if not api_key:
        log.warning("HERMES_API_KEY is empty; /v1/runs will 401 unless the api_server is open")

    relay = Relay(
        tokens_file=args.tokens_file,
        voice=VoiceClient(args.voice_url),
        hermes=HermesClient(args.hermes_url, api_key),
    )
    log.info("satellite relay on http://%s:%d (GET /health, GET /ws), %d device token(s)",
             args.host, args.port, len(relay.tokens))
    web.run_app(relay.build_app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
