#!/usr/bin/env python3
"""GPU voice daemon — local STT/TTS HTTP service for Hermes.

Warm daemon: loads a CUDA STT engine (faster-whisper) and a CUDA TTS engine
(Chatterbox) once, keeps them resident, and serves them over loopback HTTP.
Hermes reaches it through its own `type: command` provider mechanism via the
`hermes-stt` / `hermes-tts` curl shims in `bin/`.

Endpoints (127.0.0.1 only):
    GET  /health           -> {"ok", "engines": {...}, "vram_mb": {...}, "timing_ms": {...}}
    GET  /voices           -> {"ok", "engine", "default", "voices": [...]}
    POST /load[?engine=]   -> force load + warm-up (Hermes warm_command target)
    POST /release[?engine=]-> drop an engine, free VRAM
    POST /stt?language=    -> audio body (raw or multipart) in,
                              {"text", "language", "audio_s", "timing_ms": {decode, model, encode, total}} out;
                              language omitted/auto/none = detect (per utterance)
    POST /tts              -> {"text","voice","format"} in, audio out (ogg|wav);
                              text is normalised for speech (markdown, symbols,
                              units) and paragraph breaks become silence;
                              unknown voice -> 400;
                              timing in the X-Timing-Ms header (JSON), audio length
                              in X-Audio-Seconds.
    POST /tts/long         -> {"text"|"chapters","voice","format"} in, 202 + job id
                              out. Long-form reading: chapters are synthesised in
                              the background, one audio file per chapter under
                              $XDG_CACHE_HOME/voice-bot/jobs, TTL-cleaned.
    GET  /jobs             -> job list (newest first)
    GET  /jobs/<id>        -> one job: status, per-chapter progress, file paths
    POST /jobs/<id>/cancel -> stop a queued/running job; finished chapters stay
    DELETE /jobs/<id>      -> cancel and delete the job's audio

Every /stt and /tts writes one JSON line to the log (logger `voice-daemon.req`)
so latency can be graphed without parsing prose.

Invariants (do not break these):
 1. Loopback only. Refuse any other bind. No auth because no exposure.
 2. The daemon never knows about chat platforms, users, or LLMs. Input is
    audio/text; output is text/audio. If you need a chat or user id, stop.
 3. One lock per engine; engines never share a lock; model calls only inside
    `asyncio.to_thread`.
 4. Every audio in/out goes through ffmpeg; the daemon never parses containers
    and the engines never see a file — they get PCM arrays.
 5. Engine choice is env config; adding an engine is one new file under
    `engines/` implementing the Protocol plus a line in the registry.
 6. No Hermes internals are imported here, ever. The contract is the documented
    command-provider placeholders only.

This daemon has no chat surface of any kind. Hermes owns the messaging
platforms; this is only a provider.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from aiohttp import web

from engines import VoiceParams, load_stt_engine, load_tts_engine
from jobs import Job, JobCancelled, JobStore, normalize_chapters, split_chapters
from normalize import prepare_spoken_text, split_paragraphs

# ─── Config ───────────────────────────────────────────────────────
STT_ENGINE = os.environ.get("STT_ENGINE", "whisper_ct2")
TTS_ENGINE = os.environ.get("TTS_ENGINE", "chatterbox")
# both | stt | tts : which engines are loaded at startup / by a bare /load.
RESIDENT = os.environ.get("VOICE_RESIDENT", "both").lower()
HTTP_HOST = os.environ.get("VOICE_HOST", "127.0.0.1")
HTTP_PORT = int(os.environ.get("VOICE_PORT", "9877"))
OPUS_BITRATE = os.environ.get("VOICE_OPUS_BITRATE", "32k")
MAX_UPLOAD_BYTES = int(os.environ.get("VOICE_MAX_UPLOAD_BYTES", str(256 * 1024 * 1024)))
# Hard cap on text per /tts request; the per-call chunk size is the engine's.
MAX_TTS_TEXT_CHARS = int(os.environ.get("VOICE_TTS_MAX_TEXT_CHARS", "4000"))
# Cap on the FIRST chunk only: bounds time-to-first-audio on long replies.
TTS_FIRST_CHUNK_CHARS = int(os.environ.get("VOICE_TTS_FIRST_CHUNK_CHARS", "80"))
WARMUP = os.environ.get("VOICE_WARMUP", "1").lower() not in ("0", "false", "no", "off")
# Phase 3b: markdown/symbol/unit normalisation before the engine sees the text,
# and real silence (not a spoken "pause") between paragraphs.
NORMALIZE = os.environ.get("VOICE_NORMALIZE", "1").lower() not in ("0", "false", "no", "off")
PARAGRAPH_PAUSE_S = float(os.environ.get("VOICE_PARAGRAPH_PAUSE_S", "0.45"))
LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")
AUDIO_FORMATS = {"ogg": "audio/ogg", "wav": "audio/wav"}

# ─── Long-form jobs (phase 3a) ────────────────────────────────────
# Job audio is a cache, not a library: TTL-swept, lost on restart, no index.
_CACHE_HOME = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
JOBS_DIR = Path(os.environ.get("VOICE_JOBS_DIR") or (_CACHE_HOME / "voice-bot" / "jobs"))
JOBS_TTL_S = float(os.environ.get("VOICE_JOBS_TTL_S", str(24 * 3600)))
JOBS_MAX = int(os.environ.get("VOICE_JOBS_MAX", "50"))
JOBS_SWEEP_S = float(os.environ.get("VOICE_JOBS_SWEEP_S", "900"))
# A whole book, not a chat reply: this cap is per job, not per chapter.
MAX_LONG_TEXT_CHARS = int(os.environ.get("VOICE_LONG_MAX_TEXT_CHARS", str(2_000_000)))

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger("voice-daemon")
req_log = logging.getLogger("voice-daemon.req")

# ─── Engines (loaded once, kept on GPU) ───────────────────────────
_engines: Dict[str, Any] = {"stt": None, "tts": None}
# STT and TTS are separate models and may run concurrently; concurrent calls
# into the SAME model would race on CUDA state, so one lock each.
_locks = {"stt": asyncio.Lock(), "tts": asyncio.Lock()}
# Last-seen timings, surfaced on /health so a curl tells you where time goes.
_timing: Dict[str, Optional[Dict[str, Any]]] = {
    "load_stt": None, "load_tts": None, "warmup_stt": None, "warmup_tts": None,
    "last_stt": None, "last_tts": None,
}
# Long-form jobs. One worker task at a time (the TTS lock is per chunk, so a
# long job yields the GPU between chapters and a chat /tts is never starved).
_jobs = JobStore(JOBS_DIR, ttl_s=JOBS_TTL_S, max_jobs=JOBS_MAX)


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def ensure_engine(kind: str):
    """Load (and warm) the engine for *kind* ('stt'|'tts') if not resident. Blocking."""
    if kind not in _engines:
        raise ValueError(f"unknown engine kind {kind!r}")
    if _engines[kind] is None:
        t0 = time.perf_counter()
        engine = load_stt_engine(STT_ENGINE) if kind == "stt" else load_tts_engine(TTS_ENGINE)
        _timing[f"load_{kind}"] = {"total": _ms(t0)}
        _engines[kind] = engine
        if WARMUP:
            t0 = time.perf_counter()
            try:
                engine.warmup()
                warm_ms = _ms(t0)
                _timing[f"warmup_{kind}"] = {"total": warm_ms}
                log.info("%s engine warmed up in %.0f ms", kind, warm_ms)
            except Exception:  # noqa: BLE001 — a failed warm-up is not a failed load
                log.exception("%s warm-up failed; first request will be cold", kind)
    return _engines[kind]


def release_engine(kind: str) -> bool:
    """Drop the engine for *kind* and free its VRAM. True if something was freed."""
    engine = _engines.get(kind)
    if engine is None:
        return False
    try:
        engine.unload()
    except Exception:  # noqa: BLE001 — best effort
        log.exception("unload of %s engine failed", kind)
    _engines[kind] = None
    _empty_cuda_cache()
    log.info("released %s engine", kind)
    return True


def _empty_cuda_cache() -> None:
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _torch():
    """torch if it is already imported/importable, else None (never a hard dep here)."""
    try:
        import torch

        return torch
    except Exception:  # noqa: BLE001
        return None


def _kinds_for(resident: str) -> list[str]:
    return ["stt", "tts"] if resident == "both" else [resident] if resident in ("stt", "tts") else []


def _vram_mb() -> Dict[str, Optional[int]]:
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return {"free": None, "total": None, "used": None}
    free, total = torch.cuda.mem_get_info()
    return {"free": free // 2**20, "total": total // 2**20, "used": (total - free) // 2**20}


def _describe(kind: str, configured: str) -> Dict[str, Any]:
    engine = _engines.get(kind)
    if engine is None:
        return {"configured": configured, "loaded": False}
    described = engine.describe() if hasattr(engine, "describe") else {"engine": engine.name}
    return {"configured": configured, "loaded": True, **described}


# ─── Audio codec boundary (ffmpeg only; nobody else touches containers) ────

def decode_to_pcm(audio_path: str, sample_rate: int) -> np.ndarray:
    """Any container/codec ffmpeg knows -> float32 mono PCM at *sample_rate*."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", audio_path,
         "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "pipe:1"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise ValueError(f"ffmpeg could not decode audio: {proc.stderr.decode(errors='replace').strip()[:300]}")
    pcm = np.frombuffer(proc.stdout, dtype=np.float32)
    if pcm.size == 0:
        raise ValueError("audio decoded to zero samples")
    return pcm


def encode_pcm(pcm16: np.ndarray, sample_rate: int, fmt: str, out_path: str) -> None:
    encode_args = (["-c:a", "libopus", "-b:a", OPUS_BITRATE] if fmt == "ogg"
                   else ["-c:a", "pcm_s16le"])
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y",
         "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
         *encode_args, "-f", fmt, out_path],
        input=pcm16.tobytes(), capture_output=True, check=True,
    )


_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str, limit: int, first_limit: Optional[int] = None) -> List[str]:
    """Sentence-boundary chunks, each <= *limit* chars (a monster sentence is cut on a space).

    Chunks are what bound first-audio latency. With *first_limit* the first
    chunk is capped lower than the rest, so the engine's first generate()
    returns fast even for a long reply (Chatterbox is ~linear in chars).
    """
    text = " ".join(text.split())
    if not text:
        return []
    cap = min(limit, first_limit) if first_limit else limit
    if len(text) <= cap:
        return [text]
    out: list[str] = []
    buf = ""
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > cap:  # a single monster sentence
            if buf:
                out.append(buf)
                buf = ""
                cap = limit
            cut = sentence.rfind(" ", 0, cap)
            cut = cut if cut > 0 else cap
            out.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
            cap = limit
        if not buf:
            buf = sentence
        elif len(buf) + 1 + len(sentence) <= cap:
            buf = f"{buf} {sentence}"
        else:
            out.append(buf)
            buf = sentence
            cap = limit
    if buf:
        out.append(buf)
    return [chunk for chunk in out if chunk]


# ─── Spoken script: normalise, then chunk with paragraph pauses (phase 3b) ──

def spoken_script(text: str) -> str:
    """Text as it should be spoken: markdown/symbols/units expanded.

    Falls back to the raw text if normalisation leaves nothing (input that was
    only a code fence or a bare URL) — a request that asked for speech gets
    speech, not a 400.
    """
    if not NORMALIZE:
        return text
    spoken = prepare_spoken_text(text)
    if not spoken.strip():
        log.warning("normalisation emptied a %d-char request; speaking it raw", len(text))
        return text
    return spoken


def silence_pcm(seconds: float, sample_rate: int) -> np.ndarray:
    """int16 silence. A paragraph break is this, never a spoken word."""
    return np.zeros(max(0, int(seconds * sample_rate)), dtype=np.int16)


def plan_speech(text: str, limit: int, first_limit: Optional[int] = None) -> List[Any]:
    """[(kind, value)] over a prepared script: ('text', chunk) | ('pause', seconds).

    Paragraphs (blank-line separated, courtesy of `spoken_script`) are chunked
    independently so a sentence never straddles a paragraph break, and the break
    itself becomes a silence item the synthesiser turns into PCM.
    """
    paragraphs = split_paragraphs(text) or ([text] if text.strip() else [])
    plan: List[Any] = []
    for para in paragraphs:
        # first_limit bounds time-to-first-audio, so it applies to the very
        # first chunk of the whole script, not to every paragraph.
        chunks = split_sentences(para, limit, first_limit if not plan else None)
        if not chunks:
            continue
        if plan and PARAGRAPH_PAUSE_S > 0:
            plan.append(("pause", PARAGRAPH_PAUSE_S))
        plan.extend(("text", chunk) for chunk in chunks)
    return plan


# ─── Work (runs in a thread, never on the event loop) ──────────────

def transcribe(audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
    engine = ensure_engine("stt")
    t0 = time.perf_counter()
    pcm = decode_to_pcm(audio_path, engine.sample_rate)
    decode_ms = _ms(t0)
    audio_s = round(pcm.size / engine.sample_rate, 2)
    t1 = time.perf_counter()
    text, used_language = engine.transcribe(pcm, language=language)
    model_ms = _ms(t1)
    timing = {"decode": decode_ms, "model": model_ms, "encode": 0.0, "total": _ms(t0)}
    return {"text": text, "language": used_language, "audio_s": audio_s, "timing_ms": timing}


def synthesize(text: str, out_path: str, params: VoiceParams, fmt: str) -> Dict[str, Any]:
    engine = ensure_engine("tts")
    t0 = time.perf_counter()
    script = spoken_script(text)
    plan = plan_speech(script, engine.max_chars, TTS_FIRST_CHUNK_CHARS or None)
    pieces: List[np.ndarray] = []
    first_chunk_ms: Optional[float] = None
    spoken_chunks = pauses = 0
    sample_rate = engine.sample_rate
    for kind, value in plan:
        if kind == "pause":
            # Silence PCM between paragraphs; sample_rate is real by now because
            # a pause is only ever emitted after a spoken chunk.
            pieces.append(silence_pcm(value, sample_rate))
            pauses += 1
            continue
        t_piece = time.perf_counter()
        pcm16, sample_rate = engine.synthesize(value, params)
        if first_chunk_ms is None:
            first_chunk_ms = _ms(t_piece)
        spoken_chunks += 1
        if pcm16.size:
            pieces.append(pcm16)
    model_ms = _ms(t0)
    if not any(p.size for p in pieces):
        raise RuntimeError("TTS engine produced no audio")
    pcm16 = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    t1 = time.perf_counter()
    encode_pcm(pcm16, sample_rate, fmt, out_path)
    encode_ms = _ms(t1)
    audio_s = round(pcm16.size / sample_rate, 2)
    total_ms = _ms(t0)
    timing = {"decode": 0.0, "model": model_ms, "encode": encode_ms, "total": total_ms,
              "first_chunk": first_chunk_ms}
    return {"audio_s": audio_s, "chunks": spoken_chunks, "pauses": pauses,
            "chars": len(script), "chars_in": len(text), "timing_ms": timing,
            "rtf": round(total_ms / 1000 / audio_s, 3) if audio_s else None}


def _log_request(kind: str, result: Dict[str, Any], **extra: Any) -> None:
    line = {"kind": kind, **{k: v for k, v in result.items() if k != "text"}, **extra}
    _timing[f"last_{kind}"] = result["timing_ms"]
    req_log.info(json.dumps(line, separators=(",", ":")))


# ─── Long-form jobs: synthesise chapter by chapter, straight to disk ───────

class _ChapterEncoder:
    """One ffmpeg per chapter, fed PCM as it is generated.

    The point of the whole phase: a book's PCM never exists in memory. Each
    chunk goes down ffmpeg's stdin as soon as the engine returns it, so peak
    memory is one chunk regardless of document length. Opened lazily because
    the true sample rate comes back with the first chunk.
    """

    def __init__(self, fmt: str, out_path: Path):
        self.fmt, self.out_path = fmt, out_path
        self.proc: Optional[subprocess.Popen] = None
        self.samples = 0
        self.sample_rate = 0

    def write(self, pcm16: np.ndarray, sample_rate: int) -> None:
        if not pcm16.size:
            return
        proc = self.proc
        if proc is None:
            self.sample_rate = sample_rate
            encode_args = (["-c:a", "libopus", "-b:a", OPUS_BITRATE] if self.fmt == "ogg"
                           else ["-c:a", "pcm_s16le"])
            proc = self.proc = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-nostdin", "-y",
                 "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
                 *encode_args, "-f", self.fmt, str(self.out_path)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        assert proc.stdin is not None and proc.stderr is not None  # PIPE, always
        proc.stdin.write(pcm16.tobytes())
        self.samples += int(pcm16.size)

    def close(self) -> float:
        """Finish the file and return its audio seconds. Raises if ffmpeg failed."""
        proc = self.proc
        if proc is None:
            return 0.0
        assert proc.stdin is not None and proc.stderr is not None
        proc.stdin.close()
        stderr = proc.stderr.read()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace').strip()[:300]}")
        return round(self.samples / self.sample_rate, 2) if self.sample_rate else 0.0

    def abort(self) -> None:
        if self.proc is None:
            return
        self.proc.kill()
        self.proc.wait()
        self.out_path.unlink(missing_ok=True)


async def _synthesize_chapter(job: Job, chapter, params: VoiceParams) -> None:
    """Synthesise one chapter into one file. Cancellation is checked per chunk."""
    engine = await asyncio.to_thread(ensure_engine, "tts")
    plan = plan_speech(spoken_script(chapter.text), engine.max_chars)
    out_path = job.chapter_path(chapter)
    encoder = _ChapterEncoder(job.format, out_path)
    sample_rate = engine.sample_rate
    job.chapter_started(chapter)
    try:
        for kind, value in plan:
            job.raise_if_cancelled()
            if kind == "pause":
                await asyncio.to_thread(encoder.write, silence_pcm(value, sample_rate), sample_rate)
                continue
            # The lock is taken per chunk, not per chapter: a chat-reply /tts can
            # slot in between chunks instead of waiting out a whole book.
            async with _locks["tts"]:
                pcm16, sample_rate = await asyncio.to_thread(engine.synthesize, value, params)
            # ffmpeg's stdin is a 64 KB pipe and a chunk is bigger, so the write
            # blocks — off the event loop with it.
            await asyncio.to_thread(encoder.write, pcm16, sample_rate)
            del pcm16
            job.chapter_progress(len(value))
        job.raise_if_cancelled()
    except BaseException:
        encoder.abort()
        raise
    audio_s = await asyncio.to_thread(encoder.close)
    if audio_s <= 0:
        raise RuntimeError(f"chapter {chapter.index} produced no audio")
    job.chapter_done(chapter, out_path, audio_s)


async def run_job(job: Job, params: VoiceParams) -> None:
    """Job worker: chapters in order, each encoded to its own file."""
    t0 = time.perf_counter()
    job.start()
    try:
        for chapter in job.chapters:
            job.raise_if_cancelled()
            await _synthesize_chapter(job, chapter, params)
    except JobCancelled:
        job.finish("cancelled")
        log.info("job %s cancelled after %d/%d chapters", job.id,
                 sum(1 for c in job.chapters if c.status == "done"), len(job.chapters))
    except asyncio.CancelledError:
        job.finish("cancelled")
        raise
    except Exception as exc:  # noqa: BLE001 — one bad chapter fails the job, loudly
        log.exception("job %s failed", job.id)
        current = next((c for c in job.chapters if c.status == "running"), None)
        if current is not None:
            job.chapter_failed(current, str(exc))
        job.finish("error", str(exc))
    else:
        job.finish("done")
    req_log.info(json.dumps({
        "kind": "tts_long", "job": job.id, "status": job.status,
        "chapters": len(job.chapters), "chars": job.chars_total, "audio_s": job.audio_s,
        "timing_ms": {"total": _ms(t0)},
        "rtf": round(_ms(t0) / 1000 / job.audio_s, 3) if job.audio_s else None,
    }, separators=(",", ":")))


# ─── HTTP handlers ────────────────────────────────────────────────

async def handle_health(request: web.Request) -> web.Response:
    active = [j for j in _jobs.list() if j.status in ("queued", "running")]
    return web.json_response({
        "ok": True,
        "resident": RESIDENT,
        "engines": {"stt": _describe("stt", STT_ENGINE), "tts": _describe("tts", TTS_ENGINE)},
        "vram_mb": _vram_mb(),
        "timing_ms": dict(_timing),
        "jobs": {"dir": str(JOBS_DIR), "ttl_s": JOBS_TTL_S,
                 "total": len(_jobs.list()), "active": len(active)},
    })


def _requested_kinds(request: web.Request) -> list[str]:
    requested = (request.query.get("engine") or "").lower().strip()
    if not requested:
        return _kinds_for(RESIDENT)
    if requested in ("stt", "tts"):
        return [requested]
    if requested == "both":
        return ["stt", "tts"]
    raise ValueError(f"engine must be stt, tts or both (got {requested!r})")


async def handle_load(request: web.Request) -> web.Response:
    """Force engine load. Hermes' warm_command target."""
    try:
        kinds = _requested_kinds(request)
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    try:
        for kind in kinds:
            async with _locks[kind]:
                await asyncio.to_thread(ensure_engine, kind)
    except Exception as exc:  # noqa: BLE001
        log.exception("engine load failed")
        return web.json_response({"ok": False, "error": str(exc)}, status=500)
    return await handle_health(request)


async def handle_release(request: web.Request) -> web.Response:
    """Drop engines and free VRAM. Exclusive-mode / manual use; not wired by default."""
    try:
        kinds = _requested_kinds(request)
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    released = []
    for kind in kinds:
        async with _locks[kind]:
            if await asyncio.to_thread(release_engine, kind):
                released.append(kind)
    return web.json_response({"ok": True, "released": released, "vram_mb": _vram_mb()})


async def _read_audio_body(request: web.Request, dest: Path) -> int:
    """Write the request's audio payload to *dest* (multipart 'file'/'audio' or raw body)."""
    ctype = (request.headers.get("Content-Type") or "").lower()
    if ctype.startswith("multipart/"):
        reader = await request.multipart()
        while True:
            part = await reader.next()
            if part is None:
                raise ValueError("multipart request had no file part")
            if part.name in ("file", "audio"):
                break
        written = 0
        with dest.open("wb") as fh:
            while True:
                chunk = await part.read_chunk()
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise ValueError(f"audio body exceeds {MAX_UPLOAD_BYTES} bytes")
                fh.write(chunk)
        if written == 0:
            raise ValueError("empty file part")
        return written
    written = 0
    with dest.open("wb") as fh:
        async for chunk in request.content.iter_chunked(65536):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise ValueError(f"audio body exceeds {MAX_UPLOAD_BYTES} bytes")
            fh.write(chunk)
    if written == 0:
        raise ValueError("empty request body")
    return written


async def handle_stt(request: web.Request) -> web.Response:
    """Audio in (raw body or multipart), transcript out."""
    language = request.query.get("language") or None
    if language in ("auto", "none", ""):
        language = None
    t0 = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="voice-stt-") as tmpdir:
            audio_path = Path(tmpdir) / "input.audio"
            nbytes = await _read_audio_body(request, audio_path)
            async with _locks["stt"]:
                result = await asyncio.to_thread(transcribe, str(audio_path), language)
        result["timing_ms"]["request"] = _ms(t0)
        # Log the language decoded, not the one requested: with auto-detect the
        # requested value is always "auto" and tells you nothing after the fact.
        _log_request("stt", result, language=result.get("language") or "unknown",
                     bytes_in=nbytes, chars=len(result["text"]))
        return web.json_response({"ok": True, **result})
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:  # noqa: BLE001
        log.exception("/stt failed")
        return web.json_response({"ok": False, "error": str(exc)}, status=500)


async def _read_tts_request(request: web.Request) -> tuple[str, VoiceParams, str]:
    ctype = (request.headers.get("Content-Type") or "").lower()
    payload: Dict[str, Any] = {}
    if ctype.startswith("application/json"):
        payload = await request.json()
        text = str(payload.get("text") or "").strip()
    else:
        text = (await request.text()).strip()
    if not text:
        raise ValueError("missing text")
    if len(text) > MAX_TTS_TEXT_CHARS:
        raise ValueError(f"text exceeds {MAX_TTS_TEXT_CHARS} chars (got {len(text)})")
    fmt = str(request.query.get("format") or payload.get("format") or "ogg").lower().lstrip(".")
    if fmt == "opus":
        fmt = "ogg"
    if fmt not in AUDIO_FORMATS:
        raise ValueError(f"format must be one of {sorted(AUDIO_FORMATS)} (got {fmt!r})")
    params = VoiceParams(
        voice=request.query.get("voice") or payload.get("voice") or None,
        language=request.query.get("language") or payload.get("language") or None,
    )
    return text, params, fmt


async def handle_tts(request: web.Request) -> web.Response:
    """Text in (JSON {"text": ...} or text/plain body), OGG/Opus (or WAV) out.

    `voice` selects an engine voice (see `GET /voices`); an id the engine does
    not know is a 400, never a silent fallback to the default.
    Timing rides in `X-Timing-Ms` (JSON) because the body is audio.
    """
    t0 = time.perf_counter()
    try:
        text, params, fmt = await _read_tts_request(request)
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:  # noqa: BLE001 — malformed JSON etc.
        return web.json_response({"ok": False, "error": f"bad request: {exc}"}, status=400)
    try:
        with tempfile.TemporaryDirectory(prefix="voice-tts-") as tmpdir:
            out_path = str(Path(tmpdir) / f"out.{fmt}")
            async with _locks["tts"]:
                result = await asyncio.to_thread(synthesize, text, out_path, params, fmt)
            body = Path(out_path).read_bytes()
        result["timing_ms"]["request"] = _ms(t0)
        _log_request("tts", result, format=fmt, bytes_out=len(body))
        return web.Response(body=body, content_type=AUDIO_FORMATS[fmt], headers={
            "X-Timing-Ms": json.dumps(result["timing_ms"], separators=(",", ":")),
            "X-Audio-Seconds": str(result["audio_s"]),
        })
    except ValueError as exc:  # bad voice id and friends: caller error, not ours
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:  # noqa: BLE001
        log.exception("/tts failed")
        return web.json_response({"ok": False, "error": str(exc)}, status=500)


async def handle_voices(request: web.Request) -> web.Response:
    """Voice ids the loaded TTS engine accepts on /tts?voice=.

    Loads the engine if it is not resident yet; an engine with no named voices
    (Chatterbox) reports an empty list, which is not an error.
    """
    try:
        async with _locks["tts"]:
            engine = await asyncio.to_thread(ensure_engine, "tts")
    except Exception as exc:  # noqa: BLE001
        log.exception("/voices could not load the TTS engine")
        return web.json_response({"ok": False, "error": str(exc)}, status=500)
    voices = list(engine.voices()) if hasattr(engine, "voices") else []
    return web.json_response({
        "ok": True,
        "engine": engine.name,
        "default": getattr(engine, "voice", None),
        "voices": voices,
    })


async def _read_long_request(request: web.Request):
    """Parse POST /tts/long -> (chapters, params, fmt).

    Either `text` (chaptered on markdown headings, or `split=none` for one
    chapter) or an explicit `chapters` array of strings / `{title, text}`.
    """
    ctype = (request.headers.get("Content-Type") or "").lower()
    payload: Dict[str, Any] = {}
    if ctype.startswith("application/json"):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
    else:
        payload = {"text": await request.text()}

    raw_chapters = payload.get("chapters")
    text = str(payload.get("text") or "")
    if raw_chapters is not None:
        if not isinstance(raw_chapters, list):
            raise ValueError("chapters must be a list")
        chapters = normalize_chapters(raw_chapters)
    else:
        split = str(request.query.get("split") or payload.get("split") or "headings").lower()
        if split not in ("headings", "none"):
            raise ValueError("split must be 'headings' or 'none'")
        chapters = split_chapters(text, split)
    if not chapters:
        raise ValueError("missing text")
    total = sum(len(t) for _, t in chapters)
    if total > MAX_LONG_TEXT_CHARS:
        raise ValueError(f"text exceeds {MAX_LONG_TEXT_CHARS} chars (got {total})")

    fmt = str(request.query.get("format") or payload.get("format") or "ogg").lower().lstrip(".")
    if fmt == "opus":
        fmt = "ogg"
    if fmt not in AUDIO_FORMATS:
        raise ValueError(f"format must be one of {sorted(AUDIO_FORMATS)} (got {fmt!r})")
    params = VoiceParams(
        voice=request.query.get("voice") or payload.get("voice") or None,
        language=request.query.get("language") or payload.get("language") or None,
    )
    return chapters, params, fmt


async def handle_tts_long(request: web.Request) -> web.Response:
    """Start a long-form job. Returns 202 + the job snapshot immediately.

    The synthesis runs as a background task; poll `GET /jobs/<id>` for progress
    and per-chapter file paths. Batch `POST /tts` is untouched and is still the
    right endpoint for a chat reply.
    """
    try:
        chapters, params, fmt = await _read_long_request(request)
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:  # noqa: BLE001 — malformed JSON etc.
        return web.json_response({"ok": False, "error": f"bad request: {exc}"}, status=400)

    # Fail an unknown voice here, at submit time, rather than 40 chapters in.
    try:
        async with _locks["tts"]:
            engine = await asyncio.to_thread(ensure_engine, "tts")
        known = list(engine.voices()) if hasattr(engine, "voices") else []
        if params.voice and known and params.voice not in known:
            raise ValueError(f"unknown voice {params.voice!r}; known: {known}")
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:  # noqa: BLE001
        log.exception("/tts/long could not load the TTS engine")
        return web.json_response({"ok": False, "error": str(exc)}, status=500)

    job = _jobs.create(chapters, voice=params.voice, fmt=fmt)
    task = asyncio.create_task(run_job(job, params))
    request.app["job_tasks"].add(task)
    task.add_done_callback(request.app["job_tasks"].discard)
    log.info("job %s queued: %d chapters, %d chars", job.id, len(job.chapters), job.chars_total)
    return web.json_response({"ok": True, "job": job.snapshot()}, status=202)


async def handle_jobs(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "jobs": [j.snapshot() for j in _jobs.list()]})


def _job_or_404(request: web.Request) -> Job:
    job = _jobs.get(request.match_info["job_id"])
    if job is None:
        raise web.HTTPNotFound(text=json.dumps({"ok": False, "error": "no such job"}),
                               content_type="application/json")
    return job


async def handle_job(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "job": _job_or_404(request).snapshot()})


async def handle_job_cancel(request: web.Request) -> web.Response:
    """Stop a queued/running job. Chapters already written stay on disk."""
    job = _job_or_404(request)
    cancelled = job.cancel()
    return web.json_response({"ok": True, "cancelled": cancelled, "job": job.snapshot()})


async def handle_job_delete(request: web.Request) -> web.Response:
    """Cancel if needed, then delete the job and its audio."""
    job = _job_or_404(request)
    _jobs.delete(job.id)
    return web.json_response({"ok": True, "deleted": job.id})


async def _sweep_jobs(app: web.Application) -> None:
    """Background TTL sweep of the jobs cache (and of dirs a restart orphaned)."""
    while True:
        try:
            removed = await asyncio.to_thread(_jobs.purge)
            if removed:
                log.info("swept %d expired job(s): %s", len(removed), ", ".join(removed))
        except Exception:  # noqa: BLE001 — a failed sweep must not kill the daemon
            log.exception("job sweep failed")
        await asyncio.sleep(JOBS_SWEEP_S)


async def _on_startup(app: web.Application) -> None:
    app["job_tasks"] = set()
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(_jobs.purge)  # clear what the last run left behind
    app["sweeper"] = asyncio.create_task(_sweep_jobs(app))


async def _on_cleanup(app: web.Application) -> None:
    app["sweeper"].cancel()
    for task in list(app.get("job_tasks") or ()):
        task.cancel()


def build_app() -> web.Application:
    app = web.Application(client_max_size=MAX_UPLOAD_BYTES)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/voices", handle_voices)
    app.router.add_post("/load", handle_load)
    app.router.add_post("/release", handle_release)
    app.router.add_post("/stt", handle_stt)
    app.router.add_post("/tts", handle_tts)
    app.router.add_post("/tts/long", handle_tts_long)
    app.router.add_get("/jobs", handle_jobs)
    app.router.add_get("/jobs/{job_id}", handle_job)
    app.router.add_post("/jobs/{job_id}/cancel", handle_job_cancel)
    app.router.add_delete("/jobs/{job_id}", handle_job_delete)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU voice daemon (STT/TTS provider for Hermes)")
    parser.add_argument("--host", default=HTTP_HOST)
    parser.add_argument("--port", type=int, default=HTTP_PORT)
    parser.add_argument("--lazy", action="store_true",
                        help="do not load engines at startup; load on first request")
    args = parser.parse_args()

    if args.host not in LOOPBACK_HOSTS:
        raise SystemExit(f"refusing to bind non-loopback host {args.host!r}; this daemon is loopback-only")

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise SystemExit("ffmpeg not found or not working — install ffmpeg")

    if not args.lazy:
        for kind in _kinds_for(RESIDENT):
            ensure_engine(kind)

    log.info("voice daemon on http://%s:%d  (GET /health /voices /jobs, "
             "POST /load /release /stt /tts /tts/long)", args.host, args.port)
    web.run_app(build_app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
