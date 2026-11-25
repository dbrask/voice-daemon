"""faster-whisper (CTranslate2) STT engine."""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

log = logging.getLogger("voice-daemon.stt")

SAMPLE_RATE = 16000  # whisper's native input rate

# Padding around each per-language decode run. The VAD marks speech onset a hair
# late, and slicing exactly on that boundary shaves the first consonant off the
# run ("Quick check" decoded as "Put check"). A fifth of a second of context on
# each side fixes it; more starts pulling the neighbouring language back in.
RUN_PAD_S = 0.2

# Below this detection probability a chunk's language is not believed. Whisper
# detects on up to 30 s of audio; a one-word interjection ("ok", "ja") gives it
# almost nothing to go on and it will return the wrong language at p~0.4 — which
# splits one utterance into three decode runs and mislabels the middle one. A
# low-confidence chunk inherits the language of the run before it instead.
DEFAULT_LANG_MIN_PROB = 0.6

# Short, deliberately boring prompt. Whisper conditions its decode on this text,
# so it must read like ordinary prose in the target language: a bare word list
# biases the model towards emitting lists. Keep it 50-80 tokens; a long prompt
# starts steering content, not spelling.
DEFAULT_GAZETTEER = (
    "Reporting from Ukraine and the talks around it: Dobropillia, Sloviansk, "
    "Kramatorsk, Pokrovsk, Svyatohirsk and Orikhiv are place names, and "
    "Witkoff, Ushakov and Zelensky are people."
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def multilingual_enabled() -> bool:
    """Per-utterance language detection, on unless ``WHISPER_MULTILINGUAL=0``.

    The kill switch exists because the extra detect pass is the only thing
    standing between a mixed clip and a plain single-language decode: if
    detection ever starts thrashing on your audio, one env var puts the
    engine back on whole-clip detection without a deploy.
    """
    return _env_bool("WHISPER_MULTILINGUAL", True)


def speech_chunks(pcm: np.ndarray) -> list:
    """Silero VAD speech spans as ``[{"start": int, "end": int}]`` sample offsets.

    Imported lazily for the same reason the model is: ``import
    engines.whisper_ct2`` must not drag faster_whisper in.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    return get_speech_timestamps(pcm, VadOptions())


def lang_min_prob() -> float:
    """Confidence floor for believing a chunk's detected language.

    ``WHISPER_LANG_MIN_PROB`` overrides :data:`DEFAULT_LANG_MIN_PROB`; a garbage
    value falls back to the default rather than failing a transcription. Set it
    to ``0`` to trust every detection (the behaviour before the floor existed).
    """
    raw = os.environ.get("WHISPER_LANG_MIN_PROB")
    if raw is None:
        return DEFAULT_LANG_MIN_PROB
    try:
        return float(raw)
    except ValueError:
        log.warning("ignoring WHISPER_LANG_MIN_PROB=%r (not a number)", raw)
        return DEFAULT_LANG_MIN_PROB


def language_runs(chunks, detect, min_prob=None) -> list:
    """Group consecutive VAD chunks that share a language into decode runs.

    ``detect(start, end)`` returns ``(language, probability)`` for one chunk.
    Returns ``[(start, end, language)]`` covering first..last chunk of each run,
    so the silence *between* two same-language utterances is decoded with them
    (whisper handles internal pauses fine) while a language switch cuts a new run.

    A chunk detected below *min_prob* (default :func:`lang_min_prob`) is not
    allowed to cut a new run: it inherits the previous run's language. The first
    chunk has no predecessor to inherit from, so its detection stands.
    """
    if min_prob is None:
        min_prob = lang_min_prob()
    runs: list = []
    for chunk in chunks:
        start, end = int(chunk["start"]), int(chunk["end"])
        lang, prob = detect(start, end)
        if runs and prob < min_prob:
            lang = runs[-1][2]
        if runs and runs[-1][2] == lang:
            runs[-1] = (runs[-1][0], end, lang)
        else:
            runs.append((start, end, lang))
    return runs


def join_segments(segments) -> str:
    """Join faster-whisper segments into one transcript.

    faster-whisper hands back per-segment ``text`` that usually carries a
    leading space, but not always (the batched pipeline strips some), so a bare
    ``"".join`` glues words together. Strip each segment and join with a single
    space instead: punctuation and case come from the model and are left alone.
    """
    parts = []
    for seg in segments:
        text = getattr(seg, "text", seg)
        if not isinstance(text, str):
            continue
        text = " ".join(text.split())  # collapse internal newlines/runs
        if text:
            parts.append(text)
    return " ".join(parts)


def initial_prompt() -> Optional[str]:
    """The env-gated spelling hint, or ``None`` when disabled.

    ``WHISPER_GAZETTEER=1`` turns on :data:`DEFAULT_GAZETTEER`;
    ``WHISPER_INITIAL_PROMPT`` overrides the text (and implies enabled).
    """
    custom = os.environ.get("WHISPER_INITIAL_PROMPT")
    if custom is not None and custom.strip():
        return custom.strip()
    if _env_bool("WHISPER_GAZETTEER", False):
        return DEFAULT_GAZETTEER
    return None


def _cuda_free_bytes(torch) -> Optional[int]:
    """Free device memory right now, or None when there is no usable CUDA.

    Used either side of a model load to price the load. ``mem_get_info`` reads
    the driver, not torch's allocator, so it sees CTranslate2's allocations —
    which torch never made and cannot report.
    """
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        torch.cuda.synchronize()
        return torch.cuda.mem_get_info()[0]
    except Exception:  # noqa: BLE001 - a missing figure must never fail a load
        return None


def _vram_delta_mb(torch, free_before: Optional[int]) -> Optional[int]:
    """Megabytes consumed since *free_before*, or None if it cannot be measured.

    Clamped at zero: another process freeing memory mid-load would otherwise
    report the model as negatively sized. A negative raw delta still warns —
    something else freed memory, so the single-owner-GPU assumption broke.
    """
    if free_before is None:
        return None
    free_after = _cuda_free_bytes(torch)
    if free_after is None:
        return None
    raw = free_before - free_after
    if free_after > free_before:
        log.warning(
            "vram delta negative (raw=%s bytes); something else freed memory mid-load",
            raw,
        )
    return max(0, raw) // 2**20


class WhisperCT2Engine:
    """CUDA faster-whisper. Default model ``large-v3-turbo``.

    Input is float32 mono PCM at 16 kHz (the daemon decodes with ffmpeg); the
    engine never opens a file, so decode time is measurable separately.

    ``vad_filter=True`` uses the bundled Silero VAD; without it whisper
    hallucinates "Thank you." on trailing silence in voice notes.

    ``condition_on_previous_text=False`` costs nothing on short clips and
    stops one hallucinated segment poisoning the rest of a long one.

    Clips longer than ``WHISPER_BATCH_ABOVE_S`` (default 30) go through
    ``BatchedInferencePipeline``: VAD splits the audio, the chunks decode as
    one batch instead of a sequential 30 s window walk. Below that there is
    only one window, so batching is pure overhead.

    ``WHISPER_GAZETTEER=1`` prepends a short prose ``initial_prompt`` naming the
    proper nouns that whisper otherwise re-spells per segment (see
    :data:`DEFAULT_GAZETTEER`); ``WHISPER_INITIAL_PROMPT`` replaces the text.
    Off by default — a prompt is a decode bias, not a lookup table.

    ``WHISPER_COMPUTE_TYPE`` (default ``int8_float16``) picks the numeric
    precision CTranslate2 loads the weights at. int8 weights with float16
    activations halve the resident footprint against plain ``float16`` at no
    measured quality cost (see ``docs/bench-int8.md``); quantization happens at
    load, so there is one model on disk either way. ``WHISPER_COMPUTE_TYPE=float16``
    is the rollback and needs no deploy.

    No language is assumed. When the caller asks for none, the language is
    detected per VAD utterance so a clip that switches between Swedish and
    English transcribes each part in its own language rather than translating
    one; ``WHISPER_MULTILINGUAL=0`` falls back to one detection per clip. A chunk
    whose detection scores below ``WHISPER_LANG_MIN_PROB`` (default 0.6) inherits
    the previous run's language instead of cutting a new run — short interjections
    detect wrongly and would otherwise fragment an utterance into three decodes.
    """

    name = "whisper_ct2"
    sample_rate = SAMPLE_RATE

    def __init__(
        self,
        model: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ) -> None:
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        self.model_name = model or os.environ.get("WHISPER_MODEL", "large-v3-turbo")
        self.device = device or os.environ.get("WHISPER_DEVICE", "cuda")
        self.compute_type = compute_type or os.environ.get("WHISPER_COMPUTE_TYPE", "int8_float16")
        self.beam_size = int(os.environ.get("WHISPER_BEAM_SIZE", "1"))
        self.condition_on_previous_text = _env_bool("WHISPER_CONDITION_ON_PREVIOUS", False)
        self.batch_above_s = float(os.environ.get("WHISPER_BATCH_ABOVE_S", "30"))
        self.batch_size = int(os.environ.get("WHISPER_BATCH_SIZE", "8"))
        # Read once at load: the daemon is long-lived and a prompt that changes
        # per request would make transcripts irreproducible.
        self.initial_prompt = initial_prompt()
        self.multilingual = multilingual_enabled()
        self.lang_min_prob = lang_min_prob()
        torch = None
        if self.device == "cuda":
            # CTranslate2 dlopens libcublas/libcudnn at first encode. There is no
            # system CUDA toolkit here; the pip-installed torch wheels carry the
            # libraries and importing torch is what puts them on the loader path.
            # Without this, STT works only if something else imported torch first.
            import torch  # noqa: F811
        log.info("loading faster-whisper %r on %s (%s)", self.model_name, self.device, self.compute_type)
        free_before = _cuda_free_bytes(torch)
        self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        self._batched = BatchedInferencePipeline(self._model)
        self.vram_mb = _vram_delta_mb(torch, free_before)
        log.info("faster-whisper loaded (%s MB)", self.vram_mb)

    def transcribe(self, pcm: np.ndarray, language: Optional[str] = None) -> tuple:
        """Transcribe *pcm*, returning ``(text, language_used)``.

        With no *language* the engine detects one. Whisper detects a single
        language per 30 s decode window, so a clip that switches mid-way is
        otherwise decoded wholly in the dominant language — Swedish audio with
        an English tail comes back translated. On the short (sequential) path we
        therefore detect per VAD utterance and decode each run in its own
        language; ``language_used`` is then ``"sv+en"``-style, in time order.
        """
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)
        duration_s = pcm.size / SAMPLE_RATE
        model, batched = self._model, self._batched
        if model is None or batched is None:
            raise RuntimeError("STT engine is unloaded")
        if duration_s > self.batch_above_s:
            # Long content: the batched pipeline already detects once over the
            # whole clip. Per-utterance detection here would add a detect call
            # per chunk to bulk decodes, and long reads are single-language.
            segments, info = batched.transcribe(
                pcm, batch_size=self.batch_size, beam_size=self.beam_size,
                language=language or None, vad_filter=True,
                initial_prompt=self.initial_prompt,
                condition_on_previous_text=self.condition_on_previous_text)
            return join_segments(segments), language or getattr(info, "language", None)
        if language is None and self.multilingual:
            runs = self._language_runs(pcm)
            if len(runs) > 1:
                texts, langs = [], []
                pad = int(RUN_PAD_S * SAMPLE_RATE)
                for start, end, run_language in runs:
                    piece = pcm[max(0, start - pad):min(pcm.size, end + pad)]
                    segments, _info = model.transcribe(
                        piece, beam_size=self.beam_size, language=run_language,
                        vad_filter=True, initial_prompt=self.initial_prompt,
                        condition_on_previous_text=self.condition_on_previous_text)
                    text = join_segments(segments)
                    if text:
                        texts.append(text)
                        langs.append(run_language)
                if not langs:
                    # Every run decoded to nothing (silence the VAD mistook for
                    # speech). Report no language rather than a language for
                    # text that does not exist.
                    return "", None
                return " ".join(texts), "+".join(langs)
            if runs:
                # One language across the clip: decode it whole, in that
                # language, so no second detection can disagree with the first.
                language = runs[0][2]
        segments, info = model.transcribe(
            pcm, beam_size=self.beam_size, language=language or None, vad_filter=True,
            initial_prompt=self.initial_prompt,
            condition_on_previous_text=self.condition_on_previous_text)
        return join_segments(segments), language or getattr(info, "language", None)

    def _language_runs(self, pcm: np.ndarray) -> list:
        """VAD-split *pcm* and label each speech run with its detected language."""
        model = self._model
        if model is None:
            raise RuntimeError("STT engine is unloaded")

        def detect(start, end):
            lang, prob, _all = model.detect_language(audio=pcm[start:end])
            return lang, prob

        return language_runs(speech_chunks(pcm), detect, min_prob=self.lang_min_prob)

    def warmup(self) -> None:
        # 1 s of near-silence is enough to compile the CUDA kernels and touch
        # the VAD path; both decode routes are exercised so neither is cold.
        # Fixed lengths: the batched route is warmed directly, not via the
        # duration threshold, so a huge WHISPER_BATCH_ABOVE_S cannot make the
        # warm-up clip hours long.
        noise = (np.random.default_rng(0).standard_normal(SAMPLE_RATE) * 1e-3).astype(np.float32)
        model, batched = self._model, self._batched
        if model is None or batched is None:
            raise RuntimeError("STT engine is unloaded")
        # transcribe() is lazy: the model only runs when the segments are consumed.
        # No VAD here: it would strip the noise and the model would never run.
        list(model.transcribe(noise, beam_size=self.beam_size, language="en", vad_filter=False)[0])
        # Batched pipeline: without VAD it accepts a clip < 30 s as one chunk.
        list(batched.transcribe(np.tile(noise, 20), batch_size=self.batch_size,
                                beam_size=self.beam_size, language="en", vad_filter=False)[0])
        # And once through the Silero VAD so its first (slow) call is not a user's.
        list(model.transcribe(noise, beam_size=self.beam_size, language="en", vad_filter=True)[0])

    def unload(self) -> None:
        self._batched = None
        self._model = None

    def describe(self) -> dict:
        return {"engine": self.name, "model": self.model_name, "device": self.device,
                "compute_type": self.compute_type, "vram_mb": self.vram_mb,
                "beam_size": self.beam_size,
                "condition_on_previous_text": self.condition_on_previous_text,
                "initial_prompt": bool(self.initial_prompt),
                "multilingual": self.multilingual,
                "lang_min_prob": self.lang_min_prob,
                "batch_above_s": self.batch_above_s}
