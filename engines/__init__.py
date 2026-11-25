"""Engine seam for the voice daemon.

Two Protocols and a lazy registry. Adding an engine is one new file in this
package plus one line in ``_STT_REGISTRY`` / ``_TTS_REGISTRY`` — nothing in
``voice_daemon.py`` changes.

Contract:
    STTEngine.sample_rate                 -> int, PCM rate the engine wants
    STTEngine.transcribe(pcm, language)   -> (text, language_used)  (pcm: float32 mono at sample_rate)
    STTEngine.warmup()                    -> None  (run a dummy through the model)
    TTSEngine.max_chars                   -> int, longest text one synthesize() call takes well
    TTSEngine.synthesize(text, params)    -> (np.ndarray[int16] mono PCM, sample_rate)
    TTSEngine.warmup()                    -> None
    TTSEngine.voices()                    -> list[str], optional; named voice ids the
                                             engine accepts on VoiceParams.voice. An
                                             engine without named voices omits it and
                                             /voices reports an empty list.

Engines never touch HTTP, containers, or ffmpeg. The daemon owns those: it
decodes uploads to PCM before calling STT and encodes PCM after TTS, so every
audio byte in or out goes through one ffmpeg and the engines only ever see
numpy arrays. Text chunking for TTS is also the daemon's job; an engine just
says how much it can take per call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class VoiceParams:
    """Per-request synthesis knobs. Phase 0 accepts and mostly ignores these;
    phase 2 (voices/presets) gives them meaning. Engines must tolerate all
    fields being unset."""

    voice: Optional[str] = None
    language: Optional[str] = None
    exaggeration: float = 0.5
    cfg_weight: float = 0.3
    temperature: float = 0.8


@runtime_checkable
class STTEngine(Protocol):
    name: str
    sample_rate: int

    def transcribe(self, pcm: np.ndarray, language: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """Returns (transcript, language actually used). *language* None = detect."""
        ...

    def warmup(self) -> None: ...

    def unload(self) -> None: ...


@runtime_checkable
class TTSEngine(Protocol):
    name: str
    sample_rate: int
    max_chars: int

    def synthesize(self, text: str, params: VoiceParams) -> Tuple[np.ndarray, int]: ...

    def warmup(self) -> None: ...

    def unload(self) -> None: ...


# ---- registry -------------------------------------------------------------
# Values are factories imported lazily: importing chatterbox pulls torch and
# ~4 s of module import, which must not happen when only STT is wanted.

def _whisper_ct2_factory():
    from engines.whisper_ct2 import WhisperCT2Engine

    return WhisperCT2Engine


def _chatterbox_factory():
    from engines.chatterbox import ChatterboxEngine

    return ChatterboxEngine


def _kokoro_factory():
    from engines.kokoro import KokoroEngine

    return KokoroEngine


def _piper_factory():
    from engines.piper import PiperEngine

    return PiperEngine


_STT_REGISTRY: Dict[str, Callable[[], type]] = {
    "whisper_ct2": _whisper_ct2_factory,
}

# Three TTS engines: `chatterbox` is the short-chat voice (clone + emotion,
# max_chars 300), `kokoro` is the long-form reader (82M, RTF ~0.05, fixed voice
# packs, max_chars ~230 for its 100-200 phoneme sweet spot). Both need CUDA;
# `piper` is the CPU fallback (ONNX VITS, RTF ~0.03, zero VRAM) for when the
# GPU is busy or degraded. Pick with TTS_ENGINE. A fourth engine is one new
# module plus one line here.
_TTS_REGISTRY: Dict[str, Callable[[], type]] = {
    "chatterbox": _chatterbox_factory,
    "kokoro": _kokoro_factory,
    "piper": _piper_factory,
}


def stt_engine_names() -> list[str]:
    return sorted(_STT_REGISTRY)


def tts_engine_names() -> list[str]:
    return sorted(_TTS_REGISTRY)


def load_stt_engine(name: str, **kwargs) -> STTEngine:
    try:
        factory = _STT_REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown STT engine {name!r}; known: {stt_engine_names()}") from None
    return factory()(**kwargs)


def load_tts_engine(name: str, **kwargs) -> TTSEngine:
    try:
        factory = _TTS_REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown TTS engine {name!r}; known: {tts_engine_names()}") from None
    return factory()(**kwargs)
