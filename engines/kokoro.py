"""Kokoro-82M TTS engine (CUDA, torch) — the long-form reading voice.

Chatterbox stays the short-chat engine (clone + emotion). Kokoro is the
reader: 82M params, RTF ~0.05 on the 5070 Ti, no reference clip, fixed
voice packs.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np

from engines import VoiceParams

log = logging.getLogger("voice-daemon.tts")

# Kokoro's context is 510 *phonemes* per forward pass, and VOICES.md puts the
# voice packs' sweet spot at 100-200 phonemes — past ~400 the delivery rushes.
# The daemon chunks by characters, so the number below is that band converted:
# A throwaway measurement found 1.01 phonemes per character on the bench
# corpus, so 200 chars keeps every chunk inside 100-200 phonemes (measured on
# 03-long: 6 chunks, 60-188 phonemes). This is the engine's number, the daemon
# just honours it — it is NOT Chatterbox's 300.
MAX_CHARS_PER_CALL = int(os.environ.get("KOKORO_MAX_CHARS", "200"))

SAMPLE_RATE = 24000
DEFAULT_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")
DEFAULT_LANG_CODE = os.environ.get("KOKORO_LANG_CODE", "a")
REPO_ID = os.environ.get("KOKORO_REPO_ID", "hexgrad/Kokoro-82M")

# Voice packs offered per lang_code. Kokoro's packs are ~half a megabyte each,
# so the whole shortlist is preloaded at startup and a `?voice=` switch costs
# nothing at request time. Keep the list short and curated: the point is a few
# distinguishable voices, not every id hexgrad ships. Override with
# KOKORO_VOICES (comma-separated); the default voice is always included.
VOICE_CATALOGUE = {
    "a": ("af_heart", "af_bella", "am_michael"),  # American: f warm, f bright, m
    "b": ("bf_emma", "bm_george"),                # British
}


def _configured_voices(lang_code: str, default_voice: str) -> list[str]:
    raw = os.environ.get("KOKORO_VOICES", "")
    names = [v.strip() for v in raw.split(",") if v.strip()] if raw else \
        list(VOICE_CATALOGUE.get(lang_code, ()))
    if default_voice not in names:
        names.insert(0, default_voice)
    return names


class KokoroEngine:
    name = "kokoro"
    max_chars = MAX_CHARS_PER_CALL
    sample_rate = SAMPLE_RATE

    def __init__(self, device: Optional[str] = None) -> None:
        from kokoro import KPipeline

        self.device = device or os.environ.get("TTS_DEVICE", "cuda")
        self.voice = DEFAULT_VOICE
        self.lang_code = DEFAULT_LANG_CODE
        self.speed = float(os.environ.get("KOKORO_SPEED", "1.0"))
        log.info("loading Kokoro TTS on %s (lang=%s)", self.device, self.lang_code)
        self._pipe = KPipeline(lang_code=self.lang_code, repo_id=REPO_ID, device=self.device)
        # Pay the voice-pack download/load once, not on the first request. A
        # pack that fails to load is dropped from the list rather than left to
        # fail a user request later — /voices only ever lists loaded packs.
        self._voices: list[str] = []
        for name in _configured_voices(self.lang_code, self.voice):
            try:
                self._pipe.load_voice(name)
            except Exception:  # noqa: BLE001 — a bad id must not stop startup
                if name == self.voice:
                    raise
                log.exception("Kokoro voice pack %r failed to load; dropping it", name)
                continue
            self._voices.append(name)
        log.info("Kokoro TTS loaded (sr=%d, voice=%s, voices=%s)",
                 self.sample_rate, self.voice, ",".join(self._voices))

    def voices(self) -> list[str]:
        """Voice ids this engine will accept on ``VoiceParams.voice``."""
        return list(self._voices)

    def synthesize(self, text: str, params: VoiceParams) -> Tuple[np.ndarray, int]:
        """One pipeline pass. *text* should already be <= ``max_chars``.

        ``split_pattern=None`` because the daemon has already chunked; Kokoro
        still splits internally if a chunk exceeds its 510-phoneme context,
        and those pieces are concatenated here.
        """
        pipe = self._pipe
        if pipe is None:
            raise RuntimeError("TTS engine is unloaded")
        text = " ".join(text.split())
        if not text:
            return np.zeros(0, dtype=np.int16), self.sample_rate
        voice = params.voice or self.voice
        if voice not in self._voices:
            raise ValueError(
                f"unknown voice {voice!r}; known: {sorted(self._voices)}")
        pieces = [
            np.asarray(result.audio, dtype=np.float32)
            for result in pipe(text, voice=voice, speed=self.speed, split_pattern=None)
            if result.audio is not None
        ]
        if not pieces:
            return np.zeros(0, dtype=np.int16), self.sample_rate
        audio = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16), self.sample_rate

    def warmup(self) -> None:
        self.synthesize("Warming up.", VoiceParams())

    def unload(self) -> None:
        self._pipe = None

    def describe(self) -> dict:
        return {"engine": self.name, "device": self.device, "sample_rate": self.sample_rate,
                "max_chars": self.max_chars, "voice": self.voice, "lang_code": self.lang_code,
                "voices": self.voices()}
