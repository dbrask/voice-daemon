"""Chatterbox TTS engine (CUDA, torch)."""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np

from engines import VoiceParams

log = logging.getLogger("voice-daemon.tts")

# Chatterbox degrades badly past a few hundred characters per call. The daemon
# chunks text at sentence boundaries to stay under this; it is an engine
# property, so the engine owns the number.
MAX_CHARS_PER_CALL = int(os.environ.get("VOICE_TTS_MAX_CHARS", "300"))


def _patch_perth() -> None:
    """chatterbox 0.1.7 calls ``perth.PerthImplicitWatermarker()`` unconditionally.

    ``perth`` sets that symbol to None when its ``perth_net`` submodule fails to
    import (it uses ``pkg_resources``, removed from setuptools >= 81), and the
    constructor then raises ``TypeError: 'NoneType' object is not callable``.
    Fall back to the no-op watermarker; we are not distributing the audio.
    """
    import perth

    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
        log.warning("perth implicit watermarker unavailable; using DummyWatermarker")


class ChatterboxEngine:
    name = "chatterbox"
    max_chars = MAX_CHARS_PER_CALL

    def __init__(self, device: Optional[str] = None) -> None:
        _patch_perth()
        from chatterbox.tts import ChatterboxTTS

        self.device = device or os.environ.get("TTS_DEVICE", "cuda")
        log.info("loading Chatterbox TTS on %s", self.device)
        self._tts = ChatterboxTTS.from_pretrained(device=self.device)
        self.sample_rate = int(self._tts.sr)
        log.info("Chatterbox TTS loaded (sr=%d)", self.sample_rate)

    def synthesize(self, text: str, params: VoiceParams) -> Tuple[np.ndarray, int]:
        """One generate() call. *text* should already be <= ``max_chars``."""
        import torch

        tts = self._tts
        if tts is None:
            raise RuntimeError("TTS engine is unloaded")
        text = " ".join(text.split())
        if not text:
            return np.zeros(0, dtype=np.int16), self.sample_rate
        with torch.inference_mode():
            wav = tts.generate(
                text,
                exaggeration=params.exaggeration,
                cfg_weight=params.cfg_weight,
                temperature=params.temperature,
            )
        audio = wav.squeeze().detach().cpu().float().clamp(-1, 1)
        return (audio * 32767).to(torch.int16).numpy(), self.sample_rate

    def warmup(self) -> None:
        self.synthesize("Warming up.", VoiceParams())

    def unload(self) -> None:
        self._tts = None

    def describe(self) -> dict:
        return {"engine": self.name, "device": self.device, "sample_rate": self.sample_rate,
                "max_chars": self.max_chars}
