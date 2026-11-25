"""Piper TTS engine (ONNX VITS, CPU) — the zero-VRAM fallback voice.

Chatterbox and Kokoro are both torch/CUDA. When the GPU is busy or degraded
the daemon has no mouth at all, so this engine exists to keep one: a small
ONNX VITS model that runs entirely on the processor at RTF ~0.03
(measured: 5.39 s of audio in 0.15 s).

Why no `piper-tts` dependency
-----------------------------
Piper's runtime is three steps — espeak-ng IPA phonemes, the voice's own
``phoneme_id_map``, an ONNX forward pass — and every piece is already in this
venv: ``onnxruntime`` resolves CPU-only here, and ``phonemizer`` +
``espeakng_loader`` come along with kokoro/misaki. Depending on ``piper-tts``
would vendor a second onnxruntime and phonemizer pin into a venv whose torch
stack is already pinned to cu128 wheels. So the ~40 lines are inline instead.

A voice is two files in ``PIPER_VOICE_DIR`` (default ``~/piper-voices``):
``NAME.onnx`` and ``NAME.onnx.json``. Everything variable — sample rate,
espeak voice, phoneme inventory, inference scales — is read from the sidecar,
never hardcoded, because Piper voices genuinely differ (x_low is 16 kHz,
medium is 22.05 kHz). Download more from rhasspy/piper-voices; dropping the
pair in the directory is the whole install.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from engines import VoiceParams

log = logging.getLogger("voice-daemon.tts")

# Piper's four reserved symbols, per its phoneme_id_map: pad, beginning of
# sentence, end of sentence. The map carries the ids; these are the keys.
PAD, BOS, EOS = "_", "^", "$"

DEFAULT_VOICE_DIR = "~/piper-voices"

# Piper has no hard context limit the way Kokoro's 510 phonemes does — it is a
# VITS decoder over the whole id sequence. The cap is about latency and prosody
# drift on very long inputs, and about giving the daemon's chunker a number to
# honour. 400 chars is roughly a long paragraph sentence group.
MAX_CHARS_PER_CALL = int(os.environ.get("PIPER_MAX_CHARS", "400"))

CPU_PROVIDER = "CPUExecutionProvider"


def _make_session(model_path: str):
    """Build a CPU-only ONNX session.

    Split out as a module-level function so tests can substitute it, and so
    the ``providers`` list lives in exactly one place: this engine's entire
    reason to exist is that it does not touch the GPU.
    """
    import onnxruntime

    opts = onnxruntime.SessionOptions()
    opts.log_severity_level = 3  # warnings and up; ORT is chatty on load
    return onnxruntime.InferenceSession(
        model_path, sess_options=opts, providers=[CPU_PROVIDER]
    )


def _make_phonemizer(espeak_voice: str):
    """espeak-ng IPA backend, pointed at the bundled library.

    ``espeakng_loader`` ships libespeak-ng and its data inside the venv, so
    this works without the system espeak-ng package installed.
    """
    import espeakng_loader
    from phonemizer.backend import EspeakBackend
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    # phonemizer's words_mismatch="ignore" still emits a summary WARNING
    # (Ignore.process → _resume). Piper does not use word alignment.
    phonemizer_log = logging.getLogger("voice-daemon.tts.phonemizer")
    phonemizer_log.setLevel(logging.ERROR)
    return EspeakBackend(
        espeak_voice,
        preserve_punctuation=True,
        with_stress=True,
        language_switch="remove-flags",
        words_mismatch="ignore",
        logger=phonemizer_log,
    )


def discover_voices(voice_dir: Path) -> Dict[str, Path]:
    """``{name: model path}`` for every ``NAME.onnx`` with a ``NAME.onnx.json``.

    A model without its sidecar is unusable — the sidecar holds the phoneme
    map — so it is skipped rather than offered and failed later.
    """
    if not voice_dir.is_dir():
        return {}
    found = {}
    for model in sorted(voice_dir.glob("*.onnx")):
        if model.with_suffix(".onnx.json").exists():
            found[model.stem] = model
    return found


class _Voice:
    """One loaded voice: its sidecar config, ONNX session and phonemizer."""

    def __init__(self, name: str, model_path: Path) -> None:
        self.name = name
        self.model_path = model_path
        self.config = json.loads(model_path.with_suffix(".onnx.json").read_text())
        self.sample_rate = int(self.config["audio"]["sample_rate"])
        self.phoneme_id_map: Dict[str, List[int]] = self.config["phoneme_id_map"]
        inference = self.config.get("inference", {})
        self.scales = np.array(
            [
                float(inference.get("noise_scale", 0.667)),
                float(inference.get("length_scale", 1.0)),
                float(inference.get("noise_w", 0.8)),
            ],
            dtype=np.float32,
        )
        espeak_voice = self.config.get("espeak", {}).get("voice", "en-us")
        log.info("loading Piper voice %s (sr=%d, espeak=%s)",
                 name, self.sample_rate, espeak_voice)
        self.session = _make_session(str(model_path))
        active = list(self.session.get_providers())
        if active != [CPU_PROVIDER]:
            raise RuntimeError(
                f"Piper voice {name!r} got ONNX providers {active}; this engine is "
                f"the zero-VRAM fallback and must run on {CPU_PROVIDER} alone"
            )
        self.phonemizer = _make_phonemizer(espeak_voice)

    def encode(self, phonemes: str) -> List[int]:
        """Piper's id encoding: BOS, PAD, then each phoneme followed by PAD, then EOS.

        A symbol the voice does not know is dropped. espeak's inventory is
        wider than any single voice's, so a stray phoneme must degrade the word
        rather than fail the request. Punctuation espeak emits that the voice
        map lacks (em-dash, ellipsis) is dropped the same way; the daemon's
        ``normalize.prepare_spoken_text`` runs first on the live path.

        Encoding is pinned to Piper's own algorithm:
        https://github.com/rhasspy/piper-phonemize/blob/master/src/phoneme_ids.cpp
        https://github.com/OHF-Voice/piper1-gpl/blob/main/src/piper/phoneme_ids.py
        """
        pim = self.phoneme_id_map
        ids = list(pim[BOS])
        pad = pim[PAD]
        ids.extend(pad)
        for ph in phonemes:
            entry = pim.get(ph)
            if entry is None:
                log.debug("Piper voice %s has no id for phoneme %r; skipping",
                          self.name, ph)
                continue
            ids.extend(entry)
            ids.extend(pad)
        ids.extend(pim[EOS])
        return ids

    def synthesize(self, text: str) -> np.ndarray:
        phonemes = self.phonemizer.phonemize([text], strip=True)[0]
        ids = self.encode(phonemes)
        out = self.session.run(
            None,
            {
                "input": np.asarray([ids], dtype=np.int64),
                "input_lengths": np.asarray([len(ids)], dtype=np.int64),
                "scales": self.scales,
            },
        )[0]
        return np.asarray(out, dtype=np.float32).squeeze()


class PiperEngine:
    name = "piper"
    max_chars = MAX_CHARS_PER_CALL

    def __init__(self, voice_dir: Optional[str] = None,
                 default_voice: Optional[str] = None) -> None:
        raw_dir = voice_dir or os.environ.get("PIPER_VOICE_DIR") or DEFAULT_VOICE_DIR
        self.voice_dir = Path(os.path.expanduser(raw_dir))
        self._available = discover_voices(self.voice_dir)
        if not self._available:
            raise FileNotFoundError(
                f"no Piper voices in {self.voice_dir}: expected NAME.onnx next to "
                f"NAME.onnx.json (get them from rhasspy/piper-voices, or set "
                f"PIPER_VOICE_DIR)"
            )
        wanted = default_voice or os.environ.get("PIPER_VOICE")
        if wanted and wanted not in self._available:
            raise ValueError(
                f"unknown Piper voice {wanted!r}; known: {sorted(self._available)}"
            )
        self.voice = wanted or sorted(self._available)[0]
        self._voices: Dict[str, _Voice] = {}
        self._unloaded = False
        # Pay the default voice's load at construction, like the other engines:
        # a broken model must fail startup, not the first user request.
        default = self._load(self.voice)
        self.sample_rate = default.sample_rate
        log.info("Piper TTS loaded on CPU (sr=%d, voice=%s, voices=%s)",
                 self.sample_rate, self.voice, ",".join(self.voices()))

    # -- internals ---------------------------------------------------------

    def _load(self, name: str) -> _Voice:
        voice = self._voices.get(name)
        if voice is None:
            voice = _Voice(name, self._available[name])
            self._voices[name] = voice
        return voice

    def _session_for(self, name: str):
        """Test seam: the ONNX session behind a loaded voice."""
        return self._voices[name].session

    def _loaded_voice_names(self) -> List[str]:
        return sorted(self._voices)

    # -- engine contract ---------------------------------------------------

    def voices(self) -> List[str]:
        return sorted(self._available)

    def synthesize(self, text: str, params: VoiceParams) -> Tuple[np.ndarray, int]:
        if self._unloaded:
            raise RuntimeError("TTS engine is unloaded")
        name = params.voice or self.voice
        if name not in self._available:
            raise ValueError(
                f"unknown voice {name!r}; known: {sorted(self._available)}"
            )
        text = " ".join(text.split())
        if not text:
            return np.zeros(0, dtype=np.int16), self.sample_rate
        voice = self._load(name)
        audio = voice.synthesize(text)
        if audio.size == 0:
            return np.zeros(0, dtype=np.int16), voice.sample_rate
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        return pcm, voice.sample_rate

    def warmup(self) -> None:
        self.synthesize("Warming up.", VoiceParams())

    def unload(self) -> None:
        self._voices.clear()
        self._unloaded = True

    def describe(self) -> dict:
        return {"engine": self.name, "device": "cpu", "sample_rate": self.sample_rate,
                "max_chars": self.max_chars, "voice": self.voice,
                "voice_dir": str(self.voice_dir), "voices": self.voices()}
