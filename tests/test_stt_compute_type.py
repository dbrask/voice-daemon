"""Compute-type selection for the whisper STT path — CPU only, no GPU, no model.

Covers the `stt-int8-quantization` change: the compute type is chosen by
env or argument, the shipped default is the quantized one, and the engine
reports both the type and the device memory its model load cost.

``WhisperCT2Engine.__init__`` imports faster_whisper and constructs a real
``WhisperModel``, so these tests patch the module it imports from and a fake
torch, then assert on what the constructor *asked* for. Nothing here touches
CUDA; the GPU-gated companion file proves the quality/VRAM claims for real.

Run:  .venv/bin/python -m unittest discover -s tests -v
"""

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engines import whisper_ct2 as w  # noqa: E402


class FakeWhisperModel:
    """Records the kwargs the engine loaded it with, and charges the fake GPU.

    The fake torch holds free memory in a mutable box. Construction subtracts
    ``load_bytes`` so ``mem_get_info`` after the load is lower than before —
    reading "before" after constructing the model yields a zero delta.
    """

    load_bytes = 1600 * 2**20

    def __init__(self, model, device=None, compute_type=None, **kwargs):
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self.kwargs = kwargs
        torch = sys.modules.get("torch")
        cuda = getattr(torch, "cuda", None) if torch is not None else None
        box = getattr(cuda, "free", None)
        if isinstance(box, list) and box:
            box[0] -= type(self).load_bytes


class FakeBatched:
    def __init__(self, model):
        self.model = model


def _fake_torch(free=8_000 * 2**20, available=True):
    """A torch stand-in whose free memory lives in a mutable box.

    ``FakeWhisperModel`` decrements the box; ``mem_get_info`` returns the
    current value. A falling iterator (independent of when the model loads)
    is not enough — it cannot tell "before" from "after".
    """
    box = [free]
    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        mem_get_info=lambda: (box[0], 16_000 * 2**20),
        synchronize=lambda: None,
        free=box,
    )
    return types.SimpleNamespace(cuda=cuda)


def _build(env=None, torch=None, **kwargs):
    """Construct a WhisperCT2Engine with faster_whisper and torch faked out."""
    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = FakeWhisperModel
    fw.BatchedInferencePipeline = FakeBatched
    modules = {"faster_whisper": fw, "torch": torch if torch is not None else _fake_torch()}
    with mock.patch.dict(sys.modules, modules), \
            mock.patch.dict("os.environ", env or {}, clear=False):
        return w.WhisperCT2Engine(**kwargs)


class ComputeTypeSelectionTests(unittest.TestCase):
    def setUp(self):
        # Each test states its own env; inherited values would make the
        # default-value assertions lie.
        patcher = mock.patch.dict(
            "os.environ",
            {k: v for k, v in __import__("os").environ.items()
             if k not in ("WHISPER_COMPUTE_TYPE", "WHISPER_DEVICE", "WHISPER_MODEL")},
            clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_environment_variable_selects_the_compute_type(self):
        eng = _build(env={"WHISPER_COMPUTE_TYPE": "float16"})
        self.assertEqual(eng.compute_type, "float16")
        self.assertEqual(eng._model.compute_type, "float16")
        self.assertEqual(eng.describe()["compute_type"], "float16")

    def test_explicit_argument_wins_over_the_environment(self):
        eng = _build(env={"WHISPER_COMPUTE_TYPE": "float16"}, compute_type="int8_float16")
        self.assertEqual(eng.compute_type, "int8_float16")
        self.assertEqual(eng._model.compute_type, "int8_float16")

    def test_default_compute_type_is_the_quantized_one(self):
        eng = _build(env={})
        self.assertEqual(eng.compute_type, "int8_float16")
        self.assertEqual(eng._model.compute_type, "int8_float16")

    def test_rollback_to_float16_needs_only_the_env_var(self):
        # The one-word rollback the proposal promises: same code, same model id,
        # only the compute type differs.
        quantized = _build(env={})
        rolled_back = _build(env={"WHISPER_COMPUTE_TYPE": "float16"})
        self.assertEqual(rolled_back.compute_type, "float16")
        self.assertEqual(rolled_back.model_name, quantized.model_name)
        self.assertEqual(rolled_back.describe()["compute_type"], "float16")

    def test_env_example_does_not_pin_a_different_compute_type_than_the_engine_default(self):
        # .env.example is copied to .env by install.sh and loaded by systemd.
        # A pin here that disagrees with the engine default is a second source
        # of truth — the code default never reaches a deployed daemon.
        pinned = None
        example = Path(__file__).resolve().parents[1] / ".env.example"
        for line in example.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            if key.strip() == "WHISPER_COMPUTE_TYPE":
                pinned = value.strip()
        engine_default = _build(env={}).compute_type
        if pinned is not None:
            self.assertEqual(
                pinned, engine_default,
                ".env.example pins WHISPER_COMPUTE_TYPE=%r but the engine "
                "default (empty env) is %r" % (pinned, engine_default),
            )


class VramReportingTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(
            "os.environ",
            {k: v for k, v in __import__("os").environ.items()
             if k not in ("WHISPER_COMPUTE_TYPE", "WHISPER_DEVICE", "WHISPER_MODEL")},
            clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_engine_reports_the_vram_its_model_load_cost(self):
        eng = _build(torch=_fake_torch(free=8_000 * 2**20))
        self.assertEqual(eng.vram_mb, 1600)
        self.assertEqual(eng.describe()["vram_mb"], 1600)

    def test_non_cuda_device_reports_no_figure_and_still_loads(self):
        eng = _build(env={"WHISPER_DEVICE": "cpu"})
        self.assertEqual(eng.device, "cpu")
        self.assertIsNone(eng.vram_mb)
        self.assertIn("vram_mb", eng.describe())
        self.assertIsNone(eng.describe()["vram_mb"])

    def test_cuda_without_a_usable_device_does_not_crash_the_load(self):
        # torch present but CUDA unavailable: measurement is impossible, the
        # model load is not. Reporting None beats refusing to start.
        eng = _build(torch=_fake_torch(available=False))
        self.assertIsNone(eng.vram_mb)
        self.assertIsNotNone(eng._model)

    def test_negative_raw_delta_clamps_to_zero_and_warns(self):
        # Another process freeing memory mid-load would make free_after larger
        # than free_before. The reported figure stays 0; swallowing the sign
        # without a warning hid that the single-owner-GPU assumption broke.
        original = FakeWhisperModel.load_bytes
        FakeWhisperModel.load_bytes = -1_600 * 2**20
        self.addCleanup(setattr, FakeWhisperModel, "load_bytes", original)
        with self.assertLogs("voice-daemon.stt", level="WARNING") as cm:
            eng = _build(torch=_fake_torch(free=8_000 * 2**20))
        self.assertEqual(eng.vram_mb, 0)
        self.assertTrue(
            any("raw" in line.lower() or "negative" in line.lower()
                for line in cm.output),
            cm.output,
        )


if __name__ == "__main__":
    unittest.main()
