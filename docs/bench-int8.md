# STT compute type: int8_float16 vs float16

> **Clips note (public release).** These numbers were measured on an earlier
> version of `bench/clips/` that Chatterbox generated from the same texts. The
> clips shipped in this repository were regenerated with Kokoro (`af_heart`)
> and one sentence of `03-long.txt` was reworded, so a re-run will not match
> these tables to the millisecond or the exact WER.

Measured 2026-09-07 on the test machine (RTX 5070 Ti 16 GB, Ryzen 9 7950X), `large-v3-turbo`,
faster-whisper / CTranslate2, beam size 1, language auto-detect.

Reproduce:

```bash
scripts/bench.py compare --language "" --runs 3          # HTTP path, private daemons
VOICEBOT_GPU_TESTS=1 .venv/bin/python -m unittest tests.test_stt_compute_type_gpu -v   # engine path, + sv clips
```

## Memory

The model is quantized by CTranslate2 at load; the model id on disk is
unchanged. `model_vram_mb` / engine `vram_mb` is the weight-load delta only;
cuBLAS/cuDNN workspaces allocated at first encode are not included.
`daemon_used_mb` is the whole process including torch/CUDA context (that is
why daemon_used differs by ~1 GB while the model differs by 960 MB).

| compute_type | model_vram_mb | vs baseline | daemon_used_mb |
|---|---|---|---|
| float16 | 1952 | 1.00x | 6278 |
| int8_float16 | 992 | 0.51x | 5254 |

The model itself halves, as CTranslate2 documents. Process total falls by
~1 GB rather than ~1 GB exactly matching, because the CUDA context and torch
allocator are a fixed cost either way — the headroom this change was about is the
~960 MB of model weights, and that is what came back.

## Quality

WER against the reference transcripts, via the daemon (`scripts/bench.py
compare`). `dwer` is int8_float16 minus float16; the spec's tolerance is 0.05.

| clip | audio_s | wer[float16] | wer[int8_float16] | dwer |
|---|---|---|---|---|
| 01-short.ogg | 2.5 | 0.00 | 0.00 | 0.00 |
| 02-ten-sec.ogg | 11.3 | 0.12 | 0.12 | 0.00 |
| 03-long.ogg | 37.0 | 0.05 | 0.05 | 0.00 |
| 04-silence.ogg | 3.0 | 0.00 | 0.00 | 0.00 |
| 05-wav-16k.wav | 11.3 | 0.00 | 0.00 | 0.00 |

Zero degradation on every clip. The non-zero baseline WERs are pre-existing
(the 0.12 on `02-ten-sec` is casing and "3GB"/"three gigabytes" style
normalisation against a TTS-generated reference, not a quantization artifact —
it is identical at both precisions).

Swedish, via the engine directly over Common Voice sv-SE (the clips the
`stt-language-detect` change used):

| clip | wer[float16] | wer[int8_float16] | lang |
|---|---|---|---|
| common_voice_sv-SE_25158638 | 0.000 | 0.000 | sv / sv |
| common_voice_sv-SE_25158633 | 0.000 | 0.000 | sv / sv |

Both reference sentences come back verbatim at both precisions, and detection
agrees. Transcripts differed on exactly one token across all seven clips:
`Chatterbox` vs `chatterbox` — a casing choice on a proper noun, inside the
existing WER noise floor.

## Latency

Median of 3, engine path (model time only, no HTTP):

| clip | ms[float16] | ms[int8_float16] | ratio |
|---|---|---|---|
| 01-short | 169 | 152 | 0.90 |
| 02-ten-sec | 228 | 206 | 0.90 |
| 03-long | 364 | 331 | 0.91 |
| 04-silence | 81 | 74 | 0.91 |
| 05-wav-16k | 235 | 215 | 0.91 |
| sv 25158638 | 177 | 161 | 0.91 |
| sv 25158633 | 171 | 154 | 0.90 |

Consistently ~9-10% *faster*, not merely "not worse" — int8 weights mean less
memory traffic per decode step. The HTTP-path numbers show the same direction
with more variance (0.92-0.99), as expected once ffmpeg and the socket are in
the measurement.

## Conclusion

`int8_float16` is strictly better here on all three axes: half the model VRAM,
identical transcripts, slightly faster. Shipped as the `WHISPER_COMPUTE_TYPE`
default. Rollback is `WHISPER_COMPUTE_TYPE=float16` in the unit's environment —
no code change, no redeploy.

Not tested and deliberately not shipped: the prequantized
`ThomasG/faster-whisper-large-v3-turbo-int8-fp16` weights (load-time
quantization is cheap and one model on disk is simpler), and plain `int8` /
`bfloat16` (different tradeoff; `scripts/bench.py compare --compute-type int8
--compute-type bfloat16` will measure them if it ever matters).
