# STT model: large-v3 vs large-v3-turbo

> **Clips note (public release).** These numbers were measured on an earlier
> version of `bench/clips/` that Chatterbox generated from the same texts. The
> clips shipped in this repository were regenerated with Kokoro (`af_heart`)
> and one sentence of `03-long.txt` was reworded, so a re-run will not match
> these tables to the millisecond or the exact WER.

Measured 2026-09-17 on the test machine (RTX 5070 Ti 16 GB, Ryzen 9 7950X). Same CTranslate2
stack, `int8_float16`, beam size 1, language auto-detect. No engine
change; private `--spawn` daemons on a separate port only. Live unit stayed
`large-v3-turbo` + Kokoro on `:9877`.

Reproduce:

```bash
# English clips (HTTP, private daemon)
scripts/bench.py --suite stt --language "" --runs 3 \
  --spawn WHISPER_MODEL=large-v3-turbo WHISPER_COMPUTE_TYPE=int8_float16
scripts/bench.py --suite stt --language "" --runs 3 \
  --spawn WHISPER_MODEL=large-v3 WHISPER_COMPUTE_TYPE=int8_float16

# Swedish Common Voice (engine path, default turbo)
VOICEBOT_GPU_TESTS=1 .venv/bin/python -m unittest \
  tests.test_stt_language_gpu tests.test_stt_compute_type_gpu -v
```

Swedish rows in the table below are the same two CV sv-SE clips,
posted to each spawned daemon and scored with `scripts.bench.wer`.
Raw JSON: `/tmp/bench-large-v3.json` (not in git).

`nvidia-smi` was unavailable this session (`NVML library version:
615.71` mismatch). VRAM numbers are from `/health`.

## Memory

`model_vram_mb` is the engine weight-load delta (same definition as
`docs/bench-int8.md`). `gpu_used_mb` is card-wide `/health` used while
the spawn ran **next to** the live unit (~2.0 GB turbo+Kokoro), so it
is not an exclusive footprint.

| model | model_vram_mb | vs turbo | gpu_used_mb (live + spawn) |
|---|---|---|---|
| large-v3-turbo | 992 | 1.00x | 3836 |
| large-v3 | 1824 | 1.84x | 4668 |

large-v3 int8 lands at **1.8 GB of weights**, not the ~3 GB class the
earlier planning note guessed from official large-v2 fp16/int8 tables. It fits
exclusive or resident next to Kokoro.

## Quality

WER against `bench/clips/<name>.txt` (English) or the pinned Common
Voice sentences (Swedish). `dwer` is large-v3 minus turbo. Median of 3.

| clip | audio_s | wer[turbo] | wer[large-v3] | dwer | wall_ms[turbo] | wall_ms[large-v3] | ratio | rtf[turbo] | rtf[large-v3] |
|---|---|---|---|---|---|---|---|---|---|
| 01-short.ogg | 2.49 | 0.000 | 0.000 | 0.000 | 217 | 256 | 1.18 | 0.087 | 0.103 |
| 02-ten-sec.ogg | 11.29 | 0.116 | 0.093 | −0.023 | 248 | 451 | 1.82 | 0.022 | 0.040 |
| 03-long.ogg | 36.97 | 0.049 | 0.056 | +0.007 | 398 | 840 | 2.11 | 0.011 | 0.023 |
| 04-silence.ogg | 3.01 | 0.000 | 0.000 | 0.000 | 104 | 113 | 1.09 | 0.035 | 0.037 |
| 05-wav-16k.wav | 11.32 | 0.000 | 0.000 | 0.000 | 249 | 441 | 1.77 | 0.022 | 0.039 |
| common_voice_sv-SE_25158638 | 4.18 | 0.000 | 0.000 | 0.000 | 192 | 282 | 1.47 | 0.046 | 0.067 |
| common_voice_sv-SE_25158633 | 3.64 | 0.000 | 0.000 | 0.000 | 185 | 250 | 1.35 | 0.051 | 0.069 |

Swedish transcripts are byte-identical to the references on both
models; both report `language=sv`. GPU unittest suite (default turbo)
was 12/12 in 16.4 s.

The 02-ten-sec “win” is not content-path quality. Both models still
say `3GB` against a reference that writes `three gigabytes`; large-v3
only recovered the word `in` (`Chatterbox in another 4` vs turbo’s
`chatterbox and another 4`). That is the whole −0.023 dWER.

On 03-long (~37 s, the long English clip), large-v3 is **worse**:
turbo said `deep-sea cables` (ref: `deep sea cables`); large-v3 said
`Deep-C cables`. No podcast file is in `testdata/` / `bench/clips`, so
there is no named-entity miss for large-v3 to catch.

Pass rule set before the run (internal planning note, not included): large-v3 dWER vs turbo ≤ 0
on CV sv **and** on `03-long`, *or* a named-entity catch on a podcast
clip. CV sv is a tie at 0.000. `03-long` is +0.007. No podcast clip.
**Does not pass.**

## Latency

large-v3 is ~1.8–2.1× wall on the 11–37 s clips, still far realtime
(RTF 0.023 on 37 s). Fine for overnight batch; not a reason to take
it if WER does not move.

## Conclusion

`WHISPER_MODEL=large-v3` is a wash on this audio and a named-entity
regression on the only long English clip. Do not change the live unit
or the content default. Next STT lever is the Parakeet TDT 0.6B v3
experiment (not part of this repository).

Not tested: float16 large-v3, a real English podcast file, the
optional 50-clip CV sample.
