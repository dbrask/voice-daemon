# voice-daemon — local speech-to-text and text-to-speech over HTTP

A small HTTP service that keeps a speech-to-text model and a text-to-speech
model loaded on your own GPU and answers on `127.0.0.1`. You send it audio and
get text back, or send it text and get audio back. Nothing leaves the machine.

It was built as the local "ears and mouth" for the
[Hermes](https://github.com/NousResearch/hermes-agent) agent, which calls it
through small command-line shims, but the HTTP interface is plain and anything
that can `curl` can use it. It is not a chat bot: it has no chat accounts, no
language model and no stored history. Hermes (or your own program) is the brain.

The code was written with AI coding agents under the owner's direction, and
reviewed and tested on the owner's machine.

## How it fits together

    voice note ─► hermes-stt ─► POST /stt  ─┐
                                            ├─ voice_daemon.py (127.0.0.1:9877)
    reply text ─► hermes-tts ─► POST /tts  ─┘     ├─ STT: faster-whisper large-v3-turbo
                                                  └─ TTS: Kokoro-82M | Chatterbox | Piper

- `voice_daemon.py` is one aiohttp process. It loads the engines once, keeps
  them warm, and refuses to bind anything but loopback. There is no
  authentication because there is no exposure.
- `engines/` holds one file per engine behind a small Protocol: speech-to-text
  with faster-whisper (CTranslate2), text-to-speech with Kokoro-82M (default),
  Chatterbox, or Piper (CPU only, no GPU memory).
- All audio in and out goes through `ffmpeg`. Engines only ever see PCM arrays.
- `bin/` has the command-line shims (`hermes-stt`, `hermes-tts`,
  `hermes-tts-long`, `voice-speak`). They only talk HTTP to the daemon.
- `satellite/` is an optional Raspberry Pi desk speaker: a thin client on the Pi
  and a relay next to the daemon (see [Desk satellite](#desk-satellite-optional)).

## Measured numbers

Measured on one machine: NVIDIA RTX 5070 Ti (16 GB, Blackwell / sm_120), AMD
Ryzen 9 7950X, torch 2.9.1 with CUDA 12.8 wheels, faster-whisper 1.2.1,
CTranslate2 4.8.2. Warm daemon, loopback HTTP, median of 3 runs unless noted.
Raw tables and method: [`docs/bench-2026-09-07.md`](docs/bench-2026-09-07.md),
[`docs/bench-int8.md`](docs/bench-int8.md),
[`docs/bench-large-v3.md`](docs/bench-large-v3.md).

| What | Number |
|---|---|
| STT, 11 s clip, wall time over HTTP | ~180–250 ms |
| STT, 37 s clip (batched path) | ~360–400 ms |
| STT model memory, `int8_float16` vs `float16` | 992 MB vs 1952 MB, same transcripts on the bench clips, ~9 % faster |
| Kokoro TTS, real-time factor | 0.009 (48 s of audio for a 765-character text) |
| Kokoro TTS, time to first audio | 69 ms in the engine, 81 ms over HTTP |
| Chatterbox TTS, real-time factor / first chunk | 0.24–0.28 / 0.5–0.8 s |
| Piper TTS on the CPU, real-time factor | ~0.03, zero GPU memory |
| Whole process, STT + Kokoro resident | ~5.9 GB of GPU memory |

The bench clips are synthetic speech (see [Bench](#bench)), so the word error
rates in the docs are a regression tripwire, not a quality score for real
voices. The daemon also runs without a GPU (`WHISPER_DEVICE=cpu`,
`TTS_DEVICE=cpu`); it works but transcription is far slower — about 3.6 s for a
15 s clip on the CPU above.

## Install

Requirements: Linux with systemd (for the example unit), Python 3.11+, `ffmpeg`,
`curl` and `jq` on `PATH`, and an NVIDIA GPU for the default configuration.
Kokoro also wants the `espeak-ng` system package as a fallback for words it
does not know.

    git clone <this repository's URL> ~/repo/voice-daemon-public   # any path works
    cd ~/repo/voice-daemon-public
    uv venv && uv pip install -r requirements.txt --override overrides.txt
    ./install.sh          # shims -> ~/.local/bin (or $BIN_DIR), user unit, .env from .env.example
    systemctl --user enable --now voice-daemon
    curl -s http://127.0.0.1:9877/health | jq

`requirements.txt` pulls the CUDA 12.8 torch wheels, which RTX 50-series cards
need. `overrides.txt` exists because `chatterbox-tts` pins an older torch
without kernels for those cards; the override replaces that pin.

The first start downloads model weights from Hugging Face into your normal
Hugging Face cache (a few GB). Nothing is bundled in this repository.

Settings live in `.env` (copied from [`.env.example`](.env.example), which
documents every variable). The most useful ones:

| Variable | Default | Meaning |
|---|---|---|
| `VOICE_HOST` / `VOICE_PORT` | `127.0.0.1` / `9877` | Where the daemon listens. Non-loopback hosts are refused. |
| `VOICE_RESIDENT` | `both` | `both`, `stt` or `tts`: which engines to load. |
| `WHISPER_MODEL` | `large-v3-turbo` | Any faster-whisper model name. |
| `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` | `cuda` / `int8_float16` | Use `cpu` / `int8` without a GPU. |
| `TTS_ENGINE` | `chatterbox` in code, `kokoro` in `.env.example` | `kokoro`, `chatterbox` or `piper`. |
| `TTS_DEVICE` | `cuda` | `cpu` works for Kokoro. |
| `KOKORO_VOICE` | `af_heart` | Default Kokoro voice. |
| `PIPER_VOICE_DIR` | `~/piper-voices` | Folder of `NAME.onnx` + `NAME.onnx.json` Piper voices. |
| `VOICE_URL` | `http://127.0.0.1:9877` | Where the shims and the bench find the daemon. |

## HTTP interface

Loopback only. JSON responses carry `"ok": true|false`; errors also carry
`"error"`. Examples below are real responses (captured from a CPU-only run, so
`vram_mb` is empty and the timings are CPU timings).

| Route | Input | Output |
|---|---|---|
| `GET /health` | — | engines, loaded state, `vram_mb`, load/warm-up/last-request timings |
| `GET /voices` | — | voices the current TTS engine accepts |
| `POST /load[?engine=stt\|tts\|both]` | — | loads and warms the engines, returns health |
| `POST /release[?engine=…]` | — | unloads an engine to free GPU memory |
| `POST /stt[?language=xx]` | audio as the raw body or a multipart `file`/`audio` field, any format ffmpeg reads | `{"text", "language", "audio_s", "timing_ms"}` |
| `POST /tts[?format=ogg\|wav&voice=]` | `text/plain` body or JSON `{"text", "voice", "format"}` | audio bytes; `X-Timing-Ms` and `X-Audio-Seconds` headers |
| `POST /tts/long` | JSON `{"text" or "chapters", "voice", "format"}` | `202` and a job; synthesis runs in the background |
| `GET /jobs`, `GET /jobs/<id>` | — | job list / one job with per-chapter progress and file paths |
| `POST /jobs/<id>/cancel`, `DELETE /jobs/<id>` | — | cancel / cancel and delete the audio |

### Text to speech

    curl -s -D - -o hello.ogg -XPOST http://127.0.0.1:9877/tts \
      -H 'content-type: application/json' \
      -d '{"text":"Hello from the local voice daemon. It is 21°C today.","voice":"af_heart","format":"ogg"}'

    HTTP/1.1 200 OK
    X-Timing-Ms: {"decode":0.0,"model":504.2,"encode":193.4,"total":697.6,"first_chunk":501.1,"request":698.5}
    X-Audio-Seconds: 4.8
    Content-Type: audio/ogg

Text is cleaned up for speech before the engine sees it: markdown markers and
code blocks are dropped, `21°C` becomes "21 degrees Celsius", `99%` becomes
"99 percent", and a blank line between paragraphs becomes a short real pause
(`VOICE_NORMALIZE=0` turns this off). An unknown voice is an error, never a
silent fallback:

    curl -s -XPOST 'http://127.0.0.1:9877/tts?voice=nope' --data 'Hi.'
    {"ok": false, "error": "unknown voice 'nope'; known: ['af_bella', 'af_heart', 'am_michael']"}   # HTTP 400

### Speech to text

    curl -s -XPOST --data-binary @hello.ogg http://127.0.0.1:9877/stt
    {
      "ok": true,
      "text": "Hello from the local voice demon. It is 21 degrees Celsius today.",
      "language": "en",
      "audio_s": 4.8,
      "timing_ms": {"decode": 39.8, "model": 6661.7, "encode": 0.0, "total": 6701.5, "request": 6702.4}
    }

(That `model` time is the CPU. On the GPU above it is ~150 ms for 10 s of audio.)

No language is assumed. Leave `language` out (or pass `auto`) and the engine
detects it, per utterance, so a clip that switches language between sentences
comes back as e.g. `"language": "sv+en"`. Passing a language forces it — and
forcing English on Swedish audio makes Whisper translate instead of transcribe,
so only force it when you are sure.

### Long documents

    curl -s -XPOST http://127.0.0.1:9877/tts/long -H 'content-type: application/json' \
      -d '{"text":"# One\n\nFirst bit.\n\n# Two\n\nSecond bit.","voice":"af_heart"}'
    {"ok": true, "job": {"id": "45a26aa99611", "status": "queued", "progress": {"chapters_total": 2, ...}, ...}}

    curl -s http://127.0.0.1:9877/jobs/45a26aa99611
    {"ok": true, "job": {"id": "45a26aa99611", "status": "done", "audio_s": 3.19,
      "chapters": [{"index": 1, "title": "One", "status": "done", "path": ".../001-one.ogg", "audio_s": 1.62, ...},
                   {"index": 2, "title": "Two", "status": "done", "path": ".../002-two.ogg", "audio_s": 1.57, ...}]}}

The text is split at markdown headings (or pass an explicit `chapters` array)
and each chapter becomes one audio file under `$XDG_CACHE_HOME/voice-bot/jobs/`.
That folder is a cache: files expire after a day (`VOICE_JOBS_TTL_S`) and are
dropped on restart.

## Command-line shims

    hermes-stt <input_audio> [output_txt] [language]     # transcript to file and stdout
    hermes-tts <input_text> <output_audio> [format] [voice]
    hermes-tts-long [--voice ID] [--auto-chapter [CHARS]] <input_text|->   # prints one MEDIA:<path> per chapter
    voice-speak [--voice ID] [--stop] [file]             # speak clipboard / stdin / file on the local speakers

Exit codes: `0` ok, `1` daemon or HTTP error (one line on stderr), `2` usage.
`voice-speak` needs `pw-play` or `ffplay`, and `wl-paste` for the clipboard.

Smoke test:

    printf 'Hello from the local GPU.' > /tmp/t.txt
    hermes-tts /tmp/t.txt /tmp/t.ogg && hermes-stt /tmp/t.ogg

### Using it from Hermes

Hermes supports `type: command` speech providers. In Hermes' `config.yaml`:

```yaml
stt:
  provider: gpu
  providers:
    gpu:
      type: command
      command: 'hermes-stt {input_path} {output_path} {language}'
tts:
  provider: gpu
  providers:
    gpu:
      type: command
      command: 'hermes-tts {input_path} {output_path} {format} {voice}'
      format: ogg               # Opus, which chat apps play as voice notes
      voice_compatible: true
      voice: af_heart           # must be an id GET /voices lists
      warm_command: 'curl -sf -XPOST http://127.0.0.1:9877/load'
      release_command: /bin/true
```

`release_command` is a deliberate no-op so the model is not unloaded every time
speech is switched off; point it at `/release?engine=tts` if you want it freed.
Write `/bin/true`, not `true`, or YAML reads it as a boolean.

## Engines and choices

- **Kokoro-82M** is the recommended voice: fast, small (~0.6–1.2 GB of GPU
  memory), good for both short replies and long reading. English voices only
  in this setup.
- **Chatterbox** is slower and larger (~3 GB) but has an expressiveness
  ("exaggeration") setting. If its
  watermarking library cannot load, the engine falls back to Chatterbox's no-op
  watermarker and the audio is **not** watermarked; keep that in mind before
  you publish audio it made.
- **Piper** runs on the CPU with no GPU memory at all, for when the GPU is busy.
  It is implemented directly on `onnxruntime` (no `piper-tts` package). You
  download voices yourself; check each voice's licence (see below).
- **Whisper** `large-v3-turbo` with `int8_float16` is the default. `large-v3` was
  measured and was not better on the bench clips (`docs/bench-large-v3.md`).
- `WHISPER_GAZETTEER=1` prepends a short spelling hint for recurring proper
  nouns; the built-in example names places and people from news coverage of
  Ukraine. Replace it with `WHISPER_INITIAL_PROMPT=...` for your own vocabulary.

## Licences of what you download

This repository's own code is MIT (see [`LICENSE`](LICENSE)). It ships no model
weights, voices or third-party libraries; `install` downloads them, and each
keeps its own licence. Part of `normalize.py` is adapted from Hermes Agent (MIT);
see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

| Component | What it is | Licence |
|---|---|---|
| faster-whisper, CTranslate2 | STT runtime | MIT |
| Whisper large-v3-turbo weights (OpenAI; CTranslate2 conversion on Hugging Face) | STT model | MIT |
| kokoro (Python package), misaki (G2P) | TTS runtime | Apache-2.0 |
| Kokoro-82M weights and voices (`hexgrad/Kokoro-82M`) | TTS model | Apache-2.0 |
| chatterbox-tts, resemble-perth | TTS runtime, watermarker | MIT |
| Chatterbox weights (`ResembleAI/chatterbox`) | TTS model | MIT |
| phonemizer (`phonemizer-fork`, pulled in by misaki) | phonemes for Kokoro fallback and Piper | **GPL-3.0** |
| espeak-ng (system package, and bundled by `espeakng-loader`) | phoneme engine | **GPL-3.0** |
| onnxruntime | Piper inference | MIT |
| torch, torchaudio | GPU runtime | BSD-3-Clause |
| aiohttp | HTTP server/client | Apache-2.0 (and MIT for a vendored part) |
| Piper voices (`rhasspy/piper-voices`) | CPU TTS voices | **varies per voice** — see below |

Things to know before you build on it:

- **GPL parts.** Kokoro's fallback and the Piper engine call phonemizer and
  espeak-ng, which are GPL-3.0. This repository does not include them and its
  own source stays MIT, but if you *distribute* a bundle that contains them
  (for example a container image), the GPL applies to that bundle.
- **Piper voices are licensed one by one**, by the data they were trained on.
  For example `en_US-lessac` is trained on the Blizzard 2013 Lessac data, whose
  licence allows research use only and excludes commercial use; `en_US-amy`
  comes from the Mimic 3 voices (CC BY-SA 4.0). Read the `MODEL_CARD` next to
  any voice you download.
- **Chatterbox watermarking**: see the note under Engines.

## Bench

    scripts/bench.py                       # STT + TTS against the running daemon
    scripts/bench.py --suite stt --spawn WHISPER_COMPUTE_TYPE=int8_float16   # private daemon on :9879
    scripts/bench.py compare               # int8_float16 vs float16 STT
    scripts/bench.py make-clips            # regenerate bench/clips with the daemon's own TTS

`bench/texts/` holds five short texts. `bench/clips/` holds the clips made from
them by `make-clips` with Kokoro (`af_heart`), plus three seconds of silence;
they are synthetic, not recordings of a person. The numbers in `docs/` were
measured on an earlier set made the same way with Chatterbox, so a re-run will
be close but not identical.

The opt-in GPU tests (`VOICEBOT_GPU_TESTS=1`) also use two Swedish sentences
from Mozilla Common Voice (CC0), which you download yourself into
`testdata/cv-corpus-26.0-2026-06-12/sv-SE/`; without it those tests skip.

## Tests

    uv venv .venv-test && VIRTUAL_ENV=.venv-test uv pip install -r requirements-test.txt
    .venv-test/bin/python -m unittest discover -s tests

313 tests on the standard library's `unittest` and `aiohttp.test_utils`, no GPU
and no model downloads; 15 skip unless you opt in with `VOICEBOT_GPU_TESTS=1`
and have a GPU, Piper voices and the Common Voice clips on disk. They use fake
engines and stub servers on random ports, and they cover the HTTP contract,
audio decoding, chunking, the long-job store, text normalisation, the shims,
the loopback-only rule, and the satellite protocol, relay and client, and the Pi install layout.

## Desk satellite (optional)

`satellite/` turns a Raspberry Pi with a microphone, a speaker and a small touch
screen into a tap-to-talk desk speaker for Hermes.

- **Client (on the Pi)**: records while you tap, streams the audio over one
  WebSocket, plays the reply, and serves a local "face" page (state and mood)
  on `127.0.0.1:8765`. It needs only `python3-aiohttp` from the Pi OS packages
  and holds one per-device token, no other keys.
- **Relay (next to the daemon)**: binds `127.0.0.1:9878`. For each turn it sends
  the audio to `/stt`, the text to the Hermes API server (`/v1/runs`), and the
  reply to `/tts`, then sends the audio back down the socket.

Setup sketch:

    # on the daemon host
    ./scripts/satellite-token desk          # prints a token once; only its sha256 is stored
    # set HERMES_API_URL and HERMES_API_KEY in .env, then:
    cp systemd/satellite-relay.service ~/.config/systemd/user/
    systemctl --user daemon-reload && systemctl --user enable --now satellite-relay
    # publish 127.0.0.1:9878 through a TLS reverse proxy of your choice,
    # e.g. at wss://relay.example.invalid/satellite/ws

    # on the Pi, from the root of a copy of this repository:
    mkdir -p ~/satellite && cp -r satellite ~/satellite/
    # that gives ~/satellite/satellite/ (the package), which is what the unit's
    # WorkingDirectory=%h/satellite and `python3 -m satellite.client` expect.
    # Store the token in ~/.config/satellite/token (mode 600), then install the unit:
    mkdir -p ~/.config/systemd/user
    cp ~/satellite/satellite/pi/satellite-client.service ~/.config/systemd/user/
    # set SAT_RELAY_URL, SAT_SINK and SAT_MIC in ~/.config/systemd/user/satellite-client.service
    systemctl --user enable --now satellite-client

Kiosk setup for the face page: [`satellite/pi/kiosk.md`](satellite/pi/kiosk.md).
Name the speaker and microphone explicitly (`--sink`/`--mic` or
`SAT_SINK`/`SAT_MIC`); the default output is not always the one you expect.

## Not included

This is a cleaned export of a private working repository, with a fresh history.
Left out on purpose:

- Planning notes, research notes, design specs and experiment scripts
  (`spikes/`), and agent tooling configuration — working material, not product.
- The satellite's character/persona and wake-word notes, and the acceptance and
  setup scripts that were tied to one particular home setup.
- Anything about the original machine and network: host names, addresses,
  service paths.
- Model weights, voices and the Common Voice test clips (download them).
- Document extraction (PDF/EPUB to text) for long reading. That lives in Hermes;
  this daemon only ever receives text.
- Streaming STT/TTS and barge-in. Requests are batch: whole audio in, whole
  audio out (long reading is chunked into chapter files).
