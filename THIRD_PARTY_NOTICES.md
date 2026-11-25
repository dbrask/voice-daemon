# Third-party notices

This repository contains no model weights, voice files or vendored libraries.
Models and packages are downloaded by you at install or first run; their
licences are listed in the README.

## Code adapted into this repository

### Hermes Agent — `normalize.py`

The markdown/symbol regular expressions and the unit and currency tables in
`normalize.py` are adapted from `tools/tts_text_normalize.py` in Hermes Agent
(https://github.com/NousResearch/hermes-agent).

```
MIT License

Copyright (c) 2025 Nous Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Audio in this repository

`bench/clips/*.ogg` and `bench/clips/05-wav-16k.wav` are synthetic speech made
with Kokoro-82M (Apache-2.0), voice `af_heart`, from the texts in
`bench/texts/`, using this repository's own `scripts/bench.py make-clips`.
`04-silence.ogg` is generated silence. No recording of a real person is included.
