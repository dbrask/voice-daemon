#!/usr/bin/env python3
"""Latency bench for the voice daemon. Talks HTTP only; never imports engines.

    scripts/bench.py                      # warm STT + TTS over bench/ against the running daemon
    scripts/bench.py --suite stt --runs 5
    scripts/bench.py --spawn WHISPER_COMPUTE_TYPE=int8_float16 --suite stt
    scripts/bench.py compare              # int8_float16 vs float16 STT, one table
    scripts/bench.py compare --compute-type float16 --compute-type int8 --compute-type bfloat16
    scripts/bench.py make-clips           # (re)generate bench/clips/*.ogg via the daemon's TTS

--spawn starts a private daemon on --spawn-port with the given env overrides
(resident = only the engines the chosen suites need, so two daemons fit in
VRAM), benches it, and kills it. The systemd unit is never touched.

compare runs the STT suite once per --compute-type, each in its own private
daemon, **sequentially** — one model resident at a time, so a VRAM figure is a
footprint and not a reading taken next to somebody else's model. It prints a
per-clip delta against the first type listed (the baseline).

Numbers come from the daemon's own timing_ms (decode / model / encode) plus
wall time measured here, so the HTTP+curl overhead is visible as the gap.
WER is against bench/clips/<name>.txt after a crude normalisation; it is a
regression tripwire, not a benchmark score.

Output: a markdown table on stdout (paste into docs/bench-<date>.md) and,
with --json, the raw rows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
CLIPS = BENCH / "clips"
TEXTS = BENCH / "texts"
DEFAULT_URL = os.environ.get("VOICE_URL", "http://127.0.0.1:9877")

# name -> (reference text, format, sample rate). Silence has no text on purpose.
CLIP_SPECS: Dict[str, Dict[str, Any]] = {
    "01-short":      {"text": "01-short.txt",   "fmt": "ogg", "rate": 48000},
    "02-ten-sec":    {"text": "02-medium.txt",  "fmt": "ogg", "rate": 48000},
    "03-long":       {"text": ["03-long.txt", "02-medium.txt"], "fmt": "ogg", "rate": 48000},  # > 30 s: batched path
    "04-silence":    {"text": None,             "fmt": "ogg", "rate": 48000, "seconds": 3.0},
    "05-wav-16k":    {"text": "05-expressive.txt", "fmt": "wav", "rate": 16000},
}


# ─── HTTP ─────────────────────────────────────────────────────────

def _request(url: str, data: Optional[bytes] = None, headers: Optional[Dict[str, str]] = None,
             method: Optional[str] = None, timeout: float = 600) -> tuple[int, Dict[str, str], bytes]:
    req = urllib.request.Request(url, data=data, headers=headers or {},
                                 method=method or ("POST" if data is not None else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read()


def health(url: str) -> Dict[str, Any]:
    status, _h, body = _request(f"{url}/health")
    if status != 200:
        raise SystemExit(f"{url}/health -> HTTP {status}: {body[:200]!r}")
    return json.loads(body)


def stt(url: str, audio: bytes, language: Optional[str]) -> tuple[float, Dict[str, Any]]:
    q = f"?language={language}" if language else ""
    t0 = time.perf_counter()
    status, _h, body = _request(f"{url}/stt{q}", data=audio,
                                headers={"Content-Type": "application/octet-stream"})
    wall = (time.perf_counter() - t0) * 1000
    if status != 200:
        raise RuntimeError(f"/stt HTTP {status}: {body[:200]!r}")
    return wall, json.loads(body)


def tts(url: str, text: str, fmt: str) -> tuple[float, Dict[str, Any], bytes]:
    t0 = time.perf_counter()
    status, headers, body = _request(f"{url}/tts?format={fmt}", data=text.encode(),
                                     headers={"Content-Type": "text/plain; charset=utf-8"})
    wall = (time.perf_counter() - t0) * 1000
    if status != 200:
        raise RuntimeError(f"/tts HTTP {status}: {body[:200]!r}")
    timing = json.loads(headers["x-timing-ms"])
    return wall, {"timing_ms": timing, "audio_s": float(headers["x-audio-seconds"])}, body


# ─── WER ──────────────────────────────────────────────────────────

def _norm(text: str) -> List[str]:
    text = text.lower().replace("~", " tilde ")
    text = re.sub(r"[^a-z0-9åäöé\s]", " ", text)
    return text.split()


def wer(reference: str, hypothesis: str) -> float:
    r, h = _norm(reference), _norm(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


# ─── clip generation ─────────────────────────────────────────────

def make_clips(url: str) -> None:
    """Synthetic corpus: the daemon's own TTS speaks bench/texts, re-encoded like a
    chat-app voice note (48 kHz Opus) or plain 16 kHz WAV. Deterministic enough
    for latency; WER against synthetic speech is optimistic, treat it as a tripwire."""
    CLIPS.mkdir(parents=True, exist_ok=True)
    for name, spec in CLIP_SPECS.items():
        out = CLIPS / f"{name}.{spec['fmt']}"
        ref = CLIPS / f"{name}.txt"
        if spec["text"] is None:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                            "-i", f"anullsrc=r={spec['rate']}:cl=mono", "-t", str(spec["seconds"]),
                            "-c:a", "libopus", "-b:a", "32k", str(out)], check=True)
            ref.write_text("")
            print(f"  {out.name}: {spec['seconds']} s silence")
            continue
        names = spec["text"] if isinstance(spec["text"], list) else [spec["text"]]
        text = " ".join((TEXTS / n).read_text().strip() for n in names)
        _wall, _meta, wav = tts(url, text, "wav")
        codec = (["-c:a", "libopus", "-b:a", "32k"] if spec["fmt"] == "ogg" else ["-c:a", "pcm_s16le"])
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "pipe:0", "-ar", str(spec["rate"]),
                        "-ac", "1", *codec, str(out)], input=wav, check=True)
        ref.write_text(text + "\n")
        print(f"  {out.name}: {_meta['audio_s']} s from {'+'.join(names)}")


def _clip_seconds(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


# ─── suites ──────────────────────────────────────────────────────

def _med(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.median(vals) if vals else None


def bench_stt(url: str, runs: int, language: Optional[str]) -> List[Dict[str, Any]]:
    rows = []
    clips = sorted(p for p in CLIPS.iterdir() if p.suffix in (".ogg", ".wav"))
    if not clips:
        raise SystemExit("no clips in bench/clips — run: scripts/bench.py make-clips")
    for clip in clips:
        audio = clip.read_bytes()
        ref_path = clip.with_suffix(".txt")
        ref = ref_path.read_text().strip() if ref_path.exists() else None
        seconds = _clip_seconds(clip)
        samples = []
        text = ""
        for _ in range(runs):
            wall, body = stt(url, audio, language)
            t = body["timing_ms"]
            text = body["text"]
            samples.append({"wall": wall, "decode": t["decode"], "model": t["model"],
                            "total": t["total"]})
        row = {"clip": clip.name, "audio_s": round(seconds, 2), "bytes": len(audio),
               "first_ms": samples[0]["wall"],
               "wall_ms": _med(samples, "wall"), "decode_ms": _med(samples, "decode"),
               "model_ms": _med(samples, "model"), "daemon_ms": _med(samples, "total"),
               "text": text}
        row["rtf"] = round(row["wall_ms"] / 1000 / seconds, 3) if seconds else None
        row["wer"] = round(wer(ref, text), 3) if ref is not None else None
        rows.append(row)
    return rows


def bench_tts(url: str, runs: int, fmt: str) -> List[Dict[str, Any]]:
    rows = []
    texts = sorted(TEXTS.glob("*.txt"))
    if not texts:
        raise SystemExit("no texts in bench/texts")
    for path in texts:
        text = path.read_text().strip()
        samples = []
        audio_s = 0.0
        for _ in range(runs):
            wall, meta, body = tts(url, text, fmt)
            t = meta["timing_ms"]
            audio_s = meta["audio_s"]
            samples.append({"wall": wall, "model": t["model"], "encode": t["encode"],
                            "first": t.get("first_chunk"), "total": t["total"], "bytes": len(body)})
        row = {"text": path.name, "chars": len(text), "audio_s": audio_s,
               "first_ms": samples[0]["wall"],
               "wall_ms": _med(samples, "wall"), "first_chunk_ms": _med(samples, "first"),
               "model_ms": _med(samples, "model"), "encode_ms": _med(samples, "encode"),
               "daemon_ms": _med(samples, "total"), "bytes": int(_med(samples, "bytes") or 0)}
        row["rtf"] = round(row["wall_ms"] / 1000 / audio_s, 3) if audio_s else None
        rows.append(row)
    return rows


# ─── output ──────────────────────────────────────────────────────

def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.0f}" if v >= 100 else f"{v:.2f}" if v < 10 else f"{v:.1f}"
    return str(v)


def table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


STT_COLS = ["clip", "audio_s", "first_ms", "wall_ms", "decode_ms", "model_ms", "rtf", "wer"]
TTS_COLS = ["text", "chars", "audio_s", "first_ms", "wall_ms", "first_chunk_ms", "model_ms", "encode_ms", "rtf"]


# ─── private daemon ──────────────────────────────────────────────

def spawn_daemon(port: int, env_overrides: Dict[str, str], resident: str) -> subprocess.Popen:
    env = {**os.environ, **env_overrides, "VOICE_PORT": str(port), "VOICE_RESIDENT": resident,
           "PYTHONPATH": str(ROOT), "PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    python = ROOT / ".venv" / "bin" / "python"
    proc = subprocess.Popen([str(python), str(ROOT / "voice_daemon.py"), "--port", str(port)],
                            env=env, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 600
    while time.time() < deadline:
        if proc.poll() is not None:
            err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            raise SystemExit(f"spawned daemon died:\n{err[-2000:]}")
        try:
            body = health(url)
            if all(body["engines"][k]["loaded"] for k in (["stt", "tts"] if resident == "both" else [resident])):
                return proc
        except (urllib.error.URLError, SystemExit, ConnectionError, OSError):
            pass
        time.sleep(1)
    proc.kill()
    raise SystemExit("spawned daemon did not become healthy in 600 s")


# ─── compute-type comparison ─────────────────────────────────────

def compare_compute_types(compute_types: List[str], runs: int, language: Optional[str],
                          port: int) -> Dict[str, Any]:
    """Bench the STT suite once per compute type, one private daemon at a time.

    Sequential by construction: each daemon is started, measured and killed
    before the next is started. On a single-GPU host an overlapping run would
    make both the VRAM and the latency numbers meaningless.
    """
    results: Dict[str, Any] = {}
    for compute_type in compute_types:
        print(f"\n=== {compute_type} ===", file=sys.stderr)
        proc = spawn_daemon(port, {"WHISPER_COMPUTE_TYPE": compute_type}, "stt")
        url = f"http://127.0.0.1:{port}"
        try:
            h = health(url)
            stt_engine = h["engines"]["stt"]
            if stt_engine.get("compute_type") != compute_type:
                raise SystemExit(f"daemon reports compute_type "
                                 f"{stt_engine.get('compute_type')!r}, asked for {compute_type!r}")
            results[compute_type] = {
                "model_vram_mb": stt_engine.get("vram_mb"),
                "daemon_vram_mb": h["vram_mb"]["used"],
                "rows": {r["clip"]: r for r in bench_stt(url, runs, language)},
            }
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            # The next daemon must not start while this one still holds the
            # model, or its "before" VRAM reading includes the previous model.
            time.sleep(3)
    return results


def compare_table(results: Dict[str, Any], baseline: str) -> str:
    """Per-clip WER and wall time for each type, plus the delta against *baseline*."""
    others = [t for t in results if t != baseline]
    cols = ["clip", "audio_s", f"wer[{baseline}]", f"ms[{baseline}]"]
    for t in others:
        cols += [f"wer[{t}]", "dwer", f"ms[{t}]", "ratio"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for clip, base_row in results[baseline]["rows"].items():
        cells = [clip, _fmt(base_row["audio_s"]),
                 _fmt(base_row["wer"]), _fmt(base_row["wall_ms"])]
        for t in others:
            row = results[t]["rows"][clip]
            dwer = (None if row["wer"] is None or base_row["wer"] is None
                    else round(row["wer"] - base_row["wer"], 3))
            ratio = (round(row["wall_ms"] / base_row["wall_ms"], 2)
                     if base_row["wall_ms"] else None)
            cells += [_fmt(row["wer"]), _fmt(dwer), _fmt(row["wall_ms"]), _fmt(ratio)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compare_vram_table(results: Dict[str, Any], baseline: str) -> str:
    base_model = results[baseline]["model_vram_mb"]
    lines = ["| compute_type | model_vram_mb | vs baseline | daemon_used_mb |",
             "|---|---|---|---|"]
    for t, res in results.items():
        model = res["model_vram_mb"]
        ratio = (f"{model / base_model:.2f}x" if model and base_model else "-")
        lines.append(f"| {t} | {_fmt(model)} | {ratio} | {_fmt(res['daemon_vram_mb'])} |")
    return "\n".join(lines)


# ─── main ────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="bench", choices=["bench", "make-clips", "compare"])
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--suite", choices=["stt", "tts", "both"], default="both")
    ap.add_argument("--runs", type=int, default=3, help="per item; median reported")
    ap.add_argument("--language", default="en", help="STT language hint ('' for auto-detect)")
    ap.add_argument("--format", default="ogg", help="TTS output format for the bench")
    ap.add_argument("--spawn", action="append", default=[], metavar="KEY=VAL",
                    help="start a private daemon with this env (repeatable) instead of --url")
    ap.add_argument("--spawn-port", type=int, default=9879)
    ap.add_argument("--json", action="store_true", help="also print raw rows as JSON")
    ap.add_argument("--compute-type", action="append", default=[], metavar="TYPE",
                    help="compare mode: bench this compute type (repeatable; first is the baseline)")
    args = ap.parse_args()

    if args.command == "compare":
        types = args.compute_type or ["float16", "int8_float16"]
        if len(types) < 2:
            raise SystemExit("compare needs at least two --compute-type values")
        results = compare_compute_types(types, args.runs, args.language or None, args.spawn_port)
        baseline = types[0]
        print(f"\nSTT compute types (median of {args.runs}, language={args.language or 'auto'}, "
              f"baseline={baseline})\n")
        print(compare_vram_table(results, baseline))
        print()
        print(compare_table(results, baseline))
        for t, res in results.items():
            print(f"\n{t}:")
            for clip, row in res["rows"].items():
                print(f"  {clip}: {row['text'][:110]!r}")
        if args.json:
            print(json.dumps(results, indent=1))
        return

    url = args.url
    proc = None
    if args.spawn:
        overrides = dict(kv.split("=", 1) for kv in args.spawn)
        resident = args.suite
        print(f"spawning daemon on :{args.spawn_port} with {overrides} (resident={resident})", file=sys.stderr)
        proc = spawn_daemon(args.spawn_port, overrides, resident)
        url = f"http://127.0.0.1:{args.spawn_port}"

    try:
        h = health(url)
        if args.command == "make-clips":
            print(f"generating clips via {url}")
            make_clips(url)
            return
        print(f"daemon: {url}")
        print(f"engines: {json.dumps(h['engines'], separators=(',', ':'))}")
        print(f"vram_mb: {h['vram_mb']}  warmup_ms: stt={h['timing_ms'].get('warmup_stt')} "
              f"tts={h['timing_ms'].get('warmup_tts')}")
        out: Dict[str, Any] = {}
        if args.suite in ("stt", "both"):
            rows = bench_stt(url, args.runs, args.language or None)
            out["stt"] = rows
            print(f"\nSTT (median of {args.runs}, language={args.language or 'auto'})\n")
            print(table(rows, STT_COLS))
            for r in rows:
                print(f"  {r['clip']}: {r['text'][:110]!r}")
        if args.suite in ("tts", "both"):
            rows = bench_tts(url, args.runs, args.format)
            out["tts"] = rows
            print(f"\nTTS (median of {args.runs}, format={args.format})\n")
            print(table(rows, TTS_COLS))
        if args.json:
            print(json.dumps(out, indent=1))
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
