"""bin/voice-speak — desk TTS hotkey.

Route/chunk logic is imported from the script (stdlib only). The CLI is
exercised as a subprocess against a stdlib HTTP stub, with fake pw-play /
wl-paste / notify-send on PATH.

Run:  .venv/bin/python -m unittest tests.test_voice_speak -v
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin/voice-speak"


def _load():
    if not SCRIPT.is_file():
        raise AssertionError("bin/voice-speak missing")
    sys.dont_write_bytecode = True
    loader = importlib.machinery.SourceFileLoader("voice_speak", str(SCRIPT))
    spec = importlib.util.spec_from_loader("voice_speak", loader)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load bin/voice-speak")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vs = _load()

    def test_short_text_uses_batch_tts(self):
        self.assertEqual(self.vs.route_for("Hello there."), "/tts")
        self.assertEqual(self.vs.route_for("x" * 1500), "/tts")

    def test_text_over_1500_chars_uses_long_jobs(self):
        self.assertEqual(self.vs.route_for("x" * 1501), "/tts/long")

    def test_voice_query_is_appended(self):
        url = self.vs.build_url("http://127.0.0.1:9877", "/tts", voice="af_heart")
        self.assertEqual(url, "http://127.0.0.1:9877/tts?format=wav&voice=af_heart")

    def test_no_voice_keeps_format_only(self):
        url = self.vs.build_url("http://127.0.0.1:9877", "/tts/long", voice="")
        self.assertEqual(url, "http://127.0.0.1:9877/tts/long?format=wav")


class ResolveTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vs = _load()

    def test_file_argument_wins(self):
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory())) / "t.txt"
        tmp.write_text("from file", encoding="utf-8")
        text = self.vs.resolve_text(
            file_arg=str(tmp), stdin_isatty=False, stdin_read=lambda: "from stdin",
            paste=lambda: "from clip")
        self.assertEqual(text, "from file")

    def test_piped_stdin_is_used_when_no_file(self):
        text = self.vs.resolve_text(
            file_arg=None, stdin_isatty=False, stdin_read=lambda: "Hello there, this is the desk voice.\n",
            paste=lambda: "from clip")
        self.assertEqual(text, "Hello there, this is the desk voice.\n")

    def test_tty_defaults_to_clipboard(self):
        text = self.vs.resolve_text(
            file_arg=None, stdin_isatty=True, stdin_read=lambda: self.fail("must not block on tty stdin"),
            paste=lambda: "clip text")
        self.assertEqual(text, "clip text")

    def test_empty_pipe_falls_back_to_clipboard(self):
        # Hyprland bind: stdin is /dev/null, not a tty.
        text = self.vs.resolve_text(
            file_arg=None, stdin_isatty=False, stdin_read=lambda: "",
            paste=lambda: "clip text")
        self.assertEqual(text, "clip text")


class PlayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vs = _load()

    def test_prefers_pw_play(self):
        which = lambda name: "/usr/bin/pw-play" if name == "pw-play" else None
        self.assertEqual(self.vs.player_argv(which), ["pw-play"])

    def test_falls_back_to_ffplay(self):
        which = lambda name: "/usr/bin/ffplay" if name == "ffplay" else None
        self.assertEqual(self.vs.player_argv(which), ["ffplay", "-nodisp", "-autoexit"])


# ---- subprocess against a stub daemon ------------------------------------

WAV = b"RIFF____WAVEfmt "  # not a real wav; the player fake never decodes it


class _Stub(BaseHTTPRequestHandler):
    """In-memory daemon: /tts returns bytes, /tts/long returns a finished job."""

    bodies: list
    paths: list
    job_dir: Path
    fail_connect: bool = False

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        type(self).bodies.append(body)
        type(self).paths.append(self.path)
        parsed = urlparse(self.path)
        if parsed.path == "/tts":
            self._send(200, WAV, "audio/wav")
            return
        if parsed.path == "/tts/long":
            job_id = "job1"
            chap = type(self).job_dir / "001.wav"
            chap.write_bytes(WAV)
            payload = {
                "job": {
                    "id": job_id,
                    "status": "queued",
                    "progress": {"chapters_total": 2, "chapters_done": 0, "percent": 0},
                    "chapters": [],
                }
            }
            self._send(202, json.dumps(payload).encode(), "application/json")
            return
        self._send(404, b"no", "text/plain")

    def do_GET(self):  # noqa: N802
        type(self).paths.append(self.path)
        parsed = urlparse(self.path)
        if parsed.path == "/jobs/job1":
            a = type(self).job_dir / "001.wav"
            b = type(self).job_dir / "002.wav"
            a.write_bytes(WAV)
            b.write_bytes(WAV)
            payload = {
                "job": {
                    "id": "job1",
                    "status": "done",
                    "progress": {"chapters_total": 2, "chapters_done": 2, "percent": 100},
                    "chapters": [
                        {"index": 1, "status": "done", "path": str(a)},
                        {"index": 2, "status": "done", "path": str(b)},
                    ],
                }
            }
            self._send(200, json.dumps(payload).encode(), "application/json")
            return
        self._send(404, b"no", "text/plain")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        pass


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _write_exe(path: Path, body: str):
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class CliStubTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.job_dir = Path(tempfile.mkdtemp(prefix="voice-speak-job-"))
        _Stub.job_dir = cls.job_dir
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = "http://127.0.0.1:%d" % cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        shutil.rmtree(cls.job_dir, ignore_errors=True)

    def setUp(self):
        _Stub.bodies = []
        _Stub.paths = []
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.bindir = self.tmp / "bin"
        self.bindir.mkdir()
        self.playlog = self.tmp / "play.log"
        self.notifylog = self.tmp / "notify.log"
        self.runtime = self.tmp / "run"
        self.runtime.mkdir()
        _write_exe(self.bindir / "pw-play", f"#!/bin/sh\necho PLAY \"$1\" >> '{self.playlog}'\n")
        _write_exe(self.bindir / "notify-send", f"#!/bin/sh\necho \"$@\" >> '{self.notifylog}'\n")
        _write_exe(self.bindir / "wl-paste", "#!/bin/sh\nprintf 'from clipboard'\n")
        self.env = {
            "PATH": str(self.bindir) + ":/usr/bin:/bin",
            "VOICE_URL": self.url,
            "VOICE_TIMEOUT": "10",
            "XDG_RUNTIME_DIR": str(self.runtime),
            "HOME": str(self.tmp),
        }

    def _run(self, args, input_text=None, timeout=15):
        return subprocess.run(
            [str(SCRIPT), *args],
            input=input_text,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def test_stdin_posts_short_text_to_tts_wav_and_plays(self):
        proc = self._run([], input_text="Hello there, this is the desk voice.")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(len(_Stub.paths), 1)
        self.assertEqual(_Stub.paths[0], "/tts?format=wav")
        self.assertEqual(_Stub.bodies[0], b"Hello there, this is the desk voice.")
        plays = self.playlog.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(plays), 1)
        self.assertTrue(plays[0].startswith("PLAY "))

    def test_voice_flag_is_honoured(self):
        proc = self._run(["--voice", "af_bella"], input_text="hi")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(_Stub.paths[0], "/tts?format=wav&voice=af_bella")

    def test_clipboard_when_no_input(self):
        proc = self._run([])  # no stdin text, fake wl-paste
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(_Stub.bodies[0], b"from clipboard")

    def test_file_argument(self):
        src = self.tmp / "note.txt"
        src.write_text("file body", encoding="utf-8")
        proc = self._run([str(src)])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(_Stub.bodies[0], b"file body")

    def test_long_text_posts_tts_long_and_plays_chapters_in_order(self):
        text = "L" * 1501
        proc = self._run([], input_text=text)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(any(p.startswith("/tts/long?format=wav") for p in _Stub.paths), _Stub.paths)
        self.assertTrue(any(p == "/jobs/job1" for p in _Stub.paths), _Stub.paths)
        plays = [ln.split(" ", 1)[1] for ln in self.playlog.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(plays), 2, plays)
        self.assertTrue(plays[0].endswith("001.wav"))
        self.assertTrue(plays[1].endswith("002.wav"))

    def test_daemon_down_exits_1_with_systemctl_hint(self):
        self.env["VOICE_URL"] = "http://127.0.0.1:%d" % _free_port()
        proc = self._run([], input_text="hello")
        self.assertEqual(proc.returncode, 1)
        err = proc.stderr.strip()
        self.assertIn("cannot reach voice daemon", err)
        self.assertIn("systemctl --user start voice-daemon", err)
        self.assertNotIn("\n", err, "one-line stderr hint")
        notify = self.notifylog.read_text(encoding="utf-8")
        self.assertTrue(notify.strip(), "notify-send on error")


class StopTests(unittest.TestCase):
    """--stop must kill a sleeping player within 1 s."""

    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), "bin/voice-speak missing")
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.bindir = self.tmp / "bin"
        self.bindir.mkdir()
        self.runtime = self.tmp / "run"
        self.runtime.mkdir()
        self.playlog = self.tmp / "play.log"
        self.player_pid_file = self.tmp / "player.pid"
        self.wavdir = self.tmp / "wav"
        self.wavdir.mkdir()
        # Slow player so --stop has something to kill. Record $$ before exec so
        # the test can prove the player child died, not just the parent.
        _write_exe(
            self.bindir / "pw-play",
            (
                "#!/bin/sh\n"
                f"echo $$ > '{self.player_pid_file}'\n"
                f"echo PLAY >> '{self.playlog}'\n"
                "exec sleep 30\n"
            ),
        )
        _write_exe(self.bindir / "notify-send", "#!/bin/sh\nexit 0\n")
        self.job_dir = self.tmp / "job"
        self.job_dir.mkdir()
        _Stub.job_dir = self.job_dir
        _Stub.bodies = []
        _Stub.paths = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._shutdown)
        self.env = {
            "PATH": str(self.bindir) + ":/usr/bin:/bin",
            "VOICE_URL": "http://127.0.0.1:%d" % self.server.server_address[1],
            "VOICE_TIMEOUT": "10",
            "XDG_RUNTIME_DIR": str(self.runtime),
            "TMPDIR": str(self.wavdir),
            "HOME": str(self.tmp),
        }

    def _shutdown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def test_stop_silences_within_one_second(self):
        speak = subprocess.Popen(
            [str(SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=self.env,
            text=True,
        )
        assert speak.stdin is not None
        speak.stdin.write("Hello there, this is the desk voice.")
        speak.stdin.close()
        # Wait until the player has started (pidfile + play log).
        deadline = time.monotonic() + 5
        pidfile = self.runtime / "voice-speak.pid"
        while time.monotonic() < deadline:
            if pidfile.is_file() and self.playlog.is_file():
                break
            time.sleep(0.05)
        self.assertTrue(pidfile.is_file(), "pidfile not written")
        t0 = time.monotonic()
        stop = subprocess.run(
            [str(SCRIPT), "--stop"],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(stop.returncode, 0, stop.stderr)
        try:
            speak.wait(timeout=2)
        except subprocess.TimeoutExpired:
            speak.kill()
            self.fail("speak did not exit after --stop")
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertNotEqual(speak.returncode, 0)
        player_pid = int(self.player_pid_file.read_text(encoding="utf-8").strip())
        self._assert_pid_gone(player_pid)

    def test_stop_unlinks_temp_wav(self):
        speak = subprocess.Popen(
            [str(SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=self.env,
            text=True,
        )
        assert speak.stdin is not None
        speak.stdin.write("Hello there, this is the desk voice.")
        speak.stdin.close()
        deadline = time.monotonic() + 5
        pidfile = self.runtime / "voice-speak.pid"
        while time.monotonic() < deadline:
            if pidfile.is_file() and self.playlog.is_file():
                break
            time.sleep(0.05)
        self.assertTrue(pidfile.is_file(), "pidfile not written")
        wavs = list(self.wavdir.glob("voice-speak-*.wav"))
        self.assertTrue(wavs, "expected a temp wav during playback")
        stop = subprocess.run(
            [str(SCRIPT), "--stop"],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(stop.returncode, 0, stop.stderr)
        try:
            speak.wait(timeout=2)
        except subprocess.TimeoutExpired:
            speak.kill()
            self.fail("speak did not exit after --stop")
        leftover = list(self.wavdir.glob("voice-speak-*.wav"))
        self.assertEqual(leftover, [], leftover)

    def test_direct_sigterm_kills_player_and_unlinks_runtime(self):
        # Hyprland/systemd/`kill <pid>` hit _on_term, not --stop. --stop kills
        # PLAYER first so proc.wait() == -15 and main's Interrupted branch
        # cleans up before the parent's SIGTERM lands — that path does not
        # observe _on_term's body.
        speak = subprocess.Popen(
            [str(SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=self.env,
            text=True,
        )
        assert speak.stdin is not None
        speak.stdin.write("Hello there, this is the desk voice.")
        speak.stdin.close()
        deadline = time.monotonic() + 5
        pidfile = self.runtime / "voice-speak.pid"
        while time.monotonic() < deadline:
            if (
                pidfile.is_file()
                and self.playlog.is_file()
                and list(self.wavdir.glob("voice-speak-*.wav"))
            ):
                break
            time.sleep(0.05)
        self.assertTrue(pidfile.is_file(), "pidfile not written")
        self.assertTrue(self.playlog.is_file(), "player did not start")
        wavs = list(self.wavdir.glob("voice-speak-*.wav"))
        self.assertTrue(wavs, "expected a temp wav during playback")
        player_pid = int(self.player_pid_file.read_text(encoding="utf-8").strip())
        os.kill(speak.pid, signal.SIGTERM)
        try:
            speak.wait(timeout=2)
        except subprocess.TimeoutExpired:
            speak.kill()
            self.fail("speak did not exit after SIGTERM")
        self.assertEqual(speak.returncode, 143)
        self._assert_pid_gone(player_pid)
        leftover = list(self.wavdir.glob("voice-speak-*.wav"))
        self.assertEqual(leftover, [], leftover)
        self.assertFalse(pidfile.is_file(), "pidfile should be unlinked")

    def test_stop_stale_pidfile_does_not_kill_unrelated_process(self):
        sleeper = subprocess.Popen(["sleep", "30"])
        self.addCleanup(lambda: sleeper.kill() if sleeper.poll() is None else None)
        pidfile = self.runtime / "voice-speak.pid"
        pidfile.write_text(f"{sleeper.pid}\n", encoding="utf-8")
        stop = subprocess.run(
            [str(SCRIPT), "--stop"],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(stop.returncode, 0, stop.stderr)
        self.assertIsNone(sleeper.poll(), "stale pidfile killed an unrelated pid")
        self.assertFalse(pidfile.is_file(), "stale pidfile should be unlinked")
        sleeper.terminate()
        sleeper.wait(timeout=2)

    def test_stop_stale_player_pid_does_not_kill_unrelated_process(self):
        # Between chapters, PLAYER= can point at a pid that has already
        # exited and been reused. --stop must not SIGTERM that process.
        speak = subprocess.Popen(
            [str(SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=self.env,
            text=True,
        )
        assert speak.stdin is not None
        speak.stdin.write("Hello there, this is the desk voice.")
        speak.stdin.close()
        deadline = time.monotonic() + 5
        pidfile = self.runtime / "voice-speak.pid"
        while time.monotonic() < deadline:
            if pidfile.is_file() and self.playlog.is_file():
                break
            time.sleep(0.05)
        self.assertTrue(pidfile.is_file(), "pidfile not written")
        sleeper = subprocess.Popen(["sleep", "30"])
        self.addCleanup(lambda: sleeper.kill() if sleeper.poll() is None else None)
        pidfile.write_text(f"{speak.pid}\nPLAYER={sleeper.pid}\n", encoding="utf-8")
        stop = subprocess.run(
            [str(SCRIPT), "--stop"],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(stop.returncode, 0, stop.stderr)
        self.assertIsNone(sleeper.poll(), "stale PLAYER= killed an unrelated pid")
        try:
            speak.wait(timeout=2)
        except subprocess.TimeoutExpired:
            speak.kill()
            self.fail("speak did not exit after --stop")
        sleeper.terminate()
        sleeper.wait(timeout=2)

    def _assert_pid_gone(self, pid: int, timeout: float = 1.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.02)
        self.fail(f"player pid {pid} still alive")


if __name__ == "__main__":
    unittest.main()
