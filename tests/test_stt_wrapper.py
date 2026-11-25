"""`bin/hermes-stt` query-string behaviour — the shell script, run for real.

Covers the `stt-language-detect` change, scenarios "Wrapper passes the
language argument through" and "Wrapper omits the parameter for auto". The
engine tests prove the daemon does the right thing with `?language=`; nothing
proved the wrapper actually sends it, and the auto/absent asymmetry is a shell
conditional that no Python test touches.

The script is run as a subprocess against a stdlib `http.server` standing in for
the daemon, so the assertion is on the request path the daemon would really see.

Run:  .venv/bin/python -m unittest tests.test_stt_wrapper -v
"""

import json
import shutil
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin/hermes-stt"


class _Recorder(BaseHTTPRequestHandler):
    paths: list = []

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's naming
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        type(self).paths.append(self.path)
        body = json.dumps({"ok": True, "text": "x"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002 - matches the base signature
        pass  # keep the test output clean


@unittest.skipUnless(shutil.which("curl"), "hermes-stt shells out to curl")
@unittest.skipUnless(SCRIPT.is_file(), "bin/hermes-stt not present")
class WrapperQueryStringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = "http://127.0.0.1:%d" % cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        _Recorder.paths = []
        self.audio = Path(self.enterContext(tempfile.TemporaryDirectory())) / "clip.ogg"
        self.audio.write_bytes(b"not really audio")

    def _run(self, *extra_args):
        proc = subprocess.run(
            [str(SCRIPT), str(self.audio), "/dev/null", *extra_args],
            env={"PATH": "/usr/bin:/bin", "VOICE_URL": self.url, "VOICE_TIMEOUT": "20"},
            capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(len(_Recorder.paths), 1, "exactly one request")
        return _Recorder.paths[0], proc.stdout

    def test_no_language_argument_sends_no_query_parameter(self):
        path, stdout = self._run()
        self.assertEqual(path, "/stt")
        self.assertEqual(stdout.strip(), "x", "transcript is echoed to stdout")

    def test_auto_sends_no_query_parameter(self):
        path, _ = self._run("auto")
        self.assertEqual(path, "/stt")

    def test_a_named_language_is_passed_through(self):
        path, _ = self._run("sv")
        self.assertEqual(path, "/stt?language=sv")

    def test_english_is_passed_through_too(self):
        # 'en' is not special-cased anywhere: only 'auto' and empty mean detect.
        path, _ = self._run("en")
        self.assertEqual(path, "/stt?language=en")

    def test_empty_language_argument_sends_no_query_parameter(self):
        path, _ = self._run("")
        self.assertEqual(path, "/stt")


if __name__ == "__main__":
    unittest.main()
