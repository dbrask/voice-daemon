"""Pi install layout: the README's copy commands produce what the unit runs.

Runs the README's documented Pi copy commands into a throwaway HOME, then
checks that every path the docs and the unit refer to exists there and that
the unit's `python3 -m satellite.client` resolves from its WorkingDirectory.
No Pi, no audio, no network: a filesystem and module-resolution check only.

Run: .venv/bin/python -m unittest discover -s tests
"""

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UNIT = REPO / "satellite" / "pi" / "satellite-client.service"
PI_MARKER = "# on the Pi"


def readme_pi_block() -> list[str]:
    """The indented setup-sketch lines from `# on the Pi` to the end of the block."""
    lines = (REPO / "README.md").read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith(PI_MARKER))
    block = []
    for line in lines[start + 1:]:
        if line and not line.startswith("    "):
            break
        block.append(line.strip())
    return [line for line in block if line]


def copy_commands() -> list[str]:
    """The shell lines that put the package on the Pi (not comments, not systemctl)."""
    return [line for line in readme_pi_block()
            if not line.startswith("#") and re.match(r"(mkdir|cp|rsync)\b", line)
            and "systemd" not in line]


def unit_copy_sources() -> list[str]:
    """Every `cp <src> ~/.config/systemd/user/` source in the README and kiosk.md."""
    texts = [(REPO / "README.md").read_text(), (REPO / "satellite" / "pi" / "kiosk.md").read_text()]
    found = []
    for text in texts:
        found += re.findall(r"cp\s+(\S+satellite-client\.service)\s+~/\.config/systemd/user/", text)
    return found


def unit_value(key: str) -> str:
    for line in UNIT.read_text().splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1]
    raise AssertionError(f"{key} missing from {UNIT.name}")


class TestPiLayout(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        cmds = copy_commands()
        self.assertTrue(cmds, "the README's Pi setup has no runnable copy command")
        env = {"HOME": str(self.home), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        for cmd in cmds:
            # Run from the repository root, as the README says; only HOME is written.
            subprocess.run(["bash", "-euc", cmd], cwd=REPO, env=env, check=True)

    def expand(self, path: str) -> Path:
        return Path(path.replace("%h", str(self.home)).replace("~", str(self.home), 1))

    def test_the_documented_unit_copy_source_exists(self):
        sources = unit_copy_sources()
        self.assertGreaterEqual(len(sources), 2, "README and kiosk.md both copy the unit")
        for src in sources:
            with self.subTest(src=src):
                self.assertTrue(self.expand(src).is_file(), f"{src} is not there after the copy")

    def test_the_unit_documentation_path_exists(self):
        doc = unit_value("Documentation").removeprefix("file:")
        self.assertTrue(self.expand(doc).is_file(), doc)

    def test_the_unit_module_imports_from_its_working_directory(self):
        workdir = self.expand(unit_value("WorkingDirectory"))
        exec_start = unit_value("ExecStart")
        module = re.search(r"-m\s+(\S+)", exec_start).group(1)
        self.assertEqual(module, "satellite.client")
        env = {"HOME": str(self.home), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        # -s: no user site; no PYTHONPATH: only the working directory can provide `satellite`.
        proc = subprocess.run([sys.executable, "-s", "-m", module, "--help"],
                              cwd=workdir, env=env, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # And it is the copy, not the repository checkout, that was imported.
        proc = subprocess.run(
            [sys.executable, "-s", "-c", "import satellite.client as c; print(c.__file__)"],
            cwd=workdir, env=env, capture_output=True, text=True, check=True)
        self.assertTrue(Path(proc.stdout.strip()).is_relative_to(self.home), proc.stdout)

    def test_the_face_page_is_copied(self):
        workdir = self.expand(unit_value("WorkingDirectory"))
        self.assertTrue((workdir / "satellite" / "face" / "index.html").is_file())


if __name__ == "__main__":
    unittest.main()
