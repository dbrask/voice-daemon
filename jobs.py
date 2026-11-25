"""Long-form TTS jobs: chaptering, state, on-disk layout, TTL cleanup.

This module is bookkeeping only. It knows about job state, chapter text and
the cache directory; it knows nothing about engines, PCM, ffmpeg or HTTP.
`voice_daemon` owns synthesis and calls back into a `Job` to record progress.
That split keeps the GPU code in one file and makes chaptering and the store
unit-testable without an engine.

On-disk layout (`$XDG_CACHE_HOME/voice-bot/jobs`, TTL-cleaned):

    <jobs_dir>/<job_id>/job.json          snapshot, rewritten on every update
    <jobs_dir>/<job_id>/001-chapter.ogg   one file per chapter, written as it
    <jobs_dir>/<job_id>/002-chapter.ogg   finishes (never a whole-document blob)

Nothing here is a document store: a job is transient output with a TTL, there
are no bookmarks, no resume and no index across jobs.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Job lifecycle. `cancelled` and `error` are terminal like `done`; a job never
# leaves a terminal state.
QUEUED, RUNNING, DONE, CANCELLED, ERROR = "queued", "running", "done", "cancelled", "error"
TERMINAL = (DONE, CANCELLED, ERROR)

# A markdown ATX heading on its own line is the chapter marker. Callers that
# want exact control post an explicit `chapters` array instead.
_HEADING = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
_JOB_ID = re.compile(r"^[0-9a-f]{8,32}$")


class JobCancelled(Exception):
    """Raised inside a running job when someone cancelled it."""


def valid_job_id(job_id: str) -> bool:
    """True for ids this module minted. Guards path building against traversal."""
    return bool(_JOB_ID.match(job_id or ""))


def _slug(title: str, fallback: str) -> str:
    slug = _SLUG_STRIP.sub("-", title.lower()).strip("-")[:40].strip("-")
    return slug or fallback


def split_chapters(text: str, split: str = "headings") -> List[Tuple[str, str]]:
    """`(title, text)` per chapter. Markdown headings are the markers.

    Text before the first heading becomes its own untitled chapter, so a
    document with a preamble does not lose it. `split="none"` returns the whole
    text as one chapter. Chapters whose body is empty (a heading followed by
    another heading) are dropped: there is nothing to speak.
    """
    text = text.strip()
    if not text:
        return []
    if split == "none":
        return [("", text)]
    chapters: List[Tuple[str, List[str]]] = []
    current_title, current_lines = "", []
    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            chapters.append((current_title, current_lines))
            current_title, current_lines = match.group(2).strip(), []
        else:
            current_lines.append(line)
    chapters.append((current_title, current_lines))
    out = []
    for title, lines in chapters:
        body = "\n".join(lines).strip()
        if body:
            out.append((title, body))
    return out


def normalize_chapters(payload: Iterable[Any]) -> List[Tuple[str, str]]:
    """`(title, text)` from an explicit `chapters` array of strings or objects."""
    out: List[Tuple[str, str]] = []
    for item in payload:
        if isinstance(item, str):
            title, body = "", item
        elif isinstance(item, dict):
            title, body = str(item.get("title") or ""), str(item.get("text") or "")
        else:
            raise ValueError("each chapter must be a string or {title, text} object")
        body = body.strip()
        if body:
            out.append((title.strip(), body))
    return out


@dataclass
class Chapter:
    index: int  # 1-based, matches the filename prefix
    title: str
    text: str
    chars: int = 0
    status: str = QUEUED
    path: Optional[str] = None
    audio_s: float = 0.0
    error: Optional[str] = None

    def __post_init__(self) -> None:
        self.chars = len(self.text)

    def snapshot(self) -> Dict[str, Any]:
        return {"index": self.index, "title": self.title, "chars": self.chars,
                "status": self.status, "path": self.path, "audio_s": self.audio_s,
                "error": self.error}


@dataclass
class Job:
    id: str
    dir: Path
    chapters: List[Chapter]
    voice: Optional[str] = None
    format: str = "ogg"
    status: str = QUEUED
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    updated_at: float = field(default_factory=time.time)
    chars_done: int = 0
    chars_in_flight: int = 0
    audio_s: float = 0.0
    _cancel: bool = False

    # ---- state transitions (all of them touch updated_at, none of them do IO
    # except the snapshot write, so the daemon can call these from anywhere) --

    @property
    def chars_total(self) -> int:
        return sum(c.chars for c in self.chapters)

    @property
    def cancelled(self) -> bool:
        return self._cancel

    def cancel(self) -> bool:
        """Request cancellation. False if the job already finished."""
        if self.status in TERMINAL:
            return False
        self._cancel = True
        if self.status == QUEUED:  # never started: nothing to interrupt
            self.finish(CANCELLED)
        else:
            self._touch()
        return True

    def raise_if_cancelled(self) -> None:
        if self._cancel:
            raise JobCancelled(f"job {self.id} cancelled")

    def start(self) -> None:
        self.status, self.started_at = RUNNING, time.time()
        self._touch()

    def chapter_path(self, chapter: Chapter) -> Path:
        name = f"{chapter.index:03d}-{_slug(chapter.title, f'chapter-{chapter.index}')}.{self.format}"
        return self.dir / name

    def chapter_started(self, chapter: Chapter) -> None:
        chapter.status = RUNNING
        self.chars_in_flight = 0
        self._touch()

    def chapter_progress(self, chars: int) -> None:
        """Mid-chapter progress, in chunk chars.

        Chunk lengths do not re-add the whitespace `split_sentences` collapsed,
        so this undercounts slightly; `chapter_done` corrects to the chapter's
        real char count, which is why 100 % is exact at the end.
        """
        self.chars_in_flight += chars
        self._touch(write=False)  # per chunk: too hot for a disk write

    def chapter_done(self, chapter: Chapter, path: Path, audio_s: float) -> None:
        chapter.status, chapter.path, chapter.audio_s = DONE, str(path), audio_s
        self.audio_s = round(self.audio_s + audio_s, 2)
        self.chars_done += chapter.chars  # exact, replaces the chunk estimate
        self.chars_in_flight = 0
        self._touch()

    def chapter_failed(self, chapter: Chapter, error: str) -> None:
        chapter.status, chapter.error = ERROR, error
        self.chars_in_flight = 0
        self._touch()

    def finish(self, status: str, error: Optional[str] = None) -> None:
        self.status, self.error, self.finished_at = status, error, time.time()
        for chapter in self.chapters:
            if chapter.status in (QUEUED, RUNNING):
                chapter.status = CANCELLED if status == CANCELLED else status
        self._touch()

    def _touch(self, write: bool = True) -> None:
        self.updated_at = time.time()
        if write:
            self.write_snapshot()

    # ---- views ----------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        total = self.chars_total
        done = min(self.chars_done + self.chars_in_flight, total)
        return {
            "id": self.id,
            "status": self.status,
            "error": self.error,
            "voice": self.voice,
            "format": self.format,
            "dir": str(self.dir),
            "created_at": round(self.created_at, 3),
            "started_at": round(self.started_at, 3) if self.started_at else None,
            "finished_at": round(self.finished_at, 3) if self.finished_at else None,
            "updated_at": round(self.updated_at, 3),
            "elapsed_s": round((self.finished_at or time.time()) - (self.started_at or self.created_at), 2),
            "audio_s": self.audio_s,
            "progress": {
                "chapters_total": len(self.chapters),
                "chapters_done": sum(1 for c in self.chapters if c.status == DONE),
                "chars_total": total,
                "chars_done": done,
                "percent": round(100.0 * done / total, 1) if total else 0.0,
            },
            "chapters": [c.snapshot() for c in self.chapters],
        }

    def write_snapshot(self) -> None:
        try:
            (self.dir / "job.json").write_text(json.dumps(self.snapshot(), indent=2))
        except OSError:  # a job whose dir was swept out from under it still runs
            pass


class JobStore:
    """In-memory job registry over a TTL-cleaned cache directory.

    Deliberately not persistent: a daemon restart loses job state, and the
    sweeper then removes the orphaned directories once they age past the TTL.
    Long-form output is a cache, not a library.
    """

    def __init__(self, root: Path, ttl_s: float = 24 * 3600, max_jobs: int = 50):
        self.root = Path(root)
        self.ttl_s = ttl_s
        self.max_jobs = max_jobs
        self._jobs: Dict[str, Job] = {}

    def create(self, chapters: List[Tuple[str, str]], voice: Optional[str] = None,
               fmt: str = "ogg") -> Job:
        if not chapters:
            raise ValueError("no chapters to speak")
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job = Job(
            id=job_id, dir=job_dir, voice=voice, format=fmt,
            chapters=[Chapter(index=i, title=title, text=text)
                      for i, (title, text) in enumerate(chapters, start=1)],
        )
        self._jobs[job_id] = job
        job.write_snapshot()
        self._trim()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id) if valid_job_id(job_id) else None

    def list(self) -> List[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def delete(self, job_id: str) -> bool:
        """Cancel if running, then drop the job and its audio."""
        job = self.get(job_id)
        if job is None:
            return False
        job.cancel()
        self._jobs.pop(job_id, None)
        shutil.rmtree(job.dir, ignore_errors=True)
        return True

    def purge(self, now: Optional[float] = None) -> List[str]:
        """Drop finished jobs past the TTL and any orphan dirs. Returns ids removed.

        Orphans are directories with no live job — what a daemon restart leaves
        behind. They are only removed once their own mtime is past the TTL, so a
        job from a still-running older process is not deleted underneath it.
        """
        now = now if now is not None else time.time()
        removed: List[str] = []
        for job in list(self._jobs.values()):
            age_from = job.finished_at or job.created_at
            if job.status in TERMINAL and now - age_from > self.ttl_s:
                self._jobs.pop(job.id, None)
                shutil.rmtree(job.dir, ignore_errors=True)
                removed.append(job.id)
        if self.root.is_dir():
            for path in self.root.iterdir():
                if not path.is_dir() or path.name in self._jobs:
                    continue
                try:
                    if now - path.stat().st_mtime > self.ttl_s:
                        shutil.rmtree(path, ignore_errors=True)
                        removed.append(path.name)
                except OSError:
                    continue
        return removed

    def _trim(self) -> None:
        """Keep at most max_jobs; the oldest terminal jobs go first."""
        finished = [j for j in self.list() if j.status in TERMINAL]
        while len(self._jobs) > self.max_jobs and finished:
            self.delete(finished.pop().id)
