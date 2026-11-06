"""Hand-off point between the inference thread and any number of async web clients."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from dataclasses import dataclass, field


@dataclass(slots=True)
class Snapshot:
    seq: int = 0
    jpeg: bytes | None = None
    state: dict = field(default_factory=dict)


class ResultBus:
    """Holds the latest annotated frame + state and wakes async subscribers.

    ``publish`` is called from the worker thread; ``wait_next`` is awaited by
    web handlers. All clients share one inference loop, so cost doesn't scale
    with the number of viewers.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snap = Snapshot(state={"status": "starting", "people": []})
        self._loop: asyncio.AbstractEventLoop | None = None
        self._event: asyncio.Event | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._event = asyncio.Event()

    def publish(self, jpeg: bytes | None, state: dict) -> None:
        with self._lock:
            seq = self._snap.seq + 1
            self._snap = Snapshot(seq, jpeg if jpeg is not None else self._snap.jpeg, state)
            state["seq"] = seq
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._wake)

    def update_state(self, **fields) -> None:
        """Merge fields into the current state (e.g. status/error) and notify."""
        with self._lock:
            state = {**self._snap.state, **fields}
        self.publish(None, state)

    def _wake(self) -> None:
        if self._event is not None:
            self._event.set()
            self._event = asyncio.Event()

    @property
    def latest(self) -> Snapshot:
        with self._lock:
            return self._snap

    async def wait_next(self, after_seq: int, timeout: float = 5.0) -> Snapshot:
        """Wait until a snapshot newer than ``after_seq`` exists (or timeout)."""
        if self._event is None:
            raise RuntimeError("ResultBus.attach_loop() was not called")
        snap = self.latest
        if snap.seq > after_seq:
            return snap
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._event.wait(), timeout)
        return self.latest
