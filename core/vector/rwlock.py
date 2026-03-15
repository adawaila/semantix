"""Read-write lock (readers-writer lock).

Allows multiple concurrent readers OR one exclusive writer — never both.
Writers are preferred: once a writer is waiting, new readers are blocked
to prevent writer starvation.

Usage
-----
    lock = RWLock()

    # Concurrent reads
    with lock.read():
        data = index.search(query)

    # Exclusive write
    with lock.write():
        index.add(vector)
"""
from __future__ import annotations

import threading
from contextlib import contextmanager


class RWLock:
    """Writer-preferring read-write lock.

    Properties
    ----------
    - Multiple threads can hold the read lock simultaneously.
    - The write lock is exclusive: no readers or other writers allowed.
    - Writers are preferred: a waiting writer blocks all new readers,
      preventing writer starvation under read-heavy workloads.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers: int = 0        # active reader count
        self._writers_waiting: int = 0
        self._writing: bool = False

    # ------------------------------------------------------------------
    # Low-level acquire / release
    # ------------------------------------------------------------------

    def acquire_read(self) -> None:
        """Block until the read lock is acquired."""
        with self._cond:
            # Block while a writer holds or is waiting for the lock
            while self._writing or self._writers_waiting > 0:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        """Release the read lock; wake a waiting writer if last reader."""
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        """Block until the write lock is exclusively acquired."""
        with self._cond:
            self._writers_waiting += 1
            while self._writing or self._readers > 0:
                self._cond.wait()
            self._writers_waiting -= 1
            self._writing = True

    def release_write(self) -> None:
        """Release the write lock; wake all waiting readers and writers."""
        with self._cond:
            self._writing = False
            self._cond.notify_all()

    # ------------------------------------------------------------------
    # Context-manager interface  (preferred)
    # ------------------------------------------------------------------

    @contextmanager
    def read(self):
        """Context manager for a shared read lock."""
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def write(self):
        """Context manager for an exclusive write lock."""
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()
