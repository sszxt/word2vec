"""Plain periodic print-based progress reporting.

tqdm's carriage-return-updated single line renders poorly once output isn't
a real interactive terminal (piped to a log, captured by a harness, etc) --
every refresh becomes its own line, flooding the log. This prints one clean
line every `every` steps or `min_interval` seconds instead.
"""

from __future__ import annotations

import time


class LogProgress:
    def __init__(self, total: int, desc: str, every: int = 1000, min_interval: float = 5.0):
        self.total = total
        self.desc = desc
        self.every = every
        self.min_interval = min_interval
        self.start = time.time()
        self._last_print = self.start

    def update(self, step: int, **fields) -> None:
        now = time.time()
        is_last = step == self.total
        if not is_last and step % self.every != 0 and (now - self._last_print) < self.min_interval:
            return
        self._last_print = now
        elapsed = now - self.start
        rate = step / elapsed if elapsed > 0 else 0.0
        pct = 100.0 * step / self.total if self.total else 100.0
        extras = " ".join(f"{k}={v}" for k, v in fields.items())
        print(f"{self.desc} {step}/{self.total} ({pct:5.1f}%) {rate:6.0f} it/s {extras}")
