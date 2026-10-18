"""Progress lines for logs read with `tail -f`."""

from __future__ import annotations

import time


def progress_line(done: int, total: int, t_start: float, unit: str) -> str:
    """'  [HH:MM] done/total unit, elapsed s, eta m min'."""
    elapsed = time.time() - t_start
    eta = f", eta {elapsed / done * (total - done) / 60:.0f} min" if done else ""
    return f"  [{time.strftime('%H:%M')}] {done}/{total} {unit}, {elapsed:.0f} s{eta}"
