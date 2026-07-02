"""Console output. Rich progress lines and tables matching the demo script in §8 of the spec."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from rich.console import Console
from rich.table import Table

console = Console()


@contextmanager
def stage(num: int, total: int, label: str) -> Iterator[dict]:
    """Print '[n/total] label...', then on success append ✓ with elapsed time + optional detail."""
    state: dict = {"detail": ""}
    console.print(rf"[bold cyan]\[{num}/{total}][/] {label}...", end="")
    t0 = time.monotonic()
    try:
        yield state
    except Exception:
        console.print("  [red]✗ failed[/]")
        raise
    dt = time.monotonic() - t0
    detail = f"  [dim]{state['detail']}[/]" if state["detail"] else ""
    console.print(f"  [green]✓[/] [dim]{_fmt(dt)}[/]{detail}")


def cached(num: int, total: int, label: str) -> None:
    console.print(rf"[bold cyan]\[{num}/{total}][/] {label}...  [yellow]✓ cached[/]")


def clips_table(clips: list[dict]) -> None:
    t = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    t.add_column("#", justify="right", style="dim")
    t.add_column("score", justify="right")
    t.add_column("range", justify="right")
    t.add_column("hook")
    for c in clips:
        t.add_row(
            str(c.get("rank", "")),
            str(c.get("score", "")),
            f"{int(c['start'])}-{int(c['end'])}",
            (c.get("hook") or "")[:60],
        )
    console.print(t)


def _fmt(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s) // 60}m {int(s) % 60}s"
