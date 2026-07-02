"""Per-run file logging. Independent of the rich console UI in ui.py.

One detailed log file is opened per `python run.py` invocation under `logs/`.
The console keeps showing the clean progress lines; everything verbose
(HTTP request/response, subprocess stderr, server manifests, full tracebacks)
goes into the log file. Grep that file after a failure.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def make_run_id(job_id: str) -> str:
    """`<YYYYMMDD_HHMMSS>_<job_id>`. Used for both the log filename and the per-run output dir."""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job_id}"


def setup(run_id: str, log_root: Path = Path("logs")) -> Path:
    """Configure root logging to write a detailed file log for this run.

    Returns the log file path so the caller can surface it on exit.
    """
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{run_id}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)

    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return log_path
