"""Serve the dashboard, and optionally launch experiment runs from it.

    python scripts/serve_dashboard.py              # view only
    VGP_ALLOW_RUN=1 python scripts/serve_dashboard.py   # view + launch

Standard library only, so the viewing image stays free of vectorbt and numba.

SECURITY. POST /api/run starts a process, which is a command-execution surface,
so it is built to refuse by default and to have no string path into a shell:

  * Launching is OFF unless VGP_ALLOW_RUN=1. Serving is always safe.
  * Every parameter is a number, range-checked against PARAM_SPEC before use.
    Nothing from the request becomes an argument, a path or a shell word — the
    argv is fixed and the values travel as environment variables.
  * subprocess is called with a list and shell=False.
  * One run at a time, and never into the committed results/ directory.

Even so, bind this to a port you control. It is a local research tool, not a
service: there is no authentication, and anyone who can reach the port and has
VGP_ALLOW_RUN set can spend your CPU for several hours.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DASH_DIR = PROJECT_DIR / "dashboard"
RUNS_DIR = PROJECT_DIR / "runs"

PORT = int(os.environ.get("VGP_PORT", "8000"))
BIND = os.environ.get("VGP_BIND", "0.0.0.0")
ALLOW_RUN = os.environ.get("VGP_ALLOW_RUN", "") == "1"

# name -> (env var, min, max, default, integer?). The bounds are not cosmetic:
# they are what stops a request asking for a run that never finishes.
PARAM_SPEC: dict[str, tuple[str, float, float, float, bool]] = {
    "fee_bps": ("VGP_FEE_BPS", 0.0, 100.0, 10.0, False),
    "n_null_runs": ("VGP_N_NULL_RUNS", 0, 999, 99, True),
    "pop_size": ("VGP_POP_SIZE", 10, 1000, 200, True),
    "n_gen": ("VGP_N_GEN", 1, 200, 30, True),
    "n_seeds": ("VGP_N_SEEDS", 1, 10, 3, True),
    "oos_months": ("VGP_OOS_MONTHS", 1, 12, 4, True),
    "train_months": ("VGP_TRAIN_MONTHS", 3, 36, 9, True),
}


class RunState:
    """The single in-flight run, if any. Guarded by its own lock."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.started: float | None = None
        self.params: dict | None = None
        self.log_path: Path | None = None
        self.out_dir: Path | None = None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def snapshot(self) -> dict:
        with self.lock:
            running = self.is_running()
            rc = None if running or self.proc is None else self.proc.returncode
            return {
                "running": running,
                "returncode": rc,
                "started": self.started,
                "elapsed": None if self.started is None else round(time.time() - self.started),
                "params": self.params,
                "out_dir": None if self.out_dir is None else str(self.out_dir.name),
                "log": self._tail(),
            }

    def _tail(self, n_bytes: int = 8000) -> str:
        if self.log_path is None or not self.log_path.exists():
            return ""
        data = self.log_path.read_bytes()[-n_bytes:]
        text = data.decode("utf-8", errors="replace")
        # tqdm rewrites one line with \r; keep the last state of each, not every frame.
        lines = [ln.split("\r")[-1] for ln in text.splitlines()]
        return "\n".join(lines[-60:])


STATE = RunState()


def validate(payload: dict) -> dict[str, float]:
    """Coerce and range-check every parameter. Raises ValueError on anything else."""
    out: dict[str, float] = {}
    for name, (_env, lo, hi, default, is_int) in PARAM_SPEC.items():
        raw = payload.get(name, default)
        try:
            val = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{name}: not a number")
        if val != val or val in (float("inf"), float("-inf")):
            raise ValueError(f"{name}: not finite")
        if not (lo <= val <= hi):
            raise ValueError(f"{name}: {val:g} outside [{lo:g}, {hi:g}]")
        out[name] = int(val) if is_int else val
    unknown = set(payload) - set(PARAM_SPEC)
    if unknown:
        raise ValueError(f"unknown parameter(s): {', '.join(sorted(unknown))}")
    return out


def experiment_available() -> tuple[bool, str]:
    """Can this interpreter actually run the experiment?

    The viewing image deliberately ships without vectorbt, so the honest answer
    there is no — and the UI should say so rather than offering a button that
    fails.
    """
    try:
        import vectorbt  # noqa: F401
    except Exception as exc:
        return False, f"vectorbt is not importable here ({type(exc).__name__})"
    if not (PROJECT_DIR / "scripts" / "run.py").exists():
        return False, "scripts/run.py is not present in this image"
    return True, ""


def start_run(params: dict[str, float]) -> dict:
    with STATE.lock:
        if STATE.is_running():
            raise RuntimeError("a run is already in progress")

        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = RUNS_DIR / f"run-{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "run.log"

        # Values travel as environment variables, never as argv or a shell word.
        env = os.environ.copy()
        for name, (var, *_rest) in PARAM_SPEC.items():
            env[var] = str(params[name])
        env["VGP_RESULTS_DIR"] = str(out_dir)
        env["PYTHONUNBUFFERED"] = "1"

        (out_dir / "params.json").write_text(json.dumps(params, indent=2) + "\n")

        log = open(log_path, "wb")
        proc = subprocess.Popen(  # noqa: S603 — fixed argv, shell=False, bounded env
            [sys.executable, str(PROJECT_DIR / "scripts" / "run.py")],
            cwd=str(PROJECT_DIR),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            shell=False,
            start_new_session=True,
        )
        STATE.proc = proc
        STATE.started = time.time()
        STATE.params = params
        STATE.log_path = log_path
        STATE.out_dir = out_dir
        return {"ok": True, "pid": proc.pid, "out_dir": out_dir.name}


def cancel_run() -> dict:
    with STATE.lock:
        if not STATE.is_running():
            return {"ok": False, "error": "nothing is running"}
        # start_new_session put the child in its own process group, so the whole
        # worker pool goes down with it rather than being orphaned.
        os.killpg(os.getpgid(STATE.proc.pid), signal.SIGTERM)
        return {"ok": True}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(DASH_DIR), **kw)

    def log_message(self, fmt, *args):  # quieter than the default one-line-per-asset
        if "/api/" in (self.path or "") or self.command == "POST":
            sys.stderr.write(f"{self.command} {self.path}\n")

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] == "/api/state":
            can_run, reason = experiment_available()
            return self._json(
                {
                    "allow_run": ALLOW_RUN,
                    "can_run": can_run and ALLOW_RUN,
                    "reason": (
                        ""
                        if can_run and ALLOW_RUN
                        else (
                            reason or "launching is disabled; start with VGP_ALLOW_RUN=1 to enable"
                        )
                    ),
                    "params": {
                        k: {"min": lo, "max": hi, "default": d, "int": i, "env": e}
                        for k, (e, lo, hi, d, i) in PARAM_SPEC.items()
                    },
                    "run": STATE.snapshot(),
                }
            )
        return super().do_GET()

    def do_POST(self):
        path = self.path.split("?")[0]
        if path not in ("/api/run", "/api/cancel"):
            return self._json({"error": "no such endpoint"}, 404)
        if not ALLOW_RUN:
            return self._json(
                {"error": "launching is disabled. Restart with VGP_ALLOW_RUN=1."}, 403
            )
        if path == "/api/cancel":
            return self._json(cancel_run())

        length = int(self.headers.get("Content-Length") or 0)
        if length > 10_000:
            return self._json({"error": "payload too large"}, 413)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            params = validate(payload)
        except ValueError as exc:
            return self._json({"error": str(exc)}, 400)
        except Exception:
            return self._json({"error": "malformed JSON"}, 400)

        ok, reason = experiment_available()
        if not ok:
            return self._json({"error": reason}, 409)
        try:
            return self._json(start_run(params))
        except RuntimeError as exc:
            return self._json({"error": str(exc)}, 409)


def main() -> int:
    if not (DASH_DIR / "index.html").exists():
        print(
            "dashboard/index.html is missing — run `make dashboard` first.",
            file=sys.stderr,
        )
        return 2
    can_run, reason = experiment_available()
    print(f"VGP dashboard on http://localhost:{PORT}")
    if ALLOW_RUN and can_run:
        print("  launching ENABLED — this port can start multi-hour runs")
    elif ALLOW_RUN:
        print(f"  launching requested but unavailable: {reason}")
    else:
        print("  launching disabled (set VGP_ALLOW_RUN=1 to enable)")
    HTTPServer((BIND, PORT), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
