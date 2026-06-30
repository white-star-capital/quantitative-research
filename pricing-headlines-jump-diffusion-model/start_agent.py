#!/usr/bin/env python3
"""
Hourly scheduler for the jump-diffusion pricing engine.

Runs the full multi-outcome pipeline (discover top geopolitical markets → news →
local Ollama analysis → pricing → save JSON) immediately on start, then once
every hour. Each cycle shells out to the engine's own __main__, so there is no
re-orchestration here — this file is purely the scheduling loop.

Run:  python start_agent.py   (Ctrl-C to stop)
Stops cleanly on SIGINT or SIGTERM (e.g. `make stop`), terminating any
in-flight engine subprocess so nothing is left orphaned.
"""

import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = "pricing_engine_multi_outcome.py"
INTERVAL_SECONDS = 3600  # 1 hour

# Reference to the currently-running engine subprocess, so the signal handler
# can forward a termination signal to it on shutdown rather than orphaning it.
_current_child = None
# Set by the signal handler to break the main loop on the next check.
_stop = False


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _shutdown(signum, frame):
    """Request shutdown: flag the loop and forward SIGTERM to the engine child.

    Deliberately does NOT wait/reap here. This handler can fire while the main
    flow is blocked in Popen.wait() (which holds Popen's internal waitpid lock);
    reaping from inside the handler would deadlock on that lock and just spin out
    the timeout. Instead we signal the child and return — the interrupted wait()
    in run_once() resumes and reaps it cleanly, and the loop sees `_stop`.
    """
    global _stop
    _stop = True
    name = signal.Signals(signum).name
    print(f"\n👋 [{_now()}] Received {name}, shutting down...", flush=True)
    child = _current_child
    if child is not None:
        try:
            os.kill(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # child already gone


def run_once() -> None:
    """Run one full engine cycle as a subprocess. Never raises."""
    global _current_child
    engine_path = os.path.join(HERE, ENGINE)
    print(f"\n{'=' * 80}")
    print(f"🚀 [{_now()}] Starting pricing run: {ENGINE}")
    print(f"{'=' * 80}", flush=True)

    started = time.monotonic()
    try:
        _current_child = subprocess.Popen([sys.executable, engine_path], cwd=HERE)
        returncode = _current_child.wait()
        elapsed = time.monotonic() - started
        if _stop:
            print(f"\n🛑 [{_now()}] Run interrupted by shutdown", flush=True)
        else:
            status = "✅ OK" if returncode == 0 else f"⚠️  exit {returncode}"
            print(f"\n{status} [{_now()}] Run finished in {elapsed/60:.1f} min", flush=True)
    except Exception as e:
        elapsed = time.monotonic() - started
        print(f"\n❌ [{_now()}] Run failed after {elapsed/60:.1f} min: {e}", flush=True)
    finally:
        _current_child = None


def main() -> None:
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"🕐 Pricing agent started at {_now()} — running every {INTERVAL_SECONDS // 60} min")
    print("   Press Ctrl-C to stop.")

    while not _stop:
        cycle_start = time.monotonic()
        run_once()
        if _stop:
            break

        # Keep an ~hourly cadence regardless of how long the run took.
        elapsed = time.monotonic() - cycle_start
        sleep_for = max(0, INTERVAL_SECONDS - elapsed)
        next_run = datetime.now() + timedelta(seconds=sleep_for)
        print(f"\n⏱️  Sleeping {sleep_for/60:.1f} min — next run at {next_run.strftime('%H:%M:%S')}", flush=True)

        # Interruptible sleep so a stop signal is honored within ~1s.
        deadline = time.monotonic() + sleep_for
        while not _stop and time.monotonic() < deadline:
            time.sleep(min(1.0, deadline - time.monotonic()))

    print(f"\n👋 [{_now()}] Pricing agent stopped.", flush=True)


if __name__ == "__main__":
    main()
