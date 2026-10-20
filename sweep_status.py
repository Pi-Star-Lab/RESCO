#!/usr/bin/env python3
"""Show progress for the max-green sweep without touching the running job."""

import json
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "max_green_sweep_2026-10-13"
EPISODE_RE = re.compile(r"Episode:\s*(\d+)")


def last_episode(log_path: Path) -> int:
    # Logs are small enough to read, and this avoids assumptions about line size.
    matches = EPISODE_RE.findall(log_path.read_text(errors="replace"))
    return int(matches[-1]) if matches else 0


def main() -> None:
    logs = list(RESULTS.glob("*/git_FINAL/*/console.log"))
    if not logs:
        raise SystemExit(f"No active or completed runs found in {RESULTS}")

    current_log = max(logs, key=lambda path: path.stat().st_mtime)
    run_path = current_log.parent
    config = json.loads((run_path / "config.json").read_text())

    episode = last_episode(current_log)
    training_total = int(config.get("episodes", 0))
    testing_total = int(config.get("testing", 0))
    training_done = min(episode, training_total)
    testing_done = min(max(episode - training_total, 0), testing_total)

    algorithm = config.get("algorithm", "unknown")
    action_set = config.get("action_set", "unknown")
    max_green = config.get("ring_max_green")
    through_max = config.get("ring_through_max_green")

    print(f"Updated:          {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}")
    print(f"Current model:    {algorithm}")
    print(f"Action set:       {action_set}")
    if action_set == "RingAndBarrier":
        print(f"Max green:        {max_green} seconds")
        print(f"Through max:      {through_max} seconds")

    if episode <= training_total:
        percent = 100 * training_done / training_total if training_total else 100
        print(f"Stage:            training")
        print(f"Training:         {training_done}/{training_total} completed ({percent:.1f}%)")
        print(f"Currently on:     episode {min(training_done + 1, training_total)}")
    else:
        percent = 100 * testing_done / testing_total if testing_total else 100
        print(f"Stage:            testing")
        print(f"Training:         {training_total}/{training_total} completed (100.0%)")
        print(f"Testing:          {testing_done}/{testing_total} completed ({percent:.1f}%)")

    print(f"Run directory:    {run_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
