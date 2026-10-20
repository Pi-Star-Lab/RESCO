#!/usr/bin/env python3
"""Plot training-episode timeLoss for the runs read by readingResults.py."""

import argparse
import json
import math
import os
import re
import warnings
from pathlib import Path

# The workspace may not have a writable user-level Matplotlib config directory.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/resco-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from readingResults import find_run_configs, result_files


DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "figures"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot timeLoss by episode for the same result files used by "
            "readingResults.py. Training episodes are plotted by default."
        )
    )
    parser.add_argument(
        "results_file",
        nargs="?",
        type=Path,
        help="one RESCO result JSON (default: every JSON in ./results)",
    )
    parser.add_argument(
        "-n",
        "--testing-episodes",
        type=int,
        help=(
            "fallback test count when a run's config.json cannot be found; "
            "the test tail is removed from the plot"
        ),
    )
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="include testing episodes instead of showing training only",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=f"directory for plot images (default: {DEFAULT_OUTPUT_DIRECTORY.name}/)",
    )
    return parser.parse_args()


def finite_points(values):
    """Return episode/value pairs without joining a line across invalid data."""
    points = []
    for episode, value in enumerate(values, start=1):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            points.append((episode, math.nan))
            continue
        number = float(value)
        points.append((episode, number if math.isfinite(number) else math.nan))
    return points


def training_values(values, config, fallback_testing_episodes):
    """Select the training prefix using config metadata when it is available."""
    configured_episodes = config.get("episodes")
    if isinstance(configured_episodes, int) and configured_episodes >= 0:
        return values[:configured_episodes]

    testing = config.get("testing", fallback_testing_episodes)
    if not isinstance(testing, int) or testing < 0:
        raise ValueError(
            "No valid training/test episode count was found. Keep the run's "
            "config.json or provide --testing-episodes."
        )
    return values[:-testing] if testing else values


def safe_filename(value):
    """Make a short, portable component for an output filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "run"


def main():
    args = parse_args()
    if args.testing_episodes is not None and args.testing_episodes < 0:
        raise SystemExit("--testing-episodes cannot be negative")

    if args.results_file:
        sources = [args.results_file.expanduser().resolve()]
        if not sources[0].is_file():
            raise FileNotFoundError(f"Result file does not exist: {sources[0]}")
        results_directory = sources[0].parent
    else:
        results_directory = Path(__file__).resolve().parent / "results"
        sources = result_files(results_directory)

    loaded = []
    run_ids = set()
    for source in sources:
        with source.open(encoding="utf-8") as result_file:
            results = json.load(result_file)
        loaded.append((source, results))
        time_loss = results.get("timeLoss", {})
        if isinstance(time_loss, dict):
            run_ids.update(time_loss)

    run_configs = find_run_configs(results_directory, run_ids)
    output_directory = args.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    plotted = 0

    for source, results in loaded:
        runs = results.get("timeLoss", {})
        if not isinstance(runs, dict):
            continue
        for run_id, values in runs.items():
            if not isinstance(values, list) or not values:
                continue
            if not args.all_episodes:
                try:
                    values = training_values(
                        values, run_configs.get(run_id, {}), args.testing_episodes
                    )
                except ValueError as error:
                    warnings.warn(f"Skipping {source.name} ({run_id[:8]}): {error}")
                    continue
            points = finite_points(values)
            if not points or not any(math.isfinite(y) for _, y in points):
                continue

            config = run_configs.get(run_id, {})
            action_set = config.get("action_set", "Unknown action set")
            fig, ax = plt.subplots(figsize=(11, 6.5))
            ax.plot(
                [x for x, _ in points],
                [y for _, y in points],
                linewidth=1.35,
            )
            ax.set_xlabel("Episode")
            ax.set_ylabel("Mean timeLoss per vehicle (s)")
            episode_kind = "all episodes" if args.all_episodes else "training episodes"
            ax.set_title(f"{action_set} timeLoss ({episode_kind}) · {run_id[:8]}")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()

            filename = f"timeLoss_{safe_filename(action_set)}_{run_id[:8]}.png"
            fig.savefig(output_directory / filename, dpi=180)
            plt.close(fig)
            plotted += 1

    if not plotted:
        raise ValueError("No non-empty timeLoss episode data was found to plot")

    print(f"Plotted {plotted} run(s) to: {output_directory}")


if __name__ == "__main__":
    main()
