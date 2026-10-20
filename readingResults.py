#!/usr/bin/env python3
"""Summarize the testing episodes from a completed RESCO run."""

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_FILE = Path(__file__).resolve().parent / "new_results.txt"
CONFIG_HASH_IGNORES = {
    "uuid", "log_dir", "names_map", "hashed_name", "run_name", "run_path",
    "processors", "script_launcher", "log_level", "save_console_log",
    "compress_results", "trials", "error_bars", "min_max",
    "savgol_smoothing", "skip_unnamed", "xml_metrics", "csv_metrics",
    "optuna_trials", "episodes", "converged", "smoothing", "benchmark_path",
    "delete_episode_logs", "init_ender", "save_model", "save_log", "home_log",
}


def result_files(results_directory: Path):
    candidates = [
        path
        for path in results_directory.glob("*.json")
        if not path.name.endswith("_testing_statistics.json")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No RESCO result JSON files were found in {results_directory}"
        )
    return sorted(candidates, key=lambda path: path.name)


def finite_numbers(values):
    numbers = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            numbers.append(number)
    return numbers


def summarize(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "standard_deviation": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def testing_statistics(results, run_configs, fallback_testing_episodes=None):
    summaries = {}
    for metric, runs in results.items():
        if not isinstance(runs, dict):
            continue
        metric_summaries = {}
        for run_id, all_values in runs.items():
            if not isinstance(all_values, list):
                continue
            testing_episodes = run_configs.get(run_id, {}).get("testing")
            if testing_episodes is None:
                testing_episodes = fallback_testing_episodes
            if not isinstance(testing_episodes, int) or testing_episodes <= 0:
                raise ValueError(
                    f"No valid testing episode count was found for run {run_id!r}. "
                    "Keep its config.json or provide --testing-episodes as a fallback."
                )
            values = finite_numbers(all_values[-testing_episodes:])
            metric_summaries[run_id] = summarize(values)
            metric_summaries[run_id]["testing_episodes"] = testing_episodes
        if metric_summaries:
            summaries[metric] = metric_summaries
    if not summaries:
        raise ValueError("The input does not contain any RESCO metric episode data")
    return summaries


def config_hash(config):
    hashable = config.copy()
    if "load_model" in hashable:
        hashable["load_model"] = True
    for key in CONFIG_HASH_IGNORES:
        hashable.pop(key, None)
    return hashlib.sha3_256(str(hashable).encode("utf-8")).hexdigest()


def find_run_configs(results_directory, run_ids):
    run_configs = {}
    unmatched = set(run_ids)
    for config_path in results_directory.rglob("config.json"):
        try:
            with config_path.open(encoding="utf-8") as config_file:
                config = json.load(config_file)
            run_id = config_hash(config)
        except (OSError, ValueError, TypeError):
            continue
        if run_id in unmatched:
            algorithm = str(config.get("algorithm", "Unknown"))
            action_set = str(config.get("action_set", "Unknown"))
            if algorithm.upper() == "FIXED":
                action_set = "FIXED"
            run_configs[run_id] = {
                "action_set": action_set,
                "episodes": config.get("episodes"),
                "testing": config.get("testing"),
            }
            unmatched.remove(run_id)
            if not unmatched:
                break
    return run_configs


def format_table(headers, rows):
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def line(row):
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    divider = "-+-".join("-" * width for width in widths)
    return "\n".join((line(headers), divider, *(line(row) for row in rows)))


def format_report(run_summaries, run_configs):
    headers = (
        "Run", "Action set", "Test episodes", "Count", "Mean", "Median",
        "Std dev", "Min", "Max"
    )
    rows_by_metric = {}

    def format_number(value):
        return "N/A" if value is None else f"{value:.6f}"

    for source, summaries in run_summaries:
        for metric, runs in summaries.items():
            for run_id, values in runs.items():
                rows_by_metric.setdefault(metric, []).append(
                    (
                        source.stem,
                        run_configs.get(run_id, {}).get("action_set", "Unknown"),
                        str(values["testing_episodes"]),
                        str(values["count"]),
                        format_number(values["mean"]),
                        format_number(values["median"]),
                        format_number(values["standard_deviation"]),
                        format_number(values["minimum"]),
                        format_number(values["maximum"]),
                    )
                )
    sections = []
    for metric, rows in rows_by_metric.items():
        sections.append(f"{metric}\n{'=' * len(metric)}\n{format_table(headers, rows)}")
    report_tables = "\n\n".join(sections)

    return (
        "RESCO Testing Episode Statistics\n"
        f"Generated at: {datetime.now(timezone.utc).isoformat()}\n\n"
        f"{report_tables}\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate testing statistics for all completed RESCO runs."
    )
    parser.add_argument(
        "results_file",
        nargs="?",
        type=Path,
        help="one RESCO result JSON (default: every JSON in ./new_results)",
    )
    parser.add_argument(
        "-n",
        "--testing-episodes",
        type=int,
        help="fallback test count only when a run's config.json cannot be found",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.testing_episodes is not None and args.testing_episodes <= 0:
        raise SystemExit("--testing-episodes must be greater than zero")

    if args.results_file:
        sources = [args.results_file.expanduser().resolve()]
        if not sources[0].is_file():
            raise FileNotFoundError(f"Result file does not exist: {sources[0]}")
        results_directory = sources[0].parent
    else:
        results_directory = (Path(__file__).resolve().parent / "new_results")
        sources = result_files(results_directory)

    loaded_results = []
    run_ids = set()
    for source in sources:
        with source.open(encoding="utf-8") as input_file:
            results = json.load(input_file)
        loaded_results.append((source, results))
        for runs in results.values():
            if isinstance(runs, dict):
                run_ids.update(runs)

    run_configs = find_run_configs(results_directory, run_ids)
    run_summaries = [
        (
            source,
            testing_statistics(results, run_configs, args.testing_episodes),
        )
        for source, results in loaded_results
    ]
    report = format_report(run_summaries, run_configs)
    with OUTPUT_FILE.open("w", encoding="utf-8") as output_file:
        output_file.write(report)

    print(f"Testing statistics saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
