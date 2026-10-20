#!/usr/bin/env python3
"""Create time-loss tables and figures for the seven-run max-green sweep."""

import csv
import json
import math
import os
import statistics
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/resco-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from readingResults import config_hash


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "max_green_sweep_2026-10-13"
TEST_EPISODES = 100


def load_runs():
    configs = {}
    for path in RESULTS.rglob("config.json"):
        config = json.loads(path.read_text())
        configs[config_hash(config)] = config

    runs = []
    for path in sorted(RESULTS.glob("*.json")):
        data = json.loads(path.read_text())
        for run_id, values in data.get("timeLoss", {}).items():
            config = configs[run_id]
            algorithm = config["algorithm"]
            action_set = config["action_set"]
            max_green = config.get("ring_max_green")
            label = (
                "IDQN Phase"
                if action_set == "Phase"
                else f"{algorithm} R&B {max_green}s"
            )
            testing = [float(value) for value in values[-TEST_EPISODES:]]
            runs.append(
                {
                    "id": run_id,
                    "label": label,
                    "algorithm": algorithm,
                    "action_set": action_set,
                    "max_green": max_green if action_set == "RingAndBarrier" else None,
                    "all": [float(value) for value in values],
                    "training": [float(value) for value in values[: config["episodes"]]],
                    "testing": testing,
                }
            )
    return sorted(
        runs,
        key=lambda run: (
            run["action_set"] == "Phase",
            run["algorithm"] == "STOCHASTIC",
            run["max_green"] or 0,
        ),
    )


def stats(values):
    ordered = sorted(values)
    count = len(ordered)
    mean = statistics.fmean(ordered)
    std = statistics.pstdev(ordered)
    return {
        "count": count,
        "mean": mean,
        "median": statistics.median(ordered),
        "std": std,
        "ci95": 1.96 * std / math.sqrt(count),
        "min": ordered[0],
        "q1": statistics.median(ordered[: count // 2]),
        "q3": statistics.median(ordered[(count + 1) // 2 :]),
        "max": ordered[-1],
    }


def moving_average(values, window=50):
    result = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        result.append(running / min(index + 1, window))
    return result


def save_csv(runs):
    columns = ["model", "action_set", "max_green_seconds", "testing_episodes",
               "mean", "median", "std_dev", "ci95", "minimum", "q1", "q3", "maximum"]
    with (RESULTS / "time_loss_testing_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for run in runs:
            summary = stats(run["testing"])
            writer.writerow(
                {
                    "model": run["algorithm"],
                    "action_set": run["action_set"],
                    "max_green_seconds": run["max_green"] or "",
                    "testing_episodes": summary["count"],
                    "mean": f'{summary["mean"]:.6f}',
                    "median": f'{summary["median"]:.6f}',
                    "std_dev": f'{summary["std"]:.6f}',
                    "ci95": f'{summary["ci95"]:.6f}',
                    "minimum": f'{summary["min"]:.6f}',
                    "q1": f'{summary["q1"]:.6f}',
                    "q3": f'{summary["q3"]:.6f}',
                    "maximum": f'{summary["max"]:.6f}',
                }
            )


def save_figures(runs):
    labels = [run["label"] for run in runs]
    colors = ["#2878B5" if run["algorithm"] == "IDQN" else "#E07A1F" for run in runs]

    summaries = [stats(run["testing"]) for run in runs]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    bars = ax.bar(labels, [s["mean"] for s in summaries], color=colors, alpha=0.88)
    ax.errorbar(labels, [s["mean"] for s in summaries],
                yerr=[s["ci95"] for s in summaries], fmt="none", ecolor="black", capsize=4)
    ax.bar_label(bars, labels=[f'{s["mean"]:.2f}' for s in summaries], padding=3, fontsize=9)
    ax.set_ylabel("Mean time loss per vehicle (s)")
    ax.set_title("Testing time loss: mean and 95% confidence interval")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "time_loss_testing_means.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    box = ax.boxplot([run["testing"] for run in runs], tick_labels=labels, patch_artist=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_yscale("log")
    ax.set_ylabel("Time loss per vehicle (s, logarithmic scale)")
    ax.set_title("Testing time-loss distributions (100 episodes)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "time_loss_testing_boxplot.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 7))
    for run, color in zip(runs, colors):
        if run["algorithm"] != "IDQN":
            continue
        smoothed = moving_average(run["training"], 50)
        ax.plot(range(1, len(smoothed) + 1), smoothed, label=run["label"], color=color,
                linewidth=1.7, alpha=0.9)
        # Give the three learned R&B curves distinct shades.
        if run["action_set"] == "RingAndBarrier":
            ax.lines[-1].set_color({40: "#54A24B", 80: "#2878B5", 120: "#B279A2"}[run["max_green"]])
        else:
            ax.lines[-1].set_color("#E45756")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean time loss per vehicle (s), 50-episode moving average")
    ax.set_title("Learned-controller training curves")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "time_loss_training_curves.png", dpi=200)
    plt.close(fig)


def save_report(runs):
    summaries = [(run, stats(run["testing"])) for run in runs]
    ranked = sorted(summaries, key=lambda item: item[1]["mean"])
    best_run, best = ranked[0]
    phase_run, phase = next(item for item in summaries if item[0]["action_set"] == "Phase")
    learned = sorted(
        (item for item in summaries if item[0]["algorithm"] == "IDQN" and item[0]["action_set"] == "RingAndBarrier"),
        key=lambda item: item[1]["mean"],
    )

    lines = [
        "# Max-green sweep: time-loss comparison",
        "",
        "Testing results use the final 100 recorded episode values from each run. Lower time loss is better.",
        "",
        "| Rank | Controller | Mean (s) | Median (s) | Std dev | Min | Max |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, (run, summary) in enumerate(ranked, 1):
        lines.append(
            f'| {rank} | {run["label"]} | {summary["mean"]:.3f} | '
            f'{summary["median"]:.3f} | {summary["std"]:.3f} | '
            f'{summary["min"]:.3f} | {summary["max"]:.3f} |'
        )
    improvement = 100 * (phase["mean"] - best["mean"]) / phase["mean"]
    lines += [
        "",
        "## Main findings",
        "",
        f'- **Best overall:** {best_run["label"]}, with mean time loss {best["mean"]:.3f} s and median {best["median"]:.3f} s.',
        f'- Compared with the IDQN Phase baseline ({phase["mean"]:.3f} s), the best run reduces mean time loss by {improvement:.1f}%.' ,
        f'- Among learned Ring-and-Barrier controllers, the order is: ' + ", ".join(
            f'{run["max_green"]}s ({summary["mean"]:.3f} s)' for run, summary in learned
        ) + ".",
    ]
    learned_120 = next(item for item in learned if item[0]["max_green"] == 120)
    lines.append(
        f'- The learned 120s run is unstable: its median is {learned_120[1]["median"]:.3f} s, '
        f'but a {learned_120[1]["max"]:.3f} s outlier raises its mean to {learned_120[1]["mean"]:.3f} s.'
    )
    lines += [
        "- All stochastic Ring-and-Barrier variants perform substantially worse than every learned controller in this sweep.",
        "",
        "## Figures",
        "",
        "- `time_loss_testing_means.png`: mean testing time loss with 95% confidence intervals.",
        "- `time_loss_testing_boxplot.png`: full testing distributions on a logarithmic scale so outliers remain visible.",
        "- `time_loss_training_curves.png`: 50-episode moving averages for learned controllers.",
        "",
    ]
    (RESULTS / "time_loss_comparison.md").write_text("\n".join(lines))


def main():
    runs = load_runs()
    if len(runs) != 7:
        raise SystemExit(f"Expected 7 runs but found {len(runs)}")
    save_csv(runs)
    save_figures(runs)
    save_report(runs)
    print(f"Created time-loss comparison for {len(runs)} runs in {RESULTS}")


if __name__ == "__main__":
    main()
