# Max-green sweep: time-loss comparison

Testing results use the final 100 recorded episode values from each run. Lower time loss is better.

| Rank | Controller | Mean (s) | Median (s) | Std dev | Min | Max |
|---:|---|---:|---:|---:|---:|---:|
| 1 | IDQN R&B 80s | 21.467 | 21.663 | 0.688 | 19.629 | 22.834 |
| 2 | IDQN R&B 40s | 22.124 | 22.032 | 1.490 | 19.694 | 27.931 |
| 3 | IDQN Phase | 24.544 | 24.471 | 1.139 | 21.823 | 27.688 |
| 4 | IDQN R&B 120s | 33.811 | 25.508 | 67.601 | 22.656 | 703.771 |
| 5 | STOCHASTIC R&B 40s | 58.774 | 58.584 | 7.895 | 41.918 | 83.174 |
| 6 | STOCHASTIC R&B 120s | 62.629 | 61.151 | 9.291 | 44.780 | 94.852 |
| 7 | STOCHASTIC R&B 80s | 62.828 | 61.503 | 9.706 | 42.782 | 97.819 |

## Main findings

- **Best overall:** IDQN R&B 80s, with mean time loss 21.467 s and median 21.663 s.
- Compared with the IDQN Phase baseline (24.544 s), the best run reduces mean time loss by 12.5%.
- Among learned Ring-and-Barrier controllers, the order is: 80s (21.467 s), 40s (22.124 s), 120s (33.811 s).
- The learned 120s run is unstable: its median is 25.508 s, but a 703.771 s outlier raises its mean to 33.811 s.
- All stochastic Ring-and-Barrier variants perform substantially worse than every learned controller in this sweep.

## Figures

- `time_loss_testing_means.png`: mean testing time loss with 95% confidence intervals.
- `time_loss_testing_boxplot.png`: full testing distributions on a logarithmic scale so outliers remain visible.
- `time_loss_training_curves.png`: 50-episode moving averages for learned controllers.
