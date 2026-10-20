#!/usr/bin/env bash
set -euo pipefail

cd /home/ghebes/RESCO/resco_benchmark

PYTHON=/home/ghebes/miniconda3/envs/resco/bin/python
LOG_DIR=max_green_sweep_2026-10-13

for MAX_GREEN in 40 80 120; do
  "$PYTHON" main.py @cologne1 @IDQN \
    action_set:RingAndBarrier \
    ring_max_green:"$MAX_GREEN" \
    ring_through_max_green:"$MAX_GREEN" \
    episodes:1400 testing:100 \
    log_dir:"$LOG_DIR" \
    save_console_log:True

  "$PYTHON" main.py @cologne1 @STOCHASTIC \
    action_set:RingAndBarrier \
    ring_max_green:"$MAX_GREEN" \
    ring_through_max_green:"$MAX_GREEN" \
    episodes:1400 testing:100 \
    log_dir:"$LOG_DIR" \
    save_console_log:True
done

"$PYTHON" main.py @cologne1 @IDQN \
  action_set:Phase \
  episodes:1400 testing:100 \
  log_dir:"$LOG_DIR" \
  save_console_log:True
