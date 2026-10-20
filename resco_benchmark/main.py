import os
import time
import logging
import importlib
import random

import numpy as np
import torch

from resco_benchmark.config.config import config as cfg
from resco_benchmark.utils.logs import parse_logs
from resco_benchmark.utils.utils import cleanup_log_dir
from resco_benchmark.multi_signal import MultiSignal
import resco_benchmark.mdp_options.states as states
import resco_benchmark.mdp_options.rewards as rewards
import resco_benchmark.config.optuna_objectives as objectives
from resco_benchmark.mdp_options.state_builder import StateBuilder, RewardBuilder

logger = logging.getLogger(__name__)


def main():
    if cfg.optuna_objective is not None:
        # TODO add DB setup instructions to docs (see optuna docs for now)
        import optuna

        if cfg.optuna_trials is not None:
            callbacks = [
                optuna.study.MaxTrialsCallback(
                    int(cfg.optuna_trials), states=(optuna.trial.TrialState.COMPLETE,)
                )
            ]
        else:
            callbacks = None

        study = optuna.create_study(
            study_name=cfg.run_name,
            storage="mysql+pymysql://root@localhost/optuna",  # TODO not root
            load_if_exists=True,
            sampler=optuna.samplers.CmaEsSampler(),
        )

        obj = getattr(objectives, cfg.optuna_objective)
        study.optimize(lambda trial: obj(trial, run_trial), callbacks=callbacks)
    else:
        start = time.time()
        run_trial()
        print("Time taken:", time.time() - start)


def run_trial():
    print(f"cfg.run_name: {cfg.run_name}")
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
    if cfg.route is not None:
        cfg.route = str(os.path.join(os.path.dirname(__file__), cfg.route))

    if cfg == "grid4x4" or cfg == "arterial4x4":
        if not os.path.exists(cfg.route):
            raise EnvironmentError(
                "You must decompress environment files defining traffic flow"
            )

    alg = getattr(importlib.import_module("resco_benchmark.agents."+cfg.module), cfg.algorithm)
    state_fn = getattr(states, cfg.state)
    reward_fn = getattr(rewards, cfg.reward)

    if cfg.state == "state_builder":
        cfg.state_builder = StateBuilder(cfg.state_builder)
    if cfg.reward == "reward_builder":
        cfg.reward_builder = RewardBuilder(cfg.reward_builder)

    env = MultiSignal(state_fn, reward_fn)

    # Get agent id's, observation shapes, and action sizes from env
    agent = alg(env.obs_act)
    if cfg.load_model is not None:
        try:
            agent.load()
        except Exception:
            logger.exception("Could not load model; are the RESCO parameters the same?")
            raise

    if cfg.curriculum is None:
        adj_len = 1
    else:
        adj_len = len(cfg.curriculum)

    minimum_length = cfg.episodes * adj_len if cfg.training else 0

    terminated = False
    for __ in range(minimum_length):
        obs, info = env.reset()
        if terminated and cfg.delete_episode_logs:
            parse_logs()
        terminated = False
        while not terminated:
            act = agent.act(obs)
            obs, rew, terminated, truncated, info = env.step(act)
            agent.observe(obs, rew, terminated, info)

            if "init_ender" in cfg and cfg.init_ender:
                break

            if (
                terminated
                and env.cumulative_episode % cfg.save_frequency == 0
                and cfg.save_model
            ):
                agent.save()

        if "init_ender" in cfg and cfg.init_ender:
            break

    # Testing is a distinct phase.  Several agents and environment reset hooks use
    # cfg.training to decide whether to learn or advance a training curriculum.
    cfg.training = False
    agent.testing()
    terminated = False
    for __ in range(cfg.testing):
        obs, info = env.reset()
        if terminated and cfg.delete_episode_logs:
            parse_logs()
        terminated = False
        while not terminated:
            act = agent.act(obs)
            obs, rew, terminated, truncated, info = env.step(act)
            agent.observe(obs, rew, terminated, info)

    if cfg.save_model:
        agent.save()
    env.close()
    if cfg.delete_episode_logs:
        cleanup_log_dir()
    return (
        parse_logs()
    )  # TODO scan json file for full results and return them for optuna objective evaluation


if __name__ == "__main__":
    main()
