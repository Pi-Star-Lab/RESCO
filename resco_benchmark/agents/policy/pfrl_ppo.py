import os
import numpy as np

import torch
import torch.nn as nn

from pfrl.nn import Branched
from pfrl.agents import PPO
from pfrl.policies import SoftmaxCategoricalHead
import pfrl.initializers

from resco_benchmark.config.config import config as cfg
from resco_benchmark.utils.utils import compute_safe_id
from resco_benchmark.agents.agent import IndependentAgent, Agent
from resco_benchmark.agents.action_value.pfrl_dqn import build_network


def lecun_init(layer, gain=1):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        pfrl.initializers.init_lecun_normal(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    else:
        pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
        pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
        nn.init.zeros_(layer.bias_ih_l0)
        nn.init.zeros_(layer.bias_hh_l0)
    return layer


class IPPO(IndependentAgent):
    def __init__(self, obs_act):
        super().__init__(obs_act)
        for agent_id in obs_act:
            obs_space = obs_act[agent_id][0]
            act_space = obs_act[agent_id][1]
            self.agents[agent_id] = PFRLPPOAgent(agent_id, obs_space, act_space)


class PFRLPPOAgent(Agent):
    def __init__(self, agent_id, obs_space, act_space):
        super().__init__()
        self.agent_id = agent_id

        model, crelu_mult = build_network(obs_space, init_fn=lecun_init)

        model.append(
            Branched(
                nn.Sequential(
                    lecun_init(
                        nn.Linear(cfg.number_of_units * crelu_mult, act_space), 1e-2
                    ),
                    SoftmaxCategoricalHead(),
                ),
                lecun_init(nn.Linear(cfg.number_of_units * crelu_mult, 1)),
            )
        )
        self.model = model

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon
        )
        self.agent = PPO(
            self.model,
            self.optimizer,
            gpu=self.device.index,
            phi=lambda x: np.asarray(x, dtype=np.float32),
            clip_eps=cfg.clip_eps,
            update_interval=cfg.update_interval,
            minibatch_size=cfg.batch_size,
            epochs=cfg.epochs,
            standardize_advantages=cfg.standardize_advantages,
            entropy_coef=cfg.entropy_coef,
            max_grad_norm=cfg.max_grad_norm,
        )

    def act(self, observation):
        return self.agent.act(observation)

    def observe(self, observation, reward, done, info):
        self.agent.observe(observation, reward, done, False)

    def save(self):
        path = str(os.path.join(cfg.run_path, "agt_" + compute_safe_id(self.agent_id)))
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path + ".pt",
        )

    def load(self):
        path = str(os.path.join(cfg.run_path, "agt_" + compute_safe_id(self.agent_id)))
        self.model.load_state_dict(
            torch.load(path + ".pt", map_location=self.device)["model_state_dict"]
        )
        self.optimizer.load_state_dict(
            torch.load(path + ".pt", map_location=self.device)["optimizer_state_dict"]
        )
        if not cfg.training:
            self.agent.training = False

    def training(self):
        self.agent.training = True

    def testing(self):
        self.agent.training = False
