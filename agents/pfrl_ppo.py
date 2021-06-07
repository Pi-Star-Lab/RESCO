import numpy as np

import torch
import torch.nn as nn

from pfrl.nn import Branched
import pfrl.initializers
from pfrl.agents import PPO
from pfrl.policies import SoftmaxCategoricalHead

from agents.agent import IndependentAgent, Agent


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
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__(config, obs_act, map_name, thread_number)
        for key in obs_act:
            obs_space = obs_act[key][0]
            act_space = obs_act[key][1]
            self.agents[key] = PFRLPPOAgent(config, obs_space, act_space)


class PFRLPPOAgent(Agent):
    def __init__(self, config, obs_space, act_space):
        super().__init__()

        def conv2d_size_out(size, kernel_size=2, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        h = conv2d_size_out(obs_space[1])
        w = conv2d_size_out(obs_space[2])

        self.model = nn.Sequential(
            lecun_init(nn.Conv2d(obs_space[0], 64, kernel_size=(2, 2))),
            nn.ReLU(),
            nn.Flatten(),
            lecun_init(nn.Linear(h*w*64, 64)),
            nn.ReLU(),
            lecun_init(nn.Linear(64, 64)),
            nn.ReLU(),
            Branched(
                nn.Sequential(
                    lecun_init(nn.Linear(64, act_space), 1e-2),
                    SoftmaxCategoricalHead()
                ),
                lecun_init(nn.Linear(64, 1))
            )
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2.5e-4, eps=1e-5)
        self.agent = PPO(self.model, self.optimizer, gpu=self.device.index,
                         phi=lambda x: np.asarray(x, dtype=np.float32),
                         clip_eps=0.1,
                         clip_eps_vf=None,
                         update_interval=1024,
                         minibatch_size=256,
                         epochs=4,
                         standardize_advantages=True,
                         entropy_coef=0.001,
                         max_grad_norm=0.5)

    def act(self, observation):
        return self.agent.act(observation)

    def observe(self, observation, reward, done, info):
        self.agent.observe(observation, reward, done, False)

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path+'.pt')
