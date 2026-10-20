from typing import Any, Sequence
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pfrl
from pfrl import replay_buffers
from pfrl.explorer import Explorer
from pfrl.agents import DQN
from pfrl.explorers import LinearDecayEpsilonGreedy
from pfrl.q_functions import DiscreteActionValueHead
from pfrl.utils.contexts import evaluating
from pfrl import explorers
from pfrl.explorers.epsilon_greedy import select_action_epsilon_greedily
import pfrl.initializers

from resco_benchmark.config.config import config as cfg
from resco_benchmark.utils.utils import (
    conv2d_size_out,
    compute_safe_id,
)
from resco_benchmark.agents.agent import IndependentAgent, Agent


class SharedEpsGreedy(explorers.LinearDecayEpsilonGreedy):
    """
    Epsilon-greedy adjusted to return random action from the set of valid actions.
    """

    def select_action(self, t, greedy_action_func, action_value=None, num_acts=None):
        self.epsilon = self.compute_epsilon(t)
        if num_acts is None:
            fn = self.random_action_func
        else:
            fn = lambda: np.random.randint(num_acts)
        a, greedy = select_action_epsilon_greedily(self.epsilon, fn, greedy_action_func)
        greedy_str = "greedy" if greedy else "non-greedy"
        self.logger.debug("t:%s a:%s %s", t, a, greedy_str)
        if num_acts is None:
            return a
        else:
            return a, greedy


logger = logging.getLogger(__name__)


# From https://gist.github.com/lintangsutawika/f2f3fb422d6d7df28bd74e26940da2e6
class CReLU(nn.Module):

    def __init__(self, inplace=False):
        super(CReLU, self).__init__()

    def forward(self, x):
        x = torch.cat((x, -x), -1)
        return F.relu(x)


def build_network(obs_space, init_fn=None):
    model = nn.Sequential()
    crelu_mult = 1
    if len(obs_space) == 1:
        input_size = obs_space[0]
    else:
        height = conv2d_size_out(obs_space[1])
        width = conv2d_size_out(obs_space[2])
        input_size = height * width * cfg.number_of_units

        layer = nn.Conv2d(obs_space[0], cfg.number_of_units, kernel_size=(2, 2))
        if init_fn is not None:
            model.append(init_fn(layer))
        else:
            model.append(layer)

        if "crelu" in cfg:
            model.append(CReLU())
            crelu_mult *= 2
        elif "linear_model" not in cfg:
            model.append(nn.ReLU())
        model.append(nn.Flatten())

    layer = nn.Linear(input_size * crelu_mult, cfg.number_of_units * crelu_mult)
    if init_fn is not None:
        model.append(init_fn(layer))
    else:
        model.append(layer)

    if "crelu" in cfg:
        model.append(CReLU())
        crelu_mult *= 2
    elif "linear_model" not in cfg:
        model.append(nn.ReLU())
    for i in range(cfg.number_of_layers - 1):
        layer = nn.Linear(
            cfg.number_of_units * crelu_mult,
            cfg.number_of_units * crelu_mult,
        )
        if init_fn is not None:
            model.append(init_fn(layer))
        else:
            model.append(layer)
        if "crelu" in cfg:
            model.append(CReLU())
            crelu_mult *= 2
        elif "linear_model" not in cfg:
            model.append(nn.ReLU())
    return model, crelu_mult


def build_q_network(obs_space, act_space):
    model, crelu_mult = build_network(obs_space)
    model.append(nn.Linear(cfg.number_of_units * crelu_mult, act_space))
    model.append(DiscreteActionValueHead())
    return model


class IDQN(IndependentAgent):
    def __init__(self, obs_act):
        super().__init__(obs_act)
        for agent_id in obs_act:
            obs_space = obs_act[agent_id][0]
            act_space = obs_act[agent_id][1]
            self.agents[agent_id] = DQNAgent(
                agent_id, act_space, build_q_network(obs_space, act_space)
            )


class DQNAgent(Agent):
    def __init__(self, agent_id, act_space, model, num_agents=0):
        super().__init__()
        self.agent_id = agent_id
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self._training = True

        if "fasttrac" in cfg:
            try:
                from trac_optimizer import start_trac
            except ImportError:
                raise ImportError(
                    "fasttrac not installed, install 'trac-optimizer' or RESCO[cl]"
                )

            self.optimizer = start_trac(
                cfg.run_path + os.sep + "trac_log.log", torch.optim.Adam
            )(model.parameters(), lr=cfg.learning_rate)

        if cfg.buffer_type == "uniform":
            self.replay_buffer = replay_buffers.ReplayBuffer(cfg.buffer_size)
        elif cfg.buffer_type == "prioritized":
            self.replay_buffer = replay_buffers.PrioritizedReplayBuffer(cfg.buffer_size)
        else:
            raise NotImplementedError()

        if num_agents > 0:
            logger.warning("Using shared epsilon-greedy")
            explorer = SharedEpsGreedy(
                cfg.epsilon_begin,
                cfg.epsilon_end,
                num_agents * cfg.epsilon_decay_period,
                lambda: np.random.randint(act_space),
            )
        else:
            explorer = LinearDecayEpsilonGreedy(
                cfg.epsilon_begin,
                cfg.epsilon_end,
                cfg.epsilon_decay_period,
                lambda: np.random.randint(act_space),
            )

        self.explorer = explorer

        if num_agents > 0:
            logger.warning(f"Using shared network DQN {num_agents}")
            self.agent = SharedDQN(
                self.model,
                self.optimizer,
                self.replay_buffer,
                cfg.discount,
                explorer,
                gpu=self.device.index,
                minibatch_size=cfg.batch_size,
                replay_start_size=cfg.batch_size,
                phi=lambda x: np.asarray(x, dtype=np.float32),
                target_update_interval=cfg.target_update_steps * num_agents,
                update_interval=num_agents,
            )
        else:
            self.agent = DQN(
                self.model,
                self.optimizer,
                self.replay_buffer,
                cfg.discount,
                explorer,
                gpu=self.device.index,
                minibatch_size=cfg.batch_size,
                replay_start_size=cfg.batch_size,
                phi=lambda x: np.asarray(x, dtype=np.float32),
                target_update_interval=cfg.target_update_steps,
            )

    def act(self, observation, pair_to_act_map=None, reverse_valid=None):
        def select_action():
            if isinstance(self.agent, SharedDQN):
                return self.agent.act(
                    observation,
                    pair_to_act_map=pair_to_act_map,
                    reverse_valid=reverse_valid,
                )
            return self.agent.act(observation)

        if self._training:
            act = select_action()
        else:
            # Disabling updates alone does not disable PFRL's epsilon explorer.
            # Evaluation mode makes testing greedy, including after loading a
            # checkpoint whose interaction-step counter is not persisted.
            with self.agent.eval_mode():
                act = select_action()

        return act

    def observe(self, observation, reward, done, info):
        if not self._training:
            return  # By not sending observe to agent update() is never called

        if isinstance(self.agent, SharedDQN):
            # "info" in this case is actually batch_reset (see SharedAgent)
            self.agent.observe(observation, reward, done, info)
        else:
            self.agent.observe(observation, reward, done, reset=False)

        if (
            "parameter_reset_freq" in cfg
            and info["environment"].cumulative_episode % cfg.parameter_reset_freq == 0
        ):
            self.model[-2].reset_parameters()

    def save(self):
        logger.debug("Saving agent {0}".format(self.agent_id))
        path = str(os.path.join(cfg.run_path, "agt_" + compute_safe_id(self.agent_id)))
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path + ".pt",
        )
        self.replay_buffer.save(path + ".replay")

    def load(self):
        if cfg.load_model is None:
            raise ValueError("load_model is not set")
        agt_path = str(
            os.path.join(cfg.load_model, "agt_" + compute_safe_id(self.agent_id))
        )
        logger.debug("Loading agent {0} from {1}".format(self.agent_id, agt_path))
        try:
            # RESCO checkpoints contain optimizer/config objects as well as tensor
            # weights. PyTorch 2.6 changed this default to weights_only=True.
            checkpoint = torch.load(
                agt_path + ".pt", map_location=self.device, weights_only=False
            )
        except TypeError:
            # ``weights_only`` is unavailable in older supported PyTorch releases.
            checkpoint = torch.load(agt_path + ".pt", map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if cfg.load_replay:
            self.replay_buffer.load(agt_path + ".replay")
        if cfg.training:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            self.testing()

    def training(self):
        self._training = True

    def testing(self):
        logger.debug("Disabling training")
        self._training = False


class SharedDQN(DQN):
    def __init__(
        self,
        q_function: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        replay_buffer: pfrl.replay_buffer.AbstractReplayBuffer,
        discount: float,
        explorer: Explorer,
        gpu,
        minibatch_size,
        replay_start_size,
        phi,
        target_update_interval,
        update_interval,
    ):

        super().__init__(
            q_function,
            optimizer,
            replay_buffer,
            discount,
            explorer,
            gpu=gpu,
            minibatch_size=minibatch_size,
            replay_start_size=replay_start_size,
            phi=phi,
            target_update_interval=target_update_interval,
            update_interval=update_interval,
        )
        self.batch_last_action = None
        self.batch_last_obs = None

    def act(self, obs: Any, pair_to_act_map=None, reverse_valid=None) -> Any:
        return self.batch_act(
            obs, pair_to_act_map=pair_to_act_map, reverse_valid=reverse_valid
        )

    def observe(
        self,
        obs: Sequence[Any],
        reward: Sequence[float],
        done: Sequence[bool],
        reset: Sequence[bool],
    ) -> None:
        self.batch_observe(
            obs, reward, done, reset
        )  # Prevent DQN.observe from wrapping parameters

    def batch_act(
        self, batch_obs: Sequence[Any], pair_to_act_map=None, reverse_valid=None
    ) -> Sequence[Any]:
        if pair_to_act_map is None:
            return super(SharedDQN, self).batch_act(batch_obs)
        with torch.no_grad(), evaluating(self.model):
            batch_av = self._evaluate_model_and_update_recurrent_states(batch_obs)

            batch_qvals = batch_av.params[0].detach().cpu().numpy()
            batch_argmax = []
            for i in range(len(batch_obs)):
                batch_item = batch_qvals[i]
                max_val, max_idx = None, None
                for idx in pair_to_act_map[i]:
                    batch_item_qval = batch_item[idx]
                    if max_val is None:
                        max_val = batch_item_qval
                        max_idx = idx
                    elif batch_item_qval > max_val:
                        max_val = batch_item_qval
                        max_idx = idx
                batch_argmax.append(max_idx)
            batch_argmax = np.asarray(batch_argmax)

        if self.training:
            batch_action = []
            for i in range(len(batch_obs)):
                av = batch_av[i : i + 1]
                greed = batch_argmax[i]
                if isinstance(self.explorer, SharedEpsGreedy):
                    act, greedy = self.explorer.select_action(
                        self.t,
                        lambda: greed,
                        action_value=av,
                        num_acts=len(pair_to_act_map[i]),
                    )
                else:
                    act, greedy = self.explorer.select_action(
                        self.t, lambda: greed, action_value=av
                    )
                if not greedy:
                    act = reverse_valid[i][act]
                batch_action.append(act)

            self.batch_last_obs = list(batch_obs)
            self.batch_last_action = list(batch_action)
        else:
            batch_action = batch_argmax

        # Account for differing action spaces between signals
        valid_batch_action = []
        for i in range(len(batch_action)):
            valid_batch_action.append(pair_to_act_map[i][batch_action[i]])
        return valid_batch_action


class VisibleQDQN(DQN):
    """
    A DQN agent variant that exposes Q-values for visibility and analysis.

    This class extends the standard PFRL DQN agent to provide access to Q-values
    alongside actions, which can be useful for debugging, visualization, or
    custom exploration strategies.
    """

    def __init__(
        self,
        q_function: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        replay_buffer: pfrl.replay_buffer.AbstractReplayBuffer,
        discount: float,
        explorer: Explorer,
        gpu,
        minibatch_size,
        replay_start_size,
        phi,
        target_update_interval,
        agent=None,
    ):

        super().__init__(
            q_function,
            optimizer,
            replay_buffer,
            discount,
            explorer,
            gpu=gpu,
            minibatch_size=minibatch_size,
            replay_start_size=replay_start_size,
            phi=phi,
            target_update_interval=target_update_interval,
        )

        self.batch_last_action = None
        self.batch_last_obs = None
        self.agent = agent
        self.action_count = 0

    def batch_act(self, batch_obs: Sequence[Any]) -> Sequence[Any]:
        # TODO batch support has been broken to save time
        with torch.no_grad(), evaluating(self.model):
            batch_av = self._evaluate_model_and_update_recurrent_states(batch_obs)
            batch_argmax = batch_av.greedy_actions.detach().cpu().numpy()

        argmax_act = lambda: batch_argmax[0]

        if self.training:
            batch_action = [
                self.explorer.select_action(
                    self.t,
                    argmax_act,
                    action_value=batch_av[0:1],
                ),
            ]
            self.batch_last_obs = list(batch_obs)
            self.batch_last_action = list(batch_action)
        else:
            batch_action = [argmax_act()]
        return batch_action, batch_av

    def act(self, obs: Any) -> Any:
        self.action_count += 1
        batched, qvals = self.batch_act([obs])
        action = batched[0]
        qvals = qvals.q_values[0].detach().cpu().numpy()

        return action, qvals
