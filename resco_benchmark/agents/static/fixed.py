import os
import random

import numpy as np
from pfrl.explorers.epsilon_greedy import LinearDecayEpsilonGreedy
from pfrl.explorers.epsilon_greedy import select_action_epsilon_greedily

from resco_benchmark.config.config import config as cfg
from resco_benchmark.agents.agent import IndependentAgent, Agent
from resco_benchmark.utils.utils import permutations_without_rotations, compute_safe_id
import logging

logger = logging.getLogger(__name__)


class FIXED(IndependentAgent):
    def __init__(self, obs_act):
        super().__init__(obs_act)
        for key in obs_act:
            if "mgr" not in key:
                self.agents[key] = FixedAgent(obs_act[key], key)


class FixedAgent(Agent):
    def __init__(self, obs_act, key):
        super().__init__()
        self.agent_id = key
        self.acts = 0
        self.curr_idx = 0
        self.plan = list()
        self.offset = 0
        self.active_phase = 0
        self.active_phase_len = 0
        num_acts = obs_act[1]

        timings = cfg[key]["fixed_timings"]
        for i in range(num_acts):
            self.plan.append(int(timings[i]))

        phase_orders = np.asarray(list(permutations_without_rotations(range(num_acts))))
        self.phase_order = phase_orders[cfg[key]["fixed_phase_order_idx"]].tolist()

        if "fixed_offset" in cfg[key]:
            self.offset = cfg[key]["fixed_offset"]
            if self.agent_id != "B3":
                logger.warning(
                    "Fixed agent offset not implemented for " + self.agent_id
                )

        self.num_acts = num_acts

    def act(self, _=None):
        if np.all(np.array(self.plan) == 0):
            self.plan[self.active_phase] = 1  # Ensure that at least one phase is active
        if self.agent_id == "B3" and self.acts == 0:
            # TODO only implemented for 2 signal maps with A3 and B3
            tmp_offset = self.offset
            while tmp_offset > 0:
                if self.active_phase_len > tmp_offset:
                    self.active_phase_len -= tmp_offset
                    tmp_offset = 0
                else:
                    self.active_phase = (self.active_phase - 1) % len(self.plan)
                    swap = self.plan[self.active_phase]
                    swap = swap - tmp_offset
                    if swap < 0:
                        tmp_offset = swap * -1
                    else:
                        self.active_phase_len = swap
                        tmp_offset = 0

        if self.active_phase_len >= np.abs(self.plan[self.active_phase]):
            self.active_phase = (self.active_phase + 1) % len(self.plan)
            while self.plan[self.active_phase] == 0:  # Skip phases with 0 duration
                self.active_phase = (self.active_phase + 1) % len(self.plan)
            self.active_phase_len = 1
        else:
            self.active_phase_len += 1

        self.acts += 1

        return self.phase_order[self.active_phase]

    @property
    def phase_ending(self):
        return self.active_phase_len >= np.abs(self.plan[self.active_phase])

    def __getitem__(self, act):
        return self.plan[self.phase_order[act]]

    def increase_current_phase_length(self, size=1):
        self.plan[self.active_phase] += size

    def decrease_current_phase_length(self, size=1):
        self.plan[self.active_phase] -= size
        if self.plan[self.active_phase] < 0:
            self.plan[self.active_phase] = 0

    def save(self):
        with open(
            os.path.join(
                cfg.run_path, "fixd_{0}.txt".format(compute_safe_id(self.agent_id))
            ),
            "w",
        ) as f:
            f.write(str(self.plan) + "\n")
            f.write(str(self.phase_order) + "\n")
            f.write(str(self.offset) + "\n")

    def observe(self, observation, reward, done, info):
        pass
