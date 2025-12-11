from resco_benchmark.config.config import config as cfg
from resco_benchmark.agents.agent import IndependentAgent
from resco_benchmark.agents.action_value.pfrl_dqn import DQNAgent, build_q_network


class ICycle(IndependentAgent):
    def __init__(self, obs_act):
        super().__init__(obs_act)
        for agent_id in obs_act:
            obs_space = obs_act[agent_id][0]
            act_space = obs_act[agent_id][1]
            self.agents[agent_id] = CycleAgent(
                agent_id, act_space, build_q_network(obs_space, act_space)
            )

class CycleAgent(DQNAgent):
    def __init__(self, agent_id, act_space, model):
        super().__init__(agent_id, act_space, model)
        self.agent_id = agent_id
        self.action_set = None
        self.cuml_reward = 0
        self.cuml_steps = 0

    def act(self, observation, pair_to_act_map=None, reverse_valid=None):
        # Call only at the end of each phase
        if self.action_set is None or self.action_set.phase_ending[self.agent_id]:
            return self.agent.act(observation)
        return 0

    def observe(self, observation, reward, done, info):
        self.action_set = info["environment"].action_set
        self.cuml_reward += reward
        self.cuml_steps += 1
        if not self._training:
            return  # By not sending observe to agent update() is never called

        if self.action_set.phase_ending[self.agent_id]:
            self.agent.observe(observation, self.cuml_reward / self.cuml_steps, done, reset=False)
            self.cuml_reward = 0
            self.cuml_steps = 0
