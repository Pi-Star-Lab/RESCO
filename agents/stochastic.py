from agents.agent import IndependentAgent, Agent
import random


class STOCHASTIC(IndependentAgent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__(config, obs_act, map_name, thread_number)
        for key in obs_act:
            act_space = obs_act[key][1]
            self.agents[key] = STOCHASTICAgent(act_space)


class STOCHASTICAgent(Agent):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions

    def act(self, observation):
        return random.randint(0, self.num_actions-1)

    def observe(self, observation, reward, done, info):
        pass
