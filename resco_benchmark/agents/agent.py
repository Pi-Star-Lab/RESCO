import torch
from resco_benchmark.config.config import config as cfg


class Agent(object):
    def __init__(self):
        if torch.cuda.is_available():
            device = "cuda:" + str(cfg.gpu)
        else:
            device = "cpu"
        self.device = torch.device(device)
        cfg.device = torch.device(device)

    def act(self, observation):
        raise NotImplementedError

    def observe(self, observation, reward, done, info):
        raise NotImplementedError

    def save(self):
        pass

    def load(self):
        pass

    def training(self):
        pass

    def testing(self):
        pass


class IndependentAgent(Agent):
    def __init__(self, obs_act):
        super().__init__()
        self.agents = dict()
        self.last_obs = dict()

    def act(self, observation):
        self.last_obs = observation
        acts = dict()
        for agent_id in observation:
            act = self.agents[agent_id].act(observation[agent_id])

            acts[agent_id] = act
        return acts

    def observe(self, observation, reward, done, info):
        for agent_id in observation:
            self.agents[agent_id].observe(
                observation[agent_id], reward[agent_id], done, info
            )

    def save(self):
        for agent_id in self.agents:
            self.agents[agent_id].save()

    def load(self):
        for agent_id in self.agents:
            self.agents[agent_id].load()

    def training(self):
        for agent_id in self.agents:
            self.agents[agent_id].training()

    def testing(self):
        for agent_id in self.agents:
            self.agents[agent_id].testing()


class SharedAgent(Agent):
    def __init__(self, obs_act):
        super().__init__()
        self.agent = None
        self.pair_to_act_map = None
        self.reverse_valid = None

    def filter_uncontrolled(self, observation):
        removals = []
        if cfg.controlled_signals is not None:
            for agent_id in observation:
                if agent_id not in cfg.controlled_signals:
                    removals.append(agent_id)
        for r in removals:
            del observation[r]
        return observation

    def act(self, observation):
        observation = self.filter_uncontrolled(observation)
        if self.reverse_valid is None and self.pair_to_act_map is not None:
            self.reverse_valid = dict()
            for signal_id in self.pair_to_act_map:
                self.reverse_valid[signal_id] = {
                    v: k for k, v in self.pair_to_act_map[signal_id].items()
                }

        batch_obs = [observation[agent_id] for agent_id in observation.keys()]
        if self.pair_to_act_map is None:
            batch_valid = None
            batch_reverse = None
        else:
            batch_valid = [
                self.pair_to_act_map.get(agent_id) for agent_id in observation.keys()
            ]
            batch_reverse = [
                self.reverse_valid.get(agent_id) for agent_id in observation.keys()
            ]

        batch_acts = self.agent.act(
            batch_obs, pair_to_act_map=batch_valid, reverse_valid=batch_reverse
        )
        acts = dict()
        for i, agent_id in enumerate(observation.keys()):
            acts[agent_id] = batch_acts[i]
        return acts

    def observe(self, observation, reward, done, info):
        observation = self.filter_uncontrolled(observation)
        batch_obs = [observation[agent_id] for agent_id in observation.keys()]
        batch_rew = [reward[agent_id] for agent_id in observation.keys()]
        batch_done = [done] * len(batch_obs)
        batch_reset = [False] * len(batch_obs)
        self.agent.observe(batch_obs, batch_rew, batch_done, batch_reset)

    def save(self):
        self.agent.save()

    def load(self):
        self.agent.load()

    def training(self):
        self.agent.training()

    def testing(self):
        self.agent.testing()
