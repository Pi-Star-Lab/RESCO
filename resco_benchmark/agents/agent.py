import torch


class Agent(object):
    def __init__(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def act(self, observation):
        raise NotImplementedError

    def observe(self, observation, reward, done, info):
        raise NotImplementedError


class IndependentAgent(Agent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__()
        self.config = config
        self.agents = {}

    def act(self, observation):
        return {
            agent_id: self.agents[agent_id].act(observation[agent_id])
            for agent_id in observation.keys()
        }

    def observe(self, observation, reward, done, info):
        for agent_id in observation.keys():
            self.agents[agent_id].observe(
                observation[agent_id], reward[agent_id], done, info
            )
            if done and info["eps"] % 100 == 0:
                self.agents[agent_id].save(self.config["log_dir"] + "agent_" + agent_id)


class SharedAgent(Agent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__()
        self.config = config
        self.agent = None
        self.valid_acts = None
        self.reverse_valid = None

    def act(self, observation):
        if self.reverse_valid is None and self.valid_acts is not None:
            self.reverse_valid = {
                signal_id: {v: k for k, v in self.valid_acts[signal_id].items()}
                for signal_id in self.valid_acts
            }

        batch_obs = [observation[agent_id] for agent_id in observation.keys()]
        if self.valid_acts is None:
            batch_valid = None
            batch_reverse = None
        else:
            batch_valid = [
                self.valid_acts.get(agent_id) for agent_id in observation.keys()
            ]
            batch_reverse = [
                self.reverse_valid.get(agent_id) for agent_id in observation.keys()
            ]

        batch_acts = self.agent.act(
            batch_obs, valid_acts=batch_valid, reverse_valid=batch_reverse
        )
        return {
            agent_id: batch_acts[i] for i, agent_id in enumerate(observation.keys())
        }

    def observe(self, observation, reward, done, info):
        batch_obs = [observation[agent_id] for agent_id in observation.keys()]
        batch_rew = [reward[agent_id] for agent_id in observation.keys()]
        batch_done = [done] * len(batch_obs)
        batch_reset = [False] * len(batch_obs)
        self.agent.observe(batch_obs, batch_rew, batch_done, batch_reset)
        if done and info["eps"] % 100 == 0:
            self.agent.save(self.config["log_dir"] + "agent")
