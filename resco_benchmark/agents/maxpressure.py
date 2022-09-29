from resco_benchmark.agents.agent import SharedAgent
from resco_benchmark.agents.maxwave import WaveAgent
from resco_benchmark.config.signal_config import signal_configs


class MAXPRESSURE(SharedAgent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__(config, obs_act, map_name, thread_number)
        self.valid_acts = signal_configs[map_name]['valid_acts']
        self.agent = MaxAgent(signal_configs[map_name]['phase_pairs'])


class MaxAgent(WaveAgent):
    def act(self, observation, valid_acts=None, reverse_valid=None):
        repacked_obs = []
        for obs in observation:
            repacked_obs.append(obs[1:])
        return super().act(repacked_obs, valid_acts, reverse_valid)
