# import numpy as np
# from resco_benchmark.agents.agent import SharedAgent, Agent
# from resco_benchmark.config.signal_config import signal_configs


# class MAXWAVE(SharedAgent):
#     def __init__(self, config, obs_act, map_name, thread_number):
#         super().__init__(config, obs_act, map_name, thread_number)
#         self.valid_acts = signal_configs[map_name]['valid_acts']
#         self.agent = WaveAgent(signal_configs[map_name]['phase_pairs'])


# class WaveAgent(Agent):
#     def __init__(self, phase_pairs):
#         super().__init__()
#         self.phase_pairs = phase_pairs

#     def act(self, observations, valid_acts=None, reverse_valid=None):
#         acts = []
#         for i, observation in enumerate(observations):
#             if valid_acts is None:
#                 all_press = []
#                 for pair in self.phase_pairs:
#                     all_press.append(observation[pair[0]] + observation[pair[1]])
#                 acts.append(np.argmax(all_press))
#             else:
#                 max_press, max_index = None, None
#                 for idx in valid_acts[i]:
#                     pair = self.phase_pairs[idx]
#                     press = observation[pair[0]] + observation[pair[1]]
#                     if max_press is None:
#                         max_press = press
#                         max_index = idx
#                     if press > max_press:
#                         max_press = press
#                         max_index = idx
#                 acts.append(valid_acts[i][max_index])
#         return acts

#     def observe(self, observation, reward, done, info):
#         pass

#     def save(self, path):
#         pass

# class MINJUNGAGENT(SharedAgent):
#     def __init__(self, config, obs_act, map_name, thread_number):
#         super().__init__(config, obs_act, map_name, thread_number)
#         self.valid_acts = signal_configs[map_name]['valid_acts']
#         self.agent = MaxAgent(signal_configs[map_name]['phase_pairs'])


# class MinjugAgent(WaveAgent):
#     """
#     TODO:
#     [ ] - Implement the act method
#         [ ] - Get the desired observations
#         [ ] - Get the valid actions
#         [ ] - Get the reverse valid actions

#     Args:
#         WaveAgent (_type_): _description_
#     """


#     def act(self, observation, valid_acts=None, reverse_valid=None):


#          sum(
#                     _alpha * veh_speed_factor[p] + _beta * 60 * accumulated_wtime[p] + 60 * _gamma #Furuku normalization
#                     for p in set(action)
#                 )
#                 for action, _alpha, _beta, _gamma in zip(
#                     tl_actor.action_space, self.alpha[tls_ix], self.beta[tls_ix], self.gamma[tls_ix]
#                 )
#             ]
#             """ Max added that you don't need below. Action space is sorted so that the first element is the optimal action """
#             best_action = sorted(
#                 zip(tl_actor.action_space, action_priority),
#                 key=lambda x: x[1],
#                 reverse=True,
#             )[0][1]
