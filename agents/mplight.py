import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.agent import SharedAgent
from agents.pfrl_dqn import DQNAgent
from signal_config import signal_configs
from pfrl.q_functions import DiscreteActionValueHead


class MPLight(SharedAgent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__(config, obs_act, map_name, thread_number)
        phase_pairs = signal_configs[map_name]['phase_pairs']
        num_actions = len(phase_pairs)

        comp_mask = []
        for i in range(len(phase_pairs)):
            zeros = np.zeros(len(phase_pairs) - 1, dtype=np.int)
            cnt = 0
            for j in range(len(phase_pairs)):
                if i == j: continue
                pair_a = phase_pairs[i]
                pair_b = phase_pairs[j]
                if len(list(set(pair_a + pair_b))) == 3: zeros[cnt] = 1
                cnt += 1
            comp_mask.append(zeros)
        comp_mask = np.asarray(comp_mask)
        print(comp_mask)

        comp_mask = torch.from_numpy(comp_mask).to(self.device)
        self.valid_acts = signal_configs[map_name]['valid_acts']
        model = FRAP(config, num_actions, phase_pairs, comp_mask, self.device)
        self.agent = DQNAgent(config, num_actions, model, num_agents=config['num_lights'])


class FRAP(nn.Module):
    def __init__(self, config, output_shape, phase_pairs, competition_mask, device):
        super(FRAP, self).__init__()
        self.oshape = output_shape
        self.phase_pairs = phase_pairs
        self.comp_mask = competition_mask
        self.device = device
        self.demand_shape = config['demand_shape']      # Allows more than just queue to be used

        self.d_out = 4      # units in demand input layer
        self.p_out = 4      # size of phase embedding
        self.lane_embed_units = 16
        relation_embed_size = 4

        self.p = nn.Embedding(2, self.p_out)
        self.d = nn.Linear(self.demand_shape, self.d_out)

        self.lane_embedding = nn.Linear(self.p_out + self.d_out, self.lane_embed_units)

        self.lane_conv = nn.Conv2d(2*self.lane_embed_units, 20, kernel_size=(1, 1))

        self.relation_embedding = nn.Embedding(2, relation_embed_size)
        self.relation_conv = nn.Conv2d(relation_embed_size, 20, kernel_size=(1, 1))

        self.hidden_layer = nn.Conv2d(20, 20, kernel_size=(1, 1))
        self.before_merge = nn.Conv2d(20, 1, kernel_size=(1, 1))

        self.head = DiscreteActionValueHead()

    def forward(self, states):
        states = states.to(self.device)
        num_movements = int((states.size()[1]-1)/self.demand_shape)
        batch_size = states.size()[0]
        acts = states[:, 0].to(torch.int64)
        states = states[:, 1:]
        states = states.float()

        # Expand action index to mark demand input indices
        extended_acts = []
        for i in range(batch_size):
            act_idx = acts[i]
            pair = self.phase_pairs[act_idx]
            zeros = torch.zeros(num_movements, dtype=torch.int64, device=self.device)
            zeros[pair[0]] = 1
            zeros[pair[1]] = 1
            extended_acts.append(zeros)
        extended_acts = torch.stack(extended_acts)
        phase_embeds = torch.sigmoid(self.p(extended_acts))

        phase_demands = []
        for i in range(num_movements):
            phase = phase_embeds[:, i]  # size 4
            demand = states[:, i:i+self.demand_shape]
            demand = torch.sigmoid(self.d(demand))    # size 4
            phase_demand = torch.cat((phase, demand), -1)
            phase_demand_embed = F.relu(self.lane_embedding(phase_demand))
            phase_demands.append(phase_demand_embed)
        phase_demands = torch.stack(phase_demands, 1)

        pairs = []
        for pair in self.phase_pairs:
            pairs.append(phase_demands[:, pair[0]] + phase_demands[:, pair[1]])

        rotated_phases = []
        for i in range(len(pairs)):
            for j in range(len(pairs)):
                if i != j: rotated_phases.append(torch.cat((pairs[i], pairs[j]), -1))
        rotated_phases = torch.stack(rotated_phases, 1)
        rotated_phases = torch.reshape(rotated_phases,
                                       (batch_size, self.oshape, self.oshape - 1, 2 * self.lane_embed_units))
        rotated_phases = rotated_phases.permute(0, 3, 1, 2)  # Move channels up
        rotated_phases = F.relu(self.lane_conv(rotated_phases))  # Conv-20x1x1  pair demand representation

        # Phase competition mask
        competition_mask = self.comp_mask.tile((batch_size, 1, 1))
        relations = F.relu(self.relation_embedding(competition_mask))
        relations = relations.permute(0, 3, 1, 2)  # Move channels up
        relations = F.relu(self.relation_conv(relations))  # Pair demand representation

        # Phase pair competition
        combine_features = rotated_phases * relations
        combine_features = F.relu(self.hidden_layer(combine_features))  # Phase competition representation
        combine_features = self.before_merge(combine_features)  # Pairwise competition result

        # Phase score
        combine_features = torch.reshape(combine_features, (batch_size, self.oshape, self.oshape - 1))
        q_values = torch.sum(combine_features, dim=-1)
        return self.head(q_values)
