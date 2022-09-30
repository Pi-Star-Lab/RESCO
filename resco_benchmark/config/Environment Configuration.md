## Environment Creation: Example using cologne3

### Defining map_config
    'cologne3': {
        'lights': [],
        'net': 'environments/cologne3/cologne3.sumocfg',
        'route': None,
        'step_length': 10,
        'yellow_length': 3,
        'step_ratio': 1,
        'start_time': 25200,
        'end_time': 28800,
        'warmup': 0
    }
- lights: If 'lights' is an empty list then all intersections with defined <tlLogic...> in cologne3.net.xml will be used. If only a subset of lights is desired then one could configure this as ['360082', '360086']. These two values are the two of the three traffic light controller id's defined in the cologne network file.
- net: Simply defines the location of the .sumocfg or .net.xml SUMO file.
- route: If net is defined with a .sumocfg this should be 'None' as .sumocfg defines the routing file location. If a .net.xml file is used indicate the location of the routing files here.
- step_length: The time between RL agent decisions.
- step_ratio: SUMO supports internal steps of the simulation at variable intervals. This parameter can be used to adjust RESCO to account for this. For example, if the internal step interval is 0.25s setting step_ratio will move the simulation forward 4 times such that a step from an RL agent perspective is still 1s.
- start_time: Indicates the time step the simulation starts from, should match <time\><begin value\> in .sumocfg.
- end_time: Indicates the time step the simulation ends at, should match <time\><end value\> in .sumocfg.
- warmup: Steps the simulator forward this many times before activating RL agents.

### Defining signal_config
#### Action Definitions
    'cologne3': {
		'phase_pairs': [[1, 7], [2, 8], [1, 2], [7, 8], [4, 10], [5, 11], [10, 11], [4, 5], [9, 11]],
		'valid_acts': {
				'GS_cluster_2415878664_254486231_359566_359576': {4: 0, 5: 1, 0: 2, 1: 3},
				'360086': {4: 0, 5: 1, 0: 2, 1: 3},
				'360082': {4: 0, 5: 1, 1: 2},
		},
- phase_pairs: MPLight requires a defined set of traffic movements (in terms of traffic flow direction) which are consistent across traffic signals. In RESCO these movements have been assigned an index as given below. The defined phase_pairs for an environment should include all possible movement directions for all traffic signals on the map.
  - Traffic movement indices - 0: 'S-W',
1: 'S-S',
2: 'S-E',
3: 'W-N',
4: 'W-W',
5: 'W-S',
6: 'N-E',
7: 'N-N',
8: 'N-W',
9: 'E-S',
10: 'E-E',
11: 'E-N' - [e.g. Read S-W as Southbound incoming traffic turning to the Westbound outbound lanes]

- valid_acts: In shared parameter algorithms such as MPLight agents might be heterogenous in terms of their action space. Therefore, actions which two signals have in common must be remapped to correspond to the same action in each agent. 

  - RESCO uses the non-yellow phases defined in <tlLogic...> in .net.xml environment configuration to define actions for each agent - actions are indexed according to the order they are given in <tlLogic...>.
  - valid_acts are defined for each signal ID with the common act index defined by phase_pairs.
  - For example, signal '360086' has 4 actions, the first action declared by its <tlLogic..> is 'GGGggrrrrGGGggrrrr', this corresponds to the [W-W, E-E] phase pair [4, 10]. The index of that phase pair is 4, so the first valid act is defined as 4: 0. The second action is 'rrrGGrrrrrrrGGrrrr' which corresponds to [W-S, E-N] - index 5 in phase_pairs. The second action is mapped to 5: 1.  

#### Signal lane configurations
		'GS_cluster_2415878664_254486231_359566_359576': {
			'lane_sets': {
				'S-W': ['319261593#16_0'],
				'S-S': ['319261593#16_1', '319261593#16_0'],
				'S-E': ['319261593#16_1'],
				'W-N': ['-241660955#3_0'],
				'W-W': ['-241660955#3_0', '-241660955#3_1'],
				'W-S': ['-241660955#3_1'],
				'N-E': ['241660957#0_0'],
				'N-N': ['241660957#0_0', '241660957#0_1'],
				'N-W': ['241660957#0_1'],
				'E-S': ['200818108#0_0'],
				'E-E': ['200818108#0_0', '200818108#0_1'],
				'E-N': ['200818108#0_1']
			},
			'downstream': {
				'N': None,
				'E': '360086',
				'S': None,
				'W': None
			}
		},
    ...
- lane_sets: In order to support remapping common actions between traffic signals each signal needs to declare which lanes constitute  a traffic movement in each direction. For the signal ID 'GS_cluster_2415878664_254486231_359566_359576' in cologne3.net.xml the lanes have been set for each direction here. The easiest way to examine these in SUMO is to start a SUMO simulation with GUI on the scenario and then locate the intersection to view the lane IDs.
- downstream: In the case that an algorithm relies on knowing which traffic signals are adjacent, such as MPLight, downstream defines this by direction in relation to the agent. Signal '360086' is East of signal 'GS_cluster_2415878664_254486231_359566_359576'. accordingly, in the downstream configuration for signal '360086' the downstream West signal is set as 'GS_cluster_2415878664_254486231_359566_359576'.  

### Defining mdp_config
    'cologne3': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['360082', '360086'],
                'bot_mgr': ['GS_cluster_2415878664_254486231_359566_359576']
            },
            'management_neighbors': {
                'top_mgr': ['bot_mgr'],
                'bot_mgr': ['top_mgr']
            }
    }
These settings are so far only used for the FMA2C algorithm. The first 7 values are hyperparameters used in the state and reward representations for the algorithm.
- management: Defines managing agents and which sub-agents they control. top_mgr and bot_mgr are abitrary names and more can be added as needed.
- management_neighbors: Defines which managing agents each interacts with. In this case there are only two managing agents and they both interact with each other.