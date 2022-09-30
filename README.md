# RESCO
![Alt text](maps.png?raw=true "Provided SUMO scenarios")


Source code implementing the Reinforcement Learning Benchmarks for Traffic Signal Control (RESCO).

The benchmark uses the Simulation for Urban Mobility (SUMO), which must be [installed separately](https://sumo.dlr.de/docs/Installing/index.html). SUMO_HOME environment variable must be set, this is done automatically on the install of Sumo on Windows and Ubuntu. SUMO 1.9.0 and 1.9.1 have been tested.

On Ubuntu the speed of the simulation may be greatly increased by using libsumo. Set the environment variable LIBSUMO_AS_TRACI to any value and give main.py --libsumo True. Note that this can not be used with multi-threading.

Python 3.7.4 is required for tensorflow -used by the MA2C and FMA2C implementation.

agent_config defines parameters for the available agents. An agent is specified by the --agent argument to main.

map_config specifies the SUMO scenario parameters, road network, and demand files.

mdp_config supplies constants to state and reward functions (e.g. for normalization)

signal_config defines each signal of each SUMO scenario. Valid green phases are determined from the road network TLSLogic, yellow signals are inserted as required. phase_pairs gives the directional index of phase combinations following the order defined in TLSLogic. valid_acts provides a translation table for shared controllers with varying action definitions across multiple signals. For each signal inbound lanes are given by the direction of traffic. Finally, each signal defines which signals are downstream for the purposes of coordination (neighbors, pressure, etc.)

An example command to train IDQN on the Ingolstadt region scenario is:

`python main.py --agent IDQN --map ingolstadt21`

SUMO scenarios are supplied in the environments directory. All scenarios are distributed under their original licenses. Information on the Cologne scenario can be found on (https://sumo.dlr.de/docs/Data/Scenarios/TAPASCologne.html). Information on Ingolstadt scenarios can be found at (https://github.com/silaslobo/InTAS). For more scenarios please see (https://sumo.dlr.de/docs/Data/Scenarios.html)

Below the benchmark performance for baselines (Fixed Time, Greedy, Max Pressure) and learning algorithms (IDQN, IPPO, MPLight, Extended MPLight (MPLight*), FMA2C) are given.
![Alt text](delays.png?raw=true "Benchmark learning curves")

# Citing RESCO
This project was used in [Reinforcement Learning Benchmarks for Traffic Signal Control](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/f0935e4cd5920aa6c7c996a5ee53a70f-Abstract-round1.html). If you use RESCO in your work, please include a citation:
```
@inproceedings{ault2021reinforcement,
  title={Reinforcement Learning Benchmarks for Traffic Signal Control},
  author={James Ault and Guni Sharon},
  booktitle={Proceedings of the Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS 2021) Datasets and Benchmarks Track},
  month={December},
  year={2021}
}
```


# EPyMARL
RESCO has been updated to be compatible with the [EPyMARL](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/a8baa56554f96369ab93e4f3bb068c22-Abstract-round1.html) benchmark for cooperative RL algorithms. Some modifications within EPyMARL are required currently, available [here](https://github.com/Pi-Star-Lab/epymarl_resco). Clone the modified repository and execute EPyMARL algorithms against the RESCO benchmark using EPyMARL's main.py:

```main.py --config=qmix --env-config=gymma with env_args.time_limit=1 env_args.key=resco_benchmark:cologne3-qmix-v1```