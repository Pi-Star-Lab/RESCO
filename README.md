# RESCO
![Alt text](utils/maps.png?raw=true "Provided SUMO scenarios")


Source code implementing the Reinforcement Learning Benchmarks for Traffic Signal Control (RESCO).

The benchmark uses the Simulation for Urban Mobility (SUMO), which must be installed separately. SUMO_HOME environment variable must be set, this is done automatically on the install of Sumo on Windows and Ubuntu. SUMO 1.9.0 and 1.9.1 have been tested.

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
![Alt text](utils/delays.png?raw=true "Benchmark learning curves")