import numpy as np

from resco_benchmark.config.config import config as cfg
from resco_benchmark.agents.static.fixed import FIXED, FixedAgent


# Config values of phase or null will use direct 'set next phase' actions


#   Replaces action space with simplified set of [keep, increase, decrease] following fixed timing config
class FixedCyclePlan:
    def __init__(self, obs_act, signals):
        self.signals = signals
        self.fixed_agent = FIXED(obs_act)
        self.phase_ending = dict()
        self.fixed_plan = dict()
        self.length_memory = dict()
        for signal in signals:
            self.fixed_agent.agents[signal].plan = list(np.ones_like(self.fixed_agent.agents[signal].plan))
            self.fixed_plan[signal] = self.fixed_agent.agents[signal].plan
            self.length_memory[signal] = self.fixed_plan[signal].copy()
        self.num_acts = 5

    def act(self, acts):
        for signal in self.signals:
            agt_act = acts[signal]
            signal_fix = self.fixed_agent.agents[signal]
            active = signal_fix.active_phase

            if agt_act == 0:
                pass
            elif agt_act == 1:
                signal_fix.increase_current_phase_length()
            elif agt_act == 2:
                if signal_fix.plan[signal_fix.active_phase] > 1:
                    signal_fix.decrease_current_phase_length()
            elif agt_act == 3: # Deactivate
                if sum([1 for l in signal_fix.plan if l != 0]) > 2:
                    self.length_memory[signal][active] = self.fixed_plan[signal][active]
                    signal_fix.plan[active] = 0
            elif agt_act == 4: # Activate
                for i, l in enumerate(signal_fix.plan[active:]):    # Find the next inactive phase
                    pass
                if signal_fix.plan[active] == 0:
                    signal_fix.plan[active] = self.length_memory[signal][active]
            else:
                raise NotImplementedError()
        acts = self.fixed_agent.act(observation=self.signals)
        print(self.fixed_plan)
        for signal in self.signals:
            self.phase_ending[signal] = self.fixed_agent.agents[signal].phase_ending
        return acts


def resolve_idx(l, r):
    for i, phase_pair in enumerate(cfg.phase_pairs):
        if (l == phase_pair[0] and r == phase_pair[1]) or (l == phase_pair[1] and r == phase_pair[0]):
            return i

# Stay, Move top, Move bottom
class RingBarrier:
    def __init__(self, obs_act, signals):
        self.mapping = {
            "left": { "top": ['W-S', 'E-E'], "bottom": ['E-N', 'W-W']},
            "right": { "top": ['N-W', 'S-S'], "bottom": ['S-E', 'N-N']}
        }
        self.key_mapped_phase = dict()
        self.cur_phase = dict()
        for signal in signals:
            self.cur_phase[signal] = ["left", 0, 0]
        self.signals = signals
        self.num_acts = 3

    def act(self, acts):
        for signal_id in self.signals:
            curr = self.cur_phase[signal_id]
            agt_act = acts[signal_id]

            if agt_act == 0:
                pass
            elif agt_act == 1:
                if curr[1] == len(self.mapping[curr[0]]["top"])-1 and curr[2] == len(self.mapping[curr[0]]["bottom"])-1:
                    # Cross barrier
                    curr[1] = 0
                    curr[2] = 0
                    if curr[0] == "left":
                        curr[0] = "right"
                    else:
                        curr[0] = "left"
                elif curr[1] != len(self.mapping[curr[0]]["top"])-1:
                    curr[1] += 1
            elif agt_act == 2:
                if curr[1] == len(self.mapping[curr[0]]["top"])-1 and curr[2] == len(self.mapping[curr[0]]["bottom"])-1:
                    # Cross barrier
                    curr[1] = 0
                    curr[2] = 0
                    if curr[0] == "left":
                        curr[0] = "right"
                    else:
                        curr[0] = "left"
                elif curr[2] != len(self.mapping[curr[0]]["bottom"])-1:
                    curr[2] += 1
            else:
                raise IndexError()

            l = self.mapping[curr[0]]["top"][curr[1]]
            r = self.mapping[curr[0]]["bottom"][curr[2]]
            if str(curr) not in self.key_mapped_phase:
                self.key_mapped_phase[str(curr)] = resolve_idx(l, r)

            total_veh = 0
            signal = self.signals[signal_id]
            for lane_id in signal.lane_sets[l]:
                lane = signal.observation.get_lane(lane_id)
                total_veh += lane.vehicle_count
            if total_veh == 0: # Skip
                if curr[1] != len(self.mapping[curr[0]]["top"]) - 1:
                    curr[1] += 1
            total_veh = 0
            for lane_id in signal.lane_sets[r]:
                lane = signal.observation.get_lane(lane_id)
                total_veh += lane.vehicle_count
            if total_veh == 0:  # Skip
                if curr[2] != len(self.mapping[curr[0]]["bottom"]) - 1:
                    curr[2] += 1

            l = self.mapping[curr[0]]["top"][curr[1]]
            r = self.mapping[curr[0]]["bottom"][curr[2]]
            if str(curr) not in self.key_mapped_phase:
                self.key_mapped_phase[str(curr)] = resolve_idx(l, r)

            acts[signal_id] = self.key_mapped_phase[str(curr)]
        return acts



# Follow config fixed time defined cycle, choose to stay in current phase or go next
class FixedCycle:
    def __init__(self, obs_act, signals):
        self.signals = signals
        self.fixed_agent = FIXED(obs_act)
        self.num_acts = 2

        # Force all phases to 1 step length
        for signal in self.signals:
            for i in range(len(self.fixed_agent.agents[signal].plan)):
                self.fixed_agent.agents[signal].plan[i] = 1
            self.fixed_agent.agents[signal].active_phase = (
                len(self.fixed_agent.agents[signal].plan) - 2
            )

    def act(self, acts):
        if cfg.algorithm == "FIXED":
            return acts
        for signal in self.signals:
            agt_act = acts[signal]
            print(signal, agt_act)
            if agt_act == 0:  # Keep same phase
                self.fixed_agent.agents[signal].active_phase_len = 0
            elif agt_act == 1:  # Go next
                self.fixed_agent.agents[signal].active_phase_len = np.inf
            else:
                raise NotImplementedError()
        acts = self.fixed_agent.act(observation=self.signals)
        print("mask", acts)
        return acts


class PlanPick:
    def __init__(self, obs_act, signals):
        self.signals = signals
        # Only valid/tested for saltlake
        cfg["equal_plan"] = dict()
        cfg["vertical_plan"] = dict()
        cfg["horizontal_plan"] = dict()
        cfg["horizontal_plan"]["fixed_phase_order_idx"] = 0
        cfg["vertical_plan"]["fixed_phase_order_idx"] = 0
        cfg["equal_plan"]["fixed_phase_order_idx"] = 0
        cfg["equal_plan"]["fixed_timings"] = [4, 0, 4, 4, 4, 4, 4, 0]
        cfg["vertical_plan"]["fixed_timings"] = [
            4,
            0,
            4,
            4,
            4 + cfg.priority_offset,
            4 + cfg.priority_offset,
            4 + cfg.priority_offset,
            0,
        ]
        cfg["horizontal_plan"]["fixed_timings"] = [
            4 + cfg.priority_offset,
            0,
            4 + cfg.priority_offset,
            4 + cfg.priority_offset,
            4,
            4,
            4,
            0,
        ]

        self.equal_plans = dict()
        obs_act = obs_act.copy()
        obs_act[1] = 8
        for signal in self.signals:
            self.equal_plans[signal] = FixedAgent(obs_act, "equal_plan")
        self.vertical_plans = dict()
        for signal in self.signals:
            self.vertical_plans[signal] = FixedAgent(obs_act, "vertical_plan")
        self.horizontal_plans = dict()
        for signal in self.signals:
            self.horizontal_plans[signal] = FixedAgent(obs_act, "horizontal_plan")
        self.num_acts = 3

    def act(self, acts):
        new_acts = dict()
        for signal in self.signals:
            agt_act = acts[signal]
            if agt_act == 0:
                new_acts[signal] = self.equal_plans[signal].act(None)
            elif agt_act == 1:
                new_acts[signal] = self.vertical_plans[signal].act(None)
            elif agt_act == 2:
                new_acts[signal] = self.horizontal_plans[signal].act(None)
            else:
                print(signal, agt_act)
                raise NotImplementedError()
        return new_acts


# Continuous or discrete, choose current phase's length
class PhaseLength:
    pass  # TODO
