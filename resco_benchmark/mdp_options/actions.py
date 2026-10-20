import numpy as np

from resco_benchmark.config.config import config as cfg
from resco_benchmark.agents.static.fixed import FIXED, FixedAgent


# Config values of phase or null will use direct 'set next phase' actions


class RingAndBarrier:
    """Stateful dual-ring, two-barrier controller for Cologne1.

    Each ring independently occupies the left-turn (0) or through (1) phase in
    its current barrier group.  A ring may advance from left-turn to through on
    its own, but it must wait at through until both rings can cross the barrier
    together.

    FHWA Traffic Signal Timing Manual, section 4.2.2:
    https://ops.fhwa.dot.gov/publications/fhwahop08024/chapter4.htm
    """

    SIGNAL_ID = "cluster_357187_359543"
    EXPECTED_PHASE_COUNT = 8

    HOLD_BOTH = 0
    ADVANCE_RING_1 = 1
    ADVANCE_RING_2 = 2
    ADVANCE_BOTH = 3

    # (barrier, ring-1 position, ring-2 position) -> Cologne1 SUMO phase.
    # Position 0 is the protected-left phase and position 1 is the through phase.
    # The four combinations on each side of the barrier account for all eight
    # green phases in Cologne1's existing SUMO program.
    PHASE_MAP = {
        (0, 0, 0): 3,  # east-west: both protected lefts
        (0, 1, 0): 0,  # east-west: ring 1 through, ring 2 left
        (0, 0, 1): 1,  # east-west: ring 1 left, ring 2 through
        (0, 1, 1): 2,  # east-west: both through
        (1, 0, 0): 7,  # north-south: both protected lefts
        (1, 1, 0): 4,  # north-south: ring 1 through, ring 2 left
        (1, 0, 1): 5,  # north-south: ring 1 left, ring 2 through
        (1, 1, 1): 6,  # north-south: both through
    }

    def __init__(self, obs_act, signals):
        if cfg.map != "cologne1":
            raise ValueError("RingAndBarrier is defined only for cologne1")
        if set(signals) != {self.SIGNAL_ID}:
            raise ValueError(
                "The Cologne1 ring-and-barrier action set requires signal "
                f"{self.SIGNAL_ID!r}; received {sorted(signals)}"
            )
        phase_count = obs_act[self.SIGNAL_ID][1]
        if phase_count != self.EXPECTED_PHASE_COUNT:
            raise ValueError(
                f"{self.SIGNAL_ID} must expose {self.EXPECTED_PHASE_COUNT} green "
                f"phases; received {phase_count}"
            )

        self.signals = signals
        self.num_acts = 4
        self.max_green = cfg.ring_max_green
        self.through_min_green = cfg.ring_through_min_green
        self.through_max_green = cfg.ring_through_max_green
        for name, value in (
            ("ring_max_green", self.max_green),
            ("ring_through_min_green", self.through_min_green),
            ("ring_through_max_green", self.through_max_green),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or null")
        self.reset()

    def reset(self):
        """Start each episode with both rings at barrier 1 protected lefts."""
        self.barriers = {signal_id: 0 for signal_id in self.signals}
        self.ring_positions = {
            signal_id: [0, 0] for signal_id in self.signals
        }
        self.phase_elapsed = {signal_id: 0 for signal_id in self.signals}
        self.last_phases = {
            signal_id: self.PHASE_MAP[(0, 0, 0)] for signal_id in self.signals
        }

    def _safe_action(self, signal_id, action):
        """Prevent a learned policy from starving a ring indefinitely."""
        positions = self.ring_positions[signal_id]
        if (
            positions == [1, 1]
            and self.through_min_green is not None
            and self.phase_elapsed[signal_id] < self.through_min_green
        ):
            return self.HOLD_BOTH
        max_green = (
            self.through_max_green if positions == [1, 1] else self.max_green
        )
        if (
            max_green is None
            or self.phase_elapsed[signal_id] < max_green
        ):
            return action

        if positions == [0, 0] or positions == [1, 1]:
            return self.ADVANCE_BOTH
        if positions == [0, 1]:
            return self.ADVANCE_RING_1
        return self.ADVANCE_RING_2

    def _advance(self, signal_id, action):
        positions = self.ring_positions[signal_id]
        both_waiting = positions == [1, 1]

        if action in (self.ADVANCE_RING_1, self.ADVANCE_BOTH):
            positions[0] = min(1, positions[0] + 1)
        if action in (self.ADVANCE_RING_2, self.ADVANCE_BOTH):
            positions[1] = min(1, positions[1] + 1)

        # A barrier may only be crossed by an explicit ADVANCE_BOTH when both
        # rings were already waiting at their through phases before this action.
        if action == self.ADVANCE_BOTH and both_waiting:
            self.barriers[signal_id] = 1 - self.barriers[signal_id]
            positions[:] = [0, 0]

    def act(self, acts):
        mapped_acts = dict(acts)
        for signal_id in self.signals:
            action = acts[signal_id]
            if not isinstance(action, (int, np.integer)):
                raise TypeError(
                    f"RingAndBarrier action for {signal_id} must be an integer"
                )
            if action < 0 or action >= self.num_acts:
                raise ValueError(
                    f"RingAndBarrier action for {signal_id} must be between 0 "
                    f"and {self.num_acts - 1}; received {action}"
                )
            action = int(action)
            action = self._safe_action(signal_id, action)
            self._advance(signal_id, action)

            state = (
                self.barriers[signal_id],
                *self.ring_positions[signal_id],
            )
            mapped_phase = self.PHASE_MAP[state]
            if mapped_phase == self.last_phases[signal_id]:
                self.phase_elapsed[signal_id] += cfg.step_length
            else:
                self.last_phases[signal_id] = mapped_phase
                self.phase_elapsed[signal_id] = 0
            mapped_acts[signal_id] = mapped_phase
        return mapped_acts


#   Replaces action space with simplified set of [keep, increase, decrease] following fixed timing config
class FixedCyclePlan:
    def __init__(self, obs_act, signals):
        self.signals = signals
        self.fixed_agent = FIXED(obs_act)
        self.num_acts = 3

    def act(self, acts):
        for signal in self.signals:
            agt_act = acts[signal]
            if agt_act == 0:
                pass
            elif agt_act == 1:
                self.fixed_agent.agents[signal].increase_current_phase_length()
            elif agt_act == 2:
                self.fixed_agent.agents[signal].decrease_current_phase_length()
            else:
                raise NotImplementedError()
        acts = self.fixed_agent.act(observation=self.signals)
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
