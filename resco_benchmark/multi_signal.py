import os
import datetime
import logging
import pickle
import uuid

import gymnasium as gym

from resco_benchmark.agents.static.fixed import FIXED
from resco_benchmark.agents.static.maxpressure import MAXPRESSURE
from resco_benchmark.agents.static.maxwave import MAXWAVE
from resco_benchmark.mdp_options.states import wave, mplight

from resco_benchmark.config.config import config as cfg
from resco_benchmark.traffic_signal import Signal
from resco_benchmark.utils.add_flow import generate_additional_flow
from resco_benchmark.utils.csv_to_flo import generate_flow_from_csv
import resco_benchmark.mdp_options.actions as action_sets


if cfg.libsumo and not cfg.gui:
    import libsumo as traci
else:
    cfg.libsumo = False
    import traci
from sumolib import checkBinary


logger = logging.getLogger(__name__)


class MultiSignal(gym.Env):
    """
    The `MultiSignal` class is a custom Gym environment that simulates a multi-signal traffic control scenario.
    It is responsible for:

    - Initializing the simulation environment with the specified state and reward functions.
    - Managing the simulation state, including the current hour of the day, cumulative episode, episode reward, and
        episode vehicle count.
    - Building the SUMO command to run the simulation with the appropriate configuration parameters.
    - Stepping the simulation forward and observing the state of the traffic signals.
    - Resetting the simulation environment for a new episode.
    - Calculating and storing various metrics for the current simulation step, including queue lengths, maximum queues,
        and vehicle counts.
    - Saving the accumulated metrics to a CSV file.
    """

    def __init__(self, state_fn, reward_fn):
        self.reward_fn = reward_fn
        self.hour_of_day = 0
        self.cumulative_episode = 0
        self.episode_reward = 0.0
        self.episode_vehicles = 0
        self.metrics = list()
        self.original_route = cfg.route
        self.sumo_cmd = None
        self.sumo = None  # Set by start_sumo
        self.best_reward = None
        self.best_episode = None
        if "prob_test" in cfg:
            self.context = 0

        state_wrapped = state_fn
        if cfg.flat_state:

            def flat_wrapper(signals):
                flat_state = state_fn(signals)
                for signal in flat_state:
                    flat_state[signal] = flat_state[signal].ravel()
                return flat_state

            state_wrapped = flat_wrapper
        self.state_fn = state_wrapped

        self.date = self.resolve_date()

        self.sumo_start()
        self.signal_ids = self.sumo.trafficlight.getIDList()
        logger.info("lights {0} {1}".format(len(self.signal_ids), self.signal_ids))
        # self.signal_ids = [] This will pass through all control to SUMO's .net.xml TLLogic controllers

        self.signals = dict()
        # Pull signal observation shapes
        for signal_id in self.signal_ids:
            self.signals[signal_id] = Signal(self.sumo, signal_id)
            self.signals[signal_id].signals = (
                self.signals
            )  # Facilitates signal communication
            self.signals[signal_id].observe()
        observations = self.state_fn(self.signals)

        self.sumo_cmd = None  # Force regeneration of sumo command

        self.obs_act = dict()
        for signal_id in observations:
            act_size = (
                None  # Some are not a real signal (used for managers / hierarchy)
            )
            if signal_id in self.signals:
                act_size = len(self.signals[signal_id].green_phases)
            self.obs_act[signal_id] = (observations[signal_id].shape, act_size)

            if cfg.state == "state_builder":
                print("Built state:", signal_id, observations[signal_id].shape)

        # Override action space if not standard 'Phase' type
        self.action_set = None
        if cfg.action_set is not None and cfg.action_set != "Phase":
            self.action_set = getattr(action_sets, cfg.action_set)(
                self.obs_act, self.signals
            )
            if cfg.algorithm != "FIXED":
                for ts in self.obs_act:
                    self.obs_act[ts] = (self.obs_act[ts][0], self.action_set.num_acts)

        # Calculate decay period from % input for convenience
        steps_per_episode = int((cfg.end_time - cfg.start_time) / cfg.step_length)
        cfg.steps = steps_per_episode * cfg.episodes
        print(f"cfg.steps: {cfg.steps}")
        if "epsilon_decay_period" in cfg:  # Convert decay period from % to # of steps
            cfg.epsilon_decay_period = cfg.epsilon_decay_period * cfg.steps

        # Set policy for uncontrolled signals
        if cfg.controlled_signals is not None:
            if cfg.uncontrolled_policy == "FIXED":
                self.uncontrolled_policy = FIXED(self.obs_act)
                __ = self.uncontrolled_policy.act(
                    self.signals
                )  # Advance one step for correct FIXED timing
            elif cfg.uncontrolled_policy == "MAXWAVE":
                self.uncontrolled_policy = MAXWAVE(self.obs_act)
            elif cfg.uncontrolled_policy == "MAXPRESSURE":
                self.uncontrolled_policy = MAXPRESSURE(self.obs_act)
        if cfg.algorithm == "IMultiDQN":
            self.uncontrolled_policy = MAXPRESSURE(self.obs_act)

    def partial_lane_closure(self, edge_id, lane_index, position, duration=3600):
        ghost_id = "ghost_" + str(uuid.uuid4())
        traci.vehicletype.copy("Car", ghost_id)
        traci.vehicletype.setShapeClass(ghost_id, "bus/coach")
        traci.vehicletype.setColor(ghost_id, (255, 255, 255, 255))
        traci.vehicletype.setWidth(ghost_id, 3.2)
        position = traci.lane.getLength(edge_id + f"_{lane_index}") - position
        traci.route.add(ghost_id, [edge_id])
        traci.vehicle.add(
            vehID=ghost_id,
            routeID=ghost_id,
            typeID=ghost_id,
        )
        traci.vehicle.deactivateGapControl(ghost_id)
        traci.vehicle.moveTo(ghost_id, edge_id + f"_{lane_index}", position)
        traci.vehicle.setStop(
            vehID=ghost_id,
            edgeID=edge_id,
            laneIndex=lane_index,
            pos=position,
            duration=duration,
        )
        traci.vehicle.setLength(ghost_id, position)

    def resolve_date(self):
        date = None
        if cfg.run_hour is not None:
            if "peak_date" not in cfg or cfg.peak_date is None:
                # Find peak date
                cfg.peak_date, cfg.peak_hour, cfg.low_hour = self.find_peak_date()
            else:
                date = datetime.datetime.strptime(cfg.peak_date, "%Y-%m-%d")
        elif "start_date" in cfg and cfg.start_date is not None:
            date = datetime.datetime.strptime(cfg.start_date, "%Y-%m-%d")
        return date

    def find_peak_date(self):  # TODO rewrite and move out of here
        """
        Finds the peak date, peak hour, and low volume hour for the traffic data.

        This function iterates through the days and hours in the specified date range, calculating the maximum and minimum
        traffic volumes for each day. It then returns the day with the maximum traffic volume, the hour with the maximum
        traffic volume, and the hour with the minimum traffic volume.

        The function also logs the peak date, peak hour, peak volume, low volume hour, and low volume to the logger.

        Returns:
            tuple: The peak date, peak hour, and low volume hour.
        """
        logger.info("Finding peak date")
        day = datetime.datetime.strptime(cfg.start_date, "%Y-%m-%d")
        max_day = day
        hour = 0
        max_hour = 0
        max_volume = 0
        min_hour = 0
        min_volume = None
        daily_min_hour = 0
        daily_min_volume = 0
        end_date = datetime.datetime.strptime(cfg.end_date, "%Y-%m-%d")
        while True:
            cfg.start_time = hour * 3600
            cfg.end_time = cfg.start_time + 3600
            hour += 1
            if hour == 24:
                if day == max_day:
                    min_hour = daily_min_hour
                    min_volume = daily_min_volume
                day += datetime.timedelta(days=1)
                logger.info(
                    "Peak date: {0}, Peak hour: {1}, Peak volume: {2}, Low volume hour: {3}, Low volume: {4}, Current: {5}".format(
                        max_day, max_hour - 1, max_volume, min_hour - 1, min_volume, day
                    )
                )
                if day == end_date:
                    break
                hour = 0
                daily_min_hour = 0
                daily_min_volume = float("inf")
            _, vehicles = generate_flow_from_csv(day, self.original_route)

            if vehicles > max_volume:
                max_volume = vehicles
                max_day = day
                max_hour = hour
            if vehicles < daily_min_volume:
                daily_min_volume = vehicles
                daily_min_hour = hour
        logger.warning(
            "HIGHLY RECOMMENDED: Set peak_date in config to avoid recalculating peak date"
        )
        return max_day, max_hour, min_hour

    def build_sumo_cmd(self):
        """
        Builds the SUMO command to run the simulation.

        This method sets up the SUMO command with the appropriate configuration parameters, including the network file,
        route file, step length, and other options. It also handles adjusting the start and end times based on the
        configuration settings, such as running at the peak or low hour.

        The generated SUMO command is returned as a list of strings.
        """
        if self.sumo_cmd is None:
            # Find SUMO
            if cfg.gui and self.cumulative_episode != 0:
                binary = checkBinary("sumo-gui")
            else:
                binary = checkBinary("sumo")

            # Adjust start/end for config settings
            if cfg.run_hour is not None:
                if cfg.run_hour == "peak":
                    cfg.start_time = cfg.peak_hour
                elif cfg.run_hour == "low":
                    cfg.start_time = cfg.low_hour
                else:
                    raise NotImplementedError("run_hour not in [peak, low, null]")
                cfg.end_time = cfg.start_time + 3600

            self.sumo_cmd = [
                binary,
                "--net-file",
                cfg.network,
                "--route-files",
                cfg.route,
                "--step-length",
                f"{cfg.step_ratio}",
                "--random",
                "True",
                "--no-warnings",
                "True",
                "--no-step-log",
                "True",
                "--time-to-teleport",
                "-1",
                "--extrapolate-departpos",
                "True",
                "--tripinfo-output.write-unfinished",
                "True",
                "--tripinfo-output.write-undeparted",
                "True",
                "--eager-insert",  # Undeparted not written correctly without this, slows down simulation though
            ]

            if "saltlake" not in cfg.map and cfg.run_hour is None:
                self.sumo_cmd += [
                    "--begin",
                    str(cfg.start_time),
                    "--end",
                    str(cfg.end_time),
                ]

        sumo_cmd = self.sumo_cmd + [
            "--tripinfo-output",
            cfg.run_path + "tripinfo_{0}.xml".format(self.cumulative_episode),
        ]

        # Years long data requires a different start/end time between episodes
        if "saltlake" in cfg.map or cfg.run_hour is not None:
            sumo_cmd += ["--begin", str(cfg.start_time), "--end", str(cfg.end_time)]

        state_file = cfg.run_path + "state.xml.gz"
        if os.path.exists(state_file):
            sumo_cmd += ["--load-state", state_file]
        if cfg.load_sim is not None and cfg.load_sim:
            sumo_cmd += ["--load-state", cfg.load_sim]

        logger.debug(" ".join(sumo_cmd))
        return sumo_cmd

    def curriculum_next(self):
        if self.cumulative_episode % cfg.episodes == 0:
            cfg.flow = cfg.curriculum.pop(0)

    def reset(self, seed=None, options=None):
        self.sumo_close()

        if cfg.curriculum is not None:
            self.curriculum_next()

        info = {"out_of_data": False, "environment": self}
        # Start a new simulation
        # Generate flow files for altered flows / move date forward for saltlake year long runs
        if cfg.run_hour is not None:
            self.date = datetime.datetime.strptime(cfg.peak_date, "%Y-%m-%d")
            cfg.route, self.episode_vehicles = generate_flow_from_csv(
                self.date, self.original_route
            )
        elif "saltlake" in cfg.map:
            if "prob_test" in cfg:
                if self.cumulative_episode % 10 == 0 and self.cumulative_episode != 0:
                    self.context += 1
                    if self.context > 1:
                        self.context = 0
                found = False
                for _ in range(10):
                    if found:
                        logger.info(
                            f"Running day {self.date}, hour {cfg.start_time} to {cfg.end_time}, context {self.context}, vehicles {self.episode_vehicles}"
                        )
                        break
                    try:
                        for _ in range(cfg.episodes):
                            self.get_saltlake_data()
                            if self.context == 0 and 900 < self.episode_vehicles < 1100:
                                found = True
                                break  # Context 1
                            if (
                                self.context == 1
                                and 2900 < self.episode_vehicles < 3100
                            ):
                                found = True
                                break  # Context 2
                    except:
                        pass
            else:
                self.get_saltlake_data()
        elif (
            cfg.flow != 0 or cfg.curriculum is not None
        ):  # If the flow is altered, generate new flow files
            cfg.route, self.episode_vehicles = generate_additional_flow()

        self.episode_reward = 0.0
        self.cumulative_episode += 1

        self.sumo_start()

        for ts in self.signal_ids:
            self.signals[ts] = Signal(self.sumo, ts)
            self.signals[ts].signals = self.signals
            self.signals[ts].observe()

        return self.state_fn(self.signals), info

    def get_saltlake_data(self):
        cfg.start_time = self.hour_of_day * 3600
        cfg.end_time = cfg.start_time + 3600
        self.hour_of_day += 1
        if self.hour_of_day == 24:
            self.date += datetime.timedelta(days=1)
            self.hour_of_day = 0
        self.handle_missing_data()
        try:
            cfg.route, self.episode_vehicles = generate_flow_from_csv(
                self.date, self.original_route
            )
        except FileNotFoundError:
            raise RuntimeError("Out of data for saltlake, ending run")

    def handle_missing_data(self):
        """
        Handles missing data by adjusting the simulation date and hour of day when the simulation encounters a period
        of missing data.

        If the current simulation date and hour of day match the start of a missing data period, the function
        advances the simulation date and hour of day until it is past the end of the missing data period.
        """
        for start_dt_str, end_dt_str in zip(cfg.missing_start, cfg.missing_end):
            missing_start = datetime.datetime.strptime(start_dt_str, "%Y-%m-%d %H:%M")
            missing_end = datetime.datetime.strptime(end_dt_str, "%Y-%m-%d %H:%M")
            if (
                self.date.date() == missing_start.date()
                and self.hour_of_day == missing_start.hour
            ):
                while self.date < missing_end:
                    self.date += datetime.timedelta(hours=1)
                self.hour_of_day = missing_end.hour

    def uncontrolled_acts(self, act):
        """
        If some of the signals are not controlled with the agent policy, replace uncontrolled signals with the
        actions from the defined uncontrolled policy.
        """
        if cfg.controlled_signals is not None:
            if cfg.uncontrolled_policy == "FIXED":
                uncontrolled_acts = self.uncontrolled_policy.act(wave(self.signals))
            elif cfg.uncontrolled_policy == "MAXWAVE":
                uncontrolled_acts = self.uncontrolled_policy.act(wave(self.signals))
            elif cfg.uncontrolled_policy == "MAXPRESSURE":
                uncontrolled_acts = self.uncontrolled_policy.act(mplight(self.signals))
            else:
                raise NotImplementedError(
                    "uncontrolled_policy not in [FIXED, MAXWAVE, MAXPRESSURE]"
                )
            for signal_id in self.signals:
                if signal_id not in cfg.controlled_signals:
                    act[signal_id] = uncontrolled_acts[signal_id]
        return act

    def step(self, act):
        if self.action_set is not None:
            act = self.action_set.act(act)
        act = self.uncontrolled_acts(act)

        cutoff = traci.simulation.getTime()
        for signal in self.signals:
            self.signals[signal].switch_phase(act[signal])
        for __ in range(cfg.step_length):
            self.sumo.simulationStep()
            for signal in self.signal_ids:
                self.signals[signal].step()
            cutoff += 1
            if cutoff > cfg.end_time:
                break
        for signal in self.signal_ids:
            self.signals[signal].observe()

        observations = self.state_fn(self.signals)
        rewards = self.reward_fn(self.signals)

        current_time = traci.simulation.getTime() - 1

        self.episode_reward += sum(rewards.values())
        self.calc_metrics(rewards, current_time)
        terminated = current_time >= cfg.end_time
        truncated = False

        info = {
            "environment": self,
        }

        if "init_ender" in cfg:
            pending = traci.simulation.getPendingVehicles()

            max_depart = 0
            for veh_id in pending:
                veh_depart = traci.vehicle.getDepartDelay(veh_id)
                if veh_depart > max_depart:
                    max_depart = veh_depart
            if max_depart >= 1800:
                cfg.init_ender = True

        if terminated:
            # Remove ghosts (vehicles used for partial lane closures)
            all_vehicles = traci.vehicle.getIDList()
            for vehicle in all_vehicles:
                if vehicle.startswith("ghost"):
                    traci.vehicle.remove(vehicle)
            if (
                "saltlake" in cfg.map and cfg.run_hour is None
            ):  # Save state for continual simulation
                if "prob_test" in cfg:
                    for veh in self.sumo.vehicle.getIDList():
                        road = self.sumo.vehicle.getRoadID(veh)
                        self.sumo.vehicle.remove(veh)

                self.sumo.simulation.saveState(cfg.run_path + "state.xml.gz")
                for signal in self.signals:
                    self.signals[signal].sumo = None
                pickle.dump(self.signals, open(cfg.run_path + "signals.pkl", "wb"))

            if self.best_reward is None or self.episode_reward > self.best_reward:
                self.best_reward = self.episode_reward
                self.best_episode = self.cumulative_episode
            logger.info(
                "Episode: {0}, Best: {1}, Best Reward: {2}, Episode Reward: {3}".format(
                    self.cumulative_episode,
                    self.best_episode,
                    self.best_reward,
                    self.episode_reward,
                )
            )
        return observations, rewards, terminated, truncated, info

    def calc_metrics(self, rewards, current_time):
        """
        Calculates and stores various metrics for the current simulation step, including:
        - Queue lengths for each signal
        - Maximum queue lengths for each signal
        - Number of vehicles for each signal

        These metrics are appended to the `self.metrics` list, with each entry containing the current simulation
         time, the rewards for the current step, and the calculated metrics.
        """
        queue_lengths, max_queues, vehicles, phase_length = (
            dict(),
            dict(),
            dict(),
            dict(),
        )
        for signal_id in self.signals:
            signal = self.signals[signal_id]
            queue_lengths[signal_id] = signal.observation.total_queued
            max_queues[signal_id] = signal.observation.max_queue
            vehicles[signal_id] = (
                self.episode_vehicles
            )  # Match other metric formats, simplify reading
            phase_length[signal_id] = signal.observation.time_since_phase[
                signal.current_phase
            ]

        self.metrics.append(
            {
                "step": current_time,
                "rewards": rewards,
                "max_queues": max_queues,
                "queue_lengths": queue_lengths,
                "vehicles": vehicles,
                "phase_length": phase_length,
            }
        )

    def save_metrics(self):
        log = cfg.run_path + "metrics_{0}.csv".format(self.cumulative_episode)
        csv_build = list(cfg.csv_metrics)
        csv_build[-1] += "\n"

        for step in self.metrics:
            # print(f"step:{step}")
            for metric in cfg.csv_metrics:
                csv_build.append(str(step[metric]))
            csv_build[-1] += "\n"
        with open(log, "w+") as output_file:
            output_file.write(",".join(csv_build))
        self.metrics.clear()

    def render(self, mode="human"):
        if mode == "human":
            raise EnvironmentError("Set gui parameter to render GUI")

    def sumo_start(self):
        """
        Starts the SUMO simulation and sets up the necessary connections.

        If `cfg.libsumo` is True, the simulation is started using the LIBSUMO bindings. Otherwise, the simulation is started
        using the traci library over a local network connection.
        """
        cmd = self.build_sumo_cmd()
        if cfg.libsumo:
            traci.start(cmd)
            self.sumo = traci
        else:
            traci.start(cmd, label=cfg.uuid)
            self.sumo = traci.getConnection(cfg.uuid)
        if "closures" in cfg:
            for closure in cfg.closures:
                self.partial_lane_closure(
                    closure["edge_id"], closure["lane_index"], closure["position"]
                )
        if os.path.exists(cfg.run_path + "signals.pkl"):
            self.signals = pickle.load(open(cfg.run_path + "signals.pkl", "rb"))
            for signal_id in self.signals:
                signal = self.signals[signal_id]
                signal.sumo = self.sumo
            # Remove vehicles loaded inside an intersection
            for veh in self.sumo.vehicle.getIDList():
                road = self.sumo.vehicle.getRoadID(veh)
                if ":" in road or (
                    cfg.clean_nightly is not None
                    and self.hour_of_day == cfg.clean_nightly
                ):
                    self.sumo.vehicle.remove(veh)

    def sumo_close(self):
        if not cfg.libsumo:
            traci.switch(cfg.uuid)
        traci.close()
        if len(self.metrics) != 0:
            self.save_metrics()

    def close(self):
        """
        Closes the SUMO simulation and saves any accumulated metrics. If the current run has a custom route file,
        it is also removed.
        """
        self.sumo_close()
        if cfg.uuid in cfg.route:
            os.remove(cfg.route)
            cfg.route = self.original_route
