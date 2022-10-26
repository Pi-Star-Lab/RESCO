import numpy as np

from resco_benchmark.config.mdp_config import mdp_configs


def drq(signals):
    observations = {}
    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.phase
        for i, lane in enumerate(signal.lanes):
            lane_obs = []
            if i == act_index:
                lane_obs.append(1)
            else:
                lane_obs.append(0)

            lane_obs.extend(
                (
                    signal.full_observation[lane]["approach"],
                    signal.full_observation[lane]["total_wait"],
                    signal.full_observation[lane]["queue"],
                )
            )

            vehicles = signal.full_observation[lane]["vehicles"]
            total_speed = sum(vehicle["speed"] for vehicle in vehicles)
            lane_obs.append(total_speed)

            obs.append(lane_obs)
        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    return observations


def drq_norm(signals):
    observations = {}
    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.phase
        for i, lane in enumerate(signal.lanes):
            lane_obs = []
            if i == act_index:
                lane_obs.append(1)
            else:
                lane_obs.append(0)

            lane_obs.extend(
                (
                    signal.full_observation[lane]["approach"] / 28,
                    signal.full_observation[lane]["total_wait"] / 28,
                    signal.full_observation[lane]["queue"] / 28,
                )
            )

            vehicles = signal.full_observation[lane]["vehicles"]
            total_speed = sum((vehicle["speed"] / 20 / 28) for vehicle in vehicles)
            lane_obs.append(total_speed)

            obs.append(lane_obs)
        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    return observations


def mplight(signals):
    observations = {}
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [signal.phase]
        for direction in signal.lane_sets:
            # Add inbound
            queue_length = sum(
                signal.full_observation[lane]["queue"]
                for lane in signal.lane_sets[direction]
            )

            # Subtract downstream
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signalid[lane]
                if dwn_signal in signal.signals:
                    queue_length -= signal.signals[dwn_signal].full_observation[lane][
                        "queue"
                    ]
            obs.append(queue_length)
        observations[signal_id] = np.asarray(obs)
    return observations


def mplight_full(signals):
    observations = {}
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [signal.phase]
        for direction in signal.lane_sets:
            # Add inbound
            queue_length = 0
            total_wait = 0
            total_speed = 0
            tot_approach = 0
            for lane in signal.lane_sets[direction]:
                queue_length += signal.full_observation[lane]["queue"]
                total_wait += signal.full_observation[lane]["total_wait"] / 28
                total_speed = 0
                vehicles = signal.full_observation[lane]["vehicles"]
                for vehicle in vehicles:
                    total_speed += vehicle["speed"]
                tot_approach += signal.full_observation[lane]["approach"] / 28

            # Subtract downstream
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signalid[lane]
                if dwn_signal in signal.signals:
                    queue_length -= signal.signals[dwn_signal].full_observation[lane][
                        "queue"
                    ]
            obs.extend((queue_length, total_wait, total_speed, tot_approach))
        observations[signal_id] = np.asarray(obs)
    return observations


def wave(signals):
    observations = {}
    for signal_id in signals:
        signal = signals[signal_id]
        state = []
        for direction in signal.lane_sets:
            wave_sum = sum(
                signal.full_observation[lane]["queue"]
                + signal.full_observation[lane]["approach"]
                for lane in signal.lane_sets[direction]
            )

            state.append(wave_sum)
        observations[signal_id] = np.asarray(state)
    return observations


def ma2c(signals):
    ma2c_config = mdp_configs["MA2C"]

    signal_wave = {}
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane in signal.lanes:
            wave = (
                signal.full_observation[lane]["queue"]
                + signal.full_observation[lane]["approach"]
            )
            waves.append(wave)
        signal_wave[signal_id] = np.clip(
            np.asarray(waves) / ma2c_config["norm_wave"], 0, ma2c_config["clip_wave"]
        )

    observations = {}
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None:
                waves.append(ma2c_config["coop_gamma"] * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = [signal.full_observation[lane]["max_wait"] for lane in signal.lanes]
        waits = np.clip(
            np.asarray(waits) / ma2c_config["norm_wait"], 0, ma2c_config["clip_wait"]
        )

        observations[signal_id] = np.concatenate([waves, waits])
    return observations


def fma2c(signals):
    fma2c_config = mdp_configs["FMA2C"]
    management = fma2c_config["management"]
    supervisors = fma2c_config["supervisors"]  # reverse of management
    management_neighbors = fma2c_config["management_neighbors"]

    region_fringes = {manager: [] for manager in management}
    for signal_id in signals:
        signal = signals[signal_id]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is None or supervisors[neighbor] != supervisors[signal_id]:
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    lane_wave = {}
    for signal_id in signals:
        signal = signals[signal_id]
        for lane in signal.lanes:
            lane_wave[lane] = (
                signal.full_observation[lane]["queue"]
                + signal.full_observation[lane]["approach"]
            )

    manager_obs = {}
    for manager, lanes in region_fringes.items():
        waves = [lane_wave[lane] for lane in lanes]
        manager_obs[manager] = np.clip(
            np.asarray(waves) / fma2c_config["norm_wave"], 0, fma2c_config["clip_wave"]
        )

    management_neighborhood = {}
    for manager in manager_obs:
        neighborhood = [manager_obs[manager]]
        neighborhood.extend(
            fma2c_config["alpha"] * manager_obs[neighbor]
            for neighbor in management_neighbors[manager]
        )

        management_neighborhood[manager] = np.concatenate(neighborhood)

    signal_wave = {}
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane in signal.lanes:
            wave = (
                signal.full_observation[lane]["queue"]
                + signal.full_observation[lane]["approach"]
            )
            waves.append(wave)
        signal_wave[signal_id] = np.clip(
            np.asarray(waves) / fma2c_config["norm_wave"], 0, fma2c_config["clip_wave"]
        )

    observations = {}
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None and supervisors[neighbor] == supervisors[signal_id]:
                waves.append(fma2c_config["alpha"] * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = [signal.full_observation[lane]["max_wait"] for lane in signal.lanes]
        waits = np.clip(
            np.asarray(waits) / fma2c_config["norm_wait"], 0, fma2c_config["clip_wait"]
        )

        observations[signal_id] = np.concatenate([waves, waits])
    observations.update(management_neighborhood)
    return observations


def fma2c_full(signals):
    fma2c_config = mdp_configs["FMA2CFull"]
    management = fma2c_config["management"]
    supervisors = fma2c_config["supervisors"]  # reverse of management
    management_neighbors = fma2c_config["management_neighbors"]

    region_fringes = {manager: [] for manager in management}
    for signal_id in signals:
        signal = signals[signal_id]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is None or supervisors[neighbor] != supervisors[signal_id]:
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    lane_wave = {}
    for signal_id in signals:
        signal = signals[signal_id]
        for lane in signal.lanes:
            lane_wave[lane] = (
                signal.full_observation[lane]["queue"]
                + signal.full_observation[lane]["approach"]
            )

    manager_obs = {}
    for manager, lanes in region_fringes.items():
        waves = [lane_wave[lane] for lane in lanes]
        manager_obs[manager] = np.clip(
            np.asarray(waves) / fma2c_config["norm_wave"], 0, fma2c_config["clip_wave"]
        )

    management_neighborhood = {}
    for manager in manager_obs:
        neighborhood = [manager_obs[manager]]
        neighborhood.extend(
            fma2c_config["alpha"] * manager_obs[neighbor]
            for neighbor in management_neighbors[manager]
        )

        management_neighborhood[manager] = np.concatenate(neighborhood)

    signal_wave = {}
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane in signal.lanes:
            wave = (
                signal.full_observation[lane]["queue"]
                + signal.full_observation[lane]["approach"]
            )
            waves.extend((wave, signal.full_observation[lane]["total_wait"] / 28))
            vehicles = signal.full_observation[lane]["vehicles"]
            total_speed = sum((vehicle["speed"] / 20 / 28) for vehicle in vehicles)
            waves.append(total_speed)
        signal_wave[signal_id] = np.clip(
            np.asarray(waves) / fma2c_config["norm_wave"], 0, fma2c_config["clip_wave"]
        )

    observations = {}
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None and supervisors[neighbor] == supervisors[signal_id]:
                waves.append(fma2c_config["alpha"] * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = [signal.full_observation[lane]["max_wait"] for lane in signal.lanes]
        waits = np.clip(
            np.asarray(waits) / fma2c_config["norm_wait"], 0, fma2c_config["clip_wait"]
        )

        observations[signal_id] = np.concatenate([waves, waits])
    observations.update(management_neighborhood)
    return observations
