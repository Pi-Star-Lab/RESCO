import os
import numpy as np
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

log_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'results' + os.sep)
env_base = '..'+os.sep+'environments'+os.sep
names = [folder for folder in next(os.walk(log_dir))[1]]

metric = 'queue'
output_file = 'avg_{}.py'.format(metric)
run_avg = dict()

for name in names:
    split_name = name.split('-')
    print(split_name)
    map_name = split_name[2]
    average_per_episode = []
    for i in range(1, 10000):
        trip_file_name = log_dir+name + os.sep + 'metrics_'+str(i)+'.csv'
        if not os.path.exists(trip_file_name):
            print('No '+trip_file_name)
            break

        num_steps, total = 0, 0.0
        last_departure_time = 0
        last_depart_id = ''
        with open(trip_file_name) as fp:
            reward, wait, steps = 0, 0, 0
            for line in fp:
                line = line.split('}')
                queues = line[2]
                signals = queues.split(':')
                step_total = 0
                for s, signal in enumerate(signals):
                    if s == 0: continue
                    queue = signal.split(',')
                    queue = int(queue[0])
                    step_total += queue
                step_avg = step_total / len(signals)
                total += step_avg
                num_steps += 1

        average = total / num_steps
        average_per_episode.append(average)

    run_name = split_name[0]+' '+split_name[2]+' '+split_name[3]+' '+split_name[4]+' '+split_name[5]
    average_per_episode = np.asarray(average_per_episode)

    if run_name in run_avg:
        run_avg[run_name].append(average_per_episode)
    else:
        run_avg[run_name] = [average_per_episode]

alg_res = []
alg_name = []
for run_name in run_avg:
    list_runs = run_avg[run_name]
    min_len = min([len(run) for run in list_runs])
    list_runs = [run[:min_len] for run in list_runs]
    avg_delays = np.sum(list_runs, 0)/len(list_runs)
    err = np.std(list_runs, axis=0)

    alg_name.append(run_name)
    alg_res.append(avg_delays)

    alg_name.append(run_name+'_yerr')
    alg_res.append(err)

    plt.title(run_name)
    plt.plot(avg_delays)
    plt.show()


np.set_printoptions(threshold=sys.maxsize)
with open(output_file, 'a') as out:
    for i, res in enumerate(alg_res):
        out.write("'{}': {},\n".format(alg_name[i], res.tolist()))
