import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from avg_timeLoss import delays
from avg_duration import durations
from avg_waitingTime import waiting
from avg_queue import queue

map_title = {
    'grid4x4': '4x4 Grid',
    'arterial4x4': '4x4 Avenues',
    'ingolstadt1': 'Ingolstadt Single Signal',
    'ingolstadt7': 'Ingolstadt Corridor',
    'ingolstadt21': 'Ingolstadt Region',
    'cologne1': 'Cologne Single Signal',
    'cologne3': 'Cologne Corridor',
    'cologne8': 'Cologne Region'
}

alg_name = {
    'FIXED': 'Fixed Time',
    'STOCHASTIC': 'Random',
    'MAXWAVE': 'Greedy',
    'MAXPRESSURE': 'Max Pressure',
    'FULLMAXPRESSURE': 'Max Pressure w/ All phases',
    'IDQN': 'IDQN',
    'MPLight': 'MPLight',
    'MPLightFULL': 'Full State MPLight',
    'FMA2C': 'FMA2C',
    'IPPO': 'IPPO'
}

statics = ['MAXPRESSURE', 'STOCHASTIC', 'MAXWAVE', 'FIXED']

num_n = -1
num_episodes = 120
fs = 21
window_size = 5

metrics = [delays, durations, waiting, queue]
metrics_str = ['Avg. Delay', 'Avg. Trip Time', 'Avg. Wait', 'Avg. Queue']

chart = {
    'IDQN': {
        'Avg. Delay': [],
        'Avg. Wait': [],
        'Avg. Queue': [],
        'Avg. Trip Time': []
    },
    'IPPO': {
        'Avg. Delay': [],
        'Avg. Wait': [],
        'Avg. Queue': [],
        'Avg. Trip Time': []
    },
    'MPLight': {
        'Avg. Delay': [],
        'Avg. Wait': [],
        'Avg. Queue': [],
        'Avg. Trip Time': []
    },
    'FMA2C': {
        'Avg. Delay': [],
        'Avg. Wait': [],
        'Avg. Queue': [],
        'Avg. Trip Time': []
    },
    'Full State MPLight': {
        'Avg. Delay': [],
        'Avg. Wait': [],
        'Avg. Queue': [],
        'Avg. Trip Time': []
    },
}

for met_i, metric in enumerate(metrics):
    print('\n', metrics_str[met_i])
    for map in map_title.keys():
        print()
        print(map_title[map])
        dqn_max = 0
        plt.gca().set_prop_cycle(None)
        for key in metric:
            if map in key and '_yerr' not in key:
                alg = key.split(' ')[0]
                key_map = key.split(' ')[1]

                if alg == 'IDQN': dqn_max = np.max(metric[key])     # Set ylim to DQN max, it's approx. random perf.

                if len(metric[key]) == 0:   # Fixed time isn't applicable to valid. scenario, skip color for consistency
                    plt.plot([], [])
                    plt.fill_between([], [], [])
                    continue

                # Print out performance metric
                err = metric.get(key + '_yerr')
                if num_n == -1:
                    last_n_ind = np.argmin(metric[key])
                    last_n = metric[key][last_n_ind]
                else:
                    last_n_ind = np.argmin(metric[key][-num_n:])
                    last_n = metric[key][-num_n:][last_n_ind]
                last_n_err = 0 if err is None else err[last_n_ind]
                avg_tot = np.mean(metric[key])
                avg_tot = np.round(avg_tot, 2)
                last_n = np.round(last_n, 2)
                last_n_err = np.round(last_n_err, 2)

                #last_n = np.round(np.mean(err), 2) if err is not None else 0
                #last_n = last_n_ind

                # Print stats
                if alg in statics:
                    print('{} {}'.format(alg_name[alg], avg_tot))
                    do_nothing = 0
                else:
                    print('{} {} +- {}'.format(alg_name[alg], last_n, last_n_err))
                    if not(map == 'grid4x4' or map == 'arterial4x4'):
                        chart[alg_name[alg]][metrics_str[met_i]].append(str(last_n)) #+' $\pm$ '+str(last_n_err)

                # Build plots
                if alg in statics:
                    plt.plot([avg_tot]*num_episodes, '--', label=alg_name[alg])
                    plt.fill_between([], [], [])      # Advance color cycle
                elif not('FMA2C' in alg or 'IPPO' in alg):
                    windowed = []
                    queue = deque(maxlen=window_size)
                    std_q = deque(maxlen=window_size)

                    windowed_yerr = []
                    x = []
                    for i, eps in enumerate(metric[key]):
                        x.append(i)
                        queue.append(eps)
                        windowed.append(np.mean(queue))
                        if err is not None:
                            std_q.append(err[i])
                            windowed_yerr.append(np.mean(std_q))

                    windowed = np.asarray(windowed)
                    if err is not None:
                        windowed_yerr = np.asarray(windowed_yerr)
                        low = windowed - windowed_yerr
                        high = windowed + windowed_yerr
                    else:
                        low = windowed
                        high = windowed

                    plt.plot(windowed, label=alg_name[alg])
                    plt.fill_between(x, low, high, alpha=0.4)
                else:
                    if alg == 'FMA2C':  # Skip pink in color cycle
                        plt.plot([], [])
                        plt.fill_between([], [], [])
                    alg = key.split(' ')[0]
                    x = [num_episodes-1, num_episodes]
                    y = [last_n]*2
                    plt.plot(x, y, label=alg_name[alg])
                    plt.fill_between([], [], [])  # Advance color cycle

        points = np.asarray([0, 20, 40, 60, 80, 100, num_episodes])
        labels = ('0', '20', '40', '60', '80', '100', '..1400')
        plt.yticks(fontsize=fs)
        plt.xticks(points, labels, fontsize=fs)
        #plt.xlabel('Episode', fontsize=32)
        #plt.ylabel('Delay (s)', fontsize=32)
        plt.title(map_title[map], fontsize=fs)
        #plt.legend(prop={'size': 25})
        bot, top = plt.ylim()
        if bot < 0: bot = 0
        plt.ylim(bot, dqn_max)
        plt.show()

for alg in chart:
    print(alg)
    for met in metrics_str:
        print(met, ' & ', ' & '.join(chart[alg][met]), '\\\\')
