import numpy as np
import os
from utils import MINERL_DATA_ROOT, CUMULATIVE_REWARDS
import sys
import pandas


def time_to_rewards(data_set, trajectory):
    """
    Takes a data_set and a trajectory, and returns times (in ticks) to achieve each cumulative reward (from the last
    cumulative reward, not from start).
    :param data_set: data set name (for example: 'MineRLObtainDiamond-v0')
    :param trajectory: trajectory path
    :return: a list of times to achieve cumulative rewards
    """
    doc = os.path.join(MINERL_DATA_ROOT, data_set, trajectory, 'rendered.npz')
    f = np.load(doc)
    rewards = list(f['reward'])
    times = []
    c = 0
    sum_rew = 0
    for i in range(len(rewards)):
        while rewards[i] + sum_rew >= CUMULATIVE_REWARDS[c]:
            times.append(i)
            c += 1
        sum_rew += rewards[i]
    time_periods = [times[i] - times[i - 1] for i in range(1, len(times))]
    return time_periods


def main():
    if len(sys.argv) > 1:
        data_set = sys.argv[1]
    else:
        data_set = 'MineRLObtainDiamond-v0'
    path = os.path.join(MINERL_DATA_ROOT, data_set)
    trajectories = os.listdir(path)
    trajectories.sort()

    trajectory_times = []
    for trajectory in trajectories:
        time_periods = time_to_rewards(data_set, trajectory)
        trajectory_times.append(time_periods)

    reward_times = [[] for _ in range(len(CUMULATIVE_REWARDS[1:-1]))]
    for times in trajectory_times:
        for i in range(len(times)):
            reward_times[i].append(times[i])
    reward_times = [sorted(i) for i in reward_times]

    mean = [0] + [sum(i) // len(i) for i in reward_times if len(i) > 0]
    median = [0] + [i[len(i) // 2] for i in reward_times if len(i) > 0]
    counts = [len(trajectories)] + [len(i) for i in reward_times if len(i) > 0]

    d = {'mean': {}, 'median': {}, 'counts': {}}
    for i in range(len(mean)):
        d['mean'][CUMULATIVE_REWARDS[i]] = mean[i]
        d['median'][CUMULATIVE_REWARDS[i]] = median[i]
        d['counts'][CUMULATIVE_REWARDS[i]] = counts[i]

    print('\ntimes to achieve cumulative rewards(in ticks) and number of trajectories that achieve them')
    print(pandas.DataFrame.from_dict(d, orient='index').to_string())


if __name__ == "__main__":
    main()
