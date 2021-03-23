import os
import collections
from utils import actions_from_file, MINERL_DATA_ROOT
import tqdm
import sys


def main():
    if len(sys.argv) > 1:
        data_set = sys.argv[1]
    else:
        data_set = 'MineRLObtainDiamond-v0'
    path = os.path.join(MINERL_DATA_ROOT, data_set)
    trajectories = os.listdir(path)
    trajectories.sort()

    d = collections.defaultdict(lambda: [0] * len(trajectories))
    for i, trajectory in tqdm.tqdm(enumerate(trajectories)):
        actions = actions_from_file(data_set, trajectory)
        for action in actions:
            d[action][i] = 1

    frequency = [[i, sum(d[i])] for i in d]
    frequency.sort(key=lambda x: x[1], reverse=True)
    frequency = [f'{i[1]}:{i[0]}' for i in frequency]
    output_file = f'Action frequencies in trajectories {data_set}.txt'
    with open(output_file, 'w', newline='') as txt:
        for i in frequency:
            txt.write(i + '\n')


if __name__ == "__main__":
    main()
