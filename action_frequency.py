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
    d = collections.defaultdict(lambda: 0)

    for trajectory in tqdm.tqdm(trajectories):
        actions = actions_from_file(data_set, trajectory)
        for action in actions:
            d[action] += 1

    freq = [[i, d[i]] for i in d]
    freq.sort(key=lambda x: x[1], reverse=True)

    freq = [f'{i[1]}:{i[0]}' for i in freq]
    output_file = f'Action frequencies {data_set}.txt'
    with open(output_file, 'w', newline='') as txt:
        for i in freq:
            txt.write(i + '\n')


if __name__ == "__main__":
    main()
