import numpy as np
import os
from utils import actions_from_file, MINERL_DATA_ROOT
import tqdm
import sys
import pickle


def main():
    if len(sys.argv) > 1:
        data_set = sys.argv[1]
    else:
        data_set = 'MineRLObtainDiamond-v0'
    data_set_obf = data_set[:-3] + 'VectorObf' + data_set[-3:]
    path = os.path.join(MINERL_DATA_ROOT, data_set)
    path_obf = os.path.join(MINERL_DATA_ROOT, data_set_obf)
    trajectories = os.listdir(path)
    trajectories.sort()
    d = {}
    d_obf = {}  # technically not necessary, but helps run the whole thing faster at a cost of some extra memory

    for trajectory in tqdm.tqdm(trajectories):
        actions = actions_from_file(data_set, trajectory)
        f_obf = np.load(os.path.join(path_obf, trajectory, 'rendered.npz'))
        actions_obf = [tuple(i) for i in f_obf['action$vector']]
        for i in range(len(actions)):
            if actions_obf[i] not in d_obf:
                action = actions[i]
                d[action] = np.array(actions_obf[i])
                d_obf[actions_obf[i]] = action

    output_file = f'Action mapping {data_set}.pkl'
    with open(output_file, 'wb') as file:
        pickle.dump(d, file)


if __name__ == "__main__":
    main()
