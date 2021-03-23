import pickle
import joblib
import sys
from utils import deobfuscate_kmeans_actions, max_rewards

KMEANS_FILENAME = 'kmeans.joblib'
DATA_SET = 'MineRLObtainDiamond-v0'


def main():
    if len(sys.argv) == 2:
        file = sys.argv[1]
    else:
        file = KMEANS_FILENAME

    actions = set()
    pkl_filename = f'Action mapping {DATA_SET}.pkl'
    with open(pkl_filename, 'rb') as pkl:
        acts = pickle.load(pkl)

    if file.endswith('.pkl'):
        with open(file, 'rb') as pkl_filename:
            kmeans = pickle.load(pkl_filename)
    else:  # it's a .joblib file
        kmeans = joblib.load(file, mmap_mode=None)

    print(f'Deobfuscated actions of {file}:')
    _, best_actions = deobfuscate_kmeans_actions(kmeans, acts)
    for best_action in best_actions:
        for action in best_action[1:]:  # ignore camera action
            actions.add(action)
        print(best_action)

    print(f'\nMaximum possible reward: {max_rewards(actions)}')


if __name__ == "__main__":
    main()
