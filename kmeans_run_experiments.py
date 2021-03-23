import numpy as np
import os
import pickle
import random
from datetime import datetime
from sklearn.cluster import KMeans
from utils import MINERL_DATA_ROOT, deobfuscate_kmeans_actions

DATA_KMEANS = 'MineRLObtainDiamondVectorObf-v0'
N_CLUSTERS = [10, 20, 30, 50, 70, 100, 150]
N_ACTIONS = [100000]
N_ITERATIONS = 5
OUTPUT_PATH = 'out'


def train_kmeans(data_set='MineRLObtainDiamondVectorObf-v0', n_clusters=70, n_actions=100000):
    """
    Returns kmeans clustering of a given data set and parameters.
    :param data_set: name of the data set
    :param n_clusters: number of kmeans clusters
    :param n_actions: number of actions to train on
    :return: KMeans object
    """
    actions = []
    trajectories = os.listdir(os.path.join(MINERL_DATA_ROOT, data_set))
    for trajectory in trajectories:
        f = np.load(os.path.join(MINERL_DATA_ROOT, data_set, trajectory, 'rendered.npz'))
        for action in f['action$vector']:
            actions.append(action)
    random.shuffle(actions)
    actions = actions[:n_actions]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(actions)
    return kmeans


def main():
    for n_actions in N_ACTIONS:
        out_path = os.path.join(OUTPUT_PATH, DATA_KMEANS, str(n_actions))
        os.makedirs(out_path, exist_ok=True)
    pkl_path = 'Action mapping ' + DATA_KMEANS.replace('VectorObf', '') + '.pkl'
    with open(pkl_path, 'rb') as pkl:
        action_mapping_dict = pickle.load(pkl)

    for _ in range(N_ITERATIONS):
        for n_clusters in N_CLUSTERS:
            for n_actions in N_ACTIONS:
                kmeans = train_kmeans(DATA_KMEANS, n_clusters, n_actions)
                best_mses, best_actions = deobfuscate_kmeans_actions(kmeans, action_mapping_dict)

                dt = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
                output_file = os.path.join(OUTPUT_PATH, DATA_KMEANS, str(n_actions),
                                           f'kmeans deobfuscated, {n_clusters}, {dt}.txt')
                with open(output_file, 'w') as f:
                    f.write('best_mse, best_action' + '\n')
                    for i in range(len(best_actions)):
                        f.write(f'{best_mses[i]},{best_actions[i]}' + '\n')


if __name__ == "__main__":
    main()
