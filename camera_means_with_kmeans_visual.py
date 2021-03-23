import matplotlib.pyplot as plt
import os
import collections
import numpy as np
from utils import camera_stats

OUTPUT_PATH = 'out'
DATA_SET = 'MineRLObtainDiamondVectorObf-v0'
N_ACTIONS = 100000


def main():
    kmeans_data_path = os.path.join(OUTPUT_PATH, DATA_SET, str(N_ACTIONS))
    files = os.listdir(kmeans_data_path)
    files.sort()

    d_means = collections.defaultdict(lambda: [])
    for doc in files:
        stats = camera_stats(kmeans_data_path, doc)
        mean = sum(stats) / len(stats)
        d_means[int(doc.split(',')[1])].append(mean)

    x = []
    y = []
    for i in d_means:
        for j in range(len(d_means[i])):
            x.append(i)
            y.append(d_means[i][j])

    plt.plot(x, y, 'm_', ms=15)
    plt.axhline(y=0, color='black', linewidth=1, alpha=0.2)
    plt.xticks(np.arange(0, 160, step=10))
    plt.xlabel('Kmeans clusters')
    plt.ylabel('mean camera pitch (up/down) angle')
    plt.show()


if __name__ == "__main__":
    main()
