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

    d_mini = collections.defaultdict(lambda: [])
    d_maxi = collections.defaultdict(lambda: [])
    for doc in files:
        stats = camera_stats(kmeans_data_path, doc)
        mini, maxi = min(stats), max(stats)
        d_mini[int(doc.split(',')[1])].append(mini)
        d_maxi[int(doc.split(',')[1])].append(maxi)

    x = []
    y = []
    y1 = []
    for i in d_mini:
        for j in range(len(d_mini[i])):
            x.append(i)
            y.append(d_mini[i][j])
            y1.append(d_maxi[i][j])

    plt.plot(x, y, 'r_', ms=15)
    plt.plot(x, y1, 'g_', ms=15)
    plt.axhline(y=0, color='black', linewidth=1, alpha=0.2)
    plt.xticks(np.arange(0, 160, step=10))
    plt.xlabel('Kmeans clusters')
    plt.ylabel('min/max camera pitch (up/down) angle')
    plt.show()


if __name__ == "__main__":
    main()
