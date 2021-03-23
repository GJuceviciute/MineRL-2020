import collections
import os
import pandas
import copy
from utils import max_rewards, CUMULATIVE_REWARDS

OUTPUT_PATH = 'out'
DATA_SET = 'MineRLObtainDiamondVectorObf-v0'
N_ACTIONS = 100000


def main():
    kmeans_data_path = os.path.join(OUTPUT_PATH, DATA_SET, str(N_ACTIONS))
    files = os.listdir(kmeans_data_path)
    files.sort()

    cumulative = CUMULATIVE_REWARDS[:-1]
    reward_frequency = collections.defaultdict(lambda: {str(r): 0 for r in cumulative})

    for doc in files:
        mini_actions = {}
        with open(os.path.join(kmeans_data_path, doc)) as txt:
            txt.readline()
            for line in txt.readlines():
                action = eval(line[line.index(',') + 1:-1])
                for act in action:
                    mini_actions[act] = 1
        max_rew = str(max_rewards(mini_actions))

        reward_frequency[int(doc.split(',')[1])][max_rew] += 1

    print('\nNumber of kmeans action obfuscations with different possible max rewards')
    print(pandas.DataFrame.from_dict(reward_frequency, orient='index').sort_index())

    d1 = copy.deepcopy(reward_frequency)
    for i in d1:
        for j in d1[i]:
            d1[i][j] = int(round((reward_frequency[i][j] * 100) / (len(files) // len(reward_frequency)), 0))

    print('\nPercentage of kmeans action obfuscations with different possible max rewards')
    print(pandas.DataFrame.from_dict(d1, orient='index').sort_index())

    d2 = copy.deepcopy(reward_frequency)
    for i in d2:
        for j in range(len(cumulative)):
            if cumulative[j] == 0:
                d2[i][str(cumulative[j])] = len(files) // len(reward_frequency)
            else:
                d2[i][str(cumulative[j])] = d2[i][str(cumulative[j - 1])] - reward_frequency[i][str(cumulative[j - 1])]

    print('\nNumber of kmeans action obfuscations with possibility to get different rewards')
    print(pandas.DataFrame.from_dict(d2, orient='index').sort_index())

    for i in d2:
        for j in d2[i]:
            d2[i][j] = int(round((d2[i][j] * 100) / (len(files) // len(reward_frequency)), 0))

    print('\nPercentage of kmeans action obfuscations with possibility to get different rewards')
    print(pandas.DataFrame.from_dict(d2, orient='index').sort_index())


if __name__ == "__main__":
    main()
