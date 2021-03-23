import os
from utils import max_rewards

PATH = 'out/MineRLObtainDiamondVectorObf-v0/100000'


def main():
    files = os.listdir(PATH)
    files.sort()

    for file in files:
        d = {}
        with open(os.path.join(PATH, file)) as txt:
            txt.readline()
            for line in txt.readlines():
                action = eval(line[line.index(',') + 1:-1])
                for act in action:
                    d[act] = 1

        print(f'{max_rewards(d)}\t{file}')


if __name__ == "__main__":
    main()
