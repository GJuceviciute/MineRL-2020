import numpy as np
import os

MOVING_ACTIONS = ['action$forward', 'action$left', 'action$back', 'action$right', 'action$jump', 'action$sneak',
                  'action$sprint', 'action$attack']
OTHER_ACTIONS = ['action$camera', 'action$place', 'action$equip', 'action$craft', 'action$nearbyCraft',
                 'action$nearbySmelt']
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'D:\\MineRL data 2020')
CUMULATIVE_REWARDS = [0, 1, 3, 7, 11, 19, 35, 67, 99, 131, 163, 291, 547, 1571, 10000]


def actions_from_file(data_set, trajectory):
    """
    Takes a trajectory path and returns all of it's actions as a list of tuples of strings.
    An example of an action: ('camera: [0. 0.]', 'nearbyCraft: iron_pickaxe').
    :param data_set: data set name (for example: 'MineRLObtainDiamond-v0')
    :param trajectory: trajectory path
    :return: list of actions
    """
    doc = os.path.join(MINERL_DATA_ROOT, data_set, trajectory, 'rendered.npz')
    f = np.load(doc)
    actions = []
    for i in range(len(f['reward'])):
        tick_acts = tuple()
        for act in MOVING_ACTIONS:
            if f[act][i] != 0:
                tick_acts += (act[7:],)
        for act in OTHER_ACTIONS:
            if act == 'action$camera' or f[act][i] != 'none':
                tick_acts += (f'{act[7:]}: {f[act][i]}',)
        actions.append(tick_acts)
    return actions


def deobfuscate_kmeans_actions(kmeans, action_mapping_dict):
    """
    For each centroid of given KMeans, finds the nearest obfuscated action in the mapping as measured by MSE.
    :param kmeans: KMeans object
    :param action_mapping_dict: mapping from non-obfuscated to obfuscated actions
    :return: lists of deobfuscated actions and corresponding MSEs
    """
    centers = kmeans.cluster_centers_
    best_mses = []
    best_actions = []
    for centroid in centers:
        best_mse = 13
        best_action = None
        for act in action_mapping_dict:
            a = action_mapping_dict[act]
            mse = ((centroid - a) ** 2).mean()
            if mse < best_mse:
                best_mse = mse
                best_action = act

        if 'camera' not in best_action[0]:
            best_action = list(best_action)
            for i in range(1, len(best_action)):
                if 'camera' in best_action[i]:
                    best_action = tuple([best_action[i]] + best_action[:i] + best_action[i + 1:])

        best_mses.append(np.sqrt(best_mse))
        best_actions.append(best_action)
    return best_mses, best_actions


def max_rewards(actions):
    """
    Takes a set of actions and returns the maximum possible reward.
    :param actions: a set of mini actions such as 'attack', 'craft: planks'
    :return: maximum possible rewards with given mini actions
    """
    max_reward = 0
    if 'attack' in actions:
        max_reward = 1
    else:
        return max_reward
    if 'craft: planks' in actions:
        max_reward = 3
    else:
        return max_reward
    if 'craft: stick' in actions and 'craft: crafting_table' in actions:
        max_reward = 11
    elif 'craft: stick' in actions or 'craft: crafting_table' in actions:
        max_reward = 7
        return max_reward
    else:
        return max_reward
    if 'place: crafting_table' in actions and 'nearbyCraft: wooden_pickaxe' in actions:
        max_reward = 19
        if 'equip: wooden_pickaxe' in actions:
            max_reward = 35
        else:
            return max_reward
    else:
        return max_reward
    if 'nearbyCraft: stone_pickaxe' in actions:
        max_reward = 67
    else:
        return max_reward
    if 'nearbyCraft: furnace' in actions and 'equip: stone_pickaxe' in actions:
        max_reward = 163
    elif 'nearbyCraft: furnace' in actions and 'equip: stone_pickaxe' not in actions:
        max_reward = 99
        return max_reward
    elif 'equip: stone_pickaxe' in actions and 'nearbyCraft: furnace' not in actions:
        max_reward = 131
        return max_reward
    else:
        return max_reward
    if 'place: furnace' in actions and 'nearbySmelt: iron_ingot' in actions:
        max_reward = 291
    else:
        return max_reward
    if 'nearbyCraft: iron_pickaxe' in actions:
        max_reward = 547
    else:
        return max_reward
    if 'equip: iron_pickaxe' in actions:
        max_reward = 1571
    else:
        return max_reward
    return max_reward


def camera_stats(path, file):
    """
    Gets all vertical camera action angles (positive angle means down) from a given KMeans experiment.
    :param path: path to the experiments
    :param file: filename of the experiment
    :return: a list of vertical camera action angles
    """
    updown = []
    with open(os.path.join(path, file)) as txt:
        txt.readline()
        for line in txt.readlines():
            action = eval(line[line.index(',') + 1:-1])
            camera_action = [float(i) for i in action[0][action[0].index('[') + 1:-1].split()]
            updown.append(camera_action[0])
    return updown