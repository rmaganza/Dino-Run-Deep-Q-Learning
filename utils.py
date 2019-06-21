import os
import pickle
from collections import deque

import pandas as pd

from model.model_params import STARTING_EPS, REPLAY_MEMORY
from paths import scores_file_path, actions_file_path, basepath

scores_df = pd.read_csv(scores_file_path) if os.path.isfile(scores_file_path) else pd.DataFrame(columns=['scores'])
actions_df = pd.read_csv(actions_file_path) if os.path.isfile(actions_file_path) else pd.DataFrame(columns=['actions'])


def save_pickle(obj, name):
    with open(basepath + '/objects/' + name + '.pkl', 'wb') as f:  # dump files into objects folder
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(name):
    with open(basepath + '/objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def init_cache():
    save_pickle(STARTING_EPS, "epsilon")
    t = 0
    save_pickle(t, "time")
    D = deque(maxlen=REPLAY_MEMORY)
    save_pickle(D, "D")


if __name__ == '__main__':
    init_cache()
