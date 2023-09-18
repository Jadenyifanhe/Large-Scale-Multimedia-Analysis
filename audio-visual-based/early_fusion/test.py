import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np
import pandas as pd
import pickle
from utils.read_write import read_test_data

BASE_DIR = 'logs/early_fusion'
os.makedirs(BASE_DIR, exist_ok=True)


def main():
    names, audio_feat, visual_feat = read_test_data()
    X = np.concatenate([audio_feat, visual_feat], axis=-1)

    model_file = os.path.join(BASE_DIR, 'model_early.pkl')
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    # (num_samples) with integer
    pred_cats = model.predict(X)

    data = np.stack([names, pred_cats], axis=-1)
    pred_df = pd.DataFrame(data=data, columns=['Id', 'Category'])
    pred_file = os.path.join(BASE_DIR, 'pred_early.csv')
    pred_df.to_csv(pred_file, index=False)
    return


if __name__ == '__main__':
    main()
