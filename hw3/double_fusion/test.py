import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import pickle
import numpy as np
from utils.read_write import read_test_data, save_pred_csv

BASE_DIR = 'logs/double_fusion/'
os.makedirs(BASE_DIR, exist_ok=True)


def main():
    names, feat_audio, feat_visual = read_test_data()
    feat_audio_visual = np.concatenate([feat_audio, feat_visual], axis=-1)

    model_file = os.path.join(BASE_DIR, 'model_double.pkl')
    with open(model_file, 'rb') as f:
        model_double = pickle.load(f)

    model_audio = model_double['audio']
    model_visual = model_double['visual']
    model_audio_visual = model_double['audio_visual']
    model_fusion = model_double['fusion']

    prob_audio = model_audio.predict_proba(feat_audio)
    prob_visual = model_visual.predict_proba(feat_visual)
    prob_audio_visual = model_audio_visual.predict_proba(feat_audio_visual)
    prob_fusion = np.concatenate([prob_audio, prob_visual, prob_audio_visual], axis=-1)

    pred_cats = model_fusion.predict(prob_fusion)

    data = np.stack([names, pred_cats], axis=-1)
    pred_file = os.path.join(BASE_DIR, 'pred_double.csv')
    save_pred_csv(data=data, fname=pred_file)

    return


if __name__ == '__main__':
    main()
