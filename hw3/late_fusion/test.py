import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import pickle
import numpy as np
from utils.read_write import read_test_data, save_pred_csv

BASE_DIR = 'logs/late_fusion/'
os.makedirs(BASE_DIR, exist_ok=True)


def main():
    names, audio_feat, visual_feat = read_test_data()

    model_file = os.path.join(BASE_DIR, 'model_late.pkl')
    with open(model_file, 'rb') as f:
        model_late = pickle.load(f)

    model_audio = model_late['audio']
    model_visual = model_late['visual']
    model_fusion = model_late['fusion']

    prob_audio = model_audio.predict_proba(audio_feat)
    prob_visual = model_visual.predict_proba(visual_feat)
    prob_fusion = np.concatenate([prob_audio, prob_visual], axis=-1)

    pred_cats = model_fusion.predict(prob_fusion)

    data = np.stack([names, pred_cats], axis=-1)
    pred_file = os.path.join(BASE_DIR, 'pred_late.csv')
    save_pred_csv(data=data, fname=pred_file)
    return


if __name__ == '__main__':
    main()
