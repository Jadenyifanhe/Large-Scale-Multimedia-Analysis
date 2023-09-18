import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from utils.read_write import read_trainval_data
import time

BASE_DIR = 'logs/late_fusion/'
os.makedirs(BASE_DIR, exist_ok=True)


def main():
    max_iter = 1000
    names, cats, feat_audio, feat_visual = read_trainval_data()

    model_audio = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", max_iter=max_iter)
    model_audio.fit(feat_audio, cats)
    prob_audio = model_audio.predict_proba(feat_audio)

    model_visual = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", max_iter=max_iter)
    model_visual.fit(feat_visual, cats)
    prob_visual = model_visual.predict_proba(feat_visual)

    prob_fusion = np.concatenate([prob_audio, prob_visual], axis=-1)

    X_train, X_val, y_train, y_val = train_test_split(prob_fusion, cats, test_size=0.2, random_state=45)
    model_fusion = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", max_iter=max_iter)
    model_fusion.fit(X_train, y_train)

    score = model_fusion.score(X_val, y_val)
    print('val score: {}'.format(score))

    confusion_file = os.path.join(BASE_DIR, 'confusion_matrix_val.jpg')
    plot_confusion_matrix(model_fusion, X_val, y_val)
    plt.savefig(confusion_file)

    model_file = os.path.join(BASE_DIR, 'model_late.pkl')
    model_late = {
        'audio': model_audio,
        'visual': model_visual,
        'fusion': model_fusion,
    }
    with open(model_file, 'wb') as f:
        pickle.dump(model_late, f)
    return


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("Time for computation: ", (end - start))