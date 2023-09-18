import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils.read_write import read_trainval_data
import time

BASE_DIR = 'logs/early_fusion/'
os.makedirs(BASE_DIR, exist_ok=True)


def main():
    max_iter = 1000

    names, cats, audio_feat, visual_feat = read_trainval_data()
    X = np.concatenate([audio_feat, visual_feat], axis=-1)
    y = cats

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

    model_early = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", max_iter=max_iter)

    model_early.fit(X_train, y_train)
    score = model_early.score(X_val, y_val)
    print('val score: {}'.format(score))

    confusion_file = os.path.join(BASE_DIR, 'confusion_matrix_val.jpg')
    plot_confusion_matrix(model_early, X_val, y_val)
    plt.savefig(confusion_file)

    # save trained MLP in output_file
    model_file = os.path.join(BASE_DIR, 'model_early.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(model_early, f)
    return


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("Time for computation: ", (end - start))