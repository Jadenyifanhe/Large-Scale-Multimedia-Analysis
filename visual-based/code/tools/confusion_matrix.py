import os
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

SAVE_DIR = 'figs'
os.makedirs(SAVE_DIR, exist_ok=True)


def read_file(fpath):
    df = pd.read_csv(fpath)
    y_label = df['Category'].values
    y_pred = df['Pred'].values
    return y_label, y_pred


def main():
    # sift
    # train_file = 'data/mlp/sift/version_3/train_prediction.csv'
    # val_file = 'data/mlp/sift/version_3/val_prediction.csv'

    cnn
    train_file = 'data/mlp/cnn/version_7/train_prediction.csv'
    val_file = 'data/mlp/cnn/version_7/val_prediction.csv'     

    y_label, y_pred = read_file(train_file)
    ConfusionMatrixDisplay.from_predictions(y_label, y_pred)
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_train.jpg'))


    y_label, y_pred = read_file(val_file)
    ConfusionMatrixDisplay.from_predictions(y_label, y_pred)
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_val.jpg'))
    return


if __name__ == '__main__':
    main()
