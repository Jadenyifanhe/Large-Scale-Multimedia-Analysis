#!/bin/python

import argparse
import os
import pickle
import time

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import sys

# Train MLP classifier with labels

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")
parser.add_argument("--folds", type=int, default=5)

np.random.seed(0)

if __name__ == '__main__':
  start = time.time()

  args = parser.parse_args()

  # 1. read all features in one array.
  fread = open(args.list_videos, "r")
  feat_list = []
  # labels are [0-9]
  label_list = []
  # load video names and events in dict
  df_videos_label = {}
  for line in open(args.list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category

  for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    # for videos with no audio, ignored in training
    if os.path.exists(feat_filepath):
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))
      label_list.append(int(df_videos_label[video_id]))

  n = len(label_list)
  inds = np.arange(n)
  np.random.shuffle(inds)

  best_acc = -1
  all_val_acc = []
  all_train_acc = []
  best_model = None

  label_list = np.array(label_list)
  print('label_list shape ', label_list.shape)
  feat_list = np.array(feat_list)
  print('feat_list shape ', feat_list.shape)

  for fold in range(args.folds):
    start_val = int(n * (float(fold)/args.folds))
    end_val = min(int(n * (float(fold+1)/args.folds)), n)

    train_fold_inds = np.concatenate((inds[:start_val], inds[end_val:]))
    val_fold_inds = inds[start_val:end_val]

    train_label_list = label_list[train_fold_inds]
    train_feat_list = feat_list[train_fold_inds]

    val_label_list = label_list[val_fold_inds]
    val_feat_list = feat_list[val_fold_inds]

    y = train_label_list
    X = train_feat_list

    clf = MLPClassifier(hidden_layer_sizes=(4096,),
                      activation="relu",
                      solver="adam",
                      alpha=1e-3,
                      max_iter=2000)
    clf.fit(X, y)

    y_val = np.array(val_label_list)
    X_val = np.array(val_feat_list)
    acc = accuracy_score(y_val, clf.predict(X_val))
    conf = confusion_matrix(y_val, clf.predict(X_val))
    print(conf)
    ConfusionMatrixDisplay.from_predictions(y_val, clf.predict(X_val))
    plt.savefig('confusion_matrix.png')

    all_train_acc.append(accuracy_score(y, clf.predict(X)))
    all_val_acc.append(acc)
    if acc > best_acc:
      best_acc = acc
      best_model = clf

  # save trained MLP in output_file
  pickle.dump(clf, open(args.output_file, 'wb'))
  end = time.time()
  print('MLP classifier trained successfully')
  print('Average training accuracy: ', np.array(all_train_acc).mean())
  print('Average validation accuracy: ', np.array(all_val_acc).mean())
  print('Best validation accuracy: ', best_acc)
  print("Time for computation: ", (end - start))