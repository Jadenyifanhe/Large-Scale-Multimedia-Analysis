import os
import os.path as osp
import pandas as pd
import pickle
import numpy as np


def load_bow(bow_path):
    with open(bow_path, 'rb') as f:
        frame_id, bow = pickle.load(f)
    return frame_id, bow


def save_bow(bow_path, feature, frame_id=0):
    with open(bow_path, 'wb') as f:
        pickle.dump((frame_id, feature), f)
    return


def get_stats(trainval_path, bow_dir):
    df = pd.read_csv(trainval_path)
    trainval_video_ids = df['Id']
    bows = []
    bow_path_fmt = osp.join(bow_dir, '{}.pkl')
    for video_id in trainval_video_ids:
        bow_path = bow_path_fmt.format(video_id)
        _, bow = load_bow(bow_path)
        bows.append(bow)
    bows = np.stack(bows)
    mean = bows.mean(axis=0)
    std = bows.std(axis=0)
    return mean, std

def norm_bows(video_ids, src_dir=None, dst_dir=None, mean=None, std=None):
    src_fmt = osp.join(src_dir, '{}.pkl')
    dst_fmt = osp.join(dst_dir, '{}.pkl')
    for video_id in video_ids:
        frame_id, bow = load_bow(src_fmt.format(video_id))
        bow = (bow - mean) / std
        save_bow(dst_fmt.format(video_id), bow, frame_id=frame_id)
    return


def main():
    trainval_path = 'data/labels/train_val.csv'
    test_path = 'data/labels/test_for_students.csv'
    bow_dir = 'data/bow_sift_1024'
    save_dir = bow_dir + '_normed'
    os.makedirs(save_dir, exist_ok=True)

    mean, std = get_stats(trainval_path, bow_dir)

    df = pd.read_csv(trainval_path)
    trainval_video_ids = df['Id']
    norm_bows(trainval_video_ids, src_dir=bow_dir, dst_dir=save_dir, mean=mean, std=std)

    df = pd.read_csv(test_path)
    test_video_ids = df['Id']
    norm_bows(test_video_ids, src_dir=bow_dir, dst_dir=save_dir, mean=mean, std=std)
    return


if __name__ == '__main__':
    main()
