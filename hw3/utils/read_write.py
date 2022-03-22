import os
import numpy as np
import pandas as pd
import pickle


def read_feat(feat_file, ext='csv', dim=1024):
    if os.path.isfile(feat_file):
        if ext == 'csv':
            feat = pd.read_csv(feat_file, header=None).values.squeeze()
        elif ext == 'pkl':
            with open(feat_file, 'rb') as f:
                feat = pickle.load(f)[1]
        else:
            raise ValueError('NOT supported: {}'.format(ext))
    else:
        print('ERROR: {}'.format(feat_file))
        feat = np.zeros(dim)
    return feat


def read_trainval_data():
    # visual_feat_dir = '/home/ubuntu/mountpoint/hw3/data/feats/visual/cnn_resnet101'
    visual_feat_dir = '/home/ubuntu/mountpoint/hw3/data/feats/visual/cnn_resnet152'
    visual_feat_ext = 'pkl'
    visual_feat_dim = 2048

    # audio_feat_dir = '/home/ubuntu/mountpoint/hw3/data/feats/audio/panns'
    audio_feat_dir = '/home/ubuntu/mountpoint/hw3/data/feats/audio/passt/scene'
    audio_feat_ext = 'csv'
    audio_feat_dim = 1295 # for passt
    # audio_feat_dim = 2048 # for panns

    train_val = pd.read_csv('/home/ubuntu/mountpoint/hw3/data/labels/train_val.csv').values

    name_list = []
    audio_feat_list = []
    visual_feat_list = []
    cat_list = []

    print('Reading trainval data...')
    for name, cat in train_val:
        audio_feat_file = os.path.join(audio_feat_dir, name + '.{}'.format(audio_feat_ext))
        audio_feat = read_feat(audio_feat_file, ext=audio_feat_ext, dim=audio_feat_dim)
        audio_feat_list.append(audio_feat)

        visual_feat_file = os.path.join(visual_feat_dir, name + '.{}'.format(visual_feat_ext))
        visual_feat = read_feat(visual_feat_file, ext=visual_feat_ext, dim=visual_feat_dim)
        visual_feat_list.append(visual_feat)

        name_list.append(name)
        cat_list.append(cat)

    print('Reading done!')
    print("Number of samples: {}".format(len(name_list)))

    names = np.array(name_list)
    cats = np.array(cat_list)
    audio_feat = np.array(audio_feat_list)
    visual_feat = np.array(visual_feat_list)

    return names, cats, audio_feat, visual_feat


def read_test_data():
    # visual_feat_dir = '/home/ubuntu/mountpoint/hw3/data/feats/visual/cnn_resnet101'
    visual_feat_dir = '/home/ubuntu/mountpoint/hw3/data/feats/visual/cnn_resnet152'
    visual_feat_ext = 'pkl'
    visual_feat_dim = 2048

    # audio_feat_dir = '/home/ubuntu/mountpoint/hw3/data/feats/audio/panns'
    audio_feat_dir = '/home/ubuntu/mountpoint/hw3/data/feats/audio/passt/scene'
    audio_feat_ext = 'csv'
    audio_feat_dim = 1295 # for passt
    # audio_feat_dim = 2048 # for panns

    test_set = pd.read_csv('/home/ubuntu/mountpoint/hw3/data/labels/test_for_students.csv').values[:, 0]

    name_list = []
    audio_feat_list = []
    visual_feat_list = []

    print('Reading test data...')
    for name in test_set:
        audio_feat_file = os.path.join(audio_feat_dir, name + '.{}'.format(audio_feat_ext))
        audio_feat = read_feat(audio_feat_file, ext=audio_feat_ext, dim=audio_feat_dim)
        audio_feat_list.append(audio_feat)

        visual_feat_file = os.path.join(visual_feat_dir, name + '.{}'.format(visual_feat_ext))
        visual_feat = read_feat(visual_feat_file, ext=visual_feat_ext, dim=visual_feat_dim)
        visual_feat_list.append(visual_feat)

        name_list.append(name)

    print('Reading done!')
    print("Number of samples: {}".format(len(name_list)))

    names = np.array(name_list)
    audio_feat = np.array(audio_feat_list)
    visual_feat = np.array(visual_feat_list)

    return names, audio_feat, visual_feat


def save_pred_csv(data=None, fname=None):
    print('Saving pred file...')
    pred_df = pd.DataFrame(data=data, columns=['Id', 'Category'])
    pred_df.to_csv(fname, index=False)
    print('Saving done!')
    return
