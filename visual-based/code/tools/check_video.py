import os
import pandas as pd


def main():
    video_path = 'data/videos'
    train_val_file_path = 'data/labels/train_val.csv'
    test_file_path = 'data/labels/test_for_students.csv'

    df = pd.read_csv(train_val_file_path)
    train_val_video_ids = df['Id'].values.tolist()

    df = pd.read_csv(test_file_path)
    test_video_ids = df['Id'].values.tolist()

    total_video_ids = train_val_video_ids + test_video_ids
    total_video_ids.sort()
    total_video_ids = [v + '.mp4' for v in total_video_ids]

    video_names = os.listdir(video_path)
    video_names.sort()

    for i, video_id in enumerate(total_video_ids):
        if video_id not in video_names:
            print('{}: {}'.format(i, video_id))

    for i, video_id in enumerate(video_names):
        if video_id not in total_video_ids:
            print('{}: {}'.format(i, video_id))
    return


if __name__ == '__main__':
    main()
