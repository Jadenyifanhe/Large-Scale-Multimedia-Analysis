import torch
from hear21passt.base import load_model, get_scene_embeddings, get_timestamp_embeddings
import os
import librosa
import numpy as np
import pandas as pd
import time


if __name__ == '__main__':
    start = time.time()
    model = load_model().cuda()

    for idx, filename in enumerate(os.listdir('wav')):
        audio_path = os.path.join('wav', filename)

        (audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
        audio = audio[None, :]  # (batch_size, segment_samples)
        audio = torch.tensor(audio)
        # time_embed, time_stamps = get_timestamp_embeddings(audio ,model)
        # print(time_embed.shape)
        scene_embed = get_scene_embeddings(audio, model)
        print(scene_embed.shape)

        # time_embed = time_embed.numpy()
        # time_embed = pd.DataFrame(time_embed)
        # time_embed.to_csv( 'passt/timestamp/' + filename + '.csv')
        scene_embed = scene_embed.cpu()
        scene_embed = scene_embed.numpy()
        scene_embed = pd.DataFrame(scene_embed).T
        scene_embed.to_csv( 'passt/scene' + filename.split('.')[0] + '.csv', index=False, header=False)

        end = time.time()
        print(str(idx + 1))
        print("Time for computation: ", (end - start))