import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
import librosa
import os
import time
import numpy as np

if __name__ == '__main__':
    start = time.time()

    for filename in os.listdir('mp3'):
        audio_path = os.path.join('mp3', filename)

        (audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
        audio = audio[None, :]  # (batch_size, segment_samples)

        at = AudioTagging(checkpoint_path=None, device='cpu')
        (clipwise_output, embedding) = at.inference(audio)
        np.savetxt('panns/' + filename.split('.')[0] + '.csv', embedding.T, delimiter=",")

    end = time.time()
    # print('------ Sound event detection ------')
    # sed = SoundEventDetection(checkpoint_path=None, device='cpu')
    # framewise_output = sed.inference(audio)
    print("Time for computation: ", (end - start))