import os.path as osp
import numpy as np

from pyturbo import Stage, Task
from torchvision.io import read_video


class LoadVideo(Stage):

    """
    Input: video_id
    Output: [frames], each frame as [H x W x C]
    """

    def allocate_resource(self, resources, *, video_dir, file_suffix='mp4',
                          worker_per_cpu=1):
        self.video_dir = video_dir
        self.file_suffix = file_suffix
        return resources.split(len(resources.get('cpu'))) * worker_per_cpu

    def select_frames(self, frames: np.ndarray, frame_rate: float):
        """
        frames: [T x H x W x C]
        frame_rate: number of frames per second

        Return: a subset of frames, [t x H x W x C]
        """
        # TODO: select a subset of frames, 
        # potentially according to current frame rate
        # raise NotImplementedError

        # sample_rate = round(frame_rate / 2)
        # out = frames[0:-1:sample_rate]
        
        out = frames[0:-1:int(frame_rate)]

        return out

    def process(self, task):
        task.start(self)
        video_id = task.content
        video_path = osp.join(self.video_dir, f'{video_id}.{self.file_suffix}')
        frames, _, meta = read_video(video_path)
        frames = frames.numpy()
        frame_rate = meta['video_fps']
        selected_frames = self.select_frames(frames, frame_rate)
        for frame_id, frame in enumerate(selected_frames):
            sub_task = Task(meta={'frame_id': frame_id},
                            parent_task=task).start(self)
            sub_task.finish(frame)
            yield sub_task
        task.finish()
