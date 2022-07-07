import tempfile
import numpy as np
import streamlit as st
from mmaction.datasets.pipelines import Compose
import cv2
from mmcv.parallel import collate, scatter
import mmcv



class Video_Caption_Loader():
    def __init__(self, config):
        """
        config: file path for network ../swin_base_bert.py
        """
        self.confige = config
        if isinstance(config, str):
            self.cfg = mmcv.Config.fromfile(config)
        self.transformer_video = self.cfg.transformer_video
        self.transformer_video = Compose(self.transformer_video)

    def __getitem__(self, video_path):
        """Returns one data pair (images)."""

        # uniformly selecting 32 frame
        cap = cv2.VideoCapture(video_path)  # decalre cap object to read the video
        time_depth = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # compute total video frame
        frame_index = np.linspace(0, time_depth - 1, num=32, dtype=int)  # select 32 frame

        # video prepartion with mmcv library
        data = dict(
            filename=video_path,
            label=-1,
            start_index=0,
            modality='RGB',
            frame_inds=frame_index,
            clip_len=32,
            num_clips=1
        )
        data = self.transformer_video(data)
        images = collate([data], samples_per_gpu=1)[
            'imgs']  # image shape:[1, 3, 32, 224, 224] [batch size, channel, number of frames, height, width]

        images = images.squeeze(0)  # image shape:[1, 1, 3, 32, 224, 224]

        return images

    def __len__(self):
        return self.dataset['video_name'].__len__()
