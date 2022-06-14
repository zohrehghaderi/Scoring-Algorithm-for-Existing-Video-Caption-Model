# Scoring Algorithm for Existing Video Caption Model



Existing-Video-Caption-Model code is available: [a link](https://github.com/ECCV7129/ECCV2022_submission_7129)

#First Step: Data loader
1. you should have a dataloader class 
```
import numpy as np
from torch.utils.data import Dataset
from mmaction.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import mmcv
import cv2

class Video_Caption_Loader(Dataset):
   def __init__(self, config):
        """
        config: file path for network config
        """
        self.confige=config
        if isinstance(config, str):
            self.cfg = mmcv.Config.fromfile(config)
        self.transformer_video = self.cfg.transformer_video
        self.transformer_video = Compose(self.transformer_video)
       
       
   def __getitem__(self , video_path):
        """Returns one data pair (images)."""
        
        # uniformly selecting 32 frame
        cap = cv2.VideoCapture(video_path) #decalre cap object to read the video
        time_depth = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # compute total video frame
        frame_index = np.linspace(0, time_depth - 1, num=32, dtype=int) # select 32 frame


        # video prepartion with mmcv library
        data = dict(
            filename=video_file,
            label=-1,
            start_index=0,
            modality='RGB',
            frame_inds=frame_index,
            clip_len=32,
            num_clips=1
        )
        data = self.transformer_video(data)
        images = collate([data], samples_per_gpu=1)['imgs'] # image shape:[1, 3, 32, 224, 224] [batch size, channel, number of frames, height, width]
        
        images = images.squeeze(0) # image shape:[1, 1, 3, 32, 224, 224]

       

        return images

    def __len__(self):
        return self.dataset['video_name'].__len__()
```

#Second step: loading model
it is better you read how to load a model in pytorch [a link](https://forums.pytorchlightning.ai/t/how-to-load-and-use-model-checkpoint-ckpt/677)

1. you create model   
``` 
from Swin_BERT_Semantics import Swin_BERT_Semantics

model = Swin_BERT_Semantics(mlp_freeze=False, swin_freeze=True, in_size=1024, hidden_sizes=[2048, 1024], out_size=768, drop_swin=0, max_length=20,
                                            drop_mlp=0.1, drop_bert=0.3, bs=2,
                                            config_data=config,
                                            checkpoint_encoder=checkpoint_encoder)
```


