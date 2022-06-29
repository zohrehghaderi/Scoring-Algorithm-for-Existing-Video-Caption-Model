import Dataloader
import numpy as np
import tempfile
import streamlit as st
import cv2

in_video = st.file_uploader('Dataloader Test', type=['mp4'])
if in_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(in_video.read())
    mydataloader = Dataloader.Video_Caption_Loader(config="swin_base_bert.py")
    images = mydataloader.__getitem__(tfile.name)
    np.squeeze(images, )

    st.video(in_video)
    st.write(type(images))
    st.write(images.shape)


# uncomment to see pictures in cv2 window
    for batch in images:
        for frames in batch:
            for frame in frames:
                im = frame.numpy()
                st.image(im, clamp=True)
                # cv2.imshow('Pic', im)
                #
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
