import Dataloader
import numpy as np
import tempfile
import streamlit as st
import torch
from Swin_BERT_Semantics import Swin_BERT_Semantics

@st.cache
def gen_caption(device,path_model,in_video):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(in_video.read())
    mydataloader = Dataloader.Video_Caption_Loader(config="swin_base_bert.py")

    model = Swin_BERT_Semantics(mlp_freeze=False, swin_freeze=True, in_size=1024, hidden_sizes=[2048, 1024],
                                out_size=768, drop_bert=0, max_length=20)

    checkpoint = torch.load(path_model, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    images = mydataloader.__getitem__(tfile.name)
    images = images.to(device)

    output = model(images)
    generate_text = output.cpu().numpy().tolist()
    generate_converted = model.tokenizer.batch_decode(generate_text, skip_special_tokens=True)
    out_caption = generate_converted[0]
    return out_caption


in_video = st.file_uploader('Video Caption Test', type=['mp4'])
out_caption = 'None'
rating1 = 0
user_caption = ''
if in_video is not None:
    st.video(in_video)
    out_caption = gen_caption('cpu','VASTA.ckpt',in_video)


if out_caption != 'None':
    st.write('The generated caption is:')
    st.info(out_caption)
    st.write('Now, if you would, rate the generated caption:')
    rating1 = st.slider('How accurate is the caption?', 0, 10, 0)
    user_caption = st.text_input('How would you caption the video')