import Dataloader
import numpy as np
import tempfile
import streamlit as st
import torch
from Swin_BERT_Semantics import Swin_BERT_Semantics


in_video = st.file_uploader('Dataloader Test', type=['mp4'])
if in_video is not None:
    st.video(in_video)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(in_video.read())
    mydataloader = Dataloader.Video_Caption_Loader(config="swin_base_bert.py")
    device = 'cpu'
    model = Swin_BERT_Semantics(mlp_freeze=False, swin_freeze=True, in_size=1024, hidden_sizes=[2048, 1024],
                                out_size=768, drop_bert=0, max_length=20)
    path_model = 'VASTA.ckpt'
    checkpoint = torch.load(path_model, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    images = mydataloader.__getitem__(tfile.name)
    images = images.to(device)

    output = model(images)

    generate_text = output.cpu().numpy().tolist()

    generate_converted = model.tokenizer.batch_decode(generate_text, skip_special_tokens=True)

    st.write(generate_converted)
