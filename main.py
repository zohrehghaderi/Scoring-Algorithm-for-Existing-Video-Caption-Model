import Dataloader
import numpy as np
import tempfile
import streamlit as st
import torch
from csv import writer
from Swin_BERT_Semantics import Swin_BERT_Semantics

st.header('Video-Caption-Model')
if 'selectedVideo' not in st.session_state:
    st.session_state['selectedVideo'] = 0
else:
    st.write(st.session_state['selectedVideo'])

@st.experimental_memo
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


if st.session_state['selectedVideo'] == 0:
    in_video = st.file_uploader('Upload your own Video here or choose one from the list on the left hand side:', type=['mp4'], help='The video should be no longer than 10 sec. Only mp4-files will be accepted.')
else:
    in_video = st.session_state.selectedVideo
out_caption = 'None'
video_name = ''
flag_num = 0

if in_video is not None:
    video_name = in_video.name
    rating1 = 0
    user_caption = ''
    st.video(in_video)
    out_caption = gen_caption('cpu','VASTA.ckpt',in_video)



def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)




if out_caption != 'None':
    st.write('The generated caption is:')
    st.info(out_caption)

    st.header('Please rate the generated caption to help improve the model:')

    rating_match = st.radio("Does the description match the video?",('yes', 'to some degree', 'not at all'))
    rating_capture = st.radio("Is everything important captured by the caption?",('yes', 'no'))
    if rating_capture == 'no':
        user_missing = st.text_input('What is missing?')
    else: user_missing = 'empty'

    rating_accuracy = st.select_slider('How accurate is the caption?',
        options=['very vague','vague', 'decent','detailed', 'very detailed'],
        value=('decent'))

    rating_grammer = st.radio("Are there any grammatical errors in the caption?",('yes', 'no'), 1)

    user_caption = st.text_input('Please provide your own caption of the video:')

    clicked = st.button("Submit")

    if (clicked):
        append_list_as_row('ratings.csv', [video_name, out_caption, rating_match, rating_capture, rating_accuracy, rating_grammer, user_caption])
        clicked = False
        st.info('Thank you!')
