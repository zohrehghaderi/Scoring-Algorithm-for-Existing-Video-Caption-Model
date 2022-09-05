import Dataloader
import tempfile
import streamlit as st
import torch
from csv import writer
from Swin_BERT_Semantics import Swin_BERT_Semantics

st.header('Video-Caption-Model')
if 'selectedVideo' not in st.session_state:
    st.session_state['selectedVideo'] = 0

if 'useClicked' not in st.session_state:
    st.session_state['useClicked'] = False
#else:
    #st.write(st.session_state['selectedVideo'])

@st.cache(show_spinner=False)
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


#wrapper um gen_caption um Text von Spinner selber bestimmen zu können
def get_caption(device,path_model,in_video):
    with st.spinner(text="generating caption"):
        return gen_caption(device,path_model,in_video)

out_caption = 'None'
video_name = ''
flag_num = 0

in_video = st.file_uploader('Upload your own Video here or choose one from the list on the left hand side:', type=['mp4'], help='The video should be no longer than 10 sec. Only mp4-files will be accepted.')

if in_video is not None:
    st.video(in_video)
    #out_caption = get_caption('cpu','VASTA.ckpt',in_video)
    out_caption = 'TEST'
    video_name = in_video.name

else:#falls ein video aus der liste ausgesucht wurde
    if st.session_state['selectedVideo'] != 0:
        if (st.button("use selected video") or st.session_state["useClicked"] == True):
            st.session_state['useClicked'] = True
            in_video = open('videos/'+ st.session_state['selectedVideo'][0], 'rb')
            #out_caption = get_caption('cpu','VASTA.ckpt',in_video)
            out_caption = 'TEST'
            video_name = in_video.name[7:]

            video_file = open('videos/'+ st.session_state['selectedVideo'][0], 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def transform_ratings(rating_match,rating_capture,rating_accuracy,rating_grammar):
    percentage_accuracy = 0

    #35% gewicht
    if rating_match == 'yes': percentage_accuracy += 35
    elif rating_match == 'to some degree': percentage_accuracy += 17.5

    #25%
    if rating_capture == 'yes': percentage_accuracy += 25

    #25%
    if rating_accuracy == 'very detailed': percentage_accuracy += 25
    elif rating_accuracy == 'detailed': percentage_accuracy += 18.75
    elif rating_accuracy == 'decent': percentage_accuracy += 12.5
    elif rating_accuracy == 'vague': percentage_accuracy += 6.25

    #15%
    if rating_grammar == 'no': percentage_accuracy += 15

    return percentage_accuracy


if out_caption != 'None':
    st.write('The generated caption is:')
    st.info(out_caption)

    st.header('Please rate the generated caption to help improve the model:')
    rating_match = st.radio("Does the description match the video?",('yes', 'to some degree', 'not at all'))

    rating_capture = st.radio("Is everything important captured by the caption?",('yes', 'no'))
    if rating_capture == 'no':
        user_missing = st.text_input('What is missing?')
    else:
        user_missing = 'empty'

    rating_accuracy = st.select_slider('How accurate is the caption?',
        options=['very vague','vague', 'decent','detailed', 'very detailed'],
        value=('decent'))

    rating_grammar = st.radio("Are there any grammatical errors in the caption?",('yes', 'no'), 1)

    user_caption = st.text_input('Please provide your own caption of the video:')

    clicked = st.button("Submit")
    if (clicked):
        st.session_state["useClicked"] = False
        percentage_accuracy = transform_ratings(rating_match,rating_capture,rating_accuracy,rating_grammar)
        append_list_as_row('ratings.csv', [video_name, out_caption, rating_match, rating_capture, user_missing, rating_accuracy, rating_grammar, user_caption, percentage_accuracy])
        clicked = False
        st.info('Thank you!')
