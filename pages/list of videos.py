import streamlit as st
import csv


ratings = open('ratings.csv')
csvreader = csv.reader(ratings)
rows = []
for row in csvreader:
        rows.append(row)

        

def count_ratings(video_name):
    count = 0
    for i in range(1, len(rows)):
        if rows[i][0] == video_name:
            count += 1
    return count



for i in range (1, 5):

    video_name = 'example'+ str(i) + '.mp4'
    countRating = count_ratings(video_name)
    
    video_file = open('videos/'+ video_name, 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)
    st.write('ratings for this video: '+ str(countRating))
    

    if st.button("select this video", key='select'+str(i)):
        st.session_state.selectedVideo = video_name
    if countRating > 0:
        st.button("show ratings of this video", key='show'+str(i))

st.write('Session State: ' + st.session_state.selectedVideo)


