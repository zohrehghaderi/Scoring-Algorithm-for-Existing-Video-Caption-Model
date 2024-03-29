import streamlit as st
import csv
import pandas as pd
import altair as alt


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


def generate_charts(video_name):
    count = 0
    count_match_yes = 0
    count_match_tosomedegree = 0
    count_match_notatall = 0
    count_captured_yes = 0
    count_captured_no = 0
    count_accuracy_veryvague = 0
    count_accuracy_vague = 0
    count_accuracy_decent = 0
    count_accuracy_detailed = 0
    count_accuracy_verydetailed = 0
    count_grammatical_errors_yes = 0
    count_grammatical_errors_no = 0
    user_missing = []
    user_captions = []
    avg_percentage = 0

    for i in range(1, len(rows)):
        if rows[i][0] == video_name:
            count += 1
            if rows[i][2] == 'yes':
                count_match_yes += 1
            elif rows[i][2] == 'to some degree':
                count_match_tosomedegree += 1
            elif rows[i][2] == 'not at all':
                count_match_notatall += 1

            if rows[i][3] == 'yes':
                count_captured_yes += 1
            elif rows[i][3] == 'no':
                count_captured_no += 1
                user_missing.append(rows[i][4])

            if rows[i][5] == 'very detailed':
                count_accuracy_verydetailed += 1
            elif rows[i][5] == 'detailed':
                count_accuracy_detailed += 1
            elif rows[i][5] == 'decent':
                count_accuracy_decent += 1
            elif rows[i][5] == 'vague':
                count_accuracy_vague += 1
            elif rows[i][5] == 'very vague':
                count_accuracy_veryvague += 1

            if rows[i][6] == 'yes':
                count_grammatical_errors_yes += 1
            elif rows[i][6] == 'no':
                count_grammatical_errors_no += 1
            user_captions.append(rows[i][7])
            avg_percentage += float(rows[i][8])

    avg_percentage = avg_percentage/count

    st.write('Does the description match the video?')
    chart_match = pd.DataFrame({
        'Count': [count_match_yes, count_match_tosomedegree, count_match_notatall],
        'Options': ['yes', 'to some degree', 'not at all']
    })
    bar_chart1 = alt.Chart(chart_match).mark_bar().encode(
        y='Count:Q',
        x='Options:O',
    )
    st.altair_chart(bar_chart1, use_container_width=True)


    st.write('Is everything important captured by the caption?')
    chart_captured = pd.DataFrame({
        'Count': [count_captured_yes, count_captured_no],
        'Options': ['yes', 'no']
    })
    bar_chart2 = alt.Chart(chart_captured).mark_bar().encode(
        y='Count:Q',
        x='Options:O',
    )
    st.altair_chart(bar_chart2, use_container_width=True)
    st.write('What is missing?')
    st.write(user_missing)

    st.write('How accurate is the caption?')
    chart_accuracy = pd.DataFrame({
        'Count': [count_accuracy_veryvague, count_accuracy_vague, count_accuracy_decent, count_accuracy_detailed,
                  count_accuracy_verydetailed],
        'Options': ['very vague', 'vague', 'decent', 'detailed', 'very detailed']
    })
    bar_chart3 = alt.Chart(chart_accuracy).mark_bar().encode(
        y='Count:Q',
        x='Options:O',
    )
    st.altair_chart(bar_chart3, use_container_width=True)

    st.write('Are there any grammatical errors in the caption?')
    chart_grammar = pd.DataFrame({
        'Count': [count_grammatical_errors_yes,count_grammatical_errors_no],
        'Options': ['yes', 'no']
    })
    bar_chart4 = alt.Chart(chart_grammar).mark_bar().encode(
        y='Count:Q',
        x='Options:O',
    )
    st.altair_chart(bar_chart4, use_container_width=True)

    st.write('Please provide your own caption of the video')
    st.write(user_captions)



    st.write('Average Percentage: ',avg_percentage)


    return


def get_ratings(video_name):
    caption = "no caption"
    rating_match = "no rating_match"
    rating_capture = "no rating_capture"
    user_missing = "empty"
    rating_accuracy = "no rating_accuracy"
    rating_grammer = "no rating_grammer"
    user_caption = "no user caption"
    percentage = -1
    countRatings = 0

    for i in range (1, len(rows)):
        if rows[i][0] == video_name:
            caption = rows[i][1]
            rating_match = rows[i][2]
            rating_capture = rows[i][3]
            user_missing = rows[i][4]
            rating_accuracy = rows[i][5]
            rating_grammer = rows[i][6]
            user_caption = rows[i][7]
            percentage = rows[i][8]
            countRatings = count_ratings(video_name)
            break
    return [video_name, caption, rating_match, rating_capture, user_missing, rating_accuracy, rating_grammer, user_caption, percentage, countRatings]



for i in range (1, 5):

    video_name = 'example'+ str(i) + '.mp4'
    countRating = count_ratings(video_name)

    video_file = open('videos/'+ video_name, 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)
    st.write('ratings for this video: '+ str(countRating))

    if st.button("select this video", key='select'+str(i)):
        st.session_state['selectedVideo'] = get_ratings(video_name) 

    if countRating > 0:
        if st.button("Generate Statistics for this video", key='stats'+str(i)):
            generate_charts(video_name)
    

#st.write('Session State: ' + str(st.session_state['selectedVideo']))