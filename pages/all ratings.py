import streamlit as st
import csv
import pandas as pd

ratings = open('ratings.csv')
csvreader = csv.reader(ratings)
rows = []
for row in csvreader:
        rows.append(row)
df = pd.DataFrame(rows, columns=['video name','generated caption','rating match','rating capture','rating accuracy','rating grammar','user caption','percentage'])
#iloc removes first row
st.table(df.iloc[1:])
