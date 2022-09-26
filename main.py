import streamlit as st

import numpy as np
import pandas as pd

from joblib import dump, load

import warnings

warnings.filterwarnings('ignore')

st.header("Welcome to RiotCreche")
opgg = pd.read_csv("data/opgg1.csv")
ch = pd.read_csv("data/ch.csv")

mod = load("model.joblib")

st.subheader("Please select your champs")
left_column, right_column = st.columns(2)
with left_column:
    input_btop = st.selectbox(
        'Blue Top Lane:',
        np.unique(ch['champ']))
    input_bjungle = st.selectbox(
        'Blue Jungle Lane:',
        np.unique(ch['champ']))
    input_bmid = st.selectbox(
        'Blue Mid Lane:',
        np.unique(ch['champ']))
    input_bbot = st.selectbox(
        'Blue Bot Lane:',
        np.unique(ch['champ']))
    input_bsup = st.selectbox(
        'Blue Support:',
        np.unique(ch['champ']))

with right_column:
    input_rtop = st.selectbox(
        'Red Top Lane:',
        np.unique(ch['champ']))
    input_rjungle = st.selectbox(
        'Red Jungle Lane:',
        np.unique(ch['champ']))
    input_rmid = st.selectbox(
        'Red Mid Lane:',
        np.unique(ch['champ']))
    input_rbot = st.selectbox(
        'Red Bot Lane:',
        np.unique(ch['champ']))
    input_rsup = st.selectbox(
        'Red Support:',
        np.unique(ch['champ']))

dt = {'r-top': [input_rtop], 'r-jungle': [input_rjungle], 'r-mid': [input_rmid], 'r-bot': [input_rbot],
         'r-sup': [input_rsup], 'b-top': [input_btop], 'b-jungle': [input_bjungle], 'b-mid': [input_bmid],
         'b-bot': [input_bbot], 'b-sup': [input_bsup], 'b-counters': [0], 'b-countered': [0], 'r-counters': [0],
         'r-countered': [0]}

input = pd.DataFrame(data=dt)

red_champs = input[['r-top', 'r-jungle', 'r-mid', 'r-bot', 'r-sup']]
blue_champs = input[['b-top', 'b-jungle', 'b-mid', 'b-bot', 'b-sup']]
all_champs = input[['r-top', 'r-jungle', 'r-mid', 'r-bot', 'r-sup', 'b-top', 'b-jungle', 'b-mid', 'b-bot', 'b-sup']]
weak = ch[['weak-a1', 'weak-a2', 'weak-a3', 'weak-a4', 'weak-a5']]
strong = ch[['strong-a1', 'strong-a2', 'strong-a3', 'strong-a4', 'strong-a5']]

roles_map = {
    "[]": 0,
    "['Artillery']": 1,
    "['Assassin']": 2,
    "['Battlemage']": 3,
    "['Burst']": 4,
    "['Catcher']": 5,
    "['Diver']": 6,
    "['Enchanter']": 7,
    "['Juggernaut']": 8,
    "['Marksman']": 9,
    "['Skirmisher']": 10,
    "['Specialist']": 11,
    "['Vanguard']": 12,
    "['Warden']": 13
}

win_map = {
    0: "Blue-side victory",
    1: "Red-side victory"
}

for index, row in input.iterrows():
    for col in red_champs:
        name_ind = ch[ch['champ'] == input[col][index]].index.values
        for col2 in blue_champs:
            for col3 in weak:
                bool_series = input[col2][index] == ch[col3][name_ind]
                for x in bool_series:
                    if x:
                        input['r-countered'][index] = input['r-countered'][index] + 1
            for col4 in strong:
                bool_series = input[col2][index] == ch[col4][name_ind]
                for x in bool_series:
                    if x:
                        input['r-counters'][index] = input['r-counters'][index] + 1
    for col1 in blue_champs:
        name_ind = ch[ch['champ'] == input[col1][index]].index.values
        for col2 in red_champs:
            for col3 in weak:
                bool_series = input[col2][index] == ch[col3][name_ind]
                for x in bool_series:
                    if x:
                        input['b-countered'][index] = input['b-countered'][index] + 1
            for col4 in strong:
                bool_series = input[col2][index] == ch[col4][name_ind]
                for x in bool_series:
                    if x:
                        input['b-counters'][index] = input['b-counters'][index] + 1
    for col1 in all_champs:
        name_ind = ch[ch['champ'] == input[col1][index]].index.values
        archetype = ch['archetype'][name_ind].values
        input[col1][index] = roles_map[np.array_str(archetype)]

if st.button("Make Prediction"):
    prediction = mod.predict(input)
    st.write(f"Prediction is a {win_map[prediction[0]]}")

    st.write("GG!")
