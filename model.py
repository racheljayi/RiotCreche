import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

opgg = pd.read_csv("data/opgg1.csv")

# drop unnecessary columns
opgg = opgg.drop(["player-href", "web-scraper-order", "web-scraper-start-url", "player", "r-top", "r-jungle", "r-mid",
                  "r-bot", "r-sup", "b-top", "b-jungle", "b-mid", "b-bot", "b-sup", "self-champ"], axis=1)

# clear out all rows that didn't scrape correctly
for index, row in opgg.iterrows():
    if not (type(opgg['r-top-href'][index]) == str or type(opgg['r-jungle-href'][index] == str)):
        opgg = opgg.drop([index])

# this feels like a really clumsy way to do it, but here i slot the player's champs into the correct column
# and create a new column that records their side, red or blue
opgg['side'] = "other"

for index, row in opgg.iterrows():
    for column in opgg.columns:
        if type(opgg[column][index]) != str:
            opgg[column][index] = opgg['self-champ-href'][index]
            opgg['side'][index] = column.split('-')
            opgg['side'][index] = column[0]

opgg = opgg.drop("self-champ-href", axis=1)

# sort victories and defeats into red wins (1) or blue wins (0)
sidesort = {
    "r": "b",
    "b": "r"
}

winsort = {
    "r": 1,
    "b": 0,
}

for index, row in opgg.iterrows():
    if opgg["self-win"][index] == "Defeat":
        opgg["self-win"][index] = sidesort[opgg["side"][index]]
    else:
        opgg["self-win"][index] = opgg["side"][index]
    opgg["self-win"][index] = winsort[opgg["self-win"][index]]

opgg = opgg.drop("side", axis=1)

# clean champ names
for column in opgg[['r-top-href', 'r-jungle-href', 'r-mid-href', 'r-bot-href', 'r-sup-href', 'b-top-href',
                    'b-jungle-href', 'b-mid-href', 'b-bot-href', 'b-sup-href']]:
    opgg[column] = opgg[column].str.split('/')
    opgg[column] = opgg[column].str[-1]
    opgg[column] = opgg[column].str.split('=')
    opgg[column] = opgg[column].str[-1]
    opgg[column] = opgg[column].str.capitalize()

# clean column names
opgg = opgg.rename({'self-win': 'win', 'r-top-href': 'r-top', 'r-jungle-href': 'r-jungle', 'r-mid-href': 'r-mid',
                    'r-bot-href': 'r-bot', 'r-sup-href': 'r-sup', 'b-top-href': 'b-top', 'b-jungle-href': 'b-jungle',
                    'b-mid-href': 'b-mid', 'b-bot-href': 'b-bot', 'b-sup-href': 'b-sup'}, axis=1)

ch = pd.read_csv('data/ch.csv')

# time to create new data
# the data i want to create will track
# 1. counters: # of strong-against
# 2. countered: # of weak-against
# 3. champ classes of each role for each side, #1-13

opgg['b-counters'] = 0
opgg['b-countered'] = 0
opgg['r-counters'] = 0
opgg['r-countered'] = 0

red_champs = opgg[['r-top', 'r-jungle', 'r-mid', 'r-bot', 'r-sup']]
blue_champs = opgg[['b-top', 'b-jungle', 'b-mid', 'b-bot', 'b-sup']]
all_champs = opgg[['r-top', 'r-jungle', 'r-mid', 'r-bot', 'r-sup', 'b-top', 'b-jungle', 'b-mid', 'b-bot', 'b-sup']]
weak_against = ch[['weak-a1', 'weak-a2', 'weak-a3', 'weak-a4', 'weak-a5']]
strong_against = ch[['strong-a1', 'strong-a2', 'strong-a3', 'strong-a4', 'strong-a5']]

# 1-2
for index, row in opgg.iterrows():
    for col1 in red_champs:
        name_ind = ch[ch['champ'] == opgg[col1][index]].index.values
        for col2 in blue_champs:
            for col3 in weak_against:
                bool_series = opgg[col2][index] == ch[col3][name_ind]
                for x in bool_series:
                    if x:
                        opgg['r-countered'][index] = opgg['r-countered'][index] + 1
            for col4 in strong_against:
                bool_series = opgg[col2][index] == ch[col4][name_ind]
                for x in bool_series:
                    if x:
                        opgg['r-counters'][index] = opgg['r-counters'][index] + 1
    for col1 in blue_champs:
        name_ind = ch[ch['champ'] == opgg[col1][index]].index.values
    for col2 in red_champs:
        for col3 in weak_against:
            bool_series = opgg[col2][index] == ch[col3][name_ind]
            for x in bool_series:
                if x:
                    opgg['b-countered'][index] = opgg['b-countered'][index] + 1
        for col4 in strong_against:
            bool_series = opgg[col2][index] == ch[col4][name_ind]
            for x in bool_series:
                if x:
                    opgg['b-counters'][index] = opgg['b-counters'][index] + 1

# 3
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

for index, row in opgg.iterrows():
    for col1 in all_champs:
        name_ind = ch[ch['champ'] == opgg[col1][index]].index.values
        archetype = ch['archetype'][name_ind].values
        opgg[col1][index] = roles_map[np.array_str(archetype)]

# modeling
X_train = opgg.drop('win', axis=1)
y_train = opgg['win']
y_train = y_train.astype('int')

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

from joblib import dump, load
dump(clf, 'model.joblib')

