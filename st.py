import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import streamlit as st
from PIL import Image

#reading the data
stats_gerais = pd.read_excel('data/stats.xlsx', sheet_name='Stats Gerais')
roster = pd.read_excel('data/stats.xlsx', sheet_name='Roster')
totals = pd.read_excel('data/stats.xlsx', sheet_name='Totals')
awards = pd.read_excel('data/stats.xlsx', sheet_name='Awards')

#data processing
stats_gerais['Atleta'] = stats_gerais['Atleta'].apply(lambda x: x.lower())
stats_gerais.rename(columns={'ll': 'FT',
                             'rd': 'DR',
                             'ro': 'OR',
                             'pt': 'PTS'}, inplace=True)
stats_gerais.replace(np.NaN, 0, inplace=True)

roster.rename(columns={'atleta': 'Atleta'}, inplace=True)
roster['Atleta'] = roster['Atleta'].apply(lambda x: x.lower())

totals.rename(columns={'ll': 'FT',
                       'rd': 'DR',
                       'ro': 'OR',
                       'pt': 'PTS',
                       'Equipe': 'equipe'}, inplace=True)
totals['Atleta'] = totals['Atleta'].apply(lambda x: x.lower())

awards.rename(columns={'ll': 'FT',
                       'rd': 'DR',
                       'ro': 'OR',
                       'pt': 'PTS',
                       'player': 'Atleta'}, inplace=True)
awards['Atleta'] = awards['Atleta'].apply(lambda x: x.lower())
rodadas = [1, 1, 1, 1, 1,
           2, 2, 2, 2, 2,
           3, 3, 3, 3, 3,
           4, 4, 4, 4, 4,
           5, 5, 5, 5, 5]
awards['rod'] = rodadas

df = stats_gerais.copy()
df = df.merge(roster, on='Atleta')

C = df[df['pos1'] == 'C']
F = df[df['pos1'] == 'F']
G = df[df['pos1'] == 'G']

#general settings
size=18
params = {'legend.fontsize': size,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size,
          'ytick.labelsize': size}
plt.rcParams.update(params)

st.set_page_config(layout="wide")

image = Image.open('images/logo.png')
col1, col, col2 = st.columns([3,20,1])
with col:
    st.image(image, use_column_width='auto')

st.title('Local team indoor basketball championship analysis')
st.subheader(""" The main goal of this streamlit project is to extract meaningful information from a dataset from a basketball championship""")

url = "https://www.instagram.com/aacbluewings/"

st.write("The data used here was produced by an indoor basketball championship, promoted by "
         " [Associação Atlética Cosmópolis](url). \n"
         "Below we can see the dataset (other tables were used, but the main information"
         " comes from this dataset). The columns names are pretty self explanatory")
st.dataframe(df)

st.write("Firstly, below you can see the stats distribution by player and position"
         " so you can have a better feeling of the data")
col1, col2 = st.columns(2)
with col1:
    option1 = st.selectbox(
        'Select position',
        ['C', 'F', 'G']) 
    if option1 == 'G':
        option1 = G
    elif option1 == 'F':
        option1 = F
    else:
        option1 = C

with col2:
    option2 = st.selectbox(
        'Select statistic',
        df.select_dtypes([int, float]).columns[:-1])

fig, ax = plt.subplots(figsize=(20,10))
sns.barplot(option1['Atleta'], option1[option2], hue=option1['nível'])
# plt.xticks(rotation=45)
fig.savefig('button_fig.png')

col1, col, col2 = st.columns([3,20,1])
with col:
    st.image('button_fig.png', use_column_width='auto')

st.write('#')
st.write("Now we can dive into more complex analysis. Let's take a look at the distribution of stats per position "
    "all at once:")
image = Image.open('images/stats_by_position.png')
st.image(image, caption='Basic stats per position', use_column_width='auto')

st.write("""
*Before any statement is worthy to mention that due to the small number of samples we have to keep in mind that the analysis may be biased by one or other player*

Main points to note:


*   As expected, centers have the highest amount of rebounds, followed by fowards and guards, due to the height
*   Also as expected, guards have the highest amount of FTM, but we cannot state
    for sure that there is a correlation between guards x FTM,
    since we don't have data on FTA.
    (it could only be that every position goes with the same frequency to the free throw line but guards have a higher FTM %, which is also pretty reasonable)
*   Foul distribution has a low variance regardless to positions/levels.
    However, we'll further look into the relation blk x fouls and stl x fouls 
*   Contrary to what was expected, guards do not have the highest amount of steals
*   Taking into account outliers, we can say that the position does not affect on the amount of 2 and 3 pts made.
    Again, here we could explore better if FGA was available.
""")

st.write("Let's now take a look at the relation points x efficiency")
image = Image.open('images/efficiency_x.png')
st.image(image, caption='points x efficiency', use_column_width='auto')
st.write("""We do have a correlation between 2pts and efficiency, however, we
         can't state that for 3pts. This is probably because due to the  low amount of
         threes made, the impact on efficiency was also low
         (since FGA was not take into account). You can see below
         the distribution for both 2pts and 3pts""")

image = Image.open('images/pts_dist.png')
st.image(image, caption='points distribution', use_column_width='auto')
st.write("Very low number of 3PTM. What we can say about the 3PT% ?")
image = Image.open('images/points_by_pos.png')
st.image(image, caption='points by position', use_column_width='auto')
st.write("""
    This may be an evidence that the 3FG% was low. Generally guards and fowards have higher 3FG% than centers,
    but here, every position got roughly the same amount of 3PM.
    (In the current era of 3's that we live, is very odd that guards have shooted few 3 pointers.)
    Since the amount of 3 mades is almost the same for every position, what explain this is a low 3FG%

    Note that guards have higher 2PM than centers, which is curious.
""")

st.write("Now let's take a look at specific stats:")
st.subheader('Main stats')
st.write("- blocks")
image = Image.open('images/blk_per_player.png')
st.image(image, caption='blocks per player', use_column_width='auto')
st.write("""
    Bruno excelled the other playes by a huge gap, not only is his position, but in general terms as well.
    The same thing goes to Emmanuel, and a special attention to Daniel and Louzeiro whose even being guards
    have a higher zscore than any other player except by Emmanuel and Bruno.

    Obs: zscore is a measure that tells you "how far from the average" you are.
    In this case, the zscore was made taking into account each of the positions.
""")

st.write('- ast')
image = Image.open('images/ast_per_player.png')
st.image(image, caption='ast per player', use_column_width='auto')
st.write("""
    In this case what we expected happened. Guards indeed have the highest
    ast (namely, guards have twice as much as the other positions regarding
    to mean ast per position). The zscore for guards isn't that much than 
    other positions because on average guards have a higher ast per player.
""")

st.write('- reb')
image = Image.open('images/reb_per_player.png')
st.image(image, caption='reb_per_player', use_column_width='auto')
st.write("""
    Again, as expected centers have the highest amount of rebounds.
    But note that the highest difference of zscore was achivied by Eric with
    a gap of almost 1 point.
""")

st.write('- stl')

image = Image.open('images/stl_foul.png')
st.image(image, caption='stl: guards', use_column_width='auto')
st.write("""
Here we have something interesting: although Ruan and Louzeiro (leve 5 players)
are second and third in avg steals they do not have a godd stl/foul ratio.
We can say somehow that the "best" defensive players were Ranzani and Eric
(Eric was also an outlier in rebounds, which reinforces our statement.)
""")

st.subheader('Were level assumptions made correctly?')
st.write("The 'level' feature was given to balance things out. "
         "This feature was given based on the knowledge of the coach, "
         "the question is: did he make the right choices?")

image = Image.open('images/level_comparisson.png')
st.image(image, caption='level validation', use_column_width='auto')

st.write("If we only look at the stantard efficiency (not so stantard) "
         "it does look like the levels were given correctly except by a "
         "few exceptions. However, since the FGA was not take into account "
         "in this formula the measure might be biased. Therefore, a good "  
         "strategy is to look at the so called defensive efficiency, which is "
         "given by: (stl+reb+blk)/fouls. In defensive terms, these 4 attributes "
         "are pretty explanatory by themselves.")
st.write("#")
st.write("Looking at the defensive efficiency we see that Bruno and Lucao "
         "stood at the top, but surprinsingly, we had a lot of levels 3 and 4 " 
         "figuring out at the top positions")

st.subheader("Players clustering")
st.write("Last but not least, let's explore the relationship between players. "
         "What make them alike? Position? Level?")

image = Image.open('images/clustering.png')
st.image(image, caption='players clustering', use_column_width='auto')

st.write("We can see that the most 'effective' clustering was made by level, "
         "which is somewhat curious but also logical. We can also see that "
         "clustering by position had some clusters as well, having centers "
         "within a pretty separated category")
