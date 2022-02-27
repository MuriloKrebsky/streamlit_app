import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#reading the data
stats_gerais = pd.read_excel('data/stats.xlsx', sheet_name='Stats Gerais')
roster = pd.read_excel('data/stats.xlsx', sheet_name='Roster')
totals = pd.read_excel('data/stats.xlsx', sheet_name='Totals')
awards = pd.read_excel('data/stats.xlsx', sheet_name='Awards')

#data cleaning
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

size=18
params = {'legend.fontsize': size,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size,
          'ytick.labelsize': size}
plt.rcParams.update(params)

#plot1: stats by position
fig, ax = plt.subplots(2, 4, figsize=(30,15))

ax[0][0].set_title('2pt distribution by position and level', size=15)
ax[0][1].set_title('3pt distribution by position and level')
ax[0][2].set_title('reb distribution by position and level')
ax[0][3].set_title('foul distribution by position and level')
ax[1][0].set_title('blk distribution by position and level')
ax[1][1].set_title('ast distribution by position and level')
ax[1][2].set_title('stl distribution by position and level')
ax[1][3].set_title('FT distribution by position and level')

sns.barplot(x='pos1', y='2pt', data=df, hue='nível', ci=False, ax=ax[0][0])
sns.barplot(x='pos1', y='3pt', data=df, hue='nível', ci=False, ax=ax[0][1])
sns.barplot(x='pos1', y='reb', data=df, hue='nível', ci=False, ax=ax[0][2])
sns.barplot(x='pos1', y='foul', data=df, hue='nível', ci=False, ax=ax[0][3])
sns.barplot(x='pos1', y='blk', data=df, hue='nível', ci=False, ax=ax[1][0])
sns.barplot(x='pos1', y='ast', data=df, hue='nível', ci=False, ax=ax[1][1])
sns.barplot(x='pos1', y='stl', data=df, hue='nível', ci=False, ax=ax[1][2])
sns.barplot(x='pos1', y='FT', data=df, hue='nível', ci=False, ax=ax[1][3])

fig.suptitle('Averages by position and level')
plt.savefig('images/stats_by_position.png')

#plot 2: 2pts and 3pts relation with efficiency 
fig, ax = plt.subplots(1, 2, figsize=(20,10))

ax[0].scatter(df['2pt'], df['eff'], c='b')
ax[0].set_xlabel('2PTS')
ax[0].set_ylabel('EFF')
ax[0].set_title('Relation between efficiency and 2PM')

ax[1].scatter(df['3pt'], df['eff'], c='b')
ax[1].set_xlabel('3PTS')
ax[1].set_ylabel('EFF')
ax[1].set_title('Relation between efficiency and 3PM')

plt.savefig('images/efficiency_x.png')

#plot 3: 2pts and 3pts distribution
fig, ax = plt.subplots(1, 2, figsize=(20,10))

sns.distplot(df['2pt'], color='skyblue', ax=ax[0])
ax[0].set_xlabel('2PTS')
ax[0].set_title('2PTS made')

sns.distplot(df['3pt'], color='skyblue', ax=ax[1])
ax[1].set_xlabel('3PTS')
ax[1].set_title('3PTS made')

plt.savefig('images/pts_dist.png')

#plot 4: pts distribution by position
fig, ax = plt.subplots(1, 2, figsize=(20,10), sharey=True)

ax[0].bar(df.groupby(by='pos1').mean().reset_index()['pos1'], df.groupby(by='pos1').mean()['2pt'], color='skyblue')
ax[0].set_title('Average 2pts made by position')

ax[1].bar(df.groupby(by='pos1').mean().reset_index()['pos1'], df.groupby(by='pos1').mean()['3pt'], color='skyblue')
ax[1].set_title('Average 3pts made by position')

plt.savefig('images/points_by_pos.png')

#plot 5: blocks by player
C = df[df['pos1'] == 'C']
F = df[df['pos1'] == 'F']
G = df[df['pos1'] == 'G']

C['blk_zscore'] = C['blk'].apply(lambda x: (x - np.mean(C['blk']))/np.std(C['blk']))
C['blk_zscore'] = C['blk_zscore'].apply(lambda x: abs(x))

F['blk_zscore'] = F['blk'].apply(lambda x: (x - np.mean(F['blk']))/np.std(F['blk']))
F['blk_zscore'] = F['blk_zscore'].apply(lambda x: abs(x))

G['blk_zscore'] = G['blk'].apply(lambda x: (x - np.mean(G['blk']))/np.std(G['blk']))
G['blk_zscore'] = G['blk_zscore'].apply(lambda x: abs(x))

color_dict = {'C': 'g',
              'F': 'y',
              'G': 'b'}

fig, ax = plt.subplots(1,2, figsize=(25,10))
ax[0].bar(df['Atleta'], df['blk'], color=df.replace({'pos1': color_dict})['pos1'])
ax[0].xaxis.set_tick_params(rotation=90)
ax[0].set_title('Block count for each players')
colors = {'C': 'g',
           'F': 'y',
           'G': 'b'}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
ax[0].legend(handles, labels)

ax[1].bar(C['Atleta'], C['blk_zscore'], color='g')
ax[1].xaxis.set_tick_params(rotation=90)

ax[1].bar(F['Atleta'], F['blk_zscore'], color='y')
ax[1].xaxis.set_tick_params(rotation=90)

ax[1].bar(G['Atleta'], G['blk_zscore'], color='b')
ax[1].xaxis.set_tick_params(rotation=90)

ax[1].set_title('blk_zscore by player and position')
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
ax[1].legend(handles, labels)

plt.savefig('images/blk_per_player.png')

#plot 6: ast by player
C = df[df['pos1'] == 'C']
F = df[df['pos1'] == 'F']
G = df[df['pos1'] == 'G']

C['ast_zscore'] = C['ast'].apply(lambda x: (x - np.mean(C['blk']))/np.std(C['blk']))
C['ast_zscore'] = C['ast_zscore'].apply(lambda x: abs(x))

F['ast_zscore'] = F['ast'].apply(lambda x: (x - np.mean(F['blk']))/np.std(F['blk']))
F['ast_zscore'] = F['ast_zscore'].apply(lambda x: abs(x))

G['ast_zscore'] = G['ast'].apply(lambda x: (x - np.mean(G['blk']))/np.std(G['blk']))
G['ast_zscore'] = G['ast_zscore'].apply(lambda x: abs(x))

color_dict = {'C': 'g',
              'F': 'y',
              'G': 'b'}

fig, ax = plt.subplots(1,2, figsize=(25,10))
ax[0].bar(df['Atleta'], df['ast'], color=df.replace({'pos1': color_dict})['pos1'])
ax[0].xaxis.set_tick_params(rotation=90)
ax[0].set_title('ast count for each players')
colors = {'C': 'g',
           'F': 'y',
           'G': 'b'}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
ax[0].legend(handles, labels)

ax[1].bar(C['Atleta'], C['ast_zscore'], color='g')
ax[1].xaxis.set_tick_params(rotation=90)

ax[1].bar(F['Atleta'], F['ast_zscore'], color='y')
ax[1].xaxis.set_tick_params(rotation=90)

ax[1].bar(G['Atleta'], G['ast_zscore'], color='b')
ax[1].xaxis.set_tick_params(rotation=90)

ax[1].set_title('ast_zscore by player and position')
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
ax[1].legend(handles, labels)

plt.savefig('images/ast_per_player.png')

#plot 7: reb by player
C = df[df['pos1'] == 'C']
F = df[df['pos1'] == 'F']
G = df[df['pos1'] == 'G']

C['reb_zscore'] = C['reb'].apply(lambda x: (x - np.mean(C['reb']))/np.std(C['reb']))
C['reb_zscore'] = C['reb_zscore'].apply(lambda x: abs(x))

F['reb_zscore'] = F['reb'].apply(lambda x: (x - np.mean(F['reb']))/np.std(F['reb']))
F['reb_zscore'] = F['reb_zscore'].apply(lambda x: abs(x))

G['reb_zscore'] = G['reb'].apply(lambda x: (x - np.mean(G['reb']))/np.std(G['reb']))
G['reb_zscore'] = G['reb_zscore'].apply(lambda x: abs(x))

color_dict = {'C': 'g',
              'F': 'y',
              'G': 'b'}

fig, ax = plt.subplots(1,2, figsize=(25,10))
ax[0].bar(df['Atleta'], df['reb'], color=df.replace({'pos1': color_dict})['pos1'])
ax[0].xaxis.set_tick_params(rotation=90)
ax[0].set_title('reb count for each players')
colors = {'C': 'g',
           'F': 'y',
           'G': 'b'}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
ax[0].legend(handles, labels)

ax[1].bar(C['Atleta'], C['reb_zscore'], color='g')
ax[1].xaxis.set_tick_params(rotation=90)

ax[1].bar(F['Atleta'], F['reb_zscore'], color='y')
ax[1].xaxis.set_tick_params(rotation=90)

ax[1].bar(G['Atleta'], G['reb_zscore'], color='b')
ax[1].xaxis.set_tick_params(rotation=90)

ax[1].set_title('reb_zscore by player and position')
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
ax[1].legend(handles, labels)

plt.savefig('images/reb_per_player.png')

#plot 8: efficiency by level
C_grouped = C.groupby(by='nível').mean()
F_grouped = F.groupby(by='nível').mean()
G_grouped = G.groupby(by='nível').mean()

fig, ax = plt.subplots(1, 3, figsize=(30, 10), sharey=True)
ax[0].set_title('C efficiency by level')
ax[1].set_title('F efficiency by level')
ax[2].set_title('G efficiency by level')

sns.lineplot(x='nível', y='eff', data=C_grouped, ci=None, markers=True, color="#3756ee", ax=ax[0])
sns.lineplot(x='nível', y='eff', data=F_grouped, ci=None, markers=True, color="#3756ee", ax=ax[1])
sns.lineplot(x='nível', y='eff', data=G_grouped, ci=None, markers=True, color="#3756ee", ax=ax[2])

plt.savefig('images/eff_lev.png')

#plot 9: ast/foul: Guards
G['ratio'] = G['stl'] / G['foul']

fig, ax = plt.subplots(1, 3, figsize=(25, 10))
ax[0].set_title('Avg stl by player and level (guards only)')
ax[0].xaxis.set_tick_params(rotation=90)
ax[1].set_title('Avg foul by player and level (guards only)')
ax[1].xaxis.set_tick_params(rotation=90)
ax[2].set_title('Avg stl/foul by player and level (guards only)')
ax[2].xaxis.set_tick_params(rotation=90)

sns.barplot(G['Atleta'], G['stl'], hue=G['nível'], ax=ax[0])
sns.barplot(G['Atleta'], G['foul'], hue=G['nível'], ax=ax[1])
sns.barplot(G['Atleta'], G['ratio'], hue=G['nível'], ax=ax[2])

plt.savefig('images/stl_foul.png')

#plot 10: is level distribution fair?
fig, ax = plt.subplots(1, 2, figsize=(30, 10))
ax[0].set_title('Efficiency by player and level')
ax[1].set_title('Definsive efficiency by player and level')
ax[0].xaxis.set_tick_params(rotation=90)
ax[1].xaxis.set_tick_params(rotation=90)

sns.barplot('Atleta', 'eff', data=df.sort_values(by='eff', ascending=False), hue='nível', ax=ax[0])

df['def_eff'] = ((df['reb'] + df['stl'] + df['blk']) / df['foul'])
sns.barplot('Atleta', 'def_eff', data=df.sort_values(by='def_eff', ascending=False), hue='nível', ax=ax[1])

plt.savefig('images/level_comparisson.png')

#plot 11: correct levels
temp = df.sort_values(by='eff', ascending=False).reset_index() 
temp['index'] = df.sort_values(by='eff', ascending=False).reset_index().index
i = temp[['index', 'Atleta', 'nível']]

with pd.option_context('mode.use_inf_as_null', True):
    temp = temp.sort_values('def_eff', ascending=False, na_position='last')
    temp['index'] = df.sort_values(by='def_eff', ascending=False).reset_index().index
    ii = temp[['index', 'Atleta']]

sorted_ = i.merge(ii, on='Atleta')
sorted_['overall'] = (sorted_['index_x'] + sorted_['index_y']) / 2
sorted_.sort_values('overall')

sns.barplot('Atleta', 'overall', hue='nível', data=sorted_)

#plot 12: players clustering
df.drop(columns=['pos2', 'def_eff'], inplace=True)
df = pd.concat([df, pd.get_dummies(df['pos1'])], axis=1)
pos = df['pos1']
team = df['equipe']
df.drop(columns=['pos1', 'equipe'], inplace=True)

def summarize_level(level):
  if level == 1:
    return 1
  elif level == 2 or level == 3:
    return 2
  else:
    return 3

df['nivel'] = df['nível'].apply(lambda x: summarize_level(x))
level = df['nivel']
df.drop(columns='nível', inplace=True)

x = df.values[:,1:]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
projection = PCA(n_components=2)
data2d = projection.fit_transform(x_scaled)

fig, ax = plt.subplots(1,3,figsize=(30, 10))

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[0].set_title('Clustering by level')

ax[1].set_xlabel('PC1')
ax[1].set_ylabel('PC2')
ax[1].set_title('Clustering by position')

ax[2].set_xlabel('PC1')
ax[2].set_ylabel('PC2')
ax[2].set_title('Clustering by team')

sns.scatterplot(x=data2d[:,0], y=data2d[:,1], hue=level, ax=ax[0])
sns.scatterplot(x=data2d[:,0], y=data2d[:,1], hue=pos, ax=ax[1])
sns.scatterplot(x=data2d[:,0], y=data2d[:,1], hue=team, ax=ax[2])

fig.suptitle('Players clustering', fontsize=18)
plt.savefig('images/clustering.png')