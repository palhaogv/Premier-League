#Machining at the next games
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import funct

# Creating the frames by season
frames = []
for i in range(2005, 2020):
    globals()['df' + str(i)] = pd.read_csv(str(i) + '.csv', nrows=380)
    globals()['df' + str(i)].dropna()
    frames.append(globals()['df' + str(i)])

# Creating the stats for which frame (season)
for frame in frames:
    frame['FTHG_MEAN'] = frame.groupby('HomeTeam')['FTHG'].transform(lambda x: x.expanding().mean().shift()) # FTHG = Full Time Home Team Goals
    frame['FTAG_MEAN'] = frame.groupby('AwayTeam')['FTAG'].transform(lambda x: x.expanding().mean().shift()) # FTAG = Full Time Away Team Goals
    frame['HTHG_MEAN'] = frame.groupby('HomeTeam')['HTHG'].transform(lambda x: x.expanding().mean().shift()) # HTHG = Half Time Home Team Goals
    frame['HTAG_MEAN'] = frame.groupby('AwayTeam')['HTAG'].transform(lambda x: x.expanding().mean().shift()) # HTAG = Half Time Away Team Goals
    frame['HS_MEAN'] = frame.groupby('HomeTeam')['HS'].transform(lambda x: x.expanding().mean().shift()) # HS = Home Team Shots
    frame['AS_MEAN'] = frame.groupby('AwayTeam')['AS'].transform(lambda x: x.expanding().mean().shift()) # AS = Away Team Shots
    frame['HST_MEAN'] = frame.groupby('HomeTeam')['HST'].transform(lambda x: x.expanding().mean().shift())  # HST = Home Team Shots on Target
    frame['AST_MEAN'] = frame.groupby('AwayTeam')['AST'].transform(lambda x: x.expanding().mean().shift()) # AST = Away Team Shots on Target
    frame['HC_MEAN'] = frame.groupby('HomeTeam')['HC'].transform(lambda x: x.expanding().mean().shift()) # HC = Home Team Corners
    frame['AC_MEAN'] = frame.groupby('AwayTeam')['AC'].transform(lambda x: x.expanding().mean().shift()) # AC = Away Team Corners
    frame['HF_MEAN'] = frame.groupby('HomeTeam')['HF'].transform(lambda x: x.expanding().mean().shift()) # HF = Home Team Fouls Committed
    frame['AF_MEAN'] = frame.groupby('AwayTeam')['AF'].transform(lambda x: x.expanding().mean().shift()) # AF = Away Team Fouls Committed
    frame['HY_MEAN'] = frame.groupby('HomeTeam')['HY'].transform(lambda x: x.expanding().mean().shift()) # HY = Home Team Yellow Cards
    frame['AY_MEAN'] = frame.groupby('AwayTeam')['AY'].transform(lambda x: x.expanding().mean().shift()) # AY = Away Team Yellow Cards
    frame['HR_MEAN'] = frame.groupby('HomeTeam')['HR'].transform(lambda x: x.expanding().mean().shift()) # HR = Home Team Red Cards
    frame['AR_MEAN'] = frame.groupby('AwayTeam')['AR'].transform(lambda x: x.expanding().mean().shift()) # AR = Away Team Red Cards

    # Mean of odds
    frame['AvgHr'] = frame[['B365H', 'BWH', 'IWH', 'VCH', 'WHH']].mean(axis=1) # AvgHr = Market average home win odds
    frame['AvgDr'] = frame[['B365D', 'BWD', 'IWD', 'VCD', 'WHD']].mean(axis=1) # AvgDr = Market average draw win odds
    frame['AvgAr'] = frame[['B365A', 'BWA', 'IWA', 'VCA', 'WHA']].mean(axis=1) # AvgAr = Market average away win odds
    frame['PRIZE'] = np.select([frame['FTR'] == 'H', frame['FTR'] == 'A', frame['FTR'] == 'D'], [frame['AvgHr'] - 1, frame['AvgAr'] - 1, frame['AvgDr'] - 1], default=None)

    # Creating the WINS, LOSSES, DRAWS of witch team
    frame['FTR_A'] = np.select([frame['FTR'] == 'H', frame['FTR'] == 'A', frame['FTR'] == 'D'], [0, 2, 1], default=None)
    frame['FTR'] = frame['FTR'].map({'H': 2, 'A': 0, 'D': 1}) # FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)
    frame['HTR'] = frame['HTR'].map({'H': 2, 'A': 0, 'D': 1}) # HTR = Half Time Result (H=Home Win, D=Draw, A=Away Win)
    frame['H_POINTS'] = frame.groupby('HomeTeam')['FTR'].transform(lambda x: x.expanding().sum().shift())
    frame['A_POINTS'] = frame.groupby('AwayTeam')['FTR_A'].transform(lambda x: x.expanding().sum().shift())
    frame['H_DRAWS'] = frame.groupby('HomeTeam')['FTR'].transform(lambda x: (x == 1).expanding().sum().shift())
    frame['A_DRAWS'] = frame.groupby('AwayTeam')['FTR_A'].transform(lambda x: (x == 1).expanding().sum().shift())
    frame['Diff_POINTS'] = frame['H_POINTS'] - frame['A_POINTS']

    #Last 3 and 5 matches points
    frame['LAST_2_MP_H'] = frame.groupby('HomeTeam')['FTR'].transform(lambda x: x.rolling(window=2).sum().shift())
    frame['LAST_2_MP_A'] = frame.groupby('AwayTeam')['FTR_A'].transform(lambda x: x.rolling(window=2).sum().shift())
    frame['LAST_3_MP_H'] = frame.groupby('HomeTeam')['FTR'].transform(lambda x: x.rolling(window=3).sum().shift())
    frame['LAST_3_MP_A'] = frame.groupby('AwayTeam')['FTR_A'].transform(lambda x: x.rolling(window=3).sum().shift())
    frame['LAST_5_MP_H'] = frame.groupby('HomeTeam')['FTR'].transform(lambda x: x.rolling(window=5).sum().shift())
    frame['LAST_5_MP_A'] = frame.groupby('AwayTeam')['FTR_A'].transform(lambda x: x.rolling(window=5).sum().shift())



# Creating the dataframe with all data
premier_league_stats = pd.concat(frames)
columns_to_use = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTHG_MEAN', 'FTAG', 'FTR', 'FTR_A', 'Diff_POINTS', 'LAST_2_MP_H', 'LAST_2_MP_A',
                 'LAST_3_MP_H', 'LAST_3_MP_A', 'LAST_5_MP_H', 'LAST_5_MP_A', 'H_DRAWS',
                  'A_DRAWS', 'H_POINTS', 'A_POINTS', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF',
                  'AF', 'HY', 'AY', 'HR', 'AR', 'AvgHr', 'AvgDr', 'AvgAr', 'PRIZE', 'FTAG_MEAN', 'HTHG_MEAN',
                  'HTAG_MEAN', 'HS_MEAN', 'AS_MEAN', 'HST_MEAN', 'AST_MEAN', 'HC_MEAN', 'AC_MEAN', 'HF_MEAN', 'AF_MEAN',
                  'HY_MEAN', 'AY_MEAN', 'HR_MEAN', 'AR_MEAN']
premier_league_stats = premier_league_stats[columns_to_use]
premier_league_stats = premier_league_stats.dropna().reset_index(drop=True)

#Appling
features = ['LAST_2_MP_H', 'LAST_2_MP_A', 'LAST_5_MP_H', 'LAST_5_MP_A', 'LAST_3_MP_H', 'LAST_3_MP_A', 'Diff_POINTS',
            'H_DRAWS', 'A_DRAWS', 'HS_MEAN', 'AS_MEAN', 'HF_MEAN', 'AF_MEAN', 'FTHG_MEAN', 'FTAG_MEAN', 'HTHG_MEAN', 'HTAG_MEAN']
target = ['FTR']

train = premier_league_stats
X_train = train[features].copy()
y_train = train[target].copy()


next_games = frames[-1][-10:].copy()
previsoes = next_games[features]

X_train_transf, X_test_transf = funct.scaler(X_train, previsoes)

lr = LogisticRegression(penalty='l2', C=7.74263682, max_iter=4000).fit(X_train_transf, y_train)
next_games['Predicts'] = lr.predict(X_test_transf)
next_games = pd.merge(next_games, premier_league_stats, how='left')
next_games = next_games[['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'AvgHr', 'AvgDr', 'AvgAr', 'Predicts', 'PRIZE']]
next_games
print(next_games[['HomeTeam', 'AwayTeam', 'Predicts']])
