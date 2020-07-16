import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor


def frames_season():
    # Creating the frames by season
    frames = []
    new_frames = []
    for i in range(2005, 2019):
        globals()['df' + str(i)] = pd.read_csv(r"frames\\" + str(i) + '.csv', nrows=380)
        globals()['df' + str(i)].dropna()
        frames.append(globals()['df' + str(i)])

    # Creating the stats for which frame (season)
    pd.options.mode.chained_assignment = None
    for frame in frames:
        frame['ID'] = frame.index + 1
        frame['ID'] = frame['ID'].apply(lambda x: '{0:0>5}'.format(x))
        frame['FTGT'] = frame['FTHG'] + frame['FTAG']
        columns_name = ['ID', 'Date', 'Team', 'FTG', 'FTR', 'FTGT', 'HTG', 'HTR', 'S', 'ST', 'C', 'F', 'Y', 'R',
                        'B365H', 'BWH', 'IWH', 'VCH', 'WHH',
                        'B365D', 'BWD', 'IWD', 'VCD', 'WHD',
                        'B365A', 'BWA', 'IWA', 'VCA', 'WHA',
                        'BbAv>2.5', 'BbAv<2.5']

        # Home games
        frame_h = frame[['ID', 'Date', 'HomeTeam', 'FTHG', 'FTR', 'FTGT', 'HTHG', 'HTR', 'HS', 'HST', 'HC', 'HF', 'HY', 'HR',
                         'B365H', 'BWH', 'IWH', 'VCH', 'WHH',
                         'B365D', 'BWD', 'IWD', 'VCD', 'WHD',
                         'B365A', 'BWA', 'IWA', 'VCA', 'WHA',
                         'BbAv>2.5', 'BbAv<2.5']].copy()
        frame_h.columns = columns_name
        frame_h['Location'] = 1

        # Away games
        frame_A = frame[['ID', 'Date', 'AwayTeam', 'FTAG', 'FTR', 'FTGT', 'HTAG', 'HTR', 'AS', 'AST', 'AC', 'AF', 'AY', 'AR',
                         'B365H', 'BWH', 'IWH', 'VCH', 'WHH',
                         'B365D', 'BWD', 'IWD', 'VCD', 'WHD',
                         'B365A', 'BWA', 'IWA', 'VCA', 'WHA',
                         'BbAv>2.5', 'BbAv<2.5']].copy()
        frame_A.columns = columns_name
        frame_A['Location'] = 0

        # Merge and making stats
        new_data_set = pd.merge(frame_h, frame_A, how='outer').sort_values('ID')

        new_data_set['FTGT_ALL_MEAN'] = new_data_set.groupby('Team')['FTGT'].transform(lambda x: x.expanding().mean().shift())
        new_data_set['FTG_ALL_MEAN'] = new_data_set.groupby('Team')['FTG'].transform(lambda x: x.expanding().mean().shift())
        new_data_set['HTG_ALL_MEAN'] = new_data_set.groupby('Team')['HTG'].transform(lambda x: x.expanding().mean().shift())
        new_data_set['S_ALL_MEAN'] = new_data_set.groupby('Team')['S'].transform(lambda x: x.expanding().mean().shift())
        new_data_set['ST_ALL_MEAN'] = new_data_set.groupby('Team')['ST'].transform(lambda x: x.expanding().mean().shift())
        new_data_set['C_ALL_MEAN'] = new_data_set.groupby('Team')['C'].transform(lambda x: x.expanding().mean().shift())
        new_data_set['F_ALL_MEAN'] = new_data_set.groupby('Team')['F'].transform(lambda x: x.expanding().mean().shift())
        new_data_set['Y_ALL_MEAN'] = new_data_set.groupby('Team')['Y'].transform(lambda x: x.expanding().mean().shift())
        new_data_set['R_ALL_MEAN'] = new_data_set.groupby('Team')['R'].transform(lambda x: x.expanding().mean().shift())

        # Sep and merging again by game
        home_teams = new_data_set[new_data_set['Location'] == 1]
        home_teams.columns = ['ID', 'Date', 'HomeTeam', 'FTHG', 'FTR', 'FTGT', 'HTHG', 'HTR', 'HS', 'HST', 'HC', 'HF', 'HY',
                              'HR', 'B365H', 'BWH', 'IWH', 'VCH', 'WHH', 'B365D', 'BWD', 'IWD', 'VCD', 'WHD',
                              'B365A', 'BWA', 'IWA', 'VCA', 'WHA', 'BbAv>2.5', 'BbAv<2.5', 'Location', 'FTGHT_ALL_MEAN',
                              'FTHG_ALL_MEAN', 'HTHG_ALL_MEAN', 'HS_ALL_MEAN', 'HST_ALL_MEAN', 'HC_ALL_MEAN', 'HF_ALL_MEAN',
                              'HY_ALL_MEAN', 'HR_ALL_MEAN']
        #home_teams.reset_index(drop=True)

        away_teams = new_data_set[new_data_set['Location'] == 0]
        away_teams.columns = ['ID', 'Date', 'AwayTeam', 'FTAG', 'FTR', 'FTGT', 'HTAG', 'HTR', 'AS', 'AST', 'AC', 'AF', 'AY',
                              'AR', 'B365H', 'BWH', 'IWH', 'VCH', 'WHH', 'B365D', 'BWD', 'IWD', 'VCD', 'WHD',
                              'B365A', 'BWA', 'IWA', 'VCA', 'WHA', 'BbAv>2.5', 'BbAv<2.5', 'Location', 'FTGAT_ALL_MEAN',
                              'FTAG_ALL_MEAN', 'HTAG_ALL_MEAN', 'AS_ALL_MEAN', 'AST_ALL_MEAN', 'AC_ALL_MEAN', 'AF_ALL_MEAN',
                              'AY_ALL_MEAN', 'AR_ALL_MEAN']
        #away_teams.reset_index(drop=True)

        frame_merge = pd.merge(home_teams, away_teams, left_on=['ID', 'Date', 'B365H', 'BWH', 'IWH', 'VCH', 'WHH', 'B365D', 'BWD', 'IWD', 'VCD', 'WHD', 'B365A', 'BWA', 'IWA', 'VCA', 'WHA', 'BbAv>2.5', 'BbAv<2.5'],
                               right_on=['ID', 'Date', 'B365H', 'BWH', 'IWH', 'VCH', 'WHH', 'B365D', 'BWD', 'IWD', 'VCD', 'WHD', 'B365A', 'BWA', 'IWA', 'VCA', 'WHA', 'BbAv>2.5', 'BbAv<2.5'])
                               
        frame_merge = frame_merge.drop(['FTR_y', 'HTR_y', 'Location_y', 'Location_x', 'FTGT_y'], axis=1)
        frame_merge = frame_merge.rename(columns={'FTR_x': 'FTR', 'HTR_x': 'HTR', 'FTGT_x': 'FTGT'})

        frame_merge['FTGHT_MEAN'] = frame_merge.groupby('HomeTeam')['FTGT'].transform(lambda x: x.expanding().mean().shift())
        frame_merge['FTGAT_MEAN'] = frame_merge.groupby('AwayTeam')['FTGT'].transform(lambda x: x.expanding().mean().shift())
        frame_merge['FTHG_MEAN'] = frame_merge.groupby('HomeTeam')['FTHG'].transform(lambda x: x.expanding().mean().shift()) # FTHG = Full Time Home Team Goals
        frame_merge['FTAG_MEAN'] = frame_merge.groupby('AwayTeam')['FTAG'].transform(lambda x: x.expanding().mean().shift()) # FTAG = Full Time Away Team Goals
        frame_merge['HTHG_MEAN'] = frame_merge.groupby('HomeTeam')['HTHG'].transform(lambda x: x.expanding().mean().shift()) # HTHG = Half Time Home Team Goals
        frame_merge['HTAG_MEAN'] = frame_merge.groupby('AwayTeam')['HTAG'].transform(lambda x: x.expanding().mean().shift()) # HTAG = Half Time Away Team Goals
        frame_merge['HS_MEAN'] = frame_merge.groupby('HomeTeam')['HS'].transform(lambda x: x.expanding().mean().shift()) # HS = Home Team Shots
        frame_merge['AS_MEAN'] = frame_merge.groupby('AwayTeam')['AS'].transform(lambda x: x.expanding().mean().shift()) # AS = Away Team Shots
        frame_merge['HST_MEAN'] = frame_merge.groupby('HomeTeam')['HST'].transform(lambda x: x.expanding().mean().shift())  # HST = Home Team Shots on Target
        frame_merge['AST_MEAN'] = frame_merge.groupby('AwayTeam')['AST'].transform(lambda x: x.expanding().mean().shift()) # AST = Away Team Shots on Target
        frame_merge['HC_MEAN'] = frame_merge.groupby('HomeTeam')['HC'].transform(lambda x: x.expanding().mean().shift()) # HC = Home Team Corners
        frame_merge['AC_MEAN'] = frame_merge.groupby('AwayTeam')['AC'].transform(lambda x: x.expanding().mean().shift()) # AC = Away Team Corners
        frame_merge['HF_MEAN'] = frame_merge.groupby('HomeTeam')['HF'].transform(lambda x: x.expanding().mean().shift()) # HF = Home Team Fouls Committed
        frame_merge['AF_MEAN'] = frame_merge.groupby('AwayTeam')['AF'].transform(lambda x: x.expanding().mean().shift()) # AF = Away Team Fouls Committed
        frame_merge['HY_MEAN'] = frame_merge.groupby('HomeTeam')['HY'].transform(lambda x: x.expanding().mean().shift()) # HY = Home Team Yellow Cards
        frame_merge['AY_MEAN'] = frame_merge.groupby('AwayTeam')['AY'].transform(lambda x: x.expanding().mean().shift()) # AY = Away Team Yellow Cards
        frame_merge['HR_MEAN'] = frame_merge.groupby('HomeTeam')['HR'].transform(lambda x: x.expanding().mean().shift()) # HR = Home Team Red Cards
        frame_merge['AR_MEAN'] = frame_merge.groupby('AwayTeam')['AR'].transform(lambda x: x.expanding().mean().shift()) # AR = Away Team Red Cards

        # Mean of odds
        frame_merge['AvgHr'] = frame_merge[['B365H', 'BWH', 'IWH', 'VCH', 'WHH']].mean(axis=1) # AvgHr = Market average home win odds
        frame_merge['AvgDr'] = frame_merge[['B365D', 'BWD', 'IWD', 'VCD', 'WHD']].mean(axis=1) # AvgDr = Market average draw win odds
        frame_merge['AvgAr'] = frame_merge[['B365A', 'BWA', 'IWA', 'VCA', 'WHA']].mean(axis=1) # AvgAr = Market average away win odds

        # Creating the WINS, LOSSES, DRAWS of witch team
        frame_merge['FTR_A'] = np.select([frame_merge['FTR'] == 'H', frame_merge['FTR'] == 'A', frame_merge['FTR'] == 'D'], [0, 2, 1], default=None)
        frame_merge['FTR'] = frame_merge['FTR'].map({'H': 2, 'A': 0, 'D': 1}) # FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)
        frame_merge['HTR'] = frame_merge['HTR'].map({'H': 2, 'A': 0, 'D': 1}) # HTR = Half Time Result (H=Home Win, D=Draw, A=Away Win)
        frame_merge['H_POINTS'] = frame_merge.groupby('HomeTeam')['FTR'].transform(lambda x: x.expanding().sum().shift())
        frame_merge['A_POINTS'] = frame_merge.groupby('AwayTeam')['FTR_A'].transform(lambda x: x.expanding().sum().shift())
        frame_merge['H_DRAWS'] = frame_merge.groupby('HomeTeam')['FTR'].transform(lambda x: (x == 1).expanding().sum().shift())
        frame_merge['A_DRAWS'] = frame_merge.groupby('AwayTeam')['FTR_A'].transform(lambda x: (x == 1).expanding().sum().shift())
        frame_merge['Diff_POINTS'] = frame_merge['H_POINTS'] - frame_merge['A_POINTS']

        #Last 2
        frame_merge['LAST_2_MP_H'] = frame_merge.groupby('HomeTeam')['FTR'].transform(lambda x: x.rolling(window=2).sum().shift())
        frame_merge['LAST_2_MP_A'] = frame_merge.groupby('AwayTeam')['FTR_A'].transform(lambda x: x.rolling(window=2).sum().shift())
        frame_merge['LAST_2_FTGHT_MEAN'] = frame_merge.groupby('HomeTeam')['FTGT'].transform(lambda x: x.rolling(window=2).mean().shift())
        frame_merge['LAST_2_FTGAT_MEAN'] = frame_merge.groupby('AwayTeam')['FTGT'].transform(lambda x: x.rolling(window=2).mean().shift())
        frame_merge['LAST_2_FTHG_MEAN'] = frame_merge.groupby('HomeTeam')['FTHG'].transform(lambda x: x.rolling(window=2).mean().shift())
        frame_merge['LAST_2_FTAG_MEAN'] = frame_merge.groupby('AwayTeam')['FTAG'].transform(lambda x: x.rolling(window=2).mean().shift())
        frame_merge['LAST_2_HS_MEAN'] = frame_merge.groupby('HomeTeam')['HS'].transform(lambda x: x.rolling(window=2).mean().shift())
        frame_merge['LAST_2_AS_MEAN'] = frame_merge.groupby('AwayTeam')['AS'].transform(lambda x: x.rolling(window=2).mean().shift())
        frame_merge['LAST_2_HC_MEAN'] = frame_merge.groupby('HomeTeam')['HC'].transform(lambda x: x.rolling(window=2).mean().shift())
        frame_merge['LAST_2_AC_MEAN'] = frame_merge.groupby('AwayTeam')['AC'].transform(lambda x: x.rolling(window=2).mean().shift())
        frame_merge['LAST_2_HF_MEAN'] = frame_merge.groupby('HomeTeam')['HF'].transform(lambda x: x.rolling(window=2).mean().shift())
        frame_merge['LAST_2_AF_MEAN'] = frame_merge.groupby('AwayTeam')['AF'].transform(lambda x: x.rolling(window=2).mean().shift())
        frame_merge['LAST_2_HY_MEAN'] = frame_merge.groupby('HomeTeam')['HY'].transform(lambda x: x.rolling(window=2).mean().shift())
        frame_merge['LAST_2_AY_MEAN'] = frame_merge.groupby('AwayTeam')['AY'].transform(lambda x: x.rolling(window=2).mean().shift())
        frame_merge['LAST_2_HR_MEAN'] = frame_merge.groupby('HomeTeam')['HR'].transform(lambda x: x.rolling(window=2).mean().shift())
        frame_merge['LAST_2_AR_MEAN'] = frame_merge.groupby('AwayTeam')['AR'].transform(lambda x: x.rolling(window=2).mean().shift())


        #Last 3
        frame_merge['LAST_3_MP_H'] = frame_merge.groupby('HomeTeam')['FTR'].transform(lambda x: x.rolling(window=3).sum().shift())
        frame_merge['LAST_3_MP_A'] = frame_merge.groupby('AwayTeam')['FTR_A'].transform(lambda x: x.rolling(window=3).sum().shift())
        frame_merge['LAST_3_FTGHT_MEAN'] = frame_merge.groupby('HomeTeam')['FTGT'].transform(lambda x: x.rolling(window=3).mean().shift())
        frame_merge['LAST_3_FTGAT_MEAN'] = frame_merge.groupby('AwayTeam')['FTGT'].transform(lambda x: x.rolling(window=3).mean().shift())
        frame_merge['LAST_3_FTHG_MEAN'] = frame_merge.groupby('HomeTeam')['FTHG'].transform(lambda x: x.rolling(window=3).mean().shift())
        frame_merge['LAST_3_FTAG_MEAN'] = frame_merge.groupby('AwayTeam')['FTAG'].transform(lambda x: x.rolling(window=3).mean().shift())
        frame_merge['LAST_3_HS_MEAN'] = frame_merge.groupby('HomeTeam')['HS'].transform(lambda x: x.rolling(window=3).mean().shift())
        frame_merge['LAST_3_AS_MEAN'] = frame_merge.groupby('AwayTeam')['AS'].transform(lambda x: x.rolling(window=3).mean().shift())
        frame_merge['LAST_3_HC_MEAN'] = frame_merge.groupby('HomeTeam')['HC'].transform(lambda x: x.rolling(window=3).mean().shift())
        frame_merge['LAST_3_AC_MEAN'] = frame_merge.groupby('AwayTeam')['AC'].transform(lambda x: x.rolling(window=3).mean().shift())
        frame_merge['LAST_3_HF_MEAN'] = frame_merge.groupby('HomeTeam')['HF'].transform(lambda x: x.rolling(window=3).mean().shift())
        frame_merge['LAST_3_AF_MEAN'] = frame_merge.groupby('AwayTeam')['AF'].transform(lambda x: x.rolling(window=3).mean().shift())
        frame_merge['LAST_3_HY_MEAN'] = frame_merge.groupby('HomeTeam')['HY'].transform(lambda x: x.rolling(window=3).mean().shift())
        frame_merge['LAST_3_AY_MEAN'] = frame_merge.groupby('AwayTeam')['AY'].transform(lambda x: x.rolling(window=3).mean().shift())
        frame_merge['LAST_3_HR_MEAN'] = frame_merge.groupby('HomeTeam')['HR'].transform(lambda x: x.rolling(window=3).mean().shift())
        frame_merge['LAST_3_AR_MEAN'] = frame_merge.groupby('AwayTeam')['AR'].transform(lambda x: x.rolling(window=3).mean().shift())

        # Last 5
        frame_merge['LAST_5_MP_H'] = frame_merge.groupby('HomeTeam')['FTR'].transform(lambda x: x.rolling(window=5).sum().shift())
        frame_merge['LAST_5_MP_A'] = frame_merge.groupby('AwayTeam')['FTR_A'].transform(lambda x: x.rolling(window=5).sum().shift())
        frame_merge['LAST_5_FTGHT_MEAN'] = frame_merge.groupby('HomeTeam')['FTGT'].transform(lambda x: x.rolling(window=5).mean().shift())
        frame_merge['LAST_5_FTGAT_MEAN'] = frame_merge.groupby('AwayTeam')['FTGT'].transform(lambda x: x.rolling(window=5).mean().shift())
        frame_merge['LAST_5_FTHG_MEAN'] = frame_merge.groupby('HomeTeam')['FTHG'].transform(lambda x: x.rolling(window=5).mean().shift())
        frame_merge['LAST_5_FTAG_MEAN'] = frame_merge.groupby('AwayTeam')['FTAG'].transform(lambda x: x.rolling(window=5).mean().shift())
        frame_merge['LAST_5_HS_MEAN'] = frame_merge.groupby('HomeTeam')['HS'].transform(lambda x: x.rolling(window=5).mean().shift())
        frame_merge['LAST_5_AS_MEAN'] = frame_merge.groupby('AwayTeam')['AS'].transform(lambda x: x.rolling(window=5).mean().shift())
        frame_merge['LAST_5_HC_MEAN'] = frame_merge.groupby('HomeTeam')['HC'].transform(lambda x: x.rolling(window=5).mean().shift())
        frame_merge['LAST_5_AC_MEAN'] = frame_merge.groupby('AwayTeam')['AC'].transform(lambda x: x.rolling(window=5).mean().shift())
        frame_merge['LAST_5_HF_MEAN'] = frame_merge.groupby('HomeTeam')['HF'].transform(lambda x: x.rolling(window=5).mean().shift())
        frame_merge['LAST_5_AF_MEAN'] = frame_merge.groupby('AwayTeam')['AF'].transform(lambda x: x.rolling(window=5).mean().shift())
        frame_merge['LAST_5_HY_MEAN'] = frame_merge.groupby('HomeTeam')['HY'].transform(lambda x: x.rolling(window=5).mean().shift())
        frame_merge['LAST_5_AY_MEAN'] = frame_merge.groupby('AwayTeam')['AY'].transform(lambda x: x.rolling(window=5).mean().shift())
        frame_merge['LAST_5_HR_MEAN'] = frame_merge.groupby('HomeTeam')['HR'].transform(lambda x: x.rolling(window=5).mean().shift())
        frame_merge['LAST_5_AR_MEAN'] = frame_merge.groupby('AwayTeam')['AR'].transform(lambda x: x.rolling(window=5).mean().shift())
        new_frames.append(frame_merge)

    return new_frames


def data_set():
    frames = frames_season()
    premier_league_stats = pd.concat(frames)
    columns_to_use = ['Date', 'HomeTeam', 'AwayTeam', 'FTGT', 'FTHG', 'FTHG_MEAN', 'FTAG', 'FTR', 'FTR_A', 'Diff_POINTS',
                      'LAST_2_MP_H', 'LAST_2_MP_A', 'LAST_3_MP_H', 'LAST_3_MP_A', 'LAST_5_MP_H', 'LAST_5_MP_A', 'H_DRAWS',
                      'A_DRAWS', 'H_POINTS', 'A_POINTS', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF',
                      'AF', 'HY', 'AY', 'HR', 'AR', 'AvgHr', 'AvgDr', 'AvgAr', 'BbAv>2.5', 'BbAv<2.5', 'FTAG_MEAN', 'HTHG_MEAN', 'HTAG_MEAN', 'HS_MEAN',
                      'AS_MEAN', 'HST_MEAN', 'AST_MEAN', 'HC_MEAN', 'AC_MEAN', 'HF_MEAN', 'AF_MEAN', 'HY_MEAN', 'AY_MEAN', 'HR_MEAN',
                      'AR_MEAN', 'LAST_2_FTHG_MEAN', 'LAST_2_FTAG_MEAN', 'LAST_3_FTHG_MEAN', 'LAST_3_FTAG_MEAN', 'LAST_5_FTHG_MEAN', 'LAST_5_FTAG_MEAN',
                      'LAST_2_HS_MEAN', 'LAST_2_AS_MEAN', 'LAST_2_HC_MEAN', 'LAST_2_AC_MEAN', 'LAST_2_HF_MEAN', 'LAST_2_AF_MEAN',
                      'LAST_2_HY_MEAN', 'LAST_2_AY_MEAN', 'LAST_2_HR_MEAN', 'LAST_2_AR_MEAN',
                      'LAST_3_HS_MEAN', 'LAST_3_AS_MEAN', 'LAST_3_HC_MEAN', 'LAST_3_AC_MEAN', 'LAST_3_HF_MEAN',
                      'LAST_3_AF_MEAN', 'LAST_3_HY_MEAN', 'LAST_3_AY_MEAN', 'LAST_3_HR_MEAN', 'LAST_3_AR_MEAN',
                      'LAST_5_HS_MEAN', 'LAST_5_AS_MEAN', 'LAST_5_HC_MEAN', 'LAST_5_AC_MEAN', 'LAST_5_HF_MEAN',
                      'LAST_5_AF_MEAN', 'LAST_5_HY_MEAN', 'LAST_5_AY_MEAN', 'LAST_5_HR_MEAN', 'LAST_5_AR_MEAN',
                      'FTGHT_ALL_MEAN', 'FTHG_ALL_MEAN', 'HTHG_ALL_MEAN', 'HS_ALL_MEAN', 'HST_ALL_MEAN', 'HC_ALL_MEAN', 'HF_ALL_MEAN', 'HY_ALL_MEAN', 'HR_ALL_MEAN',
                      'FTGAT_ALL_MEAN', 'FTAG_ALL_MEAN', 'HTAG_ALL_MEAN', 'AS_ALL_MEAN', 'AST_ALL_MEAN', 'AC_ALL_MEAN', 'AF_ALL_MEAN',
                      'AY_ALL_MEAN', 'AR_ALL_MEAN',
                      'FTGHT_MEAN', 'FTGAT_MEAN', 'LAST_2_FTGHT_MEAN', 'LAST_2_FTGAT_MEAN', 'LAST_3_FTGHT_MEAN', 'LAST_3_FTGAT_MEAN', 'LAST_5_FTGHT_MEAN', 'LAST_5_FTGAT_MEAN']
    premier_league_stats = premier_league_stats[columns_to_use]
    premier_league_stats = premier_league_stats.dropna().reset_index(drop=True)
    return premier_league_stats


def feature_and_target(tML=str, tar=str):
    features = ['LAST_2_HS_MEAN', 'LAST_2_AS_MEAN', 'LAST_2_HC_MEAN', 'LAST_2_AC_MEAN', 'LAST_2_HF_MEAN',
            'LAST_2_AF_MEAN', 'LAST_2_HY_MEAN', 'LAST_2_AY_MEAN', 'LAST_2_HR_MEAN', 'LAST_2_AR_MEAN',
            'LAST_3_HS_MEAN', 'LAST_3_AS_MEAN', 'LAST_3_HC_MEAN', 'LAST_3_AC_MEAN', 'LAST_3_HF_MEAN',
            'LAST_3_AF_MEAN', 'LAST_3_HY_MEAN', 'LAST_3_AY_MEAN', 'LAST_3_HR_MEAN', 'LAST_3_AR_MEAN',
            'LAST_5_HS_MEAN', 'LAST_5_AS_MEAN', 'LAST_5_HC_MEAN', 'LAST_5_AC_MEAN', 'LAST_5_HF_MEAN',
            'LAST_5_AF_MEAN', 'LAST_5_HY_MEAN', 'LAST_5_AY_MEAN', 'LAST_5_HR_MEAN', 'LAST_5_AR_MEAN',
            'FTGHT_ALL_MEAN', 'FTHG_ALL_MEAN', 'HTHG_ALL_MEAN', 'HS_ALL_MEAN', 'HST_ALL_MEAN', 'HC_ALL_MEAN', 'HF_ALL_MEAN', 'HY_ALL_MEAN',
            'HR_ALL_MEAN', 'FTGAT_ALL_MEAN', 'FTAG_ALL_MEAN', 'HTAG_ALL_MEAN', 'AS_ALL_MEAN', 'AST_ALL_MEAN', 'AC_ALL_MEAN', 'AF_ALL_MEAN',
            'AY_ALL_MEAN', 'AR_ALL_MEAN', 'FTHG_MEAN', 'FTAG_MEAN', 'HTHG_MEAN', 'HTAG_MEAN', 'HS_MEAN', 'AS_MEAN', 'HST_MEAN',
            'AST_MEAN', 'HC_MEAN', 'AC_MEAN', 'HF_MEAN', 'AF_MEAN', 'HY_MEAN', 'AY_MEAN', 'HR_MEAN', 'AR_MEAN', 'H_POINTS', 'A_POINTS',
            'H_DRAWS', 'A_DRAWS', 'Diff_POINTS', 'LAST_2_MP_H', 'LAST_2_MP_A', 'LAST_2_FTHG_MEAN', 'LAST_2_FTAG_MEAN',
            'LAST_3_MP_H', 'LAST_3_MP_A', 'LAST_3_FTHG_MEAN', 'LAST_3_FTAG_MEAN', 'LAST_5_MP_H', 'LAST_5_MP_A', 'LAST_5_FTHG_MEAN',
            'LAST_5_FTAG_MEAN']

    if tML == 'classifier':
        target = [tar]

    if tML == 'regression':
        target = [tar]

    return features, target


def KNN(Xtr, ytr, Xte, yte): #KNN(near neightboor) test
    for i in range(1, 20):
        knn = KNeighborsClassifier(n_neighbors=i).fit(Xtr, ytr)
        if i == 1:
            s_te = knn.score(Xte, yte)
            s_tr = knn.score(Xtr, ytr)
            g = i
        if knn.score(Xte, yte) > s_te:
            s_te = knn.score(Xte, yte)
            s_tr = knn.score(Xtr, ytr)
            g = i
    print(f'The best n_neighbors is: {g}')
    print(f'Accuracy of K-NN ({g}) test on train set is: {s_tr}')
    print(f'Accuracy of K-NN ({g}) test on test set is: {s_te}')
    return knn.predict(Xte)


def Decision_tree(Xtr, ytr, Xte, yte): #Decision tree test
    dt = DecisionTreeClassifier().fit(Xtr, ytr)
    print(f'Accuracy of DecisionTreeClassifier test on train set is: {dt.score(Xtr, ytr)}')
    print(f'Accuracy of DecisionTreeClassifier test on test set is: {dt.score(Xte, yte)}')
    return dt.predict(Xte)


def SVC_test(Xtr, ytr, Xte, yte=None, text=None): #SVC
    for i in [0.01, 0.1, 1, 2, 10]:
        svm = SVC(gamma=i).fit(Xtr, ytr)
        if i == 0.01:
            s_te = svm.score(Xte, yte)
            s_tr = svm.score(Xtr, ytr)
            g = i
        if svm.score(Xte, yte) > s_te:
            s_te = svm.score(Xte, yte)
            s_tr = svm.score(Xtr, ytr)
            g = i
    if text is not None:
        print(f'The best params is: {g}')
        print(f'Accuracy of Logistic Regression test on train set is: {svm.score(Xtr, ytr)}')
    if yte is not None:
        print(f'Accuracy of Logistic Regression test on test set is: {svm.score(Xte, yte)}')
    return svm.predict(Xte)


def logistic_reg(Xtr, ytr, Xte, yte=None, text=None):
    lr = LogisticRegression(penalty='l2', C=7.74263682, max_iter=4000).fit(Xtr, ytr)
    y_pred = lr.predict(Xte)
    if text is not None:
        print(f'Accuracy of Logistic Regression test on train set is: {lr.score(Xtr, ytr)}')
    if yte is not None:
        print(f'Accuracy of Logistic Regression test on test set is: {lr.score(Xte, yte)}')

    return y_pred


def linear_reg(Xtr, ytr, Xte, yte=None, text=None):
    lir = LinearRegression().fit(Xtr, ytr)
    y_pred = lir.predict(Xte)
    if text is not None:
        print(f'Accuracy of Linear Regression test on train set is: {lir.score(Xtr, ytr)}')

    if yte is not None:
        ## Kpi
        print(f"R2 (explained variance):{round(r2_score(yte, y_pred), 3)}")
        print(f'Mean Absolute Percentual Error (Σ(|y-pred|/y)/n):{round(np.mean(np.abs((yte - y_pred) / y_pred)), 2)}')
        print(f'Mean Absolute Error (Σ|y-pred|/n): {(mean_absolute_error(yte, y_pred)):.2f}')
        print(f"Root Mean Squared Error (sqrt(Σ(y-pred)^2/n)): {(np.sqrt(mean_squared_error(yte, y_pred))):.2f}")

    return y_pred


def RandomF(Xtr, ytr, Xte, yte):
    '''{'max_depth': 5, 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 100}'''
    rf = RandomForestClassifier(max_depth=5, min_samples_leaf=5, min_impurity_split=2, n_estimators=100).fit(Xtr, ytr.values.ravel())
    print(f'Accuracy of Logistic Regression test on train set is: {rf.score(Xtr, ytr)}')
    print(f'Accuracy of Logistic Regression test on test set is: {rf.score(Xte, yte)}')
    return rf.predict(Xte)


def XGB(Xtr, ytr, Xte, yte=None):
    xg = XGBClassifier().fit(Xtr, ytr)
    y_pred = xg.predict(Xte)
    print(f'Accuracy of Logistic Regression test on train set is: {xg.score(Xtr, ytr)}')
    if yte is not None:
        print(f'Accuracy of Logistic Regression test on test set is: {xg.score(Xte, yte)}')
    return y_pred


def XGB_r(Xtr, ytr, Xte, yte=None, text=None):
    xgb = XGBRegressor().fit(Xtr, ytr)
    y_pred = xgb.predict(Xte)
    if text is not None:
        print(f'R2 at the train set (explained variance): {xgb.score(Xtr, ytr)}')

    if yte is not None:
        ## Kpi
        print(f"R2 at the test set (explained variance):{round(r2_score(yte, y_pred), 3)}")
        print(f'Mean Absolute Error (Σ|y-pred|/n): {(mean_absolute_error(yte, y_pred)):.2f}')
        print(f"Root Mean Squared Error (sqrt(Σ(y-pred)^2/n)): {(np.sqrt(mean_squared_error(yte, y_pred))):.2f}")

    return y_pred


def features_importances(Xtr, ytr, tML='classifier'):
    if tML == 'regression':
        ## call model
        model = GradientBoostingRegressor()
        ## Importance
        model.fit(Xtr, ytr)
        importances = model.feature_importances_
        ## Put in a pandas dtf
        dtf_importances = pd.DataFrame({"IMPORTANCE":importances, "VARIABLE":Xtr.columns.tolist()}).sort_values("IMPORTANCE", ascending=False)
        dtf_importances['cumsum'] = dtf_importances['IMPORTANCE'].cumsum(axis=0)
        dtf_importances = dtf_importances.set_index("VARIABLE")
        
        ## Plot
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=[50,70])
        fig.suptitle("Features Importance", fontsize=100)
        ax[0].title.set_text('variables')
        dtf_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(kind="barh", legend=False, ax=ax[0], fontsize=30).grid(axis="x")
        ax[0].set(ylabel="")
        ax[1].title.set_text('cumulative')
        dtf_importances[["cumsum"]].plot(kind="line", linewidth=4, legend=False, ax=ax[1])
        ax[1].set(xlabel="", xticks=np.arange(len(dtf_importances)), xticklabels=dtf_importances.index)
        plt.xticks(rotation=70)
        plt.grid(axis='both')
        plt.show()

    if tML =='classifier':
        ## call model
        model = GradientBoostingClassifier()
        ## Importance
        model.fit(Xtr, ytr)
        importances = model.feature_importances_
        ## Put in a pandas dtf
        dtf_importances = pd.DataFrame({"IMPORTANCE":importances, "VARIABLE":Xtr.columns.tolist()}).sort_values("IMPORTANCE", ascending=False)
        dtf_importances['cumsum'] = dtf_importances['IMPORTANCE'].cumsum(axis=0)
        dtf_importances = dtf_importances.set_index("VARIABLE")
        
        ## Plot
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=[50,70])
        fig.suptitle("Features Importance", fontsize=100)
        ax[0].title.set_text('variables')
        dtf_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(kind="barh", legend=False, ax=ax[0], fontsize=30).grid(axis="x")
        ax[0].set(ylabel="")
        ax[1].title.set_text('cumulative')
        dtf_importances[["cumsum"]].plot(kind="line", linewidth=4, legend=False, ax=ax[1])
        ax[1].set(xlabel="", xticks=np.arange(len(dtf_importances)), xticklabels=dtf_importances.index)
        plt.xticks(rotation=70)
        plt.grid(axis='both')
        plt.show()


def scaler(X_tr, X_te):
    scaller = MinMaxScaler()
    X_train_transf = scaller.fit_transform(X_tr)
    X_test_transf = scaller.transform(X_te)
    return X_train_transf, X_test_transf


def by_rods_C(temp_line, season=str, bet_by_rod=10, tML='classifier', tar=str):
    import matplotlib.pyplot as plt
    import seaborn as sns

    features, target = feature_and_target(tML, tar)
    data_base = data_set()
    rods = []
    index_r = []
    for i in range(temp_line, (temp_line + (10 * 28)) - 2, 10):
        rods.append(i)

    gain_by_rod = []
    gain_by_rod_f = []
    frames = []
    frames_f = []
    for item in rods:
        try:
            train = data_base[:item]
            X_train = train[features].copy()
            y_train = train[target].copy()
            test = data_base[item:(item + 10)]
            X_test = test[features].copy()
            y_test = test[target].copy()

            X_train_transf, X_test_transf = scaler(X_train, X_test)

            X_test['PREDICTS'] = logistic_reg(X_train_transf, y_train, X_test_transf, y_test)
            globals()['DB_test' + str(item)] = pd.merge(X_test, data_base, how='left')
            globals()['DB_test' + str(item)] = globals()['DB_test' + str(item)][['Date', 'HomeTeam', 'AwayTeam', 'AvgHr', 'AvgAr', 'AvgDr', 'FTR', 'PREDICTS']]
            globals()['DB_test' + str(item)]['ODD_CHOSEN'] = np.select([globals()['DB_test' + str(item)]['PREDICTS'] == 2, globals()['DB_test' + str(item)]['PREDICTS'] == 1, globals()['DB_test' + str(item)]['PREDICTS'] == 0],
                                                                       [globals()['DB_test' + str(item)]['AvgHr'] - 1, globals()['DB_test' + str(item)]['AvgDr'] - 1, globals()['DB_test' + str(item)]['AvgAr'] - 1], default=None)
            globals()['DB_test' + str(item)]['GAIN'] = np.where(globals()['DB_test' + str(item)]['FTR'] == globals()['DB_test' + str(item)]['PREDICTS'], globals()['DB_test' + str(item)]['ODD_CHOSEN'], -1)
            gain_by_rod.append(sum(globals()['DB_test' + str(item)]["GAIN"]) / len(globals()['DB_test' + str(item)]) * 100)
            globals()['DB_test_F' + str(item)] = globals()['DB_test' + str(item)][globals()['DB_test' + str(item)]['ODD_CHOSEN'] < 1]
            gain_by_rod_f.append(sum(globals()['DB_test_F' + str(item)]["GAIN"]) / len(globals()['DB_test_F' + str(item)]) * 100)
            frames.append(globals()['DB_test' + str(item)])
            frames_f.append(globals()['DB_test_F' + str(item)])


        except:
            pass

    for i in range(11, (len(gain_by_rod) + 10) + 1):
        index_r.append(i)

    df_by_rod = pd.DataFrame(data={'Gain ML': gain_by_rod, 'Gain F': gain_by_rod_f}, index=index_r)
    df_by_rod['U$D Gain ML'] = df_by_rod['Gain ML'] * bet_by_rod / 100
    df_by_rod['U$D Gain F'] = df_by_rod['Gain F'] * bet_by_rod / 100

    print(f'The gain with the normal machine learning, betting US$ {bet_by_rod:.2f} per round, would be: '
          f'US$ {df_by_rod["U$D Gain ML"].sum():.2f}, {(df_by_rod["U$D Gain ML"].sum()/(len(rods) * bet_by_rod)) * 100:.2f}%\n'
          f'The gain with the team favorites machine learning, betting US$ {bet_by_rod:.2f} per round,  would be: '
          f'US$ {df_by_rod["U$D Gain F"].sum():.2f}, {(df_by_rod["U$D Gain F"].sum()/(len(rods) * bet_by_rod)) * 100:.2f}%')

    plt.figure()
    sns.set(style="dark")
    sns.lineplot(data=df_by_rod[['Gain ML', 'Gain F']], palette="PuBuGn_d", linewidth=2.5).lines[1].set_linestyle("-")
    plt.plot([index_r[0], index_r[-1]], [0, 0], '-g', linewidth=1, alpha=0.8)
    plt.title('Season ' + season)
    plt.show()
    return pd.concat(frames), pd.concat(frames_f)


def by_rods_R(temp_line, season=str, bet_by_rod=10, tML='regression', tar=str):
    import matplotlib.pyplot as plt
    import seaborn as sns

    features, target = feature_and_target(tML, tar)
    data_base = data_set()
    rods = []
    index_r = []
    for i in range(temp_line, (temp_line + (10 * 28)) - 2, 10):
        rods.append(i)

    gain_by_rod_menor = []
    gain_by_rod_maior = []
    frames_menor = []
    frames_maior = []
    for item in rods:
        try:
            train = data_base[:item]
            X_train = train[features].copy()
            y_train = train[target].copy()
            test = data_base[item:(item + 10)]
            X_test = test[features].copy()
            y_test = test[target].copy()

            X_train_transf, X_test_transf = scaler(X_train, X_test)

            X_test['PREDICTS'] = XGB_r(X_train_transf, y_train, X_test_transf, y_test)
            globals()['DB_test' + str(item)] = pd.merge(X_test, data_base, how='left')
            globals()['DB_test' + str(item)] = globals()['DB_test' + str(item)][['Date', 'HomeTeam', 'AwayTeam', 'AvgHr', 'FTR', 'PREDICTS']]
            globals()['DB_test' + str(item)]['GAIN'] = np.where(globals()['DB_test' + str(item)]['FTR'] == 2, globals()['DB_test' + str(item)]['AvgHr'] - 1, -1)
            globals()['DB_test' + str(item)]['aposta'] = np.where(globals()['DB_test' + str(item)]['AvgHr'] > globals()['DB_test' + str(item)]['PREDICTS'], 1, 0)
            globals()['DB_test_menor' + str(item)] = globals()['DB_test' + str(item)][globals()['DB_test' + str(item)]['aposta'] == 1]
            globals()['DB_test_maior' + str(item)] = globals()['DB_test' + str(item)][globals()['DB_test' + str(item)]['aposta'] == 0]
            gain_by_rod_menor.append(sum(globals()['DB_test_menor' + str(item)]["GAIN"]) / len(globals()['DB_test_menor' + str(item)]) * 100)
            gain_by_rod_maior.append(sum(globals()['DB_test_maior' + str(item)]["GAIN"]) / len(globals()['DB_test_maior' + str(item)]) * 100)
            frames_menor.append(globals()['DB_test_menor' + str(item)])
            frames_maior.append(globals()['DB_test_maior' + str(item)])

        except:
            pass

    for i in range(11, (len(gain_by_rod_menor) + 10) + 1):
        index_r.append(i)

    df_by_rod = pd.DataFrame(data={'Gain menor': gain_by_rod_menor, 'Gain maior': gain_by_rod_maior}, index=index_r)
    df_by_rod['U$D Gain menor'] = df_by_rod['Gain menor'] * bet_by_rod / 100
    df_by_rod['U$D Gain maior'] = df_by_rod['Gain maior'] * bet_by_rod / 100

    print(f'The gain with the menor machine learning, betting US$ {bet_by_rod:.2f} per round, would be: '
          f'US$ {df_by_rod["U$D Gain menor"].sum():.2f}, {(df_by_rod["U$D Gain menor"].sum()/(len(rods) * bet_by_rod)) * 100:.2f}%\n'
          f'The gain with the maior machine learning, betting US$ {bet_by_rod:.2f} per round,  would be: '
          f'US$ {df_by_rod["U$D Gain maior"].sum():.2f}, {(df_by_rod["U$D Gain maior"].sum()/(len(rods) * bet_by_rod)) * 100:.2f}%')

    plt.figure()
    sns.set(style="dark")
    sns.lineplot(data=df_by_rod[['Gain menor', 'Gain maior']], palette="PuBuGn_d", linewidth=2.5).lines[1].set_linestyle("-")
    plt.plot([index_r[0], index_r[-1]], [0, 0], '-g', linewidth=1, alpha=0.8)
    plt.title('Season ' + season)
    plt.show()
    return pd.concat(frames_menor), pd.concat(frames_maior)


def training_test_data(tML=str, tar=str):
    if tML == 'classifier':
        premier_league_ds = data_set()
        features, target = feature_and_target(tML, tar)

        X_train, X_test, y_train, y_test = train_test_split(premier_league_ds[features], premier_league_ds[target], random_state=0)

        X_train_transf, X_test_transf = scaler(X_train, X_test)

        X_test['PREDICTS'] = SVC_test(X_train_transf, y_train, X_test_transf, y_test, text=True)
        DB_test = pd.merge(X_test, premier_league_ds, how='left')
        DB_test['ODD_CHOSEN'] = np.select([DB_test['PREDICTS'] == 2, DB_test['PREDICTS'] == 1, DB_test['PREDICTS'] == 0], [DB_test['AvgHr'] - 1, DB_test['AvgDr'] - 1, DB_test['AvgAr'] - 1], default=None)
        DB_test = DB_test[['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'PREDICTS', 'ODD_CHOSEN']]
        DB_test['GAIN'] = np.where(DB_test['FTR'] == DB_test['PREDICTS'], DB_test['ODD_CHOSEN'], -1)
        #DB_test = DB_test[DB_test['ODD_CHOSEN'] < 1]
        print(f'Warging 1 dollar in the bet of machine predict, at the end of {len(DB_test)} games, \n'
        f'the gain is {sum(DB_test["GAIN"])} and the returning is {sum(DB_test["GAIN"]) / len(DB_test) * 100:.2f}%.')

        return DB_test

    if tML == 'regression':
        premier_league_ds = data_set()
        features, target = feature_and_target(tML, tar)

        X_train, X_test, y_train, y_test = train_test_split(premier_league_ds[features], premier_league_ds[target], random_state=0)

        X_train_transf, X_test_transf = scaler(X_train, X_test)

        X_test['PREDICTS'] = XGB_r(X_train_transf, y_train, X_test_transf, y_test, text=True)
        DB_test = pd.merge(X_test, premier_league_ds, how='left')
        if tar == 'BbAv<2.5':
            DB_test['Gain'] = np.where((DB_test['FTHG'] + DB_test['FTAG']) < 2, DB_test['BbAv<2.5'] - 1, -1)
            DB_test['aposta'] = np.where(DB_test['BbAv<2.5'] > DB_test['PREDICTS'], 1, 0)
        if tar == 'BbAv>2.5':
            DB_test['Gain'] = np.where((DB_test['FTHG'] + DB_test['FTAG']) > 2, DB_test['BbAv>2.5'] - 1, -1)
            DB_test['aposta'] = np.where(DB_test['BbAv>2.5'] > DB_test['PREDICTS'], 1, 0)
        if tar == 'AvgHr':
            DB_test['Gain'] = np.where(DB_test['FTR'] == 2, DB_test['AvgHr'] - 1, -1)
            DB_test['aposta'] = np.where(DB_test['AvgHr'] > DB_test['PREDICTS'], 1, 0)
        if tar == 'AvgAr':
            DB_test['Gain'] = np.where(DB_test['FTR'] == 0, DB_test['AvgAr'] - 1, -1)
            DB_test['aposta'] = np.where(DB_test['AvgAr'] > DB_test['PREDICTS'], 1, 0)
        if tar == 'AvgDr':
            DB_test['Gain'] = np.where(DB_test['FTR'] == 1, DB_test['AvgHr'] - 1, -1)
            DB_test['aposta'] = np.where(DB_test['AvgAr'] > DB_test['PREDICTS'], 1, 0)
        DB_test = DB_test[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'AvgHr', 'AvgAr', 'AvgDr', 'BbAv>2.5', 'BbAv<2.5', 'PREDICTS', 'Gain', 'aposta']]
        db_test_menor = DB_test[DB_test['aposta'] == 1]
        db_test_maior = DB_test[DB_test['aposta'] == 0]
        print(f'Warging 1 dollar in the overrated odds, at the end of {len(db_test_maior)} games, \n'
              f'the gain is {sum(db_test_maior["Gain"]):.2f} and the returning is {sum(db_test_maior["Gain"]) / len(db_test_maior) * 100:.2f}%.')
        print(f'Warging 1 dollar in the underrated odds, at the end of {len(db_test_menor)} games, \n'
              f'the gain is {sum(db_test_menor["Gain"]):.2f} and the returning is {sum(db_test_menor["Gain"]) / len(db_test_menor) * 100:.2f}%.')
        return db_test_menor, db_test_maior


def next_games(tML=str):
    if tML == 'classifier':
        premier_league_ds = data_set()
        features, target = feature_and_target(tML='classifier')

        X_train = premier_league_ds[features].copy()
        y_train = premier_league_ds[target].copy()

        frames = frames_season()
        next_games = frames[-1][-10:].copy()
        previsoes = next_games[features]

        X_train_transf, X_test_transf = scaler(X_train, previsoes)

        next_games['Predicts'] = logistic_reg(X_train_transf, y_train, X_test_transf)
        next_games = pd.merge(next_games, premier_league_ds, how='left')
        next_games = next_games[['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'AvgHr', 'AvgDr', 'AvgAr', 'Predicts']]
        print(next_games[['HomeTeam', 'AwayTeam', 'Predicts']])

        return next_games, premier_league_ds

    if tML == 'regression':
        premier_league_ds = data_set()
        features, target = feature_and_target(tML='regression')

        X_train = premier_league_ds[features].copy()
        y_train = premier_league_ds[target].copy()

        frames = frames_season()
        next_games = frames[-1][-10:].copy()
        previsoes = next_games[features]

        X_train_transf, X_test_transf = scaler(X_train, previsoes)

        next_games['Predicts'] = linear_reg(X_train_transf, y_train, X_test_transf)
        next_games = pd.merge(next_games, premier_league_ds, how='left')
        next_games = next_games[['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'AvgHr', 'AvgDr', 'AvgAr', 'Predicts']]
        print(next_games[['HomeTeam', 'AwayTeam', 'Predicts']])

        return next_games, premier_league_ds
