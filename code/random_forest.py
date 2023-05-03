#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

#%%
df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/data/KPIs.csv')
#Remove Inf and NaN
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

#Create time columns
df['timestamp'] = pd.to_datetime(df['datetime'])
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.weekday

num_cols = [
    'users_avg',
    'prb_dl_usage_rate_avg',
    'prb_ul_usage_rate_avg', 
    'hsr_intra_freq',
    'hsr_inter_freq',
    'cell_throughput_dl_avg',
    'cell_throughput_ul_avg',
    'user_throughput_dl_avg', 
    'user_throughput_ul_avg',
    'traffic_volume_data'
]

cat_cols = [
    'datetime',
    'enodeb_name',
    'ltecell_name',
    #'timestamp',
    'day',
    'hour'
]

# %%
#Create new features
df['day/night'] = [1 if 7 < x else 0 for x in df['hour']]
df['workday'] = [1 if 4 < x else 0 for x in df['day']]

# %%
#One-Hot encode different cells
dummies = pd.get_dummies(df['ltecell_name'])
df_dummies = pd.concat([df, dummies], axis=1)
df_dummies.drop(cat_cols, axis=1, inplace=True)
# %%
# %%
#Create label from hsr_intra_freq
df_dummies['label'] = [1 if x == 100 else 0 for x in df['hsr_intra_freq']]
# %%
#Based on pairwise correlations drop columns:
final_df = df_dummies.drop(['hsr_intra_freq', 'hsr_inter_freq', 'traffic_volume_data', 'prb_dl_usage_rate_avg', 'prb_ul_usage_rate_avg', 'cell_throughput_ul_avg', 'user_throughput_dl_avg'], axis=1)

# %%
final_df.set_index('timestamp', inplace=True)
final_df.sort_index(inplace=True)

#Divide into features and label
X = final_df.drop(['label'], axis=1)
y = final_df['label']

#Split data into training and testing datasets
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                        random_state=1, 
                                        test_size=0.2, 
                                        shuffle=True)
'''
tss = TimeSeriesSplit(n_splits = 4)
for train_i, test_i in tss.split(X):
    X_train, X_test = X.iloc[train_i, :], X.iloc[test_i, :]
    y_train, y_test = y.iloc[train_i], y.iloc[test_i]

# %%
#Standard scale data
scale_cols = ['users_avg', 'cell_throughput_dl_avg', 'user_throughput_ul_avg',
            ]

ct = ColumnTransformer([
        ('somename', StandardScaler(), scale_cols)
    ], remainder='passthrough')

scaled_X_train = ct.fit_transform(X_train)
scaled_X_test = ct.transform(X_test)
# %%
rf = RandomForestClassifier(max_depth=4, random_state=0)
rf.fit(scaled_X_train, y_train)

y_train_pred = rf.predict(scaled_X_train)
y_val_pred = rf.predict(scaled_X_test)
# %%

print(len([1 for x in y_train if x == 1]) / len(y_train))
print(len([1 for x in y_test if x == 1]) / len(y_test))
print(f1_score(y_test, [1 for x in y_test], average='macro'))
print(confusion_matrix(y_train, [1 for x in y_train]))
#Print metrics
print('training accuracy: ', accuracy_score(y_train, y_train_pred))
print('validation accuracy: ', accuracy_score(y_test, y_val_pred))
print('training precision: ', precision_score(y_train, y_train_pred, average='macro'))
print('validation precision: ', precision_score(y_test, y_val_pred, average='macro'))
print('training recall: ', recall_score(y_train, y_train_pred, average='macro'))
print('validation recall: ', recall_score(y_test, y_val_pred, average='macro'))
print('training matrix: ', confusion_matrix(y_train, y_train_pred))
print('validation matrix: ', confusion_matrix(y_test, y_val_pred))
print('training f1: ', f1_score(y_train, y_train_pred, average='macro'))
print('validation f1: ', f1_score(y_test, y_val_pred, average='macro'))
# %%


# Attempt 2. Training with previous timesteps

# %%
df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/data/KPIs1_outer.csv')

#Remove Inf and NaN
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

df['timestamp'] = pd.to_datetime(df['datetime'])
df['ltecell_name'] = df['ltecell_name'].apply(lambda x: x[-5:])

num_cols = [
    'users_avg',
    'prb_dl_usage_rate_avg',
    'prb_ul_usage_rate_avg', 
    'hsr_intra_freq',
    'hsr_inter_freq',
    'cell_throughput_dl_avg',
    'cell_throughput_ul_avg',
    'user_throughput_dl_avg', 
    'user_throughput_ul_avg',
    'traffic_volume_data',
    'tp_carrier_aggr',
    'cqi_avg',
    'time_advance_avg'
]

cat_cols = [
    'datetime',
    #'enodeb_name',
    'ltecell_name',
]

#%%
cell_groups = df.groupby(['ltecell_name'])
dfs = [group.drop(cat_cols, axis=1).set_index('timestamp') for (name, group) in cell_groups]

cellwise_df = pd.concat(dfs, axis=1, join='outer')

cols = []
for cell in df['ltecell_name'].unique():
    for col in num_cols:
        cols.append(cell + '_' + col)

cellwise_df.columns = cols
split_date = pd.to_datetime('2023-02-13 00:00:00.000')
cellwise_df.dropna(inplace=True)
cellwise_df['N3L18_hsr_intra_freq'] = [1 if x == 100 else 0 for x in cellwise_df['N3L18_hsr_intra_freq']]

# %%
split_date = pd.to_datetime('2023-02-13 00:00:00.000')
train, test = cellwise_df.loc[cellwise_df.index < split_date], cellwise_df.loc[cellwise_df.index >= split_date]
# %%
train_feats = train.drop(['N3L18_hsr_intra_freq'], axis=1)
train_label = train['N3L18_hsr_intra_freq']
test_feats = test.drop(['N3L18_hsr_intra_freq'], axis=1)
test_label = test['N3L18_hsr_intra_freq']

ss = StandardScaler()
ss = ss.fit(train_feats)

scaled_train_feats = pd.DataFrame(ss.transform(train_feats), columns=train_feats.columns)
scaled_test_feats = pd.DataFrame(ss.transform(test_feats), columns=test_feats.columns)

print(train_feats.columns)
#train['hsr_intra_freq'] = sc.transform(train[['hsr_intra_freq']])
#test['hsr_intra_freq'] = sc.transform(test[['hsr_intra_freq']])
# %%
def window(x, y, window_size):
    x_values = []
    y_values = []

    for i in range(len(x) - window_size):
        x_values.append(x.iloc[i : (i + window_size)].values)
        y_values.append(y.iloc[(i + window_size)])

    return np.array(x_values), np.array(y_values)

#n_lags determines how many previous data points are considered
n_lags = 23

X_train, y_train = window(scaled_train_feats, train_label, n_lags)
X_test, y_test = window(scaled_test_feats, test_label, n_lags)

# %%
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

# %%
print(y_train)
# %%

rf = RandomForestClassifier(max_depth=4, random_state=0)
rf.fit(X_train, y_train)

y_train_pred = rf.predict(X_train)
y_val_pred = rf.predict(X_test)
# %%

print(len([1 for x in y_train if x == 1]) / len(y_train))
print(len([1 for x in y_test if x == 1]) / len(y_test))
print(f1_score(y_test, [1 for x in y_test], average='macro'))
print(confusion_matrix(y_train, [1 for x in y_train]))
#Print metrics
print('training accuracy: ', accuracy_score(y_train, y_train_pred))
print('validation accuracy: ', accuracy_score(y_test, y_val_pred))
print('training precision: ', precision_score(y_train, y_train_pred, average='macro'))
print('validation precision: ', precision_score(y_test, y_val_pred, average='macro'))
print('training recall: ', recall_score(y_train, y_train_pred, average='macro'))
print('validation recall: ', recall_score(y_test, y_val_pred, average='macro'))
print('training matrix: ', confusion_matrix(y_train, y_train_pred))
print('validation matrix: ', confusion_matrix(y_test, y_val_pred))
print('training f1: ', f1_score(y_train, y_train_pred, average='macro'))
print('validation f1: ', f1_score(y_test, y_val_pred, average='macro'))

# %%
