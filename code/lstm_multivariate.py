# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import *
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import keras.backend as K


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
print(cellwise_df.isnull().values.any())
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
#Custom f1-score function
def f1_score(y_true, y_pred):
    """Compute the F1 score of the predictions."""
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1_score = 2 * precision * recall / (precision + recall + K.epsilon())
    return f1_score

# %%
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(y_train.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()

# %%

# fit the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

# %%

y_pred = model.predict(X_test)

#for i in range(len(y_pred)):
#    print(y_pred[i], y_test[i])

# %%
#y_pred_org = ss.inverse_transform(y_pred).reshape(len(y_test))
#y_test_org = ss.inverse_transform(y_test).reshape(len(y_test))
#errors = np.abs(y_test_org - y_pred_org) 

test_org = pd.DataFrame({
    'time': test.index[n_lags-1 : -1],
    'hsr_intra_freq': y_test.reshape(len(y_test)),
    'type': 'original'
    #'error': errors
})
pred_org = pd.DataFrame({
    'time': test.index[n_lags-1 : -1],
    'hsr_intra_freq': y_pred.reshape(len(y_pred)),
    'type': 'predicted'
})
plot_data = pd.concat([test_org, pred_org], ignore_index=True)
# %%
sns.lineplot(x='time', y='hsr_intra_freq', data=plot_data, hue='type')
plt.title('Predicted HSR with {} hour sliding window'.format(n_lags))

# %%
errors = []
for i in range(0, len(y_pred)):
    error = abs(y_pred[i] - y_test[i])
    print(error, y_pred[i], y_test[i])
    errors.append(error)

# %%

print(np.mean(errors))

# %%
