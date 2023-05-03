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
from keras.layers import LSTM, Dense, Dropout, RepeatVector
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import keras.backend as K


# %%
df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/data/KPIs.csv')

#Remove Inf and NaN
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

#Create time columns
df['timestamp'] = pd.to_datetime(df['datetime'])
df['hour'] = df['timestamp'].dt.hour
df['day/night'] = [1 if 7 < x else 0 for x in df['hour']]

cat_cols = [
    'datetime',
    'enodeb_name',
    'ltecell_name',
    'hour'
]

cell = df[df['ltecell_name'] == 'CEIRA_MCO001N3L18'].drop(cat_cols, axis=1)

split_date = pd.to_datetime('2022-10-30 00:00:00')
train, test = cell.loc[cell['timestamp'] < split_date], cell.loc[cell['timestamp'] >= split_date]

# %%
train_feats = train.drop(['timestamp'], axis=1)
train_label = train['hsr_intra_freq']
test_feats = test.drop(['timestamp'], axis=1)
test_label = test['hsr_intra_freq']

ss = StandardScaler()
ss = ss.fit(train_feats)

scaled_train_feats = pd.DataFrame(ss.transform(train_feats), columns=train_feats.columns)
scaled_test_feats = pd.DataFrame(ss.transform(test_feats), columns=test_feats.columns)

print(len(scaled_train_feats))
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
n_lags = 3

X_train, y_train = window(scaled_train_feats, scaled_train_feats, n_lags)
X_test, y_test = window(scaled_test_feats, scaled_test_feats, n_lags)

# %%
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(y_train.shape[1], activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1_score])
model.summary()

# %%

# fit the model
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1, verbose=1)

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
    'time': test['timestamp'][n_lags-1 : -1],
    'hsr_intra_freq': y_test.reshape(len(y_test)),
    'type': 'original'
    #'error': errors
})
pred_org = pd.DataFrame({
    'time': test['timestamp'][n_lags-1 : -1],
    'hsr_intra_freq': y_pred.reshape(len(y_pred)),
    'type': 'predicted'
})
plot_data = pd.concat([test_org, pred_org], ignore_index=True)
# %%
sns.lineplot(x='time', y='hsr_intra_freq', data=plot_data, hue='type')
plt.title('Predicted HSR with {} hour sliding window'.format(n_lags))

# %%
for i in range(0, len(y_pred_org)):
    print(round(errors[i], 2), round(y_pred_org[i], 2), round(y_test_org[i], 2) )

# %%
