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
from tensorflow.keras import backend as K

# %%
df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/data/KPIs.csv')

#Remove Inf and NaN
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

df['timestamp'] = pd.to_datetime(df['datetime'])

cell = df[df['ltecell_name'] == 'CEIRA_MCO001N3L18']
cell = cell[['timestamp', 'hsr_intra_freq']]
print(cell)

split_date = pd.to_datetime('2023-02-10 00:00:00.000')
train, test = cell.loc[cell['timestamp'] < split_date], cell.loc[cell['timestamp'] >= split_date]

# %%
sc = StandardScaler()
sc = sc.fit(train[['hsr_intra_freq']])
train['hsr_intra_freq'] = sc.transform(train[['hsr_intra_freq']])
test['hsr_intra_freq'] = sc.transform(test[['hsr_intra_freq']])

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

X_train, y_train = window(train[['hsr_intra_freq']], train['hsr_intra_freq'], n_lags)
X_test, y_test = window(test[['hsr_intra_freq']], test['hsr_intra_freq'], n_lags)

# %%
#Custom loss function

def custom_loss(real, predicted):
    loss = 0
    for i in range(len(real)):
        if abs(real[i] - predicted[i]) > 10:
            loss += 1
    
    return loss


# %%
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

model = Sequential()
model.add(LSTM(32, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(y_train.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()

# %%

# fit the model
history = model.fit(X_train, y_train, epochs=5, batch_size=16, validation_split=0.1, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

# %%

y_pred = model.predict(X_test)

# %%
y_pred_org = sc.inverse_transform(y_pred).reshape(len(y_test))
y_test_org = sc.inverse_transform(y_test).reshape(len(y_test))
errors = np.abs(y_test_org - y_pred_org) 

test_org = pd.DataFrame({
    'time': test['timestamp'][n_lags-1 : -1],
    'hsr_intra_freq': y_test_org,
    'type': 'original',
    'error': errors
})
pred_org = pd.DataFrame({
    'time': test['timestamp'][n_lags-1 : -1],
    'hsr_intra_freq': y_pred_org,
    'type': 'predicted'
})
plot_data = pd.concat([test_org, pred_org], ignore_index=True)
# %%
sns.lineplot(x='time', y='hsr_intra_freq', data=plot_data, hue='type')
plt.title('Predicted HSR with {} hour sliding window'.format(n_lags))
print(np.mean(errors))
# %%
for i in range(0, len(y_pred_org)):
    if y_pred_org[i] - y_test_org[i] > 10:
        print(round(errors[i], 2), round(y_pred_org[i], 2), round(y_test_org[i], 2) )

# %%
print()

# %%
