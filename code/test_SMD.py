# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model

# %%
# Specify the path to the .npy file
train_npy_file_path = 'C:/Users/mirom/Desktop/IST/Thesis/Temp/OmniAnomaly/data/train/M-2.npy'
test_npy_file_path = 'C:/Users/mirom/Desktop/IST/Thesis/Temp/OmniAnomaly/data/test/M-2.npy'
'''
folder_path =  'C:/Users/mirom/Desktop/IST/Thesis/Temp/OmniAnomaly/data/test'
for i, file in enumerate(os.listdir(folder_path)):
    data = np.load(folder_path + '/' + file)
    print(data.shape)
'''
# Specify the path for the output .csv file
csv_file_path = 'C:/Users/mirom/Desktop/IST/Thesis/Temp/A-1.csv'

# Load the .npy file using np.load()
train_data = np.load(train_npy_file_path)
test_data = np.load(test_npy_file_path)
# Save the data as a .csv file using np.savetxt()
#np.savetxt(csv_file_path, data, delimiter=',')

# %%
train_df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/Temp/OmniAnomaly/ServerMachineDataset/train/machine-1-1.txt', delimiter=',', header=None)
test_df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/Temp/OmniAnomaly/ServerMachineDataset/test/machine-1-1.txt', delimiter=',', header=None)
date_range_train = pd.date_range(start='2023-07-01', periods=len(train_df), freq='T')
date_range_test = pd.date_range(start='2023-07-01', periods=len(test_df), freq='T')

train_df.index = date_range_train
test_df.index = date_range_test

# %%
train_df = train_df.resample('h').mean()
test_df = test_df.resample('h').mean()

first = train_df.index[0]
last = train_df.index[-1]

start_date = first
td = pd.Timedelta(1, 'D')
end_date = first + td

train = []
while end_date < last:
    ts = train_df[(start_date <= train_df.index) & (train_df.index < end_date)]
    # Daily index
    index = ts.index.hour + 1
    ts.set_index(index, inplace=True)
    train.append(ts)
    start_date += td
    end_date += td

first = test_df.index[0]
last = test_df.index[-1]

start_date = first
td = pd.Timedelta(1, 'D')
end_date = first + td

test = []
while end_date < last:
    ts = test_df[(start_date <= test_df.index) & (test_df.index < end_date)]
    # Daily index
    index = ts.index.hour + 1
    ts.set_index(index, inplace=True)
    test.append(ts)
    start_date += td
    end_date += td


# %%
train = np.array(train)
test = np.array(test)

print(train.shape, test.shape)
# %%
model = Sequential()
# Encoder
model.add(LSTM(128, activation='relu', input_shape=(train.shape[1], train.shape[2]), return_sequences=True))
model.add(LSTM(64, activation='relu', input_shape=(train.shape[1], train.shape[2])))
#model.add(LSTM(64, activation='relu', input_shape=(24,1)))

#model.add(RepeatVector(X_train.shape[1]))
model.add(RepeatVector(train.shape[1]))
# Decoder
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))

model.add(TimeDistributed(Dense(train.shape[2])))
#model.add(TimeDistributed(Dense(1)))

print(model.summary())
model.compile(optimizer='adam', loss='mse')
# fit model
history = model.fit(train, train, epochs=20, verbose=True)
plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')

plt.plot(history.history['loss'], label='Training loss')
plt.legend()
plt.show()

# %%
mses_train = []
yhat = model.predict(train, verbose=0)
for i in range(len(train)):
    #mse = np.mean(np.square(yhat[i].flatten() - X_train[i]))
    mse = np.mean(np.square(yhat[i] - train[i]))
    mses_train.append(mse)

mses_test = []
yhat = model.predict(test, verbose=0)
for i in range(len(test)):
    #mse = np.mean(np.square(yhat[i].flatten() - X_train[i]))
    mse = np.mean(np.square(yhat[i] - test[i]))
    mses_test.append(mse)

# %%
p = 10
print('Train: ', np.percentile(mses_train, p))
print('Test: ', np.percentile(mses_test, p))
# %%

ds_name = 'train'
ds = mses_train if ds_name == 'train' else mses_test 
sns.histplot(ds, label=ds_name)

ds_name = 'test'
ds = mses_train if ds_name == 'train' else mses_test
sns.histplot(ds, label=ds_name)
plt.xlabel('MSE')
plt.legend()
#plt.savefig('C:/Users/mirom/Desktop/IST/Thesis/imgs/New_data/{}_sec{}_recon_hist.png'.format(file[:-4], sec))
plt.plot()

# %%
plt.plot(test[4][:,6])
# %%

x = 7
y = int(np.ceil(len(train) / 7))

fig, ax = plt.subplots(y, x+1, figsize=(25,25))
#fig.subplots_adjust(hspace=1.5, wspace=0.3)

for i in range(y):
    for j in range(x):
        index = i * x + j
        #if index in day_n:
        print(index)
        ax[i, j].plot(train[index][:,0], color='r')
        
plt.show()

# %%
