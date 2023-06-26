# %%
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import random
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model

folder_path = 'C:/Users/mirom/Desktop/IST/Thesis/data/New_data/KPIs'
ref_kpi = 'traffic_volume_data'

# Loop over files in the folder
# Each file represents a base station
for i, file in enumerate(os.listdir(folder_path)):
    print(file)
    
    if i > 7:
        os.mkdir('C:/Users/mirom/Desktop/IST/Thesis/imgs/New_data/{}'.format(file[:-4]))
        df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/data/New_data/KPIs/{}'.format(file))
        df['timestamp'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.weekday
        #Create new features
        df['daytime'] = [1 if (8 <= x) and (x < 20) else 0 for x in df['hour']]
        df['workday'] = [0 if 4 < x else 1 for x in df['day']]
        cat_cols = [
            'timestamp', 'datetime', 'ltecell_name'
        ]
        num_cols = [
            'users_avg',
            'prb_dl_usage_rate_avg',
            'prb_ul_usage_rate_avg', 
            'hsr_intra_freq',
            'hsr_inter_freq',
            #'cell_throughput_dl_avg',
            #'cell_throughput_ul_avg',
            #'user_throughput_dl_avg', 
            #'user_throughput_ul_avg',
            'traffic_volume_data',
            'tp_carrier_aggr',
            #'carrier_aggr_usage',
            #'cqi_avg',
            #'time_advance_avg'
        ]
        # Helper function to divide based on sectors
        def divide_sectors(cell_name):
            sector = int(cell_name[-4])
            return sector
        # Divide into sectors
        df['sector'] = df['ltecell_name'].apply(lambda x: divide_sectors(x))

        try:
            # Loop over each three sectors
            for sec in df['sector'].unique():  
                print(sec)

                sec_df = df[(df['sector'] == sec)].groupby('timestamp', as_index=False).sum(numeric_only=True)
                cell_df = sec_df.copy()

                weekly_tss = []
                sec_weekly_tss = []
                pca_tss = []
                full_tss = []

                day_n = []
                sec_day_n = []
                # The last timestamp of the data is faulty -> drop it
                cell_df.drop(cell_df.tail(1).index,inplace=True)
                sec_df.drop(sec_df.tail(1).index,inplace=True)

                mondays_df = sec_df[sec_df['timestamp'].dt.dayofweek == 0]
                first_row = mondays_df.loc[sec_df[ref_kpi] != 0].head(1)
                first = first_row['timestamp'].values[0]
                last_row = sec_df.loc[sec_df[ref_kpi] != 0].tail(1)
                last = last_row['timestamp'].values[0]

                start_date = first
                td = pd.Timedelta(1, 'D')
                end_date = first + td
                lengths = []
                i = 0
                pca = PCA(n_components=1, random_state=42)
                sc = StandardScaler()
                while end_date < last:
                    ts = cell_df[(start_date <= cell_df['timestamp']) & (cell_df['timestamp'] < end_date)]
                    sec_ts = sec_df[(start_date <= sec_df['timestamp']) & (sec_df['timestamp'] < end_date)]

                    # Daily index
                    index = ts['timestamp'].dt.hour + 1
                    ts.set_index(index, inplace=True)
                    sec_ts.set_index(index, inplace=True)

                    ts = ts[num_cols]
                    sec_ts = sec_ts[num_cols]

                    # Drop weeks missing more than 10% of hours
                    if sum(~ts[ref_kpi].isna()) >= 24 * 1:
                        weekly_tss.append(ts[ref_kpi])
                        day_n.append(i)
                        lengths.append(len(ts))
                        # Append all numeric columns
                        full_tss.append(ts)

                    if sum(~sec_ts[ref_kpi].isna()) >= 24 * 1:
                        sec_weekly_tss.append(sec_ts[ref_kpi])
                        sec_day_n.append(i)

                    start_date += td
                    end_date += td
                    i += 1

                X = pd.concat(weekly_tss, axis=1).T
                X = X.set_axis(day_n, copy=False)
                #X.set_axis(day_n, inplace=True)
                X_sec = pd.concat(sec_weekly_tss, axis=1).T
                X_sec = X_sec.set_axis(day_n, copy=False)
                #X_sec.set_axis(sec_day_n, inplace=True)
                X_full = pd.concat(full_tss)

                X_avg = X.mean()

                # Calculate averages for each weekday
                # Use loc instead of iloc because X is missing some weeks (ie. the indices 39-44)
                daily_avgs = []
                for i in range(7):
                    indices = list(set(day_n) & set(range(i, len(X), 7)))
                    daily_avgs.insert(i, X.loc[indices].mean())

                sec_daily_avgs = []
                for i in range(7):
                    indices = list(set(sec_day_n) & set(range(i, len(X_sec), 7)))
                    sec_daily_avgs.insert(i, X_sec.loc[indices].mean())

                weekday_avg = (pd.concat(sec_daily_avgs[0:5], axis=1).mean(axis=1))
                weekend_avg = (pd.concat(sec_daily_avgs[5:7], axis=1).mean(axis=1))
                weekday_avg_denoised = weekday_avg.rolling(3, center=True).mean()
                weekend_avg_denoised = weekend_avg.rolling(3, center=True).mean()
                weekday_avg_denoised.dropna(inplace=True)
                weekend_avg_denoised.dropna(inplace=True)

                #Interpolate missing values
                for i in range(len(X)):
                    X.iloc[i] = X.iloc[i].interpolate()

                #Noise reduction
                denoised = X.copy()
                for i in range(len(denoised)):
                    denoised.iloc[i] = denoised.iloc[i].rolling(3, center=True).mean()

                denoised_avg = X_avg.rolling(3, center=True).mean()

                # Drop nan values
                denoised.dropna(inplace=True, axis=1)

                X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)
                denoised_scaled = TimeSeriesScalerMeanVariance().fit_transform(denoised)
                X_avg_scaled = TimeSeriesScalerMeanVariance().fit_transform(X_avg.values.reshape(1,-1))

                # DBSCAN
                nbrs = NearestNeighbors(n_neighbors = 5).fit(X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1]))
                # Find the k-neighbors of a point
                neigh_dist, neigh_ind = nbrs.kneighbors(X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1]))
                # sort the neighbor distances (lengths to points) in ascending order
                # axis = 0 represents sort along first axis i.e. sort along row
                sort_neigh_dist = np.sort(neigh_dist, axis = 0)

                k_dist = sort_neigh_dist[:, 4]
                #plt.plot(k_dist)
                #plt.ylabel("k-NN distance")
                #plt.xlabel("Sorted observations (5th NN)")

                kneedle = KneeLocator(x = range(1, len(neigh_dist)+1), y = k_dist, S = 1.0, 
                                      curve = "concave", direction = "increasing", online=True)

                # get the estimate of knee point
                knee = kneedle.knee_y

                clusters = DBSCAN(eps = knee, min_samples=2*24).fit(X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1]))
                # get cluster labels
                labels = list(clusters.labels_)

                missing = list(set(range(0, max(day_n))) - set(day_n))
                for m in sorted(missing):
                    labels.insert(m, 2)

                X_pad = X.copy()
                den_pad = denoised.copy()
                full_tss_pad = full_tss.copy()
                for m in sorted(missing)[::-1]:
                    X_pad.loc[m] = np.nan
                    den_pad.loc[m] = np.nan
                    full_tss_pad.insert(m, np.nan)

                X_pad = X_pad.sort_index()
                den_pad = den_pad.sort_index()

                weekend_indices = sorted(list(range(5, len(X_pad), 7)) + list(range(6, len(X_pad), 7)))

                corrs = []
                for i in range(len(X_pad)):
                    if i in weekend_indices:
                        c = (weekend_avg.corr(X_pad.iloc[i]))
                    else:
                        c = (weekday_avg.corr(X_pad.iloc[i]))
                    corrs.append(c)


                x = 7
                y = int(np.ceil(len(X_pad) / 7) + 2)

                fig, ax = plt.subplots(y, x+1, figsize=(25,25))
                fig.suptitle('Daily traffic volume data in sector {} {}'.format(sec, file[:-4]), fontsize='xx-large')

                for i in range(y):
                    for j in range(x):
                        index = i * x + j
                        if index + 1 > len(X) + len(missing):
                            continue
                        
                        ax[i, j].plot(X_pad.loc[index], color='r')
                        ax[i, j].plot(den_pad.loc[index], color='b', linestyle='--')

                        corr = corrs[index]
                        ax[i, j].set_title('{:.2f}'.format(corr))
                        if corr < 0.7:
                            ax[i, j].set_facecolor('mistyrose')


                ax[y-1, x].plot(X_avg, color='black')
                ax[y-1, x].plot(denoised_avg, color='gray', linestyle='--')

                ax[y-1, x].set_title('Average day')

                ax[y-3,x].plot(weekday_avg)
                ax[y-3, x].set_title('Average weekday sector')
                ax[y-2,x].plot(weekend_avg)
                ax[y-2, x].set_title('Average weekend sector')

                for i in range(7):
                    ax[y-2, i].plot(daily_avgs[i], color='green')
                    ax[y-2, i].set_title('Average {}'.format(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][i]))


                for l in range(len(labels)):
                    ax[l // 7, l % 7].set_facecolor(['white', 'lightblue', 'mistyrose', 'seashell', 'lightgreen', 'yellow'][labels[l]])

                plt.savefig('C:/Users/mirom/Desktop/IST/Thesis/imgs/New_data/{}/sec{}.png'.format(file[:-4], sec))
                plt.clf()

                random.seed(42)

                # Using all KPIs except TVD
                full_normal = []
                full_anom = []

                for i, l in enumerate(clusters.labels_):
                    if l == 0: 
                        full_normal.append(full_tss[i].drop([ref_kpi], axis=1))

                    elif l == -1:
                        full_anom.append(full_tss[i].drop([ref_kpi], axis=1))


                if (len(full_normal) == 0) or (len(full_anom) == 0):
                    print('no data')
                    continue

                sc = StandardScaler()

                split = round(len(full_normal) * 0.9)

                full_train = np.array(full_normal[:split])
                full_test = np.array(full_normal[split:-1])
                full_anom = np.array(full_anom)

                full_train_scaled = sc.fit_transform(full_train.reshape(-1, full_train.shape[-1])).reshape(full_train.shape)
                full_test_scaled = sc.transform(full_test.reshape(-1, full_test.shape[-1])).reshape(full_test.shape)
                full_anom_scaled = sc.transform(full_anom.reshape(-1, full_anom.shape[-1])).reshape(full_anom.shape)

                model = Sequential()
                # Encoder
                #model.add(LSTM(128, activation='relu', input_shape=(full_train_scaled.shape[1], full_train_scaled.shape[2]), return_sequences=True))
                model.add(LSTM(64, activation='relu', input_shape=(full_train_scaled.shape[1], full_train_scaled.shape[2])))
                #model.add(LSTM(64, activation='relu', input_shape=(24,1)))

                #model.add(RepeatVector(X_train.shape[1]))
                model.add(RepeatVector(full_train_scaled.shape[1]))

                # Decoder
                model.add(LSTM(64, activation='relu', return_sequences=True))
                #model.add(LSTM(128, activation='relu', return_sequences=True))

                model.add(TimeDistributed(Dense(full_train_scaled.shape[2])))
                #model.add(TimeDistributed(Dense(1)))

                #print(model.summary())
                model.compile(optimizer='adam', loss='mse')
                # fit model
                history = model.fit(full_train_scaled, full_train_scaled, epochs=30, verbose=False)
                #plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')

                #plt.plot(history.history['loss'], label='Training loss')
                #plt.legend()
                #plt.plot()

                mses_train = []
                yhat = model.predict(full_train_scaled, verbose=0)
                for i in range(len(full_train_scaled)):
                    #mse = np.mean(np.square(yhat[i].flatten() - X_train[i]))
                    mse = np.mean(np.square(yhat[i] - full_train_scaled[i]))
                    mses_train.append(mse)

                mses_test = []
                yhat = model.predict(full_test_scaled, verbose=0)
                for i in range(len(full_test_scaled)):
                    #mse = np.mean(np.square(yhat[i].flatten() - X_train[i]))
                    mse = np.mean(np.square(yhat[i] - full_test_scaled[i]))
                    mses_test.append(mse)

                mses_anom = []
                yhat = model.predict(full_anom_scaled, verbose=0)
                for i in range(len(full_anom_scaled)):
                    #mse = np.mean(np.square(yhat[i].flatten() - X_train[i]))
                    mse = np.mean(np.square(yhat[i] - full_anom_scaled[i]))
                    mses_anom.append(mse)

                p = 100
                #print('Train: ', np.percentile(mses_train, p))
                #print('Test: ', np.percentile(mses_test, p))
                #print('Anom: ', np.percentile(mses_anom, p))


                fig, ax = plt.subplots()
                fig.set_size_inches([6.4, 4.8])
                ds_name = 'train'
                ds = mses_train if ds_name == 'train' else mses_test if ds_name == 'test' else mses_anom 
                sns.histplot(ds, label=ds_name)

                ds_name = 'anom'
                ds = mses_train if ds_name == 'train' else mses_test if ds_name == 'test' else mses_anom 
                sns.histplot(ds, label=ds_name)

                ds_name = 'test'
                ds = mses_train if ds_name == 'train' else mses_test if ds_name == 'test' else mses_anom 
                sns.histplot(ds, label=ds_name)
                plt.title('Sec {} reconstruction error distributions in\n{}'.format(sec, file[:-4]))
                plt.xlabel('MSE')
                plt.legend()
                plt.savefig('C:/Users/mirom/Desktop/IST/Thesis/imgs/New_data/{}/sec{}_recon_hist.png'.format(file[:-4], sec))
                plt.clf()


                rca_cols = [x for x in num_cols if x != ref_kpi]

                # Combined bar plot
                errs = np.square(yhat - full_anom_scaled)
                daily_mses = np.round(np.mean(errs, axis=(1,2)), decimals=2)
                mses_per_kpi = []
                for i in range(len(rca_cols)):
                    mses_per_kpi.append([np.mean(x) for x in errs[:,:,i]])

                mses_per_kpi = np.array(mses_per_kpi).T
                ind = np.arange(len(mses_per_kpi))
                colors = sns.color_palette("Paired")

                fig = plt.subplots()
                labels = []
                plots = []
                sums = np.zeros(len(mses_per_kpi))
                for i in range(len(rca_cols)):
                    if i == 0:
                        plt.bar(ind, mses_per_kpi[:,i])
                        label = rca_cols[i]
                        sums += mses_per_kpi[:,i]
                    else:
                        plt.bar(ind, mses_per_kpi[:,i], bottom=sums)
                        label = rca_cols[i]
                        sums += mses_per_kpi[:,i]

                    labels.append(label)

                plt.legend(labels)
                plt.xticks(ind, daily_mses)
                plt.xlabel('Daily MSE')
                plt.ylabel('MSE per KPI')
                plt.title('Reconstruction errors of anomalies in sector {} in\n {}'.format(sec, file[:-4]))
                plt.savefig('C:/Users/mirom/Desktop/IST/Thesis/imgs/New_data/{}/sec{}_recon_KPI_bar'.format(file[:-4], sec))
                plt.clf()
    
        except Exception:
            continue
# %%    
