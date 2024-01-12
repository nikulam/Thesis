# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import random
import os
import pickle
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

# %%
# Path to the folder containing a csv-file for each BS
folder_path = 'C:/Users/mirom/Desktop/IST/Thesis/data/New_data/KPIs'
# Ref KPI is the one used for DBSCAN clustering
ref_kpi = 'traffic_volume_data'

random.seed(42)

# Variables to store information about the anomalies etc
# Not necessary for the system, but can be displayed in the end for insights
n_sectors = 0
n_abnorms = 0
n_anoms = 0
sectors = []
abnorms = []
anoms = []
p95s = []
root_causes = []

# Loop over files in the folder
# Each file represents a base station
for i, file in enumerate(os.listdir(folder_path)):

    df = pd.read_csv(folder_path + '/{}'.format(file))
    df['timestamp'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.weekday

    # Create new features
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
        'traffic_volume_data',
        'tp_carrier_aggr',
        'cqi_avg'
    ]

    # Helper function to divide based on sectors
    def divide_sectors(cell_name):
        sector = int(cell_name[-4])
        return sector
    
    # Divide into sectors
    df['sector'] = df['ltecell_name'].apply(lambda x: divide_sectors(x))

    if file == 'VIANA_SUL_LTE_NVC103B3.csv':
        try:
            # Loop over each three sectors
            for sec in df['sector'].unique():
                n_sectors += 1
                sectors.append(file[:-4] + '_' + str(sec))
                print(sec)
                # Grouping by timestamp so that same moments in time from all cells will be summed
                sec_df = df[(df['sector'] == sec)].groupby('timestamp', as_index=False).sum(numeric_only=True)

                weekly_tss = []
                full_tss = []

                day_n = []

                # Starting from monday for visualisation purposes. Not required in the system
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
                
                while end_date < last:
                    ts = sec_df[(start_date <= sec_df['timestamp']) & (sec_df['timestamp'] < end_date)]

                    # Daily index
                    index = ts['timestamp'].dt.hour + 1
                    ts.set_index(index, inplace=True)
                    ts = ts[num_cols]

                    # Drop days misisng hours of TVD
                    if sum(~ts[ref_kpi].isna()) >= 24:
                        weekly_tss.append(ts[ref_kpi])
                        day_n.append(i)
                        lengths.append(len(ts))
                        # Append all numeric columns
                        full_tss.append(ts)

                    start_date += td
                    end_date += td
                    i += 1

                # X contains n x 24 shaped data where n is the number of full days of data in a sector
                X = pd.concat(weekly_tss, axis=1).T
                X = X.set_axis(day_n, copy=False)
                print(X.shape)

                # Divide into weekdays and weekends
                global_weekend_indices = sorted(list(range(5, X.index[-1], 7)) + list(range(6, X.index[-1], 7)))
                weekday_indices = sorted(list(set(X.index) - set(global_weekend_indices)))
                weekend_indices = sorted(list(set(X.index) - set(weekday_indices)))
                weekday_X = X.loc[weekday_indices]
                weekend_X = X.loc[weekend_indices]

                weekday_X_scaled = TimeSeriesScalerMeanVariance().fit_transform(weekday_X)
                weekend_X_scaled = TimeSeriesScalerMeanVariance().fit_transform(weekend_X)

                # DBSCAN
                labels = []
                for X_scaled in [weekday_X_scaled, weekend_X_scaled]:
                
                    nbrs = NearestNeighbors(n_neighbors = 5).fit(X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1]))
                    # Find the k-neighbors of a point
                    neigh_dist, neigh_ind = nbrs.kneighbors(X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1]))
                    # Sort the neighbor distances (lengths to points) in ascending order
                    # Axis = 0 represents sort along first axis i.e. sort along row
                    sort_neigh_dist = np.sort(neigh_dist, axis = 0)
                    k_dist = sort_neigh_dist[:, 4]
                    kneedle = KneeLocator(x = range(1, len(neigh_dist) + 1), y=k_dist, S=1.0, 
                                          curve = "concave", direction = "increasing", online=True)
                    # Get the estimate of knee point
                    knee = kneedle.knee_y
                    clusters = DBSCAN(eps = knee, min_samples=24).fit(X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1]))
                    # Get cluster labels
                    print(pd.Series(clusters.labels_).value_counts())
                    labels.append(clusters.labels_)

                # This block of code combines all labels (normal and abnormal) together and inserts missing days
                # This was purely used for visualisation purposes to plot the days of a week
                weekday_labels = list(zip(weekday_indices, labels[0]))
                weekend_labels = list(zip(weekend_indices, labels[1]))
                combined_list = sorted(weekday_labels + weekend_labels, key=lambda tuple: tuple[0])
                combined_labels = [n[1] for n in combined_list]
                labels = combined_labels.copy()
                n_abnorms += len([n for n in labels if n == -1])
                abnorms.append(len([n for n in labels if n == -1]))
                missing = list(set(range(0, max(day_n))) - set(day_n))
                for m in sorted(missing):
                    labels.insert(m, 2)


                #LSTM AE
                # Using all KPIs except ref_kpi
                # Split normal and abnormal data

                full_normal = []
                full_anom = []
                for i, l in enumerate(combined_labels):
                    if l == 0: 
                        full_normal.append(full_tss[i].drop([ref_kpi], axis=1))
                    elif l == -1:
                        full_anom.append(full_tss[i].drop([ref_kpi], axis=1))
                if (len(full_normal) == 0) or (len(full_anom) == 0):
                    print('No data')
                    anoms.append(0)
                    continue

                # Standard scaling and splitting into train, test and anom (anom = abnormal days from DBSCAN)
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
                model.add(LSTM(64, activation='relu', input_shape=(full_train_scaled.shape[1], full_train_scaled.shape[2])))
                model.add(RepeatVector(full_train_scaled.shape[1]))
                # Decoder
                model.add(LSTM(64, activation='relu', return_sequences=True))
                model.add(TimeDistributed(Dense(full_train_scaled.shape[2])))
                # Compile
                model.compile(optimizer='adam', loss='mse')
                # Fit model
                history = model.fit(full_train_scaled, full_train_scaled, epochs=30, verbose=False)

                mses_train = []
                yhat = model.predict(full_train_scaled, verbose=0)
                for i in range(len(full_train_scaled)):
                    mse = np.mean(np.square(yhat[i] - full_train_scaled[i]))
                    mses_train.append(mse)

                mses_test = []
                yhat = model.predict(full_test_scaled, verbose=0)
                for i in range(len(full_test_scaled)):
                    mse = np.mean(np.square(yhat[i] - full_test_scaled[i]))
                    mses_test.append(mse)

                mses_anom = []
                yhat = model.predict(full_anom_scaled, verbose=0)
                for i in range(len(full_anom_scaled)):
                    mse = np.mean(np.square(yhat[i] - full_anom_scaled[i]))
                    mses_anom.append(mse)


                # SET THE ANOMALY THRESHOLD TO BE THE 95TH PERCENTILE OF TRAINING ERRORS
                p_95 = np.percentile(mses_train, 95)
                p95s.append(p_95)
                n_anoms += len([n for n in mses_anom if n > p_95])
                anoms.append(len([n for n in mses_anom if n > p_95]))

                # RCA
                rca_cols = [x for x in num_cols if x != ref_kpi]
                errs = np.square(yhat - full_anom_scaled)
                daily_mses = np.round(np.mean(errs, axis=(1,2)), decimals=2)
                daily_mses_per_kpi = np.round(np.mean(errs, axis=1), decimals=2)
                mses_per_kpi = []
                rcs = []
                for i in range(errs.shape[0]):
                    if daily_mses[i] > p_95:
                        kpi_errs = daily_mses_per_kpi[i,:]
                        mean_err = np.mean(kpi_errs)
                        indices = np.where(kpi_errs > mean_err)[0]
                        selected_cols = [rca_cols[i] for i in indices]
                        rcs.extend(selected_cols)
                root_causes.append(rcs)

                for i in range(len(rca_cols)):
                    mses_per_kpi.append([np.mean(x) for x in errs[:,:,i]])

                mses_per_kpi = np.array(mses_per_kpi).T
                ind = np.arange(len(mses_per_kpi))
                labels = []
                plots = []
                sums = np.zeros(len(mses_per_kpi))
                for i in range(len(rca_cols)):
                    if i == 0:
                        label = rca_cols[i]
                        sums += mses_per_kpi[:,i]
                    else:
                        label = rca_cols[i]
                        sums += mses_per_kpi[:,i]
                    labels.append(label)

        except Exception:
            continue

print(n_anoms, n_abnorms, n_sectors)
print(anoms, abnorms, sectors)
print(root_causes)

# %%
