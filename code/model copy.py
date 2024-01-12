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

from keras.models import load_model
import matplotlib.ticker as ticker
import pickle

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model

# %%
folder_path = 'C:/Users/mirom/Desktop/IST/Thesis/data/New_data/KPIs'
ref_kpi = 'traffic_volume_data'

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
    print(file)
    
    if file == 'VIANA_SUL_LTE_NVC103B3.csv':
    #if file != 'ALTO_DA_VELA_A28_LTE_NVC084B2.csv':
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
            'cqi_avg',
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
                n_sectors += 1
                sectors.append(file[:-4] + '_' + str(sec))
                print(sec)

                sec_df = df[(df['sector'] == sec)].groupby('timestamp', as_index=False).sum(numeric_only=True)
                cell_df = sec_df.copy()

                weekly_tss = []
                sec_weekly_tss = []
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

                    # Drop days misisng hours of TVD
                    # It could be possible also to include everything
                    # -> Missing data could reveal interesting anomalies
                    if sum(~ts[ref_kpi].isna()) >= 24 * 1:
                    #if ~ts.isnull().values.any() and ts.shape == (24, 8):
                        weekly_tss.append(ts[ref_kpi])
                        day_n.append(i)
                        lengths.append(len(ts))
                        # Append all numeric columns
                        full_tss.append(ts)

                    if sum(~sec_ts[ref_kpi].isna()) >= 24 * 1:
                    #if ~ts.isnull().values.any() and ts.shape == (24, 8):
                        sec_weekly_tss.append(sec_ts[ref_kpi])
                        sec_day_n.append(i)

                    start_date += td
                    end_date += td
                    i += 1

                X = pd.concat(weekly_tss, axis=1).T
                X = X.set_axis(day_n, copy=False)
                X_sec = pd.concat(sec_weekly_tss, axis=1).T
                X_sec = X_sec.set_axis(day_n, copy=False)
                X_full = pd.concat(full_tss)

                # Divide into weekdays and weekends
                global_weekend_indices = sorted(list(range(5, X.index[-1], 7)) + list(range(6, X.index[-1], 7)))
                weekday_indices = sorted(list(set(X.index) - set(global_weekend_indices)))
                weekend_indices = sorted(list(set(X.index) - set(weekday_indices)))
                weekday_X = X.loc[weekday_indices]
                weekend_X = X.loc[weekend_indices]

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

                weekday_avg = weekday_X.mean()
                weekend_avg = weekend_X.mean()
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

                weekday_X_scaled = TimeSeriesScalerMeanVariance().fit_transform(weekday_X)
                weekend_X_scaled = TimeSeriesScalerMeanVariance().fit_transform(weekend_X)

                # DBSCAN
                labels = []
                for X_scaled in [weekday_X_scaled, weekend_X_scaled]:
                
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
                    #plt.show()

                    kneedle = KneeLocator(x = range(1, len(neigh_dist)+1), y = k_dist, S = 1.0, 
                                          curve = "concave", direction = "increasing", online=True)

                    # get the estimate of knee point
                    knee = kneedle.knee_y
                    #print(knee)

                    # With *2 we got 0 anomalies, with *1 we got 2.
                    clusters = DBSCAN(eps = knee, min_samples=24).fit(X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1]))
                    # get cluster labels
                    print(pd.Series(clusters.labels_).value_counts())
                    labels.append(clusters.labels_)

              
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

                def mb_to_gb_formatter(x, pos):
                    return f"{x / 1000:.0f}"

                '''
                #x = 7
                x = 7
                y = int(np.ceil(len(X_pad) / 7) + 2)
                #y = 8

                fig, ax = plt.subplots(y, x, figsize=(25,25))
                #fig.suptitle('Daily traffic volume data in sector {} {}'.format(sec, file[:-4]), fontsize='xx-large')

                for i in range(y):
                    for j in range(x):
                        index = i * x + j
                        if index + 1 > len(X) + len(missing):
                            continue
                        
                        ax[i, j].plot(X_pad.loc[index], color='r')
                        ax[i, j].plot(den_pad.loc[index], color='b', linestyle='--')

                        corr = corrs[index]
                        ax[i, j].set_title('{:.2f}'.format(corr))
                        if corr < 0.4:
                            ax[i, j].set_facecolor('mistyrose')

                        #if i != y - 1:
                        #    ax[i, j].set_xticks([])
                        
                        #ax[i, j].tick_params(axis='y', labelsize=30)
                        #ax[i, j].yaxis.set_major_formatter(ticker.FuncFormatter(mb_to_gb_formatter))

                        #if i == y - 1:
                        #    ax[i, j].tick_params(axis='x', labelsize=30)
                
                weekday_names = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
                column_width = 1.0 / x
                for j in range(x):
                    xpos = (j + 0.5) * column_width  # Center of each column
                    ypos = 1.05  # Y-position above the existing title
                    fig.text(xpos, ypos, weekday_names[j], ha='center', fontsize=30, fontweight='bold')

                #ax[y-1, x].plot(X_avg, color='black')
                #ax[y-1, x].plot(denoised_avg, color='gray', linestyle='--')

                #ax[y-1, x].set_title('Average day')

                #ax[y-3,x].plot(weekday_avg)
                #ax[y-3, x].set_title('Average weekday sector')
                #ax[y-2,x].plot(weekend_avg)
                #ax[y-2, x].set_title('Average weekend sector')

                #for i in range(7):
                #    ax[y-2, i].plot(daily_avgs[i], color='green')
                #    ax[y-2, i].set_title('Average {}'.format(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][i]))


                for l in range(len(labels)):
                   ax[l // 7, l % 7].set_facecolor(['white', 'lightblue', 'mistyrose', 'seashell', 'lightgreen', 'yellow'][labels[l]])

                plt.savefig('C:/Users/mirom/Desktop/IST/Thesis/imgs/New_data/{}/sec{}.png'.format(file[:-4], sec))
                plt.clf()
                '''
                
                random.seed(42)

                # Using all KPIs except TVD
                full_normal = []
                full_anom = []

                for i, l in enumerate(combined_labels):
                    if l == 0: 
                        full_normal.append(full_tss[i].drop([ref_kpi], axis=1))

                    elif l == -1:
                        full_anom.append(full_tss[i].drop([ref_kpi], axis=1))


                if (len(full_normal) == 0) or (len(full_anom) == 0):
                    print('no data')
                    anoms.append(0)
                    continue

                print('jee')
                sc = StandardScaler()

                split = round(len(full_normal) * 0.9)

                full_train = np.array(full_normal[:split])
                full_test = np.array(full_normal[split:-1])
                full_anom = np.array(full_anom)

                full_train_scaled = sc.fit_transform(full_train.reshape(-1, full_train.shape[-1])).reshape(full_train.shape)
                for i in range(full_train_scaled.shape[2]):  # loop through the 3rd dimension (KPIs)
                    kpi_data = full_train_scaled[:, :, i]
                    var_val = np.var(kpi_data)
                    min_val = np.min(kpi_data)
                    max_val = np.max(kpi_data)

                    print(f"KPI-{i+1}:")
                    print(f"  Var: {var_val}")
                    print(f"  Min: {min_val}")
                    print(f"  Max: {max_val}")
                    print("----------")
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
                
                # To load and save the model
                #if sec == 1:
                #    model.save('model.h5')
                
                #model = load_model('model.h5')

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

                p_95 = np.percentile(mses_train, 95)
                #p_95 = 1.16
                p95s.append(p_95)
                n_anoms += len([n for n in mses_anom if n > p_95])
                anoms.append(len([n for n in mses_anom if n > p_95]))

                fig, ax = plt.subplots()
                fig.set_size_inches([6.4, 4.8])
                #ds_name = 'train'
                #ds = mses_train if ds_name == 'train' else mses_test if ds_name == 'test' else mses_anom 
                #sns.histplot(ds, label=ds_name)
               
                ds_name = 'anom'
                ds = mses_train if ds_name == 'train' else mses_test if ds_name == 'test' else mses_anom 
                #sns.histplot(ds, label=ds_name)
         
                #ds_name = 'test'
                #ds = mses_train if ds_name == 'train' else mses_test if ds_name == 'test' else mses_anom 
                #sns.histplot(ds, label=ds_name)
                #plt.title('Sec {} reconstruction error distributions in\n{}'.format(sec, file[:-4]))
                #plt.xlabel('MSE')
                #plt.legend()
                #plt.savefig('C:/Users/mirom/Desktop/IST/Thesis/imgs/New_data/{}/sec{}_recon_hist.png'.format(file[:-4], sec))
                #plt.clf()


                rca_cols = [x for x in num_cols if x != ref_kpi]

                # Combined bar plot
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
                colors = sns.color_palette("Paired")

                #fig = plt.subplots()
                labels = []
                plots = []
                sums = np.zeros(len(mses_per_kpi))
                for i in range(len(rca_cols)):
                    if i == 0:
                        #plt.bar(ind, mses_per_kpi[:,i])
                        label = rca_cols[i]
                        sums += mses_per_kpi[:,i]
                    else:
                        #plt.bar(ind, mses_per_kpi[:,i], bottom=sums)
                        label = rca_cols[i]
                        sums += mses_per_kpi[:,i]

                    labels.append(label)

                #plt.legend(labels)
                #plt.xticks(ind, daily_mses, rotation=90)
                #plt.xlabel('Daily MSE')
                #plt.ylabel('MSE per KPI')
                #plt.title('Reconstruction errors of anomalies in sector {} in\n {}'.format(sec, file[:-4]))
                #plt.savefig('C:/Users/mirom/Desktop/IST/Thesis/imgs/New_data/{}/sec{}_recon_KPI_bar'.format(file[:-4], sec))
                #plt.clf()

        except Exception:
            continue

print(n_anoms, n_abnorms, n_sectors)
# %%
print(root_causes)
mapping_dict = {
    "users_avg": 'USERS',
    "prb_dl_usage_rate_avg": 'PRB_DL',
    "prb_ul_usage_rate_avg": 'PRB_UL',
    "tp_carrier_aggr": 'CA_TP',
    "hsr_intra_freq": 'HSR_INTRA',
    "hsr_inter_freq": 'HSR_INTER',
    "cqi_avg": 'CQI'
}

flat_list = [item for sublist in root_causes for item in sublist]
flat_list = [mapping_dict[item] for item in flat_list]
ticks = ['USERS', 'PRB_DL', 'PRB_UL', 'CA_TP', 'HSR_INTRA', 'HSR_INTER', 'CQI']
df = pd.DataFrame(flat_list, columns=['value'])
df['value'] = pd.Categorical(df['value'], categories=ticks)

color_map = {tick: i for i, tick in enumerate(ticks)}
df['color'] = df['value'].map(color_map)

sns.histplot(df, x='value', hue='color', palette='tab10', bins=len(ticks), discrete=True, legend=False)

# Adjust ticks and labels
plt.xticks(ticks=range(len(ticks)), labels=ticks, rotation=90)
plt.plot()

# %%
print(n_sectors)
print(n_abnorms)
print(n_anoms)
print(sectors)
print(abnorms )
print(anoms )
print(np.percentile(p95s, 90))
# %%
#n_sectors = 70
#n_abnorms = 1061
#n_anoms = 156
#sectors = ['CARDIELOS_LTE_NVC092B2_1', 'CARDIELOS_LTE_NVC092B2_2', 'CARDIELOS_LTE_NVC092B2_3', 'CEIRA_LTE_MCO001B2_1', 'CEIRA_LTE_MCO001B2_2', 'CEIRA_LTE_MCO001B2_3', 'CHAO_PICA_LTE_NVC088B2_1', 'CHAO_PICA_LTE_NVC088B2_2', 'CHAO_PICA_LTE_NVC088B2_3', 'DARQUE_FONTINHA_LTE_NVC105B2_1', 'DARQUE_FONTINHA_LTE_NVC105B2_2', 'DARQUE_FONTINHA_LTE_NVC105B2_3', 'DARQUE_LTE_NVC106B2_1', 'DARQUE_LTE_NVC106B2_2', 'DARQUE_LTE_NVC106B2_3', 'FAROL_DE_MONTADOR_LTE_NVC090B2_1', 'FAROL_DE_MONTADOR_LTE_NVC090B2_2', 'FAROL_DE_MONTADOR_LTE_NVC090B2_3', 'FREIXEIRO_LTE_NVC086B2_1', 'FREIXEIRO_LTE_NVC086B2_2', 'FREIXEIRO_LTE_NVC086B2_3', 'MEADELA_PERRE_LTE_NVC094B2_1', 'MEADELA_PERRE_LTE_NVC094B2_2', 'MEADELA_PERRE_LTE_NVC094B2_3', 'ORBACEM_LTE_NVC085B2_1', 'ORBACEM_LTE_NVC085B2_2', 'ORBACEM_LTE_NVC085B2_3', 'OUTEIRO_A28_LTE_NVC091B2_1', 'OUTEIRO_A28_LTE_NVC091B2_2', 'OUTEIRO_A28_LTE_NVC091B2_3', 'OUTEIRO_VCA_LTE_NVC089B2_1', 'OUTEIRO_VCA_LTE_NVC089B2_2', 'PORTUZELO_CENTRO_LTE_NVC097B2_1', 'PORTUZELO_CENTRO_LTE_NVC097B2_2', 'PORTUZELO_CENTRO_LTE_NVC097B2_3', 'VIANA_CASTELO_ESTE_LTE_NVC096B2_1', 'VIANA_CASTELO_ESTE_LTE_NVC096B2_2', 'VIANA_CASTELO_ESTE_LTE_NVC096B2_3', 'VIANA_CASTELO_PORTUZELO_LTE_NVC095B2_1', 'VIANA_CASTELO_PORTUZELO_LTE_NVC095B2_2', 'VIANA_CASTELO_PORTUZELO_LTE_NVC095B2_3', 'VIANA_CENTRO_LTE_NVC100B2_1', 'VIANA_CENTRO_LTE_NVC100B2_2', 'VIANA_CENTRO_LTE_NVC100B2_3', 'VIANA_DO_CASTELO_AREOSA_LTE_NVC093B2_1', 'VIANA_DO_CASTELO_AREOSA_LTE_NVC093B2_2', 'VIANA_DO_CASTELO_CENTRO_LTE_NVC104B2_1', 'VIANA_DO_CASTELO_CENTRO_LTE_NVC104B2_2', 'VIANA_DO_CASTELO_CENTRO_LTE_NVC104B2_3', 'VIANA_DO_CASTELO_CENTRO_LTE_NVC104B3_1', 'VIANA_DO_CASTELO_CENTRO_LTE_NVC104B3_2', 'VIANA_DO_CASTELO_CENTRO_LTE_NVC104B3_3', 'VIANA_DO_CASTELO_NORTE_LTE_NVC101B2_1', 'VIANA_DO_CASTELO_NORTE_LTE_NVC101B2_2', 'VIANA_DO_CASTELO_NORTE_LTE_NVC101B2_3', 'VIANA_RUA_SARA_AFONSO_LTE_NVC099B2_1', 'VIANA_RUA_SARA_AFONSO_LTE_NVC099B2_2', 'VIANA_RUA_SARA_AFONSO_LTE_NVC099B2_3', 'VIANA_SHOPPING_LTE_NVC102B2_1', 'VIANA_SHOPPING_LTE_NVC102B2_2', 'VIANA_SHOPPING_LTE_NVC102B2_3', 'VIANA_SUL_LTE_NVC103B2_1', 'VIANA_SUL_LTE_NVC103B2_2', 'VIANA_SUL_LTE_NVC103B2_3', 'VIANA_SUL_LTE_NVC103B3_1', 'VIANA_SUL_LTE_NVC103B3_2', 'VIANA_SUL_LTE_NVC103B3_3', 'VILA_PRAIA_DE_ANCORA_LTE_NVC083B2_1', 'VILA_PRAIA_DE_ANCORA_LTE_NVC083B2_2', 'VILA_PRAIA_DE_ANCORA_LTE_NVC083B2_3']
#abnorms = [10, 10, 8, 8, 9, 7, 11, 14, 9, 15, 8, 54, 8, 10, 7, 7, 6, 6, 9, 8, 22, 9, 25, 20, 16, 8, 53, 11, 5, 58, 13, 5, 50, 53, 5, 6, 5, 7, 8, 9, 5, 9, 7, 13, 12, 9, 13, 6, 51, 12, 6, 8, 17, 12, 6, 14, 9, 30, 7, 10, 6, 52, 17, 12, 11, 15, 42, 19, 11, 8]
#anoms = [4, 5, 0, 2, 2, 0, 1, 0, 0, 2, 1, 3, 3, 1, 1, 2, 1, 0, 0, 0, 2, 5, 2, 0, 0, 1, 15, 1, 0, 10, 1, 1, 3, 12, 4, 0, 1, 1, 1, 1, 0, 4, 3, 4, 2, 1, 1, 4, 4, 3, 1, 0, 3, 1, 1, 5, 2, 4, 4, 1, 0, 3, 1, 0, 2, 1, 4, 4, 2, 2]
# %%
s = sorted(zip(anoms, sectors), key=lambda x: x[0])[::-1]
print(s)
print(np.corrcoef(anoms, abnorms))
# %%
plt.bar(np.arange(70), abnorms)
plt.bar(np.arange(70), anoms, bottom=abnorms)
plt.legend(['Abnormals', 'Anomalies'])
plt.xlabel('BS sector')
plt.ylabel('Number of days')
plt.xticks([])
#sns.histplot(abnorms)
#sns.histplot(anoms)
# %%
sectors = ['CARDIELOS_LTE_NVC092B2_1', 'CARDIELOS_LTE_NVC092B2_2', 'CARDIELOS_LTE_NVC092B2_3', 'CEIRA_LTE_MCO001B2_1', 'CEIRA_LTE_MCO001B2_2', 'CEIRA_LTE_MCO001B2_3', 'CHAO_PICA_LTE_NVC088B2_1', 'CHAO_PICA_LTE_NVC088B2_2', 'CHAO_PICA_LTE_NVC088B2_3', 'DARQUE_FONTINHA_LTE_NVC105B2_1', 'DARQUE_FONTINHA_LTE_NVC105B2_2', 'DARQUE_FONTINHA_LTE_NVC105B2_3', 'DARQUE_LTE_NVC106B2_1', 'DARQUE_LTE_NVC106B2_2', 'DARQUE_LTE_NVC106B2_3', 'FAROL_DE_MONTADOR_LTE_NVC090B2_1', 'FAROL_DE_MONTADOR_LTE_NVC090B2_2', 'FAROL_DE_MONTADOR_LTE_NVC090B2_3', 'FREIXEIRO_LTE_NVC086B2_1', 'FREIXEIRO_LTE_NVC086B2_2', 'FREIXEIRO_LTE_NVC086B2_3', 'MEADELA_PERRE_LTE_NVC094B2_1', 'MEADELA_PERRE_LTE_NVC094B2_2', 'MEADELA_PERRE_LTE_NVC094B2_3', 'ORBACEM_LTE_NVC085B2_1', 'ORBACEM_LTE_NVC085B2_2', 'ORBACEM_LTE_NVC085B2_3', 'OUTEIRO_A28_LTE_NVC091B2_1', 'OUTEIRO_A28_LTE_NVC091B2_2', 'OUTEIRO_A28_LTE_NVC091B2_3', 'OUTEIRO_VCA_LTE_NVC089B2_1', 'OUTEIRO_VCA_LTE_NVC089B2_2', 'PORTUZELO_CENTRO_LTE_NVC097B2_1', 'PORTUZELO_CENTRO_LTE_NVC097B2_2', 'PORTUZELO_CENTRO_LTE_NVC097B2_3', 'VIANA_CASTELO_ESTE_LTE_NVC096B2_1', 'VIANA_CASTELO_ESTE_LTE_NVC096B2_2', 'VIANA_CASTELO_ESTE_LTE_NVC096B2_3', 'VIANA_CASTELO_PORTUZELO_LTE_NVC095B2_1', 'VIANA_CASTELO_PORTUZELO_LTE_NVC095B2_2', 'VIANA_CASTELO_PORTUZELO_LTE_NVC095B2_3', 'VIANA_CENTRO_LTE_NVC100B2_1', 'VIANA_CENTRO_LTE_NVC100B2_2', 'VIANA_CENTRO_LTE_NVC100B2_3', 'VIANA_DO_CASTELO_AREOSA_LTE_NVC093B2_1', 'VIANA_DO_CASTELO_AREOSA_LTE_NVC093B2_2', 'VIANA_DO_CASTELO_CENTRO_LTE_NVC104B2_1', 'VIANA_DO_CASTELO_CENTRO_LTE_NVC104B2_2', 'VIANA_DO_CASTELO_CENTRO_LTE_NVC104B2_3', 'VIANA_DO_CASTELO_CENTRO_LTE_NVC104B3_1', 'VIANA_DO_CASTELO_CENTRO_LTE_NVC104B3_2', 'VIANA_DO_CASTELO_CENTRO_LTE_NVC104B3_3', 'VIANA_DO_CASTELO_NORTE_LTE_NVC101B2_1', 'VIANA_DO_CASTELO_NORTE_LTE_NVC101B2_2', 'VIANA_DO_CASTELO_NORTE_LTE_NVC101B2_3', 'VIANA_RUA_SARA_AFONSO_LTE_NVC099B2_1', 'VIANA_RUA_SARA_AFONSO_LTE_NVC099B2_2', 'VIANA_RUA_SARA_AFONSO_LTE_NVC099B2_3', 'VIANA_SHOPPING_LTE_NVC102B2_1', 'VIANA_SHOPPING_LTE_NVC102B2_2', 'VIANA_SHOPPING_LTE_NVC102B2_3', 'VIANA_SUL_LTE_NVC103B2_1', 'VIANA_SUL_LTE_NVC103B2_2', 'VIANA_SUL_LTE_NVC103B2_3', 'VIANA_SUL_LTE_NVC103B3_1', 'VIANA_SUL_LTE_NVC103B3_2', 'VIANA_SUL_LTE_NVC103B3_3', 'VILA_PRAIA_DE_ANCORA_LTE_NVC083B2_1', 'VILA_PRAIA_DE_ANCORA_LTE_NVC083B2_2', 'VILA_PRAIA_DE_ANCORA_LTE_NVC083B2_3']
abnorms = [10, 10, 8, 8, 9, 7, 11, 14, 9, 15, 8, 54, 8, 10, 7, 7, 6, 6, 9, 8, 22, 9, 25, 20, 16, 8, 53, 11, 5, 58, 13, 5, 50, 53, 5, 6, 5, 7, 8, 9, 5, 9, 7, 13, 12, 9, 13, 6, 51, 12, 6, 8, 17, 12, 6, 14, 9, 30, 7, 10, 6, 52, 17, 12, 11, 15, 42, 19, 11, 8]
anoms = [4, 4, 3, 2, 0, 1, 4, 3, 0, 3, 0, 9, 4, 1, 1, 2, 1, 0, 3, 0, 5, 3, 2, 3, 1, 1, 19, 2, 1, 14, 1, 1, 8, 9, 2, 1, 3, 1, 2, 1, 0, 4, 3, 3, 4, 1, 0, 1, 5, 4, 1, 0, 4, 1, 2, 3, 2, 4, 4, 1, 0, 4, 5, 3, 2, 3, 10, 2, 3, 5]
anoms2 = [4, 5, 0, 2, 2, 0, 1, 0, 0, 2, 1, 3, 3, 1, 1, 2, 1, 0, 0, 0, 2, 5, 2, 0, 0, 1, 15, 1, 0, 10, 1, 1, 3, 12, 4, 0, 1, 1, 1, 1, 0, 4, 3, 4, 2, 1, 1, 4, 4, 3, 1, 0, 3, 1, 1, 5, 2, 4, 4, 1, 0, 3, 1, 0, 2, 1, 4, 4, 2, 2]

n_same = 0
for x, y in zip(anoms, anoms2):
    if x == y:
        n_same += 1

print(len(sectors))
# %%
sns.histplot(anoms)
plt.show()
print(abnorms.index(max(abnorms)))
print(len(abnorms), len(anoms))
# %%
sec_1s = [n for n in sectors if int(n[-1]) == 1]
indices = [sectors.index(n) for n in sec_1s]
plt.bar(np.arange(70), abnorms)
plt.bar(np.arange(70), anoms, bottom=abnorms)
plt.legend(['Abnormals', 'Anomalies'])
#plt.xlabel('BS sector')
plt.ylabel('Number of days')
plt.xticks(ticks = indices, labels=[n[:-2] for n in sec_1s], rotation = 'vertical')
#sns.histplot(abnorms)
#sns.histplot(anoms)

# %%
print(sum(anoms))

# %%
