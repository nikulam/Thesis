# %%
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler, TimeSeriesScalerMinMax
from scipy.stats import chi2_contingency
from scipy.stats import ks_2samp
from scipy.stats import kstest
from sklearn.metrics import silhouette_score
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform, pdist
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from sklearn.decomposition import PCA
import random
import cv2

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
import keras
# %%
folder_path = 'C:/Users/mirom/Desktop/IST/Thesis/data/New_data/KPIs'

# Loop over files in the folder
file = os.listdir(folder_path)[2]
#df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/data/KPIs1_outer2.csv')
df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/data/New_data/KPIs/VIANA_SUL_LTE_NVC103B3.csv')
#Remove nulls and infs
#df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
#Create time columns
df['timestamp'] = pd.to_datetime(df['datetime'])
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.weekday
#Create new features
df['daytime'] = [1 if (8 <= x) and (x < 20) else 0 for x in df['hour']]
df['workday'] = [0 if 4 < x else 1 for x in df['day']]

cat_cols = [
    'timestamp', 'datetime', 'ltecell_name'
]

#Use mean and standard deviation to verify and remove outliers
'''
threshold = 5
outlier_columns = ['user_throughput_dl_avg']
for c in outlier_columns:
    mean1 = np.mean(df[c])
    std1 = np.std(df[c])
    outliers = []
    for n in df[c]:
            z_score= (n - mean1) / std1 
            if np.abs(z_score) > threshold:
                outliers.append(n)
    for o in outliers:
        print(o)
        df = df[df[c] != o]
'''

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
# %%
print(~df.iloc[2732].isnull().values.any())

# %%

w = 'weekday'
workday = 1 if w == 'weekday' else 0
sec = 1
# & (df['workday'] == workday)
sec_df = df[(df['sector'] == sec)]
sec_df.dropna(axis=0, inplace=True)
sec_df = sec_df.groupby('timestamp', as_index=False).sum()
#sec_df = df[df['sector'] == sec].groupby('timestamp', as_index=False).sum()
# 1. WEEKLY TIMESERIES
# Split into weekly timeseries
#cell = 'CEIRA_MCO001N2L08'
#cell_df = df[df['ltecell_name'] == cell]
cell_df = sec_df.copy()

weekly_tss = []
sec_weekly_tss = []
pca_tss = []
full_tss = []

day_n = []
sec_day_n = []
# The last timestamp of the data is faulty -> drop it
cell_df = cell_df[~cell_df['timestamp'].duplicated(keep='first')]
sec_df = sec_df[~sec_df['timestamp'].duplicated(keep='first')]

mondays_df = sec_df[sec_df['timestamp'].dt.dayofweek == 0]
first_row = mondays_df.loc[~sec_df.isnull().any(axis=1)].head(1)
first = first_row['timestamp'].values[0]
last_row = sec_df.loc[~sec_df.isnull().any(axis=1)].tail(1)
last = last_row['timestamp'].values[0]
#first = pd.to_datetime('2022-09-19 00:00:00')
#last = pd.to_datetime('2023-03-13 23:00:00')

#df_no_duplicates.to_csv('C:/Users/mirom/Desktop/IST/Thesis/data/no_duplicates.csv')

print(first, last)
print(first)
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
    # Weekly index
    #index = ((ts['timestamp'].dt.weekday * 24) + (ts['timestamp'].dt.hour + 1))
    # Daily index
    index = ts['timestamp'].dt.hour + 1
    ts.set_index(index, inplace=True)
    sec_ts.set_index(index, inplace=True)
    #ts.reset_index(inplace=True)
    #ts = ts['cell_throughput_dl_avg']
    ###
    #weekly_tss.append(ts['user_throughput_dl_avg'])
   

    ts = ts[num_cols]
    sec_ts = sec_ts[num_cols]
    
    # Drop weeks missing more than 10% of hours
    if ~ts.isnull().values.any() and ts.shape == (24, 8):
        print(ts.shape)
        #weekly_tss.append(pd.DataFrame(pca.fit_transform(sc.fit_transform(ts)).reshape(len(ts))))
        weekly_tss.append(ts['traffic_volume_data'])
        #print(pca.explained_variance_ratio_)
        day_n.append(i)
        lengths.append(len(ts))
        # Append all numeric columns
        full_tss.append(ts)
        #pca_tss.append(pd.DataFrame(pca.fit_transform(ts).reshape(len(ts))))
    if ~ts.isnull().values.any() and ts.shape == (24, 8):
        #sec_weekly_tss.append(pd.DataFrame(pca.fit_transform(sc.fit_transform(ts)).reshape(len(ts))))
        sec_weekly_tss.append(sec_ts['traffic_volume_data'])
        sec_day_n.append(i)


    start_date += td
    end_date += td
    i += 1

X = pd.concat(weekly_tss, axis=1).T
X.set_axis(day_n, inplace=True)
X_sec = pd.concat(sec_weekly_tss, axis=1).T
X_sec.set_axis(sec_day_n, inplace=True)
X_full = pd.concat(full_tss)

global_weekend_indices = sorted(list(range(5, X.index[-1], 7)) + list(range(6, X.index[-1], 7)))
weekday_indices = sorted(list(set(X.index) - set(global_weekend_indices)))
weekend_indices = sorted(list(set(X.index) - set(weekday_indices)))
weekday_X = X.loc[weekday_indices]
weekend_X = X.loc[weekend_indices]
#X_full.reset_index(inplace=True)

#X.fillna(value=0,inplace=True)
X_avg = X.mean()
#sec_avgs.append(X_avg)

#Calculate averages for each weekday
'''
daily_avgs = []
for i in range(7):
    daily_avgs.append(X.iloc[i::7].mean())
weekly_avgs = []
for i in range(25):
    weekly_avgs.append(X.iloc[i*7:i*7+7].mean())
'''
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
'''
weekly_avgs = []
for i in range(25):
    indices = list(set(day_n) & set(range(i, len(X), 25)))
    weekly_avgs.insert(i, X.iloc[indices].mean())
'''

#Interpolate missing values
for i in range(len(X)):
    X.iloc[i] = X.iloc[i].interpolate()

#Noise reduction
denoised = X.copy()
for i in range(len(denoised)):
    denoised.iloc[i] = denoised.iloc[i].rolling(3, center=True).mean()

denoised_avg = X_avg.rolling(3, center=True).mean()

X_diffs = []
for i in range(len(X) - 1):
    X_diffs.append((X.iloc[i] - X_avg))
X_diffs = pd.DataFrame(X_diffs)

X_change = X.diff(axis=1)

corrs_X = []
corrs_denoised = []
for i in X.index:
    if i in weekday_indices:
        corrs_X.append(weekday_avg.corr(X.loc[i]))
        corrs_denoised.append(weekday_avg_denoised.corr(denoised.loc[i]))
    else:
        corrs_X.append(weekend_avg.corr(X.loc[i]))
        corrs_denoised.append(weekend_avg_denoised.corr(denoised.loc[i]))


# Categorize
'''
X_avg_bins = pd.qcut(X_avg, 10, labels=list(range(1,11)))
X_bins = X.copy()
for i in range(len(X)):
    X_bins.iloc[i] = pd.qcut(X_bins.iloc[i], 10, labels=list(range(1,11)))


chi2 = []
for i in range(len(X)):
    stat, p, dof, expected = chi2_contingency([X_avg_bins.values, X_bins.iloc[i].values])
    chi2.append(p)
    print(p)
'''

# Drop nan values for Kshape clustering
denoised.dropna(inplace=True, axis=1)

scaled = False

#sns.histplot(corrs_denoised, bins=10)
#plt.title('Denoised correlation distribution cell {}'.format(cell))
#plt.xlabel('Correlation')

# %%

print((X.index) % 24)
X_full1 = X_full.copy()
print(len(X_full1))
for i in range(len(X_full1)):
    X_full1.iloc[i] = X_full1.iloc[i].interpolate()
print(len(X_full1))
# %%
pca = PCA(n_components=1, random_state=0)
pcas = []
for ts in full_tss:
    pcas.append(pca.fit_transform(ts))

print(pd.concat(pcas, axis=1))
#print(pca.fit_transform(full_tss[0]).shape)
#pca.explained_variance_ratio_

# %%
sc = StandardScaler()
print(np.array(full_tss).shape)
full_jee = np.array(full_tss)

X_scaled = sc.fit_transform(full_jee.reshape(-1, full_jee.shape[-1])).reshape(full_jee.shape)
print(X_scaled.shape)


print(weekend_X)
# %%
print(X_full.shape)
print(sum(~X_full.isnull()))
# %%
weekday_X_scaled = TimeSeriesScalerMeanVariance().fit_transform(weekday_X)
weekend_X_scaled = TimeSeriesScalerMeanVariance().fit_transform(weekend_X)
#weekday_X = weekday_X.to_numpy().reshape(weekday_X.shape[0], weekday_X.shape[1], 1)
#weekend_X = weekend_X.to_numpy().reshape(weekend_X.shape[0], weekend_X.shape[1], 1)
print(weekday_X_scaled.min().min())
print(weekend_X.min().min())
# %%
X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)
denoised_scaled = TimeSeriesScalerMeanVariance().fit_transform(denoised)
#X_test = sc.fit_transform(X.to_numpy().reshape(162*24, 1))
#X_scaled = X_test.reshape(162, 24, 1)
#denoised_scaled = sc.fit_transform(denoised.to_numpy().reshape(162*22,1)).reshape(162,22,1)
#X_diffs = TimeSeriesScalerMeanVariance().fit_transform(X_diffs)
#X_change = TimeSeriesScalerMeanVariance().fit_transform(X_change)
X_avg_scaled = TimeSeriesScalerMeanVariance().fit_transform(X_avg.values.reshape(1,-1))

# %%

for i, r in enumerate(range(len(weekday_X_scaled))):
    if np.isnan(weekday_X_scaled[i]).sum() > 0:
        print(i)

print(weekday_X_scaled[75])
# %%
# Time series clustering
init_clusters = np.array([weekday_avg_denoised, weekend_avg_denoised]).reshape(2,22,1)
#km = TimeSeriesKMeans(n_clusters=4, metric="euclidean", n_init=5, random_state=0)
#km = KernelKMeans(n_clusters=3, kernel="gak", n_init=5, random_state=0)
km =  KShape(n_clusters=3, verbose=True, n_init=5, random_state=0)
labels = km.fit_predict(denoised)
labels = list(labels)
print(pd.Series(labels).value_counts())
print(silhouette_score(X, labels))

# %%
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

    kneedle = KneeLocator(x = range(1, len(neigh_dist)+1), y = k_dist, S = 1.0, 
                          curve = "concave", direction = "increasing", online=True)

    # get the estimate of knee point
    #font = {'size': 15}
    #plt.rc('font', **font)
    knee = kneedle.knee_y
    print(knee)
    plt.plot(k_dist)
    plt.ylabel("k-NN distance")
    plt.xlabel("Sorted observations (5th NN)")
    plt.plot(list(k_dist).index(knee), knee, color='red', marker='o', markersize=10)
    plt.show()

    # With *2 we got 0 anomalies, with *1 we got 2.
    clusters = DBSCAN(eps = knee, min_samples=24).fit(X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1]))
    # get cluster labels
    print(pd.Series(clusters.labels_).value_counts())
    labels.append(clusters.labels_)

# %%
weekday_labels = list(zip(weekday_indices, labels[0]))
weekend_labels = list(zip(weekend_indices, labels[1]))
combined_list = sorted(weekday_labels + weekend_labels, key=lambda tuple: tuple[0])
combined_labels = [n[1] for n in combined_list]
print((weekday_labels))
normal_weekdays = np.array([weekday_X[i] for i, n in enumerate(labels[0]) if n == 0])
normal_weekends = np.array([weekend_X[i] for i, n in enumerate(labels[1]) if n == 0])
# %%
print(normal_weekends.shape)
avg_weekday = np.mean(normal_weekdays, axis=0)
avg_weekend = np.mean(normal_weekends, axis=0)
inverse_weekday = avg_weekday[::-1]
inverse_weekend = avg_weekday[::-1]
print(inverse_weekend.shape)
#imputed_weekdays = np.concatenate((normal_weekdays, inverse_weekday.reshape(1,24,1)), axis=0)
#imputed_weekends = np.concatenate((normal_weekends, inverse_weekend.reshape(1,24,1)), axis=0)
imputed_weekdays = np.concatenate((normal_weekdays, avg_weekday.reshape(1,24,1)), axis=0)
imputed_weekends = np.concatenate((normal_weekends, avg_weekend.reshape(1,24,1)), axis=0)
print(imputed_weekends.shape)
weekdays_df = pd.DataFrame(imputed_weekdays.reshape(109,24))
weekends_df = pd.DataFrame(imputed_weekends.reshape(33,24))
weekdays_df.to_csv('weekdays_df.csv', index=False)
weekends_df.to_csv('weekends_df.csv', index=False)
plt.plot(avg_weekday, label='Average weekday')
plt.plot(inverse_weekday, label='Inverse weekday')
plt.xlabel('Time (hour)', fontsize=15)
plt.ylabel('Traffic volume data (scaled)', fontsize=15)
#plt.legend(fontsize=15)

# %%
imputed_weekdays = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/code/weekdays_df.csv').to_numpy()
imputed_weekends = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/code/weekends_df.csv').to_numpy()
imputed_weekdays = TimeSeriesScalerMeanVariance().fit_transform(imputed_weekdays)
imputed_weekends = TimeSeriesScalerMeanVariance().fit_transform(imputed_weekends)
plt.plot(imputed_weekdays[-1])
plt.xlabel('Time (hour)', fontsize=15)
plt.ylabel('Traffic volume data (scaled)', fontsize=15)
# %%
# DBSCAN #2
labels = []
for X_scaled in [imputed_weekdays, imputed_weekends]:

    nbrs = NearestNeighbors(n_neighbors = 5).fit(X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1]))
    # Find the k-neighbors of a point
    neigh_dist, neigh_ind = nbrs.kneighbors(X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1]))
    # sort the neighbor distances (lengths to points) in ascending order
    # axis = 0 represents sort along first axis i.e. sort along row
    sort_neigh_dist = np.sort(neigh_dist, axis = 0)

    k_dist = sort_neigh_dist[:, 4]

    kneedle = KneeLocator(x = range(1, len(neigh_dist)+1), y = k_dist, S = 1.0, 
                          curve = "concave", direction = "increasing", online=True)

    # get the estimate of knee point
    #font = {'size': 15}
    #plt.rc('font', **font)
    knee = kneedle.knee_y
    print(knee)
    plt.plot(k_dist)
    plt.ylabel("k-NN distance")
    plt.xlabel("Sorted observations (5th NN)")
    plt.plot(list(k_dist).index(knee), knee, color='red', marker='o', markersize=10)
    plt.show()

    # With *2 we got 0 anomalies, with *1 we got 2.
    clusters = DBSCAN(eps = knee, min_samples=24).fit(X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1]))
    # get cluster labels
    print(pd.Series(clusters.labels_).value_counts())
    labels.append(clusters.labels_)

# %%
weekday_labels = list(zip(weekday_indices, labels[0]))
weekend_labels = list(zip(weekend_indices, labels[1]))
combined_list = sorted(weekday_labels + weekend_labels, key=lambda tuple: tuple[0])
combined_labels = [n[1] for n in combined_list]
# %%
labels = combined_labels.copy()

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

# %%
anoms = []
for i, l in enumerate(labels):
    if l == -1:
        anoms.append(i)

print(anoms)
# %%
font = {'family': 'normal',
            'weight': 'normal',
            'size': 20}

plt.rc('font', **font)

fig, ax = plt.subplots(1,1)

ax.plot(X_pad.iloc[0])
#ax.set_facecolor('yellow')
plt.xlabel('Time (hour)')
plt.ylabel('Data traffic volume (MB)')
#plt.show()
plt.savefig('C:/Users/mirom/Desktop/IST/Thesis/imgs/normal', bbox_inches='tight')

# %%
# OpenCV
print('jee0')
image1 = cv2.imread('C:/Users/mirom/Desktop/IST/Thesis/imgs/normal.png')
image2 = cv2.imread('C:/Users/mirom/Desktop/IST/Thesis/imgs/abnormal.png')
# Resize the images (optional if they are already of the same size)
#image1 = cv2.resize(image1, (800, 600))
#image2 = cv2.resize(image2, (800, 600))
#image3 = cv2.resize(image3, (800, 600))
print('jee2')
# Horizontally concatenate the images
combined_image = cv2.hconcat([image1, image2])
print('jee3')

# Save the combined image
cv2.imwrite('C:/Users/mirom/Desktop/IST/Thesis/imgs/anom_sample.png', combined_image)
# %%
fig, ax = plt.subplots()
'''
for a in scaled_avgs:
    print(a.reshape(-1).shape)
    ax.plot(a.reshape(-1))
'''

#
weekday_avgs = df[(df['workday'] == 1) & (df['sector'] == sec)].groupby('timestamp', as_index=False).sum()['traffic_volume_data']
weekend_avgs = df[(df['workday'] == 0) & (df['sector'] == sec)].groupby('timestamp', as_index=False).sum()['traffic_volume_data']
ax.plot(weekday_avg, color=sns.color_palette()[0])
ax.plot(weekend_avg, color=sns.color_palette()[0], linestyle='--')

line_a = Line2D([], [], color=sns.color_palette()[0], label='Weekday', linestyle='-')
line_b = Line2D([], [], color=sns.color_palette()[0], label='Weekend', linestyle='--')
#pop_a = mpatches.Patch(color=sns.color_palette()[0], label='Weekday')
#pop_b = mpatches.Patch(color=sns.color_palette()[1], label='Sector 2')
#pop_c = mpatches.Patch(color=sns.color_palette()[2], label='Sector 3')
fig.legend(handles=[line_a, line_b], ncol=3, loc='upper center', fontsize=14)
ax.set_xlabel('Time (hour)', fontsize=14)
ax.set_ylabel('Average number of users', fontsize=14)

plt.show()
#x.set_xticks(range(11, 168, 12)) 
#ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
#ax.set_xticklabels([12, 24,12, 24,12, 24,12, 24,12, 24,12, 24,12, 24,])
# %%
#x = 4
#y = 6
x = 7
y = int(np.ceil(len(X_pad) / 7) + 2)

fig, ax = plt.subplots(y, x+1, figsize=(25,25))
#fig.subplots_adjust(hspace=1.5, wspace=0.3)
fig.suptitle('Daily traffic volume data in sector {} {}'.format(sec, file[:-4]), fontsize='xx-large')

for i in range(y):
    for j in range(x):
        index = i * x + j
        #if index in day_n:
        if index + 1 > len(X) + len(missing):
            continue
        
        if scaled:
            ax[i, j].plot(X[index], color='rgby'[labels[index]])
        else:
            ax[i, j].plot(X_pad.loc[index], color='r')
            ax[i, j].plot(den_pad.loc[index], color='b', linestyle='--')
        #axs[i, j].plot(X_change.iloc[index], color='grey', linestyle='--')
        #ax[i, j].set_title('C_red {:.2f}, C_blue: {:.2f}'.format(corrs_X[index], corrs_denoised[index]), fontsize='small')
        corr = corrs[index]
        ax[i, j].set_title('{:.2f}'.format(corr))
        if corr < 0.7:
            ax[i, j].set_facecolor('mistyrose')
            
if scaled:
    ax[y-1, x-1].plot(X_avg.reshape(-1), color='black')
else:
    ax[y-1, x].plot(X_avg, color='black')
    ax[y-1, x].plot(denoised_avg, color='gray', linestyle='--')

ax[y-1, x].set_title('Average day')

ax[y-3,x].plot(weekday_avg)
ax[y-3, x].set_title('Average weekday sector')
ax[y-2,x].plot(weekend_avg)
ax[y-2, x].set_title('Average weekend sector')


for i in range(7):
    ax[y-1, i].plot(daily_avgs[i], color='green')
    ax[y-1, i].set_title('Average {}'.format(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][i]))

'''
for i in range(25):
    ax[i, x].plot(weekly_avgs[i], color='green')
    ax[i, x].set_title('Average week {}'.format(i))
'''

for l in range(len(labels)):
    #if labels[l] == 2:
    ax[l // 7, l % 7].set_facecolor(['white', 'lightblue', 'mistyrose', 'seashell', 'lightgreen', 'yellow'][labels[l]])

plt.savefig('C:/Users/mirom/Desktop/IST/Thesis/imgs/New_data/{}_sec{}.png'.format(file[:-4], sec))
plt.show()
# %%
print(combined_labels)
# %%
random.seed(42)

# Using all KPIs except TVD
full_normal_weekday = []
full_anom_weekday = []
full_normal_weekend = []
full_anom_weekend = []
full_normal = []
full_anom = []
full_test = []

test = pd.concat(full_tss).interpolate()

for i, l in enumerate(combined_labels):
    if l == 0:
        '''
        if len(full_tss[i]) < 24:
            print('jee')
            missing = list(set(range(1,25)) - set(full_tss[i].index))
            for m in missing[::-1]:
                full_tss[i].loc[m] = (full_tss[i].iloc[m-1] + full_tss[i].iloc[m+1]) / 2
            full_tss[i].sort_index(inplace=True)
        '''

        full_normal.append(full_tss[i].drop(['traffic_volume_data'], axis=1))
        #full_normal.append(full_tss[i]['traffic_volume_data'])
        #full_normal.append(test.iloc[i*24 : (i+1)*24].drop(['traffic_volume_data'], axis=1))
        #full_normal.append(full_tss[i])
        #if i in weekend_indices:
            #full_normal_weekend.append(full_tss[i].drop(['traffic_volume_data'], axis=1))
            #full_normal_weekend.append(full_tss[i])
        #else:
            #full_normal_weekday.append(full_tss[i].drop(['traffic_volume_data'], axis=1))
            #full_normal_weekday.append(full_tss[i])

    elif l == -1:
        '''
        if len(full_tss[i]) < 24:
            print('jee')
            missing = list(set(range(1,25)) - set(full_tss[i].index))
            for m in missing[::-1]:
                full_tss[i].loc[m] = (full_tss[i].iloc[m-1] + full_tss[i].iloc[m+1]) / 2
            full_tss[i].sort_index(inplace=True)
        '''

        full_anom.append(full_tss[i].drop(['traffic_volume_data'], axis=1))
        #full_anom.append(full_tss[i]['traffic_volume_data'])
        #full_anom.append(test.iloc[i*24 : (i+1)*24].drop(['traffic_volume_data'], axis=1))
        #full_anom.append(full_tss[i])
        
        #if i in weekend_indices:
            #full_anom_weekend.append(full_tss[i].drop(['traffic_volume_data'], axis=1))
            #full_anom_weekend.append(full_tss[i])
        #else:
            #full_anom_weekday.append(full_tss[i].drop(['traffic_volume_data'], axis=1))
            #full_anom_weekday.append(full_tss[i])

sc = StandardScaler()
# For all

split = round(len(full_normal) * 0.9)
'''
full_train = pd.concat(full_normal[:split])
full_test = pd.concat(full_normal[split:-1])
full_anom = pd.concat(full_anom)
'''
full_train = np.array(full_normal[:split])
full_test = np.array(full_normal[split:-1])
full_anom = np.array(full_anom)

'''
full_train_scaled = []
full_test_scaled = []
full_anom_scaled = []

for i, day in enumerate(full_train):
    if i == 0:
        full_train_scaled.append(sc.fit_transform(day))
    else:
        full_train_scaled.append(sc.fit_transform(day))

for i, day in enumerate(full_test):
    full_test_scaled.append(sc.fit_transform(day))

for i, day in enumerate(full_anom):
    full_anom_scaled.append(sc.fit_transform(day))

full_train_scaled = np.array(full_train_scaled)
full_test_scaled = np.array(full_test_scaled)
full_anom_scaled = np.array(full_anom_scaled)
'''
'''
tss = TimeSeriesScalerMeanVariance()
full_train_scaled = tss.fit_transform(full_train)
full_test_scaled = tss.transform(full_test)
full_anom_scaled = tss.transform(full_anom)
'''

full_train_scaled = sc.fit_transform(full_train.reshape(-1, full_train.shape[-1])).reshape(full_train.shape)
full_test_scaled = sc.transform(full_test.reshape(-1, full_test.shape[-1])).reshape(full_test.shape)
full_anom_scaled = sc.transform(full_anom.reshape(-1, full_anom.shape[-1])).reshape(full_anom.shape)

'''
# For weekday
split = round(len(full_normal_weekday) * 0.9)
full_train_weekday = pd.concat(full_normal_weekday[:split])
full_test_weekday = pd.concat(full_normal_weekday[split:-1])
full_anom_weekday = pd.concat(full_anom_weekday)

full_train_weekday_scaled = sc.fit_transform(full_train_weekday)
full_test_weekday_scaled = sc.transform(full_test_weekday)
full_anom_weekday_scaled = sc.transform(full_anom_weekday)

# For weekend
split = round(len(full_normal_weekend) * 0.9)
full_train_weekend = pd.concat(full_normal_weekend[:split])
full_test_weekend = pd.concat(full_normal_weekend[split:-1])
full_anom_weekend = pd.concat(full_anom_weekend)

full_train_weekend_scaled = sc.fit_transform(full_train_weekend)
full_test_weekend_scaled = sc.transform(full_test_weekend)
full_anom_weekend_scaled = sc.transform(full_anom_weekend)
'''
# %%
for i in range(7):
    print(np.max(full_train_scaled[:,:,i]))
#plt.plot(full_train_scaled[0,:,6])
# %%
# Using only TVD
normal = []
anomalies = []

for i, l in enumerate(labels):
    if l == 0:
        normal.append(X_pad.loc[i])
        
    elif l == -1:
        anomalies.append(X_pad.loc[i])

split = round(len(normal) * 0.9)  

train_days = np.array(normal[:split])
test_days = np.array(normal[split:-1])
anomalies_days = np.array(anomalies)

tss = TimeSeriesScalerMeanVariance()
train_days_scaled = tss.fit_transform(train_days)
test_days_scaled = tss.transform(test_days)
anom_days_scaled = tss.transform(anomalies_days)

train = train_days.reshape(train_days.shape[0] * train_days.shape[1], 1)
test = test_days.reshape(test_days.shape[0] * test_days.shape[1], 1)
anomalies = anomalies_days.reshape(anomalies_days.shape[0] * anomalies_days.shape[1], 1)

sc = StandardScaler()
train_scaled = sc.fit_transform(train)
test_scaled = sc.transform(test)
anom_scaled = sc.transform(anomalies)

print(train_days.shape, train_days_scaled.shape)
# %%
print(full_train_scaled.shape, full_train.shape)
#plt.plot(full_anom_scaled[0][:,3])
print(np.std(full_train_scaled[0][:,0]))
print(np.std(full_train_scaled.reshape(140*24,11)[:,10]))
plt.plot(full_train_scaled[0][:,0])
#print(np.mean(full_train_scaled[0][:,2]))

# %%
def window(ts, window_size):
    x_values = []

    for i in range(len(ts) - window_size):
        x_values.append(ts[i : (i + window_size)])

    return np.array(x_values)

#n_lags determines how many previous data points are considered
n_lags = 24

X_train = window(train_scaled, n_lags)
X_test = window(test_scaled, n_lags)
X_anom = window(anom_scaled, n_lags)

X_full_train = window(full_train_scaled, n_lags)
X_full_test = window(full_test_scaled, n_lags)
X_full_anom = window(full_anom_scaled, n_lags)
# %%
normal_df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/code/normal_df.csv')
normal_data = normal_df
train_array = normal_data.to_numpy().reshape(95,24,7)[:80]

full_train_scaled = train_array
# %%
for i in range(7):
    print(full_train_scaled[:,:,i].min())
# %%
print(full_train_scaled.shape, full_test_scaled.shape, full_anom_scaled.shape)

model = Sequential()
# Encoder
#model.add(LSTM(256, activation='relu', input_shape=(full_train_scaled.shape[1], full_train_scaled.shape[2]), return_sequences=True))
model.add(LSTM(64, activation='relu', input_shape=(full_train_scaled.shape[1], full_train_scaled.shape[2])))
#model.add(LSTM(64, activation='relu', input_shape=(24,1)))

#model.add(RepeatVector(X_train.shape[1]))
model.add(RepeatVector(full_train_scaled.shape[1]))
# Decoder
model.add(LSTM(64, activation='relu', return_sequences=True))
#model.add(LSTM(256, activation='relu', return_sequences=True))

model.add(TimeDistributed(Dense(full_train_scaled.shape[2])))
#model.add(TimeDistributed(Dense(1)))

print(model.summary())
model.compile(optimizer='adam', loss='mse')
# fit model
history = model.fit(full_train_scaled, full_train_scaled, epochs=30, verbose=True)
plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')

plt.plot(history.history['loss'], label='Training loss')
plt.legend()
plt.show()
# %%
sets = [X_train, X_test, X_anom]
s = 0
n = 50
yhat = model.predict(sets[s], verbose=0)
print(np.mean(np.square(yhat[n] - sets[s][n])))
plt.plot(yhat[n], color='blue')
plt.plot(sets[s][n], color='green')

# %%
sets = [X_full_train, X_full_test, X_full_anom]
s = 2
n = 71
kpi = 0

yhat = model.predict(sets[s], verbose=0)
for i in range(11):
    print(np.mean(np.square(yhat[n] - sets[s][n])[:,i]))

yhat = model.predict(sets[s], verbose=0)
print(np.mean(np.square(yhat[n] - sets[s][n])))
plt.plot(sets[s][n][:,kpi])
plt.plot(yhat[n][:,kpi])
# %%
yhat = model.predict(X_anom, verbose=0)
print(np.mean(np.square(yhat - X_anom)))
# %%
#reconstruction_loss = model.evaluate(anomalies, anomalies)
#print(reconstruction_loss)
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
raws_anom = []
yhat = model.predict(full_anom_scaled, verbose=0)
for i in range(len(full_anom_scaled)):
    #mse = np.mean(np.square(yhat[i].flatten() - X_train[i]))
    mse = np.mean(np.square(yhat[i] - full_anom_scaled[i]))
    mses_anom.append(mse)
    if mse > 0.1:
        raws_anom.append(i)
        if mse > 0.82:
            print(i)
        else:
            print('jee: ', i)



# %%
# Obtain set of normal days
normal_set = []
normal_index = []
original = np.array(full_normal)
original_scaled = sc.fit_transform(original.reshape(-1, original.shape[-1])).reshape(original.shape)
yhat = model.predict(original_scaled, verbose=0)
for i in range(len(yhat)):
    mse = np.mean(np.square(yhat[i] - original_scaled[i]))
    if mse < 0.5:
        normal_index.append(i)
        normal_set.append(original_scaled[i])
    elif mse > 8:
        print(i)

plt.plot(full_train_scaled[21,:,6], label='Original')
plt.plot(yhat[21,:,6], label='Reconstructed')
#plt.plot(normal_set[0])
plt.xlabel('Time (hour)', fontsize=15)
plt.ylabel('Cqi average (scaled)', fontsize=15)
plt.legend(fontsize=15)
print(len(normal_set))
normal_df = pd.DataFrame(np.array(normal_set).reshape(len(normal_set)*24,7))
normal_df.to_csv('normal_df.csv', index=False)

# %%
normal_df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/code/normal_df.csv')
normal_data = normal_df.copy()
n_days = int(normal_data.shape[0] / 24)
train_array = normal_data.to_numpy().reshape(n_days,24,7)[:60]
test_array = normal_data.to_numpy().reshape(n_days,24,7)[60:]

avgs = []
inversed = []
for n in range(7):
    avg_day = np.mean(test_array[:,:,n], axis=0)
    avgs.append(avg_day)
    inversed.append(avg_day[::-1])

imputed_array = np.concatenate((test_array, np.array(avgs).T.reshape(1,24,7)), axis=0)
imputed_array = np.concatenate((imputed_array, np.array(inversed).T.reshape(1,24,7)), axis=0)
print(imputed_array.shape)
plt.plot(imputed_array[-1,:,0])
plt.xlabel('Time (hour)', fontsize=15)
plt.ylabel('Number of users (scaled)', fontsize=15)
n_test_days = int(imputed_array.shape[0])
pd.DataFrame(imputed_array.reshape(n_test_days*24, 7)).to_csv('imputed_df.csv', index=False)

# %%
imputed_data = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/code/imputed_df.csv')
imputed = imputed_data.to_numpy().reshape(n_test_days,24,7)
plt.plot(imputed[-1,:,0])
print(train_array.shape)
print(np.min(original[:,:,0]))
print(np.min(train_array[:,:,0]))
print(np.std(full_train))
print(np.mean(full_train))
print(np.std(original))
# %%


mses = []
yhat = model.predict(imputed)
print(yhat.shape)
for i in range(len(imputed)):
    mse = np.mean(np.square(yhat[i] - imputed[i]))
    mses.append(mse)

mses_1 = []
yhat_1 = model.predict(train_array)
for i in range(len(train_array)):
    #mse = np.mean(np.square(yhat[i].flatten() - X_train[i]))
    mse = np.mean(np.square(yhat_1[i] - train_array[i]))
    mses_1.append(mse)

print(len(mses))
print('max test mse: ', np.max(mses))
print('max train mse: ', np.max(mses_1))
print('index of max: ', mses.index(np.max(mses)))
print('mse of second largest: ', np.partition(mses, -1)[-2])
print('mse of last: ', mses[-1])
plt.plot(yhat[-1])
plt.clf()
sns.histplot(mses_1, label='Train')
sns.histplot(mses, label='Test')
plt.legend(fontsize=15)
plt.xlabel('MSE', fontsize=15)
plt.ylabel('Count', fontsize=15)
# %%
plt.plot(yhat[-1,:,0])
plt.xlabel('Time (hour)', fontsize=15)
plt.ylabel('Number of users (scaled)', fontsize=15)

# %%
p = 100
print('Train: ', np.percentile(mses_train, p))
print('Test: ', np.percentile(mses_test, p))
print('Anom: ', np.percentile(mses_anom, p))
# %%
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
#plt.savefig('C:/Users/mirom/Desktop/IST/Thesis/imgs/New_data/{}_sec{}_recon_hist.png'.format(file[:-4], sec))
plt.plot()

# %%
errs = np.square(yhat - full_train_scaled)
print(errs.shape)
#print(np.mean(np.square(yhat - full_anom_scaled)))
kpi_df = pd.DataFrame(yhat[:,:,0])
kpi_df = kpi_df.stack().reset_index()
kpi_df.columns = ['Day', 'Hour', 'KPI']
print(kpi_df)
sns.lineplot(data=kpi_df, x='Hour', y='KPI', hue='Day', palette='Set1')
plt.ylabel('')
plt.title('Sec {} anom days HSR inter freq reconstructed'.format(sec))
# %%
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
    #'traffic_volume_data',
    'tp_carrier_aggr',
    #'carrier_aggr_usage',
    'cqi_avg',
    #'time_advance_avg'
]

indices = []
for r in raws_anom:
    errors = []
    for i in range(len(num_cols)):
        errors.append(np.mean(np.square(yhat[r] - full_anom_scaled[r])[:,i]))
   
    indices.append(errors.index(max(errors)))
    print(r, errors.index(max(errors)))
    #for e in errors:
    #    if e > 1.5:
    #        indices.append(errors.index(e))

vc = pd.Series([num_cols[i] for i in indices]).value_counts()
vc.plot.bar(rot=90)
plt.title('Sec {} RCA'.format(sec, w))
# %%
# Combined bar plot
errs = np.square(yhat - full_anom_scaled)
daily_mses = np.round(np.mean(errs, axis=(1,2)), decimals=2)
mses_per_kpi = []
for i in range(len(num_cols)):
    mses_per_kpi.append([np.mean(x) for x in errs[:,:,i]])

mses_per_kpi = np.array(mses_per_kpi).T
ind = np.arange(len(mses_per_kpi))
colors = sns.color_palette("Paired")

fig = plt.subplots()
labels = []
plots = []
sums = np.zeros(len(mses_per_kpi))
for i in range(len(num_cols)):
    if i == 0:
        print(mses_per_kpi[:,i])
        plt.bar(ind, mses_per_kpi[:,i])
        label = num_cols[i]
        sums += mses_per_kpi[:,i]
    else:
        plt.bar(ind, mses_per_kpi[:,i], bottom=sums)
        label = num_cols[i]
        sums += mses_per_kpi[:,i]

    labels.append(label)

print(labels)
plt.legend(labels)
plt.xticks(ind, daily_mses)
plt.xlabel('Daily MSE')
plt.ylabel('MSE per KPI')
plt.title('Reconstruction errors of anomalies in sector {} in\n {}'.format(sec, file[:-4]))
plt.savefig('C:/Users/mirom/Desktop/IST/Thesis/imgs/New_data/{}_sec{}_recon_KPI_bar'.format(file[:-4], sec))
plt.plot()
# %%
# Individual bar plot
day = 0
print(mses_per_kpi[day])
print(np.mean(mses_per_kpi[day]))
mse = np.mean(np.square(yhat[day] - full_anom_scaled[day]))
plt.bar(np.arange(len(num_cols)), mses_per_kpi[day])
plt.xticks(np.arange(len(num_cols)), num_cols, rotation=90)
plt.title('Day {} of anomalies in sector {}. Day MSE: {:.2f}'.format(day+1, sec, mse))
plt.plot()


# %%
highlight_x = raws_anom
highlight_y = anomalies[raws_anom]
plt.plot(anomalies)
plt.plot(highlight_x, highlight_y, 'ro') 
# %%
print(full_tss[0])
# %%
test = X_pad.to_numpy().reshape(X_pad.shape[0]*X_pad.shape[1], 11)
test_scaled = sc.transform(test)
test_w = window(test_scaled, n_lags)
print(test_w.shape)

yhat = model.predict(test_w)

# %%
indices = []
for i in range(len(test_w)):
    mse = np.mean(np.square(yhat[i] - test_w[i]))
    mses_anom.append(mse)
    if mse > 2.705:
        print(i)
        indices.append(i+24)

# %%
print(max(indices))
# %%
x = 7
y = 27

fig, ax = plt.subplots(y, x+1, figsize=(25,25))
#fig.subplots_adjust(hspace=1.5, wspace=0.3)
fig.suptitle('Anomalies in {} based on Traffic volume data'.format(sec), fontsize='xx-large')

for i in range(y):
    for j in range(x):
        index = i * x + j
        #if index in day_n:
        if index + 1 > len(X) + len(missing):
            continue
        
        ax[i, j].plot(X_pad.loc[index], color='b')
        print(index)

for anom_index in indices:
    i = anom_index // 24 // 7
    j = anom_index // 24 % 7
    n = int(np.round((anom_index / 24 % 7) % 1 * 24))
    print(i, j, n)
    ax[i, j].plot(n, X_pad.loc[i * x + j].iloc[n], 'ro')

for l in range(len(labels)):
    ax[l // 7, l % 7].set_facecolor(['white', 'lightblue', 'mistyrose', 'seashell', 'lightgreen', 'yellow'][labels[l]])
    

# %%
print(X_pad.loc[8 * 7 + 2].iloc[24])
# %%
print(426 // 24 // 7)
print(426 / 24 % 7)
print(np.round((826 / 24 % 7) % 1 * 24))
# %%
