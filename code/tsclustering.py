# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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
# %%

df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/data/KPIs1_outer2.csv')
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

# %%
# 1. WEEKLY TIMESERIES
# Split into weekly timeseries
cell = 'CEIRA_MCO001N3L08'
#time_df = df[(df['daytime'] == 1) & (df['workday'] == 1)]
cell_df = df[df['ltecell_name'] == cell][['timestamp', 'cell_throughput_dl_avg']]
weekly_tss = []
# The last timestamp of the data is faulty -> drop it
cell_df.dropna(inplace=True)
cell_df.drop(cell_df.tail(1).index,inplace=True)
#cell_df.set_index('timestamp', inplace=True)
# Remove missing values
#first = cell_df.index[0]
#last = cell_df.index[-1]
first = pd.to_datetime('2022-09-19 00:00:00')
last = pd.to_datetime('2023-03-15 23:00:00')
#print(sum(cell_df[(cell_df['timestamp'] >= first) & (cell_df['timestamp'] <= last)]['user_throughput_dl_avg']))
#print(first, last)
start_date = first
td = pd.Timedelta(1, 'D')
end_date = first + td

lengths = []
while end_date < last:
    ts = cell_df[(start_date <= cell_df['timestamp']) & (cell_df['timestamp'] < end_date)]
    # Weekly index
    #index = ((ts['timestamp'].dt.weekday * 24) + (ts['timestamp'].dt.hour + 1))
    # Daily index
    index = ts['timestamp'].dt.hour + 1
    ts.set_index(index, inplace=True)
    #ts.reset_index(inplace=True)
    ts = ts['cell_throughput_dl_avg']
    ###
    weekly_tss.append(ts)
    
    #Drop weeks missing more than 10% of hours
    '''
    if len(ts) >= 24 * 0.9:
        weekly_tss.append(ts)
        lengths.append(len(ts))
    '''
    start_date += td
    end_date += td

X = pd.concat(weekly_tss, axis=1).T


print(lengths)

#Noise reduction

for i in range(len(X)):
    X.iloc[i] = X.iloc[i].rolling(3).mean()

#X.fillna(value=0,inplace=True)

X_avg = X.mean()

X_diffs = []
for i in range(len(X) - 1):
    X_diffs.append((X.iloc[i] - X_avg))
X_diffs = pd.DataFrame(X_diffs)

X_change = X.diff(axis=1)

#Interpolate missing values
'''
for i in range(len(X)):
    X.iloc[i] = X.iloc[i].interpolate()
'''

corrs = []
for i in range(len(X)):
    corrs.append(X_avg.corr(X.iloc[i]))


'''
# Categorize
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


scaled = False

# %%
# Scale data

X = TimeSeriesScalerMeanVariance().fit_transform(X)
#X_diffs = TimeSeriesScalerMeanVariance().fit_transform(X_diffs)
#X_change = TimeSeriesScalerMeanVariance().fit_transform(X_change)
X_avg = TimeSeriesScalerMeanVariance().fit_transform(X_avg.values.reshape(1,-1))

print(X.shape)
#X = TimeSeriesScalerMinMax().fit_transform(X)
#X_avg = TimeSeriesScalerMinMax().fit_transform(X_avg.values.reshape(1,-1))

ks2 = []
for i in range(len(X)):
    ks2.append(kstest(X_avg[0].flatten(), X[i].flatten())[1])

scaled = True
# %%
# 2. COMPARING CELLWISE TIMESERIES
tps = []
for cell in df['ltecell_name'].unique():
    tps.append(df[df['ltecell_name'] == cell][['ltecell_name', 'timestamp', 'user_throughput_dl_avg']].set_index('timestamp'))

# %%
data = pd.DataFrame()
for tp in tps:
    data[tp['ltecell_name'][0]] = tp['user_throughput_dl_avg']
# %%
temp = data.dropna()
print(temp)
timeseries_df = temp.T
scaler = StandardScaler()
for i in range(len(timeseries_df)):
    tp = timeseries_df.iloc[i].values
    scaled = scaler.fit_transform(tp.reshape(-1, 1))
    timeseries_df.iloc[i] = scaled.reshape(-1,)

# %%
# Time series KMeans clustering
#d = pd.DataFrame(X.reshape(X.shape[0], X.shape[1]))

km = TimeSeriesKMeans(n_clusters=3, metric="dtw", random_state=1)
#km = KernelKMeans(n_clusters=3, kernel="gak", random_state=0)
#km =  KShape(n_clusters=3, verbose=True, random_state=0)

labels = km.fit_predict(X)
print(pd.Series(labels).value_counts())

print(silhouette_score(X, labels))

# %%
x = 7
y = 25

fig, ax = plt.subplots(y,x,figsize=(25,25))
for i in range(y):
    for j in range(x):
        index = i * x + j
        if index + 1 > len(X):
            continue
            
        if scaled:
            ax[i, j].plot(X[index], color='rgby'[labels[index]])
        else:
            ax[i, j].plot(X.iloc[index], color='rgby'[labels[index]])
        #axs[i, j].plot(X_change.iloc[index], color='grey', linestyle='--')
        ax[i, j].set_title('Corr: {:.2f}'.format(corrs[index]))

if scaled:
    ax[y-1, x-1].plot(X_avg.reshape(-1), color='black')
else:
    ax[y-1, x-1].plot(X_avg, color='black')
ax[y-1, x-1].set_title('Average week')
fig.suptitle('Weekly user dl throughput in cell {}'.format(cell), fontsize='xx-large')
plt.show()
# %%

clusters = []
zipped = zip(labels, list(df['ltecell_name'].unique()))
dict = {c: l for (l, c) in zipped}
def get_clust(row):
    clusters.append(dict.get(row['ltecell_name']))

df.dropna().apply(lambda x: get_clust(x), axis=1)
print(clusters)
# %%
clust_df = df.dropna()
clust_df['cluster'] = clusters
print(clust_df)
# %%
sns.lineplot(data=clust_df, x='timestamp', y='user_throughput_dl_avg', hue='cluster')
# %%
cell_df = df[df['ltecell_name'] == 'CEIRA_MCO001N1L08']
plt.plot(cell_df['timestamp'], cell_df['user_throughput_dl_avg'])
# %%
