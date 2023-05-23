#%%
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt
# %%

df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/data/KPIs1_outer2.csv')
#Remove nulls and infs
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
#Create time columns
df['timestamp'] = pd.to_datetime(df['datetime'])
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.weekday
#Create new features
df['daytime'] = [1 if 7 < x else 0 for x in df['hour']]
df['workday'] = [0 if 4 < x else 1 for x in df['day']]

cat_cols = [
    'timestamp', 'datetime', 'ltecell_name'
]
# %%
#Helper function to divide based on sectors
def divide_sectors(cell_name):
    sector = int(cell_name[-4])
    return sector

#Divide into sectors
df['sector'] = df['ltecell_name'].apply(lambda x: divide_sectors(x))

#Helper function to divide based on frequencies
def divide_freqs(cell_name):
    freq = int(cell_name[-2:])
    return freq

#Divide into frequencies
df['freq'] = df['ltecell_name'].apply(lambda x: divide_freqs(x))
# %%
# %%
#Encode cyclical features using sin/cos encoding
def encode(data, col, max):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max)
    return data

encode(df, 'hour', 24)
encode(df, 'day', 7)

# %%
#One-Hot encode different cells
#dummies = pd.get_dummies(df['ltecell_name'])
#drop_df = pd.concat([drop_df, dummies], axis=1)
cell = 'CEIRA_MCO001N1L21'
cell_df = df[df['ltecell_name'] == cell]
#cell_df = cell_df[cell_df['user_throughput_dl_avg'] < 150000]
drop_df = cell_df.drop(['timestamp', 'datetime', 'ltecell_name', 'prb_dl_usage_rate_avg','traffic_volume_data','cell_throughput_dl_avg','cqi_avg','hour_sin', 'hour_cos','workday',   'day_sin', 'day_cos','freq', 'sector',  'hour', 'day',  'users_avg', 'time_advance_avg','tp_carrier_aggr', 'carrier_aggr_usage',  'prb_ul_usage_rate_avg', 'user_throughput_ul_avg', 'hsr_intra_freq', 'hsr_inter_freq', 'cell_throughput_ul_avg'], axis=1)
num_cols = ['user_throughput_dl_avg', 'daytime']
print(drop_df.columns)
# %%
ct = ColumnTransformer([
        ('somename', StandardScaler(), num_cols)
    ], remainder='passthrough')

scaled_df = ct.fit_transform(drop_df)
print(scaled_df.shape)
# %%
linkage_data = linkage(scaled_df, method='ward', metric='euclidean')
dendrogram(linkage_data)
plt.ylabel('distance')
plt.show()

# %%
clustering = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
labels = clustering.fit_predict(scaled_df)

#%%
clusters = pd.Series(labels).value_counts()
print(clusters)
n_clusts = len(clusters)


# %%
cell_df['cluster'] = labels
sns.scatterplot(data=cell_df, x='timestamp', y='user_throughput_dl_avg', hue='cluster')
plt.title('Agglomerative clustering on cell {}'.format(cell, ', '.join(drop_df.columns)))

# %%
clust_df = cell_df.drop(cat_cols, axis=1)
feats = []

for clust in clust_df['cluster'].unique():
    feat = []
    print('clust', clust)
    for col in clust_df.columns:
        print(col)
        feat.append(clust_df[clust_df['cluster'] == clust][col].mean())
        print(clust_df[clust_df['cluster'] == clust][col].mean())
    print('mean', clust_df[col].mean())
    feats.append(feat)

# %%
print(', '.join([f'{key}: {value}' for key, value in clust_df['cluster'].value_counts().items()]))
df_info = pd.DataFrame({'Cell': cell,
            'Features': ', '.join(sorted(drop_df.columns)),
            'Clusters': clust_df['cluster'].unique()
})
df_content = pd.DataFrame(feats, columns=clust_df.columns)
df_clustered = pd.concat([df_info, df_content], axis=1)
df_clustered.loc[len(df_clustered)] = clust_df.mean()
print(df_clustered[num_cols + ['daytime', 'workday', 'cluster']])

# %%

print(np.percentile(cell_df[cell_df['daytime'] == 0]['user_throughput_dl_avg'], 50))
print(cell_df[cell_df['daytime'] == 0]['user_throughput_dl_avg'].mean())
print(cell_df[cell_df['daytime'] == 1]['user_throughput_dl_avg'].mean())
# %%
sns.lineplot(data=cell_df, x='hour', y='user_throughput_dl_avg', hue='cluster')
# %%
df_clustered.to_csv('C:/Users/mirom/Desktop/IST/Thesis/data/agglomerative/{}_f{}_c{}.csv'.format(cell, len(drop_df.columns), n_clusts))


#####################################################
# %%
low_tp = cell_df[cell_df['cluster'] == 7]
print(low_tp.max())
# %%
print(low_tp)

# %%
drop_df = low_tp.drop(['timestamp', 'datetime', 'daytime', 'ltecell_name', 'hour', 'user_throughput_dl_avg', 'prb_ul_usage_rate_avg', 'hsr_intra_freq', 'hsr_inter_freq', 'cell_throughput_dl_avg', 'cell_throughput_ul_avg',], axis=1)
num_cols = ['time_advance_avg', 'cqi_avg', 'users_avg', 'tp_carrier_aggr', 'carrier_aggr_usage', 'traffic_volume_data', 'prb_dl_usage_rate_avg',]
print(drop_df.columns)
# %%
ct = ColumnTransformer([
        ('somename', StandardScaler(), num_cols)
    ], remainder='passthrough')

scaled_df = ct.fit_transform(drop_df)

# %%
linkage_data = linkage(scaled_df, method='ward', metric='euclidean')
dendrogram(linkage_data)

plt.show()

# %%
hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(scaled_df)

#%%
print(pd.Series(labels).value_counts())
# %%
low_tp['cluster'] = labels
#sns.scatterplot(data=low_tp, x='carrier_aggr_usage', y='tp_carrier_aggr', hue='cluster')
#sns.scatterplot(data=low_tp, x='traffic_volume_data', y='prb_dl_usage_rate_avg', hue='cluster')
sns.scatterplot(data=low_tp, x='time_advance_avg', y='cqi_avg', hue='cluster')
# %%
clust_df = low_tp.drop(cat_cols, axis=1)
feats = []
for clust in clust_df['cluster'].unique():
    feat = []
    print('clust', clust)
    for col in clust_df.columns:
        print(col)
        feat.append(clust_df[clust_df['cluster'] == clust][col].mean())
        print(clust_df[clust_df['cluster'] == clust][col].mean())
    print('mean', clust_df[col].mean())
    feats.append(feat)

# %%
print(', '.join([f'{key}: {value}' for key, value in clust_df['cluster'].value_counts().items()]))
df_info = pd.DataFrame({'Cell': cell,
            'Features': ', '.join(sorted(drop_df.columns)),
            'Clusters': clust_df['cluster'].unique()
})
df_content = pd.DataFrame(feats, columns=clust_df.columns)
df_clustered = pd.concat([df_info, df_content], axis=1)
df_clustered.loc[len(df_clustered)] = clust_df.mean()
print(df_clustered)
# %%
