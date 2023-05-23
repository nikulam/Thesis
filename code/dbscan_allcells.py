# %%
import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from kneed import KneeLocator

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# %%

df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/data/KPIs1_outer.csv')
#Remove nulls and infs
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
#Create time columns
df['timestamp'] = pd.to_datetime(df['datetime'])
df['hour'] = df['timestamp'].dt.hour
#Create new features
df['daytime'] = [1 if 7 < x else 0 for x in df['hour']]
# %%
#One-Hot encode different cells
#dummies = pd.get_dummies(df['ltecell_name'])
#drop_df = pd.concat([drop_df, dummies], axis=1)
cell = 'CEIRA_MCO001N1L18'
cell_df = df[df['ltecell_name'] == cell]
drop_df = cell_df.drop(['timestamp', 'datetime', 'ltecell_name', 'time_advance_avg', 'hour','cqi_avg', 'prb_dl_usage_rate_avg', 'users_avg', 'tp_carrier_aggr', 'carrier_aggr_usage', 'traffic_volume_data', 'prb_ul_usage_rate_avg', 'user_throughput_ul_avg', 'hsr_intra_freq', 'hsr_inter_freq', 'cell_throughput_dl_avg', 'cell_throughput_ul_avg',], axis=1)
num_cols = ['user_throughput_dl_avg']
# %%
ct = ColumnTransformer([
        ('somename', StandardScaler(), num_cols)
    ], remainder='passthrough')

scaled_df = ct.fit_transform(drop_df)

# %%
# n_neighbors = 5 as kneighbors function returns distance of point to itself (i.e. first column will be zeros) 
nbrs = NearestNeighbors(n_neighbors = 5).fit(scaled_df)
# Find the k-neighbors of a point
neigh_dist, neigh_ind = nbrs.kneighbors(scaled_df)
# sort the neighbor distances (lengths to points) in ascending order
# axis = 0 represents sort along first axis i.e. sort along row
sort_neigh_dist = np.sort(neigh_dist, axis = 0)

k_dist = sort_neigh_dist[:, 4]
plt.plot(k_dist)
plt.ylabel("k-NN distance")
plt.xlabel("Sorted observations (4th NN)")
plt.show()

kneedle = KneeLocator(x = range(1, len(neigh_dist)+1), y = k_dist, S = 1.0, 
                      curve = "concave", direction = "increasing", online=True)

# get the estimate of knee point
knee = kneedle.knee_y
print(knee)


# %%
clusters = DBSCAN(eps=knee, min_samples=30).fit(scaled_df)
# get cluster labels
print(pd.Series(clusters.labels_).value_counts())

# %%
clust_df = drop_df.copy()
print(clust_df.columns)
clust_df['cluster'] = clusters.labels_
#%%
feats = []
for clust in clust_df['cluster'].unique():
    feat = []
    print('clust', clust)
    for col in [c for c in clust_df.columns if c != 'cluster']:
        print(col)
        feat.append(clust_df[clust_df['cluster'] == clust][col].mean())
        print(clust_df[clust_df['cluster'] == clust][col].mean())
    print('mean', clust_df[col].mean())
    feats.append(feat)

# %%
print(', '.join([f'{key}: {value}' for key, value in clust_df['cluster'].value_counts().items()]))
df_info = pd.DataFrame({'Cell': cell,
            'Features': ', '.join(sorted([c for c in clust_df.columns if c != 'cluster'])),
            'Clusters': clust_df['cluster'].unique()
})
df_content = pd.DataFrame(feats, columns=[c for c in clust_df.columns if c != 'cluster'])
df_clustered = pd.concat([df_info, df_content], axis=1)
df_clustered.loc[len(df_clustered)] = clust_df.drop(['cluster'], axis=1).mean()
print(df_clustered)

# %%
cell_df['cluster'] = clusters.labels_
sns.scatterplot(data=cell_df, x='timestamp', y='user_throughput_dl_avg', hue='cluster')

# %%

df_clustered.to_csv('C:/Users/mirom/Desktop/IST/Thesis/data/dbscan/clust_{}_{}.csv'.format(cell,len(clust_df.columns)))
# %%
def plotlines(ax):
    for i, row in clust_df[clust_df['cluster'] == -1].iterrows():
        ax.axvline(x=row['timestamp'], c='black', linestyle=':')

default_palette = sns.color_palette(palette='colorblind')
ax = sns.lineplot(data=clust_df, x='timestamp', y=num_cols[3], color=default_palette[0])
plotlines(ax)

# %%
#not_100 = clust_1[clust_1['hsr_intra_freq'] == 100]
#print(not_100['hsr_inter_freq'])
not_100 = clust_1[clust_1['hsr_intra_freq'] == 100]
clust_1.loc['mean'] = df[(df['ltecell_name'] == cell) & (df['daytime'] == 0)].mean()
not_100.loc['daytime_mean'] = df[(df['ltecell_name'] == cell) & (df['daytime'] == 1)].mean()
not_100.loc['nighttime_mean'] = df[(df['ltecell_name'] == cell) & (df['daytime'] == 0)].mean()
not_100[num_cols].to_csv('sample_csv_{}3.csv'.format(cell))
# %%
clust_df.to_csv('C:/Users/mirom/Desktop/IST/Thesis/data/dbscan_N1L08.csv')

# %%
