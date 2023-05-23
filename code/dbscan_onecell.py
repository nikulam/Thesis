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
# %%

df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/data/KPIs1_outer2.csv')
#Remove nulls and infs
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
#Create time columns
df['timestamp'] = pd.to_datetime(df['datetime'])
df['hour'] = df['timestamp'].dt.hour
#Create new features
df['daytime'] = [1 if 7 < x else 0 for x in df['hour']]
df.drop(['hour'], axis=1, inplace=True)
# %%
cell = 'CEIRA_MCO001N1L08'
drop_df = df[df['ltecell_name'] == cell].drop(['timestamp', 'datetime', 'ltecell_name', 'time_advance_avg', 'daytime'], axis=1)
num_cols = ['users_avg', 'prb_ul_usage_rate_avg', 
            'hsr_intra_freq', 'hsr_inter_freq',
            'cell_throughput_dl_avg', 'user_throughput_ul_avg',
            'tp_carrier_aggr', 'cqi_avg',  'traffic_volume_data', 'user_throughput_dl_avg', 'cell_throughput_ul_avg', 'prb_dl_usage_rate_avg', 'carrier_aggr_usage',]
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
clusters = DBSCAN(eps = knee, min_samples = 2*len(num_cols)).fit(scaled_df)
# get cluster labels
# %%
print(pd.Series(clusters.labels_).value_counts())
clust_df = df[df['ltecell_name'] == cell]
clust_df['cluster'] = clusters.labels_
# %%
def plotlines(ax):
    for i, row in clust_df[clust_df['cluster'] == -1].iterrows():
        ax.axvline(x=row['timestamp'], c='black', linestyle=':')

default_palette = sns.color_palette(palette='colorblind')
ax = sns.lineplot(data=clust_df, x='timestamp', y=num_cols[3], color=default_palette[0])
plotlines(ax)

# %%
clust_1 = clust_df[clust_df['cluster'] == -1]

print(len(clust_1[num_cols[3]]))
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
