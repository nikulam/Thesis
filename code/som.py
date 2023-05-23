# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import seaborn as sns
# %%
df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/data/KPIs1_outer.csv')
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
            'tp_carrier_aggr', 'carrier_aggr_usage', 'cqi_avg','traffic_volume_data', 'user_throughput_dl_avg', 'cell_throughput_ul_avg', 'prb_dl_usage_rate_avg', ]
# %%
ct = ColumnTransformer([
        ('somename', StandardScaler(), num_cols)
    ], remainder='passthrough')

scaled_df = ct.fit_transform(drop_df)
# %%
som_shape = (2,2)
som = MiniSom(x=som_shape[0], y=som_shape[1], input_len=len(num_cols), sigma=1, learning_rate=0.5)
som.random_weights_init(scaled_df)
som.train_random(scaled_df, 20000)

# %%
# Weights are:
wts = som.get_weights()
# Shape of the weight are:
print(wts.shape)
# Returns the distance map from the weights:
som.distance_map()
# %%
plt.bone()
plt.pcolor(som.distance_map().T)
plt.colorbar()
plt.show()

# %%
winner_coordinates = np.array([som.winner(x) for x in scaled_df]).T
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
print(pd.Series(cluster_index).value_counts())

# %%
clust_df = df[df['ltecell_name'] == cell]
clust_df['cluster'] = cluster_index
def plotlines(ax):
    for i, row in clust_df[clust_df['cluster'] == 2].iterrows():
        ax.axvline(x=row['timestamp'], c='black', linestyle=':')

default_palette = sns.color_palette(palette='colorblind')
ax = sns.lineplot(data=clust_df, x='timestamp', y=num_cols[2], color=default_palette[0])
plotlines(ax)

# %%
clust_df.to_csv('som_N1L08.csv')
# %%
dbscan_df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/data/dbscan_N1L08.csv')
dbscan_df['timestamp'] = pd.to_datetime(dbscan_df['timestamp'])
# %%
dbscan_df = dbscan_df.set_index('timestamp', drop=False)
clust_df = clust_df.set_index('timestamp', drop=False)
# %%
di = dbscan_df[dbscan_df['cluster'] == -1].index
ci = clust_df[clust_df['cluster'] == 2].index
sum(x in di for x in ci)

# %%
def plotlines(ax):
    for i, row in dbscan_df[dbscan_df['cluster'] == -1].iterrows():
        ax.axvline(x=row['timestamp'], c='black', linestyle=':')

default_palette = sns.color_palette(palette='colorblind')
ax = sns.lineplot(data=dbscan_df, x='timestamp', y=num_cols[2], color=default_palette[0])
plotlines(ax)
# %%
def plotlines(ax):
    for i, row in clust_df[clust_df['cluster'] == 2].iterrows():
        ax.axvline(x=row['timestamp'], c='black', linestyle=':')

default_palette = sns.color_palette(palette='colorblind')
ax = sns.lineplot(data=clust_df, x='timestamp', y=num_cols[2], color=default_palette[0])
plotlines(ax)

# %%
