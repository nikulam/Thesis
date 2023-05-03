# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
# %%
df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/data/KPIs.csv')
#Remove Inf and NaN
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

#Create time columns
df['timestamp'] = pd.to_datetime(df['datetime'])
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.weekday

num_cols = [
    'users_avg',
    'prb_dl_usage_rate_avg',
    'prb_ul_usage_rate_avg', 
    'hsr_intra_freq',
    'hsr_inter_freq',
    'cell_throughput_dl_avg',
    'cell_throughput_ul_avg',
    'user_throughput_dl_avg', 
    'user_throughput_ul_avg',
    'traffic_volume_data'
]

cat_cols = [
    'datetime',
    'enodeb_name',
    'ltecell_name',
    'timestamp',
    'day',
    'hour'
]

# %%
#Divide into normal dataset based on hsr_intra_freq
all_df = df.drop(cat_cols, axis=1)
normal_df = all_df[all_df['hsr_intra_freq'] == 100].drop(['hsr_intra_freq', 'hsr_inter_freq', 'traffic_volume_data', 'prb_dl_usage_rate_avg', 'prb_ul_usage_rate_avg', 'cell_throughput_ul_avg', 'user_throughput_dl_avg'], axis=1)
# %%
print(len(df), len(normal_df))
normalized_df = (normal_df - normal_df.mean()) / normal_df.std()

# %%
#PCA
'''
pca_t = ColumnTransformer([
        ('somename', StandardScaler(), num_cols)
    ], remainder='passthrough')

scaled_df = pca_t.fit_transform(normal_df)
'''
print(normalized_df.columns)
pca = PCA(n_components=3, random_state=0)
pca.fit_transform(normalized_df)
pca.explained_variance_ratio_

# %%
df_pca = pd.DataFrame(pca.fit_transform(scaled_df), index=df.index)
df_restored = pd.DataFrame(pca.inverse_transform(df_pca), index=df_pca.index)

# %%
def get_anomaly_scores(df_original, df_restored):
    loss = np.sum((np.array(df_original) - np.array(df_restored)) ** 2, axis=1)
    loss = pd.Series(data=loss, index=df_original.index)
    return loss

def is_anomaly(data, pca, threshold):
    pca_data = pca.transform(data)
    restored_data = pca.inverse_transform(pca_data)
    loss = np.sum((data - restored_data) ** 2)
    return loss > threshold

# %%

reconstruction_errors = get_anomaly_scores(df.drop(cat_cols, axis=1), df_restored)
print(reconstruction_errors)
# %%
