#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/data/KPIs1_outer.csv')
cellwise_df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/data/cellwise_df.csv')
inner_df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/data/KPIs1.csv')
#Set timestamp index for cellwise_df
cellwise_df['timestamp'] = pd.to_datetime(cellwise_df['timestamp'])
cellwise_df.set_index(pd.to_datetime(cellwise_df['timestamp']),inplace=True)
inner_df['timestamp'] = pd.to_datetime(inner_df['datetime'])
#Remove Inf and NaN
#df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
cellwise_df = cellwise_df[~cellwise_df.isin([np.nan, np.inf, -np.inf]).any(1)]


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
    'traffic_volume_data',
    'tp_carrier_aggr',
    'cqi_avg',
    'time_advance_avg'
]

cat_cols = [
    'datetime',
    #'enodeb_name',
    'ltecell_name',
    #'timestamp',
    'day',
    'hour',
    'workday',
    'daytime'
]

#Shorten the ltecell names
df['ltecell_name'] = df['ltecell_name'].apply(lambda x: x[-5:])
inner_df['ltecell_name'] = inner_df['ltecell_name'].apply(lambda x: x[-5:])

# %%
#Create new features
df['daytime'] = [1 if 7 < x else 0 for x in df['hour']]
df['workday'] = [0 if 4 < x else 1 for x in df['day']]
df.drop(['hour', 'day'], axis=1, inplace=True)

# %%
groups = df.groupby('ltecell_name')
for name, group in groups:
    print(name, len(group))

# %%
print(len(df.dropna()))
# %%
#Lineplot
default_palette = sns.color_palette(palette='colorblind')
x = 'hour'
y = 'carrier_aggr_usage'
cell = 'N1L08'
plot_df = inner_df[(inner_df['ltecell_name'] == cell)]
ax = sns.lineplot(data=df, x=x, y=y, hue='ltecell_name', ci=False, estimator='mean', legend=True)
#ax.axvline(x=pd.to_datetime('2023-01-30 00:00:00'), c='black', linestyle='--')
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
plt.title('Average {} by {}'.format(y, x))
# %%
#Correlation matrix
#plot_df = df[df['daytime'] == 0].drop(['hour', 'day', 'daytime'], axis=1)
cell = 'N1L08'
ho = 'hsr_intra_freq'
plot_df = df[df['ltecell_name'] == cell]
#plot_df = df[(df[ho] < 100) & (df['ltecell_name'] == cell) & (df['daytime'] == 0)]
#plot_df = df[df['ltecell_name'] == cell].drop(['timestamp', 'datetime', 'ltecell_name', 'time_advance_avg', 'traffic_volume_data', 'user_throughput_dl_avg', 'cell_throughput_ul_avg', 'prb_dl_usage_rate_avg'], axis=1)
#print(plot_df['N1L08_prb_dl_usage_rate_avg'].corr(df['N1L08_user_throughput_dl_avg'].sort_values()))
sns.heatmap(plot_df.corr(), annot=True, fmt='.1f', cmap=sns.color_palette("viridis", as_cmap=True))
plt.title('Cell {} correlations before filtering'.format(cell, ho))
# %%
plot_data = df.drop([
                     'datetime',
                     'day',
                     'hour',
                     'cell_throughput_ul_avg',
                     'cell_throughput_dl_avg',
                     'traffic_volume_data'
                     ], axis=1)

sns.pairplot(plot_data, hue='ltecell_name', corner=True)


# %%
#Helper function to divide based on sectors
def divide_sectors(cell_name):
    sector = int(cell_name[1])
    return sector

#Divide into sectors
df['sector'] = df['ltecell_name'].apply(lambda x: divide_sectors(x))

# %%
#Helper function to divide based on frequencies
def divide_freqs(cell_name):
    freq = int(cell_name[-2:])
    return freq

#Divide into frequencies
df['freq'] = df['ltecell_name'].apply(lambda x: divide_freqs(x))

# %%
#Autocorrelation
acs = []
ratios = []
for cell in df['ltecell_name'].unique():
    #print(cell)
    cell_data = df[df['ltecell_name'] == cell]
    acs.append(cell_data['hsr_intra_freq'].autocorr(lag=1))
    ratios.append(len(cell_data[cell_data['hsr_intra_freq'] < 100]) / len(cell_data))
    
cell_list = df['ltecell_name'].unique()
sns.scatterplot(ratios, acs, hue=cell_list)
plt.xlabel('Proportion of hsr_intra < 100%')
plt.ylabel('Autocorrelation hsr_intra (lag=1)')
plt.title('Autocorrelation compared to HSR')

# %%
#Correlations of cellwise features
corr_feats = []
#cellwise_df.drop(['timestamp'], axis=1, inplace=True)
for c in cellwise_df.columns:
    corrs = cellwise_df.corrwith(cellwise_df[c])
    for i, x in enumerate(corrs):
        #If correlation is strong enough between different cell KPIs
        if (c[:5] != corrs[[i]].index[0][:5]) and (c[5:-1] != corrs[[i]].index[0][5:-1]) and abs(x) > 0.55:
            corr_feats.append(c)
            #print(c, corrs[[i]])
print(len(corr_feats))
#sns.heatmap(cellwise_df[corr_feats.unique()].corr(), annot=True, fmt='.1f', cmap=sns.color_palette("viridis", as_cmap=True))
#plt.title('Correlations during day on a working day')

#%%
#CREATE CELLWISE DATAFRAME THAT INCLUDES ALL THE CELLWISE KPIS
cell_groups = df.groupby(['ltecell_name'])
dfs = [group.drop(cat_cols, axis=1).set_index('timestamp') for (name, group) in cell_groups]
cellwise_df = pd.concat(dfs, axis=1, join='outer')

cols = []
for cell in df['ltecell_name'].unique():
    for col in num_cols:
        cols.append(cell + '_' + col)

cellwise_df.columns = cols
#cellwise_df.dropna(inplace=True)
#sns.heatmap(cellwise_df.corr(), annot=False, fmt='.2f', cmap=sns.color_palette("viridis", as_cmap=True))
cellwise_df.to_csv('C:/Users/mirom/Desktop/IST/Thesis/data/cellwise_df.csv')