#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cv2
import os
from matplotlib.lines import Line2D

#%%
df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/data/KPIs1_outer2.csv')
#Remove nulls and infs
#df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
#df.dropna(inplace=True)

# %%
folder_path = 'C:/Users/mirom/Desktop/IST/Thesis/data/New_data/KPIs'
ref_kpi = 'traffic_volume_data'

x = 7
y = 10
fig, ax = plt.subplots(y, x, figsize=(25,25))

n = 0
for i, file in enumerate(os.listdir(folder_path)):
    df = pd.read_csv('C:/Users/mirom/Desktop/IST/Thesis/data/New_data/KPIs/{}'.format(file))
    df['timestamp'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.weekday
    df['workday'] = [0 if 4 < x else 1 for x in df['day']]
    
    # Helper function to divide based on sectors
    def divide_sectors(cell_name):
        sector = int(cell_name[-4])
        return sector
    # Divide into sectors
    df['sector'] = df['ltecell_name'].apply(lambda x: divide_sectors(x))

    try:
        # Loop over each three sectors
        for j, sec in enumerate(df['sector'].unique()):
            sec_df_week = df[(df['sector'] == sec) & (df['workday'] == 1) ].groupby('timestamp', as_index=False).sum(numeric_only=True)
            sec_df_end = df[(df['sector'] == sec) & (df['workday'] == 0) ].groupby('timestamp', as_index=False).sum(numeric_only=True)

            print(n)
            #fig.suptitle('Daily traffic volume data in sector {} {}'.format(sec, file[:-4]), fontsize='xx-large')
            x_index = n % x
            y_index = n // x
            sns.lineplot(data=sec_df_week, x='hour', y='traffic_volume_data', estimator='mean', ci=False, ax=ax[x_index, y_index])
            sns.lineplot(data=sec_df_end, x='hour', y='traffic_volume_data', estimator='mean', ci=False, ax=ax[x_index, y_index])
            ax[x_index, y_index].set_title(file[:-4] + ' ' + sec)
            n += 1
            print(n)
            #ax[i, j].plot(X_pad.loc[index], color='r')
            #ax[i, j].plot(den_pad.loc[index], color='b', linestyle='--')
            
    except Exception:
        continue

plt.show()
# %%
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
    'carrier_aggr_usage',
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

#Helper function to divide based on sectors
def divide_sectors(cell_name):
    sector = int(cell_name[1])
    return sector

#Divide into sectors
df['sector'] = df['ltecell_name'].apply(lambda x: divide_sectors(x))

#Create new features
df['daytime'] = [1 if (8 <= x) and (x < 20) else 0 for x in df['hour']]
df['workday'] = [0 if 4 < x else 1 for x in df['day']]

# %%
timestamp = df['timestamp']
data = df[df['ltecell_name'] == 'N1L18']
data = df.drop(['timestamp', 'datetime', 'ltecell_name', 'hsr_intra_freq', 'hsr_inter_freq', 'daytime', 'workday', 'freq', 'sector',  'prb_ul_usage_rate_avg', 'user_throughput_ul_avg', 'cell_throughput_ul_avg', 'time_advance_avg'], axis=1).dropna()
print(len(data.columns))
# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Perform PCA on the data
pca = PCA(n_components=5)
principal_components = pca.fit_transform(data_scaled)

print(pca.explained_variance_ratio_)
print(principal_components)

# Plot the data
plt.scatter(principal_components[:,0], principal_components[:,1])
plt.xlabel('Timestamp')
plt.ylabel('1st Principal Component')
plt.show()

# %%
print(df.shape)

'''
for n in num_cols:
    print(n)
    print('min', np.min(df[n]))
    print('max', np.max(df[n]))
    print('mean', np.mean(df[n]))
'''
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
for cell in df['ltecell_name'].unique():
    print(df[df['ltecell_name'] == cell]['user_throughput_dl_avg'].autocorr(2))
# %%
for cell in df['ltecell_name'].unique():
    cell = cell
    cell_df = df[(df['ltecell_name'] == cell)]
    print(max(cell_df['user_throughput_dl_avg']))
    #Use mean and standard deviation to verify and remove outliers
    threshold = 5
    outlier_columns = ['user_throughput_dl_avg']
    for c in outlier_columns:
        mean1 = np.mean(cell_df[c])
        std1 = np.std(cell_df[c])
        outliers = []
        for n in cell_df[c]:
                z_score= (n - mean1) / std1 
                if np.abs(z_score) > threshold:
                    outliers.append(n)

        for o in outliers:
            print(cell, o)
            #data = data[data[c] != o]
# %%
#Lineplot
default_palette = sns.color_palette(palette='colorblind')
x = 'day'
y = 'traffic_volume_data'
cell = 'N3L21'
plot_df = df[(df['sector'] == 1)]
for c in df['ltecell_name'].unique():
    print(len(df[df['ltecell_name'] == c]['user_throughput_dl_avg']))
print(df.columns)
#plt.plot(plot_df[x], plot_df[x])
#sns.kdeplot(data=df, x='user_throughput_ul_avg', hue='ltecell_name')
#sns.histplot(data=df, x='user_throughput_ul_avg', hue='ltecell_name')
#sns.histplot(data=df, x='user_throughput_ul_avg', hue='ltecell_name', bins=len(df), 
#             stat="density", element="step", fill=False, cumulative=True, common_norm=False)
#sns.boxplot(data=df, x='ltecell_name', y='user_throughput_ul_avg', hue='ltecell_name')
#sns.scatterplot(data=df, x='timestamp', y='traffic_volume_data')
sns.set_context("notebook")
sns.lineplot(data=df, x=x, y=y, hue='ltecell_name', estimator='mean', ci=False, legend=True)
#ax.axvline(x=pd.to_datetime('2023-01-30 00:00:00'), c='black', linestyle='--')
#plt.ylabel('number of data points')
palette = sns.color_palette()

# Customizing the legend
colors = [palette[i] for i in range(9)]  # Adjust the indices to your specific color needs
legend_labels = ['Sec1 0.8 GHz', 'Sec1 1.8 GHz', 'Sec1 2.1 GHz', 'Sec2 0.8 GHz', 'Sec2 1.8 GHz', 'Sec2 2.1 GHz', 'Sec3 0.8 GHz', 'Sec3 1.8 GHz', 'Sec3 2.1 GHz']
legend_lines = [Line2D([0], [0], color=color, lw=2) for color in colors]

plt.legend(bbox_to_anchor=(1.01, 1), handles=legend_lines, labels=legend_labels, loc='upper left')

weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

plt.xticks(ticks=range(7), labels=weekday_names, rotation=90)

# Adjusting Y-label
plt.ylabel('Data traffic volume (MB)')
#plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
#plt.title('One week of {} in cell {}'.format(y, cell))
#plt.xlim(0, 25000)

# %%
print(plt.rcParams['font.family'])
# %%
#Correlation matrix
#plot_df = df[df['daytime'] == 0].drop(['hour', 'day', 'daytime'], axis=1)
#fig, axis = plt.subplots(1, 3, figsize=(15, 5))
#ax = axis[0]
sec = 3
wd = 0
day = 'workday' if wd == 1 else 'weekend'
#plot_df = df[df['ltecell_name'] == cell]
plot_df = df[(df['sector'] == sec)].drop(['hour', 'day', 'sector', 'carrier_aggr_usage', 'daytime', 'workday'], axis=1)
#plot_df = df[(df[ho] < 100) & (df['ltecell_name'] == cell) & (df['daytime'] == 0)]
#plot_df = df[df['ltecell_name'] == cell].drop(['timestamp', 'datetime', 'ltecell_name', 'time_advance_avg', 'traffic_volume_data', 'user_throughput_dl_avg', 'cell_throughput_ul_avg', 'prb_dl_usage_rate_avg'], axis=1)
#print(plot_df['N1L08_prb_dl_usage_rate_avg'].corr(df['N1L08_user_throughput_dl_avg'].sort_values()))
sns.set(font_scale=1.0)
sns.heatmap(plot_df.corr(), annot=True, fmt='.1f', cmap=sns.color_palette("viridis", as_cmap=True))
plt.title('Sector {} correlations'.format(sec))
# %%
%matplotlib inline
# OpenCV
print('jee0')
image1 = cv2.imread('C:/Users/mirom/Desktop/IST/Thesis/imgs/Test_set/DBSCAN_3hours_anomaly.png')
image2 = cv2.imread('C:/Users/mirom/Desktop/IST/Thesis/imgs/Test_set/DBSCAN_11hours_anomaly.png')
print('jee1')
# Resize the images (optional if they are already of the same size)
#image1 = cv2.resize(image1, (800, 600))
#image2 = cv2.resize(image2, (800, 600))
#image3 = cv2.resize(image3, (800, 600))
print('jee2')
# Horizontally concatenate the images
combined_image = cv2.hconcat([image1, image2])
print('jee3')

# Save the combined image
cv2.imwrite('C:/Users/mirom/Desktop/IST/Thesis/imgs/Test_set/combined_zero_DBSCAN.png', combined_image)
# Display the combined image
cv2.imshow('Combined Image', image1)
plt.show()
#cv2.waitKey(0)
#cv2.destroyAllWindows()



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