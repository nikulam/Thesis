#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
traffic = pd.read_csv('./data/codas_eri_lte_traffic_h.csv')
ho_ecell = pd.read_csv('./data/codas_eri_lte_ho_ecell_h.csv')
pdcp = pd.read_csv('./data/codas_eri_lte_pdcp_h.csv')
rrc_sig = pd.read_csv('./data/codas_eri_lte_rrc_sig_h.csv')

#%%
sheets = [traffic, ho_ecell, pdcp, rrc_sig]
print(len(sheets[3]['pmm_datetime'].unique()))
print(sheets[1]['ltecell_name'].value_counts())

t = traffic.groupby('ltecell_name')

for group in t:
    print(group)


#%%
#Remove any instance of time with other than 9 data points
traffic = traffic.groupby('pmm_datetime', as_index=False).filter(lambda x : len(x)==9)
rrc_sig = rrc_sig.groupby('pmm_datetime', as_index=False).filter(lambda x : len(x)==9)
ho_ecell = ho_ecell.groupby('pmm_datetime', as_index=False).filter(lambda x : len(x)==9)
pdcp = pdcp.groupby('pmm_datetime', as_index=False).filter(lambda x : len(x)==9)

#%%
#Remove datetimes that don't appear in all the 4 datasets
traffic = traffic[traffic['pmm_datetime'].isin(rrc_sig['pmm_datetime'])]
traffic = traffic[traffic['pmm_datetime'].isin(pdcp['pmm_datetime'])]
traffic = traffic[traffic['pmm_datetime'].isin(ho_ecell['pmm_datetime'])]
print(traffic.shape)
traffic.reset_index(inplace=True)

rrc_sig = rrc_sig[rrc_sig['pmm_datetime'].isin(traffic['pmm_datetime'])]
print(rrc_sig.shape)
rrc_sig.reset_index(inplace=True)

ho_ecell = ho_ecell[ho_ecell['pmm_datetime'].isin(traffic['pmm_datetime'])]
print(ho_ecell.shape)
ho_ecell.reset_index(inplace=True)

pdcp = pdcp[pdcp['pmm_datetime'].isin(traffic['pmm_datetime'])]
print(pdcp.shape)
pdcp.reset_index(inplace=True)

n_windows = traffic.shape[0] / 9
#%%
'''
pd.DataFrame({'t':traffic['pmm_datetime'], 'r':rrc_sig['pmm_datetime'], 'h':ho_ecell['pmm_datetime'], 'p':pdcp['pmm_datetime']}).to_csv('test.csv')

test_df = pd.read_csv('./test.csv')
#print(test_df.iloc[5322])
print(test_df.isna().sum())
'''
#%%
#RRC connections
rrc_conns = []
rrc_conn_windows = np.array_split(rrc_sig['rrcconnlevsum'], n_windows)
rrc_samp_windows = np.array_split(rrc_sig['rrcconnlevsamp'], n_windows)
for i in range(int(n_windows)):
    value = sum(rrc_conn_windows[i] / rrc_samp_windows[i])
    rrc_conns.append(value)

#%%
#PRB DL USAGE
prb_dls = []
prb_dl_used_windows = np.array_split(traffic['prbuseddldtch'], n_windows) 
prb_dl_avail_windows = np.array_split(traffic['prbavaildl'], n_windows)
for i in range(int(n_windows)):
    value = 100 * sum(prb_dl_used_windows[i]) / sum(prb_dl_avail_windows[i])
    prb_dls.append(value)

#%%
#PRB UL USAGE
prb_uls = []
prb_ul_used_windows = np.array_split(traffic['prbuseduldtch'], n_windows) 
prb_ul_avail_windows = np.array_split(traffic['prbavailul'], n_windows)
for i in range(int(n_windows)):
    value = 100 * sum(prb_ul_used_windows[i]) / sum(prb_ul_avail_windows[i])
    prb_uls.append(value)

#%%
#QCIs
qci_features = []
drb_windows = np.array_split(traffic['activeuedrbsamp_avg'], n_windows)

for i in range(1,10):
    qcis = []
    column = 'activeuedlsumqci' + str(i)
    qci_windows = np.array_split(traffic[column], n_windows)

    for i in range(int(n_windows)):
        qci = sum(qci_windows[i]) / sum(drb_windows[i])
        qcis.append(qci)
    
    qci_features.append(qcis)


#%%
#HSR INTRA FREQ
ho_intras = []
ho_intra_suc = np.array_split(ho_ecell['hoprepsucclteintraf'], n_windows) 
ho_intra_att = np.array_split(ho_ecell['hoprepattlteintraf'], n_windows)
hoe_intra_suc = np.array_split(ho_ecell['hoexesucclteintraf'], n_windows)
hoe_intra_att = np.array_split(ho_ecell['hoexeattlteintraf'], n_windows)
for i in range(int(n_windows)):
    value = 100 * (sum(ho_intra_suc[i]) / sum(ho_intra_att[i])) * (sum(hoe_intra_suc[i]) / sum(hoe_intra_att[i]))
    ho_intras.append(value)

#%%
#HSR INTER FREQ
ho_inters = []
ho_inter_suc = np.array_split(ho_ecell['hoprepsucclteinterf'], n_windows) 
ho_inter_att = np.array_split(ho_ecell['hoprepattlteinterf'], n_windows)
hoe_inter_suc = np.array_split(ho_ecell['hoexesucclteinterf'], n_windows)
hoe_inter_att = np.array_split(ho_ecell['hoexeattlteinterf'], n_windows)
for i in range(int(n_windows)):
    value = 100 * (sum(ho_inter_suc[i]) / sum(ho_inter_att[i])) * (sum(hoe_inter_suc[i]) / sum(hoe_inter_att[i]))
    ho_inters.append(value)

#%%
#cell_throughput_dl_avg
cell_tp_dls = []
pdcp_dl = np.array_split(pdcp['pdcpvoldldrb'], n_windows) 
activity_dl = np.array_split(traffic['schedactivitycelldl'], n_windows)
for i in range(int(n_windows)):
    value = 1000 * sum(pdcp_dl[i] / activity_dl[i])
    if i == 4000 or i == 4002:
        print(i, value, pdcp_dl[i], activity_dl[i])
    cell_tp_dls.append(value)

#%%
#cell_throughput_ul_avg
cell_tp_uls = []
pdcp_ul = np.array_split(pdcp['pdcpvoluldrb'], n_windows) 
activity_ul = np.array_split(traffic['schedactivitycellul'], n_windows)
for i in range(int(n_windows)):
    value = 1000 * sum(pdcp_ul[i] / activity_ul[i])
    cell_tp_uls.append(value)

#%%
#User dl throughputs
user_tp_dls = []
pdcp_dl = np.array_split(pdcp['pdcpvoldldrb'], n_windows) 
pdcp_vol_dl = np.array_split(pdcp['pdcpvoldldrblasttti'], n_windows) 
ue_dl = np.array_split(traffic['uethptimedl'], n_windows)
for i in range(int(n_windows)):
    value = (1000 * (sum(pdcp_dl[i]) - sum(pdcp_vol_dl[i])) / sum(ue_dl[i]))
    user_tp_dls.append(value)

#%%
#User ul throughputs
user_tp_uls = []
pdcp_ul = np.array_split(pdcp['pdcpvoluldrb'], n_windows) 
ue_ul = np.array_split(traffic['uethptimeul'], n_windows)
for i in range(int(n_windows)):
    value = (1000 * sum(pdcp_ul[i]) / sum(ue_ul[i]))
    user_tp_uls.append(value)

#%%
#TRAFFIC VOLUME DATA
traffic_vols = []
pdcp_ul = np.array_split(pdcp['pdcpvoluldrb'], n_windows) 
for i in range(int(n_windows)):
    value = sum(pdcp_ul[i]) / 8.0*1000
    traffic_vols.append(value)

#%%
times = rrc_sig['pmm_datetime'][0::9]
enodebs = rrc_sig['enodeb_name'][0::9]
ltecells = rrc_sig['ltecell_name'][0::9]

print(len(times), len(enodebs), len(ltecells), len(cell_tp_uls), n_windows, traffic.shape[0] /9)
#%%
df = pd.DataFrame({'time':times,
                    'enodeb':enodebs,
                    'ltecells':ltecells,
                    'users_avg':rrc_conns,
                    'prb_dl_usage_rate_avg':prb_dls,
                    'prb_ul_usage_rate_avg':prb_uls,
                    'hsr_intra_freq':ho_intras,
                    'hsr_inter_freq':ho_inters,
                    'cell_throughput_dl_avg':cell_tp_dls,
                    'cell_throughput_ul_avg':cell_tp_uls,
                    'user_throughput_dl_avg':user_tp_dls,
                    'user_throughput_ul_avg':user_tp_uls,
                    'traffic_volume_data':traffic_vols,
                    'users_qci1':qci_features[0],
                    'users_qci2':qci_features[1],
                    'users_qci3':qci_features[2],
                    'users_qci4':qci_features[3],
                    'users_qci5':qci_features[4],
                    'users_qci6':qci_features[5],
                    'users_qci7':qci_features[6],
                    'users_qci8':qci_features[7],
                    'users_qci9':qci_features[8]
                    })

df.to_csv('KPIs.csv', index=False)
#%%

