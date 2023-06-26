# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Read csv-files
'''
traffic = pd.read_csv('./data/CEIRA_LTE_MCO001B2/codas_eri_lte_traffic_h.csv')
ho_ecell = pd.read_csv('./data/CEIRA_LTE_MCO001B2/codas_eri_lte_ho_ecell_h.csv')
pdcp = pd.read_csv('./data/CEIRA_LTE_MCO001B2/codas_eri_lte_pdcp_h.csv')
rrc_sig = pd.read_csv('./data/CEIRA_LTE_MCO001B2/codas_eri_lte_rrc_sig_h.csv')
ca = pd.read_csv('./data/CEIRA_LTE_MCO001B2/codas_eri_lte_ca_h.csv')
eut_6 = pd.read_csv('data/CEIRA_LTE_MCO001B2/codas_eri_lte_eutranfdd_6_h.csv')
eut_9 = pd.read_csv('data/CEIRA_LTE_MCO001B2/codas_eri_lte_eutranfdd_9_h.csv')
glob = pd.read_csv('data/CEIRA_LTE_MCO001B2/codas_eri_lte_global_h.csv')
'''
folder = 'C:/Users/mirom/Desktop/IST/Thesis/data'
traffic = pd.read_csv(folder + '/New_data/eri_lte_traffic_h.csv')
ho_ecell = pd.read_csv(folder + '/New_data/eri_lte_ho_ecell_h.csv')
pdcp = pd.read_csv(folder + '/New_data/eri_lte_pdcp_h.csv')
rrc_sig = pd.read_csv(folder + '/New_data/eri_lte_rrc_sig_h.csv')
ca = pd.read_csv(folder + '/New_data/eri_lte_ca_h.csv')
eut_6 = pd.read_csv(folder + '/New_data/eri_lte_eutranfdd_6_h.csv')
eut_9 = pd.read_csv(folder + '/New_data/eri_lte_eutranfdd_9_h.csv')
glob = pd.read_csv(folder + '/New_data/eri_lte_global_h.csv')

for enodeb in traffic['enodeb_name'].unique():
    print(enodeb)
    #Group based on lte cell
    tgroups = traffic[traffic['enodeb_name'] == enodeb].groupby(['ltecell_name'])
    hgroups = ho_ecell[ho_ecell['enodeb_name'] == enodeb].groupby(['ltecell_name'])
    pgroups = pdcp[pdcp['enodeb_name'] == enodeb].groupby(['ltecell_name'])
    rgroups = rrc_sig[rrc_sig['enodeb_name'] == enodeb].groupby(['ltecell_name'])
    cagroups = ca[ca['enodeb_name'] == enodeb].groupby(['ltecell_name'])
    e6groups = eut_6[eut_6['enodeb_name'] == enodeb].groupby(['ltecell_name'])
    e9groups = eut_9[eut_9['enodeb_name'] == enodeb].groupby(['ltecell_name'])
    glgroups = glob[glob['enodeb_name'] == enodeb].groupby(['ltecell_name'])

    dfs = []
    #For each cell
    for name, tgroup in tgroups:
        #To filter cells with less than 50 records
        if len(tgroup) > 50:

            hgroup = hgroups.get_group(name)
            pgroup = pgroups.get_group(name)
            rgroup = rgroups.get_group(name)
            cagroup = cagroups.get_group(name)
            e6group = e6groups.get_group(name)
            e9group = e9groups.get_group(name)
            glgroup = glgroups.get_group(name)

            #Merge dataframes. Resulting is a dataframe containing all the columns from all dataframes.

            merged_group = pd.merge(e9group, tgroup, on='pmm_datetime', how='outer', suffixes=('_y', '_y'))
            merged_group = pd.merge(merged_group, hgroup, on='pmm_datetime', how='outer', suffixes=('_y', '_y'))
            merged_group = pd.merge(merged_group, pgroup, on='pmm_datetime', how='outer', suffixes=('_y', '_y'))
            merged_group = pd.merge(merged_group, rgroup, on='pmm_datetime', how='outer', suffixes=('_y', '_y'))
            merged_group = pd.merge(merged_group, cagroup, on='pmm_datetime', how='outer', suffixes=('_y', '_y'))
            merged_group = pd.merge(merged_group, e6group, on='pmm_datetime', how='outer', suffixes=('_y', '_y'))
            merged_group = pd.merge(merged_group, glgroup, on='pmm_datetime', how='outer', suffixes=('_y', '_y'))
            merged_group.insert(0, 'ltecell_name', name)
            timestamps = pd.to_datetime(merged_group['pmm_datetime'])
            merged_group.insert(0, 'timestamp', timestamps) 
            merged_group['timestamp'].sort_values()
            dfs.append(merged_group)

    merged_df = pd.concat(dfs)

    #Remove redundant columns
    for c in merged_df.columns.unique():
        if '_y' in c:
            merged_df.drop(c, axis=1, inplace=True)

    merged_df.to_csv('C:/Users/mirom/Desktop/IST/Thesis/data/New_data/merged_outer/{}.csv'.format(enodeb), index=False)




# %%
