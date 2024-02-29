#After merging all files into one with all the raw data, the KPIs can be calculated.
# %%
import pandas as pd
import numpy as np
import os

def nullif(x):
    if x == 0:
        return np.nan
    else:
        return x

folder_path = 'C:/Users/mirom/Desktop/IST/Thesis/data/New_data/merged_outer'

# Loop over files in the folder
for filename in os.listdir(folder_path):
    file_path = folder_path + '/' + filename

    df = pd.read_csv(file_path)

    datetime = df['pmm_datetime']
    timestamp = df['timestamp'] 
    #enodeb_name = df['enodeb_name_x']
    ltecell_name = df['ltecell_name']
    users_avg = []
    prb_dl_usage_rate_avg = []
    prb_ul_usage_rate_avg = []
    hsr_intra_freq = []
    hsr_inter_freq = []
    cell_throughput_dl_avg = []
    cell_throughput_ul_avg = []
    user_throughput_dl_avg = []
    user_throughput_ul_avg = []
    traffic_volume_data = []
    tp_carrier_aggr = []
    carrier_aggr_usage = []
    cqi_avg = []
    time_advance_avg = []

    def convert_to_kpis(row):
        users_avg.append(row['rrcconnlevsum'] / nullif(row['rrcconnlevsamp']))
        prb_dl_usage_rate_avg.append(100 * row['prbuseddldtch'] / nullif(row['prbavaildl']))
        prb_ul_usage_rate_avg.append(100 * row['prbuseduldtch'] / nullif(row['prbavailul']))
        hsr_intra_freq.append(100 * (row['hoprepsucclteintraf'] / nullif(row['hoprepattlteintraf'])) * (row['hoexesucclteintraf'] / nullif(row['hoexeattlteintraf'])))
        hsr_inter_freq.append(100 * (row['hoprepsucclteinterf'] / nullif(row['hoprepattlteinterf'])) * (row['hoexesucclteinterf'] / nullif(row['hoexeattlteinterf'])))
        cell_throughput_dl_avg.append(1000 * row['pdcpvoldldrb'] / nullif(row['schedactivitycelldl']))
        cell_throughput_ul_avg.append(1000 * row['pdcpvoluldrb'] / nullif(row['schedactivitycellul']))
        user_throughput_dl_avg.append(1000 * (row['pdcpvoldldrb'] - row['pdcpvoldldrblasttti']) / nullif(row['uethptimedl']))
        user_throughput_ul_avg.append(1000 * row['pdcpvoluldrb'] / nullif(row['uethptimeul']))
        traffic_volume_data.append(row['pdcpvoldldrb'] / (8.0*1000))
        tp_carrier_aggr.append((row['pmpdcpvoldldrbca'] - row['pmpdcpvoldldrblastttica']) / (nullif(row['uethptimedlca']) / 1000))
        carrier_aggr_usage.append(100 * (row['pmradiothpvoldl2ccpcell'] + row['pmradiothpvoldl3ccpcell'] + row['pmradiothpvoldl4ccpcell'] + row['radiothpvoldlscell']) / nullif(row['radiothpvoldl']))
        #cqi_avg:
        cqis1 = [row['radiouerepcqidistr_cqi{}'.format(n)] for n in range(16)]
        cqis2 = [row['radiouerepcqidistr2_cqi{}'.format(n)] for n in range(16)]
        cqi_numerator = []
        cqi_denumerator = []
        for i, (cqi1, cqi2) in enumerate(zip(cqis1, cqis2)):
            cqi_numerator.append((cqi1 + cqi2) * i)
            cqi_denumerator.append(cqi1 + cqi2)
        cqi_avg.append(np.sum(cqi_numerator) / np.sum(cqi_denumerator))
        #time_advance_avg:
        multipliers = [1350, 2850, 4350, 5850, 7350, 8850, 10350, 11850, 13350, 14850, 16350, 30000] 
        if row['ltecell_name'][-2:] == '08':
             factor = 2
        else:
             factor = 1
        time_denumerator = [row['pmraatttadistr{}'.format(i)] for i in range(12)]
        time_numerator = [x * y * factor for (x, y) in zip(time_denumerator, multipliers)]
        time_advance_avg.append(np.sum(time_numerator) / np.sum(time_denumerator))


    df.apply(lambda row: convert_to_kpis(row), axis=1)

    kpi_df = pd.DataFrame({
        'datetime': datetime, 
        'timestamp': timestamp,
        #'enodeb_name': enodeb_name,
        'ltecell_name': ltecell_name,
        'users_avg': users_avg,
        'prb_dl_usage_rate_avg': prb_dl_usage_rate_avg,
        'prb_ul_usage_rate_avg': prb_ul_usage_rate_avg,
        'hsr_intra_freq': hsr_intra_freq,
        'hsr_inter_freq': hsr_inter_freq,
        'cell_throughput_dl_avg': cell_throughput_dl_avg,
        'cell_throughput_ul_avg': cell_throughput_ul_avg,
        'user_throughput_dl_avg': user_throughput_dl_avg,
        'user_throughput_ul_avg': user_throughput_ul_avg,
        'traffic_volume_data': traffic_volume_data,
        'tp_carrier_aggr': tp_carrier_aggr,
        'carrier_aggr_usage': carrier_aggr_usage,
        'cqi_avg': cqi_avg,
        'time_advance_avg': time_advance_avg
    })

    kpi_df.to_csv('C:/Users/mirom/Desktop/IST/Thesis/data/New_data/KPIs/{}'.format(filename), index=False)
# %%
