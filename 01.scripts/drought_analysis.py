"""
# Drought Analysis US

## Author
Name: Savalan Naser Neisary
Email: savalan.neisary@gmail.com | snaserneisary@crimson.ua.edu
Affiliation: PhD Student, University of Alabama
GitHub: savalann

## Code Description
Creation Date: 2023-02-23
This code downloads the USGS streamflow data for different states under a specific folder structure.
## License
This software is licensed under the Apache License 2.0. See the LICENSE file for more details.
"""

# %% importing the libraries

# basic packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# geo packages
# from shapely.geometry import Point
# from pyproj import CRS
# import geopandas as gpd

# system packages
import glob
import os
import platform
import warnings
warnings.filterwarnings("ignore")
import scipy.stats as stats

# my packages
from pyndat import Pyndat


# %% platform detection and address assignment

if platform.system() == 'Windows':

    onedrive_path = 'E:/OneDrive/OneDrive - The University of Alabama/10.material/01.data/usgs_data/'

    onedrive_path_new = ('E:/OneDrive/OneDrive - The University of Alabama/02.projects'
                         '/02.nidis/02.code/03.resutls/')
    onedrive_path_new_data = 'E:/OneDrive/OneDrive - The University of Alabama/02.projects/02.nidis/02.code/02.data/'


    box_path = 'C:/Users/snaserneisary/Box/Evaluation/Data_1980-2020/NWIS_sites/'

elif platform.system() == 'Darwin':

    onedrive_path = '/Users/savalan/Library/CloudStorage/OneDrive-TheUniversityofAlabama/10.material/01.data/usgs_data/'

    onedrive_path_new = ('/Users/savalan/Library/CloudStorage/OneDrive-TheUniversityofAlabama/02.projects'
                         '/02.nidis/02.code/03.resutls/')
    onedrive_path_new_data = '/Users/savalan/Library/CloudStorage/OneDrive-TheUniversityofAlabama/02.projects/02.nidis/02.code/02.data/'







#%%
path = ('E:/OneDrive/OneDrive - The University of Alabama/02.projects/02.nidis/02.code/nrt-nwm-benchmark-project/03.outputs/')
path_01 = ('E:/OneDrive/OneDrive - The University of Alabama/02.projects/02.nidis/02.code/nrt-nwm-benchmark-project/03.outputs/drought_data/')
#stream_data

end_year = 2020
data_length = 41
start_year = end_year - data_length + 1
duration_list = [i for i in range(2, 11)]
# %%
empirical_distribution_data = {}
drought_severity_data_all = {}
for dsource in ['USGS', 'NWM']:
    # create the variables
    empirical_distribution_data[dsource] = {}  # Empirical cdf data for each state, station, and duration.
    drought_severity_data_all[dsource] = {}  # Empirical cdf data for each state, station, and duration.
    distribution_data = {}  # Analytic cdf data for each state, station, and duration.
    severity_data = {}  # Severity of each state, duration, station, and return period.
    final_output = {}
    start_char = 132


    # set the directory name
    parent_dir = f'{path}stream_data/{dsource}/'
    csv_files = glob.glob(os.path.join(parent_dir, "*.csv"))  # Get the file names in the state directory.

    # Read each file and get data and generate the sdf curve
    for file_name in csv_files:
        raw_df = pd.read_csv(file_name, encoding='unicode_escape')
        raw_df['Datetime'] = pd.to_datetime(raw_df['Datetime'])
        raw_df = raw_df[raw_df[f'{dsource}_flow'] >= 0]
        raw_df = raw_df[(raw_df.Datetime >= f'{start_year}-10-01') & (raw_df.Datetime < f'{end_year}-10-01')]
        if dsource == 'NWM':
            raw_df.rename(columns={'NWM_flow': 'USGS_flow'}, inplace=True)
            start_char = 131
        temp_df_01 = Pyndat.sdf_creator(data=raw_df, figure=False)
        all_sdf, drought_severity_data = temp_df_01[0], temp_df_01[3]
        empirical_distribution_data[dsource][file_name[start_char:-4]] = all_sdf
        drought_severity_data_all[dsource][file_name[start_char:-4]] = drought_severity_data

    for key, values in empirical_distribution_data[dsource].items():
        writer = pd.ExcelWriter(f'{path_01}/{dsource}/{key}.xlsx', engine='xlsxwriter')
        df_temp_01 = empirical_distribution_data[dsource][key]
        for duration_num in duration_list:
            df_temp_02 = df_temp_01[f'Duration={duration_num}'].dropna()
            df_temp_02.to_excel(writer, sheet_name=str(duration_num))

        writer.close()


#%%

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import kendalltau


station_list = pd.read_excel(f'{path}/final_modified.xlsx')
station_list = station_list.iloc[:, 1:3].values

# def analysis_correlation(all_sdf, empirical_distribution_data):


temp_result = np.zeros((len(drought_severity_data_all[dsource]), 9))
p_correlation = np.zeros((len(drought_severity_data_all[dsource]), 9))
p_trend = np.zeros((len(drought_severity_data_all[dsource]), 9))
score_correlation = np.zeros((len(drought_severity_data_all[dsource]), 9))
score_trend = np.zeros((len(drought_severity_data_all[dsource]), 9))

drought_number = []
drought_descriptive = []

for station_index in range(len(station_list)):
    temp_drought_number = []
    temp_drought_descriptive = []
    for duration_number in duration_list:

        station_usgs = station_list[station_index, 0]
        station_nwm = station_list[station_index, 1]
        data_usgs = (drought_severity_data_all['USGS'][str(station_usgs)][f'Duration={duration_number}']).dropna()
        data_nwm = (drought_severity_data_all['NWM'][str(station_nwm)][f'Duration={duration_number}']).dropna()

        temp_data_merged = pd.merge(data_usgs, data_nwm, on='Date')
        score_correlation[station_index, duration_number-2], p_correlation[station_index, duration_number-2] =\
            spearmanr(temp_data_merged['Severity(%)_x'], temp_data_merged['Severity(%)_y'])

        score_trend[station_index, duration_number-2], p_trend[station_index, duration_number-2] =\
            kendalltau(temp_data_merged['Severity(%)_x'], temp_data_merged['Severity(%)_y'])

        temp_drought_number.append((temp_data_merged['Severity(%)_x'] < 0).sum())
        temp_drought_number.append((temp_data_merged['Severity(%)_y'] < 0).sum())

        temp_drought_descriptive.append(temp_data_merged[temp_data_merged['Severity(%)_x'] < 0]['Severity(%)_x'].max())
        temp_drought_descriptive.append(temp_data_merged[temp_data_merged['Severity(%)_x'] < 0]['Severity(%)_x'].median())
        temp_drought_descriptive.append(temp_data_merged[temp_data_merged['Severity(%)_x'] < 0]['Severity(%)_x'].min())

        temp_drought_descriptive.append((temp_data_merged['Severity(%)_y'] < 0).max())
        temp_drought_descriptive.append((temp_data_merged['Severity(%)_y'] < 0).min())
        temp_drought_descriptive.append((temp_data_merged['Severity(%)_y'] < 0).median())

    drought_number.extend([temp_drought_number])
    drought_descriptive.extend([temp_drought_descriptive])




    # for col_ind in range(0, 27, 3):




aa = np.array(drought_descriptive)





















#%%

#
# for duration_number in [i for i in range(2, 11)]:
#     temp_output['correlation'] = 0
#     temp_output['trend'] = 0
#     temp_output['event_number'] = 0
#
#     for station_index, station_number in enumerate(valid_station_numbers):
#         df_temp_01 = empirical_distribution_data[station_number][f'Duration={duration_number}']
#         df_temp_02 = df_temp_01.dropna()
#
#         temp_output.iloc[station_index, -3] =
#
#         temp_output.iloc[station_index, -2] =
#
#         temp_output.iloc[station_index, -1] = len(df_temp_02['Severity(%)'])
#
#     final_output[duration_number] = temp_output.copy()
#
