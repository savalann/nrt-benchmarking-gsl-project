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
path = ('E:/OneDrive/OneDrive - The University of Alabama/02.projects/02.nidis/02.code/Drought-Analysis-US'
        '/02.outputs/')
path_01 = ('E:/OneDrive/OneDrive - The University of Alabama/02.projects/02.nidis/02.code/Drought-Analysis-US'
        '/02.outputs/drought_data')
#stream_data

end_year = 2020
data_length = 41
start_year = end_year - data_length + 1

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
    start_char = 126


    # set the directory name
    parent_dir = f'{path}stream_data/{dsource}/'
    csv_files = glob.glob(os.path.join(parent_dir, "*.csv"))  # Get the file names in the state directory.

    # Read each file and get data and generate the sdf curve
    for file_name in csv_files:

        raw_df = pd.read_csv(file_name, encoding='unicode_escape')
        raw_df['Datetime'] = pd.to_datetime(raw_df['Datetime'])
        raw_df = raw_df[raw_df[f'{dsource}_flow'] >= 0]
        raw_df = raw_df[raw_df.Datetime.dt.year >= start_year]
        if dsource == 'NWM':
            raw_df.rename(columns={'NWM_flow': 'USGS_flow'}, inplace=True)
            start_char = 125
        temp_df_01 = Pyndat.sdf_creator(data=raw_df, figure=False)
        all_sdf, drought_severity_data = temp_df_01[0], temp_df_01[3]
        empirical_distribution_data[dsource][file_name[start_char:-4]] = all_sdf
        drought_severity_data_all[dsource][file_name[start_char:-4]] = drought_severity_data

    for key, values in empirical_distribution_data[dsource].items():
        writer = pd.ExcelWriter(f'{path_01}/{dsource}/{key}.xlsx', engine='xlsxwriter')
        df_temp_01 = empirical_distribution_data[dsource][key]
        for duration in range(2, 11):
            df_temp_02 = df_temp_01[f'Duration={duration}'].dropna()
            df_temp_02.to_excel(writer, sheet_name=str(duration))

        writer.close()


#%%

station_list = pd.read_excel(f'{path}/final_modified.xlsx')
station_list = station_list.iloc[: , 1:3].values

# def analysis_correlation(all_sdf, empirical_distribution_data):

temp_result = np.zeros((len(drought_severity_data_all[dsource]['USGS']), 9))

for station_index in range(len(station_list)):
    usgs_station = station_list.iloc[station_index, 0]
    nwm_station = station_list.iloc[station_index, 1]
    usgs_data =





































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
