"""
# Drought Analysis US

## Author
Name: Savalan Naser Neisary
Email: savalan.neisary@gmail.com | snaserneisary@crimson.ua.edu
Affiliation: PhD Student, University of Alabama
GitHub: savalann

## Code Description
Creation Date: 2023-02-23
This code adds number of years, missing value, and the ending year of time series according to the downloaded file.

## License
This software is licensed under the Apache License 2.0. See the LICENSE file for more details.
"""

# %% savalan: Import basic libraries
import numpy as np
import datetime
import pandas as pd
import glob
import os
import platform
from datetime import datetime
import matplotlib.pyplot as plt

from hydrotools.nwm_client import utils

import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS
# %% savalan: Platform detection and address assignment
if platform.system() == 'Windows':

    onedrive_path_input_data = 'E:/OneDrive/OneDrive - The University of Alabama/10.material/01.data/test_usgs_data/'

    box_path_input_data = 'C:/Users/snaserneisary/Box/Evaluation/Data_1980-2020/NWM_segments/'

elif platform.system() == 'Darwin':

    onedrive_path_input_data = '/Users/savalan/Library/CloudStorage/OneDrive-TheUniversityofAlabama/10.material/01' \
                               '.data/usgs_data/ '

# %% savalan: Functions
def statistics(state_name, start_year, end_year, path):
    # savalan: Get the name of streamflow data files.
    parent_dir = path + state_name
    csv_files = glob.glob(os.path.join(parent_dir, "*.csv"))

    # savalan: Make an empty list and dictionary for the requested state and stations.
    station_name_list = []
    station_stat_list = np.zeros([len(csv_files), 3])

    # savalan: Get time series of each station and calculate the statistics.
    for file_index, files in enumerate(csv_files):
        raw_df = pd.read_csv(files, encoding='unicode_escape')
        raw_df['Datetime'] = pd.to_datetime(raw_df['Datetime'])
        raw_df = raw_df[(raw_df['Datetime'].dt.year >= start_year) & (raw_df['Datetime'].dt.year <= end_year)]

        # savalan: Extract the station number out of the file name.
        station_name_list.append(files.replace(parent_dir, '')[1:-14])
        if len(raw_df) > 0:
            # savalan: Get number of years
            station_stat_list[file_index, 0] = len(raw_df['Datetime'].dt.year.drop_duplicates())

            # savalan: Get the missing data percentage.
            station_stat_list[file_index, 1] = abs(
                np.round(((datetime(end_year, 12, 31) - datetime(start_year, 1, 1)).days - len(raw_df)) /
                         (datetime(end_year, 12, 31) - datetime(start_year, 1, 1)).days * 100, 0))

            # savalan: Get the last year of the time series.
            station_stat_list[file_index, 2] = str(raw_df.iloc[-1, 0].year)

        else:
            station_stat_list[file_index, 0] = np.nan
            station_stat_list[file_index, 1] = np.nan
            station_stat_list[file_index, 2] = np.nan

    # Make a dataframe and dictionary for the output data.
    state_statistics = pd.DataFrame({'station': station_name_list, 'year_number': station_stat_list[:, 0],
                                     'missing_value_percent': station_stat_list[:, 1],
                                     'last_year': station_stat_list[:, 2]})
    return state_statistics


def study_area_stations(state_name, start, end, path):
    parent_dir = path + state_name
    station_info_raw = pd.read_csv(parent_dir + '/error_stations/station information.csv')
    station_stat = statistics(state_name, start, end, path)
    station_stat_temp = station_info_raw[station_info_raw.site_no.isin(station_stat.station.apply(np.int64))] \
        .reset_index(drop=True)
    station_stat_temp = station_stat_temp.rename(columns={'site_no': 'station'})
    station_stat_temp.pop('agency_cd')
    station_stat_temp['state'] = state_name
    station_stat_temp['end_date'] = pd.to_datetime(station_stat_temp['end_date'])
    station_stat['station'] = station_stat['station'].astype(np.int64, copy=False)
    station_stat_temp = station_stat_temp.merge(station_stat.reset_index(drop=True), on='station')
    station_info_modified = station_stat_temp[(station_stat_temp['huc_cd'] >= 16010000) &
                                              (station_stat_temp['huc_cd'] < 16020300)].reset_index(drop=True)
    station_info_modified = station_info_modified.dropna(subset=['last_year'])
    # &(station_stat_temp['end_date'].dt.year >= 2022)
    return station_info_modified



# Function for getting NWM data
def get_nhd_model_info(NWIS_sites):
    print('Getting NHD reaches')
   #Get NHD reach colocated with NWIS
    NHD_reaches = []
    NWIS_NHDPlus = pd.DataFrame(columns=['NWISid','NHDPlusid'])

    for site in NWIS_sites.station.tolist():



        try:
            NHD_NWIS_df = utils.crosswalk(usgs_site_codes=site)
            NHD_segment = NHD_NWIS_df.nwm_feature_id.values[0]
            NHD_reaches.append(NHD_segment)

        except:
            NHD_segment = np.nan
            NHD_reaches.append(NHD_segment)

    NWIS_NHDPlus['NWISid'] = NWIS_sites.station
    NWIS_NHDPlus['state'] = NWIS_sites.state
    NWIS_NHDPlus['NHDPlusid'] = NHD_reaches

    return NHD_reaches, NWIS_NHDPlus


def get_NWM_data(NWIS_NHDPlus, start, end):
    start = f'{start}-01-01'  # Start date.
    end = f'{end}-12-31'  # End date.
    NWM_obs = {}
    reaches = NWIS_NHDPlus['NHDPlusid'].to_list()
    for reach_index, reach in enumerate(reaches):

        if reach > 1:
            r = str(int(reach))
            s = NWIS_NHDPlus.iloc[reach_index, 2]
            reachid = f"{box_path_input_data}{s}/{r}_1980_2020.csv"
            print(reachid)


            # We use try to make sure to remove the USGS gauges that don't have any NWM reach.
            try:
                NWM_obs[r] = pd.read_csv(reachid)
                NWM_obs[r].rename(columns={'Datetime': 'datetime'}, inplace=True)
                NWM_obs[r].set_index('datetime', inplace=True)
                NWM_obs[r] = NWM_obs[r][start:end]
                NWM_obs[r] = pd.DataFrame(NWM_obs[r]['NWM_flow'])
                NWM_obs[r]['station_id'] = NWIS_NHDPlus['NWISid'][NWIS_NHDPlus['NHDPlusid'] == reach].values[0]
                NWM_obs[r].reset_index(inplace=True)
            except:
                NWM_obs[r] = -1000

        else:
            r = str(reach)

    return NWM_obs

#%%
end_year = 2020
data_length = 41
start_year = end_year - data_length + 1
state_station_list = {}
temp_df = []
writer = pd.ExcelWriter('final_stations.xlsx', engine='xlsxwriter')
for state in ['ut', 'id', 'wy', 'nv']:
    state_station_all = study_area_stations(state, start_year, end_year, onedrive_path_input_data)
    temp_df.append(state_station_all)
    state_station_all.to_excel(writer, sheet_name=state)
all_station_info_modified = pd.concat(temp_df, ignore_index=True)
all_station_info_modified.to_excel(writer, sheet_name='all')
writer.close()

data_df_modified = all_station_info_modified[(all_station_info_modified['missing_value_percent']
                                              <= 10)].reset_index(drop=True)


data_df_modified['station'] = data_df_modified['station'].astype(str)  # Change the column dtype.
NWIS_NHDPlus = get_nhd_model_info(data_df_modified)[1].dropna()

data_df_modified.rename(columns={'station': 'NWISid'}, inplace=True)
final_data_set = pd.merge(NWIS_NHDPlus.iloc[:, :-1], data_df_modified, on=['NWISid'], how='inner')


NWM_obs = get_NWM_data(NWIS_NHDPlus, start_year, end_year)


wrong_reach_list = []

for station_num in NWM_obs.keys():
    if isinstance(NWM_obs[station_num], int):
        print(station_num)
        var_temp_01 = NWIS_NHDPlus.NWISid[NWIS_NHDPlus.NHDPlusid == float(station_num)].item()
        final_data_set = final_data_set[final_data_set['NWISid'] != str(var_temp_01)]
        wrong_reach_list.append(station_num)


NWM_obs_modified = {key: value for key, value in NWM_obs.items() if key not in wrong_reach_list}




final_data_set.reset_index(drop=True).to_excel('final_modified.xlsx')


#%%

geometry = [Point(xy) for xy in zip(final_data_set['dec_long_va'], final_data_set['dec_lat_va'])]
nad83_crs = CRS.from_epsg(4269)  # NAD83 EPSG code
gdf_point = gpd.GeoDataFrame(final_data_set, geometry=geometry, crs=nad83_crs)

gdf = gpd.read_file('Shape/WBDHU8.shp')
gdf_01 = gpd.read_file('Shape/WBDHU8_01.shp')
gdf_all = gpd.GeoDataFrame(pd.concat([gdf, gdf_01], ignore_index=True), crs=gdf.crs)
lake = gpd.read_file('Shape/NHDWaterbody.shp')
lake = lake[lake.gnis_name.isin(['Great Salt Lake', 'Utah Lake'])]

gdf_all = gdf_all[gdf_all.huc8 < '16020300']
fig, ax = plt.subplots(dpi=300, figsize=(6, 8))


# Plot the polygons and color them based on your data
gdf_all.plot(facecolor='gray', edgecolor='0.8', linewidth=0.8, ax=ax)
lake.plot(linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
gdf_point.plot(ax=ax, markersize=45, legend=True, color='red',
                legend_kwds={'shrink': 0.5, 'label': 'Severity'})

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.title(f'Station Location Map')
plt.legend()
# Add a title and labels
plt.tight_layout()
plt.savefig(f'Station Location Map.png')
plt.show()
