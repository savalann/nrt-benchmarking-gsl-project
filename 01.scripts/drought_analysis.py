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
import pandas as pd
from scipy.stats import spearmanr, kendalltau
import pymannkendall as mk
import matplotlib.pyplot as plt
import seaborn as sns
import math

# system packages
import glob
import os
import platform
import warnings
warnings.filterwarnings("ignore")

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

# %%
path = (
    'E:/OneDrive/OneDrive - The University of Alabama/02.projects/02.nidis/02.code/nrt-nwm-benchmark-project/03.outputs/')
path_01 = (
    'E:/OneDrive/OneDrive - The University of Alabama/02.projects/02.nidis/02.code/nrt-nwm-benchmark-project/03.outputs/drought_data/')
# stream_data

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

# %%


station_list_raw = pd.read_excel(f'{path}/final_modified.xlsx')
station_list = station_list_raw.iloc[:, 1:3].values

# def analysis_correlation(all_sdf, empirical_distribution_data):


temp_result = np.zeros((len(drought_severity_data_all[dsource]), 9))
p_correlation = np.zeros((len(drought_severity_data_all[dsource]), 9))
score_correlation = np.zeros((len(drought_severity_data_all[dsource]), 9))
p_trend = []
slope_trend = []
drought_number = []
drought_descriptive = []

for station_index in range(len(station_list)):
    temp_p_trend = []
    temp_slope_trend = []
    temp_drought_number = []
    temp_drought_descriptive = []
    for duration_number in duration_list:
        station_usgs = station_list[station_index, 0]
        station_nwm = station_list[station_index, 1]
        data_usgs = (drought_severity_data_all['USGS'][str(station_usgs)][f'Duration={duration_number}']).dropna()
        data_nwm = (drought_severity_data_all['NWM'][str(station_nwm)][f'Duration={duration_number}']).dropna()

        temp_data_merged = pd.merge(data_usgs, data_nwm, on='Date')
        score_correlation[station_index, duration_number - 2], p_correlation[station_index, duration_number - 2] = \
            spearmanr(temp_data_merged['Severity(%)_x'], temp_data_merged['Severity(%)_y'])

        temp_p_trend.append(mk.original_test(temp_data_merged['Severity(%)_x'])[0])
        temp_p_trend.append(mk.original_test(temp_data_merged['Severity(%)_y'])[0])

        temp_slope_trend.append(round(mk.original_test(temp_data_merged['Severity(%)_x'])[7], 2))
        temp_slope_trend.append(round(mk.original_test(temp_data_merged['Severity(%)_y'])[7], 2))

        temp_drought_number.append((temp_data_merged['Severity(%)_x'] < 0).sum())
        temp_drought_number.append((temp_data_merged['Severity(%)_y'] < 0).sum())


        temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_x'] < 0]
                                              ['Severity(%)_x'].max(), 2))
        temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_x'] < 0]
                                              ['Severity(%)_x'].median(), 2))
        temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_x'] < 0]
                                              ['Severity(%)_x'].min(), 2))

        temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_y'] < 0]
                                              ['Severity(%)_y'].max(), 2))
        temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_y'] < 0]
                                              ['Severity(%)_y'].median(), 2))
        temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_y'] < 0]
                                              ['Severity(%)_y'].min(), 2))


    p_trend.extend([temp_p_trend])
    slope_trend.extend([temp_slope_trend])
    drought_number.extend([temp_drought_number])
    drought_descriptive.extend([temp_drought_descriptive])

index_1 = []
index_2 = []
index_3 = []
index_4 = []
for ii in range(2, 11):
    index_4.append(f'D{ii}')
    for name_col in ['USGS', 'NWM']:
        index_1.append(f'D{ii}_{name_col}')
        for name_metric in ['min', 'median', 'max']:
            index_2.append(f'D{ii}_{name_metric}_{name_col}')

drought_number = pd.DataFrame(drought_number, columns=index_1)
final_drought_number = pd.concat([station_list_raw.iloc[:, 1:3], drought_number], axis=1)

drought_descriptive = pd.DataFrame(drought_descriptive, columns=index_2) * -1
final_drought_descriptive = pd.concat([station_list_raw.iloc[:, 1:3], drought_descriptive], axis=1)

p_trend = pd.DataFrame(p_trend, columns=index_1)
final_p_trend = pd.concat([station_list_raw.iloc[:, 1:3], p_trend], axis=1)

slope_trend = pd.DataFrame(slope_trend, columns=index_1)
final_slope_trend = pd.concat([station_list_raw.iloc[:, 1:3], slope_trend], axis=1)


score_correlation = pd.DataFrame(np.round(score_correlation, 2), columns=index_4)
final_score_correlation = pd.concat([station_list_raw.iloc[:, 1:3], score_correlation], axis=1)

p_correlation = pd.DataFrame(np.round(p_correlation, 2), columns=index_4)
p_correlation = p_correlation.applymap(lambda x: 'significant' if x <= 0.05 else ' not significant')
final_p_correlation = pd.concat([station_list_raw.iloc[:, 1:3], p_correlation], axis=1)

final_drought_number.to_csv(f'{path_01}number_of_drought_events.csv')
final_drought_descriptive.to_csv(f'{path_01}descriptive_info.csv')
final_p_trend.to_csv(f'{path_01}trend_p_value.csv')
final_slope_trend.to_csv(f'{path_01}trend_slope.csv')
final_score_correlation.to_csv(f'{path_01}correlation_score.csv')
final_p_correlation.to_csv(f'{path_01}correlation_p_value.csv')

#%%
"""
Savy: The function in this cell will create different plots, such as heatmaps for number of drought events, correlation,
and trend, and    

"""

def create_box_plot(datasets, save_path, figsize=(12, 12)):
    n_subplots = len(datasets)
    n_cols = int(math.ceil(math.sqrt(n_subplots)))
    n_rows = int(math.ceil(n_subplots / n_cols))
    key_name = list(datasets.keys())
    # Using sharey=True and sharex=True
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True, sharex=True, dpi=300)
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, ax in enumerate(axes):
        if i < n_subplots:
            # Plotting the data with labels for the legend
            ax.boxplot(datasets[key_name[i]], notch=False, patch_artist=True, labels=['USGS', 'NWM'])
            ax.set_title(f'{key_name[i]}')
            # Setting the x-axis label for the last row
            # if i // n_cols == n_rows - 1:
            #     ax.set_xlabel('Name of Dataset')

            # Setting the y-axis label for the first column
            if i % n_cols == 0:
                ax.set_ylabel('Severity(%)')
        else:
            # Hide unused subplots
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_path}severity_boxplot.png')
    plt.show()


# Example usage

temp_data_merged = {}


for duration_number in duration_list:
    for station_index in range(len(station_list)):
        station_usgs = station_list[station_index, 0]
        station_nwm = station_list[station_index, 1]
        data_usgs = ((empirical_distribution_data['USGS'][str(station_usgs)][f'Duration={duration_number}'])
                     .dropna() * -1)
        data_nwm = (empirical_distribution_data['NWM'][str(station_nwm)][f'Duration={duration_number}']).dropna() * -1
        temp_data_merged[f'U:{station_usgs}-N:{station_nwm}'] = pd.merge(data_usgs, data_nwm, on='Date')
        temp_data_merged[f'U:{station_usgs}-N:{station_nwm}'] = temp_data_merged[f'U:{station_usgs}-N:{station_nwm}'][['Severity(%)_x', 'Severity(%)_y']]

    create_box_plot(temp_data_merged, f'{path_01}d{duration_number}_')



#%%

column_names = ['USGS', 'NWM']



fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 6), dpi=300)

sns.heatmap(final_drought_number[['D2_USGS', 'D2_NWM']], fmt=".0f", cmap="Oranges", annot=True, ax=axes[0], cbar=True,
            xticklabels=column_names, yticklabels=final_drought_number['NWISid'], cbar_kws={"orientation": "horizontal", "pad": .05})
axes[0].set_title('2-Years Duration')
sns.heatmap(final_drought_number[['D10_USGS', 'D10_NWM']], fmt=".0f", cmap="Blues", annot=True, ax=axes[1], cbar=True,
            xticklabels=column_names, yticklabels=final_drought_number['NHDPlusid'], cbar_kws={"orientation": "horizontal", "pad": .05})


# Get the current axes object
axes[1] = plt.gca()

# Move the y-axis labels to the right
axes[1].yaxis.tick_right()
axes[1].yaxis.set_label_position("right")
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)
axes[1].set_title('10-Years Duration')

# Adjust layout
plt.tight_layout()
plt.savefig(f'{path_01}drought_number_heatmap.png')
# Show the plot
plt.show()


