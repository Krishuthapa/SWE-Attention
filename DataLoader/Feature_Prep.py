#!/usr/bin/env python
# coding: utf-8


import os
import sys

import numpy as np
import pandas as pd


current_directory = os.getcwd()
data_import_path = current_directory + "/DataLoader/Data/"

snotel_locations = pd.read_csv("{}SNOTEL_CONUS_filtered.csv".format(data_import_path))
slope_aspects = pd.read_csv("{}Slope_Aspect_LandCover.csv".format(data_import_path))

slope_aspects = slope_aspects.rename(columns = {'Station Na' : 'Station Name'})
snotel_locations = snotel_locations.rename(columns = {'Station Na' : 'Station Name'})

filtered_slope_aspect = slope_aspects.loc[slope_aspects['Station Name'].isin(list(snotel_locations['Station Name'].values))]
snotel_locations_info = snotel_locations.merge(filtered_slope_aspect, on='Station Name')

# Does the filtering of locations based on the number of missing data in each year. If missing data for any location in a year is more than 10%
# we exclude that location from the experiment. If its less than 10% we simply interpolate the missing values.

def removeNanAndInterpolate(data,interpolate = False, interpolate_percent  = 10 ,interpolate_window = 10 , start_year = 2001 , end_year = 2019):
    temp_df = data.copy(deep = True)
    
    excluded_columns = []
    for year in range(start_year,end_year +1):
            new_df = temp_df.loc[(temp_df['Date'] >= "{}-01-01".format(year)) & (temp_df['Date'] <= "{}-12-31".format(year)),~temp_df.columns.isin(['Date'])]
            
            value_null_percent = ((new_df.isnull().sum())/365) * 100
            
            data_with_interpolate_percent = value_null_percent[value_null_percent <= interpolate_percent].index.to_list()
            
            if interpolate and len(data_with_interpolate_percent) > 0:
                # Carrying out the interpolation on the final cols selected for interpolation.
                temp_df.loc[new_df.index, data_with_interpolate_percent] = temp_df.loc[new_df.index,data_with_interpolate_percent].fillna(temp_df.loc[new_df.index,data_with_interpolate_percent].rolling(interpolate_window * 2 + 1 , min_periods=1, center=True).mean())
                temp_df.loc[new_df.index, data_with_interpolate_percent] = temp_df.loc[new_df.index,data_with_interpolate_percent].fillna(temp_df.loc[new_df.index,data_with_interpolate_percent].ffill().bfill())
                
                # Filtering the cols with value greater than interpolation threshold and assigning to excluded cols.
                updated_val_null = ((temp_df.loc[new_df.index, :].isnull().sum())/365) * 100
                columns = updated_val_null[updated_val_null > interpolate_percent].index.to_list()
                
                excluded_columns = excluded_columns + columns
                
            else:
                columns = value_null_percent[value_null_percent > 0].index.to_list()
                excluded_columns = excluded_columns + columns
    
    excluded_columns_unique = list(set(excluded_columns))
    
    return (temp_df , excluded_columns_unique)



def getFilteredDataByDate(startDate , endDate , data):
    temp_df = data.copy(deep = True)
    
    temp_df['Date'] = pd.to_datetime(temp_df['Date'], format='%Y-%m-%d')
    new_df = temp_df[(temp_df['Date'] >= startDate) & (temp_df['Date'] <= endDate)]
    
    return new_df


def getFilteredDataBySnotel(data, locations):
    check_columns = list(locations) + ['Date']
    data = data.loc[:,data.columns.isin(check_columns)]
    
    return data


precip_all_data = pd.read_csv("{}Precip_Data_CONUS.csv".format(data_import_path))
min_temp_all_data = pd.read_csv("{}Minimum_temperature_CONUS.csv".format(data_import_path))
max_temp_all_data = pd.read_csv("{}Maximum_temperature_CONUS.csv".format(data_import_path))
obs_temp_all_data = pd.read_csv("{}Temperature_observed_CONUS.csv".format(data_import_path))

TB_19_all_data = pd.read_csv("{}TB_19_VE_CONUS.csv".format(data_import_path))
TB_37_all_data = pd.read_csv("{}TB_37_VE_CONUS.csv".format(data_import_path))
TB_Diff_all_data = pd.read_csv("{}TB_diff_37_19_VE_CONUS.csv".format(data_import_path))

swe_all_data = pd.read_csv("{}SWE_data_CONUS.csv".format(data_import_path))


precip_all_data = precip_all_data.rename(columns={col: col.replace('_', ' ') for col in precip_all_data.columns if col != 'Date'})
min_temp_all_data = min_temp_all_data.rename(columns={col: col.replace('_', ' ') for col in min_temp_all_data.columns if col != 'Date'})
max_temp_all_data = max_temp_all_data.rename(columns={col: col.replace('_', ' ') for col in max_temp_all_data.columns if col != 'Date'})
obs_temp_all_data = obs_temp_all_data.rename(columns={col: col.replace('_', ' ') for col in obs_temp_all_data.columns if col != 'Date'})

TB_19_all_data = TB_19_all_data.rename(columns={col: col.replace('_', ' ') for col in TB_19_all_data.columns if col != 'Date'})
TB_37_all_data = TB_37_all_data.rename(columns={col: col.replace('_', ' ') for col in TB_37_all_data.columns if col != 'Date'})
TB_Diff_all_data = TB_Diff_all_data.rename(columns={col: col.replace('_', ' ') for col in TB_Diff_all_data.columns if col != 'Date'})

swe_all_data = swe_all_data.rename(columns={col: col.replace('_', ' ') for col in swe_all_data.columns if col != 'Date'})


date_filtered_swe = getFilteredDataByDate("2001-01-01", "2019-12-31", swe_all_data)
station_filtered_swe = getFilteredDataBySnotel(date_filtered_swe,snotel_locations_info['Station Name'])

swe_data, swe_exclude_columns = removeNanAndInterpolate(station_filtered_swe, interpolate = True)

date_filtered_precip = getFilteredDataByDate("2001-01-01", "2019-12-31", precip_all_data)
station_filtered_precip = getFilteredDataBySnotel(date_filtered_precip,snotel_locations_info['Station Name'])

precip_data, precip_exclude_columns = removeNanAndInterpolate(station_filtered_precip, interpolate = True)

date_filtered_max_temp = getFilteredDataByDate("2001-01-01", "2019-12-31", max_temp_all_data)
station_filtered_max_temp = getFilteredDataBySnotel(date_filtered_max_temp,snotel_locations_info['Station Name'])

max_temp_data, max_temp_exclude_columns = removeNanAndInterpolate(station_filtered_max_temp, interpolate = True)

date_filtered_obs_temp = getFilteredDataByDate("2001-01-01", "2019-12-31", obs_temp_all_data)
station_filtered_obs_temp = getFilteredDataBySnotel(date_filtered_obs_temp,snotel_locations_info['Station Name'])

obs_temp_data, obs_temp_exclude_columns = removeNanAndInterpolate(station_filtered_obs_temp, interpolate = True)

date_filtered_min_temp = getFilteredDataByDate("2001-01-01", "2019-12-31", min_temp_all_data)
station_filtered_min_temp = getFilteredDataBySnotel(date_filtered_min_temp,snotel_locations_info['Station Name'])

min_temp_data, min_temp_exclude_columns = removeNanAndInterpolate(station_filtered_min_temp,interpolate = True)

date_filtered_19V = getFilteredDataByDate("2001-01-01", "2019-12-31", TB_19_all_data)
station_filtered_19V = getFilteredDataBySnotel(date_filtered_19V,snotel_locations_info['Station Name'])

V_19_data, V_19_exclude_columns = removeNanAndInterpolate(station_filtered_19V,interpolate = True)

date_filtered_37V = getFilteredDataByDate("2001-01-01", "2019-12-31", TB_37_all_data)
station_filtered_37V = getFilteredDataBySnotel(date_filtered_37V,snotel_locations_info['Station Name'])

V_37_data, V_37_exclude_columns = removeNanAndInterpolate(station_filtered_37V,interpolate = True)

date_filtered_TB_Diff = getFilteredDataByDate("2001-01-01", "2019-12-31", TB_Diff_all_data)
station_filtered_TB_Diff = getFilteredDataBySnotel(date_filtered_TB_Diff,snotel_locations_info['Station Name'])

TB_Diff_data, TB_Diff_exclude_columns = removeNanAndInterpolate(station_filtered_TB_Diff,interpolate = True)

columns_to_exclude = precip_exclude_columns + max_temp_exclude_columns + obs_temp_exclude_columns + min_temp_exclude_columns+V_19_exclude_columns+V_37_exclude_columns+TB_Diff_exclude_columns+swe_exclude_columns
columns_to_exclude = list(set(columns_to_exclude))

remaining_columns = list(set(snotel_locations_info['Station Name'].to_list()).difference(columns_to_exclude))

# To see if any data has lesser columns than remaining columns. 
precip_included_columns = list(set(precip_data.columns.to_list()).difference(columns_to_exclude))
max_temp_included_columns = list(set(max_temp_data.columns.to_list()).difference(columns_to_exclude))
min_temp_included_columns = list(set(min_temp_data.columns.to_list()).difference(columns_to_exclude))
obs_temp_included_columns = list(set(obs_temp_data.columns.to_list()).difference(columns_to_exclude))

columns_to_include = list(set(precip_included_columns).intersection(max_temp_included_columns,min_temp_included_columns,obs_temp_included_columns))

precip_collection = precip_data.loc[:, columns_to_include]
precip_collection.set_index('Date', inplace=True)

max_temp_collection = max_temp_data.loc[:,columns_to_include]
max_temp_collection.set_index('Date', inplace=True)

min_temp_collection = min_temp_data.loc[:,columns_to_include]
min_temp_collection.set_index('Date', inplace=True)

obs_temp_collection = obs_temp_data.loc[:,columns_to_include]
obs_temp_collection.set_index('Date', inplace=True)

ve_19_collection = V_19_data.loc[:,columns_to_include]
ve_19_collection.set_index('Date', inplace=True)

ve_37_collection = V_37_data.loc[:,columns_to_include]
ve_37_collection.set_index('Date', inplace=True)

tb_diff_collection = TB_Diff_data.loc[:,columns_to_include]
tb_diff_collection.set_index('Date', inplace=True)

swe_collection = swe_data.loc[:,columns_to_include]
swe_collection.set_index('Date', inplace=True)

precip_collection.to_csv("{}Precipitation_Collection.csv".format(data_import_path), index = ['Date'])
max_temp_collection.to_csv("{}Max_Temp_Collection.csv".format(data_import_path))
min_temp_collection.to_csv("{}Min_Temp_Collection.csv".format(data_import_path))
obs_temp_collection.to_csv("{}Obs_Temp_Collection.csv".format(data_import_path))

ve_19_collection.to_csv("{}VE_19_Collection.csv".format(data_import_path))
ve_37_collection.to_csv("{}VE_37_Collection.csv".format(data_import_path))
tb_diff_collection.to_csv("{}TB_Diff_Collection.csv".format(data_import_path))

swe_collection.to_csv("{}SWE_Collection.csv".format(data_import_path))

snotel_location_filtered_v3 = snotel_locations_info.loc[snotel_locations_info['Station Name'].isin(columns_to_include),:]

snotel_location_filtered_v3.to_csv("{}Snotel_Locations_Filtered_v3.csv".format(data_import_path), index = False)

# Assuming the columns in all the files are same.

def AverageForGivenWindow(data, window, exclude_columns):
    for index, item in data.iterrows():
            data.loc[index,~data.columns.isin(exclude_columns)] = np.mean(data.loc[index-window:index+window+1 ,~data.columns.isin(exclude_columns)])
    return data.loc[:,data.columns.isin(columns_to_include + ['Date'])]

precip_collection = pd.read_csv("{}Precipitation_Collection.csv".format(data_import_path))
max_temp_collection = pd.read_csv("{}Max_Temp_Collection.csv".format(data_import_path))
min_temp_collection = pd.read_csv("{}Min_Temp_Collection.csv".format(data_import_path))
obs_temp_collection= pd.read_csv("{}Obs_Temp_Collection.csv".format(data_import_path))

ve_19_collection = pd.read_csv("{}VE_19_Collection.csv".format(data_import_path))
ve_37_collection = pd.read_csv("{}VE_37_Collection.csv".format(data_import_path))
tb_diff_collection = pd.read_csv("{}TB_Diff_Collection.csv".format(data_import_path))

# Averaging with respect to window in near days.

VE_37_data_avg = AverageForGivenWindow(ve_37_collection,7, ['Date'])
VE_19_data_avg = AverageForGivenWindow(ve_19_collection,7, ['Date'])
TB_Diff_data_avg = AverageForGivenWindow(tb_diff_collection,7, ['Date'])

Precip_data_avg = AverageForGivenWindow(precip_collection,7, ['Date'])
Max_Temp_data_avg = AverageForGivenWindow(max_temp_collection,7, ['Date'])
Min_Temp_data_avg = AverageForGivenWindow(min_temp_collection,7, ['Date'])
Obs_Temp_data_avg = AverageForGivenWindow(obs_temp_collection,7, ['Date'])

VE_19_data_avg.to_csv('{}VE_19_data_avg.csv'.format(data_import_path), index = False)
VE_37_data_avg.to_csv('{}VE_37_data_avg.csv'.format(data_import_path), index = False)
TB_Diff_data_avg.to_csv('{}TB_diff_data_avg.csv'.format(data_import_path), index = False)
Precip_data_avg.to_csv('{}Precipitation_data_avg.csv'.format(data_import_path), index = False)
Max_Temp_data_avg.to_csv('{}Max_temp_data_avg.csv'.format(data_import_path), index = False)
Min_Temp_data_avg.to_csv('{}Min_temp_data_avg.csv'.format(data_import_path), index = False)
Obs_Temp_data_avg.to_csv('{}Obs_temp_data_avg.csv'.format(data_import_path), index = False)

# Averaging across the historical data in the given window of years.

def AddHistoryForGivenWindow(data,year_window, exclude_columns):
    new_df = data.copy(deep=True)
    
    for index, item in new_df.iterrows():
        
        current_year = int(item['Date'].split('-')[0])
        month = item['Date'].split('-')[1]
        day = item['Date'].split('-')[2]
        
        history = [str(year) + '-' + month + '-' + day   for year in range(current_year - year_window , current_year + year_window + 1)]
        
        new_df.loc[index,~new_df.columns.isin(exclude_columns)] = new_df.loc[new_df['Date'].isin(history),~new_df.columns.isin(exclude_columns)].mean()
            
    return new_df.loc[:,data.columns.isin(columns_to_include + ['Date'])]

VE_19_data_avg_H_3 = AddHistoryForGivenWindow(VE_19_data_avg, 3, ['Date'])
VE_37_data_avg_H_3 = AddHistoryForGivenWindow(VE_37_data_avg, 3, ['Date'])
TB_Diff_data_avg_H_3 = AddHistoryForGivenWindow(TB_Diff_data_avg, 3, ['Date'])

Precip_data_avg_H_3 = AddHistoryForGivenWindow(Precip_data_avg, 3, ['Date'])
Max_Temp_data_avg_H_3 = AddHistoryForGivenWindow(Max_Temp_data_avg, 3, ['Date'])
Min_Temp_data_avg_H_3 = AddHistoryForGivenWindow(Min_Temp_data_avg, 3, ['Date'])
Obs_Temp_data_avg_H_3 = AddHistoryForGivenWindow(Obs_Temp_data_avg, 3, ['Date'])

VE_19_data_avg_H_3.to_csv('{}VE_19_data_avg_H_3.csv'.format(data_import_path), index = False)
VE_37_data_avg_H_3.to_csv('{}VE_37_data_avg_H_3.csv'.format(data_import_path), index = False)
TB_Diff_data_avg_H_3.to_csv('{}TB_Diff_data_avg_H_3.csv'.format(data_import_path), index = False)

Precip_data_avg_H_3.to_csv('{}Precip_data_avg_H_3.csv'.format(data_import_path), index = False)
Max_Temp_data_avg_H_3.to_csv('{}Max_Temp_data_avg_H_3.csv'.format(data_import_path), index = False)
Min_Temp_data_avg_H_3.to_csv('{}Min_Temp_data_avg_H_3.csv'.format(data_import_path), index = False)
Obs_Temp_data_avg_H_3.to_csv('{}Obs_Temp_data_avg_H_3.csv'.format(data_import_path), index = False)



