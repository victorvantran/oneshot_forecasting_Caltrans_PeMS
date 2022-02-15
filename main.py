# Caltrans dataset headers
# day, timestamp, station, district, freeway, direction, lane type, station length, samples, %observed, total flow, avg occupancy, avg speed, lane n samples, lane n flow, lane n avg, occ, lane n avg speed, lane n observed

import gzip
import shutil

import numpy as np

import os

import pandas as pd


from datetime import datetime


GZ_DIR = r'C:\Users\Victor\Desktop\CSUF\Spring 2022\CPSC 599\caltrans_dataset\2018\gz_dir'
CSV_DIR = r'C:\Users\Victor\Desktop\CSUF\Spring 2022\CPSC 599\caltrans_dataset\2018\csv_dir'
NPY_DIR = r'C:\Users\Victor\Desktop\CSUF\Spring 2022\CPSC 599\caltrans_dataset\2018\npy_dir'

CSV_EXTENSION = '.csv'
NPY_EXTENSION = '.npy'


def extract_PeMS_gz_dataset(source_directory, destination_directory):
    ''' Extracts the PeMS gz dataset located in the source directory into the destination directory as csv files '''
    for filename in os.listdir(source_directory):
        if (filename.endswith('.gz')):
            # remove .gz to get filename base
            uncompressed_filename = os.path.splitext(filename)[0]
            gz_file = os.path.join(source_directory, filename)
            with gzip.open(gz_file, 'rb') as f_in:
                convert_PeMS_txt_to_csv(f_in, destination_directory)



def convert_PeMS_txt_to_csv(file, destination_directory):
    ''' Get only the 'timestamp', 'station', 'total_flow' to reduce csv size '''
    df = pd.read_csv(file,
                     skiprows=0,
                     sep=',',
                     names=['timestamp', 'station', 'total_flow'],
                     usecols=[0, 1, 9],
                     engine='python')
    filebasename = os.path.basename(file.name)
    filerawname = os.path.splitext(os.path.splitext(filebasename)[0])[0]
    df.to_csv(os.path.join(destination_directory, filerawname + CSV_EXTENSION))
    #print(df['total_flow'].isnull().sum())


def convert_PeMS_dataset_csv_to_npy(source_directory, destination_directory):
    for filename in os.listdir(source_directory):
        convert_PeMS_csv_to_npy(os.path.join(source_directory, filename), destination_directory)
        break


def get_timestamp_range_from_npy_directory(npy_directory):
    ''' Given a directory of .npy files, find the timestamp range '''
    timestamp_range = [float('inf'), float('-inf')]

    for filename in os.listdir(npy_directory):
        np_data = np.load(os.path.join(npy_directory, filename), allow_pickle=True)
        timestamp_range[0] = min(timestamp_range[0], np_data[0].min()) # np_data[0] refers to the timestamp column
        timestamp_range[1] = max(timestamp_range[1], np_data[0].max())

    return timestamp_range


def get_station_set_from_npy_directory(npy_directory):
    ''' Given a directory of .npy files, return a set of all possible stations '''
    stations = set()
    for filename in os.listdir(npy_directory):
        np_data = np.load(os.path.join(npy_directory, filename), allow_pickle=True)
        stations.update(np_data[1]) # np_data[1] refers to the station column

    return stations


def convert_PeMS_csv_to_npy(filepath, destination_directory):
    ''' Convert PeMS csv to numpy changing the timestamp to epoch time based '''
    df = pd.read_csv(filepath)
    # Create the np array
    series = list()

    timestamps = pd.to_datetime(df.timestamp, format='%m/%d/%Y %H:%M:%S').view('int64') // 10**9
    series.append(np.array(timestamps))
    series.append(np.array(df.station))
    series.append(np.array(df.total_flow))
    #series = np.array(series, dtype=np.int64)
    series = np.array(series, dtype=object)

    filebasename = os.path.basename(filepath)
    filerawname = os.path.splitext(filebasename)[0]
    np.save(os.path.join(destination_directory, filerawname + NPY_EXTENSION), series)

    print(series.shape) # [col, row]



def create_master_series(npy_directory):
    ''' Iterate through the npy files, and based on the time range, rearrange the data to be samples of stations (rows) whose
    columns will be the timestamp. The first column is the station, and the subsequent columns are the total_flow for a time step.
    The timestep is every 5 minutes.
    '''
    series = list()

    timestamp_range = get_timestamp_range_from_npy_directory(NPY_DIR)
    timestamp_origin = timestamp_range[0]
    stations = get_station_set_from_npy_directory(NPY_DIR)


    num_timesteps = ((timestamp_range[1] - timestamp_range[0])//300) + 1 # Divide by 300 due to the data timestep bing 5 minutes, which is 300 seconds

    print("Timestep range: ", timestamp_range)
    print("Number of timesteps: ", num_timesteps)
    print("Number of stations: ", len(stations))

    for station in stations:
        series_sample = [station]
        series_sample.extend([None]*num_timesteps)
        series.append(np.array(series_sample, dtype=object))

    series = np.array(series, dtype=object)
    print(series.shape)

    df = pd.DataFrame(series)
    #print(df)


    for filename in os.listdir(npy_directory):
        np_data = np.load(os.path.join(npy_directory, filename), allow_pickle=True).transpose()
        i = 0
        for sample in np_data:
            print(i)
            i+=1
            timestamp, station, total_flow = sample[0], sample[1], sample[2]
            timestep_index = (timestamp - timestamp_origin) // 300
            #print(df.loc[df[0] == station][0]) Gets the row/sample where the first column (station) equals to station
            df.loc[df[0] == station, timestep_index + 1] = total_flow

    print(df)
    series = df.to_numpy()
    np.save(os.path.join(npy_directory, 'MASTER' + NPY_EXTENSION), series)
    return series




# Remove any station with unknown data (and mark them for future removal reference in case they do have data in this new case)
#   Can do by iterating through all the reduced csv size (entire year) and built a list of all stations which had a NaN value (check for nan)

# assert df['timestamp'].isnull().any() == False
# assert df['station'].isnull().any() == False

# print(len(df['total_flow']))
# print(df['total_flow'].isnull().sum())
# break




# day and timestamp, station, district, freeway, direction, lane type, station length, samples, %observed, total flow, avg occupancy, avg speed, lane n samples, lane n flow, lane n avg, occ, lane n avg speed, lane n observed
# https://www.epochconverter.com/
# https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values
# https://colab.research.google.com/drive/1rY7Ln6rEE1pOlfSHCYOVaqt8OvDO35J0#forceEdit=true&sandboxMode=true&scrollTo=m0jdXBRiDSzj
# 1514764800 Base: jan 01 01 2018


if __name__ == '__main__':
    #extract_PeMS_gz_dataset(GZ_DIR, CSV_DIR)
    convert_PeMS_dataset_csv_to_npy(CSV_DIR, NPY_DIR)
    #print(get_timestamp_range_from_npy_directory(NPY_DIR))
    create_master_series(NPY_DIR)



# actually need to convert the file to csv with COMMA and SPACE delimiters!
# Let's do a station count first to see if the entire year holds all the stations. We ignore newly added stations




# 1) Efficiently extract files to .csv
# 2) Clean .csv files to include only relevant information such as station number, date/time, car count to lower file size
# 3) transpose rows and columns
# 4) Figure out a way to append on the time (do i need an appending algorithm using date/time as key to rows? Or can i take advantage of the files already separated by days?)
#       Possible pitfalls is checking for newly added stations that were not there?











'''
Pitfalls:
    Stations can fail at certain periods of the day at certain days of the week at certain weeks of the month, etc.
        This will give NaN values for total_flow, causing misleading data (i.e. incomplete stations)
        One file could have 60000 incomplete stations; others could have 50000 incomplete stations that may overlap or are different
        Example January 01 - 31 incomplete sample count: 
        [62974, 63617, 63878, 53065, 53736, 52575, 52029,
        53137, 53747, 53799, 51029, 53832, 56535, 60190, 
        57267, 54933, 54205, 54407, 55830, 56257, 54959,
        59087, 58261, 63040, 61304, 62778, 61704, 62569,
        67111, 66049, 67001]
        Once we combine the data into a giant dataset, make sure to remove any incomplete station samples
'''