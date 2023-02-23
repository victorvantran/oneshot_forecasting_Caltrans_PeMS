import os
import sys
import logging
import gzip
import pickle

import math
from datetime import datetime, timezone, timedelta
import pytz

import pandas as pd
from pandas import Series

##import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path


'''___________________[SUMMARY]___________________'''
'''
Given a list of .gz files downloaded from CalTrans PEMS dataset for an entire year, convert them to single station.pickle files with proper imputations

The series of steps goes like this:
    .gz -> .csv -> .np -> .MASTER_PICKLE -> .pickle of each station
    
    Then on the training/testing main.py
    .pickle of each station -> pairwise training/testing dataset


1) Look at the USER_CONFIGURATIONS section of this script

2) Define a MAIN_DIR (the project environment)

3) Within the MAIN_DIR, provide a folder of .txt.gz files downloaded from CalTrans and name that folder: gz_district_{DISTRICT}_year_{YEAR}
    For example, use the link https://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id=3&submit=Submit
    You are downloading the year's worth of station data for District 3, Year 2023, 
    your MAIN_DIR should have a folder named: gz_district_3_year_2023 with all the downloaded .txt.gz 

4) Update the DISTRICT and YEAR variable to match your data
    Following the same example above,
    set DISTRICT = 3
    set YEAR = 2023

5) Define the columns of interest and the corresponding features
    For example, is you are only interested in timestamps, station, and total_flow data,
    set FEATURES = ['timestamp', 'station', 'total_flow']
    
6) Run the script and you will have all the files and intermediary files created!
'''
'''____________________________________________________________'''



'''___________________[USER CONFIGURATIONS]___________________'''
DISTRICT = 6
YEAR = 2023
MAIN_DIR = r'G:\one_shot_time_series_forecast_tinker'
FEATURES = ['total_flow']
#FEATURES = ['total_flow', 'avg_speed', 'avg_occupancy'] # timestamp and station features are already accounted for by default
'''____________________________________________________________'''





'''___________________[CONSTANTS]___________________'''
TOTAL_FEATURES = ['timestamp', 'station'] + FEATURES

TIME_RESOLUTION = timedelta(minutes=5) # 5 minute dataset as sampled by CalTrans; we will resample to 15 or 30 minutes during training

FEATURE_INDEX_DICT = {'timestamp' : 0,
                'station' : 1,
                'district' : 2,
                'freeway' : 3,
                'direction' : 4,
                'lane': 5,
                'length': 6,
                'samples': 7,
                'observed': 8,
                'total_flow' : 9,
                'avg_occupancy' : 10,
                'avg_speed' : 11}

FEATURE_TYPE_DICT = {'timestamp' : datetime,
                'station' : int,
                'district' : int,
                'freeway' : int,
                'direction' : str,
                'lane': str,
                'length': float,
                'samples': int,
                'observed': int,
                'total_flow' : int,
                'avg_occupancy' : float,
                'avg_speed' : float}



GZ_DIR = os.path.join(MAIN_DIR, 'gz_district_{}_year_{}'.format(DISTRICT, YEAR)) # This is where you'll store the .gz files downloaded from the CalTrans PEMs
PREPROCESSED_DATA_DIR = os.path.join(MAIN_DIR, 'preprocessed_data')
CSV_DIR = os.path.join(PREPROCESSED_DATA_DIR, os.path.join(os.path.join('district_' + str(DISTRICT), 'year_' + str(YEAR)), 'csv')) # Auxiliary folder for storing the intermediary .csv files
NPY_DIR = os.path.join(PREPROCESSED_DATA_DIR, os.path.join(os.path.join('district_' + str(DISTRICT), 'year_' + str(YEAR)), 'npy')) # Auxiliary folder for storing the intermediary .npy files
MASTER_PICKLE_DIR = os.path.join(PREPROCESSED_DATA_DIR, os.path.join(os.path.join('district_' + str(DISTRICT), 'year_' + str(YEAR)), 'master_pickle')) # For the master pickle file; this can be converted to a directory of single station.pickle files
PICKLE_DIR = os.path.join(PREPROCESSED_DATA_DIR, os.path.join(os.path.join('district_' + str(DISTRICT), 'year_' + str(YEAR)), 'pickle')) # Folder for storing the single station.pickle files to be used in creating pairwise training

MASTER_PICKLE_RAW_FILENAME = 'station_district{}_{}'
PICKLE_RAW_FILENAME = 'station_{}_{}'
NP_EXTENSION = '.npy'
CSV_EXTENSION = '.csv'
PICKLE_EXTENSION = '.pickle'
'''____________________________________________________________'''




class PeMSDataset:
    __TIMESTAMP_FORMAT = '%m/%d/%Y %H:%M:%S'
    __LOCAL_TIMEZONE = pytz.timezone('US/Pacific')

    __district = None
    __year = None
    __gz_dir = None
    __csv_dir = None
    __np_dir = None
    __master_pickle_dir = None

    __features = None
    __time_resolution = None
    __timestamp_range = None

    __dst_list = None
    __station_list = None

    def __init__(self,
                 district,
                 year,
                 gz_dir,
                 csv_dir,
                 np_dir,
                 master_pickle_dir,
                 features,
                 time_resolution):
        self.__district = district
        self.__year = year
        self.__gz_dir = gz_dir
        self.__csv_dir = csv_dir
        self.__np_dir = np_dir
        self.__master_pickle_dir = master_pickle_dir
        self.__features = features
        self.__time_resolution = time_resolution

    def create_csv_from_gv(self):
        """ Extracts the gz dataset located in the gz directory into the csv directory as csv files
         [timestamp, station, total_flow] """
        logging.info('Create .csv from .gv')

        def create_csv_from_pems_txt_gz(file):
            """ Helper function to convert PeMS.txt file (extracted from .gz) to csv with three columns of information:
             [timestamp, station, total_flow] """
            logging.info('Create csv from PeMS.txt.gz')
            df = pd.read_csv(file,
                             skiprows=0,
                             sep=',',
                             names=self.__features,
                             usecols=sorted([FEATURE_INDEX_DICT[i] for i in self.__features]),
                             engine='python')
            file_name = file.name
            file_base_name = os.path.basename(file_name)
            file_raw_name = os.path.splitext(os.path.splitext(file_base_name)[0])[0]
            df.to_csv(os.path.join(self.__csv_dir, file_raw_name+CSV_EXTENSION))

        for gz_file_base_name in os.listdir(self.__gz_dir):
            logging.info('Converting to .csv from: {}'.format(gz_file_base_name))
            gz_file_name_split = os.path.splitext(gz_file_base_name)
            gz_file_extension = gz_file_name_split[-1]
            if (gz_file_extension == '.gz'):
                gz_file_path = os.path.join(self.__gz_dir, gz_file_base_name)
                with gzip.open(gz_file_path, 'rb') as pems_txt_f_in:
                    create_csv_from_pems_txt_gz(pems_txt_f_in)

    def create_np_from_csv(self):
        """ Converts csv to numpy files """
        logging.info('Create np from csv')
        for csv_file_base_name in os.listdir(self.__csv_dir):
            logging.info('Converting to .npy from: {}'.format(csv_file_base_name))
            df = pd.read_csv(os.path.join(self.__csv_dir, csv_file_base_name))
            np_array = df.to_numpy(dtype=object)[:, 1:]
            csv_file_raw_name = os.path.splitext(csv_file_base_name)[0]
            np.save(os.path.join(self.__np_dir, csv_file_raw_name+NP_EXTENSION), np_array)

    def create_master_np(self):
        """ Combines the numpy files into one master file. Given a timestep, rectify the total flow based on the
        time resolution of the dataset. Example: Time resolution dataset of 5 minutes, with a time step of 20 minutes
        would sum (20/5 = 4) consecutive instances of total flow of 5 minutes each """
        logging.info('Create master numpy')

        master_pickle_raw_filename = MASTER_PICKLE_RAW_FILENAME.format(self.__district, self.__year)

        def create_timestamp_range_from_np_files():
            """ Find the timestamp range from the numpy files """
            logging.info('Create timestamp range from numpy files')
            #self.__timestamp_range = [None, None]
            self.__timestamp_range = {'begin':None, 'end':None}
            for file_base_name in os.listdir(self.__np_dir):
                logging.debug('Getting timestamp data from: {}'.format(file_base_name))
                np_array = np.load(os.path.join(self.__np_dir, file_base_name), allow_pickle=True)
                np_timestamp_index = 0
                timestamps = np.unique(np_array[:, np_timestamp_index])
                timestamp_str_min = timestamps.min()
                timestamp_str_max = timestamps.max()
                #print(datetime.fromtimestamp(timestamp_str_min))

                candidate_timestamp_min = datetime.strptime(timestamp_str_min, self.__TIMESTAMP_FORMAT).replace(
                    tzinfo=timezone.utc)
                candidate_timestamp_max = datetime.strptime(timestamp_str_max, self.__TIMESTAMP_FORMAT).replace(
                    tzinfo=timezone.utc)
                self.__timestamp_range['begin'] = min(self.__timestamp_range['begin'], candidate_timestamp_min) if not (self.__timestamp_range['begin'] == None) else candidate_timestamp_min
                self.__timestamp_range['end'] = max(self.__timestamp_range['end'], candidate_timestamp_max) if not (self.__timestamp_range['end'] == None) else candidate_timestamp_max
                #self.__timestamp_begin = min(self.__timestamp_begin, candidate_timestamp_min) if self.__timestamp_begin != None else candidate_timestamp_min
                #self.__timestamp_end = max(self.__timestamp_end, candidate_timestamp_max) if self.__timestamp_end != None else candidate_timestamp_max

        def create_dst_array_from_np_files():
            """ Create the daylight savings time beginning and end day(s) from the numpy. This is to be called after
            create_timestamp_range_from_np_files when self.__timestamp_begin and self.__timestamp_end is defined.
            Iterate through the timestamp range and append the daylight savings time to a list. Even index is
            beginning and odd index is end of respective dst """
            logging.info('Create daylight savings time list from numpy files')

            start_date = self.__timestamp_range['begin']
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)

            end_date = self.__timestamp_range['end']
            end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)

            delta = timedelta(days=1)

            dst_list = list()
            dt_iter_1 = start_date - delta
            dt_iter_2 = start_date
            while dt_iter_1 <= end_date + delta:
                local_dt_iter_1 = self.__LOCAL_TIMEZONE.localize(dt_iter_1)
                local_dt_iter_2 = self.__LOCAL_TIMEZONE.localize(dt_iter_2)

                if not bool(local_dt_iter_1.dst()) and bool(local_dt_iter_2.dst()):
                    dst_list.append([dt_iter_1.replace(tzinfo=timezone.utc), np.nan])
                elif bool(local_dt_iter_1.dst()) and not bool(local_dt_iter_2.dst()):
                    if not dst_list:
                        dst_list.append([np.nan, dt_iter_1.replace(tzinfo=timezone.utc)])
                    else:
                        dst_list[-1][1] = dt_iter_1.replace(tzinfo=timezone.utc)
                dt_iter_1 += delta
                dt_iter_2 += delta
            self.__dst_list = dst_list

        def create_station_list_from_np_files():
            """ Find the station list from the numpy files """
            logging.info('Create station set from numpy files')
            station_set = set()
            for file_base_name in os.listdir(self.__np_dir):
                logging.debug('Getting station number data from: {}'.format(file_base_name))
                np_array = np.load(os.path.join(self.__np_dir, file_base_name), allow_pickle=True)
                np_station_index = 1
                stations = np.unique(np_array[:, np_station_index])
                station_set.update(stations)
            self.__station_list = list(station_set)

        def fill_time_flow(master_array, stations_dict):
            """ Fill the station with its traffic flows at its respective time resolution """
            for np_file_base_name in os.listdir(self.__np_dir):
                logging.debug('Adding data from: {}'.format(np_file_base_name))
                np_data = np.load(os.path.join(self.__np_dir, np_file_base_name), allow_pickle=True)
                print("SHAPE: {}".format(np_data.shape))
                print("NUM NAN: {}".format(np.count_nonzero(pd.isna(np_data))))
                for sample in np_data:
                    '''
                    timestamp_str, station, total_flow = sample[0], int(sample[1]), sample[2]
                    timestamp_sample = datetime.strptime(timestamp_str, '%m/%d/%Y %H:%M:%S').replace(
                        tzinfo=timezone.utc)
                    timestep_index = (timestamp_sample - self.__timestamp_range['begin']) // self.__time_resolution
                    master_array[stations_dict[station], 1 + timestep_index] = total_flow
                    '''

                    timestamp_str = None
                    station = None
                    for i, feature_name in enumerate(TOTAL_FEATURES):
                        if i == 0:
                            timestamp_str = sample[i]
                        elif i == 1:
                            station = int(sample[i])
                        else:
                            feature_value = sample[i]
                            timestamp_sample = datetime.strptime(timestamp_str, '%m/%d/%Y %H:%M:%S').replace(
                                tzinfo=timezone.utc)
                            timestep_index = (timestamp_sample - self.__timestamp_range['begin']) // self.__time_resolution
                            master_array[stations_dict[station], 1 + timestep_index, i-2] = feature_value


            print("MASTER ARRAY SHAPE: {}".format(master_array.shape))
            return master_array


        create_timestamp_range_from_np_files()
        logging.info('Timestamp begin: {}'.format(self.__timestamp_range['begin']))
        logging.info('Timestamp end: {}'.format(self.__timestamp_range['end']))

        create_dst_array_from_np_files()
        logging.info('Daylight savings time list: {}'.format(self.__dst_list))

        create_station_list_from_np_files()
        logging.info('Station list size: {}'.format(len(self.__station_list)))

        logging.info('Timedelta: {}'.format(self.__timestamp_range['end'] - self.__timestamp_range['begin']))
        logging.info('Number of timesteps: {}'.format(self.get_number_timesteps(self.__time_resolution)))

        number_of_timesteps = self.get_number_timesteps(self.__time_resolution)
        master_array = np.empty((len(self.__station_list), 1 + number_of_timesteps, len(FEATURES)), dtype=np.float32)
        master_array[:] = np.nan
        print(master_array.shape)

        # Set up station hash table lookup
        stations_dict = {station_number : station_index for station_index, station_number in enumerate(self.__station_list)}
        # Set up station as column 0
        print(len(self.__station_list))

        #master_array[:, :, 0] = np.array(self.__station_list) # [!]
        master_array[:, 0, 0] = np.array(self.__station_list)
        #master_array[:, 0, 1] = np.array(self.__station_list)
        #master_array[:, 0, 2] = np.array(self.__station_list)

        # Fill the time flow data
        master_array = fill_time_flow(master_array, stations_dict)
        logging.info('Master array\n {}'.format(master_array))

        master_array_information = {'time_resolution' : self.__time_resolution,
                                    'timestamp_range' : self.__timestamp_range,
                                    'dst_list': self.__dst_list,
                                    'station_list' : self.__station_list}
        logging.info('Master array information\n {}'.format(master_array_information))

        with open(os.path.join(self.__master_pickle_dir, master_pickle_raw_filename + PICKLE_EXTENSION), 'wb') as f:
            pickle.dump(master_array_information, f)
            np.save(f, master_array, allow_pickle=True)


    def get_gz_dir(self):
        """ Returns the raw path of the gz directory """
        return self.__gz_dir

    def get_csv_dir(self):
        """ Returns the raw path of the csv directory """
        return self.__csv_dir

    def get_np_dir(self):
        """ Returns the raw path of the numpy directory """
        return self.__np_dir

    def get_timestamp_begin(self):
        """ Returns the timestamp begin of the dataset """
        return self.__timestamp_range['begin']

    def get_timestamp_end(self):
        """ Returns the timestamp end of the dataset """
        return self.__timestamp_range['end']

    def get_number_timesteps(self, timestep):
        """ Returns the number of timesteps of the dataset based on timedelta """
        return (self.__timestamp_range['end'] - self.__timestamp_range['begin'])//timestep + 1


class PeMSData:
    __NP_EXTENSION = '.npy'
    __FILE_TIMESTAMP_FORMAT = '%m_%d_%Y_%H_%M_%S'

    # Path to the master pickle file that holds the entire PeMS Data we are interested in,
    # and the respective pickle folder to store the individual station pickle files based from the master pickle file
    __master_pickle_filepath = None
    __pickle_dir = None

    # Numpy array: Holds unmanipulated, raw data (apart from preprocessing)
    _np_array = None
    _np_time_resolution = None
    _np_station_list = None
    _np_timestamp_range = None
    _np_dst_list = None

    # Dataframe: Used to manipulate, graph, etc.
    _df = None
    _df_timestep = None

    def __init__(self, master_pickle_filepath, pickle_dir):
        """ Initalize """
        self.__master_pickle_filepath = master_pickle_filepath
        self.__pickle_dir = pickle_dir

    def import_np(self):
        """ Import from .npy file """
        # Open pickle file to retrieve the np array and its information
        np_array_information = None
        with open(self.__master_pickle_filepath, 'rb') as f:
            np_array_information = pickle.load(f)
            self._np_array = np.load(f, allow_pickle=True)

        logging.debug('Loaded np_array_information: {}'.format(np_array_information))
        logging.debug('Loaded np_array: {}'.format(self._np_array))

        self.import_array_information(np_array_information)

    def import_array_information(self, np_array_information):
        """ Save information of the numpy array """
        logging.info('Preprocessing np array with its given information')
        self._np_time_resolution = np_array_information['time_resolution']
        self._np_timestamp_range = np_array_information['timestamp_range']
        self._np_station_list = np_array_information['station_list']
        self._np_dst_list = np_array_information['dst_list']

    def build_timestep(self):
        """ Build a timestep array to be used as column for the dataframe """
        logging.info('Building timestep')
        # microseconds = 1 to rectify exclusive range (also note that the time resolution cannot be less than 1 microsecond. Make an assertion to check
        dto_dataset_time_begin = self._np_timestamp_range['begin']
        dto_dataset_time_end = self._np_timestamp_range['end']
        tdo_resolution = self._np_time_resolution
        return [timedelta(microseconds=int(i - int(dto_dataset_time_begin.timestamp() * 10 ** 6))) for i in range(
            int(dto_dataset_time_begin.timestamp() * 10 ** 6),
            int((dto_dataset_time_end + timedelta(microseconds=1)).timestamp() * 10 ** 6),
            int(tdo_resolution.total_seconds() * 10 ** 6))]

    def get_timestep(self, dt_instance):
        return (dt_instance - self._np_timestamp_range['begin'])

    def create_df_from_np_array(self, preprocess=True, missing_threshold=1):
        """ Create dataframe from numpy array and its information, removing station with missing_threshold amount of sample information """
        logging.info('Creating dataframe')
        columns = ['station']
        columns.extend(self.build_timestep())
        print(len(columns))
        print(columns)

        # FOR ...
        df = pd.DataFrame(self._np_array[:,:,0], columns=columns).set_index('station')
        self._df = df

        if preprocess:
            # [!] TO DO MORE DSTs just iterate through all the possible DST RANGES instead of 1st element assumption
            for dst in self._np_dst_list:
                df = self.df_imputate_dst(df, columns,
                                          dto_day_light_savings_begin=dst[0],
                                          dto_day_light_savings_end=dst[1])
            df = self.df_remove_fruitless_stations(df, missing_threshold=missing_threshold)  # Remove any threshold
            df = self.df_imputate_single(df)  # Imputate any straggling zeroes
            # df = self.df_remove_fruitless_stations(df, missing_threshold=1)  # Remove any indication of NaN to yield the final (ignore for 2019 because missing 31st of december data final hours)
            self._df = df

        logging.info('Dataframe nan values: {}'.format(self._df.isnull().sum(axis=1)))
        logging.info('Dataframe: {}'.format(self._df))



        '''
        df = pd.DataFrame(self._np_array, columns=columns).set_index('station')
        self._df = df

        if preprocess:
            # [!] TO DO MORE DSTs just iterate through all the possible DST RANGES instead of 1st element assumption
            for dst in self._np_dst_list:
                df = self.df_imputate_dst(df, columns,
                                dto_day_light_savings_begin=dst[0],
                                dto_day_light_savings_end=dst[1])
            df = self.df_remove_fruitless_stations(df, missing_threshold=missing_threshold)  # Remove any threshold
            df = self.df_imputate_single(df) # Imputate any straggling zeroes
            #df = self.df_remove_fruitless_stations(df, missing_threshold=1)  # Remove any indication of NaN to yield the final (ignore for 2019 because missing 31st of december data final hours)
            self._df = df

        logging.info('Dataframe nan values: {}'.format(self._df.isnull().sum(axis=1)))
        logging.info('Dataframe: {}'.format(self._df))
        '''

    def df_imputate_dst(self, df, columns,
                        dto_day_light_savings_begin,
                        dto_day_light_savings_end):
        """" Imputate the days which need DST end data through averaging the missing hour """
        # [!] Need to modify in case there is no DST, a missing beginning, or a missing end
        logging.info('Imputate the dataframe to account for daylight savings time')
        # microseconds -1 as the smallest timestep deduction to rectify inclusiveness of range

        # Rectify exact hour for getting the correct elements within that hour frame; hour=2 in our calculation case
        dto_day_light_savings_begin = dto_day_light_savings_begin.replace(hour=2)
        dto_day_light_savings_end = dto_day_light_savings_end.replace(hour=2)

        # Drop columns nan for the 23-hour day
        df.drop(columns=df.loc[:, self.get_timestep(dto_day_light_savings_begin):self.get_timestep(
            dto_day_light_savings_begin + timedelta(hours=1, microseconds=-1))], inplace=True)

        # Insert columns for the 25-hour day
        for i in range(int(timedelta(hours=1)/self._np_time_resolution)):  # 12 for the number of 5 minutes in the hour
            df.insert(df.columns.get_loc(self.get_timestep(dto_day_light_savings_end) + timedelta(hours=-1)),
                      'IMPUTATED: ' + str(
                          self.get_timestep(dto_day_light_savings_end) + timedelta(hours=-1) + self._np_time_resolution * i),
                      [np.nan] * df.shape[0])

        # Reset/Refesh Column Timesteps
        df.reset_index(inplace=True)
        df.columns = columns
        df.set_index('station', inplace=True)

        # Average out the missing 1 hour in the 25-hour-day data using the hour before and hour after the dst inserted hour
        df.loc[:, self.get_timestep(dto_day_light_savings_end + timedelta(hours=-2)):self.get_timestep(
            (dto_day_light_savings_end + timedelta(hours=-1, minutes=-5)))] = np.vectorize(self.imputate_average)(
            df.loc[:, self.get_timestep((dto_day_light_savings_end + timedelta(hours=-3))):self.get_timestep(
                (dto_day_light_savings_end + timedelta(hours=-2, minutes=-5)))],
            df.loc[:, self.get_timestep((dto_day_light_savings_end + timedelta(hours=-1))):self.get_timestep(
                (dto_day_light_savings_end + timedelta(minutes=-5)))])
        return df

    def df_imputate_single(self, df):
        """ Imputate any single missing element using the average between the element before and the element after """
        logging.info('Imputate the dataframe to account for single missing')
        missing_list = df.columns[df.isna().any()].tolist()

        #print(len(missing_list))
        #i = 0
        for td_missing in missing_list:
            #i+=1
            #print(len(missing_list), i)
            station_missing = (df.loc[df[td_missing].isnull()].index).tolist()
            prev_element_timestep = td_missing - self._np_time_resolution
            next_element_timestep = td_missing + self._np_time_resolution
            if (prev_element_timestep >= df.columns[0] and next_element_timestep <= df.columns[-1]):
                df.loc[station_missing, td_missing] = np.vectorize(self.imputate_average)(
                    df.loc[station_missing, prev_element_timestep],
                    df.loc[station_missing, next_element_timestep])

        return df

    def df_remove_fruitless_stations(self, df, missing_threshold=1):
        """ Drop any rows that have #missing values >= missing threshold """
        logging.info('Drop any missing/nan with threshold: {}'.format(missing_threshold))
        if missing_threshold <= 1:
            fruitful = df.dropna()
            return fruitful
        else:
            fruitful = df.loc[df[(df.isna().sum(axis=1) <= missing_threshold)].index]
            return fruitful

    def save_np_from_df(self, file_raw_name):
        self._df.reset_index(inplace=True)
        imputated_dataset = self._df.to_numpy()
        np.save(os.path.join(MASTER_PICKLE_DIR, file_raw_name + self.__NP_EXTENSION), imputated_dataset)

    def imputate_average(self, a, b):
        """ Average out between two numbers when vecotrizing """
        #return np.nan if np.isnan(a) or np.isnan(b) else (a + b) // 2
        return np.nan if np.isnan(a) or np.isnan(b) or pd.isna(a) or pd.isna(b) else (a + b) // 2

    def reduce_df(self, classes):
        self._df = self._df.loc[classes]

    def create_single_stations(self, filled_percentage_threshold=0.9):
        """ Filled percentage threshold indicates whether to keep a station based on its total-flow data not being zero"""
        dto_dataset_time_begin = self._np_timestamp_range['begin']
        year = dto_dataset_time_begin.year
        dto_dataset_time_end = self._np_timestamp_range['end']
        tdo_resolution = self._np_time_resolution
        index = [self._np_timestamp_range['begin'] + timedelta(
            microseconds=int(i - int(dto_dataset_time_begin.timestamp() * 10 ** 6))) for i in range(
            int(dto_dataset_time_begin.timestamp() * 10 ** 6),
            int((dto_dataset_time_end + timedelta(microseconds=1)).timestamp() * 10 ** 6),
            int(tdo_resolution.total_seconds() * 10 ** 6))]

        filled_threshold = int(self._df.shape[1] * filled_percentage_threshold)
        for i in range(self._df.shape[0]):
            station = self._df.iloc[i]
            station_name = station.name

            logging.info('Station: {}'.format(int(station_name)))
            logging.info("Station shape: {}".format(station.shape))

            num_zeros = station.isna().sum()
            logging.info("Num of missing total_flow samples: {}".format(num_zeros))

            #print(self._df.shape[1] - filled_threshold)

            if not (num_zeros > self._df.shape[1] - filled_threshold):
                logging.info('Creating station_{}_{}.pickle file'.format(year, station_name))
                station_array = station.to_numpy()
                sample_station_df = pd.DataFrame(data=station_array, index=index, columns=['total_flow'])
                sample_station_df.index.name = 'timestamp'

                sample_station_df.to_pickle(os.path.join(self.__pickle_dir,
                                                         PICKLE_RAW_FILENAME.format(year, int(station_name)) + PICKLE_EXTENSION))



# READ: How to use
if __name__ == '__main__':
    # Error/Warning logger
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    np.set_printoptions(precision=3)

    # Set up directory environment
    # The gz folder should already exist with the station.txt.gz files inside,
    # but create the other folders if they do not exist yet
    Path(PREPROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(CSV_DIR).mkdir(parents=True, exist_ok=True)
    Path(NPY_DIR).mkdir(parents=True, exist_ok=True)
    Path(MASTER_PICKLE_DIR).mkdir(parents=True, exist_ok=True)
    Path(PICKLE_DIR).mkdir(parents=True, exist_ok=True)


    # Create master pickle first (which creates .csv and .np files in the process)
    pems_dataset = PeMSDataset(district=DISTRICT,
                               year=YEAR,
                               gz_dir=GZ_DIR,
                               csv_dir=CSV_DIR,
                               np_dir=NPY_DIR,
                               master_pickle_dir=MASTER_PICKLE_DIR,
                               features=TOTAL_FEATURES,
                               time_resolution=TIME_RESOLUTION)

    pems_dataset.create_csv_from_gv()
    pems_dataset.create_np_from_csv()
    pems_dataset.create_master_np()

    # Split that master pickle file into many single station pickle files to be used for training
    pems_data = PeMSData(master_pickle_filepath=os.path.join(MASTER_PICKLE_DIR, MASTER_PICKLE_RAW_FILENAME.format(DISTRICT, YEAR) + PICKLE_EXTENSION),
                         pickle_dir=PICKLE_DIR)
    pems_data.import_np()
    pems_data.create_df_from_np_array(missing_threshold=5000)
    pems_data.create_single_stations()


    print("DONE")