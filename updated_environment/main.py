# https://www.youtube.com/watch?v=5Ym-dOS9ssA&ab_channel=AladdinPersson
import os
import math

import tensorflow as tf
import numpy as np
import random


# https://anaconda.org/anaconda/pandas
import pandas as pd

# pip3 install matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import RepeatVector

from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Subtract
from tensorflow.keras.layers import Add

from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import ConvLSTM2D

from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras import initializers

from tensorflow.keras import metrics
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.regularizers import L1
from tensorflow.keras.regularizers import L2

from tensorflow.keras.utils import Sequence


from datetime import datetime, timezone, timedelta

from sklearn.utils import shuffle

import re

from tensorflow.keras.utils import plot_model



# ----------------------[USER CONFIGURATIONS]----------------------
# Path to the directory of the dataset where the pairwise training and testing files exist
DATA_DIR =  r'G:\one_shot_time_series_forecast_tinker\dataset'

# This should exist/created in your DATA_DIR folder if you already processed the pairwise training data
PREMADE_TRAINING_DATA = r'train_year2022_district12_stations3_days31_period30min_history144s_forecast24s_stride24s.npz'
PREMADE_TESTING_DATA = r'test_year2022_district3_stations3_days31_period30min_history144s_forecast24s_stride24s.npz'

MODEL_CHECKPOINT_DIR = r'G:\one_shot_time_series_forecast_tinker\models'

# The 2 historical features are (0) Traffic Total Flow and (1) The day in the week; We are only working with 2 variables here
NUM_HISTORICAL_FEATURES = 2
TOTAL_FLOW_FEATURE_INDEX = 0
WEEKLY_FEATURE_INDEX = 1
#AVG_SPEED_FEATURE_INDEX = 1
#AVG_OCCUPANCY_FEATURE_INDEX = 2
#WEEKLY_FEATURE_INDEX = 3

# TRAINING DATA PARAMETERS
TRAINING_TYPE = 'train'
TRAINING_YEAR = 2022
TRAINING_DISTRICT = 12
'''
If you are first starting off, I suggest you choose a small number for NUM_STATIONS, such as 10, to see how big the size of your file will be,
and set the RATIO_PAIRWISE_INSTANCES_KEEP = 1.0
Then extrapolate that size of the file to determine the size of the file you will get when you use the proper amount of stations.
If the file size is too big, you can then adjust the RATIO_PAIRWISE_INSTANCES_KEEP to get a workable-size dataset
For example, say your NUM_STATIONS = 10 and your resulting training_pairwise file is 10MB
If you plan to use 1000 stations, your resulting file would be 10GB
If your machine can only handle 1GB of dataset in a reasonably amount of training time, you would choose a ratio of 1/10
So in the beginning, your variables to test the filesize would be NUM_TRAINING_STATIONS=10, RATIO_PAIRWISE_INSTANCES_KEEP=1.0
So in the end when you choose the proper number of stations, your variables would be: NUM_TRAINING_STATIONS=1000, RATIO_PAIRWISE_INSTANCES_KEEP=0.10
'''
NUM_TRAINING_STATIONS = 3
TRAINING_DATETIME_BEGIN = datetime(year=TRAINING_YEAR, month=1, day=1, hour=0, minute=0)
TRAINING_DATETIME_END = datetime(year=TRAINING_YEAR, month=2, day=1, hour=5, minute=0)


# TESTING DATA PARAMETERS
TESTING_TYPE = 'test'
TESTING_YEAR = 2022
TESTING_DISTRICT = 3
NUM_TESTING_STATIONS = 3
TESTING_DATETIME_BEGIN = datetime(year=TRAINING_YEAR, month=1, day=1, hour=0, minute=0)
TESTING_DATETIME_END = datetime(year=TRAINING_YEAR, month=2, day=1, hour=0, minute=0)

# PAIRWISE WINDOWS PARAMETERS
SAMPLE_PERIOD = timedelta(minutes=30) # How much time per data sampled
HISTORICAL_WINDOW = timedelta(days=3, hours=0, minutes=0)
FORECAST_WINDOW = timedelta(days=0, hours=12, minutes=0)
STRIDE_WINDOW = timedelta(days=0, hours=12, minutes=0)

# Path Names
TRAINING_DIRECTORY = r'G:\one_shot_time_series_forecast_tinker\preprocessed_data\district_{}\year_{}\pickle'.format(TRAINING_DISTRICT, TRAINING_YEAR) # These would hold pickle files created from .gz files using the PeMS.py class
TESTING_DIRECTORY =  r'G:\one_shot_time_series_forecast_tinker\preprocessed_data\district_{}\year_{}\pickle'.format(TESTING_DISTRICT, TESTING_YEAR) # These would hold pickle files created from .gz files using the PeMS.py class


'''
When you generate pairwise instances, you would get n*(n-1)/2 instances for n is the number of windows of your station data
If n is big, n*(n-1)/2  is really big
This is compounded by having m stations
The total number of pairwise instances would be O(m*n^2)
To reduce the nubmer of pairwise instances and lower the size of our data, we will keep only a sparse ratio of pairwise instances for each station to keep
'''
RATIO_PAIRWISE_INSTANCES_KEEP = 1.0 #0.016 #0.064 #1.0

# Learning Parameters
BATCH_SIZE = 256
INIT_LEARNING_RATE = 0.0001
CLIPNORM = 0.1
NUM_EPOCHS = 2500
# --------------------------------------------------------



# ----------------------[CONSTANTS]----------------------
# Converting our windows to number of samples
NUM_HISTORICAL_SAMPLES = int(HISTORICAL_WINDOW/SAMPLE_PERIOD)
NUM_FORECAST_SAMPLES = int(FORECAST_WINDOW/SAMPLE_PERIOD)
NUM_STRIDE_SAMPLES = int(STRIDE_WINDOW/SAMPLE_PERIOD)

# The 1 forecast feature is Traffic Total Flow; We are not focusing on predicting anything else
NUM_FORECAST_FEATURES = 1

# Random seed to make "randomization" repeatable as long as the seed is the same
RANDOM_SEED = 8675309
# --------------------------------------------------------

# Configure GPU to allocate more memory if needed until maxed
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

def window_data(data, w=NUM_HISTORICAL_SAMPLES+NUM_FORECAST_SAMPLES, f=NUM_HISTORICAL_FEATURES, stride=NUM_STRIDE_SAMPLES):
    ''' Rolling window on the data, truncate end of data that does not have enough data for the next/last stride '''
    return np.lib.stride_tricks.sliding_window_view(x=data, window_shape=(w, f))[::stride].reshape(-1, w, f)



def add_weekly_dimension(data, datetime_begin, datetime_end):
    ''' Add the days-of-the-week-feature to the numpy arry [0-Sunday, 1-Monday, ... 6-Saturday] '''
    start_weekly_index = int(timedelta(days=datetime_begin.weekday(), hours=datetime_begin.hour,
                                       minutes=datetime_begin.minute) / SAMPLE_PERIOD)
    total_num_samples = int((datetime_end-datetime_begin)/SAMPLE_PERIOD)
    WEEKLY_PERIOD = timedelta(days=7, hours=0, minutes=0)
    weekly_data = np.mod(np.add(np.arange(total_num_samples), start_weekly_index), int(WEEKLY_PERIOD/SAMPLE_PERIOD))
    weekly_data = np.expand_dims(weekly_data, axis=1)
    #print("WEEKLY_DATA SHAPE: {}".format(weekly_data.shape))
    return np.column_stack((data, weekly_data))



def normalize_data(references, currents, num_historical_samples, num_forecast_samples):
    ''' Normalize each respective window using min_max normalization '''
    references = references.copy()
    currents = currents.copy()

    # Normalize reference window by the reference historical by (a_element - a_min)/(a_max-a_min)
    # We are using vectorization to deal with operations of numpy arrays. This is many mgnitudes faster than using for-loops
    # For numpy arrays and panda's dataframes, you should always use vectorization to manipulate
    references_historical = references[:, :num_historical_samples, ]
    references_historical_max = np.expand_dims(np.amax(references_historical, axis=1)[:,
                                               TOTAL_FLOW_FEATURE_INDEX], axis=1)
    references_historical_min = np.expand_dims(np.amin(references_historical, axis=1)[:,
                                               TOTAL_FLOW_FEATURE_INDEX], axis=1)
    references[:, :, TOTAL_FLOW_FEATURE_INDEX] = \
        np.divide(np.subtract(references[:, :, TOTAL_FLOW_FEATURE_INDEX], references_historical_min),
                  np.subtract(references_historical_max, references_historical_min))

    # Normalize the entire current window (historical and forecast) by the current historical by (a_element - a_min)/(a_max-a_min)
    currents_historical = currents[:, :num_historical_samples, ]
    currents_historical_max = np.expand_dims(np.amax(currents_historical, axis=1)[:,
                                               TOTAL_FLOW_FEATURE_INDEX], axis=1)
    currents_historical_min = np.expand_dims(np.amin(currents_historical, axis=1)[:,
                                               TOTAL_FLOW_FEATURE_INDEX], axis=1)

    currents[:, :, TOTAL_FLOW_FEATURE_INDEX] = \
        np.divide(np.subtract(currents[:, :, TOTAL_FLOW_FEATURE_INDEX], currents_historical_min),
                  np.subtract(currents_historical_max, currents_historical_min))

    return references, currents


def cross_data(references, currents, num_historical_samples, num_forecast_samples):
    '''
    :param references: (num_instances, num_historical_samples+num_forecast_samples, num_features)
    :param currents: (num_instances, num_historical_samples+num_forecast_samples, num_features)
    :param num_historical_samples:
    :param num_forecast_samples:
    :return: Two arrays:
        1st array is [(historical_ref_0,historical_curr_0), (historical_ref_1,historical_curr_1),..., (historical_ref_n,historical_curr_n)]
        2nd array is [(forecast_difference_0), (forecast_difference_1), (forecast_difference_2)]
    '''
    # Assertions to test that the shapes are of correct size
    assert references.shape[0] > 0
    assert currents.shape[0] > 0
    assert references.shape[1] == num_historical_samples + num_forecast_samples
    assert currents.shape[1] == num_historical_samples + num_forecast_samples

    # We will generate a list of pairwise instances for a station, given its reference and current windowed datas
    list_of_pairwise_instances = list()

    # Iterate through the reference windows in attempts to generate a pairwise data at each iteration
    for reference in references:
        # Skip attempt to generate a pair if nan or infinity is encountered.
        # This is due to the numpy having missing data (NaN) due to unclean CalTrans data,
        # or due to divide by zeros due to currents_historical_max==currents_historical_min
        if np.count_nonzero(np.isnan(reference)) or np.count_nonzero(np.isinf(reference)):
            continue
        reference_weeklys = reference[:, WEEKLY_FEATURE_INDEX]
        # For each reference window, iterate through the current windows
        for current in currents:
            # Again, skip attempt to generate a pair if nan encountered. This is due to the numpy having missing data (NaN) due to unclean CalTrans data
            if np.count_nonzero(np.isnan(current)):
                continue
            # Check the current_weeklys of each window
            # If they two windows share the same days features, then they are good windows to pair
            # For example, say we are sampling at a rate of 1 hour for a historical+forecast window length of 4 days.
            #   Say, the window starts at Monday at 12:00am
            #   The current days would be an array of [Mon_0, Mon_1, ... Mon_23, Tues_0, ..., Tues_23,  Tues_0, ..., Tues_23, Wed_0, ... Wed_23, Thur_0, ... Thur_23]
            #   It would not make sense to compare a window of [Mon_0, ... Thur_23] to a window of [Tues_0, ... Fri_23]
            #   Also, it would not make sense to compare a window of [Mon_0, ... Thur_23] to a window of [Mon_3, ... Fri_23, Sat_0, Sat_1, Sat_2, Sat_3]
            current_weeklys = current[:, WEEKLY_FEATURE_INDEX]
            if np.array_equal(reference_weeklys, current_weeklys):
                # Our reference input to our siamese twins model will be the entire reference window
                reference_input = reference

                # Our current input to our siamese twins model will be the current window WITHOUT the actual forecast; we will zero it out
                # We zero that out because our assumption (during training) is that we do not know what the actual forecast of the current window is!
                # However, we do keep the historical data of the current window to be our current input
                current_input = np.copy(current)
                current_input[num_historical_samples:, TOTAL_FLOW_FEATURE_INDEX] = 0

                # Our current output is the forecast data of our current window
                current_output = current[-num_forecast_samples:]

                # An instance of a pairwise window: Use the reference input and current input to predict the current output
                # Current window is synonymous to Test window
                pairwise_instance = np.concatenate((reference_input, current_input,
                                                current_output))


                # CULLING the data
                # Before we actually use this pairwise_instance, we will check if it is a fruitful pairwise_instance to use
                # By fruitful, I mean we are going to cull any data where the reference window and current window are vastly different
                # [!] Note that we can edit threshold values in this culling algorithm or change it completely to preprocess more fruitful data

                # Step 1: Get the difference of historical data between the reference and current window
                historical_flow_difference = np.abs(np.subtract(reference[:num_historical_samples, TOTAL_FLOW_FEATURE_INDEX],
                                                       current[:num_historical_samples, TOTAL_FLOW_FEATURE_INDEX]))

                # Step 2: Get the difference of forecast data between the reference and current window
                forecast_flow_difference = np.abs(np.subtract(reference[-num_forecast_samples:, TOTAL_FLOW_FEATURE_INDEX],
                                                    current[-num_forecast_samples:, TOTAL_FLOW_FEATURE_INDEX]))

                # If the difference of the historical or forecast between the two windows are too different, we are assuming that
                # one or both of the windows are "anomalies" (such as car crashes and stuff). This is not good for training, since we
                # don't have car crashes/other variabels for features. So we will detect these anomalies and cull them out.
                # Perhaps, if we expand the features of our data, we can account for such differences and we would not need to cull the data here

                # In the case where we only use total_flow as our feature, we will cull...
                # We cull them based on if their difference surpasses a threshold.
                # Step 3: Between the historical window, count all total_flow samples that exceed 30% difference (remember that we normalized these values)
                MAX_HISTORICAL_DIFFERENCE_THRESHOLD = 0.30
                historical_num_exceed_difference = (historical_flow_difference > MAX_HISTORICAL_DIFFERENCE_THRESHOLD).sum()

                # Step 4: Between the historical window, count all total_flow samples that exceed 30% difference (remember that we normalized these values)
                MAX_FORECAST_DIFFERENCE_THRESHOLD = 0.30
                forecast_num_exceed_difference = (forecast_flow_difference > MAX_FORECAST_DIFFERENCE_THRESHOLD).sum()

                # Step 5: Cull any instances, if over 10% of samples in the reference or forecast window exceed that exceeded 30% difference threshold
                # In other words, only accept pairwise instances that have less than 10% of samples that have exceeded the 30% difference threshold
                MAX_HISTORICAL_NUM_EXCEED_DIFFERENCE_THRESHOLD = 0.10
                MAX_FORECAST_NUM_EXCEED_DIFFERENCE_THRESHOLD = 0.10

                # Step 6: Also, cull any instances where there is over 50% of total_flow = 0 samples. These instances are deemed to have sparse, vacant data
                MAX_ZERO_THRESHOLD = 0.5
                # Cull anomalies
                if historical_num_exceed_difference <= int(MAX_HISTORICAL_NUM_EXCEED_DIFFERENCE_THRESHOLD * num_historical_samples) and \
                        forecast_num_exceed_difference <= int(MAX_FORECAST_NUM_EXCEED_DIFFERENCE_THRESHOLD * num_forecast_samples) and \
                        (reference[:num_historical_samples] >= 0.0).sum() >= int(MAX_ZERO_THRESHOLD * num_historical_samples) and \
                        (current[:num_historical_samples] >= 0.0).sum() >= int(MAX_ZERO_THRESHOLD * num_historical_samples) and \
                        (reference[-num_forecast_samples:] >= 0.0).sum() >= int(MAX_ZERO_THRESHOLD * num_forecast_samples) and \
                        (current[-num_forecast_samples:] >= 0.0).sum() >= int(MAX_ZERO_THRESHOLD * num_forecast_samples):
                        list_of_pairwise_instances.append(pairwise_instance)
                else:
                    pass #print("IGNORING PAIRWISE INSTANCE DUE TO NAN FOUND OR MAX THRESHOLD EXCEEDED")


    # After generating a list of pairwise instances for a station, we will split the list to historical and forecast arrays
    array = np.asarray(list_of_pairwise_instances)

    # Array 1: Historical data array, where an element of the array is (historical_reference, historical_test)
    # Array 2: Forecast data array, where the elemtn of the array is the actual forecast
    historical = array[:, :(num_historical_samples+num_forecast_samples + num_historical_samples+num_forecast_samples)].reshape(-1, 2, num_historical_samples+num_forecast_samples, NUM_HISTORICAL_FEATURES)
    forecast = array[:, -num_forecast_samples:, TOTAL_FLOW_FEATURE_INDEX].reshape(-1, num_forecast_samples, NUM_FORECAST_FEATURES)
    print("HISTORICAL SHAPE BEFORE RANDOM SUBSET: ", historical.shape) # Shape = (num pairwise_instances, 2, historical_horizon, num_historical_features)
    print("FORECAST SHAPE BEFORE RANDOM SUBSET: ", forecast.shape) # Shape = (num pairwise_instances, forecast_horizon, num_forecast_features=1)

    # Assert that they each hold the same number of instances (we have a paired-input for an output)
    assert(historical.shape[0] == forecast.shape[0])

    #_______________[PLOT THE DATA]_______________
    # Uncomment to see what the data looks like
    window_start_index = min(0, historical.shape[0]) # Arbitrary indices chosen
    window_end_index = min(5, historical.shape[0])
    for i in range(window_start_index, window_end_index):
        x = np.arange(num_historical_samples + num_forecast_samples)
        sample_ref_historical_total_flow = historical[i, 0, :, TOTAL_FLOW_FEATURE_INDEX]
        #sample_ref_historical_day = historical[i, 0, :, WEEKLY_FEATURE_INDEX]

        sample_can_historical_total_flow = historical[i, 1, :, TOTAL_FLOW_FEATURE_INDEX]
        sample_can_historical_day = historical[i, 1, :, WEEKLY_FEATURE_INDEX]
        #print(sample_can_historical_total_flow.shape)
        plt.title("Matplotlib demo")
        plt.xlabel("x axis caption")
        plt.ylabel("y axis caption")

        plt.plot(x, sample_ref_historical_total_flow, color='purple')
        #plt.plot(x, sample_ref_historical_day/10.0, color='grey')

        plt.plot(x, sample_can_historical_total_flow, color='green')
        #plt.plot(x, sample_can_historical_day/10.0, color='black')

        x_forecast = np.arange(num_historical_samples, num_historical_samples + num_forecast_samples)
        sample_can_forecast_total_flow = forecast[i, :, 0]
        plt.plot(x_forecast, sample_can_forecast_total_flow, color='orange')

        plt.show()
    #_____________________________________________





    # Take a random subset out of those pairwise data instances. We do not necessarily have to take every pair; that would make our dataset too big
    random_subset_index = np.arange(historical.shape[0])
    np.random.shuffle(random_subset_index)
    random_subset_index = random_subset_index[:int(len(random_subset_index)*RATIO_PAIRWISE_INSTANCES_KEEP)]
    historical = historical[random_subset_index, :]
    forecast = forecast[random_subset_index, :]
    print("HISTORICAL SHAPE AFTER RANDOM SUBSET: ", historical.shape)
    print("FORECAST SHAPE AFTER RANDOM SUBSET: ", forecast.shape)
    return historical, forecast



def generate_pairwise_data(station_data, datetime_begin, datetime_end, num_historical_samples=NUM_HISTORICAL_SAMPLES, num_forecast_samples=NUM_FORECAST_SAMPLES,
                  num_historical_features=NUM_HISTORICAL_FEATURES, stride=NUM_STRIDE_SAMPLES):
    '''
        Generate pairwise data of a station
        Step 1: Add days feature to the year-long data
        Step 2: Apply sliding window to the year-long data and get an array of many windows
        Step 3: Normalize your windows
        Step 4: Cross your windows against each other, comparing the window days between each other to get pair inputs with the respective output

        The input is the reference historical and test historical
        The output is the test forecast
    '''

    # Add weekly feature
    print("DATA SHAPE AFTER BEFORE ADDING AUXILIARY FEATURES: {}".format(station_data.shape))
    station_data = add_weekly_dimension(station_data, datetime_begin=datetime_begin, datetime_end=datetime_end)
    print("DATA SHAPE AFTER ADDING WEEKLY FEATURE: {}".format(station_data.shape))

    # Add more auxiliary features if needed...
    # {...}

    # After adding all your auxiliary features, window the data
    station_windowed_data = window_data(data=station_data,
                                        w=num_historical_samples+num_forecast_samples,
                                        f=num_historical_features,
                                        stride=stride)

    # Normalize your data
    references, currents = normalize_data(references=station_windowed_data,
                                          currents=station_windowed_data,
                                          num_historical_samples=num_historical_samples,
                                          num_forecast_samples=num_forecast_samples)

    # Using two instances of the windowed data, cross all pairs of windowed data that share the same weekly feature
    historical, forecast = cross_data(references=references,
                                      currents=currents,
                                      num_historical_samples=num_historical_samples,
                                      num_forecast_samples=num_forecast_samples)
    return historical, forecast



def create_data(directory, number_stations, datetime_begin, datetime_end, stride):
    '''
    Create the pair_wise training data given a directory of processed station_data.pickle.
    The station_data.pickle files are created from all the CalTran's .gz files downloaded from a particular year.
    Notice the contents of the .gz file (when extracted and converted to .csv file) gives a list of stations data for
    a single day. We don't want that. Rather, we want a file to be a single station.pickle file with its year's worth of data.
    Before calling this function, we need to have a directory of these station.pickle
    The proper conversions and directories are created by running the PeMS.py application

    directory: The directory where all the station pickle files resides
    num_stations: We can use as many stations as needed. For example, I used 1500 for the training as to not make the training dataset too big
                    However, if the number_stations is greater than the available number of station.pickle files in the directory, then the
                    only all the station.pickle files is considered.
    datetime_begin: datetime.datetime that determines what time of the year to begin processing the data
    datetime_end: datetime.datetime that determines what time of the year to end processing the data
    '''

    # Based on the number_stations specified in the USER CONFIGURATIONS,
    # choose a random subset of station.pickle files from your directory of station.pickle files
    candidate_files = []
    for filename in os.listdir(directory):
        candidate_files.append(filename)
    candidate_files = random.sample(candidate_files, min(len(candidate_files), number_stations))

    # Create a list of historical_reference, historical_current, and forecast this would be fed into our neural network model where
    # X=(historical_references, historical_currents) and y=(forecasts)
    # Note that the order of these lists are respective to each other, meaning
    # historical_ref[i] pairs with the window historical_cur[i], which corresponds to the forecast[i]
    historical_ref = None
    historical_curr = None
    forecast = None

    for index, filename in enumerate(candidate_files):
        print('INDEX: {}'.format(index))
        print('GENERATING PAIRWISE DATA FOR STATION: {}'.format(filename))
        file = os.path.join(directory, filename)
        if os.path.isfile(file):
            # Read the pickle file to get the sample_station_df (remember, this data is the same, data we graphed)
            sample_station_df = pd.read_pickle(file)
            # Resample the sample_station_df (sampled originally at 5 minutes) to a new USER_CONFIGURATION sample period
            sample_station_df = sample_station_df.resample(SAMPLE_PERIOD).sum()
            # Also, take a subset of time of the sample_station_df, startinf from USER_CONFIGURATION datetime_begin
            start_index = int((datetime_begin-datetime(year=datetime_begin.year,month=1,day=1))/SAMPLE_PERIOD)
            # If the USER_CONFIGURATION datetime_end ends beyond the data available in the station pickle file, just choose the end_time available
            end_index = min(int((datetime_end-datetime(year=datetime_begin.year,month=1,day=1))/SAMPLE_PERIOD), len(sample_station_df))
            sample_station_df = sample_station_df.iloc[start_index:end_index]

            # In order to optimize speed, work with purely numpy arrays when generating the pairwise data
            station_data = sample_station_df.to_numpy()

            if index == 0:
                # When you encounter the first station, initialialize the pairwise data
                historical, forecast = generate_pairwise_data(station_data=station_data, datetime_begin=datetime_begin, datetime_end=datetime_end, stride=stride)
                # Split the paired windows to separate lists of windows, one is the reference list and the other is the current list
                historical_ref = historical[:, 0, :, :] # reference is the first of the pair (0 index)
                historical_curr = historical[:, 1, :, :] # current is the second of the pair (1 index)
            else:
                # Stack the other pairwise datas for each station
                # Only add pairwise data that is valid (there exists 0 NAN and 0 inf values)
                if np.count_nonzero(np.isnan(station_data)) == 0 and np.count_nonzero(np.isinf(station_data) == 0):
                    historical_add, forecast_add = generate_pairwise_data(station_data=station_data, datetime_begin=datetime_begin, datetime_end=datetime_end, stride=stride)
                    historical_ref_add = historical_add[:, 0, :, :]
                    historical_curr_add = historical_add[:, 1, :, :]
                    historical_ref = np.append(historical_ref, historical_ref_add, axis=0)
                    historical_curr = np.append(historical_curr, historical_curr_add, axis=0)
                    forecast = np.append(forecast, forecast_add, axis=0)
                else:
                    print("NAN OR INF DETECTED; IGNORING THIS TRAIN STATION DATA WINDOW")

    print("Pairwise Historical Reference Shape: {}".format(historical_ref.shape))
    print("Pairwise Historical Candidate Shape: {}".format(historical_curr.shape))
    print("Pairwise Forecast Shape: {}".format(forecast.shape))

    historical_ref, historical_curr, forecast = shuffle(historical_ref, historical_curr, forecast, random_state=RANDOM_SEED)
    return historical_ref, historical_curr, forecast



def save_data(filename, historical_ref, historical_curr, forecast):
    # https://stackoverflow.com/questions/35133317/numpy-save-some-arrays-at-once
    np.savez_compressed(os.path.join(DATA_DIR, filename),
                        historical_ref=historical_ref,
                        historical_curr=historical_curr,
                        forecast=forecast)



def load_data(filename):
    data = np.load(os.path.join(DATA_DIR, filename))
    historical_ref = data['historical_ref'] # Note that historical_ref represents historical_top
    historical_curr = data['historical_curr'] # Note that historical_curr represents historical_bot
    forecast = data['forecast']
    return historical_ref, historical_curr, forecast



def create_window_data_file(type, directory, year, district, stations, datetime_begin, datetime_end, stride):
    ''' Save the pairwise training/testing historical data and its respective forecast data '''
    historical_ref, historical_curr, forecast = create_data(directory=directory,
                                                            number_stations=stations,
                                                            datetime_begin=datetime_begin,
                                                            datetime_end=datetime_end,
                                                            stride=stride)

    assert(np.count_nonzero(np.isnan(historical_ref)) == 0)
    assert(np.count_nonzero(np.isnan(historical_curr)) == 0)
    assert(np.count_nonzero(np.isnan(forecast)) == 0)

    data_npz = type+'_year{}_district{}_stations{}_days{}_period{}min_history{}s_forecast{}s_stride{}s.npz'.format(
        year,
        district,
        stations,
        (datetime_end-datetime_begin).days,
        SAMPLE_PERIOD.seconds // 60,
        NUM_HISTORICAL_SAMPLES,
        NUM_FORECAST_SAMPLES,
        NUM_STRIDE_SAMPLES
    )

    save_data(data_npz, historical_ref, historical_curr, forecast)



def create_tinker_model_naive_test(num_historical_samples=NUM_HISTORICAL_SAMPLES,
                        num_historical_features=NUM_HISTORICAL_FEATURES,
                        num_forecast_samples=NUM_FORECAST_SAMPLES,
                        num_forecast_features=NUM_FORECAST_FEATURES):
    ''' A model that simply does not compute a difference vector.
    It just sends the reference forecast as the predicted forecast.
    '''
    # Define two input layers
    input_shape = (num_historical_samples+num_forecast_samples, num_historical_features)

    reference_input = Input(input_shape)
    candidate_input = Input(input_shape)

    # Preprocess the shape of the input data via split vector and cropping
    # https://keras.io/api/layers/reshaping_layers/cropping1d/
    reference_forecast = tf.keras.layers.Cropping1D(cropping=(num_historical_samples, 0))(reference_input)
    reference_forecast = tf.keras.layers.Reshape((num_forecast_samples, num_historical_features), input_shape=(num_forecast_samples, num_historical_features))(reference_forecast)
    reference_forecast = reference_forecast[:,:,TOTAL_FLOW_FEATURE_INDEX] # reduce to just 1 dimension (batch, size, feature)

    model = Model(inputs=[reference_input, candidate_input], outputs=reference_forecast)
    model.summary()

    return model



def create_difference_model(num_historical_samples=NUM_HISTORICAL_SAMPLES,
                        num_historical_features=NUM_HISTORICAL_FEATURES,
                        num_forecast_samples=NUM_FORECAST_SAMPLES,
                        num_forecast_features=NUM_FORECAST_FEATURES):
    ''' Takes the difference of the reference and historical candidate data '''
    # Define two input layers
    # Notice that their shapes are the same. That is not necessary for this model, and I will specify about it layer in this function
    input_shape = (num_historical_samples+num_forecast_samples, num_historical_features)
    reference_input = Input(input_shape)
    candidate_input = Input(input_shape)

    # Preprocess the shape of the input data via split vector and cropping
    # We are feeding in two windows of size (NUM_HISTORICAL_SAMPLES+NUM_FORECAST_SAMPLES, NUM_FEATURES)
    # We first split the reference window to be (1) (NUM_HISTORICAL_SAMPLES, NUM_FEATURES) and (2) (NUM_FORECAST_SAMPLES, 1) with cropping and reshaping
    #   (1) Will be used as the candidate historical input
    #   (2) WIll be used as the reference forecast to which the difference vector will be added
    # https://keras.io/api/layers/reshaping_layers/cropping1d/
    reference_historical = tf.keras.layers.Cropping1D(cropping=(0, num_forecast_samples))(reference_input)
    reference_historical = tf.keras.layers.Reshape((num_historical_samples, num_historical_features), input_shape=(num_historical_samples, num_historical_features))(reference_historical)
    reference_forecast = tf.keras.layers.Cropping1D(cropping=(num_historical_samples, 0))(reference_input)
    reference_forecast = tf.keras.layers.Reshape((num_forecast_samples, num_historical_features), input_shape=(num_forecast_samples, num_historical_features))(reference_forecast)
    reference_forecast = reference_forecast[:,:,TOTAL_FLOW_FEATURE_INDEX] # reduce to just 1 dimension (batch, size, feature)

    # The candidate historical is given a window of (NUM_HISTORICAL_SAMPLES+NUM_FORECAST_SAMPLES, NUM_FEATURES)
    # We want to ignore the FORECAST portion of that window, so crop it out
    # Therefore, when you run an actual test with our model, you would feed in the reference window and candidate window, BUT
    # The Forecast valeus of the candidate window can be anything (zeroed out for neatness) because those values won't be used.
    # We could modify our neural network model to change the candidate input size to just be (NUM_HISTORICAL_SAMPLES, NUM_FEATURES) instead of
    # (NUM_HISTORICAL_SAMPLES+NUM_FORECAST_SAMPLES, NUM_FEATURES), but for simplciity sake in demonstration, I had those two inputs be the same dimension
    candidate_historical = tf.keras.layers.Cropping1D(cropping=(0, num_forecast_samples))(candidate_input)
    candidate_historical = tf.keras.layers.Reshape((num_historical_samples, num_historical_features), input_shape=(num_historical_samples, num_historical_features))(candidate_historical)

    # Take the difference of the reference historical and candidate historical
    # This is our Difference layer to be fed into the LSTM Model
    difference_historical = Subtract()([reference_historical, candidate_historical])

    # LSTM Model
    # The hyperparameters of LSTM such as units and return sequences are chosen via iterative testing to find the best values
    # You could tweak these hyperparameters if you want to improve (but also notice that changes in the BATCH_SIZE, learning rate, and other learning_hyperparameters
    # will have confounding affects to these model_hyperparameters.
    # Notice that that last LSTM has the dimensions (num_forecast_samples*num_forecast_features) with return sequence=False and no Bidirectionality
    #   That is the last layer to be our difference vector, so we need to make it the size of our difference vector.
    #   You could instead remove that layer and create a Dense layer of the same dimension, but that led to overfitting the training data in my tests
    lstm_model = Sequential([
        Bidirectional(LSTM(4, return_sequences=True)),
        Bidirectional(LSTM(4, return_sequences=True)),
        Bidirectional(LSTM(4, return_sequences=True)),
        LSTM(num_forecast_samples * num_forecast_features, return_sequences=False)
    ])

    # Feed the difference into the lstm model
    difference_forecast = lstm_model(difference_historical)

    # Add the difference_forecast (the difference vector) to the reference forecast
    predict = Add()([difference_forecast, reference_forecast])

    # Define out models inputs and outputs
    model = Model(inputs=[reference_input, candidate_input], outputs=predict)

    # Print out the summary of the lstm_model specifically, then the entire model
    lstm_model.summary()
    model.summary()

    return model


def create_cnn_lstm_encoder_decoder_model(num_historical_samples=NUM_HISTORICAL_SAMPLES,
                        num_historical_features=NUM_HISTORICAL_FEATURES,
                        num_forecast_samples=NUM_FORECAST_SAMPLES,
                        num_forecast_features=NUM_FORECAST_FEATURES):
    ''' Similar to the model before; but instead of taking the difference between the historical_reference and historical_candidate immediately,
     first feed them both to a Siamese CNN neural network to extract more robust feature maps. Then take the difference between those feature maps.
     Finally, feed that difference_feature_map to an LSTM model and output a difference_vector/difference_forecast. '''

    # Define two input layers
    input_shape = (num_historical_samples + num_forecast_samples, num_historical_features)

    reference_input = Input(input_shape) # (BATCH_SIZE, HISTORICAL_HORIZON+FORECAST_HORIZON, NUM_FEATURES=2)
    candidate_input = Input(input_shape) # (BATCH_SIZE, HISTORICAL_HORIZON+FORECAST_HORIZON, 1)

    # Preprocess the shape of the inputs like in the difference model
    # https://keras.io/api/layers/reshaping_layers/cropping1d/
    reference_historical = tf.keras.layers.Cropping1D(cropping=(0, num_forecast_samples))(reference_input) # axis to crop is 0, the time dimension
    ##reference_historical = tf.keras.layers.Reshape((num_historical_samples, num_historical_features),
    ##                                               input_shape=(num_historical_samples, num_historical_features))(reference_historical)

    reference_forecast = tf.keras.layers.Cropping1D(cropping=(num_historical_samples, 0))(reference_input)
    ##reference_forecast = tf.keras.layers.Reshape((num_forecast_samples, num_historical_features),
    ##                                             input_shape=(num_forecast_samples, num_historical_features))(reference_forecast)

    reference_forecast = reference_forecast[:, :,
                         TOTAL_FLOW_FEATURE_INDEX]  # reduce to just 1 dimension (batch, size, feature)

    candidate_historical = tf.keras.layers.Cropping1D(cropping=(0, num_forecast_samples))(candidate_input)
    ##candidate_historical = tf.keras.layers.Reshape((num_historical_samples, num_historical_features),
    ##                                               input_shape=(num_historical_samples, num_historical_features))(candidate_historical)

    # Create out CNN model to create feature maps
    # I used average pooling instead of max pooling to not generalize too much because we are going to further generalize by taking the difference of the feature maps
    # The number of filters are inversely proportional to the kernel_size. I thought, intuitively, I want to focus on more features of smaller sections of the historical
    # data, rather than have many features of large portions. If we had many feature for large portions, I would think that the model would overfit to the training windows
    # But you can explore different hyperparameters of this model, especially when you make it more multivariate
    # https://stackoverflow.com/questions/65006011/size-of-output-of-a-conv1d-layer-in-keras
    cnn_model = Sequential([
        Conv1D(filters=(int)(64 / 8), kernel_size=(int)(num_historical_samples / 2), strides=1, padding='causal',
               activation='relu'),
        AveragePooling1D(pool_size=4, strides=2), # regularization method
        Conv1D(filters=(int)(64 / 4), kernel_size=(int)(num_historical_samples / 4), strides=1, padding='causal',
               activation='relu'),
        Conv1D(filters=(int)(64 / 2), kernel_size=(int)(num_historical_samples / 8), strides=1, padding='causal',
               activation='relu'),
        AveragePooling1D(pool_size=3, strides=1),
        Conv1D(filters=(int)(64 / 1), kernel_size=(int)(num_historical_samples / 16), strides=1, padding='causal',
               activation='relu'),
    ])

    # Feed out inputs to the same instances of the cnn_model. Since we are using a Siamese Network architecture, the two twin
    # models would share the same weights when they are being trained. So instead of having two separate instances of the cnn_model,
    # siamese twins effectively means that we need only one instance.
    # If we had two separate instances, the model would not be siamese twins. They would akin to be regular siblings with different weights
    # We do not want that, as it does not make our model symmetric and it may have one sibling overcompensate for the other.
    reference_feature_map = cnn_model(reference_historical)
    candidate_feature_map = cnn_model(candidate_historical)

    # Subtract the difference maps
    difference_historical = Subtract()([reference_feature_map, candidate_feature_map])

    # LSTM model like in the difference model
    lstm_model = Sequential([
        Bidirectional(LSTM(4, return_sequences=True)),
        Bidirectional(LSTM(4, return_sequences=True)),
        Bidirectional(LSTM(4, return_sequences=True)),
        LSTM(num_forecast_samples * num_forecast_features, return_sequences=False) # Unwrap and reduction to the proper forecast dimension
    ])

    # Get the difference_vector/difference_forecast
    difference_forecast = lstm_model(difference_historical)

    # Our predicted forecast = difference_vector + reference_forecast
    predict = Add()([difference_forecast, reference_forecast])

    # Define our models inputs and outputs
    model = Model(inputs=[reference_input, candidate_input], outputs=predict)

    print("Plot the siamese twins model to see its configuration")
    cnn_model.summary()
    plot_model(cnn_model, to_file='cnn_siamese_model.png', show_shapes=True, show_layer_names=True,
               expand_nested=True)

    model.summary()

    print("Plot the entire model to see its configuration, and how/why we preprocessed the two inputs")
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    return model



if __name__ == '__main__':
    # Random seeding for numpy random and python random
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


    # Print time started
    begin_time = datetime.now()
    str_begin_time = begin_time.strftime("%H:%M:%S")
    print("Begin Time =", str_begin_time)

    '''
    # Step 1: These will create your training .npz file
    # Create training data
    create_window_data_file(type=TRAINING_TYPE,
                directory=TRAINING_DIRECTORY,
                year=TRAINING_YEAR,
                district=TRAINING_DISTRICT,
                stations=NUM_TRAINING_STATIONS,
                datetime_begin=TRAINING_DATETIME_BEGIN,
                datetime_end=TRAINING_DATETIME_END,
                stride=NUM_STRIDE_SAMPLES)


    # Step 2: These will create your training .npz file
    # Create testing data
    create_window_data_file(type=TESTING_TYPE,
            directory=TESTING_DIRECTORY,
            year=TESTING_YEAR,
            district=TESTING_DISTRICT,
            stations=NUM_TESTING_STATIONS,
            datetime_begin=TESTING_DATETIME_BEGIN,
            datetime_end=TESTING_DATETIME_END,
            stride=NUM_STRIDE_SAMPLES)
    '''


    # Step 3: After creating your training and testing .npz files, update your PREMADE_TRAINING_DATA and PREMADE_TESTING_DATA names to match
    print("Loading pairwise train instances")
    historical_ref_train, historical_curr_train, forecast_train = load_data(PREMADE_TRAINING_DATA)
    # Check for no NaN values in our training windows
    assert(np.count_nonzero(np.isnan(historical_ref_train)) == 0)
    assert(np.count_nonzero(np.isnan(historical_curr_train)) == 0)
    assert(np.count_nonzero(np.isnan(forecast_train)) == 0)

    print("Loading pairwise test instances")
    historical_ref_test, historical_curr_test, forecast_test = load_data(PREMADE_TESTING_DATA)
    # Check for no NaN values in our testing windows
    assert(np.count_nonzero(np.isnan(historical_ref_test)) == 0)
    assert(np.count_nonzero(np.isnan(historical_curr_test)) == 0)
    assert(np.count_nonzero(np.isnan(forecast_test)) == 0)


    # Step 4: Optional Reduction
    # We are only taking every fourth window (effectively reducing the size of our training dataset even further by a factor of n)
    # This is to be able to run some quick training to see if the model is converging properly.
    # Once it is, we can set the reduce scale to 1 to train with the entire dataset
    # You may reduce the size even more, but be wary that this may lead to overfitting the model based on your hyperparameters!
    # I would NOT adjust the size of the training dataset to adhere to your hyperparameters; rather, adjust the hyperparameters to adhere to your training set!
    REDUCE_SCALE = 4
    historical_ref_train = historical_ref_train[:historical_ref_train.shape[0]//REDUCE_SCALE]
    historical_curr_train = historical_curr_train[:historical_curr_train.shape[0]//REDUCE_SCALE]
    forecast_train = forecast_train[:forecast_train.shape[0]//REDUCE_SCALE]

    historical_ref_test = historical_ref_test[:historical_ref_test.shape[0]//REDUCE_SCALE]
    historical_curr_test = historical_curr_test[:historical_curr_test.shape[0]//REDUCE_SCALE]
    forecast_test = forecast_test[:forecast_test.shape[0]//REDUCE_SCALE]

    # Step 5: Set up training for your models here
    # model = create_tinker_model_naive_test()
    # model = create_difference_model()
    model = create_cnn_lstm_encoder_decoder_model()

    # Learning_hyper_parameters
    # We are using the Adam optimizer, for it is most common for learning
    # The initial learning rate is 0.0001 * np.sqrt(BATCH_SIZE) (but this could be tweaked by a magnitude like 0.001 for faster learning (but it may yield a less accurate model)
    # Gradient clipping = 1.0 for generalization (but this could be tweaked)
    adam = tf.keras.optimizers.Adam(learning_rate=INIT_LEARNING_RATE * np.sqrt(BATCH_SIZE), clipnorm=CLIPNORM)

    # I used Mean Absolute Error (MAE) as our loss function (rather than the more common Mean Squared Error (MSE)) because I did not
    # want one single spike or deficit of a certain sample in the window to cause a major effect in the accuracy of our forecast.
    # In other words, one sample that is an anomaly should not hold so much impact over the other histroical samples
    # However, I still tracked the MSE in the metrics argument for curiosity
    model.compile(loss='mae',
                  optimizer=adam,
                  metrics=[
                      metrics.MeanSquaredError(name='mse'),
                      metrics.MeanAbsoluteError(name='mae')])


    # When we get a latest best model during training, we will save it to a certain path
    epoch_best_model_checkpoint_filename = r'best_model_best_lstm_sequence_sequence{epoch:08d}.h5'
    #epoch_best_model_checkpoint_filename = r'best_model_best_lstm_sequence_sequence.h5'
    epoch_model_best_checkpoint_filepath = os.path.join(MODEL_CHECKPOINT_DIR,
                                                        epoch_best_model_checkpoint_filename)


    # We can also save the newest(latest created) model even if it performed worse than the best
    #https://stackoverflow.com/questions/54323960/save-keras-model-at-specific-epochs
    epoch_latest_model_checkpoint_filename = r'latest_model_latest_lstm_sequence_sequence.h5'
    epoch_model_latest_checkpoint_filepath = os.path.join(MODEL_CHECKPOINT_DIR,
                                                          epoch_latest_model_checkpoint_filename)


    # Save the best model callback for the model.fit function below
    best_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(epoch_model_best_checkpoint_filepath,
                                                               monitor='val_mae', mode='min',
                                                               verbose=1, save_weights_only=False, save_best_only=True)
    # Save the latest model callback for the model.fit function below
    latest_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(epoch_model_latest_checkpoint_filepath,
                                                                 monitor='mse',
                                                                 verbose=0,
                                                                 save_best_only=False,
                                                                 save_weights_only=False,
                                                                 mode='auto',
                                                                 save_freq=1)

    # Step 6: Train your model
    # [!] Note that when training, the program will print the progress of our model after every epoch. I used that series of printed information,
    # parsed it, and plotted a learning curve of the MAE from that data. It is important to SAVE this training history information or else you will
    # not get it back except by retraining the model all over again.
    # Though, there should be a method to automatically write out the training information to a file after every training epoch in addition to printing it out.
    # The best way to do this is to use TensorBoard library. This will automatically collect your training data and plot it. Though, you would have to the Tensorboard API for it
    # Also, you can extract the models weights by using model.get_weights(), then plot the weighs in a violin plot to see which layers are
    # saturated or not for further analysis.
    model.fit(x=(historical_ref_train, historical_curr_train), y=forecast_train,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              use_multiprocessing=True,
              verbose=1,
              validation_data=((historical_ref_test, historical_curr_test), forecast_test),
              callbacks=[best_model_checkpoint, latest_model_checkpoint],
              shuffle=True
              )


    # Print time finished
    end_time = datetime.now()
    str_end_time = end_time.strftime("%H:%M:%S")
    print("End Time =", str_end_time)

    print("timedelta: {}".format(end_time-begin_time))
    print("Done")

