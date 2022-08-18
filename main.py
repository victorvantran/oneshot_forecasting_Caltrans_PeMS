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





# How much time per data sampled
SAMPLE_PERIOD = timedelta(minutes=30)
NUM_SAMPLES_PER_DAY = int(timedelta(days=1) / SAMPLE_PERIOD)

# Start data at 140 days from the calender year to prevent any day light savings shenanigans
OFFSET_DAYS = 140

# Historical Data is the data used to forecast
NUM_HISTORICAL_DAYS = 3
HISTORICAL_LENGTH = int(NUM_SAMPLES_PER_DAY * NUM_HISTORICAL_DAYS)
# The 2 historical features are (0) Traffic Total Flow and (1) The day in the week; We are only working with 2 variables here
NUM_HISTORICAL_FEATURES = 2
TOTAL_FLOW_FEATURE_INDEX = 0
DAY_FEATURE_INDEX = 1

# Forecast data
NUM_FORECAST_DAYS = 0.5
FORECAST_LENGTH = int(NUM_SAMPLES_PER_DAY * NUM_FORECAST_DAYS)
# The 1 forecast feature is Traffic Total Flow; We are not focusing on predicting anything else
NUM_FORECAST_FEATURES = 1

# The number of days to skip when taking the next sliding window instance. This can be a floating point, such as 0.25 for 8 hours of the day
NUM_STRIDE_DAYS = 1
STRIDE_LENGTH = int(NUM_SAMPLES_PER_DAY * NUM_STRIDE_DAYS)

# Learning Hyperparameters
BATCH_SIZE = 256
INIT_LEARNING_RATE = 0.0001
CLIPNORM = 0.1

# Path to the directory of the dataset
DATA_DIR = r'dataset\model_data'

# TRAINING DATA PARAMETERS
TRAINING_TYPE = 'train'
TRAINING_DIRECTORY = r'dataset\station_data\2019\district_12' # These would hold pickle files created from .gz files using the PeMS.py class
TRAINING_YEAR = 2019
TRAINING_DISTRICT = 12
NUM_TRAINING_STATIONS = 1500
NUM_TRAINING_DAYS = 56
TRAINING_START_DAY = 2
TRAINING_OFFSET_DAY = OFFSET_DAYS

# TESTING DATA PARAMETERS
TESTING_TYPE = 'test'
TESTING_DIRECTORY = r'dataset\station_data\2019\district_3' # These would hold pickle files created from .gz files using the PeMS.py class
TESTING_YEAR = 2019
TESTING_DISTRICT = 3
NUM_TESTING_STATIONS = 400
NUM_TESTING_DAYS = 56
TESTING_START_DAY = 2
TESTING_OFFSET_DAY = OFFSET_DAYS

PREMADE_TRAINING_DATA = r'train_year2019_district12_stations1500_days56_period30min_history144s_forecast24s.npz'
PREMADE_TESTING_DATA = r'test_year2019_district3_stations400_days56_period30min_history144s_forecast24s.npz'


# Random seed to make "randomization" repeatable as long as the seed is the same
RANDOM_SEED = 8675309




# Configure GPU to allocate more memory if needed until maxed
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)



def window_data(data, w=HISTORICAL_LENGTH + FORECAST_LENGTH, f=NUM_HISTORICAL_FEATURES, stride=STRIDE_LENGTH):
    ''' Rolling window on the data, truncate end of data that does not have enough data for the next/last stride '''
    return np.lib.stride_tricks.sliding_window_view(x=data, window_shape=(w, f))[::stride].reshape(-1, w, f)


def add_days_dimension(data, sample_period=SAMPLE_PERIOD, start_day=0):
    ''' Add the days-of-the-week-feature to the numpy arry [0-Sunday, 1-Monday, ... 6-Saturday] '''
    weekly_repeat = np.repeat(np.mod(np.add(np.arange(7), start_day), 7), NUM_SAMPLES_PER_DAY)
    num_weeks = data.shape[0] // 7
    day_data = np.hstack((weekly_repeat,) * (num_weeks + 1))[:data.shape[0]]
    return np.stack((data, day_data), axis=1)

def cross_data(references, candidates, historical_length=HISTORICAL_LENGTH, forecast_length=FORECAST_LENGTH):
    '''
    :param references: (num_instances, historical+forecast length, num_features)
    :param candidates: (num_instances, historical+forecast length, num_features)
    :param historical_length:
    :param forecast_length:
    :return: Two arrays:
        1st array is [(historical_ref,historical_candidate), ...]
        2nd array is [(forecast_difference)]

    Array of [historical, historical, difference] instances given from
    two arrays of shape (numinstances, historicallength+forecastlength, numfeatures)
    Basically, cross the sliding windows of two difference year-worth-long station data.
    '''
    l = list()
    assert references.shape[0] > 0
    assert candidates.shape[0] > 0
    assert references.shape[1] == historical_length + forecast_length
    assert candidates.shape[1] == historical_length + forecast_length

    references = references.copy()
    candidates = candidates.copy()

    # Normalize reference window by the reference historical by (a_element - a_min)/(a_max-a_min)
    # We are using vectorization to deal with operations of numpy arrays. This is many mgnitudes faster than using for-loops
    # For numpy arrays and panda's dataframes, you should always use vectorization to manipulate
    references_historical = references[:, :historical_length, ]
    references_historical_max = np.expand_dims(np.amax(references_historical, axis=1)[:,
                            TOTAL_FLOW_FEATURE_INDEX], axis=1)
    references_historical_min = np.expand_dims(np.amin(references_historical, axis=1)[:,
                            TOTAL_FLOW_FEATURE_INDEX], axis=1)
    references[:, :, TOTAL_FLOW_FEATURE_INDEX] = \
        np.divide(np.subtract(references[:, :, TOTAL_FLOW_FEATURE_INDEX], references_historical_min),
                  np.subtract(references_historical_max, references_historical_min))

    # Normalize candidate window by the candidate historical by (a_element - a_min)/(a_max-a_min)
    candidates_historical = candidates[:, :historical_length, ]
    candidates_historical_max = np.expand_dims(np.amax(candidates_historical, axis=1)[:,
                            TOTAL_FLOW_FEATURE_INDEX], axis=1)
    candidates_historical_min = np.expand_dims(np.amin(candidates_historical, axis=1)[:,
                            TOTAL_FLOW_FEATURE_INDEX], axis=1)

    candidates[:, :, TOTAL_FLOW_FEATURE_INDEX] = \
        np.divide(np.subtract(candidates[:, :, TOTAL_FLOW_FEATURE_INDEX], candidates_historical_min),
                  np.subtract(candidates_historical_max, candidates_historical_min))


    for reference in references:
        # Skip if nan encountered. This is due to the numpy having missing data (NaN) due to unclean CalTrans data
        if np.count_nonzero(np.isnan(reference)):
            continue
        reference_days = reference[:, DAY_FEATURE_INDEX]
        for candidate in candidates:
            # Skip if nan encountered. This is due to the numpy having missing data (NaN) due to unclean CalTrans data
            if np.count_nonzero(np.isnan(candidate)):
                continue
            candidate_days = candidate[:, DAY_FEATURE_INDEX]
            # If they two windows share the same days features, then they are good windows to pair
            # It would not make sense to compare a window of [Monday, Tuesday, Wednesday] to a window of [Friday, Saturday, Sunday]
            if (np.array_equal(reference_days, candidate_days)):
                reference_input = reference

                candidate_input = np.copy(candidate)
                candidate_input[historical_length:, TOTAL_FLOW_FEATURE_INDEX] = 0

                candidate_output = candidate[-forecast_length:]

                # An instance of a window: Use the reference input and candidate input to get the candidate output
                # Candidate window is synonymous to Test window
                instance = np.concatenate((reference_input, candidate_input,
                                           candidate_output))

                historical_flow_difference = np.abs(np.subtract(reference[:historical_length, TOTAL_FLOW_FEATURE_INDEX],
                                                       candidate[:historical_length, TOTAL_FLOW_FEATURE_INDEX]))

                forecast_flow_difference = np.abs(np.subtract(reference[-forecast_length:, TOTAL_FLOW_FEATURE_INDEX],
                                                    candidate[-forecast_length:, TOTAL_FLOW_FEATURE_INDEX]))

                # If the difference of the historical or forecast between the two windows are too difference, we are assuming that
                # one or both of the windows are "anomalies" (such as car crashes and stuff). This is not good for training, since we
                # don't have car crashes/other variabels for features. So we will detect these anomalies and cull them out.
                MAX_HISTORICAL_DIFFERENCE_THRESHOLD = 0.30
                MAX_FORECAST_DIFFERENCE_THRESHOLD = 0.30

                historical_num_exceed_difference = (historical_flow_difference > MAX_HISTORICAL_DIFFERENCE_THRESHOLD).sum()
                forecast_num_exceed_difference = (forecast_flow_difference > MAX_FORECAST_DIFFERENCE_THRESHOLD).sum()

                MAX_HISTORICAL_NUM_EXCEED_DIFFERENCE_THRESHOLD = 0.10
                MAX_FORECAST_NUM_EXCEED_DIFFERENCE_THRESHOLD = 0.10

                MAX_ZERO_THRESHOLD = 0.5

                # Cull anomalies
                if historical_num_exceed_difference <= int(MAX_HISTORICAL_NUM_EXCEED_DIFFERENCE_THRESHOLD * historical_length) and \
                        forecast_num_exceed_difference <= int(MAX_FORECAST_NUM_EXCEED_DIFFERENCE_THRESHOLD * forecast_length) and \
                        (reference[:historical_length] >= 0.0).sum() >= int(MAX_ZERO_THRESHOLD * historical_length) and \
                        (candidate[:historical_length] >= 0.0).sum() >= int(MAX_ZERO_THRESHOLD * historical_length) and \
                        (reference[-forecast_length:] >= 0.0).sum() >= int(MAX_ZERO_THRESHOLD * forecast_length) and \
                        (candidate[-forecast_length:] >= 0.0).sum() >= int(MAX_ZERO_THRESHOLD * forecast_length):
                        l.append(instance)
                else:
                    pass #print("IGNORING INSTANCE DUE TO NAN FOUND OR MAX EXCEEDED")


    array = np.asarray(l)

    # Separate the array to two arrays:
    # Historical data array, where an element of the array: (historical_reference, historical_test)
    # Forecast data array, where the elemtn of the array: (difference of forecast between the reference_forecast and historical_forecast)
    # Note that, for example, index 0 of the historical and forecast array would coincide. You would use historical[0] to get forecast[0].
    historical = array[:, :(historical_length+forecast_length + historical_length+forecast_length)].reshape(-1, 2, historical_length+forecast_length, NUM_HISTORICAL_FEATURES)
    forecast = array[:, -forecast_length:, TOTAL_FLOW_FEATURE_INDEX].reshape(-1, forecast_length, NUM_FORECAST_FEATURES)
    print("HISTORICAL SIZE: ", historical.shape)
    print("FORECAST SIZE: ", forecast.shape)


    # Uncomment to see what the data looks like
    '''
    for i in range(320, 325):
        x = np.arange(historical_length + forecast_length)
        sample_ref_historical_total_flow = historical[i, 0, :, TOTAL_FLOW_FEATURE_INDEX]
        sample_ref_historical_day = historical[i, 0, :, DAY_FEATURE_INDEX]

        sample_can_historical_total_flow = historical[i, 1, :, TOTAL_FLOW_FEATURE_INDEX]
        sample_can_historical_day = historical[i, 1, :, DAY_FEATURE_INDEX]
        print(sample_can_historical_total_flow.shape)
        plt.title("Matplotlib demo")
        plt.xlabel("x axis caption")
        plt.ylabel("y axis caption")

        plt.plot(x, sample_ref_historical_total_flow, color='blue')
        plt.plot(x, sample_ref_historical_day/10.0, color='grey')

        plt.plot(x, sample_can_historical_total_flow, color='red')
        plt.plot(x, sample_can_historical_day/10.0, color='black')

        x_forecast = np.arange(historical_length, historical_length + forecast_length)
        sample_can_forecast_total_flow = forecast[i, :, 0]
        plt.plot(x_forecast, sample_can_forecast_total_flow, color='orange')

        plt.show()
    '''

    return historical, forecast



def generate_pairwise_data(data, historical_length=HISTORICAL_LENGTH, forecast_length=FORECAST_LENGTH,
                  num_historical_features=NUM_HISTORICAL_FEATURES, stride=STRIDE_LENGTH,
                  start_day=0):
    ''' Step 1: Add days feature to the year-long data
        Step 2: Apply sliding window to the year-long data and get an array of many windows
        Step 3: Compare the window days between each other to get pair inputs with the respective output
            The input is the reference historical and test historical
            The output is the test forecast
    '''
    data_with_days = add_days_dimension(data, start_day=start_day)
    windowed_data = window_data(data_with_days, historical_length + forecast_length, num_historical_features, stride)
    historical, forecast = cross_data(windowed_data, windowed_data, historical_length, forecast_length)
    return historical, forecast



def create_data(directory, number_stations, num_days, start_day, offset_day):
    '''
    Create the pair_wise training data given a directory of processed station_data.pickle.
    The station_data.pickle files are created from all the CalTran's .gz files downloaded from a particular year.
    Notice the contents of the .gz file (when extracted and converted to .csv file) gives a list of stations data for
    a single day. We don't want that. Rather, we want a file to be a single station.pickle file with its year's worth of data.
    The conversion to do this is done in the pycharm project named "caltrans_dataset"
    Before calling this function, we need to have a directory of these station.pickle

    directory: This is important when we add the Days feature to our training/testing windows
    num_stations: We can use as many stations as needed. For example, I used 1500 for the training as to not make the testing dataset to big
    num_days: We are using NUM_TESTING_DAYS out of the 365(year) days available in our training dataset.
    start_day: Remember, for example, January 1st, 2019 started off as a Tuesday,
                so our days_of_the_week feature will start at Tuesday=2 for the year 2019
    offset_day:  We can start the sliding window at any offset of the year.
                I offsetted by the amount that skips daylgiht savings time to avoid any misaligned data.
                Notice on the .csv files how CalTrans Actually "skips an hour" on March and essentially
                loses an hours worth of Data coming in on Novemenber.
                Our "effective" time does not match the data given.
                Yes we are moving 1 hour forward in our clocks, but we are not time traveling 1 hour.
                So we can impute the data to ignore the time shift by moving the skipped hours, 1 hour backwards.
                Then on December, recompensate for the missing hour by repeating the samples of 2:00am - 3:00am
                That sounds complicated, but you may avoid this by using an offset
                However, by doing so, you will not get the first OFFSET_DAYS of window data
    '''
    candidate_files = []
    historical_top = None
    historical_bot = None
    forecast = None

    for filename in os.listdir(directory):
        candidate_files.append(filename)

    candidate_files = random.sample(candidate_files, min(len(candidate_files), number_stations))

    for index, filename in enumerate(candidate_files):
        print('INDEX: {}'.format(index))
        file = os.path.join(directory, filename)
        if os.path.isfile(file):
            sample_station_df = pd.read_pickle(file)
            sample_station_df = sample_station_df.resample(SAMPLE_PERIOD).sum().iloc[
                                NUM_SAMPLES_PER_DAY * offset_day: NUM_SAMPLES_PER_DAY * (OFFSET_DAYS + num_days + 1)]
            data = sample_station_df.to_numpy().reshape(-1)

            if index == 0:
                # initialialize
                historical, forecast = generate_pairwise_data(data=data, start_day=start_day+(offset_day % 7))
                historical_top = historical[:, 0, :, :]
                historical_bot = historical[:, 1, :, :]
            else:
                if np.count_nonzero(np.isnan(data)) == 0:
                    historical_add, forecast_add = generate_pairwise_data(data=data, start_day=start_day+(offset_day % 7))
                    historical_top_add = historical_add[:, 0, :, :]
                    historical_bot_add = historical_add[:, 1, :, :]
                    historical_top = np.append(historical_top, historical_top_add, axis=0)
                    historical_bot = np.append(historical_bot, historical_bot_add, axis=0)
                    forecast = np.append(forecast, forecast_add, axis=0)
                else:
                    print("NAN DETECTED; IGNORING THIS TRAIN STATION DATA WINDOW")

    print(historical_top.shape)
    print(historical_bot.shape)
    print(forecast.shape)

    historical_top, historical_bot, forecast = shuffle(historical_top, historical_bot, forecast, random_state=RANDOM_SEED)
    return historical_top, historical_bot, forecast




def save_data(filename, historical_top, historical_bot, forecast):
    # https://stackoverflow.com/questions/35133317/numpy-save-some-arrays-at-once
    np.savez_compressed(DATA_DIR + r'\{}'.format(filename),
                        historical_top=historical_top,
                        historical_bot=historical_bot,
                        forecast=forecast)

def load_data(filename):
    data = np.load(DATA_DIR + r'\{}'.format(filename))
    historical_top = data['historical_top'] # historical_top = historical_ref
    historical_bot = data['historical_bot'] # historical_bot = historical_cand
    forecast = data['forecast']
    return historical_top, historical_bot, forecast



def create_window_data_file(type, directory, year, district, stations, days, start_day, offset_day):
    ''' Save the pairwise training/testing historical data and its respective forecast data '''
    historical_ref_train, historical_cand_train, forecast_train = create_data(directory=directory,
                                                                             number_stations=stations,
                                                                             num_days=days,
                                                                             start_day=start_day,
                                                                             offset_day=offset_day)

    assert(np.count_nonzero(np.isnan(historical_ref_train)) == 0)
    assert(np.count_nonzero(np.isnan(historical_cand_train)) == 0)
    assert(np.count_nonzero(np.isnan(forecast_train)) == 0)

    data_npz = type+'_year{}_district{}_stations{}_days{}_period{}min_history{}s_forecast{}s.npz'.format(
        year,
        district,
        stations,
        days,
        SAMPLE_PERIOD.seconds // 60,
        HISTORICAL_LENGTH,
        FORECAST_LENGTH
    )

    save_data(data_npz, historical_ref_train, historical_cand_train, forecast_train)




def create_tinker_model_naive_test(historical_length=HISTORICAL_LENGTH,
                        num_historical_features=NUM_HISTORICAL_FEATURES,
                        forecast_length=FORECAST_LENGTH,
                        num_forecast_features=NUM_FORECAST_FEATURES):
    ''' A model that simply does not compute a difference vector.
    It just sends the reference forecast as the predicted forecast.
    '''
    # Define two input layers
    input_shape = (historical_length+forecast_length, num_historical_features)

    reference_input = Input(input_shape)
    candidate_input = Input(input_shape)

    # Preprocess the shape of the input data via split vector and cropping
    # https://keras.io/api/layers/reshaping_layers/cropping1d/
    reference_forecast = tf.keras.layers.Cropping1D(cropping=(historical_length, 0))(reference_input)
    reference_forecast = tf.keras.layers.Reshape((forecast_length, num_historical_features), input_shape=(forecast_length, num_historical_features))(reference_forecast)
    reference_forecast = reference_forecast[:,:,TOTAL_FLOW_FEATURE_INDEX] # reduce to just 1 dimension (batch, size, feature)

    model = Model(inputs=[reference_input, candidate_input], outputs=reference_forecast)

    model.summary()

    return model





def create_difference_model(historical_length=HISTORICAL_LENGTH,
                        num_historical_features=NUM_HISTORICAL_FEATURES,
                        forecast_length=FORECAST_LENGTH,
                        num_forecast_features=NUM_FORECAST_FEATURES):
    ''' Takes the difference of the reference and historical candidate data '''
    # Define two input layers
    # Notice that their shapes are the same. That is not necessary for this model, and I will specify about it layer in this function
    input_shape = (historical_length+forecast_length, num_historical_features)
    reference_input = Input(input_shape)
    candidate_input = Input(input_shape)

    # Preprocess the shape of the input data via split vector and cropping
    # We are feeding in two windows of size (HISTORICAL_LENGTH+FORECAST_LENGTH, NUM_FEATURES)
    # We first split the reference window to be (1) (HISTORICAL_LENGTH, NUM_FEATURES) and (2) (FORECAST_LENGTH, 1) with cropping and reshaping
    #   (1) Will be used as the candidate historical input
    #   (2) WIll be used as the reference forecast to which the difference vector will be added
    # https://keras.io/api/layers/reshaping_layers/cropping1d/
    reference_historical = tf.keras.layers.Cropping1D(cropping=(0, forecast_length))(reference_input)
    reference_historical = tf.keras.layers.Reshape((historical_length, num_historical_features), input_shape=(historical_length, num_historical_features))(reference_historical)
    reference_forecast = tf.keras.layers.Cropping1D(cropping=(historical_length, 0))(reference_input)
    reference_forecast = tf.keras.layers.Reshape((forecast_length, num_historical_features), input_shape=(forecast_length, num_historical_features))(reference_forecast)
    reference_forecast = reference_forecast[:,:,TOTAL_FLOW_FEATURE_INDEX] # reduce to just 1 dimension (batch, size, feature)

    # The candidate historical is given a window of (HISTORICAL_LENGTH+FORECAST_LENGTH, NUM_FEATURES)
    # We want to ignore the FORECAST portion of that window, so crop it out
    # Therefore, when you run an actual test with our model, you would feed in the reference window and candidate window, BUT
    # The Forecast valeus of the candidate window can be anything (zeroed out for neatness) because those values won't be used.
    # We could modify our neural network model to change the candidate input size to just be (HISTORICAL_LENGTH, NUM_FEATURES) instead of
    # (HISTORICAL_LENGTH+FORECAST_LENGTH, NUM_FEATURES), but for simplciity sake in demonstration, I had those two inputs be the same dimension
    candidate_historical = tf.keras.layers.Cropping1D(cropping=(0, forecast_length))(candidate_input)
    candidate_historical = tf.keras.layers.Reshape((historical_length, num_historical_features), input_shape=(historical_length, num_historical_features))(candidate_historical)

    # Take the difference of the reference historical and candidate historical
    # This is our Difference layer to be fed into the LSTM Model
    difference_historical = Subtract()([reference_historical, candidate_historical])

    # LSTM Model
    # The hyperparameters of LSTM such as units and return sequences are chosen via iterative testing to find the best values
    # You could tweak these hyperparameters if you want to improve (but also notice that changes in the BATCH_SIZE, learning rate, and other learning_hyperparameters
    # will have confounding affects to these model_hyperparameters.
    # Notice that that last LSTM has the dimensions (forecast_length*num_forecast_features) with return sequence=False and no Bidirectionality
    #   That is the last layer to be our difference vector, so we need to make it the size of our difference vector.
    #   You could instead remove that layer and create a Dense layer of the same dimension, but that led to overfitting the training data in my tests
    lstm_model = Sequential([
        Bidirectional(LSTM(4, return_sequences=True)),
        Bidirectional(LSTM(4, return_sequences=True)),
        Bidirectional(LSTM(4, return_sequences=True)),
        LSTM(forecast_length * num_forecast_features, return_sequences=False)
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


def create_cnn_lstm_encoder_decoder_model(historical_length=HISTORICAL_LENGTH,
                        num_historical_features=NUM_HISTORICAL_FEATURES,
                        forecast_length=FORECAST_LENGTH,
                        num_forecast_features=NUM_FORECAST_FEATURES):
    ''' Similar to the model before; but instead of taking the difference between the historical_reference and historical_candidate immediately,
     first feed them both to a Siamese CNN neural network to extract more robust feature maps. Then take the difference between those feature maps.
     Finally, feed that difference_feature_map to an LSTM model and output a difference_vector/difference_forecast. '''

    # Define two input layers
    input_shape = (historical_length + forecast_length, num_historical_features)

    reference_input = Input(input_shape)
    candidate_input = Input(input_shape)

    # Preprocess the shape of the inputs like in the difference model
    # https://keras.io/api/layers/reshaping_layers/cropping1d/
    reference_historical = tf.keras.layers.Cropping1D(cropping=(0, forecast_length))(reference_input)
    reference_historical = tf.keras.layers.Reshape((historical_length, num_historical_features),
                                                   input_shape=(historical_length, num_historical_features))(
        reference_historical)
    reference_forecast = tf.keras.layers.Cropping1D(cropping=(historical_length, 0))(reference_input)
    reference_forecast = tf.keras.layers.Reshape((forecast_length, num_historical_features),
                                                 input_shape=(forecast_length, num_historical_features))(
        reference_forecast)
    reference_forecast = reference_forecast[:, :,
                         TOTAL_FLOW_FEATURE_INDEX]  # reduce to just 1 dimension (batch, size, feature)

    candidate_historical = tf.keras.layers.Cropping1D(cropping=(0, forecast_length))(candidate_input)
    candidate_historical = tf.keras.layers.Reshape((historical_length, num_historical_features),
                                                   input_shape=(historical_length, num_historical_features))(
        candidate_historical)

    # Create out CNN model to create feature maps
    # I used average pooling instead of max pooling to not generalize too much because we are going to further generalize by taking the difference of the feature maps
    # The number of filters are inversely proportional to the kernel_size. I thought, intuitively, I want to focus on more features of smaller sections of the historical
    # data, rather than have many features of large portions. If we had many feature for large portions, I would think that the model would overfit to the training windows
    # But you can explore different hyperparameters of this model, especially when you make it more multivariate
    cnn_model = Sequential([
        Conv1D(filters=(int)(64 / 8), kernel_size=(int)(HISTORICAL_LENGTH / 2), strides=1, padding='causal',
               activation='relu'),
        AveragePooling1D(pool_size=4, strides=2),
        Conv1D(filters=(int)(64 / 4), kernel_size=(int)(HISTORICAL_LENGTH / 4), strides=1, padding='causal',
               activation='relu'),
        Conv1D(filters=(int)(64 / 2), kernel_size=(int)(HISTORICAL_LENGTH / 8), strides=1, padding='causal',
               activation='relu'),
        AveragePooling1D(pool_size=3, strides=1),
        Conv1D(filters=(int)(64 / 1), kernel_size=(int)(HISTORICAL_LENGTH / 16), strides=1, padding='causal',
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
        Bidirectional(LSTM(32, return_sequences=True)),
        Bidirectional(LSTM(16, return_sequences=True)),
        Bidirectional(LSTM(8, return_sequences=True)),
        LSTM(forecast_length * num_forecast_features, return_sequences=False)
    ])

    # Get the difference_vector/difference_forecast
    difference_forecast = lstm_model(difference_historical)

    # Our predicted forecast = difference_vector + reference_forecast
    predict = Add()([difference_forecast, reference_forecast])

    # Define out models inputs and outputs
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

    # These will create your training/test .npz file
    # Find the filepath and call load_data(filepath of .npz) to load the train/test windows
    # After calling these functions, update your PREMADE_TRAINING_DATA and PREMADE_TESTING_DATA names to match
    # Now that you have your premade, comment out the creating of data and start your training!
    # (No need to redundantly create data that is already made into a file)
    '''
    # Create training data
    create_window_data_file(type=TRAINING_TYPE,
                directory=TRAINING_DIRECTORY,
                year=TRAINING_YEAR,
                district=TRAINING_DISTRICT,
                stations=NUM_TRAINING_STATIONS,
                days=NUM_TRAINING_DAYS,
                start_day=TRAINING_START_DAY,
                offset_day=TRAINING_OFFSET_DAY)
                
    # Create testing data
    create_window_data_file(type=TESTING_TYPE,
                directory=TESTING_DIRECTORY,
                year=TESTING_YEAR,
                district=TESTING_DISTRICT,
                stations=NUM_TESTING_STATIONS,
                days=NUM_TESTING_DAYS,
                start_day=TESTING_START_DAY,
                offset_day=TESTING_OFFSET_DAY)
    '''


    # Start training after creating data
    print("Loading train")
    historical_top_train, historical_bot_train, forecast_train = load_data(PREMADE_TRAINING_DATA)

    # Check for no NaN values in our training windows
    assert(np.count_nonzero(np.isnan(historical_top_train)) == 0)
    assert(np.count_nonzero(np.isnan(historical_bot_train)) == 0)
    assert(np.count_nonzero(np.isnan(forecast_train)) == 0)

    # We are only taking very fourth window (effectively reducing the size of our training dataset by 4)
    # This is done due to my hardware limitations (as I could not fit all windows in GPU memory), and time constraint when training
    # You may reduce the size even more, but be wary that this may lead to overfitting the model based on your hyperparameters!
    # I would stick to a manageble size training dataset like this one, and adjust your hyperparameters accordingly.
    # I would NOT adjust the size of the training dataset to adhere to your hyperparameters.
    historical_top_train = historical_top_train[:historical_top_train.shape[0]//4]
    historical_bot_train = historical_bot_train[:historical_bot_train.shape[0]//4]
    forecast_train = forecast_train[:forecast_train.shape[0]//4]

    print("Loading test")
    historical_top_test, historical_bot_test, forecast_test = load_data(PREMADE_TESTING_DATA)
    # Check for no NaN values in our testing windows
    assert(np.count_nonzero(np.isnan(historical_top_test)) == 0)
    assert(np.count_nonzero(np.isnan(historical_bot_test)) == 0)
    assert(np.count_nonzero(np.isnan(forecast_test)) == 0)
    # The same culling of test set size is done due to hardware limitations
    historical_top_test = historical_top_test[:historical_top_test.shape[0]//4]
    historical_bot_test = historical_bot_test[:historical_bot_test.shape[0]//4]
    forecast_test = forecast_test[:forecast_test.shape[0]//4]

    # Train your models here
    #model = create_tinker_model_naive_test()
    model = create_difference_model()
    #model = create_cnn_lstm_encoder_decoder_model()

    # Learning_hyper_parameters
    # We are using the Adam optimizer, for it is most common for learning
    # The initial learning rate is 0.0001 * np.sqrt(BATCH_SIZE) (but this could be tweaked by a magnitude like 0.001 for faster learning (but it yield a worse model))
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
    epoch_model_checkpoint_dir_base = r'F:\fewshot_time_series_forecasting\dataset_v1'
    epoch_best_model_checkpoint_filename = r'model_best_lstm_sequence_sequence{epoch:08d}.h5'
    epoch_model_best_checkpoint_filepath = os.path.join(epoch_model_checkpoint_dir_base,
                                                        epoch_best_model_checkpoint_filename)


    #https://stackoverflow.com/questions/54323960/save-keras-model-at-specific-epochs
    epoch_latest_model_checkpoint_filename = r'model_latest_lstm_sequence_sequence.h5'
    epoch_model_latest_checkpoint_filepath = os.path.join(epoch_model_checkpoint_dir_base,
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

    # Train our model
    # Note that when training, the program will print the progress of our model after every epoch. I used that series of printed information
    # to plot a learning curve of the MAE. But a more efficient/automatic way to do this is to use TensorBoard library. This will automatically
    # collect your training data and plot it.
    # Also, you can extract the models weights by using model.get_weights(), then plot the weighs in a violin plot to see which layers are
    # saturated or not. https://www.kaggle.com/code/vishnurapps/beginners-guide-to-use-keras/notebook
    model.fit(x=(historical_top_train, historical_bot_train), y=forecast_train,
              batch_size=BATCH_SIZE,
              epochs=2500,
              use_multiprocessing=True,
              verbose=1,
              validation_data=((historical_top_test, historical_bot_test), forecast_test),
              callbacks=[best_model_checkpoint, latest_model_checkpoint],
              shuffle=True
              )


    print("Done")

