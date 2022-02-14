# Caltrans dataset headers
# day, timestamp, station, district, freeway, direction, lane type, station length, samples, %observed, total flow, avg occupancy, avg speed, lane n samples, lane n flow, lane n avg, occ, lane n avg speed, lane n observed

import gzip
import shutil

import numpy as np

import os

import pandas as pd


GZ_DIR = r'C:\Users\Victor\Desktop\CSUF\Spring 2022\CPSC 599\caltrans_dataset\2018\gz_dir'
TXT_DIR = r'C:\Users\Victor\Desktop\CSUF\Spring 2022\CPSC 599\caltrans_dataset\2018\txt_dir'
CSV_DIR = r'C:\Users\Victor\Desktop\CSUF\Spring 2022\CPSC 599\caltrans_dataset\2018\csv_dir'


TXT_EXTENSION = '.txt'
CSV_EXTENSION = '.csv'



def extract_PeMS_gz_dataset(source_directory, destination_directory):
    ''' Extracts the PeMS gz dataset located in the source directory into the destination directory '''
    for filename in os.listdir(source_directory):
        if (filename.endswith('.gz')):
            # remove .gz to get filename base
            uncompressed_filename = os.path.splitext(filename)[0]
            gz_file = os.path.join(source_directory, filename)
            with gzip.open(gz_file, 'rb') as f_in:
                convert_PeMS_txt_to_csv(f_in, destination_directory)
                break



def convert_PeMS_txt_to_csv(file, destination_directory):
    ''' Get only the 'timestamp', 'station', 'total_flow' to  reduce csv size '''
    df = pd.read_csv(file,
                     skiprows=0,
                     sep=',',
                     names=['timestamp', 'station', 'total_flow'],
                     usecols=[0, 1, 9],
                     engine='python')
    filebasename = os.path.splitext(file.filename)[0]
    df.to_csv(os.path.join(destination_directory, filebasename + CSV_EXTENSION))
    print(df['total_flow'].isnull().sum())












# Remove any station with unknown data (and mark them for future removal reference in case they do have data in this new case)
#   Can do by iterating through all the reduced csv size (entire year) and built a list of all stations which had a NaN value (check for nan)

# assert df['timestamp'].isnull().any() == False
# assert df['station'].isnull().any() == False

# print(len(df['total_flow']))
# print(df['total_flow'].isnull().sum())
# break




# day and timestamp, station, district, freeway, direction, lane type, station length, samples, %observed, total flow, avg occupancy, avg speed, lane n samples, lane n flow, lane n avg, occ, lane n avg speed, lane n observed


if __name__ == '__main__':
    extract_PeMS_gz_dataset(GZ_DIR, TXT_DIR)
    #convert_PeMS_txt_dataset_to_csv(TXT_DIR, CSV_DIR)




# actually need to convert the file to csv with COMMA and SPACE delimiters!

# Let's do a station count first to see if the entire year holds all the stations. We ignore newly added stations





# 1) Efficiently extract files to .csv
# 2) Clean .csv files to include only relevant information such as station number, date/time, car count to lower file size
# 3) transpose rows and columns
# 4) Figure out a way to append on the time (do i need an appending algorithm using date/time as key to rows? Or can i take advantage of the files already separated by days?)
#       Possible pitfalls is checking for newly added stations that were not there?