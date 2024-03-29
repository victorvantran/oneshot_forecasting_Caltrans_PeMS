-------------------------------------------------
Anaconda Environment Setup
-------------------------------------------------
I used a Anaconda Environment for TensorFlow by following this tutorial: https://www.youtube.com/watch?v=hHWkvEcDBO0&ab_channel=AladdinPersson

For this project, I installed Tensorflow v2.6.0 and Python 3.9.0

Auxilliary packages such as pandas and matplotlib may need to be installed as needed in the Anaconda environment


-------------------------------------------------
Recomended Tutorials
-------------------------------------------------
Before tackling with the code in this project, I recommend completing two
tutorials that gives a practical basis on:
(1) one-shot classification (which we want to extrapolate to few-shot)
	https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d
(2) time-series forecasting
	https://medium.com/geekculture/time-series-forecast-using-deep-learning-adef5753ec85

Here are other resources that may help you get started:
(1) 3Blue1Brown video series to visually understand what a neural network is
and how it learns
	https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&ab_channel=3Blue1Brown
(2) Introduction to neural networks and techniques to train a model via tuning
hyperparameters. There is much math involved, but do not get too bogged down.
The author gives pragmatic and illustrative explainations. 
	http://neuralnetworksanddeeplearning.com/index.html
(3) Andrew Ng's Deep Learning Course
	(a) One-shot classification with Face Recognition
		https://www.youtube.com/watch?v=K_3STku0BAs&list=PLkRLdi-c79HKEWoi4oryj-Cx-e47y_NcM&index=13&ab_channel=MakingTARS
	(b) Reccurent Neural Networks and different architectures for
different applciations
		https://www.youtube.com/watch?v=IV8--Y3evjw&list=PLkRLdi-c79HKEWoi4oryj-Cx-e47y_NcM&index=15&ab_channel=MakingTARS
	(c) You are welcome to watch his other tutorials for more information
as needed

	

--------------------------------------------------------------------------------
Few-shot Time-series Forecasting of CalTrans Traffic Flow PyCharm Project Setup
--------------------------------------------------------------------------------
Now that you have completed those two tutorials before proceeding on to this particular project. 
We will be using PyCharm for our environment setup, as followed by the tutorial video mentioned above.

(1) Create a new PyCharm project
(2) Place the /dataset, /preprocessed_data, main.py, and PeMS.py directories and files into said PyCharm project.



-------------------------------------------------
preprocessed_data and dataset Directories
-------------------------------------------------
Note that the directories /preprocessed_data and /dataset are empty
I have included example directories that were already created from the PeMS.py
and main.py code. These actual directories are made public on a dropbox due to its large size.
Download them and replace the empty directories:
https://www.dropbox.com/s/t2fnuo6qmanggc6/dataset.zip?dl=0
https://www.dropbox.com/s/c7u9vbt5u0b9y6f/preprocessed_data.zip?dl=0

There are two directories:
(1) preprocessed_data
(2) dataset

Unzip those two folders and replace them into your PyCharm project

-------------------------------------------------
preprocessed_data
-------------------------------------------------
The purpose of this directory is to store the .gz files downloaded from CalTrans
PeMS dataset and transform those files into .pickle files that can be used
later for training. Take note that each individual .gz files holds data for
EVERY station of that district for a certain day. That is not the ideal format
of our data. Instead, our objective in preprocessing is to have each
individual .pickle file to hold a years-worth of data for ONE station.

Look into the preprocessed_data folder and notice that all directories are
empty except for the gz folder. This folder holds data traffic flow data of
all stations in District 3 for the year 2019. You would find this data under
the PeMS Clearninghouse page under "Available Files":
	https://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id=3&submit=Submit 

The proprocessing code is written in PeMS.py. It will take steps
to transform the .gz files to .csv, then .np, then a master .pickle, then
individual .pickle files. All these files will be stored in their respective
directories. I left all other folders besides the gz folder empty for you to
use as an opportunity to run and preprocess the data yourself. If all is
successful, your pickle directory should be filled with station.pickle files.
Then you could repeat the process for other stations in different districts
for different years. If all is successful, your pickle directory should look
like the directory of station_data/2019/district_3.



----------------------------------------------------------------------
dataset
----------------------------------------------------------------------
After acquiring a directory full of station .pickle files, you would run the
code in main.py to take those preprocessed files and transform them into
respective training and dataset files. Here, you can tune the parameters of
your dataset such as: NUM_HISTORICAL_DAYS, NUM_FORECAST_DAYS,
NUM_STRIDE_DAYS, TRAINING DATA PARAMETERS, TESTING DATA PARAMETERS,
PREMADE_TRAINING_DATA, and PREMADE_TESTING_DATA. If all is successful, your
dataset should look like:
model_data/test_year2019_district3_stations400_days56_period30min_history144s_forecast24s.npz

I have already provided you with the dataset I used for my project. Notice in
the main.py code, specifically the __main__ function, I commented out the call
to transform the .pickle files to a .npz dataset file. I recommend you
uncomment and run the code to see if your .pickle files list that you created
can successfully generate a dataset file. This process may take several
minutes depending on your dataset parameters.

The dataset directory is the environment to store your station (.pickle) files and your
dataset (.npz) files.

Congratulations! You are able to create a training and/or testing dataset to
be used in your model for pairwise training given just a list of .gz files. 
