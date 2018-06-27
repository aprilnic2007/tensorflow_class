
# coding: utf-8

# ## April McKay - Final Project - Tensorflow June 2018

# Input Data:
# > Data for this project is based on pedestrian count data gathered from 5 Motionloft sensors in New York.
# > 
# > Weather data is automatically pulled from the Darksy API and available along with our count data in the API. 
# > 
# > Weather includes hourly temperature (F), weather conditions, and amount of precipitation (inches).
# 
# 
# Goals:
# 1. Use historical pedestrian count data to train a model to learn the intracasies of this count data:
#     > Seasonality
#     > 
#     > Weekly Patterns
#     > 
#     > Daily Patterns
#     
# 2. Strech goal - Use the model to predict counts into the future based on forecast weather data (Darksky API)

# ![vimo.jpg](attachment:vimo.jpg)

# In[1]:


import pandas as pd
import numpy as np
from io import StringIO
import sys
import datetime
import getpass
import requests
import json
from requests.auth import HTTPBasicAuth
import pytz
import tensorflow as tf
import shutil
from tensorflow.contrib.learn import ModeKeys
import tensorflow.contrib.rnn as rnn
import time
from datetime import timedelta
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.python.keras.initializers import RandomUniform


# In[2]:


sess = tf.Session()


# ### Input Functions

# retrieve_csv (Input function)
# 
# > This function pulls in the required auth info and returns the CSV from the Motionlof API

# In[3]:


def retrieve_csv(resource_uid, begin_date, end_date):
    #print("in retrieve")
    request_params = dict(resource_uid=resource_uid,
            begin=begin_date,
            end=end_date,
            output_format='csv',
            language='en',
            direction='both')
    api_base_url = "https://dashboard.motionloft.com/api/v1/"
    csv_base_url = api_base_url + "location"
    userinfo_url = api_base_url + "userinfo"
    config = json.load(open('config.json'))
    api_auth_object = HTTPBasicAuth(username=config['user_id'], password=config['token'])
    csv_ = requests.get(csv_base_url, params=request_params, auth=api_auth_object)
    return csv_.text


# <hr>
# weather_forecast_data_pull (Input function)
# 
# > The weather_forecast_data_pull function uses the Motionloft API to pull data from 5 locations in New York City.
# >
# > The loop passes thru all of the locations in my CSV to create summary_df.
# >
# > The data is at the hourly level per location.
# >
# > The output of the function is a CSV that averages the data from the 5 locations by day and hour.
# >
# > Weather data is from the Darksky API.

# In[4]:


# coding=utf-8

def weather_forecast_data_pull(start_date,end_date):
    am_user_name = getpass.getuser()
    api_base_url = "https://dashboard.motionloft.com/api/v1/"
    csv_base_url = api_base_url + "location"
    #print(csv_base_url)
    userinfo_url = api_base_url + "userinfo"
    config = json.load(open('config.json'))
    api_auth_object = HTTPBasicAuth(username=config['user_id'], password=config['token'])
    data_list = pd.read_csv('NYC_Locations.csv')
    length_of_locations = data_list['nyc_locations'].count()
    summary_df = pd.DataFrame()

    for x in range(0,length_of_locations):
        ext_uid = data_list.iloc[x]
        df = retrieve_csv(ext_uid,start_date,end_date)
        df_buffer = StringIO(df)
        ext_df = pd.read_csv(df_buffer)
        ext_df=ext_df.rename(columns = {'ï»¿hour_beginning':'hour_beginning'})
        ext_df['hour_beginning'] = pd.to_datetime(ext_df['hour_beginning'],format = '%Y-%m-%d %H:%M:%S')
        ext_df['hournum']= pd.DatetimeIndex(ext_df['hour_beginning']).hour
        ext_df['Day'] = pd.DatetimeIndex(ext_df['hour_beginning']).dayofweek
        ext_df['Day_of_Year'] = pd.DatetimeIndex(ext_df['hour_beginning']).dayofyear
        ext_df['Pedestrians'].fillna(0,inplace=True)
        ext_df['temperature'].fillna(0,inplace=True)
        ext_df['precipitation'].fillna(0,inplace=True)
        ext_df['events'].fillna('no_event',inplace=True)
        ext_df['event_code'] = np.where(ext_df['events'] == 'no_event', 0, 1)
        ext_df_for_report = ext_df[['hour_beginning','Day','Day_of_Year','location','Pedestrians', 'temperature','precipitation','event_code','hournum']]
        ext_df_for_report = ext_df_for_report.rename(columns = {'event_code':'events'})
 
        if x == 0:
            summary_df = pd.DataFrame(ext_df_for_report)
        else:
            summary_df = summary_df.append(ext_df_for_report)
        #print(x)

        x = x+1

    hourly_average = ext_df_for_report.groupby(by=['hour_beginning','Day','Day_of_Year','hournum'])[['Pedestrians','temperature','precipitation','events']].mean()
    hourly_average['location'] = 'New York City'
    hourly_average = hourly_average.reset_index()
    hourly_average = hourly_average[['hour_beginning','Day','Day_of_Year','location','Pedestrians','temperature','precipitation','events','hournum']]
    return hourly_average


# <hr>
# inputs_counts - Runs the input functions
# 
# > The inputs_counts function runs the weather_forecast_data_pull function for the specified date range and
# day of the week. 
# > 
# > The percent of the data to use for training is also an input for the function.
# > 
# > The day_num is also a perameter, where Monday = 0, Tuesday - 1. If the input is <> [0-6], all the days will be used. 
# (Date format: '2016-11-01')

# In[5]:


def inputs_counts(day_num,start_date,end_date):
    ped_df = weather_forecast_data_pull(start_date,end_date)
    ped_df = ped_df[["Day",'Day_of_Year',"hournum","Pedestrians","temperature","precipitation"]]

    # segment out a day of the week
    if day_num in [0,1,2,3,4,5,6]:
            ped_df = ped_df[ped_df['Day']==day_num]
            ped_df = ped_df.drop('Day', axis=1)
            ped_df = ped_df.drop('temperature', axis=1)
            shift_days = -1
    else:
            shift_days = -7 #(-168)
    
    print(ped_df.head(30))
    return ped_df,shift_days


# <hr>
# Call the inputs_counts function, split and train
# 
# > Here's the inspiration for the model I built for this: https://www.youtube.com/watch?v=6f67zrH-_IE. 
# It is a  Recurrent Neural Network (GRU / LSTM) tutorial for time series weather data that tries to predict results
# > 
# > I want to use the data to attempt to predict the next week's pedestrian traffic
# so I'm shifting the data 1 week (168 hours) and I'm removing the NAs 
# from the shifted data and cutting off the rows that were shifted away in the other data.
# > 
# > I'm also training on 80% of the data I have
# > 
# > Because the data values can range pretty widely, I'm using the MinMaxScaler() function from sklearn to make all
# the values a number between -1 and 1.

# In[6]:


# pulls the input count and weather data
#inputs_result = inputs_counts(0,'2016-11-01','2018-06-16')
inputs_result = inputs_counts(0,'2017-11-01','2018-06-25')
ped_df = inputs_result[0]

train_split = .8
target_names = ['Pedestrians']
#shift_days = -7 #(-168) This is the original value
shift_steps = inputs_result[1]*24
df_targets = ped_df.Pedestrians.shift(shift_steps)

train_length = int((ped_df.shape[0])*train_split)
x_data_not_vals = ped_df[:shift_steps]
y_data_not_vals = df_targets.values[:shift_steps]
print(ped_df.head(30))

x_data = ped_df.values[:shift_steps]
y_data = df_targets.values[:shift_steps]

x_train = x_data[0:train_length]
x_test = x_data[train_length:]

y_train = y_data[0:train_length].reshape(-1, 1)
y_test =  y_data[train_length:].reshape(-1, 1)

x_scaler = MinMaxScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.fit_transform(x_test)

#y_scaler = MinMaxScaler()
y_train_scaled = x_scaler.fit_transform(y_train)
y_test_scaled = x_scaler.fit_transform(y_test)

print("X shape:",x_train_scaled.shape)
print("Y shape:",y_train_scaled.shape)


# Batch Generator Helper Function
# 
# > This is a function that I got from the Hvass Labs tuturial.
# > 
# > It breaks my data into batches for testing / processing based on how large I want the batches to be,
# the number of input signals and the number of output signals (1)
# 

# In[7]:


def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/23_Time-Series-Prediction.ipynb
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]
        
        yield (x_batch, y_batch)


# This line actually runs the generator script above and produces the batches

# In[8]:


batch_size = 256 #100
sequence_length = 24*7*2 #24 hours a day, 7 days, 2 weeks, original value = 24*7*2
num_x_signals = x_train_scaled.shape[1] #4
num_y_signals = 1
num_train = 500
generator = batch_generator(batch_size = batch_size, sequence_length = sequence_length)

x_batch,y_batch = next(generator)

print(x_batch.shape)
print(y_batch.shape)


# Visualize Input Batch
# 
# > To see what a batch of data looks like, I'm plotting the data from the first 2 week batch

# In[9]:


batch = 1

for q in range(num_x_signals):
    print(list(ped_df)[q])
    seq_0 = x_batch[batch,:,q]
    plt.plot(seq_0)
    plt.show()


# <hr>
# This section defines the validation data for the model, based on the scaled x and y data

# In[10]:


validation_data = (np.expand_dims(x_test_scaled, axis=0),np.expand_dims(y_test_scaled, axis=0))
print("X validation data\n",validation_data[0],"\n")
print("Y validation data - Peds only\n", validation_data[1])


# <hr>
# ### Construct the model
# 
# > The Sequential model lets me add layers one by one.
# > 
# > Because the middle layer has tons of elements, the Dense sigmoid layer rolls 
# everything up to the number of signals I want a result for, which is 1

# In[11]:


model = Sequential()
model.add(GRU(units=512,
              return_sequences=True,
              input_shape=(None, num_x_signals,)))
model.add(Dense(num_y_signals, activation='sigmoid'))

if False:
    from tensorflow.python.keras.initializers import RandomUniform
    # Maybe use lower init-ranges.
    init = RandomUniform(minval=-0.05, maxval=0.05)

    model.add(Dense(num_y_signals,
                    activation='linear',
                    kernel_initializer=init))


# <hr>
# Adding warmup steps
# 
# > Because the model is naturally going to be inaccurate early on in the training, I'm ignoring the results from the first day to give the model a fair chance at accuracy. This was initially set to 20 in the example code

# In[12]:


warmup_steps = 24


# <hr>
# ### Loss Section
# 
# > The loss_mse_warmup section was taken from the tutorial as well.
# > 
# > It uses tensorflow to calculate the Mean Squared Error to help me see how well the model
# predicted the actual values

# In[13]:


def loss_mse_warmup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.
    
    y_true is the desired output.
    y_pred is the model's output.
    """

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    loss = tf.losses.mean_squared_error(labels=y_true_slice,
                                        predictions=y_pred_slice)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire tensor, we reduce it to a
    # single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean


# <hr>
# This line sets the learning rate. The tutorial used 1e-3 or .001. 

# In[14]:


optimizer = RMSprop(lr=1e-3)


# <hr>
# This line puts all of the different parts of the model together and compiles it all

# In[15]:


# loss_mse_warmup and optimizer were provided by the tutorial
#model.compile(loss=loss_mse_warmup , optimizer=optimizer, metrics=['accuracy'])

# this variety was a suggestion
model.compile(loss='mae', optimizer='adam',metrics=['accuracy'])


# <hr>
# This line prints out a summary of what the model looks like and the dense layer has a shape of 1 for my Pedestrians

# In[16]:


model.summary()


# <hr>
# Defining checkpoints (suggested in the tutorial)
#  
# > The most interesting one is the Reduce Learning Rate on Plateau callback, which adjusts the learning rate as training goes on

# In[17]:


path_checkpoint = 'ANM_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)

callback_tensorboard = TensorBoard(log_dir='./rnn_final_logs/',
                                   histogram_freq=0,
                                   write_graph=True)

callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)

callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]


# <hr>
# 
# ### Training Section

# In[18]:


get_ipython().run_cell_magic('time', '', 'model.fit_generator(generator=generator,\n                    epochs=2,\n                    steps_per_epoch=25,\n                    validation_data=validation_data,\n                    callbacks=callbacks)')


# <hr>
# ### Evaluation Section

# In[19]:


result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))

print("loss (test-set):", result)


# <hr>
# ### Visualization Section
# 
# plot_comparison was also a helper function from the tutorial that pulls the signals from the start time I want and the length I want, reverses the MinMaxScalers to get me back to ped counts, and visualizes the results - actual counts in blue, predicted counts in orange

# In[20]:


def plot_comparison(start_idx, length=100, train=False):
    """
    Plot the predicted and true output-signals.
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """
    
    if train:
        # Use training-data.
        x = x_train_scaled
        y_true = y_train
    else:
        # Use test-data.
        x = x_test_scaled
        y_true = y_test
    
    # End-index for the sequences.
    end_idx = start_idx + length
    
    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]
    
    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)
    
    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = x_scaler.inverse_transform(y_pred[0])
    
    # For each output-signal.
    for signal in range(len(target_names)):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred_rescaled[:, signal]
        
        # Get the true output-signal from the data-set.
        signal_true = y_true[:, signal]

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(15,5))
        
        # Plot and compare the two signals.
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')
        
        # Plot grey box for warmup-period.
        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
        
        # Plot labels etc.
        plt.ylabel(target_names[signal])
        plt.legend()
        plt.show()


# In[21]:


plot_comparison(start_idx=1, length=2000, train=True)


# In[22]:


sess.close()


# <hr>
# ### Description of results
# 
# At the moment, Saturday counts are being predicted the best with the current model and the various changes I made to the input signal list. I suspect this is due to the typical pattern, as the curve is smoother and doesn't have spikes like weekdays do. Traffic on Saturdays is reletively consistent throughtout the year and doesn't suffer from random Holidays that impact weekdays quite a bit (think Memorial Day vs a typical Monday)
# 
# Removing columns from the input signals doesn't have much of an impact
# 
# In general, valleys are identified easily by the model, but peaks are not - especially weekday peaks during the day.
# 
# Model 4 had the best result: (Model 4 (Monday's only): Batch Size = 256, Epochs = 2, Steps = 25, 14min 1s). I removed Day from the input signals. I used 'mae' for loss and 'adam' as the optimizer. I also restricted the amount of input data from 11/1/2017 till 6/15/2018. I believe historical data was confusing the model.
# 
# loss: 0.0587 - acc: 0.0024 - val_loss: 0.1067 - val_acc: 0.0143
# loss (test-set): [0.10669038444757462, 0.014285714365541935]

# <hr>
# ### Documenting tweeks
# 
# Model 1 (Monday's only): Batch Size = 256, Epochs = 2, Steps = 50, 26min 57s, loss: 0.0161 - val_loss: 0.0282. 
#  > loss (test-set): 0.028185173869132996
#  > This model wasn't able to pick up peaks in the data. I also need to update the shift function. If the data only consists
#  > Of Mondays, It doesn't make sense to shift 7 Mondays
# 
# Model 2 (All days of the week): Batch Size = 256, Epochs = 2, Steps = 25, 14min 29s, loss: 0.6474 - acc: 0.0000e+00 - val_loss: 0.7432 - val_acc: 2.4378e-04
#  > loss (test-set): [0.7432200312614441, 0.00024378352100029588]
#  > Shifting a week
#  > This model was VERY inaccurate. Epoch 1 started with .6474 loss and there wasn't much improvement
#  > Including all the days of the week seems to be confusing the model
# 
# Model 3 (Monday's only): Batch Size = 256, Epochs = 2, Steps = 25, 15min 35s, loss: 0.0137 - acc: 5.3153e-04 - val_loss: 0.0214 - val_acc: 0.0053
#  > loss (test-set): [0.02144785411655903, 0.005263158120214939]
#  > I updated the shift days if statement to automatically choose shift 1 day (which is truly shifting a week 
#  because there's only 1 type of day in the dataset)
#  > I'm increasing the training split to .8
#  > I put the learning rate back to .0001
#  > Added RandomUniform back in
# 
#  Model 4 (Monday's only): Batch Size = 256, Epochs = 2, Steps = 25, 13min 57s, loss: 0.0136 - acc: 5.4501e-04 - val_loss: 0.0127 - val_acc: 0.0053
#  > loss (test-set): [0.012714440934360027, 0.005263158120214939]
#  > I removed Day from the input signals
# 
#  Model 5 (Monday's only): Batch Size = 256, Epochs = 2, Steps = 10, 6min 4s, loss: 0.0363 - acc: 5.3827e-04 - val_loss: 0.0590 - val_acc: 0.0026
#  > loss (test-set): [0.012714440934360027, 0.005263158120214939]
#  > I removed Day and Hour from the input signals
#  > Accuracy isn't really improving
# 
#  Model 6 (Monday's only): Batch Size = 256, Epochs = 5, Steps = 10, 15min 27s, loss: 0.0083 - acc: 5.7082e-04 - val_loss: 0.0149 - val_acc: 0.0053
#  > loss (test-set): [0.01494902465492487, 0.005263158120214939]
#  > I added Day and Hour back but I increased the number of epics
# 
# Model 7 (Monday's only): Batch Size = 100, Epochs = 5, Steps = 10, 8min 12s, loss: 0.0100 - acc: 4.9405e-04 - val_loss: 0.0157 - val_acc: 0.0053
# > loss (test-set): [0.01574281044304371, 0.005263158120214939]
# > I changed the batch size. This ran much faster!
# 
# Model 8 (Monday's only): Batch Size = 100, Epochs = 5, Steps = 10, 9min 37s, loss: 0.0126 - acc: 4.7619e-04 - val_loss: 0.0193 - val_acc: 0.0053
# > loss (test-set): [0.019304059445858, 0.005263158120214939]
# > I added the day of the year metric to try to train for seasonality
# 
# Model 9 (Monday's only): Batch Size = 100, Epochs = 5, Steps = 10, 7min 32s, loss: 0.0096 - acc: 5.0000e-04 - val_loss: 0.0127 - val_acc: 0.0053
# > loss (test-set): [0.012737380340695381, 0.005263158120214939]
# > Removed precip and day from input signals
# 
# Model 10 (Saturday's only): Batch Size = 100, Epochs = 5, Steps = 10, 7min 26s, loss: 0.0164 - acc: 6.2500e-05 - val_loss: 0.0194 - val_acc: 0.0052
# > loss (test-set): [0.01944039575755596, 0.0052083334885537624]
# > switched to Saturdays
# 
# Model 10 (Saturday's only): Batch Size = 100, Epochs = 5, Steps = 10, 7min 9s, loss: 0.0141 - acc: 9.5238e-05 - val_loss: 0.0136 - val_acc: 0.0052
# > loss (test-set): [0.013565653935074806, 0.0052083334885537624]
# > Removed Temp
# 
# Model 11 (Saturday's only): Batch Size = 100, Epochs = 5, Steps = 10, 7min 29s, loss: 0.0700 - acc: 1.0119e-04 - val_loss: 0.0751 - val_acc: 0.0052
# > loss (test-set): [0.0750596895813942, 0.0052083334885537624]
# > Switch to (loss='mae', optimizer='adam',metrics=['accuracy']). No column drops
