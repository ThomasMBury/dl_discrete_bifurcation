#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 15:48:34 2021

Script to train DL classifier

Key for trajectory type:
    0 : Null trajectory
    1 : Period-doubling trajectory
    2 : Neimark-Sacker trajectory
    3 : Fold trajectory
    4 : Transcritical trajectory
    5 : Pitchfork trajectory

Options for model_type
    1: random length U[50,ts_len] & random start time U[0,ts_len-L]
    2: random length U[50,ts_len] & start time = ts_len-L (use end of time series)

@author: tbury
"""

import time
start = time.time()

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description='Train DL classifier')
parser.add_argument('--model_type', type=int, help='Model type', choices=[1,2],
                    default=2)
parser.add_argument('--num_epochs', type=int, help='Number of training epochs', default=5)

args = parser.parse_args()
model_type = args.model_type
num_epochs = args.num_epochs

print('Training DL with mtype={}'.format(model_type))
seed = 1

import pandas as pd
import numpy as np
np.random.seed(seed)

from tensorflow.random import set_seed
set_seed(seed)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


# Load in training data
df = pd.read_parquet('../training_data/output/df_train.parquet')

# Make classes of equal size : currently 5 times number of samples in null section
# Get null class
tsid_null = df[df['type']==0]['tsid'].unique()
# Downsample - Take random selection 
tsid_null_down = np.random.choice(tsid_null, 
                                  size=int(len(tsid_null)/5),
                                  replace=False)
df_null = df[df['tsid'].isin(tsid_null_down)]
df_not_null = df[df['type']!=0]
df = pd.concat((df_null, df_not_null))

ts_len = 500

# Pad and normalise the data
print('Pad and normalise the data')
def prepare_series(series):
    '''
    Prepare raw series data for training.
    
    Parameters:
        series: pd.Series of length ts_len
    '''
    
    # Length of sequence to extract
    L = np.random.choice(np.arange(50,ts_len+1))
    
    # Start time of sequence to extract
    if model_type==1:
        t0 = np.random.choice(np.arange(0,ts_len-L+1))
    elif model_type==2:
        t0 = ts_len-L
        
    seq = series[t0:t0+L]
    
    # Normalise the sequence by mean of absolute values
    mean = seq.abs().mean()
    seq_norm = seq/mean
    
    # Prepend with zeros to make sequence of length ts_len
    series_out = pd.concat([
        pd.Series(np.zeros(ts_len-L)),
        seq_norm], ignore_index=True)
    
    # Keep original index
    series_out.index=series.index
    
    # print('Series with L={}, t0={} prepared'.format(L,t0))
    # print(series_out.isnull().any())
    return series_out

# Apply preprocessing to each series
ts_pad = df.groupby('tsid')['x'].transform(prepare_series)
df['x_pad'] = ts_pad

# Put into numpy array with shape (samples, timesteps, features)
inputs = df['x_pad'].to_numpy().reshape(-1, ts_len, 1)
# targets = df.groupby('tsid', sort=False)['type'].max().values.reshape(-1,1)
targets = df['type'].iloc[::ts_len].to_numpy().reshape(-1,1) 

# Shuffle data
indices_permutation = np.random.permutation(len(inputs))
inputs_shuffled = inputs[indices_permutation]
targets_shuffled = targets[indices_permutation]


# Split into training, validation and test
print('Split data into training, validation and test groups')
# split_ratio = (0.5,0.25,0.25) # (training, validation, test)
split_ratio = (0.95,0.025,0.025)
max_index_train = int(split_ratio[0]*len(inputs))
max_index_val = int((split_ratio[0]+split_ratio[1])*len(inputs))                      

inputs_train = inputs_shuffled[:max_index_train]
inputs_val = inputs_shuffled[max_index_train:max_index_val]
inputs_test = inputs_shuffled[max_index_val:]

targets_train = targets_shuffled[:max_index_train]
targets_val = targets_shuffled[max_index_train:max_index_val]
targets_test = targets_shuffled[max_index_val:]

print('Using {} training data samples'.format(len(inputs_train)))
print('Using {} validation data samples'.format(len(inputs_val)))
print('Using {} test data samples'.format(len(inputs_test)))

# Export test data
np.save('output/test_inputs_{}.npy'.format(model_type), inputs_test)
np.save('output/test_targets_{}.npy'.format(model_type), targets_test)

# Hyperparameters
pool_size = 2
learning_rate = 0.0005     
batch_size = 1024
dropout_percent = 0.10
filters = 50
mem_cells = 50
mem_cells2 = 10
kernel_size = 12
kernel_initializer = 'lecun_normal'


# Set up NN architecture
model = Sequential()

model.add(Conv1D(filters=filters, 
                 kernel_size=kernel_size, 
                 activation='relu', 
                 padding='same',
                 input_shape=(ts_len, 1),
                 kernel_initializer = kernel_initializer))

model.add(Dropout(dropout_percent))
model.add(MaxPooling1D(pool_size=pool_size))

model.add(LSTM(mem_cells, 
               return_sequences=True, 
               kernel_initializer=kernel_initializer))
model.add(Dropout(dropout_percent))

model.add(LSTM(mem_cells2,
               kernel_initializer=kernel_initializer))
model.add(Dropout(dropout_percent))

model.add(Dense(6, activation='softmax',kernel_initializer = kernel_initializer))

# Set up optimiser and checkpoints to save best model
adam = Adam(learning_rate=learning_rate)

# Classifier name for export
model_name = 'output/classifier_{}.pkl'.format(model_type)

chk = ModelCheckpoint(model_name, 
                      monitor='val_accuracy', 
                      save_best_only=True, 
                      mode='max', 
                      verbose=1)

# Compile Keras model
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=adam, 
              metrics=['accuracy'])

# Train model
print('Train NN with seed={} and model_type={}'.format(seed,model_type))
history = model.fit(inputs_train, targets_train, 
                    epochs=num_epochs, 
                    batch_size=batch_size, 
                    callbacks=[chk], 
                    validation_data=(inputs_val,targets_val))

# Export history data (metrics evaluated on training and validation sets at each epoch)
df_history = pd.DataFrame(history.history)
df_history.to_csv('output/df_history_{}.csv'.format(model_type), index=False)

end = time.time()
print('Script took {:0.1f} seconds'.format(end-start))



