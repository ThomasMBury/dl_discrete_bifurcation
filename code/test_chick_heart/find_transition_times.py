#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 12:47:52 2022

Find transition times in chick heart data based on slope of return map

@author: tbury
"""

import numpy as np
import pandas as pd

import ewstools
from sklearn.linear_model import LinearRegression

def get_return_map_slope(series):
    '''
    Compute slope of return map for pd.Series.
    Use linear regression to get slope
    
    '''
    # Fit linear regression
    x = series.values[1:].reshape(-1,1)
    y = series.shift(1).values[1:]
    
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    
    return slope

def find_repeats(L, required_number, num_repeats):
    idx = 0
    while idx < len(L):
        if [required_number]*num_repeats == L[idx:idx+num_repeats]:
            idx_repeat = idx
            return idx_repeat
        else:
            idx += 1
    return np.nan

# Parameters
rw = 10 # Rolling window size for return map slope computation
m_thresh = -0.95 # Threhsold in slope to define period-doubling bifurcation location
bw=60 # Bandwidth (in beat number) to detrend data
num_consec = 10 # Number of consecutive beats required below thresh to define onset of bifurcation

# Import PD data
df_traj = pd.read_csv('../../data/df_chick.csv')
df_pd = df_traj[df_traj['type']=='pd']

# Loop through each tsid
tsid_vals = df_pd['tsid'].unique()
list_tup = []

for tsid in tsid_vals:
    
    df = df_pd[df_pd['tsid']==tsid].copy().set_index('Beat number')
    
    # Detrend using Gaussian kernel
    s = df['IBI (s)']
    ts = ewstools.TimeSeries(s)
    ts.detrend(method='Gaussian', bandwidth=bw)
    ts.state.index.name = 'Beat #'
    
    # Compute return map slope over rolling window
    df['slope'] = ts.state['residuals'].rolling(window=rw).apply(
        lambda x: get_return_map_slope(x),
        )
    
    # Get first time where slope<thresh for num_consec consecutive beats
    list_bool = list((df['slope']<m_thresh).astype(int).values)
    idx = find_repeats(list_bool, 1, num_repeats=num_consec)
    if np.isnan(idx):
        beatnum_bif = np.nan
    else:
        # Subtract off length of rolling window
        # Bifurcation starts at the beginning of the rolling window
        beatnum_bif = df.index[idx - rw]
        
    print('Transition for tsid = {} at beat {}'.format(tsid, beatnum_bif))
    list_tup.append((tsid, beatnum_bif))
    

df_tbif = pd.DataFrame(
    {'tsid': [tup[0] for tup in list_tup],
     'transition': [tup[1] for tup in list_tup],
      })

df_tbif.to_csv('output/df_transitions.csv', index=False)





