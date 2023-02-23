#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 10:46:50 2022

-Make plot of single trajectory 

@author: tbury
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import ewstools

import tensorflow as tf
from tensorflow.keras.models import load_model


np.random.seed(0)

typ = 'pd'
tsid = 14

# Extract single trajectory
df = pd.read_csv('data/df_traj.csv')
df = df[
        (df['tsid']==tsid)&\
        (df['type']==typ)][['IBI (s)', 'Beat number']]

# fig = px.line(df, x='Time (s)', y='IBI (s)')
fig = px.line(df, x='Beat number', y='IBI (s)')

fig.write_html('temp.html')





