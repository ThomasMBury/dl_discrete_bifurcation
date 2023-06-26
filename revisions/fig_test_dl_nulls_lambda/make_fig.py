#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:23:35 2022

@author: tbury
"""

import time
start_time = time.time()

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import ewstools

import tensorflow as tf
from tensorflow.keras.models import load_model

# Load in EWS
sigma = 0.01
df_dl_full = pd.read_csv('output/df_sigma_{}.csv'.format(sigma))

# Make heatmap fig
# fig = make_subplots(1, 1, horizontal_spacing=0.15, vertical_spacing=0.08)

fig = go.Figure()

df_plot = df_dl_full.groupby(['rho','length'])[['0']].mean().reset_index().pivot(index='rho', columns='length')['0']
xvals = ['{:d}'.format(x) for x in df_plot.columns]
yvals = ['{:.1f}'.format(x) for x in df_plot.index]

row=1
fig.add_trace(go.Heatmap(z=df_plot.values, x=xvals, y=yvals, 
                           zmin=0, zmax=1, 
                            coloraxis='coloraxis',
                          ))
              # row=row,col=1)
# fig.add_trace(go.Heatmap(z=df_plot.values, x=xvals, y=yvals, 
#                            zmin=0, zmax=1, 
#                           # coloraxis='coloraxis',
#                           ),
#               row=row,col=2)
# fig.add_trace(go.Heatmap(z=df_plot.values, x=xvals, y=yvals, 
#                            zmin=0, zmax=1, 
#                           # coloraxis='coloraxis',
#                           ),
#               row=row,col=3)


fig.update_coloraxes(cmin=0, cmax=1, colorbar_title_text='DL weight<br>for null')



# Axes properties
fig.update_xaxes(title='Length')
fig.update_yaxes(title='Lag-1 autocorrelation (rho)')

fig.update_xaxes(automargin=False)
fig.update_yaxes(automargin=False)


fig.update_layout(width=300, height=500,
                  margin=dict(l=60, r=10, b=75, t=30),
                  font=dict(family='Times New Roman'))

fig.write_html('temp.html')

fig.write_image('fig_test_dl_nulls_lambda.png', scale=8)













