#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 11:45:50 2023

Make fig showing chick heart time series being detrended for different span and bandwidth

@author: tbury
"""


import numpy as np
import pandas as pd
import ewstools

import plotly.express as px


bw_vals = [2,5,10,20,40]
span_vals = [5,10,20,40,80]

# Import chick heart data
tsid = 8
df_raw = pd.read_csv('../../data/df_chick.csv')
df = df_raw[df_raw['type']=='pd'].query('tsid==@tsid')

# Load in transition times
df_transition = pd.read_csv('../../code/test_chick_heart/output/df_transitions.csv')
df_transition.set_index('tsid', inplace=True)
transition = df_transition.loc[tsid]['transition']

series = df.set_index('Beat number')['IBI (s)']


list_df = []

for idx in range(len(bw_vals)):
    
    # Detrend using Gaussian
    bw = bw_vals[idx]
    ts = ewstools.TimeSeries(series, transition=transition)
    ts.detrend(method='Gaussian', bandwidth=bw)        
    df_state = ts.state
    df_state['title'] = 'bandwidth = {:d}'.format(bw)
    list_df.append(df_state)    
    
    # Detrend using Lowess
    span = span_vals[idx]
    ts = ewstools.TimeSeries(series, transition=transition)
    ts.detrend(method='Lowess', span=span)        
    df_state = ts.state
    df_state['title'] = 'span = {:d}'.format(span)
    list_df.append(df_state)      
    
    
# for bw in bw_vals:
#     # Detrend
#     ts = ewstools.TimeSeries(series, transition=transition)
#     ts.detrend(method='Gaussian', bandwidth=bw)        
    
#     df_state = ts.state
#     df_state['title'] = 'bandwidth = {:.2f}'.format(bw)
#     list_df.append(df_state)
    
# for span in span_vals:
#     # Detrend
#     ts = ewstools.TimeSeries(series, transition=transition)
#     ts.detrend(method='Lowess', span=span)        
    
#     df_state = ts.state
#     df_state['title'] = 'span = {:.2f}'.format(span)
#     list_df.append(df_state)
    

df_plot = pd.concat(list_df).reset_index()

df_plot = df_plot.melt(id_vars = ['title','Beat number'], value_vars=['state','smoothing'])


fig = px.line(df_plot, x='Beat number', y='value', color='variable', facet_col_wrap=2, facet_col='title')
fig.update_xaxes(range=[0,260])
fig.update_yaxes(range=[0.4,1.6])
fig.for_each_annotation(lambda a: a.update(text=a.text[a.text.find("=")+1:]))

# General layout properties
fig.update_layout(height=800,
                  width=700,
                  showlegend=False,
                  margin={'l':40,'r':5,'b':60,'t':50},
                  font=dict(size=15, family='Times New Roman'),
                  paper_bgcolor='rgba(255,255,255,1)',
                  plot_bgcolor='rgba(255,255,255,1)',
                  )


font_size_letter_label = 10
font_size_titles = 12

linewidth = 0.7
linewidth_axes = 0.5
tickwidth = 0.5
ticklen = 2


# Opacity of DL probabilities for different bifs
opacity = 0.5


# dist from axis to axis label
xaxes_standoff = 0
yaxes_standoff = 0

# Global y axis properties
fig.update_yaxes(showline=True,
                  ticks="outside",
                  tickwidth=tickwidth,
                  ticklen=ticklen,
                  linecolor='black',
                  linewidth=linewidth_axes,
                  mirror=False,
                  showgrid=False,
                  automargin=False,
                  title_standoff=yaxes_standoff,
                  title='IBI (s)',
                  )

# Global x axis properties
fig.update_xaxes(showline=True,
                 linecolor='black',
                 linewidth=linewidth_axes,
                 mirror=False,
                 showgrid=False,
                 automargin=False,
                 title_standoff=xaxes_standoff
                 )


# Titles
y_pos = 1.06
title_gauss = dict(
        x=0.5,
        y=y_pos,
        text='<b>Gaussian smoothing</b>',
        xref='x domain',
        yref='paper',
        showarrow=False,
        font = dict(
                color = "black",
                size = 15)
        )

title_lowess = dict(
        x=0.5,
        y=y_pos,
        text='<b>Lowess smoothing</b>',
        xref='x2 domain',
        yref='paper',
        showarrow=False,
        font = dict(
                color = "black",
                size = 15)
        )

fig.add_annotation(title_gauss)
fig.add_annotation(title_lowess)


# Specific x axes properties
fig.update_xaxes(ticks="outside",
                  tickwidth=tickwidth,
                  ticklen=ticklen,
                  # row=5,
                  )

# fig.write_html('temp.html')

fig.write_image('figures/fig_preprocessing_timeseries.png', scale=4)








