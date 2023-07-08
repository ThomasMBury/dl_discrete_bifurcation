#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:23:35 2022

Make figure for select period-doubling trajectories in chick-heart data
for main manuscript

Make fig with rows
- trajectory and smoothing
- variance
- lag-1 ac
- dl preds

and each col a different trajectory

@author: tbury
"""
import time
start_time = time.time()

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Load in EWS data
df_ews = pd.read_csv('../test_chick_heart/output/df_ews_pd_rolling.csv')

# Load in DL data
df_dl = pd.read_csv('../test_chick_heart/output/df_dl_pd_rolling.csv')
df_dl['time'] = df_dl['Beat number']
df_dl = df_dl.groupby(['tsid','time']).mean().reset_index()
df_dl['any'] = df_dl[['1','2','3','4','5']].sum(axis=1)

# Load in transition times
df_transitions = pd.read_csv('../test_chick_heart/output/df_transitions.csv')
df_transitions.set_index('tsid', inplace=True)

tsid_plot = [8, 20, 14, 22, 16]

# Pixels to mm
mm_to_pixel = 96/25.4 # 96 dpi, 25.4mm in an inch

# Nature width of single col fig : 89mm
# Nature width of double col fig : 183mm

# Get width of single panel in pixels
fig_width = 183*mm_to_pixel # try single col width
fig_height = fig_width*0.6


# Colour scheme
# cols = px.colors.qualitative.D3 # blue, orange, green, red, purple, brown
cols = px.colors.qualitative.Plotly # blue, red, green, purple, orange, cyan, pink, light green
col_grays = px.colors.sequential.gray

col_other_bif = 'gray'
dic_colours = {
        'state':'gray',
        'smoothing':col_grays[2],
        'variance':cols[1],
        'ac':cols[2],
        'dl_any':cols[0],
        'dl_specific':cols[4],
        'dl_null':'black',
        'dl_pd':col_other_bif,  
        'dl_ns':col_other_bif,
        'dl_fold':col_other_bif,
        'dl_tc':col_other_bif,
        'dl_pf':col_other_bif,
     }



font_size = 10
font_family = 'Times New Roman'
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


# Scale up factor on image export
scale = 8 # default dpi=72 - nature=300-600

fig = make_subplots(rows=4, cols=5,
                    shared_xaxes=True,
                    vertical_spacing=0.04,
                    )

list_shapes = []
list_annotations = []


for i, tsid in enumerate(tsid_plot):
    
    df_ews_sel = df_ews[df_ews['tsid']==tsid]
    df_dl_sel = df_dl[df_dl['tsid']==tsid]
    col = i+1
    
    # Trace for trajectory
    fig.add_trace(
        go.Scatter(x=df_ews_sel['Beat number'],
                   y=df_ews_sel['state'],
                   marker_color=dic_colours['state'],
                   showlegend=False,
                   line={'width':linewidth},
                   ),
        row=1,col=col,
        )
    
    # Trace for smoothing
    fig.add_trace(
        go.Scatter(x=df_ews_sel['Beat number'],
                   y=df_ews_sel['smoothing'],
                   marker_color=dic_colours['smoothing'],
                   showlegend=False,
                   line={'width':linewidth},
                   ),
        row=1,col=col,
        )
    
    # Trace for lag-1 AC
    fig.add_trace(
        go.Scatter(x=df_ews_sel['Beat number'],
                   y=df_ews_sel['ac1'],
                   marker_color=dic_colours['ac'],
                   showlegend=False,
                   line={'width':linewidth},
                   ),
        row=3,col=col,
    
        )
    
    # Trace for variance
    fig.add_trace(
        go.Scatter(x=df_ews_sel['Beat number'],
                   y=df_ews_sel['variance'],
                   marker_color=dic_colours['variance'],
                   showlegend=False,
                   line={'width':linewidth},
                   ),
        row=2,col=col,
    
        )
    
    # Weight for any bif
    fig.add_trace(
        go.Scatter(x=df_dl_sel['time'],
                   y=df_dl_sel['any'],
                   marker_color=dic_colours['dl_any'],
                   showlegend=False,
                   line={'width':linewidth},
                   ),
        row=4,col=col,
        )
    
    
    # Weight for PD
    fig.add_trace(
        go.Scatter(x=df_dl_sel['time'],
                   y=df_dl_sel['1'],
                   marker_color=dic_colours['dl_specific'],
                   showlegend=False,
                   line={'width':linewidth},
                   ),
        row=4,col=col,
        )
    
    # Weight for NS
    fig.add_trace(
        go.Scatter(x=df_dl_sel['time'],
                   y=df_dl_sel['2'],
                   marker_color=dic_colours['dl_ns'],
                   showlegend=False,
                   line={'width':linewidth},
                   opacity=opacity,
                   ),
        row=4,col=col,
        )
    
    # Weight for Fold
    fig.add_trace(
        go.Scatter(x=df_dl_sel['time'],
                   y=df_dl_sel['3'],
                   marker_color=dic_colours['dl_fold'],
                   showlegend=False,
                   line={'width':linewidth},
                   opacity=opacity,
                   ),
        row=4,col=col,
        )
    
    # Weight for transcritical
    fig.add_trace(
        go.Scatter(x=df_dl_sel['time'],
                   y=df_dl_sel['4'],
                   marker_color=dic_colours['dl_tc'],
                   showlegend=False,
                   line={'width':linewidth},
                   opacity=opacity,
                   ),
        row=4,col=col,
        )
    
    
    # Weight for pitchfork
    fig.add_trace(
        go.Scatter(x=df_dl_sel['time'],
                   y=df_dl_sel['5'],
                   marker_color=dic_colours['dl_pf'],
                   showlegend=False,
                   line={'width':linewidth},
                   opacity=opacity,
                   ),
        row=4,col=col,
        )
    
    
    # Vertical line for where transition occurs
    transition = df_transitions.loc[tsid]['transition']
    shape = {'type': 'line', 
              'x0': transition, 
              'y0': 0, 
              'x1': transition, 
              'y1': 1, 
              'xref': 'x{}'.format(i+1),
              'yref': 'paper',
              'line': {'width':linewidth,'dash':'dot'},
          }
    list_shapes.append(shape)
    
    
    # Let x range go 15% beyond transition
    tstart = df_ews_sel['Beat number'].iloc[0]
    tend = tstart + 1.15*(transition-tstart)  
    fig.update_xaxes(range=[tstart,tend], col=col)
        
    
    
    # Arrows to indiciate rolling window
    rw = 0.5
    arrowhead=1
    arrowsize=1.5
    arrowwidth=0.4
    
    axis_numbers = [6+i, 11+i]
    
    for axis_number in axis_numbers:
    
        # Make right-pointing arrow
        annotation_arrow_right = dict(
              x=0,  # arrows' head
              y=0.1,  # arrows' head
              ax=transition*rw,  # arrows' tail
              ay=0.1,  # arrows' tail
              xref='x{}'.format(axis_number),
              yref='y{} domain'.format(axis_number),
              axref='x{}'.format(axis_number),
              ayref='y{} domain'.format(axis_number),
              text='',  # if you want only the arrow
              showarrow=True,
              arrowhead=arrowhead,
              arrowsize=arrowsize,
              arrowwidth=arrowwidth,
              arrowcolor='black'
            )       
        # Make left-pointing arrow
        annotation_arrow_left = dict(
              ax=0,  # arrows' head
              y=0.1,  # arrows' head
              x=transition*rw,  # arrows' tail
              ay=0.1,  # arrows' tail
              xref='x{}'.format(axis_number),
              yref='y{} domain'.format(axis_number),
              axref='x{}'.format(axis_number),
              ayref='y{} domain'.format(axis_number),
              text='',  # if you want only the arrow
              showarrow=True,
              arrowhead=arrowhead,
              arrowsize=arrowsize,
              arrowwidth=arrowwidth,
              arrowcolor='black'
            )
    
        # Append to annotations
        list_annotations.append(annotation_arrow_left)
        list_annotations.append(annotation_arrow_right)
        
        
        
    
fig['layout'].update(shapes=list_shapes)



# #--------------
# # Add annotations
# #----------------------



# Letter labels for each panel
import string
label_letters = string.ascii_lowercase

axes_numbers = [str(n) for n in np.arange(1,21)]
axes_numbers[0] = ''
idx=0
for axis_number in axes_numbers:
    label_annotation = dict(
            x=0.01,
            y=1.00,
            text='<b>{}</b>'.format(label_letters[idx]),
            xref='x{} domain'.format(axis_number),
            yref='y{} domain'.format(axis_number),
            showarrow=False,
            font = dict(
                    color = "black",
                    size = font_size_letter_label)
            )
    list_annotations.append(label_annotation)
    idx+=1
    


fig['layout'].update(annotations=list_annotations)







#-------------
# Axes properties
#-----------


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


# Specific x axes properties
fig.update_xaxes(title='Beat number',
                 ticks="outside",
                 tickwidth=tickwidth,
                 ticklen=ticklen,
                 row=4,
                 )

# fig.update_xaxes(mirror=False,
#                  row=1,
#                  )

# Specific y axes properties
fig.update_yaxes(title='IBI (s)',
                  row=1,col=1)

fig.update_yaxes(title='Lag-1 AC',
                  row=3,col=1)

fig.update_yaxes(title='Variance',
                  row=2,col=1)    

fig.update_yaxes(title='DL probability',
                  row=4,col=1)   


fig.update_yaxes(range=[0.45, 1.4],
                  row=1,col=3)

fig.update_yaxes(range=[0.5,1.39],
                  row=1,col=5)

fig.update_yaxes(range=[-0.05,1.05], row=4)


# General layout properties
fig.update_layout(height=fig_height,
                  width=fig_width,
                  margin={'l':40,'r':5,'b':30,'t':5},
                  font=dict(size=font_size, family=font_family),
                  paper_bgcolor='rgba(255,255,255,1)',
                  plot_bgcolor='rgba(255,255,255,1)',
                  )

fig.update_traces(connectgaps=True)


# Export as temp image
fig.write_image('temp.png',scale=8)

# fig.write_html('temp.html')

# Import image with Pil to assert dpi and export - this assigns correct
# dimensions in mm for figure.
from PIL import Image
img = Image.open('temp.png')
dpi=96*8 # (default dpi) * (scaling factor)
img.save('../../results/figure_3.png', dpi=(dpi,dpi))

# Remove temp images
import os
try:
    os.remove('temp.png')
except:
    pass


# Time taken for script to run
end_time = time.time()
time_taken = end_time - start_time
print('Ran in {:.2f}s'.format(time_taken))


# # Export time taken for script to run
# end_time = time.time()
# time_taken = end_time - start_time
# path = 'time_make_fig.txt'
# with open(path, 'w') as f:
#     f.write('{:.2f}'.format(time_taken))






