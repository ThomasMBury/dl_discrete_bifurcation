#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 11:45:50 2023

Make fig showing AUC results for different detrending (Lowess/Guassian)

@author: tbury
"""


import numpy as np
import pandas as pd
import ewstools

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


cols = np.array(px.colors.qualitative.Plotly)

bw_vals = [2,5,10,20,40]
span_vals = [5,10,20,40,80]

# Import AUC data
list_df = []
for bw in bw_vals:
    df = pd.read_csv('output/df_roc_bw_{}.csv'.format(bw))
    df_auc = df.groupby('ews')['auc'].max().to_frame().reset_index()
    df_auc['bw'] = bw
    list_df.append(df_auc)
df_auc_bw = pd.concat(list_df)
df_auc_bw['ews'] = df_auc_bw['ews'].replace({'DL bif': 'DL'})

list_df = []
for span in span_vals:
    df = pd.read_csv('output/df_roc_span_{}.csv'.format(span))
    df_auc = df.groupby('ews')['auc'].max().to_frame().reset_index()
    df_auc['span'] = span
    list_df.append(df_auc)
df_auc_span = pd.concat(list_df)

df_auc_bw['bw'] = df_auc_bw['bw'].astype(str)
df_auc_span['span'] = df_auc_span['span'].astype(str)
df_auc_span['ews'] = df_auc_span['ews'].replace({'DL bif': 'DL'})


#------------
# Bar plot of AUC score - bw
#------------

font_size_letter_label = 10
font_size_titles = 12

linewidth = 0.7
linewidth_axes = 0.5
tickwidth = 0.5
ticklen = 2

fig = px.bar(df_auc_bw, x='bw', y='auc', color='ews', barmode='group',
             color_discrete_sequence=cols[[0,2,1]],
             )

# General layout properties
fig.update_layout(height=400,
                  width=400,
                  showlegend=True,
                  margin={'l':50,'r':5,'b':50,'t':50},
                  font=dict(size=15, family='Times New Roman'),
                  paper_bgcolor='rgba(255,255,255,1)',
                  plot_bgcolor='rgba(255,255,255,1)',
                  title='<b>Gaussian smoothing</b>',
                  title_x=0.5,
                  bargap=0.4,
                  legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=0.95,
                        xanchor="right",
                        x=1,
                        title='',
                  ))

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
                  title='AUC score',
                  range=[0.5,1],
                  )

# Global x axis properties
fig.update_xaxes(showline=True,
                 linecolor='black',
                 linewidth=linewidth_axes,
                 mirror=False,
                 showgrid=False,
                 automargin=False,
                 title_standoff=xaxes_standoff,
                 title='Bandwidth'
                 )

fig.add_annotation(x=-0.3, y=1,
            text='<b>a</b>',
            showarrow=False,
            font=dict(family='Times New Roman',
                      size=20,
                      )
            )
fig.write_image('figures/temp/fig_bar_auc_bw.png', scale=4)





#------------
# Bar plot of AUC score - span
#------------

font_size_letter_label = 10
font_size_titles = 12

linewidth = 0.7
linewidth_axes = 0.5
tickwidth = 0.5
ticklen = 2

fig = px.bar(df_auc_span, x='span', y='auc', color='ews', barmode='group',
             color_discrete_sequence=cols[[0,2,1]],
)

# General layout properties
fig.update_layout(height=400,
                  width=400,
                  showlegend=True,
                  margin={'l':50,'r':5,'b':50,'t':50},
                  font=dict(size=15, family='Times New Roman'),
                  paper_bgcolor='rgba(255,255,255,1)',
                  plot_bgcolor='rgba(255,255,255,1)',
                  title='<b>Lowess smoothing</b>',
                  title_x=0.5,
                  bargap=0.4,
                  legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=0.95,
                        xanchor="right",
                        x=1,
                        title='',
                  ))

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
                  title='AUC score',
                  range=[0.5,1],
                  )

# Global x axis properties
fig.update_xaxes(showline=True,
                 linecolor='black',
                 linewidth=linewidth_axes,
                 mirror=False,
                 showgrid=False,
                 automargin=False,
                 title_standoff=xaxes_standoff,
                 title='Span'
                 )

fig.add_annotation(x=-0.3, y=1,
            text='<b>b</b>',
            showarrow=False,
            font=dict(family='Times New Roman',
                      size=20,
                      )
            )

fig.write_image('figures/temp/fig_bar_auc_span.png', scale=4)





#------------
# Box plot of DL weight for period-doubling - bw
#------------


# Import DL weights
list_df = []
for bw in bw_vals:
    df = pd.read_csv('output/df_dl_pd_fixed_bw_{}.csv'.format(bw))
    df['bw'] = bw
    list_df.append(df[['1','bw']])
df_pd_weights_bw = pd.concat(list_df)
df_pd_weights_bw['bw'] = df_pd_weights_bw['bw'].astype('str')



font_size_letter_label = 10
font_size_titles = 12

linewidth = 0.7
linewidth_axes = 0.5
tickwidth = 0.5
ticklen = 2

fig = px.box(df_pd_weights_bw, x='bw', y='1',
             color_discrete_sequence=['#f59542'],
             )
# fig.write_html('temp3.html')


# General layout properties
fig.update_layout(height=400,
                  width=400,
                  showlegend=True,
                  margin={'l':50,'r':5,'b':50,'t':50},
                  font=dict(size=15, family='Times New Roman'),
                  paper_bgcolor='rgba(255,255,255,1)',
                  plot_bgcolor='rgba(255,255,255,1)',
                  # title='Lowess smoothing',
                  title_x=0.5,
                  boxgap=0.5,
                  # legend=dict(
                  #       orientation="h",
                  #       yanchor="bottom",
                  #       y=0.95,
                  #       xanchor="right",
                  #       x=1,
                  #       title='',
                  # )
)

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
                  title='DL weight for PD',
                  range=[-0.05, 1.05],
                  )

# Global x axis properties
fig.update_xaxes(showline=True,
                 linecolor='black',
                 linewidth=linewidth_axes,
                 mirror=False,
                 showgrid=False,
                 automargin=False,
                 title_standoff=xaxes_standoff,
                 title='Bandwidth'
                 )

fig.add_annotation(x=-0.3, y=1,
            text='<b>c</b>',
            showarrow=False,
            font=dict(family='Times New Roman',
                      size=20,
                      )
            )

fig.write_image('figures/temp/fig_box_dl_weight_bw.png', scale=4)




#------------
# Box plot of DL weight for period-doubling - span
#------------

list_df = []
for span in span_vals:
    df = pd.read_csv('output/df_dl_pd_fixed_span_{}.csv'.format(span))
    df['span'] = span
    list_df.append(df[['1','span']])
df_pd_weights_span = pd.concat(list_df)
df_pd_weights_span['span'] = df_pd_weights_span['span'].astype('str')


font_size_letter_label = 10
font_size_titles = 12

linewidth = 0.7
linewidth_axes = 0.5
tickwidth = 0.5
ticklen = 2

fig = px.box(df_pd_weights_span, x='span', y='1',
             color_discrete_sequence=['#f59542'],
)
# fig.write_html('temp3.html')


# General layout properties
fig.update_layout(height=400,
                  width=400,
                  showlegend=True,
                  margin={'l':50,'r':5,'b':50,'t':50},
                  font=dict(size=15, family='Times New Roman'),
                  paper_bgcolor='rgba(255,255,255,1)',
                  plot_bgcolor='rgba(255,255,255,1)',
                  # title='Lowess smoothing',
                  title_x=0.5,
                  boxgap=0.5,
                  # legend=dict(
                  #       orientation="h",
                  #       yanchor="bottom",
                  #       y=0.95,
                  #       xanchor="right",
                  #       x=1,
                  #       title='',
                  # )
)

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
                  title='DL weight for PD',
                  range=[-0.05, 1.05],
                  )

# Global x axis properties
fig.update_xaxes(showline=True,
                 linecolor='black',
                 linewidth=linewidth_axes,
                 mirror=False,
                 showgrid=False,
                 automargin=False,
                 title_standoff=xaxes_standoff,
                 title='Span'
                 )


fig.add_annotation(x=-0.3, y=1,
            text='<b>d</b>',
            showarrow=False,
            font=dict(family='Times New Roman',
                      size=20,
                      )
            )

fig.write_image('figures/temp/fig_box_dl_weight_span.png', scale=4)





#------------
# Bar plot of DL chosen bifurcation - bw
#------------

# Import favourite bifurcations
list_df = []
for bw in bw_vals:
    df = pd.read_csv('output/df_fav_bif_bw_{}.csv'.format(bw))
    df['bw'] = bw
    list_df.append(df)
df_fav_bif_bw = pd.concat(list_df)
df_fav_bif_bw['bw'] = df_fav_bif_bw['bw'].astype('str')
df_fav_bif_bw['bif_id'] = df_fav_bif_bw['bif_id'].astype('str')


rep_dict = {'1':'PD','2':'NS', '3':'fold','4':'TC', '5':'PF'}
df_fav_bif_bw['bif_id'] = df_fav_bif_bw['bif_id'].replace(rep_dict)


fig = px.bar(df_fav_bif_bw, x='bw', y='count', color='bif_id',
             color_discrete_sequence=cols[4:])
# fig.write_html('temp5.html')



font_size_letter_label = 10
font_size_titles = 12

linewidth = 0.7
linewidth_axes = 0.5
tickwidth = 0.5
ticklen = 2

# General layout properties
fig.update_layout(height=400,
                  width=400,
                  showlegend=True,
                  margin={'l':50,'r':5,'b':50,'t':50},
                  font=dict(size=15, family='Times New Roman'),
                  paper_bgcolor='rgba(255,255,255,1)',
                  plot_bgcolor='rgba(255,255,255,1)',
                  # title='Lowess smoothing',
                  bargap=0.5,
                  title_x=0.5,
                   legend=dict(
                         orientation="h",
                         yanchor="bottom",
                         y=0.95,
                         xanchor="right",
                         x=1,
                         title='',
                   )
)

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
                  title='Count',
                  # range=[-0.05, 1.05],
                  )

# Global x axis properties
fig.update_xaxes(showline=True,
                 linecolor='black',
                 linewidth=linewidth_axes,
                 mirror=False,
                 showgrid=False,
                 automargin=False,
                 title_standoff=xaxes_standoff,
                 title='Bandwidth'
                 )

fig.add_annotation(x=-0.3, y=1.05,
                   yref='paper',
            text='<b>e</b>',
            showarrow=False,
            font=dict(family='Times New Roman',
                      size=20,
                      )
            )

fig.write_image('figures/temp/fig_box_dl_favbif_bw.png', scale=4)



#------------
# Bar plot of DL chosen bifurcation - span
#------------


# Import favourite bifurcations
list_df = []
for span in span_vals:
    df = pd.read_csv('output/df_fav_bif_span_{}.csv'.format(span))
    df['span'] = span
    list_df.append(df)
df_fav_bif_span = pd.concat(list_df)
df_fav_bif_span['span'] = df_fav_bif_span['span'].astype('str')
df_fav_bif_span['bif_id'] = df_fav_bif_span['bif_id'].astype('str')

rep_dict = {'1':'PD','2':'NS', '3':'fold','4':'TC', '5':'PF'}
df_fav_bif_span['bif_id'] = df_fav_bif_span['bif_id'].replace(rep_dict)


fig = px.bar(df_fav_bif_span, x='span', y='count', color='bif_id',
             color_discrete_sequence=cols[4:])
# fig.write_html('temp5.html')

font_size_letter_label = 10
font_size_titles = 12

linewidth = 0.7
linewidth_axes = 0.5
tickwidth = 0.5
ticklen = 2

# General layout properties
fig.update_layout(height=400,
                  width=400,
                  showlegend=True,
                  margin={'l':50,'r':5,'b':50,'t':50},
                  font=dict(size=15, family='Times New Roman'),
                  paper_bgcolor='rgba(255,255,255,1)',
                  plot_bgcolor='rgba(255,255,255,1)',
                  # title='Lowess smoothing',
                  title_x=0.5,
                  bargap=0.5,
                   legend=dict(
                         orientation="h",
                         yanchor="bottom",
                         y=0.95,
                         xanchor="right",
                         x=1,
                         title='',
                   )
)

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
                  title='Count',
                  # range=[-0.05, 1.05],
                  )

# Global x axis properties
fig.update_xaxes(showline=True,
                 linecolor='black',
                 linewidth=linewidth_axes,
                 mirror=False,
                 showgrid=False,
                 automargin=False,
                 title_standoff=xaxes_standoff,
                 title='Span'
                 )

fig.add_annotation(x=-0.3, y=1.05,
                   yref='paper',
            text='<b>f</b>',
            showarrow=False,
            font=dict(family='Times New Roman',
                      size=20,
                      )
            )

fig.write_image('figures/temp/fig_box_dl_favbif_span.png', scale=4)


#---------
# Combine figs
#----------

from PIL import Image

list_filenames = ['fig_bar_auc_bw',
                  'fig_bar_auc_span',
                  'fig_box_dl_weight_bw',
                  'fig_box_dl_weight_span',
                  'fig_box_dl_favbif_bw',
                  'fig_box_dl_favbif_span',
                  ]
list_filenames = ['figures/temp/{}.png'.format(s) for s in list_filenames]

list_img = []
for filename in list_filenames:
    img = Image.open(filename)
    list_img.append(img)

# Get heght and width of individual panels
ind_height = list_img[0].height
ind_width = list_img[0].width


# Create frame
dst = Image.new('RGB',(2*ind_width, 3*ind_height), (255,255,255))

# Paste in images
i=0
for y in np.arange(3)*ind_height:
    for x in np.arange(2)*ind_width:
        dst.paste(list_img[i], (x,y))
        i+=1


dpi=96*8 # (default dpi) * (scaling factor)
dst.save('figures/fig_preproc_perf.png',
          dpi=(dpi,dpi))

















