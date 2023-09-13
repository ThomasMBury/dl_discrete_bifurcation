#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 11:45:50 2023

Make fig showing AUC results for different rolling windows

@author: tbury
"""


import numpy as np
import pandas as pd
import ewstools

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


cols = np.array(px.colors.qualitative.Plotly)

rw_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# Import AUC data
list_df = []
for rw in rw_vals:
    df = pd.read_csv("output/df_roc_rw_{}.csv".format(rw))
    df_auc = df.groupby("ews")["auc"].max().to_frame().reset_index()
    df_auc["rw"] = rw
    list_df.append(df_auc)
df_auc = pd.concat(list_df)

df_auc["rw"] = df_auc["rw"].astype(str)


# ------------
# Bar plot of AUC score - bw
# ------------

font_size_letter_label = 10
font_size_titles = 12

linewidth = 0.7
linewidth_axes = 0.5
tickwidth = 0.5
ticklen = 2

fig = px.bar(
    df_auc,
    x="rw",
    y="auc",
    color="ews",
    barmode="group",
    color_discrete_sequence=cols[[2, 1]],
)

# General layout properties
fig.update_layout(
    height=400,
    width=400,
    showlegend=True,
    margin={"l": 50, "r": 5, "b": 50, "t": 30},
    font=dict(size=15, family="Times New Roman"),
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
    # title='Gaussian smoothing',
    title_x=0.5,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.95,
        xanchor="right",
        x=1,
        title="",
    ),
)

# Opacity of DL probabilities for different bifs
opacity = 0.5

# dist from axis to axis label
xaxes_standoff = 0
yaxes_standoff = 0

# Global y axis properties
fig.update_yaxes(
    showline=True,
    ticks="outside",
    tickwidth=tickwidth,
    ticklen=ticklen,
    linecolor="black",
    linewidth=linewidth_axes,
    mirror=False,
    showgrid=False,
    automargin=False,
    title_standoff=yaxes_standoff,
    title="AUC score",
    range=[0.5, 1],
)

# Global x axis properties
fig.update_xaxes(
    showline=True,
    linecolor="black",
    linewidth=linewidth_axes,
    mirror=False,
    showgrid=False,
    automargin=False,
    title_standoff=xaxes_standoff,
    title="Rolling window",
)

# fig.add_annotation(x=-0.3, y=1,
#             text='<b>a</b>',
#             showarrow=False,
#             font=dict(family='Times New Roman',
#                       size=20,
#                       )
#             )

# fig.write_image('figures/fig_bar_auc_rw.png', scale=4)
fig.write_image("../../results/figure_s14.png", scale=4)
