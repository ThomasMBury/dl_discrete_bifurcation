#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:23:35 2022

Make fig with rows
- simulation
- variance
- lag-1 ac
- dl preds

and cols
- period-doubling model
- NS model
- fold model
- transcritical model
- pitchfork model


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
df_plot = pd.read_csv("output/df_plot.csv")
# DL probabiliyt for *any* bifurcation
bif_labels = ["1", "2", "3", "4", "5"]
df_plot["any"] = df_plot[bif_labels].dropna().sum(axis=1)


# Pixels to mm
mm_to_pixel = 96 / 25.4  # 96 dpi, 25.4mm in an inch

# Nature width of single col fig : 89mm
# Nature width of double col fig : 183mm

# Get width of single panel in pixels
fig_width = 183 * mm_to_pixel  # try single col width
fig_height = fig_width * 0.6


# Colour scheme
# cols = px.colors.qualitative.D3 # blue, orange, green, red, purple, brown
cols = (
    px.colors.qualitative.Plotly
)  # blue, red, green, purple, orange, cyan, pink, light green
col_grays = px.colors.sequential.gray

dic_colours = {
    "state": "gray",
    "smoothing": col_grays[2],
    "variance": cols[1],
    "ac": cols[2],
    "dl_any": cols[0],
    "dl_specific": cols[4],
    "dl_null": "black",
    "dl_pd": cols[3],
    "dl_ns": cols[4],
    "dl_fold": cols[5],
    "dl_tc": cols[6],
    "dl_pf": cols[7],
}


col_other_bif = "gray"
dic_colours = {
    "state": "gray",
    "smoothing": col_grays[2],
    "variance": cols[1],
    "ac": cols[2],
    "dl_any": cols[0],
    "dl_specific": cols[4],
    "dl_null": "black",
    "dl_pd": col_other_bif,
    "dl_ns": col_other_bif,
    "dl_fold": col_other_bif,
    "dl_tc": col_other_bif,
    "dl_pf": col_other_bif,
}


font_size = 10
font_family = "Times New Roman"
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
scale = 8  # default dpi=72 - nature=300-600

fig = make_subplots(
    rows=4,
    cols=5,
    shared_xaxes=True,
    vertical_spacing=0.04,
)


# ----------------
# Col 1: period-doubling
# ------------------

col = 1

df = df_plot[df_plot["model"] == "pd"]
# Trace for trajectory
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["state"],
        marker_color=dic_colours["state"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=col,
)

# Trace for smoothing
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["smoothing"],
        marker_color=dic_colours["smoothing"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=col,
)


# Trace for variance
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["variance"],
        marker_color=dic_colours["variance"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=2,
    col=col,
)


# Trace for lag-1 AC
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["ac1"],
        marker_color=dic_colours["ac"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=3,
    col=col,
)


# Weight for any bif
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["any"],
        marker_color=dic_colours["dl_any"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=4,
    col=col,
)


# Weight for PD
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["1"],
        marker_color=dic_colours["dl_specific"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=4,
    col=col,
)

# Weight for NS
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["2"],
        marker_color=dic_colours["dl_ns"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=4,
    col=col,
)

# Weight for Fold
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["3"],
        marker_color=dic_colours["dl_fold"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=4,
    col=col,
)

# Weight for transcritical
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["4"],
        marker_color=dic_colours["dl_tc"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=4,
    col=col,
)

# Weight for pitchfork
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["5"],
        marker_color=dic_colours["dl_pf"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=4,
    col=col,
)


# ----------------
# Col 2: NS
# ------------------

col = 2

df = df_plot[df_plot["model"] == "ns"]
# Trace for trajectory
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["state"],
        marker_color=dic_colours["state"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=col,
)

# Trace for smoothing
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["smoothing"],
        marker_color=dic_colours["smoothing"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=col,
)


# Trace for variance
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["variance"],
        marker_color=dic_colours["variance"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=2,
    col=col,
)


# Trace for lag-1 AC
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["ac1"],
        marker_color=dic_colours["ac"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=3,
    col=col,
)

# Weight for any bif
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["any"],
        marker_color=dic_colours["dl_any"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=4,
    col=col,
)

# Weight for PD
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["1"],
        marker_color=dic_colours["dl_pd"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=4,
    col=col,
)

# Weight for NS
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["2"],
        marker_color=dic_colours["dl_specific"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=4,
    col=col,
)

# Weight for Fold
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["3"],
        marker_color=dic_colours["dl_fold"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=4,
    col=col,
)

# Weight for transcritical
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["4"],
        marker_color=dic_colours["dl_tc"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=4,
    col=col,
)


# Weight for pitchfork
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["5"],
        marker_color=dic_colours["dl_pf"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=4,
    col=col,
)


# ----------------
# Col 3: Fold
# ------------------

col = 3

df = df_plot[df_plot["model"] == "fold"]
# Trace for trajectory
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["state"],
        marker_color=dic_colours["state"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=col,
)

# Trace for smoothing
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["smoothing"],
        marker_color=dic_colours["smoothing"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=col,
)


# Trace for variance
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["variance"],
        marker_color=dic_colours["variance"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=2,
    col=col,
)

# Trace for lag-1 AC
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["ac1"],
        marker_color=dic_colours["ac"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=3,
    col=col,
)

# Weight for any bif
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["any"],
        marker_color=dic_colours["dl_any"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=4,
    col=col,
)


# Weight for PD
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["1"],
        marker_color=dic_colours["dl_pd"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=4,
    col=col,
)

# Weight for NS
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["2"],
        marker_color=dic_colours["dl_ns"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=4,
    col=col,
)

# Weight for Fold
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["3"],
        marker_color=dic_colours["dl_specific"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=4,
    col=col,
)

# Weight for transcritical
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["4"],
        marker_color=dic_colours["dl_tc"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=4,
    col=col,
)


# Weight for pitchfork
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["5"],
        marker_color=dic_colours["dl_pf"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=4,
    col=col,
)


# ----------------
# Col 4: transcritical Kot model
# ------------------

col = 4

df = df_plot[df_plot["model"] == "tc"]
# Trace for trajectory
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["state"],
        marker_color=dic_colours["state"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=col,
)

# Trace for smoothing
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["smoothing"],
        marker_color=dic_colours["smoothing"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=col,
)


# Trace for variance
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["variance"],
        marker_color=dic_colours["variance"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=2,
    col=col,
)


# Trace for lag-1 AC
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["ac1"],
        marker_color=dic_colours["ac"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=3,
    col=col,
)


# Weight for any bif
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["any"],
        marker_color=dic_colours["dl_any"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=4,
    col=col,
)

# Weight for PD
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["1"],
        marker_color=dic_colours["dl_pd"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=4,
    col=col,
)

# Weight for NS
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["2"],
        marker_color=dic_colours["dl_ns"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=4,
    col=col,
)

# Weight for Fold
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["3"],
        marker_color=dic_colours["dl_fold"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=4,
    col=col,
)

# Weight for transcritical
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["4"],
        marker_color=dic_colours["dl_specific"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=4,
    col=col,
)


# Weight for pitchfork
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["5"],
        marker_color=dic_colours["dl_pf"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=4,
    col=col,
)


# ----------------
# Col 5: pitchfork Lorenz model
# ------------------

col = 5

df = df_plot[df_plot["model"] == "pf"]
# Trace for trajectory
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["state"],
        marker_color=dic_colours["state"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=col,
)

# Trace for smoothing
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["smoothing"],
        marker_color=dic_colours["smoothing"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=col,
)


# Trace for variance
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["variance"],
        marker_color=dic_colours["variance"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=2,
    col=col,
)


# Trace for lag-1 AC
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["ac1"],
        marker_color=dic_colours["ac"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=3,
    col=col,
)

# Weight for any bif
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["any"],
        marker_color=dic_colours["dl_any"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=4,
    col=col,
)

# Weight for PD
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["1"],
        marker_color=dic_colours["dl_pd"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=4,
    col=col,
)

# Weight for NS
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["2"],
        marker_color=dic_colours["dl_ns"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=4,
    col=col,
)

# Weight for Fold
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["3"],
        marker_color=dic_colours["dl_fold"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=4,
    col=col,
)

# Weight for transcritical
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["4"],
        marker_color=dic_colours["dl_tc"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=4,
    col=col,
)


# Weight for pitchfork
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["5"],
        marker_color=dic_colours["dl_specific"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=4,
    col=col,
)


# --------------
# Shapes
# --------------

list_shapes = []

# Vertical lines for where transitions occur
t_transition = 500

#  Line for PD transition
shape = {
    "type": "line",
    "x0": t_transition,
    "y0": 0,
    "x1": t_transition,
    "y1": 1,
    "xref": "x",
    "yref": "paper",
    "line": {"width": linewidth, "dash": "dot"},
}
list_shapes.append(shape)

#  Line for NS transition
shape = {
    "type": "line",
    "x0": t_transition,
    "y0": 0,
    "x1": t_transition,
    "y1": 1,
    "xref": "x5",
    "yref": "paper",
    "line": {"width": linewidth, "dash": "dot"},
}
list_shapes.append(shape)


#  Line for Fold transition
shape = {
    "type": "line",
    "x0": t_transition,
    "y0": 0,
    "x1": t_transition,
    "y1": 1,
    "xref": "x9",
    "yref": "paper",
    "line": {"width": linewidth, "dash": "dot"},
}
list_shapes.append(shape)


#  Line for TC transition
shape = {
    "type": "line",
    "x0": t_transition,
    "y0": 0,
    "x1": t_transition,
    "y1": 1,
    "xref": "x13",
    "yref": "paper",
    "line": {"width": linewidth, "dash": "dot"},
}
list_shapes.append(shape)


#  Line for PF transition
shape = {
    "type": "line",
    "x0": t_transition,
    "y0": 0,
    "x1": t_transition,
    "y1": 1,
    "xref": "x17",
    "yref": "paper",
    "line": {"width": linewidth, "dash": "dot"},
}
list_shapes.append(shape)


fig["layout"].update(shapes=list_shapes)


# --------------
# Add annotations
# ----------------------

list_annotations = []


# Letter labels for each panel
import string

label_letters = string.ascii_lowercase

axes_numbers = [str(n) for n in np.arange(1, 21)]
axes_numbers[0] = ""
idx = 0
for axis_number in axes_numbers:
    label_annotation = dict(
        x=0.01,
        y=1.00,
        text="<b>{}</b>".format(label_letters[idx]),
        xref="x{} domain".format(axis_number),
        yref="y{} domain".format(axis_number),
        showarrow=False,
        font=dict(color="black", size=font_size_letter_label),
    )
    list_annotations.append(label_annotation)
    idx += 1


# Bifurcation titles
y_pos = 1.06
title_pd = dict(
    x=0.5,
    y=y_pos,
    text="Period-doubling",
    xref="x domain",
    yref="paper",
    showarrow=False,
    font=dict(color="black", size=font_size_titles),
)
title_ns = dict(
    x=0.5,
    y=y_pos,
    text="Neimark-Sacker",
    xref="x2 domain",
    yref="paper",
    showarrow=False,
    font=dict(color="black", size=font_size_titles),
)
title_fold = dict(
    x=0.5,
    y=y_pos,
    text="Fold",
    xref="x3 domain",
    yref="paper",
    showarrow=False,
    font=dict(color="black", size=font_size_titles),
)

title_tc = dict(
    x=0.5,
    y=y_pos,
    text="Transcritical",
    xref="x4 domain",
    yref="paper",
    showarrow=False,
    font=dict(color="black", size=font_size_titles),
)
title_pf = dict(
    x=0.5,
    y=y_pos,
    text="Pitchfork",
    xref="x5 domain",
    yref="paper",
    showarrow=False,
    font=dict(color="black", size=font_size_titles),
)


# # label for scaling factor of variance (10^-3)
# axes_numbers = ['7','8','9']
# for axis_number in axes_numbers:
#     label_scaling = dict(
#         x=0,
#         y=1,
#         text='&times;10<sup>-3</sup>',
#         xref='x{} domain'.format(axis_number),
#         yref='y{} domain'.format(axis_number),
#         showarrow=False,
#         font = dict(
#                 color = "black",
#                 size = font_size)
#         )
#     list_annotations.append(label_scaling)


# Arrows to indiciate rolling window
rw = 0.5
axes_numbers = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
arrowhead = 1
arrowsize = 1.5
arrowwidth = 0.4

for axis_number in axes_numbers:
    # Make right-pointing arrow
    annotation_arrow_right = dict(
        x=0,  # arrows' head
        y=0.1,  # arrows' head
        ax=500 * rw,  # arrows' tail
        ay=0.1,  # arrows' tail
        xref="x{}".format(axis_number),
        yref="y{} domain".format(axis_number),
        axref="x{}".format(axis_number),
        ayref="y{} domain".format(axis_number),
        text="",  # if you want only the arrow
        showarrow=True,
        arrowhead=arrowhead,
        arrowsize=arrowsize,
        arrowwidth=arrowwidth,
        arrowcolor="black",
    )
    # Make left-pointing arrow
    annotation_arrow_left = dict(
        ax=0,  # arrows' head
        y=0.1,  # arrows' head
        x=500 * rw,  # arrows' tail
        ay=0.1,  # arrows' tail
        xref="x{}".format(axis_number),
        yref="y{} domain".format(axis_number),
        axref="x{}".format(axis_number),
        ayref="y{} domain".format(axis_number),
        text="",  # if you want only the arrow
        showarrow=True,
        arrowhead=arrowhead,
        arrowsize=arrowsize,
        arrowwidth=arrowwidth,
        arrowcolor="black",
    )

    # Append to annotations
    list_annotations.append(annotation_arrow_left)
    list_annotations.append(annotation_arrow_right)


list_annotations.append(label_annotation)

list_annotations.append(title_pd)
list_annotations.append(title_ns)
list_annotations.append(title_fold)
list_annotations.append(title_tc)
list_annotations.append(title_pf)


fig["layout"].update(annotations=list_annotations)


# -------------
# Axes properties
# -----------

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
)

# Global x axis properties
fig.update_xaxes(
    range=[0, 650],
    showline=True,
    linecolor="black",
    linewidth=linewidth_axes,
    mirror=False,
    showgrid=False,
    automargin=False,
    title_standoff=xaxes_standoff,
)


# Specific x axes properties
fig.update_xaxes(
    title="Time",
    ticks="outside",
    tickwidth=tickwidth,
    ticklen=ticklen,
    row=4,
)

# fig.update_xaxes(mirror=False,
#                  row=1,
#                  )

# Specific y axes properties
fig.update_yaxes(title="State", row=1, col=1)

fig.update_yaxes(title="Lag-1 AC", row=3, col=1)

fig.update_yaxes(title="Variance", row=2, col=1)

fig.update_yaxes(title="DL probability", row=4, col=1)

fig.update_yaxes(range=[100, 200], row=1, col=1)

fig.update_yaxes(range=[15, 40], row=1, col=2)

fig.update_yaxes(range=[-0.5, 12], row=1, col=3)

fig.update_yaxes(range=[0.7, 1.1], row=1, col=4)

fig.update_yaxes(range=[-0.05, 0.5], row=1, col=5)

# fig.update_yaxes(range=[0.65,0.75],
#                   row=3, col=2)
fig.update_yaxes(range=[-0.05, 1.05], row=4)


# General layout properties
fig.update_layout(
    height=fig_height,
    width=fig_width,
    margin={"l": 35, "r": 5, "b": 30, "t": 35},
    font=dict(size=font_size, family=font_family),
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
)

fig.update_traces(connectgaps=True)

# Export as temp image
# fig.write_image("output/fig_model_nulls.png", scale=8)
fig.write_image("../../results/figure_s3.png", scale=8)

# # Import image with Pil to assert dpi and export - this assigns correct
# # dimensions in mm for figure.
# from PIL import Image
# img = Image.open('output/temp.png')
# dpi=96*8 # (default dpi) * (scaling factor)
# img.save('../../results/figure_2.png', dpi=(dpi,dpi))

# # Export time taken for script to run
# end_time = time.time()
# time_taken = end_time - start_time
# path = 'time_make_fig.txt'
# with open(path, 'w') as f:
#     f.write('{:.2f}'.format(time_taken))
