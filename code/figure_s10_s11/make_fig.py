#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 19:39:06 2021

Make a figure that includes:
- Trajectory and smoothing
- Variance
- Lag-1 AC
- DL predictions

For each null trajectory in chick-heart data

@author: tbury
"""


import time

start_time = time.time()

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Load in EWS data
df_ews = pd.read_csv("../test_chick_heart/output/df_ews_null_rolling.csv")

# Import DL prediction data
df_dl = pd.read_csv("../test_chick_heart/output/df_dl_null_rolling.csv")
df_dl["any"] = df_dl[["1", "2", "3", "4", "5"]].sum(axis=1)
df_dl["time"] = df_dl["Beat number"]
# df_dl = df_dl.rename({col_name:str(col_name) for col_name in df_dl.columns.values})


# Colour scheme
# cols = px.colors.qualitative.D3 # blue, orange, green, red, purple, brown
cols = (
    px.colors.qualitative.Plotly
)  # blue, red, green, purple, orange, cyan, pink, light green
col_grays = px.colors.sequential.gray
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

linewidth = 1.2
opacity = 0.5


def make_grid_figure(df_ews, df_dl, letter_label, title, transition=False):
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0,
    )
    # Trace for trajectory
    fig.add_trace(
        go.Scatter(
            x=df_ews["Beat number"],
            y=df_ews["state"],
            marker_color=dic_colours["state"],
            showlegend=False,
            line={"width": linewidth},
        ),
        row=1,
        col=1,
    )

    # Trace for smoothing
    fig.add_trace(
        go.Scatter(
            x=df_ews["Beat number"],
            y=df_ews["smoothing"],
            marker_color=dic_colours["smoothing"],
            showlegend=False,
            line={"width": linewidth},
        ),
        row=1,
        col=1,
    )

    # Trace for lag-1 AC
    fig.add_trace(
        go.Scatter(
            x=df_ews["Beat number"],
            y=df_ews["ac1"],
            marker_color=dic_colours["ac"],
            showlegend=False,
            line={"width": linewidth},
        ),
        row=2,
        col=1,
    )

    # Trace for variance
    fig.add_trace(
        go.Scatter(
            x=df_ews["Beat number"],
            y=df_ews["variance"],
            marker_color=dic_colours["variance"],
            showlegend=False,
            line={"width": linewidth},
        ),
        row=3,
        col=1,
    )

    # Weight for any bif
    fig.add_trace(
        go.Scatter(
            x=df_dl["time"],
            y=df_dl["any"],
            marker_color=dic_colours["dl_any"],
            showlegend=False,
            line={"width": linewidth},
        ),
        row=4,
        col=1,
    )

    # Weight for PD
    fig.add_trace(
        go.Scatter(
            x=df_dl["time"],
            y=df_dl["1"],
            marker_color=dic_colours["dl_specific"],
            showlegend=False,
            line={"width": linewidth},
        ),
        row=4,
        col=1,
    )

    # Weight for NS
    fig.add_trace(
        go.Scatter(
            x=df_dl["time"],
            y=df_dl["2"],
            marker_color=dic_colours["dl_ns"],
            showlegend=False,
            line={"width": linewidth},
            opacity=opacity,
        ),
        row=4,
        col=1,
    )

    # Weight for Fold
    fig.add_trace(
        go.Scatter(
            x=df_dl["time"],
            y=df_dl["3"],
            marker_color=dic_colours["dl_fold"],
            showlegend=False,
            line={"width": linewidth},
            opacity=opacity,
        ),
        row=4,
        col=1,
    )

    # Weight for transcritical
    fig.add_trace(
        go.Scatter(
            x=df_dl["time"],
            y=df_dl["4"],
            marker_color=dic_colours["dl_tc"],
            showlegend=False,
            line={"width": linewidth},
            opacity=opacity,
        ),
        row=4,
        col=1,
    )

    # Weight for pitchfork
    fig.add_trace(
        go.Scatter(
            x=df_dl["time"],
            y=df_dl["5"],
            marker_color=dic_colours["dl_pf"],
            showlegend=False,
            line={"width": linewidth},
            opacity=opacity,
        ),
        row=4,
        col=1,
    )

    # --------------
    # Add vertical line where transition occurs
    # --------------

    if transition:
        # Add vertical lines where transitions occur
        list_shapes = []

        #  Make line for start of transition transition
        shape = {
            "type": "line",
            "x0": transition,
            "y0": 0,
            "x1": transition,
            "y1": 1,
            "xref": "x",
            "yref": "paper",
            "line": {"width": 2, "dash": "dot"},
        }

        # Add shape to list
        list_shapes.append(shape)

        fig["layout"].update(shapes=list_shapes)

    # --------------
    # Add labels and titles
    # ----------------------

    list_annotations = []

    label_annotation = dict(
        # x=sum(xrange)/2,
        x=0.03,
        y=1,
        text="<b>{}</b>".format(letter_label),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color="black", size=16),
    )

    title_annotation = dict(
        # x=sum(xrange)/2,
        x=0.65,
        y=1,
        text=title,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color="black", size=14),
    )

    list_annotations.append(label_annotation)
    list_annotations.append(title_annotation)

    fig["layout"].update(annotations=list_annotations)

    # -------
    # Axes properties
    # ---------

    # # Get rate of forcing
    # rof = df_properties[
    #     df_properties['tsid']==tsid]['rate of forcing (mV/s)'].iloc[0]
    # # If rate of forcing <=40, add more space to y limits
    # if rof<=40:
    #     ymin = df_traj_plot['Pressure (kPa)'].min()
    #     ymax = df_traj_plot['Pressure (kPa)'].max()
    #     ymin_plot = ymin-0.4*(ymax-ymin)
    #     ymax_plot = ymax+0.4*(ymax-ymin)
    #     fig.update_yaxes(range=[ymin_plot, ymax_plot],row=1,col=1)

    fig.update_xaxes(
        title={"text": "Beat number", "standoff": 5},
        ticks="outside",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        row=4,
        col=1,
    )

    # Global y axis properties
    fig.update_yaxes(
        showline=True,
        ticks="outside",
        linecolor="black",
        mirror=True,
        showgrid=False,
        automargin=False,
    )

    # Global x axis properties
    fig.update_xaxes(
        showline=True,
        linecolor="black",
        mirror=False,
        showgrid=False,
    )

    fig.update_xaxes(mirror=True, row=1, col=1)

    fig.update_yaxes(
        title={
            "text": "IBI (s)",
            "standoff": 5,
        },
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title={
            "text": "Lag-1 AC",
            "standoff": 5,
        },
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title={
            "text": "Variance",
            "standoff": 5,
        },
        row=3,
        col=1,
    )

    fig.update_yaxes(
        title={
            "text": "DL probability",
            "standoff": 5,
        },
        range=[-0.05, 1.07],
        row=4,
        col=1,
    )

    fig.update_layout(
        height=400,
        width=200,
        margin={"l": 50, "r": 10, "b": 20, "t": 10},
        font=dict(size=12, family="Times New Roman"),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
    )

    fig.update_traces(mode="lines")

    return fig


# -------------
# Make ind figs for period-doubling trajectories
# -----------

list_tsid = df_ews["tsid"].unique()

# Get average prediction from each model
df_dl_av = df_dl.groupby(["tsid", "time"]).mean().reset_index()


import string

list_letter_labels = string.ascii_lowercase[: len(list_tsid)]

for i, tsid in enumerate(list_tsid):
    letter_label = list_letter_labels[i]
    df_ews_spec = df_ews[df_ews["tsid"] == tsid]
    df_dl_spec = df_dl_av[df_dl_av["tsid"] == tsid]

    # Title
    # title = 'tsid={}'.format(tsid)
    title = ""
    fig = make_grid_figure(
        df_ews_spec, df_dl_spec, letter_label, title, transition=False
    )
    # Export as png
    fig.write_image("img_{}.png".format(tsid), scale=2)
    print("Exported image {}".format(tsid))


# -----------
# Combine images into single png - part 1
# -----------

from PIL import Image

list_img = []
filename = "../../results/figure_s10.png"

for tsid in np.arange(1, 13):
    img = Image.open("img_{}.png".format(tsid))
    list_img.append(img)

# Get height and width of individual panels
ind_height = list_img[0].height
ind_width = list_img[0].width

# Create frame
dst = Image.new("RGB", (6 * ind_width, 2 * ind_height), (255, 255, 255))

# Paste in images
i = 0
for y in np.arange(0, 2) * ind_height:
    for x in np.arange(0, 6) * ind_width:
        dst.paste(list_img[i], (x, y))
        i += 1

dst.save(filename)


# -----------
# Combine images into single png - part 2
# -----------

list_img = []
filename = "../../results/figure_s11.png"

for tsid in np.arange(13, 24):
    img = Image.open("img_{}.png".format(tsid))
    list_img.append(img)

# Get height and width of individual panels
ind_height = list_img[0].height
ind_width = list_img[0].width

# Create frame
dst = Image.new("RGB", (6 * ind_width, 2 * ind_height), (255, 255, 255))

# Paste in images
i = 0
for y in np.arange(0, 2) * ind_height:
    for x in np.arange(0, 6) * ind_width:
        try:
            dst.paste(list_img[i], (x, y))
            i += 1
        except:
            pass

dst.save(filename)

# Remove temp images
import os

for i in range(1, 24):
    try:
        os.remove("img_{}.png".format(i))
    except:
        pass

# Time taken for script to run
end_time = time.time()
time_taken = end_time - start_time
print("Ran in {:.2f}s".format(time_taken))


# # Export time taken for script to run
# end_time = time.time()
# time_taken = end_time - start_time
# path = 'time_make_fig.txt'
# with open(path, 'w') as f:
#     f.write('{:.2f}'.format(time_taken))
