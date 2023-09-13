#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:49:05 2023

Example application of deep learning classifier to single time series

@author: tbury
"""


import numpy as np
import pandas as pd
import ewstools

from tensorflow.keras.models import load_model

m1 = load_model("../../data/classifier_1.pkl")
m2 = load_model("../../data/classifier_2.pkl")

# Import data
tsid = 14
df = pd.read_csv("../../data/df_chick.csv").set_index("Beat number")
series = df.query('tsid==@tsid and type=="pd"')["IBI (s)"]

# Set up time series object
ts = ewstools.TimeSeries(series, transition=300)

# Detrend
ts.detrend(method="Gaussian", bandwidth=20)

# Compute variance and lag-1 autocorrelation
ts.compute_var(rolling_window=0.5)
ts.compute_auto(rolling_window=0.5, lag=1)

# Compute DL predictions
ts.apply_classifier_inc(m1, inc=10, name="m1", verbose=0)
ts.apply_classifier_inc(m2, inc=10, name="m2", verbose=0)

fig = ts.make_plotly(ens_avg=True)

# new_names = {
#     "state": "state",
#     "smoothing": "smoothing",
#     "variance": "variance",
#     "ac1": "ac1",
#     "DL class 0": "Null",
#     "DL class 1": "Period-doubling",
#     "DL class 2": "Neimark-Sacker",
#     "DL class 3": "Fold",
#     "DL class 4": "Transcritical",
#     "DL class 5": "Pitchfork",
# }
# fig.for_each_trace(lambda t: t.update(name=new_names[t.name]))

# fig.write_html("output/example.html")
fig.write_image("figures/example.png", scale=2)
