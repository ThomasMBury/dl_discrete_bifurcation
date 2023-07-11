#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:51:19 2022

Generic functions stored here

@author: tbury
"""


# import python libraries
import numpy as np
import pandas as pd

def iterate_model(yn1, yn, a, b, c, d, sigma):
    '''
    Iterate the model one time step using the Euler-Maruyama scheme
    
    Parameters:
        yn1 : national income at time step n+1
        yn : national income at time step n
        a : autonomous expenditures
        b : parameter in S-shaped function that determines fraction of income
            consumed by agents
        c : likewise
        d : policy maker's control parameter
    '''
    
    zeta = np.random.normal()
    
    # Difference equation
    yn2 = a + (b-d)*yn1 + d*yn + c*yn1/(1+np.exp(-(yn1-yn))) + sigma*zeta
    
    return yn2, yn1



def simulate_model(y1, y0, avals, sigma):
    '''
    Simulate a single realisaiton of model
    
    Parameters:
        M0 : initial condition of memory variable
        D0 : intitial condition of apd
        avals : list of consecutive values of a to run simulation with
        sigma: noise amp

    Returns
    -------
    df_traj: pd.DataFrame
        Realisation fo model

    '''
    
    
    # Model parameters (taken from Westerhoff paper)
    # b = 0.45
    # c = 0.1
    b = 0.05
    c = (1-b)/5.5
    d = 0
    
    # Simulation parameters
    tburn = 100
    
    # Initialise arrays to store single time-series data
    t = np.arange(len(avals))
    y_vals = np.zeros(len(avals))
    
    # Run burn-in period
    for i in range(int(tburn)):
        y1, y0 = iterate_model(y1,y0,avals[0],b,c,d,sigma)
        
    # Initial condition post burn-in period
    y_vals[0]=y0
    y_vals[1]=y1
    
    # Run simulation
    for i, a in enumerate(avals[:-2]):
        y_vals[i+2], _ = iterate_model(
            y_vals[i+1],y_vals[i],a,b,c,d,sigma,
            )

    # Put data in df
    df_traj = pd.DataFrame(
        {'time':t,
         'y': y_vals,
         })

    return df_traj



def sim_rate_forcing(sigma, rate_forcing= 10/500):
    '''
    Run a simulation with the bifurcation parameter varying at some defined rate
    
    Parameters:
        sigma: noise ampltiude
        rate_forcing: (change in a)/(change in time)
    
    '''
    
    y0 = 20
    y1 = 20

    astart = 10
    # astart = 15
    acrit = 20
    afinal = 22.5
    avals = np.arange(astart, afinal, rate_forcing)
    
    # Take transition time as time at bifurcation
    transition = int((acrit-astart)/rate_forcing)

    df_forced = simulate_model(y0, y1, avals, sigma)           
    series_forced = df_forced.set_index('time')['y']
    
    # Simulate a null trajectory with the same length as the pretransition section
    df_null = simulate_model(y0, y1, [astart]*transition, sigma)   
    series_null = df_null.set_index('time')['y']
    
    return series_forced, transition, series_null
    

