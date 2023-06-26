#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:51:19 2022

Generic functions for Fox model

@author: tbury
"""

# import python libraries
import numpy as np
import pandas as pd

import plotly.graph_objects as go


def simulate_model(A, B, C, D, tau, alpha, M0, D0, Tvals, sigma_1, sigma_2):
    '''
    Simulate a single realisaiton of model
    
    Parameters:
        M0 : initial condition of memory variable
        D0 : intitial condition of apd
        Tvals : list of consecutive values of T to run simulation at
        sigma_1 : noise amplitude for M
        sigma_2 : noise amplitude for D

    Returns
    -------
    df_traj: pd.DataFrame
        Realisation of model

    '''
  
    # # Model parameters
    # A = 88
    # B = 122
    # C = 40
    # D = 28
    # tau = 180
    # alpha = 0.2

    # Simulation parameters
    tburn = 100
    
    def iterate_model(Mn,Dn,T,sigma_1,sigma_2):
        '''
        Iterate the model one time step using the Euler-Maruyama scheme
        
        Parameters:
            Mn : memory variable for beat n
            Dn : action potential duration for beat n
            T : period of pacing (bifurcation parameter)
        '''
        
        zeta_1 = np.random.normal()
        zeta_2 = np.random.normal()
        
        # Intermediate varaibles
        In = T - Dn # interval between APs.
        
        # Subsequent state variables
        Mn_next = np.exp(-In/tau)*(1+ (Mn-1)*np.exp(-Dn/tau)) + sigma_1*zeta_1
        # Dn_next = (1-alpha*Mn_next)*(A + B/(1+np.exp(-(In-C)/D))) + sigma_2*zeta_2
        Dn_next = (1-alpha*Mn_next)*(A + B/(1+np.exp(-(In-C)/D)))
        
        return Mn_next, Dn_next

    # Initialise arrays to store single time-series data
    t = np.arange(0,len(Tvals))
    M_vals = np.zeros(len(Tvals))
    D_vals = np.zeros(len(Tvals))
    
    # # Set up bifurcation parameter, that varies linearly between Tstart and Tfin
    # T = pd.Series(np.linspace(Tstart,Tfinal,len(t)),index=t)

    # Run burn-in period
    for i in range(int(tburn)):
        M0, D0 = iterate_model(M0,D0,Tvals[0],sigma_1,sigma_2)
    
    # Initial condition post burn-in period
    M_vals[0]=M0
    D_vals[0]=D0
    
    # Run simulation
    for i, T in enumerate(Tvals[:-1]):
        M_vals[i+1], D_vals[i+1] = iterate_model(
            M_vals[i],D_vals[i],T,sigma_1,sigma_2
            )

    # Put data in df
    df_traj = pd.DataFrame(
        {'time':t,
         'D': D_vals,
         'M': M_vals})

    return df_traj
    


def sim_rate_forcing(sigma, rate_forcing=100/500):
    '''
    Run a simulation with the bifurcation parameter varying at some defined rate
    
    Parameters:
        sigma: noise ampltiude
        rate_forcing: (change in T)/(change in time)
    
    Note change in T from start to bifurcation is 100.
    '''
    
    D0 = 200 
    M0 = 1

    Tstart = 300
    Tcrit = 200
    Tfinal = 150
    Tvals = np.arange(Tstart, Tfinal, -rate_forcing)
    
    # Take transition time as time at bifurcation
    transition = int((Tstart-Tcrit)/rate_forcing)

    df_forced = simulate_model(M0, D0, Tvals, sigma, sigma)           
    series_forced = df_forced.set_index('time')['D']
    
    # Simulate a null trajectory with the same length as the pretransition section
    df_null = simulate_model(M0, D0, [Tstart]*transition, sigma, sigma)   
    series_null = df_null.set_index('time')['D']
    
    return series_forced, transition, series_null
    






