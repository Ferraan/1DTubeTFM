# -*- coding: utf-8 -*-
"""
Created on Thu May 22 15:59:32 2025

@author: blas_fe

This code solves a 1 dimensional flow through a rough pipe 
with heat transfer. The aim is to recreate the experimental 
results at the outlet from the data at the inlet and 
the roughness profile.
"""

import numpy as np
import cantera as ct
from dataclasses import dataclass

# Imports functions from other files
from FrictionCorrelations import RaKsStimpson
from LookupTable import Create_lookup_table
from Initialize import Initialization
from Solver import Solver
from Plotting import Plot


# Create database for speedup instead of using Cantera at
# each instance that thermophysical properties are wanted.
# interpolators contains a list of scipy RegularGridInterpolator
# objects for cp, cv, viscosity and density given a Temperature 
# and pressure

interpolators = Create_lookup_table()


# Geometry of the channel
L = 0.06  # m
D = 1e-3  # m
Ac = np.pi * (D/2)**2 # m2

# Numerical parameters
N = 50
epsilon = 1e-8

# Rugosity of channel
Ra = 12.51e-6 # m CHANGE TO CHANGE FRICTION FACTOR OF CHANNEL

# Boundary conditions
Tin = 5.38e+01+273.15  # K
Pin = 5.38e+00*1e+5  # Pa
mflowin = 2.87e-01/1000  # kg/s

Tout = 2.95e+02+273.15  # K
Pout = ct.one_atm  # Pa, 1atm



# Create a dataclass, C-like struct to store the boundary conditions
# so that it is easier to pass to functions
@dataclass
class BoundaryConditions:
    Tin: float
    Pin: float
    mflowin: float
    Tout: float
    Pout: float

# Instantiate the dataclass
BCs = BoundaryConditions(Tin, Pin, mflowin, Tout, Pout)

# Discretization
COORD = np.linspace(0, L, N)
DeltaX = COORD[1]-COORD[0]

# Initialize the problem. Returns Air, a struct with velocity, temp, 
# pressure... arrays as members.
Air = Initialization(N, Ra, D, BCs, Ac, COORD, interpolators, L)

# Get the equivalent sand grain roughness
KsDh = RaKsStimpson(Ra/D)
# As it is given with the hydraulic diameter, multiply to obtain
# equivalent sand grain roughness with radius.
KsR = KsDh*2 


## COMPUTATIONS
# %%
# Call the solver function
Air = Solver(epsilon, Air, KsR, DeltaX, 
            D, N, mflowin, Ac, interpolators)

## PLOTTING
# %%
Plot(Air, COORD, L)
