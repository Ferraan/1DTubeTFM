# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 10:35:48 2025

@author: blas_fe
"""
import numpy as np
from FrictionCorrelations import DarcyWeisbach, frictionFactor
from LookupTable import pT_interp

def Solver(epsilon, Air, KsR, DeltaX, 
           D, N, mflowin, Ac, interpolators):
    init_var = np.zeros((N, 1))
    Rair = 287.052874 # J kg−1 K−1 Gas constant
    # Iterative solver
    error = 1
    count = 0
    while error > epsilon:
        count += 1
        # Append new iteration onto the variables
        Air.T = np.append(Air.T, Air.T[:, -1][:, np.newaxis], axis = 1)
        
        #Update pressure with Darcy-Weisbach eq
        #FERRAN recalculate
        # Compute friction factor with the previous iteration Reynolds (f,p,L,D,density,v)
        friction_factor_vector = frictionFactor(Air.Reynolds[:], KsR)
        # Compute new pressures with DarcyWeisbach
        new_p = DarcyWeisbach(friction_factor_vector[0: -1], Air.p[0: -1, -1], DeltaX, D, Air.rho[0: -1, -1], Air.u[0: -1, -1])
        new_p = np.append(Air.p[0, -1], new_p).reshape(-1,1)
        # Append the new pressures
        Air.p = np.append(Air.p, new_p, axis = 1)
        
        # Update rho, u and mu
        Air.rho = np.append(Air.rho, np.zeros_like(init_var), axis = 1)
        Air.u = np.append(Air.u, np.zeros_like(init_var), axis = 1)
        Air.mu = np.append(Air.mu, np.zeros_like(init_var), axis = 1)
        
        Air.rho[0, -1] = Air.rho[0, -2]
        Air.u[0, -1] = Air.u[0, -2]
        Air.mu[0, -1] = Air.mu[0, -2]
        # Get density at this T and P
        Air.cp[1:, -1], Air.cv[1:, -1], Air.mu[1:, -1], Air.rho[1:, -1],  Air.gamma[1:, -1] =pT_interp(interpolators,
                                                               Air.p[1:,-1], Air.T[1:,-1])
        
        
        # Compute velocity from conservation equation
        Air.u[1:, -1] = mflowin / (Ac * Air.rho[1:, -1])
        
        error = np.sum(np.abs(Air.u[:, -1] - Air.u[:, -2]))
        
        if count > 1000:
            break
    
    Air.Reynolds = Air.u[:, -1] * D * Air.rho[:, -1] / Air.mu[:, -1]
    Air.Mach = Air.u[:, -1]/np.sqrt(Air.gamma[:, -1] * Rair * Air.T[:, -1])
    return Air